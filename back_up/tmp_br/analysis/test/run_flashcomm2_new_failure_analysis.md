# run_flashcomm2_new.log 分析：加 --enforce-eager 后的新失败点

分析对象：

- 新日志：`D:/lzy/code/for_env/result/analysis/test/run_flashcomm2_new.log`
- 脚本来源：`D:/lzy/code/for_env/result/analysis/test/run_flashcomm2.sh`
- 代码：`D:/lzy/code/for_env/vllm-ascend` main 分支

## 1. 结论

`--enforce-eager` 已经生效，并且已经绕过了上一版日志里的 TorchDynamo / FlashComm2 `all_to_all_single` fake tensor 编译失败。

现在服务仍然拉起失败，但失败点已经变了：当前错误是 Ascend910B 环境不支持自定义算子 `AddRmsNormBias`。

核心报错：

```text
RuntimeError: call aclnnAddRmsNormBias failed, detail:[PID: 106010] 2026-05-12-19:33:40.020.657 AclNN_Parameter_Error(EZ1001): Get regInfo failed, The binary_info_config.json of socVersion [ascend910b] does not support opType [AddRmsNormBias].
```

## 2. --enforce-eager 已生效

日志确认 `--enforce-eager` 已进入 vLLM 配置：

```text
non-default args: {... 'enforce_eager': True, ...}
```

并且 vLLM 明确关闭了 torch.compile 和 CUDAGraph：

```text
Enforce eager set, disabling torch.compile and CUDAGraphs.
Compilation disabled, using eager mode by default
```

EngineCore 配置也显示：

```text
compilation_config={'mode': <CompilationMode.NONE: 0>, ... 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, ...}
```

所以，上一版 `run_flashcomm2.log` 中的 Dynamo fake tensor 报错已经不是当前失败原因。

## 3. FlashComm2 仍然正常开启

日志继续确认 FlashComm2 开启成功：

```text
Enable FLASHCOMM2 with flashcomm2_oproj_tensor_parallel_size = 1
```

也仍然有这两个 warning：

```text
It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance.
It is recommended to enable FLASHCOMM2 in P-scenario deployments, enable it in hybrid deployment may lead to decode performance degradation.
```

这两个 warning 仍不是直接失败原因：

- 未开启 FlashComm1 只是性能建议，不会阻断启动。
- 非 P 场景只是性能/场景建议，不会阻断启动。

## 4. 新失败点在哪里

新失败发生在 profile run / dummy run 的 Llama forward 中，但已经不是 `o_proj` 的 FlashComm2 all-to-all，而是 post attention layernorm：

```text
llama.py:331, hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
vllm_ascend/ops/layernorm.py:72, x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(...)
RuntimeError: call aclnnAddRmsNormBias failed
```

对应代码是 `vllm_ascend/ops/layernorm.py:69`：

```python
if residual is not None:
    residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
    if enable_custom_op():
        x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
            x, residual, self.weight, self.bias, self.variance_epsilon
        )
    else:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
        if self.bias is not None:
            x.add_(self.bias)
```

当前环境中 `enable_custom_op()` 返回了 True，所以代码选择了 `_C_ascend.npu_add_rms_norm_bias`。但运行时报：

```text
binary_info_config.json of socVersion [ascend910b] does not support opType [AddRmsNormBias]
```

这说明当前安装/编译出来的自定义算子包里没有适配 `ascend910b` 的 `AddRmsNormBias` regInfo，或者当前 CANN/自定义算子产物不支持这个 opType。

## 5. 根因判断

当前失败根因是自定义算子能力与当前硬件/算子包不匹配：

- 硬件识别为 `ascend910b`。
- vllm-ascend 自定义算子已注册成功，因此 `enable_custom_op()` 为 True。
- `AscendRMSNorm.forward_oot()` 进入了 `_C_ascend.npu_add_rms_norm_bias` 分支。
- ACLNN 查询 `AddRmsNormBias` 的 regInfo 失败，说明该自定义算子对当前 socVersion 不可用。

这不是 FlashComm2 参数限制导致的，也不是 `--enforce-eager` 没生效。

## 6. 如何修改 / 规避

### 6.1 推荐先验证：禁用 vllm-ascend 自定义算子路径

从代码看，只要 `enable_custom_op()` 为 False，`AscendRMSNorm.forward_oot()` 就会走 torch_npu 原生 fallback：

```python
x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
if self.bias is not None:
    x.add_(self.bias)
```

因此建议先用能关闭自定义 op 的方式验证服务是否能拉起。

可尝试在启动前增加 vLLM batch invariant 开关：

```bash
export VLLM_BATCH_INVARIANT=1
```

代码依据是 `vllm_ascend/utils.py:317`：

```python
if envs.VLLM_BATCH_INVARIANT or get_ascend_device_type() == AscendDeviceType.A5:
    _CUSTOM_OP_ENABLED = False
```

注意：`VLLM_BATCH_INVARIANT=1` 会带来额外行为，不只是关闭这个 RMSNorm bias 自定义算子。`vllm_ascend/batch_invariant.py:76` 还会设置确定性相关环境变量，并关闭部分融合路径。因此它适合作为快速验证/规避，不一定是最终性能方案。

修改后的验证片段：

```bash
export VLLM_BATCH_INVARIANT=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1

vllm serve ${MODEL_PATH}   \
    --served-model-name llama3-8b   \
    --trust-remote-code   \
    --dtype bfloat16   \
    --tensor-parallel-size 4   \
    --max-num-seqs 32   \
    --enable-chunked-prefill   \
    --no-enable-prefix-caching   \
    --async-scheduling   \
    --enforce-eager   \
    --gpu-memory-utilization 0.9   \
    --max-num-batched-tokens 2768   \
    --host 0.0.0.0   \
    --port 8000
```

### 6.2 更根本的修复：检查/重编自定义算子包

如果目标是保留自定义 op 性能路径，需要检查当前 vllm-ascend 自定义算子产物是否包含 `ascend910b` 的 `AddRmsNormBias` 支持：

- 确认 CANN 版本和 vllm-ascend 自定义算子编译版本匹配。
- 确认编译时 `SOC_VERSION` 是否正确覆盖到当前硬件，例如 Ascend910B 对应的 socVersion。
- 重新编译/安装 vllm-ascend 自定义算子，使 `binary_info_config.json` 包含 `AddRmsNormBias` 对 `ascend910b` 的 regInfo。

当前日志中的错误明确是 regInfo 查不到，不是 Python 参数形状不对。

### 6.3 仍建议开启 FlashComm1

当前日志仍提示：

```text
It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance.
```

建议脚本里补上：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

这不是修复 `AddRmsNormBias` 的关键项，但可以避免 FlashComm2 路径后续额外走全局 TP `all_gather()`，更接近代码推荐配置。

## 7. 下一步建议

建议按以下顺序排查：

1. 保留 `--enforce-eager`，增加 `export VLLM_BATCH_INVARIANT=1`，确认是否能绕过 `AddRmsNormBias` 并成功拉起。
2. 同时增加 `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`，保持 FlashComm2 推荐组合。
3. 如果 batch invariant 能拉起，说明 FlashComm2 主路径已经可运行，剩余问题是自定义 `AddRmsNormBias` 算子对 `ascend910b` 不可用。
4. 如果要追求最终性能，再回头修自定义算子包/CANN/SOC_VERSION，而不是继续调整 `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`。

## 8. 与上一版日志的区别

上一版 `run_flashcomm2.log`：

- 失败在 `Flashcomm2OProjRowParallelOp` 的 `dist.all_to_all_single()`。
- 根因是 TorchDynamo fake tensor tracing 编译 FlashComm2 all-to-all 路径失败。
- `--enforce-eager` 是正确规避。

新版 `run_flashcomm2_new.log`：

- `--enforce-eager` 已生效。
- 已绕过 TorchDynamo 编译失败。
- 新失败点变为 `_C_ascend.npu_add_rms_norm_bias`。
- 根因是当前 `ascend910b` 算子包不支持 `AddRmsNormBias`。
