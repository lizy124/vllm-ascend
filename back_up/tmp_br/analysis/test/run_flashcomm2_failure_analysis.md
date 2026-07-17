# run_flashcomm2.sh 服务拉起失败分析

分析对象：

- 脚本：`D:/lzy/code/for_env/result/analysis/test/run_flashcomm2.sh`
- 日志：`D:/lzy/code/for_env/result/analysis/test/run_flashcomm2.log`
- 代码：`D:/lzy/code/for_env/vllm-ascend` main 分支

## 1. 现象

服务没有成功拉起，失败发生在 EngineCore 初始化阶段。

日志里的最终失败是：

```text
RuntimeError: Engine core initialization failed. See root cause above.
```

更靠前的 EngineCore 根因是：

```text
RuntimeError: Worker failed with error 'Dynamo failed to run FX node with fake tensors: call_method copy_(...): got RuntimeError('expand: the requested shape has too few dimensions!')
```

失败堆栈指向 Llama 的 `o_proj`：

```text
llama.py:232, output, _ = self.o_proj(attn_output)
vllm_ascend/ops/linear.py:348, return self.custom_op.apply(input_)
vllm_ascend/ops/linear_op.py:350, input_parallel = otp_maybe_quant_comm(input_parallel)
vllm_ascend/ops/linear_op.py:338, dist.all_to_all_single(recv_buf, send_buf, group=self.odp_group.device_group)
torch/distributed/_functional_collectives.py:1130, return output.copy_(...)
```

## 2. 不是环境变量取值非法

日志证明 FlashComm2 已经被正确识别并开启：

```text
Enable FLASHCOMM2 with flashcomm2_oproj_tensor_parallel_size = 1
```

当前脚本关键配置是：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1
vllm serve ${MODEL_PATH} \
    --tensor-parallel-size 4 \
    --max-num-batched-tokens 2768
```

对照代码限制：

- `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1` 大于 0，会开启 FlashComm2。
- `--tensor-parallel-size 4` 下，`1 < 4`，满足 `flashcomm2_oproj_tp_size` 必须小于全局 TP 的限制。
- `4 % 1 == 0`，满足全局 TP 必须能整除 FlashComm2 parallel size 的限制。

所以这次失败不是 `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1` 本身不合法。

## 3. 直接原因

直接原因是：脚本没有开启 eager，默认走了 vLLM compile / ACLGraph 路径；FlashComm2 的 `o_proj` all-to-all 通信路径在 TorchDynamo fake tensor tracing 阶段失败。

日志证据：

```text
compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, ... 'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>, ...}
```

随后在 profile run / dummy run 中编译 Llama forward 时失败：

```text
determine_available_memory -> profile_run -> _dummy_run -> _model_forward -> llama.forward -> self.o_proj
```

最终失败点是 `Flashcomm2OProjRowParallelOp` 的 all-to-all：

```text
vllm_ascend/ops/linear_op.py:338, dist.all_to_all_single(recv_buf, send_buf, group=self.odp_group.device_group)
torch/distributed/_functional_collectives.py:1130, return output.copy_(...)
RuntimeError('expand: the requested shape has too few dimensions!')
```

对应代码路径：

- `vllm_ascend/ops/linear_op.py:673`：FlashComm2 开启后，`o_proj` / `out_proj` 会选择 `Flashcomm2OProjRowParallelOp`。
- `vllm_ascend/ops/linear_op.py:314`：`otp_maybe_quant_comm()` 开始做 FlashComm2 通信前的数据重排。
- `vllm_ascend/ops/linear_op.py:338`：调用 `dist.all_to_all_single()`。
- `vllm_ascend/ops/linear_op.py:350`：非 W8A8 量化时，会在 matmul 前立即执行 `otp_maybe_quant_comm()`。

当前模型是 `llama3-8b`，日志中 `quantization=None`，所以会走非 W8A8 分支，直接触发 `otp_maybe_quant_comm(input_parallel)`。

## 4. 根因判断

根因是 FlashComm2 当前脚本组合触发了“图编译 + dense BF16 Llama + FlashComm2 all_to_all_single”的不兼容路径。

这点可以从现有 e2e 用例侧面印证：`vllm-ascend/tests/e2e/multicard/2-cards/test_offline_inference_distributed.py:153` 和 `vllm-ascend/tests/e2e/multicard/2-cards/test_offline_inference_distributed.py:172` 的 FlashComm2 用例都设置了 `enforce_eager=True`，其中 `test_qwen3_moe_fc2_oshard_tp2` 旁边还有注释：

```python
enforce_eager=True,  # TODO(Levi-JQ): support graph mode for fc2 in Qwen
```

虽然注释写的是 Qwen，但本次日志说明 Llama dense 模型也会在编译路径里撞到 FlashComm2 `all_to_all_single` 的 Dynamo fake tensor 问题。

## 5. 日志中的 warning 怎么看

日志里还有两个 warning：

```text
It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance.
It is recommended to enable FLASHCOMM2 in P-scenario deployments, enable it in hybrid deployment may lead to decode performance degradation.
```

这两个都不是导致启动失败的直接原因：

- 未开启 FlashComm1 只影响推荐性能组合，不会阻断启动。
- 未配置 PD/P 节点只会给 warning，不会阻断启动。

但建议仍然按代码提示打开 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`，否则 `Flashcomm2OProjRowParallelOp` 在 FlashComm1 未启用时会额外做一次全局 TP `all_gather()`。

## 6. 修改建议

### 6.1 最小修改：强制 eager

当前失败发生在 TorchDynamo / ACLGraph 编译阶段，最小修改是在 serve 参数里增加：

```bash
--enforce-eager
```

并建议同时增加 FlashComm1：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

修改后的关键片段：

```bash
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

### 6.2 如果要验证 O-shard

如果目标是验证 FlashComm2 + O-shard 省内存路径，可以再增加：

```bash
--additional-config '{"layer_sharding": ["o_proj"]}'
```

注意：FlashComm2 下 `layer_sharding` 只能是 `["o_proj"]`，不能加 `q_b_proj` 等其他层。

### 6.3 如果想先确认基线服务可拉起

可以先临时关闭 FlashComm2：

```bash
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
```

如果关闭 FlashComm2 后同一模型和同一 TP 能成功拉起，则能进一步确认本次问题就是 FlashComm2 的 all-to-all 编译路径，而不是模型路径、设备、权重或 TP 基础配置问题。

## 7. 结论

`run_flashcomm2.sh` 失败不是因为 `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1` 对 TP=4 不合法；这个值是合法的，并且日志确认 FlashComm2 已启用。

真正失败原因是默认图编译路径在 profile/dummy run 编译 `llama3-8b` 的 `o_proj` 时，进入 `Flashcomm2OProjRowParallelOp`，其中 `dist.all_to_all_single()` 被 TorchDynamo fake tensor tracing 到 `copy_`，最终触发 `expand: the requested shape has too few dimensions!`。建议先按项目现有 FlashComm2 e2e 用例方式加 `--enforce-eager`，并同时设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`。