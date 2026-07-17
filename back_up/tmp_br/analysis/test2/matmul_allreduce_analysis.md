# VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE 环境变量梳理

## 1. 定义

**envs.py**：
```python
"VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0"))),
```

默认值：`0`（关闭）

## 2. 读取函数

**vllm_ascend/utils.py**：
```python
def matmul_allreduce_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE
```

## 3. 代码中的使用

### 3.1 Row Parallel 算子选择（核心用途）

**vllm_ascend/ops/linear_op.py:671**：

在 `get_row_parallel_op()` 函数中，`matmul_allreduce_enable()` 作为 Row Parallel 算子选择的判断条件之一：

```python
def get_row_parallel_op(prefix, layer):
    if enable_dsa_cp_with_layer_shard() and "o_proj" in prefix:
        return ShardedCPRowParallelOp(layer)
    if "down_proj" in prefix and mlp_tp_enable() and not is_moe_layer(prefix):
        return MLPRowParallelOp(layer)
    if "o_proj" in prefix and oproj_tp_enable():
        return OProjRowParallelOp(layer)
    if matmul_allreduce_enable():                          # <-- 这里
        return MatmulAllreduceRowParallelOp(layer)
    if flashcomm2_enable():
        if "o_proj" in prefix or "out_proj" in prefix:
            return Flashcomm2OProjRowParallelOp(layer)
    if enable_sp():
        ...
```

**优先级**：matmul_allreduce 在 flashcomm2 和 enable_sp 之前，但排在 DSA CP、MLP TP、OProj TP 之后。

### 3.2 MatmulAllreduceRowParallelOp 的实现

**vllm_ascend/ops/linear_op.py:385-413**：

```python
class MatmulAllreduceRowParallelOp(CustomRowParallelOp):
    def apply_impl(self, input_):
        input_parallel = self.get_input_parallel(input_)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        if self.reduce_results and self.tp_size > 1:
            output = torch_npu.npu_mm_all_reduce_base(
                input_parallel, self.layer.weight.t(), self.hcomm_info, bias=bias_
            )
        else:
            output = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
```

**核心操作**：使用 `torch_npu.npu_mm_all_reduce_base` 将矩阵乘法和 AllReduce 融合成一个算子，减少通信开销。

### 3.3 AllReduce+RMSNorm 融合（编译优化）

**vllm_ascend/compilation/passes/allreduce_rmsnorm_fusion_pass.py**：

当 `fuse_allreduce_rms` 启用时，编译 pass 会将 `matmul → allreduce → add_rms_norm` 三个操作融合为 `matmul_allreduce_add_rmsnorm` 一个算子。

### 3.4 Batch Invariant 模式下禁用

**vllm_ascend/batch_invariant.py:81**：

```python
os.environ["VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE"] = "0"
```

在 batch invariant 模式下强制关闭 matmul_allreduce，因为融合算子不能保证 batch 间的不变性。

## 4. 限制条件

| 条件 | 说明 |
|------|------|
| **仅 A2 支持** | envs.py 注释明确说明 "this feature is supported in A2"，即仅 Atlas A2 系列芯片支持 |
| **Eager 模式更优** | 注释说明 "eager mode will get better performance"，即 eager 模式下性能更好 |
| **与 FlashComm2 互斥** | 在 `get_row_parallel_op()` 中，matmul_allreduce 优先级高于 flashcomm2，开启后 flashcomm2 不会生效 |
| **与 enable_sp 互斥** | 同理，matmul_allreduce 优先级高于 enable_sp，开启后 SP 不会生效 |
| **Batch Invariant 不兼容** | batch invariant 模式下强制关闭 |
| **需要 TP > 1** | `self.reduce_results and self.tp_size > 1`，单卡无意义 |
| **仅影响 Row Parallel 层** | 只对 Row Parallel 线性层生效（o_proj、down_proj 等） |

## 5. 工作原理

```
传统流程：
  input × weight → matmul → allreduce → output

融合流程（matmul_allreduce）：
  input × weight → npu_mm_all_reduce_base → output
                    (matmul + allreduce 融合)

进一步融合（allreduce_rmsnorm_fusion）：
  input × weight → matmul_allreduce_add_rmsnorm → output + norm_output
                    (matmul + allreduce + add + rms_norm 融合)
```

**性能收益**：减少 matmul 和 allreduce 之间的显存搬运和同步开销，提升 TP 场景下的推理性能。

## 6. 验证脚本分析

当前 `run_server.sh` 中**没有设置 `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`**，即默认关闭。

如果要验证此环境变量的迁移，建议创建对比脚本：

**run_server_env.sh**（环境变量方式）：
```bash
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1
```

**run_server_config.sh**（Config 方式）：
```bash
--additional-config '{"enable_matmul_allreduce": true}'
```

**验证方法**：
1. 两个脚本分别启动服务
2. 查看日志中是否出现 `npu_mm_all_reduce_base` 相关调用
3. 对比推理性能（开启后 TP 通信延迟应降低）
4. 确认两种方式行为一致
