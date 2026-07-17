# VLLM_ASCEND_ENABLE_FLASHCOMM1 是否适合迁移到 Config 的分析

## 1. 结论

`VLLM_ASCEND_ENABLE_FLASHCOMM1` **适合迁移到 `AscendConfig.additional_config`**。

它和 `DYNAMIC_EPLB`、`VLLM_ASCEND_BALANCE_SCHEDULING` 不同：

- `DYNAMIC_EPLB` / `VLLM_ASCEND_BALANCE_SCHEDULING` 用于 Config 初始化前的 import-time monkey patch gate。
- `VLLM_ASCEND_ENABLE_FLASHCOMM1` 不控制 import-time patch，不负责替换 vLLM 核心类。
- 它主要控制运行时/初始化阶段是否启用 FlashComm1 / sequence parallel 相关路径。
- 这些读取点大多发生在 `vllm_config` / `AscendConfig` 可用之后。

因此，从时序和功能性质看，它属于可迁移的运行时优化开关。

推荐迁移形式：

```bash
--additional-config '{"enable_flashcomm1": true}'
```

但迁移时需要处理两个细节：

1. `enable_sp()` 当前有全局缓存 `_ENABLE_SP`，迁移后仍要保证 refresh / clear 逻辑正确。
2. 当前还兼容旧变量 `VLLM_ASCEND_ENABLE_FLASHCOMM`，迁移时要决定是否保留 fallback。

---

## 2. 变量定义

当前 `ascend_config` 分支中，`VLLM_ASCEND_ENABLE_FLASHCOMM1` 仍保留在 `envs.py`：

```python
# vllm_ascend/envs.py:74
# Whether to enable FlashComm optimization when tensor parallel is enabled.
# This feature will get better performance when concurrency is large.
"VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
```

它没有像 `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`、`VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`、`VLLM_ASCEND_ENABLE_NZ` 等变量一样被标记为 removed/deprecated。

这说明当前 PR/分支没有把 FlashComm1 纳入最终迁移范围，或者迁移过程中被保留了。

---

## 3. 核心读取点：`enable_sp()`

FlashComm1 的核心读取函数是 `enable_sp()`：

```python
# vllm_ascend/utils.py:818
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except AssertionError:
            vllm_config = None

    additional_config = getattr(vllm_config, "additional_config", None) if vllm_config is not None else None
    refresh = additional_config.get("refresh", False) if additional_config else False

    if _ENABLE_SP is None or refresh:
        _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1 or bool(
            int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0"))
        )

        if not _ENABLE_SP and enable_shared_expert_dp:
            _ENABLE_SP = True
            logger.info("shared_expert_dp requires enable_sp = True. has set enable_sp to True")

    return _ENABLE_SP
```

这里可以看出：

1. `VLLM_ASCEND_ENABLE_FLASHCOMM1` 是 FlashComm1 的主开关。
2. `VLLM_ASCEND_ENABLE_FLASHCOMM` 是旧兼容变量。
3. 结果会缓存到模块级变量 `_ENABLE_SP`。
4. 当 `additional_config.refresh` 为 true 时，会重新计算 `_ENABLE_SP`。
5. `enable_shared_expert_dp` 可以强制打开 `_ENABLE_SP`。

因此迁移时不能只改一处读取，还要处理缓存语义。

---

## 4. 是否是 import-time patch gate？

判断一个环境变量是否不适合迁移到 Config，最关键要看它是否在 `AscendConfig` 初始化前控制 monkey patch。

`DYNAMIC_EPLB` 和 `VLLM_ASCEND_BALANCE_SCHEDULING` 在这里被读取：

```python
# vllm_ascend/patch/platform/__init__.py
if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    import vllm_ascend.patch.platform.patch_multiproc_executor

if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule
```

但 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 没有出现在 `patch/platform/__init__.py` 的 import-time patch gate 中。

也就是说，FlashComm1 不会在 Config 初始化前决定是否替换：

- `MultiprocExecutor`
- `Scheduler`
- `EngineCoreProc.run_engine_core`
- 其他 vLLM 核心类

这和 Dynamic EPLB / Balance Scheduling 有本质区别。

结论：

```text
VLLM_ASCEND_ENABLE_FLASHCOMM1 不是 early monkey-patch gate。
```

这为迁移到 Config 提供了基础条件。

---

## 5. FlashComm1 影响的功能路径

### 5.1 forward context 中决定是否启用 FlashComm1

`ascend_forward_context.py` 会在 forward context 中设置 `flash_comm_v1_enabled`：

```python
# vllm_ascend/ascend_forward_context.py:115
is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
if is_context_moe_model:
    flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None
    mmrs_fusion = False
elif is_draft_model:
    flash_comm_v1_enabled = False
else:
    flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000
forward_context.mmrs_fusion = mmrs_fusion
forward_context.num_tokens = num_tokens
forward_context.flash_comm_v1_enabled = flash_comm_v1_enabled
```

这里的逻辑是：

- MoE 模型：只要 `enable_sp(vllm_config)` 为 true 且 `num_tokens` 存在，就启用 FlashComm1。
- dense drafter：强制关闭。
- 普通 dense 模型：`enable_sp(vllm_config)` 为 true 且 `num_tokens > 1000` 才启用。

这说明 FlashComm1 是运行时 forward 行为开关，不是 import 阶段结构 patch。

### 5.2 padding 逻辑依赖 FlashComm1 / FlashComm2

同一个上下文中还有：

```python
# vllm_ascend/ascend_forward_context.py:130
forward_context.flashcomm_v2_enabled = flashcomm2_enable() and tp_world_size > 1 and num_tokens is not None

# vllm_ascend/ascend_forward_context.py:133
if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
    pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
    forward_context.pad_size = pad_size
```

也就是说 FlashComm1 会影响 token padding 和后续通信路径。

### 5.3 线性层算子选择依赖 `enable_sp()`

Column Parallel 路径：

```python
# vllm_ascend/ops/linear_op.py:637
if enable_sp():
    if "shared_expert" in prefix:
        return None
    sp_column_prefix = [
        "gate_up_proj",
        "in_proj",
        "qkv_proj",
        "conv1d",
        "query_key_value",
    ]
    for a_prefix in sp_column_prefix:
        if a_prefix in prefix:
            return SequenceColumnParallelOp(layer)
```

Row Parallel 路径：

```python
# vllm_ascend/ops/linear_op.py:676
if enable_sp():
    if "shared_expert" in prefix:
        return None
    sp_row_prefixes = [
        "o_proj",
        "out_proj",
        "down_proj",
        "attention.dense",
    ]
    for a_prefix in sp_row_prefixes:
        if a_prefix in prefix:
            return SequenceRowParallelOp(layer)
```

这说明 FlashComm1 / SP 会影响线性层 custom op 的选择：

- Column Parallel 可能变成 `SequenceColumnParallelOp`
- Row Parallel 可能变成 `SequenceRowParallelOp`

这些选择通常发生在模型构建 / layer 初始化时，属于 Config 可以覆盖的阶段。

### 5.4 与 `matmul_allreduce`、FlashComm2 的优先级关系

Row Parallel 中相关顺序是：

```python
# vllm_ascend/ops/linear_op.py:665
if enable_dsa_cp_with_layer_shard() and "o_proj" in prefix:
    return ShardedCPRowParallelOp(layer)
if "down_proj" in prefix and mlp_tp_enable() and not is_moe_layer(prefix):
    return MLPRowParallelOp(layer)
if "o_proj" in prefix and oproj_tp_enable():
    return OProjRowParallelOp(layer)
if matmul_allreduce_enable():
    return MatmulAllreduceRowParallelOp(layer)
if flashcomm2_enable():
    if "o_proj" in prefix or "out_proj" in prefix:
        return Flashcomm2OProjRowParallelOp(layer)
if enable_sp():
    ...
```

优先级是：

```text
DSA CP / finegrained TP / matmul_allreduce / FlashComm2 > FlashComm1-SP
```

所以即使 FlashComm1 开启，也可能被更高优先级的 op 选择覆盖。

### 5.5 `AscendConfig` 初始化中也会调用 `enable_sp()`

`AscendConfig.__init__()` 中已经在部分配置校验/修正时调用 `enable_sp(vllm_config)`：

```python
# vllm_ascend/ascend_config.py:100
if self.enable_shared_expert_dp:
    assert enable_sp(vllm_config=vllm_config, enable_shared_expert_dp=True)
```

```python
# vllm_ascend/ascend_config.py:103
if vllm_config.parallel_config.prefill_context_parallel_size > 1 and enable_sp(vllm_config=vllm_config):
    tp_pcp_size = (
        vllm_config.parallel_config.tensor_parallel_size
        * vllm_config.parallel_config.prefill_context_parallel_size
    )
    if vllm_config.scheduler_config.max_num_batched_tokens % tp_pcp_size != 0:
        vllm_config.scheduler_config.max_num_batched_tokens = (
            cdiv(vllm_config.scheduler_config.max_num_batched_tokens, tp_pcp_size) * tp_pcp_size
        )
        logger.warning_once(...)
```

这里的作用是：

- `enable_shared_expert_dp` 需要 SP/FlashComm1 能力。
- 当 PCP + FlashComm1 同时启用时，调整 `max_num_batched_tokens`，确保能被 `tp_size * pcp_size` 整除。

这些逻辑都在 `AscendConfig` 初始化过程中或之后执行，因此可以由 `additional_config` 驱动。

---

## 6. 当前为什么可能还没迁移

虽然 FlashComm1 适合迁移，但当前 `ascend_config` 分支里它仍保留为 env。可能原因如下。

### 6.1 `_ENABLE_SP` 全局缓存需要谨慎处理

`enable_sp()` 使用全局缓存：

```python
# vllm_ascend/utils.py:819
_global _ENABLE_SP
```

计算后缓存：

```python
# vllm_ascend/utils.py:831
if _ENABLE_SP is None or refresh:
    _ENABLE_SP = ...
```

如果迁移到 Config，必须确保不同测试、不同 engine、不同 `VllmConfig` 间不会复用旧值。

当前 `clear_ascend_config()` 已经清理 `_ENABLE_SP`：

```python
# vllm_ascend/ascend_config.py:610
def clear_ascend_config():
    global _ASCEND_CONFIG
    _ASCEND_CONFIG = None
    from vllm_ascend.utils import clear_enable_sp

    clear_enable_sp()
```

而 `clear_enable_sp()` 会清理 `_ENABLE_SP` 和相关 cache：

```python
# vllm_ascend/utils.py:71
def clear_enable_sp():
    global _ENABLE_SP
    _ENABLE_SP = None
    enable_dsa_cp.cache_clear()
    enable_dsa_cp_with_layer_shard.cache_clear()
    enable_dsa_cp_with_o_proj_tp.cache_clear()
    _libc_getenv.cache_clear()
```

这说明迁移是可行的，但需要保持这个清理机制。

### 6.2 旧变量兼容问题

`enable_sp()` 当前还兼容旧变量：

```python
# vllm_ascend/utils.py:832
_ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1 or bool(
    int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0"))
)
```

也就是说实际有两个环境变量影响 FlashComm1：

```text
VLLM_ASCEND_ENABLE_FLASHCOMM1
VLLM_ASCEND_ENABLE_FLASHCOMM
```

迁移时需要决定：

1. 是否保留 `VLLM_ASCEND_ENABLE_FLASHCOMM` 作为 deprecated fallback。
2. 如果 `additional_config.enable_flashcomm1` 和旧 env 同时设置且冲突，谁优先。
3. 是否输出 warning。

推荐优先级：

```text
additional_config.enable_flashcomm1 > VLLM_ASCEND_ENABLE_FLASHCOMM1 > VLLM_ASCEND_ENABLE_FLASHCOMM > 默认 False
```

这样兼容性最好。

### 6.3 与 FlashComm2 的耦合

`get_flashcomm2_config_and_validate()` 里会检查 FlashComm1 是否开启：

```python
# vllm_ascend/utils.py:1162
if not envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1:
    logger.warning_once(
        "It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance."
    )
```

如果迁移 FlashComm1，这里也要同步改成：

```python
if not ascend_config.enable_flashcomm1:
    logger.warning_once(...)
```

否则就会出现：

```text
用户通过 additional_config.enable_flashcomm1=true 开启 FlashComm1，
但 FlashComm2 校验仍读 env，误以为 FlashComm1 没开，打印错误 warning。
```

### 6.4 `enable_sp_by_pass` 与编译 pass 的关系

`AscendConfig` 中还有一个 `enable_sp_by_pass`：

```python
# vllm_ascend/ascend_config.py:207
self.enable_sp_by_pass = (
    vllm_config.model_config is not None
    and not vllm_config.model_config.enforce_eager
    and vllm_config.compilation_config.pass_config.enable_sp
)
```

对应工具函数：

```python
# vllm_ascend/utils.py:814
def enable_sp_by_pass():
    return get_ascend_config().enable_sp_by_pass
```

这说明 SP/FlashComm1 相关能力有两条路径：

1. `enable_sp()`：环境变量控制的 FlashComm1 / sequence parallel custom op 路径。
2. `enable_sp_by_pass()`：编译 pass 控制的 SP 路径。

迁移 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 不应混淆这两者。

---

## 7. 是否适合迁移的判断标准

可以用以下标准判断一个 env 是否适合迁移到 Config：

| 判断项 | `VLLM_ASCEND_ENABLE_FLASHCOMM1` 情况 | 结论 |
|---|---|---|
| 是否在 `patch/platform/__init__.py` import 阶段控制 monkey patch | 否 | 适合迁移 |
| 是否依赖 Config 初始化前生效 | 否 | 适合迁移 |
| 是否主要影响运行时/模型构建/算子选择 | 是 | 适合迁移 |
| 是否已有 `vllm_config` 参与读取 | 是，`enable_sp(vllm_config)` | 适合迁移 |
| 是否存在缓存/兼容复杂度 | 是，`_ENABLE_SP` 和旧 env | 需要谨慎迁移，但不阻止迁移 |

因此最终判断：

```text
VLLM_ASCEND_ENABLE_FLASHCOMM1 适合迁移到 Config。
```

但它不是“无脑替换一行”的迁移，需要同步处理缓存、旧 env fallback、FlashComm2 warning、测试。

---

## 8. 推荐迁移方案

### 8.1 `AscendConfig` 新增字段

建议在 `AscendConfig.__init__()` 中新增：

```python
self.enable_flashcomm1 = additional_config.get("enable_flashcomm1", None)
```

这里建议默认用 `None` 而不是 `False`，用于区分：

- 用户没有在 Config 中显式设置。
- 用户显式设置为 false。

这样可以实现兼容旧 env 的优先级。

### 8.2 推荐优先级

推荐优先级：

```text
additional_config.enable_flashcomm1
  > VLLM_ASCEND_ENABLE_FLASHCOMM1
  > VLLM_ASCEND_ENABLE_FLASHCOMM
  > False
```

伪代码：

```python
config_value = get_ascend_config().enable_flashcomm1
if config_value is not None:
    _ENABLE_SP = bool(config_value)
else:
    _ENABLE_SP = (
        envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
        or bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0")))
    )
```

如果项目希望一次性彻底移除 env，也可以直接：

```python
self.enable_flashcomm1 = additional_config.get("enable_flashcomm1", False)
```

然后：

```python
_ENABLE_SP = get_ascend_config().enable_flashcomm1
```

但这样会破坏已有启动脚本兼容性。

### 8.3 修改 `enable_sp()`

迁移后建议形式：

```python
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    ...
    additional_config = getattr(vllm_config, "additional_config", None) if vllm_config is not None else None
    refresh = additional_config.get("refresh", False) if additional_config else False

    if _ENABLE_SP is None or refresh:
        try:
            config_value = get_ascend_config().enable_flashcomm1
        except RuntimeError:
            config_value = None

        if config_value is not None:
            _ENABLE_SP = bool(config_value)
        else:
            _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1 or bool(
                int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0"))
            )

        if not _ENABLE_SP and enable_shared_expert_dp:
            _ENABLE_SP = True
            logger.info("shared_expert_dp requires enable_sp = True. has set enable_sp to True")

    return _ENABLE_SP
```

如果确认 `enable_sp()` 总是在 `AscendConfig` 初始化后调用，也可以不加 `try/except RuntimeError`。但从当前 `AscendConfig.__init__()` 内部会调用 `enable_sp(vllm_config=...)` 看，函数最好继续支持传入 `vllm_config` 并避免强依赖 `get_ascend_config()` 已完成。

更稳的方案是直接从传入的 `vllm_config.additional_config` 读：

```python
config_value = None
if additional_config is not None and "enable_flashcomm1" in additional_config:
    config_value = additional_config["enable_flashcomm1"]
elif get_ascend_config() 可用:
    config_value = get_ascend_config().enable_flashcomm1
```

因为 `AscendConfig.__init__()` 调 `enable_sp(vllm_config=vllm_config)` 时，`get_ascend_config()` 还没有完成缓存。

### 8.4 修改 FlashComm2 warning

当前：

```python
if not envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1:
    logger.warning_once(...)
```

迁移后：

```python
if not ascend_config.enable_flashcomm1:
    logger.warning_once(...)
```

如果使用 `None` fallback 设计，则要用统一 helper 判断，而不是直接读字段：

```python
if not flashcomm1_config_enabled(ascend_config):
    logger.warning_once(...)
```

### 8.5 更新 envs.py 注释

可改为：

```python
# DEPRECATED: VLLM_ASCEND_ENABLE_FLASHCOMM1 env var is removed.
# Use --additional-config '{"enable_flashcomm1": true}'.
#     "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
```

如果保留兼容期，则文案建议写：

```python
# DEPRECATED: VLLM_ASCEND_ENABLE_FLASHCOMM1 env var will be removed.
# Use --additional-config '{"enable_flashcomm1": true}'.
```

---

## 9. 测试建议

迁移后建议至少覆盖以下场景。

### 9.1 `enable_sp()` 配置优先级

测试：

1. 未设置 Config，env=0，返回 false。
2. 未设置 Config，`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`，返回 true。
3. 未设置 Config，`VLLM_ASCEND_ENABLE_FLASHCOMM=1`，返回 true。
4. Config 设置 `enable_flashcomm1=true`，env=0，返回 true。
5. Config 设置 `enable_flashcomm1=false`，env=1，返回 false。
6. `additional_config.refresh=true` 时，缓存能重新计算。

### 9.2 `enable_shared_expert_dp` 自动开启

当前逻辑中：

```python
if not _ENABLE_SP and enable_shared_expert_dp:
    _ENABLE_SP = True
```

迁移后应保持行为一致。

### 9.3 PCP + FlashComm1 的 token 对齐

覆盖：

```python
if vllm_config.parallel_config.prefill_context_parallel_size > 1 and enable_sp(vllm_config=vllm_config):
    max_num_batched_tokens 调整为 tp_size * pcp_size 的倍数
```

### 9.4 FlashComm2 warning

覆盖：

- `enable_flashcomm2_parallel_size > 0` 且 `enable_flashcomm1=false` 时打印 warning。
- `enable_flashcomm2_parallel_size > 0` 且 `enable_flashcomm1=true` 时不打印 warning。

### 9.5 算子选择

覆盖：

- `enable_flashcomm1=true` 时，符合 prefix 的 column op 返回 `SequenceColumnParallelOp`。
- `enable_flashcomm1=true` 时，符合 prefix 的 row op 返回 `SequenceRowParallelOp`。
- `matmul_allreduce_enable=true` 或 `flashcomm2_enable=true` 时，确认优先级仍高于 FlashComm1。

---

## 10. 与其他变量的分类对比

| 变量 | 当前角色 | 是否适合迁移 | 说明 |
|---|---|---:|---|
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | Row Parallel 算子选择 | 是 | 已迁移为 `enable_matmul_allreduce`。 |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | FlashComm2 配置 | 是 | 已迁移为 `enable_flashcomm2_parallel_size`。 |
| `VLLM_ASCEND_ENABLE_NZ` | 权重格式转换策略 | 是 | 已迁移为 `weight_nz_mode`。 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | FlashComm1 / SP 路径选择 | 是 | 不是 early patch gate，但需处理缓存和兼容 env。 |
| `DYNAMIC_EPLB` | early executor patch gate + EPLB 启动辅助 | 不能简单迁移 | 业务配置已在 `eplb_config`，但 early patch gate 仍依赖 env。 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | early scheduler/engine monkey patch gate | 否 | Config 初始化太晚，不能简单迁移。 |

---

## 11. 最终判断

`VLLM_ASCEND_ENABLE_FLASHCOMM1` 从设计和代码时序上都适合迁移到 Config。

推荐结论表述：

```text
VLLM_ASCEND_ENABLE_FLASHCOMM1 不属于 import-time patch gate。它主要控制 FlashComm1 / sequence parallel 的运行时路径选择，理论上适合迁移到 AscendConfig.additional_config，例如 enable_flashcomm1。但迁移时需要同步处理 enable_sp() 的全局缓存、VLLM_ASCEND_ENABLE_FLASHCOMM 旧变量兼容，以及 FlashComm2 校验中的 warning 读取点。
```

建议迁移目标：

```bash
--additional-config '{"enable_flashcomm1": true}'
```

推荐保留一个兼容期：

```text
additional_config.enable_flashcomm1 > VLLM_ASCEND_ENABLE_FLASHCOMM1 > VLLM_ASCEND_ENABLE_FLASHCOMM > False
```

等文档和测试都更新后，再考虑彻底移除旧环境变量。
