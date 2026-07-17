## 4. `VLLM_ASCEND_BALANCE_SCHEDULING` 详细分析

### 4.1 变量定义

```python
# vllm_ascend/envs.py:120
# Whether to enable balance scheduling in the v1 scheduler.
# Platform validation: only PD-mixed mode (`kv_role='kv_both'` or no kv_transfer_config).
# Not supported in PD-disaggregated mode (`kv_producer` / `kv_consumer` only).
"VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
```

### 4.2 import 阶段读取点

```python
# vllm_ascend/patch/platform/__init__.py:39
if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
```

这个判断发生在 `adapt_patch(is_global_patch=True)` import `vllm_ascend.patch.platform` 的时候，早于 `init_ascend_config(vllm_config)`。

### 4.3 它真正控制的功能不是普通 if 分支，而是 monkey patch

`patch_balance_schedule.py` 的核心行为在文件末尾：

```python
# vllm_ascend/patch/platform/patch_balance_schedule.py:706
EngineCoreProc.run_engine_core = run_engine_core

# vllm_ascend/patch/platform/patch_balance_schedule.py:707
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

这个 patch 做了两件关键事：

1. 替换 `EngineCoreProc.run_engine_core`。
2. 将 vLLM 原始 `Scheduler` 类替换为 `BalanceScheduler`。

`BalanceScheduler` 是继承 vLLM 原始 `Scheduler` 的自定义调度器：

```python
# vllm_ascend/patch/platform/patch_balance_schedule.py:28
class BalanceScheduler(Scheduler):
    ...
```

它还定义了 DP 场景下的 `BalanceDPEngineCoreProc`：

```python
# vllm_ascend/patch/platform/patch_balance_schedule.py:607
class BalanceDPEngineCoreProc(DPEngineCoreProc):
    ...
```

并在新的 `run_engine_core()` 中使用它：

```python
# vllm_ascend/patch/platform/patch_balance_schedule.py:683
engine_core = BalanceDPEngineCoreProc(*args, **kwargs)
```

### 4.4 `check_and_update_config()` 中的读取只是校验，不是功能启用

后续在 `platform.py` 中也读取了这个环境变量：

```python
# vllm_ascend/platform.py:473
if envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING:
    kv_transfer_config = vllm_config.kv_transfer_config
    kv_role = getattr(kv_transfer_config, "kv_role", None)
    if kv_transfer_config is not None and kv_role != "kv_both":
        raise ValueError(...)
```

这里的作用是校验 balance scheduling 只支持 PD-mixed，不支持 PD-disaggregated。这个阶段已经可以访问 `vllm_config`，但此时如果前面的 import-time patch 没有加载，调度器替换已经错过了。

### 4.5 它还和 `profiling_chunk_config` 冲突

`AscendConfig` 初始化阶段会检查它和 `profiling_chunk_config` 是否冲突：

```python
# vllm_ascend/ascend_config.py:74
from vllm_ascend import envs as ascend_envs

# vllm_ascend/ascend_config.py:76
if self.profiling_chunk_config.enabled and ascend_envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    raise ValueError(...)
```

这说明当前设计里 `VLLM_ASCEND_BALANCE_SCHEDULING` 已经被认为是 Config 之外的全局开关。

### 4.6 如果简单迁移到 Config 会发生什么

假设只做如下迁移：

```python
self.enable_balance_scheduling = additional_config.get("enable_balance_scheduling", False)
```

并把后续校验改成：

```python
if get_ascend_config().enable_balance_scheduling:
    ...
```

这仍然无法启用 balance scheduling，因为真正启用功能的是：

```python
import vllm_ascend.patch.platform.patch_balance_schedule
```

而这个 import 发生在 `AscendConfig` 初始化之前。如果 import-time gate 不读取 env，就没有办法知道用户是否希望加载 patch。

### 4.7 对 `VLLM_ASCEND_BALANCE_SCHEDULING` 的结论

`VLLM_ASCEND_BALANCE_SCHEDULING` 当前不适合直接迁移到 `AscendConfig`。

原因不是“它不能被 Config 表达”，而是当前实现依赖 import-time monkey patch：

```text
是否启用 balance scheduling
  = 是否在 early import 阶段加载 patch_balance_schedule
  = 是否替换 vLLM Scheduler / EngineCoreProc
```

而 `AscendConfig` 初始化太晚，无法作为这个 early import 判断条件。