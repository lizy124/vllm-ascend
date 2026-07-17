# VLLM_ASCEND_BALANCE_SCHEDULING 迁移到 AscendConfig 实施文档

## 1. 问题分析

### 1.1 为什么不能简单迁移

`VLLM_ASCEND_BALANCE_SCHEDULING` 不是一个普通的运行时开关，它的启用发生在 Python import 阶段。

原始代码链路：

```text
环境变量 VLLM_ASCEND_BALANCE_SCHEDULING
  → patch/platform/__init__.py 的 import 阶段判断
  → 是否导入 patch_balance_schedule
  → 导入即执行 monkey patch：
      EngineCoreProc.run_engine_core = run_engine_core
      vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

关键时序问题：

1. `patch/platform/__init__.py` 在 `adapt_patch(is_global_patch=True)` 时被导入，发生在 `NPUPlatform.pre_register_and_update()` 阶段
2. `AscendConfig` 的初始化发生在 `NPUPlatform.check_and_update_config()` 中调用 `init_ascend_config(vllm_config)` 时
3. 阶段 1 早于阶段 2，因此在 import 阶段无法读取 `AscendConfig`

如果只是简单地把 `envs.VLLM_ASCEND_BALANCE_SCHEDULING` 替换成 `get_ascend_config().enable_balance_scheduling`，在 import 阶段 AscendConfig 还没初始化，会直接报错。

### 1.2 patch_balance_schedule.py 的不可逆替换

文件末尾直接执行全局替换：

```python
EngineCoreProc.run_engine_core = run_engine_core
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

一旦导入，替换就已经生效。这种设计适合 env gate（导入前判断），但不适合 config gate（导入时 config 还不存在）。

### 1.3 BalanceScheduler 的特殊性

`BalanceScheduler` 继承自 `Scheduler`，重写了 `schedule()` 方法。整个 `schedule()` 方法是 vLLM 原始 `Scheduler.schedule()` 的完整副本，仅增加了 2 行 balance 逻辑：

```python
balance_flag = max(t.item() for t in self.balance_queue) == self.max_num_running_reqs
if balance_flag:
    break
```

此外还有 `balance_gather()` 方法和 `BalanceDPEngineCoreProc` 类。

## 2. 方案选型

### 2.1 方案 A：无条件导入 patch，patch 内运行时按 config 分支

- 无条件 import `patch_balance_schedule`
- `BalanceScheduler` 和 `run_engine_core` 内部按 `vllm_config` 分流
- 未启用时 fallback 到原始实现

优点：可以从 `vllm_config` 读取 `additional_config`，用户入口完全迁到 AscendConfig
缺点：patch 全局安装，未启用场景也经过一层自定义类

### 2.2 方案 B：早期安装轻量 hook，实际启用延迟到 vllm_config

- 无条件导入轻量 hook
- hook 不直接替换 Scheduler，而是 patch Scheduler 创建点
- EngineCore 根据 `vllm_config` 动态选择 Scheduler

优点：最符合迁移目标，未启用时最大程度保持原始 Scheduler
缺点：需要梳理 vLLM EngineCore 创建 Scheduler 的路径，改动量大，版本耦合风险高

### 2.3 方案 C：check_and_update_config() 里动态 import patch

- AscendConfig 初始化后，如果 `enable_balance_scheduling=True`，再 import patch
- 删除 `patch/platform/__init__.py` 的 env gate

缺点：engine-core 子进程通过 plugin entry points 应用 global patch，如果只在主进程动态 import，子进程可能无法继承 patch

### 2.4 最终选择：方案 A + B 结合

- `run_engine_core` 采用方案 B 的运行时分流：保存原始引用，按 `vllm_config` 决定走原始路径还是 balance 路径
- `Scheduler` 采用方案 A 的 wrapper class：`BalanceScheduler` 继承 `Scheduler`，未启用时 `super().schedule()` 回退到原始逻辑
- 两者结合：早期安装 hook，运行时按 config 分流

选择理由：
1. `run_engine_core` 可以通过 `kwargs["vllm_config"]` 获取配置，运行时分流很干净
2. `Scheduler` 通过 wrapper class 方式，未启用时 `super().schedule()` 就是原始逻辑，风险可控
3. 不需要去 patch vLLM 内部 Scheduler 实例化点，改动量小，版本耦合低
4. 多进程/engine-core subprocess 仍能通过 global patch 机制获得 hook

## 3. 代码设计

### 3.1 核心辅助函数

```python
_ORIGINAL_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core
_ORIGINAL_SCHEDULER = Scheduler

def _balance_scheduling_enabled(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get("enable_balance_scheduling", False))
```

- `_ORIGINAL_RUN_ENGINE_CORE`：保存原始 `EngineCoreProc.run_engine_core`，未启用时直接调用
- `_ORIGINAL_SCHEDULER`：保存原始 `Scheduler` 类引用（供未来扩展使用）
- `_balance_scheduling_enabled()`：从 `vllm_config.additional_config` 读取 `enable_balance_scheduling`

### 3.2 BalanceScheduler 改造

```python
class BalanceScheduler(Scheduler):
    def __init__(self, vllm_config, ...):
        super().__init__(vllm_config, ...)
        self._balance_enabled = _balance_scheduling_enabled(vllm_config)
        if self._balance_enabled:
            self.balance_queue = [
                torch.tensor([0], dtype=torch.int, device="cpu")
                for _ in range(self.vllm_config.parallel_config.data_parallel_size)
            ]

    def balance_gather(self, dp_group):
        if not self._balance_enabled:
            return
        running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
        dist.all_gather(self.balance_queue, running_tensor, group=dp_group)

    def schedule(self) -> SchedulerOutput:
        if not self._balance_enabled:
            return super().schedule()
        # ... 完整的 balance schedule 逻辑 ...
```

关键设计点：
- `self._balance_enabled`：实例级标志，在 `__init__` 中根据 `vllm_config` 决定
- `balance_gather()`：未启用时直接 return，不做 any all-gather 操作
- `schedule()`：未启用时调用 `super().schedule()`，走原始 Scheduler 逻辑
- `balance_queue`：仅启用时初始化，避免不必要的内存分配

### 3.3 run_engine_core 改造

```python
def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    if not _balance_scheduling_enabled(vllm_config):
        return _ORIGINAL_RUN_ENGINE_CORE(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)
    # ... 完整的 balance run_engine_core 逻辑 ...
```

关键设计点：
- 通过 `kwargs.get("vllm_config")` 获取配置（vLLM 的 `run_engine_core` 签名保证 `vllm_config` 在 kwargs 中）
- 未启用时直接调用 `_ORIGINAL_RUN_ENGINE_CORE`，不进入 balance 逻辑
- `BalanceDPEngineCoreProc` 仅在启用时创建（因为整个函数体被跳过）

### 3.4 全局替换保持不变

```python
EngineCoreProc.run_engine_core = run_engine_core
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

文件末尾的全局替换仍然执行，但替换后的实现内部有分流逻辑。

## 4. 修改文件清单

### 4.1 vllm_ascend/ascend_config.py

新增 `enable_balance_scheduling` 字段：

```python
from vllm_ascend import envs as ascend_envs

self.enable_balance_scheduling = self._get_config_value(
    additional_config,
    "enable_balance_scheduling",
    "VLLM_ASCEND_BALANCE_SCHEDULING",
    ascend_envs.VLLM_ASCEND_BALANCE_SCHEDULING,
)
```

冲突校验从 env 迁移到 config：

```python
# 迁移前
if self.profiling_chunk_config.enabled and ascend_envs.VLLM_ASCEND_BALANCE_SCHEDULING:

# 迁移后
if self.profiling_chunk_config.enabled and self.enable_balance_scheduling:
```

优先级：`additional_config` 显式配置 > 旧环境变量 fallback > 默认值（False）

### 4.2 vllm_ascend/platform.py

PD 模式校验从 env 迁移到 config：

```python
# 迁移前
if envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING:
    ...
    raise ValueError("VLLM_ASCEND_BALANCE_SCHEDULING (balance scheduling) only supports PD-mixed mode ...")

# 迁移后
if ascend_config.enable_balance_scheduling:
    ...
    raise ValueError("enable_balance_scheduling only supports PD-mixed mode ...")
```

移除不再使用的 `import vllm_ascend.envs as envs_ascend`。

### 4.3 vllm_ascend/patch/platform/__init__.py

无条件导入 patch_balance_schedule：

```python
# 迁移前
if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule  # noqa

# 迁移后
import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
```

移除不再使用的 `from vllm_ascend import envs`。

### 4.4 vllm_ascend/patch/platform/patch_balance_schedule.py

核心改造（详见第 3 节）：
1. 保存原始引用 `_ORIGINAL_RUN_ENGINE_CORE` 和 `_ORIGINAL_SCHEDULER`
2. 新增 `_balance_scheduling_enabled()` 辅助函数
3. `BalanceScheduler.__init__` 有条件初始化 `balance_queue`
4. `BalanceScheduler.balance_gather()` 未启用时 return
5. `BalanceScheduler.schedule()` 未启用时 fallback 到 `super().schedule()`
6. `run_engine_core()` 未启用时调用 `_ORIGINAL_RUN_ENGINE_CORE`

### 4.5 vllm_ascend/envs.py

标记为 DEPRECATED（保留定义，作为 fallback）：

```python
# DEPRECATED: VLLM_ASCEND_BALANCE_SCHEDULING env var will be removed in a future release.
# Use --additional-config '{"enable_balance_scheduling": true}' instead.
"VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
```

### 4.6 vllm_ascend/patch/__init__.py

更新文档说明：

```python
#    How：
#       Set --additional-config '{"enable_balance_scheduling": true}' or
#       set environmental variable VLLM_ASCEND_BALANCE_SCHEDULING=1 (deprecated).
```

### 4.7 tests/ut/test_platform.py

测试从 mock env 迁移到 mock config：

```python
# 迁移前
mock_ascend_config.recompute_scheduler_enable = False
with (
    patch("vllm_ascend.platform.envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING", True, create=True),
    pytest.raises(ValueError, match=r"VLLM_ASCEND_BALANCE_SCHEDULING.*PD-mixed.*PD-disaggregated"),
    ...
):

# 迁移后
mock_ascend_config.recompute_scheduler_enable = False
mock_ascend_config.enable_balance_scheduling = True
with (
    pytest.raises(ValueError, match=r"enable_balance_scheduling.*PD-mixed.*PD-disaggregated"),
    ...
):
```

## 5. 运行时行为对比

### 5.1 未启用 balance scheduling

| 阶段 | 迁移前 | 迁移后 |
|------|--------|--------|
| import 阶段 | 不导入 patch_balance_schedule | 导入 patch_balance_schedule，全局替换 Scheduler 和 run_engine_core |
| Scheduler 创建 | 使用原始 Scheduler | 使用 BalanceScheduler，但 `schedule()` 内部 fallback 到 `super().schedule()` |
| run_engine_core | 使用原始实现 | 使用 wrapper，内部调用 `_ORIGINAL_RUN_ENGINE_CORE` |
| 功能行为 | 原始调度 | 与原始调度等价 |

### 5.2 启用 balance scheduling

| 阶段 | 迁移前 | 迁移后 |
|------|--------|--------|
| 用户入口 | `VLLM_ASCEND_BALANCE_SCHEDULING=1` | `--additional-config '{"enable_balance_scheduling": true}'` 或环境变量 |
| Scheduler 创建 | BalanceScheduler（balance 逻辑） | BalanceScheduler（balance 逻辑） |
| run_engine_core | BalanceDPEngineCoreProc | BalanceDPEngineCoreProc |
| 功能行为 | balance 调度 | 与迁移前等价 |

## 6. 风险评估

### 6.1 未启用场景经过 BalanceScheduler 类

风险：未启用时 `isinstance(scheduler, Scheduler)` 仍然为 True（BalanceScheduler 是 Scheduler 子类），不影响类型检查。

缓解：`schedule()` 未启用时直接 `return super().schedule()`，逻辑等价于原始 Scheduler。

### 6.2 多进程子进程的 config 上下文

风险：engine-core 子进程通过 plugin entry points 应用 global patch，此时 `vllm_config` 是否可用？

缓解：`run_engine_core` 的签名保证 `kwargs["vllm_config"]` 存在；`BalanceScheduler.__init__` 接收 `vllm_config` 参数。两个分流点都能正确获取配置。

### 6.3 vLLM 上游 Scheduler 变更

风险：`BalanceScheduler.schedule()` 是 vLLM `Scheduler.schedule()` 的完整副本，如果上游修改了 `schedule()` 逻辑，副本不会自动同步。

缓解：未启用时走 `super().schedule()`，只有启用时才走副本逻辑。如果上游 `Scheduler.schedule()` 变更，只需同步更新 balance 分支中的副本。

## 7. 用户迁移指南

### 7.1 迁移前

```bash
VLLM_ASCEND_BALANCE_SCHEDULING=1 vllm serve <model> --data-parallel-size 2
```

### 7.2 迁移后（推荐）

```bash
vllm serve <model> --data-parallel-size 2 --additional-config '{"enable_balance_scheduling": true}'
```

### 7.3 迁移后（兼容旧环境变量）

```bash
VLLM_ASCEND_BALANCE_SCHEDULING=1 vllm serve <model> --data-parallel-size 2
```

过渡期环境变量仍然有效，但会输出 deprecation 提示。优先级：`additional_config` > 环境变量 > 默认值。

### 7.4 Python API

```python
from vllm import LLM

llm = LLM(model, additional_config={"enable_balance_scheduling": True})
```
