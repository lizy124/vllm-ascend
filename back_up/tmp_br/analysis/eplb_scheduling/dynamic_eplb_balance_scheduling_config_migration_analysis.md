# DYNAMIC_EPLB 与 VLLM_ASCEND_BALANCE_SCHEDULING 是否适合迁移到 Config 的验证分析

## 1. 结论先行

基于本地仓库 `D:\lzy\code\for_env\vllm-ascend` 的 `ascend_config` 分支，并对照 `main` 分支，结论如下：

| 变量 | 是否适合像普通环境变量一样迁移到 `AscendConfig.additional_config` | 结论 |
|---|---:|---|
| `VLLM_ASCEND_BALANCE_SCHEDULING` | 否 | 当前它控制 import 阶段是否加载 monkey patch，Config 初始化发生得太晚，不能简单迁移。 |
| `DYNAMIC_EPLB` | 不能简单迁移 | Dynamic EPLB 的功能配置已经在 `eplb_config` 中，但 `DYNAMIC_EPLB` 仍承担 early patch gate 角色，不能直接删除或只改成 Config。 |

更精确地说，这两个变量不属于普通“运行时配置开关”，而属于 **bootstrap / import-time patch gate**：它们在 `AscendConfig` 初始化之前决定是否修改 vLLM 的核心类或进程模型。

因此，“这类 import 阶段生效的环境变量不适合迁移到 Config”这个说法，对当前代码是成立的。但对 `DYNAMIC_EPLB` 要补充一句：它的业务参数已经部分进入 Config，只是早期 patch 开关仍依赖环境变量。

---

## 2. 背景：为什么要验证这两个变量

`ascend_config` 分支正在把一批 vLLM Ascend 专属环境变量迁移到 `--additional-config`，例如：

```bash
VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE      -> --additional-config '{"enable_matmul_allreduce": true}'
VLLM_ASCEND_ENABLE_NZ                    -> --additional-config '{"weight_nz_mode": 1}'
VLLM_ASCEND_ENABLE_FUSED_MC2             -> --additional-config '{"enable_fused_mc2": 1}'
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE     -> --additional-config '{"enable_flashcomm2_parallel_size": 2}'
HCCL_SO_PATH                             -> --additional-config '{"hccl_so_path": "/path/to/libhccl.so"}'
```

这些变量的共同特点是：它们主要在模型/worker/算子初始化或运行阶段读取，此时 `AscendConfig` 已经初始化，可以安全通过 `get_ascend_config()` 读取。

但是 `DYNAMIC_EPLB` 和 `VLLM_ASCEND_BALANCE_SCHEDULING` 的特点不同：它们被用于决定是否在 import 阶段加载 patch 模块，而 patch 模块会直接替换 vLLM 的类或方法。

本分析的核心问题是：

> 这两个变量是否也能像 `enable_matmul_allreduce` 一样迁移到 `AscendConfig.additional_config`？

---

## 3. vLLM Ascend 当前初始化时序

### 3.1 global patch 入口早于 Config 初始化

`NPUPlatform.pre_register_and_update()` 会在平台预注册阶段应用全局 patch：

```python
# vllm_ascend/platform.py:134
@classmethod
def pre_register_and_update(cls, parser: FlexibleArgumentParser | None = None) -> None:
    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
```

`adapt_patch(is_global_patch=True)` 的行为是 import `vllm_ascend.patch.platform`：

```python
# vllm_ascend/utils.py:412
def adapt_patch(is_global_patch: bool = False):
    if is_global_patch:
        from vllm_ascend.patch import platform  # noqa: F401
    else:
        from vllm_ascend.patch import worker  # noqa: F401
```

也就是说，`vllm_ascend.patch.platform.__init__` 的模块顶层代码会在此时执行。

### 3.2 `AscendConfig` 初始化发生在后面

`AscendConfig` 是在 `NPUPlatform.check_and_update_config()` 中初始化的：

```python
# vllm_ascend/platform.py:273
@classmethod
def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
    ...
    ascend_config = init_ascend_config(vllm_config)
```

对应调用位置：

```python
# vllm_ascend/platform.py:281
cls._fix_incompatible_config(vllm_config)

# vllm_ascend/platform.py:284
ascend_config = init_ascend_config(vllm_config)
```

如果在这之前读取 `get_ascend_config()`，会直接失败：

```python
# vllm_ascend/ascend_config.py:618
def get_ascend_config():
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is None or not _is_ascend_config_initialized(_ASCEND_CONFIG):
        raise RuntimeError("Ascend config is not initialized. Please call init_ascend_config first.")
    return _ASCEND_CONFIG
```

### 3.3 时序结论

当前顺序是：

```text
平台插件加载 / pre_register_and_update
  -> adapt_patch(is_global_patch=True)
    -> import vllm_ascend.patch.platform
      -> 读取 DYNAMIC_EPLB / VLLM_ASCEND_BALANCE_SCHEDULING
      -> 决定是否 import 对应 patch 模块
        -> monkey-patch vLLM 类/方法

之后才进入：
check_and_update_config(vllm_config)
  -> init_ascend_config(vllm_config)
  -> get_ascend_config() 才可用
```

因此，`vllm_ascend.patch.platform.__init__` 中不能直接依赖 `AscendConfig`，因为此时 Config 尚未初始化。

---

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

---

## 5. `DYNAMIC_EPLB` 详细分析

### 5.1 变量定义

```python
# vllm_ascend/envs.py:107
# Whether to anbale dynamic EPLB
"DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
```

注意这里返回的是字符串，例如 `"true"`、`"false"`、`"1"`，不是 bool。

### 5.2 import 阶段读取点

```python
# vllm_ascend/patch/platform/__init__.py:36
if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    import vllm_ascend.patch.platform.patch_multiproc_executor  # noqa
```

这个读取点同样发生在 `AscendConfig` 初始化之前。

### 5.3 import 后加载的 patch 做了什么

`patch_multiproc_executor.py` 的核心行为是替换 vLLM 的 `MultiprocExecutor`：

```python
# vllm_ascend/patch/platform/patch_multiproc_executor.py:211
vllm.v1.executor.multiproc_executor.MultiprocExecutor = AscendMultiprocExecutor
```

`AscendMultiprocExecutor` 里最关键的差异是创建 worker 进程时设置：

```python
# vllm_ascend/patch/platform/patch_multiproc_executor.py:195
proc = context.Process(
    target=WorkerProc.worker_main,
    kwargs=process_kwargs,
    name=f"VllmWorker-{rank}",
    daemon=False,
)
```

也就是把 worker 进程从 daemon 模式改成非 daemon。

### 5.4 为什么 Dynamic EPLB 需要这个 patch

patch 文档里写得很明确：

```text
# vllm_ascend/patch/__init__.py:57
File: platform/patch_multiproc_executor.py

# vllm_ascend/patch/__init__.py:61
vLLM create child process with daemon=True, which doesn't work with EPLB case,
since EPLB will create a new process which is not allowed by daemon=True.

# vllm_ascend/patch/__init__.py:64
Set daemon=False in MultiprocExecutor.
```

Dynamic EPLB 在 worker/model runner 中确实会创建额外进程：

```python
# vllm_ascend/worker/model_runner_v1.py:395
self.dynamic_eplb = eplb_config.dynamic_eplb

# vllm_ascend/worker/model_runner_v1.py:398
if self.dynamic_eplb:
    ...

# vllm_ascend/worker/model_runner_v1.py:404
self.eplb_process = EplbProcess(shared_dict=self.shared_dict, policy_type=self.policy_type, enable_d2d=True)

# vllm_ascend/worker/model_runner_v1.py:405
self.process = self.eplb_process._launch_process()
```

如果 worker 仍是 daemon 进程，Python multiprocessing 不允许 daemon 进程再创建子进程，因此 Dynamic EPLB 需要提前修改 executor 的进程创建方式。

### 5.5 Dynamic EPLB 的业务配置已经在 Config 中

和 `VLLM_ASCEND_BALANCE_SCHEDULING` 不同，Dynamic EPLB 的业务配置已经有 `EplbConfig`：

```python
# vllm_ascend/ascend_config.py:47
eplb_config = additional_config.get("eplb_config", {})
self.eplb_config = EplbConfig(eplb_config)
```

默认字段包括：

```python
# vllm_ascend/ascend_config.py:522
_defaults = {
    "dynamic_eplb": False,
    "expert_map_path": None,
    "expert_heat_collection_interval": 400,
    "algorithm_execution_interval": 30,
    "expert_map_record_path": None,
    "num_redundant_experts": 0,
    "eplb_policy_type": 1,
}
```

运行时大量逻辑已经读取 Config：

```python
# vllm_ascend/distributed/parallel_state.py:98
if get_ascend_config().eplb_config.dynamic_eplb:
    ...
```

```python
# vllm_ascend/worker/model_runner_v1.py:395
self.dynamic_eplb = eplb_config.dynamic_eplb
```

```python
# vllm_ascend/ops/fused_moe/fused_moe.py:361
eplb_config = ascend_config.eplb_config
```

所以 Dynamic EPLB 已经不是完全依赖 `DYNAMIC_EPLB` 环境变量。

### 5.6 但当前 Config 仍反向要求 env 存在

`EplbConfig` 校验中有如下逻辑：

```python
# vllm_ascend/ascend_config.py:570
if self.config["dynamic_eplb"]:
    assert (
        os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1")
        or os.getenv("EXPERT_MAP_RECORD", "false") == "true"
    ), "The environment variable DYNAMIC_EPLB or EXPERT_MAP_RECORD of the EPLB must be set to true."
```

这说明当前设计要求：

```text
eplb_config.dynamic_eplb = true
必须同时有：
DYNAMIC_EPLB=true/1 或 EXPERT_MAP_RECORD=true
```

它不是单纯重复校验，而是在保证 early patch gate 被触发：如果只设置 Config，不设置 env，`patch_multiproc_executor` 不会被 import，后续 EPLB 创建子进程可能失败。

### 5.7 `EXPERT_MAP_RECORD` 的作用

`patch.platform.__init__` 里不仅检查 `DYNAMIC_EPLB`，还检查 `EXPERT_MAP_RECORD`：

```python
# vllm_ascend/patch/platform/__init__.py:36
if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    import vllm_ascend.patch.platform.patch_multiproc_executor  # noqa
```

`EplbConfig` 里如果设置了 `expert_map_record_path`，会自动把 `dynamic_eplb` 置为 true：

```python
# vllm_ascend/ascend_config.py:557
if self.expert_map_record_path is not None:
    self.config["dynamic_eplb"] = True
```

这类“记录 expert map”场景也需要 executor 非 daemon patch，因此 `EXPERT_MAP_RECORD` 也被纳入 early patch gate。

### 5.8 如果简单迁移 `DYNAMIC_EPLB` 到 Config 会发生什么

如果只保留：

```bash
--additional-config '{"eplb_config": {"dynamic_eplb": true}}'
```

但不设置：

```bash
DYNAMIC_EPLB=true
```

当前流程会变成：

```text
pre_register_and_update
  -> import vllm_ascend.patch.platform
    -> os.getenv("DYNAMIC_EPLB") 为 false
    -> 不加载 patch_multiproc_executor

check_and_update_config
  -> init_ascend_config
    -> eplb_config.dynamic_eplb = true
    -> 但 executor patch 已错过加载时机

worker/model runner
  -> Dynamic EPLB 尝试启动 EPLB 子进程
  -> 如果 worker 仍是 daemon 进程，可能失败
```

这就是为什么当前代码在 `EplbConfig` 中强制要求 env 同步存在。

### 5.9 对 `DYNAMIC_EPLB` 的结论

`DYNAMIC_EPLB` 的情况比 `VLLM_ASCEND_BALANCE_SCHEDULING` 更复杂：

1. Dynamic EPLB 的业务配置已经适合 Config，并且已经在 `eplb_config` 中实现。
2. 但 `DYNAMIC_EPLB` 环境变量当前仍承担 early patch gate 的职责。
3. 因此不能简单说“Dynamic EPLB 不适合 Config”，更准确说法是：
   - `eplb_config.dynamic_eplb` 适合 Config，并且已经在 Config 中。
   - `DYNAMIC_EPLB` 作为 bootstrap patch gate，当前不适合直接迁移或删除。

---

## 6. 与普通可迁移环境变量的对比

以 `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` 为例，它在 `ascend_config` 分支中已经迁移：

```python
# vllm_ascend/envs.py:70
# DEPRECATED: VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE env var is removed.
# Use --additional-config '{"enable_matmul_allreduce": true}'.
```

对应 Config 字段：

```python
# vllm_ascend/ascend_config.py:124
self.enable_matmul_allreduce = additional_config.get("enable_matmul_allreduce", False)
```

运行时读取：

```python
# vllm_ascend/utils.py:810
def matmul_allreduce_enable() -> bool:
    return get_ascend_config().enable_matmul_allreduce
```

这类变量可以迁移，是因为它的读取发生在模型/算子初始化过程中。此时 `AscendConfig` 已经初始化，不涉及 import-time monkey patch。

对比：

| 类型 | 示例 | 是否适合普通 Config 迁移 | 原因 |
|---|---|---:|---|
| 运行时算子选择 | `enable_matmul_allreduce` | 是 | 读取发生在模型/算子初始化阶段，Config 已可用。 |
| 运行时格式/优化策略 | `weight_nz_mode`、`enable_mlapo`、`enable_fused_mc2` | 是 | 读取点大多在 worker、quant、MoE、attention 初始化之后。 |
| 动态 EPLB 业务参数 | `eplb_config.dynamic_eplb` | 是 | 后续 EPLB 逻辑读 Config。 |
| early executor patch gate | `DYNAMIC_EPLB` | 不能简单迁移 | 用于 Config 初始化前决定是否替换 `MultiprocExecutor`。 |
| early scheduler patch gate | `VLLM_ASCEND_BALANCE_SCHEDULING` | 不能简单迁移 | 用于 Config 初始化前决定是否替换 `Scheduler` 和 `EngineCoreProc.run_engine_core`。 |

---

## 7. main 与 ascend_config 分支对照

已对照 `main` 与 `ascend_config` 分支，两个分支在这两个关键 early patch 判断上保持一致：

```python
# vllm_ascend/patch/platform/__init__.py:36
if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    import vllm_ascend.patch.platform.patch_multiproc_executor  # noqa

# vllm_ascend/patch/platform/__init__.py:39
if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
```

这说明 `ascend_config` 分支虽然迁移了许多环境变量，但刻意保留了这两个 import-time gate。

这也支持一个判断：PR/分支作者大概率知道这两个变量不属于普通 Config 迁移范围，因此没有把它们放进 `AscendConfig` 的普通字段列表。

---

## 8. 是否能“设计上”迁移？

可以，但不是简单替换读取点，需要重构启用方式。

### 8.1 `VLLM_ASCEND_BALANCE_SCHEDULING` 的可迁移方案

当前实现依赖 monkey patch：

```python
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
EngineCoreProc.run_engine_core = run_engine_core
```

如果想迁移到 Config，至少需要满足一个条件：

```text
在 vLLM 创建 Scheduler / EngineCoreProc 之前，能够基于 vllm_config.additional_config 决定使用哪个类或哪个入口函数。
```

可能方案：

1. 上游 vLLM 提供 scheduler class / engine core class 的正式配置入口。
2. vLLM Ascend 在 `check_and_update_config()` 中设置类似 `scheduler_config.scheduler_cls` 的字段，让 vLLM 后续按配置创建调度器。
3. 将 `BalanceDPEngineCoreProc` 的逻辑改造成可配置扩展点，而不是 import-time 替换 `EngineCoreProc.run_engine_core`。

当前代码中已经有类似 scheduler class 配置的使用，例如：

```python
# vllm_ascend/platform.py:498
if ascend_config.SLO_limits_for_dynamic_batch != -1:
    vllm_config.scheduler_config.scheduler_cls = (
        "vllm_ascend.core.scheduler_dynamic_batch.SchedulerDynamicBatch"
    )
```

但 balance scheduling 还不只是 scheduler class，它还 patch 了 `EngineCoreProc.run_engine_core`，因此迁移复杂度更高。

### 8.2 `DYNAMIC_EPLB` 的可迁移方案

Dynamic EPLB 的业务配置已经在 Config 中，剩下的问题是 early executor patch。

可选方案：

1. **无条件加载 `patch_multiproc_executor`**
   - 优点：不再需要 `DYNAMIC_EPLB` 作为 patch gate。
   - 缺点：所有场景都会使用 `AscendMultiprocExecutor` 和 `daemon=False`，需要验证是否有副作用。

2. **在更晚阶段按 Config patch executor**
   - 前提：必须保证 patch 发生在 vLLM 实例化 `MultiprocExecutor` 之前。
   - 如果 `check_and_update_config()` 到 executor 创建之间有稳定窗口，可以在这里做。
   - 需要非常明确 vLLM 上游初始化顺序，否则容易产生竞态或版本兼容问题。

3. **上游提供 worker process daemon 配置**
   - 例如允许平台通过 Config 指定 worker process 是否 daemon。
   - 这样 Dynamic EPLB 就不需要 monkey patch executor 类。

在没有这些重构前，`DYNAMIC_EPLB` 不能简单删除。

---

## 9. 风险分析

### 9.1 如果错误迁移 `VLLM_ASCEND_BALANCE_SCHEDULING`

可能结果：

1. 用户设置了 `--additional-config '{"enable_balance_scheduling": true}'`。
2. `patch_balance_schedule` 没有在 import 阶段加载。
3. vLLM 仍使用原始 `Scheduler` 和原始 `EngineCoreProc.run_engine_core`。
4. 配置看似生效，但功能实际未启用。
5. 如果只保留后续校验，还可能出现“校验认为启用了，但实际 patch 没启用”的不一致。

这是隐蔽且危险的行为错误。

### 9.2 如果错误迁移 `DYNAMIC_EPLB`

可能结果：

1. 用户只设置 `--additional-config '{"eplb_config": {"dynamic_eplb": true}}'`。
2. `patch_multiproc_executor` 没有加载。
3. worker 进程仍可能是 daemon 模式。
4. Dynamic EPLB 在 worker 内尝试启动 EPLB 子进程。
5. multiprocessing 报错，或 EPLB 子进程无法正常启动。

这会导致功能启动失败。

### 9.3 当前设计中的一致性风险

当前 Dynamic EPLB 有一点设计不够优雅：

```text
eplb_config.dynamic_eplb 是 Config
DYNAMIC_EPLB 是 env
两者必须同时满足某些条件
```

这会带来用户体验问题：用户可能以为设置 Config 就足够，但实际上还要设置 env。当前代码通过 `EplbConfig._validate_config()` 抛错来避免 silent failure，这个校验是必要的。

---

## 10. 推荐结论与处理建议

### 10.1 对当前 PR/迁移任务的建议

建议不要把以下两个变量纳入普通 Config 迁移：

```text
DYNAMIC_EPLB
VLLM_ASCEND_BALANCE_SCHEDULING
```

它们应该在环境变量迁移分析中单独归类为：

```text
import-time / bootstrap patch gate，不适合当前阶段迁移到 AscendConfig。
```

### 10.2 建议保留的注释说明

可以在 `envs.py` 或迁移说明中明确写：

```text
DYNAMIC_EPLB and VLLM_ASCEND_BALANCE_SCHEDULING are intentionally kept as environment variables because they gate import-time monkey patches before AscendConfig is initialized.
```

中文说明：

```text
DYNAMIC_EPLB 和 VLLM_ASCEND_BALANCE_SCHEDULING 当前用于 Config 初始化前的全局 monkey patch 加载判断，不能像普通运行时配置一样迁移到 additional_config。
```

### 10.3 对 `DYNAMIC_EPLB` 的更细建议

文档中不要简单写“DYNAMIC_EPLB 不适合 Config”，而应写成：

```text
Dynamic EPLB 的业务参数已经通过 additional_config.eplb_config 管理；但 DYNAMIC_EPLB 作为 early executor patch gate 当前仍需保留。
```

推荐用户仍然同时设置：

```bash
export DYNAMIC_EPLB=true
vllm serve ... \
  --additional-config '{"eplb_config": {"dynamic_eplb": true, "num_redundant_experts": 1}}'
```

具体 EPLB 参数以项目文档/场景为准。

### 10.4 对 `VLLM_ASCEND_BALANCE_SCHEDULING` 的更细建议

如果继续保留当前实现，用户应继续使用 env：

```bash
export VLLM_ASCEND_BALANCE_SCHEDULING=1
```

在文档中强调：

1. 该变量只支持 PD-mixed 模式。
2. 不支持 PD-disaggregated 的 `kv_producer` / `kv_consumer` 模式。
3. 与 `profiling_chunk_config` 互斥。
4. 它不是普通运行时调度参数，而是启用 scheduler/engine core monkey patch。

---

## 11. 最终判断

严格按当前代码验证：

```text
VLLM_ASCEND_BALANCE_SCHEDULING 不适合当前迁移到 Config。
DYNAMIC_EPLB 不能简单迁移到 Config；其运行时业务配置已在 eplb_config，但 env 仍作为 early patch gate 必须保留。
```

根因是：

```text
这两个变量决定的是“是否在 Config 初始化前修改 vLLM 的核心类/进程模型”，而不是普通的“Config 初始化后读取某个开关”。
```

如果未来想彻底迁移，需要先消除 import-time monkey patch 依赖，或者把 patch 应用时机改到能安全读取 `vllm_config.additional_config` 的阶段。
