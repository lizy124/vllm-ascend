# VLLM_ASCEND_BALANCE_SCHEDULING 迁移方案

## 背景

`VLLM_ASCEND_BALANCE_SCHEDULING` 当前不是一个普通运行时开关。它在 `vllm_ascend/patch/platform/__init__.py` 的 import 阶段决定是否加载 `patch_balance_schedule`：

```python
if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
```

`patch_balance_schedule` 导入后会立即做 monkey patch：

```python
EngineCoreProc.run_engine_core = run_engine_core
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

因此它当前的启用链路是：

```text
环境变量
  -> early import 阶段判断
  -> 是否导入 patch_balance_schedule
  -> 是否替换 Scheduler / EngineCoreProc.run_engine_core
```

`AscendConfig` 的初始化发生在 `NPUPlatform.check_and_update_config()` 中：

```python
ascend_config = init_ascend_config(vllm_config)
```

这个阶段晚于 `NPUPlatform.pre_register_and_update()` 中的全局 patch 加载。因此，如果只是新增：

```python
self.enable_balance_scheduling = additional_config.get("enable_balance_scheduling", False)
```

然后把校验改成 `get_ascend_config().enable_balance_scheduling`，功能不会真正启用，因为早期 monkey patch 已经错过。

结论：这个变量不是技术上绝对不能迁移，而是当前 import-time monkey patch 架构下不能简单迁移。迁移的核心是先解决 patch 加载时机问题。

## 迁移目标

把用户入口从环境变量：

```bash
VLLM_ASCEND_BALANCE_SCHEDULING=1
```

迁移为 `additional_config`：

```bash
vllm serve <model> --additional-config '{"enable_balance_scheduling": true}'
```

或 Python API：

```python
LLM(model, additional_config={"enable_balance_scheduling": True})
```

同时保持以下行为：

1. 启用时仍替换 vLLM v1 Scheduler 为 `BalanceScheduler`。
2. DP 场景仍使用 `BalanceDPEngineCoreProc`。
3. 仍禁止 PD-disaggregated 场景，即 `kv_role='kv_producer'` / `kv_role='kv_consumer'`。
4. 仍禁止和 `profiling_chunk_config.enabled` 同时启用。
5. 不让未启用 balance scheduling 的路径承担行为变化风险。

## 关键约束

### 1. `additional_config` 在 patch import 阶段不可用

`vllm_ascend/patch/platform/__init__.py` 被 `adapt_patch(is_global_patch=True)` 导入，发生在平台 pre-register 阶段。此时还没有完整 `VllmConfig`，也没有 `AscendConfig`。

因此 `patch/platform/__init__.py` 不能再依赖 `AscendConfig` 决定是否 import balance patch。

### 2. 当前 patch 是不可逆的全局替换

当前文件末尾直接执行：

```python
EngineCoreProc.run_engine_core = run_engine_core
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

一旦导入，替换就已经生效。这个设计适合 env gate，但不适合 config gate。

### 3. Scheduler 类替换必须早于 EngineCore 创建

是否使用 `BalanceScheduler`，最终取决于 EngineCore 初始化时引用到的 Scheduler 类。如果等 EngineCore 已经创建后才替换，就太晚。

所以迁移后仍要保证：在 EngineCore 创建 Scheduler 前，patch 已经安装，或者 EngineCore 的创建逻辑能按 `vllm_config` 动态选择 Scheduler。

## 可选方案

### 方案 A：无条件导入 patch，patch 内运行时按 config 分支

思路：

1. `patch/platform/__init__.py` 无条件 import `patch_balance_schedule`。
2. `patch_balance_schedule` 导入后仍替换 `EngineCoreProc.run_engine_core` 和 Scheduler 入口。
3. 但替换后的入口内部读取 `vllm_config.additional_config` / `AscendConfig`，只有 `enable_balance_scheduling=True` 时才走 balance 逻辑。
4. 未启用时尽量回落到原始 vLLM 行为。

优点：

- 可以从 `vllm_config` 读取 `additional_config`。
- 用户入口可以完全迁到 `AscendConfig`。
- 仍然解决了 import-time 无法读取 config 的问题，因为 import 不再需要判断开关。

缺点：

- patch 仍然全局安装，即使用户未启用 balance scheduling。
- 必须保存原始对象，否则无法干净回退。
- `Scheduler` 类替换比较敏感：如果无条件把 Scheduler 替成 wrapper，未启用场景也会经过一层自定义类，风险较高。

变体：

- 保存原始引用：

```python
_ORIGINAL_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core
_ORIGINAL_SCHEDULER = vllm.v1.core.sched.scheduler.Scheduler
```

- `run_engine_core` 内部：

```python
if not _balance_enabled(kwargs["vllm_config"]):
    return _ORIGINAL_RUN_ENGINE_CORE(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)
```

- Scheduler 替换可做成 factory/wrapper，但要确认 vLLM 创建 Scheduler 的方式是否支持动态选择。

评价：可行，但需要非常谨慎处理 Scheduler 替换，否则未启用场景也会被影响。

### 方案 B：保留早期轻量 patch，只把真正选择延迟到 `vllm_config`

思路：

1. `patch/platform/__init__.py` 无条件导入一个轻量 balance hook。
2. 这个 hook 不直接把 Scheduler 永久替换成 `BalanceScheduler`。
3. 它只 patch EngineCore 初始化/创建 Scheduler 的路径，让 Scheduler 选择变成：

```text
if vllm_config.additional_config.enable_balance_scheduling:
    scheduler_cls = BalanceScheduler
else:
    scheduler_cls = 原始 Scheduler
```

4. `run_engine_core` 也只在 `enable_balance_scheduling=True` 且 DP 场景下使用 `BalanceDPEngineCoreProc`。

优点：

- 这是最符合迁移目标的设计：patch 早安装，功能晚决策。
- 未启用时可以最大程度保持原始 Scheduler。
- `additional_config` 成为唯一控制源。

缺点：

- 需要梳理 vLLM EngineCore 是在哪里创建 Scheduler 的，改动可能比方案 A 大。
- 需要避免跟 vLLM upstream 版本强耦合。

评价：这是推荐方向。它把“是否安装 hook”和“是否启用功能”解耦：hook 可以早安装，功能由 config 晚判断。

### 方案 C：在 `check_and_update_config()` 里按 config 动态 import patch

思路：

1. `AscendConfig` 初始化后，如果 `enable_balance_scheduling=True`，再 import `patch_balance_schedule`。
2. 删除 `patch/platform/__init__.py` 的 env gate。

优点：

- 改动最小，迁移直观。
- patch 只在用户启用时导入。

缺点：

- 风险在于时机是否足够早。`check_and_update_config()` 虽然早于大多数 EngineCore 创建，但 vLLM 插件/多进程路径中是否所有进程都会先走到这里，需要验证。
- 文档中提到 engine-core 子进程也会通过 plugin entry points 应用 global patch。如果只在主进程 `check_and_update_config()` 动态 import，子进程是否能稳定继承 patch 取决于启动方式和导入顺序。
- 当前 `vllm_ascend/__init__.py` 的 `_ensure_global_patch()` 明确是为 engine-core subprocess 补全 global patches；绕开它可能破坏多进程路径。

评价：不建议作为首选。除非能通过 vLLM 启动链路验证所有 EngineCore 创建前都会执行这段动态 import，否则容易出现单进程可用、多进程失效的问题。

### 方案 D：短期双入口兼容

思路：

1. 新增 `enable_balance_scheduling` 到 `AscendConfig`。
2. 暂时保留 `VLLM_ASCEND_BALANCE_SCHEDULING` env。
3. 在早期 import 阶段仍允许 env 控制 patch。
4. 在 config 初始化后做一致性校验：如果 env 和 config 同时设置且冲突，报错。

优点：

- 兼容旧用户。
- 迁移风险低。

缺点：

- 没有真正解决“迁移到 config”的核心问题。
- 仍然需要 env 才能早期加载 patch。
- 容易形成长期双入口，配置语义变复杂。

评价：只适合作为过渡，不符合“决定迁移环境变量”的最终目标。

## 推荐方案

推荐采用方案 B：早期无条件安装轻量 hook，实际启用由 `AscendConfig.enable_balance_scheduling` 决定。

核心原则：

```text
早期 import 阶段：只安装可按 config 分流的 hook
配置初始化阶段：解析 enable_balance_scheduling 并做合法性校验
EngineCore / Scheduler 创建阶段：根据 vllm_config 决定是否走 balance 实现
```

这样可以同时满足：

- 不再依赖 env 做 early gate。
- 可以通过 `additional_config` 启用。
- 未启用时不使用 `BalanceScheduler`。
- 多进程/engine-core subprocess 仍能通过 global patch 机制获得 hook。

## 推荐实现步骤

### 步骤 1：新增 AscendConfig 字段

在 `vllm_ascend/ascend_config.py` 中新增：

```python
self.enable_balance_scheduling = additional_config.get("enable_balance_scheduling", False)
```

建议放在其他布尔型运行时开关附近。

### 步骤 2：迁移冲突校验

当前冲突校验：

```python
if self.profiling_chunk_config.enabled and ascend_envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    raise ValueError(...)
```

迁移为：

```python
if self.profiling_chunk_config.enabled and self.enable_balance_scheduling:
    raise ValueError(
        "profiling_chunk_config and balance scheduling (enable_balance_scheduling) "
        "cannot be enabled at the same time. Please disable one of them."
    )
```

并移除这里对 `ascend_envs` 的导入。

### 步骤 3：迁移 PD 模式校验

当前 `platform.py` 中：

```python
if envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING:
    ...
```

迁移为：

```python
if ascend_config.enable_balance_scheduling:
    kv_transfer_config = vllm_config.kv_transfer_config
    kv_role = getattr(kv_transfer_config, "kv_role", None)
    if kv_transfer_config is not None and kv_role != "kv_both":
        raise ValueError(
            "enable_balance_scheduling only supports PD-mixed mode "
            "(kv_role='kv_both' or no kv_transfer_config), and is not supported in "
            "PD-disaggregated mode (kv_role='kv_producer'/'kv_consumer')."
        )
```

### 步骤 4：移除 env 定义

从 `vllm_ascend/envs.py` 移除：

```python
"VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
```

也可以短期保留注释为 deprecated，但不要再作为功能判断源。

### 步骤 5：调整 patch 加载入口

当前：

```python
if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
```

推荐改成无条件导入轻量 hook：

```python
import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
```

但前提是 `patch_balance_schedule.py` 不再导入即强制启用 balance 行为，而是只安装可分流 hook。

### 步骤 6：重构 `patch_balance_schedule.py`

建议把文件结构调整为：

```python
_ORIGINAL_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core
_ORIGINAL_SCHEDULER = vllm.v1.core.sched.scheduler.Scheduler


def balance_scheduling_enabled(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get("enable_balance_scheduling", False))
```

然后把 patch 行为拆成两层：

1. 早期安装 hook。
2. hook 内根据 `vllm_config` 分流。

`run_engine_core` 的分流可以是：

```python
def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    vllm_config = kwargs["vllm_config"]
    if not balance_scheduling_enabled(vllm_config):
        return _ORIGINAL_RUN_ENGINE_CORE(
            *args,
            dp_rank=dp_rank,
            local_dp_rank=local_dp_rank,
            **kwargs,
        )
    ...  # existing balance implementation
```

Scheduler 的分流需要根据 vLLM 当前实现选择落点。优先考虑 patch Scheduler 创建点，而不是无条件替换 Scheduler 类。如果必须替换 Scheduler 类，则需要保证未启用场景完全走原始 `Scheduler` 逻辑。

一种保守写法是：

```python
class BalanceScheduler(_ORIGINAL_SCHEDULER):
    ...
```

并只在启用时让 EngineCore 使用它。不要在未启用场景把全局 Scheduler 解析成 `BalanceScheduler`。

如果 vLLM 当前只能通过 `vllm.v1.core.sched.scheduler.Scheduler` 全局变量创建 Scheduler，则可以考虑替换成一个 factory/wrapper，但这需要验证调用方是否期望它是 class，而不是函数。

### 步骤 7：测试覆盖

需要新增/调整以下测试：

1. `AscendConfig` 解析测试：
   - `additional_config={"enable_balance_scheduling": True}` 时字段为 True。
   - 默认 False。

2. 冲突测试：
   - `enable_balance_scheduling=True` 且 `profiling_chunk_config.enabled=True` 报错。

3. PD 模式校验测试：
   - `enable_balance_scheduling=True` + `kv_role='kv_producer'` 报错。
   - `enable_balance_scheduling=True` + `kv_role='kv_consumer'` 报错。
   - `enable_balance_scheduling=True` + `kv_role='kv_both'` 通过。
   - `enable_balance_scheduling=True` + `kv_transfer_config=None` 通过。

4. patch 行为测试：
   - 未启用时，`run_engine_core` 分流到原始实现。
   - 启用且 DP 场景时，使用 `BalanceDPEngineCoreProc`。
   - 启用时 Scheduler 选择为 `BalanceScheduler`。

5. env 迁移测试：
   - `VLLM_ASCEND_BALANCE_SCHEDULING=1` 不再启用功能，或者如果选择过渡兼容，则测试它会触发 deprecation warning 并映射到 config。

## 推荐提交拆分

建议拆成 3 个提交，降低 review 难度：

1. `[Refactor] Decouple balance scheduler patch loading`
   - 先把 import-time gate 改成早期 hook + config 分流。
   - 不急着删除 env。

2. `[Refactor] Migrate balance scheduling to AscendConfig`
   - 新增 `enable_balance_scheduling`。
   - 迁移 `AscendConfig` / `platform.py` 校验。
   - 更新测试。

3. `[Cleanup] Remove balance scheduling env switch`
   - 从 `envs.py` 删除 `VLLM_ASCEND_BALANCE_SCHEDULING`。
   - 更新文档和报错信息。

如果希望一次性提交，也建议在 PR 描述里按这三个逻辑块解释。

## 风险点

### 风险 1：无条件导入 patch 改变默认行为

如果 `patch_balance_schedule.py` 仍在导入时执行全局替换，那么无条件导入会让所有用户默认启用 balance scheduling，这是不可接受的。

规避：必须先把 patch 文件改成“安装 hook，但 hook 内按 config 判断”。

### 风险 2：Scheduler 替换时机不对

如果 Scheduler 已经被实例化，再替换类就无效。

规避：hook 必须在 EngineCore 创建前安装；实际 Scheduler 选择必须发生在 EngineCore 初始化或 Scheduler 创建前。

### 风险 3：EngineCore 子进程没有 config 上下文

`run_engine_core` 接收 `kwargs["vllm_config"]`，这是可以读取 `additional_config` 的关键点。迁移方案应优先在这个函数内部基于 `vllm_config` 分流。

### 风险 4：多版本 vLLM 兼容

`patch_balance_schedule.py` 直接引用 vLLM 内部类和路径，任何对 Scheduler 创建路径的进一步 patch 都可能增加版本耦合。

规避：尽量复用当前已有 patch 入口，少 patch 新路径；必要时用 `vllm_version_is()` 做版本分支。

## 最终建议

不要把 `VLLM_ASCEND_BALANCE_SCHEDULING` 直接替换成 `get_ascend_config().enable_balance_scheduling`。正确迁移路径应该是：

```text
先改 patch 架构：import-time gate -> early hook + runtime config gate
再迁移配置源：env -> AscendConfig.enable_balance_scheduling
最后删除 env：VLLM_ASCEND_BALANCE_SCHEDULING
```

推荐优先实现方案 B。它能从根上解决 import-time monkey patch 问题，同时把用户入口迁移到 `additional_config`，并尽量降低未启用场景的行为风险。
