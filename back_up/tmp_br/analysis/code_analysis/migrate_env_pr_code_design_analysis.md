# migrate_env 分支 PR 代码与设计意图讲解

## 1. PR 目标

本 PR 的目标是把 vllm-ascend 中一批原本只能通过环境变量控制的功能，迁移到 `AscendConfig` / `additional_config` 配置体系中。

迁移后用户可以通过：

```bash
--additional-config '{"enable_matmul_allreduce": true}'
```

或 Python API 中的：

```python
LLM(..., additional_config={"enable_matmul_allreduce": True})
```

来配置这些 Ascend 侧开关。

同时，为了兼容已有部署脚本，本 PR 没有直接删除原环境变量，而是保留一段过渡期 fallback。

最终优先级是：

```text
additional_config 显式配置 > 旧环境变量 fallback > 默认值
```

## 2. 迁移范围

本 PR 迁移以下 10 个环境变量：

| 旧环境变量 | 新 config 字段 | 类型 | 默认值 |
|---|---|---|---|
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `enable_balance_scheduling` | bool | false |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `enable_flashcomm1` | bool | false |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `enable_matmul_allreduce` | bool | false |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | `enable_flashcomm2_parallel_size` | int | 0 |
| `MSMONITOR_USE_DAEMON` | `msmonitor_use_daemon` | bool | false |
| `VLLM_ASCEND_ENABLE_MLAPO` | `enable_mlapo` | bool | true |
| `VLLM_ASCEND_ENABLE_NZ` | `weight_nz_mode` | int | 1 |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | `enable_context_parallel` | bool | false |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `enable_fused_mc2` | int | 0 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `enable_transpose_kv_cache_by_block` | bool | true |

明确不迁移的变量：

| 环境变量 | 不迁移原因 |
|---|---|
| `HCCL_SO_PATH` | 运行时库路径类配置，应继续作为环境变量 |
| `VLLM_VERSION` | 构建/版本选择类配置，不适合放入运行时 `AscendConfig` |

## 3. 设计原则

### 3.1 Config 优先，env 兼容

新逻辑不是简单删除 env，而是在 `AscendConfig` 初始化时把两类来源统一起来。

优先级：

```text
1. 用户显式传入 additional_config 字段
2. 用户没有传 config 时，读取旧环境变量
3. 环境变量也没有设置时，使用 envs.py 中已有默认值
```

这样可以保证：

```text
新用户可以使用 --additional-config
老用户原有启动脚本不需要立刻修改
后续版本可以逐步移除 deprecated env
```

### 3.2 避免在业务路径中散落 env 读取

迁移前，不同模块直接读取 `vllm_ascend.envs`：

```python
envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2
envs_ascend.VLLM_ASCEND_ENABLE_NZ
envs_ascend.MSMONITOR_USE_DAEMON
```

迁移后，业务逻辑统一读取：

```python
get_ascend_config().enable_fused_mc2
get_ascend_config().weight_nz_mode
get_ascend_config().msmonitor_use_daemon
```

这样配置来源集中在 `AscendConfig`，下游只关心最终配置值。

### 3.3 兼容初始化时序

部分路径在 `AscendConfig` 初始化过程中就会调用某些 helper，例如 `enable_sp()`。如果简单改成只读 `get_ascend_config()`，可能遇到 singleton 还未初始化的问题。

因此 FlashComm1 的迁移做了特殊处理：

```text
enable_sp(vllm_config) 优先读传入 vllm_config.additional_config
否则尝试读 get_ascend_config().enable_flashcomm1
如果 AscendConfig 未初始化，则 fallback 到旧 env
```

这样可以兼容：

```text
AscendConfig.__init__ 内部调用 enable_sp()
worker / ops 中无 current config 的 enable_sp() 调用
旧环境变量启动方式
```

## 4. 核心文件改动说明

## 4.1 vllm_ascend/ascend_config.py

这是本 PR 的核心改动文件。

### 4.1.1 新增 _get_config_value()

新增 helper：

```python
@staticmethod
def _get_config_value(additional_config: dict[str, Any], config_key: str, env_key: str, env_value: Any) -> Any:
    if config_key in additional_config:
        value = additional_config[config_key]
        logger.info_once(f"AscendConfig.{config_key} is set from additional_config with value {value}.")
        return value
    logger.info_once(
        f"AscendConfig.{config_key} falls back to environment variable {env_key} with value {env_value}."
    )
    return env_value
```

设计意图：

```text
把“config 优先、env fallback”的逻辑收敛到一个函数里，避免每个字段重复写 additional_config.get(..., env)。
```

同时它会打印 `info_once`，方便用户和开发者确认最终值来自 config 还是 env。

### 4.1.2 新增迁移字段

在 `AscendConfig.__init__()` 中新增这些字段：

```python
self.enable_balance_scheduling
self.enable_flashcomm1
self.enable_context_parallel
self.enable_matmul_allreduce
self.enable_fused_mc2
self.enable_mlapo
self.enable_flashcomm2_parallel_size
self.msmonitor_use_daemon
self.enable_transpose_kv_cache_by_block
self.weight_nz_mode
```

每个字段都通过 `_get_config_value()` 初始化。

设计意图：

```text
AscendConfig 成为这些迁移配置的唯一统一入口。
```

### 4.1.3 profiling_chunk_config 与 balance scheduling 冲突检查改为 config 字段

迁移前检查：

```python
if self.profiling_chunk_config.enabled and ascend_envs.VLLM_ASCEND_BALANCE_SCHEDULING:
```

迁移后检查：

```python
if self.profiling_chunk_config.enabled and self.enable_balance_scheduling:
```

设计意图：

```text
如果用户通过 --additional-config 开启 enable_balance_scheduling，也能触发同样的冲突检查。
```

### 4.1.4 clear_ascend_config() 同步清理 enable_sp 缓存

新增：

```python
from vllm_ascend.utils import clear_enable_sp
clear_enable_sp()
```

设计意图：

```text
enable_sp() 有模块级缓存 _ENABLE_SP。
AscendConfig 被 clear 后，FlashComm1/SP 的缓存也必须清掉，否则测试或 refresh 场景可能复用旧值。
```

## 4.2 vllm_ascend/envs.py

本 PR 没有删除旧环境变量，只是补充 deprecated 说明。

例如：

```python
# DEPRECATED: use additional_config.enable_flashcomm1 instead.
"VLLM_ASCEND_ENABLE_FLASHCOMM1": ...
```

以及：

```python
# DEPRECATED: VLLM_ASCEND_BALANCE_SCHEDULING env var will be removed in a future release.
# Use --additional-config '{"enable_balance_scheduling": true}' instead.
"VLLM_ASCEND_BALANCE_SCHEDULING": ...
```

设计意图：

```text
保留兼容性，但明确推荐用户迁移到 additional_config。
```

## 4.3 vllm_ascend/utils.py

这是第二个核心改动文件，主要覆盖多个被业务路径广泛调用的 helper。

### 4.3.1 新增 clear_enable_sp()

新增：

```python
def clear_enable_sp():
    global _ENABLE_SP
    _ENABLE_SP = None
    enable_dsa_cp.cache_clear()
    enable_dsa_cp_with_layer_shard.cache_clear()
    enable_dsa_cp_with_o_proj_tp.cache_clear()
    _libc_getenv.cache_clear()
```

设计意图：

```text
统一清理 FlashComm1/SP 及 DSA-CP 相关缓存，配合 clear_ascend_config() 使用。
```

### 4.3.2 _should_trans_nz() 使用 weight_nz_mode

迁移前：

```python
envs_ascend.VLLM_ASCEND_ENABLE_NZ
```

迁移后：

```python
config = get_ascend_config()
nz_mode = config.weight_nz_mode
```

语义保持：

```text
0：关闭 NZ
1：仅量化场景启用 NZ
2：BF16/FP16 等可用场景也启用 NZ
```

设计意图：

```text
NZ 行为跟随 AscendConfig，支持 --additional-config '{"weight_nz_mode": 2}'。
```

### 4.3.3 matmul_allreduce_enable() 使用 config

迁移前：

```python
return envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE
```

迁移后：

```python
return get_ascend_config().enable_matmul_allreduce
```

设计意图：

```text
所有调用 matmul_allreduce_enable() 的路径自动支持 config 方式。
```

### 4.3.4 enable_sp() 支持 enable_flashcomm1

这是 FlashComm1 迁移的关键。

迁移前主要读：

```python
envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
```

并且还兼容旧变量：

```python
VLLM_ASCEND_ENABLE_FLASHCOMM
```

迁移后逻辑：

```python
if additional_config is not None and "enable_flashcomm1" in additional_config:
    _ENABLE_SP = bool(additional_config["enable_flashcomm1"])
else:
    try:
        _ENABLE_SP = get_ascend_config().enable_flashcomm1
    except RuntimeError:
        _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
```

设计意图：

```text
1. enable_sp() 是 FlashComm1/SP 的统一入口。
2. 优先支持 additional_config.enable_flashcomm1。
3. AscendConfig 初始化完成后读取 AscendConfig.enable_flashcomm1。
4. 无 AscendConfig 上下文时 fallback 到旧 env，避免 worker / 初始化路径异常。
5. 不再重新引入旧变量 VLLM_ASCEND_ENABLE_FLASHCOMM。
```

保留 shared expert DP 的特殊逻辑：

```python
if not _ENABLE_SP and enable_shared_expert_dp:
    _ENABLE_SP = True
```

这表示 shared expert DP 需要 SP 路径时，仍会强制启用。

### 4.3.5 prefill_context_parallel_enable() 使用 config

迁移后：

```python
return get_ascend_config().enable_context_parallel
```

设计意图：

```text
context parallel 相关判断走统一 config。
```

### 4.3.6 flashcomm2_enable() 和 get_flashcomm2_config_and_validate() 使用 config

迁移前 FlashComm2 parallel size 从 env 读取。

迁移后：

```python
config_val = get_ascend_config().enable_flashcomm2_parallel_size
return config_val > 0
```

`get_flashcomm2_config_and_validate()` 中也改为读取：

```python
ascend_config.enable_flashcomm2_parallel_size
```

同时 FlashComm2 warning 改为：

```python
if not ascend_config.enable_flashcomm1:
```

设计意图：

```text
如果用户通过 additional_config.enable_flashcomm1=true 开启 FlashComm1，FlashComm2 校验不能再因为 env=false 而误报 warning。
```

## 4.4 balance scheduling patch 相关改动

涉及文件：

```text
vllm_ascend/patch/platform/__init__.py
vllm_ascend/patch/platform/patch_balance_schedule.py
vllm_ascend/patch/__init__.py
```

### 4.4.1 patch import 从 env gate 改为总是 import

迁移前：

```python
if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule
```

迁移后：

```python
import vllm_ascend.patch.platform.patch_balance_schedule
```

设计原因：

```text
VLLM_ASCEND_BALANCE_SCHEDULING 原来是 import-time gate。
如果继续用 env 控制是否 import patch，则 additional_config 还没解析时无法决定是否加载 patch。
因此 patch 必须总是加载，但运行时根据 vllm_config.additional_config 决定是否真正启用 balance scheduling。
```

### 4.4.2 patch 内部新增运行时开关判断

新增：

```python
def _balance_scheduling_enabled(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get("enable_balance_scheduling", False))
```

在 `BalanceScheduler` 中：

```python
self._balance_enabled = _balance_scheduling_enabled(vllm_config)
```

如果没启用：

```python
def balance_gather(self, dp_group):
    if not self._balance_enabled:
        return


def schedule(self) -> SchedulerOutput:
    if not self._balance_enabled:
        return super().schedule()
```

设计意图：

```text
patch 可以提前加载，但默认行为仍然完全走原 Scheduler。
只有用户显式设置 additional_config.enable_balance_scheduling=true 时才启用 balance scheduling 逻辑。
```

### 4.4.3 EngineCoreProc.run_engine_core 运行时选择原始或 balance 版本

新增保存原始入口：

```python
_ORIGINAL_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core
```

运行时判断：

```python
if not _balance_scheduling_enabled(vllm_config):
    return _ORIGINAL_RUN_ENGINE_CORE(...)
```

设计意图：

```text
避免未启用 balance scheduling 时改变 EngineCoreProc 的启动路径。
```

## 4.5 vllm_ascend/platform.py

### 4.5.1 balance scheduling 平台校验改读 config

迁移前：

```python
if envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING:
```

迁移后：

```python
if ascend_config.enable_balance_scheduling:
```

报错信息也从环境变量名改为 config 字段名：

```text
enable_balance_scheduling only supports PD-mixed mode
```

设计意图：

```text
通过 --additional-config 开启 balance scheduling 时，仍能触发平台兼容性校验。
```

### 4.5.2 fused mc2 与 hierarchy communication 冲突检查改读 config

迁移后：

```python
if ascend_config.enable_mc2_hierarchy_comm and get_ascend_config().enable_fused_mc2:
```

设计意图：

```text
enable_fused_mc2 迁移后，冲突检查也必须跟随 config 值。
```

## 4.6 profiler 相关改动

文件：

```text
vllm_ascend/profiler/torch_npu_profiler.py
```

迁移前直接读：

```python
envs_ascend.MSMONITOR_USE_DAEMON
```

迁移后：

```python
msmonitor_use_daemon = envs_ascend.MSMONITOR_USE_DAEMON
with suppress(RuntimeError):
    msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
```

设计意图：

```text
1. AscendConfig 初始化后，优先读 config。
2. 如果 profiler 初始化发生在 AscendConfig 尚未初始化的路径，仍 fallback 到旧 env。
3. 保持 MSMONITOR_USE_DAEMON 与 torch profiler 不能同时开启的约束。
```

## 4.7 其他业务调用点改动

以下模块主要是把直接 env 读取替换成 `get_ascend_config()` 字段读取。

### 4.7.1 MoE / Fused MC2 路径

涉及：

```text
vllm_ascend/ascend_forward_context.py
vllm_ascend/eplb/adaptor/vllm_adaptor.py
vllm_ascend/ops/fused_moe/fused_moe.py
vllm_ascend/ops/fused_moe/moe_comm_method.py
vllm_ascend/quantization/methods/w8a8_dynamic.py
```

迁移字段：

```text
VLLM_ASCEND_ENABLE_FUSED_MC2 -> enable_fused_mc2
```

设计意图：

```text
MoE 通信方法选择、fused MC2 权重格式处理、W8A8 scale 处理、EPLB 权重名处理都必须使用同一个 enable_fused_mc2 配置值。
```

否则会出现用户通过 config 开启 fused MC2，但某些子路径仍按 env=false 行为执行的问题。

### 4.7.2 MLAPO 路径

涉及：

```text
vllm_ascend/attention/sfa_v1.py
vllm_ascend/attention/utils.py
```

迁移字段：

```text
VLLM_ASCEND_ENABLE_MLAPO -> enable_mlapo
```

设计意图：

```text
MLAPO 是否启用由 AscendConfig 决定，同时保留 A5 / decode instance 等原有业务限制。
```

### 4.7.3 NZ 路径

涉及：

```text
vllm_ascend/utils.py
vllm_ascend/worker/worker.py
vllm_ascend/xlite/xlite.py
vllm_ascend/batch_invariant.py
```

迁移字段：

```text
VLLM_ASCEND_ENABLE_NZ -> weight_nz_mode
```

设计意图：

```text
NZ 不再是简单 bool，而是 0/1/2 三态配置。
0 表示关闭，1 表示仅量化场景，2 表示 BF16/FP16 等场景也尽量启用。
```

`worker.wake_up()` 的提示也从：

```text
Please set VLLM_ASCEND_ENABLE_NZ=0.
```

改为：

```text
Please set weight_nz_mode=0 via --additional-config.
```

`batch_invariant.override_envs_for_invariance()` 不再修改 env，而是直接修改已初始化的 `ascend_config`：

```python
ascend_config.weight_nz_mode = 0
ascend_config.enable_matmul_allreduce = False
```

设计意图：

```text
迁移后运行期逻辑主要读取 AscendConfig，继续改 env 已经不能影响已初始化的 config。
```

### 4.7.4 Mooncake / KV transfer fused transpose 路径

涉及：

```text
vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
```

迁移字段：

```text
VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK -> enable_transpose_kv_cache_by_block
```

设计意图：

```text
Mooncake KV transfer 是否使用 fused transpose op 跟随 AscendConfig。
```

### 4.7.5 context parallel / matmul allreduce helper

涉及：

```text
vllm_ascend/utils.py
```

迁移字段：

```text
VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL -> enable_context_parallel
VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE -> enable_matmul_allreduce
```

设计意图：

```text
调用 helper 的下游路径无需关心 env/config 来源，只读取统一结果。
```

## 5. 为什么不是简单 additional_config.get(...)

这个 PR 里没有在每个调用点写：

```python
vllm_config.additional_config.get(...)
```

而是统一走：

```python
get_ascend_config().xxx
```

原因：

```text
1. 下游调用点不一定都能拿到 vllm_config。
2. AscendConfig 本来就是 additional_config 的插件侧统一封装。
3. 配置来源应该在 AscendConfig 内集中处理，业务代码只读最终值。
4. 方便后续移除 env fallback，只需要改 AscendConfig 初始化逻辑。
```

## 6. 为什么 balance scheduling 需要特殊处理

`VLLM_ASCEND_BALANCE_SCHEDULING` 和普通运行期配置不同，它原来控制的是：

```text
是否 import patch_balance_schedule
```

这是 import-time 行为。

但 `additional_config` 只有在 vLLM config 初始化后才可用，无法在 import 阶段读取。

因此迁移方案是：

```text
1. patch_balance_schedule 总是 import
2. patch 内部保存原始 Scheduler / EngineCoreProc.run_engine_core
3. 运行时根据 vllm_config.additional_config.enable_balance_scheduling 决定是否启用 balance 逻辑
4. 未启用时回退原始 Scheduler / EngineCoreProc 行为
```

这保证了：

```text
config 可以控制 balance scheduling
未开启时行为尽量不变
后续移除 env 时无需再依赖 import-time env gate
```

## 7. 为什么 FlashComm1 需要特殊处理

FlashComm1 旧开关：

```text
VLLM_ASCEND_ENABLE_FLASHCOMM1
```

代码侧核心入口不是到处直接读取 env，而是：

```python
enable_sp()
```

很多路径通过 `enable_sp()` 判断是否启用 FlashComm1/SP 通信路径。

因此迁移 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 时，关键不是只新增：

```python
AscendConfig.enable_flashcomm1
```

还必须修改：

```python
enable_sp()
```

并保持安全 fallback。

最终设计：

```text
传入 vllm_config.additional_config.enable_flashcomm1
  > 已初始化 AscendConfig.enable_flashcomm1
  > 旧环境变量 VLLM_ASCEND_ENABLE_FLASHCOMM1
```

这能覆盖：

```text
1. AscendConfig.__init__ 内部调用 enable_sp(vllm_config=...)
2. 正常推理路径中 AscendConfig 已初始化
3. Worker / 初始化 / 无 current config 上下文路径
```

## 8. 兼容性设计

### 8.1 老启动脚本继续可用

例如原来：

```bash
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1
```

如果用户不传 `additional_config.enable_matmul_allreduce`，仍会 fallback 到 env。

### 8.2 新 config 显式覆盖旧 env

如果用户同时设置：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

并传入：

```bash
--additional-config '{"enable_flashcomm1": false}'
```

最终以 config 为准，FlashComm1 关闭。

这是为了保证：

```text
显式 config 的优先级高于环境变量。
```

### 8.3 后续移除 env fallback 更容易

当前 env fallback 集中在：

```text
AscendConfig._get_config_value()
enable_sp() 的特殊 fallback
profiler 的未初始化 fallback
```

后续真正移除 env 时，不需要全仓搜索大量业务路径，只需要收敛修改这些入口。

## 9. UT 改动简要说明

UT 不是本次讲解重点，但可以概括为三类。

### 9.1 AscendConfig 迁移优先级测试

文件：

```text
tests/ut/test_ascend_config.py
```

覆盖：

```text
config 未设置时 fallback env
config 显式设置时覆盖 env
enable_flashcomm1 的 env/config 优先级
enable_sp 无 current config 时 fallback env
FlashComm2 warning 使用 config 后不误报
```

### 9.2 下游调用点 mock 更新

多个 UT 原来 mock env，现在改为 mock `get_ascend_config()` 或 config 字段。

涉及：

```text
tests/ut/attention/test_sfa_v1.py
tests/ut/ops/test_linear.py
tests/ut/quantization/methods/*
tests/ut/spec_decode/test_eagle_proposer.py
tests/ut/test_utils.py
```

目的：

```text
让测试跟随新的 config 读取路径，而不是继续依赖 env。
```

### 9.3 profiler / platform / worker 行为测试

覆盖：

```text
msmonitor_use_daemon 与 torch profiler 冲突
balance scheduling 平台校验
NZ wake_up 限制
```

## 10. 对外 PR 描述建议

可以这样描述本 PR：

```text
This PR migrates selected vllm-ascend environment variables from direct env access to AscendConfig / additional_config. The new behavior supports --additional-config as the preferred user-facing configuration path while keeping existing environment variables as deprecated fallbacks for compatibility.

The precedence is:
additional_config explicit value > deprecated environment variable fallback > default value.
```

重点强调：

```text
1. 不是删除 env，而是过渡迁移。
2. 不是简单字段替换，而是把配置读取统一收敛到 AscendConfig。
3. balance scheduling 和 FlashComm1 有特殊时序问题，做了专门兼容设计。
4. 下游业务路径已改为读取最终 config 值，避免 config/env 行为不一致。
```

## 11. 需要 reviewers 重点关注的地方

### 11.1 balance scheduling patch 行为是否等价

因为它从 import-time env gate 变成 always import + runtime config gate，需要重点确认：

```text
未开启 enable_balance_scheduling 时，是否完全回退原始 Scheduler / EngineCoreProc 行为。
```

### 11.2 enable_sp() fallback 是否覆盖所有场景

需要重点确认：

```text
AscendConfig 初始化中调用 enable_sp(vllm_config=...)
推理路径中调用 enable_sp()
无 current vLLM config 上下文时 fallback env
shared_expert_dp 强制 enable_sp 的原行为
```

### 11.3 所有下游路径是否不再误读旧 env

尤其是：

```text
FlashComm2 warning
Fused MC2 MoE 路径
NZ format 路径
MLAPO 路径
MSMONITOR profiler 冲突检查
Mooncake fused transpose 路径
```

### 11.4 env 过渡期行为是否符合预期

需要确认：

```text
旧 env 仍生效
config 显式值覆盖 env
默认值保持不变
```

## 12. 总结

本 PR 的核心设计是：

```text
把 vllm-ascend 的一批运行期环境变量迁移到 AscendConfig，让 additional_config 成为首选配置入口，同时保留旧 env fallback 作为过渡兼容。
```

开发代码的主线是：

```text
1. AscendConfig 新增迁移字段和统一 helper
2. 下游业务逻辑从 envs_xxx 改为 get_ascend_config().xxx
3. balance scheduling 从 import-time env gate 改为 runtime config gate
4. FlashComm1 的 enable_sp() 保持 config 优先、env fallback、无上下文安全
5. UT 更新为验证 config/env 优先级和关键业务路径
```

这使得后续真正删除旧环境变量时，改动面更小、风险更可控。
