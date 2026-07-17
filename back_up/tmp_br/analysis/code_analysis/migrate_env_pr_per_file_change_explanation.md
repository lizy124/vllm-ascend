# migrate_env / ascend_config PR 逐文件代码改动说明

## 文档目的

本文用于在 PR 合入社区前，向 reviewer 逐文件解释当前 `ascend_config` 分支相对 `upstream/main` 的代码改动和设计意图。

当前 PR 的核心目标是：

```text
将 vllm-ascend 中一批运行期环境变量迁移到 AscendConfig / additional_config 配置体系。
```

迁移后的过渡期优先级是：

```text
additional_config 显式配置 > 旧环境变量 fallback > env 默认值
```

本 PR 不是立刻删除旧环境变量，而是先让业务主路径改读 `AscendConfig`，同时保留 deprecated env fallback，降低对已有部署脚本的影响。

另外，当前 PR 还包含两处 CI/测试稳定性修复：

1. non-triton E2E 步骤卸载 Triton 后补装 `regex`，避免 CI log summary 脚本缺依赖。
2. `test_model_runner_v1_with_device.py` 使用本地 `tests/ut/fake_weight`，避免 UT 初始化时访问 ModelScope 下载 `facebook/opt-125m`。

## 变更文件总览

当前 PR 相对 `upstream/main` 净变更 37 个文件：

```text
1 个 CI workflow 文件
19 个开发源码文件
17 个 UT 文件
```

CI workflow 文件：

```text
.github/workflows/_e2e_test.yaml
```

开发源码文件：

```text
vllm_ascend/ascend_config.py
vllm_ascend/ascend_forward_context.py
vllm_ascend/attention/sfa_v1.py
vllm_ascend/attention/utils.py
vllm_ascend/batch_invariant.py
vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
vllm_ascend/envs.py
vllm_ascend/eplb/adaptor/vllm_adaptor.py
vllm_ascend/ops/fused_moe/fused_moe.py
vllm_ascend/ops/fused_moe/moe_comm_method.py
vllm_ascend/patch/__init__.py
vllm_ascend/patch/platform/__init__.py
vllm_ascend/patch/platform/patch_balance_schedule.py
vllm_ascend/platform.py
vllm_ascend/profiler/torch_npu_profiler.py
vllm_ascend/quantization/methods/w8a8_dynamic.py
vllm_ascend/utils.py
vllm_ascend/worker/worker.py
vllm_ascend/xlite/xlite.py
```

UT 文件：

```text
tests/ut/attention/test_sfa_v1.py
tests/ut/batch_invariant/test_batch_invariant.py
tests/ut/distributed/test_parallel_state.py
tests/ut/eplb/adaptor/test_vllm_adaptor.py
tests/ut/ops/test_fused_moe.py
tests/ut/ops/test_linear.py
tests/ut/ops/test_prepare_finalize.py
tests/ut/profiler/test_torch_npu_profiler.py
tests/ut/quantization/methods/test_w8a16.py
tests/ut/quantization/methods/test_w8a8_dynamic.py
tests/ut/quantization/methods/test_w8a8_static.py
tests/ut/spec_decode/test_eagle_proposer.py
tests/ut/test_ascend_config.py
tests/ut/test_platform.py
tests/ut/test_utils.py
tests/ut/worker/test_model_runner_v1_with_device.py
tests/ut/worker/test_worker_v1.py
```

## 一、CI workflow 文件改动说明

### `.github/workflows/_e2e_test.yaml`

改动内容：在 non-triton E2E 测试步骤里，卸载 `triton-ascend` 和 `triton` 后补装：

```bash
python3 -m pip install regex
```

改动原因：后续 summary 步骤会运行 `.github/workflows/scripts/ci_log_summary.py`，该脚本按项目规范使用 `import regex as re`。non-triton 步骤卸载 Triton 相关包后，CI 中曾出现 summary 阶段 `ModuleNotFoundError: No module named 'regex'`。这里补装 `regex`，保证日志汇总脚本可用。

## 二、开发源码文件改动说明

### `vllm_ascend/ascend_config.py`

改动内容：这是本 PR 的核心文件，新增统一 helper：

```python
_get_config_value(additional_config, config_key, env_key, env_value)
```

该 helper 的行为是：

1. 如果 `additional_config` 显式包含 `config_key`，使用该值，并打印一次 config 来源日志。
2. 否则返回旧 env 解析后的 `env_value`。
3. 只有用户显式设置了对应 `env_key` 时，才打印 env fallback 日志，并提示改用 `additional_config.<config_key>`，因为旧环境变量会在下一个 release 移除。

新增迁移字段：

```text
enable_balance_scheduling
enable_flashcomm1
enable_context_parallel
enable_matmul_allreduce
enable_fused_mc2
enable_mlapo
enable_flashcomm2_parallel_size
msmonitor_use_daemon
enable_transpose_kv_cache_by_block
weight_nz_mode
```

其他改动：

- `profiling_chunk_config` 与 balance scheduling 的冲突检查改读 `self.enable_balance_scheduling`。
- `clear_ascend_config()` 同步调用 `clear_enable_sp()`，避免 FlashComm1/SP 的模块级缓存跨测试或 refresh 场景复用旧值。

改动原因：`AscendConfig` 是 `additional_config` 的统一封装入口。把 env fallback 集中在这里，可以让业务代码只依赖最终配置值，后续真正删除旧 env 时改动面更小。

### `vllm_ascend/envs.py`

改动内容：保留旧环境变量定义，但补充 deprecated 注释，提示迁移到 `additional_config`。

改动原因：本 PR 是过渡期迁移，不是立即删除 env。保留旧 env 可以保证已有部署脚本继续可用，同时通过注释和 runtime log 引导用户迁移。

### `vllm_ascend/utils.py`

改动内容：多个公共 helper 从直接读取 env 改为读取 `AscendConfig`：

- 新增 `clear_enable_sp()`。
- `_should_trans_nz()` 改读 `get_ascend_config().weight_nz_mode`。
- `matmul_allreduce_enable()` 改读 `get_ascend_config().enable_matmul_allreduce`。
- `enable_sp()` 支持 `enable_flashcomm1` config 优先，并保留 env fallback。
- `prefill_context_parallel_enable()` 改读 `get_ascend_config().enable_context_parallel`。
- `flashcomm2_enable()` 改读 `get_ascend_config().enable_flashcomm2_parallel_size`。
- `get_flashcomm2_config_and_validate()` 使用 `ascend_config.enable_flashcomm2_parallel_size` 和 `ascend_config.enable_flashcomm1`。

改动原因：`utils.py` 是多条业务路径的公共入口。如果这些 helper 仍读 env，即使 `AscendConfig` 新增字段，下游行为也不会真正迁移。

### `vllm_ascend/patch/platform/__init__.py`

改动内容：balance scheduling patch 从 env 控制 import 改为总是 import：

```python
import vllm_ascend.patch.platform.patch_balance_schedule
```

改动原因：`additional_config` 只有在 vLLM config 初始化后才可用，不能在 import 阶段决定是否加载 patch。因此 patch 必须 always import，运行时再按 config 判断是否启用。

### `vllm_ascend/patch/platform/patch_balance_schedule.py`

改动内容：新增运行时 gate：

```python
def _balance_scheduling_enabled(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get("enable_balance_scheduling", False))
```

并在 scheduler / engine core 入口中：

- 未启用时回退原始 `Scheduler` / `EngineCoreProc.run_engine_core` 行为。
- 启用时才使用 `BalanceScheduler` / `BalanceDPEngineCoreProc` 逻辑。

改动原因：patch 现在总是 import，但默认行为必须不变。runtime gate 可以同时满足 additional_config 初始化时序和默认兼容性。

### `vllm_ascend/patch/__init__.py`

改动内容：更新注释说明，把 balance scheduling 推荐入口从只提 env 改为推荐：

```text
--additional-config '{"enable_balance_scheduling": true}'
```

同时说明旧 env 仍是 deprecated fallback。

改动原因：patch 说明需要和新的用户入口保持一致。

### `vllm_ascend/platform.py`

改动内容：平台级校验从 env 改读 `AscendConfig`：

- balance scheduling 校验使用 `ascend_config.enable_balance_scheduling`。
- `enable_mc2_hierarchy_comm` 与 fused mc2 的冲突检查使用 `get_ascend_config().enable_fused_mc2`。
- 报错信息从旧 env 名调整为 config 字段名。

改动原因：平台校验必须和最终配置值一致，否则用户通过 config 开关功能时，校验和运行行为会不一致。

### `vllm_ascend/profiler/torch_npu_profiler.py`

改动内容：`MSMONITOR_USE_DAEMON` 与 torch profiler 的互斥检查改为 config 优先、env fallback：

```python
msmonitor_use_daemon = envs_ascend.MSMONITOR_USE_DAEMON
with suppress(RuntimeError):
    msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
```

改动原因：profiler 创建时机可能早于 `AscendConfig` 初始化，因此需要兼容未初始化场景；但一旦 config 可用，应以 config 为准。

### `vllm_ascend/ascend_forward_context.py`

改动内容：`select_moe_comm_method()` 中 fused mc2 判断从 env 改为：

```python
fused_mc2_enable = get_ascend_config().enable_fused_mc2
```

并保留主线合入后的 draft model guard：

```python
dispatch_ffn_combine_enable = get_ep_group().world_size <= 32 and (not is_draft_model)
```

改动原因：MoE 通信方法选择是 fused mc2 的核心运行路径，必须跟随 `enable_fused_mc2` 的最终 config 值。

### `vllm_ascend/attention/sfa_v1.py`

改动内容：MLAPO 开关改读：

```python
get_ascend_config().enable_mlapo
```

改动原因：SFA/MLA 主路径中的 MLAPO 行为需要跟随 `AscendConfig.enable_mlapo`。

### `vllm_ascend/attention/utils.py`

改动内容：`enabling_mlapo()` 改为 config 优先读取 `get_ascend_config().enable_mlapo`，原有 A5 和 decode instance 限制保留。

改动原因：MLAPO 公共判断函数必须和 SFA 主路径使用同一配置来源。

### `vllm_ascend/batch_invariant.py`

改动内容：`override_envs_for_invariance()` 对 NZ / matmul allreduce 的覆盖从修改 env 改为修改已初始化的 `AscendConfig`：

```python
ascend_config.weight_nz_mode = 0
ascend_config.enable_matmul_allreduce = False
```

HCCL/LCCL deterministic 相关 env 仍保留。

改动原因：迁移后运行路径读取 config，继续修改 env 已不能影响已初始化配置。

### `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

改动内容：Mooncake KV transfer 中 fused transpose 开关改读：

```python
get_ascend_config().enable_transpose_kv_cache_by_block
```

改动原因：KV transfer 中是否使用 fused `transpose_kv_cache_by_block` op 需要跟随 config。

### `vllm_ascend/eplb/adaptor/vllm_adaptor.py`

改动内容：EPLB adaptor 中 fused mc2 判断从 env 改为：

```python
get_ascend_config().enable_fused_mc2
```

覆盖 W8A8 和 W4A8 的 fused MC2 相关判断。

改动原因：EPLB 权重名和 fused scale 处理必须和实际 fused mc2 模式一致。

### `vllm_ascend/ops/fused_moe/fused_moe.py`

改动内容：`process_weights_after_loading()` 中 fused mc2 判断改读：

```python
get_ascend_config().enable_fused_mc2
```

改动原因：fused mc2 开启时 MoE 权重需要走对应格式转换，不能继续只看 env。

### `vllm_ascend/ops/fused_moe/moe_comm_method.py`

改动内容：`FusedMC2CommImpl` 中所有 fused mc2 mode 判断改读 `get_ascend_config().enable_fused_mc2`，包括 mode 1、mode 2 和非法值报错。

改动原因：这是 fused mc2 通信算子执行路径，必须与 config 开关一致。

### `vllm_ascend/quantization/methods/w8a8_dynamic.py`

改动内容：W8A8 dynamic MoE 量化路径中 fused mc2 判断改读 `get_ascend_config().enable_fused_mc2`，影响 fused scale flag、scale tensor 生成和清理逻辑。

改动原因：W8A8 dynamic fused MoE 的权重 scale 结构必须与 fused mc2 mode 一致。

### `vllm_ascend/worker/worker.py`

改动内容：

- `wake_up()` 中 NZ 限制改读 `get_ascend_config().weight_nz_mode`，报错提示改为 `additional_config.weight_nz_mode`。
- 执行模型时 msMonitor step 改读 `get_ascend_config().msmonitor_use_daemon`。

改动原因：worker 的 sleep/RL 限制和 msMonitor 执行路径都需要使用迁移后的 config。

### `vllm_ascend/xlite/xlite.py`

改动内容：Xlite weight NZ 判断改为：

```python
xlite_config.weight_nz = get_ascend_config().weight_nz_mode == 2
```

改动原因：`weight_nz_mode` 是三态配置，只有值为 2 时才表示 BF16/FP16 等场景也启用 NZ。

## 三、UT 文件改动说明

### `tests/ut/test_ascend_config.py`

改动内容：新增和扩展 AscendConfig 迁移测试，覆盖：

- config 未设置时 fallback 到 env。
- config 显式设置时覆盖 env。
- 只有显式设置 env 时才打印 fallback log。
- fallback log 提示推荐使用 `additional_config`，旧 env 会在下一个 release 移除。
- `enable_flashcomm1` config/env 优先级。
- `enable_sp()` 在无 current config 时 fallback env。
- FlashComm2 warning 使用 `enable_flashcomm1` config，不再误读 env。

改动原因：这是本 PR 最核心的迁移测试，用来证明 `additional_config > env fallback > 默认值` 的优先级真实生效。

### `tests/ut/profiler/test_torch_npu_profiler.py`

改动内容：扩展 `_create_profiler()` 测试，覆盖：

- AscendConfig 未初始化时 env enabled 仍然报错。
- config disabled 可以覆盖 env enabled。
- config enabled 可以覆盖 env disabled 并报错。

改动原因：profiler 是特殊初始化路径，必须验证 config 优先和 env fallback 都可用。

### `tests/ut/ops/test_fused_moe.py`

改动内容：把 fused MoE 测试中的 env mock 改成 `get_ascend_config().enable_fused_mc2` mock。

改动原因：源码已删除对应 env 读取，测试必须跟随新的配置入口。

### `tests/ut/attention/test_sfa_v1.py`

改动内容：更新 SFA 测试 mock，补充迁移后路径需要的依赖，例如 DSA-CP 和 TP group 相关 mock。

改动原因：配置迁移后测试路径会触发更多 runtime helper，需要让测试继续聚焦 SFA metadata 构造逻辑，而不是依赖真实分布式环境。

### `tests/ut/batch_invariant/test_batch_invariant.py`

改动内容：测试从验证 env 被写入，改为验证 mock config 字段被覆盖：

```python
weight_nz_mode == 0
enable_matmul_allreduce is False
```

改动原因：`override_envs_for_invariance()` 迁移后覆盖的是 `AscendConfig`，不是 env。

### `tests/ut/distributed/test_parallel_state.py`

改动内容：测试中的 mock AscendConfig 补齐迁移后的字段，例如 FlashComm2/context parallel 等。

改动原因：下游 helper 现在会读取更多 config 字段，mock 对象需要覆盖这些属性。

### `tests/ut/eplb/adaptor/test_vllm_adaptor.py`

改动内容：把 fused mc2 相关 env mock 改为 config mock。

改动原因：`VllmEplbAdaptor` 已通过 `get_ascend_config().enable_fused_mc2` 判断 fused scale 权重名。

### `tests/ut/ops/test_linear.py`

改动内容：更新 matmul allreduce / config 相关 mock，使测试读取 `get_ascend_config().enable_matmul_allreduce`。

改动原因：`matmul_allreduce_enable()` 已迁移到 config。

### `tests/ut/ops/test_prepare_finalize.py`

改动内容：补充与 `enable_sp()` / FlashComm1 迁移相关的 mock 或断言。

改动原因：prepare/finalize 路径依赖 FlashComm1/SP 判断，测试需要覆盖迁移后的 config 路径。

### `tests/ut/quantization/methods/test_w8a16.py`

改动内容：补充 config mock，使量化路径中 NZ / fused mc2 相关判断通过 AscendConfig 提供。

改动原因：量化测试不应继续依赖 env。

### `tests/ut/quantization/methods/test_w8a8_dynamic.py`

改动内容：把 W8A8 dynamic 中 fused mc2 的测试 mock 改为 config mock。

改动原因：源码中 `enable_fused_mc2` 已从 env 迁移到 config，需要验证 mode 1 / mode 2 下 scale 处理逻辑仍正确。

### `tests/ut/quantization/methods/test_w8a8_static.py`

改动内容：补充或更新 config mock，匹配迁移后的 NZ / fused mc2 读取路径。

改动原因：W8A8 static 测试依赖公共量化路径，公共路径已经迁移到 AscendConfig。

### `tests/ut/spec_decode/test_eagle_proposer.py`

改动内容：补充 `get_ascend_config()` mock，并调整 FlashComm1/SP、context parallel、fused mc2 等字段依赖。

改动原因：Eagle proposer / spec decode 路径会间接调用多个 config helper，测试需要提供完整 mock config。

### `tests/ut/test_platform.py`

改动内容：更新 platform 校验测试，让 balance scheduling 和 fused mc2 相关校验走 AscendConfig 字段。

改动原因：`platform.py` 已经从 env 判断迁移到 config 判断。

### `tests/ut/test_utils.py`

改动内容：更新 utils helper 测试，包括：

```text
matmul_allreduce_enable
prefill_context_parallel_enable
flashcomm2_enable
enable_sp
weight_nz_mode 相关逻辑
```

改动原因：`utils.py` 是公共入口，UT 必须覆盖 config 优先、env fallback 和 helper 返回值。

### `tests/ut/worker/test_model_runner_v1_with_device.py`

改动内容：把测试模型从远程：

```python
model="facebook/opt-125m"
```

改为本地：

```python
model=FAKE_WEIGHT_PATH
skip_tokenizer_init=True
```

其中 `FAKE_WEIGHT_PATH` 指向 `tests/ut/fake_weight`。

改动原因：CI 中 ModelScope 对 `facebook/opt-125m` 下载接口曾返回 400，导致 UT setup 在测试前失败。该测试只需要 OPT config 元信息，不需要真实下载模型，因此改用仓库内已有 fake OPT config，避免外网依赖。

### `tests/ut/worker/test_worker_v1.py`

改动内容：更新 worker 测试中 NZ / msmonitor 的 mock，从 env 改为 AscendConfig 字段。

改动原因：`NPUWorker.wake_up()` 和 `NPUWorker.execute_model()` 已经分别读取 `weight_nz_mode` 和 `msmonitor_use_daemon` config。

## 四、逐文件变更原因汇总表

| 文件 | 改动原因 |
|---|---|
| `.github/workflows/_e2e_test.yaml` | non-triton E2E 卸载 Triton 后补装 `regex`，保证 CI log summary 可运行 |
| `vllm_ascend/ascend_config.py` | 新增迁移配置字段，集中实现 config 优先、env fallback，并优化 fallback 日志 |
| `vllm_ascend/envs.py` | 保留旧 env，并标记 deprecated，提示迁移到 config |
| `vllm_ascend/utils.py` | 公共 helper 从 env 改为 config，处理 FlashComm1 特殊 fallback |
| `vllm_ascend/patch/platform/__init__.py` | balance scheduling patch 不能再由 env 做 import-time gate |
| `vllm_ascend/patch/platform/patch_balance_schedule.py` | 改成 always import + runtime config gate |
| `vllm_ascend/patch/__init__.py` | 更新用户说明，推荐 additional_config |
| `vllm_ascend/platform.py` | 平台校验改读 config，避免 config/env 行为不一致 |
| `vllm_ascend/profiler/torch_npu_profiler.py` | profiler 互斥检查 config 优先，AscendConfig 未初始化时 env fallback |
| `vllm_ascend/ascend_forward_context.py` | MoE 通信方法选择使用 enable_fused_mc2 config，并保留 draft model guard |
| `vllm_ascend/attention/sfa_v1.py` | SFA/MLAPO 主路径使用 enable_mlapo config |
| `vllm_ascend/attention/utils.py` | MLAPO helper 使用 enable_mlapo config |
| `vllm_ascend/batch_invariant.py` | 迁移后运行期覆盖 config，而不是修改 env |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` | Mooncake fused transpose 开关使用 config |
| `vllm_ascend/eplb/adaptor/vllm_adaptor.py` | EPLB fused scale 权重名判断使用 enable_fused_mc2 config |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | Fused MoE 权重格式处理使用 enable_fused_mc2 config |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | Fused MC2 通信算子选择使用 enable_fused_mc2 config |
| `vllm_ascend/quantization/methods/w8a8_dynamic.py` | W8A8 dynamic scale 处理使用 enable_fused_mc2 config |
| `vllm_ascend/worker/worker.py` | worker wake_up / msmonitor 路径使用 config |
| `vllm_ascend/xlite/xlite.py` | Xlite weight NZ 判断使用 weight_nz_mode config |
| `tests/ut/worker/test_model_runner_v1_with_device.py` | 使用本地 fake OPT config，避免 UT 依赖 ModelScope 下载 |
| `tests/ut/*` | 测试从 mock env 更新为 mock config，并补充优先级/兼容测试 |

## 五、对 reviewer 的解释重点

建议向 reviewer 强调：

```text
1. 本 PR 不是删除 env，而是迁移到 config，并保留 deprecated env fallback。
2. 业务主路径已经从 env 读取切换到 AscendConfig。
3. additional_config 显式值优先于旧 env。
4. 默认 env 未显式设置时不再打印 fallback 日志，减少日志噪音。
5. 如果用户显式设置 deprecated env，会打印一次提示，建议改用 additional_config，并说明 env 会在下一个 release 移除。
6. balance scheduling 因为原来是 import-time env gate，所以采用 always import + runtime config gate。
7. FlashComm1 因为代码侧统一入口是 enable_sp()，所以必须特殊处理初始化时序和 fallback。
8. non-triton CI 的 regex 补装和本地 fake weight UT 是流水稳定性修复，不改变运行功能。
9. 后续真正删除 env 时，只需要清理 AscendConfig fallback、enable_sp fallback 和 profiler fallback 等集中入口。
```

## 六、当前是否已经完全脱离 env

当前 PR 是过渡期状态。

用户可以不传 env，只通过 config 使用这些功能：

```bash
--additional-config '{"enable_flashcomm1": true, "enable_fused_mc2": 1}'
```

但代码层面还保留 env fallback，因此还不能直接删除 `envs.py` 中这些 deprecated env 定义。

后续删除 env 前，需要再做一轮清理：

```text
1. AscendConfig 中的 env fallback 改为 hardcoded default。
2. enable_sp() 中的 VLLM_ASCEND_ENABLE_FLASHCOMM1 fallback 改为 False 或强依赖 config。
3. torch_npu_profiler.py 中的 MSMONITOR_USE_DAEMON fallback 改为 False。
4. 删除 envs.py 中 deprecated env 定义和相关文档。
```

最终目标状态是：

```text
additional_config 显式配置 > hardcoded default
```

不再依赖旧环境变量。
