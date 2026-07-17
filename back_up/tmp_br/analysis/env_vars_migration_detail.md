# 环境变量迁移到 AscendConfig 详细复核分析

## 0. 本次复核结论

本文档复核 `D:\lzy\code\for_env\vllm-ascend` 仓库中 `main` 分支与 `ascend_config` 分支关于环境变量迁移的实际代码。

原文档中有几处变量名不准确，已修正：

| 原错误名称 | 正确名称 | 说明 |
|---|---|---|
| `VLLM_ASCEND_HCCL_SO_PATH` | `HCCL_SO_PATH` | `main` 分支真实定义就是 `HCCL_SO_PATH`。 |
| `VLLM_ASCEND_VLLM_VERSION` | `VLLM_VERSION` | `main` 分支真实定义就是 `VLLM_VERSION`。 |
| `VLLM_ASCEND_WEIGHT_NZ_MODE` | `VLLM_ASCEND_ENABLE_NZ` | `main` 分支真实定义是 `VLLM_ASCEND_ENABLE_NZ`，语义映射到 `weight_nz_mode`。 |
| `VLLM_ASCEND_ENABLE_SHARED_EXPERT_DP` | 不存在于 `envs.py` | `enable_shared_expert_dp` 是 `additional_config` 原生字段，不是本次从 envs.py 迁移的环境变量。 |

另外，原文档说“10 个环境变量”不准确：

- `main` 分支中确实有 10 个被 `ascend_config` 分支移除/注释的 env 定义。
- 但其中不包含 `VLLM_ASCEND_ENABLE_SHARED_EXPERT_DP`。
- 真实 10 个是：
  1. `HCCL_SO_PATH`
  2. `VLLM_VERSION`
  3. `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`
  4. `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`
  5. `MSMONITOR_USE_DAEMON`
  6. `VLLM_ASCEND_ENABLE_MLAPO`
  7. `VLLM_ASCEND_ENABLE_NZ`
  8. `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL`
  9. `VLLM_ASCEND_ENABLE_FUSED_MC2`
  10. `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK`

按归属重新分类：

| 环境变量 | 归属/来源判断 | 是否应迁移到 vllm-ascend Config | 复核结论 |
|---|---|---:|---|
| `HCCL_SO_PATH` | HCCL/CANN 生态路径变量，被 vllm-ascend 用于 pyhccl | 不建议作为“vllm-ascend 专属迁移项”强迁 | 可保留 env；如保留 `hccl_so_path`，应作为显式覆盖项而非替代 CANN/HCCL env。 |
| `VLLM_VERSION` | 通用名称，vllm-ascend 用于覆盖 vLLM 包版本判断；不是 `VLLM_ASCEND_*` 专属变量 | 不建议迁移 | 更适合删除 override 或保持与 vLLM 版本来源一致，避免 Config 覆盖导致不一致。 |
| `VLLM_ASCEND_ENABLE_NZ` | vllm-ascend 专属运行时策略 | 可以迁移 | 已映射为 `weight_nz_mode`。 |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | vllm-ascend 专属运行时优化 | 可以迁移 | 已映射为 `enable_matmul_allreduce`。 |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | vllm-ascend 专属运行时优化 | 可以迁移 | 已映射为 `enable_flashcomm2_parallel_size`。 |
| `MSMONITOR_USE_DAEMON` | msMonitor 工具环境变量，非 vllm-ascend 专属命名 | 谨慎迁移 | 可提供 `msmonitor_use_daemon` 作为 vllm-ascend 启动配置，但不应宣称原 env 是 vllm-ascend 专属。 |
| `VLLM_ASCEND_ENABLE_MLAPO` | vllm-ascend 专属运行时优化 | 可以迁移 | 已映射为 `enable_mlapo`。 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | vllm-ascend 专属运行时优化 | 可以迁移 | 已映射为 `enable_fused_mc2`。 |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | vllm-ascend 专属运行时并行策略 | 可以迁移 | 已映射为 `enable_context_parallel`。 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | vllm-ascend 专属运行时融合算子开关 | 可以迁移 | 已映射为 `enable_transpose_kv_cache_by_block`。 |

---

## 1. `HCCL_SO_PATH` → `hccl_so_path`

### 1.1 正确变量名

`main` 分支真实定义：

```python
# vllm_ascend/envs.py, main
# The path for HCCL library, it's used by pyhccl communicator backend. If
# not set, the default value is libhccl.so.
"HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
```

`ascend_config` 分支中的注释：

```python
# vllm_ascend/envs.py, ascend_config
# DEPRECATED: HCCL_SO_PATH env var is removed. Use --additional-config '{"hccl_so_path": "/path/to/libhccl.so"}'.
#     "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
```

因此原文档中的 `VLLM_ASCEND_HCCL_SO_PATH` 是错误名称，应全部改为 `HCCL_SO_PATH`。

### 1.2 当前迁移代码

`ascend_config` 分支新增字段：

```python
# vllm_ascend/ascend_config.py
self.hccl_so_path = additional_config.get("hccl_so_path", None)
```

`find_hccl_library()` 当前读取 Config：

```python
# vllm_ascend/utils.py
config = get_ascend_config()
so_file = config.hccl_so_path
```

错误提示也改为 Config：

```python
# vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py
"config hccl_so_path via --additional-config"
```

### 1.3 归属判断

`HCCL_SO_PATH` 不是 `VLLM_ASCEND_*` 命名，语义是 HCCL shared library 路径。HCCL 是 CANN/Ascend 通信库，严格讲这个变量更接近底层 HCCL/CANN 生态路径配置，而不是 vllm-ascend 自己的业务开关。

vllm-ascend 只是消费它来定位 `libhccl.so`。

### 1.4 是否适合迁移

技术上可以通过 Config 传递 `hccl_so_path`，因为 `find_hccl_library()` 的调用发生在通信库初始化阶段，此时 `AscendConfig` 通常已经初始化。

但从归属和语义看，不建议把它作为“vllm-ascend 专属环境变量迁移”来处理：

1. 它是 HCCL/CANN 路径类变量，不是 vllm-ascend 业务配置。
2. 用户可能已经通过环境或系统安装路径管理 HCCL。
3. 迁移到 `additional_config` 会让一个底层库路径配置变成 vLLM 请求/服务配置，边界不够清晰。

建议结论：

```text
HCCL_SO_PATH 名称必须修正。它不应被归类为 VLLM_ASCEND_* 专属变量。是否提供 hccl_so_path 作为 additional_config override 可以保留，但不建议把 HCCL_SO_PATH 视为必须迁移到 Config 的环境变量。
```

---

## 2. `VLLM_VERSION` → `vllm_version`

### 2.1 正确变量名

`main` 分支真实定义：

```python
# vllm_ascend/envs.py, main
"VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
```

`ascend_config` 分支中的注释：

```python
# DEPRECATED: VLLM_VERSION env var is removed.
# Use --additional-config '{"vllm_version": "0.9.0"}'.
#     "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
```

因此原文档中的 `VLLM_ASCEND_VLLM_VERSION` 是错误名称，应全部改为 `VLLM_VERSION`。

### 2.2 当前迁移代码

`ascend_config` 分支新增字段：

```python
self.vllm_version = additional_config.get("vllm_version", None)
```

`vllm_version_is()` 中当前逻辑：

```python
config_version = None
with suppress(RuntimeError):
    config_version = get_ascend_config().vllm_version
if config_version is not None:
    vllm_version = config_version
else:
    import vllm
    vllm_version = vllm.__version__
```

### 2.3 归属判断

`VLLM_VERSION` 是非常通用的名称，语义上指 vLLM 包版本，不是 vllm-ascend 专属变量。

本地 `D:\lzy\code\for_env\vllm\vllm\envs.py` 没有定义 `VLLM_VERSION`，所以它不是当前本地 vLLM core 的正式 env 定义。但它表达的是 vLLM 的版本，不是 Ascend 后端的运行时功能。

从注释看，它主要用于开发者本地从源码安装 vLLM 时，实际 `vllm.__version__` 可能偏离目标版本，用 env 手动覆盖版本判断。

### 2.4 是否适合迁移

不建议迁移到 `AscendConfig`，原因：

1. `VLLM_VERSION` 表达的是 vLLM 包版本，应该尽量来自 `vllm.__version__` 或 vLLM 自身版本机制。
2. 把它放进 `additional_config` 会造成“单个 vLLM engine 配置可以覆盖全局 vLLM 包版本”的语义错位。
3. `vllm_version_is()` 可能在 `AscendConfig` 初始化前被调用，当前代码只能通过 `suppress(RuntimeError)` fallback，说明它本身不是自然的 Config 字段。
4. 为保持与 vLLM 实际版本一致，最好减少手动覆盖，而不是把覆盖能力迁移到运行时 Config。

建议结论：

```text
VLLM_VERSION 名称必须修正。它不是 VLLM_ASCEND_* 专属变量，也不建议迁移到 Config。更合理的是优先使用 vllm.__version__；如确需开发调试 override，应保留为开发/环境层机制，而不是用户运行时 additional_config。
```

---

## 3. `VLLM_ASCEND_ENABLE_NZ` → `weight_nz_mode`

### 3.1 正确变量名

`main` 分支真实定义：

```python
"VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
```

`ascend_config` 分支迁移说明：

```python
# DEPRECATED: VLLM_ASCEND_ENABLE_NZ env var is removed.
# Use --additional-config '{"weight_nz_mode": 1}'.
```

原文档中的 `VLLM_ASCEND_WEIGHT_NZ_MODE` 是错误名称。

### 3.2 当前迁移代码

```python
self.weight_nz_mode = additional_config.get("weight_nz_mode", 1)
```

使用点包括：

```python
# vllm_ascend/utils.py
nz_mode = config.weight_nz_mode
```

```python
# vllm_ascend/worker/worker.py
nz_mode = get_ascend_config().weight_nz_mode
```

```python
# vllm_ascend/xlite/xlite.py
xlite_config.weight_nz = get_ascend_config().weight_nz_mode == 2
```

`batch_invariant` 场景也改为直接修改 Config：

```python
ascend_config.weight_nz_mode = 0
```

### 3.3 归属与迁移判断

这是 vllm-ascend 专属运行时策略，控制权重是否转 FRACTAL_NZ，以及 BF16/FP16 是否也转换。

适合迁移到 Config。

---

## 4. `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` → `enable_matmul_allreduce`

### 4.1 真实定义

```python
"VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0"))),
```

`ascend_config` 分支注释：

```python
# Use --additional-config '{"enable_matmul_allreduce": true}'.
```

### 4.2 当前迁移代码

```python
self.enable_matmul_allreduce = additional_config.get("enable_matmul_allreduce", False)
```

```python
def matmul_allreduce_enable() -> bool:
    return get_ascend_config().enable_matmul_allreduce
```

### 4.3 归属与迁移判断

这是 vllm-ascend 专属运行时优化，影响 Row Parallel 算子是否使用 `MatmulAllreduceRowParallelOp`。

适合迁移到 Config。

---

## 5. `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` → `enable_flashcomm2_parallel_size`

### 5.1 真实定义

```python
"VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
```

`ascend_config` 分支注释：

```python
# Use --additional-config '{"enable_flashcomm2_parallel_size": 2}'.
```

### 5.2 当前迁移代码

```python
self.enable_flashcomm2_parallel_size = additional_config.get("enable_flashcomm2_parallel_size", 0)
```

```python
def flashcomm2_enable() -> bool:
    config_val = get_ascend_config().enable_flashcomm2_parallel_size
    return config_val > 0
```

`get_flashcomm2_config_and_validate()` 也改为读 `ascend_config.enable_flashcomm2_parallel_size`。

### 5.3 归属与迁移判断

这是 vllm-ascend 专属 FlashComm2 配置，适合迁移到 Config。

需要注意：当前 `get_flashcomm2_config_and_validate()` 里仍有一处对 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 的 env 读取，用于 warning。这是 FlashComm1 迁移时需要同步处理的问题，不影响 FlashComm2 自身适合迁移的结论。

---

## 6. `MSMONITOR_USE_DAEMON` → `msmonitor_use_daemon`

### 6.1 真实定义

```python
"MSMONITOR_USE_DAEMON": lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0"))),
```

`ascend_config` 分支注释：

```python
# Use --additional-config '{"msmonitor_use_daemon": true}'.
```

### 6.2 当前迁移代码

```python
self.msmonitor_use_daemon = additional_config.get("msmonitor_use_daemon", False)
```

使用点：

```python
if get_ascend_config().msmonitor_use_daemon:
    dp.step()
```

以及 profiler 冲突检查：

```python
if get_ascend_config().msmonitor_use_daemon:
    raise RuntimeError("MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled...")
```

### 6.3 归属与迁移判断

`MSMONITOR_USE_DAEMON` 不是 `VLLM_ASCEND_*` 命名，语义上属于 msMonitor 工具开关，不是严格的 vllm-ascend 专属变量。

但 vllm-ascend 在 worker 执行路径中消费它，用于控制是否调用 msMonitor daemon 步进逻辑。从技术上可以提供 `additional_config.msmonitor_use_daemon`。

建议结论：

```text
可作为 vllm-ascend 对 msMonitor 的显式运行配置保留，但文档中应标明原 env 归属于 msMonitor/工具生态，不应称为 vllm-ascend 专属环境变量。
```

---

## 7. `VLLM_ASCEND_ENABLE_MLAPO` → `enable_mlapo`

### 7.1 真实定义

```python
"VLLM_ASCEND_ENABLE_MLAPO": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
```

`ascend_config` 分支注释：

```python
# Use --additional-config '{"enable_mlapo": true}'.
```

### 7.2 当前迁移代码

```python
self.enable_mlapo = additional_config.get("enable_mlapo", True)
```

使用点：

```python
config_val = get_ascend_config().enable_mlapo
```

```python
self.enable_mlapo = get_ascend_config().enable_mlapo
```

### 7.3 归属与迁移判断

这是 vllm-ascend 专属 MLAPO 优化开关，适合迁移到 Config。

---

## 8. `VLLM_ASCEND_ENABLE_FUSED_MC2` → `enable_fused_mc2`

### 8.1 真实定义

```python
"VLLM_ASCEND_ENABLE_FUSED_MC2": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")),
```

`ascend_config` 分支注释：

```python
# Use --additional-config '{"enable_fused_mc2": 1}'.
```

### 8.2 当前迁移代码

```python
self.enable_fused_mc2 = additional_config.get("enable_fused_mc2", 0)
```

迁移涉及 MoE 通信、量化、EPLB adaptor、forward context、platform 校验等多个点，统一模式是：

```python
get_ascend_config().enable_fused_mc2
```

### 8.3 归属与迁移判断

这是 vllm-ascend 专属 fused MC2 优化开关，适合迁移到 Config。

---

## 9. `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` → `enable_context_parallel`

### 9.1 真实定义

```python
"VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL", "0"))),
```

`ascend_config` 分支注释：

```python
# Use --additional-config '{"enable_context_parallel": true}'.
```

### 9.2 当前迁移代码

```python
self.enable_context_parallel = additional_config.get("enable_context_parallel", False)
```

```python
def prefill_context_parallel_enable() -> bool:
    return get_ascend_config().enable_context_parallel
```

### 9.3 归属与迁移判断

这是 vllm-ascend 专属上下文并行开关，适合迁移到 Config。

---

## 10. `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` → `enable_transpose_kv_cache_by_block`

### 10.1 真实定义

```python
"VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": lambda: bool(
    int(os.getenv("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1"))
),
```

`ascend_config` 分支注释：

```python
# Use --additional-config '{"enable_transpose_kv_cache_by_block": true}'.
```

### 10.2 当前迁移代码

```python
self.enable_transpose_kv_cache_by_block = additional_config.get("enable_transpose_kv_cache_by_block", True)
```

使用点：

```python
# vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
use_fused_op = get_ascend_config().enable_transpose_kv_cache_by_block
```

### 10.3 归属与迁移判断

这是 vllm-ascend 专属 KV cache reformat fused op 开关，适合迁移到 Config。

---

## 11. 被原文档误列的 `VLLM_ASCEND_ENABLE_SHARED_EXPERT_DP`

原文档第 9 项写为：

```text
VLLM_ASCEND_ENABLE_SHARED_EXPERT_DP → enable_shared_expert_dp
```

复核后结论：

1. `main` 分支 `vllm_ascend/envs.py` 中没有 `VLLM_ASCEND_ENABLE_SHARED_EXPERT_DP`。
2. `ascend_config` 分支也没有从 env 中注释掉这个变量。
3. `enable_shared_expert_dp` 是 `AscendConfig` 中已有/新增的 `additional_config` 字段，不是本次 10 个 env 迁移项。

相关代码：

```python
self.enable_shared_expert_dp = (
    additional_config.get("enable_shared_expert_dp", False)
    and vllm_config.parallel_config.enable_expert_parallel
    and vllm_config.parallel_config.tensor_parallel_size > 1
)
```

因此文档中应删除该项，不应作为环境变量迁移分析对象。

---

## 12. 最终建议表

| 正确环境变量 | Config 字段 | vllm-ascend 专属？ | 外部/底层归属 | 是否建议迁移 | 备注 |
|---|---|---:|---|---:|---|
| `HCCL_SO_PATH` | `hccl_so_path` | 否 | HCCL/CANN | 谨慎/不建议强迁 | 可保留 Config override，但不应说成 vllm-ascend 专属迁移。 |
| `VLLM_VERSION` | `vllm_version` | 否 | vLLM 版本语义 | 不建议 | 应优先保持与 `vllm.__version__` 一致。 |
| `VLLM_ASCEND_ENABLE_NZ` | `weight_nz_mode` | 是 | vllm-ascend | 是 | 原文档变量名已修正。 |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `enable_matmul_allreduce` | 是 | vllm-ascend | 是 | 运行时优化。 |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | `enable_flashcomm2_parallel_size` | 是 | vllm-ascend | 是 | 运行时优化。 |
| `MSMONITOR_USE_DAEMON` | `msmonitor_use_daemon` | 否 | msMonitor 工具 | 可选/谨慎 | 可作为 vllm-ascend 对工具集成的运行配置。 |
| `VLLM_ASCEND_ENABLE_MLAPO` | `enable_mlapo` | 是 | vllm-ascend | 是 | 运行时优化。 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `enable_fused_mc2` | 是 | vllm-ascend | 是 | MoE 通信优化。 |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | `enable_context_parallel` | 是 | vllm-ascend | 是 | 上下文并行开关。 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `enable_transpose_kv_cache_by_block` | 是 | vllm-ascend | 是 | KV cache fused op 开关。 |

---

## 13. 修正后的总判断

原文档里“10 个都可以迁移”的判断需要修正：

```text
严格按归属和语义看，10 个里并不是全部都适合迁移。
```

建议分类：

### 不建议作为 vllm-ascend Config 迁移项

```text
HCCL_SO_PATH
VLLM_VERSION
```

原因：

- `HCCL_SO_PATH` 更接近 HCCL/CANN 底层库路径配置。
- `VLLM_VERSION` 表达 vLLM 包版本，应尽量与 vLLM 自身版本机制一致。

### 可谨慎提供 Config override

```text
MSMONITOR_USE_DAEMON -> msmonitor_use_daemon
```

原因：

- 原 env 属于 msMonitor 工具语义，不是 vllm-ascend 专属命名。
- 但 vllm-ascend worker 确实消费它作为运行时行为开关。

### 适合迁移到 Config

```text
VLLM_ASCEND_ENABLE_NZ -> weight_nz_mode
VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE -> enable_matmul_allreduce
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE -> enable_flashcomm2_parallel_size
VLLM_ASCEND_ENABLE_MLAPO -> enable_mlapo
VLLM_ASCEND_ENABLE_FUSED_MC2 -> enable_fused_mc2
VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL -> enable_context_parallel
VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK -> enable_transpose_kv_cache_by_block
```

这些都是 vllm-ascend 专属运行时功能/优化开关，迁移到 `additional_config` 是合理的。
