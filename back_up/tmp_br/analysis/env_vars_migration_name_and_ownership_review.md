# env_vars_migration_detail.md 变量名与归属复核结果

## 1. 复核目标

复核文件：

`D:\lzy\code\for_env\result\analysis\env_vars_migration_detail.md`

复核仓库：

`D:\lzy\code\for_env\vllm-ascend`

复核分支：

- `main`
- `ascend_config`

复核重点：

1. 文档涉及的 10 个环境变量名称是否真实存在。
2. `main` 分支和 `ascend_config` 分支中这些变量的变化是否一致。
3. 哪些变量是 vllm-ascend 专属，哪些更像外部系统/CANN/vLLM/msMonitor 的变量。
4. 哪些变量确实适合迁移到 Config，哪些不建议迁移。

---

## 2. 变量名修正

原文档中有 4 个明显问题。

| 原文档名称 | 正确名称 | 复核结论 |
|---|---|---|
| `VLLM_ASCEND_HCCL_SO_PATH` | `HCCL_SO_PATH` | `main` 中真实定义为 `HCCL_SO_PATH`，没有 `VLLM_ASCEND_` 前缀。 |
| `VLLM_ASCEND_VLLM_VERSION` | `VLLM_VERSION` | `main` 中真实定义为 `VLLM_VERSION`，没有 `VLLM_ASCEND_` 前缀。 |
| `VLLM_ASCEND_WEIGHT_NZ_MODE` | `VLLM_ASCEND_ENABLE_NZ` | `main` 中真实定义为 `VLLM_ASCEND_ENABLE_NZ`。 |
| `VLLM_ASCEND_ENABLE_SHARED_EXPERT_DP` | 不在 `envs.py` 中 | 这是 `additional_config.enable_shared_expert_dp` 字段，不是 env 迁移项。 |

---

## 3. main 分支真实的 10 个待迁移/已迁移变量

`main` 分支中，后续在 `ascend_config` 分支被注释为 deprecated/removed 的真实 10 个变量是：

```text
HCCL_SO_PATH
VLLM_VERSION
VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
MSMONITOR_USE_DAEMON
VLLM_ASCEND_ENABLE_MLAPO
VLLM_ASCEND_ENABLE_NZ
VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL
VLLM_ASCEND_ENABLE_FUSED_MC2
VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
```

不包括：

```text
VLLM_ASCEND_ENABLE_SHARED_EXPERT_DP
```

---

## 4. ascend_config 分支中的映射

`ascend_config` 分支中对应映射如下：

| 环境变量 | Config 字段 |
|---|---|
| `HCCL_SO_PATH` | `hccl_so_path` |
| `VLLM_VERSION` | `vllm_version` |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `enable_matmul_allreduce` |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | `enable_flashcomm2_parallel_size` |
| `MSMONITOR_USE_DAEMON` | `msmonitor_use_daemon` |
| `VLLM_ASCEND_ENABLE_MLAPO` | `enable_mlapo` |
| `VLLM_ASCEND_ENABLE_NZ` | `weight_nz_mode` |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | `enable_context_parallel` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `enable_fused_mc2` |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `enable_transpose_kv_cache_by_block` |

`enable_shared_expert_dp` 虽然也在 `AscendConfig` 中，但不是从 `envs.py` 迁移来的环境变量。

---

## 5. 归属分析

### 5.1 不应归为 vllm-ascend 专属环境变量

#### `HCCL_SO_PATH`

`HCCL_SO_PATH` 是 HCCL shared library 路径变量，HCCL 属于 Ascend/CANN 通信库生态。

vllm-ascend 中原本用它在 `find_hccl_library()` 中定位 `libhccl.so`。这说明 vllm-ascend 是消费者，但该变量语义不是 vllm-ascend 自己的模型/推理配置。

结论：

```text
不建议把 HCCL_SO_PATH 作为 vllm-ascend 专属配置强制迁移。可以提供 additional_config.hccl_so_path 作为显式 override，但不应替代底层 HCCL/CANN 环境配置语义。
```

#### `VLLM_VERSION`

`VLLM_VERSION` 表达的是 vLLM 包版本。虽然本地 vLLM core 的 `vllm/envs.py` 未定义它，但它不是 `VLLM_ASCEND_*` 命名，也不是 Ascend 后端运行时功能。

它在 vllm-ascend 中的作用是：开发者本地源码安装 vLLM 时，手工覆盖版本判断。

结论：

```text
不建议迁移到 vllm-ascend Config。版本判断应优先依赖 vllm.__version__ 或 vLLM 自身版本机制，避免 additional_config 覆盖导致与实际包版本不一致。
```

#### `MSMONITOR_USE_DAEMON`

`MSMONITOR_USE_DAEMON` 语义来自 msMonitor 工具，不是 `VLLM_ASCEND_*` 命名。

但 vllm-ascend worker 确实消费它控制 msMonitor daemon step 和 profiler 冲突检查。

结论：

```text
可以提供 msmonitor_use_daemon 作为 vllm-ascend 对 msMonitor 集成的显式运行配置，但文档中应标明原变量归属是 msMonitor 工具，不是 vllm-ascend 专属变量。
```

### 5.2 vllm-ascend 专属、适合迁移的变量

以下变量均是 `VLLM_ASCEND_*` 命名，控制 vllm-ascend 运行时优化/并行/融合策略，适合迁移到 `additional_config`：

```text
VLLM_ASCEND_ENABLE_NZ
VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
VLLM_ASCEND_ENABLE_MLAPO
VLLM_ASCEND_ENABLE_FUSED_MC2
VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL
VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
```

---

## 6. 最终建议

建议将这 10 个变量分成三类写入迁移结论。

### 6.1 不建议迁移

```text
HCCL_SO_PATH
VLLM_VERSION
```

原因：

- `HCCL_SO_PATH` 属于 HCCL/CANN 底层库路径配置。
- `VLLM_VERSION` 属于 vLLM 包版本语义，不应由 Ascend runtime Config 覆盖。

### 6.2 可谨慎保留 Config override

```text
MSMONITOR_USE_DAEMON -> msmonitor_use_daemon
```

原因：

- 原变量更像 msMonitor 工具变量。
- 但 vllm-ascend worker 有明确消费场景。

### 6.3 适合迁移

```text
VLLM_ASCEND_ENABLE_NZ -> weight_nz_mode
VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE -> enable_matmul_allreduce
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE -> enable_flashcomm2_parallel_size
VLLM_ASCEND_ENABLE_MLAPO -> enable_mlapo
VLLM_ASCEND_ENABLE_FUSED_MC2 -> enable_fused_mc2
VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL -> enable_context_parallel
VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK -> enable_transpose_kv_cache_by_block
```

原因：

- 均为 vllm-ascend 专属运行时功能开关。
- 不属于 package build 阶段变量。
- 不属于 import-time monkey patch gate。
- 读取点基本都在 `AscendConfig` 初始化之后。

---

## 7. 已更新文件

已更新：

`D:\lzy\code\for_env\result\analysis\env_vars_migration_detail.md`

主要更新内容：

1. 修正错误变量名。
2. 删除 `VLLM_ASCEND_ENABLE_SHARED_EXPERT_DP` 作为 env 迁移项的描述。
3. 补充 `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK`。
4. 增加变量归属判断。
5. 修正迁移建议：`HCCL_SO_PATH` 和 `VLLM_VERSION` 不建议作为普通 vllm-ascend Config 迁移项。
