# vllm-ascend AscendConfig 配置指南

## 背景

vllm-ascend 之前通过环境变量控制 Ascend 平台特有功能开关。从 PR [#9064](https://github.com/vllm-project/vllm-ascend/pull/9064) 起，这些配置统一迁移到 `AscendConfig`，通过 `--additional-config` 参数传入。

**过渡期内旧环境变量仍然生效**，但会在下个版本移除，建议尽快迁移到新方式。

---

## 配置方式对比

### 旧方式：环境变量

```bash
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
vllm serve deepseek-ai/DeepSeek-V3
```

缺点：
- 环境变量散落在启动脚本中，难以统一管理
- 无法与 vllm 的配置体系集成
- 无法在 Python API 中方便地设置

### 新方式：additional-config

**命令行：**

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --additional-config '{"enable_matmul_allreduce": true, "enable_fused_mc2": 1}'
```

**Python API：**

```python
from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    additional_config={
        "enable_matmul_allreduce": True,
        "enable_fused_mc2": 1,
    },
)
```

---

## 优先级规则

当 config 和环境变量同时设置时，优先级为：

```
additional_config > 环境变量 > 默认值
```

即：如果 `additional_config` 中显式设置了某个字段，环境变量的值会被忽略。

---

## 配置项完整列表

### 1. enable_balance_scheduling

| 项目 | 说明 |
|------|------|
| 功能 | 控制 MoE 模型的负载均衡调度策略 |
| 类型 | bool |
| 默认值 | false |
| 约束 | 不能和 `profiling_chunk_config` 同时启用 |

**旧方式：**
```bash
export VLLM_ASCEND_BALANCE_SCHEDULING=1
```

**新方式：**
```bash
vllm serve ... --additional-config '{"enable_balance_scheduling": true}'
```

---

### 2. enable_flashcomm1

| 项目 | 说明 |
|------|------|
| 功能 | 启用 FlashComm1 通信优化（tensor parallel 场景） |
| 类型 | bool |
| 默认值 | false |

**旧方式：**
```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

**新方式：**
```bash
vllm serve ... --additional-config '{"enable_flashcomm1": true}'
```

---

### 3. enable_matmul_allreduce

| 项目 | 说明 |
|------|------|
| 功能 | 启用 MatmulAllReduce 融合核（tensor parallel 场景，A2 支持，eager 模式性能更佳） |
| 类型 | bool |
| 默认值 | false |

**旧方式：**
```bash
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1
```

**新方式：**
```bash
vllm serve ... --additional-config '{"enable_matmul_allreduce": true}'
```

---

### 4. enable_flashcomm2_parallel_size

| 项目 | 说明 |
|------|------|
| 功能 | 启用 FlashComm2 并设置 O 矩阵 TP group size |
| 类型 | int |
| 默认值 | 0（关闭） |
| 取值 | 0 = 关闭，>0 = 开启且值为 parallel size |

**旧方式：**
```bash
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=2
```

**新方式：**
```bash
vllm serve ... --additional-config '{"enable_flashcomm2_parallel_size": 2}'
```

---

### 5. msmonitor_use_daemon

| 项目 | 说明 |
|------|------|
| 功能 | 以守护进程方式启用 msMonitor 性能监控工具 |
| 类型 | bool |
| 默认值 | false |
| 约束 | 与 torch profiler 互斥，不能同时启用 |

**旧方式：**
```bash
export MSMONITOR_USE_DAEMON=1
```

**新方式：**
```bash
vllm serve ... --additional-config '{"msmonitor_use_daemon": true}'
```

---

### 6. enable_mlapo

| 项目 | 说明 |
|------|------|
| 功能 | 启用 DeepSeek W8A8 系列模型的 MLAPO 优化（默认开启，提升性能但消耗更多 NPU 内存） |
| 类型 | bool |
| 默认值 | true |

**旧方式：**
```bash
export VLLM_ASCEND_ENABLE_MLAPO=0
```

**新方式：**
```bash
vllm serve ... --additional-config '{"enable_mlapo": false}'
```

---

### 7. weight_nz_mode

| 项目 | 说明 |
|------|------|
| 功能 | 控制权重格式是否转为 FRACTAL_NZ |
| 类型 | int |
| 默认值 | 1 |
| 取值 | 0 = 关闭 NZ / 1 = 仅量化场景启用 NZ / 2 = 尽可能启用 NZ |

**旧方式：**
```bash
export VLLM_ASCEND_ENABLE_NZ=2
```

**新方式：**
```bash
vllm serve ... --additional-config '{"weight_nz_mode": 2}'
```

> 注意：config 字段名从环境变量的 `ENABLE_NZ` 改为 `weight_nz_mode`，语义更清晰。

---

### 8. enable_context_parallel

| 项目 | 说明 |
|------|------|
| 功能 | 启用上下文并行 (Context Parallelism) |
| 类型 | bool |
| 默认值 | false |

**旧方式：**
```bash
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
```

**新方式：**
```bash
vllm serve ... --additional-config '{"enable_context_parallel": true}'
```

---

### 9. enable_fused_mc2

| 项目 | 说明 |
|------|------|
| 功能 | 控制 MoE 通信的融合算子选择 |
| 类型 | int |
| 默认值 | 0 |
| 取值 | 0 = 默认 ALLTOALL/MC2 / 1 = 使用 dispatch_ffn_combine / 2 = 使用 dispatch_gmm_combine_decode |
| 约束 | 值为 1 时：仅适用于 W8A8、EP≤32、非 MTP、非 dynamic-eplb |
| 约束 | 值为 2 时：仅适用于 decode 节点 W8A8，且 MTP 层必须 W8A8 |

**旧方式：**
```bash
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
```

**新方式：**
```bash
vllm serve ... --additional-config '{"enable_fused_mc2": 1}'
```

---

### 10. enable_transpose_kv_cache_by_block

| 项目 | 说明 |
|------|------|
| 功能 | 控制 KV cache 是否使用 fused transpose by block 算子 |
| 类型 | bool |
| 默认值 | true |

**旧方式：**
```bash
export VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK=0
```

**新方式：**
```bash
vllm serve ... --additional-config '{"enable_transpose_kv_cache_by_block": false}'
```

---

## 迁移速查表

| 旧环境变量 | 新 config 字段 | 类型转换 |
|-----------|---------------|---------|
| `VLLM_ASCEND_BALANCE_SCHEDULING=1` | `"enable_balance_scheduling": true` | `"1"` → `true` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | `"enable_flashcomm1": true` | `"1"` → `true` |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1` | `"enable_matmul_allreduce": true` | `"1"` → `true` |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=2` | `"enable_flashcomm2_parallel_size": 2` | 整数不变 |
| `MSMONITOR_USE_DAEMON=1` | `"msmonitor_use_daemon": true` | `"1"` → `true` |
| `VLLM_ASCEND_ENABLE_MLAPO=0` | `"enable_mlapo": false` | `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_NZ=2` | `"weight_nz_mode": 2` | 整数不变，字段名变更 |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1` | `"enable_context_parallel": true` | `"1"` → `true` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2=1` | `"enable_fused_mc2": 1` | 整数不变 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK=0` | `"enable_transpose_kv_cache_by_block": false` | `"0"` → `false` |

> 类型转换说明：环境变量值都是字符串，旧代码中 bool 类型通过 `bool(int("1"))` 转换。新 config 中直接使用 Python 原生类型，bool 用 `true`/`false`，int 直接用数字。

---

## 常见配置示例

### DeepSeek-V3 W8A8 量化 + Fused MC2

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --additional-config '{"enable_fused_mc2": 1, "enable_mlapo": true}'
```

### 启用 FlashComm2 + 上下文并行

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --additional-config '{"enable_flashcomm2_parallel_size": 2, "enable_context_parallel": true}'
```

### 负载均衡调度

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --additional-config '{"enable_balance_scheduling": true}'
```

### 多项组合配置

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --additional-config '{
    "enable_matmul_allreduce": true,
    "enable_flashcomm2_parallel_size": 4,
    "enable_fused_mc2": 1,
    "enable_mlapo": true,
    "weight_nz_mode": 2
  }'
```

### Python API 多项配置

```python
from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
    additional_config={
        "enable_matmul_allreduce": True,
        "enable_flashcomm2_parallel_size": 4,
        "enable_fused_mc2": 1,
        "enable_mlapo": True,
        "weight_nz_mode": 2,
    },
)
```

---

## 仍保留为环境变量的配置

以下配置不适合迁移到 `additional_config`，继续使用环境变量：

| 环境变量 | 用途 |
|----------|------|
| `HCCL_SO_PATH` | HCCL 库路径，属于底层运行时配置 |
| `VLLM_VERSION` | vLLM 包版本选择，属于构建时配置 |

---

## 过渡期说明

- 当前版本中，如果只设置了环境变量而未设置 `additional_config`，环境变量值仍然生效
- 使用环境变量时，日志中会出现类似以下警告：

```
AscendConfig.enable_fused_mc2 falls back to environment variable VLLM_ASCEND_ENABLE_FUSED_MC2 with value 1.
Please use additional_config.enable_fused_mc2 instead, because VLLM_ASCEND_ENABLE_FUSED_MC2 will be removed in the next release.
```

- **下个版本将移除环境变量支持**，请尽快迁移到 `additional_config` 方式
