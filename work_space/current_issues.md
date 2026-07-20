# Nightly 池化（kv_pool）用例缺口分析

## 一、池化 Connector 全景

代码中 `kv_transfer` 下有两个子目录，对应不同职责：

| 目录 | 职责 | 典型 connector |
|---|---|---|
| `kv_p2p/` | PD 分离的 KV 传输（prefill → decode 跨节点） | MooncakeConnectorV1、MooncakeLayerwiseConnector、MooncakeHybridConnector |
| `kv_pool/` | 池化：KV cache 存储/复用（单节点或跨请求） | AscendStoreConnector、SimpleCPUOffload、RecomputeCPUOffload 等 |

本文档只关注 **`kv_pool/`（池化）**。

`vllm_ascend/distributed/kv_transfer/__init__.py` 注册了 5 个池化 connector：

| Connector | 注册名 | 源文件 | 用途 |
|---|---|---|---|
| `AscendStoreConnector` | `MooncakeConnectorStoreV1` / `AscendStoreConnector` | `kv_pool/ascend_store/` | 基于 Mooncake 后端的 KV pool 存储/复用 |
| `LMCacheAscendConnector` | `LMCacheAscendConnector` | `kv_pool/lmcache_ascend_connector.py` | 封装上游 `LMCacheConnectorV1`，需 `lmcache_ascend` 外部库 |
| `UCMConnector` | `UCMConnector` | `kv_pool/ucm_connector.py` | 统一缓存管理，需 `ucm` 外部库 |
| `SimpleCPUOffloadConnector` | `SimpleCPUOffloadConnector` | `kv_pool/simple_cpu_offload/` | NPU 适配的 CPU KV offload，无外部依赖 |
| `RecomputeCPUOffloadConnector` | `RecomputeCPUOffloadConnector` | `kv_pool/recompute_cpu_offload/` | 重计算 + CPU offload，无外部依赖 |

## 二、Nightly 覆盖情况

| Connector | 现有测试 | 所在位置 | Nightly 状态 |
|---|---|---|---|
| `AscendStoreConnector` | `test_qwen3_30b_acc.py` | `weekly/`（`a3.multi_card` 调度） | **已覆盖** |
| `SimpleCPUOffloadConnector` | `test_simple_cpu_offload.py` | `pull_request/one_card/` | **仅 PR 级别** |
| `RecomputeCPUOffloadConnector` | `test_recompute_cpu_offload.py` | `tests/ut/` | **仅 UT** |
| `LMCacheAscendConnector` | 无 | — | **无覆盖** |
| `UCMConnector` | 无 | — | **无覆盖** |

## 三、缺口分析

### 3.1 `SimpleCPUOffloadConnector` — 建议补充到 nightly

**现状：** 已有 PR 级别的 e2e 测试（`pull_request/one_card/test_simple_cpu_offload.py`），模型 `Qwen/Qwen3-0.6B`，1 卡，测试 KV cache CPU offload 的精度和稳定性。

**为什么不直接在 nightly 跑：** 该测试已存在但只在 PR 触发，不在 nightly 看护。

**建议：** 在 `nightly_config.yaml` 的 `a3.multi_card`（或其他 1 卡位置）新增一条，将已有的 `test_simple_cpu_offload.py` 纳入 nightly 看护。

### 3.2 `RecomputeCPUOffloadConnector` — 建议补充 e2e 测试

**现状：** 仅有 UT（`tests/ut/kv_offload/test_recompute_cpu_offload.py`），没有 e2e 测试。

**影响：** 重计算 + CPU offload 的完整链路（scheduler → worker → 重计算 → CPU 读写）在 e2e 层面完全未覆盖。

**建议：** 参考 `test_simple_cpu_offload.py` 的写法，新增 e2e 测试，验证 offload 后的精度一致性。

### 3.3 `LMCacheAscendConnector` — 暂不补充

**原因：** 依赖外部库 `lmcache_ascend`，nightly 环境未必具备依赖。需确认环境支持后再评估。

### 3.4 `UCMConnector` — 暂不补充

**原因：** 依赖外部 `ucm` 库和 UCM 服务端，nightly 环境不具备。需环境就绪后再评估。

## 四、建议补充的用例

| 优先级 | Connector | 改动 | 备注 |
|---|---|---|---|
| P0 | `SimpleCPUOffloadConnector` | nightly_config 新增 1 条（复用已有测试） | 改动最小，已有测试直接用 |
| P1 | `RecomputeCPUOffloadConnector` | 新增 e2e 测试文件 + nightly_config 注册 | 需新写测试，参考 SimpleCPUOffload |
| P2 | `LMCacheAscendConnector` | 待环境就绪 | 依赖外部库 |
| P2 | `UCMConnector` | 待环境就绪 | 依赖外部服务 |

## 五、验证方式

在 PR 评论区发送 `/nightly <case_name>` 触发对应用例。