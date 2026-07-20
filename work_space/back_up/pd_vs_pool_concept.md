# PD 分离与池化：概念与代码层面的区分

## 一、目录层面的证据

`vllm_ascend/distributed/kv_transfer/` 下有两个平级子目录，这就是最直接的区分：

```
kv_transfer/
├── kv_p2p/          ← PD 分离
│   ├── mooncake_connector.py
│   ├── mooncake_layerwise_connector.py
│   └── mooncake_hybrid_connector.py
└── kv_pool/         ← 池化
    ├── ascend_store/        (AscendStoreConnector)
    ├── simple_cpu_offload/  (SimpleCPUOffloadConnector)
    ├── recompute_cpu_offload/ (RecomputeCPUOffloadConnector)
    ├── lmcache_ascend_connector.py
    └── ucm_connector.py
```

两者是 `kv_transfer` 下的两个独立子模块，职责不同，没有父子或从属关系。

## 二、PD 分离（kv_p2p）：prefill → decode 直传

### 2.1 核心机制

PD 分离的本质是：**prefill 节点算完 KV cache 后，直接通过网络传给 decode 节点**。这是一个"生产者→消费者"的直传模型。

### 2.2 代码证据

**底层传输引擎：**

```python
# kv_p2p/mooncake_connector.py:25
from mooncake.engine import TransferEngine
```

三个 connector 都依赖 `mooncake.engine.TransferEngine` 做底层 P2P 传输。

**角色分工：producer 和 consumer：**

```python
# kv_p2p/mooncake_connector.py:2327
if self.kv_role == "kv_producer":
    self.kv_send_thread = KVCacheSendingThread(...)  # producer 发送 KV

# kv_p2p/mooncake_connector.py:2382
if self.kv_role == "kv_consumer":
    self.kv_recv_thread = KVCacheRecvingThread(...)  # consumer 接收 KV
```

`kv_producer` 跑在 prefill 节点，`kv_consumer` 跑在 decode 节点。producer 算完一层就发一层，consumer 收一层用一层。

**三种 connector 的差异：**

| Connector | 源文件 | 差异 |
|---|---|---|
| `MooncakeConnectorV1` | `mooncake_connector.py` | 按 request 粒度传输，一批请求算完后一起发 |
| `MooncakeLayerwiseConnector` | `mooncake_layerwise_connector.py` | 按 layer 粒度逐层传输，算完一层立即发一层，延迟更低 |
| `MooncakeHybridConnector` | `mooncake_hybrid_connector.py` | 继承 V1，但 `use_hybrid=True`（L1220），处理 MLA/Full Attention 混合 block size |

三者都是 P2P 直传，只是传输粒度或 block 管理策略不同。

### 2.3 一句话总结

> **PD 分离 = prefill 节点算出来 → 直接发给 decode 节点 → decode 节点用。**
> 
> 数据流是单向的、一次性的：producer 发完就完，consumer 用完就丢。

## 三、池化（kv_pool）：存储 → 查找 → 加载

### 3.1 核心机制

池化的本质是：**KV cache 先存到一个"池子"里，后续请求通过 key 查找命中，再加载回来复用**。这是一个"存储→查找→加载"的模型。

### 3.2 代码证据

**池化专用组件：**

```python
# kv_pool/ascend_store/ascend_store_connector.py:32-36
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,    # 池化调度器
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker         # 池化 worker
)
```

`KVPoolScheduler` 和 `KVPoolWorker` 是池化独有的，kv_p2p 中没有。

**存储（Save）：**

```python
# kv_pool/ascend_store/ascend_store_connector.py:224-233
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
    if not self.use_layerwise:
        return
    if self.kv_role == "kv_consumer":
        return  # consumer 不保存
    self.connector_worker.save_kv_layer(self._get_connector_metadata())
```

```python
# kv_pool/ascend_store/pool_worker.py:1399
def save_kv_layer(self, connector_metadata):
    ...  # 将 KV 保存到池中
```

**加载（Load）：**

```python
# kv_pool/ascend_store/ascend_store_connector.py:200-217
def start_load_kv(self, forward_context, **kwargs):
    metadata = self._get_connector_metadata()
    # 日志中可以看到 load_spec 信息：
    #   can_load      - 是否可以加载
    #   vllm_cached_tokens    - vllm 自身已缓存的 token 数
    #   kvpool_cached_tokens  - 池中已缓存的 token 数
    self.connector_worker.start_load_kv(metadata)
```

**查找（Lookup）：**

```python
# kv_pool/ascend_store/pool_scheduler.py:1108
def lookup(self, ...):
    ...  # 按 block hash 查找池中是否已有缓存
```

池化通过 `LookupKeyServer`/`LookupKeyClient` 做 key 查找，判断哪些 block 已缓存。

**外部缓存池：**

```python
# kv_pool/ascend_store/coordinator.py:27
class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by external AscendStore key existence."""
```

`ExternalCachedBlockPool` 是一个"外部缓存的 block pool"，它通过检查 AscendStore 中的 key 是否存在来判断 block 是否命中——这是池化的核心数据结构。

### 3.3 五个池化 connector

| Connector | 源文件 | 用途 | 外部依赖 |
|---|---|---|---|
| `AscendStoreConnector` | `ascend_store/ascend_store_connector.py` | 基于 Mooncake 后端的 KV pool 存储/复用 | Mooncake 服务 |
| `SimpleCPUOffloadConnector` | `simple_cpu_offload/simple_cpu_offload_connector.py` | NPU 适配的 CPU KV offload，继承上游 `SimpleCPUOffloadConnector`（L14-15） | 无 |
| `RecomputeCPUOffloadConnector` | `recompute_cpu_offload/recompute_cpu_offload_connector.py` | CPU KV cache 保存，用于重计算被抢占的请求（L44 docstring） | 无 |
| `LMCacheAscendConnector` | `lmcache_ascend_connector.py` | 封装上游 `LMCacheConnectorV1`（L2-3: `import lmcache_ascend`） | `lmcache_ascend` 库 |
| `UCMConnector` | `ucm_connector.py` | 统一缓存管理，继承外部 `ucm.integration.vllm.ucm_connector.UCMConnector`（L4） | `ucm` 库 + UCM 服务 |

### 3.4 一句话总结

> **池化 = 先存到池子里 → 后续请求按 key 查找 → 命中就加载复用。**
> 
> 数据流是"存-查-取"的循环：同一个 KV block 可能被多个请求命中加载。

## 四、关键区别对照

| 维度 | PD 分离（kv_p2p） | 池化（kv_pool） |
|---|---|---|
| 目录 | `kv_p2p/` | `kv_pool/` |
| 核心组件 | `TransferEngine`（Mooncake P2P） | `KVPoolScheduler` + `KVPoolWorker` + `ExternalCachedBlockPool` |
| 角色 | `kv_producer`（发） / `kv_consumer`（收） | `kv_both`（既存又取） |
| 数据流 | producer → consumer（单向一次性） | store → pool → lookup → load（循环复用） |
| 关键方法 | `send_kv` / `recv_kv`（P2P 传输） | `save_kv_layer` / `start_load_kv` / `lookup`（存储/加载/查找） |
| 是否跨节点 | 是（prefill 节点 → decode 节点） | 否（单节点内，或通过 Mooncake 后端在单节点内共享） |
| 典型场景 | 长序列 prefill 与 decode 资源分离 | KV cache 跨请求复用，减少重复计算 |

## 五、常见误解

1. **"PD 分离和池化是一回事"** —— 错。一个在 `kv_p2p/`，一个在 `kv_pool/`，是两套独立代码。
2. **"PD 分离是池化的一种"** —— 错。PD 分离是直传，池化是存储复用，数据流完全不同。
3. **"池化需要 PD 分离"** —— 错。`AscendStoreConnector` 单节点 `kv_both` 模式，不需要 PD 分离。
4. **"`pull_request/one_card/pooling/*` 是池化"** —— 错。那是 embedding/classification 的 pooling runner（表示层池化），与 KV cache 池化无关。
5. **"多节点就是 PD 分离"** —— 错。多节点 YAML 可能是纯分布式部署（如 `DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml`），不含 `disaggregated_prefill` 字段。