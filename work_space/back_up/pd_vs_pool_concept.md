# PD 分离与池化：概念与代码层面的区分

> **参考文档：** 官方设计文档 [KV_Cache_Pool_Guide.md](../docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md) / [disaggregated_prefill.md](../docs/source/developer_guide/Design_Documents/disaggregated_prefill.md)

## 零、先搞清楚：Mooncake 是什么

Mooncake 是底层基础设施，PD 分离和池化都用它，但用的是不同能力：

| | PD 分离用的 Mooncake 能力 | 池化用的 Mooncake 能力 |
|---|---|---|
| **组件** | `TransferEngine`（P2P 传输引擎） | Mooncake Store（KV 存储 + master 服务） |
| **用途** | P2P 直传，不落地 | 持久化存储，按 key 查找加载 |
| **数据生命周期** | 一次性传输，用完即丢 | 持久化，跨请求复用 |

**两者可以共存：** 通过 `MultiConnector` 同时启用池化 + PD 分离——池化负责 prefix cache 复用，PD 分离负责 P2P 传输。

---

## 一、目录层面的证据

`vllm_ascend/distributed/kv_transfer/` 下有两个平级子目录：

```
kv_transfer/
├── kv_p2p/          ← PD 分离（prefill→decode 直传）
│   ├── mooncake_connector.py              (MooncakeConnectorV1)
│   ├── mooncake_layerwise_connector.py    (MooncakeLayerwiseConnector)
│   └── mooncake_hybrid_connector.py       (MooncakeHybridConnector)
└── kv_pool/         ← 池化（存储→查找→复用）
    ├── ascend_store/                      (AscendStoreConnector)
    ├── simple_cpu_offload/                (SimpleCPUOffloadConnector)
    ├── recompute_cpu_offload/             (RecomputeCPUOffloadConnector)
    ├── lmcache_ascend_connector.py        (LMCacheAscendConnector)
    └── ucm_connector.py                   (UCMConnector)
```

两者是 `kv_transfer` 下的两个独立子模块，职责不同，没有父子或从属关系。

---

## 二、PD 分离（kv_p2p）：prefill → decode 直传

### 2.1 核心机制

PD 分离的本质是：**prefill 节点算完 KV cache 后，直接 P2P 传给 decode 节点**。数据流是单向的、一次性的：producer 发完就完，consumer 用完就丢。

### 2.2 架构

```
外部请求 → 全局 Proxy
              │
              ├─→ Prefill 节点 (kv_producer)  ─┐
              │     计算 KV cache               │ P2P 直传 KV cache
              │                                 │ (不经过 Proxy)
              └─→ Decode 节点 (kv_consumer)  ←─┘
                    接收 KV cache 后 decode
```

### 2.3 两种传输模式

| 模式 | Connector | 工作原理 |
|---|---|---|
| **Pull**（D 拉取） | `MooncakeConnectorV1` | Proxy 路由到 P 完成 prefill → D 节点主动从 P 拉取 KV cache |
| **Push**（P 推送） | `MooncakeLayerwiseConnector` | P 逐层算完一层立即推送给 D → D 逐层接收后开始 decode，延迟更低 |

### 2.4 代码证据

**底层传输引擎：**

```python
# kv_p2p/mooncake_connector.py:25
from mooncake.engine import TransferEngine
```

三个 connector 都依赖 `mooncake.engine.TransferEngine` 做底层 P2P 传输。

**角色分工：**

```python
# kv_p2p/mooncake_connector.py:2327
if self.kv_role == "kv_producer":
    self.kv_send_thread = KVCacheSendingThread(...)  # producer 发送 KV

# kv_p2p/mooncake_connector.py:2382
if self.kv_role == "kv_consumer":
    self.kv_recv_thread = KVCacheRecvingThread(...)  # consumer 接收 KV
```

`kv_producer` 跑在 prefill 节点，`kv_consumer` 跑在 decode 节点。

### 2.5 三种 connector 的差异

| Connector | 传输模式 | 差异 |
|---|---|---|
| `MooncakeConnectorV1` | Pull | 按 request 粒度传输，一批请求算完后一起发 |
| `MooncakeLayerwiseConnector` | Push | 按 layer 粒度逐层传输，算完一层立即发一层，延迟更低 |
| `MooncakeHybridConnector` | Pull | 继承 V1，处理 MLA/Full Attention 混合 block size 的模型（如 DeepSeek-V4） |

### 2.6 一句话总结

> **PD 分离 = prefill 节点算出来 → P2P 直接发给 decode 节点 → decode 节点用。**
> 
> 数据流是单向的、一次性的，不持久化。

---

## 三、池化（kv_pool）：存储 → 查找 → 加载

### 3.1 核心机制

池化的本质是：**KV cache 先存到一个"池子"里，后续请求通过 key 查找命中，再加载回来复用**。数据流是"存→查→取"的循环，同一个 KV block 可能被多个请求命中加载。

### 3.2 两种部署模式

| 部署模式 | kv_role | 说明 |
|---|---|---|
| **PD-Mixed** | `kv_both` | 单实例自己存自己取，池子作为共享 prefix cache |
| **PD 分离 + 池化** | `kv_producer` / `kv_consumer` | P 节点存入池子，D 节点从池子加载，通过 `MultiConnector` 组合 `MooncakeConnectorV1` + `AscendStoreConnector` |

### 3.3 代码证据

**池化专用组件（kv_p2p 中没有）：**

```python
# kv_pool/ascend_store/ascend_store_connector.py:32-36
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,    # 池化调度器
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker         # 池化 worker
)
```

**存储（Save）：**

```python
# kv_pool/ascend_store/ascend_store_connector.py:224-233
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
    self.connector_worker.save_kv_layer(self._get_connector_metadata())
```

**加载（Load）：**

```python
# kv_pool/ascend_store/ascend_store_connector.py:200-217
def start_load_kv(self, forward_context, **kwargs):
    metadata = self._get_connector_metadata()
    self.connector_worker.start_load_kv(metadata)
```

**查找（Lookup）：**

```python
# kv_pool/ascend_store/pool_scheduler.py:1108
def lookup(self, ...):
    ...  # 按 block hash 查找池中是否已有缓存
```

**外部缓存池：**

```python
# kv_pool/ascend_store/coordinator.py:27
class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by external AscendStore key existence."""
```

### 3.4 五个池化 connector

| Connector | 后端 | 部署模式 | 外部依赖 |
|---|---|---|---|
| `AscendStoreConnector` | Mooncake Store / Memcache / Yuanrong | `kv_both`、`kv_producer`/`kv_consumer` | Mooncake/Memcache 服务 |
| `SimpleCPUOffloadConnector` | CPU DRAM | `kv_both` | 无 |
| `RecomputeCPUOffloadConnector` | CPU DRAM | `kv_both` | 无 |
| `LMCacheAscendConnector` | LMCache 后端 | `kv_both` | `lmcache_ascend` 库 |
| `UCMConnector` | UCM | `kv_both` | `ucm` 库 + UCM 服务 |

> `AscendStoreConnector` 还支持 **Layerwise 模式**（`use_layerwise: true`），以逐层方式 save/load KV cache，减少首 token 延迟。当前仅支持 Memcache 后端。

### 3.5 一句话总结

> **池化 = 先存到池子里 → 后续请求按 key 查找 → 命中就加载复用。**
> 
> 数据流是"存→查→取"的循环，KV cache 持久化在池中，跨请求复用。

---

## 四、关键区别对照

| 维度 | PD 分离（kv_p2p） | 池化（kv_pool） |
|---|---|---|
| 本质 | KV cache **传输** | KV cache **存储** |
| 目录 | `kv_p2p/` | `kv_pool/` |
| 核心组件 | `TransferEngine`（Mooncake P2P 传输引擎） | `KVPoolScheduler` + `KVPoolWorker` + `ExternalCachedBlockPool` |
| 角色 | `kv_producer`（发） / `kv_consumer`（收） | `kv_both`（既存又取） |
| 数据流 | producer → consumer（单向，一次性） | store → pool → lookup → load（循环，复用） |
| 数据生命周期 | 不持久化，用完即丢 | 持久化在池中（DRAM/SSD），跨请求复用 |
| 关键方法 | `send_kv` / `recv_kv` | `save_kv_layer` / `start_load_kv` / `lookup` |
| Mooncake 作用 | P2P Transfer Engine（RDMA 直传） | Mooncake Store（持久化存储 + master 服务） |
| 可共存 | 可以！通过 `MultiConnector` 同时启用 |

---

## 五、常见误解

1. **"PD 分离和池化是一回事"** —— 错。一个在 `kv_p2p/`，一个在 `kv_pool/`，是两套独立代码，本质不同（传输 vs 存储）。

2. **"PD 分离是池化的一种"** —— 错。PD 分离是直传（不持久化），池化是存储复用（持久化），数据流完全不同。

3. **"池化不能跨节点"** —— 错。`AscendStoreConnector` 通过 Mooncake Store 后端，可以跨节点存取 KV cache。PD 分离 + 池化部署模式下，P 节点存入池子，D 节点从池子加载。

4. **"池化需要 PD 分离"** —— 错。`AscendStoreConnector` 在 PD-Mixed 模式（`kv_both`）下单节点独立运行，不需要 PD 分离。

5. **"`pull_request/one_card/pooling/*` 是 KV cache 池化"** —— 错。那是 embedding/classification 的 pooling runner（表示层池化，对向量做平均/最大池化），与 KV cache 池化无关。

6. **"多节点就是 PD 分离"** —— 错。多节点 YAML 可能是纯分布式部署（如 `DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml`），不含 `disaggregated_prefill` 字段。

---

## 六、附录：Mooncake 如何做到 "不存储，直接传输"？

PD 分离中，Mooncake 的 P2P 直传基于 **RDMA 单边读**，不经过任何中间存储：

```
P 节点 (NPU 显存)                          D 节点 (NPU 显存)
     │                                          │
     │ ① initialize("P2PHANDSHAKE", "ascend")   │
     │    register_memory(ptr, size)             │
     │    把 NPU 显存地址注册到 TransferEngine   │
     │                                          │
     │ ② 通过 ZMQ side channel 发送：            │
     │    - session_id（远端内存句柄）             │
     │    - block 内存布局信息                    │
     │                                          │
     │                                   ③ batch_transfer_sync_read(
     │                                          session_id,     ← 远端内存句柄
     │                                          src_ptrs,       ← 远端 NPU 地址
     │                                          dst_ptrs,       ← 本地 NPU 地址
     │                                          lengths         ← 数据长度
     │                                      )
     │                                          │
     │     ╔════════ RDMA 单边读 ═══════════╗     │
     │     ║ P 节点 NPU 显存 ──────────→ D 节点 NPU 显存
     │     ║ 不经过 CPU，不经过磁盘       ║
     │     ╚══════════════════════════════╝
     │                                          │
     │ ④ ZMQ 通知 DONE_RECVING                   │
     │    → P 释放 KV cache                      │
```

**关键代码：**

```python
# mooncake_transfer_engine.py — 初始化传输引擎
self.transfer_engine = TransferEngine()
self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
self.transfer_engine.register_memory(ptr, size)  # 注册 NPU 显存，暴露给远端

# mooncake_connector.py — D 节点直接读 P 节点的 NPU 显存
session_id = f"{remote_host}:{remote_transfer_port}"
self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
```

**核心要点：**

- `register_memory(ptr, size)` 把本地 NPU 显存地址注册到 Mooncake TransferEngine，使其可被远端访问
- `session_id` 是远端内存的句柄（`host:port` 格式），携带它就能访问远端已注册的内存
- `batch_transfer_sync_read` 是 **RDMA 单边读**（one-sided RDMA read）——D 节点像读自己本地内存一样直接读 P 节点的 NPU 显存
- 全程数据不落盘、不经过 CPU，从 NPU 到 NPU 直通
- 传输完成后，P 节点释放 KV cache，不持久化任何数据

这就是为什么叫 "不存储，直接传输"——Mooncake 的 TransferEngine 提供的是对远端 NPU 显存的直接 RDMA 访问能力，数据从一块 NPU 显存直通到另一块 NPU 显存，不存在 "先存到某个地方再取出来" 的中间环节。