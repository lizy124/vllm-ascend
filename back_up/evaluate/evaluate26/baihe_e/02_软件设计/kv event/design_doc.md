# Layerwise KV Cache Events 设计文档

## 1. 背景与动机

### 1.1 问题背景

在分布式推理场景下，KV cache events 用于追踪和同步 KV cache 的存储状态。现有的非 layerwise 模式存在以下问题：

- **同步不精确**：每个 chunk 独立生成 event，无法精确追踪最后一层的完成状态
- **事件分散**：多个 event 分散在不同时间点，难以统一管理
- **分布式场景需求**：PD disaggregated 模式下，需要在最后一层精确同步

### 1.2 目标

实现 layerwise KV cache events 支持：
- 只在最后一层生成累积 event，包含所有层的 block 信息
- 确保分布式推理场景的精确同步
- 保持线程安全和内存管理正确性

---

## 2. 设计目标

| 目标 | 说明 |
|-----|------|
| 精确同步 | 在最后一层生成包含所有层信息的累积 event |
| 线程安全 | 多线程环境下保护共享数据结构 |
| 内存管理 | 正确清理 `layerwise_event_starts`，防止内存泄漏 |
| 向后兼容 | 默认关闭 (`use_layerwise=false`)，不影响现有行为 |

---

## 3. 核心概念

### 3.1 Layerwise 模式

**定义**：按 layer 存储 KV cache，而非按 chunk。

| 模式 | 存储粒度 | Event 生成时机 |
|-----|---------|---------------|
| 非 layerwise | 按 chunk | 每个 chunk 生成独立 event |
| layerwise | 按 layer | 只在最后一层生成累积 event |

### 3.2 累积 Event

**定义**：最后一层生成的 `BlockStored` event，包含所有层的 block 信息。

**特点**：
- 前面的层只记录信息到 `layerwise_event_starts`
- 最后一层从 `layerwise_event_starts` 取出所有信息，生成累积 event
- 确保 event 包含完整的 layer 信息

### 3.3 请求生命周期

**流程**：
```
Layer 0 → Layer 1 → ... → Layer N-1 (最后一层)
   ↓         ↓              ↓
  记录      记录         生成累积 event + 清理
```

---

## 4. 数据结构设计

### 4.1 LayerMultiBlockReqMeta

**位置**：[config_data.py:726](file:///D:/lzy/code/for_usability/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L726)

```python
@dataclass(init=False)
class LayerMultiBlockReqMeta:
    # 请求标识
    req_id: str
    
    # 缓存键（每层）
    keys: list[LayerPoolKey]
    
    # Token 位置
    starts: list[int]             # token 起始位置
    ends: list[int]               # token 结束位置
    
    # Block 信息
    block_ids_by_group: list[list[int]]
    block_hashes: list[Any]       # block hash 列表
    
    # Layer 信息
    layer_id: int                 # 当前层 ID
    is_last_chunk: bool | None    # 是否是最后一个 chunk
    
    # KV Event 相关
    token_ids: list[int] | None   # token ID 列表（用于 KV event）
    original_block_size: list[int] | int | None  # 原始 block size
    kv_cache_group_id: int        # KV cache group ID
    
    # 同步
    current_event: torch.npu.Event | None  # NPU 同步事件
```

### 4.2 与 ReqMeta 的对比

| 字段 | ReqMeta (非 layerwise) | LayerMultiBlockReqMeta (layerwise) |
|-----|------------------------|-----------------------------------|
| 存储粒度 | chunk | layer |
| `keys` | `PoolKey` | `LayerPoolKey` |
| `layer_id` | 无 | 有 |
| `starts/ends` | 通过 `process_tokens` 计算 | 直接传入 |
| `is_last_chunk` | 有 | 有 |
| `token_ids` | 有 | 有 |
| `current_event` | 有 | 有 |

---

## 5. 线程架构设计

### 5.1 KVCacheStoreLayerSendingThread

**位置**：[kv_transfer.py](file:///D:/lzy/code/for_usability/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py)

**职责**：按 layer 发送 KV cache 到远程存储

**核心属性**：

```python
class KVCacheStoreLayerSendingThread(KVTransferThread):
    # 最后一层 ID
    self.final_layer_id = num_layers - 1
    
    # 存储步长（用于 TP sharding）
    self.put_step = put_step
    
    # KV event 开关
    self.enable_kv_event = enable_kv_event
    
    # 记录每层的 event start 位置
    self.layerwise_event_starts: dict[str, set[int]] = defaultdict(set)
    
    # 正在处理的请求计数
    self.stored_requests: dict[str, int] = defaultdict(int)
    
    # 线程锁
    self.done_task_lock = threading.Lock()      # 保护 stored_requests
    self.layerwise_event_lock = threading.Lock() # 保护 layerwise_event_starts
```

### 5.2 KVCacheStoreLayerRecvingThread

**职责**：按 layer 从远程存储接收 KV cache

**核心属性**：

```python
class KVCacheStoreLayerRecvingThread(KVTransferThread):
    # 接收完成通知
    self.get_event = get_event
    
    # 失效 block 记录
    self._invalid_block_ids: set[int]
    self._invalid_block_ids_lock = threading.Lock()
```

### 5.3 与非 layerwise 线程的对比

| 特性 | KVCacheStoreSendingThread | KVCacheStoreLayerSendingThread |
|-----|--------------------------|-------------------------------|
| 存储粒度 | chunk | layer |
| Event 生成 | 每个 chunk | 只在最后一层 |
| 数据结构 | `ReqMeta` | `LayerMultiBlockReqMeta` |
| 线程锁 | `done_task_lock` | `done_task_lock` + `layerwise_event_lock` |
| 额外数据 | 无 | `layerwise_event_starts` |

---

## 6. 线程安全设计

### 6.1 锁的使用场景

| 锁 | 保护对象 | 使用场景 |
|---|---------|---------|
| `layerwise_event_lock` | `layerwise_event_starts` | 记录层信息、取出层信息、清理 |
| `done_task_lock` | `stored_requests` | 增加/减少请求计数、删除请求 |
| `completed_events_lock` | `completed_events` | 标记完成事件 |

### 6.2 关锁时机

**记录层信息**：
```python
def _record_layerwise_event_starts(self, req_meta: LayerMultiBlockReqMeta, starts: list[int]):
    with self.layerwise_event_lock:
        self.layerwise_event_starts[req_meta.req_id].update(starts)
```

**取出层信息**：
```python
def _build_stored_events(self, req_meta: LayerMultiBlockReqMeta):
    with self.layerwise_event_lock:
        starts_set = self.layerwise_event_starts.pop(req_meta.req_id, set())
```

**清理请求**：
```python
if layer_id == self.final_layer_id and is_last_chunk:
    with self.layerwise_event_lock:
        self.layerwise_event_starts.pop(req_meta.req_id, None)
    self.dec_stored_request(req_meta.req_id)
    self.set_finished_request(req_meta.req_id)
```

---

## 7. KV Event 生成逻辑

### 7.1 记录层信息

**`_record_layerwise_event_starts()`**：

```python
def _record_layerwise_event_starts(self, req_meta: LayerMultiBlockReqMeta, starts: list[int]):
    """记录当前层的 event start 位置"""
    with self.layerwise_event_lock:
        self.layerwise_event_starts[req_meta.req_id].update(starts)
```

**时机**：每次成功存储 KV cache 后调用

### 7.2 构建累积 Event

**`_build_stored_events()`**：

```python
def _build_stored_events(self, req_meta: LayerMultiBlockReqMeta) -> list[BlockStored]:
    """从 layerwise_event_starts 构建累积 event"""
    stored_events: list[BlockStored] = []
    group_block_size = self._get_block_size(req_meta.kv_cache_group_id)
    new_block_hashes = [maybe_convert_block_hash(bh) for bh in req_meta.block_hashes]
    
    # 取出所有层的 start 位置
    with self.layerwise_event_lock:
        starts_set = self.layerwise_event_starts.pop(req_meta.req_id, set())
    
    # 按顺序生成 BlockStored event
    for start in sorted(starts_set):
        block_idx = start // group_block_size
        if block_idx >= len(new_block_hashes):
            continue
        block_hash = new_block_hashes[block_idx]
        parent_block_hash = new_block_hashes[block_idx - 1] if block_idx > 0 else None
        end = min(start + group_block_size, len(req_meta.token_ids or []))
        token_ids = req_meta.token_ids[start:end] if req_meta.token_ids is not None else None
        
        stored_event = BlockStored(
            block_hashes=[block_hash],
            parent_block_hash=parent_block_hash,
            token_ids=token_ids,
            block_size=block_size,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        stored_events.append(stored_event)
    
    return stored_events
```

### 7.3 只在最后一层生成的原理

**`_handle_request()` 中的逻辑**：

```python
def _handle_request(self, req_meta: LayerMultiBlockReqMeta):
    layer_id = req_meta.layer_id
    
    # ... 存储 KV cache ...
    
    # 记录层信息（所有层）
    self._record_layerwise_event_starts(req_meta, starts)
    
    # 只在最后一层生成 event
    if layer_id == self.final_layer_id:
        stored_events = self._build_stored_events(req_meta)
        if stored_events:
            self.update_kv_event(stored_events)
    
    # 只在最后一层的最后一个 chunk 清理
    if layer_id == self.final_layer_id and is_last_chunk:
        with self.layerwise_event_lock:
            self.layerwise_event_starts.pop(req_meta.req_id, None)
        self.dec_stored_request(req_meta.req_id)
        self.set_finished_request(req_meta.req_id)
```

---

## 8. 请求生命周期管理

### 8.1 请求计数

**增加计数**：
```python
def add_stored_request(self, req_id: str):
    with self.done_task_lock:
        self.stored_requests[req_id] += 1
```

**减少计数**：
```python
def dec_stored_request(self, req_id: str):
    with self.done_task_lock:
        if req_id in self.stored_requests:
            self.stored_requests[req_id] -= 1
```

**删除请求**：
```python
def delete_finished_stored_request(self, req_id: str):
    with self.done_task_lock:
        if req_id in self.stored_requests:
            del self.stored_requests[req_id]
    with self.layerwise_event_lock:
        self.layerwise_event_starts.pop(req_id, None)
```

### 8.2 完成标记

```python
def set_finished_request(self, req_id):
    with self.done_task_lock:
        self.finished_requests.add(req_id)
```

### 8.3 清理逻辑

**时机**：最后一层的最后一个 chunk 处理完成

**清理内容**：
1. `layerwise_event_starts[req_id]` - 防止内存泄漏
2. `stored_requests[req_id]` - 减少计数
3. `finished_requests.add(req_id)` - 标记完成

---

## 9. 配置与使用

### 9.1 配置项

**`use_layerwise`**：在 `KVPoolWorker.__init__` 中传入

```python
class KVPoolWorker:
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwize: bool,  # 启用 layerwise 模式
        kv_cache_config: KVCacheConfig | None = None,
    ):
        self.use_layerwise = use_layerwize
```

### 9.2 行为对比

| 配置 | `use_layerwise=false` | `use_layerwise=true` |
|-----|----------------------|---------------------|
| 存储粒度 | chunk | layer |
| Event 生成 | 每个 chunk | 只在最后一层 |
| 线程类 | `KVCacheStoreSendingThread` | `KVCacheStoreLayerSendingThread` |
| 数据结构 | `ReqMeta` | `LayerMultiBlockReqMeta` |

### 9.3 向后兼容

- **默认值**：`use_layerwise=false`
- **现有行为**：不受影响
- **无 breaking change**：API 和配置不变

---

## 10. 测试覆盖

### 10.1 测试文件

**位置**：`tests/e2e/distributed/ascend_store/test_kv_transfer.py`

### 10.2 测试场景

| 场景 | 说明 |
|-----|------|
| Layerwise 存储 | 验证按 layer 存储 KV cache |
| 累积 Event 生成 | 验证最后一层生成累积 event |
| 线程安全 | 验证多线程环境下的数据一致性 |
| 内存清理 | 验证 `layerwise_event_starts` 正确清理 |

---

## 11. 文件改动总结

| 文件 | 改动行数 | 说明 |
|-----|---------|------|
| `config_data.py` | +14 | 新增 `LayerMultiBlockReqMeta` 数据结构 |
| `kv_transfer.py` | +136 | 新增 `KVCacheStoreLayerSendingThread` 和 `KVCacheStoreLayerRecvingThread` |
| `pool_worker.py` | +12 | 支持 `use_layerwise` 配置和线程初始化 |
| `test_kv_transfer.py` | +69 | 新增测试覆盖 |

---

## 12. 总结

Layerwise KV Cache Events 设计文档实现了 layerwise KV cache events 支持，核心设计：

1. **累积 Event**：只在最后一层生成包含所有层信息的 `BlockStored`
2. **线程安全**：使用 `layerwise_event_lock` 和 `done_task_lock` 保护共享数据
3. **内存管理**：正确清理 `layerwise_event_starts`，防止内存泄漏
4. **向后兼容**：默认关闭，不影响现有行为

该设计为分布式推理场景提供了精确的 KV cache 同步机制。