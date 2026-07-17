# PR #6593 与 PR #9468 详细对比分析

## 文档概述

本文档详细对比了 vLLM Ascend 项目中两个关于 KV Cache 事件的重要 PR：
- **PR #6593**: [Misc] gen kv events in ascendconnector (2026 年 2 月)
- **PR #9468**: [Feature] Support layerwise KV cache events (2026 年 5 月)

这两个 PR 共同构成了 vLLM Ascend 的 KV Cache 事件系统，支持从简单到复杂的不同应用场景。

### 核心理解（必读！）

**PR #9468 的真实目的**：
> ❌ **不是为了上报更多信息！**
> ✅ **而是为了让 layerwise 模式也支持 KV event 上报功能！**

**演进历程**：
- **PR #6593 之前**：两种模式都不支持 KV event 上报
- **PR #6593 之后**：非 layerwise 模式 ✅ 支持，layerwise 模式 ❌ 仍然不支持
- **PR #9468 之后**：两种模式 ✅ 都支持 KV event 上报！

**一句话总结**：PR #9468 的核心贡献是让 layerwise 存储模式也能用上 KV event 上报功能！

---

## 一、PR #6593: 基础 KV 事件框架

### 1.1 基本信息

| 属性 | 值 |
|------|-----|
| **标题** | [Misc] gen kv events in ascendconnector |
| **作者** | yejj710 |
| **合并时间** | 2026 年 2 月 12 日 |
| **vLLM 版本** | v0.15.0 |
| **代码改动** | 171 行新增，5 行删除 |
| **PR 链接** | https://github.com/vllm-project/vllm-ascend/pull/6593 |

### 1.2 核心目标

在 AscendConnector 中实现完整的 KV Cache 事件生成和发布机制，使 vLLM Ascend 能够适配 vLLM 上游的 KV Cache 事件发布流程。

**参考 Issue**: [#6391](https://github.com/vllm-project/vllm-ascend/issues/6391) - [RFC]: Adapt KV Cache Events Publishing Mechanism

### 1.3 主要改动

#### 1.3.1 新增 AscendStoreKVEvents 类

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`

```python
class AscendStoreKVEvents(KVConnectorKVEvents):
    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "AscendStoreKVEvents":
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()
```

**功能**:
- 实现 `KVConnectorKVEvents` 接口（与 vLLM 上游的 `LMCacheKVEvents` 实现完全一致）
- 使用 `KVEventAggregator` 聚合多 worker 的事件
- 提供事件的添加、聚合、获取功能

#### 1.3.2 Connector 接口实现

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`

```python
def update_connector_output(self, connector_output: KVConnectorOutput):
    # Get the KV events
    kv_cache_events = connector_output.kv_cache_events
    if not kv_cache_events or not isinstance(kv_cache_events, AscendStoreKVEvents):
        return

    if self._kv_cache_events is None:
        self._kv_cache_events = kv_cache_events
    else:
        self._kv_cache_events.add_events(kv_cache_events.get_all_events())
        self._kv_cache_events.increment_workers(kv_cache_events.get_number_of_workers())
    return

def take_events(self) -> Iterable["KVCacheEvent"]:
    if self._kv_cache_events is not None:
        self._kv_cache_events.aggregate()
        kv_cache_events = self._kv_cache_events.get_all_events()
        yield from kv_cache_events
        self._kv_cache_events.clear_events()
        self._kv_cache_events = None

def get_kv_connector_kv_cache_events(self) -> AscendStoreKVEvents | None:
    events = self.connector_worker.get_kv_events()
    if not events:
        return None

    ascend_store_kv_events = AscendStoreKVEvents(num_workers=1)
    ascend_store_kv_events.add_events(events)
    return ascend_store_kv_events
```

#### 1.3.3 元数据扩展

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`

在 `ReqMeta` 中新增字段：
- `token_ids`: 令牌 ID 列表
- `original_block_size`: 原始 block 大小

#### 1.3.4 事件生成逻辑

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

在 `KVCacheStoreSendingThread._handle_request()` 中生成 `BlockStored` 事件：

```python
for index, start in enumerate(starts):
    # ... 处理 KV 存储 ...
    
    # Create KV event
    if self.enable_kv_event:
        token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
        block_size = (
            req_meta.original_block_size[group_id]
            if isinstance(req_meta.original_block_size, list)
            else req_meta.original_block_size
        )
        stored_event = BlockStored(
            block_hashes=[new_block_hashes[index]],
            parent_block_hash=prev_key,
            token_ids=token_ids,
            block_size=block_size,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        stored_events.append(stored_event)
        prev_key = new_block_hashes[index]

if self.enable_kv_event and stored_events:
    self.update_kv_event(stored_events)
```

### 1.4 工作流程

```
1. 请求到达 → 构建 ReqMeta (包含 token_ids, original_block_size)
   ↓
2. KVCacheStoreSendingThread 处理请求
   ↓
3. 存储 KV Cache 到后端 (Mooncake/Memcache)
   ↓
4. 如果 enable_kv_event=True → 生成 BlockStored 事件
   ↓
5. 事件存储到 kv_events 列表
   ↓
6. Scheduler 通过 get_kv_connector_kv_cache_events() 收集事件
   ↓
7. Scheduler 通过 take_events() 获取聚合后的事件
   ↓
8. 事件传递给 vLLM 的 kv_cache_manager
```

---

## 二、PR #9468: 分层 KV 事件增强

### 2.1 基本信息

| 属性 | 值 |
|------|-----|
| **标题** | [Feature] Support layerwise KV cache events |
| **作者** | lizy124 |
| **创建时间** | 2026 年 5 月 22 日 |
| **vLLM 版本** | v0.20.2 |
| **状态** | Open (待合并) |
| **PR 链接** | https://github.com/vllm-project/vllm-ascend/pull/9468 |

### 2.2 核心目标

支持分层 KV Cache 事件，在多层模型架构中，只在处理完所有层后（最后一层）才生成和发布 KV Cache 事件，确保事件的完整性和准确性。

### 2.3 主要改动

#### 2.3.1 增强 KVCacheStoreLayerSendingThread 类（重要修正）

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

**注意**: `KVCacheStoreLayerSendingThread` 类不是本 PR 新增的！
- 该类在 PR #9468 之前就已存在
- 本 PR 是在该类中添加分层事件追踪机制

**新增属性**：
```python
class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        enable_kv_event: bool = False,
    ):
        super().__init__(...)
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.enable_kv_event = enable_kv_event
        
        # ⭐ PR #9468 新增：分层事件追踪属性 ⭐
        self.layerwise_event_starts: dict[str, set[int]] = defaultdict(set)
        self.stored_requests: dict[str, int] = defaultdict(int)
        self.done_task_lock = threading.Lock()
```

#### 2.3.2 新增请求追踪方法

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

```python
def add_stored_request(self, req_id: str):
    with self.done_task_lock:
        self.stored_requests[req_id] += 1

def dec_stored_request(self, req_id: str):
    with self.done_task_lock:
        if req_id in self.stored_requests:
            self.stored_requests[req_id] -= 1

def delete_finished_stored_request(self, req_id: str):
    with self.done_task_lock:
        if req_id in self.stored_requests:
            del self.stored_requests[req_id]
    self.layerwise_event_starts.pop(req_id, None)
```

#### 2.3.3 新增分层事件追踪机制

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

**记录每层的 missing blocks**：
```python
def _record_layerwise_event_starts(self, req_meta: LayerMultiBlockReqMeta, starts: list[int]) -> None:
    if self.enable_kv_event:
        self.layerwise_event_starts[req_meta.req_id].update(starts)
```

**只在最后一层构建事件**（核心方法）：
```python
def _build_stored_events(self, req_meta: LayerMultiBlockReqMeta) -> list[BlockStored]:
    if not self.enable_kv_event or req_meta.layer_id != self.final_layer_id:
        return []  # ⭐ 关键：只在最后一层生成事件
    
    block_size = (
        req_meta.original_block_size[req_meta.kv_cache_group_id]
        if isinstance(req_meta.original_block_size, list)
        else req_meta.original_block_size
    )
    stored_events: list[BlockStored] = []
    prev_key = None
    group_block_size = self._get_block_size(req_meta.kv_cache_group_id)
    new_block_hashes = [maybe_convert_block_hash(bh) for bh in req_meta.block_hashes]
    
    # 使用所有层累积的 layerwise_event_starts 构建完整事件
    for start in sorted(self.layerwise_event_starts.pop(req_meta.req_id, set())):
        block_idx = start // group_block_size
        if block_idx >= len(new_block_hashes):
            continue
        block_hash = new_block_hashes[block_idx]
        end = min(start + group_block_size, len(req_meta.token_ids or []))
        token_ids = req_meta.token_ids[start:end] if req_meta.token_ids is not None else None
        
        stored_event = BlockStored(
            block_hashes=[block_hash],
            parent_block_hash=prev_key,
            token_ids=token_ids,
            block_size=block_size,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        stored_events.append(stored_event)
        prev_key = block_hash
        logger.debug("Added layerwise kv cache event '%s' to kv cache events queue", stored_event)
    
    return stored_events
```

#### 2.3.4 修改 _handle_request 方法

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

```python
def _handle_request(self, req_meta: LayerMultiBlockReqMeta):
    # ... 原有逻辑 ...
    
    if not keys:
        if layer_id == self.final_layer_id:
            stored_events = self._build_stored_events(req_meta)
            if stored_events:
                self.update_kv_event(stored_events)
        if is_last_chunk:
            self.dec_stored_request(req_meta.req_id)
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()
        return

    # ... 处理 missing blocks ...
    
    if not missing_indices:
        if layer_id == self.final_layer_id:
            stored_events = self._build_stored_events(req_meta)
            if stored_events:
                self.update_kv_event(stored_events)
        if is_last_chunk:
            self.dec_stored_request(req_meta.req_id)
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()
        return
    
    # ... 存储 missing blocks ...
    
    self.m_store.put(key_list, addr_list, size_list)
    self._record_layerwise_event_starts(req_meta, starts)  # ⭐ 记录
    stored_events = self._build_stored_events(req_meta)     # ⭐ 尝试构建
    if stored_events:
        self.update_kv_event(stored_events)

    if layer_id == self.final_layer_id and is_last_chunk:
        self.layerwise_event_starts.pop(req_meta.req_id, None)  # ⭐ 清理
        self.dec_stored_request(req_meta.req_id)
        self.set_finished_request(req_meta.req_id)
    self.request_queue.task_done()
```

#### 2.3.5 元数据扩展

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`

在 `LayerMultiBlockReqMeta` 中新增字段：
- `token_ids`: 令牌 ID 列表
- `original_block_size`: 原始 block 大小
- `block_hashes`: block 哈希列表（新增默认值）
- `kv_cache_group_id`: KV 缓存组 ID

#### 2.3.6 更新 pool_worker.py

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`

```python
def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
    if self.current_layer == 0:
        self.layerwise_storers = []
        current_event = None
        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue
            current_event = torch.npu.Event()
            current_event.record()
            break
        
        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue

            request.skip_null_blocks_by_group = self.group_uses_align_state
            request.disable_tp_key_sharding = (self.use_mla or self.use_sparse) and self.put_step > 1
            request.current_event = current_event
            self.kv_send_thread.add_stored_request(request.req_id)  # ⭐ 新增
            layerwise_storer = self.store_layer(request, current_event)
            self.layerwise_storers.append(layerwise_storer)
    
    for layerwise_storer in self.layerwise_storers:
        try:
            next(layerwise_storer)
        except Exception:
            raise
    
    self.current_layer = self.current_layer + 1

def store_layer(self, request: RequestTracker, current_event: torch.npu.Event) -> ...:
    # ... 构建 LayerMultiBlockReqMeta ...
    req_meta = LayerMultiBlockReqMeta(
        req_id=request.req_id,
        keys=keys_multi_layer,
        # ... 其他字段 ...
        token_ids=request.token_ids,                      # ⭐ 新增
        original_block_size=request.original_block_size,   # ⭐ 新增
        block_hashes=group_block_hashes,                   # ⭐ 新增
        kv_cache_group_id=group_id,                        # ⭐ 新增
    )
    self.kv_send_thread.add_request(req_meta)
```

#### 2.3.7 新增测试用例

**文件**: `tests/ut/distributed/ascend_store/test_kv_transfer.py`

新增 3 个分层事件测试：
- `test_layerwise_kv_event_published_on_final_layer`: 验证在最后一层发布事件
- `test_layerwise_kv_event_not_published_before_final_layer`: 验证中间层不发布事件
- `test_layerwise_kv_event_uses_missing_blocks_from_previous_layers`: 验证使用之前层的 missing blocks

### 2.4 分层工作流程

```
Layer 0:
  存储 KV Cache → 调用 _record_layerwise_event_starts 记录 → 不生成事件
  ↓
Layer 1:
  存储 KV Cache → 调用 _record_layerwise_event_starts 记录 → 不生成事件
  ↓
...
  ↓
Layer N-1 (最后一层):
  存储 KV Cache → 调用 _build_stored_events 使用累积信息 → 生成 BlockStored 事件 → 发布事件
  ↓
清理临时记录 layerwise_event_starts[req_id]
```

---

## 三、详细对比

### 3.1 相同点

| 方面 | 描述 |
|------|------|
| **核心目标** | 都实现 KV Cache 事件生成机制 |
| **基础架构** | 都使用 `BlockStored` 事件类型 |
| **元数据扩展** | 都使用 `token_ids`, `original_block_size` |
| **启用方式** | 都通过 `enable_kv_event` 参数控制 |
| **事件锁** | 都使用 `kv_event_lock` 保护事件列表 |
| **Connector 接口** | 都实现 `get_kv_connector_kv_cache_events` 等方法 |

### 3.2 不同点

| 对比维度 | PR #6593 | PR #9468 |
|---------|----------|----------|
| **发布时间** | 2026 年 2 月 | 2026 年 5 月（晚了 3 个月） |
| **vLLM 版本** | v0.15.0 | v0.20.2 |
| **适用场景** | 普通 KV Cache 存储（非分层） | 分层 KV Cache 存储 |
| **事件生成时机** | 存储完成后**一次性生成所有事件** | **只在最后一层生成事件** |
| **核心类** | `KVCacheStoreSendingThread` | `KVCacheStoreLayerSendingThread`（注意：不是新增！） |
| **事件完整性** | ✅ 完整事件（一次性生成） | ✅ 完整事件（最后一层累积生成） |
| **missing blocks 追踪** | 无需追踪（一次性处理） | **使用 layerwise_event_starts 累积跨层信息** |
| **测试覆盖** | 基础事件生成测试 | **新增分层事件专用测试**（3 个测试用例） |
| **元数据类** | `ReqMeta` | `LayerMultiBlockReqMeta`（扩展更多字段） |
| **日志输出** | `Added kv cache event` | `Added layerwise kv cache event` |

### 3.3 技术差异详解

#### 3.3.1 事件生成逻辑对比

**PR #6593 - 每层生成**:
```python
for index, start in enumerate(starts):
    # ... 存储 KV ...
    
    if self.enable_kv_event:
        # 立即为当前层生成事件
        stored_event = BlockStored(...)
        stored_events.append(stored_event)
    
if self.enable_kv_event and stored_events:
    self.update_kv_event(stored_events)
```

**PR #9468 - 只在最后一层生成**:
```python
def _handle_request(self, req_meta: LayerMultiBlockReqMeta):
    # ... 存储 KV ...
    
    self._record_layerwise_event_starts(req_meta, starts)  # 记录，不生成
    stored_events = self._build_stored_events(req_meta)     # 只在最后一层构建
    if stored_events:
        self.update_kv_event(stored_events)

def _build_stored_events(self, req_meta: LayerMultiBlockReqMeta) -> list[BlockStored]:
    if req_meta.layer_id != self.final_layer_id:
        return []  # 不是最后一层，不生成
    
    # 使用所有层累积的信息构建事件
    for start in sorted(self.layerwise_event_starts.pop(req_meta.req_id, set())):
        # 构建事件
```

---

## 四、总结

### 4.1 PR #6593 贡献

1. ✅ **基础框架**: 实现了 AscendConnector 的 KV Cache 事件生成能力
2. ✅ **上游适配**: 使 vLLM Ascend 能够适配上游 vLLM 的事件发布机制
3. ✅ **可观测性**: 提供了 KV Cache 操作的追踪能力
4. ✅ **扩展性**: 为后续增强（如 PR #9468）奠定了基础

### 4.2 PR #9468 贡献

1. ✅ **分层支持**: 针对分层 KV Cache 存储场景优化
2. ✅ **事件完整性**: 确保事件反映完整的 KV Cache 状态
3. ✅ **性能优化**: 减少多层模型中的事件生成次数
4. ✅ **测试覆盖**: 提供了完善的分层事件测试

### 4.3 关键修正

- ❌ **KVCacheStoreLayerSendingThread 不是 PR #9468 新增的**
- ✅ PR #9468 是在已有的 `KVCacheStoreLayerSendingThread` 中添加分层事件追踪机制

---

## 五、深入分析：PR #9468 的信息完整性验证

### 5.1 核心问题

**问题**：PR #9468 只在最后一层生成事件，是否会导致信息缺失？

**结论**：✅ **不会！** PR #9468 的设计是正确的，没有信息缺失。

### 5.2 关键发现

#### 5.2.1 每一层的 `LayerMultiBlockReqMeta` 信息是相同的

看 `pool_worker.py` 的 `store_layer` 方法：

```python
def store_layer(self, request: ReqMeta, current_event: torch.npu.Event | None):
    starts = []
    ends = []
    keys = []
    group_block_hashes = get_block_hashes(request.block_hashes, ...)  # ⭐ 计算一次
    
    for start, end, key in ...:
        starts.append(start)
        ends.append(end)
        keys.append(...)
    
    for layer_id, keys_multi_chunk in enumerate(keys):
        req_meta = LayerMultiBlockReqMeta(
            request.req_id,
            keys_multi_chunk,
            starts,                      # ⭐ 每一层都用相同的 starts
            ends,                        # ⭐ 每一层都用相同的 ends
            request.block_ids,
            layer_id,
            request.is_last_chunk,
            current_event,
            token_ids=request.token_ids,           # ⭐ 每一层都用相同的 token_ids
            original_block_size=request.original_block_size,  # ⭐ 相同
            block_hashes=group_block_hashes,       # ⭐ 每一层都用相同的 block_hashes
            kv_cache_group_id=group_id,
        )
```

**结论**：每一层的 `req_meta.block_hashes` 和 `req_meta.token_ids` 都是**完全相同的**！

#### 5.2.2 只记录该层实际存储的 blocks

看 `kv_transfer.py` 的 `_handle_request` 方法：

```python
def _handle_request(self, req_meta: LayerMultiBlockReqMeta):
    # ... 计算 missing_indices (该层实际需要存储的 blocks)
    missing_indices = [index for index, exists in enumerate(exists_states) if not exists]
    
    starts = [starts[index] for index in missing_indices]  # ⭐ 只保留该层实际存储的
    ends = [ends[index] for index in missing_indices]
    
    # ... 存储 KV cache
    self.m_store.put(key_list, addr_list, size_list)
    
    # ⭐ 只记录该层实际存储的 starts
    self._record_layerwise_event_starts(req_meta, starts)
```

#### 5.2.3 最后一层使用累积信息构建事件

```python
def _build_stored_events(self, req_meta: LayerMultiBlockReqMeta) -> list[BlockStored]:
    if req_meta.layer_id != self.final_layer_id:
        return []
    
    # ⭐ 使用所有层累积的 starts（只包含实际存储的）
    for start in sorted(self.layerwise_event_starts.pop(req_meta.req_id, set())):
        block_idx = start // group_block_size
        block_hash = new_block_hashes[block_idx]  # ⭐ 使用最后一层的 req_meta，信息完整
        token_ids = req_meta.token_ids[start:end]  # ⭐ 使用最后一层的 req_meta，信息完整
```

### 5.3 两种方案详细对比

**⚠️ 重要更正**：之前的分析错误，**PR #6593 不是每层一个事件！**

| 对比项 | PR #6593（非 Layerwise） | PR #9468（Layerwise） |
|--------|-------------------------|----------------------|
| **存储模式** | 一次性存储所有层 | 分层存储 |
| **使用线程类** | `KVCacheStoreSendingThread` | `KVCacheStoreLayerSendingThread` |
| **事件生成时机** | 一次性生成（存储完成后立即） | 只在最后一层生成 |
| **token_ids** | ✅ 完整 | ✅ 完整（相同） |
| **block_hashes** | ✅ 完整 | ✅ 完整（相同） |
| **信息完整性** | ✅ 完整 | ✅ 完整 |

### 5.4 总结

**PR #9468 的设计是正确的，没有信息缺失！**

原因：
1. ✅ 每一层的 `req_meta.block_hashes` 和 `req_meta.token_ids` 都是相同的
2. ✅ 最后一层的 `req_meta` 包含完整的信息
3. ✅ `layerwise_event_starts` 只累积实际存储的 blocks 的 `starts`
4. ✅ 使用最后一层的 `req_meta` 和累积的 `starts` 构建的事件，与 PR #6593 构建的事件**信息完全一致**

**PR #9468 的优势**：
- ✅ 适配分层存储架构
- ✅ 语义更清晰（一个事件表示完整的 KV Cache 存储）
- ✅ 没有信息缺失

---

## 六、上报信息详细对比

### 6.1 单个事件的信息结构

两种方案生成的**单个事件信息完全相同**！

每个 `BlockStored` 事件包含：

```python
BlockStored(
    block_hashes=[block_hash],        # ⭐ 单个 block 的 hash
    parent_block_hash=prev_key,       # ⭐ 前一个 block 的 hash（链式关系）
    token_ids=token_ids,              # ⭐ 该 block 对应的 token ids
    block_size=block_size,            # ⭐ block 大小
    lora_id=None,                     # ⭐ LoRA ID（None）
    medium="cpu",                     # ⭐ 存储介质（cpu）
    lora_name=None,                   # ⭐ LoRA 名称（None）
)
```

### 6.2 PR #6593（非 Layerwise）上报方案

#### 6.2.1 核心概念
- **非 Layerwise 模式不是分层存储的！**
- 使用 `KVCacheStoreSendingThread`，一次性处理所有层
- 一次性存储，一次性生成事件

#### 6.2.2 事件生成时机
- 存储完成后**一次性生成所有事件**

#### 6.2.3 事件数量
- 假设 M 个 missing blocks（所有层）
- 总事件数 = **M**

#### 6.2.4 上报流程
```
一次性存储所有层 KV Cache
  ↓
生成 M 个 BlockStored 事件（包含所有层的 blocks）
  ↓
一次性上报
```

### 6.3 PR #9468（Layerwise）上报方案

#### 6.3.1 核心概念
- Layerwise 模式是分层存储的
- 使用 `KVCacheStoreLayerSendingThread`，逐层处理
- 前 N-1 层：记录 starts，不生成事件
- 最后一层：使用所有层累积的 starts，一次性生成所有事件

#### 6.3.2 事件生成时机
- **只在最后一层生成事件**

#### 6.3.3 事件数量
- 假设 N 层，每层有 M 个 missing blocks
- 总事件数 = **N × M**（累积所有层的 missing blocks）

#### 6.3.4 上报流程
```
Layer 0:
  存储 KV Cache → 记录 M 个 starts → 不生成事件

Layer 1:
  存储 KV Cache → 记录 M 个 starts → 不生成事件

...

Layer N-1 (最后一层):
  存储 KV Cache → 记录 M 个 starts
  → 使用累积的 N×M 个 starts → 生成 N×M 个 BlockStored 事件 → 上报
```

### 6.4 两种方案详细对比

| 对比项 | PR #6593（非 Layerwise） | PR #9468（Layerwise） |
|--------|-------------------------|----------------------|
| **存储模式** | 一次性存储所有层 | 分层存储 |
| **使用线程类** | `KVCacheStoreSendingThread` | `KVCacheStoreLayerSendingThread` |
| **单个事件信息** | ✅ 完整（block_hash, parent_hash, token_ids 等） | ✅ 完整（**完全相同**） |
| **事件生成时机** | 一次性生成 | 只在最后一层生成 |
| **事件总数量** | M（所有层总 blocks） | N × M（分层累积） |
| **事件粒度** | 所有层的 blocks（一次性） | 所有层的 blocks（最后一层） |
| **parent_block_hash 关系** | 所有层链式（一次性） | 所有层链式（累积） |
| **中间层上报** | ❌ 无（一次性） | ❌ 无（只在最后一层） |
| **语义清晰度** | 好（一次性上报完整 KV Cache） | 好（适配分层架构） |
| **事件系统压力** | 低（一次性上报） | 低（一次性上报） |

### 6.5 关键结论

**⚠️ 重要更正**：
1. ❌ **之前错误**：PR #6593 不是每层一个事件！
2. ✅ **正确理解**：
   - PR #6593（非 Layerwise）：一次性存储，一次性生成事件
   - PR #9468（Layerwise）：分层存储，最后一层一次性生成事件

**正确对比**：
1. ✅ **单个事件信息完全相同**：两种方案生成的 `BlockStored` 事件内容完全一致
2. ✅ **事件数量不同**：
   - PR #6593：M 个事件（所有层一次性）
   - PR #9468：N × M 个事件（分层累积）
3. ✅ **根本区别**：
   - PR #6593：适配**一次性存储**架构
   - PR #9468：适配**分层存储**架构

### 6.6 parent_block_hash 的说明

**两种方案都支持链式关系**：
- PR #6593（非 Layerwise）：一次性链式
- PR #9468（Layerwise）：累积链式

两种方案的 `parent_block_hash` 关系是一致的，都是跨层的完整链式。

---

## 七、Layerwise 方案的意义

### 7.1 核心问题：为什么需要 Layerwise？

**关键理解**：
> ❌ Layerwise 方案不是为了改变事件内容
> ✅ Layerwise 方案是为了让分层存储架构能够正常工作！

### 7.2 演进历程

| 阶段 | 非 Layerwise | Layerwise |
|------|-------------|-----------|
| **PR #6593 之前** | ❌ 不支持 KV event | ❌ 不支持 KV event |
| **PR #6593 之后** | ✅ 支持 KV event | ❌ **仍然不支持！**（这就是问题） |
| **PR #9468 之后** | ✅ 支持 KV event | ✅ **现在也支持 KV event 了！**（这就是贡献） |

### 7.3 Layerwise 方案的真实意义

#### 7.3.1 架构适配

Layerwise 方案的核心价值：
- **不是为了改变事件上报**
- **而是为了支持分层存储架构！**

**如果没有 PR #9468 的 Layerwise 方案**：
```
分层存储架构
  ↓
每层存储完就想生成事件（但只有该层的信息）
  ↓
❌ 事件系统无法正常工作！（因为无法获取完整信息）
```

**有了 PR #9468 的 Layerwise 方案后**：
```
分层存储架构
  ↓
每层只记录 starts，不生成事件
  ↓
最后一层累积所有 starts
  ↓
✅ 生成完整事件！（适配分层架构）
```

#### 7.3.2 分层存储本身的意义

Layerwise 存储架构本身就有重要价值：

| 分层存储的优势 | 说明 |
|--------------|------|
| **内存效率** | 可以逐层释放内存，减少峰值内存占用 |
| **并行处理** | 可以并行处理不同层，提高性能 |
| **调度灵活** | 支持更精细的资源调度和负载均衡 |
| **架构演进** | 为未来可能的硬件架构（如专用芯片）准备 |

#### 7.3.3 实际场景举例

假设模型有 32 层：

**非 Layerwise（一次性存储）**：
```
一次性处理所有 32 层
  ↓
需要同时存储 32 层的中间数据
  ↓
内存峰值高 ✗
```

**Layerwise（分层存储）**：
```
处理第 0 层 → 存储 → 释放该层内存
  ↓
处理第 1 层 → 存储 → 释放该层内存
  ↓
...
  ↓
处理第 31 层 → 存储 → 生成事件
  ↓
内存峰值低 ✓
```

### 7.4 总结

**Layerwise 方案的意义不在于改变事件内容**，而在于：

1. ✅ **架构适配**：让分层存储架构能够正常集成到事件系统
2. ✅ **性能优化**：通过分层存储提高内存使用效率
3. ✅ **保持兼容**：生成的事件与非 Layerwise 模式完全相同，下游系统无需修改
4. ✅ **功能完整**：既支持分层存储，又支持完整的 KVCache 事件上报

**一句话总结**：Layerwise 方案不是为了改事件，而是为了支持分层存储架构！

---

## 八、参考资料

### 8.1 PR 链接

- PR #6593: https://github.com/vllm-project/vllm-ascend/pull/6593
- PR #9468: https://github.com/vllm-project/vllm-ascend/pull/9468

### 7.2 核心文件

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`

---

**文档创建时间**: 2026-05-28  
**作者**: AI Assistant  
**版本**: 6.0（添加核心理解和Layerwise方案意义详解）  
**分支**: tmp_br  
**路径**: `analysis/kv_events_pr_comparison.md`
