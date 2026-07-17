# PR #9468: Support layerwise KV cache events 完整讲解

**讲解人**: lizy124  
**PR链接**: https://github.com/vllm-project/vllm-ascend/pull/9468  
**日期**: 2026-05-29

---

## 目录

1. [需求背景](#1-需求背景)
2. [问题分析](#2-问题分析)
3. [设计思路](#3-设计思路)
4. [代码改动详解](#4-代码改动详解)
5. [工作流程](#5-工作流程)
6. [测试覆盖](#6-测试覆盖)
7. [常见问题FAQ](#7-常见问题faq)

---

## 1. 需求背景

### 1.1 演进历程

| 阶段 | 非 Layerwise 模式 | Layerwise 模式 |
|------|-----------------|----------------|
| **PR #6593 之前** | ❌ 不支持 KV event 上报 | ❌ 不支持 KV event 上报 |
| **PR #6593 之后** | ✅ 支持 KV event 上报 | ❌ **仍然不支持！**（这就是我们的问题） |
| **PR #9468 之后** | ✅ 支持 KV event 上报 | ✅ **现在也支持了！**（这就是我们的贡献） |

### 1.2 核心需求

**一句话总结**: 让 layerwise 存储模式也能用上 KV event 上报功能！

**关键点**:
- ❌ 不是为了改变事件内容
- ❌ 不是为了上报更多信息
- ✅ 是为了让分层存储架构能够正常工作
- ✅ 保持与非 layerwise 模式的事件兼容性

---

## 2. 问题分析

### 2.1 什么是 Layerwise 存储？

**Layerwise**: 分层存储，逐层处理模型的每一层

**非 Layerwise**: 一次性存储所有层

**Layerwise 的优势**:
- 内存效率高：逐层释放内存，减少峰值内存占用
- 并行处理：可以并行处理不同层
- 调度灵活：支持更精细的资源调度

### 2.2 为什么之前 Layerwise 不支持 KV Event？

**核心难题**: 如果每层都生成事件的话:
- ❌ 只有该层的信息，不完整
- ❌ 事件系统会收到 N 倍的事件（N=层数）
- ❌ 下游系统难以处理

**之前的状况**: layerwise 模式根本就没有 KV event 上报功能！

---

## 3. 设计思路

### 3.1 核心设计理念

**分治 + 聚合**:
- **分治**: 逐层处理，每层只记录 starts
- **聚合**: 最后一层统一生成完整事件

### 3.2 数据结构设计

新增数据结构用于 layerwise 模式:

```python
# 在 KVCacheStoreLayerSendingThread 中新增
self.layerwise_event_starts: dict[str, set[int]] = defaultdict(set)
# 用于记录每个 request 的所有层的 starts
```

### 3.3 设计优势

| 优势 | 说明 |
|------|------|
| ✅ 事件完整 | 最后一层生成完整事件，包含所有层信息 |
| ✅ 减少开销 | N 层只生成 1 次事件，而不是 N 次 |
| ✅ 保持兼容 | 生成的事件与非 layerwise 模式完全相同 |
| ✅ 语义清晰 | 事件表示完整的 KV cache 存储 |

---

## 4. 代码改动详解

### 4.1 改动文件概览

总共改动 4 个文件:
- `config_data.py`: 元数据结构增强
- `kv_transfer.py`: 核心功能实现
- `pool_worker.py`: 集成调用
- `test_kv_transfer.py`: 测试用例

---

### 4.2 config_data.py 改动详解

#### 4.2.1 LayerMultiBlockReqMeta 增强

```python
# 新增字段:
token_ids: Optional[list[int]] = None
original_block_size: Optional[int] = None
block_hashes: Optional[list[int]] = None
is_last_chunk: bool = False
kv_cache_group_id: Optional[int] = None
```

**设计目的**: 这些字段和 ReqMeta 保持一致，确保信息完整

**关键点**: `block_hashes` 设置默认值为 None，避免类型错误

---

### 4.3 kv_transfer.py 改动详解（核心！）

#### 4.3.1 KVCacheStoreLayerSendingThread 初始化

```python
def __init__(...):
    # ... 原有代码
    self.final_layer_id = num_layers - 1  # 记录最后一层 ID
    self.put_step = put_step
    self.enable_kv_event = enable_kv_event
    
    # 新增：layerwise 事件追踪
    self.layerwise_event_starts: dict[str, set[int]] = defaultdict(set)
    self.stored_requests: dict[str, int] = defaultdict(int)
    self.done_task_lock = threading.Lock()
```

**新增字段解释**:
- `layerwise_event_starts`: 保存每个 request 的所有 starts
- `stored_requests`: 追踪 request 计数
- `done_task_lock`: 线程安全锁

---

#### 4.3.2 新增方法: add_stored_request

```python
def add_stored_request(self, req_id: str):
    """
    增加一个 stored request 计数
    
    pool_worker 会在每层调用这个方法
    """
    with self.done_task_lock:
        self.stored_requests[req_id] += 1
```

**设计考虑**:
- 使用线程锁确保安全
- 每层调用一次，计数累加

---

#### 4.3.3 新增方法: dec_stored_request

```python
def dec_stored_request(self, req_id: str):
    """
    减少一个 stored request 计数
    """
    with self.done_task_lock:
        if req_id in self.stored_requests:
            self.stored_requests[req_id] -= 1
```

---

#### 4.3.4 新增方法: delete_finished_stored_request

```python
def delete_finished_stored_request(self, req_id: str):
    """
    删除已完成的 stored request
    """
    with self.done_task_lock:
        if req_id in self.stored_requests:
            del self.stored_requests[req_id]
```

---

#### 4.3.5 新增方法: _record_layerwise_event_starts

```python
def _record_layerwise_event_starts(self, req_meta: LayerMultiBlockReqMeta, starts: list[int]):
    """
    记录某一层的 starts，但不生成事件
    
    参数:
    - req_meta: 层请求元数据
    - starts: 该层实际存储的 start 列表
    """
    if not self.enable_kv_event:
        return
    
    req_id = req_meta.req_id
    for start in starts:
        self.layerwise_event_starts[req_id].add(start)
```

**关键设计**:
- 只记录，不生成事件
- 使用 set 自动去重
- 累积所有层的信息

---

#### 4.3.6 新增方法: _build_stored_events

```python
def _build_stored_events(self, req_meta: LayerMultiBlockReqMeta) -> list[BlockStored]:
    """
    构建完整的 stored events（只在最后一层调用！）
    
    参数:
    - req_meta: 层请求元数据（最后一层）
    
    返回:
    - list[BlockStored]: 完整的事件列表
    """
    if not self.enable_kv_event:
        return []
    
    if req_meta.layer_id != self.final_layer_id:
        return []  # 不是最后一层，直接返回空
    
    req_id = req_meta.req_id
    if req_id not in self.layerwise_event_starts:
        return []
    
    stored_events: list[BlockStored] = []
    prev_key = None
    block_hashes = req_meta.block_hashes
    original_block_size = req_meta.original_block_size
    token_ids = req_meta.token_ids
    group_block_size = original_block_size
    
    # 获取所有层积累的 starts，并排序
    sorted_starts = sorted(self.layerwise_event_starts.pop(req_id))
    
    for start in sorted_starts:
        # 找到对应的 end
        # ... 计算 end
        end = ...
        
        # 计算 block index
        block_idx = start // group_block_size
        
        # 从最后一层的 req_meta 获取 block_hash
        if block_hashes and block_idx < len(block_hashes):
            bh = block_hashes[block_idx]
        else:
            bh = None
        
        if bh is None:
            continue
        
        new_hash = maybe_convert_block_hash(bh)
        
        # 获取 token_ids
        tids = None
        if token_ids is not None:
            tids = token_ids[start:end]
        
        # 创建 BlockStored 事件
        stored_event = BlockStored(
            block_hashes=[new_hash],
            parent_block_hash=prev_key,
            token_ids=tids,
            block_size=original_block_size,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        stored_events.append(stored_event)
        prev_key = new_hash
        logger.debug("Added layerwise kv cache event '%s' to kv cache events queue", stored_event)
    
    return stored_events
```

**核心逻辑解释**:
1. 检查是否是最后一层，不是则返回空
2. 取出之前累积的所有 starts
3. 从最后一层的 req_meta 获取完整信息（因为每一层的 req_meta 都有完整信息！）
4. 排序 starts，确保事件正确顺序
5. 构建完整的事件链，设置 parent_block_hash
6. 清理数据

---

#### 4.3.7 _handle_request 方法增强

```python
def _handle_request(self, req_meta: LayerMultiBlockReqMeta):
    req_id = req_meta.req_id
    
    # 新增：检查 req_id 是否在 stored_requests 中
    if req_id not in self.stored_requests:
        self.request_queue.task_done()
        return
    
    # ... 原有代码
    
    # 计算 missing_indices（该层实际存储的 blocks）
    exists_states = self.lookup(key_list)
    missing_indices = [...]
    
    starts = [starts[index] for index in missing_indices]
    ends = [ends[index] for index in missing_indices]
    key_list = [key_list[index] for index in missing_indices]
    
    # ... 存储 KV cache
    
    # 新增：记录该层的 starts（不生成事件）
    self._record_layerwise_event_starts(req_meta, starts)
    
    # 新增：尝试构建事件（只有最后一层才会真正构建）
    stored_events = self._build_stored_events(req_meta)
    if stored_events:
        self.update_kv_event(stored_events)
    
    # 新增：减少 stored_requests 计数
    self.dec_stored_request(req_id)
    
    # 修改：标记完成的条件 - 只要 is_last_chunk 就可以，不一定是最后一层
    if is_last_chunk:
        self.set_finished_request(req_id)
        self.delete_finished_stored_request(req_id)
    elif layer_id == self.final_layer_id and is_last_chunk:
        self.set_finished_request(req_id)
    
    self.request_queue.task_done()
```

**关键改动点**:
1. 检查 `stored_requests` 中的 req_id
2. 调用 `_record_layerwise_event_starts` 记录 starts
3. 调用 `_build_stored_events` 构建事件
4. 调用 `dec_stored_request` 减少计数
5. 修改完成条件：只要 `is_last_chunk` 就可以

---

### 4.4 pool_worker.py 改动详解

#### 4.4.1 store_layer 方法增强

```python
def store_layer(self, request: ReqMeta, current_event: torch.npu.Event | None):
    # ... 原有代码
    
    for layer_id, keys_multi_chunk in enumerate(keys):
        # 构建 LayerMultiBlockReqMeta，包含完整信息！
        req_meta = LayerMultiBlockReqMeta(
            request.req_id,
            keys_multi_chunk,
            starts,
            ends,
            request.block_ids,
            layer_id,
            request.is_last_chunk,
            current_event,
            token_ids=request.token_ids,              # 新增：完整信息
            original_block_size=request.original_block_size,  # 新增
            block_hashes=group_block_hashes,           # 新增：完整信息
            kv_cache_group_id=group_id,
        )
        
        # 新增：告诉发送线程这是一个 stored request
        if self.kv_send_thread is not None:
            self.kv_send_thread.add_stored_request(request.req_id)
        
        self.kv_send_thread.add_request(req_meta)
        yield None
```

**关键点**: 每一层的 LayerMultiBlockReqMeta 都包含完整的信息！

---

### 4.5 test_kv_transfer.py 改动详解

新增测试用例，覆盖 layerwise 模式的 KV event 功能

```python
class TestKVCacheStoreLayerSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, num_layers=2, enable_kv_event=False):
        # ... 修改：添加 enable_kv_event 参数
    
    def test_enable_kv_event(self):
        """测试 layerwise 模式的 KV event 上报"""
        # 创建线程，启用 KV event
        thread = self._make_thread(enable_kv_event=True)
        
        # 创建测试请求
        req_meta = LayerMultiBlockReqMeta(
            req_id="test_req",
            keys=[...],
            starts=[0, 16],
            ends=[16, 32],
            block_ids=[0, 1],
            layer_id=0,
            token_ids=[1, 2, ..., 32],
            original_block_size=16,
            block_hashes=[123, 456],
            is_last_chunk=False
        )
        
        # 测试第 0 层：只记录，不生成事件
        thread._handle_request(req_meta)
        events = thread.get_kv_events()
        self.assertEqual(len(events), 0)
        
        # 测试第 1 层（最后一层）：会生成事件
        req_meta.layer_id = 1
        req_meta.is_last_chunk = True
        thread._handle_request(req_meta)
        events = thread.get_kv_events()
        
        # 验证事件数量和内容
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].block_hashes, [123])
        self.assertEqual(events[1].block_hashes, [456])
```

---

## 5. 工作流程

### 5.1 完整流程图示

假设有 3 层模型:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                      Layer 0 (第 1 层)                     │
  ├─────────────────────────────────────────────────────────────┤
  │ 1. pool_worker.store_layer() 调用                          │
  │ 2. 创建 LayerMultiBlockReqMeta（包含完整信息！）            │
  │ 3. 调用 kv_send_thread.add_stored_request(req_id)          │
  │ 4. 调用 kv_send_thread.add_request(req_meta)               │
  ├─────────────────────────────────────────────────────────────┤
  │                    KVCacheStoreLayerSendingThread          │
  │ 5. _handle_request() 被调用                                │
  │ 6. 计算 missing_indices，确定实际存储的 blocks             │
  │ 7. 存储 KV cache                                           │
  │ 8. 调用 _record_layerwise_event_starts(starts)             │
  │    → layerwise_event_starts["req_id"] = {0, 10, 20}        │
  │ 9. 调用 _build_stored_events() → 不是最后一层，返回空     │
  │ 10. 调用 dec_stored_request(req_id)                        │
  │ 11. is_last_chunk 是 false，不标记完成                     │
  └─────────────────────────────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────┐
  │                      Layer 1 (第 2 层)                     │
  ├─────────────────────────────────────────────────────────────┤
  │ 类似 Layer 0 的步骤 1-11                                    │
  │ layerwise_event_starts["req_id"] = {0,10,20,30,40,50}      │
  └─────────────────────────────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────┐
  │                    Layer 2 (最后一层！)                    │
  ├─────────────────────────────────────────────────────────────┤
  │ 1. 执行步骤 1-7（和前面一样）                              │
  │ 2. _record_layerwise_event_starts(starts)                  │
  │    → layerwise_event_starts["req_id"] = {0,10,20,30,40,50,60,70,80} │
  │ 3. 调用 _build_stored_events() → 是最后一层！构建事件！   │
  │    → 排序 starts: [0,10,20,30,40,50,60,70,80]              │
  │    → 构建 9 个 BlockStored 事件                             │
  │ 4. 调用 update_kv_event(stored_events)                     │
  │ 5. dec_stored_request(req_id)                              │
  │ 6. is_last_chunk = true → 标记完成                        │
  └─────────────────────────────────────────────────────────────┘
                           ↓
              KV event 上报完成！
```

### 5.2 数据流详解

**关键数据流转**:
1. `pool_worker.store_layer()` 每层调用
2. 每层的 `req_meta` 都有完整信息
3. `layerwise_event_starts` 累积所有层的 starts
4. 最后一层统一构建事件

---

## 6. 测试覆盖

### 6.1 新增测试用例

| 测试用例 | 测试内容 |
|---------|---------|
| `test_enable_kv_event` | layerwise 模式下 KV event 正确生成 |
| `test_stored_requests_tracking` | stored_requests 计数正确 |
| `test_final_layer_only_generates_events` | 只有最后一层生成事件 |

### 6.2 测试要点

1. 验证中间层不生成事件
2. 验证最后一层生成完整事件
3. 验证事件内容正确
4. 验证 stored_requests 计数正确

---

## 7. 常见问题FAQ

### Q1: 为什么每一层的 req_meta 都包含完整的 block_hashes？

**A**: 因为在 pool_worker.py 中，每一层都用相同的 block_hashes 构建 req_meta。这样最后一层可以获取到所有层的 block 信息！

### Q2: 为什么使用 set 来存储 layerwise_event_starts？

**A**: set 可以自动去重，避免重复记录相同的 start。

### Q3: 为什么要引入 stored_requests？

**A**: 为了追踪 request 计数，确保只处理已登记的 request。

### Q4: 如果中间某层失败了怎么办？

**A**: 只要最后一层成功，就能生成完整事件。但如果最后一层失败，就不会生成事件。

### Q5: 生成的事件和非 layerwise 模式有区别吗？

**A**: 没有！完全相同，下游系统不需要任何改动！

### Q6: 如何启用 layerwise 模式的 KV event？

**A**: 和非 layerwise 一样，设置 `enable_kv_event=true` 即可。

### Q7: 为什么 _build_stored_events 只在最后一层调用？

**A**: 因为只有最后一层才能确定所有层的信息，确保事件完整性。

### Q8: parent_block_hash 是如何工作的？

**A**: 和非 layerwise 模式一样，形成链式关系，支持 block 链追踪。

---

## 附录 A: 代码统计

| 指标 | 数值 |
|------|------|
| 改动文件数 | 4 |
| 新增代码行数 | ~138 |
| 删除代码行数 | 3 |
| 净增代码行数 | ~135 |
| 新增测试用例 | 3 个 |

---

## 附录 B: 关键提交信息

```
[Feature] Support layerwise KV cache events

- Support layerwise KV cache events
- Fix layerwise KV event metadata typing
- Add default value to block_hashes field in LayerMultiBlockReqMeta
- Mark request as finished when is_last_chunk=True regardless of layer_id
- Fix stored_requests tracking for layerwise KV events mode
- Trigger CI pipeline verification

Signed-off-by: lizy124 <1950471827@qq.com>
```

---

**文档版本**: 1.0  
**最后更新**: 2026-05-29
