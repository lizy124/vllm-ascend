# MindIE-LLM: KV Cache 池化异步写设计文档

## 1. 背景与动机

### 1.1 问题背景

MindIE-LLM 的 KV Cache 池化特性（Prefix Cache）之前采用**同步写**模式：

| 问题 | 影响 |
|-----|------|
| 同步写阻塞推理 | Prefill 阶段写入 KV Cache 时，阻塞后续计算 |
| 性能瓶颈 | 长序列场景下，写入延迟显著影响吞吐量 |
| 资源利用率低 | NPU 计算资源在写入期间闲置 |

### 1.2 设计目标

1. **异步写**：Prefill 阶段异步写入 KV Cache，不阻塞计算
2. **流间同步**：通过 Event 实现图内部与图外部的同步
3. **特性叠加**：明确支持的模型和特性组合，避免不可预期的报错

### 1.3 依赖关系

- **Prefix Cache 特性**：KV Cache 池化的基础功能
- **EventManager**：已有的 Event 管理组件

---

## 2. 配置设计

### 2.1 新增配置项

```json
"kvPoolConfig" : {"backend":"", "configPath":"", "asyncWrite": true}
```

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| `backend` | string | "" | 指定使用的池化后端 |
| `configPath` | string | "" | 池化后端所需要的配置文件路径 |
| `asyncWrite` | bool | false | 是否开启池化异步写 |

### 2.2 配置优先级

```
用户配置 asyncWrite=true → MemPoolType::ASYNC_WRITE
用户配置 asyncWrite=false → MemPoolType::SYNC_WRITE
未配置 asyncWrite → MemPoolType::SYNC_WRITE（默认）
```

---

## 3. MemPoolType 枚举设计

### 3.1 枚举定义

```cpp
enum MemPoolType {
    DISABLED = 0,    // 未启用 KV Cache 池化
    SYNC_WRITE = 1,  // 同步写模式
    ASYNC_WRITE = 2  // 异步写模式
};
```

### 3.2 模式说明

| 模式 | 行为 | 适用场景 |
|-----|------|---------|
| `DISABLED` | 不使用 KV Cache 池化 | 无 Prefix Cache 需求 |
| `SYNC_WRITE` | Prefill 完成后同步写入 | 短序列、低延迟场景 |
| `ASYNC_WRITE` | Prefill 过程中异步写入 | 长序列、高吞吐场景 |

### 3.3 参数传递

```cpp
// model_param.h
MemPoolType memPoolType = MemPoolType::DISABLED;
std::string memPoolEventPipeKey = "default";  // 异步写时的 event pipeKey

// model_param.cpp
if (paramJson.contains("memPoolType")) {
    this->memPoolType = FetchJsonParam<MemPoolType>(paramJson, "memPoolType");
}
if (paramJson.contains("pipeKey")) {
    this->memPoolEventPipeKey = FetchJsonParam<std::string>(paramJson, "pipeKey");
}
```

---

## 4. Event 同步机制

### 4.1 EventManager 新增数据结构

```cpp
// 用于图外部与图内部的流间同步
std::map<std::string, std::tuple<int, std::vector<aclrtEvent>, aclrtStream>> eventsForExternal_;
```

**结构说明**：

| 元素 | 类型 | 说明 |
|-----|------|------|
| `int` | 索引 | 当前 event 索引，循环使用 |
| `std::vector<aclrtEvent>` | Event 队列 | 循环 event 队列 |
| `aclrtStream` | Stream | 用于同步的 sub stream |

### 4.2 EventManager 新增方法

#### CheckPipeKey()

```cpp
EventManagerStatus EventManager::CheckPipeKey(const std::string &pipeKey)
{
    if (eventsForExternal_.find(pipeKey) == eventsForExternal_.end()) {
        // 从 eventQueues_ 中取出 event，创建 sub stream
        std::vector<aclrtEvent> queue;
        while (!eventQueues_[pipeKey].empty()) {
            queue.push_back(eventQueues_[pipeKey].front());
            eventQueues_[pipeKey].pop();
        }
        aclrtStream subStream;
        aclrtCreateStream(&subStream);
        aclrtSetStreamFailureMode(subStream, ACL_STOP_ON_FAILURE);
        eventsForExternal_[pipeKey] = std::make_tuple(0, queue, subStream);
    }
    return EM_SUCCESS;
}
```

#### RecordEvent()

```cpp
EventManagerStatus EventManager::RecordEvent(const std::string &pipeKey)
{
    auto rt = CheckPipeKey(pipeKey);
    if (rt != EM_SUCCESS) return rt;
    
    auto& currentEventIdx = std::get<0>(eventsForExternal_[pipeKey]);
    auto& eventsForExtel = std::get<1>(eventsForExternal_[pipeKey]);
    auto& stream = std::get<2>(eventsForExternal_[pipeKey]);
    
    aclrtRecordEvent(eventsForExtel[currentEventIdx], stream);
    currentEventIdx = (currentEventIdx + 1) % eventsForExtel.size();
    return EM_SUCCESS;
}
```

#### WaitEvent()

```cpp
EventManagerStatus EventManager::WaitEvent(const std::string &pipeKey)
{
    auto rt = CheckPipeKey(pipeKey);
    if (rt != EM_SUCCESS) return rt;
    
    auto& currentEventIdx = std::get<0>(eventsForExternal_[pipeKey]);
    auto& eventsForExtel = std::get<1>(eventsForExternal_[pipeKey]);
    auto& stream = std::get<2>(eventsForExternal_[pipeKey]);
    
    aclrtStreamWaitEvent(stream, eventsForExtel[currentEventIdx]);
    aclrtResetEvent(eventsForExtel[currentEventIdx], stream);
    aclrtSynchronizeStream(stream);
    currentEventIdx = (currentEventIdx + 1) % eventsForExtel.size();
    return EM_SUCCESS;
}
```

### 4.3 Event 循环使用机制

```
Event Queue: [E0, E1, E2, E3]
currentEventIdx: 0 → 1 → 2 → 3 → 0 → 1 → ... (循环)
```

---

## 5. Prefill/Decode 阶段同步流程

### 5.1 Prefill 阶段

```
┌─────────────────────────────────────────────────────────────┐
│                    Prefill Phase                             │
│                                                              │
│  Layer 0 ──▶ Layer 1 ──▶ ... ──▶ Layer N                    │
│                                              │               │
│                                              ▼               │
│                                    ┌─────────────────┐       │
│                                    │ RecordEvent     │       │
│                                    │ (pipeKey)       │       │
│                                    └─────────────────┘       │
│                                              │               │
│                                              ▼               │
│                                    ┌─────────────────┐       │
│                                    │ Prefix Cache    │       │
│                                    │ Save (异步)     │       │
│                                    └─────────────────┘       │
│                                              │               │
│                                              ▼               │
│                                    Event E0 recorded         │
│                                    (等待 Decode Wait)        │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Decode 阶段

```
┌─────────────────────────────────────────────────────────────┐
│                    Decode Phase                              │
│                                                              │
│  ┌─────────────────┐                                        │
│  │ Prefix Cache    │                                        │
│  │ Load            │                                        │
│  └─────────────────┘                                        │
│          │                                                   │
│          ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ WaitEvent       │                                        │
│  │ (pipeKey)       │                                        │
│  │ (等待异步写完成) │                                        │
│  └─────────────────┘                                        │
│          │                                                   │
│          ▼                                                   │
│  Layer 0 ──▶ Layer 1 ──▶ ... ──▶ Layer N                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 RecordEventBeforePrefixCacheSave()

```cpp
atb::Status DecoderModel::RecordEventBeforePrefixCacheSave()
{
    if (param.memPoolType == atb_speed::base::MemPoolType::ASYNC_WRITE && param.isPrefill) {
        // 异步写模式 + prefill 阶段
        atb::Operation *op = nullptr;
        atb_speed::EventManager::GetInstance().RecordEvent(
            op, atb_speed::EventAction::PUSH, param.memPoolEventPipeKey);
        
        atb_speed::Model::Node recordSaveNode;
        recordSaveNode.inTensors = {};
        recordSaveNode.outTensors = {};
        recordSaveNode.operation.reset(op);
        graph_.nodes.push_back(recordSaveNode);
    }
    return atb::NO_ERROR;
}
```

### 5.4 时序图

```
Prefill Worker                Event Queue              Pool Backend
     │                            │                        │
     │  Layer N 计算              │                        │
     │───────────────────────────▶│                        │
     │                            │                        │
     │  RecordEvent(E0)           │                        │
     │───────────────────────────▶│                        │
     │                            │   E0 recorded          │
     │                            │                        │
     │  Prefix Cache Save         │                        │
     │────────────────────────────────────────────────────▶│
     │                            │                        │
     │  继续计算...               │   (异步写入)            │
     │                            │                        │
     │                            │                        │
Decode Worker                  │                        │
     │                            │                        │
     │  Prefix Cache Load         │                        │
     │────────────────────────────────────────────────────▶│
     │                            │                        │
     │  WaitEvent(E0)             │                        │
     │───────────────────────────▶│                        │
     │                            │   E0 wait + sync       │
     │                            │                        │
     │  Layer N 计算              │                        │
     │                            │                        │
```

---

## 6. Event Adapter 设计

### 6.1 Event 类接口

```cpp
// event.h
namespace atb_speed {

class Event {
public:
    void Record(const std::string& pipeKey);
    void Wait(const std::string& pipeKey);
};

} // namespace atb_speed
```

### 6.2 Event 类实现

```cpp
// event.cpp
namespace atb_speed {

void Event::Record(const std::string& pipeKey) {
    EventManager::GetInstance().RecordEvent(pipeKey);
}

void Event::Wait(const std::string& pipeKey) {
    EventManager::GetInstance().WaitEvent(pipeKey);
}

} // namespace atb_speed
```

### 6.3 Python/C++ 交互

Event Adapter 提供 Python 绑定，允许 Python 层调用：

```python
# Python 层使用示例
event = Event()
event.Record("default")  # 在当前 stream 上 record event
event.Wait("default")    # 等待 event 完成
```

---

## 7. 限制与约束

### 7.1 支持的模型

| 模型 | 说明 |
|-----|------|
| Qwen 稠密（非 MOE） | 支持 Qwen 系列稠密模型 |
| DeepSeek V3/V3.1/R1 | 支持 DeepSeek 系列模型 |

### 7.2 支持叠加的特性

| 模型 | 支持叠加的特性 |
|-----|---------------|
| Qwen 稠密 | 异步调度、Prefix Cache、Function Call、思考解析、Yarn |
| DeepSeek V3/V3.1/R1 | 异步推理、Prefix Cache、Context Parallel、Sequence Parallel |

### 7.3 不支持叠加的特性

| 特性 | 说明 |
|-----|------|
| SplitFuse | 与异步写时序冲突 |
| Micro Batch | 与异步写时序冲突 |
| Multi-Lora | 与异步写时序冲突 |

### 7.4 约束说明

> **重要**：异步写特性当前仅支持上述模型和特性组合。叠加其他特性或模型可能导致不可预期的报错。

---

## 8. 文件改动总结

### 8.1 改动文件列表

| 文件 | 改动类型 | 改动内容 |
|-----|---------|---------|
| `docs/zh/user_guide/feature/mempool.md` | 新增 + 移动 | 异步写配置 + 限制约束 |
| `docs/zh/user_guide/feature/kv_cache_pool.md` | 修改 | 更新链接引用 |
| `event_manager.h` | 新增 | RecordEvent/WaitEvent 方法声明 |
| `event_manager.cpp` | 新增 | RecordEvent/WaitEvent 实现 + 析构函数清理 |
| `model_param.h` | 新增 | MemPoolType 枚举 + memPoolType/memPoolEventPipeKey 参数 |
| `model_param.cpp` | 新增 | 参数解析逻辑 |
| `decoder_model.h` | 新增 | RecordEventBeforePrefixCacheSave 方法声明 |
| `decoder_model.cpp` | 新增 | RecordEventBeforePrefixCacheSave 实现 |
| `event.h` | 新增 | Event 类声明 |
| `event.cpp` | 新增 | Event 类实现 |

### 8.2 改动统计

| 类型 | 文件数 | 改动行数 |
|-----|-------|---------|
| 文档 | 2 | ~30 行 |
| C++ Header | 3 | ~50 行 |
| C++ Source | 4 | ~120 行 |
| 新增文件 | 2 | ~60 行 |

---

## 9. 测试与验证

### 9.1 测试场景

| 场景 | 测试内容 |
|-----|---------|
| 异步写开启 | Prefill 阶段异步写入，Decode 阶段正确等待 |
| 异步写关闭 | 同步写模式正常工作 |
| Event 循环 | 多次 Prefill/Decode 循环，Event 正确循环使用 |
| 特性叠加 | 支持的特性组合正常工作 |

### 9.2 特性叠加测试

| 组合 | 测试结果 |
|-----|---------|
| Qwen + 异步调度 + Prefix Cache | ✅ 通过 |
| Qwen + Function Call + 异步写 | ✅ 通过 |
| DeepSeek V3 + Context Parallel + 异步写 | ✅ 通过 |
| Qwen + SplitFuse + 异步写 | ❌ 不支持 |
| Qwen + Multi-Lora + 异步写 | ❌ 不支持 |

---

## 10. 总结

### 10.1 实现的功能

| 功能 | 说明 |
|-----|------|
| 异步写配置 | `asyncWrite` 配置项 |
| Event 同步机制 | RecordEvent/WaitEvent 方法 |
| MemPoolType 枚举 | DISABLED/SYNC_WRITE/ASYNC_WRITE |
| Event Adapter | Python/C++ 交互接口 |
| 限制约束文档 | 支持的模型和特性组合 |

### 10.2 性能提升效果

| 场景 | 同步写 | 异步写 | 提升 |
|-----|-------|-------|------|
| 长序列 Prefill | 阻塞计算 | 不阻塞 | 吞吐量提升 |
| 多轮对话 | 每轮阻塞 | 异步叠加 | 延迟降低 |

### 10.3 后续改进方向

1. **支持更多模型**：扩展支持更多模型类型
2. **支持更多特性叠加**：解决与 SplitFuse、Micro Batch 的时序冲突
3. **Event 优化**：减少 Event 同步开销
4. **错误处理**：异步写失败时的错误恢复机制