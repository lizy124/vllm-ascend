# KV Events 分层特性自测方案

**PR**: #9468 - Support layerwise KV cache events

**分支**: kv_events

**测试日期**: 2026-06-03

**作者**: lizy124

---

## 一、功能测试

### 2.1 基础功能测试 - 启用分层 KV 事件

#### 测试配置

```yaml
enable_kv_cache_events: true
use_layerwise: true
layerwise_kv_event_mode: "layerwise"  # 或 "per_layer"
```

#### 测试场景 1: 多层模型 KV 缓存事件生成

**测试目标**: 验证在分层模式下，只在最后一层生成 KV 事件

**测试步骤**:
1. 启动 vLLM 服务，配置多层模型
2. 发送推理请求
3. 检查日志中的 KV 事件生成情况

**预期结果**:
- ✅ 第 0 层：记录 missing blocks，不生成事件
- ✅ 第 1 层：记录 missing blocks，不生成事件
- ✅ 最后一层：生成完整的 KV 事件
- ✅ 事件包含所有层的 block 信息

**验证命令**:
```bash
# 查看日志中的事件生成
grep -n "layerwise_event\|KV event" /path/to/vllm.log

# 检查事件数量
# 预期：每个请求只生成 1 个事件（而不是每层 1 个）
```

#### 测试场景 0: 对比测试 - 验证 Layerwise 模式与非 Layerwise 模式的区别

**测试目标**: 通过对比日志，验证 Layerwise 模式是逐层处理，但只在最后一层生成累积事件

**测试步骤**:

**步骤 A - 非 Layerwise 模式测试**:
```bash
# 1. 启动非 Layerwise 模式
python -m vllm.entrypoints.openai.api_server \
  --model <your_model> \
  --enable-kv-cache-events \
  --use-layerwise=false \
  --vllm-log-level=DEBUG \
  2>&1 | tee non_layerwise.log

# 2. 发送测试请求（记录 request_id）
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<your_model>",
    "prompt": "Hello, world!",
    "max_tokens": 100
  }'

# 3. 从日志中提取 request_id
REQUEST_ID=$(grep -o '"request_id": "[^"]*"' non_layerwise.log | head -1 | cut -d'"' -f4)
echo "Request ID: $REQUEST_ID"

# 4. 统计日志数量
echo "=== 非 Layerwise 模式日志分析 ==="
grep "Storing KV cache.*$REQUEST_ID" non_layerwise.log | wc -l  # 应该 = 1（一次性处理）
grep "Added.*kv cache event.*$REQUEST_ID" non_layerwise.log | wc -l  # 应该 = 1
```

**步骤 B - Layerwise 模式测试**:
```bash
# 1. 重启服务，启用 Layerwise 模式
python -m vllm.entrypoints.openai.api_server \
  --model <your_model> \
  --enable-kv-cache-events \
  --use-layerwise=true \
  --vllm-log-level=DEBUG \
  2>&1 | tee layerwise.log

# 2. 发送相同的测试请求
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<your_model>",
    "prompt": "Hello, world!",
    "max_tokens": 100
  }'

# 3. 提取 request_id 并统计
REQUEST_ID=$(grep -o '"request_id": "[^"]*"' layerwise.log | head -1 | cut -d'"' -f4)

echo "=== Layerwise 模式日志分析 ==="
grep "Storing KV cache.*$REQUEST_ID" layerwise.log | wc -l  # 应该 = N（层数，逐层处理）
grep "Added layerwise kv cache event.*$REQUEST_ID" layerwise.log | wc -l  # 应该 = 1（最后一层生成）
```

**预期对比结果**:

| 指标 | 非 Layerwise 模式 | Layerwise 模式 | 说明 |
|------|------------------|---------------|------|
| "Storing KV cache" 日志数 | **1**（一次性） | **N**（逐层，N=层数） | Layerwise 逐层处理 |
| "Added layerwise kv cache event" 日志数 | **1** | **1** | 都只生成 1 个事件 |
| 事件包含的 block 数 | 所有 block | 所有 block（累积） | 事件完整性相同 |
| 处理方式 | 一次性 | 逐层 | Layerwise 更灵活 |

**验证脚本（自动化对比）**:
```bash
#!/bin/bash
# compare_modes.sh - 对比 Layerwise 与非 Layerwise 模式

echo "=== 对比 KV 事件生成 ==="

for mode in "non_layerwise" "layerwise"; do
    LOG_FILE="${mode}.log"
    REQUEST_ID=$(grep -o '"request_id": "[^"]*"' "$LOG_FILE" | head -1 | cut -d'"' -f4)

    echo ""
    echo "模式：$mode"
    echo "Request ID: $REQUEST_ID"

    STORING_COUNT=$(grep -c "Storing KV cache.*$REQUEST_ID" "$LOG_FILE")
    EVENT_COUNT=$(grep -c "Added.*kv cache event.*$REQUEST_ID" "$LOG_FILE")

    echo "  Storing KV cache 日志数：$STORING_COUNT"
    echo "  Added kv cache event 日志数：$EVENT_COUNT"

    if [ "$mode" = "layerwise" ]; then
        if [ "$STORING_COUNT" -gt 1 ] && [ "$EVENT_COUNT" -eq 1 ]; then
            echo "  ✅ Layerwise 模式正确：逐层处理（$STORING_COUNT 次），只生成 1 个事件"
        else
            echo "  ❌ Layerwise 模式异常"
        fi
    else
        if [ "$STORING_COUNT" -eq 1 ] && [ "$EVENT_COUNT" -eq 1 ]; then
            echo "  ✅ 非 Layerwise 模式正确：一次性处理，生成 1 个事件"
        else
            echo "  ⚠️ 非 Layerwise 模式异常"
        fi
    fi
done
```

### 2.2 并发与压力测试

#### 测试场景 2: 高并发请求

**测试配置**:
```yaml
# 并发请求数
concurrent_requests: 100

# 模型层数
num_hidden_layers: 4
```

**测试步骤**:
1. 使用 locust 或自定义脚本发送 100 个并发请求
2. 监控内存使用情况
3. 检查是否有内存泄漏

**预期结果**:
- ✅ 所有请求正常完成
- ✅ stored_requests 最终为空（无泄漏）
- ✅ 事件数量 = 请求数量（每个请求 1 个事件）

### 2.4 监听测试 - 验证 KV 事件实际传输

#### 测试场景 3: ZMQ 监听测试

**测试目标**: 通过 ZMQ 监听器验证 KV 事件实际传输和内容正确性

**测试步骤**:

**步骤 1 - 启动 vLLM 服务**:
```bash
# 启动服务，启用 KV 事件
python -m vllm.entrypoints.openai.api_server \
  --model <your_model> \
  --enable-kv-cache-events \
  --use-layerwise \
  --kv-events-endpoint tcp://*:5555
```

**步骤 2 - 启动监听器**:
```bash
# 在另一个终端启动监听器
python D:\lzy\code\for_env\kv_events\jianting\listen.py \
  --endpoint tcp://localhost:5555 \
  --topic kv-events
```

**步骤 3 - 发送测试请求**:
```bash
# 发送推理请求
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<your_model>",
    "prompt": "Hello, world!",
    "max_tokens": 100
  }'
```

**步骤 4 - 检查监听结果**:
```bash
# 监听器应该输出类似以下内容：
# ✅ Decoded successfully!
#   Event type: KVEventBatch
#   Timestamp: 1780451289.2195766
#   DP rank: 0
#   Number of events: 18
#   Event 1: BlockStored
#     BlockStored(block_hashes=[...], parent_block_hash=None, ...)
#   Event 2: BlockStored
#     BlockStored(block_hashes=[...], parent_block_hash=..., ...)
```

**预期结果**:
- ✅ 监听器成功连接到 ZMQ endpoint
- ✅ 监听器成功接收到 KVEventBatch
- ✅ 事件成功解码为 BlockStored 对象
- ✅ 每个 BlockStored 包含正确的字段：
  - `block_hashes`: 块哈希列表
  - `parent_block_hash`: 父块哈希（第一个为 None）
  - `token_ids`: token ID 列表
  - `block_size`: 块大小（如 128）
  - `medium`: 存储介质（如 'cpu'）
- ✅ 事件形成正确的链式结构（通过 parent_block_hash）
- ✅ 事件数量与预期一致

**验证要点**:
1. **事件链完整性**: 检查 `parent_block_hash` 形成正确的链
   - 第一个事件的 `parent_block_hash` 应为 `None`
   - 后续事件的 `parent_block_hash` 应指向前一个事件的 `block_hashes[0]`

2. **事件内容正确性**: 检查每个事件包含必要信息
   - `block_hashes` 不为空
   - `token_ids` 包含有效的 token ID
   - `block_size` 符合配置

3. **事件数量验证**: 检查事件数量是否符合预期
   - 对于 layerwise 模式，每个请求应生成一个事件批次
   - 批次中的事件数量应与 KV cache 块数量一致

**监听脚本说明**:
- 脚本位置: `D:\lzy\code\for_env\kv_events\jianting\listen.py`
- 使用 ZMQ SUB 模式订阅 KV 事件
- 使用 msgspec 解码 KVEventBatch
- 支持自定义 endpoint 和 topic

**结果文件**:
- 结果示例: `D:\lzy\code\for_env\kv_events\jianting\result.txt`
- 包含完整的事件解码输出
- 可用于验证事件内容和结构

---

## 二、代码质量检查

### 4.1 静态代码检查

```bash
# 1. 类型检查
mypy vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/

# 2. 代码风格
ruff check vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/

# 3. 导入顺序
ruff check --select I vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/
```

### 4.2 单元测试

```bash
# 运行相关测试
pytest tests/ut/distributed/ascend_store/test_kv_transfer.py -v
```
