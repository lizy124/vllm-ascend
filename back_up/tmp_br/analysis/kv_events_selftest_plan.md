# KV Events 分层特性自测方案

**PR**: #9468 - Support layerwise KV cache events  
**分支**: kv_events  
**测试日期**: 2026-05-29  
**作者**: lizy124

---

## 零、快速验证指南（必读）

### 0.1 如何从日志验证功能是否生效

#### 核心概念理解

**Layerwise 模式的含义**：
- **非 layerwise** (`use_layerwise=false`): 一次性处理所有层的 KV cache，**一次性生成一个事件**
- **Layerwise** (`use_layerwise=true`): **逐层处理** KV cache，但只在**最后一层生成一个累积事件**

**关键区别**：
- 非 layerwise：一次性处理 → 一个事件（包含所有 block）
- Layerwise：逐层处理 → 前 N-1 层只记录信息，最后一层生成一个事件（包含所有层的 block）

#### 关键日志指标

1. **查看日志中的 INFO 级别输出**（`KVCacheStoreLayerSendingThread`）:
```
INFO: Storing KV cache for X out of Y blocks (missing_count=Z) for request <req_id>
```
- **Layerwise 模式**：每个请求应该看到 **多层** 这样的日志（每层一次）
- **非 Layerwise 模式**：每个请求只看到 **一次** 这样的日志（一次性处理）

2. **查看 DEBUG 日志**（需要设置 `VLLM_LOGGING_LEVEL=DEBUG`）:
```
DEBUG: Added layerwise kv cache event '<BlockStored ...>' to kv cache events queue
```
- **Layerwise 模式**：只在 **最后一层** 出现一次
- **非 Layerwise 模式**：出现一次（一次性生成）

3. **关键区别对比**：

| 场景 | "Storing KV cache" 日志 | "Added layerwise kv cache event" 日志 | 事件数量 |
|------|---------------------|----------------------------------|---------|
| **Layerwise** (use_layerwise=true) | N 次（每层一次） | **1 次**（最后一层） | 1 个事件（累积所有层） |
| **非 Layerwise** (use_layerwise=false) | **1 次**（一次性） | **1 次**（一次性） | 1 个事件 |
| **禁用事件** (enable_kv_cache_events=false) | 有 | 无 | 0 个事件 |

#### 实际日志示例

**非 Layerwise 模式（use_layerwise=false）日志示例**：
```
DEBUG: Added kv cache event 'BlockStored(block_hashes=[0x1234..., 0x5678..., 0x9abc...], ...)' to kv cache events queue
INFO: Storing KV cache for 30 out of 30 blocks (missing_count=30) for request req-abc123
```
说明：一次性处理 3 层（共 30 个 block），生成**一个事件**（包含 3 个 block_hash）。

**Layerwise 模式（use_layerwise=true）日志示例**：
```
INFO: Storing KV cache for 10 out of 10 blocks (missing_count=10) for request req-abc123  # layer 0
INFO: Storing KV cache for 10 out of 10 blocks (missing_count=10) for request req-abc123  # layer 1
DEBUG: Added layerwise kv cache event 'BlockStored(block_hashes=[0x1234..., 0x5678..., 0x9abc...], ...)' to kv cache events queue  # layer 2 (最后一层)
INFO: Storing KV cache for 10 out of 10 blocks (missing_count=10) for request req-abc123  # layer 2
```
说明：逐层处理 3 层，前 2 层只记录信息，最后一层生成**一个累积事件**（包含所有层的 block）。

**验证要点**：
- 数 "Storing KV cache" 日志数量：
  - Layerwise 模式：N 条（N=层数）
  - 非 Layerwise 模式：1 条
- 数 "Added layerwise kv cache event" 日志数量：
  - Layerwise 模式：1 条（最后一层）
  - 非 Layerwise 模式：1 条（一次性）

#### 验证脚本

```bash
#!/bin/bash
# verify_kv_events.sh - 验证 KV 事件生成功能

LOG_FILE="/path/to/vllm.log"
REQUEST_ID="your_request_id"

echo "=== 验证 KV 事件生成 ==="

# 1. 统计 "Storing KV cache" 日志数量（应该 = 层数）
STORING_COUNT=$(grep -c "Storing KV cache.*$REQUEST_ID" "$LOG_FILE")
echo "Storing KV cache 日志数：$STORING_COUNT (应该 = 模型层数)"

# 2. 统计 "Added kv cache event" 日志数量
EVENT_COUNT=$(grep -c "Added kv cache event.*$REQUEST_ID" "$LOG_FILE")
echo "Added kv cache event 日志数：$EVENT_COUNT"

# 3. 判断模式
if [ "$EVENT_COUNT" -eq 1 ]; then
    echo "✅ 分层模式生效：每个请求只生成 1 个事件"
elif [ "$EVENT_COUNT" -gt 1 ]; then
    echo "⚠️  非分层模式：每个请求生成 $EVENT_COUNT 个事件（可能 = 层数）"
else
    echo "❌ 事件未生成：检查 enable_kv_cache_events 配置"
fi
```

**运行示例**：
```bash
# 设置 debug 日志级别
export VLLM_LOGGING_LEVEL=DEBUG

# 启动服务（带上你的配置）
python -m vllm.entrypoints.openai.api_server \
    --model <your_model> \
    --enable-kv-cache-events \
    --use-layerwise

# 发送测试请求后，运行验证脚本
bash verify_kv_events.sh
```

---

## 一、测试环境准备

### 1.1 硬件环境要求
- 华为 Ascend NPU 设备（如 910B）
- 内存：≥ 64GB
- 显存：≥ 64GB

### 1.2 软件环境
```bash
# Python 版本
Python 3.10+

# 关键依赖
vllm-ascend (kv_events 分支)
torch_npu
CANN 8.0.RC1+
```

### 1.3 环境检查脚本
```bash
# 1. 检查分支
git branch | grep kv_events

# 2. 检查关键文件是否存在
ls -la vllm/ascend/kv_pool/ascend_store/kv_transfer.py
ls -la vllm/ascend/kv_pool/ascend_store/config_data.py
ls -la vllm/ascend/kv_pool/ascend_store/pool_worker.py

# 3. 检查代码变更
git diff origin/main..kv_events --stat
```

---

## 二、功能测试

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
1. 启动 vLLM 服务，配置 2 层模型
2. 发送推理请求
3. 检查日志中的 KV 事件生成情况

**预期结果**:
- ✅ 第 0 层：记录 missing blocks，不生成事件
- ✅ 第 1 层（最后一层）：生成完整的 KV 事件
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
|------|-----------------|---------------|------|
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
            echo "  ⚠️  非 Layerwise 模式异常"
        fi
    fi
done
```

#### 测试场景 2: stored_requests 引用计数
**测试目标**: 验证 stored_requests 正确追踪请求的层处理状态

**测试步骤**:
1. 启用分层模式
2. 发送多个并发请求
3. 监控 stored_requests 的变化

**预期结果**:
- ✅ 请求开始时：stored_requests[req_id] = 总层数
- ✅ 每处理完一层：stored_requests[req_id] -= 1
- ✅ 处理完最后一层：stored_requests[req_id] = 0，从字典中移除

**验证方法**:
```python
# 在代码中添加调试日志
print(f"stored_requests: {self.stored_requests}")
# 观察输出是否符合预期
```

---

### 2.2 边界条件测试

#### 测试场景 3: 单层模型
**配置**:
```yaml
enable_kv_cache_events: true
use_layerwise: true
num_hidden_layers: 1
```

**预期**:
- ✅ 单层模型正常工作
- ✅ 第 0 层即为最后一层，直接生成事件

#### 测试场景 4: 禁用 KV 事件
**配置**:
```yaml
enable_kv_cache_events: false
use_layerwise: true
```

**预期**:
- ✅ 不生成任何 KV 事件
- ✅ 不影响正常推理功能

#### 测试场景 5: 启用事件但不启用分层
**配置**:
```yaml
enable_kv_cache_events: true
use_layerwise: false
```

**预期**:
- ✅ 每层都生成 KV 事件（旧行为）
- ✅ 向后兼容性验证

---

### 2.3 并发与压力测试

#### 测试场景 6: 高并发请求
**测试配置**:
```bash
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

#### 测试场景 7: 长时间运行稳定性
**测试步骤**:
1. 持续发送请求 1 小时
2. 定期检查内存使用
3. 检查日志中的错误

**预期结果**:
- ✅ 无内存泄漏
- ✅ 无 AttributeError 或其他异常
- ✅ 事件生成正常

---

## 三、异常场景测试

### 3.1 错误处理测试

#### 测试场景 8: 请求中途失败
**测试步骤**:
1. 发送请求
2. 在中间层模拟错误（如修改代码抛出异常）
3. 观察 stored_requests 清理情况

**预期结果**:
- ✅ stored_requests 正确清理
- ✅ 无资源泄漏
- ✅ 后续请求不受影响

#### 测试场景 9: is_last_chunk=True 场景
**配置**:
```yaml
# 模拟流式输出的最后一个 chunk
```

**预期结果**:
- ✅ 无论 layer_id 是多少，请求都正确标记为完成
- ✅ stored_requests 正确递减

---

## 四、性能测试

### 4.1 性能基准对比

#### 测试场景 10: 性能对比测试
**对比组**:
- 组 A: `enable_kv_cache_events=false`
- 组 B: `enable_kv_cache_events=true, use_layerwise=false`
- 组 C: `enable_kv_cache_events=true, use_layerwise=true`

**测试指标**:
| 指标 | 测量方法 |
|------|---------|
| 吞吐量 (tokens/s) | 每秒生成的 token 数 |
| 延迟 (ms) | 端到端推理延迟 |
| 内存使用 (GB) | 峰值内存占用 |
| 事件数量 | 每个请求生成的事件数 |

**预期结果**:
- ✅ 组 C 的事件数量显著少于组 B（N 层 vs 1 个）
- ✅ 组 C 的性能开销应小于或等于组 B
- ✅ 组 C 的内存使用应合理

**测试脚本**:
```bash
# 使用 vLLM 内置性能测试工具
python -m vllm.benchmarks.benchmark_throughput \
    --model <your_model> \
    --dataset <your_dataset> \
    --num-prompts 1000
```

---

## 五、代码质量检查

### 5.1 静态代码检查
```bash
# 1. 类型检查
mypy vllm/ascend/kv_pool/ascend_store/

# 2. 代码风格
ruff check vllm/ascend/kv_pool/ascend_store/

# 3. 导入顺序
ruff check --select I vllm/ascend/kv_pool/ascend_store/
```

### 5.2 单元测试
```bash
# 运行相关测试
pytest tests/ascend/kv_pool/test_kv_transfer.py -v
pytest tests/ascend/kv_pool/test_pool_worker.py -v
```

---

## 六、测试结果记录模板

### 测试执行记录表

| 测试场景 | 测试日期 | 测试结果 | 备注 |
|---------|---------|---------|------|
| 场景 1: 多层模型事件生成 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 场景 2: stored_requests 计数 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 场景 3: 单层模型 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 场景 4: 禁用 KV 事件 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 场景 5: 启用事件不启用分层 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 场景 6: 高并发请求 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 场景 7: 长时间运行 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 场景 8: 请求中途失败 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 场景 9: is_last_chunk=True | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 场景 10: 性能对比 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |
| 代码质量检查 | 2026-05-29 | ⬜ 通过 ⬜ 失败 | |

---

## 七、问题追踪

### 发现问题记录

**问题 1**:
- **发现日期**: 
- **问题描述**: 
- **严重级别**: 高/中/低
- **解决方案**: 
- **状态**: 已解决/待解决

---

## 八、测试结论

### 总体评估
- [ ] 所有测试通过，可以合并
- [ ] 大部分测试通过，存在小问题（详见问题追踪）
- [ ] 存在严重问题，需要修复后重新测试

### 关键指标达成情况
| 指标 | 目标 | 实际结果 | 达成 |
|------|------|---------|------|
| 功能正确性 | 100% | % | ⬜ |
| 性能开销 | ≤5% | % | ⬜ |
| 内存泄漏 | 0 |  | ⬜ |
| 代码质量 | 通过检查 | | ⬜ |

### 签字确认
**测试者**: ___________  
**日期**: ___________

---

## 附录：快速测试命令

```bash
# 1. 快速功能验证
python -c "
from vllm.ascend.kv_pool.ascend_store.kv_transfer import KVCacheSender
print('Import successful')
"

# 2. 检查关键代码路径
grep -n "stored_requests\|layerwise_event" vllm/ascend/kv_pool/ascend_store/kv_transfer.py

# 3. 运行单元测试
pytest tests/ -k kv_event -v

# 4. 性能快速测试
python examples/offline_inference.py \
    --model <your_model> \
    --enable-kv-cache-events \
    --use-layerwise
```
