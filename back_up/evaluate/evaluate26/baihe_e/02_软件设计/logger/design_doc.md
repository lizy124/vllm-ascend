# vLLM-Ascend 日志质量改进设计文档

## 1. 背景与动机

### 1.1 问题背景

vLLM-Ascend 代码仓的日志质量存在以下问题，影响生产环境的调试和排查效率：

- **错误上下文不足**：错误日志缺少参数快照、根因分析和排查建议
- **组件归属不明确**：难以快速定位日志来源组件
- **日志级别误用**：正常流程打 ERROR，异常情况打 INFO
- **格式风格不统一**：参数格式、占位符风格不一致
- **关键事件遗漏**：线程启动/关闭、连接建立等关键状态缺少日志

### 1.2 参考规范

本 PR 参考 `private-skills` 中的日志质量规范：

| Skill | 说明 |
|-------|------|
| `log-quality-standard` | 8 个核心标准定义 |
| `log-quality-write` | 日志编写工作流程 |
| `log-quality-rewrite` | 日志整改工作流程 |
| `log-quality-scan-code` | 代码仓日志扫描 |

### 1.3 目标

- 改进日志质量，提升生产环境可调试性
- 遵循 log-quality-standard 规范
- 保持向后兼容，无 breaking change

---

## 2. 参考规范：log-quality-standard

### 2.1 核心标准

| 标准 | 名称 | 核心要求 |
|-----|------|---------|
| 标准 1 | 事件必记 | 状态改变、关键动作、硬件资源、告警、维护、组件交互必须有日志 |
| 标准 2 | 运维管理 | 自动转储、老化规则、存储要求、日志安全 |
| 标准 3 | 分级清晰 | 有问题的日志必须是 ERROR/WARNING，正常流程打 INFO |
| 标准 4 | 描述充分 | ERROR/WARNING 必须回答：什么错 + 为什么 + 查哪里 |
| 标准 5 | 组件归属明确 | 每条日志必须带组件标识 `[仓库/子模块]` |
| 标准 6 | 防刷屏 | 同一错误短时内重复必须合并为一条带计数 |
| 标准 7 | 铵路追踪 | ERROR/WARNING 必须携带 trace_id |
| 标准 8 | 隐私保护 | 禁止打印用户原始输入、PII、完整 API Key |

### 2.2 log-quality-write 工作流程

1. **第零步**：确认目标代码仓的语言和格式（必须先执行）
2. **第一步**：分析代码逻辑
3. **第二步**：确定打日志的位置
4. **第三步**：按所有标准编写每条日志
5. **第四步**：输出代码模板
6. **第五步**：自检清单

### 2.3 log-quality-rewrite 工作流程

1. **第零步**：确认目标代码仓的语言和格式
2. **第一步**：读取扫描报告
3. **第二步**：按标准逐条重写
4. **第三步**：生成修改对比表
5. **第四步**：生成代码 Diff
6. **第五步**：汇总统计

---

## 3. 改动策略

本 PR 针对以下标准进行改进：

| 标准 | 改动策略 | 优先级 |
|-----|---------|-------|
| 标准 1（事件必记） | 添加线程启动/关闭、连接建立等关键事件日志 | 高 |
| 标准 3（分级清晰） | 修正日志级别，异常不打 INFO，正常不打 ERROR | 高 |
| 标准 4（描述充分） | 添加参数快照、错误类型、分析建议 | 高 |
| 标准 5（组件归属） | 添加组件标识 `[ComponentName]` | 中 |
| 格式统一 | 统一参数格式、占位符风格 | 中 |

---

## 4. 改动分类详解

### 4.1 增强错误上下文（标准 4：描述充分）

**改动模式**：
- 添加参数快照：`type=%s, error=%s`
- 添加分析建议：`Check network and remote store.`
- 添加计数信息：`failed_count=%d, failed_blocks=%s`

**改动前后对比**：

```python
# 改动前（kv_transfer.py）
logger.error("Error in KVCacheTransferThread: %s", e)

# 改动后
logger.error(
    "Error in KVCacheTransferThread. type=%s, error=%s. Check thread state and request processing.",
    type(e).__name__,
    e,
)
```

```python
# 改动前（kv_transfer.py）
logger.error("Failed to load blocks: %s", failed_blocks)

# 改动后
logger.error(
    "Failed to load blocks. failed_count=%d, failed_blocks=%s. Check block availability and memory state.",
    len(failed_blocks),
    failed_blocks,
)
```

**涉及文件**：
- kv_transfer.py
- pool_worker.py
- mooncake_connector.py
- mooncake_hybrid_connector.py
- mooncake_layerwise_connector.py
- backend 文件（mooncake_backend.py, yuanrong_backend.py, memcache_backend.py）

### 4.2 添加组件标识（标准 5：组件归属明确）

**改动模式**：
- 在日志开头添加 `[ComponentName]` 标识

**改动前后对比**：

```python
# 改动前（mooncake_connector.py）
logger.error("Mooncake transfer server start failed: %s", e)

# 改动后
logger.error(
    "[MooncakeConnector] Mooncake transfer server start failed. "
    "local_segment_name=%s, error=%s. Check mooncake config and network.",
    local_segment_name,
    e,
)
```

**涉及文件及组件标识**：

| 文件 | 组件标识 |
|-----|---------|
| mooncake_connector.py | `[MooncakeConnector]` |
| mooncake_hybrid_connector.py | `[MooncakeHybridConnector]` |
| mooncake_layerwise_connector.py | `[MooncakeLayerwiseConnector]` |
| pool_worker.py | `[KVPoolWorker]` |

### 4.3 修正日志级别（标准 3：分级清晰）

**改动模式**：
- 异常情况不打 INFO，改为 WARNING 或 ERROR
- 正常流程不打 ERROR，改为 INFO 或 DEBUG
- 循环内日志降级为 DEBUG

**改动前后对比**：

```python
# 改动前（platform.py）
logger.warning("Model config is missing...")

# 改动后（测试场景下这是预期行为）
logger.info("Got empty model config...")
```

```python
# 改动前（pool_worker.py）
logger.info("Layerwise get failed")

# 改动后（非阻断性问题，保持 INFO 但补充分析）
logger.info(
    "Layerwise get failed. Timeout waiting for get_event. Check receiver thread status."
)
```

### 4.4 统一格式风格

**改动模式**：
- 参数格式统一：`parameter=value` 格式
- action/solution 格式：`impact: ..., solution: ...`
- 占位符修正：`%s` 而非 `{variable}`

**改动前后对比**：

```python
# 改动前（platform.py）
logger.warning(
    "'--ubatch-size' is currently ignored on Ascend NPU because it "
    "depends on the generic DBO path..."
)

# 改动后
logger.warning(
    "Parameter is currently ignored on Ascend. "
    "parameter=ubatch_size, value=%d, action: resetting to 0. ",
    ubatch_size,
)
```

```python
# 改动前（utils.py）
logger.warning(
    "Currently, communication is performed using FFTS+ method, which reduces "
    "the number of available streams..."
)

# 改动后
logger.warning(
    "Currently, communication is performed using FFTS+ method. "
    "impact: reduces available streams, limits runtime shapes. "
    "solution: set HCCL_OP_EXPANSION_MODE=AIV to improve performance and increase supported shapes. "
)
```

### 4.5 添加事件覆盖（标准 1：事件必记）

**改动模式**：
- 线程启动/关闭事件
- 连接建立事件
- Store 初始化事件

**改动前后对比**：

```python
# 改动前（kv_transfer.py）
logger.warning("Received a None request!")

# 改动后（补充说明含义）
logger.warning("Received a None request. This indicates queue shutdown or invalid request.")
```

---

## 5. 模块改动汇总

| 模块 | 改动行数 | 主要改动类型 |
|-----|---------|-------------|
| mooncake_connector.py | +106/-27 | 组件标识 + 错误上下文 + 连接事件 |
| mooncake_hybrid_connector.py | +105/-24 | 同上 |
| mooncake_layerwise_connector.py | +54/-18 | 同上 |
| platform.py | +64/-64 | 参数格式统一 + action 格式 |
| utils.py | +45/-45 | 错误上下文 + 分析建议 |
| pool_worker.py | +17/-7 | 错误上下文 |
| kv_transfer.py | +20/-10 | 错误上下文 + 计数信息 |
| ascend_config.py | +18/-18 | 参数格式统一 |
| mooncake_backend.py | +34/-10 | 错误上下文 + 组件标识 |
| yuanrong_backend.py | +29/-5 | 错误上下文 |
| memcache_backend.py | +29/-5 | 错误上下文 |
| cpu_binding.py | +27/-7 | 错误上下文 |
| pyhccl_wrapper.py | +7/-3 | 错误上下文 |
| mla_v1.py | +8/-4 | 错误上下文 |
| sfa_v1.py | +4/-2 | 错误上下文 |

**总计**：17 个文件，+429/-186 行

---

## 6. 测试与验证

### 6.1 修改的测试用例

| 测试文件 | 测试用例 | 改动原因 |
|---------|---------|---------|
| test_kv_transfer_failures.py | `test_logs_failed_blocks` | 新日志格式包含多个参数 |
| test_platform.py | `test_check_and_update_config_no_model_config_warning` | 日志级别从 WARNING 改为 INFO |
| test_platform.py | `test_check_and_update_config_enforce_eager_mode` | 修复日志捕获时机 |
| test_platform.py | `test_check_and_update_config_unsupported_compilation_level` | 修复日志捕获时机 |

### 6.2 CI 测试结果

```
- 1202 passed
- 18 skipped
- 0 failed
```

### 6.3 代码风格检查

- `ruff check`：通过（linting rules including SIM117）
- `ruff format`：通过（formatting standards）

---

## 7. 用户影响

### 7.1 日志格式变化

| 变化类型 | 说明 |
|---------|------|
| 更长的错误消息 | 包含更多上下文信息，更有用但更长 |
| 参数格式统一 | `parameter=value` 格式便于解析 |
| 组件标识 | `[ComponentName]` 便于过滤和搜索 |

**示例**：
```
# 改动前
[ERROR] Failed to load blocks: {2, 3}

# 改动后
[ERROR] Failed to load blocks. failed_count=2, failed_blocks={2, 3}. Check block availability and memory state.
```

### 7.2 日志级别调整

| 场景 | 改动前 | 改动后 |
|-----|-------|-------|
| Model config missing（测试场景） | WARNING | INFO |
| 正常 patch 操作 | WARNING | INFO/DEBUG |

### 7.3 向后兼容性

- **无 breaking change**：所有改动向后兼容
- **功能不变**：只改日志格式和内容，不改业务逻辑
- **用户收益**：更易调试和排查问题

---

## 8. 总结

### 8.1 质量提升效果

| 标准 | 改进效果 |
|-----|---------|
| 标准 1（事件必记） | 添加关键事件日志 |
| 标准 3（分级清晰） | 修正级别误用 |
| 标准 4（描述充分） | 添加参数快照和分析建议 |
| 标准 5（组件归属） | 添加组件标识 |
| 格式统一 | 统一参数格式和占位符风格 |

### 8.2 后续改进方向

- 标准 6（防刷屏）：循环内日志合并计数
- 标准 7（链路追踪）：添加 trace_id 支持
- 标准 8（隐私保护）：检查是否有敏感信息泄露

---

## 附录：改动示例汇总

### A. 错误上下文增强示例

```python
# pool_worker.py
- logger.error("Remote connection failed in contains: %s", e)
+ logger.error(
+     "Remote connection failed in get_common_prefix_length. type=%s, error=%s. "
+     "Check network and remote store.",
+     type(e).__name__,
+     e,
+ )
```

### B. 组件标识示例

```python
# mooncake_connector.py
- logger.error("Mooncake transfer server start failed: %s", e)
+ logger.error(
+     "[MooncakeConnector] Mooncake transfer server start failed. "
+     "local_segment_name=%s, error=%s. Check mooncake config and network.",
+     local_segment_name,
+     e,
+ )
```

### C. 格式统一示例

```python
# platform.py
- logger.warning("'--ubatch-size' is currently ignored on Ascend NPU...")
+ logger.warning(
+     "Parameter is currently ignored on Ascend. "
+     "parameter=ubatch_size, value=%d, action: resetting to 0. ",
+     ubatch_size,
+ )
```

### D. 分析建议示例

```python
# utils.py
- logger.warning("Currently, communication is performed using FFTS+ method...")
+ logger.warning(
+     "Currently, communication is performed using FFTS+ method. "
+     "impact: reduces available streams, limits runtime shapes. "
+     "solution: set HCCL_OP_EXPANSION_MODE=AIV to improve performance and increase supported shapes. "
+ )
```