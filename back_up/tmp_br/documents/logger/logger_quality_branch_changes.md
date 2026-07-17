# logger_quality 分支改动梳理

## 改动文件列表

| # | 文件 | 改动行数 | 主要改动类型 |
|---|------|---------|-------------|
| 1 | `ascend_config.py` | +14/-5 | 参数格式改进 |
| 2 | `attention/mla_v1.py` | +4/-4 | 消息简化 |
| 3 | `attention/sfa_v1.py` | +2/-2 | 消息简化 |
| 4 | `cpu_binding.py` | +17/-10 | 参数格式改进、添加关键参数 |
| 5 | `pyhccl_wrapper.py` | +4/-3 | 添加错误参数 |
| 6 | `mooncake_connector.py` | +82/-24 | 大量改进，添加排查方向 |
| 7 | `mooncake_hybrid_connector.py` | +96/-19 | 大量改进，添加排查方向 |
| 8 | `mooncake_layerwise_connector.py` | +45/-11 | 大量改进，添加排查方向 |
| 9 | `memcache_backend.py` | +8/-3 | 参数格式改进 |
| 10 | `mooncake_backend.py` | +8/-3 | 参数格式改进 |
| 11 | `yuanrong_backend.py` | +8/-3 | 参数格式改进 |
| 12 | `kv_transfer.py` | +15/-6 | 参数格式改进 |
| 13 | `pool_worker.py` | +10/-5 | 参数格式改进 |
| 14 | `platform.py` | +32/-32 | 参数格式改进 |
| 15 | `utils.py` | +26/-19 | 参数格式改进 |

---

## 详细改动分析

### 1. ascend_config.py

**改动类型**: 参数格式改进

**改动详情**:

```python
# 改动前
logger.warning(
    "max_num_batched_tokens (%d) is smaller than "
    "profiling_chunk_config.min_chunk (%d). "
    "Clamping min_chunk to %d to avoid it being silently ignored.",
    max_batched, min_chunk, max_batched)

# 改动后
logger.warning(
    "max_num_batched_tokens is smaller than profiling_chunk_config.min_chunk. "
    "max_num_batched_tokens=%d, min_chunk=%d. "
    "Clamping min_chunk to %d to avoid it being silently ignored.",
    max_batched, min_chunk, max_batched)
```

**改进点**:
- ✅ 参数格式统一为 `key=%d`
- ✅ 消息结构更清晰

---

### 2. attention/mla_v1.py

**改动类型**: 消息简化、添加排查提示

**改动详情**:

```python
# 改动前
logger.warning_once(
    f"Layer '{layer_name}' not found in kwargs for layer sharding, skipping sharding configuration")

# 改动后
logger.warning_once(
    f"Layer '{layer_name}' not found in kwargs, skipping sharding. "
    f"Check layer_sharding config and model layer names.")
```

**改进点**:
- ✅ 消息更简洁
- ✅ 添加排查提示

---

### 3. attention/sfa_v1.py

**改动类型**: 消息简化、添加排查提示

**改动详情**:

```python
# 改动前
logger.warning_once(
    f"[SFAImpl init] Layer '{layer_name}' not found in kwargs for layer sharding, "
    "skipping sharding configuration")

# 改动后
logger.warning_once(
    f"Layer '{layer_name}' not found in kwargs, skipping sharding. "
    f"Check layer_sharding config and model layer names.")
```

**改进点**:
- ✅ 移除冗余的 `[SFAImpl init]` 前缀
- ✅ 添加排查提示

---

### 4. cpu_binding.py

**改动类型**: 参数格式改进、添加关键参数

**改动详情**:

| 行号 | 改动前 | 改动后 |
|------|--------|--------|
| 35 | `"Unknown CPU architecture '%s', CPU binding will be disabled.", arch` | `"Unknown CPU architecture. arch=%s, action: disabling CPU binding.", arch` |
| 340 | `"NPU topo affinity not found, fallback to global-slice CPU binding."` | `"NPU topo affinity not found. action: fallback to global-slice CPU binding."` |
| 411 | `"[migrate] rank:%s -> NPU%s has no CPU pool, skip memory binding.", self.rank_id, npu` | `"[migrate] NPU has no CPU pool. rank=%s, npu=%s.", self.rank_id, npu` |
| 419 | `"[migrate] NPU:%s -> NUMA %s not found, skip memory binding.", npu, target_numa` | `"[migrate] NUMA node not found. npu=%s, numa=%s. ", npu, target_numa` |
| 452 | `"[irq] rank:%s -> NPU%s has no cpu pool, skip irq binding.", self.rank_id, current_npu` | `"[irq] NPU has no CPU pool. rank=%s, npu=%s. ", self.rank_id, current_npu` |
| 476 | `"[irq] NPU%s cpu pool too small (<2), skip irq binding.", npu` | `"[irq] CPU pool too small. npu=%s, cpu_count=%d, min_required=2. ", npu, len(cpus)` |
| 498 | `"Can't find pci address of NPU%s .", npu` | `"Can't find PCI address. npu=%s. ", npu` |
| 504 | `"The msi_irqs folder cannot be found under /sys/bus/pci/devices/%s .", pci_addr` | `"The msi_irqs folder cannot be found. pci_addr=%s. ", pci_addr` |
| 514 | `"The sq_send_trigger_irq of NPU%s is not found.", npu` | `"The sq_send_trigger_irq is not found. npu=%s. ", npu` |

**改进点**:
- ✅ 参数格式统一为 `key=%s`
- ✅ 添加 `action:` 说明采取的动作
- ✅ 添加关键参数（如 `cpu_count`, `min_required`）

---

### 5. pyhccl_wrapper.py

**改动类型**: 添加错误参数

**改动详情**:

```python
# 改动前
logger.error(
    "Failed to load HCCL library from %s. "
    "It is expected if you are not running on Ascend NPUs."
    "Otherwise, the hccl library might not exist, be corrupted "
    "or it does not support the current platform %s. "
    "If you already have the library, please set the "
    "environment variable HCCL_SO_PATH"
    " to point to the correct hccl library path.",
    so_file,
    platform.platform())

# 改动后
logger.error(
    "Failed to load HCCL library. "
    "so_file=%s, error=%s. "
    "The hccl library might not exist, be corrupted "
    "or it does not support the current platform %s. "
    "If you already have the library, please set the "
    "environment variable HCCL_SO_PATH"
    " to point to the correct hccl library path.",
    so_file,
    e,
    platform.platform())
```

**改进点**:
- ✅ 添加 `error=%s` 参数，显示具体错误
- ✅ 参数格式统一

---

### 6. mooncake_connector.py

**改动类型**: 大量改进，添加排查方向

**改动详情**:

| 场景 | 改动前 | 改动后 |
|------|--------|--------|
| finish req not in reqs | `"MooncakeConnector finish req not in reqs to process."` | 添加 `request_id=%s` 和 `Possible cause:`, `Check:` |
| Invalid message format | `"Invalid message format: %s", frames` | 添加 `Expected:`, `Actual:`, `Check:` |
| Connection listener exception | `"Connection listener error: %s", e` | 添加 `Exception type:`, `Context:`, `Check:` |
| Mooncake transfer failed | `"Mooncake transfer failed, ret: %d", ret` | 添加 `remote_request_id=%s, ret=%d` |
| Failed to receive ACK | `"Failed to receive ACK for request %s", request_id` | 添加 `source=%s:%d` |

**典型改动示例**:

```python
# 改动前
logger.error("Invalid message format: %s", frames)

# 改动后
logger.error(
    "Invalid message format in KVCacheSendingThread. "
    "Expected: at least 2 frames (identity + payload). "
    "Actual: %d frames. "
    "Frames: %s. "
    "Check: Verify message sender implementation.",
    len(frames),
    frames,
)
```

**改进点**:
- ✅ 添加关键参数（`request_id`, `remote_request_id`, `source`）
- ✅ 添加期望值和实际值对比
- ✅ 添加排查方向（`Possible cause:`, `Check:`）
- ✅ 添加上下文信息（`Exception type:`, `Context:`）

---

### 7. mooncake_hybrid_connector.py

**改动类型**: 大量改进，添加排查方向

**改动详情**: 与 `mooncake_connector.py` 类似，主要改进：
- ✅ 添加关键参数
- ✅ 添加期望值和实际值对比
- ✅ 添加排查方向

---

### 8. mooncake_layerwise_connector.py

**改动类型**: 大量改进，添加排查方向

**改动详情**:

| 场景 | 改动前 | 改动后 |
|------|--------|--------|
| Failed to transfer KV cache | 无详细参数 | 添加 `layer_idx=%s, error=%s` 和排查提示 |
| Mooncake transfer failed | 无详细参数 | 添加 `req_ids=%s, destination=%s, ret=%d` |
| Invalid message format | 简单错误 | 添加 `expected>=2 frames, got %d. frames=%s` |
| Failed to connect to metaserver | 无详细参数 | 添加 `url=%s, retry=%d` |

**改进点**:
- ✅ 添加关键参数
- ✅ 添加排查方向
- ✅ 添加重试信息

---

### 9-11. backend 文件 (memcache_backend.py, mooncake_backend.py, yuanrong_backend.py)

**改动类型**: 参数格式改进

**改动详情**: 统一参数格式为 `key=%s`

---

### 12. kv_transfer.py

**改动类型**: 参数格式改进

**改动详情**: 添加关键参数，统一参数格式

---

### 13. pool_worker.py

**改动类型**: 参数格式改进

**改动详情**: 添加关键参数，统一参数格式

---

### 14. platform.py

**改动类型**: 参数格式改进

**改动详情**:

```python
# 改动前
logger.warning(
    "NPU does not support compilation mode. "
    "mode=%s, action: setting CUDAGraphMode to NONE.",
    compilation_config.mode)

# 改动后（保持不变，已经是正确格式）
logger.warning(
    "NPU does not support compilation mode. mode=%s, action: setting CUDAGraphMode to NONE.",
    compilation_config.mode)
```

**改进点**:
- ✅ 参数格式统一
- ✅ 添加 `action:` 说明

---

### 15. utils.py

**改动类型**: 参数格式改进

**改动详情**:

```python
# 改动前
logger.warning(
    "Failed to register custom ops, all custom ops will be disabled. "
    "The custom ops library might not be installed or the environment is not configured correctly. "
    "Please check the custom ops installation and environment variables.")

# 改动后
logger.warning(
    "Failed to register custom ops, all custom ops will be disabled. "
    "The custom ops library might not be installed or the environment is not configured correctly. "
    "Please check the custom ops installation and environment variables.")
```

**改进点**:
- ✅ 保持原有格式（已经符合标准）

---

## 改动模式总结

### 模式 1: 参数格式统一

**改动前**:
```python
logger.warning("Unknown CPU architecture '%s', CPU binding will be disabled.", arch)
```

**改动后**:
```python
logger.warning("Unknown CPU architecture. arch=%s, action: disabling CPU binding.", arch)
```

**特点**:
- 参数使用 `key=%s` 格式
- 添加 `action:` 说明采取的动作

---

### 模式 2: 添加关键参数

**改动前**:
```python
logger.error("Mooncake transfer failed, ret: %d", ret)
```

**改动后**:
```python
logger.error("Mooncake transfer failed for request. remote_request_id=%s, ret=%d.", remote_request_id, ret)
```

**特点**:
- 添加与错误相关的关键参数
- 参数格式统一

---

### 模式 3: 添加期望值和实际值

**改动前**:
```python
logger.error("Invalid message format: %s", frames)
```

**改动后**:
```python
logger.error(
    "Invalid message format in KVCacheSendingThread. "
    "Expected: at least 2 frames (identity + payload). "
    "Actual: %d frames. "
    "Frames: %s. "
    "Check: Verify message sender implementation.",
    len(frames),
    frames,
)
```

**特点**:
- 明确说明期望值
- 明确说明实际值
- 添加排查方向

---

### 模式 4: 添加排查方向

**改动前**:
```python
logger.warning("Layer not found in kwargs, skipping sharding configuration")
```

**改动后**:
```python
logger.warning(
    "Layer not found in kwargs, skipping sharding. "
    "Check layer_sharding config and model layer names.")
```

**特点**:
- 简短精准的排查提示
- 具体分析可定位方向

---
