# vLLM-Ascend 环境变量配置迁移设计文档

## 1. 背景与动机

### 1.1 当前问题

vLLM-Ascend 项目长期使用环境变量作为 Ascend 特有功能的配置方式，存在以下问题：

1. **配置分散，难以发现**：环境变量散落在代码各处，用户需要翻阅源码或文档才能找到可配置项，学习成本高。

2. **配置入口不一致**：vLLM 核心配置使用 `--additional-config` 参数，而 Ascend 特有配置使用环境变量，用户需要掌握两种不同的配置方式。

3. **配置难以追溯**：环境变量通常在 shell 中设置，不会体现在启动命令中，导致：
   - 难以审计配置来源
   - 难以复现问题
   - 配置无法纳入版本控制

4. **类型安全性差**：环境变量均为字符串类型，需要手动类型转换，容易引入错误。

5. **与 vLLM 生态不对齐**：vLLM 主项目使用 `additional_config` 机制，Ascend 扩展使用不同范式，增加用户认知负担。

### 1.2 目标用户

- **vLLM-Ascend 用户**：需要配置 Ascend 特有功能以优化推理性能
- **运维人员**：需要管理和部署 vLLM-Ascend 服务
- **开发者**：需要添加新的 Ascend 特有配置项

## 2. 设计目标

1. **配置集中化**：所有 Ascend 特有配置统一在 `AscendConfig` 中管理
2. **易用性提升**：统一配置入口，降低用户学习成本
3. **可追溯性**：配置显式可见，便于审计和问题排查
4. **生态对齐**：与 vLLM 主项目配置范式保持一致
5. **平滑迁移**：保持向后兼容，给用户充足的迁移时间

## 3. 技术方案

### 3.1 配置优先级机制

配置获取遵循以下优先级顺序：

```
additional_config 显式值 > 环境变量 > 默认值
```

### 3.2 统一配置获取方法

在 `AscendConfig` 类中新增 `_get_config_value()` 静态方法：

```python
@staticmethod
def _get_config_value(additional_config: dict[str, Any], config_key: str, env_key: str, env_value: Any) -> Any:
    # 优先级 1: additional_config 显式值
    if config_key in additional_config:
        value = additional_config[config_key]
        logger.info_once(f"AscendConfig.{config_key} is set from additional_config with value {value}.")
        return value

    # 优先级 2: 环境变量（带弃用警告）
    if env_key in os.environ:
        logger.info_once(
            f"AscendConfig.{config_key} falls back to environment variable {env_key} with value {env_value}. "
            f"Please use additional_config.{config_key} instead, because {env_key} will be removed in the "
            "next release."
        )
    return env_value
```

### 3.3 配置项使用示例

```python
# 在 AscendConfig.__init__ 中使用
self.enable_mlapo = self._get_config_value(
    additional_config,
    "enable_mlapo",
    "VLLM_ASCEND_ENABLE_MLAPO",
    ascend_envs.VLLM_ASCEND_ENABLE_MLAPO,
)
```

## 4. 迁移的环境变量清单

**迁移数量：10 个**

| 序号 | 环境变量 | 配置项 | 默认值 | 说明 |
|-----|---------|--------|-------|------|
| 1 | `VLLM_ASCEND_BALANCE_SCHEDULING` | `enable_balance_scheduling` | `False` | 启用负载均衡调度 |
| 2 | `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `enable_flashcomm1` | `False` | 启用 FlashComm1 |
| 3 | `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `enable_matmul_allreduce` | `False` | 启用 Matmul AllReduce |
| 4 | `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | `enable_flashcomm2_parallel_size` | `0` | FlashComm2 并行大小 |
| 5 | `MSMONITOR_USE_DAEMON` | `msmonitor_use_daemon` | `False` | MSMonitor 使用守护进程 |
| 6 | `VLLM_ASCEND_ENABLE_MLAPO` | `enable_mlapo` | `True` | 启用 MLAPO |
| 7 | `VLLM_ASCEND_ENABLE_NZ` | `weight_nz_mode` | `1` | 权重 NZ 模式 (0: 禁用, 1: 仅量化, 2: 全部) |
| 8 | `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | `enable_context_parallel` | `False` | 启用上下文并行 |
| 9 | `VLLM_ASCEND_ENABLE_FUSED_MC2` | `enable_fused_mc2` | `0` | 启用 Fused MC2 |
| 10 | `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `enable_transpose_kv_cache_by_block` | `True` | 按块转置 KV Cache |

### 4.1 迁移挑战分析

在 10 个环境变量中，有 2 个存在特殊的迁移挑战，需要在 `AscendConfig` 初始化之前被读取。

#### 4.1.1 `VLLM_ASCEND_ENABLE_FLASHCOMM1` → `enable_flashcomm1`

**挑战原因**：

该配置项控制 Sequence Parallel (SP) 功能，需要在早期阶段确定：
- Worker 初始化阶段
- Parallel state 设置阶段
- 这些阶段可能在 `AscendConfig` 完整初始化之前发生

**解决方案**：

在 `vllm_ascend/utils.py` 的 `enable_sp()` 函数中实现了三层 fallback 逻辑：

```python
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    # ... 获取 vllm_config ...

    additional_config = getattr(vllm_config, "additional_config", None)

    if _ENABLE_SP is None or refresh:
        # 第一层：直接从 additional_config 获取
        if additional_config is not None and "enable_flashcomm1" in additional_config:
            _ENABLE_SP = bool(additional_config["enable_flashcomm1"])
        else:
            # 第二层：尝试从 AscendConfig 获取
            try:
                _ENABLE_SP = get_ascend_config().enable_flashcomm1
            except RuntimeError:
                # 第三层：fallback 到环境变量
                _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1

    return bool(_ENABLE_SP)
```

**关键设计点**：
- 使用全局变量 `_ENABLE_SP` 缓存结果，避免重复计算
- 三层 fallback 确保在任何阶段都能正确获取配置
- 测试覆盖：`test_enable_sp_falls_back_to_env_without_current_config` 验证了 AscendConfig 未初始化时的 fallback 行为

#### 4.1.2 `MSMONITOR_USE_DAEMON` → `msmonitor_use_daemon`

**挑战原因**：

该配置项控制 MSMonitor 工具的使用，与 torch profiler 存在互斥关系：
- Profiler 初始化可能在 `AscendConfig` 初始化之前
- 需要在 profiler 创建时立即检查配置，避免两者同时启用

**解决方案**：

在 `vllm_ascend/profiler/torch_npu_profiler.py` 中使用 `suppress` 处理：

```python
@staticmethod
def _create_profiler(profiler_config: ProfilerConfig, trace_name: str) -> Any:
    # 第一层：直接读取环境变量
    msmonitor_use_daemon = envs_ascend.MSMONITOR_USE_DAEMON

    # 第二层：尝试从 AscendConfig 获取（可能失败）
    with suppress(RuntimeError):
        msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon

    if msmonitor_use_daemon:
        raise RuntimeError("MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.")
    # ... 创建 profiler ...
```

**关键设计点**：
- 使用 `suppress(RuntimeError)` 静默处理 AscendConfig 未初始化的情况
- 环境变量作为兜底方案，确保配置始终可读
- **注意**：当前缺少针对此场景的专门测试覆盖

#### 4.1.3 迁移挑战总结

| 环境变量 | 挑战类型 | 解决方案 | 测试覆盖 |
|---------|---------|---------|---------|
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 早期阶段读取 | 三层 fallback + 全局缓存 | ✅ 有测试 |
| `MSMONITOR_USE_DAEMON` | Profiler 初始化顺序 | suppress(RuntimeError) | ❌ 缺少测试 |

**建议**：为 `MSMONITOR_USE_DAEMON` 的 fallback 场景补充专门的测试用例。

## 5. 向后兼容性设计

### 5.1 过渡期策略

1. **环境变量继续有效**：在过渡期内，环境变量仍然可以正常工作
2. **弃用警告**：当使用环境变量时，输出弃用警告提示用户迁移
3. **版本计划**：
   - 当前版本：环境变量可用，输出弃用警告
   - 下个版本：环境变量继续可用，警告升级
   - 未来版本：移除环境变量支持

### 5.2 弃用警告信息

```
AscendConfig.enable_mlapo falls back to environment variable VLLM_ASCEND_ENABLE_MLAPO with value True.
Please use additional_config.enable_mlapo instead, because VLLM_ASCEND_ENABLE_MLAPO will be removed in the next release.
```

## 6. 用户迁移指南

### 6.1 配置方式对比

**旧方式（环境变量）：**
```bash
export VLLM_ASCEND_ENABLE_MLAPO=true
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_BALANCE_SCHEDULING=true
vllm serve model
```

**新方式（additional_config）：**
```bash
vllm serve model --additional-config '{
  "enable_mlapo": true,
  "weight_nz_mode": 2,
  "enable_balance_scheduling": true
}'
```

### 6.2 迁移步骤

1. **识别现有配置**：检查当前使用的环境变量
2. **创建配置文件**：将环境变量转换为 JSON 配置
3. **更新启动命令**：使用 `--additional-config` 参数
4. **验证功能**：确保配置生效，功能正常
5. **移除环境变量**：清理旧的环境变量设置

### 6.3 配置文件示例

创建 `ascend_config.json`：
```json
{
  "enable_mlapo": true,
  "weight_nz_mode": 2,
  "enable_balance_scheduling": false,
  "enable_flashcomm1": false,
  "enable_matmul_allreduce": true,
  "enable_context_parallel": false,
  "enable_fused_mc2": 0,
  "enable_transpose_kv_cache_by_block": true,
  "msmonitor_use_daemon": false,
  "enable_flashcomm2_parallel_size": 0
}
```

启动命令：
```bash
vllm serve model --additional-config @ascend_config.json
```

## 7. 开发者指南

### 7.1 添加新的配置项

1. **在 `vllm_ascend/envs.py` 中定义环境变量**（可选）：
```python
VLLM_ASCEND_NEW_FEATURE = bool(int(os.getenv("VLLM_ASCEND_NEW_FEATURE", "0")))
```

2. **在 `AscendConfig.__init__` 中添加配置项**：
```python
self.enable_new_feature = self._get_config_value(
    additional_config,
    "enable_new_feature",
    "VLLM_ASCEND_NEW_FEATURE",
    ascend_envs.VLLM_ASCEND_NEW_FEATURE,
)
```

3. **添加单元测试**：
```python
def test_new_feature_config():
    # 测试 additional_config 优先级
    config = create_test_config(additional_config={"enable_new_feature": True})
    assert config.enable_new_feature == True

    # 测试环境变量回退
    os.environ["VLLM_ASCEND_NEW_FEATURE"] = "1"
    config = create_test_config(additional_config={})
    assert config.enable_new_feature == True
```

### 7.2 读取配置

在代码中通过 `get_ascend_config()` 获取配置：
```python
from vllm_ascend import get_ascend_config

ascend_config = get_ascend_config()
if ascend_config.enable_new_feature:
    # 执行新功能逻辑
    pass
```

### 7.3 测试要求

- 必须测试 `additional_config` 优先级
- 必须测试环境变量回退逻辑
- 必须测试弃用警告输出
- 必须测试默认值

## 8. 风险评估与测试

### 8.1 风险评估

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| 环境变量用户未迁移 | 功能中断 | 提供过渡期，输出弃用警告 |
| 配置项名称不一致 | 用户困惑 | 保持配置项名称与环境变量语义一致 |
| 类型转换错误 | 运行时错误 | JSON 配置支持原生类型，减少转换 |

### 8.2 测试覆盖

1. **单元测试**：`tests/ut/test_ascend_config.py`
   - 配置优先级测试
   - 环境变量回退测试
   - 弃用警告测试
   - 默认值测试

2. **集成测试**：
   - 端到端配置验证
   - 多配置项组合测试

3. **回归测试**：
   - 确保现有功能不受影响
   - 性能回归测试

## 9. 总结

本次配置迁移是 vLLM-Ascend 易用性改进的重要一步，通过：

1. 统一配置入口，降低用户学习成本
2. 提高配置可追溯性，便于问题排查
3. 与 vLLM 生态对齐，提供一致的用户体验
4. 保持向后兼容，确保平滑迁移

为后续添加更多 Ascend 特有配置建立了清晰的范式。