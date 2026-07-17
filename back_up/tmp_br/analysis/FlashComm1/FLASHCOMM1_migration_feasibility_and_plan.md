# VLLM_ASCEND_ENABLE_FLASHCOMM1 迁移可行性与方案

## 结论

`VLLM_ASCEND_ENABLE_FLASHCOMM1` 可以迁移到 `AscendConfig.additional_config`，但不能直接删除环境变量，也不能把 `enable_sp()` 简单改成只读 `get_ascend_config().enable_flashcomm1`。

推荐采用过渡迁移：

```text
additional_config.enable_flashcomm1 > VLLM_ASCEND_ENABLE_FLASHCOMM1 > False
```

也就是：

- 新增 config 入口：`--additional-config '{"enable_flashcomm1": true}'`
- 暂时保留旧环境变量：`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`
- 显式 config 优先于环境变量
- 后续再逐步废弃并移除环境变量

## 当前代码依据

### 1. 它不是 import-time monkey patch gate

当前 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 定义在：

```python
# vllm_ascend/envs.py
"VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
```

它没有出现在 `vllm_ascend/patch/platform/__init__.py` 的 early import gate 中。

因此它不像 `VLLM_ASCEND_BALANCE_SCHEDULING` 那样，在 `AscendConfig` 初始化前决定是否导入 monkey patch。

这说明它具备迁移到 config 的基础条件。

### 2. 核心入口是 `enable_sp()`

当前核心读取点在：

```python
# vllm_ascend/utils.py

def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except AssertionError:
            vllm_config = None

    additional_config = getattr(vllm_config, "additional_config", None) if vllm_config is not None else None
    refresh = additional_config.get("refresh", False) if additional_config else False

    if _ENABLE_SP is None or refresh:
        _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1

        if not _ENABLE_SP and enable_shared_expert_dp:
            _ENABLE_SP = True
            logger.info("shared_expert_dp requires enable_sp = True. has set enable_sp to True")

    return _ENABLE_SP
```

注意这里有两个关键点：

1. `enable_sp()` 支持无参调用。
2. 如果 `get_current_vllm_config()` 不可用，会捕获 `AssertionError` 并回退到 env。

这说明当前代码已经考虑到某些调用场景没有当前 vLLM config 上下文。

### 3. 当前有大量无参调用

当前仓里存在不少无参 `enable_sp()` 调用，例如：

- `vllm_ascend/worker/worker.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/ops/linear.py`
- `vllm_ascend/ops/linear_op.py`
- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `vllm_ascend/ops/fused_moe/prepare_finalize.py`
- `vllm_ascend/spec_decode/eagle_proposer.py`
- `vllm_ascend/_310p/ops/fla/gdn_310.py`

这也是 FlashComm1 迁移的主要风险点。

如果迁移后 `enable_sp()` 强依赖 `get_current_vllm_config()` 或 `get_ascend_config()`，就可能在 Worker / 初始化 / 无上下文路径重新触发：

```text
AssertionError: Current vLLM config is not set
```

所以迁移方案必须保留无 config 上下文下的安全 fallback。

### 4. 当前已不再读取旧变量 `VLLM_ASCEND_ENABLE_FLASHCOMM`

当前仓中 `enable_sp()` 只读取：

```python
envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
```

没有再读取旧变量：

```text
VLLM_ASCEND_ENABLE_FLASHCOMM
```

因此迁移方案不建议重新引入 `VLLM_ASCEND_ENABLE_FLASHCOMM`，否则会扩大兼容面并增加配置语义复杂度。

## 推荐迁移方案

### 1. `AscendConfig` 新增字段

在 `vllm_ascend/ascend_config.py` 中新增：

```python
self.enable_flashcomm1 = additional_config.get(
    "enable_flashcomm1",
    ascend_envs.VLLM_ASCEND_ENABLE_FLASHCOMM1,
)
```

含义：

- 用户显式传 `additional_config.enable_flashcomm1` 时，以 config 为准。
- 用户没有传 config 时，继续使用旧环境变量。
- 默认值仍是 false。

### 2. 修改 `enable_sp()`

推荐实现：

```python
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except AssertionError:
            vllm_config = None

    additional_config = getattr(vllm_config, "additional_config", None) if vllm_config is not None else None
    refresh = additional_config.get("refresh", False) if additional_config else False

    if _ENABLE_SP is None or refresh:
        if additional_config is not None and "enable_flashcomm1" in additional_config:
            _ENABLE_SP = bool(additional_config["enable_flashcomm1"])
        else:
            try:
                _ENABLE_SP = bool(get_ascend_config().enable_flashcomm1)
            except RuntimeError:
                _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1

        if not _ENABLE_SP and enable_shared_expert_dp:
            _ENABLE_SP = True
            logger.info("shared_expert_dp requires enable_sp = True. has set enable_sp to True")

    return _ENABLE_SP
```

这里的优先级是：

```text
传入的 vllm_config.additional_config.enable_flashcomm1
  > 已初始化的 AscendConfig.enable_flashcomm1
  > VLLM_ASCEND_ENABLE_FLASHCOMM1
```

这样可以覆盖三类场景：

1. `AscendConfig.__init__()` 中调用 `enable_sp(vllm_config=...)`，此时 `_ASCEND_CONFIG` 还没完成缓存，但传入的 `vllm_config.additional_config` 可用。
2. 推理路径中 `AscendConfig` 已初始化，可以读取 `get_ascend_config().enable_flashcomm1`。
3. Worker / 初始化 / 无当前 config 上下文路径中，继续回退到环境变量，不抛错。

### 3. 保留 `_ENABLE_SP` 缓存与 refresh 语义

当前 `enable_sp()` 使用模块级缓存 `_ENABLE_SP`。

迁移后仍应保留：

```python
if _ENABLE_SP is None or refresh:
    ...
```

并继续依赖 `clear_ascend_config()` 调用 `clear_enable_sp()` 清理缓存。

这样可以避免重复计算，也能保持测试和 refresh 行为一致。

### 4. 修改 FlashComm2 warning 判断

当前 `get_flashcomm2_config_and_validate()` 中仍直接读取 env：

```python
if not envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1:
    logger.warning_once(
        "It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance."
    )
```

迁移后应改成基于统一 config 结果判断，否则会出现：

```text
用户通过 additional_config.enable_flashcomm1=true 开启 FlashComm1，
但 FlashComm2 校验仍读 env=false，误打印 warning。
```

推荐新增 helper：

```python
def flashcomm1_enabled(ascend_config=None, vllm_config=None) -> bool:
    if vllm_config is not None:
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        if "enable_flashcomm1" in additional_config:
            return bool(additional_config["enable_flashcomm1"])
    if ascend_config is not None:
        return bool(ascend_config.enable_flashcomm1)
    return enable_sp(vllm_config)
```

然后改成：

```python
if not flashcomm1_enabled(ascend_config, vllm_config):
    logger.warning_once(
        "It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance."
    )
```

### 5. 保留 env 过渡期

`vllm_ascend/envs.py` 中继续保留：

```python
"VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
```

可以把注释调整为：

```python
# DEPRECATED: VLLM_ASCEND_ENABLE_FLASHCOMM1 will be removed in a future release.
# Use --additional-config '{"enable_flashcomm1": true}'.
```

但不要在本阶段删除 env。

## 需要同步补充的测试

### 1. `enable_sp()` 优先级测试

至少覆盖：

1. config 未设置，env false，返回 false。
2. config 未设置，env true，返回 true。
3. config true，env false，返回 true。
4. config false，env true，返回 false。
5. `additional_config.refresh=true` 后能重新计算 `_ENABLE_SP`。
6. `get_current_vllm_config()` 抛 `AssertionError` 时，仍回退 env，不抛错。

### 2. `AscendConfig` fallback 测试

覆盖：

1. `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 且 config 未设置时，`ascend_config.enable_flashcomm1 is True`。
2. `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 且 config 显式 `enable_flashcomm1=False` 时，`ascend_config.enable_flashcomm1 is False`。
3. `VLLM_ASCEND_ENABLE_FLASHCOMM1=0` 且 config 显式 `enable_flashcomm1=True` 时，`ascend_config.enable_flashcomm1 is True`。

### 3. FlashComm2 warning 测试

覆盖：

1. FlashComm2 开启，FlashComm1 env/config 都 false，打印 warning。
2. FlashComm2 开启，`enable_flashcomm1=True`，不打印 warning。
3. FlashComm2 开启，env true 且 config 未设置，不打印 warning。

### 4. 下游行为测试保留现状

之前已经把很多 UT 改成 mock `get_ascend_config()` 或 mock config 字段，这部分不需要回滚。

迁移后应该额外补 compatibility tests，而不是把新 config 路径测试改回 env 路径。

## 风险与规避

### 风险 1：Worker 子进程没有 current vLLM config

规避：`enable_sp()` 不能强依赖 `get_current_vllm_config()`，必须保留 fallback。

### 风险 2：`AscendConfig.__init__()` 内部调用 `enable_sp()` 时 singleton 未完成初始化

规避：`enable_sp(vllm_config=...)` 优先从传入的 `vllm_config.additional_config` 读取，不要先强制 `get_ascend_config()`。

### 风险 3：缓存 `_ENABLE_SP` 复用旧值

规避：保留 `refresh` 判断，并确保 `clear_ascend_config()` 继续调用 `clear_enable_sp()`。

### 风险 4：FlashComm2 warning 与实际配置不一致

规避：`get_flashcomm2_config_and_validate()` 不再直接读 env，改用统一 helper 或 `ascend_config.enable_flashcomm1`。

## 最终建议

可以迁移 `VLLM_ASCEND_ENABLE_FLASHCOMM1`，但要采用兼容期迁移：

```text
additional_config.enable_flashcomm1 > VLLM_ASCEND_ENABLE_FLASHCOMM1 > False
```

不要直接删除 env，也不要把 `enable_sp()` 改成只读 `AscendConfig`。

推荐实施顺序：

1. `AscendConfig` 新增 `enable_flashcomm1`，默认 fallback 到 env。
2. `enable_sp()` 改为 config 优先、env fallback，并保留无 current config 的安全路径。
3. `get_flashcomm2_config_and_validate()` 改用统一的 FlashComm1 判断。
4. 补 `enable_sp()`、`AscendConfig`、FlashComm2 warning 的兼容测试。
5. 文档标记 env 为 deprecated，后续版本再移除。
