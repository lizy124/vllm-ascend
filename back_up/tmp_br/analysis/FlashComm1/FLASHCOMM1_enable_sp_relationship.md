# VLLM_ASCEND_ENABLE_FLASHCOMM1 与 enable_sp() 的关系说明

## 结论

`VLLM_ASCEND_ENABLE_FLASHCOMM1` 和 `enable_sp()` 强相关，是因为在 vLLM Ascend 代码里，FlashComm1 的运行期开关并不是到处直接读取环境变量，而是集中通过 `enable_sp()` 对外暴露。

迁移前的核心关系是：

```text
VLLM_ASCEND_ENABLE_FLASHCOMM1 -> envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1 -> enable_sp() -> 各个 FlashComm1/SP 调用点
```

迁移后的核心关系是：

```text
additional_config.enable_flashcomm1
  > AscendConfig.enable_flashcomm1
  > VLLM_ASCEND_ENABLE_FLASHCOMM1 fallback
  -> enable_sp()
  -> 各个 FlashComm1/SP 调用点
```

因此，迁移 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 时，不能只在 `AscendConfig` 里新增字段，还必须同步改 `enable_sp()`。

## 为什么名字是 enable_sp()，但控制的是 FlashComm1

`enable_sp()` 里的 `sp` 可以理解为 sequence parallel / SP 相关通信路径的总开关。在当前实现中，FlashComm1 是这条 SP 通信优化路径的主要启用条件。

也就是说，代码层面没有把所有调用点都命名成 `enable_flashcomm1()`，而是历史上通过 `enable_sp()` 来表示：当前是否启用 FlashComm1/SP 这套并行通信优化路径。

所以虽然环境变量叫：

```text
VLLM_ASCEND_ENABLE_FLASHCOMM1
```

但下游判断通常写成：

```python
enable_sp(...)
```

这就是二者强相关的原因。

## enable_sp() 是 FlashComm1 的统一入口

当前 `enable_sp()` 位于：

```text
vllm_ascend/utils.py
```

它负责把用户配置转换成一个稳定的布尔值，并缓存到模块级变量 `_ENABLE_SP` 中。

迁移后逻辑是：

```python
if additional_config is not None and "enable_flashcomm1" in additional_config:
    _ENABLE_SP = bool(additional_config["enable_flashcomm1"])
else:
    try:
        _ENABLE_SP = get_ascend_config().enable_flashcomm1
    except RuntimeError:
        _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
```

这说明 `enable_sp()` 是 `enable_flashcomm1` config 和旧环境变量 fallback 的汇合点。

## 为什么不能只改 AscendConfig

很多下游代码并不直接访问：

```python
get_ascend_config().enable_flashcomm1
```

而是调用：

```python
enable_sp()
```

或：

```python
enable_sp(vllm_config)
```

如果只新增 `AscendConfig.enable_flashcomm1`，但不改 `enable_sp()`，这些调用点仍然会继续走旧逻辑，迁移不会真正生效。

更重要的是，`enable_sp()` 支持无参调用。部分 worker、op、初始化路径可能没有 current vLLM config 上下文。如果把所有逻辑强行改成只读 `get_ascend_config().enable_flashcomm1`，这些路径可能触发：

```text
RuntimeError: Ascend config is not initialized. Please call init_ascend_config first.
```

或者：

```text
AssertionError: Current vLLM config is not set
```

因此，`enable_sp()` 必须保留安全 fallback。

## 典型调用链

FlashComm1/SP 判断会影响多类路径：

```text
模型/worker/op 逻辑
  -> enable_sp(vllm_config) 或 enable_sp()
  -> _ENABLE_SP
  -> additional_config.enable_flashcomm1 / AscendConfig.enable_flashcomm1 / VLLM_ASCEND_ENABLE_FLASHCOMM1
```

典型调用包括：

- `ascend_forward_context.py`：根据 `enable_sp(vllm_config)` 判断是否走 FlashComm1 通信路径。
- `worker` / `model_runner` 路径：通过 `enable_sp()` 决定 SP/FlashComm1 相关行为。
- `ops` 路径：部分算子逻辑通过 `enable_sp()` 判断是否启用对应通信优化。
- `shared_expert_dp_enabled()`：组合判断 `enable_shared_expert_dp`、`enable_sp()`、`enable_sp_by_pass()`。

所以 `enable_sp()` 是比环境变量本身更靠近业务行为的入口。

## shared_expert_dp 的特殊关系

`enable_sp()` 还有一个额外逻辑：

```python
if not _ENABLE_SP and enable_shared_expert_dp:
    _ENABLE_SP = True
```

也就是说，即使用户没有显式开启 FlashComm1，如果 `shared_expert_dp` 需要 SP 路径，`enable_sp()` 也会被强制置为 true。

这进一步说明 `enable_sp()` 不是简单的环境变量 getter，而是 FlashComm1/SP 启用状态的统一决策函数。

## FlashComm2 warning 也要跟随 enable_flashcomm1

迁移前，FlashComm2 校验里曾直接读取：

```python
envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
```

这样会导致一个问题：

```text
用户通过 additional_config.enable_flashcomm1=true 开启 FlashComm1，
但 FlashComm2 warning 仍然看到 env=false，误报建议开启 FlashComm1。
```

因此迁移后 FlashComm2 warning 应该改为读取统一后的配置结果，例如：

```python
ascend_config.enable_flashcomm1
```

这样 config 和 env fallback 的行为才一致。

## 迁移设计要点

迁移 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 时，需要同时满足三点：

1. `AscendConfig` 新增 `enable_flashcomm1` 字段。
2. `enable_sp()` 改为优先读取 `additional_config.enable_flashcomm1` / `AscendConfig.enable_flashcomm1`，旧环境变量只作为 fallback。
3. 所有直接依赖 FlashComm1 状态的判断，例如 FlashComm2 warning，不能继续只读旧环境变量。

最终目标是把用户入口迁移到：

```bash
--additional-config '{"enable_flashcomm1": true}'
```

过渡期仍兼容：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

## 总结

`VLLM_ASCEND_ENABLE_FLASHCOMM1` 是用户侧旧开关，`enable_sp()` 是代码侧统一行为入口。

迁移这个环境变量，本质上不是简单替换一个 env 读取点，而是要把 `enable_sp()` 这个统一入口切换到 config 优先、env fallback 的模式。只有这样，下游所有 FlashComm1/SP 调用路径才能同时完成迁移，并保持无 config 上下文场景下的兼容性。
