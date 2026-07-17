# VLLM_ASCEND_ENABLE_FLASHCOMM1 迁移排除分析

## 1. 背景

本 PR 的目标是将 vllm-ascend 中的环境变量迁移到 AscendConfig 配置体系，通过 `--additional-config` 参数传入。共涉及 11 个环境变量的迁移，其中 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 被排除，仍保留环境变量读取方式。

## 2. 问题现象

将 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 迁移到 AscendConfig 后，e2e 分布式推理测试失败，报错：

```
AssertionError: Current vLLM config is not set
```

错误发生在 `enable_sp()` 函数内部调用 `get_current_vllm_config()` 时。

## 3. 根因分析

### 3.1 enable_sp() 的调用场景

`enable_sp()` 在以下场景被调用（共 23 处）：

| 场景 | 调用位置 | vLLM Config 可用性 |
|------|---------|-------------------|
| Worker 初始化 | `worker/worker.py:416, 439` | ❌ Worker 子进程中不可用 |
| 模型推理 | `worker/model_runner_v1.py` 多处 | ✅ 推理上下文中可用 |
| MoE 通信 | `ops/fused_moe/fused_moe.py:346` | ✅ 推理上下文中可用 |
| 线性层操作 | `ops/linear.py:278`, `ops/linear_op.py:460,637,676` | ✅ 推理上下文中可用 |
| MoE prepare/finalize | `ops/fused_moe/prepare_finalize.py:327,440` | ✅ 推理上下文中可用 |
| 推测解码 | `spec_decode/eagle_proposer.py:658` | ✅ 推理上下文中可用 |
| 310P FLA | `_310p/ops/fla/gdn_310.py` 多处 | ⚠️ 取决于上下文 |

### 3.2 核心问题：Worker 子进程的 Config 上下文缺失

在分布式推理场景下：

1. **主进程**通过 `set_current_vllm_config()` 设置了 vLLM Config 上下文
2. **Worker 子进程**通过 `multiprocessing` 启动，**不会继承主进程的 Config 上下文**
3. Worker 初始化阶段调用 `enable_sp()` → 尝试 `get_current_vllm_config()` → 触发 `AssertionError`

关键代码路径：
```
worker.py:Worker.init_worker()
  → worker.py:416: if enable_sp():
    → utils.py:enable_sp()
      → get_current_vllm_config()  # AssertionError!
```

### 3.3 为什么其他环境变量没有这个问题

其他 10 个环境变量（如 `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`、`VLLM_ASCEND_ENABLE_MLAPO` 等）的读取函数：

1. **调用时机**：都在推理上下文已设置之后才被调用（如 `set_ascend_forward_context()` 之后）
2. **调用位置**：都在主进程或有完整 Config 上下文的环境中
3. **不涉及 Worker 初始化阶段**

而 `enable_sp()` 的特殊性在于：
- 它在 **Worker 初始化阶段** 就被调用
- 它在 **模型加载前** 就需要确定是否启用 SP
- 它的调用时机早于 vLLM Config 上下文的设置

### 3.4 尝试过的修复方案

| 方案 | 做法 | 结果 |
|------|------|------|
| try/except 回退 | `enable_sp()` 中 catch AssertionError，回退到 `get_ascend_config()` | 治标不治本，`get_ascend_config()` 在 Worker 子进程中也可能未初始化 |
| 传入 vllm_config 参数 | 让调用方显式传入 `vllm_config` | 需要修改 23 处调用点，侵入性太大 |
| Worker 进程中设置 Config | 在 Worker 初始化时设置 `set_current_vllm_config()` | 需要修改 vLLM 核心逻辑，超出 PR 范围 |

## 4. 结论

`VLLM_ASCEND_ENABLE_FLASHCOMM1` 的迁移需要解决 Worker 子进程中 vLLM Config 上下文缺失的问题，这涉及 vLLM 核心的分布式初始化流程，不适合在本 PR 中解决。

**当前策略**：保留 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 的环境变量读取方式，待 vLLM 上游完善 Worker 子进程的 Config 上下文传递机制后，再进行迁移。

## 5. 未来方案

1. 等待 vLLM 上游在 Worker 子进程初始化时自动设置 `set_current_vllm_config()`
2. 或在 vllm-ascend 的 Worker 初始化流程中主动设置 Config 上下文
3. 上述任一方案落地后，即可将 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 迁移到 AscendConfig
