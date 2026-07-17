# PR #9064 - vllm_ascend/utils.py 改动详解

## 改动总览

utils.py 的改动分为 6 类，共涉及约 150 行代码变更：

1. 环境变量迁移到 Config 读取（7 个函数）
2. enable_sp() 重构（worker 进程安全处理）
3. clear_enable_sp() 新增函数（测试状态重置）
4. vllm_version_is() 安全改造
5. 删除 bootstrap_custom_op_env 相关代码
6. 310P MRotaryEmbedding 注册方式调整

---

## 1. 环境变量迁移到 Config 读取

### 1.1 matmul_allreduce_enable()

```python
# 改动前
def matmul_allreduce_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE

# 改动后
def matmul_allreduce_enable() -> bool:
    return get_ascend_config().enable_matmul_allreduce
```

**说明**：直接从 `envs_ascend` 读取环境变量改为从 `AscendConfig` 读取配置。`enable_matmul_allreduce` 默认值为 `False`，与原环境变量默认值 `"0"` 一致。

**影响范围**：`vllm_ascend/ops/linear_op.py` 中的 `get_row_parallel_op()` 函数，决定是否使用 `MatmulAllreduceRowParallelOp` 融合算子。

---

### 1.2 prefill_context_parallel_enable()

```python
# 改动前
def prefill_context_parallel_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL

# 改动后
def prefill_context_parallel_enable() -> bool:
    return get_ascend_config().enable_context_parallel
```

**说明**：Context Parallel（上下文并行）开关，默认 `False`。

**影响范围**：控制 prefill 阶段是否启用上下文并行，影响长序列的 prefill 性能。

---

### 1.3 flashcomm2_enable()

```python
# 改动前
def flashcomm2_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE > 0

# 改动后
def flashcomm2_enable() -> bool:
    config_val = get_ascend_config().enable_flashcomm2_parallel_size
    return config_val > 0
```

**说明**：FlashComm2 的开关判断逻辑不变（值 > 0 即开启），只是数据来源从环境变量改为 Config。

**影响范围**：`vllm_ascend/ops/linear_op.py` 中的 `get_row_parallel_op()` 和 `get_col_parallel_op()` 函数，决定是否使用 FlashComm2 优化算子。

---

### 1.4 get_flashcomm2_config_and_validate()

```python
# 改动前
def get_flashcomm2_config_and_validate(ascend_config, vllm_config):
    flashcomm2_oproj_tp_size = envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
    global_tp_size = vllm_config.parallel_config.tensor_parallel_size
    if not flashcomm2_enable():
        return 0
    ...

# 改动后
def get_flashcomm2_config_and_validate(ascend_config, vllm_config):
    flashcomm2_oproj_tp_size = ascend_config.enable_flashcomm2_parallel_size
    global_tp_size = vllm_config.parallel_config.tensor_parallel_size
    if ascend_config.enable_flashcomm2_parallel_size <= 0:
        return 0
    ...
```

**说明**：此函数已经接收 `ascend_config` 参数，所以直接从参数读取，不再调用 `envs_ascend` 或 `flashcomm2_enable()`。同时把 `if not flashcomm2_enable()` 改为 `if ascend_config.enable_flashcomm2_parallel_size <= 0`，避免在已有 ascend_config 的情况下重复调用 `get_ascend_config()`。

**影响范围**：FlashComm2 配置校验和初始化。

---

### 1.5 find_hccl_library()

```python
# 改动前
def find_hccl_library() -> str:
    so_file = envs_ascend.HCCL_SO_PATH
    if so_file:
        logger.info("Found hccl from environment variable HCCL_SO_PATH=%s", so_file)
    ...

# 改动后
def find_hccl_library() -> str:
    config = get_ascend_config()
    so_file = config.hccl_so_path
    if so_file:
        logger.info("Found hccl from Config hccl_so_path=%s", so_file)
    ...
```

**说明**：HCCL 库路径的配置来源从环境变量 `HCCL_SO_PATH` 改为 Config 参数 `hccl_so_path`。日志信息也做了相应更新。

**影响范围**：HCCL 通信库的加载路径，直接影响分布式训练/推理的通信初始化。

---

### 1.6 _should_trans_nz()

```python
# 改动前
def _should_trans_nz(weight: torch.Tensor) -> bool:
    if is_310p():
        return True
    if not envs_ascend.VLLM_ASCEND_ENABLE_NZ:
        return False
    if weight.dtype in {torch.bfloat16, torch.float16}:
        return envs_ascend.VLLM_ASCEND_ENABLE_NZ == 2
    return True

# 改动后
def _should_trans_nz(weight: torch.Tensor) -> bool:
    if is_310p():
        return True
    config = get_ascend_config()
    nz_mode = config.weight_nz_mode
    if not nz_mode:
        return False
    if weight.dtype in {torch.bfloat16, torch.float16}:
        return nz_mode == 2
    return True
```

**说明**：NZ（NZ 格式）权重转换的配置从 `VLLM_ASCEND_ENABLE_NZ` 改为 `weight_nz_mode`。语义保持一致：
- `0`：禁用 NZ 转换
- `1`：仅量化场景启用 NZ（默认）
- `2`：BF16/FP16 也启用 NZ

**影响范围**：权重加载时的格式转换，影响模型权重的内存布局和计算性能。

---

### 1.7 vllm_version_is()

```python
# 改动前
@functools.cache
def vllm_version_is(target_vllm_version: str):
    if envs_ascend.VLLM_VERSION is not None:
        vllm_version = envs_ascend.VLLM_VERSION
    else:
        import vllm
        vllm_version = vllm.__version__
    ...

# 改动后
@functools.cache
def vllm_version_is(target_vllm_version: str):
    config_version = None
    with suppress(RuntimeError):
        config_version = get_ascend_config().vllm_version
    if config_version is not None:
        vllm_version = config_version
    else:
        import vllm
        vllm_version = vllm.__version__
    ...
```

**说明**：vLLM 版本判断的配置来源从环境变量改为 Config。关键改动是使用 `suppress(RuntimeError)` 保护，因为此函数可能在 AscendConfig 初始化之前被调用（模块导入时），此时 `get_ascend_config()` 会抛出 `RuntimeError: Ascend config is not initialized`。

错误提示也做了更新：
```python
# 改动前
"Set the environment variable VLLM_VERSION to control it by hand."

# 改动后
'Use --additional-config \'{"vllm_version": "x.y.z"}\' to override.'
```

**影响范围**：vLLM 版本兼容性判断，影响不同版本 vLLM 的适配逻辑。

---

## 2. enable_sp() 重构

这是最复杂的改动，解决了 **worker 子进程没有 vLLM config 上下文** 的问题。

### 改动前

```python
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    if _ENABLE_SP is None:
        if vllm_config is None:
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()  # worker 进程会抛 AssertionError
        _ENABLE_SP = (
            envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
            or bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0")))
        )
        if not _ENABLE_SP and enable_shared_expert_dp:
            _ENABLE_SP = True
            logger.info("shared_expert_dp requires enable_sp = True. has set enable_sp to True")
    return _ENABLE_SP
```

### 改动后

```python
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
        except AssertionError:
            vllm_config = None  # worker 进程安全回退

    additional_config = getattr(vllm_config, "additional_config", None) if vllm_config is not None else None
    refresh = additional_config.get("refresh", False) if additional_config else False

    if _ENABLE_SP is None or refresh:
        _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1 or bool(
            int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0"))
        )
        if not _ENABLE_SP and enable_shared_expert_dp:
            _ENABLE_SP = True
            logger.info("shared_expert_dp requires enable_sp = True. has set enable_sp to True")
    return _ENABLE_SP
```

### 关键改动点

#### 2.1 try/except AssertionError 保护

**问题**：在分布式推理中，主进程设置了 vLLM config 上下文，但 worker 子进程通过 `multiprocessing` fork 出来后，`get_current_vllm_config()` 会抛出 `AssertionError: Current vLLM config is not set`。

**解决**：捕获 AssertionError，将 `vllm_config` 设为 None，让后续逻辑走环境变量回退路径（读取 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 和旧的 `VLLM_ASCEND_ENABLE_FLASHCOMM`）。

#### 2.2 refresh 机制

**问题**：`_ENABLE_SP` 是全局变量，一旦设置就不会再更新。但在某些场景下（如测试或动态配置），需要强制刷新。

**解决**：从 `vllm_config.additional_config` 中读取 `refresh` 标志，如果为 True 则重新计算 `_ENABLE_SP`。

#### 2.3 FLASHCOMM1 保留环境变量读取

**说明**：`VLLM_ASCEND_ENABLE_FLASHCOMM1` 没有迁移到 Config，仍然通过 `envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1` 读取。同时保留了 `os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0")` 作为向后兼容（注意：`VLLM_ASCEND_ENABLE_FLASHCOMM` 是旧的环境变量名，已不在 envs.py 中定义，但仍通过 os.getenv 直接读取以兼容旧用户）。这是因为 FLASHCOMM1 在 worker 子进程中也需要读取，而 worker 子进程没有 vLLM config 上下文，迁移到 Config 会导致 worker 进程无法获取配置。

---

## 3. clear_enable_sp() 新增函数

```python
def clear_enable_sp():
    global _ENABLE_SP
    _ENABLE_SP = None
    enable_dsa_cp.cache_clear()
    enable_dsa_cp_with_layer_shard.cache_clear()
    enable_dsa_cp_with_o_proj_tp.cache_clear()
    _libc_getenv.cache_clear()
```

**说明**：此函数用于重置 `enable_sp` 相关的全局状态，包括：
- `_ENABLE_SP` 全局变量
- `enable_dsa_cp` 的 LRU 缓存
- `enable_dsa_cp_with_layer_shard` 的 LRU 缓存
- `enable_dsa_cp_with_o_proj_tp` 的 LRU 缓存
- `_libc_getenv` 的 LRU 缓存

**调用位置**：`ascend_config.py` 中的 `clear_ascend_config()` 函数。

**解决的问题**：单元测试中，多个测试用例共享进程，全局变量和 LRU 缓存不会自动重置，导致测试间状态污染。此函数确保每次 `clear_ascend_config()` 时也清除 `enable_sp` 的缓存。

---

## 4. vllm_version_is() 安全改造

```python
config_version = None
with suppress(RuntimeError):
    config_version = get_ascend_config().vllm_version
```

**说明**：使用 `contextlib.suppress(RuntimeError)` 替代 try/except，更简洁。这是因为 `vllm_version_is()` 被装饰了 `@functools.cache`，且可能在模块导入时（AscendConfig 尚未初始化）被调用。

**时序问题**：
1. Python 解释器启动
2. 导入 vllm_ascend 模块
3. 某些模块在 import 时调用 `vllm_version_is()`
4. 此时 AscendConfig 还没初始化，`get_ascend_config()` 抛 RuntimeError
5. suppress 捕获异常，`config_version` 保持 None
6. 回退到 `vllm.__version__` 读取

---

## 5. 删除 bootstrap_custom_op_env 相关代码

### 删除的代码

```python
_CUSTOM_OP_VENDOR_DIR = "custom_transformer"
_CUSTOM_OP_BASE_DIR = (
    os.path.dirname(__file__) if os.path.isabs(__file__) else os.path.abspath(os.path.dirname(__file__))
)

def _prepend_env_path(env_name: str, path: str) -> None:
    current_value = os.environ.get(env_name, "")
    path_entries = [entry for entry in current_value.split(":") if entry]
    if path not in path_entries:
        path_entries.insert(0, path)
        os.environ[env_name] = ":".join(path_entries)

def bootstrap_custom_op_env(*, include_vendor_lib: bool = False) -> None:
    vendor_path = os.path.join(_CUSTOM_OP_BASE_DIR, "_cann_ops_custom", "vendors", _CUSTOM_OP_VENDOR_DIR)
    if not os.path.exists(vendor_path):
        return
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_path)
    if include_vendor_lib:
        vendor_lib_path = os.path.join(vendor_path, "op_api", "lib")
        if os.path.exists(vendor_lib_path):
            _prepend_env_path("LD_LIBRARY_PATH", vendor_lib_path)
```

### enable_custom_op() 简化

```python
# 改动前
def enable_custom_op():
    ...
    try:
        if not torch.compiler.is_compiling():
            bootstrap_custom_op_env()
        import vllm_ascend.vllm_ascend_C
        ...
        _CUSTOM_OP_ENABLED = True
    except ImportError as e:
        if (not torch.compiler.is_compiling()) and "libcust_opapi.so" in str(e):
            try:
                bootstrap_custom_op_env(include_vendor_lib=True)
                import vllm_ascend.meta_registration
                import vllm_ascend.vllm_ascend_C
                _CUSTOM_OP_ENABLED = True
            except ImportError:
                _CUSTOM_OP_ENABLED = False
                logger.warning(...)
        else:
            _CUSTOM_OP_ENABLED = False
            logger.warning(...)

# 改动后
def enable_custom_op():
    ...
    try:
        import vllm_ascend.vllm_ascend_C
        ...
        _CUSTOM_OP_ENABLED = True
    except ImportError:
        _CUSTOM_OP_ENABLED = False
        logger.warning("Warning: Failed to register custom ops, all custom ops will be disabled")
```

**说明**：删除了手动修改 `LD_LIBRARY_PATH` 和 `ASCEND_CUSTOM_OPP_PATH` 的逻辑。现在依赖 CANN 的 rpath 机制自动查找库文件，不再需要手动设置环境变量。这简化了 custom op 注册流程，也避免了环境变量污染。

---

## 6. 310P MRotaryEmbedding 注册方式调整

```python
# 改动前
from vllm_ascend._310p.ops.rotary_embedding import AscendMRotaryEmbedding310, AscendRotaryEmbedding310

ops_dict = {
    ...
    "MRotaryEmbedding": AscendMRotaryEmbedding310,
}

# 改动后
from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310

ops_dict = {
    ...
    # MRotaryEmbedding 不再在 dict 中
}

REGISTERED_ASCEND_OPS.pop("MRotaryEmbedding", None)
```

**说明**：将 MRotaryEmbedding 从 310P 的注册字典中移出，改为先让通用注册完成，再用 `pop` 移除。这样可以避免 310P 的 MRotaryEmbedding 实现覆盖通用实现，确保在非 310P 场景下使用通用实现。

---

## 改动影响总结

| 改动类别 | 影响范围 | 风险等级 |
|---------|---------|---------|
| 环境变量迁移 | 所有使用这些配置的功能路径 | 中（需要确保 Config 已初始化） |
| enable_sp() 重构 | 分布式推理的 worker 进程 | 高（涉及多进程通信） |
| clear_enable_sp() | 单元测试 | 低（仅影响测试状态重置） |
| vllm_version_is() 安全改造 | 版本兼容性判断 | 低（有 suppress 保护） |
| 删除 bootstrap_custom_op_env | custom op 注册 | 低（依赖 rpath 替代） |
| MRotaryEmbedding 调整 | 310P 的 MRoPE 功能 | 低（逻辑不变，注册方式调整） |
