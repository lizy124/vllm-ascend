# vllm_ascend/envs.py 中 12 个环境变量是否适合迁移到 Config 的复核分析

## 1. 结论总览

分析对象：`D:\lzy\code\for_env\vllm-ascend\vllm_ascend\envs.py` 当前仍保留的 12 个环境变量。

结论：之前“基本都不适合迁移到 Config，除了 `VLLM_ASCEND_ENABLE_FLASHCOMM1`”这个大方向基本成立，但原因需要分两类说清楚：

1. **9 个构建/安装/编译期变量**：在 package build、CMake configure、custom kernel 编译或编译宏检测阶段使用，发生在 vLLM 运行和 `AscendConfig` 初始化之前，不适合迁移到 `additional_config`。
2. **2 个 import-time patch gate 变量**：`DYNAMIC_EPLB` 和 `VLLM_ASCEND_BALANCE_SCHEDULING` 在 `AscendConfig` 初始化前决定是否加载 monkey patch，也不适合简单迁移。
3. **1 个运行时优化变量**：`VLLM_ASCEND_ENABLE_FLASHCOMM1` 不是 early patch gate，主要控制 FlashComm1 / sequence parallel 路径，适合迁移到 Config。

| 变量 | 当前用途分类 | 是否适合迁移到 Config | 结论 |
|---|---|---:|---|
| `MAX_JOBS` | 构建并发度 | 否 | setup.py 构建扩展时使用。 |
| `CMAKE_BUILD_TYPE` | CMake 构建类型 | 否 | CMake configure 阶段使用。 |
| `COMPILE_CUSTOM_KERNELS` | 是否编译 custom kernels | 否 | setup.py 决定是否构建扩展/ACNN，自安装前生效。 |
| `CXX_COMPILER` | C++ 编译器选择 | 否 | CMake configure 阶段使用。 |
| `C_COMPILER` | C 编译器选择 | 否 | CMake configure 阶段使用。 |
| `SOC_VERSION` | 目标芯片/构建产物选择 | 否 | setup.py、CMake、build info 生成阶段使用。 |
| `VERBOSE` | CMake verbose makefile | 否 | CMake configure 阶段使用。 |
| `ASCEND_HOME_PATH` | CANN toolkit 路径 | 否 | setup.py / CMake 查找 CANN headers/libs 时使用。 |
| `VLLM_ASCEND_ENABLE_BATCH_MEMCPY` | CANN API 编译宏开关 | 否 | CMake 编译期检测/宏定义使用。 |
| `DYNAMIC_EPLB` | early executor patch gate | 否，不能简单迁移 | Config 前决定是否 patch `MultiprocExecutor`。 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | early scheduler patch gate | 否 | Config 前决定是否 patch `Scheduler` / `EngineCoreProc`。 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | FlashComm1 / SP 运行时优化 | 是 | 不是 import-time patch gate，适合迁移但要处理缓存和兼容。 |

---

## 2. 当前 `envs.py` 中的 12 个变量

当前仍在 `env_variables` 中实际定义的变量如下：

```python
# vllm_ascend/envs.py:30
env_variables: dict[str, Callable[[], Any]] = {
    "MAX_JOBS": ...,
    "CMAKE_BUILD_TYPE": ...,
    "COMPILE_CUSTOM_KERNELS": ...,
    "CXX_COMPILER": ...,
    "C_COMPILER": ...,
    "SOC_VERSION": ...,
    "VERBOSE": ...,
    "ASCEND_HOME_PATH": ...,
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": ...,
    "DYNAMIC_EPLB": ...,
    "VLLM_ASCEND_BALANCE_SCHEDULING": ...,
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": ...,
}
```

已经迁移或标记移除的环境变量，例如 `HCCL_SO_PATH`、`VLLM_VERSION`、`VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`、`VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`、`MSMONITOR_USE_DAEMON`、`VLLM_ASCEND_ENABLE_MLAPO`、`VLLM_ASCEND_ENABLE_NZ`、`VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL`、`VLLM_ASCEND_ENABLE_FUSED_MC2`、`VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK`，当前已经不属于这 12 个实际定义变量。

---

## 3. 判断标准

一个变量是否适合迁移到 `AscendConfig.additional_config`，关键看它的生效阶段。

### 3.1 适合迁移的典型特征

适合迁移到 Config 的变量通常满足：

1. 运行时读取。
2. 读取发生在 `init_ascend_config(vllm_config)` 之后。
3. 控制模型、worker、算子、attention、MoE、通信策略等运行时行为。
4. 不参与 package build、CMake configure、编译宏定义。
5. 不参与 Config 初始化前的 monkey patch gate。

例如 `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` 已迁移为：

```bash
--additional-config '{"enable_matmul_allreduce": true}'
```

### 3.2 不适合迁移的典型特征

不适合迁移到 Config 的变量通常属于以下之一：

1. **构建/安装/编译期变量**：在 `pip install`、`setup.py`、CMake configure/build 过程中使用，此时还没有 vLLM runtime，也没有 `VllmConfig` / `AscendConfig`。
2. **import-time patch gate**：在 `vllm_ascend.patch.platform.__init__` 顶层执行时读取，用于决定是否 import patch 模块并替换 vLLM 核心类。这个阶段早于 `AscendConfig` 初始化。
3. **系统环境路径/编译器选择变量**：本质上是构建系统输入，不是推理运行参数。

---

## 4. 9 个构建/安装/编译期变量逐项分析

### 4.1 `MAX_JOBS`

定义：

```python
# vllm_ascend/envs.py:31
"MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
```

用途：控制 CMake build 的并发数。

读取点：

```python
# setup.py:238
# Determine number of compilation jobs

# setup.py:241
num_jobs = envs.MAX_JOBS

# setup.py:242
if num_jobs is not None:
    num_jobs = int(num_jobs)
    logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
```

最终用于：

```python
# setup.py:392
build_args = [
    "--build",
    ".",
    f"-j={num_jobs}",
    ...
]
```

分析：

`MAX_JOBS` 只影响构建扩展时的并行编译数量，发生在安装阶段。`additional_config` 是 vLLM 启动后构造 `VllmConfig` 时才存在，无法影响已经完成的编译。

结论：不适合迁移到 Config。

---

### 4.2 `CMAKE_BUILD_TYPE`

定义：

```python
# vllm_ascend/envs.py:35
"CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
```

用途：选择 CMake 编译类型，例如 `Release`、`Debug`、`RelWithDebugInfo`。

读取点：

```python
# setup.py:267
if envs.CMAKE_BUILD_TYPE is None or envs.CMAKE_BUILD_TYPE not in [
    "Debug",
    "Release",
    "RelWithDebugInfo",
]:
    envs.CMAKE_BUILD_TYPE = "Release"

# setup.py:273
cmake_args += [f"-DCMAKE_BUILD_TYPE={envs.CMAKE_BUILD_TYPE}"]
```

CMake 中也会读取：

```cmake
# CMakeLists.txt:34
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()
```

分析：

这是典型 CMake configure/build 阶段变量。它影响生成的二进制扩展和编译优化级别，运行时 Config 无法改变已经编译好的产物。

结论：不适合迁移到 Config。

---

### 4.3 `COMPILE_CUSTOM_KERNELS`

定义：

```python
# vllm_ascend/envs.py:38
"COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
```

用途：决定是否编译 custom kernels。注释明确说明主要用于无 NPU 环境跑 UT，不应在普通场景关闭。

setup.py 中用于决定是否声明扩展模块：

```python
# setup.py:459
ext_modules = []

# setup.py:460
if envs.COMPILE_CUSTOM_KERNELS:
    ext_modules = [CMakeExtension(name="vllm_ascend.vllm_ascend_C")]
```

用于决定是否执行 build：

```python
# setup.py:365
def build_extensions(self) -> None:
    if not envs.COMPILE_CUSTOM_KERNELS:
        return
```

用于决定是否先构建 ACLNN：

```python
# setup.py:436
def run(self):
    if envs.COMPILE_CUSTOM_KERNELS:
        self.run_command("build_aclnn")
```

运行时也有一个 warning：

```python
# vllm_ascend/worker/worker.py:93
if not envs_ascend.COMPILE_CUSTOM_KERNELS:
    logger.warning(
        "COMPILE_CUSTOM_KERNELS is set to False. "
        "In most scenarios, without custom kernels, vllm-ascend will not function correctly."
    )
```

分析：

虽然 worker 中也读取它，但运行时读取只是提示用户当前安装产物可能缺 custom kernels，不是用它决定是否编译。真正的功能点在 setup.py 构建阶段：是否生成 `vllm_ascend_C` 扩展、是否构建 ACLNN/custom kernels。

如果把它迁到 Config，用户启动 vLLM 时才设置已经太晚，因为 custom kernels 是否存在在安装时已经确定。

结论：不适合迁移到 Config。

---

### 4.4 `CXX_COMPILER`

定义：

```python
# vllm_ascend/envs.py:44
"CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
```

用途：指定 C++ 编译器。

读取点：

```python
# setup.py:276
if envs.CXX_COMPILER is not None:
    cmake_args += [f"-DCMAKE_CXX_COMPILER={envs.CXX_COMPILER}"]
```

分析：

这是 CMake configure 阶段变量。编译器选择必须发生在生成构建系统之前，运行时 Config 无法改变编译器。

结论：不适合迁移到 Config。

---

### 4.5 `C_COMPILER`

定义：

```python
# vllm_ascend/envs.py:47
"C_COMPILER": lambda: os.getenv("C_COMPILER", None),
```

用途：指定 C 编译器。

读取点：

```python
# setup.py:278
if envs.C_COMPILER is not None:
    cmake_args += [f"-DCMAKE_C_COMPILER={envs.C_COMPILER}"]
```

分析：

和 `CXX_COMPILER` 一样，这是构建系统输入，必须在 CMake configure 阶段生效。

结论：不适合迁移到 Config。

---

### 4.6 `SOC_VERSION`

定义：

```python
# vllm_ascend/envs.py:50
"SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
```

用途：指定目标 Ascend 芯片型号，用于 package building。

自动探测逻辑：

```python
# setup.py:134
if not envs.SOC_VERSION:
    soc_version = get_chip_type()
    if not soc_version:
        raise RuntimeError(...)
    envs.SOC_VERSION = soc_version
```

生成 build info：

```python
# setup.py:150
def gen_build_info():
    soc_version = envs.SOC_VERSION
    ...

# setup.py:186
package_dir = os.path.join(ROOT_DIR, "vllm_ascend", "_build_info.py")

# setup.py:188
f.write("# Auto-generated file\n")

# setup.py:189
f.write(f"__device_type__ = '{device_type}'\n")
```

构建 ACLNN：

```python
# setup.py:224
subprocess.check_call(["bash", "csrc/build_aclnn.sh", ROOT_DIR, envs.SOC_VERSION])
```

传给 CMake：

```python
# setup.py:321
CANN_SOC_VERSION = soc_version_map.get(envs.SOC_VERSION, envs.SOC_VERSION)

# setup.py:322
cmake_args += [f"-DSOC_VERSION={CANN_SOC_VERSION}"]
```

CMake 中使用：

```cmake
# CMakeLists.txt:31
set(SOC_VERSION ${SOC_VERSION})

# CMakeLists.txt:69
if(SOC_VERSION MATCHES "ascend950")
    ...
endif()

# CMakeLists.txt:75
if(SOC_VERSION MATCHES "ascend310p.*")
    message(STATUS "310P hardware detected: skip vllm_ascend_kernels compile")
else()
    ascendc_library(vllm_ascend_kernels SHARED ...)
endif()

# CMakeLists.txt:162
if(SOC_VERSION MATCHES "ascend310p.*")
    target_compile_definitions(vllm_ascend_C PRIVATE -DASCEND_PLATFORM_310P)
endif()
```

分析：

`SOC_VERSION` 不只是运行时硬件信息，它决定：

1. `_build_info.py` 中的 `__device_type__`。
2. ACLNN 构建参数。
3. CMake 编译哪些 custom ops。
4. 是否跳过部分 kernel。
5. 是否定义 `ASCEND_PLATFORM_310P` 编译宏。

这些都发生在包构建/安装阶段。运行时再设置 Config 已经无法改变编译产物。

结论：不适合迁移到 Config。

---

### 4.7 `VERBOSE`

定义：

```python
# vllm_ascend/envs.py:54
"VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
```

用途：开启编译过程 verbose 日志。

读取点：

```python
# setup.py:280
if envs.VERBOSE:
    cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]
```

分析：

这个变量控制 CMake 构建日志详细程度，不是 vLLM 推理日志。只能在 configure/build 阶段生效。

结论：不适合迁移到 Config。

---

### 4.8 `ASCEND_HOME_PATH`

定义：

```python
# vllm_ascend/envs.py:56
"ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
```

用途：指定 CANN toolkit 安装路径。

setup.py 中用于传入 CMake：

```python
# setup.py:283
check_or_set_default_env(
    cmake_args,
    "ASCEND_HOME_PATH",
    envs.ASCEND_HOME_PATH,
    "/usr/local/Ascend/ascend-toolkit/latest",
)
```

`check_or_set_default_env()` 会把值写回环境变量并加入 CMake 参数：

```python
# setup.py:60
if env_name == "ASCEND_HOME_PATH":
    os.environ["ASCEND_HOME_PATH"] = env_variable

# setup.py:62
cmake_args += [f"-D{env_name}={env_variable}"]
```

CMake 中用于查找 CANN CMake 文件、headers、libs：

```cmake
# CMakeLists.txt:42
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

# CMakeLists.txt:43
if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    ...
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    ...
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    ...
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist...")
endif()

# CMakeLists.txt:101
${ASCEND_HOME_PATH}/include

# CMakeLists.txt:171
${ASCEND_HOME_PATH}/lib64
```

分析：

这是 CANN toolkit 的构建路径。没有它，CMake 无法找到 AscendC/CANN headers 和 libraries。运行时 Config 无法替代 CMake configure 阶段路径。

结论：不适合迁移到 Config。

---

### 4.9 `VLLM_ASCEND_ENABLE_BATCH_MEMCPY`

定义：

```python
# vllm_ascend/envs.py:130
# Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
# "1": force enable, "0": force disable, None: auto-detect from CANN headers.
"VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
```

setup.py 中把它传给 CMake：

```python
# setup.py:342
# Pass VLLM_ASCEND_ENABLE_BATCH_MEMCPY to CMake if explicitly set.
# When unset (None), CMake will auto-detect from CANN headers.
if envs.VLLM_ASCEND_ENABLE_BATCH_MEMCPY is not None:
    cmake_args += [f"-DVLLM_ASCEND_ENABLE_BATCH_MEMCPY={envs.VLLM_ASCEND_ENABLE_BATCH_MEMCPY}"]
```

CMake 中使用它决定是否定义编译宏：

```cmake
# CMakeLists.txt:114
# Detect aclrtMemcpyBatchAsync availability (CANN 8.5+)
# Can be overridden via VLLM_ASCEND_ENABLE_BATCH_MEMCPY env var

# CMakeLists.txt:125
if(DEFINED VLLM_ASCEND_ENABLE_BATCH_MEMCPY)
  if("${VLLM_ASCEND_ENABLE_BATCH_MEMCPY}" STREQUAL "1")
    message(STATUS "aclrtMemcpyBatchAsync: force enabled via VLLM_ASCEND_ENABLE_BATCH_MEMCPY=1")
    target_compile_definitions(vllm_ascend_C PRIVATE CANN_MEMCPY_BATCH_ASYNC)
  else()
    message(STATUS "aclrtMemcpyBatchAsync: force disabled via VLLM_ASCEND_ENABLE_BATCH_MEMCPY=0")
  endif()
else()
  check_cxx_source_compiles(... HAVE_ACLRT_MEMCPY_BATCH_ASYNC)
  if(HAVE_ACLRT_MEMCPY_BATCH_ASYNC)
    target_compile_definitions(vllm_ascend_C PRIVATE CANN_MEMCPY_BATCH_ASYNC)
  endif()
endif()
```

分析：

这个变量控制的是 C++ 扩展编译时是否启用 `CANN_MEMCPY_BATCH_ASYNC` 宏。宏是否定义决定编译进 `.so` 的代码路径。运行时 `additional_config` 无法重新定义编译宏，也无法改变已经编译好的二进制。

它虽然名字像运行时功能开关，但本质是 **compile path selection**。

结论：不适合迁移到 Config。

---

## 5. 2 个 import-time patch gate 变量复核

### 5.1 `DYNAMIC_EPLB`

定义：

```python
# vllm_ascend/envs.py:107
"DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
```

import-time 使用点：

```python
# vllm_ascend/patch/platform/__init__.py:36
if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    import vllm_ascend.patch.platform.patch_multiproc_executor  # noqa
```

被加载的 patch 会替换 vLLM 的 `MultiprocExecutor`：

```python
# vllm_ascend/patch/platform/patch_multiproc_executor.py:211
vllm.v1.executor.multiproc_executor.MultiprocExecutor = AscendMultiprocExecutor
```

关键原因是 Dynamic EPLB 会在 worker 中再创建 EPLB 子进程，而 vLLM 默认 daemon worker 不允许继续创建子进程。patch 会把 worker 进程设置为 `daemon=False`：

```python
# vllm_ascend/patch/platform/patch_multiproc_executor.py:195
proc = context.Process(..., daemon=False)
```

`DYNAMIC_EPLB` 不是单纯运行时开关，而是 Config 初始化前决定是否修改 executor 进程模型。

结论：不能简单迁移到 Config。

补充：Dynamic EPLB 的业务参数已经在 `additional_config.eplb_config` 中，例如 `dynamic_eplb`、`expert_map_path`、`num_redundant_experts` 等。但 `DYNAMIC_EPLB` 作为 early executor patch gate 当前仍需保留。

---

### 5.2 `VLLM_ASCEND_BALANCE_SCHEDULING`

定义：

```python
# vllm_ascend/envs.py:120
"VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
```

import-time 使用点：

```python
# vllm_ascend/patch/platform/__init__.py:39
if envs.VLLM_ASCEND_BALANCE_SCHEDULING:
    import vllm_ascend.patch.platform.patch_balance_schedule  # noqa
```

被加载的 patch 会替换 vLLM 的 scheduler / engine core 入口：

```python
# vllm_ascend/patch/platform/patch_balance_schedule.py:706
EngineCoreProc.run_engine_core = run_engine_core

# vllm_ascend/patch/platform/patch_balance_schedule.py:707
vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
```

这同样发生在 `AscendConfig` 初始化之前。

后续 `platform.py` 中对该变量的读取只是合法性校验：

```python
# vllm_ascend/platform.py:473
if envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING:
    ...
```

真正启用功能的是早期 import patch，不是这段校验。

结论：不适合迁移到 Config，除非先重构为非 import-time monkey patch 机制。

---

## 6. 唯一适合迁移的变量：`VLLM_ASCEND_ENABLE_FLASHCOMM1`

定义：

```python
# vllm_ascend/envs.py:74
"VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
```

核心读取函数：

```python
# vllm_ascend/utils.py:818
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    ...
    if _ENABLE_SP is None or refresh:
        _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1 or bool(
            int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0"))
        )
        ...
    return _ENABLE_SP
```

它影响 forward context：

```python
# vllm_ascend/ascend_forward_context.py:118
flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None

# vllm_ascend/ascend_forward_context.py:125
flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000
```

它影响线性层 op 选择：

```python
# vllm_ascend/ops/linear_op.py:637
if enable_sp():
    ...
    return SequenceColumnParallelOp(layer)
```

```python
# vllm_ascend/ops/linear_op.py:676
if enable_sp():
    ...
    return SequenceRowParallelOp(layer)
```

分析：

`VLLM_ASCEND_ENABLE_FLASHCOMM1` 不在 `patch/platform/__init__.py` 中控制 monkey patch，也不参与 setup.py / CMake。它主要控制运行时 FlashComm1 / sequence parallel 路径，因此适合迁移到 Config。

迁移建议：

```bash
--additional-config '{"enable_flashcomm1": true}'
```

迁移时需要注意：

1. `enable_sp()` 有 `_ENABLE_SP` 全局缓存，需要保留 refresh / clear 行为。
2. 当前还兼容旧变量 `VLLM_ASCEND_ENABLE_FLASHCOMM`，迁移时要明确兼容优先级。
3. `get_flashcomm2_config_and_validate()` 中对 FlashComm1 的 warning 也要同步改成读 Config。

推荐优先级：

```text
additional_config.enable_flashcomm1 > VLLM_ASCEND_ENABLE_FLASHCOMM1 > VLLM_ASCEND_ENABLE_FLASHCOMM > False
```

---

## 7. 为什么 9 个构建变量不能迁移到 Config

这 9 个变量共同特点是：

```text
它们在 pip install / setup.py / CMake configure / CMake build 阶段生效。
```

而 `AscendConfig` 的生命周期是：

```text
用户启动 vLLM
  -> vLLM 构造 VllmConfig
  -> NPUPlatform.check_and_update_config(vllm_config)
  -> init_ascend_config(vllm_config)
  -> AscendConfig 可用
```

这个阶段远晚于 package build。

如果把构建变量迁移到 Config，会出现根本性时序错误：

```text
用户在运行时设置 Config
  -> 但 custom kernels / CMake 编译类型 / 编译宏 / CANN headers 检测 / _build_info.py 已经在安装时确定
  -> Config 无法改变已经生成的 .so 和 Python build info
```

因此这 9 个变量应继续作为环境变量、构建参数或安装文档中的 build env，而不是 vLLM runtime Config。

---

## 8. 对原结论的修正与确认

原说法：

```text
envs.py 里面有 12 个环境变量，基本都不适合迁移到 Config，除了 VLLM_ASCEND_ENABLE_FLASHCOMM1。DYNAMIC_EPLB 和 VLLM_ASCEND_BALANCE_SCHEDULING 是 import 阶段变量，另外 9 个是构建相关变量。
```

复核后结论：

```text
这个分类基本正确。
```

更精确版本：

```text
当前 envs.py 中 12 个实际定义变量里：
- 9 个是构建/安装/编译期变量，不适合迁移到 Config。
- 2 个是 Config 初始化前的 import-time patch gate，不适合简单迁移到 Config。
- 1 个 VLLM_ASCEND_ENABLE_FLASHCOMM1 是运行时优化开关，适合迁移到 Config，但需要处理 enable_sp 缓存、旧 env 兼容和 FlashComm2 warning。
```

---

## 9. 建议写入迁移文档的最终表述

建议在环境变量迁移说明里把这 12 个变量分为三类：

### 9.1 Build-time envs：保留为环境变量

```text
MAX_JOBS
CMAKE_BUILD_TYPE
COMPILE_CUSTOM_KERNELS
CXX_COMPILER
C_COMPILER
SOC_VERSION
VERBOSE
ASCEND_HOME_PATH
VLLM_ASCEND_ENABLE_BATCH_MEMCPY
```

说明：

```text
These variables are consumed by setup.py/CMake before vLLM runtime configuration exists. They control package build, compiler selection, CANN path discovery, target SOC selection, or compile-time feature macros, so they should remain environment variables/build options rather than additional_config fields.
```

### 9.2 Bootstrap patch gate envs：暂不迁移

```text
DYNAMIC_EPLB
VLLM_ASCEND_BALANCE_SCHEDULING
```

说明：

```text
These variables gate import-time monkey patches before AscendConfig is initialized. They cannot be migrated to additional_config without redesigning the patch application timing.
```

### 9.3 Runtime config candidate：可迁移

```text
VLLM_ASCEND_ENABLE_FLASHCOMM1 -> additional_config.enable_flashcomm1
```

说明：

```text
This variable controls FlashComm1/sequence-parallel runtime path selection and is not an import-time patch gate. It can be migrated to additional_config with compatibility handling for existing env vars and enable_sp cache semantics.
```

---

## 10. 最终结论

严格按代码验证，`envs.py` 当前 12 个实际定义变量中：

```text
适合迁移到 Config：
- VLLM_ASCEND_ENABLE_FLASHCOMM1

不适合迁移到 Config，构建/编译期变量：
- MAX_JOBS
- CMAKE_BUILD_TYPE
- COMPILE_CUSTOM_KERNELS
- CXX_COMPILER
- C_COMPILER
- SOC_VERSION
- VERBOSE
- ASCEND_HOME_PATH
- VLLM_ASCEND_ENABLE_BATCH_MEMCPY

不适合简单迁移到 Config，early patch gate：
- DYNAMIC_EPLB
- VLLM_ASCEND_BALANCE_SCHEDULING
```

因此，可以确认：

```text
“12 个里只有 VLLM_ASCEND_ENABLE_FLASHCOMM1 适合迁移，其余 11 个不适合普通迁移到 Config”这个结论成立。
```
