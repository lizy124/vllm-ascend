# vLLM-Ascend 日志系统设计与实现文档

## 1. 设计背景与目标

### 1.1 问题背景

vLLM-Ascend 作为 vLLM 的硬件插件，其日志消息与上游 vLLM 的日志混在一起输出到控制台，用户无法区分哪些日志来自 vLLM-Ascend、哪些来自上游 vLLM。此外，之前 vLLM-Ascend 没有持久化的日志文件，生产环境中出问题后难以回溯排查。

### 1.2 设计目标

1. **日志可识别** — 所有 vLLM-Ascend 模块的日志自动携带 `[vllm-ascend]` 前缀，一眼可区分
2. **默认文件日志** — 自动将日志写入文件，支持轮转和保留策略
3. **可配置** — 用户可通过 `additional_config` 自定义日志路径
4. **不引入新环境变量** — 遵循 vLLM-Ascend 的开发规范
5. **零侵入上游代码** — 通过 patch vLLM 的 Formatter 实现，不修改 vLLM 源码

---

## 2. 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         vLLM 日志系统                              │
│                                                                  │
│  vllm.logger.init_logger()                                       │
│       │                                                          │
│       ▼                                                          │
│  logging.getLogger("vllm")                                       │
│       │                                                          │
│       ├── StreamHandler ─── NewLineFormatter / ColoredFormatter  │
│       │                                                          │
│       │   ◄── vllm_ascend.logger._patch_vllm_formatter()         │
│       │       替换为 AscendFormatter / AscendColoredFormatter    │
│       │                                                          │
│       └── TimedRotatingFileHandler  ◄── _setup_file_logging()   │
│           使用 AscendFormatter                                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

启动流程:
  1. Python 导入 vllm_ascend.logger 模块
     → _patch_vllm_formatter() 替换 Formatter（模块级自动执行）

  2. platform.py: check_and_update_config()
     → init_ascend_config(vllm_config)
     → configure_ascend_file_logging()
       读取 AscendConfig.ascend_log_path → 创建文件日志 Handler
```

### 2.1 关键设计决策

| 决策点 | 方案 | 原因 |
|--------|------|------|
| Formatter patch 时机 | 模块导入时立即执行 | 必须在任何 vLLM 日志输出之前完成 |
| 文件日志 setup 时机 | 延迟到 `platform.py` 初始化后 | 需要读取 `additional_config`，此时 `AscendConfig` 才就绪 |
| 幂等保护 | `_file_logging_configured` 标志位 | 防止重复创建 Handler |
| 配置读取方式 | 先尝试 `get_ascend_config()`，失败 fallback 到默认值 | 兼容 AscendConfig 未初始化的场景（如部分测试） |

---

## 3. 详细代码实现

### 3.1 文件结构

```
vllm_ascend/
├── logger.py              # 日志扩展核心模块
├── ascend_config.py       # 新增 ascend_log_path 配置项
└── platform.py            # 启动流程中调用 configure_ascend_file_logging()

tests/ut/
└── test_logger.py         # 单元测试（11 个测试用例）
```

### 3.2 logger.py — 日志扩展核心

#### 3.2.1 常量定义

```python
_FORMAT = "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

_LOG_DIR = os.path.join(os.path.expanduser("~"), "ascend", "log", "vllm_ascend")
```

- `_LOG_DIR` — 默认日志目录：`~/ascend/log/vllm_ascend/`

#### 3.2.2 模块识别：`_is_ascend_module()`

```python
def _is_ascend_module(pathname: str) -> bool:
    if not pathname:
        return False
    return "vllm_ascend" in pathname.replace("\\", "/")
```

通过检查 `LogRecord.pathname`（日志调用者的文件路径）中是否包含 `vllm_ascend` 来判断是否为 ascend 模块的日志。兼容 Windows/Linux 路径分隔符。

#### 3.2.3 模块名推断：`_infer_module_name()`

```python
def _infer_module_name(pathname: str) -> str:
    if not pathname:
        return "core"
    parts = pathname.replace("\\", "/").split("/")
    try:
        idx = parts.index("vllm_ascend")
        if idx + 1 >= len(parts):
            return "core"
        item = parts[idx + 1]
        if idx + 2 >= len(parts):
            return item[:-3] if item.endswith(".py") else item
        return item
    except ValueError:
        return "core"
```

从路径中提取 `vllm_ascend` 后的第一个目录/文件名作为模块名：
- `/vllm_ascend/platform.py` → `"platform"`
- `/vllm_ascend/compilation/acl_graph.py` → `"compilation"`
- `/vllm/model.py` → `"core"`（非 ascend 模块，fallback）

#### 3.2.4 日志格式化：`_format_with_ascend_prefix()`

```python
def _format_with_ascend_prefix(self, record, super_format):
    if not _is_ascend_module(record.pathname):
        return super_format(record)  # 非 ascend 日志原样输出
    module = _infer_module_name(record.pathname)
    # 避免重复：文件名已暗示模块名时只加 [vllm-ascend]
    if record.filename == module + ".py":
        prefix = "[vllm-ascend]"
    else:
        prefix = f"[vllm-ascend] [{module}]"
    # 安全地修改 LogRecord，使用 try/finally 保证恢复
    orig_msg = record.msg
    orig_args = record.args
    try:
        record.msg = f"{prefix} - {record.getMessage()}"
        record.args = ()
        return super_format(record)
    finally:
        record.msg = orig_msg
        record.args = orig_args
```

关键细节：
- **非 ascend 日志直通**：直接调用父类 format，零开销
- **防重复前缀**：当 `record.filename` 就是 `{module}.py` 时，不再追加 `[{module}]`，避免出现 `[ascend_config.py:683] [vllm-ascend] [ascend_config]` 这种重复
- **安全 mutation**：临时修改 `record.msg` 和 `record.args`，finally 块中恢复原值，防止影响后续 handler

#### 3.2.5 Formatter 子类

```python
class AscendFormatter(NewLineFormatter):
    def format(self, record):
        return _format_with_ascend_prefix(self, record, super().format)

class AscendColoredFormatter(ColoredFormatter):
    def format(self, record):
        return _format_with_ascend_prefix(self, record, super().format)
```

继承 vLLM 的两个 Formatter，仅覆盖 `format()` 方法，格式化逻辑复用共享函数 `_format_with_ascend_prefix()`。

#### 3.2.6 Formatter 替换：`_patch_vllm_formatter()`

```python
def _patch_vllm_formatter() -> None:
    vllm_logger = logging.getLogger("vllm")

    for handler in vllm_logger.handlers:
        _patch_handler(handler)

    _original_add_handler = vllm_logger.addHandler

    def _patched_add_handler(handler):
        _patch_handler(handler)
        _original_add_handler(handler)

    vllm_logger.addHandler = _patched_add_handler
```

采用双重策略保证鲁棒性：
1. **即时 patch**：遍历已有 handler，替换 formatter
2. **Monkey-patch `addHandler`**：拦截后续添加的 handler，自动替换 formatter。这解决了导入顺序敏感的问题——即使某些 handler 在 `logger.py` 加载之后才被添加，也能被正确 patch

#### 3.2.7 文件日志：`_setup_file_logging()`

```python
_file_logging_configured = False

def _setup_file_logging(log_dir=None):
    global _file_logging_configured
    if _file_logging_configured:
        return  # 幂等保护
    target_dir = log_dir or _LOG_DIR
    os.makedirs(target_dir, exist_ok=True)
    file_handler = RotatingAscendFileHandler(target_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.addHandler(file_handler)
    _file_logging_configured = True
```

- **目录自动创建**：`os.makedirs(target_dir, exist_ok=True)`
- **按大小轮转**：`RotatingAscendFileHandler`，单文件超过 20MB 自动切分
- **文件命名**：`vllm_ascend_{timestamp}_{PID}.log`，轮转后追加 `_002.log` 等
- **INFO 级别**：`file_handler.setLevel(logging.INFO)`
- **复用 AscendFormatter**：文件日志也带 `[vllm-ascend]` 前缀

#### 3.2.8 公共入口：`configure_ascend_file_logging()`

```python
def configure_ascend_file_logging():
    log_dir = _LOG_DIR
    try:
        from vllm_ascend.ascend_config import get_ascend_config
        ascend_config = get_ascend_config()
        log_dir = ascend_config.ascend_log_path
    except Exception:
        pass  # AscendConfig 未初始化时使用默认路径
    _setup_file_logging(log_dir)
```

使用 lazy import 避免循环依赖，`try/except` 兜底保证在 AscendConfig 未就绪的场景下也能正常工作。

### 3.3 ascend_config.py — 配置项

在 `AscendConfig.__init__()` 中新增：

```python
self.ascend_log_path = additional_config.get(
    "ascend_log_path",
    os.path.join(os.path.expanduser("~"), "ascend", "log", "vllm_ascend"),
)
```

遵循现有 `additional_config` 的读取模式，提供默认值。

### 3.4 platform.py — 启动集成

在 `NPUPlatform.check_and_update_config()` 中，`init_ascend_config()` 之后调用：

```python
ascend_config = init_ascend_config(vllm_config)

from vllm_ascend.logger import configure_ascend_file_logging
configure_ascend_file_logging()
```

此时 `AscendConfig` 已初始化，`additional_config` 中的 `ascend_log_path` 已可用。

---

## 4. 使用方法

### 4.1 默认行为（无需任何配置）

直接启动 vLLM serve，日志自动输出：

```
# 控制台输出示例：
INFO 06-11 10:00:00 [vllm_ascend] [platform.py:384] - Platform initialized
INFO 06-11 10:00:01 [vllm] [worker.py:123] - Worker started

# 文件日志自动写入：
~/ascend/log/vllm_ascend/vllm_ascend.log
```

### 4.2 自定义日志路径

通过 `additional_config` 配置：

**命令行：**

```bash
vllm serve Qwen/Qwen3-8B \
    --additional-config '{"ascend_log_path": "/data/logs/my_vllm_ascend"}'
```

**Python API：**

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-8B",
    additional_config={"ascend_log_path": "/data/logs/my_vllm_ascend"},
)
```

配置后日志写入 `/data/logs/my_vllm_ascend/vllm_ascend.log`。

### 4.3 日志轮转

- **轮转**：按文件大小自动轮转，单文件超过 20MB 时自动切分
- **命名**：新文件命名为 `vllm_ascend_{timestamp}_{PID}.log`，轮转后追加 `_002.log`、`_003.log` 等
- **清理**：日志清理由用户自行管理，插件不自动删除旧日志

### 4.4 日志格式详解

```
INFO 06-11 10:00:00 [vllm_ascend] [platform.py:384] - Message text
│    │           │               │                   │
│    │           │               │                   └── 原始日志消息
│    │           │               └── 文件名:行号（vLLM 标准格式）
│    │           └── ascend 前缀 + 可选模块名
│    └── 时间戳
└── 日志级别
```

前缀规则：
- 根级别文件（如 `platform.py`）：`[vllm-ascend] - message`
- 子模块文件（如 `compilation/acl_graph.py`）：`[vllm-ascend] [compilation] - message`
- 非 ascend 文件：无前缀，保持 vLLM 原始格式

---

## 5. 测试

### 5.1 测试文件

`tests/ut/test_logger.py` — 包含 11 个测试用例：

| 测试用例 | 覆盖内容 |
|----------|----------|
| `test_is_ascend_module_with_ascend_path` | ascend 路径正确识别 |
| `test_is_ascend_module_with_vllm_path` | vLLM 路径不误识别，空字符串边界 |
| `test_infer_module_name_root_file` | 根级别 .py 文件提取模块名 |
| `test_infer_module_name_nested_file` | 嵌套子目录提取模块名 |
| `test_infer_module_name_edge_cases` | 空路径、非 ascend 路径 fallback |
| `test_ascend_formatter_adds_prefix_root_file` | 根文件前缀（不重复模块名） |
| `test_ascend_formatter_adds_prefix_nested_file` | 子模块前缀 |
| `test_ascend_formatter_pass_through_vllm_logs` | vLLM 日志直通（不加前缀） |
| `test_ascend_colored_formatter_adds_prefix` | ColoredFormatter 前缀 |
| `test_log_dir_constant` | 常数值验证 |
| `test_setup_file_logging_creates_handler` | Handler 创建、类型、级别、轮转、目录创建 |

### 5.2 运行测试

```bash
pytest tests/ut/test_logger.py -v
```

---

## 6. 设计原则总结

1. **零侵入** — 不修改 vLLM 上游任何代码
2. **自动生效** — 导入 vllm-ascend 即自动启用，无需用户配置
3. **向后兼容** — 不影响现有 vLLM 日志行为，非 ascend 日志完全直通
4. **鲁棒性** — Monkey-patch `addHandler` 保证后续 handler 也被正确处理；`try/finally` 保证 LogRecord 不被污染
5. **可配置** — 通过 `additional_config` 支持路径自定义，遵循项目现有配置模式
6. **幂等** — `_file_logging_configured` 防重复创建
7. **延迟初始化** — 文件日志延迟到 AscendConfig 就绪，避免时序问题