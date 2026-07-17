# vLLM-Ascend 日志系统设计文档

## 1. 背景与动机

### 1.1 问题背景

vLLM-Ascend 作为 vLLM 的 Ascend 后端扩展，之前的日志系统存在以下问题：

| 问题 | 影响 |
|-----|------|
| 日志混在一起 | vLLM-Ascend 日志与 vLLM 核心日志混在一起，难以区分来源 |
| 无法追踪模块 | 无法知道日志来自哪个 Ascend 模块（compilation、worker、distributed 等） |
| 缺少持久化 | 没有文件日志，生产环境问题难以事后调试 |
| 路径不可配置 | 日志路径固定，无法适应不同部署环境 |
| 多进程问题 | 子进程重复配置 logger 导致异常 |

### 1.2 设计目标

1. **日志隔离**：不修改 vLLM 全局 logging state，安全用于 upstream tests
2. **来源追踪**：每条日志带 `[vllm-ascend] [module]` 前缀，便于定位问题
3. **持久化存储**：轮转文件日志用于事后调试
4. **灵活配置**：支持自定义日志路径
5. **安全设计**：支持 multiprocessing，子进程不会重复配置

---

## 2. 架构设计

### 2.1 Logger 层级结构

```
┌─────────────────────────────────────────────────────────────┐
│                     vLLM Logger                              │
│  handlers: [ConsoleHandler, FileHandler(AscendFormatter)]   │
│  (vLLM 核心日志，保持不变)                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ propagate (FileHandler 同时挂载)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 vllm_ascend Logger                           │
│  handlers: [ConsoleHandler(AscendFormatter),                │
│             FileHandler(AscendFormatter)]                   │
│  propagate: False (Console 不传播到 vllm logger)             │
│  (Ascend 特有日志，独立 namespace)                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Handler 架构

| Handler | 挂载位置 | 功能 |
|---------|---------|------|
| ConsoleHandler | 仅 `vllm_ascend` logger | 输出带 `[vllm-ascend] [module]` 前缀的日志到终端 |
| FileHandler | `vllm` + `vllm_ascend` logger | 输出所有日志到轮转文件（包括 vLLM 核心日志） |

### 2.3 Formatter 设计

| Formatter | 功能 | 使用场景 |
|----------|------|---------|
| `AscendFormatter` | 添加 `[vllm-ascend] [module]` 前缀 | 非彩色输出 |
| `AscendColoredFormatter` | 添加前缀 + 颜色 | 彩色输出（终端） |

---

## 3. 核心组件详解

### 3.1 AscendFormatter / AscendColoredFormatter

**功能**：给日志添加 `[vllm-ascend] [module]` 前缀

**核心逻辑**：

```python
def _format_with_ascend_prefix(self, record, super_format):
    if not _is_ascend_module(record.pathname):
        return super_format(record)  # vLLM 核心日志，不添加前缀
    
    module = _infer_module_name(record.pathname)
    if record.filename == module + ".py":
        prefix = "[vllm-ascend]"  # 根目录文件
    else:
        prefix = f"[vllm-ascend] [{module}]"  # 子目录文件
    
    record.msg = f"{prefix} - {record.getMessage()}"
    return super_format(record)
```

**输出示例**：

```
[vllm-ascend] - INFO message from platform.py
[vllm-ascend] [compilation] - INFO message from compilation/acl_graph.py
[vllm-ascend] [distributed] - INFO message from distributed/kv_transfer.py
```

### 3.2 RotatingAscendFileHandler

**功能**：文件大小轮转 handler

**轮转机制**：

| 参数 | 值 | 说明 |
|-----|---|------|
| `max_bytes` | 20MB | 单文件最大大小 |
| 轮转触发 | `>= max_bytes` | 写入前检查文件大小 |

**文件命名规范**：

```
vllm_ascend_{timestamp}_{pid}.log          <- 第一个文件
vllm_ascend_{timestamp}_{pid}_002.log       <- 第二个文件
vllm_ascend_{timestamp}_{pid}_003.log       <- 第三个文件
```

**示例**：

```
~/ascend/log/vllm_ascend/vllm_ascend_20260615_120000_12345.log
~/ascend/log/vllm_ascend/vllm_ascend_20260615_120000_12345_002.log
```

### 3.3 模块名推断逻辑

**`_is_ascend_module()`**：判断日志来源是否为 vllm_ascend

```python
def _is_ascend_module(pathname: str) -> bool:
    if not pathname:
        return False
    return "vllm_ascend" in pathname.replace("\\", "/")
```

**`_infer_module_name()`**：从文件路径推断模块名

```python
def _infer_module_name(pathname: str) -> str:
    parts = pathname.replace("\\", "/").split("/")
    idx = parts.index("vllm_ascend")
    
    if idx + 1 >= len(parts):
        return "core"  # vllm_ascend 目录本身
    
    item = parts[idx + 1]
    if idx + 2 >= len(parts):
        return item[:-3] if item.endswith(".py") else item  # 根目录文件
    
    return item  # 子目录，返回目录名
```

**推断示例**：

| 文件路径 | 推断模块名 |
|---------|----------|
| `/vllm_ascend/platform.py` | `platform` |
| `/vllm_ascend/compilation/acl_graph.py` | `compilation` |
| `/vllm_ascend/distributed/kv_transfer.py` | `distributed` |
| `/vllm/model.py` | `core`（非 Ascend 模块） |

---

## 4. Console Logging 设计

### 4.1 独立 Logger Namespace

```python
ascend_logger = logging.getLogger("vllm_ascend")
ascend_logger.propagate = False  # 不传播到 vllm logger
```

**意义**：
- 不修改 vLLM 全局 logging state
- 安全用于 upstream tests
- 子进程不会重复配置

### 4.2 配置函数

```python
def configure_ascend_logging() -> None:
    """Configure vllm_ascend logger with Ascend formatters."""
    ascend_logger = logging.getLogger("vllm_ascend")
    if ascend_logger.handlers:
        return  # 已配置，跳过
    
    handler = logging.StreamHandler(stream)
    handler.setLevel(envs.VLLM_LOGGING_LEVEL)
    
    if _use_color():
        handler.setFormatter(AscendColoredFormatter(...))
    else:
        handler.setFormatter(AscendFormatter(...))
    
    ascend_logger.addHandler(handler)
    ascend_logger.setLevel(envs.VLLM_LOGGING_LEVEL)
    ascend_logger.propagate = False
```

### 4.3 颜色输出控制

```python
def _use_color() -> bool:
    """Determine if colored output should be used."""
    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":
        return False
    if envs.VLLM_LOGGING_COLOR == "1":
        return True
    # 默认：终端输出时使用颜色
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
```

---

## 5. File Logging 设计

### 5.1 文件轮转机制

```python
class RotatingAscendFileHandler(logging.FileHandler):
    def emit(self, record) -> None:
        # 写入前检查文件大小
        if os.path.getsize(self.baseFilename) >= self._max_bytes:
            self._rotate()  # 超过 20MB，轮转
        super().emit(record)
    
    def _rotate(self) -> None:
        self.stream.close()
        self._sequence += 1
        new_file = f"{self._base_name}_{self._sequence:03d}.log"
        self.baseFilename = new_file
        self.stream = self._open()
```

### 5.2 日志路径配置

**默认路径**：

```python
_LOG_DIR = os.path.join(os.path.expanduser("~"), "ascend", "log", "vllm_ascend")
```

**自定义路径**：

```python
def configure_ascend_file_logging() -> None:
    try:
        ascend_config = get_ascend_config()
        log_dir = ascend_config.ascend_log_path  # 用户配置的路径
    except Exception:
        log_dir = _LOG_DIR  # 使用默认路径
    
    _setup_file_logging(log_dir)
```

### 5.3 同时挂载到两个 Logger

```python
def _setup_file_logging(log_dir: str) -> None:
    file_handler = RotatingAscendFileHandler(log_dir)
    
    vllm_logger = logging.getLogger("vllm")
    ascend_logger = logging.getLogger("vllm_ascend")
    
    vllm_logger.addHandler(file_handler)    # 捕获 vLLM 核心日志
    ascend_logger.addHandler(file_handler)  # 捕获 Ascend 日志
```

**意义**：文件日志捕获所有日志（包括 vLLM 核心日志），便于完整调试。

---

## 6. 配置与使用

### 6.1 默认行为

```bash
# 默认：日志写入 ~/ascend/log/vllm_ascend/
vllm serve model
```

**日志文件**：

```
~/ascend/log/vllm_ascend/vllm_ascend_20260615_120000_12345.log
```

### 6.2 自定义日志路径

```bash
# 通过 additional-config 配置
vllm serve model --additional-config '{"ascend_log_path": "/var/log/vllm-ascend"}'
```

### 6.3 与 vLLM Logging 配置的关系

| vLLM 配置 | 影响 |
|----------|------|
| `VLLM_LOGGING_LEVEL` | Ascend logger 使用相同级别 |
| `VLLM_LOGGING_STREAM` | Console handler 使用相同 stream |
| `VLLM_LOGGING_COLOR` | Ascend formatter 使用相同颜色设置 |

---

## 7. 测试覆盖

### 7.1 测试用例

| 测试 | 功能点 |
|-----|-------|
| `test_is_ascend_module_*` | `_is_ascend_module()` 判断逻辑 |
| `test_infer_module_name_*` | `_infer_module_name()` 推断逻辑 |
| `test_ascend_formatter_*` | Formatter 前缀添加逻辑 |
| `test_log_dir_constant` | 默认路径和保留天数 |
| `test_setup_file_logging_*` | File logging 配置逻辑 |
| `test_rotating_handler_*` | 文件轮转机制 |
| `test_cleanup_old_logs` | 旧日志清理逻辑 |

### 7.2 测试覆盖的功能点

- 模块名推断（根目录文件、子目录文件、边缘情况）
- Formatter 前缀添加（Ascend 模块 vs vLLM 核心模块）
- 文件轮转（大小触发、命名规范）
- 日志清理（保留天数、文件过滤）

---

## 8. 总结

### 8.1 实现的功能

| 功能 | 说明 |
|-----||------|
| Console Logging | 独立 `vllm_ascend` logger，带前缀 |
| File Logging | 轮转文件日志，可配置路径 |
| 模块名推断 | 从文件路径推断模块名 |
| 颜色输出 | 继承 vLLM 颜色配置 |
| 文件轮转 | 20MB 大小轮转 |
| 日志清理 | 7 天保留（已移除，委托用户） |

### 8.2 设计亮点

1. **不修改 vLLM 全局 state**：安全用于 upstream tests
2. **独立 namespace**：`vllm_ascend` logger 与 vLLM logger 隔离
3. **来源追踪**：每条日志带 `[vllm-ascend] [module]` 前缀
4. **完整捕获**：File handler 同时挂载到两个 logger
5. **灵活配置**：支持自定义日志路径
6. **安全多进程**：子进程不会重复配置

### 8.3 后续改进方向

1. **日志级别独立配置**：支持 Ascend logger 独立设置级别
2. **日志格式可配置**：支持用户自定义日志格式
3. **日志压缩**：轮转后的文件自动压缩
4. **日志上报**：支持上报到远程日志系统