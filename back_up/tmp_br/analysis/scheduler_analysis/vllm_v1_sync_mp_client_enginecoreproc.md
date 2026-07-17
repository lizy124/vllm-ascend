# vLLM v1 SyncMPClient 与 EngineCoreProc 进程关系说明

## 结论

在 vLLM v1 的 synchronous + multiprocess 模式下，可以这样理解：

```text
process0：前端 / 客户端进程
  核心对象：SyncMPClient

process1：后台 EngineCore 进程
  核心对象：EngineCoreProc
```

这两个对象通过 ZMQ / Queue 通信，共同完成请求提交、调度、模型执行和结果返回。

更准确的表述是：

```text
在 vLLM v1 的同步多进程模式下，前端进程维护 SyncMPClient，后台 EngineCore 进程运行 EngineCoreProc。
SyncMPClient 负责把请求发送给 EngineCoreProc 并接收输出；EngineCoreProc 内部运行 EngineCore busy loop，负责调度、执行 model executor、更新请求状态并返回结果。
```

但这个结论不能泛化成所有 vLLM 模式都固定只有两个进程。vLLM 还支持 in-process、async multiprocess、data parallel 多 EngineCore 等模式。

## 相关代码位置

### EngineCoreClient 抽象类

位置：

```text
vllm/v1/engine/core_client.py
```

`EngineCoreClient` 的注释说明了不同 client 类型：

```python
class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """
```

这说明：

```text
SyncMPClient = 同步接口 + 多进程 EngineCore
AsyncMPClient = 异步接口 + 多进程 EngineCore
InprocClient = 当前进程内 EngineCore
```

### make_client() 如何选择 client

位置：

```text
vllm/v1/engine/core_client.py
```

核心逻辑：

```python
if multiprocess_mode and asyncio_mode:
    return EngineCoreClient.make_async_mp_client(
        vllm_config, executor_class, log_stats
    )

if multiprocess_mode and not asyncio_mode:
    return SyncMPClient(vllm_config, executor_class, log_stats)

return InprocClient(vllm_config, executor_class, log_stats)
```

因此，在同步多进程模式下：

```text
multiprocess_mode=True
asyncio_mode=False
```

会创建：

```text
SyncMPClient
```

## SyncMPClient 是什么

位置：

```text
vllm/v1/engine/core_client.py
```

定义：

```python
class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""
```

它的职责是：

```text
1. 在前端进程里作为 EngineCore 的客户端代理
2. 启动或连接后台 EngineCore 进程
3. 通过 ZMQ input socket 把请求发送给 EngineCoreProc
4. 通过 ZMQ output socket 接收 EngineCoreProc 返回的 EngineCoreOutputs
5. 提供同步方法，例如 add_request()、get_output()、abort_requests()、reset_prefix_cache() 等
```

它不是实际执行模型的对象，也不直接运行 Scheduler。它更像是：

```text
前端进程中的 EngineCore 远程代理 / RPC client
```

## MPClient 初始化时做了什么

`SyncMPClient` 继承自 `MPClient`。

在 `MPClient.__init__()` 中，会完成几类事情：

### 1. 创建 ZMQ 上下文和 socket

```python
sync_ctx = zmq.Context(io_threads=2)
self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx
```

同步模式下使用普通 `zmq.Context`。

随后创建：

```text
input_socket：前端发请求到 EngineCore
output_socket：前端接收 EngineCore 输出
```

### 2. 如果没有外部传入 EngineCore 地址，则由 client 启动 EngineCore 进程

核心逻辑：

```python
with launch_core_engines(
    vllm_config, executor_class, log_stats, addresses
) as (engine_manager, coordinator, addresses, tensor_queue):
    self.resources.coordinator = coordinator
    self.resources.engine_manager = engine_manager
```

也就是说，默认情况下，`SyncMPClient` 不只是连接 EngineCore，还会负责启动后台 EngineCore 进程。

### 3. 等待 EngineCore ready

MPClient 会等待每个 EngineCore 发送 ready message：

```python
while identities:
    ...
    identity, _ = sync_input_socket.recv_multipart()
    identities.remove(identity)
```

只有 EngineCore 初始化完成并发出 ready 后，client 初始化才算完成。

### 4. 启动 EngineCore 进程监控线程

```python
self.start_engine_core_monitor()
```

如果后台 EngineCore 进程异常退出，client 会标记 engine dead，后续请求会抛出 `EngineDeadError`。

## EngineCoreProc 是什么

位置：

```text
vllm/v1/engine/core.py
```

定义：

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
```

这句话很关键：

```text
EngineCoreProc 是运行在后台进程中的 EngineCore 包装器。
```

它继承自 `EngineCore`，所以真正的调度、执行、状态更新能力来自 `EngineCore`；而 `EngineCoreProc` 额外负责：

```text
1. ZMQ 通信
2. 后台进程启动后的 handshake
3. input/output socket 线程
4. EngineCore busy loop
5. shutdown / engine dead 通知
```

## EngineCoreProc 是怎么启动的

启动位置：

```text
vllm/v1/engine/utils.py
```

核心代码：

```python
context.Process(
    target=EngineCoreProc.run_engine_core,
    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
    kwargs=common_kwargs | {"dp_rank": global_index, "local_dp_rank": local_index},
)
```

这说明后台进程的入口函数是：

```text
EngineCoreProc.run_engine_core
```

如果不是 data parallel，进程名通常是：

```text
EngineCore
```

如果是 data parallel，可能是：

```text
EngineCore_DP0
EngineCore_DP1
...
```

## EngineCoreProc 初始化时做了什么

在 `EngineCoreProc.__init__()` 中，主要做：

### 1. 创建本地 input/output queue

```python
self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
```

这些 queue 是 EngineCoreProc 内部线程之间使用的队列。

### 2. 执行 handshake

```python
with self._perform_handshakes(...) as addresses:
```

Handshake 用于建立 EngineCoreProc 和前端 client 之间的 ZMQ 地址、identity、ready 通知等。

### 3. 初始化 EngineCore 本体

```python
super().__init__(
    vllm_config,
    executor_class,
    log_stats,
    executor_fail_callback,
    internal_dp_balancing,
)
```

这里会初始化真正的 EngineCore，包括：

```text
Scheduler
ModelExecutor
模型执行相关资源
```

### 4. 启动 input socket 线程

```python
input_thread = threading.Thread(
    target=self.process_input_sockets,
    ...
)
input_thread.start()
```

这个线程负责从 ZMQ input socket 收前端请求，再放入 `self.input_queue`。

### 5. 启动 output socket 线程

```python
self.output_thread = threading.Thread(
    target=self.process_output_sockets,
    ...
)
self.output_thread.start()
```

这个线程负责从 `self.output_queue` 取 EngineCoreOutputs，再通过 ZMQ output socket 发回前端 client。

## EngineCoreProc 的主循环

`EngineCoreProc` 的核心循环是：

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()

    raise SystemExit
```

可以拆成两步：

```text
1. _process_input_queue()
   处理前端发来的请求，例如 add_request、abort、utility call 等。

2. _process_engine_step()
   如果有未完成请求，则推进一次 EngineCore step，并把结果放入 output_queue。
```

## EngineCore step 做什么

`EngineCore.step()` 的核心逻辑是：

```python
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

也就是：

```text
1. Scheduler 决定这一轮调度哪些请求
2. ModelExecutor 根据 SchedulerOutput 执行模型 forward
3. Scheduler 根据模型输出更新 request 状态
4. 返回 EngineCoreOutputs
```

因此，`Scheduler` 不在前端 SyncMPClient 进程里，而是在 EngineCoreProc 所在进程里的 EngineCore 内部。

## process0 和 process1 的关系

在同步多进程模式下，可以画成：

```text
process0：前端进程

  LLM / Engine client
      |
      v
  SyncMPClient
      |
      |  ZMQ input socket: EngineCoreRequest
      v

--------------------------------------------------

process1：EngineCore 后台进程

  EngineCoreProc
      |
      |  input socket thread -> input_queue
      v
  EngineCore busy loop
      |
      |-- _process_input_queue()
      |
      |-- _process_engine_step()
              |
              |-- Scheduler.schedule()
              |-- ModelExecutor.execute_model()
              |-- Scheduler.update_from_output()
      |
      v
  output_queue -> output socket thread
      |
      |  ZMQ output socket: EngineCoreOutputs
      v

--------------------------------------------------

process0：SyncMPClient 接收 EngineCoreOutputs
```

## 请求流转流程

一个普通请求大致流转如下：

```text
1. 前端调用 SyncMPClient.add_request()

2. SyncMPClient 将 EngineCoreRequest 序列化后发到 ZMQ input socket

3. EngineCoreProc 的 input socket thread 收到请求

4. input socket thread 把请求放进 EngineCoreProc.input_queue

5. EngineCore busy loop 调用 _process_input_queue()

6. EngineCore 把 request 加入 Scheduler

7. EngineCore 调用 step()

8. Scheduler.schedule() 生成 SchedulerOutput

9. ModelExecutor.execute_model() 执行模型 forward

10. Scheduler.update_from_output() 更新 request 状态，生成 EngineCoreOutputs

11. EngineCoreProc 把 EngineCoreOutputs 放进 output_queue

12. output socket thread 通过 ZMQ 发回 SyncMPClient

13. SyncMPClient.get_output() 返回结果给前端
```

## 为什么不能说所有 vLLM 都固定两个进程

`process0 = SyncMPClient`、`process1 = EngineCoreProc` 这个说法只适用于特定模式：

```text
vLLM v1
multiprocess_mode=True
asyncio_mode=False
非 data parallel 或单 EngineCore 场景
```

不能泛化到所有场景。

### InprocClient 场景

`InprocClient` 用于 in-process EngineCore：

```text
EngineCore 和 client 在同一个进程里
没有后台 EngineCoreProc 进程
```

### AsyncMPClient 场景

`AsyncLLM` 场景下使用：

```text
AsyncMPClient
```

不是 `SyncMPClient`。

它仍然可以连接后台 EngineCoreProc，但前端 client 是 asyncio-compatible 的。

### Data Parallel 场景

如果 data parallel size > 1，可能有多个 EngineCore 进程：

```text
EngineCore_DP0
EngineCore_DP1
...
```

此时前端 client 可能管理多个 EngineCore，而不是简单的 process0/process1 两进程模型。

### Executor / Worker 可能引入更多进程或 actor

`EngineCoreProc` 内部会调用 `ModelExecutor`，具体执行模型时可能还会由 executor 管理更多 worker、Ray actor 或多进程资源。

这取决于：

```text
executor backend
TP/PP/DP 配置
是否使用 Ray
是否启用多进程 worker
```

所以更稳妥的说法是：

```text
SyncMPClient 和 EngineCoreProc 描述的是 vLLM v1 前端 client 与后台 EngineCore 的一层进程边界，不一定覆盖整个模型执行拓扑中的所有进程。
```

## SyncMPClient 和 EngineCoreProc 的职责区别

| 对象 | 所在位置 | 主要职责 |
|---|---|---|
| `SyncMPClient` | 前端进程 | 同步客户端代理，发送请求，接收输出，管理后台 EngineCore 进程生命周期 |
| `EngineCoreProc` | 后台 EngineCore 进程 | 包装 EngineCore，处理 ZMQ 通信，运行 busy loop |
| `EngineCore` | EngineCoreProc 内部 | 管理 Scheduler、ModelExecutor、请求状态推进 |
| `Scheduler` | EngineCore 内部 | 决定每轮调度哪些请求、分配 token/KV blocks、处理 preempt/resume/finish |
| `ModelExecutor` | EngineCore 内部或其管理的 worker 中 | 根据 SchedulerOutput 执行模型 forward |

## 总结

用户的说法：

```text
vllm采用2个不同的进程（process0，process1）来完成整个推理。
process0上维护的核心对象是SyncMPClient，简称客户端。
process1上维护的核心对象是EngineCoreProc，简称EngineCore。
```

在 vLLM v1 的同步多进程模式下基本正确，但建议改成更精确的表述：

```text
在 vLLM v1 的同步多进程模式下，前端进程维护 SyncMPClient，后台 EngineCore 进程运行 EngineCoreProc。SyncMPClient 通过 ZMQ 向 EngineCoreProc 发送请求并接收输出；EngineCoreProc 内部运行 EngineCore busy loop，负责调用 Scheduler 和 ModelExecutor 完成调度、模型执行和状态更新。
```

同时需要补充限制：

```text
这不是所有 vLLM 模式的固定进程结构。InprocClient、AsyncMPClient、Data Parallel、Ray executor 等场景会改变 client 类型或 EngineCore 进程数量。
```

## 为什么不是所有 vLLM 模式都固定为两个进程

上面的 `process0 = SyncMPClient`、`process1 = EngineCoreProc` 只描述 vLLM v1 的同步多进程模式。vLLM 的实际运行拓扑会受 API 入口、是否启用多进程、并行配置和 executor backend 影响。

### 1. InprocClient：没有单独的 EngineCoreProc 进程

如果使用 `InprocClient`，EngineCore 在当前进程内运行，不会启动后台 `EngineCoreProc`。

这种模式更接近：

```text
process0
  |
  |-- InprocClient
  |-- EngineCore
  |-- Scheduler
  |-- ModelExecutor
```

也就是说，client 和 EngineCore 在同一个进程里，不存在 `process1 = EngineCoreProc` 这个后台进程。

因此，两进程模型不适用于 `InprocClient`。

### 2. AsyncMPClient：前端 client 类型不同

如果是 AsyncLLM / asyncio 模式，前端使用的不是 `SyncMPClient`，而是 `AsyncMPClient`。

它仍然可以连接后台 `EngineCoreProc`，但前端侧对象变成：

```text
process0：AsyncMPClient
process1：EngineCoreProc
```

因此，虽然仍可能是多进程结构，但不能说 process0 的核心对象一定是 `SyncMPClient`。

### 3. Data Parallel：EngineCoreProc 可能不止一个

如果开启 data parallel，vLLM 可能启动多个 EngineCore 进程，例如：

```text
process0：SyncMPClient / AsyncMPClient
process1：EngineCore_DP0
process2：EngineCore_DP1
process3：EngineCore_DP2
...
```

此时不是一个前端进程对应一个 EngineCore 进程，而是一个 client 可能管理多个 EngineCore ranks。

在内部 load balancing 场景下，client 还需要在多个 DP EngineCore 之间分发请求。

因此，两进程模型不适用于 DP 多 EngineCore 场景。

### 4. Ray executor：模型执行拓扑可能继续扩展

即使前端和 EngineCore 仍然是多进程结构，`EngineCoreProc` 内部的 `ModelExecutor` 也可能通过 Ray actor 或其他 worker 机制把模型执行分发到更多进程 / actor 上。

这种情况下，整体拓扑可能类似：

```text
process0：Client
process1：EngineCoreProc
process2+：Ray workers / executor workers
```

所以 `SyncMPClient + EngineCoreProc` 只覆盖 client 到 EngineCore 这一层边界，不代表完整推理系统只有两个进程。

### 5. TP / PP / worker backend 也会影响执行结构

Tensor parallel、pipeline parallel、不同 executor backend 都可能让模型执行侧出现额外 worker、rank 或 actor。

这些 worker 不一定改变 `SyncMPClient -> EngineCoreProc` 这层关系，但会改变完整推理链路中的进程数量。

### 最终理解

更准确的理解方式是：

```text
SyncMPClient / AsyncMPClient / InprocClient 描述的是前端如何访问 EngineCore。
EngineCoreProc 描述的是 EngineCore 是否运行在独立后台进程中。
Executor / Worker 描述的是模型 forward 如何实际执行。
```

因此，不能把 vLLM 简化为永远两个进程。只能说：

```text
在 vLLM v1 同步多进程、单 EngineCore 的典型场景下，可以近似理解为两个进程：前端 SyncMPClient 进程和后台 EngineCoreProc 进程。
```
