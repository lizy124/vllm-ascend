# vLLM v1 Scheduler 类整体介绍

## 位置

`Scheduler` 类位于：

```text
vllm/v1/core/sched/scheduler.py
```

类定义位置：

```python
class Scheduler(SchedulerInterface):
```

`Scheduler` 是 vLLM v1 EngineCore 中的请求调度中枢。它不负责真正执行模型 forward，而是负责决定：

```text
这一轮哪些 request 要跑
每个 request 跑多少 token
需要分配哪些 KV cache blocks
哪些 request 要 preempt / resume / finish
最后把调度结果打包成 SchedulerOutput 给 model runner
```

## 核心生命周期

一个 request 在 Scheduler 中的典型生命周期是：

```text
add_request()
  -> 请求进入 waiting 队列

schedule()
  -> 从 running / waiting 中选择请求
  -> 分配 token budget 和 KV cache blocks
  -> 生成 SchedulerOutput

model runner forward
  -> 执行模型计算

update_from_output()
  -> 根据模型输出更新 request 状态
  -> 处理 stop、logprobs、spec decode、KV transfer、finished request
```

可以把 Scheduler 理解成：

```text
请求状态机 + 资源分配器 + 调度输出构造器
```

## Scheduler 管理的核心状态

### 请求集合

初始化中有几个核心容器：

```python
self.requests: dict[str, Request] = {}
self.waiting = create_request_queue(self.policy)
self.skipped_waiting = create_request_queue(self.policy)
self.running: list[Request] = []
self.finished_req_ids: set[str] = set()
```

含义：

| 字段 | 含义 |
|---|---|
| `requests` | 所有尚未彻底释放的 request，按 `request_id` 索引 |
| `waiting` | 等待被调度的新请求，或 preempt 后等待恢复的请求 |
| `skipped_waiting` | 暂时不能调度的 waiting 请求，例如等待 remote KV、grammar、streaming input |
| `running` | 已经进入运行队列、拥有运行状态的请求 |
| `finished_req_ids` | 两轮调度之间结束的请求 ID，用于通知 worker 清理缓存状态 |

### 调度约束

Scheduler 初始化时会设置几个调度上限：

```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
self.max_num_scheduled_tokens = ...
self.max_model_len = vllm_config.model_config.max_model_len
```

含义：

| 字段 | 含义 |
|---|---|
| `max_num_running_reqs` | 最多同时处于 running 队列的请求数量 |
| `max_num_scheduled_tokens` | 一轮调度最多安排多少 token |
| `max_model_len` | 单个 request 最多能推进到的模型长度 |

### Cache 管理器

Scheduler 还持有 cache 相关组件：

```python
self.kv_cache_manager = KVCacheManager(...)
self.encoder_cache_manager = ...
```

Scheduler 不执行 attention，但它负责决定：

```text
哪些 request 能拿到 KV cache block
哪些 request 的 KV cache 要释放
哪些 encoder inputs 能进入 encoder cache
```

## add_request()

`add_request()` 负责把新请求放入 Scheduler 管理范围。

它主要做：

```text
1. 检查 request_id 是否重复
2. 把 Request 放进 self.requests
3. 根据 request 状态放进 waiting 或 skipped_waiting
```

普通新请求一般进入：

```text
waiting
```

如果请求暂时被阻塞，例如：

- 等 structured output grammar
- 等 remote KV
- 等 streaming input

则进入：

```text
skipped_waiting
```

## schedule()

`schedule()` 是 `Scheduler` 最核心的方法。一次 `schedule()` 调用可以理解为一轮调度。

一轮调度大致做三件事：

```text
1. 选择这一轮要执行哪些请求
2. 给每个请求分配本轮要计算的 token 数和 KV cache blocks
3. 构造 SchedulerOutput，交给 model runner 执行一次 forward
```

### schedule() 中的四类请求列表

`schedule()` 开头会创建四个列表：

```python
scheduled_new_reqs: list[Request] = []
scheduled_resumed_reqs: list[Request] = []
scheduled_running_reqs: list[Request] = []
preempted_reqs: list[Request] = []
```

含义：

| 列表 | 来源 | 含义 |
|---|---|---|
| `scheduled_running_reqs` | `self.running` | 已经在 running 队列里，本轮继续执行 |
| `scheduled_new_reqs` | `waiting`，状态是 `WAITING` | 第一次被调度的新请求 |
| `scheduled_resumed_reqs` | `waiting`，状态是 `PREEMPTED` | 之前被抢占，本轮恢复执行 |
| `preempted_reqs` | `self.running` | 本轮因为资源不足被抢占出去 |

## schedule() 第一阶段：调度 running 请求

`schedule()` 会先调度 `self.running` 中已有的请求：

```python
while req_index < len(self.running) and token_budget > 0:
```

含义：

```text
只要 running 队列还有请求没检查完，并且这一轮还有 token 预算，就继续尝试调度 running 请求。
```

为什么先调度 running 请求？

```text
running 请求已经进入执行状态，优先推进它们可以减少频繁切换，也能避免已运行请求被新请求饿死。
```

为什么用 `while` 而不是 `for`？

因为调度过程中 `self.running` 可能会被修改。例如 KV cache 不足时，Scheduler 可能会 preempt 某个 running request：

```python
self.running.remove(preempted_req)
```

或者：

```python
preempted_req = self.running.pop()
```

使用 `while + req_index` 可以在列表变化时手动控制遍历位置。

这一阶段主要做：

```text
1. 计算 request 本轮需要多少 num_new_tokens
2. 检查 max_model_len、long_prefill_token_threshold、encoder inputs、mamba block 对齐等约束
3. 调 KVCacheManager.allocate_slots() 分配 KV blocks
4. 如果 KV blocks 不够，可能 preempt 低优先级 running request
5. 分配成功则加入 scheduled_running_reqs
```

## preempt：抢占请求

当 KV cache blocks 不足时，Scheduler 可能会抢占某个 running request。

核心方法：

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
```

它主要做：

```text
1. 释放这个 request 的 KV cache
2. 释放 encoder cache
3. request.status = PREEMPTED
4. request.num_computed_tokens = 0
5. 清掉 spec tokens
6. request.num_preemptions += 1
7. 放回 waiting 队列头部
```

所以 preempt 不是 finish。它只是让请求暂时让出 cache 资源，后续还可以恢复执行。

这也是为什么 Scheduler 需要 `scheduled_resumed_reqs`：

```text
被 preempt 的请求以后重新调度时，不是全新请求，但也不能按普通 running 请求处理。
```

## schedule() 第二阶段：调度 waiting 请求

调度完 running 后，Scheduler 会尝试调度 waiting 队列：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

注意：如果本轮发生了 preemption，就不会继续调度 waiting 请求。

原因是：

```text
preemption 已经说明资源紧张，此时再引入 waiting 请求会让调度更复杂，也可能导致更多抢占。
```

waiting 队列里常见两类请求：

### 新请求

状态是 `WAITING`：

```python
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
```

这种请求第一次被调度，需要发送完整 `NewRequestData` 给 worker。

### 恢复请求

状态是 `PREEMPTED`：

```python
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)
```

这种请求之前运行过，但被抢占释放了 KV cache。本轮重新分配 KV blocks 后恢复执行。

## SchedulerOutput

`schedule()` 最终会构造 `SchedulerOutput`：

```python
SchedulerOutput(
    scheduled_new_reqs=new_reqs_data,
    scheduled_cached_reqs=cached_reqs_data,
    num_scheduled_tokens=num_scheduled_tokens,
    scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
    scheduled_encoder_inputs=scheduled_encoder_inputs,
    preempted_req_ids={req.request_id for req in preempted_reqs},
    finished_req_ids=self.finished_req_ids,
)
```

核心字段含义：

| 字段 | 含义 |
|---|---|
| `scheduled_new_reqs` | 新请求的完整数据 |
| `scheduled_cached_reqs` | running / resumed 请求的增量数据 |
| `num_scheduled_tokens` | 每个 request 本轮调度多少 token |
| `scheduled_spec_decode_tokens` | speculative decoding 的 draft tokens |
| `scheduled_encoder_inputs` | 多模态 / encoder 输入本轮要处理哪些 |
| `preempted_req_ids` | 本轮被抢占的 request IDs |
| `finished_req_ids` | 通知 worker 清理已结束请求的缓存状态 |

## _make_cached_request_data()

`_make_cached_request_data()` 会把：

```python
scheduled_running_reqs
scheduled_resumed_reqs
```

合成：

```python
CachedRequestData
```

其中一个关键字段是：

```python
resumed_req_ids
```

它告诉 worker 哪些请求是 resumed。

区别在于：

```text
普通 running request：new_block_ids 追加到已有 block IDs 后面
resumed request：new_block_ids 替换旧 block IDs
```

原因是 preempt 时旧 KV cache blocks 已经释放，恢复时分配的是新的 block。

所以 `scheduled_resumed_reqs` 不能简单并入 `scheduled_running_reqs`。

## update_from_output()

`model runner` 执行完 forward 后，会返回 `ModelRunnerOutput`。Scheduler 使用 `update_from_output()` 更新状态。

核心方法：

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
```

它主要做：

```text
1. 读取 sampled_token_ids / logprobs / pooler_outputs
2. 处理 spec decode 中 accepted / rejected tokens
3. 更新 request output token
4. 检查 stop condition
5. 推进 structured output grammar
6. 释放 finished request 的 cache
7. 处理 KV connector 输出
8. 生成 EngineCoreOutputs 返回给上层
```

如果请求结束，会走：

```python
_handle_stopped_request()
_free_request()
```

如果请求是 streaming request，可能不会彻底结束，而是重新进入 waiting，等待后续 streaming input。

## async scheduling 特殊逻辑

Scheduler 中有一些 async scheduling 相关逻辑，例如：

```python
request.num_output_placeholders
```

async scheduling 会提前给请求增加 output placeholders，表示：

```text
这一轮已经安排了输出 token，但模型真实输出还没有完全同步回来。
```

因此 `schedule()` 中有逻辑用于避免多调度一步：

```python
if (
    request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens
):
    req_index += 1
    continue
```

它的含义是：

```text
如果即使 speculative draft tokens 全部被拒绝，请求也已经保守达到 max_tokens，就不要再调度它。
```

这可以避免 async scheduling 下多跑一次不必要的 decode step。

## KV Connector / Remote KV 相关能力

Scheduler 还负责 P/D disaggregation、KV offload、remote KV loading 相关状态。

相关字段包括：

```python
self.connector
self.finished_recving_kv_req_ids
self.failed_recving_kv_req_ids
```

相关方法包括：

- `_connector_finished()`
- `_update_waiting_for_remote_kv()`
- `_try_promote_blocked_waiting_request()`
- `_update_from_kv_xfer_finished()`
- `_update_requests_with_invalid_blocks()`
- `_handle_invalid_blocks()`

这些逻辑主要用于：

```text
某些 request 不能立刻调度，因为它还在等远端 KV。
等 KV 到了，再从 skipped_waiting 提升回可调度状态。
如果 KV load 失败，则根据策略 recompute 或报错。
```

## Encoder / Multimodal 相关能力

对于 encoder-decoder 或 multimodal 模型，Scheduler 还要调度 encoder inputs。

核心方法：

```python
_try_schedule_encoder_inputs()
```

它受两个预算约束：

```text
encoder_compute_budget
encoder cache capacity
```

因此一个请求即使 token budget 足够，也可能因为 encoder budget 或 encoder cache 不够而本轮无法调度。

## Prefix cache reset 能力

Scheduler 管理 KV cache，因此提供 prefix cache reset 能力：

```python
reset_prefix_cache()
```

它可以：

```text
1. reset KV cache manager
2. reset connector cache
3. 必要时 preempt running requests
4. 清理 async scheduling 下的 placeholder 状态
```

## stats / metrics

Scheduler 还能生成调度统计：

```python
make_stats()
make_spec_decoding_stats()
```

统计内容包括：

- running / waiting 请求数量
- KV cache usage
- prefix cache stats
- spec decoding 接受率
- encoder cache usage
- cudagraph stats
- perf metrics

## 总体流程图

```text
add_request(request)
  |
  v
waiting / skipped_waiting
  |
  v
schedule()
  |
  |-- 先调度 running
  |     |-- 成功：scheduled_running_reqs
  |     |-- 资源不足：preempted_reqs
  |
  |-- 再调度 waiting
  |     |-- WAITING：scheduled_new_reqs
  |     |-- PREEMPTED：scheduled_resumed_reqs
  |
  v
SchedulerOutput
  |
  v
model runner forward
  |
  v
update_from_output()
  |
  |-- 更新输出 token
  |-- 处理 stop / finish
  |-- 处理 spec decode
  |-- 处理 KV transfer
  |-- 释放 finished request
  |
  v
EngineCoreOutputs
```

## 总结

`Scheduler` 是 vLLM v1 的请求调度核心。它的职责不是执行模型，而是在每一轮调度中综合考虑：

```text
token budget
KV cache blocks
encoder cache
running / waiting 队列
preemption / resume
speculative decoding
async scheduling
remote KV transfer
request stop / finish 状态
```

然后生成 `SchedulerOutput`，告诉 model runner 这一轮应该执行哪些请求、每个请求执行多少 token、使用哪些 cache blocks。
