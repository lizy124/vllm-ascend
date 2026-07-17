# Async Scheduling 中 num_output_placeholders 跳过逻辑说明

## 背景

分析代码位置：

```text
vllm/v1/core/sched/scheduler.py
```

相关逻辑位于 `Scheduler.schedule()` 中调度 `self.running` 请求的阶段：

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    if (
        request.num_output_placeholders > 0
        and request.num_computed_tokens + 2 - request.num_output_placeholders
        >= request.num_prompt_tokens + request.max_tokens
    ):
        req_index += 1
        continue
```

这段逻辑只在请求存在 `num_output_placeholders` 时生效，主要服务于 async scheduling 场景。

## vLLM scheduler 是一轮一轮调度的

vLLM v1 scheduler 是 step-based / round-based 的。一次 `Scheduler.schedule()` 调用可以理解为一轮调度。

每一轮调度大致做三件事：

```text
1. 选择这一轮要执行哪些请求
2. 给每个请求分配本轮要计算的 token 数和 KV cache blocks
3. 构造 SchedulerOutput，交给 model runner 执行一次 forward
```

所以一轮调度不是把一个请求完整执行完，而是把所有请求按当前资源预算推进一小步。

`token_budget` 表示这一轮最多能调度多少 token：

```python
token_budget = self.max_num_scheduled_tokens
```

每成功调度一个请求，都会扣减：

```python
token_budget -= num_new_tokens
```

因此，`while req_index < len(self.running) and token_budget > 0` 的含义是：

```text
只要 running 队列还有请求没检查完，并且这一轮还有 token 预算，就继续尝试调度 running 请求。
```

## 为什么先调度 running 请求

`schedule()` 中调度顺序是：

```text
1. 先调度 self.running 中已经在运行的请求
2. 如果本轮没有发生 preemption，再调度 waiting / skipped_waiting 中的新请求或恢复请求
```

这样做可以优先推进已经进入运行队列的请求，减少已经运行请求被新请求饿死或频繁切换的概率。

## 为什么用 while 而不是 for

调度 `self.running` 时使用：

```python
while req_index < len(self.running) and token_budget > 0:
```

而不是简单的 `for request in self.running`，原因是循环过程中 `self.running` 可能会被修改。

例如，当 KV cache block 不足时，scheduler 可能 preempt 某个 running request：

```python
self.running.remove(preempted_req)
```

或者：

```python
preempted_req = self.running.pop()
```

如果使用 `for`，遍历过程中修改列表容易导致跳过元素或索引错乱。使用 `while + req_index` 可以手动控制：

- 当前 request 是否继续检查
- preempt 后是否需要回退 `req_index`
- running 队列长度变化后是否还能继续遍历

## num_output_placeholders 是什么

`num_output_placeholders` 是 async scheduling 中使用的字段，定义在：

```text
vllm/v1/request.py
```

初始化为：

```python
self.num_output_placeholders = 0
```

在 async scheduler 中，每当一个非 prefill 请求被调度后，scheduler 会提前认为这个请求将生成：

```text
1 个正常输出 token + 若干 speculative draft tokens
```

于是会增加 placeholder：

```python
cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
request.num_output_placeholders += 1 + cur_num_spec_tokens
```

位置：

```text
vllm/v1/core/sched/async_scheduler.py
```

这些 placeholder 的作用是让 scheduler 在模型真实输出返回之前，也能把“已安排但尚未确认的输出 token”计入调度状态。

当模型输出回来后，会根据实际生成的 token 数减少 placeholder：

```python
request.num_output_placeholders -= len(new_token_ids)
```

因此：

```text
num_output_placeholders > 0
```

表示这个请求还有上一轮 async scheduling 预留的输出 token / draft token 尚未完全结算。

## 这段 if 判断在做什么

原始判断：

```python
if (
    request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens
):
    req_index += 1
    continue
```

它的目标是：

```text
如果一个请求已经可以确定达到 max_tokens，就不要再给它多调度一步。
```

这主要是为了避免 async scheduling 下多执行一次不必要的 decode step。

## 右边表达式含义

```python
request.num_prompt_tokens + request.max_tokens
```

表示请求最多允许计算到的位置。

例如：

```text
prompt tokens = 10
max_tokens = 5
```

那么这个请求最多需要生成到：

```text
10 + 5 = 15
```

也就是 prompt 加最多输出 token 的总长度上限。

## 左边表达式含义

```python
request.num_computed_tokens + 2 - request.num_output_placeholders
```

代码注释中写的是：

```python
(num_computed_tokens + 1) - (num_output_placeholders - 1)
```

化简后就是：

```python
num_computed_tokens + 2 - num_output_placeholders
```

为什么要这样算？

因为 `num_output_placeholders` 包含：

```text
1 个正常输出 token placeholder + N 个 draft token placeholders
```

其中 speculative draft tokens 可能全部被拒绝。为了保守判断是否已经达到 `max_tokens`，代码需要把 draft token placeholders 从 computed count 里扣掉。

也就是扣掉：

```python
num_output_placeholders - 1
```

保留至少会产生的那个正常输出 token。

所以：

```python
(num_computed_tokens + 1) - (num_output_placeholders - 1)
```

可以理解为：

```text
即使所有 draft tokens 都被拒绝，当前请求至少已经推进到的位置。
```

## 举例说明

假设：

```text
prompt tokens = 10
max_tokens = 5
最大允许位置 = 15

num_computed_tokens = 17
num_output_placeholders = 3
```

这里的 3 个 placeholders 可以理解为：

```text
1 个正常输出 token + 2 个 draft tokens
```

如果 2 个 draft tokens 最后都被拒绝，那么保守有效位置是：

```text
17 + 2 - 3 = 16
```

也就是：

```python
request.num_computed_tokens + 2 - request.num_output_placeholders
```

结果为 16。

由于：

```text
16 >= 15
```

说明即使 draft tokens 全部失败，请求也已经足够达到 `max_tokens`。因此 scheduler 不应该再给这个请求多安排一次 forward。

所以代码执行：

```python
req_index += 1
continue
```

跳过这个 request，继续检查下一个 running request。

## 为什么不调度 partial draft tokens

注释中提到：

```text
We don't schedule partial draft tokens since this prevents uniform decode optimizations.
```

意思是：不要为了只补一点 draft token 而单独调度这个请求。

uniform decode 优化通常希望 batch 内 decode 形态尽量一致。如果某个请求只需要很少的 partial draft tokens，会破坏 batch 的统一性，降低优化效果。

因此，当 scheduler 已经能确定请求会达到 `max_tokens` 时，直接跳过它，而不是安排一次很小的、形态特殊的 decode step。

## 和 prefill / decode 的关系

这段逻辑主要影响 decode 阶段，尤其是 async scheduling + speculative decoding 场景。

普通 prefill 阶段通常不会有 `num_output_placeholders > 0`，因为 placeholders 是 async scheduling 在非 prefill step 后添加的。

所以这段 if 判断可以理解为：

```text
针对异步 decode 中还存在未结算 placeholder 的请求，判断是否已经可以安全停止继续调度。
```

## 总结

这段代码的核心作用是：

```text
在 async scheduling 场景下，如果一个 running request 已经保守地达到 max_tokens，跳过它，避免多调度一次无意义的 decode step。
```

它依赖 `num_output_placeholders` 来处理异步调度带来的“已安排但尚未确认”的输出 token。

表达式：

```python
request.num_computed_tokens + 2 - request.num_output_placeholders
```

是在扣除 speculative draft token placeholders 后，估算即使 draft 全部被拒绝，请求至少已经推进到哪里。

如果这个保守位置已经达到：

```python
request.num_prompt_tokens + request.max_tokens
```

则说明请求不需要继续调度，直接跳过即可。
