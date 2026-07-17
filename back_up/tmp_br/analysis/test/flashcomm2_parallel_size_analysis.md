# VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE 环境变量梳理

分析对象：`D:/lzy/code/for_env/vllm-ascend` 的 `main` 分支。

## 1. 变量定义

`vllm_ascend/envs.py:75` 定义了该环境变量：

```python
"VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
```

含义：

- 默认值是 `0`，表示关闭 FlashComm2。
- 取值 `> 0` 时开启 FlashComm2。
- 这个数值本身会作为 FlashComm2 的 O projection TP group size，也就是代码里的 `flashcomm2_oproj_tensor_parallel_size`。

## 2. 开关与配置落点

`vllm_ascend/utils.py:1145` 的 `flashcomm2_enable()` 只判断环境变量是否大于 0：

```python
def flashcomm2_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE > 0
```

`vllm_ascend/ascend_config.py:149` 在构造 `AscendConfig` 时调用校验函数，并把结果保存到配置对象：

```python
self.flashcomm2_oproj_tensor_parallel_size = get_flashcomm2_config_and_validate(self, vllm_config)
```

因此后续代码不再直接读环境变量值，而是使用 `get_ascend_config().flashcomm2_oproj_tensor_parallel_size`。

## 3. 代码中怎么用

### 3.1 配置校验

`vllm_ascend/utils.py:1156` 的 `get_flashcomm2_config_and_validate()` 是核心校验点：

- 环境变量未开启时返回 `0`。
- 开启时打印 `Enable FLASHCOMM2 with flashcomm2_oproj_tensor_parallel_size = ...`。
- 允许 `layer_sharding` 为空，或仅为 `["o_proj"]`。
- 如果未同时开启 `VLLM_ASCEND_ENABLE_FLASHCOMM1`，只给 warning，不阻断启动。
- 与 `additional_config.finegrained_tp_config.oproj_tensor_parallel_size` 互斥。
- 环境变量值必须小于全局 TP size。
- 全局 TP size 必须能被环境变量值整除。
- D 节点不允许使用，即 `kv_transfer_config.is_kv_consumer == True` 会直接报错。
- 没有 `kv_transfer_config` 的 mixed/普通部署只给 warning，提示更推荐 P 场景，混合场景可能导致 decode 性能下降。

### 3.2 通信组划分

`vllm_ascend/distributed/parallel_state.py:151` 在 `init_ascend_model_parallel()` 里根据该值创建 FlashComm2 专用通信组：

- `flashcomm2_otp_size = get_ascend_config().flashcomm2_oproj_tensor_parallel_size`
- `num_fc2_oproj_tensor_parallel_groups = global_tp_size // flashcomm2_otp_size`
- `_FLASHCOMM2_OTP`：O projection 的 TP 组。
- `_FLASHCOMM2_ODP`：FlashComm2 的 output data parallel 组。

当 `flashcomm2_otp_size == 1` 时不会创建新的 OTP 组，`get_flashcomm2_otp_group()` 返回 `None`，对应算子里 `tp_size == 1`、`tp_rank == 0`，但 `_FLASHCOMM2_ODP` 会复用 `get_tp_group()`。

### 3.3 O projection 算子替换

`vllm_ascend/ops/linear_op.py:673` 在 Row Parallel 算子选择时，如果 FlashComm2 开启，并且层名包含 `o_proj` 或 `out_proj`，会使用 `Flashcomm2OProjRowParallelOp`：

```python
if flashcomm2_enable():
    if "o_proj" in prefix or "out_proj" in prefix:
        return Flashcomm2OProjRowParallelOp(layer)
```

`Flashcomm2OProjRowParallelOp` 的主要逻辑在 `vllm_ascend/ops/linear_op.py:272`：

- 先按 FlashComm2 的 batch/rank 布局重排输入。
- 用 `_FLASHCOMM2_ODP` 做 `all_to_all_single`。
- 再做 O projection 计算。
- 如果 `flashcomm2_oproj_tensor_parallel_size > 1`，使用 `_FLASHCOMM2_OTP.reduce_scatter()` 聚合结果。
- 如果 FlashComm1 没启用，则最后还会走全局 TP `all_gather()` 把结果收回来。

### 3.4 batch padding

`vllm_ascend/ascend_forward_context.py:130` 和 `vllm_ascend/platform.py:743` 都会在 forward context 里设置 `flashcomm_v2_enabled`：

```python
flashcomm_v2_enabled = flashcomm2_enable() and tp_world_size > 1 and num_tokens is not None
```

开启后会把 token 数 padding 到 `tp_world_size` 的倍数，而不是 padding 到 `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` 的倍数。

### 3.5 O-shard / layer_sharding 联动

`vllm_ascend/ops/flashcomm2_oshard_manager.py:34` 判断 O-shard 是否开启：

```python
return flashcomm2_enable() and o_shard_enable()
```

其中 `o_shard_enable()` 来自 `vllm_ascend/utils.py:1149`，只看 `additional_config.layer_sharding` 是否包含 `o_proj`。这意味着：

- 仅设置 `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` 会开启 FlashComm2，但不会开启 O-shard。
- 要开启 O-shard，需要同时加 `--additional-config '{"layer_sharding": ["o_proj"]}'`。
- FlashComm2 场景下，`layer_sharding` 只能是 `["o_proj"]`，不能带 `q_b_proj` 等其他层。

## 4. 使用限制总结

| 限制项 | 代码依据 | 说明 |
| --- | --- | --- |
| 必须是正整数才开启 | `vllm_ascend/utils.py:1145` | `0` 或未设置表示关闭。 |
| 必须小于全局 TP size | `vllm_ascend/utils.py:1182` | `global_tp_size <= value` 会报错，所以 TP=4 时 value 不能是 4。 |
| 全局 TP size 必须能整除该值 | `vllm_ascend/utils.py:1187` | TP=4 时合法值只有 `1`、`2`；`4` 因上一条不合法。 |
| 不能和 finegrained TP 的 oproj TP 同时开 | `vllm_ascend/utils.py:1178` | `oproj_tensor_parallel_size > 0` 会报错。 |
| FlashComm2 下 layer_sharding 只能是 `["o_proj"]` | `vllm_ascend/utils.py:1165` | 其他 layer sharding 配置会报错。 |
| D 节点不能用 | `vllm_ascend/utils.py:1197` | `kv_transfer_config.is_kv_consumer` 时会报错。 |
| 更推荐 P 场景 | `vllm_ascend/utils.py:1192` | 没有 `kv_transfer_config` 时只是 warning，不阻断。 |
| 建议同时开启 FlashComm1 | `vllm_ascend/utils.py:1174` | 不开 FlashComm1 只 warning，但性能可能不是最优，并且算子末尾会额外 all_gather。 |
| 仅实际替换 `o_proj`/`out_proj` Row Parallel 层 | `vllm_ascend/ops/linear_op.py:673` | 不是全模型所有线性层都走 FlashComm2。 |
| forward 时需要 TP > 1 且 num_tokens 非空 | `vllm_ascend/ascend_forward_context.py:130` | context 中的 `flashcomm_v2_enabled` 才会生效并做 padding。 |

## 5. 对 `run_flashcomm2.sh` 的判断

脚本路径：`D:/lzy/code/for_env/result/analysis/test/run_flashcomm2.sh`。

关键配置：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1
vllm serve ${MODEL_PATH} \
    --tensor-parallel-size 4 \
    --max-num-batched-tokens 2768 \
    ...
```

结论：这个脚本从代码限制看是可以启动 FlashComm2 的，`VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1` 对 `--tensor-parallel-size 4` 是合法值。

理由：

- `1 > 0`，会开启 FlashComm2。
- 全局 TP size 是 `4`，环境变量值是 `1`，满足 `1 < 4`。
- `4 % 1 == 0`，满足整除要求。
- 脚本没有设置 `finegrained_tp_config.oproj_tensor_parallel_size`，不触发互斥。
- 脚本没有设置 `kv_transfer_config` 为 D 节点，不触发 D 场景禁止。
- `--max-num-batched-tokens 2768` 能被 TP size `4` 整除，forward padding 不是问题。

但这个脚本不是代码里最推荐的性能写法：

- 未设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`。代码只 warning，不会报错；但 `Flashcomm2OProjRowParallelOp` 在 FlashComm1 未启用时会额外执行全局 TP `all_gather()`，性能可能不如同时开 FlashComm1。
- 未设置 `--additional-config '{"layer_sharding": ["o_proj"]}'`。这不影响 FlashComm2 启动，但不会启用 O-shard，也就没有文档里提到的 O projection 权重分片省内存效果。
- 当前模型是 `llama3-8b` dense 模型，不是 MoE。FlashComm2 的 O projection 路径仍可命中 `o_proj`，但项目现有 e2e 覆盖示例主要是 Qwen MoE TP2，不能据此断言这个脚本一定有最佳收益。

建议如果目标是“验证 FlashComm2 功能是否能走通”，当前写法基本正确。若目标是“按推荐性能组合验证”，建议至少改成：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1
```

如果还想验证 FlashComm2 + O-shard 省内存路径，则在 serve 参数中增加：

```bash
--additional-config '{"layer_sharding": ["o_proj"]}'
```

这三项同时使用时仍满足代码限制，因为 FlashComm2 只允许 `layer_sharding` 为 `["o_proj"]`。

## 6. TP=4 下的取值建议

对当前脚本的 `--tensor-parallel-size 4`：

- `0`：关闭 FlashComm2。
- `1`：合法，现脚本使用值；OTP 计算侧相当于单 rank，ODP 复用 TP group。
- `2`：合法，会创建大小为 2 的 FlashComm2 OTP 组和对应 ODP 组。
- `3`：不合法，`4 % 3 != 0`。
- `4`：不合法，代码要求该值不能大于等于 global TP size。

因此当前脚本的 `1` 是保守合法值；如果要测试真正的 FlashComm2 OTP 分组通信，可考虑对比 `2`。