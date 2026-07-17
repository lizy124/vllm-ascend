# migrate_env 分支测试思路分析

## 结论

你的测试思路是正确的：

```text
main 分支：用旧环境变量 export 启动服务
migrate_env 分支：用 --additional-config 配置启动服务
对比两边功能是否等价
```

这是验证“环境变量迁移到 config 后用户可见行为不变”的核心测试方法。

但它还不够全面。因为当前 `migrate_env` 分支已经进入过渡期设计：

```text
显式 additional_config 优先
未配置 additional_config 时回退旧环境变量
```

所以除了 main vs migrate_env 的等价性，还需要验证 migrate_env 分支自己的兼容行为。

推荐测试目标分成三类：

1. **跨分支等价性**：main env 行为 == migrate_env config 行为。
2. **过渡期兼容性**：migrate_env 上旧 env 仍然可用。
3. **优先级正确性**：migrate_env 上 config 显式配置应覆盖 env。

## 为什么你的思路是必要的

这次迁移的风险不是代码能不能启动，而是：

- 旧 env 开启的功能，在 config 开启后是否真的走同一条路径。
- config 名字、类型、默认值是否正确。
- 初始化阶段、Worker 阶段、推理阶段读取到的值是否一致。
- 旧脚本在过渡期是否仍可运行。

所以直接启动服务做端到端验证是必要的，尤其是这些变量多数影响运行时算子选择、通信路径、profiling、KV cache 处理等行为，单元测试不能完全覆盖。

## 基础测试方法

### 1. main 分支基线测试

在 `main` 分支，用环境变量启动服务。

示例：

```bash
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1
vllm serve <model> ...
```

然后跑固定请求，记录：

- 服务是否启动成功。
- 首 token / 后续 token 是否正常。
- 输出文本是否合理。
- 日志中是否出现预期功能启用信息。
- 是否出现错误、warning、fallback。

### 2. migrate_env 分支 config 等价测试

切到 `migrate_env` 分支，取消对应 env，用 `--additional-config` 启动。

示例：

```bash
unset VLLM_ASCEND_ENABLE_NZ
unset VLLM_ASCEND_ENABLE_MLAPO
unset VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE

vllm serve <model> \
  --additional-config '{"weight_nz_mode": 2, "enable_mlapo": false, "enable_matmul_allreduce": true}' \
  ...
```

对比 main 分支的结果：

- 功能是否都能启动。
- 日志是否显示走到相同/等价路径。
- 请求输出是否正常。
- 性能指标不要要求完全一致，但不能出现明显退化或功能未启用。

### 3. migrate_env 分支 env fallback 测试

在 `migrate_env` 分支，继续使用旧 env，不传 additional-config。

示例：

```bash
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1
vllm serve <model> ...
```

预期：行为应与 main 分支 env 基线一致。

这是过渡期兼容性的关键测试。

### 4. migrate_env 分支 config 优先级测试

在 `migrate_env` 分支，同时设置 env 和 config，且二者冲突。

示例：

```bash
export VLLM_ASCEND_ENABLE_MLAPO=0
vllm serve <model> \
  --additional-config '{"enable_mlapo": true}' \
  ...
```

预期：以 config 为准，即 `enable_mlapo=true`。

再反向测试：

```bash
export VLLM_ASCEND_ENABLE_MLAPO=1
vllm serve <model> \
  --additional-config '{"enable_mlapo": false}' \
  ...
```

预期：以 config 为准，即 `enable_mlapo=false`。

这类测试能验证“过渡期不是 env 覆盖 config”。

## 推荐测试矩阵

### 迁移变量映射

| 旧环境变量 | 新 config 字段 | 推荐测试方式 |
|---|---|---|
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `enable_matmul_allreduce` | 启动 MoE/TP 场景，观察 row parallel 算子/日志 |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | `enable_flashcomm2_parallel_size` | TP > config 值，验证 FlashComm2 初始化和 warning |
| `MSMONITOR_USE_DAEMON` | `msmonitor_use_daemon` | 与 torch profiler 冲突/daemon 启用行为 |
| `VLLM_ASCEND_ENABLE_MLAPO` | `enable_mlapo` | DeepSeek MLA / W8A8 场景，验证 MLAPO 开关 |
| `VLLM_ASCEND_ENABLE_NZ` | `weight_nz_mode` | 量化/非量化权重加载，验证 NZ cast 行为 |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | `enable_context_parallel` | CP/长序列场景，验证 context parallel 开关 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `enable_fused_mc2` | MoE W8A8 场景，验证 fused MC2 选择 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `enable_transpose_kv_cache_by_block` | Mooncake/KV transfer 场景，验证 fused transpose 开关 |

### 每个变量建议至少测三种形态

对每个变量，至少覆盖：

```text
main: env 开启/关闭
migrate_env: config 开启/关闭
migrate_env: env fallback 开启/关闭
```

如果时间允许，再测：

```text
migrate_env: env 与 config 冲突时 config 优先
```

## 测试优先级建议

如果资源有限，不建议 8 个变量都做完整 e2e。可以分层。

### P0：必须端到端验证

这些变量影响通信/算子路径，最容易出真实运行问题：

1. `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` / `enable_flashcomm2_parallel_size`
2. `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` / `enable_matmul_allreduce`
3. `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` / `enable_context_parallel`
4. `VLLM_ASCEND_ENABLE_FUSED_MC2` / `enable_fused_mc2`

### P1：建议端到端验证

这些也影响功能路径，但可能依赖特定模型或部署形态：

1. `VLLM_ASCEND_ENABLE_MLAPO` / `enable_mlapo`
2. `VLLM_ASCEND_ENABLE_NZ` / `weight_nz_mode`
3. `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` / `enable_transpose_kv_cache_by_block`

### P2：可以用 UT/轻量测试覆盖

`MSMONITOR_USE_DAEMON` / `msmonitor_use_daemon` 更偏 profiler 冲突逻辑，可以用 UT 覆盖 config/env 优先级，再做一次手工 smoke test。

## 测试脚本设计建议

建议脚本不要只判断服务进程启动成功，还要做一次实际请求。

流程：

```text
1. 清理旧服务进程
2. 设置 env 或 additional-config
3. 启动 vllm serve
4. 等待服务 ready
5. 发起固定 prompt 请求
6. 检查 HTTP 状态码
7. 检查返回内容非空
8. grep 日志中的关键启用/警告信息
9. 停止服务
10. 保存日志
```

建议每次测试保存：

```text
branch_name
commit_id
启动命令
env 列表
additional_config JSON
请求 payload
服务日志
请求响应
```

这样方便对比 main 和 migrate_env。

## 对比时不要过度依赖输出完全一致

LLM 输出可能受以下因素影响：

- sampling 参数
- 并发状态
- 图编译/缓存
- 通信路径差异
- 设备状态

因此建议固定：

```text
temperature=0
max_tokens 固定
top_p=1
seed 如果接口支持则固定
```

判断标准建议是：

- 服务成功启动
- 请求成功返回
- 无异常 traceback
- 功能相关日志/路径符合预期
- 输出非空且格式正确

不要强要求两个分支 token-by-token 完全一致，除非使用完全确定性配置且模型/运行路径支持确定性。

## 需要特别注意的点

### 1. 确保 env 清理干净

测试 config 路径时，要先 unset 旧 env，否则可能误以为 config 生效，其实是 env fallback 生效。

示例：

```bash
unset VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE
unset VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
unset MSMONITOR_USE_DAEMON
unset VLLM_ASCEND_ENABLE_MLAPO
unset VLLM_ASCEND_ENABLE_NZ
unset VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL
unset VLLM_ASCEND_ENABLE_FUSED_MC2
unset VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
```

### 2. 确认 config JSON 类型正确

布尔值要用 JSON bool：

```json
{"enable_mlapo": false}
```

不要写成字符串：

```json
{"enable_mlapo": "0"}
```

整数值保持数字：  

```json
{"weight_nz_mode": 2}
```

### 3. 注意优先级测试

migrate_env 当前策略是：

```text
显式 additional_config > env fallback
```

所以冲突测试非常重要。

### 4. 注意模型/部署前提

有些变量只有特定场景才有效：

- `enable_fused_mc2` 需要 MoE/W8A8/特定 EP 场景。
- `enable_mlapo` 主要影响 DeepSeek MLA / W8A8 场景。
- `enable_context_parallel` 需要 CP/长序列场景。
- `enable_transpose_kv_cache_by_block` 需要 KV transfer / Mooncake 场景。
- `enable_flashcomm2_parallel_size` 需要 TP size 满足约束。

如果场景不满足，服务正常启动不代表该变量真的被覆盖测试到了。

## 推荐最小测试集

如果只想快速验证迁移质量，建议先跑：

### Case 1：FlashComm2

main：

```bash
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=2
```

migrate_env config：

```bash
--additional-config '{"enable_flashcomm2_parallel_size": 2}'
```

migrate_env env fallback：

```bash
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=2
```

检查：服务启动、请求成功、FlashComm2 日志/警告符合预期。

### Case 2：MatmulAllReduce

main：

```bash
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1
```

migrate_env config：

```bash
--additional-config '{"enable_matmul_allreduce": true}'
```

检查：TP 场景下 row parallel 路径正常。

### Case 3：NZ

main：

```bash
export VLLM_ASCEND_ENABLE_NZ=2
```

migrate_env config：

```bash
--additional-config '{"weight_nz_mode": 2}'
```

检查：模型加载成功，相关权重转换路径不报错。

### Case 4：MLAPO

main：

```bash
export VLLM_ASCEND_ENABLE_MLAPO=0
```

migrate_env config：

```bash
--additional-config '{"enable_mlapo": false}'
```

检查：DeepSeek MLA/W8A8 场景下请求成功，日志/行为符合预期。

### Case 5：优先级冲突

migrate_env：

```bash
export VLLM_ASCEND_ENABLE_MLAPO=0
vllm serve <model> --additional-config '{"enable_mlapo": true}' ...
```

预期：以 config true 为准。

反向：

```bash
export VLLM_ASCEND_ENABLE_MLAPO=1
vllm serve <model> --additional-config '{"enable_mlapo": false}' ...
```

预期：以 config false 为准。

## 最终建议

你的测试思路是正确的，但建议扩展为四层：

```text
1. main env 基线
2. migrate_env config 等价
3. migrate_env env fallback 兼容
4. migrate_env env/config 冲突时 config 优先
```

这样才能完整覆盖当前分支的真实目标：

```text
既验证迁移后的 config 可用，
也验证过渡期旧 env 不破坏，
还验证显式 config 优先级正确。
```
