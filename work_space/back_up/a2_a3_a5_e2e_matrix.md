# tests/e2e 中 A2 / A3 / A5 平台覆盖对照表（按池化 / PD 分离 / 图模式口径）

范围：`tests/e2e`

## 先澄清术语

这份表把“池化”和“PD 分离”拆开看：

- “池化”更对应 `kv_pool` 的 store / lookup / load / save 语义。
- “PD 分离”更对应 `disaggregated_prefill` / producer-consumer / `routing.type` 这类直传链路。

不包括：

- `tests/e2e/pull_request/one_card/pooling/*`
- `runner="pooling"`
- `embed()` / `classify()` / `score()`

上面这些是 embedding / reranker / classification 的 pooling runner，不是这里说的“池化”。

## 判断口径

判断某个平台是否“有 e2e”，这里按三层证据区分：

1. 最强证据：pytest 测试代码里直接有 `@pytest.mark.e2e_coverage(... hardware="A2/A3/A5" ...)`。
2. 次强证据：nightly / weekly / models 的 YAML 配置文件名、`test_name`、平台专属环境变量、关键链路字段明确指向某个平台。
3. 不能单独算 e2e 的痕迹：
   - 只在 `coverage_taxonomy.py` 里出现平台枚举；
   - 只在底层分支里有平台特化逻辑；
   - 只因为文件名是多节点；
   - 只因为模型名里出现 `A22B` / `A3B`。

对“PD 分离”的判断，优先认下面这些直接证据：

- `disaggregated_prefill:`
- `kv_role: kv_producer / kv_consumer`
- `routing.type: "disaggregated_prefill"`

对“池化”的判断，不能只拿上面这些字段代替，应该更偏 `kv_pool` 的 store / lookup / load / save 语义。

## 总表

| 平台 | 图模式 PR 级显式 pytest e2e | PD 分离 YAML / 配置证据 | 池化（kv_pool 语义）证据 | 仅有平台痕迹 | 当前结论 |
|---|---|---|---|---|---|
| A2 | 有 | 证据不足 | 证据不足 | 有 | A2 的图模式覆盖明确，但 PD 分离和池化当前都不能确认已有明确平台覆盖 |
| A3 | 有 | 有 | 不能仅凭现有 e2e 直接确认 | 有 | A3 的图模式和 PD 分离链路覆盖较明确，但池化不应直接跟着落结论 |
| A5 | 无 | 无 | 无 | 有 | A5 目前没有真正落地的图模式、PD 分离或池化 e2e |

## A2

### 1. 直接能看出来的 PR 级 A2 e2e

这些文件里直接写了 `hardware="A2"`：

- `tests/e2e/pull_request/one_card/test_qwen3_0_6b.py:30`
- `tests/e2e/pull_request/one_card/test_batch_invariant.py:101`
- `tests/e2e/pull_request/one_card/test_batch_invariant.py:219`
- `tests/e2e/pull_request/one_card/test_batch_invariant.py:419`
- `tests/e2e/pull_request/one_card/test_batch_invariant.py:470`
- `tests/e2e/pull_request/one_card/lora/test_qwen3_multi_loras.py:43`

这些能直接证明：

- A2 的 PR 级图模式 / batch invariant / LoRA 覆盖明确存在。

### 2. A2 的 PD 分离与池化证据为什么都不能直接算成立

A2 命名的多节点 YAML 确实有，例如：

- `tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-A2-dual-nodes.yaml:1`
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-A22B-A2.yaml:1`
- `tests/e2e/nightly/multi_node/internal_dp/config/Kimi-K2_5-W4A8-A2-dual-nodes.yaml:1`

但这些证据还不够，因为：

- 多节点不等于 disaggregated prefill；
- 不能因为文件名里出现 `A2` 就断定它测的是 KV producer-consumer 链路；
- 也不能因为模型名里有 `A22B` 就断定它是 A2 平台的 PD 分离或池化覆盖。

所以 A2 这里更准确的结论是：

- 图模式明确有；
- PD 分离当前证据不足，不能确认已有明确平台覆盖；
- 池化当前证据也不足，不能确认已有明确平台覆盖。

### 3. 补充：仓库中还有 7 个 PD 分离 YAML 没有平台标记

以下 YAML 虽然配置了 `disaggregated_prefill`，但没有任何 `ASCEND_A2_ENABLE` / `ASCEND_A3_ENABLE` 等平台专属环境变量，因此无法确定平台归属：

- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-disagg-pd.yaml`（含 `kv_producer` / `kv_consumer`）
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-VL-235B-disagg-pd.yaml`
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-W8A8.yaml`
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-W8A8-EPLB.yaml`（含 `kv_producer`）
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-W8A8-longseq.yaml`
- `tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-R1-W8A8-EPLB.yaml`（含 `kv_producer`）
- `tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-R1-W8A8-longseq.yaml`

这些 YAML 能证明 PD 分离链路本身存在，但由于没有平台标记，不能作为 A2 或 A3 平台的 PD 分离覆盖证据。

## A3

### 1. 直接能看出来的 PR 级 A3 e2e

这些文件里直接写了 `hardware="A3"`：

- `tests/e2e/pull_request/four_card/test_deepseek_v4.py:36`
- `tests/e2e/pull_request/four_card/test_deepseek_v4.py:90`
- `tests/e2e/pull_request/two_card/test_qwen3_30b_a3b.py:36`

但要注意：

- `tests/e2e/pull_request/two_card/test_qwen3_30b_a3b.py:27` 当前整体 `skip`。

所以 A3 的 PR 级图模式 / MoE 测试代码是明确存在的，但部分当前不是稳定有效覆盖。

### 2. A3 的 PD 分离证据较强，但池化不能直接跟着成立

A3 这边的强证据主要来自两部分叠加：

1. YAML 本身明确是解耦链路：
   - `tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-V3_2-W8A8-EP.yaml:23`
   - `tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-EP.yaml:28`
   - `tests/e2e/nightly/multi_node/external_dp/config/GLM5_1-W8A8-EP-external.yaml`

2. 配置里明确带 A3 平台变量：
   - `tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-V3_2-W8A8-EP.yaml:17`
   - `tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-EP.yaml:17`
   - `tests/e2e/nightly/multi_node/external_dp/config/GLM5_1-W8A8-EP-external.yaml:63`

常见信号包括：

- `ASCEND_A3_ENABLE`
- `ASCEND_A3_EBA_ENABLE`

所以 A3 更准确的结论是：

- 图模式明确有；
- PD 分离链路也有明确证据；
- 但池化不能仅凭这些 PD 分离证据直接下"已明确覆盖"的结论。

### 3. 补充：有 A3 标记不等于有 PD 分离

`tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml` 虽然配置了 `ASCEND_A3_EBA_ENABLE: 1`（A3 平台标记），但该 YAML 中**没有** `disaggregated_prefill` 字段，只是普通的多节点 DP 部署。这说明：

- 平台标记和 PD 分离是两个独立维度，不能互相替代；
- 只有同时满足"PD 分离链路字段 + 平台专属环境变量"的 YAML，才能作为该平台 PD 分离的 e2e 证据。

## A5

### 1. 当前没有真正的 A5 图模式 / PD 分离 / 池化链路 e2e

当前仓库状态下，没有找到：

- `hardware="A5"` 的 pytest e2e 测试；
- 明确面向 A5 的图模式 PR 用例；
- 明确面向 A5 的 disaggregated prefill / external DP YAML 配置；
- 明确面向 A5 的 `kv_pool` 语义证据。

所以当前 `tests/e2e` 里没有真正落地的 A5 图模式、PD 分离或池化链路 e2e。

### 2. 当前只剩下 A5 痕迹，不算覆盖

A5 现在主要只有两类痕迹：

1. coverage 枚举里允许 A5：
   - `tests/e2e/coverage_taxonomy.py:96`
2. 底层实现分支里有 A5 特化：
   - `tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor_metadata.py:43`

但这些不等于“有 A5 e2e 测试跑起来了”。

## 三个平台对比结论

| 维度 | A2 | A3 | A5 |
|---|---|---|---|
| 显式 `hardware=` pytest 用例 | 有 | 有 | 无 |
| 图模式 PR 级证据 | 有 | 有 | 无 |
| PD 分离明确证据 | 证据不足 | 有 | 无 |
| 池化（kv_pool 语义）明确证据 | 证据不足 | 不能仅凭现有 e2e 直接确认 | 无 |
| 仅枚举或底层分支痕迹 | 有 | 有 | 有 |
| 最终判断 | 图模式明确，PD 分离和池化暂不能确认 | 图模式明确，PD 分离明确，池化暂不直接下结论 | 暂无真正落地 e2e |

## 本次结论

- A2：明确有图模式覆盖；但 PD 分离和池化当前都不能确认已有明确平台覆盖。
- A3：明确有图模式覆盖，也明确有 PD 分离链路覆盖；但池化暂不能仅凭现有 e2e 直接确认。
- A5：当前没有真正落地的图模式、PD 分离或池化链路 e2e。

## 需要明确排除的误解

以后如果继续整理 `tests/e2e`，要明确把这两类东西分开：

1. `pull_request/one_card/pooling/*`
   - 这是 embedding / classification / scoring 的 pooling runner 测试。
   - 不是这里定义的“池化”。

2. `multi_node/internal_dp` / `multi_node/external_dp`
   - 这里才是 PD 分离 / producer-consumer 直传链路的主要测试区域。
   - 但也只有在配置里看到 `disaggregated_prefill`、`kv_role`、`routing.type` 时，才能把它算成这里说的 PD 分离。
   - 如果要进一步写成“池化”，还需要更接近 `kv_pool` 的 store / lookup / load / save 语义支撑。