# vllm-ascend e2e：按“池化 / PD 分离 / 图模式”口径审计 A2 / A3 / A5

范围：`tests/e2e`

## 先澄清术语

这次把“池化”和“PD 分离”彻底拆开：

- “池化”更对应 `vllm_ascend/distributed/kv_transfer/kv_pool/*` 这一支，核心语义是 store / lookup / load / save。
- “PD 分离”更对应 `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` 这一支，核心语义是 prefill 端与 decode 端之间的直连 KV 传输。
- `ascend_multi_connector.py` 是组合与编排层，不单独算“池化”或“PD 分离”。

也就是说，本文里的“池化”不是下面这些：

- `tests/e2e/pull_request/one_card/pooling/*`
- `runner="pooling"`
- `embed()` / `classify()` / `score()`

上面这些是 embedding / reranker / classification 语义下的 pooling runner，不是这里讨论的 KV transfer 语义。

## 审计口径

本次只按仓库里的真实代码与配置判断，不按历史草稿、口头预期或未提交代码判断。

判定优先级：

1. 最强证据：pytest 代码里直接出现 `@pytest.mark.e2e_coverage(... hardware="A2/A3/A5" ...)`。
2. 次强证据：nightly / weekly / models 的 YAML 文件里，平台、链路类型、环境变量、`test_name` 明确指向某个平台。
3. 对“PD 分离链路”的直接证据，优先认这类配置：
   - `disaggregated_prefill:`
   - `kv_role: kv_producer / kv_consumer`
   - `routing.type: "disaggregated_prefill"`
4. 对“池化”的直接证据，应该优先认 `kv_pool` 语义，也就是 store / lookup / load / save 这一类实现与配置；不能把所有 producer-consumer 链路直接等同成池化。
5. 不能单独算“该平台已有池化或 PD 分离 e2e”的情况：
   - 只因为是多节点 YAML；
   - 只因为文件名里出现 `A2` / `A3`；
   - 只因为模型名里有 `A22B` / `A3B`；
   - 只在 `coverage_taxonomy.py` 里有平台枚举。

特别说明：

- `Qwen3-235B-A22B` 里的 `A22B` 是模型名，不等于平台 A2。
- `Qwen3-30B-A3B` 里的 `A3B` 也是模型名，不单独等于平台 A3。
- `pull_request/one_card/pooling/*` 只能说明 embedding/classification/score 的 pooling runner 存在，不能说明你这里定义的“池化”存在。

## 结论摘要

按现在更细的口径，当前 `tests/e2e` 的结论应拆成三类看：

- A2：图模式明确有；PD 分离链路当前没有足够强的 A2 平台证据；池化也没有足够强的 A2 直接证据。
- A3：图模式明确有；PD 分离链路证据较强；但不能因为 PD 分离证据存在，就直接把池化也写成已明确覆盖。
- A5：图模式没有；PD 分离没有；池化也没有。

那么当前更准确的结论表是：

| 平台 | 池化（kv_pool 语义） | PD 分离（kv_p2p / disaggregated_prefill 语义） | 图模式 | 结论 |
|---|---|---|---|---|
| A2 | 证据不足，不能确认已明确覆盖 | 证据不足，不能确认已明确覆盖 | 明确有 | A2 目前主要能明确确认图模式 |
| A3 | 不能仅凭现有 e2e 直接确认 | 明确有 | 明确有 | A3 的图模式和 PD 分离证据较强，但池化不应直接跟着落结论 |
| A5 | 无 | 无 | 无 | A5 暂无真正落地 e2e |

## 1. 代码里哪些位置分别对应“池化”和“PD 分离”

### 1.1 PD 分离入口

internal DP 入口：

- `tests/e2e/nightly/multi_node/internal_dp/scripts/test_multi_node.py`

external DP 入口：

- `tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py`

这两类目录更直接对应本文里的“PD 分离链路”入口。

### 1.2 判定 PD 分离要看的关键字段

internal DP 型链路，重点看：

- `disaggregated_prefill:`
- `kv_role: kv_producer`
- `kv_role: kv_consumer`

external DP 型链路，重点看：

- `routing:`
- `type: "disaggregated_prefill"`

只出现“多节点”还不够；必须看到上面这些链路字段，才能说它测的是 PD 分离 / producer-consumer 直传链路。

### 1.3 判定池化要看的关键语义

池化不能再直接用“disaggregated_prefill 字段出现了”来替代判断。

更准确地说，池化应对应：

- `kv_pool` 这一支的 store / lookup / load / save 语义；
- 带有 pool / store / lookup server / load spec 一类机制的链路；
- 也就是“先存到池里，再按 key 命中并加载”的路径。

因此，PD 分离和池化虽然同属 KV transfer 大类，但在文档里不应再混成一个判断口径。

## 2. A2 审计结果

### 2.1 A2 的图模式证据是明确的

A2 的显式 PR 级 pytest 证据包括：

- `tests/e2e/pull_request/one_card/test_qwen3_0_6b.py:30`
- `tests/e2e/pull_request/one_card/test_batch_invariant.py:101`
- `tests/e2e/pull_request/one_card/test_batch_invariant.py:219`
- `tests/e2e/pull_request/one_card/test_batch_invariant.py:419`
- `tests/e2e/pull_request/one_card/test_batch_invariant.py:470`
- `tests/e2e/pull_request/one_card/lora/test_qwen3_multi_loras.py:43`

这里能直接看出 `hardware="A2"`，所以 A2 的图模式 / batch invariant / LoRA 等 PR 级覆盖是明确存在的。

### 2.2 A2 的 PD 分离与池化都不能轻易下结论

当前仓库里，确实有 A2 命名的多节点 YAML，例如：

- `tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-A2-dual-nodes.yaml:1`
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-A22B-A2.yaml:1`
- `tests/e2e/nightly/multi_node/internal_dp/config/Kimi-K2_5-W4A8-A2-dual-nodes.yaml:1`

但这些文件名本身只能说明：

- 有 A2 命名的多节点场景；
- 不能自动说明它就是 disaggregated prefill / KV producer-consumer 直传链路；
- 也不能因为模型名里有 `A22B` 就把它算成 A2 平台的 PD 分离或池化覆盖。

当前更稳妥的表述应是：

- A2 有多节点配置；
- 但就 PD 分离而言，当前没有足够强的 A2 直接证据；
- 就池化而言，也没有足够强的 `kv_pool` 语义证据；
- 所以不能写成"A2 的池化 e2e 已明确存在"，也不能直接写成"A2 的 PD 分离 e2e 已明确存在"。

### 2.3 补充：仓库中还有 7 个 PD 分离 YAML 没有平台标记

以下 YAML 虽然配置了 `disaggregated_prefill`，但没有任何 `ASCEND_A2_ENABLE` / `ASCEND_A3_ENABLE` 等平台专属环境变量，因此无法确定平台归属：

- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-disagg-pd.yaml`（含 `kv_producer` / `kv_consumer`）
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-VL-235B-disagg-pd.yaml`
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-W8A8.yaml`
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-W8A8-EPLB.yaml`（含 `kv_producer`）
- `tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-W8A8-longseq.yaml`
- `tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-R1-W8A8-EPLB.yaml`（含 `kv_producer`）
- `tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-R1-W8A8-longseq.yaml`

这些 YAML 能证明 PD 分离链路本身存在，但由于没有平台标记，不能作为 A2 或 A3 平台的 PD 分离覆盖证据。

## 3. A3 审计结果

### 3.1 A3 的图模式证据是明确的

A3 的显式 PR 级 pytest 证据包括：

- `tests/e2e/pull_request/four_card/test_deepseek_v4.py:36`
- `tests/e2e/pull_request/four_card/test_deepseek_v4.py:90`
- `tests/e2e/pull_request/two_card/test_qwen3_30b_a3b.py:36`

其中：

- `test_deepseek_v4.py` 可直接作为 A3 图模式 / MoE 证据。
- `test_qwen3_30b_a3b.py` 虽然有 `hardware="A3"`，但文件当前整体是 `skip`：
  - `tests/e2e/pull_request/two_card/test_qwen3_30b_a3b.py:27`

所以更准确的说法是：

- A3 图模式测试代码明确存在；
- 但部分 A3 PR 用例当前不是稳定有效覆盖。

### 3.2 A3 的 PD 分离证据较强，但不能直接等同成池化证据

当前能把 A3 和 PD 分离链路关联起来的强证据，主要来自：

1. 明确的 disaggregated prefill 配置，例如：
   - `tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-V3_2-W8A8-EP.yaml:23`
   - `tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-EP.yaml:28`
   - `tests/e2e/nightly/multi_node/external_dp/config/GLM5_1-W8A8-EP-external.yaml`

2. 这些链路配置里带有 A3 专属环境变量，例如：
   - `tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-V3_2-W8A8-EP.yaml:17`
   - `tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-EP.yaml:17`
   - `tests/e2e/nightly/multi_node/external_dp/config/GLM5_1-W8A8-EP-external.yaml:63`

常见的 A3 平台信号包括：

- `ASCEND_A3_ENABLE`
- `ASCEND_A3_EBA_ENABLE`

因此更准确的结论是：

- A3 平台上，确实存在明确的 PD 分离 / KV producer-consumer 直传链路测试证据；
- 证据主要来自"PD 链路字段 + A3 专属平台变量"的组合；
- 但这还不应直接改写成"A3 的池化已明确覆盖"；
- 如果要下"池化已明确覆盖"的结论，还需要更直接的 `kv_pool` / store / lookup / load / save 语义证据。

### 3.3 补充：有 A3 标记不等于有 PD 分离

`tests/e2e/nightly/multi_node/internal_dp/config/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml` 虽然配置了 `ASCEND_A3_EBA_ENABLE: 1`（A3 平台标记），但该 YAML 中**没有** `disaggregated_prefill` 字段，只是普通的多节点 DP 部署。这说明：

- 平台标记和 PD 分离是两个独立维度，不能互相替代；
- 只有同时满足"PD 分离链路字段 + 平台专属环境变量"的 YAML，才能作为该平台 PD 分离的 e2e 证据。

## 4. A5 审计结果

### 4.1 A5 的图模式当前没有明确 e2e

当前没有发现：

- `hardware="A5"` 的图模式 pytest 用例；
- A5 graph 专项目录；
- A5 graph YAML 场景。

所以 A5 图模式当前没有真正落地的 e2e。

### 4.2 A5 的 PD 分离与池化当前都没有明确 e2e

当前没有发现：

- 明确 A5 平台的 `disaggregated_prefill` YAML；
- 明确 A5 平台的 `routing.type: "disaggregated_prefill"` external DP 配置；
- `hardware="A5"` 的相关 pytest 用例；
- 明确指向 `kv_pool` 语义的 A5 平台证据。

所以 A5 的 PD 分离当前没有真正落地的 e2e，池化当前也没有真正落地的 e2e。

### 4.3 A5 当前只剩平台痕迹，不算覆盖

A5 目前主要只剩两类痕迹：

1. coverage 枚举里允许 A5：
   - `tests/e2e/coverage_taxonomy.py:96`
2. 少量底层平台分支：
   - `tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor_metadata.py:43`

但这些都不等于真正的 A5 e2e 覆盖。

## 5. 必须从本文剔除的误解

下面这些原先容易被混写，现在必须明确分开：

### 5.1 `pull_request/one_card/pooling/*` 不是本文里的“池化”

- `tests/e2e/pull_request/one_card/pooling/test_embedding.py`
- `tests/e2e/pull_request/one_card/pooling/test_classification.py`
- `tests/e2e/pull_request/one_card/pooling/test_scoring.py`

这些文件只能说明：

- 仓库里有 embedding / classify / score 的 pooling runner 测试；
- 它们测的是表示层 pooling 能力；
- 不是节点间 KV cache 传输链路；
- 不能再拿来当你定义的“池化”证据。

### 5.2 多节点不等于 PD 分离，更不等于池化

- “multi-node” 只能说明跨节点；
- 只有当 YAML 里明确出现 `disaggregated_prefill`、`kv_role`、`routing.type` 这类字段时，才能说它在测 PD 分离 / producer-consumer 直传链路；
- 要写成“池化”，还需要更接近 `kv_pool` 的 store / lookup / load / save 语义支撑。

### 5.3 模型名不等于平台

- `A22B`、`A3B` 这类模型名片段不能直接拿来当平台结论。

## 6. 最终结论

按现在修正后的口径，`work_space` 里关于“池化 / PD 分离”的正确表述应统一为：

1. “池化”不再泛指所有 KV cache 传输 / 解耦推理链路，而应更偏 `kv_pool` 的 store / lookup / load / save 语义。
2. “PD 分离”对应 `disaggregated_prefill` / producer-consumer / `routing.type` 这类直传链路证据。
3. `pull_request/one_card/pooling/*` 不属于这里的“池化”，应从结论中剔除。
4. 当前 `tests/e2e` 里：
   - A2：图模式明确有；PD 分离证据不足；池化证据也不足。
   - A3：图模式明确有；PD 分离证据较强；池化暂不能仅凭现有 e2e 直接确认。
   - A5：图模式没有；PD 分离没有；池化也没有。

如果后面你继续补文档或补测试，建议继续沿用这套拆分口径，不再把 `runner="pooling"`、PD 分离链路、`kv_pool` 语义混成一个词。