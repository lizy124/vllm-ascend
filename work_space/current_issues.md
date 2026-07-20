# Nightly 池化用例缺口分析

## 一、概念区分：PD 分离 vs 池化

这是两个独立维度，之前文档混为一谈：

| 维度 | 是什么 | YAML 体现 |
|---|---|---|
| **PD 分离** | 架构拓扑：prefill 和 decode 跑在不同节点 | `disaggregated_prefill: enabled: true` 或 `routing.type: "disaggregated_prefill"` |
| **池化（KV 传输）** | KV cache 传输机制：用什么 connector、什么策略传输 | `--kv-transfer-config` 参数，指定 `kv_connector` 类型 |

**两者的关系：**
- PD 分离必然需要 KV 传输（prefill 的 KV cache 要传给 decode）
- 但池化不限于 PD 分离：`AscendStoreConnector` 在单节点就能做 KV 复用
- 同一个 PD 分离拓扑，可以搭配不同的 connector（V1 / Layerwise / Hybrid）

## 二、Nightly 多节点测试全景

nightly 多节点测试分两类：

### 2.1 无 PD 分离（纯分布式推理）

| YAML | 模型 | 节点数 | 调度位置 |
|---|---|---|---|
| `DeepSeek-V3.1-BF16.yaml` | DeepSeek-V3.1 | 2 | `a3.multi_node` (size=4) |
| `Qwen3-235B-A22B.yaml` | Qwen3-235B-A22B | 2 | `a3.double_node` |
| `GLM5_1-W8A8-A3-dual-nodes.yaml` | GLM-5.1-W8A8 | 2 | `a3.double_node` |
| `GLM5_2-W8A8-A3-dual-nodes.yaml` | GLM-5.2-W8A8 | 2 | `a3.multi_node` |
| `DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml` | DeepSeek-V3.2-W8A8 | 2 | `a3.double_node` |
| `Kimi-K2_5-W4A8-A2-dual-nodes.yaml` | Kimi-K2.5-W4A8 | 2 | `a2.multi_node` |
| `Qwen3-235B-A22B-A2.yaml` | Qwen3-235B-A22B | 2 | `a2.multi_node` |

这类测试**没有 PD 分离，也没有 kv-transfer-config**，只是多卡放不下的模型跨节点部署。

### 2.2 有 PD 分离（全部用了 MooncakeConnectorV1）

| YAML | 模型 | 节点数 | Connector |
|---|---|---|---|
| `Qwen3-235B-disagg-pd.yaml` | Qwen3-235B | 2 | `MooncakeConnectorV1` |
| `Qwen3-235B-W8A8.yaml` | Qwen3-235B-W8A8 | 2 | `MooncakeConnectorV1` |
| `Qwen3-235B-W8A8-longseq.yaml` | Qwen3-235B-W8A8 | 2 | `MooncakeConnectorV1` |
| `Qwen3-235B-W8A8-EPLB.yaml` | Qwen3-235B-W8A8 | 2 | `MooncakeConnectorV1` |
| `Qwen3-VL-235B-disagg-pd.yaml` | Qwen3-VL-235B | 2 | `MooncakeConnectorV1` |
| `DeepSeek-V3_2-W8A8-EP.yaml` | DeepSeek-V3.2-W8A8 | 4 | `MooncakeConnectorV1` |
| `DeepSeek-R1-W8A8-EPLB.yaml` | DeepSeek-R1-W8A8 | 2 | `MooncakeConnectorV1` |
| `DeepSeek-R1-W8A8-longseq.yaml` | DeepSeek-R1-W8A8 | 2 | `MooncakeConnectorV1` |
| `GLM5_1-W8A8-EP.yaml` | GLM-5.1-W8A8 | 2 | `MooncakeConnectorV1` |
| `GLM5_1-W8A8-EP-external.yaml` | GLM-5.1-W8A8 | 2 | `MooncakeConnectorV1` (external_dp) |

**结论：nightly 的 10 个 PD 分离测试，connector 全部是 `MooncakeConnectorV1`，没有其他类型。**

## 三、代码中的池化 Connector 全景

`vllm_ascend/distributed/kv_transfer/__init__.py` 注册了 4 种 KV 传输 connector：

| Connector 名称 | 实现类 | 源文件 | 与 V1 的差异 |
|---|---|---|---|
| `MooncakeConnectorV1` | `MooncakeConnector` | `kv_p2p/mooncake_connector.py` | 基准：按 request 粒度传输 |
| `MooncakeLayerwiseConnector` | `MooncakeLayerwiseConnector` | `kv_p2p/mooncake_layerwise_connector.py` | 按 layer 粒度逐层传输，延迟更低 |
| `MooncakeHybridConnector` | `MooncakeConnector` | `kv_p2p/mooncake_hybrid_connector.py` | 开启 `use_hybrid`，处理 MLA/Full Attention 混合 block size |
| `AscendStoreConnector` | `AscendStoreConnector` | `kv_pool/ascend_store/ascend_store_connector.py` | 单节点 KV pool 复用，与 PD 分离无关 |

**`MooncakeHybridConnector` 的关键差异：**虽然注册到同一个 `MooncakeConnector` 类，但 `mooncake_hybrid_connector.py#L1220` 会设置 `self.use_hybrid = True`，`#L1273` 和 `#L1620` 根据此标志走完全不同的 block 管理路径。V1 的测试中 `use_hybrid` 始终为 `False`，**覆盖不到这条路径。**

## 四、Nightly 池化覆盖 vs 缺口

### 4.1 涉及 PD 分离的 connector 覆盖

| Connector | Nightly | Weekly | 缺口 |
|---|---|---|---|
| `MooncakeConnectorV1` | 10 个 YAML | 19 个 YAML | 无 |
| `MooncakeLayerwiseConnector` | **0** | 2 个（GLM-4.7 internal_dp + DeepSeek-V3.1T external_dp） | **需补充** |
| `MooncakeHybridConnector` | **0** | 1 个（DeepSeek-V4-flash-w8a8-PD，4 节点） | **待评估** |

### 4.2 不涉及 PD 分离的 connector 覆盖

| Connector | Nightly | Weekly |
|---|---|---|
| `AscendStoreConnector` | 1 个（`qwen3-30b-acc`，`a3.multi_card` 调度） | 同左（共用） |

**`AscendStoreConnector` 已有覆盖，无需补充。**

## 五、需补充的用例

### 缺口 1：`MooncakeLayerwiseConnector` — 已补充，待合入

**为什么必须补：**
- 与 V1 走完全不同的 KV 传输路径（逐层传输 vs 请求级传输）
- 生产环境中降低首 token 延迟的关键优化
- Nightly 的 10 个 PD 分离测试全部覆盖不到

**补充内容：**
- 新增 YAML：`tests/e2e/nightly/multi_node/internal_dp/config/GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml`（从 weekly 同名文件复制）
- 注册：`nightly_config.yaml` 的 `a3.multi_node`（及 `a3-560t.multi_node`）新增条目：
  ```yaml
  - name: multi-node-glm-4.7-mooncake-layerwise
    config_file_path: GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml
    size: 2
  ```
- 模型：GLM-4.7-W8A8C8，2 节点 × 16 NPU，PD 分离 + `MooncakeLayerwiseConnector`
- 验证命令：`/nightly multi-node-glm-4.7-mooncake-layerwise`

### 缺口 2：`MooncakeHybridConnector` — 待评估

**为什么需要补：**
- `use_hybrid` 模式走不同 block 管理路径，V1 测试完全覆盖不到
- 适用于 MLA + Full Attention 混合模型（如 DeepSeek-V4）

**为什么待评估：**
- Weekly 只有 `DeepSeek-V4-flash-w8a8-PD`（4 节点 external_dp），资源消耗大
- 直接搬进 nightly 可能过重
- 建议评估是否可以用更轻量的模型或降为 2 节点

## 六、验证方式

在 PR 评论区发送：

```
/nightly multi-node-glm-4.7-mooncake-layerwise
```

CI 会检出 PR 分支代码，在 2 节点 × 16 NPU 的 A3 环境运行测试（功能验证 + 精度基准）。

## 七、改动清单

| 文件 | 改动 | 分支 |
|---|---|---|
| `tests/e2e/nightly/multi_node/internal_dp/config/GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml` | 新增（从 weekly 复制） | e2e_pool |
| `.github/workflows/configs/nightly_config.yaml` | `a3.multi_node` + `a3-560t.multi_node` 各新增 1 条 | e2e_pool |