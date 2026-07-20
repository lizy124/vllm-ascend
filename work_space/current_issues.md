# Nightly 池化用例缺口分析

## 一、代码中的池化 Connector 全景

### 1.1 注册表（源码位置）

文件：`vllm_ascend/distributed/kv_transfer/__init__.py`

共注册了 **8 个 connector**，其中与 KV cache 池化/传输相关的 **4 个**：

| Connector 名称 | 注册方式 | 实现类 | 源文件 |
|---|---|---|---|
| `MooncakeConnectorV1` | `KVConnectorFactory.register_connector` | `MooncakeConnector` | `kv_p2p/mooncake_connector.py` |
| `MooncakeHybridConnector` | 同上 | `MooncakeConnector` | `kv_p2p/mooncake_hybrid_connector.py` |
| `MooncakeLayerwiseConnector` | 同上 | `MooncakeLayerwiseConnector` | `kv_p2p/mooncake_layerwise_connector.py` |
| `AscendStoreConnector` | 同上 | `AscendStoreConnector` | `kv_pool/ascend_store/ascend_store_connector.py` |

另外 4 个（`MultiConnector`, `MooncakeConnectorStoreV1`, `UCMConnector`, `LMCacheAscendConnector`, `SimpleCPUOffloadConnector`, `RecomputeCPUOffloadConnector`）不属于本次分析范围。

### 1.2 各 Connector 场景与差异

**`MooncakeConnectorV1`**
- 标准 P2P 传输，按 **request 粒度** 传输 KV cache
- PD 分离场景：prefill 节点完成整个请求后，将 KV cache 一次性传给 decode 节点
- 适用场景：标准 PD 分离，所有模型通用
- kv_role：`kv_producer`（prefill 侧）/ `kv_consumer`（decode 侧）

**`MooncakeLayerwiseConnector`**
- 按 **layer 粒度** 逐层传输 KV cache
- prefill 完成一层就传一层，decode 可以提前开始消费，降低首 token 延迟
- 同样适用 PD 分离，但需要 prefill/decode 侧都配置 dp_size、tp_size
- kv_role：`kv_producer` / `kv_consumer`
- kv_connector_extra_config 需包含 `prefill` 和 `decode` 的 dp/tp 配置

**`MooncakeHybridConnector`**
- 注册到同一个 `MooncakeConnector` 类，但通过 `use_hybrid` 参数走不同路径
- 源码证据：`mooncake_hybrid_connector.py#L1220` 设置 `self.use_hybrid`，`#L1273` 根据 `use_hybrid` 决定 block 管理方式
- 适用于 **MLA + Full Attention 混合模型**（如 DeepSeek-V4），不同 attention 类型使用不同的 block size
- 在 nightly 已有的 10+ 个 `MooncakeConnectorV1` 测试中，`use_hybrid` 始终为 `False`，因此这些测试**完全覆盖不到 hybrid 路径**

**`AscendStoreConnector`**
- 单节点场景，将 KV cache 存入共享的 KV pool
- 后续请求可复用，减少重复计算
- kv_role：`kv_both`（既是 producer 也是 consumer）
- kv_connector_extra_config 中的 `register_buffer` 为 `True`

## 二、Nightly 现有池化覆盖

### 2.1 多节点 PD 分离（MooncakeConnectorV1）

nightly 共有 **11 个 YAML 文件** 包含 `--kv-transfer-config`，**全部使用 `MooncakeConnectorV1`**：

#### internal_dp（10 个 YAML）

| YAML 文件 | 模型 | kv_transfer 配置 |
|---|---|---|
| `Qwen3-235B-disagg-pd.yaml` | Qwen3-235B | `MooncakeConnectorV1`, kv_producer/kv_consumer |
| `Qwen3-235B-W8A8.yaml` | Qwen3-235B-W8A8 | `MooncakeConnectorV1`, kv_producer/kv_consumer |
| `Qwen3-235B-W8A8-longseq.yaml` | Qwen3-235B-W8A8 | `MooncakeConnectorV1`, kv_producer/kv_consumer |
| `Qwen3-235B-W8A8-EPLB.yaml` | Qwen3-235B-W8A8 | `MooncakeConnectorV1`, kv_producer/kv_consumer |
| `Qwen3-VL-235B-disagg-pd.yaml` | Qwen3-VL-235B | `MooncakeConnectorV1`, kv_producer/kv_consumer |
| `DeepSeek-V3_2-W8A8-EP.yaml` | DeepSeek-V3.2-W8A8 | `MooncakeConnectorV1`, kv_producer/kv_consumer |
| `DeepSeek-R1-W8A8-EPLB.yaml` | DeepSeek-R1-W8A8 | `MooncakeConnectorV1`, kv_producer/kv_consumer |
| `DeepSeek-R1-W8A8-longseq.yaml` | DeepSeek-R1-W8A8 | `MooncakeConnectorV1`, kv_producer/kv_consumer |
| `DeepSeek-V3.1-BF16.yaml` | DeepSeek-V3.1 | `MooncakeConnectorV1`, kv_producer/kv_consumer |
| `GLM5_1-W8A8-EP.yaml` | GLM-5.1-W8A8 | `MooncakeConnectorV1`, kv_producer/kv_consumer |

#### external_dp（1 个 YAML）

| YAML 文件 | 模型 | kv_transfer 配置 |
|---|---|---|
| `GLM5_1-W8A8-EP-external.yaml` | GLM-5.1-W8A8 | `MooncakeConnectorV1`, kv_producer/kv_consumer |

### 2.2 单节点精度测试（AscendStoreConnector）

| 测试 | 位置 | 池化方式 |
|---|---|---|
| `qwen3-30b-acc` | `tests/e2e/weekly/single_node/models/test_qwen3_30b_acc.py` | `AscendStoreConnector`, `kv_role: kv_both` |

这个测试虽然在 weekly 目录下，但由 nightly 的 `a3.multi_card` 调度执行。它通过 `MooncakeLauncher` 启动 Mooncake 服务，同时配置 `AscendStoreConnector` 和 `Eagle3` 推测解码。

### 2.3 nightly_config.yaml 中的调度映射

```
a3:
  multi_node (size=4):      1 个 PD 测试 (DeepSeek-V3.2-W8A8-EP)
  double_node (size=2):     8 个测试，其中 5 个含 PD 分离
  single_node (size=16):    15 个测试，无 PD 分离
  multi_card (size=2-4):    8 个测试，含 1 个 AscendStoreConnector

a3-560t:                   与 a3 完全对称（cn12-001 集群）
```

### 2.4 覆盖总结

| Connector | Nightly 覆盖 | 详情 |
|---|---|---|
| `MooncakeConnectorV1` | **已有（11 个 YAML，10+ 个 CI 条目）** | 覆盖 internal_dp + external_dp，模型包括 DeepSeek-V3.2/R1、Qwen3-235B、GLM-5.1 |
| `MooncakeLayerwiseConnector` | **缺失（0）** | 仅 weekly 有（GLM-4.7 internal_dp + DeepSeek-V3.1T external_dp） |
| `MooncakeHybridConnector` | **缺失（0）** | 仅 weekly 有（DeepSeek-V4-flash-w8a8-PD，4 节点 external_dp） |
| `AscendStoreConnector` | **已有（1 个 pytest）** | `qwen3-30b-acc`，通过 `a3.multi_card` 调度 |

## 三、Weekly 池化覆盖（对比参考）

weekly 的池化覆盖比 nightly 更广：

| Connector | Weekly 覆盖 | 文件 |
|---|---|---|
| `MooncakeConnectorV1` | 17 个 external_dp + 3 个 internal_dp YAML | 几乎所有 weekly 多节点 PD 分离测试 |
| `MooncakeLayerwiseConnector` | 2 个 | `GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml` (internal_dp) + `DeepSeek_V3.1T_layerwise_PD.yaml` (external_dp) |
| `MooncakeHybridConnector` | 1 个 | `DeepSeek-V4-flash-w8a8-PD.yaml` (external_dp, 4 节点) |
| `AscendStoreConnector` | 1 个 | `test_qwen3_30b_acc.py`（与 nightly 共用） |

## 四、真正的缺口

### 缺口 1：`MooncakeLayerwiseConnector` — 已补充，待合入

**为什么需要补充：**
- `MooncakeLayerwiseConnector` 与 `MooncakeConnectorV1` 走完全不同的 KV 传输路径（逐层传输 vs 请求级传输）
- nightly 的 11 个 `MooncakeConnectorV1` 测试覆盖不到 layerwise 路径
- 这个 connector 是生产环境中降低首 token 延迟的关键优化

**补充的用例：**

YAML 文件：`tests/e2e/nightly/multi_node/internal_dp/config/GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml`
（从 weekly 的同名文件复制，内容不变）

```yaml
test_name: "test GLM-4.7-W8A8C8 PD separation with mooncake layerwise connector"
model: "vllm-ascend/GLM-4.7-W8A8C8"
num_nodes: 2
npu_per_node: 16

disaggregated_prefill:
  enabled: true
  prefiller_host_index: [0]
  decoder_host_index: [1]

deployment:
  - envs: ...
    server_cmd: >
        --kv-transfer-config
        '{"kv_connector": "MooncakeLayerwiseConnector",
          "kv_role": "kv_producer",
          "kv_port": "30000",
          "kv_connector_extra_config": {
              "prefill": {"dp_size": 2, "tp_size": 8},
              "decode": {"dp_size": 2, "tp_size": 8}
          }}'
  - envs: ...
    server_cmd: >
        --kv-transfer-config
        '{"kv_connector": "MooncakeLayerwiseConnector",
          "kv_role": "kv_consumer",
          "kv_port": "30200",
          "kv_connector_extra_config": {
              "prefill": {"dp_size": 2, "tp_size": 8},
              "decode": {"dp_size": 2, "tp_size": 8}
          }}'

benchmarks:
  acc:
    baseline: 95
    threshold: 10
```

**注册方式**（在 `nightly_config.yaml` 的 `a3.multi_node.test_config` 下新增）：

```yaml
a3:
  multi_node:
    test_config:
      - name: multi-node-glm-4.7-mooncake-layerwise
        config_file_path: GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml
        size: 2
```

同样需要在 `a3-560t.multi_node` 下同步添加。

### 缺口 2：`MooncakeHybridConnector` — 待评估

**为什么需要补充：**
- `MooncakeHybridConnector` 注册到同一个 `MooncakeConnector` 类，但启用 `use_hybrid` 模式
- 源码 `mooncake_hybrid_connector.py#L1220`：
  ```python
  self.use_hybrid = (
      self._is_mla_model and parent_connector.hybrid_kv_cache_enabled
  )
  ```
- 当 `use_hybrid=True` 时，走不同的 block 管理路径（`#L1273`、`#L1620`）
- nightly 的 11 个 `MooncakeConnectorV1` 测试中，`use_hybrid` 始终为 `False`，**完全覆盖不到 hybrid 路径**

**为什么待评估：**
- weekly 的 `DeepSeek-V4-flash-w8a8-PD` 是 **4 节点 external_dp** 测试，资源消耗大
- 直接搬进 nightly 可能过重（nightly 每天跑，4 节点 × 16 NPU 成本高）
- 建议评估是否可以用更轻量的模型覆盖 hybrid 路径，或降级为 2 节点

## 五、合入前验证方式

新增用例不能直接合入，需要先在 PR 中验证测试用例本身没问题。

### 5.1 验证流程

```
1. 将用例改动推到 e2e_pool 分支
2. 在 GitHub 上创建 PR（目标分支 main）
3. 在 PR 评论区发送 /nightly 命令触发测试
4. CI 自动检出 PR 分支代码，在 A3 平台运行测试
5. 测试通过后，说明用例本身没问题，可以合入
```

### 5.2 触发命令

```
/nightly multi-node-glm-4.7-mooncake-layerwise
```

这条命令会：
- 校验你有 triage+ 权限
- 读取 `nightly_config.yaml`，找到 `name: "multi-node-glm-4.7-mooncake-layerwise"` 对应的配置
- 自动分发到 2 节点 × 16 NPU 的 A3 环境
- 检出 PR 分支代码，运行 GLM-4.7 + MooncakeLayerwiseConnector 的 PD 分离精度测试

### 5.3 测试内容

根据 `test_multi_node.py` 的逻辑，验证分为两步：
1. **功能验证**：启动 vllm serve → 发 OpenAI API 请求 → 断言返回非空
2. **基准验证**：跑 aisbench 精度基准（gsm8k-lite）→ 对比 baseline（95%）和 threshold（10%）

## 六、改动清单

| 文件 | 改动 | 分支 |
|---|---|---|
| `tests/e2e/nightly/multi_node/internal_dp/config/GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml` | 新增（从 weekly 复制） | e2e_pool |
| `.github/workflows/configs/nightly_config.yaml` | 在 `a3.multi_node` 和 `a3-560t.multi_node` 下新增注册条目 | e2e_pool |
| `work_space/` 目录 | 删除（不提交） | e2e_pool |