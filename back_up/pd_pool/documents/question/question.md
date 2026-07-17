# KV 连接器架构问题汇总

## Q1: 为什么需要 KV 连接器注册机制？

**简要回答**：
vLLM 是通用推理框架，需支持多种硬件平台。上游只定义接口规范，各厂商实现自己的连接器，通过注册机制"告诉"上游有哪些实现可用。实现"接口在上游，实现在下游"的分层架构。

---

## Q2: `register_connector(name, module_path, class_name)` 三个参数的含义？

**简要回答**：
- `name`：配置名称，用户在 YAML 中使用的字符串
- `module_path`：Python 模块路径（延迟加载）
- `class_name`：模块中的实际类名

注册时只记录路径，真正 import 发生在使用时，实现延迟加载。

---

## Q3: 为什么一个类（AscendStoreConnector）同时用于 Scheduler 和 Worker？

**简要回答**：
vLLM 的 Scheduler 进程和 Worker 进程读取**同一个配置文件**，只能配置一个连接器名称。一个类通过 `role` 参数内部分支：
- `role=SCHEDULER` → 创建 `KVPoolScheduler`
- `role=WORKER` → 创建 `KVPoolWorker`

这样同一个配置名称，两个进程都能正常工作。

---

## Q4: 既然有 KVPoolScheduler 和 KVPoolWorker，为什么还需要 AscendStoreConnector？

**简要回答**：
AscendStoreConnector 是**适配器/包装器**：
1. 满足 vLLM 接口规范（必须同时实现 Scheduler 和 Worker 方法）
2. 适配工厂模式（一个类名同时支持两种进程）
3. 组合内部组件（将 Scheduler 和 Worker 逻辑分离）
4. 代理方法调用（根据 role 决定代理到哪个内部组件）

---

## Q5: 为什么 vLLM 要求一个接口同时实现 Scheduler 和 Worker 方法？

**简要回答**：
简化用户配置和工厂设计：
- 用户只需配置一个名称，不用分别配置 Scheduler 和 Worker
- 工厂只需一个注册表，不用两个
- 避免配置匹配问题（如 Scheduler 和 Worker 配置不兼容）

设计理念：**用户友好 + 代码简洁** > **严格的接口分离**