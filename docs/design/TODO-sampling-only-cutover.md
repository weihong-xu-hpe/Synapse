# Synapse TODO — Sampling 单线收口（不修改 skill 文件）

> **文档状态**: 执行 TODO
> **日期**: 2026-03-16
> **前提**: Streamable MCP + sampling 主路径已经在真实 server / synthetic client 组合下跑通
> **约束**: `memory-write` / `memory-lifecycle` skill 文件暂时**不修改、不移动、不删除**
> **目标**: 在不动 skill 文件的前提下，把 Synapse 的正式运行路径彻底收口到 sampling 高层工具，并统一读/写候选检索语义

### 当前进度（2026-03-16）

- ✅ **Phase A 已完成**：active docs 已明确 skill 保留但不是正式入口，正式 public surface 只剩 sampling 高层工具 + `search_memory` / `get_node`
- ✅ **Phase B 已完成**：读取检索与高层写前候选检索已统一到同一内部候选原语，`query_hint` 改为增强式联合召回
- ✅ **Phase C 已完成**：public MCP surface 不再暴露低层 canonical tools，运行时默认 guidance 已收口到高层工具
- 🔄 **Phase D 进行中**：回归验证已通过，历史兼容材料已改为 retired 说明；真实 host compatibility 证据仍需持续补充

---

## 1. 这份 TODO 要解决什么

当前系统已经证明：

- Streamable MCP session 可建立
- sampling request / response loop 可闭环
- `integrate_memory_with_sampling` / `review_memory_cluster` / `condense_memory_cluster` / `promote_memory_candidate` 可执行
- `create`、`complement`、lifecycle `execute_safe_actions` 已经在受控验证中跑通

但系统仍然存在两个不该继续拖着的分叉：

1. **skill 分支仍然作为潜在执行入口存在**
2. **高层写入候选检索与普通读取检索不是同一个内部原语**

这份 TODO 的任务不是再讨论方向，而是：

> **在不动 skill 文件的前提下，把产品执行路径完全收口到 sampling。**

---

## 2. 已批准的原则

本 TODO 默认以下原则已经成立：

### 2.1 正式入口只有一条

对外正式入口只保留：

- `search_memory`
- `get_node`
- `decide_memory_write`
- `integrate_memory_with_sampling`
- `review_memory_cluster`
- `condense_memory_cluster`
- `promote_memory_candidate`

### 2.2 skill 文件先保留，但不再作为执行主线

`memory-write` / `memory-lifecycle` 当前仍保留为仓库内资产，但它们的地位变为：

- 语义规则来源
- prompt 设计参考
- 审计与测试参考

它们**不是**正式 public workflow，也**不是**推荐的运行时入口。

### 2.3 不降低 target 校验安全边界

对于 `complement` / `supersede`：

- target 必须来自候选集
- 不允许 host 任意指向未提供的节点

问题要通过**提高候选召回一致性**解决，而不是通过放宽校验解决。

---

## 3. 立即收口的决策

## 3.1 暂时 block skill 执行分支，但不动 skill 文件

由于 skill 文件暂时不改，收口动作必须发生在：

- agent routing / discovery policy
- public tool surface
- active docs
- runtime guidance

### 要求

1. 运行时与文档中，不再把 skill 描述成正式执行路径
2. 外部 agent 的推荐路径统一改为高层 MCP 工具
3. 低层 write contract 不再作为默认调用建议
4. 如果仍需引用 skill，只能以“policy reference”身份引用

### 验收标准

- 活跃文档中不再出现“可以直接按 skill 走低层工具”的主叙事
- 对外推荐语只剩 sampling 高层工具主线
- skill 文件保留不动，但不再被视作产品入口

---

## 3.2 统一读取检索与写前候选检索

这是当前最重要的代码层整改项。

### 当前问题

用户可见读取路径：

- `search_memory` → `RetrievalPipeline`

高层写入候选路径：

- `decide_memory_write` / `integrate_memory_with_sampling`
- `build_memory_write_query(...)`
- `_search_existing_nodes_payload(...)`

这导致：

- 读路径能看到的相关节点
- 写前候选集不一定能看到

从而出现：

- host 做出合理 `complement` / `supersede`
- server 因 candidate set 为空而拒绝响应

### 必须达成的新原则

> **给用户读的检索语义，与给写入决策喂候选的检索语义，必须来自同一个内部候选原语。**

---

## 4. 新的检索设计

## 4.1 建立统一的内部候选检索核心

新增一层内部能力，例如：

- `retrieve_candidates(...)`
- 或 `collect_memory_candidates(...)`

它负责：

- 统一处理 query → candidate set
- 统一 lexical / vector / rerank 语义
- 为读取、写入、lifecycle 提供可复用候选集

### 输出至少包含

- `node_id`
- `title`
- `score`
- `status`
- `tier`
- `sensitivity`
- `file_path`
- `snippet`（读取场景需要）
- `match_reason`（可选，但建议加）

---

## 4.2 `search_memory` 改为同一原语的展示包装

`search_memory` 不再代表另一条检索世界，而只是：

- 调统一候选检索核心
- 追加 context / snippet / anchor 展示信息
- 以阅读友好的形式返回

换句话说：

> `search_memory` 应该是统一候选原语的“read view”，而不是独立实现。

---

## 4.3 高层写入工具复用同一候选原语

以下工具必须改为复用统一候选检索核心：

- `decide_memory_write`
- `integrate_memory_with_sampling`

### 新要求

1. sampling 前使用统一候选原语拿到 candidate set
2. 再从候选集中抽取 top full nodes 作为 evidence
3. sampling response 的 target 校验仍严格要求 target ∈ candidate_ids

这样可以同时保住：

- 安全边界
- 行为一致性
- `complement` / `supersede` 的可用性

---

## 4.4 `query_hint` 改为增强而不是覆盖

当前 `query_hint` 不应继续作为“覆盖最终 query”的机制。

### 目标语义

- `base_query = build_memory_write_query(title, content)`
- `query_hint` 只作为增强信息
- 最终候选集合来自 `base_query` 与 `query_hint` 的联合召回，而不是简单替换

### 推荐做法

1. 先用 `base_query` 检索
2. 再用 `query_hint` 检索（若存在）
3. 合并去重
4. 重新排序
5. 截取 top candidates

### 验收标准

- 提供 `query_hint` 不会意外压缩掉原本可见的候选
- 无 `query_hint` 与有 `query_hint` 的差异应表现为“增强召回”，而不是“改变世界”

---

## 4.5 放宽前置召回，严格保留后置校验

### 调整方向

- 候选召回阶段：放宽
- target 合法性校验阶段：继续严格

### 原则

候选搜索负责：

- 别漏掉真正相关节点

sampling 负责：

- 别把不该合并的节点乱合并

server 校验负责：

- 不允许越权指向未提供 target

### 禁止的错误修法

- 不能因为候选集不稳定，就取消 target 必须属于 candidate set 的约束

---

## 5. 如何在不动 skill 文件的前提下关闭 skill 分支

## 5.1 文档层关闭

active docs 必须明确：

- skill 线不是正式运行时入口
- skill 只作为 policy reference 存在
- 正式执行面只剩 sampling 高层工具

### 需要做的事

1. 在 active design 文档或新增说明中明确写出：
   - skill 文件保留，但不再代表正式 public integration path
2. 删除所有把 skill 描述成“你也可以直接这样做”的推荐语
3. 所有对外 guidance 统一指向高层 MCP 工具

---

## 5.2 agent routing 层关闭

在 agent / instruction / prompt routing 层，新增或修改规则，使其满足：

- 普通写入请求不再优先触发 `memory-write` skill
- lifecycle 请求不再优先触发 `memory-lifecycle` skill
- 如果触发 skill 相关语义，也应优先走 sampling 高层工具

### 目标

做到“skill 文件还在，但默认不会被当成执行路线”。

### 验收标准

- 常见 memory write / lifecycle 请求被路由到 MCP 高层工具，而不是低层手工编排

---

## 5.3 public surface 层关闭

低层工具不应再作为正式外部推荐面：

- `search_existing_nodes`
- `integrate_knowledge`
- `update_node_status`

### 它们的新定位

- internal canonical execution layer
- server 内部编译落点
- 测试 / 回放 / 审计地基

### 验收标准

- 面向 agent 的推荐 surface 里，不再把这些低层工具列为默认调用路径

---

## 6. 需要新增的测试

## 6.1 检索一致性测试

必须新增测试，验证：

- 同一输入 draft 下
- `search_memory` 可见的高相似节点
- 高层写入候选检索也能看见

### 至少覆盖

1. 无 hint
2. 有 hint
3. hint 偏窄
4. lexical 命中强 / vector 命中弱
5. vector 命中强 / lexical 命中弱

---

## 6.2 正向 `complement` / `supersede` 回归测试

必须新增：

- 候选已召回时，`complement` 可稳定执行
- 候选已召回时，`supersede` 可稳定执行
- 结果正确更新 links / status / superseded_by

---

## 6.3 skill 分支被关闭的行为测试

新增面向 routing / guidance 的测试或检查项，验证：

- 默认 agent workflow 不再推荐 skill 低层编排
- 默认产品说明不再把 skill 当入口
- skill 文件仍保留，不触发误删 / 误归档

---

## 7. 实施顺序

## Phase A — 先做收口声明

1. ✅ 新增本 TODO 文档
2. ✅ 在 active docs 中明确“skill 保留但非正式入口”的结论
3. ✅ 明确唯一正式入口是 sampling 高层工具

## Phase B — 统一检索核心

1. ✅ 提取统一候选检索原语
2. ✅ `search_memory` 复用该原语
3. ✅ `decide_memory_write` / `integrate_memory_with_sampling` 复用该原语
4. ✅ `query_hint` 改为增强式联合召回

## Phase C — 收口执行面

1. ✅ 对外 guidance 中移除低层工具推荐
2. ✅ 低层工具明确降为 internal canonical layer
3. ✅ agent routing / runtime guidance 不再把 skill 视为默认执行路径

## Phase D — 重建验收

1. ✅ 检索一致性测试
2. ✅ `create` / `complement` / `supersede` 端到端回归
3. ✅ lifecycle 执行回归
4. 🔄 host compatibility 记录更新

---

## 8. 完成标准

本 TODO 完成时，必须满足：

- 正式运行路径只剩 sampling 高层工具
- skill 文件仍保留，但不再是执行入口
- 读取检索与写前候选检索共享同一个内部候选原语
- `query_hint` 不再导致候选集意外缩窄
- `complement` / `supersede` 的失败不再主要来自候选漏召回
- target 必须属于 candidate set 的安全边界保持不变

---

## 9. 最终原则

这一轮收口不是“把 skill 删掉”，而是：

> **保留 skill 作为策略资产，关闭它作为执行分支的地位；保留 sampling 作为唯一正式入口，并把检索语义统一到同一个内部候选原语。**

这才叫真正转向 sampling，而不是嘴上转向 sampling。
