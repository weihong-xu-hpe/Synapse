# Synapse — MCP Sampling 高层工具分层设计

> **文档状态**: 草稿，待审阅  
> **日期**: 2026-03-15  
> **适用范围**: Synapse MCP 接口层、外部 Agent 集成层、Lifecycle 维护工作流  
> **相关文档**:  
> - `docs/design/external-memory-skill-design.md`  
> - `agentic-hybrid-memory-architecture-design.md`  
> - `docs/design/TODO-14-code-architecture-audit.md`

---

## 1. 背景与问题定义

当前 Synapse 已经形成一条清晰的写入路径：

- 外部 Agent 负责理解上下文、生成记忆草稿、检索相似节点、做出 `create | supersede | complement` 决策
- Synapse 只负责执行写入、状态流转、索引同步与机械生命周期管理

这条路径通过外部 Skills（如 `memory-write`、`memory-lifecycle`）来约束调用方行为，具备以下优点：

- 职责边界清晰
- Synapse 保持“零智能、纯执行层”定位
- 低层接口显式、稳定、可脚本化
- 不依赖特定 MCP Host 的附加能力

但该设计也有现实约束：

- 调用方必须**正确加载并遵守**相应 Skill
- 弱 Agent / 弱 Host / 简单自动化脚本难以稳定复用整套语义判断流程
- Lifecycle 场景天然更像系统级维护流程，而不是当前对话 Agent 的即时思考

因此需要评估一种增强方案：

> 在保留现有 low-level 显式写入合同不变的前提下，增加一组仅在 MCP transport 上暴露的、依赖 sampling 的高层工具（high-level tools / workflows），用于封装语义判断与流程编排。

---

## 2. 核心结论

**本设计不使用 sampling 替代现有显式写入接口，而是在其之上新增一层 sampling-powered 高层工具。**

换句话说：

- `integrate_knowledge` 等现有接口仍然是**唯一 canonical write contract**
- sampling 层只负责：
  1. 组织候选检索
  2. 请求 MCP Client / Host 使用其侧 LLM 进行判断
  3. 将判断结果编译为底层显式参数
  4. 最终调用现有 low-level 工具执行落盘

因此：

> **sampling 层是编排层（orchestration layer），不是新的存储语义层。**

这保证了系统仍然保有一个清晰、稳定、可回放、可测试的最低执行语义。

---

## 3. 对 MCP Sampling 的精确定义

在本文语境中，MCP sampling 指：

- Synapse 作为 MCP Server，在执行某个高层工作流时
- 向 MCP Client / Host 发出一个 sampling 请求
- 由 Client / Host 使用其侧 LLM 根据给定 prompt 做语义分析
- 将结构化决策结果返回给 Synapse
- 再由 Synapse 继续执行底层工具调用

### 3.1 关键边界

Sampling **不是**：
- Synapse 本地自带模型推理能力
- Synapse 直接调用一个它自己托管的 LLM
- 普通 REST 调用的自然扩展

Sampling **是**：
- Synapse 借用调用方 Host 的模型能力，完成一次受协议约束的推理
- 一种“Server 发起、Client 执行、结果回传”的双向协作模式

### 3.2 直接推论

如果某个高层工具依赖 sampling，则：

- 支持 sampling 的 MCP Host 可正常使用
- 不支持 sampling 的普通 REST / JSON-RPC client **无法直接完成该能力**
- 该工具必须定义清晰的降级或失败语义

---

## 4. 设计目标与非目标

## 4.1 目标

1. **保留 low-level 合同稳定性**  
   不破坏现有 `search_existing_nodes` / `integrate_knowledge` 等基础工具。

2. **降低高层调用心智负担**  
   让弱 Agent 或维护类工作流无需显式手写完整 decision pipeline。

3. **将 sampling 限定为增强层**  
   只在需要语义判断的高层工具中使用，避免污染全局接口语义。

4. **优先增强 lifecycle 场景**  
   使 cluster review、archive condensation、promotion review 等治理动作更自然。

5. **保持结果可审计**  
   高层工具返回的结果必须能追溯：看到了哪些候选、做了什么判断、最终执行了什么底层动作。

## 4.2 非目标

1. **不新增第二套写入语义**  
   不引入与 `create | supersede | complement` 并列的新底层动作集合。

2. **不要求所有 MCP 工具都支持 sampling**  
   绝大多数只读或纯执行工具仍保持无 sampling 依赖。

3. **不把 sampling 设计成 REST 主路径能力**  
   若强行通过 REST 模拟 sampling 协作，将引入额外的异步回调协议复杂度，不在本设计范围内。

4. **不改变 Synapse 的零智能执行层本体**  
   真正执行写入和状态变更的仍然是既有 low-level contract；sampling 只是前置编排。

---

## 5. 三层接口模型

推荐将 Synapse 面向外部的能力划分为三层：

### 5.1 第 0 层：Canonical Execution Layer

这是唯一真实的底层执行合同。

建议持续保留如下工具：

- `search_memory`
- `search_existing_nodes`
- `get_node`
- `integrate_knowledge`
- `update_node_status`

这些工具的共同特点：

- 不要求 sampling
- 输入参数显式
- 输出结果稳定
- 可用于 REST、MCP、测试、脚本、回放、审计
- 不依赖调用方是否具备模型能力

> 这层相当于 Synapse 的“指令集（ISA）”。

### 5.2 第 1 层：Sampling-Powered Semantic Tools

这是本设计新增的增强层。

它们通过：

1. 调用第 0 层工具检索候选
2. 请求 Client / Host 通过 sampling 完成语义判断
3. 将判断结果编译为第 0 层显式参数
4. 视需要继续调用第 0 层工具执行写入

这些工具只建议通过 MCP 暴露，不作为普通 REST 主接口。

### 5.3 第 2 层：Lifecycle / Maintenance Workflows

这层不是单一工具，而是长流程工作流，例如：

- archive backlog condensation
- disputed cluster review
- stale note promotion review
- periodic memory hygiene pass

这些工作流内部可复用第 1 层 sampling 工具，但最终仍然必须回到第 0 层执行合同。

---

## 6. 为什么 sampling 高层接口不应替代 low-level 合同

如果直接将写入主路径改为 sampling 驱动，会引入以下问题：

1. **Host 兼容性下降**  
   不支持 sampling 的 client 将无法完整使用关键写入能力。

2. **测试复杂度上升**  
   系统测试将不再只是工具级测试，还需要覆盖 sampling prompt 组织、结构化解析、host 能力协商等行为。

3. **可审计性变差**  
   若不强制回落到显式 contract，写入决策将被隐藏在一次黑箱式 sampling 返回中。

4. **职责边界变模糊**  
   Synapse 会从“纯执行层”滑向“语义裁决与执行的混合层”。

因此必须坚持一个原则：

> **Sampling interfaces should compile down to the existing explicit write contract, never replace it.**

---

## 7. 推荐新增的 Sampling 高层工具

本节给出建议的第一批高层工具集合。

### 7.1 `decide_memory_write`

**用途**：给出记忆草稿后，由 Synapse 组织候选检索，并通过 sampling 请求 Host LLM 输出显式写入决策，但**不立即执行写入**。

#### 输入

- `title`
- `content`
- `tier`
- 可选：`type`
- 可选：`sensitivity`
- 可选：`query_hint`
- 可选：`similarity_threshold`

#### 内部流程

1. 提炼 query（可由 server 做确定性摘要，也可直接使用 `query_hint`）
2. 调用 `search_existing_nodes`
3. 必要时调用 `get_node` 拉取候选全文
4. 发起 sampling，请 Host LLM 输出结构化决策：
   - `action`
   - `target_node_ids`
   - `reasoning`
   - 可选：`confidence`
5. 返回 decision，不调用 `integrate_knowledge`

#### 适用场景

- 需要“先判断、后确认”的 UI / IDE workflow
- 人工审阅后再执行写入
- 调试 Skill 与 sampling 行为差异

---

### 7.2 `integrate_memory_with_sampling`

**用途**：一步完成“候选检索 → sampling 决策 → 调用底层写入”。

#### 输入

- `title`
- `content`
- `tier`
- 可选：`type`
- 可选：`sensitivity`
- 可选：`query_hint`
- 可选：`allow_default_create_fallback`
- 可选：`require_confident_decision`

#### 内部流程

1. 调用 `decide_memory_write`
2. 如 decision 满足执行条件，则调用 `integrate_knowledge`
3. 返回：
   - decision
   - evidence
   - execution result

#### 适用场景

- 弱 Agent
- 希望用一条高层工具完成安全写入的调用方
- IDE 中的“smart save memory”体验

---

### 7.3 `review_memory_cluster`

**用途**：对一组候选节点进行 cluster 级别的语义审阅，输出治理建议或执行计划。

#### 输入

- `node_ids` 或 `query`
- 可选：`cluster_type`（如 `disputed`, `archive`, `stale_notes`, `overlap`）
- 可选：`mode`（`plan_only` / `execute_safe_actions`）

#### 可能输出

- `no_op`
- `recommend_manual_review`
- `promote_note_to_memory`
- `create_summary_node`
- `supersede_node`
- `complement_nodes`

#### 设计说明

此工具更偏 lifecycle，用于系统级语义治理，不建议替代普通单条写入。

---

### 7.4 `condense_memory_cluster`

**用途**：针对 archive / stale cluster 生成高层 summary draft，并根据 policy 决定是否直接写入。

#### 输入

- `node_ids`
- 可选：`target_tier`
- 可选：`plan_only`

#### 内部流程

1. 聚合节点
2. sampling 让 Host LLM 判断：
   - 是否值得总结
   - 总结后的 retrieval 价值是否更高
   - 是否存在 unresolved contradiction
3. 若可生成 summary，则得到 draft
4. 将 summary draft 交给 `decide_memory_write` 或 `integrate_memory_with_sampling`

#### 特点

这是一个“sampling + existing write contract” 的二段式工具：

- 第一段：产出 summary draft
- 第二段：仍用普通写入决策协议落库

---

### 7.5 `promote_memory_candidate`

**用途**：对现有 `note` 做“是否提升为 `memory`”的高层审核。

#### 输入

- `node_id`
- 可选：最近访问统计 / retrieval hit 统计
- 可选：`plan_only`

#### 输出

- `promote`
- `keep_as_note`
- `rewrite_then_promote`
- `manual_review`

#### 说明

该工具适合周期性 lifecycle pass，不建议在普通写入路径中滥用。

---

## 8. 不建议暴露为 Sampling 高层工具的能力

以下能力不值得做成 sampling 工具：

### 8.1 `search_memory`

原因：
- 已经是确定性检索工具
- sampling 不会提升其语义边界，反而会把“读取路径”复杂化

### 8.2 `get_node`

原因：
- 纯读取，无需 LLM 参与

### 8.3 `update_node_status`

原因：
- 这是显式修正工具
- 应保持手术刀属性，不应引入 sampling 黑箱

---

## 9. MCP 与 REST 的暴露策略

### 9.1 REST 保持 low-level only

REST API 继续暴露：

- 搜索
- 获取节点
- 显式写入
- 显式更新状态

理由：

- REST 调用语义是单向请求，不适合表达“server 反向请求 client 做 sampling”
- 若用 REST 模拟 sampling，将不得不引入异步任务、回调地址、状态机和会话追踪
- 这会把简单系统演化成一套新的 orchestration protocol

### 9.2 MCP 暴露 low-level + high-level

MCP transport 可同时暴露：

- 第 0 层 low-level 工具
- 第 1 层 sampling-powered 高层工具

前提是 Host 支持 sampling。

### 9.3 Host 能力协商

高层工具执行前应明确检测：

- 当前 MCP Host 是否支持 sampling
- 是否支持结构化输出约束
- 是否支持所需上下文长度

若不支持，则返回明确错误，而不是静默降级为不可解释行为。

---

## 10. 返回结构设计

高层 sampling 工具的返回值不应只包含“最终写入结果”，而应至少包含三类信息：

### 10.1 `decision`

用于记录模型裁决结果：

```json
{
  "action": "create",
  "target_node_ids": [],
  "reasoning": "未发现足够高相似且需替代的 active 节点",
  "confidence": 0.82
}
```

### 10.2 `evidence`

用于说明决策所基于的候选和上下文：

```json
{
  "query": "关于 sampling 高层写入接口分层的设计决策",
  "candidate_count": 3,
  "candidates": [
    {"node_id": "...", "score": 0.88, "status": "active"}
  ],
  "fetched_full_nodes": ["node_a"]
}
```

### 10.3 `execution`

用于说明是否真正执行了底层动作：

```json
{
  "executed": true,
  "tool": "integrate_knowledge",
  "result": {
    "node_id": "mem_..."
  }
}
```

### 10.4 为什么要分三段

因为这能清晰区分：

- LLM 是怎么判断的
- 判断基于什么证据
- 最终执行了什么动作

从而使 sampling 层仍然保有较强的可审计性。

---

## 11. 错误与降级语义

sampling 高层工具必须定义清晰失败语义。

### 11.1 `SAMPLING_UNAVAILABLE`

含义：
- 当前 Host 不支持 sampling
- 或当前会话禁用了 sampling

行为：
- 不执行写入
- 返回建议：改用 low-level 接口手工编排

### 11.2 `INVALID_SAMPLING_RESPONSE`

含义：
- Host 返回的内容不满足结构化 schema
- 无法解析为显式 decision

行为：
- 若配置允许，返回 `plan_only` 失败，不执行写入
- 不应自动伪造 decision

### 11.3 `LOW_CONFIDENCE_DECISION`

含义：
- sampling 返回的判断置信度过低
- 或 reasoning 明确标记无法安全裁决

推荐行为：
- 默认不执行 destructive action
- 可由 policy 决定是否退化为 `create`

### 11.4 `EXECUTION_FAILED_AFTER_DECISION`

含义：
- decision 已经形成
- 但底层 `integrate_knowledge` 或 `update_node_status` 执行失败

行为：
- 明确区分“判断阶段成功”和“执行阶段失败”
- 避免混淆为模型错误

---

## 12. 审计与可观测性

为了避免 sampling 把语义判断重新变成黑箱，建议增加专门审计结构。

### 12.1 建议记录的事件

- high-level tool invocation
- candidate retrieval summary
- sampling request metadata（不一定保存完整 prompt，可保存摘要或 hash）
- sampling response parse result
- compiled low-level action
- final execution result

### 12.2 日志目标

至少能够回答以下问题：

1. 这个高层工具是否真正用到了 sampling？
2. sampling 看到的候选有哪些？
3. 最终 decision 是什么？
4. decision 是否被执行？
5. 若失败，失败发生在判断阶段还是执行阶段？

---

## 13. 推荐的渐进式落地顺序

不建议一次性把所有 sampling 高层接口做全。推荐按以下顺序推进：

### Phase 1：能力试点

优先实现：

1. `decide_memory_write`
2. `integrate_memory_with_sampling`

目的：
- 验证 Host sampling 兼容性
- 验证结构化 decision schema
- 验证 evidence / execution 回包模型

### Phase 2：Lifecycle 增强

在 Phase 1 成功后，再实现：

3. `review_memory_cluster`
4. `promote_memory_candidate`

目的：
- 将 sampling 价值集中释放在治理类 workflow 中
- 减少对普通即时写入链路的扰动

### Phase 3：Condensation 扩展

最后评估：

5. `condense_memory_cluster`

原因：
- 该工具的 prompt 设计和输出验证最复杂
- 它会涉及 summary draft 质量和 unresolved contradiction handling
- 最适合在前两阶段稳定后再推进

---

## 14. 对现有 Skills 的影响

新增 sampling 高层工具后，外部 Skills 不会失效，反而更应被重新定位。

### 14.1 `memory-write` 的新定位

- 对强 Agent：继续作为主路径
- 对弱 Agent：可退化为“优先使用 sampling 高层工具”的行为规范

### 14.2 `memory-lifecycle` 的新定位

- 从“完全手工编排生命周期写入”
- 变成“决定何时触发 lifecycle 高层工具、何时保持 no-op 或人工复核”的治理规程

### 14.3 不应发生的事情

不应让 Skill 与 sampling 各自定义一套冲突的决策矩阵。正确做法是：

- 让 Skill 和 sampling prompt 共享同一套 references / policy files
- 保持 `create | supersede | complement` 语义一致

---

## 15. 最终推荐

Synapse 应采用如下原则：

1. **现有显式 low-level 写入接口保持不变，并继续作为唯一 canonical contract**
2. **在 MCP transport 上新增 sampling-powered 高层工具，但不将其作为 REST 主接口**
3. **sampling 高层工具只能编排和编译到底层显式动作，不得创造第二套存储语义**
4. **优先在 lifecycle / governance 场景上释放 sampling 价值，而不是一开始就替换普通写入主路径**
5. **所有高层工具必须返回 decision / evidence / execution 三段式结果，确保可审计性**

简化成一句话就是：

> **保留显式底层合同作为地基，在其之上增加 sampling 高层工作流；让 sampling 负责“帮你想”，但不负责重定义“怎么写”。**

---

## 16. 待决策问题

在正式实现前，仍需明确以下设计决策：

1. **是否要求 sampling 返回 `confidence` 字段？**  
   若要求，需要定义置信度的语义边界与阈值策略。

2. **server 是否允许在 `LOW_CONFIDENCE_DECISION` 时自动退化为 `create`？**  
   这关系到系统保守性与误写率。

3. **query 提炼是 server 内部确定性摘要，还是完全交给 Host LLM？**  
   前者更可控，后者更灵活。

4. **高层工具是否支持 `plan_only` 与 `execute` 两种模式？**  
   若支持，应尽早统一命名和返回 schema。

5. **sampling request / response 的审计粒度到什么程度？**  
   这涉及隐私、日志体积与可调试性的平衡。

---

*文档版本: 2026-03-15*