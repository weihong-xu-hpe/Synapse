# Synapse — 双线接口与传输收敛总设计

> **文档状态**: 提案，待审批  
> **日期**: 2026-03-15  
> **适用范围**: Synapse 对外接口、Agent 集成方式、远程传输策略、sampling 高层工具  
> **本文定位**: 新的总设计文档，作为后续实现与文档整理的主依据  
> **相关旧文档**:
> - `docs/design/external-memory-skill-design.md`
> - `docs/design/mcp-sampling-high-level-tools-design.md`
> - `docs/design/http-sampling-transport-design.md`
> - `docs/design/http-sampling-transport-phases.md`

---

## 1. 执行摘要

Synapse 的对外集成方式应当收敛为**两条主线**，不再让用户在 REST、普通 HTTP MCP、SSE、stdio、sampling 之间自己拼装出一套心智模型。

新的产品级收敛结论是：

### Line A — REST + External Skills

这是 **稳定、显式、可脚本化** 的执行路径。

- 面向：强 Agent、脚本、自动化、远程服务集成
- 语义：调用方自己做语义判断，Synapse 只负责执行
- 适合：`memory-write` / `memory-lifecycle` 这类外部 skill 驱动流程
- 不支持：server 发起 sampling

### Line B — Streamable MCP + Sampling

这是 **面向 MCP host / IDE / 智能编排** 的远程主路径。

- 面向：支持 MCP sampling 的 host/client
- 语义：Synapse 负责组织候选与流程，host 负责语义判断，Synapse 继续执行
- 适合：高层 sampling 工具、远程 MCP 连接、IDE 内交互体验
- 是未来远程 MCP 集成的**目标形态**

### 补充结论

- `stdio` 继续保留，作为**本地/兼容/回退路径**，不是远程传输的长期架构中心。
- 现有 **HTTP + SSE sampling** 方案应降级为**过渡性兼容实现**，不再作为最终对外架构目标。
- **普通 REST 不承载 sampling**；需要 sampling 时，必须走 MCP 的状态化双向 transport，而不是把 REST 硬扩展成 callback 协议。

一句话总结：

> **稳定显式路径走 REST + skills，智能远程路径走 Streamable MCP + sampling；两条线共享同一个 canonical execution layer。**

---

## 2. 为什么要重做总设计

当前设计材料分别讨论了：

- 外部 skills
- sampling 高层工具
- HTTP/SSE sampling transport
- stdio 的推荐地位

每一篇单独看都成立，但放在一起时存在三个问题：

### 2.1 用户心智分裂

用户需要自己回答这些问题：

- 我到底该用 REST 还是 MCP？
- sampling 是不是 HTTP 也能做？
- SSE 是不是正式方案？
- stdio 是推荐路径，那远程连接怎么办？
- low-level contract 到底该从哪条线暴露？

这会让“接口选择”压过“记忆系统本身”的价值。

### 2.2 远程目标不够明确

现在已经可以看清：

- **远程连接是明确需求**
- **sampling 也是明确需求**
- **普通 REST 不适合承载 sampling**

因此，远程智能路径必须收敛到一种正式的 MCP remote transport 形态，而不是长期停留在“HTTP/SSE + callback 拼装”的实验结构。

### 2.3 产品层没有把“显式执行”和“语义编排”分开

Synapse 的核心优势之一，是它一直保留：

- 低层执行合同稳定
- 高层语义判断可替换

新的总设计必须把这件事表达得更清楚：

> **不是所有接口都要既显式又智能。应该让两条线各司其职。**

---

## 3. 新的顶层原则

## 3.1 一个 canonical execution layer

无论调用方走哪条线，最终都必须收敛到同一个底层执行合同：

- `search_memory`
- `search_existing_nodes`
- `get_node`
- `integrate_knowledge`
- `update_node_status`

这层是真实的系统语义地基。

它必须保持：

- 显式
- 可审计
- 可回放
- 不依赖 host 是否有模型能力

## 3.2 两条公开产品线，不再增加第三条主路径

对外只保留两条主线：

1. **REST + external skills**
2. **Streamable MCP + sampling**

其他 transport/形态只作为：

- 兼容层
- 过渡层
- 本地回退层

而不是继续作为并列主产品叙事。

## 3.3 sampling 是编排增强，不是新存储语义

sampling 高层工具永远只能：

1. 组织候选
2. 向 host 请求判断
3. 将判断编译为显式动作
4. 调用 canonical execution layer

它不能发明第二套写入动作，也不能取代显式写入合同。

## 3.4 远程 first-class 路径必须是状态化 MCP transport

远程连接如果需要 sampling，就必须具备：

- 会话
- 双向消息
- capability 协商
- server → client 的 sampling request
- client → server 的 sampling response

因此远程智能路径应当明确收敛到 **Streamable MCP**，而不是普通 REST。

## 3.5 文档必须区分三种状态

以后所有文档都必须明确区分：

1. **server capability** — Synapse server 是否实现了这项能力
2. **transport capability** — 某种 transport 是否能表达这项能力
3. **host compatibility** — 某个真实 IDE/host/client 是否真的配合完成这项能力

这三者不能再混写。

---

## 4. 收敛后的双线架构

## 4.1 Line A — REST + External Skills

这是 Synapse 的**稳定显式路径**。

### 定位

- 面向强 Agent、脚本、服务编排器、远程自动化
- 面向“调用方自己做语义判断”的工作模式
- 作为无需 host sampling 支持时的长期稳定方案

### 职责边界

外部 Agent / Skill 负责：

- 是否值得写
- `note` / `memory` tier 选择
- 相似候选检索与阅读
- `create | supersede | complement` 决策
- lifecycle 语义治理决策

Synapse 负责：

- Markdown I/O
- SQLite 索引同步
- 状态流转
- 图边维护
- lifecycle 机械动作执行

### 暴露面

REST 继续暴露 canonical execution layer：

- 搜索
- 获取节点
- 显式写入
- 显式更新状态

### 为什么这条线必须保留

因为它是：

- 最透明的路径
- 最适合调试与脚本化的路径
- 最不依赖宿主能力的路径
- 远程服务场景下最可靠的 fallback

### 这条线的产品承诺

> **只要你能发普通 HTTP 请求，并愿意自己做判断，你就能稳定使用 Synapse。**

---

## 4.2 Line B — Streamable MCP + Sampling

这是 Synapse 的**远程智能编排主线**。

### 定位

- 面向 IDE host、MCP client、交互式 agent 环境
- 面向“host 有 LLM 能力，Synapse 负责流程组织”的工作模式
- 面向远程连接而非本地子进程绑定

### 职责边界

Synapse 负责：

- 组织 deterministic context
- 检索候选与必要证据
- 发起 sampling 请求
- 校验 host 返回结构
- 编译为显式低层动作
- 执行 canonical execution layer

Host / Client 负责：

- 提供 sampling handler
- 使用自己的 LLM 做语义判断
- 返回结构化 JSON 决策

### 这条线为什么必须是 Streamable MCP

因为它需要的不是“远程调用工具”这么简单，而是完整的 MCP 会话能力：

- capabilities negotiation
- 双向消息语义
- sampling/createMessage
- tool result resume
- transport-managed session lifecycle

这正是 Streamable MCP 这类 transport 的定位。

### 这条线的产品承诺

> **如果 host 支持 Streamable MCP + sampling，Synapse 可以把高层记忆编排流程远程化，而不放弃 canonical execution layer。**

---

## 5. 传输策略收敛

## 5.1 正式保留的三种传输角色

| 形态 | 角色 | sampling | 远程适配 | 推荐状态 |
|---|---|---:|---:|---|
| REST | 显式执行路径 | 否 | 是 | **稳定主线** |
| Streamable MCP | 远程智能路径 | 是（取决于 host） | 是 | **目标主线** |
| stdio MCP | 本地/兼容/回退路径 | 是 | 否 | **稳定回退** |

## 5.2 降级处理的形态

### 传统 HTTP + SSE sampling

现有的 HTTP + SSE + sampling-response 模式不再作为最终产品架构表述，而应重命名为：

- **兼容实现**
- **过渡性 adapter**
- **实验性 transport bridge**

原因不是它完全错误，而是它在产品层会制造误导：

- 用户容易把它理解成“普通 HTTP 也支持 sampling”
- 实际上它本质上仍然是在手工拼出一套状态化 MCP transport 行为

所以新的总设计里，它只能是：

> **为了逐步演进到 Streamable MCP 而保留的过渡技术实现，不是北极星。**

---

## 6. 接口暴露策略

## 6.1 REST 暴露什么

REST 只暴露 low-level / explicit contract。

### 保留

- `search_memory`
- `search_existing_nodes`
- `get_node`
- `integrate_knowledge`
- `update_node_status`

### 不做

- 不在 REST 上直接暴露 sampling 高层工具
- 不把 REST 扩展成“server 发起 client 回调”的主路径

### 原则

REST 就做一件事：

> **把 Synapse 当成一个可预测、可脚本化、无隐式智能的执行服务。**

## 6.2 Streamable MCP 暴露什么

Streamable MCP 应成为**agent-facing 智能路径**。

### 默认 public surface

建议默认暴露：

- `search_memory`
- `get_node`
- `decide_memory_write`
- `integrate_memory_with_sampling`
- `review_memory_cluster`
- `condense_memory_cluster`
- `promote_memory_candidate`

### 默认不公开的 low-level write helpers

以下低层工具不再作为默认 public MCP surface 的重点：

- `integrate_knowledge`
- `search_existing_nodes`
- `update_node_status`

这些能力：

- 仍然存在
- 仍然是 canonical execution layer
- 但优先通过 REST + skills 使用
- 或通过 trusted/internal profile 暴露

### 为什么这样切分

因为这样两条线就很清晰：

- **REST**：显式执行
- **Streamable MCP**：高层编排

而不是让每条线都既显式又隐式、既脚本化又宿主依赖。

---

## 7. sampling 高层工具在新设计中的位置

新的总设计不推翻现有 sampling 高层工具思路，只是重新给它们找到了正确的宿主与传输位置。

### 它们仍然是正确的一层

- `decide_memory_write`
- `integrate_memory_with_sampling`
- `review_memory_cluster`
- `condense_memory_cluster`
- `promote_memory_candidate`

### 但它们应当绑定到 Streamable MCP 这条线

不是绑定到：

- 普通 REST
- 手工 callback 协议
- 模糊的“HTTP 也许可以”叙事

而是绑定到：

> **支持 sampling 的状态化 MCP transport。**

### 结果格式保持不变

继续坚持：

- `decision`
- `evidence`
- `execution`

三段式结构，因为这正是 sampling 编排可审计性的关键。

---

## 8. 为什么不能把 sampling 做成普通 REST 功能

这是本设计必须明确写死的一条边界。

### 根因

sampling 不是一次单向 API 调用，而是一次工具执行中的嵌套协作：

$$
client \rightarrow server \rightarrow client \rightarrow server
$$

server 在处理中途要“反问” client。

### REST 的问题

如果强行把它做进普通 REST，会引入：

- 异步任务状态机
- callback registration
- request correlation
- client availability 检测
- retry / timeout / replay 控制

最后会把 REST 变成一个劣化版 transport 层。

### 正确做法

REST 继续单向。

需要 server-initiated reasoning 时，走 MCP 的状态化 transport。

---

## 9. 为什么目标必须从 SSE 收敛到 Streamable

SSE 本身不是问题；问题在于它不是你想长期对外讲述的主架构语言。

### SSE 适合什么

- 兼容已有实现
- 快速验证 sampling 闭环
- 作为过渡 adapter

### SSE 不适合什么

- 作为未来产品架构的中心叙事
- 承担“正式远程 MCP transport”这个角色
- 让用户以为这是普通 HTTP 的自然延长

### Streamable 的价值

把关注点从“我们内部用 SSE 还是别的”转回到正确层级：

- 这是一个**远程 MCP transport**
- 它天然服务于状态化、可恢复、能力协商的交互
- 它更符合主流客户端生态对“远程生产 transport”的认知

因此本设计的原则不是“明天必须删掉 SSE”，而是：

> **以后所有设计和文档都以 Streamable MCP 为北极星，SSE 只作为中间实现细节或兼容层存在。**

---

## 10. stdio 在新设计中的重新定位

stdio 不应被删除，但必须重新定位。

### 保留原因

- 本地开发最简单
- 对 sampling host 兼容性最好
- 对测试最友好
- 仍然是没有远程需求时的最佳 fallback

### 不再承担的角色

- 不再作为远程智能架构的中心叙事
- 不再作为“未来长期唯一推荐路径”的产品结论

### 新定位

> **stdio 是本地稳定回退路径；Streamable MCP 是远程目标路径。**

---

## 11. 安全与审计原则

新的总设计要求把安全边界直接写进架构，而不是等实现阶段再补。

## 11.1 REST 线

REST 线强调：

- stateless auth
- 显式请求
- 无 server-initiated callback
- 易于代理、审计和脚本化

## 11.2 Streamable MCP 线

Streamable MCP 线必须具备：

- auth-bound session
- sampling request 唯一 ID
- timeout 与 cancel 语义
- duplicate/late response 防护
- structured audit logs

## 11.3 日志边界

默认记录：

- tool name
- session/request id
- candidate summary
- action / outcome
- execution result

默认不完整记录：

- 整段 prompt
- 整段敏感 content

除非处于显式 debug / secure opt-in 模式。

---

## 12. Host compatibility 在新设计中的地位

新的总设计明确承认一个现实：

> **Streamable MCP 是目标 transport，不等于所有 host 今天已经兼容。**

因此后续文档必须统一使用下面这套表述：

### 可以说

- Synapse server supports Streamable-MCP-oriented sampling semantics.
- Streamable MCP is the strategic remote transport direction.
- stdio remains the stable fallback when host compatibility is missing.

### 不应提前说

- 所有 MCP host 现在都能远程 sampling
- VS Code HTTP/remote sampling 已生产就绪
- 任何支持 HTTP 的客户端都支持 sampling

这不是保守，而是避免把 server capability 误写成 host certification。

---

## 13. 对现有文档体系的整理建议

这份文档批准后，建议文档体系按下面方式重组：

### 13.1 本文成为新的总设计入口

它回答的是：

- 为什么是两条线
- 每条线做什么
- transport 为什么这样收敛
- 旧设计材料各自退到什么位置

### 13.2 旧文档改为附属文档

#### `external-memory-skill-design.md`

保留，但重命名或定位为：

- **Line A 细化设计**
- 即 REST + skills 路径的工作细则

#### `mcp-sampling-high-level-tools-design.md`

保留，但定位为：

- **Line B 工具层细化设计**

#### `http-sampling-transport-design.md`

建议降级为：

- **legacy / transitional transport note**
- 说明其价值在于验证与兼容，而不是长期北极星

#### `http-sampling-transport-phases.md`

建议被新的实现路线文档替代

因为后续阶段划分应围绕：

- REST line 保持稳定
- Streamable line 逐步落地

而不是围绕“HTTP/SSE 方案本身”展开。

---

## 14. 实施路线（审批后）

审批通过后，建议按以下顺序干活。

## Phase 1 — 文档与边界收敛

目标：

- 统一术语
- 明确两条线
- 降低旧文档冲突

产出：

- 新总设计文档生效
- 旧文档重命名/加 superseded note/改角色说明

## Phase 2 — 固化 REST 线

目标：

- 明确 REST 线是 canonical explicit path
- 文档上把 skills 与 REST 绑定清楚

产出：

- REST 使用与 external skills 的正式文档
- 低层合同的稳定性声明

## Phase 3 — Streamable MCP 传输落地

目标：

- 将远程 MCP transport 正式收敛到 Streamable 语义
- 让 sampling 高层工具运行在正确的 transport 抽象上

产出：

- Streamable transport adapter
- sampling lifecycle、session、timeout、安全语义

## Phase 4 — Host compatibility matrix

目标：

- 验证真实 host
- 区分 server support 与 host support

产出：

- compatibility matrix
- 推荐用法说明
- fallback guidance

---

## 15. 需要你审批的关键决策

这份设计稿里，真正需要你拍板的是下面几件事：

### 决策 A — 是否接受“两条主线”作为唯一公开叙事

- REST + external skills
- Streamable MCP + sampling

如果你认可，后续文档和实现都按这个框架收敛。

### 决策 B — 是否同意把现有 HTTP/SSE sampling 降级为过渡/兼容实现

也就是：

- 可以保留
- 可以继续用来验证
- 但不再作为未来主产品叙事

### 决策 C — 是否同意 REST 永不承载 sampling

如果同意，后续就不再设计任何“REST callback 模拟 sampling”的正式方案。

### 决策 D — 是否同意 Streamable MCP 成为远程智能路径的唯一目标 transport

如果同意，后续 transport 设计都围绕它收敛，而不是继续扩展平行方案。

### 决策 E — 是否同意 stdio 重新定位为稳定回退，而非远程主路径

这会影响后续文档用词与推荐策略。

---

## 16. 最终推荐

我给出的最终建议是：

1. **接受两条线收敛**：REST + skills / Streamable MCP + sampling
2. **保留一个 canonical execution layer**，不改写底层语义
3. **停止把普通 HTTP + sampling 描述成长期方向**
4. **把 Streamable MCP 明确写成远程目标 transport**
5. **把 stdio 明确写成本地稳定 fallback**

简化成一句话：

> **Synapse 对外只讲两件事：显式执行走 REST，智能远程走 Streamable MCP；其余形态全部退居兼容或回退层。**

---

*文档版本: 2026-03-15* 
