# Synapse — Streamable MCP 单线架构总设计

> **文档状态**: 当前有效架构  
> **日期**: 2026-03-16  
> **适用范围**: Synapse 对外接口、Agent 集成模式、远程 transport、sampling 高层工具  
> **本文定位**: 当前唯一总设计入口，后续实现与文档整理均以此为准

---

## 1. 核心结论

Synapse 应当收敛为**单线架构**：

> **只保留 Streamable MCP + Sampling 这一条正式对外路径。**

这意味着：

- `REST` 不再作为正式对外接口保留
- `stdio` 不再作为正式 transport 保留
- 现有 `HTTP + SSE + sampling-response` 过渡方案不再保留为产品形态
- 所有“兼容 fallback”叙事一律移除

项目仍保留内部的 canonical execution layer，但它不再被包装成一套长期并行的公开产品线。

一句话总结：

> **Synapse 是一个以 Streamable MCP 为唯一入口、以 sampling 为核心协作模式、以显式执行层为内部地基的单线记忆系统。**

### 1.1 当前执行状态（2026-03-16）

当前已经完成并确认的收口动作：

- public MCP surface 已收口到 `search_memory`、`get_node` 与五个 sampling 高层工具
- `integrate_knowledge`、`search_existing_nodes`、`update_node_status` 已退回 internal-only 角色
- 仓库中的 `memory-write` / `memory-lifecycle` skill 文件被保留为 policy/reference 资产，而非正式执行入口
- 高层写入候选检索已与读取侧候选检索统一到同一内部候选原语

---

## 2. 为什么这次要做单线收敛

当前项目同时存在过多并行叙事：

- REST
- stdio MCP
- HTTP/SSE MCP
- external skills
- sampling 高层工具
- host compatibility fallback

在一个早期项目里，这种“多线并行 + 兼容优先”的设计成本大于收益。

### 2.1 早期项目最怕的是方向分叉

Synapse 现在还在非常早的阶段。这个阶段最重要的不是“尽量兼容所有入口”，而是：

- 把产品核心做对
- 把唯一主路径做通
- 把坏的、半工作的、靠 fallback 维持的线路砍掉

如果系统一边说：

- REST 也可以
- stdio 也可以
- HTTP/SSE 也可以
- 远程 Streamable 以后再说

那最后很容易变成：

- 没有一条路径真正被做成 first-class
- 所有设计都在给兼容性让路
- sampling 反而沦为“文档里有、现实里要靠 fallback 才能用”的能力

### 2.2 “不工作”应当修复，而不是用 fallback 掩盖

这个原则应写进设计本身：

> **如果目标能力是远程 sampling，就应该把远程 sampling 修好，而不是通过 stdio fallback、REST fallback、SSE fallback 来掩盖主路径没有做完。**

### 2.3 sampling 决定了 transport 形态

Synapse 的高价值能力不是“能远程调用工具”这么简单，而是：

- server 在工具执行中途组织证据
- server 请求 host 进行 sampling 推理
- host 返回结构化决策
- server 恢复原始调用并执行低层动作

这类交互天然要求：

- 会话
- 双向消息
- capability negotiation
- server-initiated sampling requests
- request/response correlation

这不是普通 REST 的语义，也不是为了兼容本地子进程而设计的 stdio 的长期目标。

因此，产品线应直接收敛到最匹配这类交互的 transport：

> **Streamable MCP**

---

## 3. 新的顶层原则

## 3.1 单一正式入口原则

Synapse 只保留一个正式对外入口：

- **Streamable MCP**

不再对外维持“REST 主线”“stdio 主线”“SSE sampling 主线”等并行叙事。

## 3.2 sampling 是一等公民，不是附加能力

Synapse 的对外接口设计必须默认围绕 sampling 进行，而不是把 sampling 视为“部分 transport 才有的可选增强”。

这意味着：

- public surface 的设计要服务于 sampling workflows
- transport 设计要原生承载 sampling lifecycle
- host integration 要以 sampling-capable MCP client 为前提

## 3.3 canonical execution layer 保留，但退居内部地基

系统内部仍然保留稳定的显式执行层，用于：

- 检索
- 写入
- 状态流转
- 生命周期执行

但它的定位变成：

- **内部 canonical layer**
- **测试与实现地基**
- **高层 sampling 工具编译落点**

而不是一条与 Streamable MCP 并列的长期公开产品线。

## 3.4 不为兼容保留错误方向

凡是不符合单线目标的能力，都不应继续保留为长期正式方案：

- 不能做 sampling 的 REST，不保留
- 为了过渡硬拼出的 HTTP/SSE sampling bridge，不保留为目标形态
- 为本地子进程便利而存在的 stdio，不保留为正式 transport

## 3.5 文档和实现必须一致

新设计下，文档不再写“多条都可以，按需选择”的中庸叙事，而是明确写：

- 哪条是目标路径
- 哪些是将被移除的旧路径
- 当前未完成的地方是待修复项，不是永久 fallback 设计

---

## 4. 单线架构的系统形态

## 4.1 总体结构

新的系统结构可以概括为三层：

### A. Streamable MCP Transport Layer

对外唯一正式入口。

职责：

- 会话管理
- capability negotiation
- sampling request / response lifecycle
- 远程 host/client 对接
- auth / timeout / cancellation / audit 边界

### B. Sampling-Orchestrated Tool Layer

这是 agent-facing 的真正工具表面。

建议正式对外暴露：

- `search_memory`
- `get_node`
- `decide_memory_write`
- `integrate_memory_with_sampling`
- `review_memory_cluster`
- `condense_memory_cluster`
- `promote_memory_candidate`

职责：

- 组织候选与证据
- 生成结构化 sampling prompt
- 校验 host 返回值
- 编译为显式低层动作

### C. Internal Canonical Execution Layer

这是内部真实执行层。

建议内部保留：

- `search_existing_nodes`
- `integrate_knowledge`
- `update_node_status`

职责：

- 低层执行
- 可测试性
- 可回放性
- 作为 sampling 工具的稳定编译目标

这个层级仍然是系统语义地基，但不再作为长期公开集成主线。

---

## 5. 为什么 REST 必须移除

## 5.1 根因不是“REST 不好”，而是“REST 不匹配目标”

REST 擅长的是：

- 单向请求/响应
- stateless service interface
- 显式脚本调用

但 sampling 需要的是：

$$
client \rightarrow server \rightarrow client \rightarrow server
$$

也就是 server 在处理中途反向请求 client。

这意味着，如果继续保留 REST，就会出现两个坏结果之一：

1. REST 永远无法承载核心能力，于是成为功能残缺的平行入口
2. 为了让 REST 也“像是能做 sampling”，再造一整套 callback / state machine / async task 协议

前者会制造产品分裂，后者会制造架构污染。

因此在 0.01 阶段，最好的决策不是“继续兼容”，而是：

> **直接移除 REST，避免系统长期维护一条与目标能力不匹配的假主线。**

## 5.2 internal execution layer 仍然存在

移除 REST 不等于移除显式执行语义。

应该保留的是：

- internal service API
- internal canonical contracts
- internal testing surface

被移除的是：

- 作为正式外部协议暴露的 REST API

也就是说：

> **删除的是 transport 和产品线，不是显式语义本身。**

---

## 6. 为什么 stdio 必须移除

stdio 的价值主要来自：

- 本地开发简单
- host compatibility 通常更容易
- 不需要远程 transport

但这些优点在新的产品目标下都不再构成保留它的理由。

### 6.1 stdio 天然偏向本地子进程模型

而新的目标是：

- 远程连接
- 持久服务
- MCP over network
- host-driven sampling

stdio 会让系统一直背着一个与远程目标不一致的 transport 心智模型。

### 6.2 stdio fallback 会掩盖 Streamable 主路径的问题

如果保留 stdio，任何远程 sampling 没做通的地方，团队都很容易说：

- 先用 stdio 吧
- 反正 sampling 在 stdio 下能跑

结果就是目标路径迟迟不会被真正修好。

因此新设计应明确：

> **stdio 不是降级保留，而是直接移出正式架构。**

---

## 7. 为什么 HTTP/SSE sampling 也不应保留

现有 HTTP/SSE + sampling-response 方案的价值只在于两点：

- 它验证了 server-side HTTP sampling loop 是可实现的
- 它帮助我们理解了 session / timeout / duplicate response / close semantics

但这并不意味着它应继续作为正式对外架构存在。

### 7.1 它解决的是“验证问题”，不是“产品收敛问题”

它适合：

- prototype
- synthetic test harness
- transport semantics 验证

它不适合成为最终产品叙事，因为它会让用户误以为：

- 普通 HTTP + SSE 就是最终 MCP remote transport
- sampling 只是多挂一个 SSE 就行

### 7.2 目标应是 Streamable MCP，而不是长期保留 bridge

因此：

- HTTP/SSE sampling 作为设计参考可以保留在 archive 文档中
- 但作为正式 transport 目标应被移除

---

## 8. 对外工具表面

在单线架构中，对外表面不再围绕“不同 transport 各暴露一套工具”设计，而是直接围绕 Streamable MCP 的 agent-facing workflow 设计。

### 8.1 正式 public surface

- `search_memory`
- `get_node`
- `decide_memory_write`
- `integrate_memory_with_sampling`
- `review_memory_cluster`
- `condense_memory_cluster`
- `promote_memory_candidate`

### 8.2 internal-only execution helpers

- `search_existing_nodes`
- `integrate_knowledge`
- `update_node_status`

### 8.3 原则

对外表面必须服务于单线目标：

- 面向 sampling-capable host
- 面向远程 Streamable MCP session
- 面向高层 orchestration

而不是继续服务于手工编排或多 transport 并存。

---

## 9. 与 external skills 的关系

新的单线设计并不否认 external skill 这类提示工程/agent governance 的价值，但它改变了它们的系统地位。

### 9.1 它们不再对应一条正式 transport/product line

以前 external skills 对应的是：

- 一条显式外部判断路径
- 通常配合 REST 或低层接口使用

现在这条产品线被收掉了。

### 9.2 它们可以保留为内部 prompt discipline / sampling prompt reference

也就是说，`memory-write` / `memory-lifecycle` 的思想仍然有价值，但它们的定位应变成：

- 高层 sampling prompt 的语义参考
- host-side reasoning policy 的来源
- 测试与审计标准的一部分

而不是“正式公开的一条并行集成路径”。

所以新的系统表达应为：

> **skills remain as policy knowledge, not as a separate public integration line.**

---

## 10. 新设计下的成功标准

单线架构成立，不是因为文档删掉了 fallback，而是因为以下能力真正成立：

## 10.1 远程 Streamable session 成立

必须支持：

- session open / attach / close
- capability negotiation
- client initialized lifecycle
- request correlation

## 10.2 sampling loop 成立

必须支持：

- server 发 sampling request
- client 返回结构化 decision / plan
- server 继续原始调用

## 10.3 高层工具成立

至少要跑通：

- `integrate_memory_with_sampling`
- `review_memory_cluster`
- `promote_memory_candidate`
- `condense_memory_cluster`

## 10.4 failure semantics 成立

必须明确定义并测通：

- sampling unavailable
- invalid sampling response
- timeout
- duplicate response
- session closed
- auth mismatch

## 10.5 host compatibility 被诚实记录

即使单线目标确定了，也仍然必须区分：

- server implemented
- host verified

不允许用文档乐观主义代替真实 compatibility matrix。

---

## 11. 文档体系重整原则

这次文档重整应遵循三个原则：

### 11.1 active docs 只保留当前战略方向

active 设计文档中不再保留：

- 双线叙事
- HTTP/SSE sampling 主线叙事
- stdio 推荐叙事
- REST 公开接口主线叙事

### 11.2 旧材料进入 archive，而不是继续混在 active 区

旧文档不一定删除，但必须移出活跃设计目录，并明确标记：

- superseded
- historical
- transitional reference only

### 11.3 implementation plan 与 architecture 分离

以后 active 文档至少应分成：

1. **总设计文档** — 回答“系统最终是什么”
2. **实现拆分文档** — 回答“先做什么、后做什么”

避免把原则、现状、过渡、实现步骤混在同一份文档里。

---

## 12. 需要明确移除的东西

新设计批准后，应明确把以下内容列入移除清单：

### 对外接口层

- REST API
- stdio transport
- legacy HTTP/SSE sampling transport

### 文档层

- external-memory-skill 作为独立产品线的叙事
- dual-path / fallback 叙事
- HTTP sampling compatibility 作为长期产品说明

### 心智模型层

- “先保留多个入口，以后再慢慢收敛”
- “有问题先 fallback，主路径以后再补”

---

## 13. 已确认的关键决策

这份单线设计已经按下面四项原则落地推进：

### 决策 A — Streamable MCP 是唯一正式对外路径

后续不再把 REST / stdio / SSE sampling 叙事当成正式产品线。

### 决策 B — sampling 是系统默认能力，而非 transport 附加项

后续所有对外 surface 都围绕 sampling-first 设计。

### 决策 C — internal canonical layer 保留但不再作为公开主线

低层显式语义继续存在，但只作为内部稳定地基。

### 决策 D — 坏的主路径必须修复，而不是用 fallback 掩盖

后续实现优先顺序保持为：

- 删除旧路径叙事
- 修复主路径
- 再做 host compatibility 验证

而不是反过来。

---

## 14. 当前结论

当前结论是：

1. **单线架构已被采纳并执行**
2. **REST / stdio / legacy HTTP-SSE sampling 不再具有正式地位**
3. **internal canonical execution layer 继续作为实现地基保留**
4. **后续实现与文档统一围绕 Streamable MCP + sampling 展开**

压缩成一句话：

> **Synapse 不再是“多入口 memory backend”，而是“一个以 Streamable MCP sampling 为唯一正式接口的远程记忆系统”。**

---

*文档版本: 2026-03-16*
