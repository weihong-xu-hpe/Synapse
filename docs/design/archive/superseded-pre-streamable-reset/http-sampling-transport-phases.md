# Synapse — HTTP Sampling Transport 四阶段实施拆解

> **文档状态**: 草稿，待审阅  
> **日期**: 2026-03-15  
> **适用范围**: Synapse HTTP MCP transport、sampling-powered 高层工具、remote SSH / local Mac 使用场景  
> **主设计文档**: `docs/design/http-sampling-transport-design.md`

---

## 1. 文档目的

这份文档不是重新讨论“要不要做 HTTP sampling”，而是把主设计文档中的推荐方向拆成 **四个可审阅、可验收、可暂停** 的阶段。

拆解原则：

- 每个 Phase 都有清晰边界
- 每个 Phase 都能单独验收
- 每个 Phase 都尽量不推翻前一阶段
- 尽量优先建立正确的 transport 与 session 基础，再引入更高层的 host compatibility 复杂度

这意味着：

> **先把 server 端会话模型和 transport 基础打稳，再把 HTTP sampling 变成真正可用能力，最后再验证 VS Code / Host 兼容性。**

---

## 2. 四阶段总览

### Phase 0 — Runtime & Session Refactor

目标：
- 把当前 stdio-only 假设从 MCP runtime 中剥离出来
- 建立 transport-agnostic 的 session state 抽象
- 为 HTTP sampling 打下地基

产出：
- runtime / session 的职责边界清晰
- stdio 继续可用
- HTTP 可以开始拥有独立 session 语义

### Phase 1 — Sessionful HTTP Sampling Transport

目标：
- 在 HTTP / SSE transport 上真正建立 sampling request / response 闭环
- 让 synthetic client 可以通过 HTTP 成功运行 sampling-powered 高层工具

产出：
- `HttpMcpSessionManager`
- per-session SSE stream
- sampling result 回传 endpoint
- `HttpSamplingClient`

### Phase 2 — Hardening, Security, Observability

目标：
- 把 HTTP sampling 从“能跑”提升到“可控、可审计、可超时、可恢复”
- 解决认证绑定、超时、重复提交、会话回收、日志与审计问题

产出：
- auth-bound session
- timeout / cancellation 语义
- 完整审计事件
- failure matrix 与回归测试

### Phase 3 — Host Compatibility & Productization

目标：
- 验证 VS Code / host 在 HTTP MCP 模式下的 sampling 兼容性
- 根据真实兼容性决定是否把 HTTP sampling 提升为推荐路径

产出：
- compatibility matrix
- user-facing guidance
- 可能的 fallback 策略
- 是否升格为正式推荐能力的结论

---

## 3. Phase 0 — Runtime & Session Refactor

## 3.1 目标

当前 `SynapseMCPServer` 在一个对象里同时承载：

- tool registry
- protocol dispatch
- session state
- transport-specific sampling glue

这在 stdio 下成立，因为 stdio 默认只有一个 client / 一个进程 / 一个会话。

但 HTTP sampling 的前提是：

- 一个 server 实例下可存在多个独立 session
- 每个 session 有自己的 capabilities / initialized 状态 / sampling request counter
- transport-specific sampling glue 必须从 runtime 中解耦

因此 Phase 0 的唯一核心目标是：

> **让 MCP runtime 成为“session-aware but transport-agnostic”的核心。**

## 3.2 本阶段必须完成的事

1. 抽出 session state 概念  
   至少包含：
   - protocol version
   - client capabilities
   - initialized flag
   - request counter
   - tool profile

2. 抽出 runtime / registry 概念  
   保证 tool registry、tool exposure profile、request dispatch 与 transport-specific state 脱钩。

3. 让 stdio 改为“使用一个显式 session 对象”  
   这样 stdio 和 HTTP 的差异只剩 transport，不再是“逻辑层”差异。

4. 保持现有 stdio sampling 全部回归通过  
   Phase 0 绝不能把现有 sampling 高层工具搞坏。

## 3.3 本阶段不做的事

- 不做 HTTP sampling request queue
- 不做 SSE 升级
- 不做新 endpoint
- 不做 auth/timeout hardening
- 不做 VS Code 兼容性验证

## 3.4 交付标准

如果满足以下条件，Phase 0 可以算完成：

- stdio 下所有 sampling 测试仍然通过
- runtime 不再依赖 app-global 的 client state
- session state 已具备被 HTTP 复用的结构
- 代码结构上已经能自然挂接 `HttpSamplingClient`

## 3.5 主要风险

- refactor 期间把 stdio sampling 回路打断
- runtime / session 边界拆不干净，HTTP 阶段继续返工
- 为 future-proof 过度设计，反而拖慢落地

## 3.6 退出条件

退出本阶段前，必须能回答：

> 如果今天要给 HTTP 创建第二个 session，它的 capabilities / initialized / request counter 会不会和第一个 session 混掉？

如果答案还是“会”，那 Phase 0 还没完成。

---

## 4. Phase 1 — Sessionful HTTP Sampling Transport

## 4.1 目标

这一阶段的目标不是“让 HTTP 看起来像支持 sampling”，而是：

> **让 HTTP transport 真正可以完成一次 sampling-powered high-level tool call。**

也就是让以下链路成立：

$$
\text{tools/call over HTTP} \rightarrow \text{server emits sampling request} \rightarrow \text{client responds} \rightarrow \text{tool resumes and returns final result}
$$

## 4.2 本阶段必须完成的事

1. 增加显式 session open / close 语义  
   推荐：
   - `POST /mcp/session/open`
   - `DELETE /mcp/session/{session_id}`

2. 将 `/mcp/sse` 升级为 per-session 事件流  
   不再只是输出 ready/tools，而是可输出 sampling request。

3. 增加 sampling result 回传 endpoint  
   推荐：
   - `POST /mcp/sampling/respond`

4. 引入 `HttpSamplingClient`  
   行为上要与 `StdioSamplingClient` 等价：
   - 发 request
   - 等 response
   - 解析 JSON
   - 恢复原始工具执行

5. 增加 synthetic client 端到端测试  
   至少覆盖：
   - open session
   - initialize
   - notifications/initialized
   - tools/call
   - SSE 收到 sampling request
   - respond endpoint 回传
   - 最终工具调用成功返回

## 4.3 本阶段不做的事

- 不解决多 worker 分布式会话共享
- 不解决复杂 reconnect 策略
- 不做完善的 host compatibility 适配
- 不把 HTTP sampling 立即写成用户主推荐路径

## 4.4 交付标准

如果满足以下条件，Phase 1 可以算完成：

- 可以通过 synthetic HTTP client 成功跑通 `integrate_memory_with_sampling`
- HTTP transport 可以在单进程内支撑多个独立 session
- SSE 不再只是广播 ready/tools，而是能承载 sampling request
- tool 执行中的暂停/恢复语义已被 server 正确支持

## 4.5 主要风险

- 长请求阻塞导致 HTTP 超时行为不可控
- SSE 通道和 RPC 会话绑定不牢，导致 request 送错 client
- session open / close 设计不清晰，后面难以兼容真实 host

## 4.6 退出条件

退出本阶段前，必须能回答：

> 如果一个 HTTP client 真正需要执行 `integrate_memory_with_sampling`，server 是否已经能在没有 stdio 的前提下独立完成 sampling 闭环？

如果答案还是“不能，只能列出 tools”，那 Phase 1 还没完成。

---

## 5. Phase 2 — Hardening, Security, Observability

## 5.1 目标

Phase 1 解决的是“能跑”。

Phase 2 的目标是：

> **让 HTTP sampling 成为可控、可审计、可恢复的能力，而不是 demo 级能力。**

## 5.2 本阶段必须完成的事

1. 认证绑定到 session  
   保证：
   - session 创建者
   - `/mcp`
   - `/mcp/sse`
   - `/mcp/sampling/respond`
   处于同一认证上下文。

2. 增加 timeout 与 cancellation 语义  
   至少区分：
   - sampling timeout
   - session closed while waiting
   - transport disconnected

3. 增加重复提交与冲突保护  
   例如：
   - 同一 `request_id` 的二次提交
   - 已超时 request 的晚到结果

4. 增加审计与结构化日志  
   至少有：
   - session open/close
   - sampling requested
   - sampling responded
   - timeout
   - resume succeeded/failed

5. 增加失败矩阵测试  
   必测场景：
   - missing session
   - stale session
   - no active SSE channel
   - response timeout
   - duplicate response
   - auth mismatch
   - malformed sampling result

## 5.3 本阶段不做的事

- 不保证所有第三方 host 已兼容
- 不承诺 HTTP sampling 成为默认推荐使用方式
- 不做全分布式 deployment 方案

## 5.4 交付标准

如果满足以下条件，Phase 2 可以算完成：

- 所有关键错误都有明确 error code
- session 不会因为异常路径泄漏 pending request
- audit log 足以回答 sampling request 生命周期问题
- 非法/迟到/重复 sampling response 不会污染原始调用

## 5.5 主要风险

- 为了补齐安全语义引入太多复杂度，拖慢迭代
- timeout/cancel 语义不统一，导致 client 行为不确定
- 日志记录粒度过粗或过细，分别损伤可调试性或隐私边界

## 5.6 退出条件

退出本阶段前，必须能回答：

> 当一次 HTTP sampling 请求失败时，我们能否明确知道它失败在：会话、认证、传输、模型响应、还是恢复执行？

如果答案还是“不确定”，那 Phase 2 还没完成。

---

## 6. Phase 3 — Host Compatibility & Productization

## 6.1 目标

这一阶段关注的不是 server 内部，而是：

> **真实 MCP host/client（尤其是 VS Code 相关环境）是否真的会配合 HTTP sampling transport。**

这是一个产品化阶段，而不是纯 server 编码阶段。

## 6.2 本阶段必须完成的事

1. 建立 compatibility matrix  
   至少区分：
   - stdio sampling：是否支持
   - HTTP sampling：是否支持
   - SSE + respond flow：是否支持

2. 验证 VS Code / Remote SSH 真实场景  
   包括：
   - 本地 Mac 跑 Synapse HTTP server
   - remote SSH IDE 通过 tunnel 访问
   - 高层 sampling 工具是否真正工作

3. 明确用户文档策略  
   根据真实兼容性，决定文档怎么写：
   - 正式推荐
   - 实验性能力
   - 兼容性受限能力

4. 如果兼容性不一致，定义 fallback guidance  
   例如：
   - host 不支持 HTTP sampling → 建议切回 stdio
   - remote/local split 必须用 HTTP → 暂时只能用非-sampling 路径

## 6.3 本阶段不做的事

- 不继续扩展更多 transport 形态，除非 Phase 3 评估后明确需要
- 不在未验证 host 兼容性的情况下承诺“HTTP sampling 已普适可用”

## 6.4 交付标准

如果满足以下条件，Phase 3 可以算完成：

- 已知目标 host 的行为被实测记录
- 用户能够清晰知道“什么时候该用 stdio，什么时候可以用 HTTP sampling”
- 文档与实现边界一致，不会出现过度承诺

## 6.5 主要风险

- server 已做完，但 host 根本不支持该模式
- 某些 host 名义上支持 sampling，但不支持 HTTP transport 下的该交互模型
- 文档写得太乐观，导致用户以为 remote SSH + local Mac + HTTP sampling 已经无条件可用

## 6.6 退出条件

退出本阶段前，必须能回答：

> 对目标用户而言，HTTP sampling 现在是“设计上可行”、"server 端可行"、还是“端到端已可稳定使用”？

这三个状态必须被明确区分。

---

## 7. 推荐实施顺序

推荐严格按顺序推进，不建议跳 Phase：

1. **Phase 0** — 先拆 runtime/session
2. **Phase 1** — 再做 sessionful HTTP sampling 闭环
3. **Phase 2** — 再补安全、超时、审计与失败语义
4. **Phase 3** — 最后做 host compatibility 和产品化文档

原因是：

- 如果不先做 Phase 0，后续 HTTP session 很容易和 app-global 状态打架
- 如果不做 Phase 2，就算能跑也不适合长期服务场景
- 如果不做 Phase 3，就无法诚实回答用户“我到底现在能不能用”

---

## 8. 每个阶段的审阅重点

为了便于先审方案再动手，建议你看每个阶段时重点关注：

### 审 Phase 0 时
看是否同意：
- session state 必须独立化
- stdio 与 HTTP 应共享 runtime 核心，而不是共享 app-global client state

### 审 Phase 1 时
看是否同意：
- 采用 SSE 下行 + HTTP 上行回传，而不是第一阶段直接上 WebSocket
- 显式 session open/close 是值得引入的

### 审 Phase 2 时
看是否同意：
- HTTP sampling 一旦做，就必须把 auth / timeout / audit 一起做进来
- 不能让它停留在 demo 级能力

### 审 Phase 3 时
看是否同意：
- host compatibility 必须单独验证
- server 可做 ≠ 用户今天就可无脑使用

---

## 9. 最终建议

如果你的目标是：

> **先看清楚这件事要怎么做，再决定是否投入实现**

那么这四个 Phase 已经足够把“原则、实现、加固、产品化”四层分开。

一句话总结就是：

> **Phase 0 先打地基，Phase 1 先跑通，Phase 2 让它可靠，Phase 3 才讨论是不是对用户正式承诺。**

---

*文档版本: 2026-03-15*
