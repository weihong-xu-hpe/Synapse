# Synapse — HTTP Transport Sampling Support Design

> **文档状态**: 草稿，待审阅  
> **日期**: 2026-03-15  
> **适用范围**: Synapse MCP HTTP/SSE transport、remote SSH / VS Code 集成、sampling-powered 高层工具  
> **相关文档**:
> - `docs/design/mcp-sampling-high-level-tools-design.md`
> - `docs/design/external-memory-skill-design.md`
> - `docs/design/http-sampling-transport-phases.md`
> - `docs/usage.md`
> - `docs/configuration.md`

---

## 1. 背景

当前 Synapse 已经具备两套 transport 形态：

- **stdio MCP**
- **HTTP / SSE MCP + REST**

其中，sampling-powered 高层工具（如 `integrate_memory_with_sampling`、`review_memory_cluster`）在 **stdio** 下已经形成完整闭环，因为 stdio transport 会在进程内注入 `StdioSamplingClient`，使 server 可以：

1. 在工具执行中途向 client 发出 `sampling/createMessage`
2. 等待 client/host 侧 LLM 返回结构化 JSON
3. 再继续执行既有 low-level write contract

但在 **HTTP** 侧，当前实现只有：

- `POST /mcp`：单次 JSON-RPC 请求
- `GET /mcp/sse`：单向 server → client 事件流

这还不足以支撑 sampling 闭环。

因此当前结论是：

- HTTP transport 可以提供 REST 和基础 MCP 能力
- 但 **不能完成 sampling-backed 高层工具**
- 这直接影响一种非常重要的使用场景：

> **本地 Mac 上运行 Synapse，remote SSH 连接过来的 VS Code 想继续使用 sampling 高层工具。**

---

## 2. 问题定义

目标不是让 HTTP transport “看起来像支持 sampling”，而是让它真正具备以下能力：

1. Synapse 在 HTTP MCP 工具执行过程中，能够向当前会话对应的 client 发起 sampling 请求
2. client 能够接收该请求，并将 sampling 结果回传
3. Synapse 能够恢复原始工具调用，并完成高层工具逻辑
4. 全流程支持多会话并发、会话隔离、超时控制与审计

换句话说，需要让 HTTP transport 具备如下语义：

$$
\text{HTTP MCP session} \rightarrow \text{sampling request} \rightarrow \text{sampling response} \rightarrow \text{resume original tool call}
$$

---

## 3. 为什么当前 HTTP 还不行

## 3.1 当前实现是“单请求/单响应”模型

`POST /mcp` 当前只是：

- 收一个 JSON-RPC payload
- 调一次 `mcp_server.handle_request(...)`
- 立即返回结果

但 sampling 场景不是单跳完成，而是：

1. `tools/call`
2. server 中途需要 sampling
3. server 反向向 client 发请求
4. client 再回一条结果
5. server 才能完成原始 `tools/call`

也就是说，HTTP sampling 要求 **一次工具调用内部存在“暂停/恢复”语义**。

## 3.2 当前 `/mcp/sse` 只是单向广播型 ready/tools 输出

当前 `GET /mcp/sse` 只会输出：

- `ready`
- `tools`

它不是：

- per-session 的事件流
- 也不带 pending sampling request 队列
- 更没有 request/response 对应关系

## 3.3 当前 `SynapseMCPServer` 混合了“协议逻辑”和“会话状态”

当前对象里直接存放：

- `client_capabilities`
- `client_initialized`
- `protocol_version`
- `_client_request_ids`

这些状态在 stdio 下没有问题，因为 stdio 是单 client、单进程、单会话。

但 HTTP 下如果有多个 client 并发连接，则这些状态必须变成 **per-session**，而不能挂在一个 app 级单例 `mcp_server` 上。

## 3.4 当前 HTTP 启动路径没有自动注入 `sampling_client`

`create_app(...)` 虽然接受 `sampling_client` 参数，但 CLI 的 HTTP 启动路径没有像 stdio 那样自动注入一个可运行的 transport-specific sampling bridge。

最终 sampling 高层工具会在 service 层遇到：

- `_require_sampling_client()`
- `SAMPLING_UNAVAILABLE`

所以当前 HTTP 问题不是“只有 UI 没接上”，而是 **transport 级能力缺失**。

---

## 4. 设计目标

1. **让 HTTP transport 真正支持 sampling-powered 高层工具**  
   不仅能列出 tool，还能执行完整闭环。

2. **支持 remote SSH / local Mac 分离部署**  
   允许 Synapse 作为本地长期运行的 HTTP 服务，而 remote 侧 VS Code/agent 通过网络访问并继续使用 sampling。

3. **保持现有 low-level contract 不变**  
   HTTP sampling 仍然只能编排并下沉到 `integrate_knowledge` 等 canonical execution layer。

4. **保持默认 public MCP surface 不变**  
   暴露面仍然以高层闭环工具组为主，不因为 transport 扩展而重新放开底层工具。

5. **支持可审计、可超时、可取消、可恢复**  
   sampling 请求必须有明确的生命周期和日志边界。

---

## 5. 非目标

1. **不把 REST API 改造成 sampling 主路径**  
   REST 仍然保持明确、显式、低层、单向调用模型。

2. **不重写高层 sampling 工具语义**  
   `decide_memory_write` 等工具的语义保持不变，变的是 transport 实现。

3. **不要求第一阶段就解决多进程、多 worker 水平扩展**  
   Synapse 当前是本地优先系统，第一阶段可以接受单进程 session manager。

4. **不假设所有 MCP host 的 HTTP 模式都已支持 sampling**  
   设计文档必须明确区分：
   - “Synapse server 端已支持”
   - “某个具体 host/client 已验证支持”

---

## 6. 关键设计结论

## 6.1 推荐方向：Sessionful HTTP MCP + SSE 下行 + HTTP 上行回传

推荐的增量实现路径不是直接切 WebSocket，而是：

- 保留 `POST /mcp` 作为 client → server JSON-RPC 入口
- 将 `GET /mcp/sse` 升级为 **per-session sampling event stream**
- 新增一个 **sampling result 回传 endpoint**（client → server）
- 在 server 内部建立 pending request / response waiter 机制

也就是：

### 下行通道
server → client：
- `GET /mcp/sse?session_id=...`
- 用于发送：
  - ready / tools
  - sampling request
  - 未来可扩展的 cancel / status event

### 上行通道
client → server：
- `POST /mcp`
- `POST /mcp/sampling/respond`

### 控制语义
- 原始 `tools/call` 会在 server 内部阻塞等待 sampling response
- sampling response 到达后恢复执行并返回原始 `tools/call` 的最终结果

这是当前 FastAPI + 现有 `/mcp` 与 `/mcp/sse` 结构下**最小侵入、最可落地**的方案。

---

## 7. 为什么不直接推荐 WebSocket

WebSocket 当然也能实现双向会话，但当前不推荐直接把它作为第一阶段方案，原因有三：

1. **现有代码已经有 `/mcp` + `/mcp/sse` 结构**  
   继续增量演化成本更低。

2. **local-first / IDE 集成场景对 SSE 足够友好**  
   很多客户端更容易先接受 HTTP + SSE + POST 的混合模式。

3. **当前真正的风险不在全双工本身，而在会话管理和 host 兼容性**  
   即便改成 WebSocket，也仍然要解决：
   - pending request map
   - session capabilities
   - sampling lifecycle
   - host/client 是否真正支持该模式

因此：

> **先把 sessionful HTTP transport 设计正确，再考虑是否要在未来增加 WebSocket transport。**

---

## 8. 推荐架构

## 8.1 新增的核心抽象

### `HttpMcpSessionManager`

职责：
- 创建/查找/回收 session
- 维护 session timeout
- 管理 pending sampling requests

### `HttpMcpSession`

每个 session 持有：
- `session_id`
- `protocol_version`
- `client_capabilities`
- `client_initialized`
- `tool_profile`
- `request_counter`
- `event_queue`
- `pending_sampling`
- `auth_context`
- `last_seen_at`

### `HttpSamplingClient`

实现 `SamplingClient` 协议，职责是：
- 在需要 sampling 时，向当前 session 的 event queue 推入一个 sampling request
- 阻塞等待 client 回传结果
- 超时后返回 `SAMPLING_FAILED` / `SAMPLING_TIMEOUT`

### `PendingSamplingRequest`

建议字段：
- `request_id`
- `prompt`
- `system_prompt`
- `max_tokens`
- `model_hints`
- `created_at`
- `expires_at`
- `result`
- `error`
- `waiter`（event/condition/future）

---

## 8.2 必须做的对象职责拆分

当前 `SynapseMCPServer` 同时承担：

- tool registry
- protocol handler
- session state
- transport-specific sampling glue（stdio）

HTTP sampling 要落地，建议拆成至少两层：

### A. `MCPRuntime`（或保留 `SynapseMCPServer` 但变轻）
负责：
- tool registry
- tool exposure profile
- request dispatch
- 通用错误包装

### B. `MCPSessionState`
负责：
- protocol version
- capabilities
- initialized flag
- request id allocator
- transport-bound sampling channel

这样：

- stdio 可以继续是单 session 运行时
- HTTP 可以为每个 session 创建独立状态

---

## 9. HTTP transport 协议草案

## 9.1 Session lifecycle

推荐引入显式 session 概念。

### 方案 A：专门的 session open endpoint（推荐）

新增：

- `POST /mcp/session/open`
- `DELETE /mcp/session/{session_id}`

`POST /mcp/session/open` 返回：

```json
{
  "session_id": "sess_...",
  "sse_url": "/mcp/sse?session_id=sess_...",
  "rpc_url": "/mcp",
  "sampling_response_url": "/mcp/sampling/respond"
}
```

然后客户端：

1. open session
2. 打开 SSE
3. 调 `initialize`
4. 发 `notifications/initialized`
5. 开始 `tools/call`

### 方案 B：在 `initialize` 时隐式创建 session

不推荐作为第一阶段方案，因为：
- 更难清晰表达 SSE 绑定关系
- 更难做 reconnect
- 调试体验更差

因此推荐 **显式 session open**。

---

## 9.2 RPC request

客户端仍通过：

- `POST /mcp`

发送 JSON-RPC 请求，但需要带上：

- `X-Synapse-MCP-Session: <session_id>`

server 根据该 header 找到对应 `HttpMcpSession`。

如果缺失：
- 返回 `INVALID_SESSION`

---

## 9.3 SSE event stream

客户端通过：

- `GET /mcp/sse?session_id=<session_id>`

建立会话专属 SSE。

事件类型建议包括：

### `ready`
包含：
- server info
- session id
- current tool profile

### `tools`
包含：
- 当前 session 对应的 tools/list

### `sampling_request`
包含：

```json
{
  "request_id": 10042,
  "method": "sampling/createMessage",
  "params": {
    "messages": [...],
    "systemPrompt": "...",
    "maxTokens": 600,
    "modelPreferences": {...}
  }
}
```

### `sampling_cancel`（可选，Phase 2）
server 在原始调用取消/超时时发出。

---

## 9.4 Sampling result 回传

客户端通过：

- `POST /mcp/sampling/respond`

提交：

```json
{
  "session_id": "sess_...",
  "request_id": 10042,
  "result": {
    "role": "assistant",
    "content": {
      "type": "text",
      "text": "{\"action\":\"create\",...}"
    },
    "model": "gemini-3-flash",
    "stopReason": "endTurn"
  }
}
```

或：

```json
{
  "session_id": "sess_...",
  "request_id": 10042,
  "error": {
    "code": 500,
    "message": "sampling failed"
  }
}
```

server 收到后：
- 找到 pending request
- 写入 result/error
- 唤醒等待中的 `HttpSamplingClient`
- 原始 `tools/call` 继续执行

---

## 10. 原始工具调用如何“暂停/恢复”

关键机制是：

1. `POST /mcp` 收到 `tools/call`
2. tool handler 进入 service 层
3. service 层调用 `HttpSamplingClient.sample_json(...)`
4. `HttpSamplingClient`：
   - 生成 `request_id`
   - 将 sampling request 放进 session 的 SSE event queue
   - 等待 `PendingSamplingRequest.waiter`
5. client 从 SSE 收到 sampling request
6. client 跑本地/host 侧 LLM
7. client 调 `POST /mcp/sampling/respond`
8. server 唤醒 waiter
9. service 层继续执行
10. 原始 `POST /mcp` 返回最终工具结果

这意味着：

- `POST /mcp` 可能是一个长请求
- 但不需要额外的任务系统
- 也不需要改变高层工具语义

---

## 11. 与当前 stdio 设计的关系

HTTP sampling 不应重写 stdio 逻辑，而应复用同一套抽象。

推荐做法：

- 保留 `SamplingClient` 协议不变
- 保留 `StdioSamplingClient`
- 新增 `HttpSamplingClient`
- 将高层 service 完全依赖 `SamplingClient` 抽象

这样高层工具对 transport 无感知：

- stdio 时注入 `StdioSamplingClient`
- HTTP 时注入 `HttpSamplingClient`

这也是当前代码结构已经具备的良好基础。

---

## 12. VS Code / Remote SSH 使用场景

## 12.1 目标场景

用户希望：

- Synapse 跑在本地 Mac
- 记忆库和 `.synapse` 都留在本地 Mac
- remote SSH 连接进另一台主机上的 VS Code workspace
- 该 remote VS Code/agent 仍然能使用 Synapse 的 sampling 高层工具

## 12.2 为什么 HTTP sampling 是这个场景的关键

因为 stdio 不适合“跨机器连接已有进程”：

- stdio 要求 client 直接管理 server 子进程
- remote VS Code workspace 中的 stdio server 往往会在 remote 侧启动
- 它无法自然接入一个已经在本地 Mac 上运行的 stdio Synapse 进程

而 HTTP transport 则允许：

- Synapse 作为本地常驻服务运行在 Mac 上
- remote 侧通过 SSH tunnel / reverse tunnel 访问
- 同时保有 sampling 能力

也就是说，HTTP sampling 的核心价值之一就是：

> **让“本地长期运行的记忆服务器”与“远端 IDE/agent”之间仍然保持高层 sampling 闭环。**

## 12.3 关键前提

这里必须明确一个风险：

> 即便 Synapse server 端实现了 HTTP sampling，最终是否能在某个具体 VS Code MCP host 中工作，还取决于该 host 是否支持在 HTTP MCP transport 上接收并回传 sampling request。

因此设计上必须区分：

- **Server capability**：Synapse 是否能做
- **Client compatibility**：VS Code / host 是否会配合做

第一阶段实现可以先完成 server capability，并用 synthetic HTTP test client 验证；
与 VS Code host 的真实兼容性验证应作为独立阶段。

---

## 13. 安全设计

HTTP sampling 一旦引入 session 和异步回传，安全边界比当前更重要。

## 13.1 必须要求 auth_token

如果启用 HTTP sampling，建议：

- 不允许空 `auth_token`
- 或至少在非 localhost 暴露时强制要求 token

## 13.2 Session 绑定认证上下文

每个 session 必须绑定：
- 创建 session 时的 auth principal
- 后续 `/mcp`、`/mcp/sse`、`/mcp/sampling/respond` 必须使用同一认证上下文

防止：
- A client 发起 tool call
- B client 冒充提交 sampling result

## 13.3 Session TTL

建议：
- idle timeout（如 5–15 分钟）
- pending sampling timeout（如 30–120 秒）
- session close 时清理所有 waiter 与事件队列

## 13.4 回放与重放保护

`request_id` 必须：
- 单 session 唯一
- 一次性消费
- 已完成 request 不可再次写入结果

---

## 14. 可观测性与审计

建议新增以下审计事件：

- `http_session_opened`
- `http_session_closed`
- `http_sampling_requested`
- `http_sampling_response_received`
- `http_sampling_timeout`
- `http_sampling_cancelled`
- `http_sampling_resume_succeeded`
- `http_sampling_resume_failed`

建议记录字段：

- `session_id`
- `request_id`
- `tool_name`
- `sampling_provider`
- `transport=http`
- `latency_ms`
- `result_size`
- `client_capabilities`
- prompt hash / prompt metadata（而不是默认记录全文）

---

## 15. 错误语义

除已有错误码外，建议新增：

### `INVALID_SESSION`
- session 不存在、过期或未绑定

### `SAMPLING_CHANNEL_NOT_CONNECTED`
- 工具执行时 session 没有活跃 SSE sampling channel

### `SAMPLING_TIMEOUT`
- client 未在规定时间内提交 sampling result

### `SAMPLING_RESPONSE_CONFLICT`
- 同一 `request_id` 收到重复结果

### `SESSION_AUTH_MISMATCH`
- sampling result 的认证上下文与原始 session 不一致

---

## 16. 分阶段落地建议

更细的四阶段实施拆解见：

- `docs/design/http-sampling-transport-phases.md`

## Phase 0：Server-side design refactor

目标：
- 拆分 `SynapseMCPServer` 的 session state 与 runtime/registry
- 为 transport-specific session 打基础

产出：
- `MCPSessionState`
- transport-agnostic runtime core

## Phase 1：HTTP sessionful transport

目标：
- 增加 session manager
- 增加 `POST /mcp/session/open`
- 增加 per-session SSE queue
- 增加 `POST /mcp/sampling/respond`
- 实现 `HttpSamplingClient`

产出：
- synthetic client 可跑通 sampling 高层工具

## Phase 2：测试与审计

目标：
- 增加 HTTP sampling 端到端测试
- 增加超时 / 重复提交 / 认证不一致测试
- 增加日志与审计事件

## Phase 3：VS Code / Host compatibility validation

目标：
- 验证具体 host 是否真的支持 HTTP 模式下的 sampling 协作
- 若不支持，则记录 capability matrix

可能结论：
- Synapse server 已支持
- 但某些 host 仍只能在 stdio 模式下使用 sampling

这不是失败，而是正确的兼容性结论。

---

## 17. 对用户体验的影响

实现完成后，理想体验是：

### 本地 Mac 侧
- 启动长期运行的 Synapse HTTP server
- 记忆库仍在本地

### remote SSH / remote IDE 侧
- 通过 tunnel 连到本地 Synapse
- 仍可用高层 sampling 工具
- 不必把 `.synapse` 搬到 remote

这会把 Synapse 从“本地 IDE 内置 memory helper”提升为：

> **可作为个人长期记忆服务器运行，同时继续服务于远端开发环境中的 MCP agent。**

---

## 18. 推荐最终结论

### 结论 1
**HTTP sampling 是值得做的。**

因为它直接解决一个高价值场景：
- memory server 常驻本地 Mac
- remote SSH / 远端 IDE 继续使用 sampling 高层工具

### 结论 2
**推荐采用“sessionful HTTP MCP + SSE 下行 + HTTP 上行回传”的增量架构。**

原因：
- 与现有 `/mcp` + `/mcp/sse` 结构最兼容
- 比直接改 WebSocket 风险更低
- 能复用现有 `SamplingClient` 抽象

### 结论 3
**必须先拆 session state，再做 transport 扩展。**

否则当前 app 级 `mcp_server` 会把多 client HTTP 会话混在一起。

### 结论 4
**必须把 host/client compatibility 当成独立验证项。**

Server 端可做，不代表 VS Code 当前 HTTP MCP host 就一定会配合完成 sampling。

### 结论 5
**第一阶段不应试图同时解决所有部署拓扑。**

先做到：
- 单进程
- 单实例
- 多 session
- 有 auth
- 有 timeout
- 有审计

已经足够为 remote SSH / local Mac 的个人使用场景提供高价值能力。

---

## 19. 待决策问题

1. `session_id` 是 server 生成还是 client 生成？  
   推荐：server 生成。

2. 是否要求 sampling 时必须有活跃 SSE 连接？  
   推荐：必须要求，否则直接返回 `SAMPLING_CHANNEL_NOT_CONNECTED`。

3. 是否允许一个 session 并发多个 pending sampling request？  
   推荐：允许，但第一阶段可以限制为同一 session 顺序执行，以降低复杂度。

4. 是否把 HTTP sampling 作为默认公开能力写进用户文档？  
   推荐：只有在 host compatibility 验证完成后才作为推荐路径写入；在此之前应标为“实验性 transport capability”。

5. 是否将 long-running `tools/call` 的超时交给反向代理控制？  
   推荐：server 内部必须有自己的 timeout；代理层超时只能作为外层保护。

---

*文档版本: 2026-03-15*
