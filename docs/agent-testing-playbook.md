# Synapse Agent Testing Playbook

> 面向 MCP agent-client 的分阶段验证指南。

**核心原则：所有测试步骤均由 AI agent（如 Copilot）驱动执行，不是编写独立测试脚本。**

本文档是给 agent 的操作手册——你（人类）负责启动服务、确认阶段性结果、在需要时提供 sampling 回复；agent 负责构造请求、发送调用、解读返回、验证检查点。每个 Phase 是一次对 agent 的指令，agent 按步骤自主完成并汇报。

---

## 0. 通用基础

> 服务启动由操作者自行完成，此处不再赘述。  
> 后续所有请求目标: 本地运行的 Synapse MCP 服务。

### 0.1 会话初始化 (每个测试阶段前执行)

**Step 1 — Initialize**

```
POST /mcp
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": { "sampling": {} },
    "clientInfo": { "name": "agent-test", "version": "1.0" }
  }
}
```

- 返回 header `mcp-session-id` → 记为 `$SESSION`
- `capabilities.sampling` 必须存在，否则 sampling 工具不可用

**Step 2 — Notify initialized**

```
POST /mcp
Content-Type: application/json
mcp-session-id: $SESSION

{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

- 期望 HTTP 202

**Step 3 — Verify tools/list**

```
POST /mcp
Content-Type: application/json
mcp-session-id: $SESSION

{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}
```

- 确认返回 3 个工具: `search_memory`, `write_memory`, `run_dreamer`

### 0.3 SSE 流监听 (sampling 工具必须)

对于需要 sampling 的工具，请求时设置 `Accept: text/event-stream`，或在单独连接上打开 GET SSE 流：

```
GET /mcp
mcp-session-id: $SESSION
Accept: text/event-stream
```

这条连接保持打开，服务器会通过它推送 `sampling/createMessage` 请求。

### 0.4 Sampling 响应模式

当 SSE 流推送一条 `sampling/createMessage` 请求时，格式如下：

```json
{
  "jsonrpc": "2.0",
  "id": 10000,
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      { "role": "user", "content": { "type": "text", "text": "<prompt>" } }
    ],
    "maxTokens": 600,
    "systemPrompt": "..."
  }
}
```

你需要通过 `POST /mcp` 回送响应（id 必须匹配）：

```
POST /mcp
Content-Type: application/json
mcp-session-id: $SESSION

{
  "jsonrpc": "2.0",
  "id": 10000,
  "result": {
    "role": "assistant",
    "model": "manual-test",
    "content": { "type": "text", "text": "<你的 JSON 回答>" }
  }
}
```

---

## Phase 1: Read — 纯读取验证

> 目标: 验证检索和节点查询，不涉及 sampling。

### 1.1 search_memory (空库)

```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "method": "tools/call",
  "params": {
    "name": "search_memory",
    "arguments": { "query": "test query", "top_k": 5 }
  }
}
```

**检查点**:
- [ ] HTTP 200, `result.content[0].type == "text"`
- [ ] `structuredContent.results` 为空数组
- [ ] `structuredContent.query == "test query"`
- [ ] `structuredContent.top_k == 5`

### 1.2 tools/call 未知工具

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "nonexistent_tool",
    "arguments": {}
  }
}
```

**检查点**:
- [ ] 返回错误码 `TOOL_NOT_FOUND`

### 1.4 参数验证

```json
{
  "jsonrpc": "2.0",
  "id": 13,
  "method": "tools/call",
  "params": {
    "name": "search_memory",
    "arguments": { "query": "", "top_k": 5 }
  }
}
```

**检查点**:
- [ ] 返回错误码 `INVALID_ARGUMENTS`(query min_length=1 被违反)

---

## Phase 2: Write — Sampling 写入验证

> 目标: 验证 write_memory 写入流程。需要 SSE 流 + 手动 sampling 回复。

### 2.1 write_memory — 空库创建决策

**发送** (带 `Accept: text/event-stream`):

```json
{
  "jsonrpc": "2.0",
  "id": 20,
  "method": "tools/call",
  "params": {
    "name": "write_memory",
    "arguments": {
      "title": "Python asyncio patterns",
      "content": "Use asyncio.gather for concurrent coroutines.",
      "type": "persistent",
      "sensitivity": "internal"
    }
  }
}
```

**监听 SSE** → 收到 `sampling/createMessage`，prompt 包含:
- Draft 标题和内容
- Candidate summaries（空库时为 `<no candidates>`）
- JSON shape 指引

**回送 sampling 响应**:

```json
{
  "jsonrpc": "2.0",
  "id": <匹配 sampling 请求的 id>,
  "result": {
    "role": "assistant",
    "model": "manual-test",
    "content": {
      "type": "text",
      "text": "{\"action\":\"create\",\"target_node_ids\":[],\"reasoning\":\"No existing candidates\",\"confidence\":0.95}"
    }
  }
}
```

**检查点**:
- [ ] SSE 流最终推送工具结果
- [ ] `execution.executed == true`
- [ ] `decision.action == "create"`
- [ ] `decision.confidence == 0.95`
- [ ] `evidence.candidates` 为空
- [ ] `execution.result.node` 包含有效 `node_id`, `title`, `file_path`
- [ ] 文件已写入 `memory_root/nodes/` 目录

→ **记下返回的 `node_id`**，后续步骤需要。

### 2.2 search_memory — 验证写入可检索

```json
{
  "jsonrpc": "2.0",
  "id": 22,
  "method": "tools/call",
  "params": {
    "name": "search_memory",
    "arguments": { "query": "asyncio gather", "top_k": 3 }
  }
}
```

**检查点**:
- [ ] `results` 非空
- [ ] 包含 Phase 2.1 写入的节点
- [ ] 每条 result 包含完整 `node` 对象（含 `content`, `links`, `metadata`）
- [ ] `score > 0`

### 2.3 write_memory — supersede 决策

先创建第二个节点（重复 2.1 流程，内容稍有不同）：

```json
{
  "jsonrpc": "2.0",
  "id": 24,
  "method": "tools/call",
  "params": {
    "name": "write_memory",
    "arguments": {
      "title": "Python asyncio best practices",
      "content": "Prefer asyncio.TaskGroup over asyncio.gather for structured concurrency.",
      "type": "persistent",
      "sensitivity": "internal"
    }
  }
}
```

这次 sampling 回复选择 **supersede**：

```json
{
  "text": "{\"action\":\"supersede\",\"target_node_ids\":[\"<2.1 的 node_id>\"],\"reasoning\":\"Updated best practice replaces old pattern\",\"confidence\":0.9}"
}
```

**检查点**:
- [ ] `execution.result.action == "supersede"`
- [ ] `execution.result.target_node_ids` 包含被取代的 node_id
- [ ] 旧节点 status 变为 `superseded`
- [ ] 新节点创建成功，status 为 `active`
- [ ] `search_memory` 查旧节点标题，确认 result 中 `node.metadata.status == "superseded"` 且 `node.metadata.superseded_by` 指向新节点

### 2.4 write_memory — complement 决策

```json
{
  "jsonrpc": "2.0",
  "id": 26,
  "method": "tools/call",
  "params": {
    "name": "write_memory",
    "arguments": {
      "title": "Python asyncio error handling",
      "content": "Always wrap TaskGroup bodies in try/except to handle per-task errors.",
      "type": "persistent",
      "sensitivity": "internal"
    }
  }
}
```

Sampling 回复选择 **complement**：

```json
{
  "text": "{\"action\":\"complement\",\"target_node_ids\":[\"<2.3 的新 node_id>\"],\"reasoning\":\"Error handling supplements the best practices node\",\"confidence\":0.85}"
}
```

**检查点**:
- [ ] `execution.result.action == "complement"`
- [ ] 新节点和目标节点之间建立了 wiki-link
- [ ] 两个节点都保持 `active`

### 2.5 低置信度 fallback 行为

```json
{
  "jsonrpc": "2.0",
  "id": 27,
  "method": "tools/call",
  "params": {
    "name": "write_memory",
    "arguments": {
      "title": "Edge case note",
      "content": "Something ambiguous",
      "confidence_threshold": 0.8
    }
  }
}
```

Sampling 回复给出 **低置信度**：

```json
{
  "text": "{\"action\":\"supersede\",\"target_node_ids\":[],\"reasoning\":\"not sure\",\"confidence\":0.3}"
}
```

**检查点**:
- [ ] `execution.fallback_applied == true`
- [ ] 实际执行的是 `create`（安全降级）
- [ ] 节点仍然被成功创建

---

## Phase 3: Dream — 生命周期整合验证

> 目标: 验证 Dreamer 的六阶段流水线（Scan → Triage → Link Weaving → Conflict Resolution → Execute → Report）。  
> 前提: Phase 2 已创建若干节点。

### 3.0 前置数据准备

Dreamer 需要特定状态的节点才能触发各阶段。通过 Phase 2 的 integrate 工具创建以下数据：

**a) 制造"陈旧"节点（供 Triage 扫描）**

Dreamer 的 scan 根据 `janitor_days` 配置查找长期未访问的节点。在测试环境中可能需要：
- 调低 `config.toml` 中的 `[decay] janitor_days = 0`
- 或通过数据库直接修改节点的 `last_accessed` 时间戳

用 integrate 创建 2-3 个普通节点，然后手动修改它们的时间戳让它们看起来"陈旧"。

**b) 制造"语义相近但无 link"的节点对（供 Link Weaving）**

创建两个内容相关但没有显式 wiki-link 的节点：

```
Node A: "Kubernetes pod scheduling"
  Content: "Kubernetes uses the kube-scheduler to assign pods to nodes based on resource requests."

Node B: "K8s resource management"
  Content: "Kubernetes resource requests and limits control how pods are scheduled to cluster nodes."
```

**c) 制造"已 superseded"节点（供自动归档）**

通过 2.5 的 supersede 流程，被取代的旧节点不需要额外操作。Dreamer 会在 superseded > 7 天后自动归档。测试时可修改时间戳。

### 3.1 run_dreamer — 基础调用

```json
{
  "jsonrpc": "2.0",
  "id": 30,
  "method": "tools/call",
  "params": {
    "name": "run_dreamer",
    "arguments": { "batch_size": 8 }
  }
}
```

Dreamer 运行过程中可能发出 **多轮 sampling 请求**，每轮对应一个阶段：

### 3.2 Triage 阶段 — sampling 请求

SSE 收到的 prompt 包含关键词: `"Synapse Dreamer triage"`, `"NREM slow-wave consolidation"`

System prompt: `"You are a memory-lifecycle agent inside Synapse..."`

prompt 中列出陈旧节点，每个节点包括 ID, Title, Type, Status, Content。

**回送 triage 决策**:

```json
{
  "text": "{\"decisions\":[{\"node_id\":\"<stale-node-1>\",\"decision\":\"keep\",\"reason\":\"Still useful reference\"},{\"node_id\":\"<stale-node-2>\",\"decision\":\"archive\",\"reason\":\"Outdated information\"},{\"node_id\":\"<stale-node-3>\",\"decision\":\"condense\",\"reason\":\"Overlaps with other nodes\"}]}"
}
```

**检查点**:
- [ ] 每个 stale 节点都必须在 decisions 中出现
- [ ] `decision` 值只能是 `keep | condense | archive`

### 3.3 Link Weaving 阶段 — sampling 请求

SSE 收到的 prompt 包含关键词: `"Synapse Dreamer link weaving"`, `"REM associative dreaming"`

prompt 中列出语义相近但无 link 的节点对。

**回送 link 决策**:

```json
{
  "text": "{\"decisions\":[{\"node_a_id\":\"<node-a-id>\",\"node_b_id\":\"<node-b-id>\",\"link\":true}]}"
}
```

**检查点**:
- [ ] 决策中 `link: true` 表示应该互链
- [ ] `link: false` 表示保持独立

### 3.4 Conflict Resolution 阶段 — sampling 请求

SSE 收到的 prompt 包含关键词: `"Synapse Dreamer conflict resolution"`, `"interference clearance"`

prompt 中列出 disputed 节点对及其完整内容。

**回送冲突决策**:

```json
{
  "text": "{\"decisions\":[{\"node_a_id\":\"<node-a>\",\"node_b_id\":\"<node-b>\",\"decision\":\"both_valid\",\"reason\":\"Different perspectives, no real conflict\"}]}"
}
```

**检查点**:
- [ ] `decision` 值只能是 `supersede_a | supersede_b | both_valid`
- [ ] `supersede_a` → A 退役，B 保留；`supersede_b` → 反之

### 3.5 验证 Dreamer Report

所有 sampling 轮次完成后，SSE 流推送最终工具结果。

**检查点 — Report 结构**:

- [ ] `started_at` / `completed_at` 为 ISO 8601 UTC 时间
- [ ] `scanned` 对象包含: `stale`, `superseded`, `disputed`, `missing_link_pairs` 四个整数

**检查点 — Triage 执行**:

- [ ] `triage` 数组中每个条目包含 `node_id`, `decision`, `reason`
- [ ] `decision == "keep"` 的节点: `last_accessed` 已刷新
- [ ] `decision == "archive"` 的节点: 出现在 `archived` 数组中
- [ ] `decision == "condense"` 的节点:
  - 出现在 `condensed` 数组中
  - `condensed[].source_ids` 包含原始节点
  - `condensed[].new_node_id` 存在
  - 新节点标题格式为 `"Archive Condensation <date>"`
  - 原始节点被归档

**检查点 — Link Weaving 执行**:

- [ ] `links_added` 数组反映了 sampling 回复中 `link: true` 的所有对
- [ ] 用 `search_memory` 检查双向 link:
  - Node A 的 content 包含 `[[node-b-id]]`
  - Node B 的 content 包含 `[[node-a-id]]`

**检查点 — Conflict Resolution 执行**:

- [ ] `conflicts_resolved` 数组反映 sampling 决策
- [ ] `supersede_a` 决策: Node A status → `superseded`, `superseded_by → Node B`
- [ ] `supersede_b` 决策: 反之
- [ ] `both_valid` 决策: 两个节点均保持 `active`

**检查点 — 清理**:

- [ ] `deleted_archive_paths`: 超过 retention_days 的归档文件已清理
- [ ] `sync` 对象存在，确认 delta sync 完成

**检查点 — Warnings**:

- [ ] 如果某轮 sampling 失败（超时等），`warnings` 中包含对应告警
- [ ] warning 格式: `{code, message, node_id?}`

### 3.6 Dreamer 空运行

在所有节点都是 active 且刚刚访问过的库上运行：

```json
{
  "jsonrpc": "2.0",
  "id": 31,
  "method": "tools/call",
  "params": {
    "name": "run_dreamer",
    "arguments": { "batch_size": 8 }
  }
}
```

**检查点**:
- [ ] `scanned` 各项均为 0（或很小）
- [ ] **没有 sampling 请求**（无候选节点 → 跳过所有 sampling 阶段）
- [ ] `triage`, `links_added`, `conflicts_resolved` 均为空数组
- [ ] Report 仍然返回成功，`sync` 仍然执行

---

## 附录 A: 错误场景速查

| 场景 | 操作 | 期望 |
|------|------|------|
| 无 session header | 不带 `mcp-session-id` 调 tools/call | 400/404 |
| 无效 session | 随机 UUID 作为 session-id | 404 `MCP_SESSION_NOT_FOUND` |
| Sampling 超时 | 不回送 sampling 响应等 30s | `SAMPLING_TIMEOUT` 错误 |
| 重复 sampling 响应 | 对同一 request_id 回送两次 | 409 `MCP_DUPLICATE_RESPONSE` |
| 非 JSON sampling 回复 | `content.text` 填非法 JSON | `INVALID_SAMPLING_RESPONSE` |
| 缺少 sampling 能力 | Initialize 时不传 `capabilities.sampling` | sampling 工具调用返回错误 |
| 关闭会话 | `DELETE /mcp` + session header | 返回 `closed_at` + 取消 pending 请求数 |

## 附录 B: 数据库直接修改（加速测试）

当需要修改节点时间戳来触发 Dreamer 各阶段时：

```python
import sqlite3
from datetime import datetime, timedelta, timezone

db_path = "/tmp/synapse-agent-test/.synapse/synapse.db"
conn = sqlite3.connect(db_path)

# 让节点看起来"陈旧" — 触发 triage
old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
conn.execute("UPDATE nodes SET last_accessed = ? WHERE node_id = ?", (old_time, "<node-id>"))

# 让 superseded 节点超过 7 天 — 触发自动归档
conn.execute("UPDATE nodes SET updated_at = ? WHERE node_id = ? AND status = 'superseded'",
             (old_time, "<superseded-node-id>"))

conn.commit()
conn.close()
```

## 附录 C: curl 示例

```bash
SESSION="..."  # 从 initialize 响应 header 获取

# Initialize
curl -s -D- http://127.0.0.1:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{"sampling":{}},"clientInfo":{"name":"curl-test","version":"1.0"}}}'

# search_memory
curl -s http://127.0.0.1:8765/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"search_memory","arguments":{"query":"test","top_k":3}}}'

# SSE 流 (另一个终端)
curl -N http://127.0.0.1:8765/mcp \
  -H "mcp-session-id: $SESSION" \
  -H "Accept: text/event-stream"

# write_memory (通过 SSE)
curl -s http://127.0.0.1:8765/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: $SESSION" \
  -H "Accept: text/event-stream" \
  -d '{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"write_memory","arguments":{"title":"Test note","content":"Some content"}}}'
```
