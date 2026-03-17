# TODO-07: MCP Server & REST API

## Status: COMPLETED
## Priority: P0 (External interface — LLM & IDE communicate through this)
## Design Doc Section: §2 (Architecture), §4.2.2, §7.1.1

---

## Summary

实现 Synapse 的 MCP（Model Context Protocol）Server 和可选的 REST API，作为 Cloud LLM 和 IDE（Claude Desktop, Cursor 等）与本地记忆系统交互的唯一入口。支持搜索、写入、链接等 Tool Calls。

---

## Detailed Requirements

### 1. MCP Server Implementation

实现标准 MCP 协议的 Server，支持以下 Tool 定义：

#### Tool: `search_memory`
```json
{
  "name": "search_memory",
  "description": "Search the knowledge graph for relevant context. Returns top 1-3 highly relevant nodes.",
  "parameters": {
    "query": {"type": "string", "description": "Natural language search query"},
    "top_k": {"type": "integer", "default": 3, "description": "Number of results to return"}
  }
}
```
- 内部调用 TODO-06 的 Retrieval Pipeline
- 先用 TODO-04 的 Embedding Engine 编码 query
- 返回 Top-K 最终上下文

#### Tool: `write_node`
```json
{
  "name": "write_node",
  "description": "Create or update a knowledge node in the memory graph.",
  "parameters": {
    "title": {"type": "string"},
    "content": {"type": "string", "description": "Markdown body content"},
    "tier": {"type": "string", "enum": ["concept", "decision", "reference"]},
    "type": {"type": "string", "enum": ["transient", "persistent"], "default": "transient"},
    "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
    "links": {"type": "array", "items": {"type": "string"}, "description": "Wiki-link targets to embed in content"},
    "sensitivity": {"type": "string", "enum": ["public", "internal", "private"], "default": "internal"}
  }
}
```
- 生成 Node ID、构建 frontmatter
- 写入 `.md` 文件（atomic write）
- File Watcher 自动同步到 DB

#### Tool: `search_existing_nodes`
```json
{
  "name": "search_existing_nodes",
  "description": "Search for existing nodes similar to given content. Used for linking and conflict detection.",
  "parameters": {
    "query": {"type": "string"},
    "similarity_threshold": {"type": "number", "default": 0.5}
  }
}
```
- 返回相似节点列表及相似度分数
- 用于写入路径的 Retrieval-Augmented Linking（§6 Step 3）和冲突检测（§6 Step 4）

#### Tool: `update_node_status`
```json
{
  "name": "update_node_status",
  "description": "Update a node's status (for supersession/dispute resolution).",
  "parameters": {
    "node_id": {"type": "string"},
    "status": {"type": "string", "enum": ["active", "superseded", "disputed"]},
    "superseded_by": {"type": "string", "description": "ID of the superseding node (if status=superseded)"}
  }
}
```

#### Tool: `get_node`
```json
{
  "name": "get_node",
  "description": "Retrieve a specific node by ID.",
  "parameters": {
    "node_id": {"type": "string"}
  }
}
```

### 2. REST API Layer (Optional but recommended)

提供等价的 HTTP 端点，供非 MCP 客户端使用：

```
GET    /api/v1/search?q={query}&top_k={k}
POST   /api/v1/nodes                        # write_node
GET    /api/v1/nodes/{node_id}               # get_node
PATCH  /api/v1/nodes/{node_id}/status        # update_node_status
GET    /api/v1/health                        # 健康检查
GET    /api/v1/stats                         # 知识库统计
```

使用 FastAPI 或 Starlette 框架。

### 3. Server Configuration (§4.2.2)

```toml
[server]
host = "0.0.0.0"     # Bind to all interfaces
port = 8765
cors_allowed_origins = ["*"]
auth_token = ""       # Optional static auth token
```

- **CORS**: 允许跨域请求（IDE 可能从不同来源连接）
- **绑定 `0.0.0.0`**：允许局域网内任何设备连接

### 4. Authentication (§7.1.1)

简单的静态 Token 认证：
- 如果 `config.toml` 设置了 `auth_token`，所有请求必须在 header 中携带：
  ```
  Authorization: Bearer <token>
  ```
- 未设置 `auth_token` 时不强制认证（本地开发模式）
- 认证失败返回 401

### 5. Transport Layer

MCP Server 支持两种传输方式：
- **stdio**：适用于 Claude Desktop 等通过进程管道通信的客户端
- **HTTP/SSE**：适用于 Cursor、Web IDE 等通过网络连接的客户端

### 6. Error Response Format

统一错误响应格式：
```json
{
  "error": {
    "code": "NODE_NOT_FOUND",
    "message": "Node with id 'mem_xxx' does not exist",
    "details": {}
  }
}
```

### 7. Request/Response Logging

- 所有请求/响应记录到 `.synapse/.logs/mcp-daemon.log`
- 包含：timestamp, tool name, parameters (redacted if sensitive), response size, latency

---

## Dependencies
- **TODO-01**: Config, logging, project structure
- **TODO-02**: Node model (for write_node)
- **TODO-03**: SQLite (for direct queries if needed)
- **TODO-04**: Embedding engine (for query embedding)
- **TODO-06**: Retrieval pipeline (search_memory)

## Blocks
- TODO-08 (Write path uses MCP tools)
- TODO-11 (Daemonization starts MCP server)

## Acceptance Criteria
- [x] MCP Server 启动并响应 Tool Calls
- [x] `search_memory` 返回正确的 Top-K 结果
- [x] `write_node` 创建 `.md` 文件并触发 sync
- [x] `search_existing_nodes` 返回相似节点及分数
- [x] REST API 端点与 MCP Tools 等价
- [x] `auth_token` 认证工作正常（设置时拒绝无 token 请求）
- [x] CORS 配置生效
- [x] 请求/响应日志记录正常
- [x] stdio 和 HTTP 两种传输方式可用
- [x] 健康检查端点返回服务状态
