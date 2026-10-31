# TODO: MCP Tool Surface Consolidation (5 → 3)

> **Status**: Implemented  
> **Scope**: Merge public MCP tools from 5 to 3, simplify agent cognitive load  
> **Breaking**: Yes — tool names change, response schemas change

## Motivation

| Problem | Detail |
|---|---|
| `get_node` is dead weight | `search_memory` 的 snippet 已经返回节点全文（`_format_context_block` = title + path + status + content）。Agent 拿到 search 结果后无法有效判断"该看谁"，现实中只能全看。默认 top_k=3、每篇 ~500 字，3 条全文 + metadata ≈ 2600 tokens，远不会炸 context。 |
| `decide_memory_write` 是死胡同 | 返回决策后 agent 无处可去——`integrate_knowledge` 不是公开 MCP 工具，agent 无法手动执行决策。唯一能执行的 `integrate_memory_with_sampling` 会重新走 sampling，可能拿到不同决策。 |
| 5 个工具增加 agent 选择负担 | Agent 需区分两对几乎等价的工具（search/get_node、decide/integrate），浪费 tool-selection token。 |

## Target State

| # | New tool name | Old tool(s) | Sampling? | 一句话 |
|---|---|---|---|---|
| 1 | `search_memory` | `search_memory` + `get_node` | No | 语义检索，返回完整节点（含 content + metadata + links） |
| 2 | `write_memory` | `integrate_memory_with_sampling` (吸收 `decide_memory_write`) | Yes | Sampling 决策 + 执行写入 |
| 3 | `run_dreamer` | `run_dreamer` | Yes | 后台记忆整合，不变 |

**Removed tools**: `get_node`, `decide_memory_write`

---

## Phase 1: `search_memory` 吸收 `get_node`

### 1.1 变更 search_memory 返回格式

**文件**: `synapse/server/service.py` — `search_memory()`

当前 `_serialize_retrieval_item` 返回摘要格式（snippet + scores）。改为返回完整节点信息：

```python
# Before
{
    "node_id": "...",
    "title": "...",
    "score": 0.92,
    "anchor_score": ...,
    "rerank_score": ...,
    "is_anchor": true,
    "file_path": "active/...",
    "status": "active",
    "markers": [],
    "snippet": "full text already here"
}

# After — each result 包含完整 node 对象
{
    "node_id": "...",
    "title": "...",
    "score": 0.92,
    "anchor_score": ...,
    "rerank_score": ...,
    "is_anchor": true,
    "markers": [],
    "node": {                         # ← 新增：完整节点
        "id": "...",
        "title": "...",
        "content": "...",
        "file_path": "active/...",
        "links": ["[[other_node]]"],
        "metadata": { ... }
    }
}
```

**要点**:
- `_serialize_retrieval_item` 调用 `_serialize_node(item.node)` 嵌入完整 node
- 删除顶级 `snippet` 字段（内容已在 `node.content` 中）
- 保留 `file_path` / `status` 顶级字段以兼容消费端过渡期

### 1.2 移除 get_node MCP 工具注册

| File | Change |
|---|---|
| `synapse/server/mcp.py` | 从 `_build_tool_registry` 删除 `get_node` MCPToolDefinition |
| `synapse/server/schemas.py` | 移除 `GetNodeToolRequest`（或标记 deprecated 保留内部用） |

**保留** `service.get_node()` 方法——内部（Dreamer、write_path）仍在用，只是不再注册为 MCP 公开工具。

### 1.3 测试更新

| File | Scope |
|---|---|
| `tests/test_server_api.py` | 工具列表断言从 5 改 4（Phase 1）；删除 get_node 相关 MCP-level 测试；更新 search_memory 返回断言 |
| `tests/test_streamable_runtime.py` | 工具列表断言同步更新 |
| `tests/test_write_path.py` | 内部 `service.get_node()` 调用不受影响 |
| `tests/test_sqlite_store.py` | 不受影响（storage layer） |

---

## Phase 2: 合并 decide + integrate → `write_memory`

### 2.1 重命名工具

| File | Change |
|---|---|
| `synapse/server/mcp.py` | 将 `integrate_memory_with_sampling` 注册改名为 `write_memory`；删除 `decide_memory_write` 注册 |
| `synapse/server/schemas.py` | `IntegrateMemoryWithSamplingRequest` → `WriteMemoryRequest`；删除 `DecideMemoryWriteRequest`（`WriteMemoryRequest` 不再继承它，直接包含所有字段） |
| `synapse/server/service.py` | `integrate_memory_with_sampling()` → `write_memory()`；`decide_memory_write()` 标记 deprecated 或删除 |

### 2.2 简化参数

`write_memory` 继承 `integrate_memory_with_sampling` 的全部参数，但可以简化默认值：

```python
class WriteMemoryRequest(BaseModel):
    title: str = Field(min_length=1)
    content: str = ""
    type: NodeType = NodeType.TRANSIENT
    links: list[str] = Field(default_factory=list)
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL
    query_hint: str | None = None
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # 保留 confidence 控制
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
```

**移除** `allow_default_create_fallback` 和 `require_confident_decision`——低置信度时默认 fallback 到 create，不再需要两个布尔值组合：
- 原来 `require_confident_decision=True, allow_default_create_fallback=True` → 新行为默认如此
- 原来 `require_confident_decision=True, allow_default_create_fallback=False` → 罕见场景，可通过把 `confidence_threshold` 设为 0 来"关闭"

### 2.3 更新 sampling 工具名集合

| File | Change |
|---|---|
| `synapse/server/mcp.py` | `_SAMPLING_TOOL_NAMES` 改为 `{"write_memory", "run_dreamer"}` |
| `synapse/server/stdio_proxy.py` | 同步修改 `_SAMPLING_TOOL_NAMES` |

### 2.4 更新 streamable_runtime.py

| File | Change |
|---|---|
| `synapse/server/streamable_runtime.py` | `decide_memory_write()` wrapper 删除或改名 |

### 2.5 测试更新

| File | Scope |
|---|---|
| `tests/test_server_api.py` | 工具列表断言改 3；重命名测试函数/调用名；删除 decide_memory_write 独立测试 |
| `tests/test_streamable_runtime.py` | 工具列表和调用名同步 |

---

## Phase 3: 文档和 Skill 更新

### 3.1 核心文档

| File | Change |
|---|---|
| `README.md` | 更新工具列表描述 |
| `docs/usage.md` | "Default public MCP tool surface" 从 5 改 3；更新 mermaid 流程图 |
| `docs/configuration.md` | 更新工具名引用 |
| `docs/agent-testing-playbook.md` | 全面更新：测试用例、JSON-RPC payload、curl 示例 |

### 3.2 设计文档

| File | Change |
|---|---|
| `docs/design/streamable-mcp-single-path-architecture.md` | 更新公开工具列表 |
| `docs/design/streamable-mcp-implementation-plan.md` | 更新工具引用 |
| `docs/design/TODO-sampling-only-cutover.md` | 更新工具面引用 |
| `docs/design/TODO-lifecycle-simplification.md` | 更新功能表 |

### 3.3 Skill 文件

| File | Change |
|---|---|
| `.github/skills/memory-write/SKILL.md` | `decide_memory_write` / `integrate_memory_with_sampling` → `write_memory` |
| `.github/skills/memory-lifecycle/SKILL.md` | 更新 `get_node` 引用 |
| `.github/skills/memory-shared/references/retrieval-guidelines.md` | `search_memory` + `get_node` → 只有 `search_memory` |
| `.github/skills/memory-write/references/retrieval-guidelines.md` | 同上 |
| `.github/skills/memory-lifecycle/references/retrieval-guidelines.md` | 同上 |

---

## Phase 4: 验证

### 4.1 单元测试

```bash
python -m pytest tests/ -x -q
```

确认项：
- [ ] `list_tools` 返回恰好 3 个工具：`search_memory`, `write_memory`, `run_dreamer`
- [ ] `search_memory` 每条 result 包含完整 `node` 对象
- [ ] `write_memory` 走 sampling → 执行完整写入流程
- [ ] 调用已删除工具名返回 `TOOL_NOT_FOUND`
- [ ] `run_dreamer` 行为不变

### 4.2 Agent 端到端

按更新后的 `agent-testing-playbook.md` 重跑：
- [ ] initialize → tools/list 返回 3 工具
- [ ] search_memory 返回完整节点，无需再调 get_node
- [ ] write_memory 完成 sampling + 写入
- [ ] run_dreamer 正常执行

---

## 不做的事

| 不做 | 原因 |
|---|---|
| 给 `search_memory` 加 `node_id` 精确查询模式 | Agent 不需要——search 默认 top_k=3 已经返回全文。做 id 查询用内部 `service.get_node()` 即可 |
| 给 `write_memory` 加 `dry_run` / `execute` 模式 | 没有实际场景需要"只看决策不执行"——决策返回给 agent 后无法人工执行 |
| 降低 `top_k` 上限 | 当前 max=25 足够安全。极端场景（25 × 3500 字）可以未来加 content truncation，现在不需要 |
| 删除 `service.get_node()` 方法 | 内部（Dreamer、write_path）仍在使用 |
| 删除 `service.decide_memory_write()` 内部方法 | `_decide_memory_write_payload` 仍被 `write_memory` 使用，只是不再作为独立公开工具 |

---

## Execution Order

```
Phase 1 (search_memory enrichment + get_node removal from MCP)
  → Phase 2 (decide/integrate merge → write_memory)
    → Phase 3 (docs + skills)
      → Phase 4 (validation)
```

Phase 1 和 Phase 2 代码改动互不依赖，可以并行。Phase 3 等代码稳定后统一更新。Phase 4 贯穿始终。
