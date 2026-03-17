# TODO-02: Markdown Node Model & File I/O

## Status: COMPLETED
## Priority: P0 (Core data model — most TODOs depend on this)
## Design Doc Section: §3.1, §3.2

---

## Summary

实现 Synapse 的核心数据模型——Markdown 节点（Node）的定义、YAML frontmatter 解析/序列化、以及 Markdown 文件的读写工具。这是 "Text is Graph" 理念的基石。

---

## Detailed Requirements

### 1. Node Data Model (Pydantic)

定义三层节点模型，对应设计文档 §3.1 的分层系统：

| Tier | Word Limit | Token Budget (中英混合) | Use Case | Decay Rate |
|------|-----------|----------------------|----------|------------|
| `concept` | ≤ 500 words | ~700 tokens | 定义、术语、快速参考 | Fast |
| `decision` | ≤ 2,000 words | ~2,800 tokens | 设计决策 + 理由 + 权衡 | Medium |
| `reference` | ≤ 3,500 words | ~5,000 tokens | 完整设计文档、深度分析 | Slow |

用 Pydantic model 表示：

```python
class NodeTier(str, Enum):
    CONCEPT = "concept"
    DECISION = "decision"
    REFERENCE = "reference"

class NodeType(str, Enum):
    TRANSIENT = "transient"
    PERSISTENT = "persistent"

class NodeStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"

class NodeMetadata(BaseModel):
    id: str                              # e.g. mem_20260301_rate_limiting
    title: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5              # [0.0 - 1.0]
    type: NodeType = NodeType.TRANSIENT
    tier: NodeTier = NodeTier.CONCEPT
    status: NodeStatus = NodeStatus.ACTIVE
    supersedes: list[str] = []
    superseded_by: str | None = None
    sensitivity: str = "internal"        # public | internal | private (§7.2)

class Node(BaseModel):
    metadata: NodeMetadata
    content: str                         # Markdown body (sans frontmatter)
    file_path: Path                      # Relative path to .md file
```

### 2. YAML Frontmatter Parser

实现双向转换：
- **读取**: `.md` file → `Node` object（解析 `---` 分隔的 YAML frontmatter + markdown body）
- **写入**: `Node` object → `.md` file（序列化 YAML frontmatter + markdown body）

使用 `python-frontmatter` 库或手工解析 `---` 块。

Frontmatter 格式严格遵循设计文档：
```yaml
---
id: mem_20260301_rate_limiting
title: Token-based Microservice Rate Limiting
created_at: 2026-03-01T10:00:00Z
last_accessed: 2026-03-05T14:30:00Z
access_count: 5
importance: 0.8
type: persistent
tier: decision
status: active
supersedes: []
superseded_by: null
sensitivity: internal
---
```

### 3. Wiki-Link Extraction (Edge Generation)

实现 `[[...]]` 语法扫描，从 markdown body 中提取所有出链（outgoing edges）：

```python
def extract_wiki_links(content: str) -> list[str]:
    """Extract all [[Node_Name]] references from markdown content."""
    # Regex: \[\[([^\]]+)\]\]
    # Returns: ["API_Gateway_Design", "AuthZ_Architecture_V2"]
```

这是 §3.2 "Bi-directional Linking" 的实现基础。反向链接（In-Degree）由 SQLite edges 表在 TODO-03 中追踪。

### 4. Node ID Generation

标准化 Node ID 生成规则：
- 格式: `mem_{date}_{slug}`
- 示例: `mem_20260301_rate_limiting`
- `slug` 由 title 自动生成（lowercase, underscore-separated, ASCII-safe）

### 5. File I/O Utilities

- **Atomic Write**: 写入 `.tmp` 文件后 `os.rename()` 到目标路径（§9.2 write atomicity）
- **Node File Path Convention**: `{base_path}/active/{node_id}.md`
- **Archive Path**: `{archive_path}/{node_id}.md`
- **批量扫描**: 扫描目录下所有 `.md` 文件并解析为 `Node` 列表

### 6. Word Count Validation

实现分层字数限制校验（基于 tier）：
- `concept`: ≤ 500 words
- `decision`: ≤ 2,000 words
- `reference`: ≤ 3,500 words

超限时返回 warning（不阻断写入，但记录日志）。

### 7. Supersession Banners (§5.4.3)

写入时支持自动添加 supersession 相关的 markdown banner：

被替代节点：
```markdown
> ⚠️ **SUPERSEDED** by [[new_node_id]] on 2026-03-15.
> This node is retained for historical context but its conclusions are outdated.
```

替代节点：
```markdown
> **Supersedes**: [[old_node_id]] — [reason]
```

---

## Dependencies
- **TODO-01**: Project structure, Pydantic, config

## Blocks
- TODO-03 (SQLite schema relies on Node model)
- TODO-05 (File watcher parses markdown files)
- TODO-08 (Write path creates Node objects)

## Acceptance Criteria
- [x] `Node` Pydantic model 定义完整，含所有 frontmatter 字段
- [x] `.md` 文件 ↔ `Node` 对象双向无损转换
- [x] Wiki-link `[[...]]` 提取正确
- [x] Node ID 生成规则一致
- [x] 原子写入实现（`.tmp` → `rename`）
- [x] 字数校验按 tier 分层工作
- [x] Supersession banner 自动添加/移除
- [x] 单元测试覆盖所有 edge case（空 frontmatter, 缺失字段, 中英混合内容）

## Implementation Notes

- Added Pydantic node models with tier/type/status/sensitivity enums under `synapse/models/node.py`
- Added Markdown frontmatter parsing/serialization, atomic writes, directory scanning, wiki-link extraction, and banner helpers under `synapse/storage/markdown.py`
- Added tests for round-tripping, missing optional fields, mixed-language content, link extraction, scan helpers, and banner behavior
