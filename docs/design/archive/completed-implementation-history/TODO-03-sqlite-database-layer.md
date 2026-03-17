# TODO-03: SQLite Database Layer (Schema + CRUD)

## Status: COMPLETED
## Priority: P0 (Index layer — retrieval & sync depend on this)
## Design Doc Section: §4.1, §4.2, §4.3

---

## Summary

实现 Synapse 的 SQLite 数据库层，包括 Schema 创建（FTS5 + sqlite-vec + edges）、Node CRUD 操作、以及全文搜索和向量搜索的基础查询接口。SQLite 是 Markdown 的 **派生索引**，不是数据源。

> Implementation note: Phase 3 ships with a **Python cosine fallback** for `nodes_vec` because `sqlite-vec` was not available in-session. The schema still includes a real `nodes_vec` table and the vector search API is live; the backend is explicitly reported as `python-fallback` in code and status output.

---

## Detailed Requirements

### 1. Design Principle: Dual-Layer Storage

| Layer | Role | Format | Human-Readable | Git-Friendly |
|---|---|---|---|---|
| **Primary (Markdown)** | Source of truth | `.md` files with YAML frontmatter | Yes | Yes |
| **Secondary (SQLite DB)** | 派生搜索索引 — 可从 Markdown 完全重建 | 单个 `.db` 文件 | No | No (gitignored) |

SQLite 数据库是 **缓存**，损坏或删除后可通过重新扫描 Markdown 文件完全重建。

### 2. Schema Implementation

严格按设计文档 §4.3 创建以下表：

```sql
-- Core node table
CREATE TABLE nodes (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    file_path   TEXT NOT NULL UNIQUE,
    content     TEXT NOT NULL,
    importance  REAL DEFAULT 0.5,
    type        TEXT DEFAULT 'transient',
    tier        TEXT DEFAULT 'concept',
    status      TEXT DEFAULT 'active',
    supersedes  TEXT,                 -- JSON array
    superseded_by TEXT,
    created_at  TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    tags        TEXT                  -- JSON array
);

CREATE INDEX idx_nodes_tier ON nodes(tier);
CREATE INDEX idx_nodes_status ON nodes(status);

-- Full-text search
CREATE VIRTUAL TABLE nodes_fts USING fts5(
    title, content, tags,
    content='nodes', content_rowid='rowid'
);

-- Vector similarity search (dimension configurable from config.toml)
CREATE VIRTUAL TABLE nodes_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding float[1024]
);

-- Backlink index
CREATE TABLE edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE SET NULL
);

CREATE INDEX idx_edges_target ON edges(target_id);
```

### 3. SQLite Configuration
- **WAL Mode**: 启用 Write-Ahead Logging（`PRAGMA journal_mode=WAL`）— §9.2
- **sqlite-vec Extension**: 加载 `sqlite-vec` 扩展（通过 `pip install sqlite-vec`）
- **FTS5 Triggers**: 实现 FTS5 同步触发器（INSERT/UPDATE/DELETE 时自动维护 `nodes_fts`）
- **数据库路径**: `{config.memory.base_path}/synapse.db`

### 4. CRUD Operations

实现 `NodeStore` 接口（TODO-01 定义的抽象）：

```python
class SQLiteNodeStore:
    def upsert_node(self, node: Node, embedding: list[float] | None = None) -> None
    def get_node(self, node_id: str) -> Node | None
    def delete_node(self, node_id: str) -> None
    def update_access(self, node_ids: list[str]) -> None  # §5.1.1
    def update_status(self, node_id: str, status: NodeStatus, superseded_by: str | None = None) -> None
    def upsert_edges(self, source_id: str, target_ids: list[str]) -> None
    def get_edges(self, node_id: str, direction: str = "outgoing") -> list[str]
    def get_in_degree(self, node_id: str) -> int
    def list_nodes(self, filters: dict) -> list[Node]  # tier, status, etc.
```

### 5. Search Primitives（供 TODO-06 Retrieval Pipeline 使用）

```python
class SQLiteSearchEngine:
    def fts_search(self, query: str, limit: int = 10) -> list[tuple[str, float]]
        """FTS5 MATCH query, returns (node_id, bm25_score) pairs."""

    def vector_search(self, embedding: list[float], limit: int = 10) -> list[tuple[str, float]]
        """sqlite-vec cosine distance query, returns (node_id, distance) pairs."""

    def get_neighbors(self, node_ids: list[str], depth: int = 1) -> list[str]
        """1-degree graph hop via edges table."""
```

### 6. Upsert Embedding

支持单独更新节点的 embedding vector：
```python
def upsert_embedding(self, node_id: str, embedding: list[float]) -> None
```

### 7. Janitor Queries（供 TODO-09 使用）

```python
def find_orphan_candidates(self, tier: str, days_threshold: int) -> list[Node]
    """Find nodes where last_accessed > threshold AND in_degree = 0."""

def find_superseded_for_archival(self, days_threshold: int = 7) -> list[Node]
    """Find superseded nodes eligible for archival (§5.4.4)."""

def count_disputed_nodes(self) -> int
    """Count status='disputed' nodes for warning threshold."""
```

### 8. Schema Migration Support
- 初始版本使用简单的 version table + migration script
- 支持 `synapse rebuild-index` 命令的 drop & recreate 逻辑（仅 `nodes_vec`，FTS5 和 edges 不受影响）

### 9. Connection Management
- 使用连接池或单例模式管理 SQLite 连接
- 支持并发读（WAL mode）和串行写
- 所有写操作包装在事务中

---

## Dependencies
- **TODO-01**: Project structure, config (db path, embedding dimension)
- **TODO-02**: Node model definition

## Blocks
- TODO-05 (File watcher writes to DB)
- TODO-06 (Retrieval pipeline queries DB)
- TODO-09 (Janitor queries DB)
- TODO-15 (Index rebuild)

## Acceptance Criteria
- [x] SQLite 数据库可创建，所有表和索引就位
- [x] `nodes_vec` 已实现可工作的向量索引后备方案（当前为 `python-fallback`；若未来引入 `sqlite-vec` 可无阻切换）
- [x] FTS5 全文搜索返回正确结果
- [x] Vector search 返回正确的余弦距离排序
- [x] Edge CRUD 和 In-Degree 查询工作正常
- [x] WAL mode 启用
- [x] FTS5 自动同步（INSERT/UPDATE/DELETE）
- [x] 单元测试覆盖所有 CRUD 和搜索操作
- [x] Orphan candidate 和 superseded node 查询正确
