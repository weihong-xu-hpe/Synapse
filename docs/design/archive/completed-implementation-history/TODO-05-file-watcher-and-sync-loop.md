# TODO-05: File Watcher & Sync Loop

## Status: COMPLETED
## Priority: P1 (Keeps DB in sync with Markdown source of truth)
## Design Doc Section: §4.4, §4.5 Phase A

---

## Summary

实现文件系统监听器（File Watcher），监控 Markdown 目录的变更（创建/修改/删除），并自动同步到 SQLite 数据库（包括 nodes 表、FTS5、nodes_vec、edges 表）。这是 "Markdown as Source of Truth + SQLite as Derived Index" 架构的核心同步机制。

---

## Detailed Requirements

### 1. File Watcher 实现

使用 `watchdog` 库监控 `.synapse/active/` 目录：

- **监控事件**：
  - 文件创建（`FileCreatedEvent`）
  - 文件修改（`FileModifiedEvent`）
  - 文件删除（`FileDeletedEvent`）
  - 文件移动（`FileMovedEvent`）
- **过滤器**：只监控 `*.md` 文件
- **递归监控**：支持子目录

### 2. Sync Loop 逻辑（§4.4）

#### On File Create/Modify:
1. Parse frontmatter + body → 构建 `Node` 对象（使用 TODO-02 的解析器）
2. Upsert into `nodes` table（包括 `tier` 字段）
3. 更新 FTS5 触发器（自动通过 SQL trigger）
4. 计算 embedding（via TODO-04 的 `EmbeddingEngine`，CPU ONNX INT8）
5. Upsert into `nodes_vec`
6. 解析 `[[links]]`（使用 TODO-02 的 `extract_wiki_links()`）
7. Upsert into `edges` table

**对于文件修改**，同时更新 `last_accessed = file_mtime`（§5.1.1 Human Edit 信号）

#### On File Delete/Move:
- 从 `nodes`、`nodes_fts`、`nodes_vec` 删除
- Dangling edges 通过 `ON DELETE CASCADE` / `SET NULL` 自动清理

### 3. Debouncing（§4.4）

批量处理文件事件：
- **500ms 窗口**：在 500ms 内聚合同一文件的多次变更事件
- **防抖动**：避免 Git checkout、批量编辑等场景下的 thrashing
- 实现方式：事件队列 + 定时消费

### 4. Asynchronous Write (Fire-and-Forget) — §4.5 Phase A

- Agent 通过 MCP 写入 Markdown 文件后立即返回
- File Watcher 在后台异步完成 embedding 计算和边链接
- **不阻塞写入方**

### 5. Delta Sync on Startup（§9.1）

MCP daemon 启动时执行增量同步：
- 对比文件 `mtime` 与数据库 `last_modified` 
- 仅处理变更文件，避免全量重建
- 处理启动前的离线变更（如用户在 Obsidian 中编辑了文件）

### 6. Concurrent Write Handling（§9.1）

- **Last-write-wins**：与 Git 一致的策略
- SQLite upsert 是幂等的
- 关键冲突场景（IDE agent + Obsidian 同时编辑同一文件）：创建 `.conflict` 副本供人工解决

### 7. Graph Auto-Healing（§5.3.1）

- 如果存活节点引用了已归档节点 `[[Archived_Node]]`
- Sync pipeline 在索引时优雅忽略（不报错，不创建 edge）
- 类似 Obsidian 的 "dead link" 处理

### 8. Error Resilience

- Embedding 计算失败：跳过该节点的向量更新，记录 warning，其他索引照常更新
- 文件解析失败（格式错误）：跳过，记录 error，不影响其他文件
- 所有操作幂等：重复处理同一文件产生相同结果

---

## Dependencies
- **TODO-01**: Config, logging
- **TODO-02**: Node parser, wiki-link extractor
- **TODO-03**: SQLite CRUD, upsert operations
- **TODO-04**: Embedding engine

## Blocks
- TODO-07 (MCP server 需要 file watcher 保持索引同步)

## Acceptance Criteria
- [x] 新建 `.md` 文件后可通过 sync manager 快速进入 SQLite 索引
- [x] 修改文件后 FTS5、vector、edges 均自动更新
- [x] 删除文件后所有相关索引清理干净
- [x] 500ms debouncing 正常工作（批量操作不 thrash）
- [x] 启动时 delta sync 正确处理离线变更
- [x] `last_accessed` 在文件修改时更新（human edit signal）
- [x] Dead link 被优雅忽略
- [x] Embedding 失败不阻断其他同步
- [x] 集成测试：创建/修改/删除 → 验证 DB 状态
