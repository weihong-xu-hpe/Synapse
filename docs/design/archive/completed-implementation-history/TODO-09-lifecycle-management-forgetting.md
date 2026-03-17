# TODO-09: Lifecycle Management (Decay + Janitor + Condensation)

## Status: COMPLETED
## Priority: P1 (Knowledge hygiene — prevents concept bloat)
## Design Doc Section: §5 (Full Section)

---

## Summary

实现 Synapse 的四级遗忘机制（Biological Forgetting）——算法衰减、图隔离/驱逐、手动凝缩、以及冲突检测后的清理。包括 CRON Nightly Janitor 自动化守护进程和 `synapse condense` 手动凝缩命令。

---

## Detailed Requirements

### 1. Level 1: Algorithmic Decay (Passive Forgetting) — §5.1

**已在 TODO-06 的 Retrieval Pipeline 中实现**。此处仅需确保衰减配置可由 Janitor 引用。

衰减因子（来自 config.toml）：

| Tier | Decay Factor | Half-life | Janitor Threshold |
|------|-------------|-----------|-------------------|
| `concept` | 0.90 | ~7 days | 7 days |
| `decision` | 0.977 | ~30 days | 30 days |
| `reference` | 0.992 | ~90 days | 90 days |

### 2. Level 2: Graph Isolation & Eviction (Nightly Janitor) — §5.2

实现 CRON 定时任务（"Nightly Janitor"），**纯本地执行，零 API 成本**：

#### 2.1 Orphan Detection

按 tier 分层扫描孤儿候选：

```python
def find_orphan_candidates(self) -> list[Node]:
    orphans = []
    # concept: last_accessed > 7 days AND in_degree = 0
    orphans += self.db.find_orphan_candidates("concept", days=7)
    # decision: last_accessed > 30 days AND in_degree = 0
    orphans += self.db.find_orphan_candidates("decision", days=30)
    # reference: last_accessed > 90 days AND in_degree = 0 AND importance < 0.5
    orphans += self.db.find_orphan_candidates("reference", days=90, max_importance=0.5)
    return orphans
```

#### 2.2 Automatic Eviction

将孤儿节点从 `.synapse/active/` 移动到 `.synapse/.archive/`：

```python
def evict_orphans(self, orphans: list[Node]) -> EvictionReport:
    for node in orphans:
        # 1. Move .md file: active/ → .archive/
        shutil.move(active_path, archive_path)
        # 2. DB cleanup handled by file watcher (delete event)
    return EvictionReport(evicted=len(orphans), ...)
```

**不涉及 LLM**。

#### 2.3 Superseded Node Archival (§5.4.4)

Janitor 附加扫描：

```python
def archive_superseded_nodes(self) -> list[Node]:
    candidates = self.db.find_superseded_for_archival(days_threshold=7)
    safe_to_archive = []
    for node in candidates:
        # Verify superseder still exists and is active
        superseder = self.db.get_node(node.metadata.superseded_by)
        if superseder and superseder.metadata.status == "active":
            safe_to_archive.append(node)
        else:
            # Superseder missing/also-superseded: escalate to human review
            self.log_warning(f"Cannot archive {node.id}: superseder invalid")
    return safe_to_archive
```

#### 2.4 Disputed Node Warning

```python
disputed_count = self.db.count_disputed_nodes()
if disputed_count > 5:
    self.surface_warning(f"{disputed_count} knowledge conflicts need your review.")
```

### 3. Level 3: Manual Condensation (Deep Brain Sleep) — §5.3

实现 `synapse condense` 命令：

#### 3.1 Trigger
- 手动执行：`synapse condense`
- 或可选定时调度（每周）

#### 3.2 Process

```python
def condense(self) -> CondensationReport:
    # 1. Gather recent archive backlog
    archived_nodes = self.scan_archive_backlog()
    
    # 2. Group by semantic similarity (optional clustering)
    groups = self.cluster_archived_nodes(archived_nodes)
    
    # 3. For each group, call Cloud LLM to synthesize
    for group in groups:
        prompt = """
        Evaluate these disconnected, archived memories. 
        Merge overlapping concepts into a single high-level active summary note, 
        and permanently discard trivial observations.
        """
        synthesized = self.call_llm(prompt, nodes=group)
        
    # 4. Write synthesized reference node to active/
    self.write_node(synthesized, tier="reference")
    
    return report
```

#### 3.3 Error Handling (§9.1)
- LLM 调用失败：指数退避重试（最多 3 次）
- 全部失败：记录错误，归档节点安全保留在 `.archive/`
- LLM 产生不良合并：合并摘要保存在原始节点旁（不删除原始节点），frontmatter 记录 `merged_from: [node_a, node_b, ...]`，用户可手动回滚

### 4. CRON Scheduling

#### 4.1 Nightly Janitor 调度
- 默认每天凌晨 3:00 执行
- 通过 Python scheduler（`schedule` 库）或 OS-level cron
- 集成到 `synapse serve` daemon 中（随 server 启动）

#### 4.2 Execution Log
- 每次执行记录到 `.synapse/.logs/janitor.log`
- 包含：执行时间、扫描节点数、驱逐数、归档数、警告数

### 5. Archive Hygiene (§7.3)

`.archive/` 目录清理：
- 超过配置保留期（默认 90 天）的归档文件永久删除
- 通过二级 CRON 调度执行

### 6. Graph Auto-Healing (§5.3.1)

存活节点引用已归档节点时：
- `[[Archived_Node]]` 被视为 "dead link"
- Sync pipeline 在索引时优雅忽略
- 不创建 edge，不报错

**Note**: 此逻辑大部分已在 TODO-05 File Watcher 中实现，此处确保 Janitor 的归档操作不破坏现有链接。

---

## Dependencies
- **TODO-01**: Config (decay factors, janitor thresholds, archive retention)
- **TODO-02**: Node model (file move utilities)
- **TODO-03**: SQLite (orphan queries, superseded queries, disputed count)
- **TODO-05**: File watcher (handles DB cleanup on file move/delete)

## Blocks
- None (standalone lifecycle management)

## Acceptance Criteria
- [x] Nightly Janitor 正确识别各 tier 的孤儿节点
- [x] 孤儿节点从 `active/` 移到 `.archive/`
- [x] Superseded 节点在 7 天后自动归档（验证 superseder 仍存活）
- [x] Disputed 节点 > 5 时产生警告
- [x] `synapse condense` 支持真实的本地可测试归档凝缩流程
- [x] Condensation 失败时安全回退（不丢失数据）
- [x] `.archive/` 超期文件被永久清理
- [x] Janitor 日志记录完整
- [x] 集成测试：创建过期孤儿 → 运行 Janitor → 验证归档
- [x] 集成测试：superseded 节点 → 验证归档 + superseder 仍存在检查
