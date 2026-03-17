# TODO-12: Error Handling, Resilience & Observability

## Status: COMPLETED (Phase 4 startup/delta sync hooks integrated)
## Priority: P1 (System reliability)
## Design Doc Section: §9, §4.6

---

## Summary

实现 Synapse 的错误处理、故障恢复、一致性保证、以及可观测性基础设施。包括完整的 Failure Mode 处理矩阵、Index Rebuild 命令、结构化日志、以及系统一致性不变量。

> Phase 3 delivered the SQLite / rebuild / health / integrity core. Phase 4 completes the remaining `delta sync` and `startup sync` integration through the sync manager and startup checks.

---

## Detailed Requirements

### 1. Failure Mode Matrix (§9.1)

为以下故障场景实现处理逻辑：

| 故障 | 影响 | 检测 | 恢复方式 |
|------|------|------|---------|
| **MCP daemon crash** | 搜索/写入不可用 | launchd/systemd | 自动重启 + 启动时 delta sync |
| **Janitor file-move 失败** | 过期节点未归档 | Non-zero exit code | 下次重试。节点安全留在 `active/` |
| **Condensation LLM 调用失败** | 归档碎片未合成 | Exit code + retry counter | 指数退避重试（max 3）|
| **Condensation 产生坏合并** | LLM 幻觉 | Post-merge validation | 原始节点保留在 `.archive/`，可手动回滚 |
| **并发写入冲突** | 同文件同时编辑 | rapid successive writes | Last-write-wins + `.conflict` 副本 |
| **SQLite DB 损坏** | 搜索不可用 | `PRAGMA integrity_check` | `synapse rebuild-index` 从 Markdown 重建 |
| **Embedding 模型加载失败** | 向量搜索不可用 | Model file missing | Graceful degradation: FTS5 + graph hop 继续工作 |
| **Embedding 模型切换** | 向量索引过期 | Config change detected | Auto-trigger `rebuild-index` |
| **冲突检测 LLM 误判** | 节点被错误标记 | 用户发现 | 手动编辑 frontmatter 恢复 |
| **Disputed 节点积累** | 检索质量下降 | Janitor 计数 | > 5 时发出警告 |
| **磁盘满** | 无法写入新节点 | OS error | MCP 返回错误。SQLite 事务回滚，.md 写入使用 atomic rename |

### 2. Index Rebuild Command (§4.6)

实现 `synapse rebuild-index` 命令：

```bash
synapse rebuild-index --brain-dir ./.synapse/active/
```

重建流程：
1. 对 SQLite DB 执行 `PRAGMA integrity_check`
2. 根据配置确定是否需要重建向量表（模型切换时 drop & recreate `nodes_vec`）
3. 扫描所有 `.md` 文件
4. 对每个文件：解析 frontmatter → 重新计算 embedding → 重建 FTS5 + vec + edges
5. FTS5 和 edges 表在模型切换时**不受影响**（模型无关）

特性：
- **幂等**：安全重复执行
- **进度报告**：显示处理进度 (e.g., `[42/150] Processing rate_limiting.md...`)
- **并行可选**：大知识库时使用 batch embedding 加速

### 3. Consistency Guarantees (§9.2)

#### 3.1 Core Invariant
> **Markdown 文件系统永远是 canonical state。所有其他存储（SQLite, `.archive/`）是派生或次要的。**

#### 3.2 Write Atomicity
```python
def atomic_write(filepath: Path, content: str) -> None:
    """Write to temp file, then atomic rename."""
    tmp = filepath.with_suffix('.tmp')
    tmp.write_text(content, encoding='utf-8')
    tmp.rename(filepath)  # atomic on same filesystem
```

#### 3.3 SQLite WAL Mode
- 启动时设置 `PRAGMA journal_mode=WAL`
- 读写不互相阻塞
- Crash 中途事务自动回滚

#### 3.4 Idempotent Sync
- File watcher sync loop 是幂等的
- 对同一文件重复处理产生相同 DB 状态
- 任何故障后安全重执行

### 4. Startup Integrity Checks

`synapse serve` 启动时执行：

```python
def startup_checks(self) -> StartupReport:
    # 1. Verify .synapse/ directory exists and has correct permissions (700)
    # 2. Check SQLite DB integrity: PRAGMA integrity_check
    # 3. If DB corrupted or missing: auto-trigger rebuild-index
    # 4. Check embedding model availability
    # 5. If model unavailable: log warning, continue with FTS5-only mode
    # 6. Delta sync: compare file mtimes vs DB last_modified (§9.1 row 1)
    # 7. Check for config changes (e.g., embedding model switch)
    # 8. If model switched: auto-trigger rebuild-index
```

### 5. Observability (§9.3)

#### 5.1 Structured Logs

所有组件输出结构化日志到 `.synapse/.logs/`：

```
.logs/
  mcp-daemon.log       # MCP server 请求/响应
  file-watcher.log     # Sync 事件, 冲突
  janitor.log          # Nightly 执行结果, LLM 调用
  audit.log            # 所有 Cloud 数据传输
```

#### 5.2 Log Format
```json
{
  "timestamp": "2026-03-01T10:30:00Z",
  "level": "INFO",
  "component": "retrieval",
  "message": "Hybrid search completed",
  "data": {
    "query_length": 42,
    "fts_results": 15,
    "vec_results": 20,
    "rrf_anchors": 3,
    "hop_candidates": 9,
    "final_results": 3,
    "latency_ms": 1250
  }
}
```

#### 5.3 Log Rotation
- 7 天保留
- 每文件最大 50MB
- 使用 Python `logging.handlers.RotatingFileHandler`
- 可在 `config.toml → [logging]` 配置

### 6. Health Check Endpoint

`GET /api/v1/health` 返回：

```json
{
  "status": "healthy",
  "components": {
    "sqlite": "ok",
    "embedding_model": "ok",
    "reranker_model": "ok",
    "file_watcher": "running",
    "janitor": "last_run: 2026-03-01T03:00:00Z"
  },
  "stats": {
    "total_nodes": 150,
    "active_nodes": 130,
    "superseded_nodes": 12,
    "disputed_nodes": 3,
    "archived_nodes": 45
  }
}
```

---

## Dependencies
- **TODO-01**: Config, logging infrastructure, CLI
- **TODO-02**: Node parser (for rebuild-index)
- **TODO-03**: SQLite (integrity check, WAL mode)
- **TODO-04**: Embedding engine (for rebuild-index re-embedding)
- **TODO-05**: File watcher (delta sync, startup sync)

## Blocks
- None (cross-cutting concern, integrates with all modules)

## Acceptance Criteria
- [x] `synapse rebuild-index` 从 Markdown 完全重建 SQLite
- [x] Rebuild 幂等、安全重复执行
- [x] Rebuild 显示进度
- [x] Startup integrity checks 全部通过（SQLite / embedding / rebuild trigger core）
- [x] DB 损坏时自动触发 rebuild
- [x] 模型切换时自动触发 rebuild
- [x] Atomic write 实现（`.tmp` → `rename`）
- [x] WAL mode 启用验证
- [x] 结构化日志写入 `.logs/` 各文件
- [x] 日志轮转正常工作（7 天 / 50MB）
- [x] Health check 返回正确的系统状态
- [x] Graceful degradation: embedding 不可用时 FTS5 继续工作
- [x] 磁盘满时不产生半写文件
- [x] TODO-05 最终集成：真实 sync manager 驱动的 delta sync / startup sync
