# TODO-08: Write Path & Knowledge Conflict Detection

## Status: COMPLETED
## Priority: P1 (Write flow orchestration with conflict detection)
## Design Doc Section: §6, §5.4

---

## Summary

实现 Synapse 的完整写入路径——从会话终止到知识图谱集成的全流程。包括上下文压缩、分层节点创建、Retrieval-Augmented Linking、以及核心的知识冲突检测机制（Supersession Chain）。

---

## Detailed Requirements

### 1. Write Path Flow (§6 — 6 Steps)

完整写入流程：

```
Session End → Context Compression → Tiered Node Creation
  → Retrieval-Augmented Linking → Conflict Detection
  → Node Instantiation → Graph Integration (via File Watcher)
```

### 2. Step 1-2: Context Compression & Tiered Node Creation

Cloud LLM 将会话总结为分层节点：
- 简单定义/术语 → `concept` 节点（≤500 words）
- 设计决策+理由 → `decision` 节点（≤2,000 words）
- 完整架构 → `reference` 节点（≤3,500 words）
- 超过 3,500 words 的总结 **必须** 拆分为多个互联节点（via `[[wiki-links]]`）

**Implementation**: 这部分主要由 Cloud LLM 完成（prompt engineering），本地系统负责：
- 提供 write_node MCP tool（TODO-07）
- 验证字数限制（TODO-02 word count validation）
- 验证 tier 分配合理性

### 3. Step 3: Retrieval-Augmented Linking

写入前，LLM 调用 `search_existing_nodes(query="{topic}")` 查找相关现有节点：
- 本地返回现有节点列表 + 相似度分数
- LLM 在新节点 body 中嵌入 `[[existing_node_id]]` 链接

### 4. Step 4: Conflict Detection (§5.4 — THE CORE)

当 `search_existing_nodes` 返回的节点中有 `similarity > 0.80` 的高相似节点时，触发冲突检测：

#### 4.1 The Supersession Chain Model (§5.4.1)

```
高相似节点发现 → LLM Judge 判断关系 → 三种结果之一：
  - SUPERSEDE: 新节点替代旧节点
  - COMPLEMENT: 互补共存，互相链接
  - CONFLICT_UNCLEAR: 标记为 disputed，等待人工审查
```

#### 4.2 LLM Judge Prompt (§5.4.2)

```
The following existing node(s) have high semantic similarity (>0.80) to the
new knowledge you are about to save. Classify the relationship:

NEW: [new node content summary]
EXISTING: [existing node S content summary]

Choose ONE:
A) SUPERSEDE — New knowledge corrects, updates, or replaces the existing.
B) COMPLEMENT — Both are valid. Different aspects of the same topic.
C) CONFLICT_UNCLEAR — They contradict but you cannot determine which is correct.

Respond with: {action: "SUPERSEDE"|"COMPLEMENT"|"CONFLICT_UNCLEAR",
              reasoning: "one-line explanation"}
```

**关键**：此 prompt 附加到已有的上下文压缩/链接 LLM 调用中，**零额外 API 调用成本**。

#### 4.3 SUPERSEDE 处理 (§5.4.3)

**新节点**（superseder）：
```yaml
status: active
supersedes: [old_node_id]
```
Body 添加：
```markdown
> **Supersedes**: [[old_node_id]] — [reason]
```

**旧节点**（superseded）自动更新：
```yaml
status: superseded
superseded_by: new_node_id
```
Body 前置：
```markdown
> ⚠️ **SUPERSEDED** by [[new_node_id]] on YYYY-MM-DD.
> This node is retained for historical context but its conclusions are outdated.
```

#### 4.4 COMPLEMENT 处理
- 正常写入新节点
- 新旧节点互相添加 `[[wiki-links]]`

#### 4.5 CONFLICT_UNCLEAR 处理
- 两个节点都写入
- 两个节点的 `status` 都设为 `disputed`
- 等待人工审查

### 5. Step 5-6: Node Instantiation & Graph Integration

- 通过 `write_node` MCP tool 写入 `.md` 文件
- File Watcher（TODO-05）自动检测并同步到 SQLite

### 6. Edge Cases (§5.4.5)

| 场景 | 处理方式 |
|------|---------|
| **链式替代** (A→B→C) | 支持。`superseded_by` 单值（最新替代者）。A/B 都被惩罚，C 是当前权威。 |
| **部分替代** | LLM 返回 `COMPLEMENT` 并创建针对性链接 |
| **误判替代** | 可恢复：用户手动编辑 frontmatter 恢复 `status: active` |
| **Disputed 累积** | Janitor 报告数量，>5 时警告（TODO-09 实现） |
| **离线写入** | 跳过冲突检测，保存为 `status: active`。联网后后台对账 |

### 7. Offline Fallback (§7.4)

Cloud LLM 不可用时：
- 原始会话笔记保存为 `type: draft` 节点——未总结、未链接
- 后台队列在连接恢复后处理
- 文件标记为 draft 以区分

### 7.1 Implemented Pragmatic Fallback

当前代码库没有新增 `draft` 枚举，而是采用**最小侵入式 draft-like 持久化**：

- 新节点保存为 `type: transient`
- 新节点保存为 `status: disputed`
- frontmatter `tags` 自动追加：`draft`, `offline_fallback`, `pending_conflict_judgement`
- Markdown 仍写入 canonical `active/<node_id>.md`，并通过现有 sync 流程进入 SQLite

这样可以在**不扩展现有状态/类型枚举**的前提下，把“未完成冲突裁决”的节点明确标记出来，并降低其在检索中的权重（复用 TODO-06 的 `disputed` 惩罚逻辑）。

---

## Dependencies
- **TODO-01**: Config (similarity threshold)
- **TODO-02**: Node model, frontmatter writer, supersession banners
- **TODO-03**: SQLite (status update, edge upsert)
- **TODO-04**: Embedding engine (for similarity check)
- **TODO-07**: MCP tools (write_node, search_existing_nodes, update_node_status)

## Blocks
- None (end of write path)

## Acceptance Criteria
- [x] 写入路径完整 6 步可执行
- [x] `similarity > 0.80` 时触发冲突检测
- [x] SUPERSEDE 正确更新新旧节点的 frontmatter 和 body
- [x] COMPLEMENT 正确添加双向 wiki-links
- [x] CONFLICT_UNCLEAR 正确标记两个节点为 `disputed`
- [x] 链式替代 (A→B→C) 正确处理
- [x] 离线 fallback 以 draft-like 本地表示安全持久化
- [x] LLM Judge prompt 不产生额外 API 调用
- [x] 集成测试：写入相似节点 → 验证冲突检测结果
