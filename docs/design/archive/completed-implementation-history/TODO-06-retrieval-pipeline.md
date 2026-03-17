# TODO-06: Retrieval Pipeline (Hybrid Search + Rerank)

## Status: COMPLETED
## Priority: P0 (Core read path — the brain's "spinal cord")
## Design Doc Section: §4.5 Phase B, §5.1

---

## Summary

实现 Synapse 的核心检索管道——从接收查询到返回 Top 1-3 高密度上下文节点的完整流程。包括 FTS5 + sqlite-vec 混合搜索、RRF 融合、1-度图跳转、CPU Cross-Encoder 重排序、算法衰减、冲突感知评分、以及最终上下文组装。

---

## Detailed Requirements

### 1. Pipeline Overview (§4.5 Phase B: Read/Hop/Rerank)

完整流程（7 步）：

```
Query → Hybrid Search (FTS5 ∥ Vec) → RRF Fusion → 3 Anchor Nodes
  → 1-Degree Graph Hop → ~9 Candidates
  → CPU Cross-Encoder Rerank → Conflict-Aware Scoring → Decay Penalty
  → Top 1-3 Final Context → Return to LLM
```

### 2. Step 1: Hybrid Query (RRF Core)

FTS5 和 sqlite-vec **并行执行**，结果通过 **Reciprocal Rank Fusion (RRF)** 融合：

```
RRF_Score(d) = Σ 1/(k + rank_i(d))
```

- `k = 60`（来自 config.toml `[retrieval] rrf_k`）
- `rank_i` 是每个子系统（keyword vs vector）中的排名
- Top 3 by combined RRF score → **Core Anchor Nodes**

实现细节：
```python
def hybrid_search(self, query: str, query_embedding: list[float], limit: int = 3) -> list[SearchResult]:
    # 1. FTS5 search → ranked list
    fts_results = self.db.fts_search(query, limit=20)
    # 2. Vector search → ranked list  
    vec_results = self.db.vector_search(query_embedding, limit=20)
    # 3. RRF fusion
    rrf_scores = self._compute_rrf(fts_results, vec_results, k=self.config.retrieval.rrf_k)
    # 4. Return top-limit by RRF score
    return sorted(rrf_scores, key=lambda x: x.score, reverse=True)[:limit]
```

### 3. Step 2: 1-Degree Graph Hop

从 3 个 Anchor Nodes 出发，通过 `edges` 表拉取直接链接的邻居节点：

```python
def graph_hop(self, anchor_ids: list[str], max_neighbors: int = 6) -> list[str]:
    """Pull explicitly linked neighbor nodes from anchors.
    Returns up to max_neighbors unique neighbor IDs (excluding anchors themselves)."""
```

结果：~9 个未精炼的候选节点（3 anchors + ~6 neighbors）

### 4. Step 3: CPU Cross-Encoder Rerank (The Gatekeeper)

使用 `bge-reranker-v2-m3`（TODO-04）对 ~9 个候选节点进行交叉评估：

```python
def rerank_candidates(self, query: str, candidates: list[Node]) -> list[tuple[Node, float]]:
    """Cross-evaluate query against candidate documents.
    Returns (node, relevance_score) sorted by score desc.
    Bounded by config.reranker.max_candidates (default 9)."""
```

Reranker 只评估 post-hop 候选集，**不是**整个知识库。

### 5. Step 4: Algorithmic Decay (§5.1)

对每个候选节点应用 **tier-dependent 时间衰减**：

```
Final Score = Reranker_Score × (Tier_Decay_Factor ^ days_since_last_access)
```

| Tier | Decay Factor | Half-life |
|------|-------------|-----------|
| `concept` | 0.90 | ~7 days |
| `decision` | 0.977 | ~30 days |
| `reference` | 0.992 | ~90 days |

```python
def apply_decay(self, node: Node, base_score: float) -> float:
    days = (now() - node.metadata.last_accessed).days
    factor = self.config.decay.get_factor(node.metadata.tier)
    return base_score * (factor ** days)
```

### 6. Step 5: Conflict-Aware Scoring (§5.4.3)

对 `superseded` 和 `disputed` 状态的节点施加惩罚：

| Status | Score Multiplier | Effect |
|--------|:---:|---|
| `active` | 1.0 | 正常排名 |
| `superseded` | 0.1 | 90% 惩罚 — 几乎不可能进入 Top-3 |
| `disputed` | 0.5 | 50% 惩罚 — 可能上浮但携带 `[DISPUTED]` 标签 |

```python
def apply_status_penalty(self, node: Node, score: float) -> float:
    multipliers = {"active": 1.0, "superseded": 0.1, "disputed": 0.5}
    return score * multipliers.get(node.metadata.status, 1.0)
```

### 7. Step 6: Final Context Assembly

- 按最终分数排序
- 取 Top K（默认 3，来自 `config.retrieval.top_k`）
- `disputed` 节点在返回内容中标记 `[DISPUTED]`
- 格式化为 LLM 可消费的上下文块

### 8. Step 7: Access Signal Update (§5.1.1)

**关键**：只有进入最终 Top-K 返回给 LLM 的节点才更新 `last_accessed`：

```python
# Only nodes that survived the full pipeline earn a life extension
self.db.update_access(returned_node_ids)
```

被 Reranker 淘汰的候选节点 **不更新** `last_accessed`——防止 "free-rider" 节点利用 Graph Hop 邻近热门节点白蹭生命值。

### 9. Scaling Note

sqlite-vec 使用暴力 KNN。超过中等规模知识库时，考虑通过 `config.toml → [retrieval] engine = "lancedb"` 切换到 LanceDB（IVF-PQ）以获得亚线性搜索。当前阶段暂不实现 LanceDB，但接口需预留扩展点。

---

## Dependencies
- **TODO-01**: Config (RRF k, top_k, decay factors)
- **TODO-03**: SQLite search primitives (FTS5, vec, edges)
- **TODO-04**: Embedding engine (query embedding), Reranker engine

## Blocks
- TODO-07 (MCP server 调用 retrieval pipeline)

## Acceptance Criteria
- [x] FTS5 + Vec 并行搜索正常返回
- [x] RRF 融合正确计算（验证公式）
- [x] 1-度 Graph Hop 返回正确的邻居节点
- [x] Cross-Encoder Rerank 排序结果合理
- [x] Decay 公式按 tier 正确衰减
- [x] `superseded` 节点得分 ×0.1
- [x] `disputed` 节点得分 ×0.5 且标记 `[DISPUTED]`
- [x] 只有最终返回的节点更新 `last_accessed`
- [x] 端到端集成测试：插入测试数据 → 查询 → 验证返回 Top-K
- [x] 性能目标：当前测试数据下完整 pipeline 正常完成；更大规模性能调优留待后续阶段
