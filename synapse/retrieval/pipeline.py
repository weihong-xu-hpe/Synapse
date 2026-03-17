"""Hybrid retrieval pipeline for Synapse Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Iterable, Sequence

from synapse.config import SynapseConfig
from synapse.embedding import create_embedding_engine, create_reranker_engine
from synapse.interfaces import SearchQuery
from synapse.models import Node, NodeStatus
from synapse.storage import SQLiteNodeStore
from synapse.utils.documents import render_node_document
from synapse.utils.runtime import RuntimePaths, get_runtime_paths


STATUS_MULTIPLIERS: dict[NodeStatus, float] = {
    NodeStatus.ACTIVE: 1.0,
    NodeStatus.SUPERSEDED: 0.1,
    NodeStatus.DISPUTED: 0.5,
}


@dataclass(slots=True, frozen=True)
class RetrievalItem:
    """A fully scored retrieval result returned to callers."""

    node: Node
    score: float
    anchor_score: float
    rerank_score: float
    decay_multiplier: float
    status_multiplier: float
    is_anchor: bool = False
    context_text: str = ""
    markers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class RetrievalResponse:
    """Structured output for the Synapse retrieval API."""

    query: str
    anchors: tuple[RetrievalItem, ...]
    candidates: tuple[RetrievalItem, ...]
    results: tuple[RetrievalItem, ...]
    context: str


@dataclass(slots=True, frozen=True)
class _AnchorCandidate:
    node: Node
    score: float


class RetrievalPipeline:
    """Hybrid search → RRF → graph hop → rerank → decay/status scoring."""

    def __init__(
        self,
        config: SynapseConfig,
        *,
        store: SQLiteNodeStore | None = None,
        runtime_paths: RuntimePaths | None = None,
        embedding_engine=None,
        reranker_engine=None,
        now_fn=None,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths or get_runtime_paths(config)
        self._store = store
        self._owns_store = store is None
        self._embedding_engine = embedding_engine or create_embedding_engine(config.embedding, providers=config.providers)
        self._reranker_engine = reranker_engine or create_reranker_engine(config.reranker, providers=config.providers)
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def close(self) -> None:
        if self._owns_store and self._store is not None:
            self._store.close()
            self._store = None

    def __enter__(self) -> "RetrievalPipeline":
        self._get_store()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def search(
        self,
        query: str | SearchQuery,
        *,
        top_k: int | None = None,
        update_access: bool = True,
    ) -> RetrievalResponse:
        """Run the full retrieval pipeline and update access only for final results."""

        request = query if isinstance(query, SearchQuery) else SearchQuery(text=str(query), top_k=top_k or self.config.retrieval.top_k)
        final_top_k = top_k or request.top_k or self.config.retrieval.top_k
        query_embedding = self._embed_query(request.text)
        anchors = self.hybrid_search(request.text, query_embedding, limit=min(3, self.config.reranker.max_candidates))
        anchor_ids = [anchor.node.id for anchor in anchors]
        neighbor_ids = self.graph_hop(anchor_ids, max_neighbors=max(0, self.config.reranker.max_candidates - len(anchor_ids)))

        anchor_id_set = set(anchor_ids)
        candidate_ids = anchor_ids + [node_id for node_id in neighbor_ids if node_id not in anchor_id_set]
        candidate_nodes = self._get_store().get_nodes(candidate_ids)
        anchor_score_map = {anchor.node.id: anchor.score for anchor in anchors}
        reranked = self.rerank_candidates(request.text, candidate_nodes)

        candidate_items: list[RetrievalItem] = []
        for node, rerank_score in reranked:
            anchor_score = anchor_score_map.get(node.id, 0.0)
            scored_item = self._score_candidate(node, rerank_score, anchor_score=anchor_score, is_anchor=node.id in anchor_score_map)
            candidate_items.append(scored_item)

        final_results = tuple(sorted(candidate_items, key=lambda item: (-item.score, item.node.id))[:final_top_k])
        if update_access and final_results:
            self._get_store().update_access([item.node.id for item in final_results])

        anchor_items = tuple(
            self._score_candidate(anchor.node, anchor.score, anchor_score=anchor.score, is_anchor=True)
            for anchor in anchors
        )
        context = self._assemble_context(final_results)
        return RetrievalResponse(
            query=request.text,
            anchors=anchor_items,
            candidates=tuple(candidate_items),
            results=final_results,
            context=context,
        )

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        *,
        limit: int = 3,
        per_source_limit: int = 20,
    ) -> list[_AnchorCandidate]:
        """Fuse FTS and vector retrieval with reciprocal rank fusion."""

        store = self._get_store()
        fts_results = store.fts_search(query, limit=per_source_limit)
        vector_results = store.vector_search(query_embedding, limit=per_source_limit) if query_embedding else []
        rrf_scores = self.compute_rrf((fts_results, vector_results), k=self.config.retrieval.rrf_k)
        ranked_ids = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        nodes = {node.id: node for node in store.get_nodes([node_id for node_id, _ in ranked_ids])}
        return [_AnchorCandidate(node=nodes[node_id], score=score) for node_id, score in ranked_ids if node_id in nodes]

    def graph_hop(self, anchor_ids: Sequence[str], *, max_neighbors: int = 6) -> list[str]:
        """Return unique 1-degree linked neighbors for anchor nodes."""

        if not anchor_ids or max_neighbors <= 0:
            return []
        return self._get_store().get_linked_neighbors(list(anchor_ids), limit=max_neighbors)

    def rerank_candidates(self, query: str, candidates: Sequence[Node]) -> list[tuple[Node, float]]:
        """Rerank a bounded candidate set using the configured reranker."""

        bounded_candidates = list(candidates)[: self.config.reranker.max_candidates]
        if not bounded_candidates:
            return []
        documents = [render_node_document(node) for node in bounded_candidates]
        ranked = self._reranker_engine.rerank(query, documents, limit=len(documents))
        rerank_scores = {bounded_candidates[index].id: score for index, score in ranked if 0 <= index < len(bounded_candidates)}
        results = [(node, rerank_scores.get(node.id, 0.0)) for node in bounded_candidates]
        results.sort(key=lambda item: (-item[1], item[0].id))
        return results

    @staticmethod
    def compute_rrf(result_sets: Iterable[Sequence[tuple[str, float]]], *, k: int) -> dict[str, float]:
        """Compute reciprocal rank fusion scores from ranked result sets."""

        scores: dict[str, float] = {}
        for results in result_sets:
            for rank, (node_id, _) in enumerate(results, start=1):
                scores[node_id] = scores.get(node_id, 0.0) + (1.0 / (k + rank))
        return scores

    def apply_decay(self, node: Node, base_score: float) -> tuple[float, float]:
        """Apply tier-sensitive time decay to a score."""

        now = self._now_fn()
        last_accessed = node.metadata.last_accessed.astimezone(UTC)
        elapsed_days = max(0.0, (now - last_accessed).total_seconds() / 86_400.0)
        factor = self.config.decay.factor
        multiplier = factor ** elapsed_days
        return base_score * multiplier, multiplier

    def apply_status_penalty(self, node: Node, score: float) -> tuple[float, float]:
        """Apply conflict-aware penalties for disputed or superseded nodes."""

        multiplier = STATUS_MULTIPLIERS.get(node.metadata.status, 1.0)
        return score * multiplier, multiplier

    def _score_candidate(self, node: Node, rerank_score: float, *, anchor_score: float, is_anchor: bool) -> RetrievalItem:
        decayed_score, decay_multiplier = self.apply_decay(node, rerank_score)
        final_score, status_multiplier = self.apply_status_penalty(node, decayed_score)
        markers = ("DISPUTED",) if node.metadata.status is NodeStatus.DISPUTED else ()
        return RetrievalItem(
            node=node,
            score=final_score,
            anchor_score=anchor_score,
            rerank_score=rerank_score,
            decay_multiplier=decay_multiplier,
            status_multiplier=status_multiplier,
            is_anchor=is_anchor,
            context_text=self._format_context_block(node),
            markers=markers,
        )

    def _assemble_context(self, results: Sequence[RetrievalItem]) -> str:
        return "\n\n---\n\n".join(item.context_text for item in results)

    def _format_context_block(self, node: Node) -> str:
        dispute_prefix = "[DISPUTED] " if node.metadata.status is NodeStatus.DISPUTED else ""
        return (
            f"{dispute_prefix}{node.title}\n"
            f"Path: {node.file_path.as_posix()}\n"
            f"Status: {node.metadata.status.value}\n\n"
            f"{node.content.strip()}"
        ).strip()

    def _embed_query(self, query: str) -> list[float] | None:
        try:
            vector = self._embedding_engine.embed(query)
        except (OSError, RuntimeError, ValueError):
            return None
        if not vector:
            return None
        return vector

    def _get_store(self) -> SQLiteNodeStore:
        if self._store is None:
            self._store = SQLiteNodeStore(
                self.runtime_paths.base / "synapse.db",
                embedding_dimension=self.config.embedding.dimension or 0,
            )
        return self._store