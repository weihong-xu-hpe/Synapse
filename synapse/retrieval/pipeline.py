"""Hybrid retrieval pipeline for Synapse Phase 4."""

from __future__ import annotations

import logging
import math

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable, Iterable, Sequence

from synapse.config import SynapseConfig
from synapse.embedding import create_embedding_engine, create_reranker_engine
from synapse.interfaces import SearchQuery
from synapse.models import Node, NodeStatus, NodeType
from synapse.storage import SQLiteNodeStore
from synapse.utils.documents import render_node_document
from synapse.utils.runtime import RuntimePaths, get_runtime_paths



LOGGER = logging.getLogger(__name__)

STATUS_MULTIPLIERS: dict[NodeStatus, float] = {
    NodeStatus.ACTIVE: 1.0,
    NodeStatus.SUPERSEDED: 0.1,
    NodeStatus.DISPUTED: 0.5,
}
# Persistent knowledge receives a fixed mild logit-space penalty instead of an
# access-driven decay clock (see RetrievalPipeline.apply_decay).
DECAY_PERSISTENT_PENALTY = -0.2
# Multi-part recall queries are split on this separator for per-chunk dense
# embedding (bridge recall joins messages with "\n---\n").
_QUERY_CHUNK_SEPARATOR = "\n---\n"
# Chunks shorter than this add embedding cost without semantic value.
_MIN_CHUNK_CHARS = 24


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
    # Server-side inject decision (see RetrievalSettings inject_* keys): True
    # when the result clears the logit floor and relative margin. Clients
    # (bridge) inject on this instead of the raw ``score > 0`` heuristic.
    is_injectable: bool = False
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
        result_filter: Callable[[Node], bool] | None = None,
    ) -> RetrievalResponse:
        """Run the full retrieval pipeline and update access only for final results.

        ``result_filter`` drops candidates BEFORE scoring and the top_k cut so
        excluded nodes leave no holes — the next ranked results fill the slots.
        """

        request = query if isinstance(query, SearchQuery) else SearchQuery(text=str(query), top_k=top_k or self.config.retrieval.top_k)
        final_top_k = top_k or request.top_k or self.config.retrieval.top_k
        query_embedding = self._embed_query(request.text)
        # Rerank the full fused top-max_candidates: the anchor cut at 3 was the
        # pipeline's dominant recall bottleneck (the reranker — the only
        # cross-encoder — never saw ranks 4..9 of the fused list). The exclude
        # filter also applies BEFORE fusion so excluded transcripts cannot
        # consume fused candidate slots.
        anchors = self.hybrid_search(
            request.text,
            query_embedding,
            limit=self.config.reranker.max_candidates,
            exclude=result_filter,
        )
        anchor_ids = [anchor.node.id for anchor in anchors]
        # Graph-hop neighbours APPEND after fused hits (slots permitting);
        # they never displace fused hits.
        neighbor_ids = self.graph_hop(anchor_ids, max_neighbors=max(0, self.config.reranker.max_candidates - len(anchor_ids)))

        anchor_id_set = set(anchor_ids)
        candidate_ids = anchor_ids + [node_id for node_id in neighbor_ids if node_id not in anchor_id_set]
        candidate_nodes = self._get_store().get_nodes(candidate_ids)
        if result_filter is not None:
            candidate_nodes = [node for node in candidate_nodes if result_filter(node)]
        anchor_score_map = {anchor.node.id: anchor.score for anchor in anchors}
        reranked = self.rerank_candidates(request.text, candidate_nodes)

        candidate_items: list[RetrievalItem] = []
        for node, rerank_score in reranked:
            anchor_score = anchor_score_map.get(node.id, 0.0)
            scored_item = self._score_candidate(node, rerank_score, anchor_score=anchor_score, is_anchor=node.id in anchor_score_map)
            candidate_items.append(scored_item)

        final_results = tuple(sorted(candidate_items, key=lambda item: (-item.score, item.node.id))[:final_top_k])
        final_results = self._apply_inject_decisions(final_results)
        if update_access and final_results:
            self._get_store().update_access([item.node.id for item in final_results])

        anchor_items = tuple(
            self._score_candidate(anchor.node, anchor.score, anchor_score=anchor.score, is_anchor=True)
            for anchor in anchors
            if result_filter is None or result_filter(anchor.node)
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
        """Fuse FTS and vector retrieval with reciprocal rank fusion.

        Dense leg: long multi-part recall queries (bridge recall concatenates
        the last 3 user messages) dilute a single embedding across topics, so
        the query is ALSO embedded per ``\\n---\\n`` chunk and every chunk
        votes in the fusion. The whole-query embedding participates too.
        """

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        *,
        limit: int = 3,
        per_source_limit: int = 20,
        exclude: Callable[[Node], bool] | None = None,
    ) -> list[_AnchorCandidate]:
        """Fuse FTS and vector retrieval with reciprocal rank fusion.

        Dense leg: long multi-part recall queries (bridge recall concatenates
        the last 3 user messages) dilute a single embedding across topics, so
        the query is ALSO embedded per ``\\n---\\n`` chunk and every chunk
        votes in the fusion. The whole-query embedding participates too.

        ``exclude`` drops nodes BEFORE fusion so filtered transcripts cannot
        consume fused candidate slots (with the default search excluding
        represented transcripts, they otherwise crowd out knowledge).
        """

        store = self._get_store()
        fts_results = store.fts_search(query, limit=per_source_limit)
        if query_embedding is None:
            vector_sets: list[list[tuple[str, float]]] = []
        else:
            vector_sets = [store.vector_search(query_embedding, limit=per_source_limit)]
            for chunk in query.split(_QUERY_CHUNK_SEPARATOR):
                chunk = chunk.strip()
                if len(chunk) < _MIN_CHUNK_CHARS or chunk.casefold() == query.strip().casefold():
                    continue
                chunk_embedding = self._embed_query(chunk)
                if chunk_embedding:
                    vector_sets.append(store.vector_search(chunk_embedding, limit=per_source_limit))
        if exclude is not None:
            # ``exclude`` follows the service result_filter convention: it
            # returns True for nodes to KEEP.
            def _keep(node_id: str) -> bool:
                node = store.get_node(node_id)
                return node is not None and exclude(node)

            fts_results = [(node_id, score) for node_id, score in fts_results if _keep(node_id)]
            vector_sets = [
                [(node_id, score) for node_id, score in vs if _keep(node_id)]
                for vs in vector_sets
            ]
        ranked_ids = self.fuse_rankings((fts_results, *vector_sets), k=self.config.retrieval.rrf_k, limit=limit)
        ranked_id_list = [node_id for node_id, _ in ranked_ids]
        nodes = {node.id: node for node in store.get_nodes(ranked_id_list)}
        return [_AnchorCandidate(node=nodes[node_id], score=score) for node_id, score in ranked_ids if node_id in nodes]

    @staticmethod
    def fuse_rankings(
        result_sets: Iterable[Sequence[tuple[str, float]]],
        *,
        k: int,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Fuse ranked lists by keeping each item's BEST per-list RRF score.

        Max-fusion instead of sum: multi-chunk recall queries embed each user
        message separately, and generic transcripts appear in every chunk list
        (consensus noise) while a specific rank-1 match appears in only one.
        Sum-fusion lets consensus beat specificity; max keeps every leg's
        strongest evidence. For a single list this is identical to plain RRF.
        """

        best: dict[str, float] = {}
        for results in result_sets:
            for rank, (node_id, _) in enumerate(results, start=1):
                score = 1.0 / (k + rank)
                if node_id not in best or score > best[node_id]:
                    best[node_id] = score
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:limit]

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
        reranker = self._reranker_engine
        backend = getattr(reranker, "backend_name", "")
        documents = [render_node_document(node) for node in bounded_candidates]
        ranked = reranker.rerank(query, documents, limit=len(documents))
        if backend == "remote_api" and not reranker.is_available():
            # Deterministic fallback returned all-zero scores: reordering by
            # them would shuffle candidates arbitrarily. Keep the incoming
            # (RRF) order and log the degradation.
            LOGGER.warning("Reranker degraded; keeping RRF order for %d candidates", len(bounded_candidates))
            return [(node, 0.0) for node in bounded_candidates]
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
        """Apply tier-sensitive time decay to a score.

        Sign-safe: decay is applied as an additive penalty in logit space
        (``ln(decay_factor^days)``), so stale/superseded irrelevant results can
        never be moved UP toward zero, and persistent knowledge is not punished
        for simply not being accessed recently (decay targets transient
        material; persistent knowledge keeps a fixed mild penalty after a
        grace period).
        """

        now = self._now_fn()
        last_accessed = node.metadata.last_accessed.astimezone(UTC)
        elapsed_days = max(0.0, (now - last_accessed).total_seconds() / 86_400.0)
        factor = self.config.decay.factor
        if node.metadata.type is NodeType.PERSISTENT:
            # Knowledge: no access-driven decay clock (access frequency is a
            # popularity signal, not a freshness signal). A fixed mild penalty
            # keeps fresh, just-written context ahead of long-tail knowledge.
            penalty = DECAY_PERSISTENT_PENALTY
        else:
            penalty = math.log(factor) * elapsed_days
        return base_score + penalty, math.exp(penalty)

    def apply_status_penalty(self, node: Node, score: float) -> tuple[float, float]:
        """Apply conflict-aware penalties for disputed or superseded nodes.

        Additive in logit space (see apply_decay): a negative penalty on
        superseded/disputed nodes shifts their logits strictly down, which is
        monotone in relevance — unlike multiplication, it cannot turn a very
        negative (irrelevant) logit into a competitive one.
        """

        multiplier = STATUS_MULTIPLIERS.get(node.metadata.status, 1.0)
        penalty = math.log(multiplier) if multiplier > 0 else -1000.0
        return score + penalty, multiplier

    def _apply_inject_decisions(self, results: tuple[RetrievalItem, ...]) -> tuple[RetrievalItem, ...]:
        """Set ``is_injectable`` on final results (server-side inject decision).

        Rule (config [retrieval]): a result is injectable when
        ``rerank_score >= inject_logit_floor`` AND
        ``rerank_score >= best_logit - inject_relative_margin``. The floor
        catches degraded reranker outputs (long conversational queries depress
        absolute logits, but so do genuinely irrelevant ones); the relative
        margin keeps order-calibrated results injectable even when the whole
        query sits far below zero.
        """

        settings = self.config.retrieval
        if not results:
            return results
        best_logit = max(item.rerank_score for item in results)
        floor = settings.inject_logit_floor
        margin = settings.inject_relative_margin
        decided: list[RetrievalItem] = []
        for item in results:
            injectable = (
                item.rerank_score >= floor
                and item.rerank_score >= best_logit - margin
            )
            decided.append(
                RetrievalItem(
                    node=item.node,
                    score=item.score,
                    anchor_score=item.anchor_score,
                    rerank_score=item.rerank_score,
                    decay_multiplier=item.decay_multiplier,
                    status_multiplier=item.status_multiplier,
                    is_anchor=item.is_anchor,
                    is_injectable=injectable,
                    context_text=item.context_text,
                    markers=item.markers,
                )
            )
        return tuple(decided)

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
        engine = self._embedding_engine
        backend = getattr(engine, "backend_name", "")
        if backend == "unavailable":
            return None
        try:
            vector = engine.embed(query)
        except (OSError, RuntimeError, ValueError):
            return None
        if not vector:
            return None
        if backend == "remote_api" and not engine.is_available():
            # Remote provider degraded this batch to deterministic hash
            # vectors; searching with them would silently poison the vector
            # leg. Skip it — FTS + rerank still serve the query.
            LOGGER.warning("Skipping vector search leg: remote embedding provider degraded")
            return None
        return vector

    def _get_store(self) -> SQLiteNodeStore:
        if self._store is None:
            self._store = SQLiteNodeStore(
                self.runtime_paths.base / "synapse.db",
                embedding_dimension=self.config.embedding.dimension or 0,
            )
        return self._store