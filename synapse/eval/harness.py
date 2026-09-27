"""Evaluation harness: run golden queries through the real retrieval pipeline."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from synapse.config import SynapseConfig
from synapse.eval.golden import GoldenQuery, RelevantDoc, slice_of
from synapse.retrieval import RetrievalPipeline
from synapse.utils.runtime import RuntimePaths, get_runtime_paths


@dataclass(slots=True)
class QueryResult:
    """Per-query evaluation outcome."""

    query_id: str
    slice: str
    latencies_ms: list[float] = field(default_factory=list)
    results_per_run: list[list[str]] = field(default_factory=list)
    positive_scores_per_run: list[list[float]] = field(default_factory=list)
    inject_flags_per_run: list[list[bool]] = field(default_factory=list)
    recall5: float = 0.0
    mrr10: float = 0.0
    # Same metrics but counting only results the bridge would inject (score > 0).
    recall5_positive: float = 0.0
    mrr10_positive: float = 0.0
    # Bridge-effective metric: must nodes the server's inject decision admits.
    recall5_inject: float = 0.0
    top1_relevant: bool = False
    lexical_hits: int = 0
    positive_count: int = 0  # score > 0 among returned results (last run)
    inject_count: int = 0  # inject=true among returned results (last run)
    returned_count: int = 0
    # For expect_no_result queries: True when NO returned result had score > 0.
    correctly_empty: bool = True
    # Same gate but on the inject decision (server-side threshold).
    correctly_empty_inject: bool = True


@dataclass(slots=True)
class EvalReport:
    """Aggregated evaluation report."""

    per_query: list[QueryResult] = field(default_factory=list)
    created_at: str = ""
    config_path: str = ""
    # Per query id: number of labels dropped (archived or missing at eval time).
    stale_labels: dict[str, int] = field(default_factory=dict)
    # Total number of labels dropped across all queries.
    stale_labels_total: int = 0
    # Query ids whose labels were ALL stale — reported, not scored as misses.
    queries_without_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def _percentile(values: list[float], pct: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            k = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
            return ordered[k]

        slices: dict[str, list[QueryResult]] = defaultdict(list)
        no_label = set(self.queries_without_labels)
        for result in self.per_query:
            slices[result.slice].append(result)
        overall: dict[str, Any] = {}
        slice_metrics: dict[str, Any] = {}
        for name, results in sorted(slices.items()):
            ranked_results = [r for r in results if r.query_id not in no_label]
            metrics = {
                "n": len(results),
                "recall@5": round(_mean([r.recall5 for r in ranked_results]), 4),
                "recall@5_pos": round(_mean([r.recall5_positive for r in ranked_results]), 4),
                "recall@5_inject": round(_mean([r.recall5_inject for r in ranked_results]), 4),
                "mrr@10": round(_mean([r.mrr10 for r in ranked_results]), 4),
                "mrr@10_pos": round(_mean([r.mrr10_positive for r in ranked_results]), 4),
                "top1_relevance": round(_mean([1.0 if r.top1_relevant else 0.0 for r in ranked_results]), 4),
                "lexical_zero_hit_rate": round(
                    _mean([1.0 if r.lexical_hits == 0 else 0.0 for r in results]), 4
                ),
                "positive_score_rate": round(
                    _mean(
                        [
                            (r.positive_count / r.returned_count) if r.returned_count else 0.0
                            for r in results
                        ]
                    ),
                    4,
                ),
                "inject_rate": round(
                    _mean(
                        [
                            (r.inject_count / r.returned_count) if r.returned_count else 0.0
                            for r in results
                        ]
                    ),
                    4,
                ),
                "p50_latency_ms": round(_percentile([l for r in results for l in r.latencies_ms], 50), 1),
                "p95_latency_ms": round(_percentile([l for r in results for l in r.latencies_ms], 95), 1),
                "stale_labels": sum(self.stale_labels.get(r.query_id, 0) for r in results),
            }
            slice_metrics[name] = metrics
        # 'overall' covers all queries with labels (excludes expect_no_result and
        # fully-stale-label queries for ranking metrics but keeps their
        # latency/positive-score signal).
        no_label = set(self.queries_without_labels)
        labeled = [r for r in self.per_query if r.slice != "none" and r.query_id not in no_label]
        empties = [r for r in self.per_query if r.slice == "none"]
        all_lat = [l for r in self.per_query for l in r.latencies_ms]
        overall = {
            "n": len(self.per_query),
            "recall@5": round(_mean([r.recall5 for r in labeled]), 4),
            "recall@5_pos": round(_mean([r.recall5_positive for r in labeled]), 4),
            "recall@5_inject": round(_mean([r.recall5_inject for r in labeled]), 4),
            "mrr@10": round(_mean([r.mrr10 for r in labeled]), 4),
            "mrr@10_pos": round(_mean([r.mrr10_positive for r in labeled]), 4),
            "top1_relevance": round(_mean([1.0 if r.top1_relevant else 0.0 for r in labeled]), 4),
            "lexical_zero_hit_rate": round(
                _mean([1.0 if r.lexical_hits == 0 else 0.0 for r in self.per_query]), 4
            ),
            "positive_score_rate": round(
                _mean(
                    [
                        (r.positive_count / r.returned_count) if r.returned_count else 0.0
                        for r in self.per_query
                    ]
                ),
                4,
            ),
            "inject_rate": round(
                _mean(
                    [
                        (r.inject_count / r.returned_count) if r.returned_count else 0.0
                        for r in self.per_query
                    ]
                ),
                4,
            ),
            "no_result_correct_rate": round(
                _mean([1.0 if r.correctly_empty else 0.0 for r in empties]), 4
            )
            if empties
            else None,
            "no_result_correct_rate_inject": round(
                _mean([1.0 if r.correctly_empty_inject else 0.0 for r in empties]), 4
            )
            if empties
            else None,
            "p50_latency_ms": round(_percentile(all_lat, 50), 1),
            "p95_latency_ms": round(_percentile(all_lat, 95), 1),
            "stale_labels": self.stale_labels_total,
        }
        return {
            "created_at": self.created_at,
            "config_path": self.config_path,
            "stale_labels": dict(self.stale_labels),
            "stale_labels_total": self.stale_labels_total,
            "queries_without_labels": list(self.queries_without_labels),
            "overall": overall,
            "slices": slice_metrics,
            "queries": [
                {
                    "id": r.query_id,
                    "slice": r.slice,
                    "recall@5": r.recall5,
                    "recall@5_pos": r.recall5_positive,
                    "recall@5_inject": r.recall5_inject,
                    "mrr@10": r.mrr10,
                    "mrr@10_pos": r.mrr10_positive,
                    "top1_relevant": r.top1_relevant,
                    "lexical_hits": r.lexical_hits,
                    "positive": r.positive_count,
                    "inject": r.inject_count,
                    "returned": r.returned_count,
                    "latency_ms": round(_mean(r.latencies_ms), 1),
                    "results": r.results_per_run[-1] if r.results_per_run else [],
                    "correctly_empty": r.correctly_empty,
                    "correctly_empty_inject": r.correctly_empty_inject,
                }
                for r in self.per_query
            ],
        }


def _lexical_hit_count(pipeline: RetrievalPipeline, query: str) -> int:
    """Number of nodes the lexical (FTS) leg alone returns."""

    store = pipeline._get_store()
    return len(store.fts_search(query, limit=20))


def resolve_labels(
    queries: list[GoldenQuery],
    pipeline: RetrievalPipeline,
) -> tuple[dict[str, GoldenQuery], dict[str, int], list[str]]:
    """Resolve golden label node ids against live node state.

    Superseded labels follow ``superseded_by`` to the terminal active node
    (cycle-safe). Archived or missing nodes are dropped and counted as stale.
    A query whose labels are ALL stale is reported via ``queries_without_labels``
    and excluded from ranking metrics rather than scored as a miss.
    """

    from synapse.models.node import NodeStatus

    store = pipeline._get_store()
    resolved: dict[str, GoldenQuery] = {}
    stale_per_query: dict[str, int] = {}
    no_label_queries: list[str] = []

    def _terminal(node_id: str) -> str | None:
        seen: set[str] = set()
        current = node_id
        while current and current not in seen:
            seen.add(current)
            node = store.get_node(current)
            # Archived nodes are removed from the index; missing == archived.
            if node is None:
                return None
            if node.metadata.status == NodeStatus.SUPERSEDED and node.metadata.superseded_by:
                current = node.metadata.superseded_by
                continue
            return current
        return None  # cycle or dangling chain

    for gq in queries:
        if gq.expect_no_result or not gq.relevant:
            resolved[gq.id] = gq
            continue
        stale_count = 0
        resolved_docs = []
        for doc in gq.relevant:
            terminal = _terminal(doc.node_id)
            if terminal is None:
                stale_count += 1
                continue
            resolved_docs.append(RelevantDoc(node_id=terminal, tier=doc.tier))
        # Dedupe: two labels may resolve to the same terminal node.
        seen_ids: set[str] = set()
        unique_docs = []
        for doc in resolved_docs:
            if doc.node_id not in seen_ids:
                seen_ids.add(doc.node_id)
                unique_docs.append(doc)
        stale_per_query[gq.id] = stale_count
        if not unique_docs:
            no_label_queries.append(gq.id)
        resolved[gq.id] = replace(gq, relevant=tuple(unique_docs))
    return resolved, stale_per_query, no_label_queries


def run_eval(
    config: SynapseConfig,
    queries: list[GoldenQuery],
    *,
    top_k: int = 5,
    runs: int = 1,
    runtime_paths: RuntimePaths | None = None,
) -> EvalReport:
    """Run all golden queries through the real retrieval pipeline.

    Labels are matched against ``must`` tier ids for recall/MRR/top-1 (a ``may``
    hit does not count against the query but is not required). Latency excludes
    nothing — it is the full pipeline wall time per run.
    """

    from datetime import UTC, datetime

    paths = runtime_paths or get_runtime_paths(config)
    report = EvalReport(
        created_at=datetime.now(UTC).isoformat(),
        config_path=str(config.config_path),
    )

    from synapse.okf.transcripts import is_distilled_current, is_session_transcript

    def _default_include(node) -> bool:
        # Mirror the service default: distilled-current transcripts (represented
        # OR zero-item) are excluded; only not-yet-distilled remain searchable.
        if is_session_transcript(node) and is_distilled_current(node):
            return False
        return True

    import inspect

    # Eval searches must not pollute usage stats: pass source="eval" when the
    # pipeline supports tagging (shared contract with SynapseServerService).
    _pass_source = "source" in inspect.signature(RetrievalPipeline.search).parameters

    with RetrievalPipeline(config, runtime_paths=paths) as pipeline:
        resolved_queries, stale_per_query, no_label_queries = resolve_labels(queries, pipeline)
        report.stale_labels = stale_per_query
        report.stale_labels_total = sum(stale_per_query.values())
        report.queries_without_labels = no_label_queries
        for gq in queries:
            result = QueryResult(query_id=gq.id, slice=gq.language)
            effective = resolved_queries[gq.id]
            must = effective.must_ids
            scored = bool(must) and gq.id not in no_label_queries
            for run_index in range(runs):
                start = time.perf_counter()
                search_kwargs: dict[str, Any] = {
                    "query": gq.query,
                    "top_k": top_k,
                    "update_access": False,
                    "result_filter": _default_include,
                }
                if _pass_source:
                    search_kwargs["source"] = "eval"
                response = pipeline.search(**search_kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                result.latencies_ms.append(elapsed_ms)
                ranked_ids = [item.node.id for item in response.results]
                scores = [item.score for item in response.results]
                inject_flags = [bool(getattr(item, "is_injectable", False)) for item in response.results]
                result.results_per_run.append(ranked_ids)
                result.positive_scores_per_run.append(scores)
                result.inject_flags_per_run.append(inject_flags)
                result.positive_count = sum(1 for s in scores if s > 0)
                result.inject_count = sum(1 for flag in inject_flags if flag)
                result.returned_count = len(ranked_ids)
                if run_index == runs - 1:
                    result.lexical_hits = _lexical_hit_count(pipeline, gq.query)

            if runs > 1:
                # Metrics on the median-latency run keeps one warm measurement.
                ordered = sorted(range(runs), key=lambda i: result.latencies_ms[i])
                take = ordered[runs // 2]
            else:
                take = 0
            ranked_ids = result.results_per_run[take]
            inject_flags = result.inject_flags_per_run[take]

            if gq.expect_no_result:
                scores = result.positive_scores_per_run[take]
                result.correctly_empty = not any(s > 0 for s in scores)
                result.correctly_empty_inject = not any(inject_flags)
            elif scored:
                hits = [1 if node_id in must else 0 for node_id in ranked_ids[:5]]
                result.recall5 = sum(hits[:5]) / len(must) if must else 0.0
                result.recall5 = min(1.0, result.recall5)
                mrr = 0.0
                for rank, node_id in enumerate(ranked_ids[:10], start=1):
                    if node_id in must:
                        mrr = 1.0 / rank
                        break
                result.mrr10 = mrr
                # Bridge-visible metrics: only results with score > 0 get injected.
                scores = result.positive_scores_per_run[take]
                pos_ids = [nid for nid, s in zip(ranked_ids, scores) if s > 0]
                pos_hits = [1 if node_id in must else 0 for node_id in pos_ids[:5]]
                result.recall5_positive = min(1.0, sum(pos_hits[:5]) / len(must)) if must else 0.0
                result.mrr10_positive = next(
                    (1.0 / rank for rank, node_id in enumerate(pos_ids[:10], start=1) if node_id in must),
                    0.0,
                )
                # Bridge-effective metrics: the server's inject decision.
                inj_ids = [nid for nid, flag in zip(ranked_ids, inject_flags) if flag]
                inj_hits = [1 if node_id in must else 0 for node_id in inj_ids[:5]]
                result.recall5_inject = min(1.0, sum(inj_hits[:5]) / len(must)) if must else 0.0
                result.top1_relevant = bool(ranked_ids) and ranked_ids[0] in must
            report.per_query.append(result)
    return report


def write_report(report: EvalReport, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
