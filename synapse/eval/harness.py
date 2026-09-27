"""Evaluation harness: run golden queries through the real retrieval pipeline."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synapse.config import SynapseConfig
from synapse.eval.golden import GoldenQuery, slice_of
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
    recall5: float = 0.0
    mrr10: float = 0.0
    # Same metrics but counting only results the bridge would inject (score > 0).
    recall5_positive: float = 0.0
    mrr10_positive: float = 0.0
    top1_relevant: bool = False
    lexical_hits: int = 0
    positive_count: int = 0  # score > 0 among returned results (last run)
    returned_count: int = 0
    # For expect_no_result queries: True when NO returned result had score > 0.
    correctly_empty: bool = True


@dataclass(slots=True)
class EvalReport:
    """Aggregated evaluation report."""

    per_query: list[QueryResult] = field(default_factory=list)
    created_at: str = ""
    config_path: str = ""

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
        for result in self.per_query:
            slices[result.slice].append(result)
        overall: dict[str, Any] = {}
        slice_metrics: dict[str, Any] = {}
        for name, results in sorted(slices.items()):
            metrics = {
                "n": len(results),
                "recall@5": round(_mean([r.recall5 for r in results]), 4),
                "recall@5_pos": round(_mean([r.recall5_positive for r in results]), 4),
                "mrr@10": round(_mean([r.mrr10 for r in results]), 4),
                "mrr@10_pos": round(_mean([r.mrr10_positive for r in results]), 4),
                "top1_relevance": round(_mean([1.0 if r.top1_relevant else 0.0 for r in results]), 4),
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
                "p50_latency_ms": round(_percentile([l for r in results for l in r.latencies_ms], 50), 1),
                "p95_latency_ms": round(_percentile([l for r in results for l in r.latencies_ms], 95), 1),
            }
            slice_metrics[name] = metrics
        # 'overall' covers all queries with labels (excludes expect_no_result for
        # ranking metrics but keeps their latency/positive-score signal).
        labeled = [r for r in self.per_query if r.slice != "none"]
        empties = [r for r in self.per_query if r.slice == "none"]
        all_lat = [l for r in self.per_query for l in r.latencies_ms]
        overall = {
            "n": len(self.per_query),
            "recall@5": round(_mean([r.recall5 for r in labeled]), 4),
            "recall@5_pos": round(_mean([r.recall5_positive for r in labeled]), 4),
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
            "no_result_correct_rate": round(
                _mean([1.0 if r.correctly_empty else 0.0 for r in empties]), 4
            )
            if empties
            else None,
            "p50_latency_ms": round(_percentile(all_lat, 50), 1),
            "p95_latency_ms": round(_percentile(all_lat, 95), 1),
        }
        return {
            "created_at": self.created_at,
            "config_path": self.config_path,
            "overall": overall,
            "slices": slice_metrics,
            "queries": [
                {
                    "id": r.query_id,
                    "slice": r.slice,
                    "recall@5": r.recall5,
                    "recall@5_pos": r.recall5_positive,
                    "mrr@10": r.mrr10,
                    "mrr@10_pos": r.mrr10_positive,
                    "top1_relevant": r.top1_relevant,
                    "lexical_hits": r.lexical_hits,
                    "positive": r.positive_count,
                    "returned": r.returned_count,
                    "latency_ms": round(_mean(r.latencies_ms), 1),
                    "results": r.results_per_run[-1] if r.results_per_run else [],
                    "correctly_empty": r.correctly_empty,
                }
                for r in self.per_query
            ],
        }


def _lexical_hit_count(pipeline: RetrievalPipeline, query: str) -> int:
    """Number of nodes the lexical (FTS) leg alone returns."""

    store = pipeline._get_store()
    return len(store.fts_search(query, limit=20))


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

    with RetrievalPipeline(config, runtime_paths=paths) as pipeline:
        for gq in queries:
            result = QueryResult(query_id=gq.id, slice=gq.language)
            must = gq.must_ids
            for run_index in range(runs):
                start = time.perf_counter()
                response = pipeline.search(gq.query, top_k=top_k, update_access=False)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                result.latencies_ms.append(elapsed_ms)
                ranked_ids = [item.node.id for item in response.results]
                scores = [item.score for item in response.results]
                result.results_per_run.append(ranked_ids)
                result.positive_scores_per_run.append(scores)
                result.positive_count = sum(1 for s in scores if s > 0)
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

            if gq.expect_no_result:
                scores = result.positive_scores_per_run[take]
                result.correctly_empty = not any(s > 0 for s in scores)
            elif must:
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
                result.top1_relevant = bool(ranked_ids) and ranked_ids[0] in must
            report.per_query.append(result)
    return report


def write_report(report: EvalReport, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
