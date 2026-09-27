"""Golden-set loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True, frozen=True)
class RelevantDoc:
    """A labeled relevant node for a golden query."""

    node_id: str
    tier: str = "must"  # "must" (must recall) or "may" (bonus)


@dataclass(slots=True, frozen=True)
class GoldenQuery:
    """One evaluation query with labeled relevant node ids."""

    id: str
    query: str
    language: str = "en"  # "zh" | "en" | "mixed" | "code" | "none"
    relevant: tuple[RelevantDoc, ...] = ()
    expect_no_result: bool = False
    notes: str = ""

    @property
    def must_ids(self) -> set[str]:
        return {doc.node_id for doc in self.relevant if doc.tier == "must"}

    @property
    def may_ids(self) -> set[str]:
        return {doc.node_id for doc in self.relevant if doc.tier == "may"}


def load_golden(path: str | Path) -> list[GoldenQuery]:
    """Load a golden set JSON file into validated GoldenQuery objects.

    Schema::

        {
          "queries": [
            {
              "id": "zh-001",
              "query": "用户的 user group 变化时监听哪些 kafka event",
              "language": "mixed",
              "relevant": [{"node_id": "mem_...", "tier": "must"}, ...],
              "expect_no_result": false,
              "notes": "optional"
            }
          ]
        }
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("queries") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError(f"Golden set {path} must contain a 'queries' list")

    queries: list[GoldenQuery] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Golden entry #{index} is not an object")
        qid = str(entry.get("id") or f"q{index:03d}")
        if qid in seen_ids:
            raise ValueError(f"Duplicate golden query id: {qid}")
        seen_ids.add(qid)
        text = str(entry.get("query", "")).strip()
        if not text:
            raise ValueError(f"Golden query {qid} has empty query text")
        relevant = tuple(
            RelevantDoc(node_id=str(doc["node_id"]), tier=str(doc.get("tier", "must")))
            for doc in entry.get("relevant", [])
            if isinstance(doc, dict) and doc.get("node_id")
        )
        for doc in relevant:
            if doc.tier not in {"must", "may"}:
                raise ValueError(f"Golden query {qid}: invalid tier {doc.tier!r} (must|may)")
        if not relevant and not entry.get("expect_no_result"):
            raise ValueError(f"Golden query {qid} has no relevant ids and expect_no_result is false")
        queries.append(
            GoldenQuery(
                id=qid,
                query=text,
                language=str(entry.get("language", "en")),
                relevant=relevant,
                expect_no_result=bool(entry.get("expect_no_result", not relevant)),
                notes=str(entry.get("notes", "")),
            )
        )
    return queries


def slice_of(query: GoldenQuery) -> str:
    """Slices reported in metrics: language plus the aggregate 'overall'."""

    return query.language if query.language in {"zh", "en", "mixed", "code"} else "other"
