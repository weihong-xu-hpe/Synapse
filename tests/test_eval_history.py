"""Tests for eval history lines and live-data label resolution."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from synapse.eval.golden import GoldenQuery, RelevantDoc, load_golden
from synapse.eval.history import append_history_line, file_sha256, git_sha
from synapse.eval.harness import resolve_labels
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType, SensitivityLevel
from synapse.retrieval import RetrievalPipeline
from synapse.storage import SQLiteNodeStore
from synapse.utils.runtime import bootstrap_runtime_directories

NOW = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)


def _make_node(node_id: str, *, status: NodeStatus = NodeStatus.ACTIVE, superseded_by: str | None = None) -> Node:
    metadata = NodeMetadata(
        id=node_id,
        title=node_id.title(),
        created_at=NOW - timedelta(days=30),
        last_accessed=NOW,
        access_count=0,
        type=NodeType.PERSISTENT,
        status=status,
        superseded_by=superseded_by,
        tags=[],
        sensitivity=SensitivityLevel.INTERNAL,
    )
    return Node(metadata=metadata, content=f"content of {node_id}", file_path=Path(f"active/{node_id}.md"))


def _write_golden(path: Path, queries: list[dict]) -> Path:
    path.write_text(json.dumps({"queries": queries}), encoding="utf-8")
    return path


def test_history_line_format(tmp_path: Path) -> None:
    golden = _write_golden(
        tmp_path / "golden.json",
        [{"id": "q1", "query": "demo", "relevant": [{"node_id": "n1", "tier": "must"}]}],
    )
    report = {
        "overall": {"n": 1, "recall@5": 1.0},
        "slices": {"en": {"n": 1, "recall@5": 1.0}},
        "stale_labels_total": 0,
        "queries_without_labels": [],
    }
    history_path = tmp_path / "history.jsonl"
    fixed = datetime(2026, 3, 7, 10, 0, tzinfo=UTC)

    line = append_history_line(
        history_path,
        report,
        golden_path=golden,
        repo_dir=tmp_path,
        now=fixed,
    )

    assert line["ts"] == fixed.isoformat()
    assert line["golden_sha256"] == file_sha256(golden)
    assert line["n_queries"] == 1
    assert line["overall"]["recall@5"] == 1.0
    assert line["slices"]["en"]["n"] == 1
    assert line["stale_labels"] == 0
    assert line["queries_without_labels"] == []
    # git_sha is present for the real repo; None when repo_dir has no git.
    assert (line["git_sha"] is None) or (len(line["git_sha"]) == 40)

    written = json.loads(history_path.read_text(encoding="utf-8").splitlines()[-1])
    assert written == line

    # A second run appends rather than overwrites.
    append_history_line(history_path, report, golden_path=golden, repo_dir=tmp_path, now=fixed)
    assert len(history_path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_git_sha_real_repo() -> None:
    sha = git_sha(Path(__file__).resolve().parent)
    assert sha is None or len(sha) == 40


def test_resolve_labels_superseded_chain_and_stale(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "builtin"
model = "bge-m3"
dimension = 3

[reranker]
provider = "builtin"
model = "bge-reranker-v2-m3"
max_candidates = 9

[retrieval]
rrf_k = 60
top_k = 3
""".strip(),
        encoding="utf-8",
    )
    runtime_paths = bootstrap_runtime_directories(_load(config_path))

    # a -> b -> c (terminal active); stale1 archived (missing); stale2 missing.
    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        store.upsert_node(_make_node("a", status=NodeStatus.SUPERSEDED, superseded_by="b"), embedding=[1.0, 0.0, 0.0])
        store.upsert_node(_make_node("b", status=NodeStatus.SUPERSEDED, superseded_by="c"), embedding=[1.0, 0.0, 0.0])
        store.upsert_node(_make_node("c"), embedding=[1.0, 0.0, 0.0])

        pipeline = RetrievalPipeline(
            _load(config_path),
            store=store,
            runtime_paths=runtime_paths,
            embedding_engine=_FakeEmbedding(),
            reranker_engine=_FakeReranker(),
            now_fn=lambda: NOW,
        )

        queries = [
            GoldenQuery(id="chain", query="chain", relevant=(RelevantDoc(node_id="a"),)),
            GoldenQuery(id="mixed", query="mixed", relevant=(RelevantDoc(node_id="a"), RelevantDoc(node_id="stale1"))),
            GoldenQuery(id="all-stale", query="all stale", relevant=(RelevantDoc(node_id="stale2"),)),
            GoldenQuery(id="none-q", query="none", expect_no_result=True),
        ]
        resolved, stale, no_labels = resolve_labels(queries, pipeline)

    # Superseded chain resolves to the terminal active node.
    assert [d.node_id for d in resolved["chain"].relevant] == ["c"]
    # Mixed: chain label resolves, archived label dropped and counted.
    assert [d.node_id for d in resolved["mixed"].relevant] == ["c"]
    assert stale["mixed"] == 1
    # All-stale query keeps no labels and is reported, not scored.
    assert resolved["all-stale"].relevant == ()
    assert stale["all-stale"] == 1
    assert no_labels == ["all-stale"]
    assert stale["chain"] == 0
    # expect_no_result queries pass through untouched.
    assert resolved["none-q"].expect_no_result is True


def test_resolve_labels_cycle_safe(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "builtin"
model = "bge-m3"
dimension = 3

[reranker]
provider = "builtin"
model = "bge-reranker-v2-m3"
max_candidates = 9

[retrieval]
rrf_k = 60
top_k = 3
""".strip(),
        encoding="utf-8",
    )
    runtime_paths = bootstrap_runtime_directories(_load(config_path))
    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        store.upsert_node(_make_node("x", status=NodeStatus.SUPERSEDED, superseded_by="y"), embedding=[1.0, 0.0, 0.0])
        store.upsert_node(_make_node("y", status=NodeStatus.SUPERSEDED, superseded_by="x"), embedding=[1.0, 0.0, 0.0])
        pipeline = RetrievalPipeline(
            _load(config_path),
            store=store,
            runtime_paths=runtime_paths,
            embedding_engine=_FakeEmbedding(),
            reranker_engine=_FakeReranker(),
            now_fn=lambda: NOW,
        )
        queries = [GoldenQuery(id="cyc", query="cyc", relevant=(RelevantDoc(node_id="x"),))]
        resolved, stale, no_labels = resolve_labels(queries, pipeline)
    assert resolved["cyc"].relevant == ()
    assert stale["cyc"] == 1
    assert no_labels == ["cyc"]


def _load(config_path: Path):
    from synapse.config import load_config

    return load_config(config_path)


class _FakeEmbedding:
    model_name = "fake"
    dimension = 3

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def is_available(self) -> bool:
        return True


class _FakeReranker:
    model_name = "fake"

    def rerank(self, query: str, documents: list[str], limit: int | None = None) -> list[tuple[int, float]]:
        return [(i, 1.0) for i in range(len(documents))][: limit or len(documents)]

    def is_available(self) -> bool:
        return True
