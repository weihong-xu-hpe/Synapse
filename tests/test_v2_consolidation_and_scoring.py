"""V2 memory-consolidation and usage-signal tests.

Covers the design doc's Pillar 2 (three-level usage counters, project-active
session clock, ACT-R activation, hub penalty) and Pillar 1 (daytime
deterministic delta merge, Dreamer 2.0 semantic merge / oversized split with
anchor-token verification, event-driven trigger).
"""

from __future__ import annotations

import math
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from synapse.config import SynapseConfig
from synapse.lifecycle.condensation import DeterministicArchiveCondenser
from synapse.lifecycle.dreamer import Dreamer
from synapse.lifecycle.distiller_scheduler import DistillerScheduler
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType
from synapse.retrieval.pipeline import (
    ACTIVATION_ALPHA,
    HUB_PENALTY_LAMBDA,
    RetrievalPipeline,
)
from synapse.server.service import deterministic_section_merge
from synapse.storage import SQLiteNodeStore, write_node_file
from synapse.storage.markdown import node_to_markdown
from synapse.utils.runtime import RuntimePaths, bootstrap_runtime_directories


NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(tmp_path: Path) -> SynapseConfig:
    config = SynapseConfig.with_defaults(tmp_path)
    base = tmp_path / "copy"
    config.memory.base_path = base
    config.memory.archive_path = base / ".archive"
    config.embedding.provider = "builtin"
    config.reranker.provider = "builtin"
    return config


def _env(tmp_path: Path) -> tuple[SynapseConfig, RuntimePaths, SQLiteNodeStore]:
    config = _config(tmp_path)
    runtime_paths = bootstrap_runtime_directories(config)
    store = SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3)
    return config, runtime_paths, store


def _node(
    node_id: str,
    *,
    title: str | None = None,
    content: str = "Body line.",
    project: str | None = None,
    sources: list[str] | None = None,
    tags: list[str] | None = None,
    okf_type: str | None = "fact",
    last_accessed: datetime | None = None,
    status: NodeStatus = NodeStatus.ACTIVE,
) -> Node:
    metadata = NodeMetadata(
        id=node_id,
        title=title or node_id.replace("_", " ").title(),
        type=NodeType.PERSISTENT,
        status=status,
        okf_type=okf_type,
        project=project,
        sources=sources if sources is not None else ["mem_session_a"],
        tags=tags or [],
        last_accessed=last_accessed or NOW,
    )
    return Node(metadata=metadata, content=content, file_path=Path("active") / f"{node_id}.md")


class FakeSamplingClient:
    """Deterministic stand-in for the Tier B gatekeeper."""

    name = "fake-gatekeeper"

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.prompts: list[str] = []
        self.closed = False

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.closed = True


class FakeEmbeddingEngine:
    model_name = "fake-embedding"
    dimension = 3

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Pillar 2: three-level counters, read/write decoupling
# ---------------------------------------------------------------------------


def test_update_access_splits_retrieved_from_injected(tmp_path: Path) -> None:
    """Retrieval bumps only `retrieved_count`; the inject gate owns exposure."""

    _config_value, _paths, store = _env(tmp_path)
    with store:
        store.upsert_node(_node("mem_alpha", content="Alpha body."))
        store.upsert_node(_node("mem_beta", content="Beta body."))

        store.update_access(["mem_alpha", "mem_beta"], injected_ids=["mem_alpha"])

        alpha = store.get_node("mem_alpha")
        beta = store.get_node("mem_beta")
    assert alpha is not None and beta is not None
    assert alpha.metadata.retrieved_count == 1
    assert alpha.metadata.injected_count == 1
    assert alpha.metadata.used_count == 0
    assert alpha.metadata.access_count == 1
    assert beta.metadata.retrieved_count == 1
    assert beta.metadata.injected_count == 0
    assert beta.metadata.access_count == 0  # read/write decoupled: no popularity feedback


def test_legacy_update_access_call_still_counts_as_injected(tmp_path: Path) -> None:
    """The pre-P2 `update_access(ids)` contract keeps its meaning."""

    _config_value, _paths, store = _env(tmp_path)
    with store:
        store.upsert_node(_node("mem_legacy", content="Legacy body."))
        store.update_access(["mem_legacy"])
        node = store.get_node("mem_legacy")
    assert node is not None
    assert node.metadata.injected_count == 1
    assert node.metadata.access_count == 1


def test_record_memory_usage_credits_used_and_injected(tmp_path: Path) -> None:
    """Credit assignment: only actually-used nodes get `used_count`."""

    _config_value, _paths, store = _env(tmp_path)
    with store:
        store.upsert_node(_node("mem_used", content="Used body."))
        store.upsert_node(_node("mem_seen", content="Seen body."))

        store.record_memory_usage(injected_ids=["mem_used", "mem_seen"], used_ids=["mem_used"])

        used = store.get_node("mem_used")
        seen = store.get_node("mem_seen")
    assert used is not None and seen is not None
    assert used.metadata.used_count == 1
    assert used.metadata.injected_count == 1
    assert used.metadata.access_count == 2  # injected+used both count as exposure
    assert seen.metadata.used_count == 0
    assert seen.metadata.injected_count == 1
    assert seen.metadata.access_count == 1  # exposure aggregate = injected + used


def test_counters_are_monotone_across_markdown_resync(tmp_path: Path) -> None:
    """A markdown re-sync (stale node row) must not erase recorded usage."""

    _config_value, _paths, store = _env(tmp_path)
    with store:
        store.upsert_node(_node("mem_mono", content="Monotone body."))
        store.record_memory_usage(injected_ids=["mem_mono"], used_ids=["mem_mono"])
        refreshed = store.get_node("mem_mono")
        assert refreshed is not None
        # Re-upsert the markdown-derived node, which carries counters at 0.
        store.upsert_node(Node(metadata=refreshed.metadata.model_copy(update={"used_count": 0, "injected_count": 0, "access_count": 0}), content=refreshed.content, file_path=refreshed.file_path))
        after = store.get_node("mem_mono")
    assert after is not None
    assert after.metadata.used_count == 1
    assert after.metadata.injected_count == 1
    assert after.metadata.access_count == 2


def test_legacy_database_gains_counter_columns(tmp_path: Path) -> None:
    """A pre-P2 database is migrated in place on open."""

    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            content TEXT NOT NULL,
            file_path TEXT NOT NULL,
            session_key TEXT,
            distilled_turn INTEGER,
            distilled_hash TEXT,
            created_at TEXT NOT NULL,
            last_accessed TEXT NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            verified INTEGER NOT NULL DEFAULT 0,
            superseded_by TEXT,
            tags TEXT,
            okf_meta TEXT
        );
        """
    )
    connection.commit()
    connection.close()

    with SQLiteNodeStore(db_path, embedding_dimension=3) as store:
        columns = {
            str(row["name"])
            for row in store._connection.execute("PRAGMA table_info(nodes)").fetchall()  # noqa: SLF001
        }
    assert {"retrieved_count", "injected_count", "used_count"} <= columns


def test_get_node_degrees_counts_both_directions(tmp_path: Path) -> None:
    _config_value, _paths, store = _env(tmp_path)
    with store:
        for node_id in ("mem_hub", "mem_a", "mem_b", "mem_c"):
            store.upsert_node(_node(node_id, content=f"{node_id} body."))
        store.upsert_edges("mem_hub", ["mem_a", "mem_b"])
        store.upsert_edges("mem_c", ["mem_hub"])
        degrees = store.get_node_degrees(["mem_hub", "mem_a"])
    assert degrees["mem_hub"] == 3
    assert degrees["mem_a"] == 1


def test_project_session_clock_counts_distinct_sessions(tmp_path: Path) -> None:
    config, _paths, store = _env(tmp_path)
    config.embedding.dimension = 3
    with store:
        store.upsert_node(_node("mem_p1", project="proj-a", sources=["mem_s1"]))
        store.upsert_node(_node("mem_p2", project="proj-a", sources=["mem_s2"]))
        store.upsert_node(_node("mem_p3", project="proj-b", sources=["mem_s3"]))
        counts = store.get_project_session_counts()
    assert counts.get("proj-a") == 2
    assert counts.get("proj-b") == 1


# ---------------------------------------------------------------------------
# Pillar 2: ACT-R activation + hub penalty
# ---------------------------------------------------------------------------


def _pipeline(config: SynapseConfig, runtime_paths: RuntimePaths, store: SQLiteNodeStore) -> RetrievalPipeline:
    return RetrievalPipeline(
        config,
        store=store,
        runtime_paths=runtime_paths,
        embedding_engine=FakeEmbeddingEngine(),
        reranker_engine=None,
        now_fn=lambda: NOW,
    )


def test_actr_activation_boosts_used_nodes(tmp_path: Path) -> None:
    config, paths, store = _env(tmp_path)
    with store:
        pipeline = _pipeline(config, paths, store)
        cold = _node("mem_cold", content="Cold body.")
        warm = _node("mem_warm", content="Warm body.")
        warm = Node(
            metadata=warm.metadata.model_copy(update={"used_count": 4, "injected_count": 2}),
            content=warm.content,
            file_path=warm.file_path,
        )

        cold_score, _ = pipeline.apply_decay(cold, 0.0)
        warm_score, _ = pipeline.apply_decay(warm, 0.0)
        same_clock_score, _ = pipeline.apply_decay(warm, 0.0, project_sessions_elapsed=0.0)
        stale_clock_score, _ = pipeline.apply_decay(warm, 0.0, project_sessions_elapsed=400.0)

    assert warm_score > cold_score
    # A negative t_eff would explode the boost; the clock floor keeps it bounded.
    assert abs(same_clock_score - warm_score) < 10.0
    assert stale_clock_score < warm_score
    assert warm_score - cold_score <= ACTIVATION_ALPHA * 5  # bounded nudge


def test_hub_penalty_prefers_specific_node(tmp_path: Path) -> None:
    config, paths, store = _env(tmp_path)
    with store:
        pipeline = _pipeline(config, paths, store)
        specific = _node("mem_specific", content="Specific body.")
        hub = _node("mem_hub", content="Hub body.")

        plain, _ = pipeline.apply_decay(specific, 0.0)
        penalised, _ = pipeline.apply_decay(hub, 0.0, degree=20)

    assert penalised < plain
    assert penalised == pytest.approx(plain - HUB_PENALTY_LAMBDA * math.log(21.0))


def test_project_clock_lookup_uses_project_sessions(tmp_path: Path) -> None:
    config, paths, store = _env(tmp_path)
    with store:
        store.upsert_node(_node("mem_proj", project="proj-a", sources=["mem_s1"]))
        store.upsert_node(_node("mem_proj2", project="proj-a", sources=["mem_s2"]))
        pipeline = _pipeline(config, paths, store)
        context = pipeline._build_scoring_context(["mem_proj"])  # noqa: SLF001
        node = store.get_node("mem_proj")
    assert node is not None
    assert context.for_node(node)["project_sessions_elapsed"] == 2.0


# ---------------------------------------------------------------------------
# Pillar 1: daytime deterministic delta merge
# ---------------------------------------------------------------------------


def test_deterministic_section_merge_keeps_every_existing_line() -> None:
    existing = (
        "## Takeaway\n\n缓存键保持稳定。\n\n"
        "## Details\n\n- 重试复用同一缓存键。\n- 失败后重试同一请求。\n\n"
        "## Sources\n\n- mem_session_a\n"
    )
    candidate = (
        "## Takeaway\n\n完全不同。\n\n"
        "## Details\n\n- 失败后重试同一请求。\n- 新增：连接池上限为 8。\n\n"
        "## Sources\n\n- mem_session_a\n- mem_session_b\n"
    )

    merged = deterministic_section_merge(existing, candidate)

    assert "缓存键保持稳定。" in merged  # existing takeaway wins
    assert "完全不同。" not in merged
    assert "- 重试复用同一缓存键。" in merged
    assert merged.count("- 失败后重试同一请求。") == 1  # de-duplicated
    assert "- 新增：连接池上限为 8。" in merged
    assert merged.count("mem_session_a") == 1
    assert "mem_session_b" in merged


def test_deterministic_section_merge_is_lossless_for_rewritten_bodies() -> None:
    existing = "## Takeaway\n\nT.\n\n## Details\n\n- alpha detail.\n- beta detail.\n"
    candidate = "## Details\n\n- alpha detail.\n"

    merged = deterministic_section_merge(existing, candidate)

    from synapse.server.service import _omitted_knowledge_lines

    assert _omitted_knowledge_lines(existing, merged) == []


def test_condensation_summarize_skips_structural_lines() -> None:
    transcript = Node(
        metadata=NodeMetadata(
            id="mem_session_x",
            title="Session summary",
            type=NodeType.TRANSIENT,
            session_key="k",
        ),
        content=(
            "## T1 User\n---\n\n<!-- internal note -->\n: User\n\nThe cache key stays stable for retried writes.\n"
        ),
        file_path=Path("transcripts/mem_session_x.md"),
    )
    draft = DeterministicArchiveCondenser().synthesize([transcript], now=NOW)

    assert "The cache key stays stable for retried writes." in draft.content
    assert "---" not in draft.content
    assert "internal note" not in draft.content
    assert "## T1 User" not in draft.content
    assert "## User" not in draft.content
    assert ": User" not in draft.content


# ---------------------------------------------------------------------------
# Dreamer 2.0: semantic merge, anchor gate, oversized split
# ---------------------------------------------------------------------------


def _write(store: SQLiteNodeStore, runtime_paths: RuntimePaths, node: Node, vector: list[float]) -> None:
    write_node_file(node, base_path=runtime_paths.base)
    store.upsert_node(node, embedding=vector)


def test_semantic_merge_supersedes_members_and_writes_winner(tmp_path: Path) -> None:
    config, paths, store = _env(tmp_path)
    config.dreamer.thresholds.semantic_merge_cosine = 0.9
    config.dreamer.thresholds.min_semantic_merge_cluster = 2
    with store:
        member_a = _node(
            "mem_merge_a",
            title="Cache Key Stability",
            content="## Takeaway\n\nRetries reuse one cache key.\n\n## Details\n\nKeep the key stable.\n",
            project="proj-a",
        )
        member_b = _node(
            "mem_merge_b",
            title="Retry Cache Reuse",
            content="## Takeaway\n\nRetried writes reuse the key.\n\n## Details\n\nNever rotate it.\n",
            project="proj-a",
        )
        _write(store, paths, member_a, [1.0, 0.0, 0.0])
        _write(store, paths, member_b, [1.0, 0.0, 0.0])

        gatekeeper = FakeSamplingClient(
            {
                "title": "Stable Cache Keys Across Retries",
                "okf_type": "fact",
                "takeaway": "Retries must reuse one cache key.",
                "body": "## Details\n\n- Retries reuse one cache key.\n- Never rotate it.\n",
            }
        )
        dreamer = Dreamer(config, runtime_paths=paths, sampling_client=gatekeeper, gatekeeper_client=gatekeeper)
        warnings: list[Any] = []
        try:
            merged_results = dreamer._run_semantic_merge(store, warnings)  # noqa: SLF001
        finally:
            dreamer.close()

        assert len(merged_results) == 1
        merged = merged_results[0]
        winner = store.get_node(merged.winner_node_id)
        members = {member_id: store.get_node(member_id) for member_id in merged.member_ids}

    assert set(merged.member_ids) == {"mem_merge_a", "mem_merge_b"}
    assert warnings == []
    assert winner is not None
    assert winner.metadata.status is NodeStatus.ACTIVE
    assert winner.metadata.t_valid
    assert "mem_merge_a" in winner.metadata.sources
    assert "mem_merge_b" in winner.metadata.sources
    assert winner.metadata.okf_type == "fact"
    for member_id, member in members.items():
        assert member is not None
        assert member.metadata.status is NodeStatus.SUPERSEDED
        assert member.metadata.superseded_by == merged.winner_node_id
        assert member.metadata.t_invalid
        assert (paths.archive / f"{member_id}.md").exists()


def test_semantic_merge_rejects_anchor_token_loss(tmp_path: Path) -> None:
    config, paths, store = _env(tmp_path)
    config.dreamer.thresholds.semantic_merge_cosine = 0.9
    config.dreamer.thresholds.min_semantic_merge_cluster = 2
    with store:
        member_a = _node(
            "mem_anchor_a",
            title="Gateway Timeout Budget",
            content=(
                "## Takeaway\n\nTimeout is 30s.\n\n## Details\n\n"
                "- GET /api/v1/gateway times out after 30s.\n- ERROR_CODE=GW_504 is returned.\n- Retry budget is 3 attempts.\n"
            ),
            project="proj-a",
        )
        member_b = _node(
            "mem_anchor_b",
            title="Gateway Retry Budget",
            content="## Takeaway\n\nRetry 3 times.\n\n## Details\n\n- Retry budget is 3 attempts.\n",
            project="proj-a",
        )
        _write(store, paths, member_a, [1.0, 0.0, 0.0])
        _write(store, paths, member_b, [1.0, 0.0, 0.0])

        gatekeeper = FakeSamplingClient(
            {
                "title": "Gateway Timeout And Retry Policy",
                "okf_type": "fact",
                "takeaway": "Gateway calls are bounded.",
                "body": "## Details\n\n- Some calls are bounded.\n",
            }
        )
        dreamer = Dreamer(config, runtime_paths=paths, sampling_client=gatekeeper, gatekeeper_client=gatekeeper)
        warnings: list[Any] = []
        try:
            merged_results = dreamer._run_semantic_merge(store, warnings)  # noqa: SLF001
        finally:
            dreamer.close()
        members = {
            member_id: store.get_node(member_id) for member_id in ("mem_anchor_a", "mem_anchor_b")
        }

    assert merged_results == []
    assert any(warning.code == "anchor_token_loss" for warning in warnings)
    for member in members.values():
        assert member is not None and member.metadata.status is NodeStatus.ACTIVE


def test_oversized_near_duplicate_is_deduped_into_keeper(tmp_path: Path) -> None:
    """§3.4(2): an oversized near-duplicate is retired without spending an LLM call."""

    config, paths, store = _env(tmp_path)
    config.dreamer.thresholds.oversized_compaction_bytes = 300
    shared = "\n".join(f"- shared gateway retry detail number {index}" for index in range(20))
    keeper = _node(
        "mem_keeper",
        title="Gateway Retry Notes",
        content=f"## Takeaway\n\nRetries are bounded.\n\n## Details\n\n{shared}\n",
    )
    duplicate = _node(
        "mem_keeper_copy",
        title="Gateway Retry Notes Copy",
        content=(
            "## Takeaway\n\nRetries are bounded.\n\n## Details\n\n"
            f"{shared}\n- the connection pool cap is 8\n"
        ),
    )
    assert len(keeper.content.encode("utf-8")) > 300
    assert len(duplicate.content.encode("utf-8")) > 300

    with store:
        _write(store, paths, keeper, [1.0, 0.0, 0.0])
        _write(store, paths, duplicate, [1.0, 0.0, 0.0])

        gatekeeper = FakeSamplingClient(error=RuntimeError("gatekeeper must not be needed"))
        dreamer = Dreamer(config, runtime_paths=paths, sampling_client=gatekeeper, gatekeeper_client=gatekeeper)
        warnings: list[Any] = []
        try:
            compacted = dreamer._run_oversized_compaction(store, warnings)  # noqa: SLF001
        finally:
            dreamer.close()

        assert compacted[0].bytes_after == 0  # deterministic dedupe, no LLM involved
        node_ids = ("mem_keeper", "mem_keeper_copy")
        retired_ids = [
            node_id
            for node_id in node_ids
            if store.get_node(node_id).metadata.status is NodeStatus.SUPERSEDED
        ]
        survivor_ids = [node_id for node_id in node_ids if node_id not in retired_ids]
        assert len(retired_ids) == 1
        survivor = store.get_node(survivor_ids[0])
        retired = store.get_node(retired_ids[0])

    assert survivor is not None
    assert "- the connection pool cap is 8" in survivor.content
    assert "- shared gateway retry detail number 19" in survivor.content
    assert retired is not None
    assert retired.metadata.superseded_by == survivor.id
    assert (paths.archive / f"{retired.id}.md").exists()


def test_oversized_node_is_split_into_atomic_siblings(tmp_path: Path) -> None:
    config, paths, store = _env(tmp_path)
    config.dreamer.thresholds.oversized_compaction_bytes = 400
    oversized_body = (
        "## Takeaway\n\nBig node.\n\n"
        + "\n\n".join(
            f"## Section {index}\n\n" + "\n".join(f"- detail {index}.{row} with padding text" for row in range(8))
            for index in range(1, 7)
        )
        + "\n"
    )
    oversized = _node("mem_oversized", title="Huge Gateway Note", content=oversized_body)
    assert len(oversized_body.encode("utf-8")) > 400

    with store:
        _write(store, paths, oversized, [0.0, 1.0, 0.0])

        gatekeeper = FakeSamplingClient(error=RuntimeError("gatekeeper offline"))
        dreamer = Dreamer(config, runtime_paths=paths, sampling_client=gatekeeper, gatekeeper_client=gatekeeper)
        warnings: list[Any] = []
        try:
            compacted = dreamer._run_oversized_compaction(store, warnings)  # noqa: SLF001
        finally:
            dreamer.close()

        parts = [
            node
            for node in store.list_nodes({"status": NodeStatus.ACTIVE, "type": NodeType.PERSISTENT})
            if "dreamer-split" in node.metadata.tags
        ]
        source = store.get_node("mem_oversized")

    assert len(compacted) == 1
    assert len(parts) >= 2
    assert source is not None and source.metadata.status is NodeStatus.SUPERSEDED
    assert source.metadata.superseded_by == parts[0].id
    assert (paths.archive / "mem_oversized.md").exists()
    rebuilt = "\n".join(part.content for part in parts)
    for index in range(1, 7):
        assert f"## Section {index}" in rebuilt  # every section survives verbatim
        assert f"- detail {index}.7 with padding text" in rebuilt
    # Sibling links exist in both the graph and the markdown (re-index safe).
    assert all(part.id in rebuilt for part in parts)


def test_oversized_node_without_section_boundaries_is_kept(tmp_path: Path) -> None:
    config, paths, store = _env(tmp_path)
    config.dreamer.thresholds.oversized_compaction_bytes = 200
    body = "## Details\n\n" + ("- one huge unsplittable line about gateways.\n" * 20)
    with store:
        _write(store, paths, _node("mem_monolith", title="Monolith Note", content=body), [0.0, 1.0, 0.0])
        gatekeeper = FakeSamplingClient(error=RuntimeError("gatekeeper offline"))
        dreamer = Dreamer(config, runtime_paths=paths, sampling_client=gatekeeper, gatekeeper_client=gatekeeper)
        warnings: list[Any] = []
        try:
            compacted = dreamer._run_oversized_compaction(store, warnings)  # noqa: SLF001
        finally:
            dreamer.close()
        node = store.get_node("mem_monolith")

    assert compacted == []
    assert any(warning.code == "oversized_split_impossible" for warning in warnings)
    assert node is not None and node.metadata.status is NodeStatus.ACTIVE


def test_dreamer_rejects_frontier_background_model(tmp_path: Path) -> None:
    from synapse.config import ensure_background_model_allowed

    with pytest.raises(ValueError):
        ensure_background_model_allowed("claude-4.5-opus", purpose="test")
    ensure_background_model_allowed("claude-haiku-5.5", purpose="test")


# ---------------------------------------------------------------------------
# Event-driven trigger cadence
# ---------------------------------------------------------------------------


def test_distiller_sweep_triggers_dreamer_after_threshold(tmp_path: Path) -> None:
    """§3.4(3): N new knowledge nodes fire one extra sleep cycle."""

    config, paths, store = _env(tmp_path)
    config.dreamer.event_trigger_new_nodes = 3
    config.dreamer.schedule_hour = None  # isolate the event cadence from wall-clock catch-up
    store.close()

    scheduler = DistillerScheduler(config, runtime_paths=paths)
    fired = threading.Event()

    def fake_trigger() -> None:
        fired.set()

    scheduler._run_triggered_dreamer = fake_trigger  # type: ignore[method-assign]  # noqa: SLF001
    runs = [
        type("Run", (), {"ops": [{"action": "created"}, {"action": "updated"}]})(),
        type("Run", (), {"ops": [{"action": "created"}]})(),
    ]
    scheduler._maybe_trigger_dreamer(runs)  # noqa: SLF001

    assert fired.wait(timeout=5.0)
    with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=0) as check:
        assert check.get_meta("dreamer_pending_knowledge_nodes") == "0"


def test_distiller_sweep_counts_below_threshold(tmp_path: Path) -> None:
    config, paths, store = _env(tmp_path)
    config.dreamer.event_trigger_new_nodes = 30
    config.dreamer.schedule_hour = None  # isolate the event cadence from wall-clock catch-up
    store.close()

    scheduler = DistillerScheduler(config, runtime_paths=paths)
    started: list[bool] = []
    scheduler._run_triggered_dreamer = lambda: started.append(True)  # type: ignore[method-assign]  # noqa: SLF001
    scheduler._maybe_trigger_dreamer([type("Run", (), {"ops": [{"action": "created"}]})()])  # noqa: SLF001

    with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=0) as check:
        assert check.get_meta("dreamer_pending_knowledge_nodes") == "1"
    assert started == []
