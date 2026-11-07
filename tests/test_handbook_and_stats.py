"""Tests for the handbook compiler (§5.1–§5.4) and §6.5 north-star stats."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from synapse.cli import app
from synapse.config import SynapseConfig, load_config
from synapse.handbook import HANDBOOK_GENERATED_MARK, build_handbook, clean_slug, infer_project
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType, project_domains
from synapse.observability import (
    collect_daily_snapshot,
    collect_north_star_metrics,
    render_north_star_stats,
)
from synapse.storage import SearchEventMetrics, SQLiteNodeStore
from synapse.utils.runtime import bootstrap_runtime_directories, get_runtime_paths

runner = CliRunner()
FIXED_TIME = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _pin_builtin_project_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate inference from machine-local ~/.synapse/project_domains.json.

    An empty file env value disables custom domains entirely, so tests only
    ever see the built-in examples unless they point the variable at their
    own fixture file.
    """

    monkeypatch.setenv("SYNAPSE_PROJECT_DOMAINS_FILE", "")


def _config(tmp_path: Path) -> SynapseConfig:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[embedding]
provider = "builtin"
model = "bge-m3"
dimension = 1024

[reranker]
provider = "builtin"
model = "bge-reranker-v2-m3"
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    bootstrap_runtime_directories(config)
    return config


def _store(config: SynapseConfig) -> SQLiteNodeStore:
    return SQLiteNodeStore(
        get_runtime_paths(config).base / "synapse.db",
        embedding_dimension=config.embedding.dimension or 0,
    )


def _seed_node(
    store: SQLiteNodeStore,
    node_id: str,
    *,
    title: str,
    content: str,
    okf_type: str | None = "fact",
    sources: list[str] | None = None,
    project: str | None = None,
    access_count: int = 0,
    verified: bool | None = None,
    tags: list[str] | None = None,
    status: NodeStatus = NodeStatus.ACTIVE,
    node_type: NodeType = NodeType.PERSISTENT,
    session_key: str | None = None,
) -> Node:
    node = Node(
        metadata=NodeMetadata(
            id=node_id,
            title=title,
            created_at=FIXED_TIME,
            last_accessed=FIXED_TIME,
            type=node_type,
            status=status,
            access_count=access_count,
            okf_type=okf_type,
            okf_version=1 if okf_type else None,
            sources=sources or [],
            project=project,
            verified=verified,
            tags=tags or [],
            session_key=session_key,
        ),
        content=content,
        file_path=Path(f"active/{node_id}.md"),
    )
    store.upsert_node(node)
    return node


def _search_event(store: SQLiteNodeStore, results: list[dict[str, Any]], *, source: str = "bridge") -> None:
    store.record_search_event(
        SearchEventMetrics(
            ts=FIXED_TIME.isoformat().replace("+00:00", "Z"),
            source=source,
            session_hash=None,
            query="helm hook timeout",
            query_chars=18,
            cjk_ratio=0.0,
            top_k=5,
            include="knowledge",
            latency_ms=12.5,
            lexical_hit_count=None,
            results=tuple(results),
        )
    )


def _okf_pitfall(sources: int = 1) -> str:
    bodies = "\n".join(f"- [[mem_20261001_hook_source_{index}]]" for index in range(sources))
    return (
        "## Takeaway\nDo not forget the --no-hooks cleanup flag.\n\n"
        "## Symptom\nRelease stuck in pending hooks.\n\n"
        "## Cause\nLeftover pre-upgrade hooks.\n\n"
        "## Fix\nRun helm with --no-hooks then retry.\n\n"
        f"## Sources\n{bodies}\n"
    )


# ---------------------------------------------------------------------------
# Handbook: project inference + notes slugs
# ---------------------------------------------------------------------------


def test_infer_project_uses_explicit_project_first_and_domain_markers_second() -> None:
    from synapse.models import Node as _Node

    explicit = _Node(
        metadata=NodeMetadata(id="mem_20261010_x", title="T", type="persistent", project="  Payments-Svc "),
        content="anything about cluster-ops node-7",
    )
    assert infer_project(explicit) == "payments-svc"

    cluster = _Node(
        metadata=NodeMetadata(id="mem_20261010_y", title="Ingress bootstrap on cluster_ops", type="persistent"),
        content="cluster-ops upgrade drains node pools",
    )
    assert infer_project(cluster) == "cluster-ops"

    synapse = _Node(
        metadata=NodeMetadata(id="mem_20261010_z", title="Distiller watermark", type="persistent"),
        content="incremental distillation advances distilled_turn per okf contract",
    )
    assert infer_project(synapse) == "synapse"

    unknown = _Node(
        metadata=NodeMetadata(id="mem_20261010_w", title="Unrelated", type="persistent"),
        content="nothing matching any known workspace domain",
    )
    assert infer_project(unknown) == "general"


def test_infer_project_loads_custom_domains_from_local_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from synapse.models import Node as _Node

    assert infer_project(
        _Node(metadata=NodeMetadata(id="mem_20261010_c", title="Widget ledger", type="persistent"), content="widget ledger drift")
    ) == "general"

    domains_file = tmp_path / "project_domains.json"
    domains_file.write_text(
        json.dumps([{"project": "Widgets-Inc", "markers": ["widget ledger"]}, {"project": "general", "markers": ["x"]}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("SYNAPSE_PROJECT_DOMAINS_FILE", str(domains_file))

    # Custom entry claims the node; the `general` slug entry is ignored (FALLBACK reserved).
    assert infer_project(
        _Node(metadata=NodeMetadata(id="mem_20261010_c", title="Widget ledger", type="persistent"), content="widget ledger drift")
    ) == "widgets-inc"
    # Built-ins remain active for unmatched custom markers.
    assert infer_project(
        _Node(metadata=NodeMetadata(id="mem_20261010_s", title="Dreamer cadence", type="persistent"), content="dreamer cadence")
    ) == "synapse"
    # Malformed files degrade to built-ins, never fatal.
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("SYNAPSE_PROJECT_DOMAINS_FILE", str(broken))
    from synapse.models.node import _BUILTIN_PROJECT_DOMAINS

    assert project_domains() == _BUILTIN_PROJECT_DOMAINS


def test_clean_slug_strips_the_mem_date_prefix() -> None:
    assert clean_slug("mem_20261012_helm-hook-timeout-cleanup") == "helm-hook-timeout-cleanup"
    assert clean_slug("mem_fixture_gateway") == "mem_fixture_gateway"


# ---------------------------------------------------------------------------
# Handbook: compilation tree
# ---------------------------------------------------------------------------


def test_build_handbook_groups_projects_and_filters_non_knowledge_nodes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store(config)
    try:
        _seed_node(
            store,
            "mem_20261001_helm-hook-timeout",
            title="Helm hook timeout cleanup",
            content=_okf_pitfall(2),
            okf_type="pitfall",
            sources=[f"mem_20261001_hook_source_{i}" for i in range(2)],
            project="payments-svc",
            access_count=7,
        )
        _seed_node(
            store,
            "mem_20261002_single-source-pitfall",
            title="Single source pitfall",
            content=_okf_pitfall(1),
            okf_type="pitfall",
            sources=["mem_20261002_only"],
            project="payments-svc",
            access_count=99,
        )
        _seed_node(
            store,
            "mem_20261003_sync-decision",
            title="Adopt delta bullet append",
            content=(
                "## Takeaway\nDelta bullets beat full rewrites.\n\n## Context\nLLM rewrites dropped lines.\n\n"
                "## Decision\nAppend bullets deterministically.\n\n## Consequences\nZero loss by construction.\n\n"
                "## Sources\n- [[mem_a]]\n- [[mem_b]]\n- [[mem_c]]\n"
            ),
            okf_type="decision",
            sources=["mem_a", "mem_b", "mem_c"],
            project="payments-svc",
        )
        _seed_node(
            store,
            "mem_20261004_fact-page",
            title="Enforce latency budget",
            content="## Takeaway\nEnforce p99 must stay under 80ms.\n\n## Details\nBudget from SLO.\n\n## Sources\n- [[mem_x]]\n",
            okf_type="fact",
            sources=["mem_x"],
            project="payments-svc",
        )
        # Domain-inferred (no explicit project) -> cluster-ops.
        _seed_node(
            store,
            "mem_20261005_cluster-ops-bootstrap",
            title="cluster_ops bootstrap",
            content="## Takeaway\nIngress needs the OOM node fix.\n\n## Details\nStaging cluster-ops mirror.\n\n## Sources\n- [[mem_y]]\n",
            okf_type="procedure",
            sources=["mem_y"],
        )
        # Excluded: transcript, verified==False, superseded, transient.
        _seed_node(store, "mem_session_abc123", title="Session summary — proj", content="raw", okf_type=None, session_key="abc")
        _seed_node(
            store,
            "mem_20261006_falsified",
            title="Falsified note",
            content="## Takeaway\nWrong.\n\n## Sources\n- [[mem_z]]\n",
            sources=["mem_z"],
            verified=False,
        )
        _seed_node(
            store,
            "mem_20261007_superseded",
            title="Old decision",
            content="## Takeaway\nOld.\n\n## Sources\n- [[mem_q]]\n",
            okf_type="decision",
            sources=["mem_q"],
            project="payments-svc",
            status=NodeStatus.SUPERSEDED,
        )
        _seed_node(store, "mem_20261008_transient", title="Transient", content="x", node_type=NodeType.TRANSIENT)
        store.connection.commit()
    finally:
        store.close()

    result = build_handbook(config, get_runtime_paths(config))

    assert result["total_nodes"] == 5
    assert result["projects"] == {"payments-svc": 4, "cluster-ops": 1}
    target = Path(result["out_dir"])
    assert target == (get_runtime_paths(config).base / "handbook").resolve()

    index = (target / "INDEX.md").read_text(encoding="utf-8")
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert HANDBOOK_GENERATED_MARK in index
    assert index == readme
    assert "[payments-svc](./payments-svc/README.md)" in index
    assert "Multi-Source Consolidated Highlights" in index
    assert "Delta bullets beat full rewrites." in index  # 3-source decision takeaway highlighted
    assert "mem_20261006_falsified" not in index
    assert "mem_session_abc123" not in index
    assert "mem_20261007_superseded" not in index

    pitfalls = (target / "payments-svc" / "pitfalls-and-caveats.md").read_text(encoding="utf-8")
    # Multi-source pitfall sorts before the single-source one despite lower access_count.
    assert pitfalls.index("Helm hook timeout cleanup") < pitfalls.index("Single source pitfall")
    assert "Do not forget the --no-hooks cleanup flag." in pitfalls
    assert "**Symptom:**" in pitfalls and "**Cause:**" in pitfalls and "**Fix:**" in pitfalls

    decisions = (target / "payments-svc" / "architecture-decisions.md").read_text(encoding="utf-8")
    assert "**Context:**" in decisions and "**Decision:**" in decisions and "**Consequences:**" in decisions

    # notes/ pages use the clean date-stripped slug and keep fact/procedure bodies.
    note = (target / "payments-svc" / "notes" / "fact-page.md").read_text(encoding="utf-8")
    assert HANDBOOK_GENERATED_MARK in note and "Enforce p99 must stay under 80ms." in note
    assert (target / "cluster-ops" / "notes" / "cluster-ops-bootstrap.md").exists()
    assert not (target / "payments-svc" / "notes" / "mem_20261004_fact-page.md").exists()

    project_readme = (target / "payments-svc" / "README.md").read_text(encoding="utf-8")
    assert "pitfalls-and-caveats.md" in project_readme and "architecture-decisions.md" in project_readme


def test_build_handbook_is_idempotent_and_prunes_stale_generated_pages(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store(config)
    try:
        _seed_node(
            store,
            "mem_20261001_keep-me",
            title="Keep",
            content="## Takeaway\nKeep this fact.\n\n## Details\nBody.\n\n## Sources\n- [[mem_s]]\n",
            sources=["mem_s"],
            project="synapse",
        )
        _seed_node(
            store,
            "mem_20261001_drop-me",
            title="Drop",
            content="## Takeaway\nDrop this fact.\n\n## Details\nBody.\n\n## Sources\n- [[mem_s]]\n",
            sources=["mem_s"],
            project="synapse",
        )
        store.connection.commit()
    finally:
        store.close()

    first = build_handbook(config, get_runtime_paths(config))
    target = Path(first["out_dir"])
    stale = target / "synapse" / "notes" / "drop-me.md"
    assert stale.exists()
    hand_written = target / "synapse" / "HAND_NOTES.md"
    hand_written.write_text("my curated notes\n", encoding="utf-8")

    store = _store(config)
    try:
        store.delete_node("mem_20261001_drop-me")
    finally:
        store.close()

    second = build_handbook(config, get_runtime_paths(config))
    assert second["total_nodes"] == first["total_nodes"] - 1
    assert not stale.exists(), "stale generated page survived a rebuild"
    assert (target / "synapse" / "notes" / "keep-me.md").exists()
    assert hand_written.exists(), "pruning must preserve hand-written files"
    assert "Drop" not in (target / "INDEX.md").read_text(encoding="utf-8")


def test_build_handbook_respects_custom_out_dir(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store(config)
    try:
        _seed_node(
            store,
            "mem_20261001_note",
            title="One fact",
            content="## Takeaway\nOnly fact.\n\n## Details\nBody.\n\n## Sources\n- [[mem_s]]\n",
            sources=["mem_s"],
            project="omp",
        )
        store.connection.commit()
    finally:
        store.close()
    custom = tmp_path / "elsewhere"
    result = build_handbook(config, get_runtime_paths(config), out_dir=custom)
    assert Path(result["out_dir"]) == custom.resolve()
    assert (custom / "omp" / "notes" / "note.md").exists()


# ---------------------------------------------------------------------------
# §6.5 north-star metrics
# ---------------------------------------------------------------------------


def test_north_star_counters_from_node_columns(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store(config)
    try:
        _seed_node(
            store,
            "mem_20261001_multi",
            title="Multi",
            content="b",
            sources=["mem_s1", "mem_s2"],
            project="synapse",
        )
        _seed_node(store, "mem_20261002_single", title="Single", content="b", sources=["mem_s3"], project="synapse")
        store.connection.commit()
        if {"retrieved_count", "injected_count", "used_count"} <= {
            str(row["name"]) for row in store.connection.execute("PRAGMA table_info(nodes)").fetchall()
        }:
            store.connection.execute(
                "UPDATE nodes SET retrieved_count = 40, injected_count = 10, used_count = 4 WHERE id = 'mem_20261001_multi'"
            )
            store.connection.execute(
                "UPDATE nodes SET retrieved_count = 10, injected_count = 0, used_count = 0 WHERE id = 'mem_20261002_single'"
            )
            store.connection.commit()
        metrics = collect_north_star_metrics(store, runtime_paths=get_runtime_paths(config))
    finally:
        store.close()

    consolidation = metrics["consolidation"]
    assert consolidation["multi_source_nodes"] == 1
    assert consolidation["single_source_nodes"] == 1
    assert consolidation["total_knowledge_nodes"] == 2
    assert consolidation["consolidation_ratio"] == 0.5

    signal = metrics["signal_to_noise"]
    if signal["counter_source"] == "nodes.counters":
        assert signal["retrieved_total"] == 50
        assert signal["injected_total"] == 10
        assert signal["used_total"] == 4
        assert signal["inject_gate_pass_rate"] == 0.2
        assert signal["utility_conversion_rate"] == 0.4


def test_north_star_search_events_fallback(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    store = _store(config)
    try:
        # Even when the P1B columns exist, force the fallback path (§6.5
        # backwards-compat requirement: old DBs yield a search_events baseline).
        monkeypatch.setattr("synapse.observability._node_usage_counter_columns", lambda store: set())
        _search_event(
            store,
            [
                {"node_id": "mem_20261001_x", "inject": True},
                {"node_id": "mem_20261002_y", "inject": True},
                {"node_id": "mem_20261003_z", "inject": False},
            ],
            source="bridge",
        )
        _search_event(store, [{"node_id": "mem_20261001_x", "inject": False}], source="mcp")
        metrics = collect_north_star_metrics(store, runtime_paths=get_runtime_paths(config))
    finally:
        store.close()

    signal = metrics["signal_to_noise"]
    assert signal["counter_source"] == "search_events"
    assert signal["total_searches"] == 2
    assert signal["bridge_searches"] == 1
    assert signal["retrieved_total"] == 4
    assert signal["injected_total"] == 2
    assert signal["inject_gate_pass_rate"] == 0.5
    assert signal["used_total"] is None
    assert signal["utility_conversion_rate"] is None


def test_north_star_graph_orphans_and_hubs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store(config)
    try:
        hub = _seed_node(
            store,
            "mem_20261001_hub",
            title="Payments hub",
            content="b",
            sources=["mem_20261002_linked", "mem_20261003_orphan"],
            project="payments-svc",
            tags=["dreamer-hub"],
        )
        linked = _seed_node(store, "mem_20261002_linked", title="Linked", content="b", sources=["mem_s"], project="payments-svc")
        _seed_node(store, "mem_20261003_orphan", title="Orphan", content="b", sources=["mem_s"], project="cluster-ops")
        store.upsert_edges(hub.id, [linked.id])
        store.connection.commit()
        metrics = collect_north_star_metrics(store, runtime_paths=get_runtime_paths(config))
    finally:
        store.close()

    graph = metrics["graph_connectivity"]
    assert graph["active_knowledge_nodes"] == 3
    assert graph["orphan_nodes"] == 1
    assert graph["orphan_rate"] == round(1 / 3, 4)
    assert graph["hub_nodes_count"] == 1
    assert graph["projects_with_hub_ratio"] == 0.5


def test_north_star_llm_gate_health_from_distiller_tables_and_quarantine_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store(config)
    quarantine_dir = get_runtime_paths(config).base / ".distiller-quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    (quarantine_dir / "20261029.jsonl").write_text(
        json.dumps({"validator_findings": [{"code": "okf_update_loss_risk", "message": "missing line"}]})
        + "\n"
        + json.dumps({"validator_findings": [{"code": "okf_update_loss_risk", "message": "missing line"}]})
        + "\n"
        + json.dumps({"validator_findings": [{"code": "okf_takeaway_invalid", "message": "empty"}]})
        + "\n",
        encoding="utf-8",
    )
    try:
        store.connection.execute(
            """
            INSERT INTO distiller_runs (
                started_at, completed_at, duration_ms, transcripts_scanned,
                transcripts_distilled, transcripts_skipped, items_extracted, items_invalid,
                items_quarantined, decider_creates, decider_complements, decider_supersedes,
                decider_downgrades, failures, archived_transcripts, "trigger", attempts_json
            ) VALUES (?, ?, 1, 5, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'idle', '[]')
            """,
            ("2026-10-30T01:00:00Z", "2026-10-30T01:00:05Z"),
        )
        store.connection.execute(
            """
            INSERT INTO distiller_runs (
                started_at, completed_at, duration_ms, transcripts_scanned,
                transcripts_distilled, transcripts_skipped, items_extracted, items_invalid,
                items_quarantined, decider_creates, decider_complements, decider_supersedes,
                decider_downgrades, failures, archived_transcripts, "trigger", attempts_json
            ) VALUES (?, ?, 1, 5, 4, 1, 8, 2, 2, 0, 0, 0, 0, 0, 0, 'idle', '[]')
            """,
            ("2026-10-30T02:00:00Z", "2026-10-30T02:00:05Z"),
        )
        store.connection.execute(
            """
            INSERT INTO write_memory_events (
                created_at, node_type, action, candidate_count, similarity_threshold,
                warning_codes, sampling_provider, execution_succeeded, route
            ) VALUES (?, 'persistent', 'updated', 1, 0.8, ?, 'test', 0, 'test')
            """,
            ("2026-10-29T03:00:00Z", json.dumps(["okf_update_loss_risk"])),
        )
        store.connection.commit()
        metrics = collect_north_star_metrics(store, runtime_paths=get_runtime_paths(config))
    finally:
        store.close()

    gate = metrics["llm_gate_health"]
    assert gate["runs"] == 2
    assert gate["items_extracted"] == 8
    assert gate["items_quarantined"] == 2
    assert gate["quarantine_rate"] == round(2 / 10, 4)
    assert gate["zero_item_sweep_count"] == 1
    assert gate["zero_item_sweep_ratio"] == 0.5
    assert gate["okf_update_loss_risk"] >= 2
    assert gate["okf_takeaway_invalid"] >= 1
    assert gate["quarantine_codes_files"].get("okf_update_loss_risk") == 2
    assert gate["quarantine_codes_db"].get("okf_update_loss_risk") == 1
    assert "runs_last_7d" in gate["dreamer_recent"]


def test_daily_snapshot_includes_north_star_only_with_stock_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store(config)
    try:
        _seed_node(store, "mem_20261001_x", title="X", content="b", sources=["mem_s"], project="synapse")
        store.connection.commit()
        paths = get_runtime_paths(config)
        payload = collect_daily_snapshot(config, paths, day=date(2026, 10, 30), store=store)
        assert "north_star" in payload
        assert "consolidation" in payload["north_star"]
        lean = collect_daily_snapshot(config, paths, day=date(2026, 10, 29), store=store, include_stock_metrics=False)
        assert "north_star" not in lean
    finally:
        store.close()


def test_render_north_star_stats_covers_all_four_metrics() -> None:
    metrics = {
        "consolidation": {"multi_source_nodes": 3, "single_source_nodes": 7, "total_knowledge_nodes": 10, "consolidation_ratio": 0.3},
        "signal_to_noise": {
            "total_searches": 5,
            "bridge_searches": 3,
            "retrieved_total": 20,
            "injected_total": 4,
            "used_total": 1,
            "inject_gate_pass_rate": 0.2,
            "utility_conversion_rate": 0.25,
            "counter_source": "nodes.counters",
        },
        "graph_connectivity": {"active_knowledge_nodes": 10, "orphan_nodes": 2, "orphan_rate": 0.2, "hub_nodes_count": 1, "projects_with_hub_ratio": 0.5},
        "llm_gate_health": {
            "runs": 2,
            "items_extracted": 8,
            "items_quarantined": 2,
            "quarantine_rate": 0.2,
            "okf_update_loss_risk": 1,
            "okf_takeaway_invalid": 0,
            "zero_item_sweep_ratio": 0.5,
            "dreamer_recent": {"runs_last_7d": 1},
        },
    }
    rendered = render_north_star_stats(metrics)
    for heading in ("Consolidation ratio", "Recall signal-to-noise", "Wiki graph connectivity", "LLM gate & cost health"):
        assert heading in rendered
    assert "30.0%" in rendered and "25.0%" in rendered


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_handbook_build_and_stats(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store(config)
    try:
        _seed_node(
            store,
            "mem_20261001_cli-fact",
            title="CLI fact",
            content="## Takeaway\nCLI works.\n\n## Details\nYes.\n\n## Sources\n- [[mem_s]]\n",
            sources=["mem_s"],
            project="synapse",
        )
        store.connection.commit()
    finally:
        store.close()

    result = runner.invoke(app, ["--config", str(config.config_path), "handbook", "build"])
    assert result.exit_code == 0, result.output
    assert "Handbook written" in result.output
    handbook_dir = get_runtime_paths(config).base / "handbook"
    assert (handbook_dir / "synapse" / "notes" / "cli-fact.md").exists()

    stats_result = runner.invoke(app, ["--config", str(config.config_path), "stats", "--json"])
    assert stats_result.exit_code == 0, stats_result.output
    payload = json.loads(stats_result.output)
    assert set(payload) == {"consolidation", "signal_to_noise", "graph_connectivity", "llm_gate_health"}

    table_result = runner.invoke(app, ["--config", str(config.config_path), "stats"])
    assert table_result.exit_code == 0, table_result.output
    assert "Consolidation ratio" in table_result.output
    assert "Wiki graph connectivity" in table_result.output


def test_project_top_takeaways_prioritizes_multi_source_and_tracks_compression_ratio(tmp_path: Path) -> None:
    from synapse.handbook import project_top_takeaways

    config = _config(tmp_path)
    store = _store(config)
    try:
        _seed_node(
            store,
            "mem_20261001_single-high-used",
            title="Single source high used",
            content="## Takeaway\nSingle source takeaway.\n\n## Details\nBody.\n\n## Sources\n- [[mem_s1]]\n",
            sources=["mem_s1"],
            project="synapse",
            access_count=10,
        )
        _seed_node(
            store,
            "mem_20261002_multi-corroborated",
            title="Multi source corroborated",
            content="## Takeaway\nMulti source takeaway.\n\n## Details\nBody.\n\n## Sources\n- [[mem_s1]]\n- [[mem_s2]]\n",
            sources=["mem_s1", "mem_s2"],
            project="synapse",
            access_count=1,
        )
        _seed_node(
            store,
            "mem_session_tx1",
            title="Session summary — synapse",
            content="## T1 User\n" + ("x" * 500),
            okf_type=None,
            node_type=NodeType.TRANSIENT,
            session_key="tx1",
        )
        store.connection.commit()

        index = project_top_takeaways(store, "Synapse", limit=5)
        assert index["project"] == "synapse"
        assert index["total_nodes"] == 2
        assert [item["node_id"] for item in index["items"]] == [
            "mem_20261002_multi-corroborated",
            "mem_20261001_single-high-used",
        ]
        assert index["items"][0]["takeaway"] == "Multi source takeaway."

        metrics = collect_north_star_metrics(store, runtime_paths=get_runtime_paths(config))
        assert metrics["consolidation"]["compression_ratio"] is not None
        assert metrics["consolidation"]["knowledge_bytes"] > 0
        assert metrics["consolidation"]["transcript_bytes"] > 500
    finally:
        store.close()
