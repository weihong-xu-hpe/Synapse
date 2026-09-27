"""Behaviour tests for observability: search events, snapshots, report, audit."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from synapse.config import SynapseConfig, load_config
from synapse.observability import (
    collect_daily_snapshot,
    cjk_ratio,
    parse_since_days,
    render_report,
    run_injection_audit,
    session_hash,
    write_missing_snapshots,
)
from synapse.server.decider import LocalLLMDecider
from synapse.server.service import SynapseServerService
from synapse.storage import SQLiteNodeStore, SearchEventMetrics
from synapse.utils.runtime import bootstrap_runtime_directories, get_runtime_paths


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

[decider]
provider = "local_llm"
base_url = "http://localhost:1/v1"

[observability]
eval_history_path = "eval-history.jsonl"
audit_history_path = "audit-history.jsonl"
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    bootstrap_runtime_directories(config)
    return config


class _FakeAuditDecider:
    """LLM decider stub returning relevance by node-title marker."""

    name = "fake-audit"

    def __init__(self, settings: Any = None) -> None:
        pass

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
        if "directly useful sentinel" in prompt:
            relevance, reason = 2, "answers the query"
        elif "related sentinel" in prompt:
            relevance, reason = 1, "tangentially related"
        else:
            relevance, reason = 0, "unrelated content"
        body = json.dumps({"relevance": relevance, "reason": reason})
        return {"choices": [{"message": {"content": body}}]}


def _store(config: SynapseConfig) -> SQLiteNodeStore:
    return SQLiteNodeStore(
        get_runtime_paths(config).base / "synapse.db",
        embedding_dimension=config.embedding.dimension or 0,
    )


# ---------------------------------------------------------------------------
# search events
# ---------------------------------------------------------------------------


def test_search_memory_writes_event_with_source_and_inject(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = SynapseServerService(config, runtime_paths=get_runtime_paths(config))
    created = service.integrate_knowledge(
        title="Observability Search Sentinel",
        content="Observability search event coverage sentinel body.",
    )
    payload = service.search_memory("Observability search sentinel coverage", source="bridge")

    with _store(config) as store:
        rows = store.connection.execute("SELECT * FROM search_events").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "bridge"
    assert row["session_hash"] is None
    assert row["query_chars"] > 0
    assert row["latency_ms"] >= 0.0
    results = json.loads(str(row["results"]))
    assert isinstance(results, list)
    injected_ids = {item["node_id"] for item in results if item["inject"]}
    # The only indexed node must appear; inject flag is a bool per result.
    assert created["node"]["id"] in {item["node_id"] for item in results}
    assert all(isinstance(item["inject"], bool) for item in results)
    assert all(set(item) >= {"rank", "node_id", "rerank_logit", "score", "inject"} for item in results)
    del injected_ids


def test_search_event_session_hash_is_sha1_prefix(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = SynapseServerService(config, runtime_paths=get_runtime_paths(config))
    session_key = "omp-session-abc123"
    service.search_memory("any query at all", source="bridge", exclude_session_key=session_key)
    with _store(config) as store:
        row = store.connection.execute("SELECT session_hash FROM search_events").fetchone()
    assert row["session_hash"] == hashlib.sha1(session_key.encode()).hexdigest()[:12]


def test_rest_search_assigns_bridge_or_rest_source(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from synapse.server.app import create_app

    config = _config(tmp_path)
    app = create_app(config, runtime_paths=get_runtime_paths(config))
    with TestClient(app) as client:
        client.post("/api/search", json={"query": "first probe query"})
        client.post("/api/search", json={"query": "second probe query", "exclude_session_key": "key-1"})
    with _store(config) as store:
        sources = [str(row["source"]) for row in store.connection.execute("SELECT source FROM search_events ORDER BY id").fetchall()]
    assert sources == ["rest", "bridge"]


def test_mcp_search_assigns_mcp_source(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from synapse.server.app import create_app

    config = _config(tmp_path)
    app = create_app(config, runtime_paths=get_runtime_paths(config))
    with TestClient(app) as client:
        init = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "obs-test", "version": "1.0"},
                },
            },
        )
        session_header = {"mcp-session-id": init.headers["mcp-session-id"]}
        client.post(
            "/mcp",
            headers=session_header,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        response = client.post(
            "/mcp",
            headers=session_header,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search_memory", "arguments": {"query": "mcp probe query"}},
            },
        )
    assert response.status_code == 200
    with _store(config) as store:
        sources = [str(row["source"]) for row in store.connection.execute("SELECT source FROM search_events").fetchall()]
    assert sources == ["mcp"]


def test_pipeline_search_bypasses_service_so_eval_is_not_recorded(tmp_path: Path) -> None:
    """Eval calls RetrievalPipeline.search directly — no search_events row may appear."""
    from synapse.retrieval import RetrievalPipeline

    config = _config(tmp_path)
    service = SynapseServerService(config, runtime_paths=get_runtime_paths(config))
    service.integrate_knowledge(title="Eval Exclusion Sentinel", content="Eval exclusion sentinel body.")
    with RetrievalPipeline(config, runtime_paths=get_runtime_paths(config)) as pipeline:
        pipeline.search("eval exclusion sentinel body", top_k=3, update_access=False)
    with _store(config) as store:
        count = int(store.connection.execute("SELECT COUNT(*) AS c FROM search_events").fetchone()["c"])
    assert count == 0


def test_search_event_failure_never_breaks_read_path(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    service = SynapseServerService(config, runtime_paths=get_runtime_paths(config))
    service.integrate_knowledge(title="Broken Store Sentinel", content="Broken store sentinel body.")

    def explode(*args: Any, **kwargs: Any):
        raise sqlite3.DatabaseError("injected failure")

    monkeypatch.setattr(SQLiteNodeStore, "record_search_event", explode)
    payload = service.search_memory("broken store sentinel body")
    assert payload["query"]


def test_search_event_latency_bounded_overhead(tmp_path: Path, monkeypatch) -> None:
    """The event insert must not add noticeable latency to the read path."""
    import time

    config = _config(tmp_path)
    service = SynapseServerService(config, runtime_paths=get_runtime_paths(config))
    service.integrate_knowledge(title="Latency Probe Sentinel", content="Latency probe sentinel body.")
    service.search_memory("latency probe sentinel body")  # warm
    start = time.perf_counter()
    for _ in range(5):
        service.search_memory("latency probe sentinel body")
    per_search_with_event = (time.perf_counter() - start) / 5

    def no_record(self, metrics):
        return None

    monkeypatch.setattr(SQLiteNodeStore, "record_search_event", no_record)
    start = time.perf_counter()
    for _ in range(5):
        service.search_memory("latency probe sentinel body")
    per_search_without_event = (time.perf_counter() - start) / 5
    # Small inserts on a tiny local DB: overhead budget 10 ms.
    assert per_search_with_event - per_search_without_event < 0.010


def test_search_events_retention_prunes_old_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = SynapseServerService(config, runtime_paths=get_runtime_paths(config))
    old_ts = (datetime.now(UTC) - timedelta(days=400)).isoformat().replace("+00:00", "Z")
    with _store(config) as store:
        store.record_search_event(
            SearchEventMetrics(
                ts=old_ts,
                source="api",
                session_hash=None,
                query="ancient query",
                query_chars=13,
                cjk_ratio=0.0,
                top_k=3,
                include="default",
                latency_ms=1.0,
                lexical_hit_count=None,
                results=(),
            )
        )
    service.search_memory("fresh query today")
    with _store(config) as store:
        queries = [str(row["query"]) for row in store.connection.execute("SELECT query FROM search_events").fetchall()]
    assert "ancient query" not in queries
    assert any("fresh query today" in q for q in queries)


def test_write_memory_dedupe_guard_records_unchanged_event(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = SynapseServerService(config, runtime_paths=get_runtime_paths(config))
    service.write_node(title="Guard Coverage Node", content="Guard coverage node body.", node_type="transient")
    result = service.write_memory(title="Guard Coverage Node", content="Guard coverage node body.", route="mcp")
    assert result["action"] == "unchanged"
    with _store(config) as store:
        row = store.connection.execute(
            "SELECT action, route FROM write_memory_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row["action"] == "unchanged"
    assert row["route"] == "mcp"


def test_write_routes_are_attributed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = SynapseServerService(config, runtime_paths=get_runtime_paths(config))
    service.upsert_session_memory(session_key="route-probe", title="Session route node", content="Session route body.")
    service.write_node(title="Write node route", content="Write node body.")
    with _store(config) as store:
        rows = store.connection.execute("SELECT route FROM write_memory_events ORDER BY id").fetchall()
    routes = [str(row["route"]) for row in rows]
    assert "session_upsert" in routes
    assert "write_node" in routes


# ---------------------------------------------------------------------------
# daily snapshots
# ---------------------------------------------------------------------------


def test_snapshot_is_idempotent_per_day_and_backfills(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = get_runtime_paths(config)
    first = write_missing_snapshots(config, paths, backfill_days=2)
    second = write_missing_snapshots(config, paths, backfill_days=2)
    assert len(first) == 3  # today + 2 backfill days
    assert second == []
    with _store(config) as store:
        dates = [str(row["date"]) for row in store.connection.execute("SELECT date FROM metrics_snapshots").fetchall()]
    assert len(dates) == len(set(dates))


def test_snapshot_payload_contains_expected_sections(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = get_runtime_paths(config)
    service = SynapseServerService(config, runtime_paths=paths)
    service.integrate_knowledge(
        title="Snapshot Section Sentinel",
        content="Snapshot section sentinel body.",
        okf_type="fact",
        project="probe-project",
    )
    payload = collect_daily_snapshot(config, paths, day=datetime.now().astimezone().date())
    assert payload["nodes"]["total"] >= 1
    assert "fact" in payload["nodes"]["by_okf_type"]
    assert "probe-project" in payload["nodes"]["by_project"]
    assert set(payload["transcripts"]) == {"represented", "zero_item", "undistilled"}
    assert set(payload["writes"]) >= {"total", "by_route", "by_action"}
    assert "distiller" in payload and "searches" in payload and "storage" in payload


def test_snapshot_ignores_days_before_first_event(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = get_runtime_paths(config)
    # No events at all yet: only today's row is allowed to exist.
    written = write_missing_snapshots(config, paths, backfill_days=5)
    with _store(config) as store:
        count = int(store.connection.execute("SELECT COUNT(*) AS c FROM metrics_snapshots").fetchone()["c"])
    assert len(written) == count


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_renders_with_empty_data(tmp_path: Path) -> None:
    config = _config(tmp_path)
    content = render_report(config, get_runtime_paths(config), since_days=30)
    assert "# Synapse lookback report (30d)" in content
    assert "No search events recorded" in content
    assert "No eval runs recorded" in content


def test_report_renders_weekly_buckets_and_sources(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = get_runtime_paths(config)
    service = SynapseServerService(config, runtime_paths=paths)
    service.integrate_knowledge(title="Report Bucket Sentinel", content="Report bucket sentinel body.")
    service.search_memory("report bucket sentinel body", source="bridge")
    service.search_memory("report bucket sentinel body", source="mcp", exclude_session_key="sess-1")
    content = render_report(config, paths, since_days=30)
    assert "| Week |" in content
    assert "`bridge`" in content and "`mcp`" in content
    assert "Sessions with zero injections" in content


def test_report_includes_snapshot_storage_table(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = get_runtime_paths(config)
    write_missing_snapshots(config, paths, backfill_days=0)
    content = render_report(config, paths, since_days=30)
    assert "| Date | DB bytes | Active bytes | Archive bytes |" in content


def test_report_out_path_writes_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    out = tmp_path / "reports" / "lookback.md"
    render_report(config, get_runtime_paths(config), since_days=7, out_path=out)
    assert out.exists() and "Synapse lookback report" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def _seed_injected_search_events(config: SynapseConfig, node_ids: list[str]) -> None:
    with _store(config) as store:
        for index, node_id in enumerate(node_ids):
            store.record_search_event(
                SearchEventMetrics(
                    ts=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    source="bridge" if index % 2 == 0 else "mcp",
                    session_hash=None,
                    query=f"audit probe query {index} directly useful sentinel" if index % 2 else f"audit probe query {index} related sentinel",
                    query_chars=20,
                    cjk_ratio=0.0,
                    top_k=3,
                    include="default",
                    latency_ms=5.0,
                    lexical_hit_count=None,
                    results=(
                        {"rank": 1, "node_id": node_id, "rerank_logit": 1.0, "score": 1.0, "inject": True},
                    ),
                )
            )


def test_parse_since_days_units() -> None:
    assert parse_since_days("30d") == 30
    assert parse_since_days("12w") == 84
    assert parse_since_days("6m") == 180
    assert parse_since_days("14") == 14
    with pytest.raises(ValueError):
        parse_since_days("bogus")


def test_cjk_ratio_and_session_hash() -> None:
    assert cjk_ratio("") == 0.0
    assert cjk_ratio("hello") == 0.0
    assert cjk_ratio("你好") == 1.0
    assert cjk_ratio("a你好") == round(2 / 3, 4)
    assert session_hash(None) is None
    assert session_hash("k") == hashlib.sha1(b"k").hexdigest()[:12]


def test_audit_sampling_judging_and_history(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    paths = get_runtime_paths(config)
    service = SynapseServerService(config, runtime_paths=paths)
    useful = service.integrate_knowledge(title="Directly useful sentinel node", content="Directly useful sentinel body.")
    related = service.integrate_knowledge(title="Related sentinel node", content="Related sentinel body.")
    _seed_injected_search_events(config, [useful["node"]["id"], related["node"]["id"]])

    monkeypatch.setattr(LocalLLMDecider, "__init__", lambda self, settings=None, client=None: None)
    monkeypatch.setattr(LocalLLMDecider, "sample_json", _FakeAuditDecider.sample_json)
    monkeypatch.setattr(LocalLLMDecider, "_complete", lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("not used")))

    summary = run_injection_audit(config, paths, since_days=30, sample_size=10, seed=7)
    assert summary["judged"] == 2
    assert summary["precision_useful"] == pytest.approx(0.5)
    assert summary["precision_related"] == pytest.approx(1.0)
    assert summary["by_source"]["bridge"]["judged"] == 1
    assert summary["by_source"]["mcp"]["judged"] == 1
    history_line = json.loads(config.observability.resolved_audit_history_path(config).read_text(encoding="utf-8").splitlines()[-1])
    assert history_line["precision_related"] == pytest.approx(1.0)


def test_audit_worst_case_listing_in_report(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    paths = get_runtime_paths(config)
    service = SynapseServerService(config, runtime_paths=paths)
    unrelated = service.integrate_knowledge(title="Unrelated filler node", content="Unrelated filler body.")
    _seed_injected_search_events(config, [unrelated["node"]["id"]])

    monkeypatch.setattr(LocalLLMDecider, "__init__", lambda self, settings=None, client=None: None)
    monkeypatch.setattr(LocalLLMDecider, "sample_json", _FakeAuditDecider.sample_json)

    report_path = paths.logs / "audit.md"
    run_injection_audit(config, paths, since_days=30, sample_size=5, report_path=report_path, seed=3)
    text = report_path.read_text(encoding="utf-8")
    assert "Worst cases" in text
    assert "Precision (related" in text


def test_audit_swallows_llm_failures(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    paths = get_runtime_paths(config)
    service = SynapseServerService(config, runtime_paths=paths)
    node = service.integrate_knowledge(title="Failing llm node", content="Failing llm body.")
    _seed_injected_search_events(config, [node["node"]["id"]])

    def fail(self, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(LocalLLMDecider, "__init__", lambda self, settings=None, client=None: None)
    monkeypatch.setattr(LocalLLMDecider, "sample_json", fail)
    summary = run_injection_audit(config, paths, since_days=30, sample_size=5, seed=1)
    assert summary["judged"] == 0
    assert summary["precision_related"] is None


def test_store_metrics_snapshot_round_trip(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with _store(config) as store:
        assert store.record_metrics_snapshot(date="2026-09-26", created_at="now", payload={"a": 1}) is True
        assert store.record_metrics_snapshot(date="2026-09-26", created_at="now", payload={"a": 2}) is False
        snapshots = store.get_metrics_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0]["payload"] == {"a": 1}


def test_schema_version_bumped_and_new_tables_exist(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with _store(config) as store:
        version = store.get_meta("schema_version")
        tables = {
            str(row["name"])
            for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    assert version == "7"
    assert {"search_events", "metrics_snapshots"}.issubset(tables)


def test_route_column_added_to_write_events(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with _store(config) as store:
        columns = {str(row["name"]) for row in store.connection.execute("PRAGMA table_info(write_memory_events)").fetchall()}
    assert "route" in columns


def test_healthy_db_open_is_write_free(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db_path = get_runtime_paths(config).base / "synapse.db"
    with _store(config):
        pass
    data_version_before = sqlite3.connect(db_path).execute("PRAGMA data_version").fetchone()[0]
    with _store(config):
        pass
    conn = sqlite3.connect(db_path)
    data_version_after = conn.execute("PRAGMA data_version").fetchone()[0]
    conn.close()
    assert data_version_before == data_version_after
