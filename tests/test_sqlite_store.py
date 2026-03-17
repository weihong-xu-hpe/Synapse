from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synapse.models import Node, NodeMetadata, NodeStatus, NodeType, SensitivityLevel
from synapse.storage import SQLiteNodeStore


NOW = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)


def make_node(
    *,
    node_id: str,
    title: str,
    content: str,
    file_name: str,
    status: NodeStatus = NodeStatus.ACTIVE,
    last_accessed: datetime | None = None,
    tags: list[str] | None = None,
) -> Node:
    metadata = NodeMetadata(
        id=node_id,
        title=title,
        created_at=NOW - timedelta(days=14),
        last_accessed=last_accessed or NOW,
        access_count=2,
        type=NodeType.PERSISTENT,
        status=status,
        supersedes=[],
        superseded_by=None,
        tags=tags or [],
        sensitivity=SensitivityLevel.INTERNAL,
    )
    return Node(metadata=metadata, content=content, file_path=Path(f"active/{file_name}"))


def test_sqlite_store_crud_and_wal_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "synapse.db"
    node = make_node(
        node_id="mem_20260307_rate_limiting",
        title="Rate Limiting",
        content="Token bucket design for API gateway decisions.",
        file_name="mem_20260307_rate_limiting.md",
        tags=["gateway", "rate-limit"],
    )

    with SQLiteNodeStore(db_path, embedding_dimension=3) as store:
        store.upsert_node(node, embedding=[1.0, 0.0, 0.0])

        stored = store.get_node(node.id)
        assert stored is not None
        assert stored.title == node.title
        assert stored.metadata.tags == ["gateway", "rate-limit"]
        assert stored.metadata.sensitivity is SensitivityLevel.INTERNAL
        assert store.is_wal_enabled() is True
        assert store.vector_backend == "python-fallback"
        assert store.count_nodes() == 1

        store.update_access([node.id])
        updated = store.get_node(node.id)
        assert updated is not None
        assert updated.metadata.access_count == 3

        store.update_status(node.id, NodeStatus.SUPERSEDED, superseded_by="mem_20260308_new_limit")
        superseded = store.get_node(node.id)
        assert superseded is not None
        assert superseded.metadata.status is NodeStatus.SUPERSEDED
        assert superseded.metadata.superseded_by == "mem_20260308_new_limit"

        listed = store.list_nodes({"status": NodeStatus.SUPERSEDED})
        assert [item.id for item in listed] == [node.id]

        store.delete_node(node.id)
        assert store.get_node(node.id) is None
        assert store.count_nodes() == 0


def test_fts_search_vector_search_edges_and_neighbors(tmp_path: Path) -> None:
    db_path = tmp_path / "synapse.db"
    alpha = make_node(
        node_id="mem_20260307_alpha",
        title="Gateway Rate Limits",
        content="API gateway token bucket rate limiting and quotas.",
        file_name="mem_20260307_alpha.md",
        tags=["gateway"],
    )
    beta = make_node(
        node_id="mem_20260307_beta",
        title="Logging Architecture",
        content="Structured logs and retention policy for services.",
        file_name="mem_20260307_beta.md",
        tags=["logging"],
    )
    gamma = make_node(
        node_id="mem_20260307_gamma",
        title="Gateway Retries",
        content="Retry budgets for API gateway clients.",
        file_name="mem_20260307_gamma.md",
        tags=["gateway", "retries"],
    )

    with SQLiteNodeStore(db_path, embedding_dimension=3) as store:
        store.upsert_node(alpha, embedding=[1.0, 0.0, 0.0])
        store.upsert_node(beta, embedding=[0.0, 1.0, 0.0])
        store.upsert_node(gamma, embedding=[0.8, 0.2, 0.0])
        store.upsert_edges(alpha.id, [beta.id, gamma.id])
        store.upsert_edges(gamma.id, [beta.id])

        fts_results = store.fts_search("gateway", limit=3)
        assert {node_id for node_id, _ in fts_results[:2]} == {alpha.id, gamma.id}
        assert beta.id not in [node_id for node_id, _ in fts_results[:2]]

        vector_results = store.vector_search([1.0, 0.0, 0.0], limit=3)
        assert [node_id for node_id, _ in vector_results] == [alpha.id, gamma.id, beta.id]
        assert vector_results[0][1] == pytest.approx(0.0)

        assert store.get_edges(alpha.id) == [beta.id, gamma.id]
        assert store.get_edges(beta.id, direction="incoming") == [alpha.id, gamma.id]
        assert store.get_in_degree(beta.id) == 2
        assert store.get_neighbors([alpha.id], depth=1) == [beta.id, gamma.id]


def test_fts_search_safely_handles_date_like_tokens(tmp_path: Path) -> None:
    db_path = tmp_path / "synapse.db"
    dated = make_node(
        node_id="mem_20260315_http_test",
        title="HTTP MCP Full Test",
        content="Agent perspective MCP full test sentinel 2026-03-15.",
        file_name="mem_20260315_http_test.md",
        tags=["http", "sampling"],
    )

    with SQLiteNodeStore(db_path, embedding_dimension=3) as store:
        store.upsert_node(dated, embedding=[1.0, 0.0, 0.0])

        results = store.fts_search("agent perspective mcp full test sentinel 2026-03-15", limit=3)

        assert [node_id for node_id, _ in results] == [dated.id]


def test_janitor_queries_and_integrity_report(tmp_path: Path) -> None:
    db_path = tmp_path / "synapse.db"
    orphan = make_node(
        node_id="mem_20260307_orphan",
        title="Old Concept",
        content="A stale concept node.",
        file_name="mem_20260307_orphan.md",
        last_accessed=NOW - timedelta(days=20),
    )
    anchored = make_node(
        node_id="mem_20260307_anchor",
        title="Linked Concept",
        content="A concept with inbound references.",
        file_name="mem_20260307_anchor.md",
        last_accessed=NOW - timedelta(days=20),
    )
    superseded = make_node(
        node_id="mem_20260307_old_policy",
        title="Old Policy",
        content="Historical decision kept for context.",
        file_name="mem_20260307_old_policy.md",
        status=NodeStatus.SUPERSEDED,
        last_accessed=NOW - timedelta(days=10),
    )
    disputed = make_node(
        node_id="mem_20260307_disputed",
        title="Disputed Note",
        content="This note is disputed.",
        file_name="mem_20260307_disputed.md",
        status=NodeStatus.DISPUTED,
    )
    referrer = make_node(
        node_id="mem_20260307_referrer",
        title="Referrer",
        content="Reference to an anchored node.",
        file_name="mem_20260307_referrer.md",
    )
    low_reference = make_node(
        node_id="mem_20260307_low_reference",
        title="Low Importance Reference",
        content="Low importance archival candidate.",
        file_name="mem_20260307_low_reference.md",
        last_accessed=NOW - timedelta(days=120),
    )
    high_reference = make_node(
        node_id="mem_20260307_high_reference",
        title="High Importance Reference",
        content="High importance archival candidate.",
        file_name="mem_20260307_high_reference.md",
        last_accessed=NOW - timedelta(days=120),
    )

    with SQLiteNodeStore(db_path, embedding_dimension=3) as store:
        for node in (orphan, anchored, superseded, disputed, referrer, low_reference, high_reference):
            store.upsert_node(node, embedding=[1.0, 0.0, 0.0])
        store.upsert_edges(referrer.id, [anchored.id])

        orphan_candidates = store.find_orphan_candidates(days_threshold=7)
        assert orphan.id in [node.id for node in orphan_candidates]
        long_orphan_candidates = store.find_orphan_candidates(days_threshold=90)
        assert {node.id for node in long_orphan_candidates} >= {low_reference.id, high_reference.id}

        archival_candidates = store.find_superseded_for_archival(days_threshold=7)
        assert [node.id for node in archival_candidates] == [superseded.id]

        assert store.count_disputed_nodes() == 1

        integrity = store.check_integrity()
        assert integrity.ok is True
        assert integrity.integrity_check_result == "ok"
        assert integrity.wal_mode_enabled is True
        assert integrity.total_nodes == 7
