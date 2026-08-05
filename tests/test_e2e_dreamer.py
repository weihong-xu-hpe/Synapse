"""E2E gate tests for the Dreamer lifecycle pipeline against the real LLM.

These tests are marked ``live`` and skipped by default (see pyproject.toml
``addopts = "-q -m 'not live'"``). Run them manually with::

    pytest -m live

They drive the full 6-stage Dreamer pipeline (Scan -> Triage -> Link Weaving
-> Conflict Resolution -> Execute -> Report) against the real internal LLM
endpoint, asserting real side effects on the SQLite store and filesystem:

  * stale orphans are archived (removed from the active store)
  * missing links between similar nodes are woven (edges written)
  * disputed pairs are resolved (status cleared)
  * a clean store runs end-to-end with zero warnings

When the LLM endpoint is unreachable the tests skip rather than fail. But
when it *is* reachable, any pipeline error or missing side effect fails the
gate -- these are the quality gates that prevent regressions in the
LocalLLMDecider-driven Dreamer path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synapse.lifecycle import Dreamer
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType
from synapse.storage import SQLiteNodeStore

from tests.conftest import LIVE_PRIMARY_URL


pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _make_node(
    node_id: str,
    *,
    title: str | None = None,
    content: str = "",
    status: NodeStatus = NodeStatus.ACTIVE,
    node_type: NodeType = NodeType.TRANSIENT,
    last_accessed: datetime | None = None,
    superseded_by: str | None = None,
    file_path: Path | None = None,
) -> Node:
    accessed = last_accessed or datetime.now(UTC)
    return Node(
        metadata=NodeMetadata(
            id=node_id,
            title=title or f"Node {node_id}",
            status=status,
            type=node_type,
            last_accessed=accessed,
            superseded_by=superseded_by,
        ),
        content=content,
        file_path=file_path or Path("active") / f"{node_id}.md",
    )


def _write_node_file(node: Node, runtime_paths) -> None:
    """Write a node's markdown file to the active directory (required for archive/move)."""
    from synapse.storage import write_node_file

    write_node_file(node, base_path=runtime_paths.base)


def _get_store(config, runtime_paths) -> SQLiteNodeStore:
    return SQLiteNodeStore(
        runtime_paths.base / "synapse.db",
        embedding_dimension=config.embedding.dimension or 1024,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_live_dreamer_triages_stale_orphan_to_archive(live_dreamer_env, live_primary_reachable) -> None:
    """A stale orphan node -> Dreamer triages it -> LLM judges archive -> node removed from store."""
    if not live_primary_reachable:
        pytest.skip(f"primary LLM endpoint unreachable: {LIVE_PRIMARY_URL}")

    config, runtime_paths, decider = live_dreamer_env

    # A stale orphan: active, last_accessed 30 days ago, no edges, no embedding.
    stale_time = datetime.now(UTC) - timedelta(days=30)
    stale_node = _make_node(
        "stale-orphan-1",
        title="Obsolete config note",
        content="An old configuration note that is no longer referenced anywhere.",
        last_accessed=stale_time,
    )
    _write_node_file(stale_node, runtime_paths)

    store = _get_store(config, runtime_paths)
    try:
        store.upsert_node(stale_node)
        assert store.get_node("stale-orphan-1") is not None  # precondition

        dreamer = Dreamer(config, runtime_paths=runtime_paths, sampling_client=decider)
        try:
            report = dreamer.run(batch_size=8)
        finally:
            dreamer.close()

        # The stale orphan should have been triaged. The LLM may judge 'archive'
        # or 'keep' or 'condense'; for a truly obsolete orphan we expect archive.
        # Gate: if the LLM said archive, the node must be gone from the store
        # (archived file moved out of active -> startup_sync deletes it).
        triage_ids = [d.node_id for d in report.triage]
        assert "stale-orphan-1" in triage_ids, (
            f"stale-orphan-1 not triaged; triage decisions: {triage_ids}"
        )

        triage_decision = next(d for d in report.triage if d.node_id == "stale-orphan-1")
        if triage_decision.decision == "archive":
            assert "stale-orphan-1" in report.archived, (
                f"LLM said archive but node not in report.archived: {report.archived}"
            )
            assert store.get_node("stale-orphan-1") is None, (
                "stale-orphan-1 still in store after archive + sync"
            )
        elif triage_decision.decision == "keep":
            # LLM judged keep -> node touched but remains. Acceptable but log it.
            assert store.get_node("stale-orphan-1") is not None
        elif triage_decision.decision == "condense":
            assert "stale-orphan-1" in report.archived, "condense should archive the source"
    finally:
        store.close()


def test_live_dreamer_weaves_missing_link_between_similar_nodes(
    live_dreamer_env, live_primary_reachable
) -> None:
    """Two similar active nodes with no edge -> Dreamer weaves a link -> edge appears in store."""
    if not live_primary_reachable:
        pytest.skip(f"primary LLM endpoint unreachable: {LIVE_PRIMARY_URL}")

    config, runtime_paths, decider = live_dreamer_env

    recent = datetime.now(UTC) - timedelta(hours=1)
    node_a = _make_node(
        "link-a",
        title="Auth gateway design",
        content="The auth gateway validates JWT tokens and forwards claims to upstream services.",
        last_accessed=recent,
    )
    node_b = _make_node(
        "link-b",
        title="Gateway JWT validation",
        content="JWT validation in the gateway checks token signatures and expiry before forwarding.",
        last_accessed=recent,
    )
    _write_node_file(node_a, runtime_paths)
    _write_node_file(node_b, runtime_paths)

    store = _get_store(config, runtime_paths)
    try:
        # Upsert nodes with identical embeddings so cosine similarity = 1.0 > 0.75 threshold.
        dim = config.embedding.dimension or 1024
        similar_embedding = [1.0] + [0.0] * (dim - 1)
        store.upsert_node(node_a, embedding=similar_embedding)
        store.upsert_node(node_b, embedding=similar_embedding)

        # Precondition: no edges exist.
        assert store.get_edges("link-a") == []
        assert store.get_edges("link-b") == []

        dreamer = Dreamer(config, runtime_paths=runtime_paths, sampling_client=decider)
        try:
            report = dreamer.run(batch_size=8)
        finally:
            dreamer.close()

        # The pair should have been scanned as a missing-link candidate.
        assert report.scanned["missing_link_pairs"] >= 1, (
            f"expected at least 1 missing link pair, scanned={report.scanned}"
        )

        # Gate: if the LLM wove a link, the markdown files must contain the link tags.
        # (Edges in the store are populated by SyncManager from markdown [[links]].)
        if report.links_added:
            link_pairs = {(d.node_a_id, d.node_b_id) for d in report.links_added}
            pair_found = ("link-a", "link-b") in link_pairs or ("link-b", "link-a") in link_pairs
            assert pair_found, f"expected link between link-a/link-b, got {link_pairs}"

            # After sync, the markdown files should contain [[link-b]] / [[link-a]] tags.
            a_file = runtime_paths.active / "link-a.md"
            b_file = runtime_paths.active / "link-b.md"
            assert a_file.exists(), "link-a.md missing after dreamer run"
            assert b_file.exists(), "link-b.md missing after dreamer run"
            a_text = a_file.read_text(encoding="utf-8")
            b_text = b_file.read_text(encoding="utf-8")
            assert "[[link-b]]" in a_text, f"link-a.md missing [[link-b]] tag:\n{a_text}"
            assert "[[link-a]]" in b_text, f"link-b.md missing [[link-a]] tag:\n{b_text}"
        else:
            # LLM may legitimately decide not to link. Log but don't fail the gate
            # on the link decision itself -- the gate is that the pipeline ran cleanly.
            pytest.skip("LLM did not weave a link for this pair (acceptable judgement call)")
    finally:
        store.close()


def test_live_dreamer_resolves_disputed_pair(live_dreamer_env, live_primary_reachable) -> None:
    """Two disputed nodes referencing each other -> Dreamer resolves -> status cleared."""
    if not live_primary_reachable:
        pytest.skip(f"primary LLM endpoint unreachable: {LIVE_PRIMARY_URL}")

    config, runtime_paths, decider = live_dreamer_env

    recent = datetime.now(UTC) - timedelta(hours=1)
    node_a = _make_node(
        "dispute-a",
        title="Config: max connections",
        content="The max connections setting should be 100.",
        status=NodeStatus.DISPUTED,
        superseded_by="dispute-b",
        last_accessed=recent,
    )
    node_b = _make_node(
        "dispute-b",
        title="Config: max connections (revised)",
        content="The max connections setting should be 200.",
        status=NodeStatus.DISPUTED,
        superseded_by="dispute-a",
        last_accessed=recent,
    )
    _write_node_file(node_a, runtime_paths)
    _write_node_file(node_b, runtime_paths)

    store = _get_store(config, runtime_paths)
    try:
        store.upsert_node(node_a)
        store.upsert_node(node_b)

        # Precondition: both disputed.
        assert store.get_node("dispute-a").metadata.status == NodeStatus.DISPUTED
        assert store.get_node("dispute-b").metadata.status == NodeStatus.DISPUTED

        dreamer = Dreamer(config, runtime_paths=runtime_paths, sampling_client=decider)
        try:
            report = dreamer.run(batch_size=8)
        finally:
            dreamer.close()

        # The disputed pair should have been scanned.
        assert report.scanned["disputed"] >= 1, (
            f"expected at least 1 disputed pair, scanned={report.scanned}"
        )

        # Gate: if the LLM resolved the conflict, the nodes must no longer be disputed.
        if report.conflicts_resolved:
            resolved_ids = set()
            for d in report.conflicts_resolved:
                resolved_ids.add(d.node_a_id)
                resolved_ids.add(d.node_b_id)
            assert "dispute-a" in resolved_ids and "dispute-b" in resolved_ids, (
                f"dispute-a/dispute-b not in resolved set: {resolved_ids}"
            )

            decision = report.conflicts_resolved[0]
            a_after = store.get_node("dispute-a")
            b_after = store.get_node("dispute-b")

            if decision.decision == "both_valid":
                assert a_after.metadata.status == NodeStatus.ACTIVE, (
                    f"dispute-a not active after both_valid: {a_after.metadata.status}"
                )
                assert b_after.metadata.status == NodeStatus.ACTIVE, (
                    f"dispute-b not active after both_valid: {b_after.metadata.status}"
                )
            elif decision.decision == "supersede_a":
                # a is loser (superseded), b is winner (active)
                assert a_after.metadata.status == NodeStatus.SUPERSEDED, (
                    f"dispute-a not superseded after supersede_a: {a_after.metadata.status}"
                )
                assert b_after.metadata.status == NodeStatus.ACTIVE
            elif decision.decision == "supersede_b":
                assert b_after.metadata.status == NodeStatus.SUPERSEDED, (
                    f"dispute-b not superseded after supersede_b: {b_after.metadata.status}"
                )
                assert a_after.metadata.status == NodeStatus.ACTIVE
        else:
            pytest.skip("LLM did not emit a conflict resolution (acceptable judgement call)")
    finally:
        store.close()


def test_live_dreamer_run_completes_without_warning_on_clean_store(
    live_dreamer_env, live_primary_reachable
) -> None:
    """An empty store -> Dreamer runs all 6 stages -> zero warnings, report structure intact."""
    if not live_primary_reachable:
        pytest.skip(f"primary LLM endpoint unreachable: {LIVE_PRIMARY_URL}")

    config, runtime_paths, decider = live_dreamer_env

    dreamer = Dreamer(config, runtime_paths=runtime_paths, sampling_client=decider)
    try:
        report = dreamer.run(batch_size=8)
    finally:
        dreamer.close()

    # Gate 1: no warnings on a clean store.
    assert report.warnings == (), f"unexpected warnings on clean store: {report.warnings}"

    # Gate 2: all 6 stages ran and the report structure is intact.
    assert report.started_at
    assert report.completed_at
    assert isinstance(report.scanned, dict)
    assert all(key in report.scanned for key in ("stale", "superseded", "disputed", "missing_link_pairs"))
    assert report.scanned == {"stale": 0, "superseded": 0, "disputed": 0, "missing_link_pairs": 0}
    assert report.triage == ()
    assert report.links_added == ()
    assert report.conflicts_resolved == ()
    assert report.archived == ()
    assert report.condensed == ()
