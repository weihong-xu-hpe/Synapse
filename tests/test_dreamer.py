from __future__ import annotations

from pathlib import Path

from synapse.config import SynapseConfig
from synapse.lifecycle import Dreamer
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType
from synapse.storage import SQLiteNodeStore
from synapse.utils.runtime import RuntimePaths, bootstrap_runtime_directories


class FailingSamplingClient:
    name = "failing-test-client"

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
        del prompt, system_prompt, max_tokens, model_hints
        raise RuntimeError("sampling unavailable")

    def decide_memory_write(self, request):  # pragma: no cover - not used by Dreamer
        raise AssertionError(f"Unexpected write decision request: {request}")


def make_runtime(tmp_path: Path) -> tuple[SynapseConfig, RuntimePaths]:
    config = SynapseConfig.with_defaults(tmp_path)
    runtime_paths = bootstrap_runtime_directories(config)
    return config, runtime_paths


def make_node(node_id: str, *, status: NodeStatus = NodeStatus.ACTIVE) -> Node:
    return Node(
        metadata=NodeMetadata(id=node_id, title=f"Node {node_id}", status=status),
        content="Reusable test memory.",
        file_path=Path("active") / f"{node_id}.md",
    )


def make_dreamer(tmp_path: Path) -> Dreamer:
    config, runtime_paths = make_runtime(tmp_path)
    return Dreamer(config, runtime_paths=runtime_paths, sampling_client=FailingSamplingClient())


def test_triage_sampling_failure_skips_batch_without_archive_decisions(tmp_path: Path) -> None:
    dreamer = make_dreamer(tmp_path)
    warnings = []

    run_triage = getattr(dreamer, "_run_triage")
    decisions = run_triage(
        [make_node("stale-1"), make_node("stale-2")],
        batch_size=2,
        warnings=warnings,
    )

    assert decisions == []
    assert len(warnings) == 1
    assert warnings[0].code == "triage_sampling_failed"
    assert "Skipping batch (no decisions emitted)." in warnings[0].message


def test_link_weaving_sampling_failure_skips_batch(tmp_path: Path) -> None:
    dreamer = make_dreamer(tmp_path)
    warnings = []

    run_link_weaving = getattr(dreamer, "_run_link_weaving")
    decisions = run_link_weaving(
        [(make_node("node-a"), make_node("node-b"))],
        batch_size=1,
        warnings=warnings,
    )

    assert decisions == []
    assert warnings[0].code == "link_weaving_sampling_failed"


def test_conflict_resolution_sampling_failure_skips_batch(tmp_path: Path) -> None:
    dreamer = make_dreamer(tmp_path)
    warnings = []

    run_conflict_resolution = getattr(dreamer, "_run_conflict_resolution")
    decisions = run_conflict_resolution(
        [(make_node("node-a"), make_node("node-b"))],
        batch_size=1,
        warnings=warnings,
    )

    assert decisions == []
    assert warnings[0].code == "conflict_resolution_sampling_failed"


def test_dreamer_records_run_metrics_for_completed_cycle(tmp_path: Path) -> None:
    config, runtime_paths = make_runtime(tmp_path)
    dreamer = Dreamer(config, runtime_paths=runtime_paths, sampling_client=FailingSamplingClient())

    try:
        report = dreamer.run(batch_size=3)
    finally:
        dreamer.close()

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=1024) as store:
        summary = store.get_dreamer_metrics_summary()

    assert report.scanned == {"stale": 0, "superseded": 0, "disputed": 0, "missing_link_pairs": 0}
    assert summary["runs"]["total"] == 1
    assert summary["decision_totals"]["triage_keep"] == 0


def test_condensation_product_is_okf_format() -> None:
    """Sleep (condensation) products must be OKF-structured persistent nodes."""
    from datetime import UTC, datetime

    from synapse.lifecycle.condensation import DeterministicArchiveCondenser

    nodes = [
        Node(
            metadata=NodeMetadata(id="mem_aaa", title="Decision A"),
            content="## Context\nOld setup.\n\n## Decision\nSwitch to X.",
            file_path=Path("active/mem_aaa.md"),
        ),
        Node(
            metadata=NodeMetadata(id="mem_bbb", title="Decision B"),
            content="## Context\nNew constraint.\n\n## Decision\nAdopt Y.",
            file_path=Path("active/mem_bbb.md"),
        ),
    ]
    condenser = DeterministicArchiveCondenser()
    draft = condenser.synthesize(nodes, now=datetime.now(UTC))

    # OKF requires these three sections plus Takeaway/Sources (docs/okf.md).
    assert "## Context" in draft.content
    assert "## Decision" in draft.content
    assert "## Consequences" in draft.content
    assert "## Takeaway" in draft.content
    # Source provenance lives in the typed ## Sources section.
    assert "## Sources" in draft.content
    assert "[[mem_aaa]]" in draft.content
    assert "[[mem_bbb]]" in draft.content

def test_archive_superseded_resolves_chain_and_missing_superseded_by(tmp_path: Path) -> None:
    """Chained supersession archives via the ACTIVE chain terminal; missing
    superseded_by archives directly (chain end with no successor)."""
    from synapse.models import NodeStatus

    config, runtime_paths = make_runtime(tmp_path)
    dreamer = make_dreamer(tmp_path)
    store = dreamer._get_store()

    # Chain: old -> middle -> terminal(ACTIVE, file exists on disk)
    old = make_node("mem_old", status=NodeStatus.SUPERSEDED)
    middle = make_node("mem_middle", status=NodeStatus.SUPERSEDED)
    terminal = make_node("mem_terminal", status=NodeStatus.ACTIVE)
    old = old.model_copy(update={"metadata": old.metadata.model_copy(update={"superseded_by": "mem_middle"})})
    middle = middle.model_copy(update={"metadata": middle.metadata.model_copy(update={"superseded_by": "mem_terminal"})})

    from synapse.storage import write_node_file
    for n in (old, middle, terminal):
        write_node_file(n, base_path=runtime_paths.base)
    store.upsert_node(old)
    store.upsert_node(middle)
    store.upsert_node(terminal)

    # Missing superseded_by, file on disk
    orphan = make_node("mem_orphan", status=NodeStatus.SUPERSEDED)
    write_node_file(orphan, base_path=runtime_paths.base)
    store.upsert_node(orphan)

    archived_ids = dreamer._archive_superseded([old, orphan], store, [])

    assert "mem_old" in archived_ids      # chain terminal ACTIVE -> archived
    assert "mem_orphan" in archived_ids   # missing superseded_by -> archived
    from synapse.storage import locate_active_node_file

    assert locate_active_node_file(runtime_paths.base, "mem_old") is None
    assert locate_active_node_file(runtime_paths.base, "mem_orphan") is None
    assert locate_active_node_file(runtime_paths.base, "mem_terminal") is not None  # terminal untouched
    dreamer.close()


def test_archive_superseded_kept_for_disputed_terminal_archived_for_missing(tmp_path: Path) -> None:
    """DISPUTED chain terminal keeps the node (live disagreement); a terminal
    missing from the index (archived/deleted) makes the node archivable."""
    from synapse.models import NodeStatus

    config, runtime_paths = make_runtime(tmp_path)
    dreamer = make_dreamer(tmp_path)
    store = dreamer._get_store()

    # Case 1: DISPUTED terminal -> kept.
    old_disputed = make_node("mem_old_disputed", status=NodeStatus.SUPERSEDED)
    disputed = make_node("mem_term_disputed", status=NodeStatus.DISPUTED)
    old_disputed = old_disputed.model_copy(
        update={"metadata": old_disputed.metadata.model_copy(update={"superseded_by": "mem_term_disputed"})}
    )
    # Case 2: terminal NOT in the index (archived/deleted) -> archived.
    old_missing = make_node("mem_old_missing", status=NodeStatus.SUPERSEDED)
    old_missing = old_missing.model_copy(
        update={"metadata": old_missing.metadata.model_copy(update={"superseded_by": "mem_gone_terminal"})}
    )

    from synapse.storage import write_node_file
    for n in (old_disputed, disputed, old_missing):
        write_node_file(n, base_path=runtime_paths.base)
    store.upsert_node(old_disputed)
    store.upsert_node(disputed)
    store.upsert_node(old_missing)
    # NOTE: mem_gone_terminal intentionally never upserted (absent from index).

    warnings = []
    archived_ids = dreamer._archive_superseded([old_disputed, old_missing], store, warnings)

    assert archived_ids == ["mem_old_missing"]  # disputed kept, missing terminal archived
    assert any(w.code == "disputed_superseder" for w in warnings)
    from synapse.storage import locate_active_node_file

    assert locate_active_node_file(runtime_paths.base, "mem_old_disputed") is not None
    assert locate_active_node_file(runtime_paths.base, "mem_old_missing") is None
    dreamer.close()


def test_archive_superseded_cyclic_chain_logged_and_kept(tmp_path: Path) -> None:
    """A cyclic chain (A -> B -> A) is kept with a warning, not archived."""
    from synapse.models import NodeStatus

    config, runtime_paths = make_runtime(tmp_path)
    dreamer = make_dreamer(tmp_path)
    store = dreamer._get_store()

    a = make_node("mem_cyc_a", status=NodeStatus.SUPERSEDED)
    b = make_node("mem_cyc_b", status=NodeStatus.SUPERSEDED)
    a = a.model_copy(update={"metadata": a.metadata.model_copy(update={"superseded_by": "mem_cyc_b"})})
    b = b.model_copy(update={"metadata": b.metadata.model_copy(update={"superseded_by": "mem_cyc_a"})})

    from synapse.storage import write_node_file
    for n in (a, b):
        write_node_file(n, base_path=runtime_paths.base)
    store.upsert_node(a)
    store.upsert_node(b)

    warnings = []
    archived_ids = dreamer._archive_superseded([a], store, warnings)

    assert archived_ids == []
    assert any(w.code == "cyclic_supersession" for w in warnings)
    dreamer.close()


class _SynthesisSamplingClient:
    """Returns one hub payload per synthesis call; records prompts."""

    name = "fake-synthesis"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
        self.calls.append(prompt)
        return {
            "title": "Kafka consumer offset handling across services",
            "takeaway": "All services resume from committed kafka offsets.",
            "details": "Members establish that offsets gate resume behavior.",
            "sources": [],
        }

    def decide_memory_write(self, request):  # pragma: no cover - not used by Dreamer
        raise AssertionError(f"Unexpected write decision request: {request}")


def _typed_cluster_node(node_id: str, *, project: str = "kafka", okf_type: str = "fact") -> Node:
    metadata = NodeMetadata(
        id=node_id,
        title=f"Kafka fact {node_id}",
        type=NodeType.PERSISTENT,
        okf_type=okf_type,
        okf_version=1,
        project=project,
    )
    return Node(
        metadata=metadata,
        content=f"## Takeaway\nfact {node_id}\n\n## Details\ncontent {node_id}\n\n## Sources\n- x",
        file_path=Path("active") / f"{node_id}.md",
    )


def test_synthesis_creates_fact_hub_for_typed_cluster(tmp_path: Path) -> None:
    """P3: a project-scoped cluster of >= min size typed nodes above the cosine
    threshold produces ONE fact hub; hubs are not re-clustered as members."""
    config, runtime_paths = make_runtime(tmp_path)
    config.dreamer.thresholds.synthesis_cosine = 0.80
    config.dreamer.thresholds.min_synthesis_cluster = 4
    client = _SynthesisSamplingClient()
    dreamer = Dreamer(config, runtime_paths=runtime_paths, sampling_client=client)

    base = [0.0, 0.0, 1.0]
    try:
        with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
            for i in range(4):
                node = _typed_cluster_node(f"mem_kafka_{i}")
                jitter = 0.01 * i
                store.upsert_node(node, embedding=[base[0] + jitter, base[1], base[2] - jitter])
            # Untyped + transcript-adjacent nodes must not join the cluster.
            store.upsert_node(
                Node(
                    metadata=NodeMetadata(id="mem_untyped", title="Untyped note"),
                    content="plain note",
                    file_path=Path("active/mem_untyped.md"),
                ),
                embedding=[0.0, 0.0, 1.0],
            )

            results = dreamer._run_synthesis(store, warnings=[])

        assert len(results) == 1
        result = results[0]
        assert result.hub_node_id.startswith("mem_")
        assert set(result.member_ids) == {f"mem_kafka_{i}" for i in range(4)}
        assert client.calls and "Cluster members:" in client.calls[0]

        from synapse.storage import locate_active_node_file, read_node_file

        hub_path = locate_active_node_file(runtime_paths.base, result.hub_node_id)
        assert hub_path is not None
        hub = read_node_file(hub_path)
        assert hub.metadata.okf_type == "fact"
        assert hub.metadata.type is NodeType.PERSISTENT
        assert "## Sources" in hub.content
        assert all(f"- mem_kafka_{i}" in hub.content for i in range(4))

        # Second run: cluster now covered by the hub -> skipped.
        with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
            store.upsert_node(hub, embedding=[0.0, 0.0, 1.0])
            calls_before = len(client.calls)
            results2 = dreamer._run_synthesis(store, warnings=[])
        assert results2 == []
        assert len(client.calls) == calls_before
    finally:
        dreamer.close()


def test_synthesis_skips_small_clusters(tmp_path: Path) -> None:
    config, runtime_paths = make_runtime(tmp_path)
    config.dreamer.thresholds.min_synthesis_cluster = 4
    client = _SynthesisSamplingClient()
    dreamer = Dreamer(config, runtime_paths=runtime_paths, sampling_client=client)
    try:
        with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
            for i in range(3):
                store.upsert_node(_typed_cluster_node(f"mem_small_{i}"), embedding=[0.0, 0.0, 1.0])
            results = dreamer._run_synthesis(store, warnings=[])
        assert results == []
        assert client.calls == []
    finally:
        dreamer.close()


def test_repoint_superseded_links_updates_wiki_links_and_sources(tmp_path: Path) -> None:
    """When A -> B -> C (ACTIVE), active node D referencing [[mem_a]] and source mem_a
    is deterministically repointed to [[mem_c]] and source mem_c."""
    from synapse.storage import locate_active_node_file, read_node_file, write_node_file

    config, runtime_paths = make_runtime(tmp_path)
    dreamer = make_dreamer(tmp_path)
    store = dreamer._get_store()
    try:
        node_a = make_node("mem_a", status=NodeStatus.SUPERSEDED).model_copy(
            update={"metadata": NodeMetadata(id="mem_a", title="Node A", status=NodeStatus.SUPERSEDED, superseded_by="mem_b")}
        )
        node_b = make_node("mem_b", status=NodeStatus.SUPERSEDED).model_copy(
            update={"metadata": NodeMetadata(id="mem_b", title="Node B", status=NodeStatus.SUPERSEDED, superseded_by="mem_c")}
        )
        node_c = make_node("mem_c", status=NodeStatus.ACTIVE)
        node_d = Node(
            metadata=NodeMetadata(
                id="mem_d",
                title="Referencing node D",
                status=NodeStatus.ACTIVE,
                type=NodeType.PERSISTENT,
                sources=["mem_a"],
            ),
            content="See [[mem_a]] and [[mem_b|Alias B]] for details.",
            file_path=Path("active/mem_d.md"),
        )
        for n in (node_a, node_b, node_c, node_d):
            write_node_file(n, base_path=runtime_paths.base)
            store.upsert_node(n)

        repointed = dreamer._repoint_superseded_links(store)
        assert repointed == 2

        d_path = locate_active_node_file(runtime_paths.base, "mem_d")
        assert d_path is not None
        updated_d = read_node_file(d_path)
        assert "[[mem_c]]" in updated_d.content
        assert "[[mem_c|Alias B]]" in updated_d.content
        assert "[[mem_a]]" not in updated_d.content
        assert updated_d.metadata.sources == ["mem_c"]
    finally:
        dreamer.close()


def test_archive_divergent_collision_quarantines_and_preserves_canonical(tmp_path: Path) -> None:
    """When canonical .archive/<id>.md already exists with divergent content,
    quarantine the old file to .archive/.collisions/ and promote the retiring file."""
    from synapse.storage import archive_node_path, read_node_file, write_node_file

    config, runtime_paths = make_runtime(tmp_path)
    dreamer = make_dreamer(tmp_path)
    try:
        old_arch = Node(
            metadata=NodeMetadata(
                id="mem_col_1",
                title="Collision Node",
                type=NodeType.PERSISTENT,
                status=NodeStatus.SUPERSEDED,
                superseded_by="mem_old_winner",
            ),
            content="Old archived snapshot",
        )
        arch_path = archive_node_path(runtime_paths.archive, "mem_col_1")
        arch_path.parent.mkdir(parents=True, exist_ok=True)
        arch_path.write_text(f"---\nid: mem_col_1\ntitle: Collision Node\nstatus: superseded\nsuperseded_by: mem_old_winner\n---\n\nOld archived snapshot\n", encoding="utf-8")

        active_retiring = Node(
            metadata=NodeMetadata(
                id="mem_col_1",
                title="Collision Node",
                type=NodeType.PERSISTENT,
                status=NodeStatus.SUPERSEDED,
                superseded_by="mem_new_winner",
            ),
            content="Newer retiring body",
        )
        written = write_node_file(active_retiring, base_path=runtime_paths.base)
        retiring = active_retiring.model_copy(update={"file_path": written.relative_to(runtime_paths.base)})
        warnings: list = []
        archived = dreamer._archive_nodes([retiring], reason="semantic_merge", warnings=warnings)
        assert len(archived) == 1
        assert not any(w.code == "archive_collision" for w in warnings)
        assert not written.exists()
        canonical = read_node_file(arch_path)
        assert canonical.metadata.superseded_by == "mem_new_winner"
        collisions = list((runtime_paths.archive / ".collisions").glob("mem_col_1.*.md"))
        assert len(collisions) == 1
        assert read_node_file(collisions[0]).metadata.superseded_by == "mem_old_winner"
    finally:
        dreamer.close()


def test_complete_linkage_prevents_chaining_and_respects_project_mode(tmp_path: Path) -> None:
    """Complete-linkage rejects transitive A-B-C chains when cos(A,C) < floor_cosine."""
    config, runtime_paths = make_runtime(tmp_path)
    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        a = _typed_cluster_node("mem_a").model_copy(update={"metadata": _typed_cluster_node("mem_a").metadata.model_copy(update={"project": "Synapse"})})
        b = _typed_cluster_node("mem_b").model_copy(update={"metadata": _typed_cluster_node("mem_b").metadata.model_copy(update={"project": "synapse"})})
        c = _typed_cluster_node("mem_c").model_copy(update={"metadata": _typed_cluster_node("mem_c").metadata.model_copy(update={"project": "synapse"})})
        # A=(1,0,0), B=(0.9, 0.43589, 0), C=(0.62, 0.7846, 0): cos(A,B)=0.9, cos(B,C)~0.9, cos(A,C)=0.62 < 0.84
        store.upsert_node(a, embedding=[1.0, 0.0, 0.0])
        store.upsert_node(b, embedding=[0.92, 0.391918, 0.0])
        store.upsert_node(c, embedding=[0.64, 0.768375, 0.0])

        clusters = store.find_complete_linkage_clusters(
            cosine_threshold=0.86,
            floor_cosine=0.84,
            min_size=2,
            max_size=6,
            project_mode="explicit_slug",
        )
        assert len(clusters) == 1
        assert [n.id for n in clusters[0]] == ["mem_a", "mem_b"]


def test_three_way_provenance_excludes_unknown_and_cycles(tmp_path: Path) -> None:
    """Same-root dedups and unknown/cyclic leaves must not inflate verified_independent_multi_source."""
    from synapse.observability import _three_way_provenance_stats

    config, runtime_paths = make_runtime(tmp_path)
    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        # 1) Real 2-session node
        n1 = _typed_cluster_node("mem_real").model_copy(
            update={"metadata": _typed_cluster_node("mem_real").metadata.model_copy(update={"sources": ["mem_session_1", "mem_session_2"]})}
        )
        # 2) Same-root dedup (two intermediates from one session)
        m1 = _typed_cluster_node("mem_m1").model_copy(
            update={"metadata": _typed_cluster_node("mem_m1").metadata.model_copy(update={"sources": ["mem_session_x"], "status": NodeStatus.SUPERSEDED})}
        )
        m2 = _typed_cluster_node("mem_m2").model_copy(
            update={"metadata": _typed_cluster_node("mem_m2").metadata.model_copy(update={"sources": ["mem_session_x"], "status": NodeStatus.SUPERSEDED})}
        )
        n2 = _typed_cluster_node("mem_dedup").model_copy(
            update={"metadata": _typed_cluster_node("mem_dedup").metadata.model_copy(update={"sources": ["mem_m1", "mem_m2"]})}
        )
        # 3) Unknown + cycle node
        n3 = _typed_cluster_node("mem_unk").model_copy(
            update={"metadata": _typed_cluster_node("mem_unk").metadata.model_copy(update={"sources": ["mem_missing_leaf", "mem_unk"]})}
        )
        for n in (n1, m1, m2, n2, n3):
            store.upsert_node(n)

        stats = _three_way_provenance_stats(store, runtime_paths=runtime_paths)
        assert stats["verified_independent_multi_source"] == 1
        assert stats["same_root_dedup_count"] == 1
        assert stats["unknown_or_cyclic_provenance_count"] == 1


def test_eval_strict_stale_as_miss_scores_zero_on_missing_and_follows_archive(tmp_path: Path) -> None:
    """resolve_labels follows superseded_by in .archive/<id>.md; strict_stale_as_miss=True scores missing queries as 0.0."""
    from synapse.eval.golden import GoldenQuery, RelevantDoc
    from synapse.eval.harness import EvalReport, QueryResult, resolve_labels
    from synapse.retrieval import RetrievalPipeline
    from synapse.storage import archive_node_path, write_node_file

    config, runtime_paths = make_runtime(tmp_path)
    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        winner = _typed_cluster_node("mem_winner")
        write_node_file(winner, base_path=runtime_paths.base)
        store.upsert_node(winner)

    arch_path = archive_node_path(runtime_paths.archive, "mem_retired")
    arch_path.parent.mkdir(parents=True, exist_ok=True)
    arch_path.write_text(
        "---\nid: mem_retired\ntitle: Retired\nstatus: superseded\nsuperseded_by: mem_winner\n---\n\nOld\n",
        encoding="utf-8",
    )

    queries = [
        GoldenQuery(id="q1", query="kafka", relevant=(RelevantDoc(node_id="mem_retired", tier="must"),)),
        GoldenQuery(id="q2", query="gone", relevant=(RelevantDoc(node_id="mem_deleted", tier="must"),)),
    ]
    with RetrievalPipeline(config, runtime_paths=runtime_paths) as pipeline:
        resolved, stale, no_label = resolve_labels(queries, pipeline, archive_dir=runtime_paths.archive)
    assert [d.node_id for d in resolved["q1"].relevant] == ["mem_winner"]
    assert stale["q1"] == 0
    assert no_label == ["q2"]

    r1 = QueryResult(query_id="q1", slice="en", recall5=1.0, mrr10=1.0, top1_relevant=True)
    r2 = QueryResult(query_id="q2", slice="en", recall5=0.0, mrr10=0.0, top1_relevant=False)
    non_strict = EvalReport(per_query=[r1, r2], queries_without_labels=["q2"], strict_stale_as_miss=False).to_dict()
    strict = EvalReport(per_query=[r1, r2], queries_without_labels=["q2"], strict_stale_as_miss=True).to_dict()
    assert non_strict["overall"]["recall@5"] == 1.0
    assert strict["overall"]["recall@5"] == 0.5


def test_archive_collision_crash_recovery_startup_sync(tmp_path: Path) -> None:
    """Simulate crash between canonical replacement and source unlink; startup_sync keeps canonical superseded intact."""
    from synapse.storage import archive_node_path, read_node_file, write_node_file
    from synapse.sync import SyncManager

    config, runtime_paths = make_runtime(tmp_path)
    retired = Node(
        metadata=NodeMetadata(
            id="mem_crash_1",
            title="Crash Node",
            type=NodeType.PERSISTENT,
            status=NodeStatus.SUPERSEDED,
            superseded_by="mem_winner_final",
        ),
        content="Retiring node body",
    )
    active_path = write_node_file(retired, base_path=runtime_paths.base)
    arch_path = archive_node_path(runtime_paths.archive, "mem_crash_1")
    arch_path.parent.mkdir(parents=True, exist_ok=True)
    arch_path.write_text(active_path.read_text(encoding="utf-8"), encoding="utf-8")

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        SyncManager(config, runtime_paths=runtime_paths, store=store, debounce_seconds=0.0).startup_sync()
        loaded = store.get_node("mem_crash_1")
        assert loaded is not None
        assert loaded.metadata.status is NodeStatus.SUPERSEDED
        assert loaded.metadata.superseded_by == "mem_winner_final"
        assert read_node_file(arch_path).metadata.superseded_by == "mem_winner_final"


def test_similar_pairs_matches_scalar_cosine_for_every_pair() -> None:
    """The numpy scan must return exactly the pairs the scalar helper would, across block boundaries."""
    import random

    from synapse.storage.sqlite import _safe_cosine_similarity, _similar_pairs

    rng = random.Random(3)

    def unit(dim: int) -> list[float]:
        v = [rng.gauss(0, 1) for _ in range(dim)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v]

    base = unit(16)
    embeddings: dict[str, list[float]] = {}
    for i in range(40):
        v = [b + rng.gauss(0, 0.15) for b in base] if i % 2 else unit(16)  # near-duplicates mixed with noise
        norm = sum(x * x for x in v) ** 0.5
        embeddings[f"n{i:02d}"] = [x / norm for x in v]
    embeddings["odd_dim"] = unit(8)  # a different length must never match anything
    ids = sorted(embeddings)

    for threshold in (0.0, 0.5, 0.84, 0.9):
        expected: dict[tuple[str, str], float] = {}
        for index, id_a in enumerate(ids):
            for id_b in ids[index + 1 :]:
                similarity = _safe_cosine_similarity(embeddings[id_a], embeddings[id_b])
                if similarity is not None and similarity >= threshold:
                    expected[(id_a, id_b)] = similarity
        got = {(a, b): s for a, b, s in _similar_pairs(ids, embeddings, threshold, block_rows=7)}
        assert got.keys() == expected.keys()
        assert all(abs(got[key] - expected[key]) < 1e-9 for key in got)
        if threshold == 0.5:
            assert expected, "test data must produce pairs or the comparison is vacuous"
        assert not any("odd_dim" in key for key in got)
