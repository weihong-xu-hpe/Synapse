from __future__ import annotations

from pathlib import Path

from synapse.config import SynapseConfig
from synapse.lifecycle import Dreamer
from synapse.models import Node, NodeMetadata, NodeStatus
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

    # OKF requires these three sections.
    assert "## Context" in draft.content
    assert "## Decision" in draft.content
    assert "## Consequences" in draft.content
    # Source provenance is preserved as an appendix section.
    assert "## Merged From" in draft.content
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
    assert not (runtime_paths.base / "active" / "mem_old.md").exists()
    assert not (runtime_paths.base / "active" / "mem_orphan.md").exists()
    assert (runtime_paths.base / "active" / "mem_terminal.md").exists()  # terminal untouched
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
    assert (runtime_paths.base / "active" / "mem_old_disputed.md").exists()
    assert not (runtime_paths.base / "active" / "mem_old_missing.md").exists()
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
