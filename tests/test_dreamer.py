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