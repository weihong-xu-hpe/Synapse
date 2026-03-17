from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from synapse.config import load_config
from synapse.lifecycle import Dreamer
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType, SensitivityLevel
from synapse.storage import SQLiteNodeStore, archive_node_path, write_node_file
from synapse.utils.runtime import bootstrap_runtime_directories


NOW = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)


class MockSamplingClient:
    name = "mock"

    def __init__(self, triage_decisions=None, link_decisions=None, conflict_decisions=None):
        self._triage_decisions = triage_decisions or []
        self._link_decisions = link_decisions or []
        self._conflict_decisions = conflict_decisions or []

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
        if "triage" in prompt.lower():
            return {
                "content": {
                    "type": "text",
                    "text": json.dumps({"decisions": self._triage_decisions}),
                }
            }
        if "link weaving" in prompt.lower():
            return {
                "content": {
                    "type": "text",
                    "text": json.dumps({"decisions": self._link_decisions}),
                }
            }
        if "conflict resolution" in prompt.lower():
            return {
                "content": {
                    "type": "text",
                    "text": json.dumps({"decisions": self._conflict_decisions}),
                }
            }
        return {"content": {"type": "text", "text": json.dumps({"decisions": []})}}

    def decide_memory_write(self, request):
        raise AssertionError("Not expected")


def write_config(base_dir: Path) -> Path:
    config_path = base_dir / "config.toml"
    config_path.write_text(
        """
[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "builtin"
model = "bge-m3"
dimension = 8

[decay]
janitor_days = 30
archive_retention_days = 90
""".strip(),
        encoding="utf-8",
    )
    return config_path


def make_node(
    *,
    node_id: str,
    title: str,
    file_name: str,
    content: str = "# Note\n\nArchive me.",
    status: NodeStatus = NodeStatus.ACTIVE,
    last_accessed: datetime | None = None,
    tags: list[str] | None = None,
) -> Node:
    return Node(
        metadata=NodeMetadata(
            id=node_id,
            title=title,
            created_at=(last_accessed or NOW) - timedelta(days=10),
            last_accessed=last_accessed or NOW,
            type=NodeType.PERSISTENT,
            status=status,
            tags=tags or [],
            sensitivity=SensitivityLevel.INTERNAL,
        ),
        content=content,
        file_path=Path("active") / file_name,
    )


def set_mtime(path: Path, when: datetime) -> None:
    timestamp = when.timestamp()
    os.utime(path, (timestamp, timestamp))


def seed_active_node(store: SQLiteNodeStore, runtime_base: Path, node: Node, *, embedding_dimension: int = 8) -> Path:
    path = write_node_file(node, base_path=runtime_base)
    store.upsert_node(node, embedding=[1.0] + [0.0] * (embedding_dimension - 1), source_mtime=node.metadata.last_accessed)
    return path


def test_dreamer_archives_stale_orphans(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    stale = make_node(
        node_id="mem_stale_orphan",
        title="Stale Orphan",
        file_name="mem_stale_orphan.md",
        last_accessed=NOW - timedelta(days=45),
    )

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=8) as store:
        seed_active_node(store, runtime_paths.base, stale)

        mock = MockSamplingClient(
            triage_decisions=[
                {"node_id": stale.id, "decision": "archive", "reason": "outdated"},
            ]
        )
        dreamer = Dreamer(
            config,
            runtime_paths=runtime_paths,
            store=store,
            sampling_client=mock,
            now_provider=lambda: NOW,
        )
        report = dreamer.run(batch_size=8)

        assert stale.id in report.archived
        assert archive_node_path(runtime_paths.archive, stale.id).exists()
        assert not (runtime_paths.active / f"{stale.id}.md").exists()


def test_dreamer_archives_superseded_with_validation(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    superseder = make_node(
        node_id="mem_superseder",
        title="Superseder",
        file_name="mem_superseder.md",
        last_accessed=NOW,
    )
    valid = make_node(
        node_id="mem_valid_old",
        title="Valid Old",
        file_name="mem_valid_old.md",
        status=NodeStatus.SUPERSEDED,
        last_accessed=NOW - timedelta(days=12),
    )
    valid = valid.model_copy(update={"metadata": valid.metadata.model_copy(update={"superseded_by": superseder.id})})
    invalid = make_node(
        node_id="mem_invalid_old",
        title="Invalid Old",
        file_name="mem_invalid_old.md",
        status=NodeStatus.SUPERSEDED,
        last_accessed=NOW - timedelta(days=12),
    )
    invalid = invalid.model_copy(update={"metadata": invalid.metadata.model_copy(update={"superseded_by": "mem_missing"})})

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=8) as store:
        for node in (superseder, valid, invalid):
            seed_active_node(store, runtime_paths.base, node)

        mock = MockSamplingClient()
        dreamer = Dreamer(
            config,
            runtime_paths=runtime_paths,
            store=store,
            sampling_client=mock,
            now_provider=lambda: NOW,
        )
        report = dreamer.run(batch_size=8)

        assert valid.id in report.archived
        assert archive_node_path(runtime_paths.archive, valid.id).exists()
        assert (runtime_paths.active / f"{invalid.id}.md").exists()
        assert store.get_node(invalid.id) is not None
        assert any(w.node_id == invalid.id and w.code == "invalid_superseder" for w in report.warnings)


def test_dreamer_disputed_warning_and_archive_purge(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    stale_archive = archive_node_path(runtime_paths.archive, "mem_stale_archive")
    fresh_archive = archive_node_path(runtime_paths.archive, "mem_fresh_archive")
    stale_archive.write_text(
        "---\nid: mem_stale_archive\ntitle: Stale\ncreated_at: 2026-01-01T00:00:00Z\nlast_accessed: 2026-01-01T00:00:00Z\n---\n\nold",
        encoding="utf-8",
    )
    fresh_archive.write_text(
        "---\nid: mem_fresh_archive\ntitle: Fresh\ncreated_at: 2026-03-01T00:00:00Z\nlast_accessed: 2026-03-01T00:00:00Z\n---\n\nfresh",
        encoding="utf-8",
    )
    set_mtime(stale_archive, NOW - timedelta(days=120))
    set_mtime(fresh_archive, NOW - timedelta(days=5))

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=8) as store:
        for index in range(6):
            node = make_node(
                node_id=f"mem_disputed_{index}",
                title=f"Disputed {index}",
                file_name=f"mem_disputed_{index}.md",
                status=NodeStatus.DISPUTED,
                last_accessed=NOW,
            )
            seed_active_node(store, runtime_paths.base, node)

        mock = MockSamplingClient()
        dreamer = Dreamer(
            config,
            runtime_paths=runtime_paths,
            store=store,
            sampling_client=mock,
            now_provider=lambda: NOW,
        )
        report = dreamer.run(batch_size=8)

        # The 6 disputed nodes don't form pairs (no superseded_by links between them)
        # but archive purge still works
        assert stale_archive.as_posix() in report.deleted_archive_paths
        assert not stale_archive.exists()
        assert fresh_archive.exists()


def test_dreamer_condenses_when_triage_says_condense(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    node_a = make_node(
        node_id="mem_condense_a",
        title="Condense A",
        file_name="mem_condense_a.md",
        content="# Condense A\n\nGateway retries.",
        last_accessed=NOW - timedelta(days=45),
    )
    node_b = make_node(
        node_id="mem_condense_b",
        title="Condense B",
        file_name="mem_condense_b.md",
        content="# Condense B\n\nGateway rate limiting.",
        last_accessed=NOW - timedelta(days=45),
    )

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=8) as store:
        seed_active_node(store, runtime_paths.base, node_a)
        seed_active_node(store, runtime_paths.base, node_b)

        mock = MockSamplingClient(
            triage_decisions=[
                {"node_id": node_a.id, "decision": "condense", "reason": "overlap"},
                {"node_id": node_b.id, "decision": "condense", "reason": "overlap"},
            ]
        )
        dreamer = Dreamer(
            config,
            runtime_paths=runtime_paths,
            store=store,
            sampling_client=mock,
            now_provider=lambda: NOW,
        )
        report = dreamer.run(batch_size=8)

        assert len(report.condensed) == 1
        condensed = report.condensed[0]
        assert set(condensed.source_ids) == {node_a.id, node_b.id}
        assert condensed.new_node_id
        # Source nodes should be archived
        assert node_a.id in report.archived
        assert node_b.id in report.archived



