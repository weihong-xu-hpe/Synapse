from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synapse.config import SynapseConfig
from synapse.lifecycle.episode_distiller import EpisodeDistiller, EpisodeOp, EpisodeRun, build_episode_chunks
from synapse.models import Node, NodeMetadata, NodeType
from synapse.okf import EvidenceEntry
from synapse.okf.transcript_format import TranscriptTurn, V2_MARKER, parse_transcript
from synapse.storage import SQLiteNodeStore, write_node_file
from synapse.sync import SyncManager
from synapse.utils.runtime import RuntimePaths, bootstrap_runtime_directories


class FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.prompts: list[str] = []

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response


def _setup(tmp_path: Path) -> tuple[SynapseConfig, RuntimePaths, Node]:
    config = SynapseConfig.with_defaults(tmp_path)
    base = tmp_path / "copy"
    config.memory.base_path = base
    config.memory.archive_path = base / ".archive"
    config.embedding.provider = "builtin"
    config.reranker.provider = "builtin"
    paths = bootstrap_runtime_directories(config)
    transcript = Node(
        metadata=NodeMetadata(
            id="mem_session_test_fixture",
            title="Session summary — test",
            type=NodeType.TRANSIENT,
            session_key="test-session-key",
        ),
        content=(
            V2_MARKER
            + "\n\n## T1 User\nThe cache key remains stable for retried writes."
            + "\n\n## T2 Assistant\nThe implementation preserves the same key."
        ),
        file_path=Path("active") / "mem_session_test_fixture.md",
    )
    write_node_file(transcript, base_path=paths.base)
    manager = SyncManager(config, runtime_paths=paths, debounce_seconds=0.0)
    try:
        manager.startup_sync()
    finally:
        manager.close()
    return config, paths, transcript


def _runner(config: SynapseConfig, paths: RuntimePaths, client: FakeClient) -> EpisodeDistiller:
    return EpisodeDistiller(config, runtime_paths=paths, sampling_client=client)


def test_map_chunks_preserve_turn_boundaries() -> None:
    turns = [
        TranscriptTurn(number=1, role="User", body="a" * 30),
        TranscriptTurn(number=2, role="Assistant", body="b" * 30),
        TranscriptTurn(number=3, role="User", body="c" * 30),
    ]
    chunks = build_episode_chunks(turns, max_chars=50)

    assert len(chunks) == 3
    reconstructed: list[str] = []
    for text, turn_ids in chunks:
        parsed = parse_transcript(V2_MARKER + "\n\n" + text)
        assert [turn.id for turn in parsed.turns] == turn_ids
        reconstructed.extend(turn.id for turn in parsed.turns)
    assert reconstructed == ["T1", "T2", "T3"]


def test_failed_summary_does_not_advance_watermark(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient(error=RuntimeError("temporary model error")))
    try:
        with pytest.raises(RuntimeError, match="temporary model error"):
            runner.run(transcript.id)
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            stored = store.get_node(transcript.id)
        assert stored is not None
        assert stored.metadata.distilled_turn is None
        assert stored.metadata.distilled_hash is None
    finally:
        runner.close()


def test_unknown_update_target_is_quarantined(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient())
    turn = TranscriptTurn(number=1, role="User", body="The cache key remains stable for retried writes.")
    op = EpisodeOp(
        action="update",
        title="Stable Cache Key Behavior",
        okf_type="fact",
        target_id="mem_20261017_missing_target",
        content=(
            "## Takeaway\n\nRetrying a write preserves its cache key.\n\n"
            "## Details\n\nThe same key is reused on retry.\n\n## Sources\n\n- source"
        ),
        evidence=[EvidenceEntry(turn="T1", quote="The cache key remains stable for retried writes.")],
    )
    run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode-test", 60, 1, 1)
    try:
        result = runner._apply_op(
            transcript,
            [turn],
            op,
            run,
            allowed_target_ids={op.target_id},
            allowed_evidence_turns={"T1"},
        )
        assert result["action"] == "quarantined"
        assert result["target_id"] == op.target_id
        quarantine = list((paths.base / ".distiller-quarantine").glob("*.jsonl"))
        assert len(quarantine) == 1
        assert op.target_id in quarantine[0].read_text(encoding="utf-8")
    finally:
        runner.close()


def test_create_retry_uses_canonical_id_and_is_idempotent(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient())
    turn = TranscriptTurn(number=1, role="User", body="The cache key remains stable for retried writes.")
    op = EpisodeOp(
        action="create",
        title="Stable Cache Key Behavior",
        okf_type="fact",
        justification="No existing node describes this retry invariant.",
        content=(
            "## Takeaway\n\nRetrying a write preserves its cache key.\n\n"
            "## Details\n\nThe same key is reused on retry.\n\n## Sources\n\n- source"
        ),
        evidence=[EvidenceEntry(turn="T1", quote="The cache key remains stable for retried writes.")],
    )
    run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode-test", 60, 1, 1)
    try:
        first = runner._apply_op(
            transcript, [turn], op, run, allowed_target_ids=set(), allowed_evidence_turns={"T1"}
        )
        second = runner._apply_op(
            transcript, [turn], op, run, allowed_target_ids=set(), allowed_evidence_turns={"T1"}
        )
        assert first["action"] == "created"
        assert second["action"] == "unchanged"
        assert first["node_id"] == second["node_id"]
        assert first["node_id"].startswith("mem_2026")
    finally:
        runner.close()
