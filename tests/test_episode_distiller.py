from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from synapse.config import SynapseConfig
from synapse.lifecycle.episode_distiller import EpisodeDistiller, EpisodeOp, EpisodeRun, Lesson, _parse_lessons, _parse_ops, _turn_evidence_records, build_episode_chunks
from synapse.models import Node, NodeMetadata, NodeType
from synapse.okf import EvidenceEntry
from synapse.okf.episode import render_episode_map_prompt
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

def test_map_prompt_marks_only_contract_eligible_turns() -> None:
    turns = [
        TranscriptTurn(number=1, role="User", body="The selected scope is constrained by membership."),
        TranscriptTurn(number=2, role="User", body="Why is the scope constrained?"),
        TranscriptTurn(number=3, role="Assistant", body="Assistant prose is context, not evidence."),
        TranscriptTurn(
            number=4,
            role="Assistant",
            body="Tool call context.",
            tool_sections=["### Tool T4.1 read\ncall: {}\nresult [ok]: scope membership is checked before allow"],
        ),
    ]
    records = _turn_evidence_records(turns)
    by_turn = {record["turn"]: record for record in records}
    assert by_turn["T1"]["eligible"] is True
    assert by_turn["T2"]["eligible"] is False
    assert by_turn["T3"]["eligible"] is False
    assert by_turn["T4"]["eligible"] is True
    prompt = render_episode_map_prompt(
        turns="\n\n".join(f"## {turn.id} {turn.role}\n{turn.body}" for turn in turns),
        turn_eligibility=records,
        chunk_index=1,
        chunk_count=1,
    )
    assert '"turn":"T3","role":"Assistant","eligible":false' in prompt
    assert "assistant_prose_is_not_evidence" in prompt
    assert "scope membership is checked before allow" in prompt

    with pytest.raises(ValueError, match="outside its supplied eligible-turn set"):
        _parse_lessons(
            {"lessons": [{"topic": "scope", "claim": "claim", "evidence_turns": ["T3"], "evidence_quotes": [{"turn": "T3", "quote": "Assistant prose is context, not evidence."}]}]},
            {"T1": "The selected scope is constrained by membership.", "T4": "scope membership is checked before allow"},
        )

    source = "The selected scope is constrained by membership."
    parsed = _parse_lessons(
        {
            "lessons": [{
                "topic": "scope",
                "claim": "membership constrains the scope",
                "evidence_turns": ["T1"],
                "evidence_quotes": [{"turn": "T1", "quote": "scope is constrained by membership"}],
            }]
        },
        {"T1": source},
    )
    assert parsed[0].evidence_quotes == [EvidenceEntry("T1", "scope is constrained by membership")]
    with pytest.raises(ValueError, match="appear verbatim"):
        _parse_lessons(
            {
                "lessons": [{
                    "topic": "scope",
                    "claim": "unverified",
                    "evidence_turns": ["T1"],
                    "evidence_quotes": [{"turn": "T1", "quote": "a fabricated quote"}],
                }]
            },
            {"T1": source},
        )


def test_integrate_evidence_overflow_splits_batches_without_dropping_excerpts(tmp_path: Path) -> None:
    config, paths, _transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient())
    turns = [
        TranscriptTurn(number=1, role="User", body="First exact evidence quote supports a durable rule."),
        TranscriptTurn(number=2, role="User", body="Second exact evidence quote supports another durable rule."),
    ]
    lessons = [
        Lesson("first topic", "first claim", ["T1"], [EvidenceEntry("T1", turns[0].body)]),
        Lesson("second topic", "second claim", ["T2"], [EvidenceEntry("T2", turns[1].body)]),
    ]
    try:
        one_sizes = [
            len(json.dumps(runner._evidence_context(turns, [lesson]), ensure_ascii=False))
            for lesson in lessons
        ]
        combined_size = len(json.dumps(runner._evidence_context(turns, lessons), ensure_ascii=False))
        runner.settings.evidence_context_token_budget = max(one_sizes) + 1
        assert combined_size > runner.settings.evidence_context_token_budget
        batches = runner._integrate_lesson_batches(turns, lessons)
        assert [[lesson.topic for lesson in batch] for batch in batches] == [["first topic"], ["second topic"]]
        excerpt_texts = [
            record["text"]
            for batch in batches
            for block in runner._evidence_context(turns, batch)
            for record in block["evidence"]
        ]
        assert excerpt_texts == [turns[0].body, turns[1].body]
    finally:
        runner.close()


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
        record = json.loads(quarantine[0].read_text(encoding="utf-8").splitlines()[0])
        assert record["target_node_ids"] == [op.target_id]
        assert record["failure_reason"] == f"Unknown or inactive update target: {op.target_id}"
        assert record["evidence"] == [{"turn": "T1", "quote": "The cache key remains stable for retried writes."}]
    finally:
        runner.close()

def test_malformed_integrate_operation_is_quarantined(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient())
    op = _parse_ops({"ops": [None]}, lesson_count=1)[0]
    run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode-invalid-op", 0, 0, 1)
    try:
        result = runner._apply_op(
            transcript, [], op, run, allowed_target_ids=set(), allowed_evidence_turns=set()
        )
        assert result["action"] == "quarantined"
        quarantine = list((paths.base / ".distiller-quarantine").glob("*.jsonl"))
        record = json.loads(quarantine[0].read_text(encoding="utf-8").splitlines()[0])
        assert record["failure_reason"] == "Integrate operation 0 must be a JSON object"
        assert record["evidence"] == []
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

def test_overlong_takeaway_uses_focused_repair_before_write(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    repaired = "Retry writes with the same cache key."
    client = FakeClient(response={"takeaway": repaired})
    runner = _runner(config, paths, client)
    turn = TranscriptTurn(number=1, role="User", body="The cache key remains stable for retried writes.")
    op = EpisodeOp(
        action="create",
        title="Stable Cache Key Behavior",
        okf_type="fact",
        justification="No matching node describes this retry invariant.",
        content=(
            "## Takeaway\n\n" + "A durable retry statement " + "x" * 210 + "\n\n"
            "## Details\n\nThe same key is reused on retry.\n\n## Sources\n\n- source"
        ),
        evidence=[EvidenceEntry(turn="T1", quote="The cache key remains stable for retried writes.")],
    )
    run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode-repair-test", 60, 1, 1)
    try:
        result = runner._apply_op(
            transcript, [turn], op, run, allowed_target_ids=set(), allowed_evidence_turns={"T1"}
        )
        assert result["action"] == "created"
        assert len(client.prompts) == 1
        assert "Rewrite only the supplied OKF Takeaway line" in client.prompts[0]
        assert "Retry writes with the same cache key." in runner._load_node(result["node_id"]).content
        assert run.stages["takeaway_repair"].calls == 1
    finally:
        runner.close()
