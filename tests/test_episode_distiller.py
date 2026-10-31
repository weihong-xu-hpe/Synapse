from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from synapse.config import DeciderEndpointSettings, SynapseConfig
from synapse.lifecycle import episode_distiller as episode_module
from synapse.lifecycle.episode_distiller import (
    EpisodeDistiller,
    EpisodeOp,
    EpisodeRun,
    Lesson,
    _dehydrate_turn,
    _parse_lessons,
    _parse_ops,
    build_episode_chunks,
)
from synapse.models import Node, NodeMetadata, NodeType
from synapse.okf import EvidenceEntry
from synapse.okf.transcript_format import TranscriptTurn, V2_MARKER, parse_transcript
from synapse.okf.transcripts import content_sha256
from synapse.server.app import create_app
from synapse.server.service import SynapseServiceError, SynapseServerService, _omitted_knowledge_lines
from synapse.storage import SQLiteNodeStore, write_node_file
from synapse.sync import SyncManager
from synapse.utils.runtime import RuntimePaths, bootstrap_runtime_directories


class FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.prompts: list[str] = []
        self.closed = False

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.closed = True


def _transcript_content(*, user_chars: int = 40, assistant_chars: int = 40) -> str:
    return (
        V2_MARKER
        + "\n\n## T1 User\nThe cache key remains stable for retried writes. "
        + "u" * user_chars
        + "\n\n## T2 Assistant\nThe implementation preserves the same key. "
        + "a" * assistant_chars
    )


def _setup(
    tmp_path: Path,
    *,
    content: str | None = None,
    watermark: int | None = None,
    previous_hash: str | None = None,
) -> tuple[SynapseConfig, RuntimePaths, Node]:
    config = SynapseConfig.with_defaults(tmp_path)
    base = tmp_path / "copy"
    config.memory.base_path = base
    config.memory.archive_path = base / ".archive"
    config.embedding.provider = "builtin"
    config.reranker.provider = "builtin"
    paths = bootstrap_runtime_directories(config)
    transcript = Node(
        metadata=NodeMetadata(
            id="mem_fixture_session_test",
            title="Session summary — test",
            type=NodeType.TRANSIENT,
            session_key="test-session-key",
            distilled_turn=watermark,
            distilled_hash=previous_hash,
        ),
        content=content if content is not None else _transcript_content(),
        file_path=Path("transcripts") / "mem_fixture_session_test.md",
    )
    write_node_file(transcript, base_path=paths.base)
    manager = SyncManager(config, runtime_paths=paths, debounce_seconds=0.0)
    try:
        manager.startup_sync()
    finally:
        manager.close()
    return config, paths, transcript


def _runner(
    config: SynapseConfig,
    paths: RuntimePaths,
    scanner: FakeClient,
    gatekeeper: FakeClient | None = None,
) -> EpisodeDistiller:
    return EpisodeDistiller(
        config,
        runtime_paths=paths,
        sampling_client=scanner,
        gatekeeper_client=gatekeeper,
    )



def _write_persistent_fact(
    runner: EpisodeDistiller,
    transcript: Node,
    *,
    node_id: str,
    title: str,
) -> None:
    runner.service.integrate_knowledge(
        title=title,
        content=(
            "## Takeaway\n\n缓存键保持稳定。\n\n"
            "## Details\n\n重试复用同一缓存键。\n\n"
            f"## Sources\n\n- [[{transcript.id}]]\n"
        ),
        node_type=NodeType.PERSISTENT,
        action="create",
        okf_type="fact",
        okf_version=1,
        sources=[transcript.id],
        node_id=node_id,
        validate_okf=True,
    )


def test_episode_pipeline_settings_build_scanner_pool_and_gatekeeper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import synapse.server.decider as decider_module

    config = SynapseConfig.with_defaults(tmp_path)
    config.distiller.episode.scanner_endpoints = [
        DeciderEndpointSettings(base_url="http://scanner-a/v1", model="scan-a"),
        DeciderEndpointSettings(base_url="http://scanner-b/v1", model="scan-b"),
    ]
    captured: list[Any] = []
    monkeypatch.setattr(decider_module, "LocalLLMDecider", lambda settings: captured.append(settings) or settings)

    scanner_settings = EpisodeDistiller.make_scanner_client(config)
    config.distiller.episode.gatekeeper_endpoint = None
    gatekeeper_settings = EpisodeDistiller.make_gatekeeper_client(config)

    assert len(scanner_settings.endpoints) == 2
    assert [endpoint.extra_body["reasoning_effort"] for endpoint in scanner_settings.endpoints] == ["low", "low"]
    expected_concurrency = max(16, config.decider.endpoint_max_concurrency)
    assert scanner_settings.endpoint_max_concurrency == expected_concurrency
    assert gatekeeper_settings.endpoint_max_concurrency == expected_concurrency
    assert config.distiller.episode.chunk_chars == 400_000
    assert len(captured) == 2


def test_episode_chunks_preserve_turn_boundaries() -> None:
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


def test_dehydration_strips_noise_and_drops_empty_assistant_turns() -> None:
    noise_only = TranscriptTurn(
        number=1,
        role="Assistant",
        body="",
        tool_sections=[
            "### Tool T1.1 todo\nresult [ok]: todo marker",
            '### Tool T1.2 hub\ncall: {"op":"wait"}\nresult [ok]: poll marker',
        ],
    )
    assert _dehydrate_turn(noise_only) is None

    mixed = TranscriptTurn(
        number=2,
        role="Assistant",
        body="Durable assistant analysis",
        tool_sections=[
            "### Tool T2.1 vibe_wait\nresult [ok]: transient polling marker",
            '### Tool T2.2 hub\ncall: {"op":"status"}\nresult [ok]: transient hub marker',
            '### Tool T2.3 hub\ncall: {"op":"search"}\nresult [ok]: durable query result',
        ],
    )
    dehydrated = _dehydrate_turn(mixed)
    assert dehydrated is not None
    assert len(dehydrated.tool_sections) == 1
    assert "durable query result" in dehydrated.tool_sections[0]


def test_short_episode_is_deferred_until_forced_close(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    scanner = FakeClient(response={"lessons": []})
    runner = _runner(config, paths, scanner)
    try:
        result = runner.run(transcript.id)
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            stored = store.get_node(transcript.id)
        assert result.deferred and not result.stamped
        assert scanner.prompts == []
        assert stored is not None
        assert stored.metadata.distilled_turn is None
        assert stored.metadata.deferred_hash == content_sha256(stored.content)

        forced = runner.run(transcript.id, force=True)
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            stored = store.get_node(transcript.id)
        assert forced.stamped and not forced.deferred
        assert len(scanner.prompts) == 1
        assert stored is not None and stored.metadata.distilled_turn == 2
        assert stored.metadata.deferred_hash is None
    finally:
        runner.close()


def test_failed_forced_short_episode_remains_retryable_by_sweep(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    scanner = FakeClient(error=ConnectionError("scanner unavailable"))
    runner = _runner(config, paths, scanner)
    try:
        assert runner.run(transcript.id).deferred
        with pytest.raises(ConnectionError) as caught:
            runner.run(transcript.id, force=True)
        assert caught.value.distiller_stage == "scanner"
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            stored = store.get_node(transcript.id)
        assert stored is not None and stored.metadata.deferred_hash is None
        assert stored.metadata.distilled_turn is None and stored.metadata.force_retry is True
        old = time.time() - 7200
        os.utime(paths.transcripts / f"{transcript.id}.md", (old, old))
        selected = runner._distiller_helpers.select_transcripts()
        assert [node.id for node in selected] == [transcript.id]
        scanner.error = None
        scanner.response = {"lessons": []}
        from synapse.lifecycle.distiller_scheduler import DistillerScheduler

        runs, attempts = DistillerScheduler(config, runtime_paths=paths)._distill(selected, runner)
        assert len(runs) == 1 and runs[0].stamped
        assert attempts[0]["outcome"] == "skipped"
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            recovered = store.get_node(transcript.id)
        assert recovered is not None and recovered.metadata.distilled_turn == 2
        assert recovered.metadata.force_retry is None
    finally:
        runner.close()

def test_extract_episode_defers_transcript_stamp_until_integration(tmp_path: Path) -> None:
    content = V2_MARKER + "\n\n## T1 User\n" + "u" * 350 + "\n\n## T2 Assistant\n" + "a" * 350
    config, paths, transcript = _setup(tmp_path, content=content)
    scanner = FakeClient(response={"lessons": []})
    runner = _runner(config, paths, scanner)
    try:
        extracted = runner.extract_episode(transcript.id)
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            after_extraction = store.get_node(transcript.id)
        assert extracted[0].id == transcript.id
        assert len(extracted[1]) == 2
        assert len(extracted[2]) == 2
        assert after_extraction is not None
        assert after_extraction.metadata.distilled_turn is None
        assert after_extraction.metadata.distilled_hash is None

        result = runner.integrate_episode(*extracted)
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            after_integration = store.get_node(transcript.id)
        assert result.stamped is True
        assert after_integration is not None and after_integration.metadata.distilled_turn == 2
    finally:
        runner.close()


def test_stage_one_dehydrates_noise_and_scans_full_user_spine(tmp_path: Path) -> None:
    user_one = "Initial request: map the cache ownership boundary. " + "u" * 300
    assistant_body = "The cache is owned by the persistence component. " + "a" * 330
    user_two = "No, change the boundary to the retry adapter. " + "v" * 120
    noise = [
        "todo",
        "vibe_wait",
        "vibe_list",
        "vibe_kill",
        "vibe_send",
        "wait",
    ]
    sections = [
        f"### Tool T2.{index} {name}\ncall: {{}}\nresult [ok]: NOISE_{name.upper()}_MARKER_{'n' * 80}"
        for index, name in enumerate(noise, 1)
    ]
    sections.extend(
        f'### Tool T2.{index} hub\ncall: {{"op":"{operation}"}}\nresult [ok]: NOISE_HUB_{operation.upper()}_MARKER'
        for index, operation in enumerate(("wait", "list", "status", "poll", "kill"), 10)
    )
    sections.append(
        '### Tool T2.20 hub\ncall: {"op":"search","query":"cache owner"}\nresult [ok]: '
        "RETAINED_TOOL_EVIDENCE"
    )
    content = (
        V2_MARKER
        + f"\n\n## T1 User\n{user_one}"
        + f"\n\n## T2 Assistant\n{assistant_body}\n" + "\n".join(sections)
        + f"\n\n## T3 User\n{user_two}"
        + "\n\n## T4 Assistant\n### Tool T4.1 wait\ncall: {}\nresult [ok]: DROP_EMPTY_ASSISTANT"
    )
    config, paths, updated = _setup(tmp_path, content=content)
    scanner = FakeClient(response={"lessons": []})
    gatekeeper = FakeClient()
    runner = _runner(config, paths, scanner, gatekeeper)
    try:
        result = runner.run(updated.id)
        prompt = scanner.prompts[0]
        assert result.turns == 3
        assert "T1: Initial request" in prompt
        assert "T3: No, change the boundary" in prompt
        assert "NOISE_" not in prompt
        assert "DROP_EMPTY_ASSISTANT" not in prompt
        assert "RETAINED_TOOL_EVIDENCE" in prompt
        assert gatekeeper.prompts == []
    finally:
        runner.close()


def test_incremental_global_user_spine_includes_old_and_corrective_turns(tmp_path: Path) -> None:
    old_intent = "Original intent: keep the old route. " + "o" * 380
    correction = "不对，改成让 retry adapter owns the route. " + "c" * 330
    content = (
        V2_MARKER
        + f"\n\n## T1 User\n{old_intent}"
        + "\n\n## T2 Assistant\n" + "b" * 350
        + f"\n\n## T3 User\n{correction}"
        + "\n\n## T4 Assistant\n" + "d" * 350
    )
    previous_content = content.split("\n\n## T3 User", 1)[0]
    config, paths, transcript = _setup(tmp_path, content=content, watermark=2, previous_hash=content_sha256(previous_content))
    scanner = FakeClient(response={"lessons": []})
    runner = _runner(config, paths, scanner)
    try:
        result = runner.run(transcript.id)
        prompt = scanner.prompts[0]
        spine = prompt.split("Global User Spine (all User turns in the complete transcript, chronological):\n", 1)[1]
        spine = spine.split("\nDehydrated episode turns:", 1)[0]
        old_line = next(line for line in spine.splitlines() if line.startswith("T1: "))
        assert result.turns == 2
        assert len(old_line.removeprefix("T1: ")) == 350
        assert "T3: 不对，改成" in spine
        assert "T1 User" not in prompt.split("Dehydrated episode turns:", 1)[1]
    finally:
        runner.close()


def test_stage_one_lessons_are_not_rejected_for_turn_eligibility() -> None:
    parsed = _parse_lessons(
        {
            "lessons": [
                {
                    "okf_type": "pitfall",
                    "topic": "Assistant Source Analysis",
                    "claim": "助手从源代码分析确认契约边界。",
                    "evidence_turns": ["T2", "T999"],
                }
            ]
        }
    )
    assert parsed == [
        Lesson(
            topic="Assistant Source Analysis",
            claim="助手从源代码分析确认契约边界。",
            evidence_turns=["T2", "T999"],
            okf_type="pitfall",
        )
    ]


def test_evidence_context_includes_assistant_body_and_tool_sections() -> None:
    turn = TranscriptTurn(
        number=12,
        role="Assistant",
        body="Source analysis points to exact error ROOT_CAUSE_TOKEN.",
        tool_sections=["### Tool T12.1 read\nresult [ok]: file path src/service.py"],
    )
    lesson = Lesson("Service Error", "服务根因来自 source analysis。", ["T12"], "pitfall")
    context = EpisodeDistiller._evidence_context([turn], [lesson])
    excerpt = context[0]["evidence"][0]
    assert excerpt["role"] == "Assistant"
    assert "ROOT_CAUSE_TOKEN" in excerpt["text"]
    assert "src/service.py" in excerpt["text"]

    long_turn = TranscriptTurn(number=13, role="Assistant", body="x" * 2000)
    long_context = EpisodeDistiller._evidence_context(
        [long_turn], [Lesson("Long", "long claim", ["T13"])]
    )
    assert len(long_context[0]["evidence"][0]["text"]) == 1500

    mixed_turn = TranscriptTurn(
        number=14,
        role="Assistant",
        body="ASSISTANT_BODY_TOKEN " + "a" * 1200,
        tool_sections=["### Tool T14.1 read\nresult [ok]: TOOL_RESULT_TOKEN " + "t" * 1200],
    )
    mixed_context = EpisodeDistiller._evidence_context(
        [mixed_turn], [Lesson("Mixed", "mixed claim", ["T14"])]
    )
    mixed_excerpt = mixed_context[0]["evidence"][0]["text"]
    assert len(mixed_excerpt) == 1500
    assert "ASSISTANT_BODY_TOKEN" in mixed_excerpt
    assert "TOOL_RESULT_TOKEN" in mixed_excerpt


def test_gatekeeper_receives_stage_two_context_and_owns_integrate_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = (
        V2_MARKER
        + "\n\n## T1 User\nInitial question about ownership. " + "u" * 300
        + "\n\n## T2 Assistant\nAssistant source analysis: exact root cause is in service/cache.py. " + "a" * 260
        + "\n### Tool T2.1 read\ncall: {\"path\":\"service/cache.py\"}\nresult [ok]: retry token is reused."
        + "\n\n## T3 User\nNo, preserve the token only across the same request. " + "c" * 120
    )
    config, paths, transcript = _setup(tmp_path, content=content)
    scanner = FakeClient(
        response={
            "lessons": [
                {
                    "okf_type": "pitfall",
                    "topic": "Retry Token Ownership",
                    "claim": "`service/cache.py` 通过复用同一 retry token 保持幂等；不同请求必须使用新 token。",
                    "evidence_turns": ["T2", "T3"],
                }
            ]
        }
    )
    gatekeeper = FakeClient(
        response={"ops": [{"action": "none", "lesson_index": 0, "reason": "test gate"}]}
    )

    class EmptyRetrieval:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def search(self, query: str, *, top_k: int, update_access: bool, result_filter):
            assert top_k == 5
            assert "Retry Token Ownership" in query
            return SimpleNamespace(results=[])

    monkeypatch.setattr(episode_module, "RetrievalPipeline", lambda *args, **kwargs: EmptyRetrieval())
    runner = _runner(config, paths, scanner, gatekeeper)
    monkeypatch.setattr(runner, "_same_session_nodes", lambda _transcript: [])
    try:
        result = runner.run(transcript.id)
        assert result.lessons[0]["okf_type"] == "pitfall"
        assert len(scanner.prompts) == 1
        assert len(gatekeeper.prompts) == 1
        gate_prompt = gatekeeper.prompts[0]
        assert "Assistant source analysis" in gate_prompt
        assert "service/cache.py" in gate_prompt
        assert "retry token is reused" in gate_prompt
        assert result.ops[0]["action"] == "none"
    finally:
        runner.close()


def test_gatekeeper_verified_false_is_written_without_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient(), FakeClient())
    node_id = "mem_fixture_stable_cache_behavior"
    stored_node = Node(
        metadata=NodeMetadata(
            id=node_id,
            title="Stable Cache Behavior",
            type=NodeType.PERSISTENT,
            okf_type="fact",
            sources=[transcript.id],
        ),
        content="## Takeaway\n\n重试继续使用原有 cache key。\n\n## Details\n\n同一写入的 retry 会复用 key。\n",
        file_path=Path("active") / f"{node_id}.md",
    )
    captured: dict[str, Any] = {}

    def integrate_knowledge(**kwargs: Any) -> dict[str, str]:
        captured.update(kwargs)
        return {"action": "created"}

    monkeypatch.setattr(runner, "_stable_create_id", lambda *_args: node_id)
    monkeypatch.setattr(runner, "_load_node_or_none", lambda _node_id: None)
    monkeypatch.setattr(runner, "_load_node", lambda _node_id: stored_node)
    monkeypatch.setattr(runner, "_repair_takeaway_only", lambda _op, content, _sources, _run: (content, None))
    monkeypatch.setattr(runner.service, "integrate_knowledge", integrate_knowledge)
    op = EpisodeOp(
        action="create",
        title="Stable Cache Behavior",
        okf_type="fact",
        justification="",
        content=(
            "## Takeaway\n\n重试继续使用原有 cache key。\n\n"
            "## Details\n\n同一写入的 retry 会复用 key。\n"
        ),
        evidence=[EvidenceEntry(turn="T1", quote="not a verbatim excerpt")],
        verified=False,
        reason="",
    )
    run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode-test", 50, 1, 1)
    try:
        result = runner._apply_op(
            transcript,
            [TranscriptTurn(number=1, role="User", body="The actual user source differs.")],
            op,
            run,
            allowed_target_ids=set(),
        )
        assert result["action"] == "created"
        assert op.justification == "Gatekeeper approved new knowledge node"
        assert result["verified"] is False
        assert captured["verified"] is False
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            event = store.connection.execute(
                "SELECT node_id, node_type, action, route FROM write_memory_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert event["node_id"] == node_id
        assert event["node_type"] == "persistent"
        assert event["action"] == "created"
        assert event["route"] == "distiller"
        assert not list((paths.base / ".distiller-quarantine").glob("*.jsonl"))
    finally:
        runner.close()


def test_update_preserves_and_merges_related_knowledge_links(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient(), FakeClient())
    old_related_id = "mem_fixture_related_old"
    new_related_id = "mem_fixture_related_new"
    target_id = "mem_fixture_related_target"
    try:
        _write_persistent_fact(runner, transcript, node_id=old_related_id, title="Original Cache Rule")
        _write_persistent_fact(runner, transcript, node_id=new_related_id, title="Retry Cache Rule")
        runner.service.integrate_knowledge(
            title="Cache Ownership Rule",
            content=(
                "## Takeaway\n\n缓存归属保持稳定。\n\n"
                "## Details\n\n更新会保留原有缓存归属。\n\n"
                f"## Sources\n\n- [[{transcript.id}]]\n"
            ),
            node_type=NodeType.PERSISTENT,
            links=[old_related_id],
            action="create",
            okf_type="fact",
            okf_version=1,
            sources=[transcript.id],
            node_id=target_id,
            validate_okf=True,
        )
        op = EpisodeOp(
            action="update",
            title="Cache Ownership Rule",
            okf_type="fact",
            content=(
                "## Takeaway\n\n缓存归属保持稳定。\n\n"
                "## Details\n\n更新会保留原有缓存归属。\n\n更新内容不应清除相关知识链接。\n\n"
                f"## Related\n\n- [[{old_related_id}]]\n- [[{old_related_id}]]\n\n"
                f"## Related\n\n- [[{new_related_id}]]\n- [[{transcript.id}]]\n\n"
                f"## Sources\n\n- [[{transcript.id}]]\n"
            ),
            target_id=target_id,
            related_ids=[new_related_id, old_related_id, target_id, transcript.id, "mem_fixture_not_allowed"],
        )
        result = runner._apply_op(
            transcript,
            [],
            op,
            EpisodeRun(transcript.id, transcript.metadata.session_key, "episode-test", 50, 1, 1),
            allowed_target_ids={target_id, old_related_id, new_related_id, transcript.id},
        )

        assert result["action"] == "updated"
        updated = runner._load_node(target_id)
        assert updated.content.count("## Related") == 1
        related_section = updated.content.split("## Related", 1)[1].split("## Sources", 1)[0]
        assert related_section.count(f"- [[{old_related_id}]]") == 1
        assert related_section.count(f"- [[{new_related_id}]]") == 1
        assert target_id not in related_section
        assert transcript.id not in related_section
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            related_edges = store.connection.execute(
                "SELECT target_id FROM edges WHERE source_id = ? AND target_id IN (?, ?)",
                (target_id, old_related_id, new_related_id),
            ).fetchall()
        assert {str(row["target_id"]) for row in related_edges} == {old_related_id, new_related_id}
    finally:
        runner.close()


def test_create_related_ids_write_related_section_and_knowledge_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient(), FakeClient())
    related_id = "mem_fixture_create_related"
    created_id = "mem_fixture_created_knowledge"
    try:
        _write_persistent_fact(runner, transcript, node_id=related_id, title="Stable Retry Contract")
        monkeypatch.setattr(runner, "_stable_create_id", lambda *_args: created_id)
        op = EpisodeOp(
            action="create",
            title="Retry Ownership Contract",
            okf_type="fact",
            content=(
                "## Takeaway\n\n重试由同一组件继续负责。\n\n"
                "## Details\n\n重试路径继续使用稳定缓存键。\n"
            ),
            related_ids=[related_id],
        )
        result = runner._apply_op(
            transcript,
            [],
            op,
            EpisodeRun(transcript.id, transcript.metadata.session_key, "episode-test", 50, 1, 1),
            allowed_target_ids={related_id},
        )

        assert result["action"] == "created"
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            created = store.get_node(created_id)
            edge = store.connection.execute(
                "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ?",
                (created_id, related_id),
            ).fetchone()
        assert created is not None
        assert f"## Related\n\n- [[{related_id}]]" in created.content
        assert edge is not None
    finally:
        runner.close()




def test_gatekeeper_verified_flag_is_parsed_and_assistant_quotes_verify() -> None:
    op = _parse_ops(
        {
            "ops": [
                {
                    "action": "create",
                    "lesson_index": 0,
                    "title": "Verified Source Anchor",
                    "okf_type": "fact",
                    "verified": False,
                    "content": "body",
                    "justification": "",
                    "evidence": [{"turn": "T2", "quote": "src/authz/handler.go"}],
                    "related_ids": [
                        "mem_fixture_related_one",
                        "mem_fixture_related_one",
                        "mem_fixture_related_two",
                        "mem_fixture_related_three",
                        "mem_fixture_related_four",
                    ],
                    "reason": "new knowledge",
                }
            ]
        },
        lesson_count=1,
    )[0]
    assert op.related_ids == [
        "mem_fixture_related_one",
        "mem_fixture_related_two",
        "mem_fixture_related_three",
    ]
    assert op.justification == "new knowledge"
    turns = [TranscriptTurn(number=2, role="Assistant", body="Source analysis found src/authz/handler.go handles the error.")]
    verified, unverified = episode_module.verify_evidence(
        op.evidence,
        turns={"T2": episode_module._turn_verification_text(turns[0])},
    )
    assert verified == op.evidence
    assert unverified == []




def test_rest_episode_close_returns_disabled_without_starting_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, _transcript = _setup(tmp_path)
    config.distiller.enabled = False
    scanner = FakeClient()
    factory_called = threading.Event()

    def forbidden_factory(_config: SynapseConfig) -> FakeClient:
        factory_called.set()
        return FakeClient()

    monkeypatch.setattr(EpisodeDistiller, "make_scanner_client", staticmethod(forbidden_factory))
    monkeypatch.setattr(EpisodeDistiller, "make_gatekeeper_client", staticmethod(forbidden_factory))
    app = create_app(config, runtime_paths=paths, sampling_client=scanner)
    with TestClient(app) as client:
        response = client.post(
            "/api/episode/close",
            json={"session_key": "test-session-key", "reason": "idle"},
        )
    assert response.status_code == 200
    assert response.json() == {"status": "disabled", "session_key": "test-session-key"}
    assert scanner.prompts == []
    assert not factory_called.wait(timeout=0.1)


def test_rest_close_resolves_session_key_and_serializes_with_real_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, _ = _setup(tmp_path)
    session_key = "rest-close-session"
    service = SynapseServerService(config, runtime_paths=paths)
    service.upsert_session_memory(session_key=session_key, title="Session summary — rest", content=_transcript_content())
    node_id = service._session_node_id(session_key)
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    class BlockingScanner(FakeClient):
        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, **kwargs: Any) -> Any:
            calls.append(prompt)
            started.set()
            if not release.wait(5):
                raise TimeoutError("test scanner was not released")
            return {"lessons": []}

    scanner = BlockingScanner()
    monkeypatch.setattr(EpisodeDistiller, "make_scanner_client", staticmethod(lambda _config: scanner))
    monkeypatch.setattr(EpisodeDistiller, "make_gatekeeper_client", staticmethod(lambda _config: FakeClient()))
    app = create_app(config, runtime_paths=paths, sampling_client=FakeClient())
    try:
        with TestClient(app) as client:
            first = client.post("/api/episode/close", json={"session_key": session_key, "reason": "shutdown"})
            assert first.status_code == 200 and first.json()["status"] == "accepted"
            assert started.wait(5)
            second = client.post("/api/episode/close", json={"session_key": session_key, "reason": "switch"})
            assert second.status_code == 200
            release.set()
            deadline = time.monotonic() + 10
            rows = []
            stored = None
            while time.monotonic() < deadline:
                with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
                    rows = store.connection.execute(
                        "SELECT \"trigger\", failures, attempts_json FROM distiller_runs WHERE \"trigger\" = 'close'"
                    ).fetchall()
                    stored = store.get_node(node_id)
                if len(rows) == 2:
                    break
                time.sleep(0.02)
        assert len(rows) == 2
        assert all(row["failures"] == 0 for row in rows)
        assert all(json.loads(row["attempts_json"])[0]["session"] == node_id for row in rows)
        assert stored is not None and stored.metadata.distilled_turn == 2
        assert len(calls) == 1
    finally:
        release.set()


def test_single_long_turn_is_scanned_on_forced_close(tmp_path: Path) -> None:
    content = V2_MARKER + "\n\n## T1 User\n" + "durable finding " + "x" * 700
    config, paths, transcript = _setup(tmp_path, content=content)
    scanner = FakeClient(response={"lessons": []})
    runner = _runner(config, paths, scanner)
    try:
        result = runner.run(transcript.id, force=True)
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            stored = store.get_node(transcript.id)
        assert result.stamped and len(scanner.prompts) == 1
        assert stored is not None and stored.metadata.distilled_turn == 1
    finally:
        runner.close()


def test_large_episode_uses_map_then_lesson_reduce(tmp_path: Path) -> None:
    bodies = ["u" * 420, "a" * 420, "v" * 420, "b" * 420]
    content = V2_MARKER + "\n\n" + "\n\n".join(
        f"## T{index} {role}\n{body}"
        for index, (role, body) in enumerate(zip(("User", "Assistant", "User", "Assistant"), bodies), 1)
    )
    config, paths, transcript = _setup(tmp_path, content=content)
    config.distiller.episode.chunk_chars = 1_000
    scanner = FakeClient(response={"lessons": []})
    runner = _runner(config, paths, scanner)
    try:
        result = runner.run(transcript.id)
        assert result.chunks == 2
        assert len(scanner.prompts) == 3
        assert "inspect chunk 1 of 2" in scanner.prompts[0]
        assert "inspect chunk 2 of 2" in scanner.prompts[1]
        assert "Stage 1 lesson_reduce" in scanner.prompts[2]
        assert result.stages["map"].calls == 2
        assert result.stages["lesson_reduce"].calls == 1
    finally:
        runner.close()


def test_evidence_budget_splits_batches_without_omitting_turns(tmp_path: Path) -> None:
    config, paths, _transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient())
    turns = [
        TranscriptTurn(number=1, role="User", body="first evidence " + "x" * 1400),
        TranscriptTurn(number=2, role="Assistant", body="second evidence " + "y" * 1400),
    ]
    lessons = [
        Lesson("First", "第一条。", ["T1"]),
        Lesson("Second", "第二条。", ["T2"]),
    ]
    try:
        single_sizes = [
            len(json.dumps(runner._evidence_context(turns, [lesson]), ensure_ascii=False))
            for lesson in lessons
        ]
        runner.settings.evidence_context_token_budget = max(single_sizes) + 1
        batches = runner._integrate_lesson_batches(turns, lessons)
        assert [[lesson.topic for lesson in batch] for batch in batches] == [["First"], ["Second"]]
        excerpt_turns = [
            excerpt["turn"]
            for batch in batches
            for item in runner._evidence_context(turns, batch)
            for excerpt in item["evidence"]
        ]
        assert excerpt_turns == ["T1", "T2"]
    finally:
        runner.close()


def test_takeaway_only_warning_uses_gatekeeper_repair_before_write(tmp_path: Path) -> None:
    config, paths, _transcript = _setup(tmp_path)
    scanner = FakeClient()
    gatekeeper = FakeClient(response={"takeaway": "重试保留同一请求的 cache key。"})
    runner = _runner(config, paths, scanner, gatekeeper)
    op = EpisodeOp(action="create", title="Stable Retry Key", okf_type="fact")
    content = (
        "## Takeaway\n\n" + "x" * 201 + "\n\n"
        "## Details\n\n同一请求的重试沿用原有 cache key。\n\n"
        "## Sources\n\n- [[mem_fixture_session_test]]\n"
    )
    run = EpisodeRun("mem_fixture_session_test", "test-session-key", "episode-repair", 0, 0, 0)
    try:
        repaired, error = runner._repair_takeaway_only(
            op, content, ["mem_fixture_session_test"], run
        )
        assert error is None
        assert "重试保留同一请求的 cache key。" in repaired
        assert len(gatekeeper.prompts) == 1
        assert scanner.prompts == []
    finally:
        runner.close()


def test_changed_processed_prefix_is_not_skipped(tmp_path: Path) -> None:
    old = V2_MARKER + "\n\n## T1 User\n" + "old " * 100 + "\n\n## T2 Assistant\n" + "answer " * 100
    content = old.replace("old ", "corrected ", 1) + "\n\n## T3 User\nnew evidence"
    config, paths, transcript = _setup(
        tmp_path, content=content, watermark=2, previous_hash=content_sha256(old)
    )
    scanner = FakeClient(response={"lessons": []})
    runner = _runner(config, paths, scanner)
    try:
        result = runner.run(transcript.id)
        assert result.stamped
        assert result.turns == 3
        assert "corrected" in scanner.prompts[0]
    finally:
        runner.close()


def test_final_repair_transport_failure_remains_retryable(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    scanner = FakeClient(error=ConnectionError("repair endpoint unavailable"))
    runner = _runner(config, paths, scanner)
    op = EpisodeOp(
        action="create",
        title="Repair Transport Failure",
        okf_type="fact",
        justification="new fact",
        content="## Takeaway\n\nA supported fact.\n",
    )
    run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode", 10, 2, 1)
    try:
        with pytest.raises(SynapseServiceError) as caught:
            runner._apply_op(transcript, parse_transcript(transcript.content).turns, op, run, allowed_target_ids=set())
        assert caught.value.details.get("repair_error")
        assert runner.failure_stage == "gatekeeper"
        assert not list((paths.base / ".distiller-quarantine").glob("*.jsonl"))
    finally:
        runner.close()


def test_rest_close_records_source_lookup_failure_without_raw_session_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, _ = _setup(tmp_path)
    scanner = FakeClient()
    monkeypatch.setattr(EpisodeDistiller, "make_scanner_client", staticmethod(lambda _config: scanner))
    monkeypatch.setattr(EpisodeDistiller, "make_gatekeeper_client", staticmethod(lambda _config: FakeClient()))
    app = create_app(config, runtime_paths=paths, sampling_client=FakeClient())
    with TestClient(app) as client:
        response = client.post("/api/episode/close", json={"session_key": "missing-session", "reason": "idle"})
        assert response.status_code == 200
        deadline = time.monotonic() + 10
        row = None
        while time.monotonic() < deadline:
            with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
                row = store.connection.execute(
                    "SELECT failures, attempts_json FROM distiller_runs WHERE \"trigger\" = 'close'"
                ).fetchone()
            if row is not None:
                break
            time.sleep(0.02)
    assert row is not None and row["failures"] == 1
    attempt = json.loads(row["attempts_json"])[0]
    assert attempt["stage"] == "source_lookup" and attempt["outcome"] == "failed"
    assert attempt["session"].startswith("mem_session_") and "missing-session" not in attempt["session"]
    assert scanner.prompts == []


def test_legacy_transcript_credentials_are_redacted_before_scanner(tmp_path: Path) -> None:
    secret = "opaqueHistoricalToken1234567890~example"
    content = (
        V2_MARKER
        + f'\n\n## T1 User\nTOKEN = "{secret}"'
        + "\n\n## T2 Assistant\nThis historical record contains a credential."
    )
    config, paths, transcript = _setup(tmp_path, content=content)
    scanner = FakeClient(response={"lessons": []})
    runner = _runner(config, paths, scanner)
    try:
        result = runner.run(transcript.id, force=True)
        assert result.stamped
        assert len(scanner.prompts) == 1
        assert secret not in scanner.prompts[0]
        assert "[REDACTED_SECRET]" in scanner.prompts[0]
        assert secret in (paths.transcripts / f"{transcript.id}.md").read_text()
    finally:
        runner.close()


def test_final_okf_repair_redacts_credentials_before_model_call(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    secret = "opaqueFinalRepairToken1234567890"
    client = FakeClient(response={
        "title": "Safe Repair",
        "content": "## Takeaway\n\nA finding.\n\n## Details\n\nSupported details.\n\n## Sources\n\n- [[mem_fixture_session_test]]\n",
    })
    service = SynapseServerService(config, runtime_paths=paths, sampling_client=client)
    service._validate_distiller_final_body(
        title="Safe Repair",
        content=f'## Takeaway\n\nA finding.\n\nTOKEN = "{secret}"\n',
        okf_type="fact",
        sources=[transcript.id],
        links=[],
    )
    assert len(client.prompts) == 1
    assert secret not in client.prompts[0]
    assert "[REDACTED_SECRET]" in client.prompts[0]


def test_legacy_session_key_resolves_nonhashed_transcript(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    scanner = FakeClient(response={"lessons": []})
    runner = _runner(config, paths, scanner)
    try:
        result = runner.run("test-session-key", force=True)
        assert result.transcript_id == transcript.id
        assert result.stamped
    finally:
        runner.close()


def test_model_cannot_mark_unmatched_evidence_verified(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient())
    op = EpisodeOp(
        action="create",
        title="Unmatched Evidence Claim",
        okf_type="fact",
        justification="new claim",
        content="## Takeaway\n\nA claim.\n\n## Details\n\nA claim with a mismatched quote.\n",
        evidence=[EvidenceEntry(turn="T1", quote="this quotation is not in the source")],
        verified=True,
    )
    run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode", 10, 2, 1)
    try:
        result = runner._apply_op(transcript, parse_transcript(transcript.content).turns, op, run, allowed_target_ids=set())
        assert result["action"] == "created" and result["verified"] is False
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            stored = store.get_node(result["node_id"])
        assert stored is not None and stored.metadata.verified is False
    finally:
        runner.close()


def test_automatic_update_that_drops_existing_facts_merges_deterministically(tmp_path: Path) -> None:
    """§3.3 daytime path: a lossy candidate is merged by Python, not quarantined.

    The update body is rebuilt from the target body verbatim plus the
    candidate's genuinely new detail lines, so the byte-identical loss gate
    cannot fire, no LLM repair call is spent, and no knowledge is dropped.
    """

    config, paths, transcript = _setup(tmp_path)
    gatekeeper = FakeClient()
    runner = _runner(config, paths, FakeClient(), gatekeeper)
    target_id = "mem_fixture_preserved_fact"
    try:
        _write_persistent_fact(runner, transcript, node_id=target_id, title="Preserved Cache Contract")
        before = runner._load_node(target_id)
        op = EpisodeOp(
            action="update",
            title=before.title,
            okf_type="fact",
            target_id=target_id,
            content="## Takeaway\n\n完全不同的结论。\n\n## Details\n\n缺少先前的缓存约束。\n",
        )
        run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode", 10, 2, 1)
        result = runner._apply_op(transcript, parse_transcript(transcript.content).turns, op, run, allowed_target_ids={target_id})
        assert result["action"] == "updated"
        written = runner._load_node(target_id)
        assert written.content != before.content
        assert "重试复用同一缓存键。" in written.content  # existing knowledge survived
        assert "缺少先前的缓存约束。" in written.content  # candidate delta appended
        assert "缓存键保持稳定。" in written.content  # existing takeaway wins
        assert _omitted_knowledge_lines(before.content, written.content) == []
        assert list((paths.base / ".distiller-quarantine").glob("*.jsonl")) == []
        assert gatekeeper.prompts == []  # deterministic merge needs no LLM
    finally:
        runner.close()


def test_update_loss_risk_is_resolved_without_an_llm_repair(tmp_path: Path) -> None:
    """A configured gatekeeper is never asked to repair a loss-risk merge."""

    config, paths, transcript = _setup(tmp_path)
    target_id = "mem_fixture_repaired_fact"
    gatekeeper = FakeClient(
        {
            "title": "Preserved Cache Contract",
            "content": (
                "## Takeaway\n\n缓存键保持稳定。\n\n"
                "## Details\n\n重试复用同一缓存键。\n\n失败后重试同一请求。\n\n"
                f"## Sources\n\n- [[{transcript.id}]]\n"
            ),
        }
    )
    runner = _runner(config, paths, FakeClient(), gatekeeper)
    try:
        _write_persistent_fact(runner, transcript, node_id=target_id, title="Preserved Cache Contract")
        before = runner._load_node(target_id)
        assert "重试复用同一缓存键。" in before.content
        op = EpisodeOp(
            action="update",
            title=before.title,
            okf_type="fact",
            target_id=target_id,
            content="## Takeaway\n\n失败后重试同一请求。\n\n## Details\n\n失败后重试同一请求。\n",
        )
        run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode", 10, 2, 1)
        result = runner._apply_op(transcript, parse_transcript(transcript.content).turns, op, run, allowed_target_ids={target_id})
        assert result["action"] == "updated"
        written = runner._load_node(target_id)
        assert written.content != before.content
        assert "重试复用同一缓存键。" in written.content
        assert "失败后重试同一请求。" in written.content
        assert _omitted_knowledge_lines(before.content, written.content) == []
        assert list((paths.base / ".distiller-quarantine").glob("*.jsonl")) == []
        assert gatekeeper.prompts == []
    finally:
        runner.close()


def test_update_loss_risk_still_quarantines_when_the_deterministic_merge_cannot_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fallback safety net: a merge that still drops lines is repaired then quarantined.

    The deterministic merge is lossless by construction, so this case is only
    reachable when it fails (here: forced to return a still-lossy body). The
    gatekeeper gets exactly one repair attempt and, if knowledge is still
    missing, the write is quarantined instead of committed.
    """

    config, paths, transcript = _setup(tmp_path)
    target_id = "mem_fixture_repair_failed_fact"
    gatekeeper = FakeClient(
        {
            "title": "Preserved Cache Contract",
            "content": (
                "## Takeaway\n\n失败后重试同一请求。\n\n"
                "## Details\n\n失败后重试同一请求。\n\n"
                f"## Sources\n\n- [[{transcript.id}]]\n"
            ),
        }
    )
    runner = _runner(config, paths, FakeClient(), gatekeeper)
    monkeypatch.setattr(
        "synapse.server.service.deterministic_section_merge",
        lambda _existing, candidate: candidate + "\n\n## Details\n\n仍然缺少先前的缓存约束。\n",
    )
    try:
        _write_persistent_fact(runner, transcript, node_id=target_id, title="Preserved Cache Contract")
        before = runner._load_node(target_id)
        op = EpisodeOp(
            action="update",
            title=before.title,
            okf_type="fact",
            target_id=target_id,
            content="## Takeaway\n\n失败后重试同一请求。\n\n## Details\n\n失败后重试同一请求。\n",
        )
        run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode", 10, 2, 1)
        result = runner._apply_op(transcript, parse_transcript(transcript.content).turns, op, run, allowed_target_ids={target_id})
        assert result["action"] == "quarantined"
        assert runner._load_node(target_id).content == before.content
        assert len(gatekeeper.prompts) == 1
        files = list((paths.base / ".distiller-quarantine").glob("*.jsonl"))
        assert len(files) == 1
        record = json.loads(files[0].read_text().splitlines()[0])
        assert any(item["code"] == "okf_update_loss_risk" for item in record["validator_findings"])
    finally:
        runner.close()


def test_gatekeeper_credentials_are_scrubbed_from_the_written_node(tmp_path: Path) -> None:
    """A gatekeeper body or verbatim quote must not persist a secret value.

    The gatekeeper is asked for exact identifiers and verbatim excerpts, and
    evidence quotes are written into the node, so a copied credential would land
    in durable knowledge. Scrubbing happens after verification, so the node stays
    `verified`.
    """

    secret = "Cli3ntSecr3tValue"
    content = (
        V2_MARKER
        + f"\n\n## T1 User\ncurl --data-urlencode 'client_secret={secret}' https://issuer/oauth2/token"
        + "\n\n## T2 Assistant\nThe request needs the service-identity secret."
    )
    config, paths, transcript = _setup(tmp_path, content=content)
    runner = _runner(config, paths, FakeClient())
    op = EpisodeOp(
        action="create",
        title="Service Identity Token Request Needs The Secret",
        okf_type="procedure",
        justification="new claim",
        content=(
            "## Takeaway\n\n调用 token 端点需要 service-identity 的 client_secret。\n\n"
            f"## Steps\n\n1. 从 secret 读取 client_secret={secret}\n2. 用它换 token。\n"
        ),
        evidence=[EvidenceEntry(turn="T1", quote=f"client_secret={secret}")],
        verified=True,
    )
    run = EpisodeRun(transcript.id, transcript.metadata.session_key, "episode", 10, 2, 1)
    try:
        result = runner._apply_op(
            transcript, parse_transcript(transcript.content).turns, op, run, allowed_target_ids=set()
        )
        assert result["action"] == "created"
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            stored = store.get_node(result["node_id"])
        assert stored is not None
        assert secret not in stored.content
        assert "client_secret=[REDACTED_CREDENTIAL]" in stored.content
        # The persisted evidence quote is scrubbed too, without unverifying the node.
        assert f'- T1: "client_secret=[REDACTED_CREDENTIAL]"' in stored.content
        assert stored.metadata.verified is True
    finally:
        runner.close()
