from __future__ import annotations

import json
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
from synapse.okf.episode import render_episode_map_prompt
from synapse.okf.transcript_format import TranscriptTurn, V2_MARKER, parse_transcript
from synapse.server.app import create_app
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
        ),
        content=content if content is not None else _transcript_content(),
        file_path=Path("active") / "mem_fixture_session_test.md",
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


def test_short_episode_stamps_watermark_without_llm_calls(tmp_path: Path) -> None:
    config, paths, transcript = _setup(tmp_path)
    scanner = FakeClient()
    gatekeeper = FakeClient()
    runner = _runner(config, paths, scanner, gatekeeper)
    try:
        result = runner.run(transcript.id)
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            stored = store.get_node(transcript.id)
        assert result.stamped is True
        assert result.stages == {}
        assert scanner.prompts == []
        assert gatekeeper.prompts == []
        assert stored is not None and stored.metadata.distilled_turn == 2
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
        assert "Do NOT filter or reject lessons based on `turn_eligibility`" in prompt
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
    config, paths, transcript = _setup(tmp_path, content=content, watermark=2)
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
    prompt = render_episode_map_prompt(
        turns="## T2 Assistant\nAssistant analysis and a Tool result",
        global_user_spine="T1: user intent",
        chunk_index=1,
        chunk_count=1,
    )
    assert "Do NOT filter or reject lessons based on `turn_eligibility`" in prompt
    assert "assistant_prose_is_not_evidence" in prompt
    assert "T1: user intent" in prompt


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



def test_episode_run_metrics_are_recorded_with_episode_funnel(tmp_path: Path) -> None:
    config, paths, _transcript = _setup(tmp_path)
    runner = _runner(config, paths, FakeClient(), FakeClient())
    runs = [
        EpisodeRun(
            "mem_fixture_session_one",
            "session-one",
            "episode-one",
            100,
            2,
            1,
            lessons=[{}, {}],
            ops=[
                {"action": "created"},
                {"action": "updated"},
                {"action": "none"},
                {"action": "quarantined"},
            ],
            stamped=True,
        ),
        EpisodeRun("mem_fixture_session_two", "session-two", "episode-two", 10, 1, 1, stamped=True),
    ]
    try:
        runner.record_run_metrics(
            runs,
            started_at="2026-10-17T00:00:00Z",
            completed_at="2026-10-17T00:00:02Z",
            duration_ms=2000,
            transcripts_scanned=3,
            llm_failures=2,
            archived_transcripts=4,
        )
    finally:
        runner.close()

    with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        row = store.connection.execute("SELECT * FROM distiller_runs ORDER BY id DESC LIMIT 1").fetchone()
    assert row["transcripts_scanned"] == 3
    assert row["transcripts_distilled"] == 1
    assert row["transcripts_skipped"] == 1
    assert row["items_extracted"] == 2
    assert row["items_invalid"] == 1
    assert row["items_quarantined"] == 1
    assert row["decider_creates"] == 1
    assert row["decider_complements"] == 1
    assert row["decider_supersedes"] == row["decider_downgrades"] == 0
    assert row["llm_failures"] == 2
    assert row["archived_transcripts"] == 4

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
                    "reason": "new knowledge",
                }
            ]
        },
        lesson_count=1,
    )[0]
    assert op.justification == "new knowledge"
    turns = [TranscriptTurn(number=2, role="Assistant", body="Source analysis found src/authz/handler.go handles the error.")]
    verified, unverified = episode_module.verify_evidence(
        op.evidence,
        turns={"T2": episode_module._turn_verification_text(turns[0])},
    )
    assert verified == op.evidence
    assert unverified == []


def test_integrate_prompt_defines_gatekeeper_policy_and_chinese_okf_contract() -> None:
    prompt = episode_module.render_episode_integrate_prompt(
        lessons=[Lesson("API Boundary", "接口返回 404 时代表 group 不存在。", ["T1"], "pitfall").as_dict()],
        same_session_nodes=[],
        retrieved_nodes=[],
        evidence_context=[{"lesson_index": 0, "evidence": [{"turn": "T1", "role": "User", "text": "404"}]}],
        global_user_spine="T1: user later corrected the 404 hypothesis",
    )
    for required in (
        "Strict Gatekeeper",
        "refuted by later turns",
        "default to update",
        "specific English title",
        "verified=true",
        "evidence",
        "Sources",
        "Global User Spine",
    ):
        assert required in prompt


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


def test_rest_episode_close_runs_in_background_and_serializes_same_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, paths, _transcript = _setup(tmp_path)
    config.distiller.enabled = True
    scanner_clients: list[FakeClient] = []
    gatekeeper_clients: list[FakeClient] = []

    def make_scanner(_config: SynapseConfig) -> FakeClient:
        client = FakeClient()
        scanner_clients.append(client)
        return client

    def make_gatekeeper(_config: SynapseConfig) -> FakeClient:
        client = FakeClient()
        gatekeeper_clients.append(client)
        return client

    monkeypatch.setattr(EpisodeDistiller, "make_scanner_client", staticmethod(make_scanner))
    monkeypatch.setattr(EpisodeDistiller, "make_gatekeeper_client", staticmethod(make_gatekeeper))
    lock = threading.Lock()
    completed = threading.Event()
    active = 0
    max_active = 0
    completed_count = 0

    def fake_run(_runner: EpisodeDistiller, _session_key: str) -> None:
        nonlocal active, max_active, completed_count
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
            completed_count += 1
            if completed_count == 2:
                completed.set()

    monkeypatch.setattr(EpisodeDistiller, "run", fake_run)
    app = create_app(config, runtime_paths=paths, sampling_client=FakeClient())
    with TestClient(app) as client:
        first = client.post("/api/episode/close", json={"session_key": "same-key", "reason": "shutdown"})
        second = client.post("/api/episode/close", json={"session_key": "same-key", "reason": "switch"})
        assert first.status_code == second.status_code == 200
        assert first.json() == {"status": "accepted", "session_key": "same-key", "reason": "shutdown"}
        assert second.json() == {"status": "accepted", "session_key": "same-key", "reason": "switch"}
        assert completed.wait(timeout=3)
    assert max_active == 1
    assert len(scanner_clients) == 2
    assert len(gatekeeper_clients) == 2
    assert all(client.closed for client in scanner_clients + gatekeeper_clients)


def test_single_large_pending_turn_short_skips_and_stamps(tmp_path: Path) -> None:
    content = V2_MARKER + "\n\n## T1 User\n" + "one turn is not enough " + "x" * 700
    config, paths, transcript = _setup(tmp_path, content=content)
    scanner = FakeClient()
    runner = _runner(config, paths, scanner)
    try:
        result = runner.run(transcript.id)
        with SQLiteNodeStore(paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            stored = store.get_node(transcript.id)
        assert result.turns == 1
        assert result.stages == {}
        assert scanner.prompts == []
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
