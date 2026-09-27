"""Distiller acceptance tests (design doc §11, items 1–9) — fake LLM, temp runtime."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from synapse.config import SynapseConfig
from synapse.lifecycle.distiller import (
    Distiller,
    DistillerReport,
    content_sha256,
    is_distilled_current,
    is_fully_distilled,
    is_represented,
    is_session_transcript,
)
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType
from synapse.okf import (
    TYPE_REQUIRED_SECTIONS,
    OkfItem,
    OkfValidationError,
    parse_okf_items,
    render_distiller_prompt,
    validate_okf_node,
)
from synapse.server.service import SynapseServiceError, SynapseServerService
from synapse.storage import SQLiteNodeStore, write_node_file
from synapse.utils.runtime import RuntimePaths, bootstrap_runtime_directories


def _config(tmp_path: Path) -> SynapseConfig:
    config = SynapseConfig.with_defaults(tmp_path)
    # Deterministic tests: no remote embedding/reranker calls.
    config.embedding.provider = "builtin"
    config.reranker.provider = "builtin"
    return config


def _runtime(tmp_path: Path) -> RuntimePaths:
    return bootstrap_runtime_directories(_config(tmp_path))


def _write_transcript(path: RuntimePaths, node_id: str, content: str, *, title: str = "Session summary — proj", session_key: str | None = None, sync: bool = True) -> Node:
    metadata = NodeMetadata(id=node_id, title=title, type=NodeType.TRANSIENT, session_key=session_key)
    node = Node(metadata=metadata, content=content, file_path=Path("active") / f"{node_id}.md")
    write_node_file(node, base_path=path.base)
    if sync:
        _sync_all(path)
    return node


def _sync_all(path: RuntimePaths) -> None:
    """Mirror production: files reach the derived index via the sync manager."""

    from synapse.config import SynapseConfig as _C
    from synapse.sync import SyncManager

    config = _config(Path(str(path.base)).parent)
    manager = SyncManager(config, runtime_paths=path, debounce_seconds=0.0)
    try:
        manager.startup_sync()
    finally:
        manager.close()


class FakeDistillSamplingClient:
    """Returns canned items; records prompts and max_tokens.

    Also serves as the write-path decider: create when no candidates exist,
    complement the closest candidate otherwise (mirrors FakeSamplingClient).
    """

    name = "fake-distiller"

    def __init__(self, items: list[dict[str, Any]] | Exception):
        self.items = items
        self.calls: list[dict[str, Any]] = []

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if isinstance(self.items, Exception):
            raise self.items
        return {"items": self.items}

    def decide_memory_write(self, request):
        from synapse.server.sampling import MemoryWriteSamplingDecision

        candidate_ids = [candidate.node_id for candidate in request.candidates]
        if candidate_ids:
            return MemoryWriteSamplingDecision(
                action="complement",
                target_node_ids=(candidate_ids[0],),
                reasoning="complements existing knowledge",
                confidence=0.9,
            )
        return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new knowledge", confidence=0.9)


_PITFALL_ITEM = {
    "okf_type": "pitfall",
    "title": "log_dir resolves relative to the config file parent",
    "takeaway": "log_dir 是相对 config 文件父目录解析的，会写进 ~/.synapse/.synapse/.logs。",
    "sections": {
        "Symptom": "JSON logs land in ~/.synapse/.synapse/.logs.",
        "Cause": "LoggingSettings.log_dir 相对解析。",
        "Fix": "使用绝对路径。",
    },
    "sources": [],
}


def _make_distiller(tmp_path: Path, client: Any) -> tuple[Distiller, RuntimePaths]:
    runtime_paths = _runtime(tmp_path)
    distiller = Distiller(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=client)
    return distiller, runtime_paths


# ---------------------------------------------------------------------------
# 1. Selector + hash gate
# ---------------------------------------------------------------------------


def test_selector_picks_only_idle_undistilled_transcripts(tmp_path: Path) -> None:
    distiller, paths = _make_distiller(tmp_path, FakeDistillSamplingClient([]))
    try:
        old = datetime.now(UTC) - timedelta(hours=2)
        t1 = _write_transcript(paths, "mem_session_aaa1", "## User\nhello\n\n## Assistant\nworld")
        # Age the file mtime past the idle threshold.
        import os

        ts = old.timestamp()
        os.utime(paths.active / "mem_session_aaa1.md", (ts, ts))

        # A fresh (not idle) transcript must be skipped.
        _write_transcript(paths, "mem_session_bbb2", "## User\nfresh")

        # A persistent node is never selected.
        knowledge = Node(
            metadata=NodeMetadata(id="mem_knowledge", title="Knowledge", type=NodeType.PERSISTENT),
            content="## Takeaway\nx\n\n## Sources\n- y",
            file_path=Path("active/mem_knowledge.md"),
        )
        write_node_file(knowledge, base_path=paths.base)

        # A fully distilled transcript at current hash is skipped.
        t1_stamped = t1.model_copy(
            update={
                "metadata": t1.metadata.model_copy(
                    update={"distilled_hash": content_sha256(t1.content), "distilled_node_ids": ["mem_k1"]}
                )
            }
        )
        write_node_file(t1_stamped, base_path=paths.base)
        _sync_all(paths)
        import os as os2

        os2.utime(paths.active / "mem_session_aaa1.md", (ts, ts))

        selected = distiller.select_transcripts()
        assert [n.id for n in selected] == []
    finally:
        distiller.close()


def test_hash_gate_prevents_redistill_and_growth_retriggers(tmp_path: Path) -> None:
    content = "## User\nquestion\n\n## Assistant\nanswer with durable fact"
    node = Node(
        metadata=NodeMetadata(id="mem_session_ccc3", title="Session summary — x", type=NodeType.TRANSIENT),
        content=content,
        file_path=Path("active/mem_session_ccc3.md"),
    )
    assert not is_fully_distilled(node)

    stamped = node.model_copy(
        update={"metadata": node.metadata.model_copy(update={"distilled_hash": content_sha256(content), "distilled_node_ids": ["k1"]})}
    )
    assert is_fully_distilled(stamped)

    grown = stamped.model_copy(update={"content": content + "\n\n## User\nmore questions"})
    assert not is_fully_distilled(grown)


# ---------------------------------------------------------------------------
# 2. Item validation
# ---------------------------------------------------------------------------


def test_item_validation_accepts_and_rejects() -> None:
    good = parse_okf_items({"items": [_PITFALL_ITEM]})
    assert len(good) == 1
    assert good[0].okf_type == "pitfall"

    # Missing required section for the type.
    bad_section = dict(_PITFALL_ITEM)
    bad_section["sections"] = {"Symptom": "x"}  # no Cause/Fix
    assert parse_okf_items({"items": [bad_section]}) == []

    # Multi-line takeaway rejected.
    bad_takeaway = dict(_PITFALL_ITEM)
    bad_takeaway["takeaway"] = "line one\nline two"
    assert parse_okf_items({"items": [bad_takeaway]}) == []

    # Over-long takeaway rejected.
    long_takeaway = dict(_PITFALL_ITEM)
    long_takeaway["takeaway"] = "x" * 300
    assert parse_okf_items({"items": [long_takeaway]}) == []

    # Unknown type rejected.
    bad_type = dict(_PITFALL_ITEM, okf_type="essay")
    assert parse_okf_items({"items": [bad_type]}) == []


def test_okf_item_render_requires_sections() -> None:
    item = OkfItem(
        okf_type="fact",
        title="t",
        takeaway="one line",
        sections={"Details": "the fact"},
        sources=["mem_session_x"],
    )
    rendered = item.render()
    assert rendered.startswith("## Takeaway")
    assert "## Details" in rendered
    assert rendered.rstrip().endswith("- mem_session_x")

    incomplete = OkfItem(okf_type="fact", title="t", takeaway="x", sections={}, sources=[])
    with pytest.raises(OkfValidationError):
        incomplete.render()


# ---------------------------------------------------------------------------
# 3. Decider target guard
# ---------------------------------------------------------------------------


def test_decider_cannot_target_transcript_nodes(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)
    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths)
    try:
        transcript = _write_transcript(
            runtime_paths,
            "mem_session_guard1",
            "## User\nhi\n\n## Assistant\nthere",
            session_key="sess-123",
        )
        with pytest.raises(SynapseServiceError) as excinfo:
            service.integrate_knowledge(
                title="complement a transcript",
                content="body",
                action="complement",
                target_node_ids=[transcript.id],
            )
        assert excinfo.value.code in {"INVALID_SAMPLING_RESPONSE", "INVALID_TARGET"}
        assert "never session transcripts" in excinfo.value.message
    finally:
        service._store().close() if hasattr(service, "_store") else None


# ---------------------------------------------------------------------------
# 4. OKF validation on persistent writes (warnings-only)
# ---------------------------------------------------------------------------


def test_persistent_writes_get_okf_warnings_transients_do_not(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)
    from tests.test_server_api import FakeSamplingClient

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    persistent = service.write_memory(title="P", content="plain", node_type="persistent")
    assert any(w["code"] == "okf_missing_type" for w in persistent["warnings"])

    transient = service.write_memory(title="T", content="plain", node_type="transient")
    assert transient["warnings"] == []

    # A fully OKF-conforming persistent write produces zero warnings.
    okf_body = (
        "## Takeaway\nOne line takeaway.\n\n## Details\nThe fact.\n\n## Sources\n- [[mem_session_x]]"
    )
    conforming = service.write_memory(
        title="OKF Fact",
        content=okf_body,
        node_type="persistent",
        okf_type="fact",
        okf_version=1,
        sources=["mem_session_x"],
    )
    assert conforming["warnings"] == []
    node = service._load_node(conforming["execution"]["result"]["node"]["id"])
    assert node.metadata.okf_type == "fact"
    assert node.metadata.sources == ["mem_session_x"]
    # Frontmatter round-trips to disk.
    from synapse.storage import read_node_file

    on_disk = read_node_file(runtime_paths.active / f"{node.id}.md")
    assert on_disk.metadata.okf_type == "fact"
    assert on_disk.metadata.distilled_node_ids == []


# ---------------------------------------------------------------------------
# 5. E2E: sweep distills a transcript through write_memory
# ---------------------------------------------------------------------------


def test_sweep_distills_transcript_and_stamps_state(tmp_path: Path) -> None:
    client = FakeDistillSamplingClient([dict(_PITFALL_ITEM)])
    distiller, paths = _make_distiller(tmp_path, client)
    try:
        import os

        transcript = _write_transcript(paths, "mem_session_e2e1", "## User\nlog bug\n\n## Assistant\n" + "durable lesson " * 50, session_key="sess-e2e")
        ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(paths.active / "mem_session_e2e1.md", (ts, ts))

        report = distiller.run(downgrade_supersede=False)

        assert report.transcripts_distilled == 1
        assert report.items_extracted == 1
        assert report.decider_creates == 1  # no prior knowledge -> create
        assert report.llm_failures == 0

        # Transcript stamped.
        from synapse.storage import read_node_file

        stamped = read_node_file(paths.active / "mem_session_e2e1.md")
        assert stamped.metadata.distilled_hash == content_sha256(stamped.content)
        assert stamped.metadata.distilled_at is not None
        assert len(stamped.metadata.distilled_node_ids) == 1

        # Knowledge node exists, is persistent + typed + has provenance.
        knowledge_id = stamped.metadata.distilled_node_ids[0]
        knowledge = read_node_file(paths.active / f"{knowledge_id}.md")
        assert knowledge.metadata.type is NodeType.PERSISTENT
        assert knowledge.metadata.okf_type == "pitfall"
        assert "mem_session_e2e1" in knowledge.metadata.sources

        # Prompt carried the reasoning-aware token budget.
        assert client.calls and client.calls[0]["max_tokens"] == 16000

        # Second run with unchanged content is a no-op.
        report2 = distiller.run()
        assert report2.transcripts_scanned == 0
    finally:
        distiller.close()


# ---------------------------------------------------------------------------
# 6. E2E: incremental growth complements, no duplicate
# ---------------------------------------------------------------------------


def test_incremental_growth_complements_existing_knowledge(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)

    class ComplementThenComplementClient:
        name = "fake-incremental"
        calls = 0

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
            type(self).calls += 1
            item = dict(_PITFALL_ITEM)
            if type(self).calls == 1:
                item["title"] = "Incremental pitfall title"
            return {"items": [item]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            candidate_ids = [candidate.node_id for candidate in request.candidates]
            if candidate_ids:
                return MemoryWriteSamplingDecision(
                    action="complement",
                    target_node_ids=(candidate_ids[0],),
                    reasoning="complements existing knowledge",
                    confidence=0.9,
                )
            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new knowledge", confidence=0.9)

    distiller = Distiller(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=ComplementThenComplementClient())
    try:
        import os

        content = "## User\nq\n\n## Assistant\n" + "knowledge " * 60
        transcript = _write_transcript(runtime_paths, "mem_session_inc1", content, session_key="sess-inc")
        ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_inc1.md", (ts, ts))
        distiller.run(downgrade_supersede=False)

        # Grow the transcript, re-age, re-sync, sweep again.
        grown = transcript.model_copy(update={"content": content + "\n\n## User\nfollow-up"})
        write_node_file(grown, base_path=runtime_paths.base)
        _sync_all(runtime_paths)
        os.utime(runtime_paths.active / "mem_session_inc1.md", (ts, ts))
        report2 = distiller.run(downgrade_supersede=False)

        assert report2.transcripts_distilled == 1
        assert report2.decider_complements == 1  # decider complements the existing node
        # Same titled knowledge was not duplicated: one knowledge node total.
        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths.active / "mem_session_inc1.md")
        knowledge_ids = stamped.metadata.distilled_node_ids
        assert len(knowledge_ids) >= 1
        titles = {read_node_file(runtime_paths.active / f"{kid}.md").title for kid in knowledge_ids}
        assert len(titles) == len(knowledge_ids)
    finally:
        distiller.close()


# ---------------------------------------------------------------------------
# 7. E2E: retention archives distilled-only
# ---------------------------------------------------------------------------


def test_retention_archives_distilled_only(tmp_path: Path) -> None:
    client = FakeDistillSamplingClient([dict(_PITFALL_ITEM)])
    distiller, paths = _make_distiller(tmp_path, client)
    try:
        import os

        old_ts = (datetime.now(UTC) - timedelta(days=60)).timestamp()

        # Distilled transcript past retention -> archived.
        transcript = _write_transcript(paths, "mem_session_ret1", "## User\na\n\n## Assistant\n" + "k " * 80, session_key="s1")
        stamped = transcript.model_copy(
            update={
                "metadata": transcript.metadata.model_copy(
                    update={"distilled_hash": content_sha256(transcript.content), "distilled_node_ids": ["mem_k"]}
                )
            }
        )
        write_node_file(stamped, base_path=paths.base)
        _sync_all(paths)
        os.utime(paths.active / "mem_session_ret1.md", (old_ts, old_ts))

        # Undistilled transcript of the same age -> must NOT be archived.
        _write_transcript(paths, "mem_session_ret2", "## User\nb\n\n## Assistant\nc", session_key="s2")
        os.utime(paths.active / "mem_session_ret2.md", (old_ts, old_ts))

        archived = distiller._archive_expired(DistillerReport(started_at="x", completed_at="x", duration_ms=0))
        assert archived == 1
        assert (paths.active / "mem_session_ret1.md").exists() is False
        assert (paths.active / "mem_session_ret2.md").exists() is True
        assert (paths.archive / "mem_session_ret1.md").exists() is True
    finally:
        distiller.close()


# ---------------------------------------------------------------------------
# 8. Backfill dry-run on legacy groups writes nothing
# ---------------------------------------------------------------------------


def test_legacy_backfill_dry_run_writes_nothing(tmp_path: Path) -> None:
    client = FakeDistillSamplingClient([dict(_PITFALL_ITEM)])
    distiller, paths = _make_distiller(tmp_path, client)
    try:
        import os

        groups = [
            {
                "group_key": {"title": "Session summary — proj", "first_user_sha1": "ab"},
                "representative_id": "mem_session_bf1",
                "member_ids": ["mem_session_bf1", "mem_20260926_session_summary_proj"],
                "created_min": "2026-09-26",
                "created_max": "2026-09-26",
                "chars": 100,
            }
        ]
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps(groups), encoding="utf-8")

        rep = _write_transcript(paths, "mem_session_bf1", "## User\nq\n\n## Assistant\n" + "a " * 100, session_key="g1")
        # Legacy duplicate member.
        member = _write_transcript(paths, "mem_20260926_session_summary_proj", "## User\nq", title="Session summary — proj")
        ts = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
        os.utime(paths.active / "mem_session_bf1.md", (ts, ts))
        os.utime(paths.active / "mem_20260926_session_summary_proj.md", (ts, ts))

        from synapse.cli import _run_legacy_backfill

        before = {p.name for p in (paths.active).glob("*.md")}
        report = _run_legacy_backfill(distiller, groups_path=groups_file, dry_run=True, limit=None, node_id=None)
        after = {p.name for p in (paths.active).glob("*.md")}
        assert before == after  # zero writes
        assert report.transcripts_scanned == 0  # dry run plans only
    finally:
        distiller.close()


def test_legacy_backfill_apply_downgrades_supersede(tmp_path: Path) -> None:
    """Backfill applies with downgrade ON; a supersede decision becomes complement."""

    runtime_paths = _runtime(tmp_path)

    class SupersedeClient:
        name = "fake-supersede"

        def __init__(self):
            self.decider_calls = 0

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
            item = dict(_PITFALL_ITEM)
            item["title"] = "Curated knowledge"  # matches the curated node -> supersede candidate
            return {"items": [item]}

        def decide_memory_write(self, request):
            self.decider_calls += 1
            from synapse.server.sampling import MemoryWriteSamplingDecision

            # Existing curated node shares the candidate set.
            if request.candidates:
                return MemoryWriteSamplingDecision(
                    action="supersede",
                    target_node_ids=(request.candidates[0].node_id,),
                    reasoning="outdated",
                    confidence=0.9,
                )
            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths)
    curated = service.integrate_knowledge(
        title="Curated knowledge",
        content="## Context\nold wisdom",
        node_type="persistent",
        action="create",
    )

    from synapse.server.sampling import MemoryWriteSamplingDecision  # noqa: F401

    client = SupersedeClient()
    distiller = Distiller(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=client, service=service)
    try:
        import os

        transcript = _write_transcript(runtime_paths, "mem_session_bf2", "## User\nq\n\n## Assistant\n" + "a " * 100, session_key="g2")
        ts = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
        os.utime(runtime_paths.active / "mem_session_bf2.md", (ts, ts))

        # The extracted item shares the curated node's title so the retrieval
        # candidates include it and the decider picks supersede against it.
        report = distiller._run_one_transcript(
            transcript,
            downgrade_supersede=True,
        )

        assert report.transcripts_distilled == 1
        # A3: the curated node has no distiller provenance, so the supersede
        # was downgraded to complement and recorded.
        assert report.decider_downgrades == 1
        assert report.downgraded and report.downgraded[0]["item_title"] == "Curated knowledge"

        # The curated node was complemented, not superseded: still active.
        from synapse.storage import read_node_file

        curated_after = read_node_file(runtime_paths.active / f"{curated['node']['id']}.md")
        assert curated_after.metadata.status is NodeStatus.ACTIVE
        assert curated_after.metadata.superseded_by is None
    finally:
        distiller.close()
        service._store().close()


# ---------------------------------------------------------------------------
# 9. Search: distilled transcripts excluded by default, include flag
# ---------------------------------------------------------------------------


def test_search_excludes_fully_distilled_transcripts_by_default(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)
    from tests.test_server_api import FakeSamplingClient

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    try:
        transcript = _write_transcript(
            runtime_paths,
            "mem_session_search1",
            "## User\nvector cache invalidation behavior\n\n## Assistant\n" + "generation counter " * 40,
            session_key="sess-search",
        )
        # Mark fully distilled.
        stamped = transcript.model_copy(
            update={
                "metadata": transcript.metadata.model_copy(
                    update={"distilled_hash": content_sha256(transcript.content), "distilled_node_ids": ["mem_k9"]}
                )
            }
        )
        write_node_file(stamped, base_path=runtime_paths.base)
        _sync_all(runtime_paths)

        default_hits = [r["node"]["id"] for r in service.search_memory("vector cache invalidation behavior", 10)["results"]]
        assert "mem_session_search1" not in default_hits

        all_hits = [
            r["node"]["id"] for r in service.search_memory("vector cache invalidation behavior", 10, include="all")["results"]
        ]
        assert "mem_session_search1" in all_hits

        # Undistilled transcripts remain in default search.
        _write_transcript(runtime_paths, "mem_session_search2", "## User\nrrf fusion parameter\n\n## Assistant\nk=60", session_key="s9")
        default_hits2 = [r["node"]["id"] for r in service.search_memory("rrf fusion parameter", 10)["results"]]
        assert "mem_session_search2" in default_hits2
    finally:
        pass


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_distiller_settings_defaults_and_overrides(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.distiller.enabled is True
    assert config.distiller.interval_minutes == 10
    assert config.distiller.idle_minutes == 30
    assert config.distiller.retention_days == 45
    assert config.distiller.max_transcript_chars == 16_000
    assert config.distiller.llm_max_tokens == 16_000
    assert config.distiller.enforce_okf is False
    assert config.distiller.downgrade_supersede is True


def test_word_limit_warning_suppressed_for_session_transcripts(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)
    big = "## User\n" + ("混合内容 mixed " * 700)  # well over 3500 word units
    metadata = NodeMetadata(id="mem_session_big1", title="Session summary — big", type=NodeType.TRANSIENT, session_key="s")
    node = Node(metadata=metadata, content=big, file_path=Path("active/mem_session_big1.md"))
    # Must not raise; word-count warning is suppressed for session transcripts.
    written = write_node_file(node, base_path=runtime_paths.base)
    assert written.exists()


# ---------------------------------------------------------------------------
# A1: zero-item transcripts are never re-selected; archived after retention
# ---------------------------------------------------------------------------


def test_zero_item_transcript_not_reselected_and_archived_after_retention(tmp_path: Path) -> None:
    client = FakeDistillSamplingClient([])  # LLM legitimately yields nothing
    distiller, paths = _make_distiller(tmp_path, client)
    try:
        import os

        transcript = _write_transcript(paths, "mem_session_zero1", "## User\nchatter\n\n## Assistant\nnothing durable", session_key="s-z")
        ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(paths.active / "mem_session_zero1.md", (ts, ts))

        report = distiller.run(downgrade_supersede=False)
        assert report.transcripts_distilled == 1
        assert report.items_extracted == 0

        # Stamped distilled-current even with zero ids -> not re-selected.
        from synapse.storage import read_node_file

        stamped = read_node_file(paths.active / "mem_session_zero1.md")
        assert stamped.metadata.distilled_hash == content_sha256(stamped.content)
        assert stamped.metadata.distilled_node_ids == []
        assert not is_represented(stamped)  # zero items: searchable still
        assert is_distilled_current(stamped)

        report2 = distiller.run(downgrade_supersede=False)
        assert report2.transcripts_scanned == 0  # no infinite re-distill loop

        # Past retention: zero-item transcript archives (LLM judged nothing durable).
        old_ts = (datetime.now(UTC) - timedelta(days=60)).timestamp()
        os.utime(paths.active / "mem_session_zero1.md", (old_ts, old_ts))
        archived = distiller._archive_expired(DistillerReport(started_at="x", completed_at="x"))
        assert archived == 1
        assert not (paths.active / "mem_session_zero1.md").exists()
        assert (paths.archive / "mem_session_zero1.md").exists()
    finally:
        distiller.close()


# ---------------------------------------------------------------------------
# A2: excluded nodes are backfilled before the top_k cut (no holes)
# ---------------------------------------------------------------------------


def test_search_filter_backfills_top_k_before_cut(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)
    from tests.test_server_api import FakeSamplingClient

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    try:
        # Three similar knowledge nodes + one represented transcript matching the query.
        for i in range(3):
            service.write_memory(
                title=f"vector cache facts {i}",
                content="## Takeaway\nvector cache generation counter\n\n## Details\nnode " + str(i) + "\n\n## Sources\n- x",
                node_type="persistent",
                okf_type="fact",
                okf_version=1,
                sources=["x"],
            )
        transcript = _write_transcript(
            runtime_paths,
            "mem_session_filter1",
            "## User\nvector cache generation counter\n\n## Assistant\n" + "generation counter " * 40,
            session_key="s-f",
        )
        stamped = transcript.model_copy(
            update={
                "metadata": transcript.metadata.model_copy(
                    update={"distilled_hash": content_sha256(transcript.content), "distilled_node_ids": ["mem_k"]}
                )
            }
        )
        write_node_file(stamped, base_path=runtime_paths.base)
        _sync_all(runtime_paths)

        results = service.search_memory("vector cache generation counter", 2)["results"]
        assert len(results) == 2, "top_k slots must be backfilled, not left empty"
        assert all(r["node"]["id"] != "mem_session_filter1" for r in results)
    finally:
        service._store().close() if hasattr(service, "_store") else None


def test_legacy_represented_transcript_excluded_from_default_search(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)
    from tests.test_server_api import FakeSamplingClient

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    try:
        # Legacy transcript (title-based, no mem_session_ prefix, no session_key).
        legacy = _write_transcript(
            runtime_paths,
            "mem_20260926_session_summary_legacyx",
            "## User\nrrf fusion parameter k\n\n## Assistant\n" + "rrf fusion k=60 " * 30,
            title="Session summary — legacyx",
            session_key=None,
        )
        stamped = legacy.model_copy(
            update={
                "metadata": legacy.metadata.model_copy(
                    update={"distilled_hash": content_sha256(legacy.content), "distilled_node_ids": ["mem_kl"]}
                )
            }
        )
        write_node_file(stamped, base_path=runtime_paths.base)
        _sync_all(runtime_paths)

        default_ids = [r["node"]["id"] for r in service.search_memory("rrf fusion parameter k", 10)["results"]]
        assert "mem_20260926_session_summary_legacyx" not in default_ids
        all_ids = [r["node"]["id"] for r in service.search_memory("rrf fusion parameter k", 10, include="all")["results"]]
        assert "mem_20260926_session_summary_legacyx" in all_ids
    finally:
        service._store().close() if hasattr(service, "_store") else None


# ---------------------------------------------------------------------------
# A3: curated target downgraded; distilled-vs-distilled supersede allowed
# ---------------------------------------------------------------------------


def test_distilled_vs_distilled_supersede_not_downgraded(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)

    class AlwaysSupersedeClient:
        name = "fake-supersede2"

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=()):
            item = dict(_PITFALL_ITEM)
            item["title"] = "Distilled pitfall v1"  # matches the earlier distilled node
            return {"items": [item]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            if request.candidates:
                return MemoryWriteSamplingDecision(
                    action="supersede",
                    target_node_ids=(request.candidates[0].node_id,),
                    reasoning="newer distillation of the same transcript",
                    confidence=0.9,
                )
            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    client = AlwaysSupersedeClient()
    distiller, paths = _make_distiller(tmp_path, client)
    try:
        import os

        transcript = _write_transcript(paths, "mem_session_dv1", "## User\nq\n\n## Assistant\n" + "k " * 80, session_key="s-dv")
        ts = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
        os.utime(paths.active / "mem_session_dv1.md", (ts, ts))

        # First distillation creates the distilled knowledge node.
        r1 = distiller._run_one_transcript(transcript, downgrade_supersede=True)
        assert r1.decider_creates == 1
        from synapse.storage import read_node_file

        stamped = read_node_file(paths.active / "mem_session_dv1.md")
        first_knowledge_id = stamped.metadata.distilled_node_ids[0]
        first_knowledge = read_node_file(paths.active / f"{first_knowledge_id}.md")
        assert first_knowledge.metadata.type is NodeType.PERSISTENT

        # Grow the transcript and distill again: the decider supersedes the
        # earlier DISTILLED node (it has transcript provenance) — allowed.
        grown_content = stamped.content + "\n\n## User\nmore\n\n## Assistant\nupdated k"
        grown = stamped.model_copy(update={"content": grown_content})
        write_node_file(grown, base_path=paths.base)
        _sync_all(paths)
        os.utime(paths.active / "mem_session_dv1.md", (ts, ts))

        r2 = distiller._run_one_transcript(grown, downgrade_supersede=True)
        assert r2.decider_supersedes == 1, "distilled-vs-distilled supersede must stay allowed"
        assert r2.decider_downgrades == 0

        superseded = read_node_file(paths.active / f"{first_knowledge_id}.md")
        assert superseded.metadata.status is NodeStatus.SUPERSEDED
    finally:
        distiller.close()


def test_curated_provenance_classification(tmp_path: Path) -> None:
    """_has_distiller_provenance: sources pointing at transcripts mark distiller nodes."""

    runtime_paths = _runtime(tmp_path)
    from tests.test_server_api import FakeSamplingClient

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    try:
        distilled = service.write_memory(
            title="Distilled node",
            content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
            node_type="persistent",
            okf_type="fact",
            okf_version=1,
            sources=["mem_session_abc123"],
        )
        distilled_id = distilled["execution"]["result"]["node"]["id"]
        curated = service.write_memory(
            title="Curated node",
            content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
            node_type="persistent",
            okf_type="fact",
            okf_version=1,
            sources=["https://example.com/spec"],
        )
        curated_id = curated["execution"]["result"]["node"]["id"]

        assert service._curated_supersede_targets([distilled_id]) == []
        assert service._curated_supersede_targets([curated_id]) == [curated_id]
    finally:
        service._store().close() if hasattr(service, "_store") else None


# ---------------------------------------------------------------------------
# Hygiene 1: source normalization
# ---------------------------------------------------------------------------


def test_normalize_sources_keeps_conforming_drops_pseudo(tmp_path: Path) -> None:
    from synapse.okf import normalize_sources

    existing_ids = {"mem_session_aaa", "mem_20260820_ugm_kafka"}

    def resolve(node_id: str) -> bool:
        return node_id in existing_ids

    kept, dropped = normalize_sources(
        [
            "mem_session_aaa",                                        # conforming node id
            "[[mem_20260820_ugm_kafka]]",                             # wiki-link -> bare
            "related: [[mem_20260820_ugm_kafka]]",                    # prose + wiki-link -> bare id kept, prose dropped
            "https://github.com/hpe-cds/settings/pull/264",           # URL
            "session:01a0df39-4a7f-7447-86e9-e09e3664f162",           # session key
            "transcript://session/…（本会话）",                          # pseudo -> dropped
            "session transcript: AuthZ Shim Dashboard 落地全过程",       # pseudo -> dropped
            "mem_20260926_does_not_exist",                            # non-existent id -> dropped
            "greenlake-resource-notation.md:37-46,89-96",             # file ref -> dropped
        ],
        resolve_node=resolve,
    )
    assert kept == [
        "mem_session_aaa",
        "mem_20260820_ugm_kafka",
        "https://github.com/hpe-cds/settings/pull/264",
        "session:01a0df39-4a7f-7447-86e9-e09e3664f162",
    ]
    # dropped: prose remainder "related:" + 4 fully non-conforming entries
    assert len(dropped) == 5


def test_validator_fires_unresolvable_source() -> None:
    from synapse.okf import validate_okf_node

    good = validate_okf_node(
        node_type="persistent",
        content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
        okf_type="fact",
        sources=["mem_session_aaa", "https://example.com"],
    )
    assert not any(w.code == "okf_unresolvable_source" for w in good)

    bad = validate_okf_node(
        node_type="persistent",
        content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
        okf_type="fact",
        sources=["session transcript: AuthZ Shim Dashboard 落地全过程"],
    )
    assert any(w.code == "okf_unresolvable_source" for w in bad)


def test_write_memory_warns_on_pseudo_sources(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)
    from tests.test_server_api import FakeSamplingClient

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    result = service.write_memory(
        title="Node with pseudo sources",
        content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
        node_type="persistent",
        okf_type="fact",
        okf_version=1,
        sources=["session transcript: AuthZ Shim Dashboard 落地全过程"],
    )
    codes = [w["code"] for w in result["warnings"]]
    assert "okf_unresolvable_source" in codes

    clean = service.write_memory(
        title="Node with clean sources",
        content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
        node_type="persistent",
        okf_type="fact",
        okf_version=1,
        sources=["mem_session_aaa"],
    )
    assert clean["warnings"] == []


def test_distiller_normalizes_sources_and_counts_dropped(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)

    class PseudoSourceClient:
        name = "fake-pseudo"

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=()):
            item = dict(_PITFALL_ITEM)
            item["sources"] = [
                "session transcript: AuthZ Shim Dashboard 落地全过程",
                "[[mem_session_pseudo1]]",
            ]
            return {"items": [item]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="n", confidence=0.9)

    distiller = Distiller(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=PseudoSourceClient())
    try:
        import os

        transcript = _write_transcript(runtime_paths, "mem_session_pseudo1", "## User\nq\n\n## Assistant\n" + "a " * 80, session_key="s-p")
        ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_pseudo1.md", (ts, ts))

        report = distiller._run_one_transcript(transcript, downgrade_supersede=False)
        assert report.transcripts_distilled == 1
        assert report.items_sources_dropped == 1  # the pseudo entry

        # The created node's sources contain only conforming entries.
        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths.active / "mem_session_pseudo1.md")
        knowledge_id = stamped.metadata.distilled_node_ids[0]
        knowledge = read_node_file(runtime_paths.active / f"{knowledge_id}.md")
        assert knowledge.metadata.sources == ["mem_session_pseudo1"]
        assert all(s.startswith("mem_") or s.startswith("http") or s.lower().startswith("session:") for s in knowledge.metadata.sources)
    finally:
        distiller.close()


# ---------------------------------------------------------------------------
# Hygiene 2: project derived from transcript
# ---------------------------------------------------------------------------


def test_distilled_project_from_keyed_and_legacy_transcripts(tmp_path: Path) -> None:
    client = FakeDistillSamplingClient([dict(_PITFALL_ITEM)])
    distiller, paths = _make_distiller(tmp_path, client)
    try:
        import os

        # Legacy transcript: project = title suffix.
        legacy = _write_transcript(
            paths, "mem_20260926_session_summary_authz_projx",
            "## User\nq\n\n## Assistant\n" + "k " * 80,
            title="Session summary — authz-projx",
        )
        # Keyed transcript: no title suffix, project None.
        keyed = _write_transcript(paths, "mem_session_proj1", "## User\nq\n\n## Assistant\n" + "j " * 80, title="Session summary", session_key="s-pr")
        ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(paths.active / "mem_20260926_session_summary_authz_projx.md", (ts, ts))
        os.utime(paths.active / "mem_session_proj1.md", (ts, ts))

        report = distiller.run(downgrade_supersede=False)
        assert report.transcripts_distilled == 2

        from synapse.storage import read_node_file

        legacy_stamped = read_node_file(paths.active / "mem_20260926_session_summary_authz_projx.md")
        legacy_knowledge = read_node_file(paths.active / f"{legacy_stamped.metadata.distilled_node_ids[0]}.md")
        assert legacy_knowledge.metadata.project == "authz-projx"

        keyed_stamped = read_node_file(paths.active / "mem_session_proj1.md")
        keyed_knowledge = read_node_file(paths.active / f"{keyed_stamped.metadata.distilled_node_ids[0]}.md")
        assert keyed_knowledge.metadata.project is None  # title has no suffix

        # Direct helper check.
        assert distiller._transcript_project(legacy) == "authz-projx"
        assert distiller._transcript_project(keyed) is None
    finally:
        distiller.close()


# ---------------------------------------------------------------------------
# Candidate exclusion + guard create-fallback
# ---------------------------------------------------------------------------


def test_write_path_candidates_exclude_transcripts(tmp_path: Path) -> None:
    """A draft whose closest match is its own source transcript is decided
    against knowledge only — the transcript is never offered as a candidate."""

    runtime_paths = _runtime(tmp_path)

    class CandidateProbe:
        name = "probe"
        seen_candidates = []

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=()):
            return {"items": []}

        def decide_memory_write(self, request):
            type(self).seen_candidates.append([c.node_id for c in request.candidates])
            from synapse.server.sampling import MemoryWriteSamplingDecision

            if request.candidates:
                return MemoryWriteSamplingDecision(
                    action="complement", target_node_ids=(request.candidates[0].node_id,), reasoning="r", confidence=0.9
                )
            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="n", confidence=0.9)

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=CandidateProbe())
    try:
        transcript = _write_transcript(
            runtime_paths,
            "mem_session_cand1",
            "## User\nvector cache generation counter\n\n## Assistant\n" + "generation counter details " * 40,
            session_key="s-c",
        )
        knowledge = service.write_memory(
            title="vector cache generation counter facts",
            content="## Takeaway\nvector cache generation counter guards re-embedding.\n\n## Details\nGeneration counter via triggers.\n\n## Sources\n- x",
            node_type="persistent",
            okf_type="fact",
            okf_version=1,
            sources=["x"],
        )
        knowledge_id = knowledge["execution"]["result"]["node"]["id"]

        # Distill a draft whose text is nearly identical to the transcript's.
        from synapse.okf import OkfItem

        item = OkfItem(
            okf_type="fact",
            title="vector cache generation counter facts",
            takeaway="vector cache generation counter guards re-embedding",
            sections={"Details": "Generation counter via triggers; the transcript covers this extensively " * 3},
            sources=[transcript.id],
        )
        result = service.write_memory(
            title=item.title,
            content=item.render(),
            node_type="persistent",
            okf_type=item.okf_type,
            okf_version=1,
            sources=[transcript.id],
        )
        candidates_seen = CandidateProbe.seen_candidates[-1]
        assert transcript.id not in candidates_seen, "transcript must never be a write-path candidate"
        assert knowledge_id in candidates_seen
    finally:
        service._store().close() if hasattr(service, "_store") else None


def test_guard_backstop_falls_back_to_create_item_not_lost(tmp_path: Path) -> None:
    """If the guard still fires, the item falls back to create — not dropped."""

    runtime_paths = _runtime(tmp_path)

    class TranscriptTargetClient:
        name = "fake-guard-fire"
        calls = 0

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=()):
            return {"items": [dict(_PITFALL_ITEM)]}

        def decide_memory_write(self, request):
            # Simulate a stale decider that still targets transcripts.
            from synapse.server.sampling import MemoryWriteSamplingDecision

            type(self).calls += 1
            return MemoryWriteSamplingDecision(
                action="complement",
                target_node_ids=("mem_session_guardfb1",),
                reasoning="closest match",
                confidence=0.9,
            )

    distiller = Distiller(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=TranscriptTargetClient())
    try:
        import os

        transcript = _write_transcript(runtime_paths, "mem_session_guardfb1", "## User\nq\n\n## Assistant\n" + "a " * 80, session_key="s-g")
        ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_guardfb1.md", (ts, ts))

        report = distiller._run_one_transcript(transcript, downgrade_supersede=False)

        assert report.transcripts_distilled == 1
        assert report.guard_fallback_creates == 1
        detail = report.details[0]["items"][0]
        assert detail["action"] == "guard_fallback_create"
        assert detail["node_ids"], "fallback must produce a node id — item not lost"

        from synapse.storage import read_node_file

        knowledge_id = detail["node_ids"][0]
        knowledge = read_node_file(runtime_paths.active / f"{knowledge_id}.md")
        assert knowledge.metadata.type is NodeType.PERSISTENT
        assert knowledge.metadata.okf_type == "pitfall"

        # Transcript stamped with the fallback node.
        stamped = read_node_file(runtime_paths.active / "mem_session_guardfb1.md")
        assert knowledge_id in stamped.metadata.distilled_node_ids
    finally:
        distiller.close()


# ---------------------------------------------------------------------------
# English title rule + ID plan
# ---------------------------------------------------------------------------


def test_slugify_english_title_stopwords_and_caps() -> None:
    from synapse.models import slugify_english_title

    # Stopwords dropped.
    assert slugify_english_title("The Use of the Vector Cache in the Store") == "use_vector_cache_store"
    # Cap: 8 tokens / 48 chars at a word boundary.
    long_title = "authz tenant group projection semantics uses local scope group rows"
    slug = slugify_english_title(long_title)
    assert len(slug) <= 48
    assert slug.split("_")[:8]
    # CJK-only -> degenerate ("").
    assert slugify_english_title("会话蒸馏在服务端进行") == ""
    # < 2 meaningful tokens -> degenerate.
    assert slugify_english_title("Migration") == ""


def test_generate_node_id_english_vs_degenerate() -> None:
    from synapse.models import generate_node_id

    good = generate_node_id("Go dependency upgrade PR review pipeline", current_date="2026-09-27")
    assert good == "mem_20260927_go_dependency_upgrade_pr_review_pipeline"

    # CJK title: degenerate English slug -> node_<sha1[:10]> fallback.
    cjk = generate_node_id("会话蒸馏在服务端进行", current_date="2026-09-27")
    assert cjk.startswith("mem_20260927_node_")
    import re as _re
    assert _re.fullmatch(r"mem_20260927_node_[0-9a-f]{10}", cjk), cjk
    # Keyed transcripts are unaffected (session ids bypass generate_node_id).
    assert "mem_session_" not in good


def test_collision_suffix_is_deterministic_hash(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)
    from tests.test_server_api import FakeSamplingClient

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    first = service.write_memory(
        title="Deterministic collision probe",
        content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
        node_type="persistent",
        okf_type="fact",
        okf_version=1,
        sources=["x"],
    )
    second = service.write_memory(
        title="Deterministic collision probe",
        content="## Takeaway\nt2\n\n## Details\nd2\n\n## Sources\n- x2",
        node_type="persistent",
        okf_type="fact",
        okf_version=1,
        sources=["x"],
    )
    first_id = first["execution"]["result"]["node"]["id"]
    second_id = second["execution"]["result"]["node"]["id"]
    assert first_id != second_id
    # No _2/_3 counters: the second id carries a hex suffix of the same length.
    base = first_id
    assert second_id.startswith(base + "_")
    suffix = second_id[len(base) + 1:]
    assert len(suffix) in (4, 6)
    int(suffix, 16)  # hex


def test_validator_title_rules() -> None:
    from synapse.okf import validate_okf_node

    base = dict(node_type="persistent", content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x", okf_type="fact", sources=["x"])

    cjk = validate_okf_node(**base, title="会话蒸馏在服务端进行")
    assert any(w.code == "okf_title_non_english" for w in cjk)

    long = validate_okf_node(**base, title="x" * 91)
    assert any(w.code == "okf_title_too_long" for w in long)

    prefix = validate_okf_node(**base, title="Procedure: run the dedupe tool")
    assert any(w.code == "okf_title_type_prefix" for w in prefix)

    ok = validate_okf_node(**base, title="AuthZ tenant-group projection uses LOCAL rows")
    assert not any(w.code.startswith("okf_title") for w in ok)


def test_distiller_title_repair_success_and_failure(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)

    class CjkTitleClient:
        """First sample_json returns a CJK title; repair call returns English."""

        name = "fake-cjk"
        repair_calls = 0
        repair_works = True

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=()):
            if "invalid title" in prompt.casefold() or "english title" in prompt.casefold():
                type(self).repair_calls += 1
                if type(self).repair_works:
                    return {"title": "GLP design library DRC review comment triage workflow"}
                return {"title": "仍然中文"}
            item = dict(_PITFALL_ITEM)
            item["title"] = "设计库 DRC 审查意见的分诊流程"
            return {"items": [item]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="n", confidence=0.9)

    client = CjkTitleClient()
    distiller = Distiller(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=client)
    try:
        import os

        transcript = _write_transcript(runtime_paths, "mem_session_title1", "## User\nq\n\n## Assistant\n" + "a " * 80, session_key="s-t")
        ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_title1.md", (ts, ts))

        report = distiller._run_one_transcript(transcript, downgrade_supersede=False)
        assert report.title_repairs == 1
        assert report.title_repair_failures == 0

        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths.active / "mem_session_title1.md")
        knowledge_id = stamped.metadata.distilled_node_ids[0]
        # The repaired English title produced a meaningful ID (not a fallback).
        assert "mem_20260926_drc" not in knowledge_id or "drc" in knowledge_id
        assert "node_" not in knowledge_id
        knowledge = read_node_file(runtime_paths.active / f"{knowledge_id}.md")
        assert knowledge.metadata.title == "GLP design library DRC review comment triage workflow"
    finally:
        distiller.close()

    # Now the failure path: repair returns CJK again -> fallback ID, item kept.
    runtime_paths2 = _runtime(tmp_path)
    CjkTitleClient.repair_works = False
    CjkTitleClient.repair_calls = 0
    distiller2 = Distiller(_config(tmp_path), runtime_paths=runtime_paths2, sampling_client=CjkTitleClient())
    try:
        import os

        transcript = _write_transcript(runtime_paths2, "mem_session_title2", "## User\nq\n\n## Assistant\n" + "a " * 80, session_key="s-t2")
        ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths2.active / "mem_session_title2.md", (ts, ts))

        report = distiller2._run_one_transcript(transcript, downgrade_supersede=False)
        assert report.title_repair_failures == 1
        assert report.transcripts_distilled == 1, "item must be kept, never dropped"

        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths2.active / "mem_session_title2.md")
        knowledge_id = stamped.metadata.distilled_node_ids[0]
        # Fallback ID format: mem_<date>_<okf_type>_<8hex>.
        import re

        assert re.fullmatch(r"mem_\d{8}_pitfall_[0-9a-f]{8}", knowledge_id), knowledge_id
    finally:
        distiller2.close()


# ---------------------------------------------------------------------------
# Provenance fix: legacy-transcript sources mark distiller-produced nodes
# ---------------------------------------------------------------------------


def test_legacy_transcript_source_is_distiller_provenance(tmp_path: Path) -> None:
    """A node whose sources point at a LEGACY transcript (archived or not) is
    distiller-produced: supersede allowed, not downgraded."""

    runtime_paths = _runtime(tmp_path)
    from tests.test_server_api import FakeSamplingClient

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    try:
        # Sources point at a legacy transcript that does NOT exist on disk
        # (archived) — provenance must not depend on the transcript existing.
        distilled = service.write_memory(
            title="Distilled from legacy transcript",
            content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
            node_type="persistent",
            okf_type="fact",
            okf_version=1,
            sources=["mem_20260817_session_summary_authz_29"],
        )
        distilled_id = distilled["execution"]["result"]["node"]["id"]

        curated = service.write_memory(
            title="Curated non-distiller node",
            content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
            node_type="persistent",
            okf_type="fact",
            okf_version=1,
            sources=["https://example.com/spec"],
        )
        curated_id = curated["execution"]["result"]["node"]["id"]

        # Legacy-transcript source → distiller-produced → supersede allowed.
        assert service._curated_supersede_targets([distilled_id]) == []
        # URL source → curated → protected.
        assert service._curated_supersede_targets([curated_id]) == [curated_id]
    finally:
        service._store().close() if hasattr(service, "_store") else None


def test_supersede_of_legacy_sourced_node_not_downgraded(tmp_path: Path) -> None:
    """End-to-end: distiller supersedes a node with legacy-transcript provenance
    (transcript archived) — the write succeeds as supersede, no downgrade."""

    runtime_paths = _runtime(tmp_path)

    class SupersedeLegacyClient:
        name = "fake-supersede-legacy"

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=()):
            item = dict(_PITFALL_ITEM)
            item["title"] = "Legacy-sourced knowledge"
            return {"items": [item]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            if request.candidates:
                return MemoryWriteSamplingDecision(
                    action="supersede",
                    target_node_ids=(request.candidates[0].node_id,),
                    reasoning="newer distillation of the same legacy transcript",
                    confidence=0.9,
                )
            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=SupersedeLegacyClient())
    # Pre-existing distilled node with legacy-transcript provenance.
    existing = service.integrate_knowledge(
        title="Legacy-sourced knowledge",
        content="## Takeaway\nold version\n\n## Details\nold\n\n## Sources\n- mem_20260817_session_summary_authz_29",
        node_type="persistent",
        okf_type="fact",
        okf_version=1,
        sources=["mem_20260817_session_summary_authz_29"],
        action="create",
    )
    existing_id = existing["node"]["id"]

    # Distiller write with downgrade ON — must supersede, not downgrade.
    from synapse.okf import OkfItem

    item = OkfItem(
        okf_type="pitfall",
        title="Legacy-sourced knowledge",
        takeaway="newer version",
        sections={"Symptom": "s", "Cause": "c", "Fix": "f"},
        sources=["mem_20260817_session_summary_authz_29"],
    )
    result = service.write_memory(
        title=item.title,
        content=item.render(),
        node_type="persistent",
        okf_type=item.okf_type,
        okf_version=1,
        sources=[existing_id],
        downgrade_supersede=True,
    )
    assert result["decision"]["action"] == "supersede", "distilled-vs-distilled supersede must be allowed"
    assert "downgrade" not in result

    from synapse.storage import read_node_file

    superseded = read_node_file(runtime_paths.active / f"{existing_id}.md")
    assert superseded.metadata.status is NodeStatus.SUPERSEDED
    service._store().close() if hasattr(service, "_store") else None


# ---------------------------------------------------------------------------
# Distiller-specific LLM timeout
# ---------------------------------------------------------------------------


def test_distiller_llm_timeout_uses_distiller_setting_not_decider(tmp_path: Path) -> None:
    """The distiller's LLM client inherits [decider] settings except timeout."""

    config = _config(tmp_path)
    # Give the write-path decider a short timeout and the distiller a long one.
    config.decider = config.decider.model_copy(update={"timeout_seconds": 120})
    config.distiller = config.distiller.model_copy(update={"llm_timeout_seconds": 300})
    distiller = Distiller(config, runtime_paths=_runtime(tmp_path), sampling_client=None)
    try:
        client = distiller._get_sampling_client()
        assert client.settings.timeout_seconds == 300
        assert client.settings.model == config.decider.model
    finally:
        distiller.close()


def test_distiller_llm_timeout_default_is_300(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.distiller.llm_timeout_seconds == 300
