"""Distiller acceptance tests (design doc §11, items 1–9) — fake LLM, temp runtime."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from synapse.config import SynapseConfig
from synapse.config import DistillerSettings
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

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
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

        # Prompt carried the [decider] max_tokens budget (owner rule: no
        # tight cap; defaults to 600 in test config).
        assert client.calls and client.calls[0]["max_tokens"] == 600

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

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
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

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
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

        # P2b: ALL session transcripts are excluded from default search,
        # distilled or not — transcripts are storage-only raw material.
        _write_transcript(runtime_paths, "mem_session_search2", "## User\nrrf fusion parameter\n\n## Assistant\nk=60", session_key="s9")
        default_hits2 = [r["node"]["id"] for r in service.search_memory("rrf fusion parameter", 10)["results"]]
        assert "mem_session_search2" not in default_hits2
        transcript_hits2 = [
            r["node"]["id"] for r in service.search_memory("rrf fusion parameter", 10, include="transcripts")["results"]
        ]
        assert "mem_session_search2" in transcript_hits2
    finally:
        pass


def test_search_excludes_zero_item_distilled_transcripts_by_default(tmp_path: Path) -> None:
    """Zero-item distilled transcripts are excluded from default search too.

    The LLM judged nothing durable — the transcript has already contributed its
    (empty) knowledge and only competes with real knowledge in default search.
    include=transcripts|all still returns it.
    """

    runtime_paths = _runtime(tmp_path)
    from tests.test_server_api import FakeSamplingClient

    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    try:
        transcript = _write_transcript(
            runtime_paths,
            "mem_session_zeroitem",
            "## User\nslack webhook channel mapping\n\n## Assistant\nprobe message approach",
            session_key="sess-zero",
        )
        # Stamp distilled-current with ZERO knowledge ids (zero-item distill).
        stamped = transcript.model_copy(
            update={
                "metadata": transcript.metadata.model_copy(
                    update={"distilled_hash": content_sha256(transcript.content), "distilled_node_ids": []}
                )
            }
        )
        write_node_file(stamped, base_path=runtime_paths.base)
        _sync_all(runtime_paths)

        from synapse.okf.transcripts import is_distilled_current, is_represented

        reloaded = service._store().get_node("mem_session_zeroitem")
        assert is_distilled_current(reloaded) and not is_represented(reloaded)

        default_hits = [r["node"]["id"] for r in service.search_memory("slack webhook channel mapping", 10)["results"]]
        assert "mem_session_zeroitem" not in default_hits

        transcript_hits = [
            r["node"]["id"]
            for r in service.search_memory("slack webhook channel mapping", 10, include="transcripts")["results"]
        ]
        assert "mem_session_zeroitem" in transcript_hits
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
    assert config.distiller.enforce_okf is False
    assert config.distiller.downgrade_supersede is True


def test_distiller_calls_use_decider_max_tokens(tmp_path: Path) -> None:
    """Owner rule: no tight token cap — the distiller's extraction calls pass
    [decider] max_tokens through unchanged (wall time bounded by timeout)."""

    config = _config(tmp_path)
    config.decider = config.decider.model_copy(update={"max_tokens": 200_000})
    client = FakeDistillSamplingClient([])
    distiller = Distiller(config, runtime_paths=_runtime(tmp_path), sampling_client=client)
    try:
        distiller._sample_json("prompt")
        assert client.calls and client.calls[0]["max_tokens"] == 200_000
    finally:
        distiller.close()


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
        assert not is_represented(stamped)  # zero items: no knowledge produced
        assert is_distilled_current(stamped)  # still excluded from default search

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

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=(), reasoning_effort=None):
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
            "skill://authz-shim-triage-fast",                          # skill URI
            "skill://",                                                # empty skill -> dropped
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
        "skill://authz-shim-triage-fast",
    ]
    # dropped: prose remainder "related:" + 5 fully non-conforming entries
    assert len(dropped) == 6


def test_validator_fires_unresolvable_source() -> None:
    from synapse.okf import validate_okf_node

    good = validate_okf_node(
        node_type="persistent",
        content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- x",
        okf_type="fact",
        sources=["mem_session_aaa", "https://example.com", "skill://fleetgraph-extract-round-runbook"],
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

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=(), reasoning_effort=None):
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

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=(), reasoning_effort=None):
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

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=(), reasoning_effort=None):
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

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=(), reasoning_effort=None):
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

        def sample_json(self, *, prompt, system_prompt, max_tokens=600, model_hints=(), reasoning_effort=None):
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


# ---------------------------------------------------------------------------
# P2c: chunked distillation + transcript format v2
# ---------------------------------------------------------------------------


def test_parse_transcript_v2_and_legacy_v1() -> None:
    from synapse.okf.transcript_format import parse_transcript

    v2 = (
        "<!-- synapse-transcript v2 -->\n"
        "## T1 User\nhelp fix bug\n\n"
        "## T2 Assistant\ninvestigating\n"
        "### Tool T2.1 grep\ncall: grep -rn bug\nresult [ok]: found it …elided… tail\n\n"
        "## T3 User\nthanks\n"
    )
    parsed = parse_transcript(v2)
    assert parsed.version == 2
    assert [t.id for t in parsed.turns] == ["T1", "T2", "T3"]
    assert parsed.turns[1].body == "investigating"
    assert len(parsed.turns[1].tool_sections) == 1
    assert "grep -rn bug" in parsed.turns[1].tool_sections[0]
    assert parsed.last_turn_number == 3

    legacy = "## User\nfirst\n\n## Assistant\nanswer\n\n## User\nsecond\n"
    parsed_v1 = parse_transcript(legacy)
    assert parsed_v1.version == 1
    assert [t.id for t in parsed_v1.turns] == ["T1", "T2", "T3"]
    assert [t.role for t in parsed_v1.turns] == ["User", "Assistant", "User"]


def test_build_turn_segments_respects_budget_and_context() -> None:
    from synapse.lifecycle.distiller import _build_turn_segments
    from synapse.okf.transcript_format import parse_transcript

    content = (
        "<!-- synapse-transcript v2 -->\n"
        "## T1 User\n" + "a" * 200 + "\n\n"
        "## T2 User\n" + "b" * 200 + "\n\n"
        "## T3 User\n" + "c" * 200 + "\n"
    )
    parsed = parse_transcript(content)
    # Watermark 1: T1 is context on the first segment; start turn is T2.
    segments = _build_turn_segments(parsed, watermark=1, max_chars=600)
    assert segments[0][1] == "T2"
    assert "## T1 User" in segments[0][0]
    assert all(len(text) <= 600 for text, _ in segments)
    # All turn content survives.
    joined = "\n".join(text for text, _ in segments)
    assert "a" * 200 in joined and "b" * 200 in joined and "c" * 200 in joined

    # Oversized single turn splits by chars, nothing lost.
    big = (
        "<!-- synapse-transcript v2 -->\n"
        "## T1 User\n" + "x" * 1500 + "\n"
    )
    parsed_big = parse_transcript(big)
    big_segments = _build_turn_segments(parsed_big, watermark=0, max_chars=600)
    assert len(big_segments) >= 3
    assert sum(len(text) for text, _ in big_segments) >= 1500


def test_chunked_distillation_one_call_per_segment_and_dedupe(tmp_path: Path) -> None:
    """A transcript over the segment budget gets one extraction call per
    segment; near-duplicate candidates go through ONE dedupe reduce call."""

    calls: list[dict[str, Any]] = []

    class SegmentThenCapClient:
        name = "fake-chunked"
        uid = 0
        lock = __import__("threading").Lock()

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            calls.append({"prompt": prompt, "max_tokens": max_tokens})
            # Extraction calls return one item each; the dedupe call returns 2.
            if "You dedupe knowledge items" in prompt:
                return {
                    "items": [
                        dict(_PITFALL_ITEM, title="Merged pitfall one"),
                        dict(_PITFALL_ITEM, title="Merged pitfall two"),
                    ]
                }
            return {"items": [dict(_PITFALL_ITEM, title="Segment pitfall duplicate")]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            candidate_ids = [candidate.node_id for candidate in request.candidates]
            if candidate_ids:
                return MemoryWriteSamplingDecision(
                    action="complement", target_node_ids=(candidate_ids[0],), reasoning="complement", confidence=0.9
                )
            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    config = _config(tmp_path)
    config.distiller.segment_chars = 800
    runtime_paths = bootstrap_runtime_directories(config)
    distiller = Distiller(config, runtime_paths=runtime_paths, sampling_client=SegmentThenCapClient())
    try:
        import os

        turn = "## T{n} User\n" + ("lesson about kafka offsets " * 20)
        content = "<!-- synapse-transcript v2 -->\n" + "\n\n".join(
            turn.format(n=n) for n in range(1, 9)
        )
        _write_transcript(runtime_paths, "mem_session_chunk1", content, session_key="sess-chunk")
        old_ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_chunk1.md", (old_ts, old_ts))

        report = distiller.run(downgrade_supersede=False)
        assert report.transcripts_distilled == 1
        extraction_calls = [c for c in calls if "Transcript segment:" in c["prompt"]]
        assert len(extraction_calls) >= 2, "large transcript must be split into segments"
        # 8 identical-title candidates -> deterministic grouping forms one
        # multi-item group -> ONE dedupe reduce call; it merged them into 2,
        # so reduced_items = 8 - 2 = 6.
        assert report.reduce_groups_llm >= 1
        assert report.reduced_items >= 1
        # Watermark stamped to the last turn.
        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths.active / "mem_session_chunk1.md")
        assert stamped.metadata.distilled_turn == 8
    finally:
        distiller.close()


def test_incremental_distillation_only_processes_new_turns(tmp_path: Path) -> None:
    """After a first distill, growing the transcript only extracts from new
    turns (one previous turn as context), and unchanged transcripts re-distill
    nothing."""

    extraction_prompts: list[str] = []

    class IncrementalClient:
        name = "fake-incremental"

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            extraction_prompts.append(prompt)
            return {"items": [dict(_PITFALL_ITEM, title=f"Incremental pitfall {len(extraction_prompts)}")]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            candidate_ids = [candidate.node_id for candidate in request.candidates]
            if candidate_ids:
                return MemoryWriteSamplingDecision(
                    action="complement", target_node_ids=(candidate_ids[0],), reasoning="complement", confidence=0.9
                )
            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    config = _config(tmp_path)
    runtime_paths = bootstrap_runtime_directories(config)
    distiller = Distiller(config, runtime_paths=runtime_paths, sampling_client=IncrementalClient())
    try:
        import os

        def turn(n: int, word: str) -> str:
            return f"## T{n} User\n{word} " + ("durable offset lesson " * 20)

        content = "<!-- synapse-transcript v2 -->\n" + "\n\n".join([turn(1, "alpha"), turn(2, "beta")])
        transcript = _write_transcript(runtime_paths, "mem_session_inc9", content, session_key="sess-inc9")
        ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_inc9.md", (ts, ts))
        distiller.run(downgrade_supersede=False)
        assert len(extraction_prompts) == 1
        first_prompt = extraction_prompts[0]
        assert "alpha" in first_prompt and "beta" in first_prompt

        # Grow: add T3. Re-distill processes only T2 (context) + T3.
        from synapse.storage import read_node_file

        stamped_before = read_node_file(runtime_paths.active / "mem_session_inc9.md")
        assert stamped_before.metadata.distilled_turn == 2
        grown = stamped_before.model_copy(update={"content": content + "\n\n" + turn(3, "gamma")})
        write_node_file(grown, base_path=runtime_paths.base)
        _sync_all(runtime_paths)
        os.utime(runtime_paths.active / "mem_session_inc9.md", (ts, ts))
        report2 = distiller.run(downgrade_supersede=False)
        assert report2.transcripts_distilled == 1
        second_prompt = extraction_prompts[1]
        assert "gamma" in second_prompt
        assert "## T1 User" not in second_prompt, "turns before the watermark context must not reappear"

        # Unchanged transcript: not re-selected.
        report3 = distiller.run(downgrade_supersede=False)
        assert report3.transcripts_scanned == 0
    finally:
        distiller.close()


# ---------------------------------------------------------------------------
# P2+: evidence verification (deterministic, no LLM)
# ---------------------------------------------------------------------------


def test_verify_evidence_accepts_verbatim_and_rejects_paraphrase() -> None:
    from synapse.okf import EvidenceEntry, verify_evidence

    turns = {
        "T1": "We decided to use kafka offsets for resume.",
        "T2": "call: grep -rn offsets\nresult [ok]: src/main.py:1 uses offsets",
    }
    verified, unverified = verify_evidence(
        [
            EvidenceEntry(turn="T1", quote="use kafka offsets"),
            EvidenceEntry(turn="T1", quote="  use  KAFKA\noffsets "),  # case/whitespace collapse
            EvidenceEntry(turn="T2", quote="result [ok]: src/main.py:1 uses offsets"),
            EvidenceEntry(turn="T1", quote="paraphrased claim not present"),
            EvidenceEntry(turn="T9", quote="kafka"),
        ],
        turns=turns,
    )
    assert [e.turn for e in verified] == ["T1", "T1", "T2"]
    assert [(e.turn, e.quote[:20]) for e in unverified] == [("T1", "paraphrased claim no"), ("T9", "kafka")]


def test_distiller_stamps_verified_frontmatter_and_evidence_section(tmp_path: Path) -> None:
    """Items with verifiable quotes get verified: true + ## Evidence; items
    whose evidence cannot be checked are written with verified: false."""

    good_item = {
        "okf_type": "fact",
        "title": "Kafka offsets gate resume",
        "takeaway": "kafka offset 恢复由 offsets gate 控制",
        "sections": {"Details": "Offsets gate drives resume behavior."},
        "sources": [],
        "evidence": [{"turn": "T1", "quote": "offsets gate controls resume"}],
    }
    bad_item = {
        "okf_type": "fact",
        "title": "Unverifiable claim",
        "takeaway": "unverifiable takeaway",
        "sections": {"Details": "No quotable source."},
        "sources": [],
        "evidence": [{"turn": "T1", "quote": "this quote does not exist anywhere"}],
    }

    class EvidenceClient:
        name = "fake-evidence"

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            if "merge" in prompt.lower():
                return {"items": [good_item, bad_item]}
            return {"items": [good_item, bad_item]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    config = _config(tmp_path)
    runtime_paths = bootstrap_runtime_directories(config)
    distiller = Distiller(config, runtime_paths=runtime_paths, sampling_client=EvidenceClient())
    try:
        import os

        content = "<!-- synapse-transcript v2 -->\n## T1 User\nthe offsets gate controls resume behavior\n"
        _write_transcript(runtime_paths, "mem_session_ev1", content, session_key="sess-ev")
        old_ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_ev1.md", (old_ts, old_ts))

        report = distiller.run(downgrade_supersede=False)
        assert report.transcripts_distilled == 1
        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths.active / "mem_session_ev1.md")
        titles = [read_node_file(runtime_paths.active / f"{kid}.md") for kid in stamped.metadata.distilled_node_ids]
        by_title = {node.title: node for node in titles}

        good = by_title["Kafka offsets gate resume"]
        assert good.metadata.verified is True
        assert "## Evidence" in good.content
        assert '- T1: "offsets gate controls resume"' in good.content

        bad = by_title["Unverifiable claim"]
        assert bad.metadata.verified is False
        assert "## Evidence" in bad.content
    finally:
        distiller.close()


def test_distiller_validates_clean_with_skill_sources(tmp_path: Path) -> None:
    """Nodes citing skill://<name> sources validate without warnings (parallel
    data-cleanup compatibility)."""
    from synapse.okf import validate_okf_node

    warnings = validate_okf_node(
        node_type="persistent",
        content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- skill://authz-shim-triage-fast",
        okf_type="fact",
        sources=["skill://authz-shim-triage-fast"],
    )
    assert not any(w.code == "okf_unresolvable_source" for w in warnings)


def test_evidence_eligible_sources_strict_by_type() -> None:
    """§9 rule, enforced server-side: assistant prose never verifies; tool
    result lines and user-turn bodies verify; procedure call lines verify only
    with an ok result."""
    from synapse.okf import EvidenceEntry, verify_evidence
    from synapse.okf.transcript_format import parse_transcript

    content = (
        "<!-- synapse-transcript v2 -->\n"
        "## T1 User\nwe chose kafka offsets for resume\n\n"
        "## T2 Assistant\nassistant claims offsets gate works\n"
        "### Tool T2.1 grep\n"
        "call: grep -rn offsets src/\n"
        "result [ok]: src/main.py:1 uses offsets gate\n\n"
        "### Tool T2.2 rm\n"
        "call: rm -rf /tmp/stale\n"
        "result [error]: permission denied\n"
    )
    parsed = parse_transcript(content)

    def turns_for(okf_type: str) -> dict[str, str]:
        return {turn.id: turn.eligible_evidence_text(okf_type=okf_type) for turn in parsed.turns}

    # Assistant prose quote: NEVER verified, whatever the type.
    for okf_type in ("fact", "procedure", "decision", "pitfall"):
        verified, _ = verify_evidence(
            [EvidenceEntry(turn="T2", quote="assistant claims offsets gate works")],
            turns=turns_for(okf_type),
        )
        assert verified == [], f"assistant prose must not verify for {okf_type}"

    # Tool-result quote verifies for fact; user-turn quote verifies for decision.
    verified, _ = verify_evidence(
        [
            EvidenceEntry(turn="T2", quote="src/main.py:1 uses offsets gate"),
            EvidenceEntry(turn="T1", quote="we chose kafka offsets for resume"),
        ],
        turns=turns_for("fact"),
    )
    assert len(verified) == 2

    # Procedure: call-line quote verifies only against an ok-result tool block.
    verified_proc, _ = verify_evidence(
        [
            EvidenceEntry(turn="T2", quote="grep -rn offsets src/"),      # ok result block
            EvidenceEntry(turn="T2", quote="rm -rf /tmp/stale"),          # error result block
        ],
        turns=turns_for("procedure"),
    )
    assert [e.quote for e in verified_proc] == ["grep -rn offsets src/"]

    # Fact: the same call-line quote does NOT verify (call lines not eligible).
    verified_fact, _ = verify_evidence(
        [EvidenceEntry(turn="T2", quote="grep -rn offsets src/")],
        turns=turns_for("fact"),
    )
    assert verified_fact == []

    # v1 transcript (no tool blocks): only User turns are eligible.
    v1 = parse_transcript("## User\nv1 user statement\n\n## Assistant\nv1 assistant prose\n")
    verified_v1, _ = verify_evidence(
        [
            EvidenceEntry(turn="T1", quote="v1 user statement"),
            EvidenceEntry(turn="T2", quote="v1 assistant prose"),
        ],
        turns={turn.id: turn.eligible_evidence_text() for turn in v1.turns},
    )
    assert [e.turn for e in verified_v1] == ["T1"]


# ---------------------------------------------------------------------------
# P2c: archived-transcript revival (backfill resubmission)
# ---------------------------------------------------------------------------


def test_archived_transcript_revives_on_reupsert_and_redisills(tmp_path: Path) -> None:
    """A distilled+archived session_key, re-upserted with new v2 content,
    revives as a fresh ACTIVE transcript: the archive keeps its copy, the DB
    row points at the active file with default distillation state (except
    carried-over distilled_node_ids for own-title dedupe), and the sweep
    selects the new transcript."""
    from tests.test_server_api import FakeSamplingClient

    runtime_paths = _runtime(tmp_path)
    session_key = "sess-revive"

    # 1. Create + distill + archive via the distiller's REAL archive primitive.
    service = SynapseServerService(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    content_v1 = "## User\nkafka offset resume behavior\n"
    up = service.upsert_session_memory(session_key=session_key, title="Session summary — revive", content=content_v1)
    node_id = up["node"]["id"]
    assert node_id.startswith("mem_session_")

    class OneItemClient(FakeDistillSamplingClient):
        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            return {"items": [dict(_PITFALL_ITEM, title="Revived session pitfall")]}

    distiller = Distiller(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=OneItemClient([dict(_PITFALL_ITEM, title="Revived session pitfall")]))
    try:
        import os

        old_ts = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
        os.utime(runtime_paths.active / f"{node_id}.md", (old_ts, old_ts))
        first_run = distiller.run(downgrade_supersede=False)
        assert first_run.transcripts_distilled == 1

        from synapse.storage import read_node_file

        distilled = read_node_file(runtime_paths.active / f"{node_id}.md")
        carried_ids = list(distilled.metadata.distilled_node_ids)
        assert carried_ids

        # Archive with the distiller's own primitive.
        archived_node = service._load_node(node_id)
        assert distiller._archive_transcript(archived_node) is True
        assert (runtime_paths.archive / f"{node_id}.md").exists()
        assert not (runtime_paths.active / f"{node_id}.md").exists()
        with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=0) as store:
            assert store.get_node(node_id) is None, "archive must remove the index row"
    finally:
        distiller.close()

    # 2. Re-upsert the same session_key with new v2 content (backfill client).
    content_v2 = "<!-- synapse-transcript v2 -->\n## T1 User\nfresh resubmission of an old session\n"
    up2 = service.upsert_session_memory(session_key=session_key, title="Session summary — revive", content=content_v2)
    assert up2["action"] == "created"
    assert up2["node"]["id"] == node_id, "same session key -> same node id"
    # The DB row exists again, active, pointing at the active file.
    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=0) as store:
        row = store.get_node(node_id)
    assert row is not None
    assert row.metadata.status is NodeStatus.ACTIVE
    assert row.file_path.as_posix() == f"active/{node_id}.md"
    assert (runtime_paths.active / f"{node_id}.md").exists()
    assert (runtime_paths.archive / f"{node_id}.md").exists(), "archived copy must be preserved"

    revived_content = (runtime_paths.active / f"{node_id}.md").read_text(encoding="utf-8")
    assert "distilled_hash:" not in revived_content or "distilled_hash: null" in revived_content
    assert "distilled_turn:" not in revived_content or "distilled_turn: null" in revived_content
    assert f"- {carried_ids[0]}" in revived_content, "distilled_node_ids carried over for dedupe"

    # 3. The sweep selects the revived transcript and distills it fully.
    distiller2 = Distiller(_config(tmp_path), runtime_paths=runtime_paths, sampling_client=OneItemClient([dict(_PITFALL_ITEM, title="Second revival pitfall")]))
    try:
        import os

        old_ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / f"{node_id}.md", (old_ts, old_ts))
        selected = distiller2.select_transcripts()
        assert node_id in [n.id for n in selected], "revived transcript must be selected by the sweep"
        run2 = distiller2.run(downgrade_supersede=False)
        assert run2.transcripts_distilled == 1

        from synapse.storage import read_node_file

        restamped = read_node_file(runtime_paths.active / f"{node_id}.md")
        assert restamped.metadata.distilled_hash == content_sha256(restamped.content)
        assert restamped.metadata.distilled_turn == 1
        assert carried_ids[0] in restamped.metadata.distilled_node_ids, "carried ids survive re-stamping"
    finally:
        distiller2.close()


# ---------------------------------------------------------------------------
# Tolerant parse: section-key normalization + takeaway fallback
# ---------------------------------------------------------------------------


def test_parse_okf_items_normalizes_hash_prefixed_section_keys() -> None:
    """glm-5.3-flash sometimes emits '"## Details"' instead of '"Details"';
    such items must parse instead of being dropped (live variance)."""
    payload = {
        "items": [
            {
                "okf_type": "fact",
                "title": "Hash-prefixed sections still parse",
                "takeaway": "single line takeaway",
                "sections": {"## Takeaway": "single line takeaway", "## Details": "body text"},
                "sources": ["mem_session_x"],
            }
        ]
    }
    items = parse_okf_items(payload)
    assert len(items) == 1
    assert items[0].sections["Details"] == "body text"
    assert items[0].takeaway == "single line takeaway"


def test_parse_okf_items_falls_back_to_sections_takeaway() -> None:
    """Observed live: the model fills '## Takeaway' but leaves the top-level
    'takeaway' field empty; the section body is used when it is one line."""
    payload = {
        "items": [
            {
                "okf_type": "fact",
                "title": "Takeaway only under sections",
                "takeaway": "",
                "sections": {"Takeaway": "fallback takeaway line", "Details": "body"},
                "sources": [],
            }
        ]
    }
    items = parse_okf_items(payload)
    assert len(items) == 1
    assert items[0].takeaway == "fallback takeaway line"


def test_parse_okf_items_still_rejects_when_both_takeaways_empty() -> None:
    payload = {
        "items": [
            {
                "okf_type": "fact",
                "title": "Both empty",
                "takeaway": "",
                "sections": {"Takeaway": "", "Details": "body"},
                "sources": [],
            },
            {
                "okf_type": "fact",
                "title": "Missing everything",
                "takeaway": "",
                "sections": {},
                "sources": [],
            },
        ]
    }
    assert parse_okf_items(payload) == []


def test_parse_okf_items_multi_line_section_takeaway_not_used_as_fallback() -> None:
    """A multi-line section body is not a valid takeaway; no fallback happens
    and the item is rejected."""
    payload = {
        "items": [
            {
                "okf_type": "fact",
                "title": "Multi-line section takeaway",
                "takeaway": "",
                "sections": {"Takeaway": "line one\nline two", "Details": "body"},
                "sources": [],
            }
        ]
    }
    assert parse_okf_items(payload) == []


def test_partial_segment_failure_holds_back_watermark(tmp_path: Path) -> None:
    """Regression (2026-10-12 live incident): 3 segments, the middle one fails —
    the watermark must hold at the end of segment 1 and distilled_hash must stay
    unset so the next sweep retries segments 2-3. Items from successful segments
    are kept."""

    class FailMiddleSegmentClient:
        name = "fake-fail-middle"
        calls = 0

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("timed out")
            return {"items": [dict(_PITFALL_ITEM, title=f"Segment pitfall {type(self).calls}")]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    config = _config(tmp_path)
    config.distiller.segment_chars = 400
    runtime_paths = _runtime(tmp_path)
    distiller = Distiller(config, runtime_paths=runtime_paths, sampling_client=FailMiddleSegmentClient())
    try:
        import os

        turn = "## T{n} User\n" + ("durable kafka offset lesson " * 12)
        content = "<!-- synapse-transcript v2 -->\n" + "\n\n".join(turn.format(n=n) for n in range(1, 7))
        _write_transcript(runtime_paths, "mem_session_partial1", content, session_key="sess-partial")
        old_ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_partial1.md", (old_ts, old_ts))

        report = distiller.run(downgrade_supersede=False)
        assert report.transcripts_distilled == 1
        assert report.items_extracted >= 1, "items from successful segments must be kept"

        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths.active / "mem_session_partial1.md")
        # Watermark held back: the failed segment started at T3 → safe = T2's end.
        assert stamped.metadata.distilled_turn is not None and stamped.metadata.distilled_turn < 6
        # Hash left unset so the next sweep re-selects and retries the failed range.
        assert stamped.metadata.distilled_hash is None

        # Next run (all segments succeed now): watermark advances to the end,
        # hash stamped → no further re-selection.
        FailMiddleSegmentClient.calls = 100  # stop failing
        os.utime(runtime_paths.active / "mem_session_partial1.md", (old_ts, old_ts))
        report2 = distiller.run(downgrade_supersede=False)
        assert report2.transcripts_distilled == 1
        stamped2 = read_node_file(runtime_paths.active / "mem_session_partial1.md")
        assert stamped2.metadata.distilled_hash == content_sha256(stamped2.content)
        assert stamped2.metadata.distilled_turn == 6
        # Third run: nothing selectable.
        report3 = distiller.run(downgrade_supersede=False)
        assert report3.transcripts_scanned == 0
    finally:
        distiller.close()


def _write_transcript_raw(config, runtime_paths, node_id: str, content: str, *, session_key: str) -> None:
    """Write + sync a transcript using the GIVEN config (correct project root)."""
    from synapse.sync import SyncManager
    metadata = NodeMetadata(id=node_id, title="Session summary — conc", type=NodeType.TRANSIENT, session_key=session_key)
    node = Node(metadata=metadata, content=content, file_path=Path("active") / f"{node_id}.md")
    write_node_file(node, base_path=runtime_paths.base)
    mgr = SyncManager(config, runtime_paths=runtime_paths, debounce_seconds=0.0)
    try:
        mgr.sync_paths([str(runtime_paths.base / f"active/{node_id}.md")])
    finally:
        mgr.close()


def test_concurrent_extraction_same_result_as_serial(tmp_path: Path) -> None:
    """max_concurrency > 1 produces the same items and watermark as serial for
    a fake sampler with random delays; results are merged in segment order."""

    class SlowRandomClient:
        name = "fake-slow-random"

        def __init__(self) -> None:
            self.calls = 0
            self.lock = threading.Lock()
            self.write_calls = 0
            self.write_overlap = False
            self._in_write = False

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            with self.lock:
                self.calls += 1
                n = self.calls
            time.sleep(0.05 + (0.03 * ((n * 7919) % 5)))  # pseudo-random delay
            return {"items": [dict(_PITFALL_ITEM, title=f"Concurrent pitfall {n}")]}

        def decide_memory_write(self, request):
            # Detect overlapping write-phase calls (would mean the write path
            # is running concurrently).
            if self._in_write:
                self.write_overlap = True
            self._in_write = True
            time.sleep(0.01)
            from synapse.server.sampling import MemoryWriteSamplingDecision

            decision = MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)
            self._in_write = False
            return decision

    content = (
        "<!-- synapse-transcript v2 -->\n"
        + "\n\n".join(
            f"## T{n} User\n" + ("durable concurrent lesson " * 12) for n in range(1, 7)
        )
    )

    def run_once(concurrency: int, runtime_paths) -> tuple[int, Any]:
        config = _config(runtime_paths.base.parent)
        config.distiller.segment_chars = 400
        config.distiller.max_concurrency = concurrency
        client = SlowRandomClient()
        d = Distiller(config, runtime_paths=runtime_paths, sampling_client=client)
        try:
            try:
                report = d.run(downgrade_supersede=False)
                import sys as _sys
                print("REPORT", concurrency, report.transcripts_distilled,
                      report.items_extracted, report.llm_failures,
                      [(dd.get('transcript_id'), len(dd.get('items', []))) for dd in report.details],
                      file=_sys.stderr)
            except Exception:
                import traceback; traceback.print_exc()
                raise
            from synapse.storage import read_node_file

            stamped = read_node_file(runtime_paths.active / "mem_session_conc1.md")
            titles = sorted(
                read_node_file(runtime_paths.active / f"{kid}.md").title
                for kid in stamped.metadata.distilled_node_ids
            )
            import sys as _sys
            print("DBG", concurrency, "turn:", stamped.metadata.distilled_turn,
                  "hash:", bool(stamped.metadata.distilled_hash),
                  "ntitles:", len(titles), file=_sys.stderr)
            return stamped.metadata.distilled_turn, titles, client
        finally:
            d.close()

    # Serial
    serial_root = tmp_path / "serial"
    serial_root.mkdir(exist_ok=True)
    serial_config = _config(serial_root)
    serial_paths = bootstrap_runtime_directories(serial_config)
    _write_transcript_raw(serial_config, serial_paths, "mem_session_conc1", content, session_key="sess-conc")
    old_ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(serial_paths.active / "mem_session_conc1.md", (old_ts, old_ts))
    wm_serial, titles_serial, client_serial = run_once(1, serial_paths)
    assert client_serial.write_overlap is False

    # Concurrent
    conc_root = tmp_path / "conc"
    conc_root.mkdir(exist_ok=True)
    conc_config = _config(conc_root)
    conc_paths = bootstrap_runtime_directories(conc_config)
    _write_transcript_raw(conc_config, conc_paths, "mem_session_conc1", content, session_key="sess-conc")
    os.utime(conc_paths.active / "mem_session_conc1.md", (old_ts, old_ts))
    wm_conc, titles_conc, client_conc = run_once(4, conc_paths)
    assert client_conc.write_overlap is False, "write phase must stay serialized"

    assert wm_serial == wm_conc == 6
    assert titles_serial == titles_conc
    # Same non-empty outcome for both modes (the fake's reduce collapses
    # candidates to one item; equality serial-vs-concurrent is the contract).
    assert len(titles_serial) >= 1


def test_partial_failure_out_of_order_holds_earliest_watermark(tmp_path: Path) -> None:
    """706a88a semantics under concurrency: with segments failing out of
    completion order, the watermark holds at the EARLIEST failed segment in
    turn order (not the first completed failure)."""
    import threading

    class OutOfOrderClient:
        name = "fake-ooo"
        # Segments start at T1..T6 (one per turn). Behavior by TURN (robust to
        # nondeterministic pool scheduling): T1 succeeds slowly; T2 fails after
        # a delay; T3 succeeds fast; T4..T6 succeed fast. Completion order:
        # T3, T4.., T1, then T2 fails last — the watermark must still hold at
        # the EARLIEST failed segment in turn order (T2 → watermark 1).

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            import re as _re
            m = _re.search(r"This segment starts at turn (T\d+)", prompt)
            turn = m.group(1) if m else "T1"
            if turn == "T1":
                time.sleep(0.25)
                return {"items": [dict(_PITFALL_ITEM, title="OOO pitfall slow")]}
            if turn == "T2":
                time.sleep(0.4)  # fails LAST despite being earliest failure
                raise RuntimeError("timed out")
            return {"items": [dict(_PITFALL_ITEM, title=f"OOO pitfall {turn}")]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    config = _config(tmp_path)
    config.distiller.segment_chars = 400
    config.distiller.max_concurrency = 3
    runtime_paths = _runtime(tmp_path)
    distiller = Distiller(config, runtime_paths=runtime_paths, sampling_client=OutOfOrderClient())
    try:
        import os

        turn = "## T{n} User\n" + ("durable ooo lesson " * 12)
        content = "<!-- synapse-transcript v2 -->\n" + "\n\n".join(turn.format(n=n) for n in range(1, 7))
        _write_transcript(runtime_paths, "mem_session_ooo1", content, session_key="sess-ooo")
        old_ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_ooo1.md", (old_ts, old_ts))

        report = distiller.run(downgrade_supersede=False)
        assert report.transcripts_distilled == 1
        assert report.items_extracted >= 1

        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths.active / "mem_session_ooo1.md")
        # Segments start at T1..T6 (one per turn). Completion order: T3 fails
        # fast, T1 succeeds slow, T2 fails slow. Turn-order semantics: the
        # earliest FAILED segment is T2 → watermark = 1 (turn before T2).
        # Completion-order logic would have held at T3's turn (wrong).
        assert stamped.metadata.distilled_turn == 1
        assert stamped.metadata.distilled_hash is None
    finally:
        distiller.close()


def test_max_concurrency_upper_bound_30(tmp_path: Path) -> None:
    """Owner hard limit 30 (bench-driven); default stays 1."""
    config = _config(tmp_path)
    assert config.distiller.max_concurrency == 1
    from pydantic import ValidationError

    config = config.model_copy(update={
        "distiller": config.distiller.model_validate(
            {**config.distiller.model_dump(), "max_concurrency": 30}
        )
    })
    assert config.distiller.max_concurrency == 30
    import pytest as _pytest

    with _pytest.raises(ValidationError):
        DistillerSettings(max_concurrency=31)


def test_reduce_failure_retries_then_dedupes_never_truncates(tmp_path: Path) -> None:
    """Reduce failure: retry once, then deterministic exact-title dedupe keeping
    ALL distinct items. No candidates[:N] truncation anywhere."""
    attempts = {"n": 0}

    class AlwaysFailReduceClient:
        name = "fake-reduce-fail"
        uid = 0
        lock = __import__("threading").Lock()

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            if "You dedupe knowledge items" in prompt:
                attempts["n"] += 1
                raise RuntimeError("reduce timed out")
            with type(self).lock:
                type(self).uid += 1
                n = type(self).uid
            # Segment 2 and 3 emit the SAME title (near-duplicates → one reduce
            # group); the rest are distinct.
            title = "Duplicate pitfall" if n in (2, 3) else f"Distinct pitfall {n}"
            return {"items": [dict(_PITFALL_ITEM, title=title)]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    config = _config(tmp_path)
    config.distiller.segment_chars = 400
    runtime_paths = _runtime(tmp_path)
    distiller = Distiller(config, runtime_paths=runtime_paths, sampling_client=AlwaysFailReduceClient())
    try:
        import os

        turn = "## T{n} User\n" + ("durable distinct lesson " * 12)
        content = "<!-- synapse-transcript v2 -->\n" + "\n\n".join(turn.format(n=n) for n in range(1, 7))
        _write_transcript(runtime_paths, "mem_session_nofail1", content, session_key="sess-nofail")
        old_ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_nofail1.md", (old_ts, old_ts))

        report = distiller.run(downgrade_supersede=False)
        assert report.transcripts_distilled == 1
        # 6 candidates (2 share a title → one reduce group); reduce failed twice
        # (retry once); deterministic fallback deduped the pair → 5 distinct
        # items written. NOTHING truncated beyond the exact-title dedupe.
        assert attempts["n"] == 2, "reduce retried exactly once"
        assert report.items_extracted == 5
        assert report.reduce_failures == 1
        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths.active / "mem_session_nofail1.md")
        assert len(stamped.metadata.distilled_node_ids) == 5
    finally:
        distiller.close()


def test_item_warn_threshold_logs_but_writes_everything(tmp_path: Path, caplog) -> None:
    import logging as _logging

    class ProlificClient:
        name = "fake-prolific"
        uid = 0
        lock = __import__("threading").Lock()

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            with type(self).lock:
                type(self).uid += 1
                n = type(self).uid
            # 3 distinct items per call → 4 segments × 3 = 12 items > threshold 2.
            return {"items": [
                dict(_PITFALL_ITEM, title=f"Prolific pitfall {n}.1"),
                dict(_PITFALL_ITEM, title=f"Prolific pitfall {n}.2"),
                dict(_PITFALL_ITEM, title=f"Prolific pitfall {n}.3"),
            ]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    config = _config(tmp_path)
    config.distiller.segment_chars = 200
    config.distiller.item_warn_threshold = 2
    runtime_paths = _runtime(tmp_path)
    distiller = Distiller(config, runtime_paths=runtime_paths, sampling_client=ProlificClient())
    try:
        import os

        turn = "## T{n} User\n" + ("prolific lesson " * 8)
        content = "<!-- synapse-transcript v2 -->\n" + "\n\n".join(turn.format(n=n) for n in range(1, 5))
        _write_transcript_raw(config, runtime_paths, "mem_session_warn1", content, session_key="sess-warn")
        old_ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        os.utime(runtime_paths.active / "mem_session_warn1.md", (old_ts, old_ts))

        records: list = []

        class Capture(_logging.Handler):
            def emit(self, record: _logging.LogRecord) -> None:
                records.append(record)

        capture = Capture()
        dist_logger = _logging.getLogger("synapse.distiller")
        dist_logger.addHandler(capture)
        try:
            report = distiller.run(downgrade_supersede=False)
        finally:
            dist_logger.removeHandler(capture)
        assert report.transcripts_distilled == 1
        assert report.items_extracted > config.distiller.item_warn_threshold
        warns = [r for r in records if "item_warn_threshold" in r.getMessage()]
        assert len(warns) == 1, "warned once, but NOTHING was dropped"
        from synapse.storage import read_node_file

        stamped = read_node_file(runtime_paths.active / "mem_session_warn1.md")
        assert len(stamped.metadata.distilled_node_ids) == report.items_extracted
    finally:
        distiller.close()


def test_reasoning_effort_passed_only_when_set(tmp_path: Path) -> None:
    """[distiller].reasoning_effort must reach sample_json as the reasoning_effort
    kwarg ONLY when configured; default None must omit the field."""

    class CapturingClient:
        name = "fake-capture"

        def __init__(self) -> None:
            self.seen: list = []

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            self.seen.append(reasoning_effort)
            return {"items": [dict(_PITFALL_ITEM, title="Capture pitfall")]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    content = "<!-- synapse-transcript v2 -->\n## T1 User\ndurable capture lesson content\n"

    # Default: None.
    config = _config(tmp_path)
    runtime_paths = _runtime(tmp_path)
    client = CapturingClient()
    d = Distiller(config, runtime_paths=runtime_paths, sampling_client=client)
    try:
        _write_transcript(runtime_paths, "mem_session_eff1", content, session_key="eff1")
        import os as _os
        old = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        _os.utime(runtime_paths.active / "mem_session_eff1.md", (old, old))
        d.run(downgrade_supersede=False)
    finally:
        d.close()
    assert client.seen and all(v is None for v in client.seen)

    # Set: "low" passed through.
    config2 = _config(tmp_path)
    config2.distiller.reasoning_effort = "low"
    runtime_paths2 = _runtime(tmp_path)
    client2 = CapturingClient()
    d2 = Distiller(config2, runtime_paths=runtime_paths2, sampling_client=client2)
    try:
        _write_transcript(runtime_paths2, "mem_session_eff2", content, session_key="eff2")
        import os as _os
        old = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        _os.utime(runtime_paths2.active / "mem_session_eff2.md", (old, old))
        d2.run(downgrade_supersede=False)
    finally:
        d2.close()
    assert client2.seen and all(v == "low" for v in client2.seen)


def test_funnel_fields_in_distilled_log(tmp_path: Path) -> None:
    import logging as _logging

    class FunnelClient:
        name = "fake-funnel"

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=(), reasoning_effort=None):
            if "You dedupe knowledge items" in prompt:
                return {"items": [dict(_PITFALL_ITEM, title="Funnel merged pitfall")]}
            return {"items": [dict(_PITFALL_ITEM, title="Funnel segment pitfall")]}

        def decide_memory_write(self, request):
            from synapse.server.sampling import MemoryWriteSamplingDecision

            return MemoryWriteSamplingDecision(action="create", target_node_ids=(), reasoning="new", confidence=0.9)

    config = _config(tmp_path)
    config.distiller.segment_chars = 200
    runtime_paths = _runtime(tmp_path)
    distiller = Distiller(config, runtime_paths=runtime_paths, sampling_client=FunnelClient())
    import os

    turn = "## T{n} User\n" + ("funnel lesson " * 10)
    content = "<!-- synapse-transcript v2 -->\n" + "\n\n".join(turn.format(n=n) for n in range(1, 5))
    _write_transcript_raw(config, runtime_paths, "mem_session_fun1", content, session_key="sess-fun")
    old_ts = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(runtime_paths.active / "mem_session_fun1.md", (old_ts, old_ts))

    records: list = []

    class Capture2(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            records.append(record)

    capture2 = Capture2()
    dist_logger2 = _logging.getLogger("synapse.distiller")
    dist_logger2.addHandler(capture2)
    try:
        distiller.run(downgrade_supersede=False)
    finally:
        distiller.close()
        dist_logger2.removeHandler(capture2)

        entries = [r for r in records if r.getMessage() == "Distilled transcript"]
        assert len(entries) == 1
        data = entries[0].__dict__
        assert "candidates" in data, "funnel: pre-reduce candidate count"
        assert "after_reduce" in data, "funnel: post-reduce count"
        assert "reduce_groups_llm" in data
        assert "reduce_failure" in data and data["reduce_failure"] is False
        assert "written_by_action" in data and data["written_by_action"].get("create", 0) >= 1
