"""Tests for the shared transcript and OKF lifecycle primitives."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synapse.config import SynapseConfig
from synapse.lifecycle.distiller import (
    Distiller,
    DistillerReport,
    content_sha256,
    is_fully_distilled,
)
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType
from synapse.okf import OkfItem, OkfValidationError, validate_okf_node
from synapse.server.service import SynapseServiceError, SynapseServerService
from synapse.storage import write_node_file
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

    from synapse.sync import SyncManager

    config = _config(Path(str(path.base)).parent)
    manager = SyncManager(config, runtime_paths=path, debounce_seconds=0.0)
    try:
        manager.startup_sync()
    finally:
        manager.close()



def _make_distiller(tmp_path: Path) -> tuple[Distiller, RuntimePaths]:
    runtime_paths = _runtime(tmp_path)
    return Distiller(_config(tmp_path), runtime_paths=runtime_paths), runtime_paths




# ---------------------------------------------------------------------------
# 1. Selector + hash gate
# ---------------------------------------------------------------------------


def test_selector_picks_only_idle_undistilled_transcripts(tmp_path: Path) -> None:
    distiller, paths = _make_distiller(tmp_path)
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

def test_stamp_transcript_persists_episode_progress(tmp_path: Path) -> None:
    distiller, paths = _make_distiller(tmp_path)
    transcript = _write_transcript(
        paths,
        "mem_session_stamp1",
        "## T1 User\nrequest\n\n## T2 Assistant\nresponse",
        session_key="stamp-test",
    )
    try:
        distiller._stamp_transcript(transcript, ["mem_knowledge1"], distilled_turn=2)
        from synapse.storage import read_node_file

        stamped = read_node_file(paths.active / f"{transcript.id}.md")
        assert stamped.metadata.distilled_hash == content_sha256(stamped.content)
        assert stamped.metadata.distilled_node_ids == ["mem_knowledge1"]
        assert stamped.metadata.distilled_turn == 2
        assert stamped.metadata.distilled_at is not None
    finally:
        distiller.close()


def test_invalid_final_okf_item_is_quarantined(tmp_path: Path) -> None:
    from synapse.okf import EvidenceEntry

    distiller, paths = _make_distiller(tmp_path)
    transcript = _write_transcript(paths, "mem_session_quarantine1", "## T1 User\nsource quote", session_key="quarantine-test")
    item = OkfItem(
        okf_type="fact",
        title="Quarantine test item",
        takeaway="The claim is retained for repair.",
        sections={"Details": "Supporting detail."},
        sources=[transcript.id],
        evidence=[EvidenceEntry(turn="T1", quote="source quote")],
    )
    exc = SynapseServiceError(
        "INVALID_OKF_FINAL_BODY",
        "Final OKF body is invalid.",
        details={
            "final_content": "not an OKF body",
            "warnings": [{"code": "okf_missing_section:Details"}],
            "decision": {"action": "complement", "target_node_ids": ["mem_target1"]},
        },
    )
    try:
        path = distiller._quarantine_invalid_item(
            transcript,
            item,
            {"sources": [transcript.id]},
            exc,
            context={"merge_update_target": "mem_target2"},
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        assert distiller._is_final_okf_validation_error(exc)
        assert record["final_candidate_body"] == "not an OKF body"
        assert record["intended_action"] == "complement"
        assert record["target_node_ids"] == ["mem_target1", "mem_target2"]
        assert record["evidence"] == [{"turn": "T1", "quote": "source quote"}]
    finally:
        distiller.close()


# ---------------------------------------------------------------------------
# OKF item rendering
# ---------------------------------------------------------------------------





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
# 45-day transcript retention
# ---------------------------------------------------------------------------


def test_retention_archives_distilled_only(tmp_path: Path) -> None:
    distiller, paths = _make_distiller(tmp_path)
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
    assert config.distiller.idle_minutes == 60
    assert config.distiller.retention_days == 45
    assert config.distiller.enforce_okf is False
    assert config.distiller.llm_timeout_seconds == 300
    assert config.distiller.max_transcripts_per_run == 500




def test_word_limit_warning_suppressed_for_session_transcripts(tmp_path: Path) -> None:
    runtime_paths = _runtime(tmp_path)
    big = "## User\n" + ("混合内容 mixed " * 700)  # well over 3500 word units
    metadata = NodeMetadata(id="mem_session_big1", title="Session summary — big", type=NodeType.TRANSIENT, session_key="s")
    node = Node(metadata=metadata, content=big, file_path=Path("active/mem_session_big1.md"))
    # Must not raise; word-count warning is suppressed for session transcripts.
    written = write_node_file(node, base_path=runtime_paths.base)
    assert written.exists()


# ---------------------------------------------------------------------------
# Transcript search exclusion and result backfill
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
# Distiller provenance and curated supersede protection
# ---------------------------------------------------------------------------





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
# OKF source validation
# ---------------------------------------------------------------------------





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


def test_validator_flags_duplicate_required_headings() -> None:
    warnings = validate_okf_node(
        node_type="persistent",
        content="## Takeaway\nt\n\n## Details\nfirst\n\n## Details\nsecond\n\n## Sources\n- mem_session_source",
        okf_type="fact",
        sources=["mem_session_source"],
        title="Duplicate details test",
    )
    assert any(warning.code == "okf_duplicate_section:Details" for warning in warnings)


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





# ---------------------------------------------------------------------------
# Write-path candidate exclusion
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

    long = validate_okf_node(**base, title="x" * 120)
    assert not any(w.code.startswith("okf_title") for w in long)

    prefix = validate_okf_node(**base, title="Procedure: run the dedupe tool")
    assert any(w.code == "okf_title_type_prefix" for w in prefix)

    ok = validate_okf_node(**base, title="AuthZ tenant-group projection uses LOCAL rows")
    assert not any(w.code.startswith("okf_title") for w in ok)





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
    """The write path preserves supersede behavior for legacy-transcript sources."""

    runtime_paths = _runtime(tmp_path)

    class SupersedeLegacyClient:
        name = "fake-supersede-legacy"

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

    # Write-path supersede for transcript-sourced knowledge must stay allowed.
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
# Shared long-stage timeout
# ---------------------------------------------------------------------------
def test_distiller_llm_timeout_default_is_300(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.distiller.llm_timeout_seconds == 300


# ---------------------------------------------------------------------------
# Transcript format parsing
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











# ---------------------------------------------------------------------------
# P2+: evidence verification (deterministic, no LLM)
# ---------------------------------------------------------------------------


def test_verify_evidence_accepts_verbatim_and_rejects_paraphrase() -> None:
    from synapse.okf import EvidenceEntry, verify_evidence

    turns = {
        "T1": "We decided to use kafka offsets for resume.",
        "T2": "call: grep -rn offsets\nresult [ok]: src/main.py:1 uses offsets",
        "T3": "Could this be the cause?",
    }
    verified, unverified = verify_evidence(
        [
            EvidenceEntry(turn="T1", quote="use kafka offsets"),
            EvidenceEntry(turn="T1", quote="  use  KAFKA\noffsets "),  # case/whitespace collapse
            EvidenceEntry(turn="T2", quote="result [ok]: src/main.py:1 uses offsets"),
            EvidenceEntry(turn="T1", quote="paraphrased claim not present"),
            EvidenceEntry(turn="T9", quote="kafka"),
            EvidenceEntry(turn="T3", quote="this be the cause"),
        ],
        turns=turns,
        question_only_turns={"T3"},
    )
    assert [e.turn for e in verified] == ["T1", "T1", "T2"]
    assert [(e.turn, e.quote[:20]) for e in unverified] == [
        ("T1", "paraphrased claim no"),
        ("T9", "kafka"),
        ("T3", "this be the cause"),
    ]





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


def test_evidence_across_read_line_number_prefixes_verifies() -> None:
    """§9 verifier boundary (judge report_v3 §1): read-tool results carry
    per-line numbering (``12:`` / ``12-`` / ``12-18:``) embedded in the result
    line. A quote that is verbatim modulo those prefixes verifies — the
    stripped form is eligible alongside the raw text. Assistant prose and
    fabricated quotes still fail."""

    from synapse.okf import EvidenceEntry, verify_evidence
    from synapse.okf.transcript_format import parse_transcript

    content = (
        "<!-- synapse-transcript v2 -->\n"
        "## T1 User\nshow me the loader config\n\n"
        "## T2 Assistant\nthe loader config is at line 10\n"
        "### Tool T2.1 read\n"
        "call: read conf/loader.toml\n"
        "result [ok]: [conf/loader.toml] 10: workers = 4\n12: timeout = 30\n13-16: retries = [1, 2]\n"
    )
    parsed = parse_transcript(content)
    turns = {turn.id: turn.eligible_evidence_text() for turn in parsed.turns}

    # A multi-line quote spanning numbered read lines (model dropped the
    # prefixes) verifies via the stripped variant.
    verified, _ = verify_evidence(
        [EvidenceEntry(turn="T2", quote="workers = 4 timeout = 30 retries = [1, 2]")],
        turns=turns,
    )
    assert len(verified) == 1, "numbering-stripped quote must verify"

    # The RAW form (prefixes kept) still verifies — existing behavior kept.
    verified_raw, _ = verify_evidence(
        [EvidenceEntry(turn="T2", quote="12: timeout = 30")],
        turns=turns,
    )
    assert len(verified_raw) == 1, "raw form with prefixes must stay verified"

    # Assistant prose still fails.
    verified_prose, _ = verify_evidence(
        [EvidenceEntry(turn="T2", quote="the loader config is at line 10")],
        turns=turns,
    )
    assert verified_prose == []

    # A fabricated quote still fails.
    verified_fake, _ = verify_evidence(
        [EvidenceEntry(turn="T2", quote="workers = 99 concurrency = 8")],
        turns=turns,
    )
    assert verified_fake == []
