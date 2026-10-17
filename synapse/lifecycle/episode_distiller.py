"""Episode-scoped summarization and explicit OKF integration for pilot/backfill."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from synapse.config import DeciderEndpointSettings, SynapseConfig
from synapse.lifecycle.distiller import Distiller, content_sha256
from synapse.models import Node, NodeStatus, NodeType, generate_node_id, slugify_english_title
from synapse.okf import (
    EvidenceEntry,
    TYPE_REQUIRED_SECTIONS,
    EVIDENCE_QUOTE_MAX_CHARS,
    TAKEAWAY_MAX_CHARS,
    OkfItem,
    render_evidence_section,
    verify_evidence,
    validate_okf_node,
)
from synapse.okf.episode import (
    render_episode_integrate_prompt,
    render_episode_lesson_reduce_prompt,
    render_episode_map_prompt,
    render_episode_summary_prompt,
    render_episode_takeaway_repair_prompt,
)
from synapse.okf.transcript_format import TranscriptTurn, parse_transcript
from synapse.okf.transcripts import is_session_transcript
from synapse.retrieval import RetrievalPipeline
from synapse.server.service import SynapseServiceError, SynapseServerService
from synapse.storage import DistillerRunMetrics, SQLiteNodeStore
from synapse.utils.runtime import RuntimePaths


@dataclass(slots=True)
class StageMetrics:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: int = 0
    usage_available: bool = True

    def add(self, *, elapsed_ms: float, usage: Mapping[str, Any] | None, calls: int = 1) -> None:
        self.calls += calls
        self.latency_ms += round(elapsed_ms)
        if usage is None or not usage.get("available", True):
            self.usage_available = False
            return
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.reasoning_tokens += int(usage.get("reasoning_tokens", 0))

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not self.usage_available:
            result["prompt_tokens"] = None
            result["completion_tokens"] = None
            result["reasoning_tokens"] = None
        return result


@dataclass(slots=True)
class EpisodeRun:
    transcript_id: str
    session_key: str | None
    episode_id: str
    chars: int
    turns: int
    chunks: int
    lessons: list[dict[str, Any]] = field(default_factory=list)
    ops: list[dict[str, Any]] = field(default_factory=list)
    node_bodies: list[dict[str, str]] = field(default_factory=list)
    stages: dict[str, StageMetrics] = field(default_factory=dict)
    retrieval_calls: int = 0
    stamped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "transcript_id": self.transcript_id,
            "session_key": self.session_key,
            "episode_id": self.episode_id,
            "chars": self.chars,
            "turns": self.turns,
            "chunks": self.chunks,
            "lessons": self.lessons,
            "ops": self.ops,
            "node_bodies": self.node_bodies,
            "stages": {name: metrics.as_dict() for name, metrics in self.stages.items()},
            "retrieval_calls": self.retrieval_calls,
            "stamped": self.stamped,
        }


@dataclass(slots=True)
class Lesson:
    topic: str
    claim: str
    evidence_turns: list[str]
    okf_type: str = "fact"

    def as_dict(self) -> dict[str, Any]:
        return {
            "okf_type": self.okf_type,
            "topic": self.topic,
            "claim": self.claim,
            "evidence_turns": self.evidence_turns,
        }


@dataclass(slots=True)
class EpisodeOp:
    action: str
    title: str = ""
    okf_type: str = ""
    content: str = ""
    target_id: str | None = None
    justification: str = ""
    evidence: list[EvidenceEntry] = field(default_factory=list)
    verified: bool | None = None
    reason: str = ""
    lesson_index: int | None = None

class EpisodeDistiller:
    """Run the new pilot pipeline against one explicitly supplied runtime copy."""

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths,
        sampling_client: Any,
        gatekeeper_client: Any | None = None,
    ) -> None:
        self.config = config
        self.settings = config.distiller.episode
        self.runtime_paths = runtime_paths
        self.sampling_client = sampling_client
        self.gatekeeper_client = gatekeeper_client if gatekeeper_client is not None else sampling_client
        self.service = SynapseServerService(
            config,
            runtime_paths=runtime_paths,
            sampling_client=sampling_client,
        )
        self._distiller_helpers = Distiller(
            config,
            runtime_paths=runtime_paths,
            sampling_client=sampling_client,
        )
        self._commit_lock = threading.Lock()

    @staticmethod
    def make_scanner_client(config: SynapseConfig) -> Any:
        from synapse.server.decider import LocalLLMDecider

        episode = config.distiller.episode
        endpoints = episode.scanner_endpoints or episode.endpoints or [episode.endpoint]
        scanner_endpoints = [
            endpoint.model_copy(
                update={
                    "extra_body": {
                        **endpoint.extra_body,
                        "reasoning_effort": endpoint.extra_body.get("reasoning_effort", "low"),
                    }
                }
            )
            for endpoint in endpoints
        ]
        settings = config.decider.model_copy(
            update={
                "provider": "local_llm",
                "endpoints": scanner_endpoints,
                "timeout_seconds": episode.timeout_seconds,
                "max_tokens": episode.max_tokens,
                "temperature": episode.temperature,
                "endpoint_max_concurrency": max(16, config.decider.endpoint_max_concurrency),
            }
        )
        return LocalLLMDecider(settings)

    @staticmethod
    def make_gatekeeper_client(config: SynapseConfig) -> Any:
        from synapse.server.decider import LocalLLMDecider

        episode = config.distiller.episode
        endpoint = episode.gatekeeper_endpoint or DeciderEndpointSettings(
            base_url="http://127.0.0.1:47870/v1",
            model="claude-haiku-5.5",
        )
        settings = config.decider.model_copy(
            update={
                "provider": "local_llm",
                "endpoints": [endpoint],
                "max_tokens": episode.max_tokens,
                "temperature": episode.temperature,
                "endpoint_max_concurrency": max(16, config.decider.endpoint_max_concurrency),
            }
        )
        return LocalLLMDecider(settings)

    def close(self) -> None:
        self._distiller_helpers.close()

    def record_run_metrics(
        self,
        runs: list[EpisodeRun],
        *,
        started_at: str,
        completed_at: str,
        duration_ms: int,
        transcripts_scanned: int | None = None,
        llm_failures: int = 0,
        archived_transcripts: int = 0,
    ) -> None:
        operations = [op for run in runs for op in run.ops]
        metrics = DistillerRunMetrics(
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            transcripts_scanned=transcripts_scanned or len(runs),
            transcripts_distilled=sum(1 for run in runs if run.stamped and (run.lessons or run.ops)),
            transcripts_skipped=sum(1 for run in runs if not run.lessons and not run.ops),
            items_extracted=sum(len(run.lessons) for run in runs),
            items_invalid=sum(op.get("action") == "none" for op in operations),
            items_quarantined=sum(op.get("action") == "quarantined" for op in operations),
            decider_creates=sum(op.get("action") == "created" for op in operations),
            decider_complements=sum(op.get("action") == "updated" for op in operations),
            decider_supersedes=0,
            decider_downgrades=0,
            llm_failures=llm_failures,
            archived_transcripts=archived_transcripts,
        )
        with SQLiteNodeStore(
            self.runtime_paths.base / "synapse.db",
            embedding_dimension=self.config.embedding.dimension or 0,
        ) as store:
            store.record_distiller_run(metrics)

    def run(self, session: str) -> EpisodeRun:
        return self.integrate_episode(*self.extract_episode(session))

    def extract_episode(
        self, session: str
    ) -> tuple[Node, list[TranscriptTurn], list[TranscriptTurn], list[TranscriptTurn], str, list[Lesson], EpisodeRun]:
        """Read and summarize one transcript without consulting or changing the knowledge graph."""
        transcript = self._resolve_transcript(session)
        if not is_session_transcript(transcript):
            raise ValueError(f"Not a session transcript: {transcript.id}")
        parsed = parse_transcript(transcript.content)
        watermark = transcript.metadata.distilled_turn or 0
        original_pending = parsed.turns_after(watermark)
        if not original_pending:
            episode_id = self._episode_id(transcript, watermark, watermark, transcript.content)
            with self._commit_lock:
                self._distiller_helpers._stamp_transcript(transcript, [], distilled_turn=watermark)
            run = EpisodeRun(transcript.id, transcript.metadata.session_key, episode_id, 0, 0, 0, stamped=True)
            return transcript, [], [], parsed.turns, "", [], run

        all_turns: list[TranscriptTurn] = []
        for turn in parsed.turns:
            dehydrated = _dehydrate_turn(turn)
            if dehydrated is not None:
                all_turns.append(dehydrated)
        dehydrated_by_number = {turn.number: turn for turn in all_turns}
        pending = [dehydrated_by_number[turn.number] for turn in original_pending if turn.number in dehydrated_by_number]
        source_text = "\n\n".join(_render_turn(turn) for turn in pending)
        global_user_spine = _render_global_user_spine(parsed.turns)
        chunks = build_episode_chunks(pending, self.settings.chunk_chars)
        run = EpisodeRun(
            transcript_id=transcript.id,
            session_key=transcript.metadata.session_key,
            episode_id=self._episode_id(
                transcript, original_pending[0].number, original_pending[-1].number, source_text
            ),
            chars=len(source_text),
            turns=len(pending),
            chunks=len(chunks),
        )

        lesson_rows: list[Lesson] = []
        if len(pending) >= 2 and len(source_text) >= 600:
            if len(chunks) == 1:
                text, _ids = chunks[0]
                payload = self._call(
                    run,
                    "summarize",
                    render_episode_summary_prompt(turns=text, global_user_spine=global_user_spine),
                )
                lesson_rows = _parse_lessons(payload)
            else:
                mapped: list[Lesson] = []
                for index, (text, _ids) in enumerate(chunks, 1):
                    payload = self._call(
                        run,
                        "map",
                        render_episode_map_prompt(
                            turns=text,
                            global_user_spine=global_user_spine,
                            chunk_index=index,
                            chunk_count=len(chunks),
                        ),
                    )
                    mapped.extend(_parse_lessons(payload))
                reduced = self._call(
                    run,
                    "lesson_reduce",
                    render_episode_lesson_reduce_prompt(
                        chunk_lessons=[lesson.as_dict() for lesson in mapped],
                        global_user_spine=global_user_spine,
                    ),
                )
                lesson_rows = _parse_lessons(reduced)

        run.lessons = [lesson.as_dict() for lesson in lesson_rows]
        return transcript, all_turns, pending, parsed.turns, global_user_spine, lesson_rows, run

    def integrate_episode(
        self,
        transcript: Node,
        all_turns: list[TranscriptTurn],
        pending_turns: list[TranscriptTurn],
        verification_turns: list[TranscriptTurn],
        global_user_spine: str,
        lessons: list[Lesson],
        run: EpisodeRun,
    ) -> EpisodeRun:
        """Retrieve, integrate, commit, and stamp one extracted episode."""
        if run.stamped:
            return run
        original_pending = [
            turn for turn in verification_turns if turn.number > (transcript.metadata.distilled_turn or 0)
        ]
        if not original_pending:
            run.stamped = True
            return run

        if lessons:
            self._integrate(
                transcript,
                all_turns,
                lessons,
                run,
                global_user_spine=global_user_spine,
                verification_turns=verification_turns,
            )
        with self._commit_lock:
            self._distiller_helpers._stamp_transcript(
                transcript,
                [str(op.get("node_id")) for op in run.ops if op.get("node_id")],
                distilled_turn=original_pending[-1].number,
            )
        run.stamped = True
        return run

    def _resolve_transcript(self, session: str) -> Node:
        with SQLiteNodeStore(
            self.runtime_paths.base / "synapse.db",
            embedding_dimension=self.config.embedding.dimension or 0,
        ) as store:
            node = store.get_node(session)
            if node is not None:
                return node
            matches = [
                candidate
                for candidate in store.list_nodes({"type": NodeType.TRANSIENT, "status": NodeStatus.ACTIVE})
                if candidate.metadata.session_key == session and is_session_transcript(candidate)
            ]
        if len(matches) != 1:
            raise ValueError(f"Expected one transcript matching session {session!r}; found {len(matches)}")
        return matches[0]

    @staticmethod
    def _episode_id(transcript: Node, start: int, end: int, content: str) -> str:
        digest = hashlib.sha256(f"{transcript.id}:{start}:{end}:{content_sha256(content)}".encode()).hexdigest()[:20]
        return f"{transcript.id}:{start}-{end}:{digest}"

    def _call(self, run: EpisodeRun, stage: str, prompt: str) -> Any:
        client = self.gatekeeper_client if stage in {"integrate", "takeaway_repair"} else self.sampling_client
        metrics = run.stages.setdefault(stage, StageMetrics())
        started = time.perf_counter()
        call_context = getattr(client, "call_context", None)
        context = None
        try:
            if callable(call_context):
                with call_context(
                    transcript_id=run.transcript_id,
                    stage=f"episode_{stage}",
                    timeout_seconds=self.settings.timeout_seconds,
                ) as context:
                    payload = client.sample_json(
                        prompt=prompt,
                        system_prompt="",
                        max_tokens=self.settings.max_tokens,
                    )
            else:
                payload = client.sample_json(
                    prompt=prompt,
                    system_prompt="",
                    max_tokens=self.settings.max_tokens,
                )
            return payload
        finally:
            usage = context.get("usage") if isinstance(context, Mapping) else None
            endpoints = context.get("endpoints", {}) if isinstance(context, Mapping) else {}
            calls = sum(int(stats.get("calls", 0)) for stats in endpoints.values()) if isinstance(endpoints, Mapping) else 0
            metrics.add(elapsed_ms=(time.perf_counter() - started) * 1000, usage=usage, calls=calls or 1)

    def _commit_call(self, run: EpisodeRun, callback: Callable[[], Any]) -> Any:
        call_context = getattr(self.gatekeeper_client, "call_context", None)
        if not callable(call_context):
            return callback()
        context = None
        started = time.perf_counter()
        try:
            with call_context(
                transcript_id=run.transcript_id,
                stage="episode_final_okf_repair",
                timeout_seconds=self.settings.timeout_seconds,
            ) as context:
                return callback()
        finally:
            endpoints = context.get("endpoints", {}) if isinstance(context, Mapping) else {}
            calls = sum(int(stats.get("calls", 0)) for stats in endpoints.values()) if isinstance(endpoints, Mapping) else 0
            if calls:
                metrics = run.stages.setdefault("final_okf_repair", StageMetrics())
                metrics.add(
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    usage=context.get("usage") if isinstance(context, Mapping) else None,
                    calls=calls,
                )

    def _integrate(
        self,
        transcript: Node,
        turns: list[TranscriptTurn],
        lessons: list[Lesson],
        run: EpisodeRun,
        *,
        global_user_spine: str,
        verification_turns: list[TranscriptTurn],
    ) -> None:
        if not lessons:
            return
        helper_token = self.service.push_sampling_client(self.gatekeeper_client)
        try:
            prior_nodes = self._same_session_nodes(transcript)
            with RetrievalPipeline(self.config, runtime_paths=self.runtime_paths) as retrieval:
                for batch in self._integrate_lesson_batches(turns, lessons):
                    candidates = self._retrieve_for_lessons(retrieval, batch, run)
                    allowed_target_ids = {node["id"] for node in candidates}
                    allowed_target_ids.update(node["id"] for node in prior_nodes)
                    evidence_context = self._evidence_context(turns, batch)
                    payload = self._call(
                        run,
                        "integrate",
                        render_episode_integrate_prompt(
                            lessons=[lesson.as_dict() for lesson in batch],
                            same_session_nodes=prior_nodes,
                            retrieved_nodes=candidates,
                            evidence_context=evidence_context,
                            global_user_spine=global_user_spine,
                        ),
                    )
                    for op in _parse_ops(payload, len(batch)):
                        result = self._apply_op(
                            transcript,
                            verification_turns,
                            op,
                            run,
                            allowed_target_ids=allowed_target_ids,
                        )
                        run.ops.append(result)
                        if result.get("node_id"):
                            node = self._load_node(str(result["node_id"]))
                            prior_nodes = _upsert_prior_node(prior_nodes, node)
                            if result.get("action") in {"created", "updated"}:
                                run.node_bodies.append({"node_id": node.id, "title": node.title, "content": node.content})
        finally:
            self.service.reset_sampling_client(helper_token)
    def _integrate_lesson_batches(
        self, turns: list[TranscriptTurn], lessons: list[Lesson]
    ) -> list[list[Lesson]]:
        batches: list[list[Lesson]] = []
        current: list[Lesson] = []
        for lesson in lessons:
            candidate = [*current, lesson]
            excerpt_size = len(json.dumps(self._evidence_context(turns, candidate), ensure_ascii=False))
            if current and (
                len(candidate) > self.settings.integrate_lessons_per_call
                or excerpt_size > self.settings.evidence_context_token_budget
            ):
                batches.append(current)
                current = []
                excerpt_size = len(json.dumps(self._evidence_context(turns, [lesson]), ensure_ascii=False))
            if excerpt_size > self.settings.evidence_context_token_budget:
                raise ValueError(
                    f"Evidence excerpts for one lesson require {excerpt_size} tokens by conservative estimate; "
                    f"budget is {self.settings.evidence_context_token_budget}; refusing to drop evidence"
                )
            current.append(lesson)
        if current:
            batches.append(current)
        return batches

    def _same_session_nodes(self, transcript: Node) -> list[dict[str, Any]]:
        ids = transcript.metadata.distilled_node_ids
        if not ids:
            return []
        with SQLiteNodeStore(self.runtime_paths.base / "synapse.db", embedding_dimension=self.config.embedding.dimension or 0) as store:
            nodes = store.get_nodes(ids)
        return [_compact_node(node) for node in nodes if node.metadata.type is NodeType.PERSISTENT and node.metadata.status is NodeStatus.ACTIVE]

    def _retrieve_for_lessons(
        self,
        retrieval: RetrievalPipeline,
        lessons: list[Lesson],
        run: EpisodeRun,
    ) -> list[dict[str, Any]]:
        by_id: dict[str, tuple[float, Node]] = {}
        for lesson in lessons:
            query = f"{lesson.topic}\n{lesson.claim}"
            response = retrieval.search(
                query,
                top_k=self.settings.retrieval_top_k_per_lesson,
                update_access=False,
                result_filter=lambda node: node.metadata.type is NodeType.PERSISTENT and node.metadata.status is NodeStatus.ACTIVE and not is_session_transcript(node),
            )
            run.retrieval_calls += 1
            for item in response.results:
                current = by_id.get(item.node.id)
                if current is None or item.score > current[0]:
                    by_id[item.node.id] = (item.score, item.node)
        ordered = sorted(by_id.values(), key=lambda pair: (-pair[0], pair[1].id))
        output: list[dict[str, Any]] = []
        remaining = self.settings.retrieval_body_token_budget
        for score, node in ordered:
            body = node.content
            estimate = len(body)  # one token per character is a conservative upper bound for the context budget
            include_body = estimate <= remaining
            if include_body:
                remaining -= estimate
            output.append(
                {
                    "id": node.id,
                    "title": node.title,
                    "okf_type": node.metadata.okf_type,
                    "takeaway": _takeaway(node.content),
                    "score": score,
                    "content": body if include_body else None,
                }
            )
        return output

    @staticmethod
    def _evidence_context(turns: list[TranscriptTurn], lessons: list[Lesson]) -> list[dict[str, Any]]:
        turns_by_id = {turn.id: turn for turn in turns}
        result: list[dict[str, Any]] = []
        for index, lesson in enumerate(lessons):
            excerpts: list[dict[str, str]] = []
            for turn_id in dict.fromkeys(lesson.evidence_turns):
                turn = turns_by_id.get(turn_id)
                if turn is None:
                    excerpts.append(
                        {"turn": turn_id, "role": "missing", "text": "[turn not present in the transcript]"}
                    )
                    continue
                excerpts.append({"turn": turn.id, "role": turn.role, "text": _turn_evidence_excerpt(turn)})
            result.append(
                {
                    "lesson_index": index,
                    "topic": lesson.topic,
                    "claim": lesson.claim,
                    "evidence": excerpts,
                }
            )
        return result

    def _repair_takeaway_only(
        self,
        op: EpisodeOp,
        content: str,
        sources: list[str],
        run: EpisodeRun,
    ) -> tuple[str, str | None]:
        findings = validate_okf_node(
            node_type="persistent",
            content=content,
            okf_type=op.okf_type,
            sources=sources,
            title=op.title,
        )
        if len(findings) != 1 or findings[0].code != "okf_takeaway_invalid":
            return content, None
        original = _takeaway_section(content)
        payload = self._call(
            run,
            "takeaway_repair",
            render_episode_takeaway_repair_prompt(
                title=op.title,
                okf_type=op.okf_type,
                takeaway=original,
            ),
        )
        takeaway = payload.get("takeaway") if isinstance(payload, Mapping) else None
        if not isinstance(takeaway, str) or not _valid_takeaway(takeaway, hard_limit=TAKEAWAY_MAX_CHARS):
            return content, "Takeaway-only repair did not return one non-empty line within the 200-character limit"
        return _replace_takeaway(content, takeaway.strip()), None

    def _apply_op(
        self,
        transcript: Node,
        turns: list[TranscriptTurn],
        op: EpisodeOp,
        run: EpisodeRun,
        *,
        allowed_target_ids: set[str],
    ) -> dict[str, Any]:
        turn_by_id = {turn.id: turn for turn in turns}
        evidence_turns = {turn.id: _turn_verification_text(turn) for turn in turns}
        _verified_quotes, unverified = verify_evidence(op.evidence, turns=evidence_turns)
        verified_flag = (
            bool(op.verified)
            if op.verified is not None
            else bool(op.evidence) and not unverified
        )
        evidence_sources: list[str] = []
        for entry in op.evidence:
            turn = turn_by_id.get(entry.turn)
            if turn is None:
                source = "unknown"
            elif turn.role == "User":
                source = "user"
            elif verify_evidence([entry], turns={turn.id: turn.body})[0]:
                source = "assistant"
            elif any(verify_evidence([entry], turns={turn.id: section})[0] for section in turn.tool_sections):
                source = "tool_result"
            else:
                source = "assistant"
            if source not in evidence_sources:
                evidence_sources.append(source)
        def no_op(reason: str) -> dict[str, Any]:
            return {
                "action": "none",
                "title": op.title,
                "target_id": op.target_id,
                "reason": reason,
                "verified": verified_flag,
                "evidence_sources": evidence_sources,
            }

        if op.action in {"none", "invalid"}:
            return no_op(op.reason or ("Malformed Integrate operation" if op.action == "invalid" else ""))
        if not op.title or not slugify_english_title(op.title):
            return no_op("Operation title must be an English knowledge title")
        if op.okf_type not in TYPE_REQUIRED_SECTIONS:
            return no_op("Unknown OKF type")
        if op.action == "create" and not op.justification.strip():
            op.justification = op.reason.strip() or "Gatekeeper approved new knowledge node"
        if op.action == "update" and not op.target_id:
            return no_op("Update operation requires target_id")
        if op.action not in {"create", "update"}:
            return no_op("Unsupported operation")

        target: Node | None = None
        sources = [transcript.id]
        if op.action == "update":
            if op.target_id not in allowed_target_ids:
                return no_op(f"Update target was not retrieved: {op.target_id}")
            with self._commit_lock:
                target = self._load_node_or_none(op.target_id or "")
            if target is None or target.metadata.type is not NodeType.PERSISTENT or target.metadata.status is not NodeStatus.ACTIVE:
                return no_op(f"Unknown or inactive update target: {op.target_id}")
            sources = list(dict.fromkeys([*target.metadata.sources, transcript.id]))
        final_content = _with_provenance(op.content, sources, op.evidence)
        final_content, repair_error = self._repair_takeaway_only(op, final_content, sources, run)
        if repair_error:
            warnings = validate_okf_node(
                node_type="persistent",
                content=final_content,
                okf_type=op.okf_type,
                sources=sources,
                title=op.title,
            )
            error = SynapseServiceError(
                "INVALID_OKF_FINAL_BODY",
                repair_error,
                status_code=422,
                details={"warnings": [finding.as_dict() for finding in warnings], "final_content": final_content},
            )
            with self._commit_lock:
                return self._quarantine(
                    transcript, op, repair_error, run, error=error, content=final_content, target_id=op.target_id,
                    verified=verified_flag, evidence_sources=evidence_sources,
                )
        with self._commit_lock:
            try:
                if op.action == "update":
                    payload = self._commit_call(
                        run,
                        lambda: self.service._merge_update_node(
                            target_id=op.target_id or "",
                            content=final_content,
                            title=op.title,
                            sources=[transcript.id],
                            verified=verified_flag,
                            route="distiller",
                            okf_type=op.okf_type,
                            validated_title=op.title,
                        ),
                    )
                    node = self._load_node(op.target_id or "")
                    action = str(payload.get("action", "updated"))
                else:
                    stable_id = self._stable_create_id(transcript, op)
                    existing = self._load_node_or_none(stable_id)
                    if existing is not None:
                        if existing.metadata.sources and transcript.id in existing.metadata.sources:
                            return {
                                "action": "unchanged", "node_id": existing.id, "title": existing.title,
                                "target_id": existing.id, "target_title": existing.title,
                                "verified": verified_flag, "evidence_sources": evidence_sources,
                            }
                        return no_op("Deterministic create ID collides with unrelated knowledge")
                    payload = self._commit_call(
                        run,
                        lambda: self.service.integrate_knowledge(
                            title=op.title,
                            content=final_content,
                            node_type=NodeType.PERSISTENT,
                            action="create",
                            okf_type=op.okf_type,
                            okf_version=1,
                            sources=sources,
                            project=transcript.metadata.project,
                            node_id=stable_id,
                            verified=verified_flag,
                            validate_okf=True,
                        ),
                    )
                    node = self._load_node(stable_id)
                    action = "created"
                    self.service._record_route_write_event(
                        node_id=node.id,
                        node_type=node.metadata.type.value,
                        action="created",
                        route="distiller",
                    )
            except SynapseServiceError as exc:
                if exc.code == "INVALID_OKF_FINAL_BODY":
                    return self._quarantine(
                        transcript, op, exc.message, run, error=exc, content=final_content,
                        target_id=op.target_id, verified=verified_flag, evidence_sources=evidence_sources,
                    )
                raise
        return {
            "action": action,
            "node_id": node.id,
            "title": node.title,
            "target_id": op.target_id or node.id,
            "target_title": node.title,
            "verified": verified_flag,
            "evidence_sources": evidence_sources,
        }

    def _quarantine(
        self,
        transcript: Node,
        op: EpisodeOp,
        reason: str,
        run: EpisodeRun,
        *, 
        error: SynapseServiceError | None = None,
        content: str | None = None,
        target_id: str | None = None,
        verified: bool = False,
        evidence_sources: list[str] | None = None,
    ) -> dict[str, Any]:
        error = error or SynapseServiceError("INVALID_EPISODE_OPERATION", reason, status_code=422)
        sources = [transcript.id]
        body = content if content is not None else op.content
        item = _item_from_op(op, body, sources, op.evidence, verified=verified)
        self._distiller_helpers._quarantine_invalid_item(
            transcript,
            item,
            {"sources": sources},
            error,
            context={
                "intended_action": op.action,
                "target_node_ids": [target_id or op.target_id] if (target_id or op.target_id) else [],
                "candidate_content": body,
                "merge_update_target": target_id or op.target_id,
                "failure_reason": reason,
            },
        )
        return {
            "action": "quarantined",
            "title": op.title,
            "target_id": target_id or op.target_id,
            "reason": reason,
            "verified": verified,
            "evidence_sources": evidence_sources or [],
        }

    def _stable_create_id(self, transcript: Node, op: EpisodeOp) -> str:
        return generate_node_id(op.title, transcript.metadata.created_at)

    def _load_node_or_none(self, node_id: str) -> Node | None:
        with SQLiteNodeStore(self.runtime_paths.base / "synapse.db", embedding_dimension=self.config.embedding.dimension or 0) as store:
            return store.get_node(node_id)

    def _load_node(self, node_id: str) -> Node:
        node = self._load_node_or_none(node_id)
        if node is None:
            raise SynapseServiceError("NODE_NOT_FOUND", f"Node does not exist: {node_id}", status_code=404)
        return node


def build_episode_chunks(turns: list[TranscriptTurn], max_chars: int = 400_000) -> list[tuple[str, list[str]]]:
    """Group complete numbered turns; no chunk boundary can split a turn."""
    chunks: list[tuple[str, list[str]]] = []
    blocks: list[str] = []
    ids: list[str] = []
    size = 0
    for turn in turns:
        block = _render_turn(turn)
        added = len(block) + (2 if blocks else 0)
        if blocks and size + added > max_chars:
            chunks.append(("\n\n".join(blocks), ids))
            blocks, ids, size = [], [], 0
            added = len(block)
        blocks.append(block)
        ids.append(turn.id)
        size += added
    if blocks:
        chunks.append(("\n\n".join(blocks), ids))
    return chunks


def _render_turn(turn: TranscriptTurn) -> str:
    lines = [f"## {turn.id} {turn.role}"]
    if turn.body:
        lines.append(turn.body)
    lines.extend(turn.tool_sections)
    return "\n".join(lines)

def _turn_verification_text(turn: TranscriptTurn) -> str:
    parts = [turn.body] if turn.body else []
    parts.extend(turn.tool_sections)
    return "\n".join(parts)

def _turn_evidence_excerpt(turn: TranscriptTurn, limit: int = 1500) -> str:
    if not turn.tool_sections:
        return turn.body[:limit]
    tool_chars = sum(len(section) for section in turn.tool_sections) + len(turn.tool_sections) - 1
    body_budget = min(len(turn.body), limit - min(tool_chars, limit // 2))
    body_excerpt = turn.body[:body_budget]
    parts = [body_excerpt] if body_excerpt else []
    remaining = limit - len(body_excerpt)
    for index, section in enumerate(turn.tool_sections):
        separators = 1 if parts else 0
        sections_left = len(turn.tool_sections) - index
        share = max(0, (remaining - separators * (sections_left - 1)) // sections_left)
        excerpt = section[:share]
        if excerpt:
            parts.append(excerpt)
            remaining -= separators + len(excerpt)
    return "\n".join(parts)[:limit]



_NOISE_TOOL_NAMES = {"todo", "vibe_wait", "vibe_list", "vibe_kill", "vibe_send", "wait"}
_NOISE_HUB_OPS = {"wait", "list", "status", "poll", "kill"}
_TOOL_HEADER_RE = re.compile(r"^### Tool T\d+\.\d+\s+(.+?)\s*$", re.IGNORECASE)


def _is_noise_tool_section(section: str) -> bool:
    header = section.splitlines()[0] if section else ""
    match = _TOOL_HEADER_RE.match(header)
    if match is None:
        return False
    tool_name = match.group(1).strip().strip("`").split()[0].casefold()
    if tool_name in _NOISE_TOOL_NAMES:
        return True
    if tool_name != "hub":
        return False
    call_match = re.search(r"(?m)^\s*call:\s*", section)
    if call_match is None:
        return False
    decoder = json.JSONDecoder()
    try:
        call, _end = decoder.raw_decode(section[call_match.end():].lstrip())
    except json.JSONDecodeError:
        return False
    return isinstance(call, Mapping) and str(call.get("op") or "").casefold() in _NOISE_HUB_OPS


def _dehydrate_turn(turn: TranscriptTurn) -> TranscriptTurn | None:
    tools = [section for section in turn.tool_sections if not _is_noise_tool_section(section)]
    if turn.role == "Assistant" and not turn.body and not tools:
        return None
    return TranscriptTurn(number=turn.number, role=turn.role, body=turn.body, tool_sections=tools)


def _render_global_user_spine(turns: list[TranscriptTurn]) -> str:
    lines = []
    for turn in turns:
        if turn.role != "User":
            continue
        body = turn.body
        excerpt = body if len(body) <= 350 else body[:347] + "..."
        lines.append(f"{turn.id}: {excerpt}")
    return "\n".join(lines) if lines else "(No User turns.)"


def _parse_lessons(payload: Any) -> list[Lesson]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("lessons"), list):
        raise ValueError("Episode stage response must contain a lessons array")
    allowed_types = set(TYPE_REQUIRED_SECTIONS)
    result: list[Lesson] = []
    for row in payload["lessons"]:
        if not isinstance(row, Mapping):
            continue
        topic = str(row.get("topic") or "").strip()
        claim = str(row.get("claim") or "").strip()
        if not topic or not claim:
            continue
        raw_type = str(row.get("okf_type") or "fact").strip().casefold()
        okf_type = raw_type if raw_type in allowed_types else "fact"
        raw_turns = row.get("evidence_turns")
        evidence_turns = (
            list(dict.fromkeys(str(turn).strip() for turn in raw_turns if str(turn).strip()))
            if isinstance(raw_turns, list)
            else []
        )
        result.append(Lesson(topic=topic, claim=claim, evidence_turns=evidence_turns, okf_type=okf_type))
    return result
def _parse_ops(payload: Any, lesson_count: int) -> list[EpisodeOp]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("ops"), list):
        raise ValueError("Integrate response must contain an ops array")
    result: list[EpisodeOp] = []
    for op_index, row in enumerate(payload["ops"]):
        if not isinstance(row, Mapping):
            result.append(EpisodeOp(action="invalid", reason=f"Integrate operation {op_index} must be a JSON object"))
            continue
        action = str(row.get("action") or "").strip().lower()
        raw_lesson_index = row.get("lesson_index")
        lesson_index = int(raw_lesson_index) if isinstance(raw_lesson_index, int) and not isinstance(raw_lesson_index, bool) else None
        if lesson_index is not None and not 0 <= lesson_index < lesson_count:
            raise ValueError("Integrate operation lesson_index is outside this batch")
        if action == "none":
            result.append(EpisodeOp(action="none", reason=str(row.get("reason") or ""), lesson_index=lesson_index))
            continue
        evidence: list[EvidenceEntry] = []
        evidence_raw = row.get("evidence")
        if isinstance(evidence_raw, list):
            for item in evidence_raw:
                if not isinstance(item, Mapping):
                    continue
                turn = str(item.get("turn") or "").strip()
                quote = str(item.get("quote") or "").strip()
                if turn and quote and len(quote) <= EVIDENCE_QUOTE_MAX_CHARS:
                    evidence.append(EvidenceEntry(turn=turn, quote=quote))
        verified = row.get("verified") if isinstance(row.get("verified"), bool) else None
        reason = str(row.get("reason") or "").strip()
        justification = str(row.get("justification") or "").strip()
        if action == "create" and not justification:
            justification = reason or "Gatekeeper approved new knowledge node"
        result.append(
            EpisodeOp(
                action=action,
                title=str(row.get("title") or "").strip(),
                okf_type=str(row.get("okf_type") or "").strip(),
                content=str(row.get("content") or "").strip(),
                target_id=str(row.get("target_id") or "").strip() or None,
                justification=justification,
                evidence=evidence,
                verified=verified,
                reason=reason,
                lesson_index=lesson_index,
            )
        )
    return result



def _takeaway_section(content: str) -> str:
    match = re.search(r"(?ms)^## Takeaway\s*\n(.*?)(?=^## |\Z)", content)
    return match.group(1).strip() if match else ""


def _valid_takeaway(value: str, *, hard_limit: int) -> bool:
    nonempty = [line.strip() for line in value.splitlines() if line.strip()]
    return len(nonempty) == 1 and len(nonempty[0]) <= hard_limit


def _replace_takeaway(content: str, takeaway: str) -> str:
    replacement = f"## Takeaway\n\n{takeaway.strip()}\n\n"
    return re.sub(r"(?ms)^## Takeaway\s*\n.*?(?=^## |\Z)", replacement, content, count=1)
def _takeaway(content: str) -> str:
    match = re.search(r"(?ms)^## Takeaway\s*\n(.*?)(?=^## |\Z)", content)
    return match.group(1).strip().splitlines()[0] if match and match.group(1).strip() else ""


def _compact_node(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "title": node.title,
        "okf_type": node.metadata.okf_type,
        "takeaway": _takeaway(node.content),
    }


def _upsert_prior_node(prior: list[dict[str, Any]], node: Node) -> list[dict[str, Any]]:
    compact = _compact_node(node)
    result = [item for item in prior if item.get("id") != node.id]
    result.append(compact)
    return result


def _with_provenance(content: str, sources: list[str], evidence: list[EvidenceEntry]) -> str:
    sections = re.compile(r"(?ms)^## (?:Sources|Evidence)\s*\n.*?(?=^## |\Z)")
    body = sections.sub("", content).strip()
    body += "\n\n## Sources\n\n" + "\n".join(f"- [[{source}]]" for source in sources)
    if evidence:
        body += "\n\n" + render_evidence_section(evidence).strip()
    return body.strip() + "\n"


def _item_from_op(
    op: EpisodeOp,
    content: str,
    sources: list[str],
    evidence: list[EvidenceEntry],
    *,
    verified: bool,
) -> OkfItem:
    takeaway = _takeaway(content)
    sections: dict[str, str] = {}
    for name in TYPE_REQUIRED_SECTIONS.get(op.okf_type, ()):
        match = re.search(rf"(?ms)^## {re.escape(name)}\s*\n(.*?)(?=^## |\Z)", content)
        sections[name] = match.group(1).strip() if match else ""
    return OkfItem(
        okf_type=op.okf_type,
        title=op.title,
        takeaway=takeaway,
        sections=sections,
        sources=sources,
        evidence=evidence,
        verified=verified,
    )
