"""Episode-scoped summarization and explicit OKF integration for pilot/backfill."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from synapse.lifecycle.distiller import Distiller, content_sha256
from synapse.models import Node, NodeStatus, NodeType, generate_node_id, slugify_english_title
from synapse.okf import (
    EvidenceEntry,
    TYPE_REQUIRED_SECTIONS,
    OkfItem,
    render_evidence_section,
    verify_evidence,
)
from synapse.okf.episode import (
    render_episode_integrate_prompt,
    render_episode_lesson_reduce_prompt,
    render_episode_map_prompt,
    render_episode_summary_prompt,
)
from synapse.okf.transcript_format import TranscriptTurn, parse_transcript
from synapse.okf.transcripts import is_session_transcript
from synapse.retrieval import RetrievalPipeline
from synapse.server.service import SynapseServiceError, SynapseServerService
from synapse.storage import SQLiteNodeStore
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

    def as_dict(self) -> dict[str, Any]:
        return {"topic": self.topic, "claim": self.claim, "evidence_turns": self.evidence_turns}


@dataclass(slots=True)
class EpisodeOp:
    action: str
    title: str = ""
    okf_type: str = ""
    content: str = ""
    target_id: str | None = None
    justification: str = ""
    evidence: list[EvidenceEntry] = field(default_factory=list)
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
    ) -> None:
        self.config = config
        self.settings = config.distiller.episode
        self.runtime_paths = runtime_paths
        self.sampling_client = sampling_client
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

    @staticmethod
    def make_sampling_client(config: SynapseConfig) -> Any:
        from synapse.server.decider import LocalLLMDecider

        endpoint = config.distiller.episode.endpoint
        settings = config.decider.model_copy(
            update={
                "provider": "local_llm",
                "endpoints": [endpoint],
                "timeout_seconds": config.distiller.episode.timeout_seconds,
                "max_tokens": config.distiller.episode.max_tokens,
                "temperature": config.distiller.episode.temperature,
                "endpoint_max_concurrency": 4,
            }
        )
        return LocalLLMDecider(settings)

    def close(self) -> None:
        self._distiller_helpers.close()

    def run(self, session: str) -> EpisodeRun:
        transcript = self._resolve_transcript(session)
        if not is_session_transcript(transcript):
            raise ValueError(f"Not a session transcript: {transcript.id}")
        parsed = parse_transcript(transcript.content)
        watermark = transcript.metadata.distilled_turn or 0
        pending = parsed.turns_after(watermark)
        if not pending:
            episode_id = self._episode_id(transcript, watermark, watermark, transcript.content)
            return EpisodeRun(transcript.id, transcript.metadata.session_key, episode_id, 0, 0, 0, stamped=True)

        source_text = "\n\n".join(_render_turn(turn) for turn in pending)
        chunks = build_episode_chunks(pending, self.settings.chunk_chars)
        run = EpisodeRun(
            transcript_id=transcript.id,
            session_key=transcript.metadata.session_key,
            episode_id=self._episode_id(transcript, pending[0].number, pending[-1].number, source_text),
            chars=len(source_text),
            turns=len(pending),
            chunks=len(chunks),
        )
        lesson_rows: list[Lesson] = []
        if len(chunks) == 1:
            text, ids = chunks[0]
            payload = self._call(
                run,
                "summarize",
                render_episode_summary_prompt(turns=text, turn_ids=ids),
            )
            lesson_rows = _parse_lessons(payload, set(ids))
        else:
            mapped: list[Lesson] = []
            for index, (text, ids) in enumerate(chunks, 1):
                payload = self._call(
                    run,
                    "map",
                    render_episode_map_prompt(
                        turns=text,
                        turn_ids=ids,
                        chunk_index=index,
                        chunk_count=len(chunks),
                    ),
                )
                mapped.extend(_parse_lessons(payload, set(ids)))
            reduced = self._call(
                run,
                "lesson_reduce",
                render_episode_lesson_reduce_prompt(chunk_lessons=[lesson.as_dict() for lesson in mapped]),
            )
            all_turn_ids = {turn.id for turn in pending}
            lesson_rows = _parse_lessons(reduced, all_turn_ids)

        run.lessons = [lesson.as_dict() for lesson in lesson_rows]
        try:
            self._integrate(transcript, pending, lesson_rows, run)
        except Exception:
            # Never stamp partial model/commit failures; already-applied creates use deterministic IDs.
            raise
        self._distiller_helpers._stamp_transcript(
            transcript,
            [str(op.get("node_id")) for op in run.ops if op.get("node_id")],
            distilled_turn=pending[-1].number,
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
        metrics = run.stages.setdefault(stage, StageMetrics())
        started = time.perf_counter()
        call_context = getattr(self.sampling_client, "call_context", None)
        context = None
        try:
            if callable(call_context):
                with call_context(
                    transcript_id=run.transcript_id,
                    stage=f"episode_{stage}",
                    timeout_seconds=self.settings.timeout_seconds,
                ) as context:
                    payload = self.sampling_client.sample_json(
                        prompt=prompt,
                        system_prompt="",
                        max_tokens=self.settings.max_tokens,
                    )
            else:
                payload = self.sampling_client.sample_json(
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
        call_context = getattr(self.sampling_client, "call_context", None)
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
    ) -> None:
        if not lessons:
            return
        helper_token = self.service.push_sampling_client(self.sampling_client)
        try:
            prior_nodes = self._same_session_nodes(transcript)
            with RetrievalPipeline(self.config, runtime_paths=self.runtime_paths) as retrieval:
                for offset in range(0, len(lessons), self.settings.integrate_lessons_per_call):
                    batch = lessons[offset : offset + self.settings.integrate_lessons_per_call]
                    candidates = self._retrieve_for_lessons(retrieval, batch, run)
                    evidence_context = self._evidence_context(turns, batch)
                    payload = self._call(
                        run,
                        "integrate",
                        render_episode_integrate_prompt(
                            lessons=[lesson.as_dict() for lesson in batch],
                            same_session_nodes=prior_nodes,
                            retrieved_nodes=candidates,
                            evidence_context=evidence_context,
                        ),
                    )
                    allowed_target_ids = {node["id"] for node in candidates}
                    allowed_target_ids.update(node["id"] for node in prior_nodes)
                    ops = _parse_ops(payload, len(batch))
                    batch_evidence_turns = {turn_id for lesson in batch for turn_id in lesson.evidence_turns}
                    for op in ops:
                        result = self._apply_op(
                            transcript, turns, op, run,
                            allowed_target_ids=allowed_target_ids,
                            allowed_evidence_turns=batch_evidence_turns,
                        )
                        run.ops.append(result)
                        if result.get("node_id"):
                            node = self._load_node(str(result["node_id"]))
                            prior_nodes = _upsert_prior_node(prior_nodes, node)
                            if result.get("action") in {"created", "updated"}:
                                run.node_bodies.append({"node_id": node.id, "title": node.title, "content": node.content})
        finally:
            self.service.reset_sampling_client(helper_token)

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
    def _evidence_context(turns: list[TranscriptTurn], lessons: list[Lesson]) -> dict[str, str]:
        eligible_ids = {turn_id for lesson in lessons for turn_id in lesson.evidence_turns}
        result: dict[str, str] = {}
        for turn in turns:
            if turn.id not in eligible_ids:
                continue
            if turn.role == "User":
                text, question_only = Distiller._user_turn_evidence(turn.body)
                if question_only:
                    continue
            else:
                text = turn.eligible_evidence_text()
            if text:
                result[turn.id] = text
        return result

    def _apply_op(
        self,
        transcript: Node,
        turns: list[TranscriptTurn],
        op: EpisodeOp,
        run: EpisodeRun,
        *,
        allowed_target_ids: set[str],
        allowed_evidence_turns: set[str],
    ) -> dict[str, Any]:
        if op.action == "none":
            return {"action": "none", "reason": op.reason, "lesson_index": op.lesson_index}
        if not op.title or not slugify_english_title(op.title):
            return self._quarantine(transcript, op, "Operation title must be an English knowledge title", run)
        if op.okf_type not in TYPE_REQUIRED_SECTIONS:
            return self._quarantine(transcript, op, "Unknown OKF type", run)
        if op.action == "create" and not op.justification.strip():
            return self._quarantine(transcript, op, "Create operation requires a justification", run)
        if op.action == "update" and not op.target_id:
            return self._quarantine(transcript, op, "Update operation requires target_id", run)
        if op.action not in {"create", "update"}:
            return self._quarantine(transcript, op, "Unsupported operation", run)

        evidence = [entry for entry in op.evidence if entry.turn in allowed_evidence_turns]
        evidence_turns: dict[str, str] = {}
        question_only: set[str] = set()
        for turn in turns:
            if turn.id not in allowed_evidence_turns:
                continue
            if turn.role == "User":
                text, is_question = Distiller._user_turn_evidence(turn.body)
                if is_question:
                    question_only.add(turn.id)
            else:
                text = turn.eligible_evidence_text(okf_type=op.okf_type)
            evidence_turns[turn.id] = text
        verified, _unverified = verify_evidence(evidence, turns=evidence_turns, question_only_turns=question_only)
        if not verified:
            return self._quarantine(transcript, op, "Operation has no verifiable episode evidence", run)

        target: Node | None = None
        sources = [transcript.id]
        if op.action == "update":
            if op.target_id not in allowed_target_ids:
                return self._quarantine(transcript, op, f"Update target was not retrieved: {op.target_id}", run)
            target = self._load_node_or_none(op.target_id or "")
            if target is None or target.metadata.type is not NodeType.PERSISTENT or target.metadata.status is not NodeStatus.ACTIVE:
                return self._quarantine(transcript, op, f"Unknown or inactive update target: {op.target_id}", run)
            sources = list(dict.fromkeys([*target.metadata.sources, transcript.id]))
        final_content = _with_provenance(op.content, sources, verified)
        try:
            if op.action == "update":
                payload = self._commit_call(
                    run,
                    lambda: self.service._merge_update_node(
                        target_id=op.target_id or "",
                        content=final_content,
                        title=op.title,
                        sources=[transcript.id],
                        verified=True,
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
                        return {"action": "unchanged", "node_id": existing.id, "title": existing.title, "target_id": existing.id, "target_title": existing.title}
                    return self._quarantine(transcript, op, "Deterministic create ID collides with unrelated knowledge", run)
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
                        verified=True,
                        validate_okf=True,
                    ),
                )
                node = self._load_node(stable_id)
                action = "created"
        except SynapseServiceError as exc:
            if exc.code == "INVALID_OKF_FINAL_BODY":
                return self._quarantine(transcript, op, exc.message, run, error=exc, content=final_content, target_id=op.target_id)
            raise
        return {"action": action, "node_id": node.id, "title": node.title, "target_id": op.target_id or node.id, "target_title": node.title}

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
    ) -> dict[str, Any]:
        error = error or SynapseServiceError("INVALID_EPISODE_OPERATION", reason, status_code=422)
        sources = [transcript.id]
        body = content if content is not None else op.content
        item = _item_from_op(op, body, sources, op.evidence)
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
            },
        )
        return {"action": "quarantined", "title": op.title, "target_id": target_id or op.target_id, "reason": reason}

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


def build_episode_chunks(turns: list[TranscriptTurn], max_chars: int = 100_000) -> list[tuple[str, list[str]]]:
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


def _parse_lessons(payload: Any, eligible_turns: set[str]) -> list[Lesson]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("lessons"), list):
        raise ValueError("Episode stage response must contain a lessons array")
    result: list[Lesson] = []
    for row in payload["lessons"]:
        if not isinstance(row, Mapping):
            raise ValueError("Lesson must be a JSON object")
        topic = str(row.get("topic") or "").strip()
        claim = str(row.get("claim") or "").strip()
        raw_turns = row.get("evidence_turns")
        if not topic or not claim or not isinstance(raw_turns, list):
            raise ValueError("Lesson requires topic, claim, and evidence_turns")
        turn_ids = list(dict.fromkeys(str(turn).strip() for turn in raw_turns if str(turn).strip()))
        if any(turn not in eligible_turns for turn in turn_ids):
            raise ValueError("Lesson cites a turn outside its supplied transcript range")
        if not turn_ids:
            raise ValueError("Durable lesson requires at least one evidence turn")
        result.append(Lesson(topic, claim, turn_ids))
    return result


def _parse_ops(payload: Any, lesson_count: int) -> list[EpisodeOp]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("ops"), list):
        raise ValueError("Integrate response must contain an ops array")
    result: list[EpisodeOp] = []
    for row in payload["ops"]:
        if not isinstance(row, Mapping):
            raise ValueError("Integrate operation must be an object")
        action = str(row.get("action") or "").strip().lower()
        index = row.get("lesson_index")
        lesson_index = int(index) if isinstance(index, int) and not isinstance(index, bool) else None
        if lesson_index is not None and not 0 <= lesson_index < lesson_count:
            raise ValueError("Integrate operation lesson_index is outside this batch")
        if action == "none":
            result.append(EpisodeOp(action="none", reason=str(row.get("reason") or ""), lesson_index=lesson_index))
            continue
        evidence_raw = row.get("evidence")
        evidence: list[EvidenceEntry] = []
        if isinstance(evidence_raw, list):
            for item in evidence_raw:
                if isinstance(item, Mapping) and item.get("turn") and item.get("quote"):
                    evidence.append(EvidenceEntry(turn=str(item["turn"]).strip(), quote=str(item["quote"]).strip()))
        result.append(
            EpisodeOp(
                action=action,
                title=str(row.get("title") or "").strip(),
                okf_type=str(row.get("okf_type") or "").strip(),
                content=str(row.get("content") or "").strip(),
                target_id=str(row.get("target_id") or "").strip() or None,
                justification=str(row.get("justification") or "").strip(),
                evidence=evidence,
                reason=str(row.get("reason") or "").strip(),
                lesson_index=lesson_index,
            )
        )
    return result


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


def _item_from_op(op: EpisodeOp, content: str, sources: list[str], evidence: list[EvidenceEntry]) -> OkfItem:
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
        verified=bool(evidence),
    )
