"""Server-side session distiller: turns transcript nodes into typed OKF knowledge.

Design: server-side session distiller (spec: docs/okf.md).

The sweep selects idle session transcripts whose content has changed since the
last distillation, extracts durable knowledge items through the configured
LLM decider endpoint, writes each item through the normal write_memory decider
path (so create/complement/supersede run against the real corpus), stamps
distillation state back onto the transcript, and archives fully-distilled
transcripts past the retention window. A CLI mode supports backfill over
legacy session-summary groups with supersede-downgrade safety.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
import threading
from dataclasses import dataclass, field
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from synapse.config import SynapseConfig
from synapse.models import Node, NodeStatus, NodeType
from synapse.okf import (
    TYPE_REQUIRED_SECTIONS,
    OkfItem,
    normalize_sources,
    parse_okf_items,
    parse_title_repair,
    render_distiller_prompt,
    render_evidence_section,
    render_merge_update_prompt,
    render_reduce_prompt,
    render_title_repair_prompt,
    verify_evidence,
)
from synapse.okf.transcript_format import ParsedTranscript, parse_transcript
from synapse.okf.transcripts import is_session_transcript
from synapse.models import slugify_english_title
from synapse.server.service import SynapseServiceError, SynapseServerService
from synapse.storage import DistillerRunMetrics, SQLiteNodeStore, archive_node_path
from synapse.storage.markdown import write_node_file
from synapse.utils.runtime import RuntimePaths

LOGGER = logging.getLogger("synapse.distiller")
_QUESTION_ONLY_PREFIX = re.compile(
    r"^(?:what|why|how|when|where|who|which|whose|whom|can|could|would|will|do|does|did|is|are|was|were|should|may|might|must|have|has|had)\b|^(?:请问|为什么|如何|怎么|是否|能否|是不是|什么|什么时候|哪里|谁)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class DistillerReport:
    """Structured summary of one distiller run (sweep or CLI backfill)."""

    started_at: str
    completed_at: str
    duration_ms: int = 0
    transcripts_scanned: int = 0
    transcripts_distilled: int = 0
    transcripts_skipped: int = 0
    items_extracted: int = 0
    items_invalid: int = 0
    items_sources_dropped: int = 0
    reduced_items: int = 0
    reduce_failures: int = 0
    reduce_groups_llm: int = 0
    guard_fallback_creates: int = 0
    title_repairs: int = 0
    title_repair_failures: int = 0
    decider_creates: int = 0
    decider_complements: int = 0
    decider_supersedes: int = 0
    decider_downgrades: int = 0
    # Near-duplicate gate: items merged INTO an existing node (same id,
    # in-place body update) instead of written as a new sibling.
    merged_updates: int = 0
    llm_failures: int = 0
    segments_total: int = 0
    endpoint_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    actions_by_endpoint: dict[str, dict[str, int]] = field(default_factory=dict)
    archived_transcripts: int = 0
    # Backfill-safety record: supersede decisions downgraded to complement.
    downgraded: list[dict[str, str]] = field(default_factory=list)
    # Per-transcript detail for CLI reports (kept out of the metrics row).
    details: list[dict[str, Any]] = field(default_factory=list)

    def metrics(self) -> DistillerRunMetrics:
        return DistillerRunMetrics(
            started_at=self.started_at,
            completed_at=self.completed_at,
            duration_ms=self.duration_ms,
            transcripts_scanned=self.transcripts_scanned,
            transcripts_distilled=self.transcripts_distilled,
            transcripts_skipped=self.transcripts_skipped,
            items_extracted=self.items_extracted,
            items_invalid=self.items_invalid,
            decider_creates=self.decider_creates,
            decider_complements=self.decider_complements,
            decider_supersedes=self.decider_supersedes,
            decider_downgrades=self.decider_downgrades,
            llm_failures=self.llm_failures,
            archived_transcripts=self.archived_transcripts,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "transcripts_scanned": self.transcripts_scanned,
            "transcripts_distilled": self.transcripts_distilled,
            "transcripts_skipped": self.transcripts_skipped,
            "items_extracted": self.items_extracted,
            "items_invalid": self.items_invalid,
            "items_sources_dropped": self.items_sources_dropped,
            "reduced_items": self.reduced_items,
            "reduce_failures": self.reduce_failures,
            "guard_fallback_creates": self.guard_fallback_creates,
            "title_repairs": self.title_repairs,
            "title_repair_failures": self.title_repair_failures,
            "decider": {
                "create": self.decider_creates,
                "complement": self.decider_complements,
                "supersede": self.decider_supersedes,
                "downgraded_to_complement": self.decider_downgrades,
            },
            "llm_failures": self.llm_failures,
            "segments_total": self.segments_total,
            "endpoint_calls": self.endpoint_calls,
            "archived_transcripts": self.archived_transcripts,
            "downgraded": self.downgraded,
        }


# Shared predicates live in synapse.okf.transcripts (single source of truth,
# used by the distiller, the service search filter, and the markdown writer).
from synapse.okf.transcripts import (  # noqa: F401  (re-exported for callers/tests)
    content_sha256,
    is_distilled_current,
    is_represented,
    is_session_transcript,
)

# Backwards-compatible alias: pre-A1 name meant "distilled at current revision".
is_fully_distilled = is_distilled_current


def _build_turn_segments(
    parsed: ParsedTranscript,
    *,
    watermark: int,
    max_chars: int,
) -> list[tuple[str, str]]:
    """Split parsed turns into segments of at most ``max_chars`` chars.

    The first segment includes the watermark turn (if any) as verbatim context
    so extraction sees continuity; its start turn is the first NEW turn.
    An oversized single turn is split by characters — every slice is kept,
    nothing is truncated away. Returns ``(segment_text, start_turn_id)``
    pairs in order.
    """

    def turn_block(turn) -> str:  # TranscriptTurn
        lines = [f"## {turn.id} {turn.role}"]
        if turn.body:
            lines.append(turn.body)
        lines.extend(turn.tool_sections)
        return "\n".join(lines)

    pending = [turn for turn in parsed.turns if turn.number > watermark]
    if not pending:
        return []

    context_turn = parsed.turn(watermark)
    context_block = turn_block(context_turn) if context_turn is not None else ""

    segments: list[tuple[str, str]] = []
    current_chunks: list[str] = []
    current_len = 0
    current_start = pending[0].id

    def flush() -> None:
        nonlocal current_chunks, current_len, current_start
        if current_chunks:
            segments.append(("\n\n".join(current_chunks), current_start))
            current_chunks = []
            current_len = 0
            current_start = pending[0].id if not segments else current_start

    for index, turn in enumerate(pending):
        block = turn_block(turn)
        # Context only rides along on the very first segment.
        effective = f"{context_block}\n\n{block}" if index == 0 and context_block else block
        if current_chunks and current_len + len(effective) > max_chars:
            flush()
            # After a flush the context no longer applies.
            effective = block
        if len(effective) > max_chars:
            # Oversized single turn: hard-split by chars, all slices kept.
            slices = [effective[i : i + max_chars] for i in range(0, len(effective), max_chars)]
            for slice_index, slice_text in enumerate(slices):
                if current_chunks:
                    flush()
                start_id = turn.id if slice_index == 0 else f"{turn.id}#{slice_index + 1}"
                segments.append((slice_text, start_id))
            current_start = turn.id
            continue
        if not current_chunks:
            current_start = turn.id
        current_chunks.append(effective)
        current_len += len(effective)

    flush()
    return segments


def _extract_section(content: str, name: str) -> str | None:
    """Body of the ``## <name>`` section in *content*, or None when absent."""

    pattern = re.compile(rf"(?ms)^## {re.escape(name)}\s*\n(.*?)(?=^## |\Z)")
    match = pattern.search(content)
    return match.group(1).strip() if match else None


def _first_section_line(content: str, name: str) -> str:
    body = _extract_section(content, name) or ""
    return body.splitlines()[0].strip() if body else ""


class Distiller:
    """Executes one distillation sweep over the node store."""

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths,
        sampling_client: Any | None = None,
        service: SynapseServerService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths
        self.settings = config.distiller
        self._sampling_client = sampling_client
        self._service = service
        self._logger = logger or LOGGER
        self._store: SQLiteNodeStore | None = None
        self._store_local = threading.local()
        self._owner_thread_id = threading.get_ident()
        self._llm_executor: ThreadPoolExecutor | None = None
        self._llm_executor_lock = threading.Lock()
        self._write_commit_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Store / service access
    # ------------------------------------------------------------------

    def _get_store(self) -> SQLiteNodeStore:
        store = getattr(self._store_local, "store", None)
        if store is None:
            store = SQLiteNodeStore(
                self.runtime_paths.base / "synapse.db",
                embedding_dimension=self.config.embedding.dimension or 0,
            )
            self._store_local.store = store
            if threading.get_ident() == self._owner_thread_id:
                self._store = store
        return store

    def _close_thread_store(self) -> None:
        store = getattr(self._store_local, "store", None)
        if store is not None:
            store.close()
            self._store_local.store = None
            if threading.get_ident() == self._owner_thread_id:
                self._store = None

    def close(self) -> None:
        if self._llm_executor is not None:
            self._llm_executor.shutdown(wait=True)
            self._llm_executor = None
        self._close_thread_store()

    def _get_llm_executor(self) -> ThreadPoolExecutor:
        if self._llm_executor is None:
            with self._llm_executor_lock:
                if self._llm_executor is None:
                    workers = len(self.config.decider.endpoints) * self.config.decider.endpoint_max_concurrency
                    self._llm_executor = ThreadPoolExecutor(
                        max_workers=max(1, workers),
                        thread_name_prefix="synapse-distiller-llm",
                    )
        return self._llm_executor

    def _sampling_call_context(self, transcript: Node, stage: str):
        client = self._get_sampling_client()
        call_context = getattr(client, "call_context", None)
        if not callable(call_context):
            return nullcontext()
        timeout_seconds = (
            self.config.decider.timeout_seconds
            if stage in {"write_decider", "write_commit"}
            else self.settings.llm_timeout_seconds
        )
        try:
            return call_context(
                transcript_id=transcript.id,
                stage=stage,
                timeout_seconds=timeout_seconds,
            )
        except TypeError:
            return call_context(transcript_id=transcript.id, stage=stage)

    def _llm_calls_per_transcript(self) -> int:
        total = len(self.config.decider.endpoints) * self.config.decider.endpoint_max_concurrency
        return max(1, min(self.settings.max_concurrency, total // self.settings.transcript_concurrency))

    def _get_service(self) -> SynapseServerService:
        if self._service is None:
            self._service = SynapseServerService(
                self.config,
                runtime_paths=self.runtime_paths,
                logger=self._logger,
                sampling_client=self._get_sampling_client(),
            )
        return self._service

    def _get_sampling_client(self) -> Any:
        if self._sampling_client is None:
            from synapse.server.decider import LocalLLMDecider

            # Distiller-specific LLM timeout: distillation prompts (long zh
            # transcripts + reasoning models) routinely exceed the write-path
            # decider timeout, so the sweep reuses [decider] settings but with
            # its own [distiller] llm_timeout_seconds (default 300 s).
            settings = self.config.decider.model_copy(
                update={"timeout_seconds": self.config.distiller.llm_timeout_seconds}
            )
            self._sampling_client = LocalLLMDecider(settings)
        return self._sampling_client

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_transcripts(self, *, limit: int | None = None) -> list[Node]:
        """Idle, changed-since-distill session transcripts, oldest update first."""

        store = self._get_store()
        now = datetime.now(UTC)
        idle_cutoff = now - timedelta(minutes=self.settings.idle_minutes)
        candidates: list[Node] = []
        for node in store.list_nodes({"status": NodeStatus.ACTIVE, "type": NodeType.TRANSIENT}):
            if not is_session_transcript(node):
                continue
            updated_at = self._content_updated_at(node)
            if updated_at is None or updated_at > idle_cutoff:
                continue
            if is_distilled_current(node):
                continue
            candidates.append(node)
        candidates.sort(key=lambda n: self._content_updated_at(n) or now)
        if limit is not None:
            candidates = candidates[:limit]
        return candidates

    def _content_updated_at(self, node: Node) -> datetime | None:
        """Content mtime on disk (growing sessions update it); fall back to created_at."""

        path = self.runtime_paths.base / node.file_path
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(mtime, tz=UTC)

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    def run(self, *, limit: int | None = None, dry_run: bool = False, downgrade_supersede: bool | None = None) -> DistillerReport:
        started = datetime.now(UTC)
        report = DistillerReport(started_at=started.isoformat().replace("+00:00", "Z"), completed_at="")
        downgrade = self.settings.downgrade_supersede if downgrade_supersede is None else downgrade_supersede
        max_transcripts = self.settings.max_transcripts_per_run if limit is None else max(0, limit)
        self._get_service()
        self._get_llm_executor()

        seen: set[str] = set()
        in_flight: dict[Future[DistillerReport], Node] = {}
        with ThreadPoolExecutor(
            max_workers=self.settings.transcript_concurrency,
            thread_name_prefix="synapse-distiller-transcript",
        ) as pool:
            while in_flight or len(seen) < max_transcripts:
                free_slots = self.settings.transcript_concurrency - len(in_flight)
                if free_slots and len(seen) < max_transcripts:
                    candidates = [node for node in self.select_transcripts() if node.id not in seen]
                    for transcript in candidates[: min(free_slots, max_transcripts - len(seen))]:
                        seen.add(transcript.id)
                        report.transcripts_scanned += 1
                        future = pool.submit(
                            self._run_transcript_task,
                            transcript,
                            dry_run=dry_run,
                            downgrade_supersede=downgrade,
                        )
                        in_flight[future] = transcript
                if not in_flight:
                    break
                completed, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    in_flight.pop(future)
                    self._merge_transcript_report(report, future.result())

        if not dry_run:
            report.archived_transcripts = self._archive_expired(report)

        completed_at = datetime.now(UTC)
        report.completed_at = completed_at.isoformat().replace("+00:00", "Z")
        for endpoint_stats in report.endpoint_calls.values():
            endpoint_stats["avg_latency_ms"] = (
                endpoint_stats["latency_total_ms"] / endpoint_stats["calls"]
                if endpoint_stats["calls"]
                else 0.0
            )
            endpoint_stats["wall_time_ms"] = endpoint_stats["latency_total_ms"]
        report.duration_ms = int((completed_at - started).total_seconds() * 1000)
        if not dry_run:
            self._get_store().record_distiller_run(report.metrics())
        self._logger.info(
            "Distiller run complete",
            extra={
                "transcripts_scanned": report.transcripts_scanned,
                "transcripts_distilled": report.transcripts_distilled,
                "transcripts_skipped": report.transcripts_skipped,
                "segments": report.segments_total,
                "items_extracted": report.items_extracted,
                "decider_actions": {
                    "create": report.decider_creates,
                    "complement": report.decider_complements,
                    "supersede": report.decider_supersedes,
                    "updated": report.merged_updates,
                },
                "llm_failures": report.llm_failures,
                "archived_transcripts": report.archived_transcripts,
                "duration_ms": report.duration_ms,
                "endpoint_calls": report.endpoint_calls,
            },
        )
        return report

    @staticmethod
    def _is_final_okf_validation_error(exc: Exception) -> bool:
        if not isinstance(exc, SynapseServiceError):
            return False
        if exc.code == "INVALID_OKF_FINAL_BODY":
            return True
        execution_error = exc.details.get("execution_error")
        return isinstance(execution_error, Mapping) and execution_error.get("code") == "INVALID_OKF_FINAL_BODY"

    def _run_transcript_task(
        self,
        transcript: Node,
        *,
        dry_run: bool,
        downgrade_supersede: bool,
    ) -> DistillerReport:
        started = datetime.now(UTC)
        report = DistillerReport(started_at=started.isoformat().replace("+00:00", "Z"), completed_at="")
        try:
            detail = self._distill_transcript(
                transcript,
                report,
                dry_run=dry_run,
                downgrade_supersede=downgrade_supersede,
            )
            report.details.append(detail)
        except Exception as exc:  # noqa: BLE001 — one bad transcript must not stop the sweep
            report.llm_failures += 1
            report.transcripts_skipped += 1
            validation_skip = self._is_final_okf_validation_error(exc)
            call_metrics = getattr(self._get_sampling_client(), "call_metrics", None)
            self._logger.warning(
                "Distiller final OKF validation skipped write; transcript remains unstamped and retryable"
                if validation_skip
                else "Distill failed for transcript; backing off",
                extra={
                    "node_id": transcript.id,
                    "error": str(exc),
                    "endpoint_calls": call_metrics(transcript_id=transcript.id)
                    if callable(call_metrics)
                    else {},
                },
            )
            if not dry_run and not validation_skip:
                try:
                    self._mark_backoff(transcript)
                except Exception as backoff_exc:  # noqa: BLE001 — retry eligibility is retained without the touch
                    self._logger.warning(
                        "Failed to update transcript backoff timestamp",
                        extra={"node_id": transcript.id, "error": str(backoff_exc)},
                    )
        finally:
            client = self._get_sampling_client()
            metrics = getattr(client, "call_metrics", None)
            if callable(metrics):
                report.endpoint_calls = metrics(transcript_id=transcript.id, reset=True)
                for endpoint, endpoint_stats in report.endpoint_calls.items():
                    stages = endpoint_stats.get("stages", {})
                    endpoint_stats.update(
                        transcripts=1,
                        segments=stages.get("extract", {}).get("successes", 0),
                        items=stages.get("write_decider", {}).get("successes", 0),
                        actions=report.actions_by_endpoint.get(endpoint, {}),
                    )
            for detail in report.details:
                detail["endpoint_calls"] = report.endpoint_calls
            self._close_thread_store()
        completed_at = datetime.now(UTC)
        report.completed_at = completed_at.isoformat().replace("+00:00", "Z")
        report.duration_ms = int((completed_at - started).total_seconds() * 1000)
        return report

    @staticmethod
    def _merge_transcript_report(target: DistillerReport, source: DistillerReport) -> None:
        for name in (
            "transcripts_distilled", "transcripts_skipped", "items_extracted", "items_invalid",
            "items_sources_dropped", "reduced_items", "reduce_failures", "reduce_groups_llm",
            "guard_fallback_creates", "title_repairs", "title_repair_failures", "decider_creates",
            "decider_complements", "decider_supersedes", "decider_downgrades", "merged_updates",
            "llm_failures", "segments_total",
        ):
            setattr(target, name, getattr(target, name) + getattr(source, name))
        target.downgraded.extend(source.downgraded)
        target.details.extend(source.details)
        for endpoint, source_stats in source.endpoint_calls.items():
            target_stats = target.endpoint_calls.setdefault(endpoint, {
                "base_url": source_stats["base_url"],
                "model": source_stats["model"],
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "latency_total_ms": 0.0,
                "latency_max_ms": 0.0,
                "transcripts": 0,
                "segments": 0,
                "items": 0,
                "stages": {},
                "actions": {},
            })
            for name in ("calls", "successes", "failures", "latency_total_ms", "transcripts", "segments", "items"):
                target_stats[name] += source_stats.get(name, 0)
            target_stats["latency_max_ms"] = max(
                target_stats["latency_max_ms"], source_stats.get("latency_max_ms", 0.0)
            )
            for stage, source_stage in source_stats.get("stages", {}).items():
                target_stage = target_stats["stages"].setdefault(
                    stage, {"calls": 0, "successes": 0, "failures": 0, "latency_total_ms": 0.0}
                )
                for name in ("calls", "successes", "failures", "latency_total_ms"):
                    target_stage[name] += source_stage.get(name, 0)
            for action, count in source_stats.get("actions", {}).items():
                target_stats["actions"][action] = target_stats["actions"].get(action, 0) + count

    # ------------------------------------------------------------------
    # Per-transcript distillation
    # ------------------------------------------------------------------

    def _run_one_transcript(self, transcript: Node, *, dry_run: bool = False, downgrade_supersede: bool = False) -> DistillerReport:
        """Distill exactly one transcript (CLI --id and backfill paths)."""

        started = datetime.now(UTC)
        report = DistillerReport(started_at=started.isoformat().replace("+00:00", "Z"), completed_at="")
        report.transcripts_scanned = 1
        try:
            detail = self._distill_transcript(
                transcript,
                report,
                dry_run=dry_run,
                downgrade_supersede=downgrade_supersede,
            )
            report.details.append(detail)
        except Exception as exc:  # noqa: BLE001
            report.llm_failures += 1
            report.transcripts_skipped += 1
            validation_skip = self._is_final_okf_validation_error(exc)
            self._logger.warning(
                "Distiller final OKF validation skipped write; transcript remains unstamped and retryable"
                if validation_skip
                else "Distill failed for transcript",
                extra={"node_id": transcript.id, "error": str(exc)},
            )
            if not dry_run and not validation_skip:
                self._mark_backoff(transcript)
        completed = datetime.now(UTC)
        report.completed_at = completed.isoformat().replace("+00:00", "Z")
        report.duration_ms = int((completed - started).total_seconds() * 1000)
        if not dry_run:
            self._get_store().record_distiller_run(report.metrics())
        return report

    def _distill_transcript(
        self,
        transcript: Node,
        report: DistillerReport,
        *,
        dry_run: bool,
        downgrade_supersede: bool,
    ) -> dict[str, Any]:
        parsed = parse_transcript(transcript.content)
        # Incremental distillation (P2c): only turns after the watermark are
        # processed; one previous turn is included as context.
        watermark = transcript.metadata.distilled_turn or 0
        pending = parsed.turns_after(watermark)
        if not pending:
            # Nothing new since the last distillation — treat as current.
            if not dry_run:
                self._stamp_transcript(transcript, [])
            report.transcripts_distilled += 1
            return {"transcript_id": transcript.id, "items": [], "skipped_new_turns": 0}

        segments = self._build_segments(parsed, watermark=watermark, max_chars=self.settings.segment_chars)
        report.segments_total += len(segments)
        own_titles = self._titles_for_ids(transcript.metadata.distilled_node_ids)
        # known_titles computed ONCE: identical for every segment, and the
        # store's SQLite connection must not be touched from pool threads.
        known_titles = self._known_titles()

        # One extraction call per segment (same call size as the old
        # single-call flow); candidates merge afterwards.
        candidates: list[OkfItem] = []
        extraction_failures = 0
        first_failed_turn: int | None = None
        # Extraction shares one executor across all transcript workers. Batch
        # per transcript so a very long transcript cannot monopolize it.
        extraction_results: list[tuple[int, str, list[OkfItem] | Exception]] = []

        def _extract(segment_text: str, start_turn: str) -> list[OkfItem] | Exception:
            prompt = render_distiller_prompt(
                transcript=segment_text,
                known_titles=known_titles,
                own_distilled_titles=own_titles,
                start_turn=start_turn,
            )
            try:
                with self._sampling_call_context(transcript, "extract"):
                    payload = self._sample_json(prompt)
                return parse_okf_items(payload)
            except Exception as exc:  # noqa: BLE001 — one bad segment must not lose the others
                return exc

        executor = self._get_llm_executor()
        concurrency = max(1, min(self._llm_calls_per_transcript(), len(segments)))
        for offset in range(0, len(segments), concurrency):
            batch = segments[offset : offset + concurrency]
            futures = [
                (executor.submit(_extract, segment_text, start_turn), offset + index, start_turn)
                for index, (segment_text, start_turn) in enumerate(batch)
            ]
            for future, index, start_turn in futures:
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = exc
                extraction_results.append((index, start_turn, result))

        for _index, start_turn, result in sorted(extraction_results, key=lambda item: item[0]):
            if isinstance(result, Exception):
                extraction_failures += 1
                if first_failed_turn is None:
                    m = re.match(r"T(\d+)", start_turn)
                    first_failed_turn = int(m.group(1)) if m else parsed.last_turn_number
                self._logger.warning(
                    "Segment extraction failed",
                    extra={"node_id": transcript.id, "start_turn": start_turn, "error": str(result)},
                )
            else:
                candidates.extend(result)
        if extraction_failures and not candidates:
            raise RuntimeError(f"All {extraction_failures} segment extraction(s) failed")

        with self._sampling_call_context(transcript, "reduce"):
            candidates = self._reduce_candidates(transcript, candidates, own_titles, report, dry_run=dry_run)

        # Deterministic evidence verification (P2+, no LLM): quotes must
        # appear in the named turn's ELIGIBLE text only (docs/okf.md §9) —
        # user-turn bodies and tool result lines (call lines only for
        # procedure with an ok result). Assistant prose never verifies.
        for item in candidates:
            evidence_turns: dict[str, str] = {}
            question_only_turns: set[str] = set()
            for turn in parsed.turns:
                if turn.role == "User":
                    evidence_text, question_only = self._user_turn_evidence(turn.body)
                    if question_only:
                        question_only_turns.add(turn.id)
                else:
                    evidence_text = turn.eligible_evidence_text(okf_type=item.okf_type)
                evidence_turns[turn.id] = evidence_text
            verified, unverified = verify_evidence(
                item.evidence,
                turns=evidence_turns,
                question_only_turns=question_only_turns,
            )
            item.evidence = [*verified, *unverified]
            item.verified = bool(verified)




        service = self._get_service()
        preparations: list[dict[str, Any] | None] = [None] * len(candidates)
        concurrency = max(1, min(self._llm_calls_per_transcript(), len(candidates)))
        executor = self._get_llm_executor()
        for offset in range(0, len(candidates), concurrency):
            batch = candidates[offset : offset + concurrency]
            futures = [
                (executor.submit(self._prepare_write_item, service, transcript, item, dry_run=dry_run), offset + index)
                for index, item in enumerate(batch)
            ]
            preparation_error: Exception | None = None
            for future, index in futures:
                try:
                    preparations[index] = future.result()
                except Exception as exc:  # leave the transcript retryable; do not partially commit this batch
                    preparation_error = preparation_error or exc
            if preparation_error is not None:
                raise preparation_error

        produced_ids: list[str] = []
        action_counts: dict[str, int] = {}
        detail: dict[str, Any] = {"transcript_id": transcript.id, "items": [], "segments": len(segments)}
        for item, preparation in zip(candidates, preparations):
            assert preparation is not None
            report.title_repairs += int(preparation["title_repaired"])
            report.title_repair_failures += int(preparation["title_repair_failed"])
            dropped = preparation["dropped_sources"]
            if dropped:
                report.items_sources_dropped += len(dropped)
                self._logger.info(
                    "Dropped non-conforming sources",
                    extra={"node_id": transcript.id, "item": item.title, "dropped": dropped},
                )
            prepared = preparation["prepared"]
            if prepared is not None:
                item.title = prepared["title"]
                item.sources = list(prepared["sources"] or [])
            action_note = self._write_item(
                service,
                transcript,
                item,
                report,
                prepared=prepared,
                decision_endpoint=preparation["decision_endpoint"],
                guard_fallback=preparation["guard_fallback"],
                dry_run=dry_run,
                downgrade_supersede=downgrade_supersede,
            )
            produced_ids.extend(action_note.get("node_ids", []))
            detail["items"].append(
                {"title": item.title, "okf_type": item.okf_type, "verified": item.verified, **action_note}
            )
            action = action_note.get("action", "")
            action_counts[action] = action_counts.get(action, 0) + 1
        report.items_extracted += len(candidates)
        report.transcripts_distilled += 1

        # Runaway guard (owner 2026-10-12): warn (never drop) when a transcript
        # yields an unusual number of items.
        if len(candidates) > self.settings.item_warn_threshold:
            self._logger.warning(
                "Transcript yielded more than item_warn_threshold items; all written",
                extra={"node_id": transcript.id, "items": len(candidates), "threshold": self.settings.item_warn_threshold},
            )

        if not dry_run:
            # Watermark safety (2026-10-12 live incident): on partial segment
            # failure, advance the watermark only to the turn BEFORE the first
            # failed segment — failed segments' knowledge would otherwise be
            # permanently skipped. distilled_hash is still stamped (content
            # hash gate) but the reduced watermark makes the next sweep
            # re-process the failed range (via distilled_turn, provided the
            # transcript hasn't changed since — the hash gate blocks same-
            # content retries). To guarantee the retry, leave the hash unset
            # when any segment failed so the next sweep re-selects it.
            if extraction_failures:
                safe_turn = max(watermark, first_failed_turn - 1) if first_failed_turn else watermark
                self._stamp_transcript(
                    transcript, produced_ids, distilled_turn=safe_turn, mark_current=False
                )
                self._mark_backoff(transcript)
                self._logger.warning(
                    "Distilled transcript with segment failures; watermark held back for retry",
                    extra={
                        "node_id": transcript.id,
                        "segments": len(segments),
                        "failed_segments": extraction_failures,
                        "safe_watermark": safe_turn,
                        "items": len(candidates),
                    },
                )
            else:
                # The watermark covers everything parsed NOW (the client may
                # have appended turns while we distilled); late turns surface
                # on the next sweep via the content-hash gate.
                self._stamp_transcript(transcript, produced_ids, distilled_turn=parsed.last_turn_number)
            client_metrics = getattr(self._get_sampling_client(), "call_metrics", None)
            endpoint_calls = client_metrics(transcript_id=transcript.id) if callable(client_metrics) else {}
            self._logger.info(
                "Distilled transcript",
                extra={
                    "node_id": transcript.id,
                    "segments": len(segments),
                    "candidates": len(candidates),
                    "after_reduce": len(candidates) - report.reduced_items if report.reduced_items else len(candidates),
                    "reduce_groups_llm": report.reduce_groups_llm,
                    "reduce_failure": report.reduce_failures > 0,
                    "items": len(candidates),
                    "written_by_action": action_counts,
                    "types": sorted({item.okf_type for item in candidates}),
                    "knowledge_ids": produced_ids,
                    "distilled_turn": parsed.last_turn_number,
                    "segment_failures": extraction_failures,
                    "endpoint_calls": endpoint_calls,
                },
            )
        return detail

    @staticmethod
    def _user_turn_evidence(text: str) -> tuple[str, bool]:
        fragments = [
            fragment.strip()
            for fragment in re.split(r"(?<=[?？])\s*|\n+", text)
            if fragment.strip()
        ]
        if not fragments:
            return text, False
        assertive = [
            fragment
            for fragment in fragments
            if not (
                fragment.endswith(("?", "？"))
                or (_QUESTION_ONLY_PREFIX.match(fragment) is not None and not fragment.endswith((".", "。")))
            )
        ]
        return "\n".join(assertive), not assertive

    @staticmethod
    def _build_segments(parsed: ParsedTranscript, *, watermark: int, max_chars: int) -> list[tuple[str, str]]:
        """Split the transcript into ~max_chars blocks at turn boundaries.

        Returns ``(segment_text, start_turn_id)`` pairs. The turn containing
        the watermark is included as context for the first segment; a single
        oversized turn is split by chars. Nothing is truncated away.
        """

        return _build_turn_segments(parsed, watermark=watermark, max_chars=max_chars)

    # Prompt-size guard: only the 200 most recently accessed persistent
    # titles go into the "already known" block (design A5).
    MAX_KNOWN_TITLES = 200
    # Limit one reranker request to vector top-8, the 20 newest transcript-owned
    # nodes, and the top 20 title-Jaccard matches (48 documents maximum).
    NEAR_DUP_VECTOR_TOP_K = 8
    NEAR_DUP_OWN_TOP_K = 20
    NEAR_DUP_TITLE_JACCARD_TOP_K = 20
    NEAR_DUP_RERANK_MAX_CANDIDATES = 48

    def _transcript_project(self, transcript: Node) -> str | None:
        """Derive the project for distilled nodes from the transcript.

        Keyed transcripts: the ``project`` frontmatter if present. Legacy
        transcripts: the title suffix after "Session summary — ".
        """

        if transcript.metadata.project:
            return transcript.metadata.project
        title = transcript.metadata.title
        if title.startswith("Session summary"):
            for separator in ("\u2014", "\u2013", "--"):  # em dash, en dash, double hyphen
                if separator in title:
                    project = title.split(separator, 1)[1].strip()
                    if project:
                        return project
        return None

    # ------------------------------------------------------------------
    # Title rule: English titles + one-shot repair
    # ------------------------------------------------------------------

    @staticmethod
    def _title_needs_repair(title: str) -> bool:
        """CJK title or a degenerate ID slug (docs/okf.md §3 / §7 ID plan)."""

        if any("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" for ch in title):
            return True
        if len(title) > 90:
            return True
        return not slugify_english_title(title)

    def _repair_title(self, item: OkfItem) -> str | None:
        """ONE short repair LLM call asking only for an English title."""

        from synapse.okf import validate_okf_node

        body_excerpt = "\n".join(
            f"## {section}\n{(item.sections.get(section) or '').strip()}"
            for section in TYPE_REQUIRED_SECTIONS[item.okf_type]
        )
        prompt = render_title_repair_prompt(title=item.title, takeaway=item.takeaway, body_excerpt=body_excerpt)
        try:
            payload = self._get_sampling_client().sample_json(
                prompt=prompt, system_prompt="", max_tokens=self.config.decider.max_tokens
            )
        except Exception as exc:  # noqa: BLE001 — repair failure falls back, never drops
            self._logger.warning("Title repair call failed", extra={"item": item.title, "error": str(exc)})
            return None
        repaired = parse_title_repair(payload)
        if repaired is None:
            return None
        # A repaired title must not itself be degenerate.
        if not slugify_english_title(repaired):
            return None
        return repaired

    def _known_titles(self) -> list[str]:
        store = self._get_store()
        persistent = [
            node for node in store.list_nodes({"status": NodeStatus.ACTIVE, "type": NodeType.PERSISTENT})
        ]
        persistent.sort(key=lambda n: n.metadata.last_accessed or n.metadata.created_at, reverse=True)
        return [node.title for node in persistent[: self.MAX_KNOWN_TITLES]]

    def _build_prompt(self, *, transcript_content: str, start_turn: str, own_titles: list[str]) -> str:
        return render_distiller_prompt(
            transcript=transcript_content,
            known_titles=self._known_titles(),
            own_distilled_titles=own_titles,
            start_turn=start_turn,
        )

    def _reduce_candidates(
        self,
        transcript: Node,
        candidates: list[OkfItem],
        own_titles: list[str],
        report: DistillerReport,
        *,
        dry_run: bool,
    ) -> list[OkfItem]:
        """Dedupe per-segment candidates — NEVER compress to a count cap.

        Grouping is deterministic: normalized-title exact match or prefix
        relation groups items that are likely the same claim. Only groups with
        >= 2 members go to the LLM (in bounded batches so the reduce prompt
        stays small); singletons pass through untouched. The LLM may merge
        same-claim items and split composites; every other item must come back
        unchanged. On reduce failure: retry once, then a deterministic
        exact-title dedupe keeps ALL distinct items. No truncation anywhere.
        """

        if not candidates:
            return []

        def _norm(title: str) -> str:
            return " ".join(title.split()).casefold()

        # Deterministic grouping (owner rework 2026-10-13): union items whose
        # normalized titles are equal/prefix-related OR share >=3 significant
        # tokens (stopwords removed) — same-component siblings like
        # genie_app_healthz* / hub_* / oauth_* families. Calibrated on the
        # 32-item 41975f14 run: 22 groups, 7 multi-groups to the LLM, max 5
        # members (judge ideal ~12 nodes). Groups of size 1 pass through.
        parent = list(range(len(candidates)))

        _STOP = {'the', 'a', 'an', 'for', 'and', 'or', 'of', 'to', 'in', 'on',
                 'with', 'is', 'are', 'not', 'from', 'by', 'at', 'as'}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        normed = [_norm(item.title) for item in candidates]
        token_sets = [
            {w for w in re.split(r'[^a-z0-9]+', item.title.lower()) if len(w) > 3 and w not in _STOP}
            for item in candidates
        ]
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = normed[i], normed[j]
                linked = a == b or a.startswith(b) or b.startswith(a)
                if not linked and len(token_sets[i] & token_sets[j]) >= 3:
                    linked = True
                if linked:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(len(candidates)):
            groups.setdefault(find(i), []).append(i)
        multi_groups = [g for g in groups.values() if len(g) >= 2]
        report.reduce_groups_llm = len(multi_groups)

        if not multi_groups:
            return candidates  # all singletons → nothing to dedupe
        if dry_run:
            return candidates

        # LLM reduce over the multi-item groups, in bounded batches.
        merged_extra: list[OkfItem] = []
        reduce_failed = False
        batch_size = 12  # groups per LLM call — keeps the prompt small
        for batch_start in range(0, len(multi_groups), batch_size):
            batch = multi_groups[batch_start : batch_start + batch_size]
            batch_items = [item for g in batch for item in (candidates[i] for i in g)]
            candidates_json = json.dumps(
                {"items": [{"okf_type": item.okf_type, "title": item.title, "takeaway": item.takeaway,
                            "sections": item.sections, "sources": item.sources,
                            "evidence": [{"turn": e.turn, "quote": e.quote} for e in item.evidence]}
                           for item in batch_items]},
                ensure_ascii=False,
            )
            merged: list[OkfItem] | None = None
            for attempt in (1, 2):  # retry once on failure
                try:
                    payload = self._sample_json(
                        render_reduce_prompt(
                            candidates_json=candidates_json,
                            known_titles=self._known_titles(),
                            own_distilled_titles=own_titles,
                        )
                    )
                    merged = parse_okf_items(payload)
                    break
                except Exception as exc:  # noqa: BLE001 — retry once, then deterministic fallback
                    self._logger.warning(
                        "Reduce call failed",
                        extra={"node_id": transcript.id, "attempt": attempt, "error": str(exc)},
                    )
            if not merged:
                reduce_failed = True
                continue
            merged_extra.extend(merged)

        if reduce_failed:
            report.reduce_failures += 1
            # Deterministic fallback: exact-normalized-title dedupe keeping ALL
            # distinct items — never truncate. merged results from successful
            # batches are dropped in favor of the originals so nothing the LLM
            # returned is silently half-applied.
            seen: set[str] = set()
            deduped: list[OkfItem] = []
            for item in candidates:
                key = _norm(item.title)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            report.reduced_items += max(0, len(candidates) - len(deduped))
            return deduped

        # Successful reduce: multi-group members are replaced by the LLM's
        # merged output; singletons pass through untouched.
        multi_indexes = {i for g in multi_groups for i in g}
        result: list[OkfItem] = []
        for i, item in enumerate(candidates):
            if i in multi_indexes:
                continue
            result.append(item)
        result.extend(merged_extra)
        report.reduced_items += max(0, len(candidates) - len(result))
        return result

    def _titles_for_ids(self, node_ids: list[str]) -> list[str]:
        if not node_ids:
            return []
        nodes = self._get_store().get_nodes(node_ids)
        return [node.title for node in nodes]

    def _sample_json(self, prompt: str) -> dict[str, Any]:
        client = self._get_sampling_client()
        # Owner rule: no tight token cap — wall time is bounded by the LLM
        # timeout, so extraction/reduce calls use [decider] max_tokens as-is.
        # [distiller].reasoning_effort (optional) is passed as the OpenAI
        # reasoning_effort field on the distiller's OWN calls only; the
        # write-path decider is unaffected.
        kwargs: dict[str, Any] = {}
        if self.settings.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.settings.reasoning_effort
        return client.sample_json(
            prompt=prompt, system_prompt="", max_tokens=self.config.decider.max_tokens, **kwargs
        )

    # ------------------------------------------------------------------
    # Item → write path
    # ------------------------------------------------------------------

    @staticmethod
    def _title_jaccard(a: str, b: str) -> float:
        _STOP = {'the', 'a', 'an', 'for', 'and', 'or', 'of', 'to', 'in', 'on',
                 'with', 'is', 'are', 'not', 'from', 'by', 'at', 'as'}
        ta = {w for w in re.split(r'[^a-z0-9]+', a.lower()) if len(w) > 3 and w not in _STOP}
        tb = {w for w in re.split(r'[^a-z0-9]+', b.lower()) if len(w) > 3 and w not in _STOP}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @staticmethod
    def _takeaway_key(text: str) -> str:
        return " ".join(text.split()).casefold()

    def _find_near_duplicate(
        self, item: OkfItem, transcript: Node, store: SQLiteNodeStore
    ) -> Node | None:
        """Find a near-duplicate using full-corpus Jaccard and bounded reranking.

        The reranker sees vector top-8, up to 20 transcript-owned nodes, and
        the 20 highest title-Jaccard matches (at most 48 documents total).
        Jaccard still scans all active persistent nodes so unembedded knowledge
        remains eligible for the existing fallback thresholds.

        Merge thresholds are calibrated on the judge's dup pairs plus
        related-same-surface and clear non-dups:
          rerank >= 3.0 AND jaccard >= 0.25, OR
          rerank >= 1.5 AND jaccard >= 0.5, OR
          (reranker unavailable) jaccard >= 0.5 alone.
        """

        query_text = f"{item.title}\n{item.takeaway}".strip()
        candidates_nodes: list[Node] = []
        candidate_index_by_id: dict[str, int] = {}
        vector_indices: list[int] = []
        own_indices: list[int] = []

        def add_candidate(node: Node) -> int:
            existing = candidate_index_by_id.get(node.id)
            if existing is not None:
                return existing
            index = len(candidates_nodes)
            candidates_nodes.append(node)
            candidate_index_by_id[node.id] = index
            return index

        # Embedding leg: top-k over stored vectors (knowledge only; vector
        # search covers active+archived rows, filter to ACTIVE persistent).
        query_vector: list[float] | None = None
        try:
            from synapse.embedding import create_embedding_engine

            engine = create_embedding_engine(self.config.embedding, providers=self.config.providers)
            if getattr(engine, "backend_name", "") != "unavailable" and engine.is_available():
                query_vector = engine.embed(query_text)
        except (OSError, RuntimeError, ValueError):
            query_vector = None
        if query_vector:
            for node_id, _distance in store.vector_search(query_vector, limit=self.NEAR_DUP_VECTOR_TOP_K):
                node = store.get_node(node_id)
                if (
                    node is not None
                    and node.metadata.status is NodeStatus.ACTIVE
                    and node.metadata.type is NodeType.PERSISTENT
                ):
                    index = add_candidate(node)
                    if index not in vector_indices:
                        vector_indices.append(index)

        # Own distilled nodes remain priority candidates, followed by the full
        # active persistent corpus for the inexpensive title-Jaccard pass.
        for node_id in transcript.metadata.distilled_node_ids[-self.NEAR_DUP_OWN_TOP_K:]: 
            node = store.get_node(node_id)
            if node is not None:
                index = add_candidate(node)
                if index not in own_indices:
                    own_indices.append(index)
        for node in store.list_nodes({"status": NodeStatus.ACTIVE, "type": NodeType.PERSISTENT}):
            add_candidate(node)
        if not candidates_nodes:
            return None

        takeaway_key = self._takeaway_key(item.takeaway)
        if takeaway_key:
            for candidate in candidates_nodes:
                existing_takeaway = self._takeaway_key(_extract_section(candidate.content, "Takeaway") or "")
                if (
                    candidate.metadata.status is NodeStatus.ACTIVE
                    and candidate.metadata.type is NodeType.PERSISTENT
                    and existing_takeaway == takeaway_key
                ):
                    return candidate
        jaccards = [self._title_jaccard(item.title, candidate.title) for candidate in candidates_nodes]
        if max(jaccards) <= 0.0:
            return None

        from heapq import nlargest

        shortlist_indices: list[int] = []
        shortlist_seen: set[int] = set()

        def add_to_shortlist(index: int) -> None:
            if (
                index not in shortlist_seen
                and len(shortlist_indices) < self.NEAR_DUP_RERANK_MAX_CANDIDATES
            ):
                shortlist_seen.add(index)
                shortlist_indices.append(index)

        for index in vector_indices[: self.NEAR_DUP_VECTOR_TOP_K]:
            add_to_shortlist(index)
        for index in own_indices[: self.NEAR_DUP_OWN_TOP_K]:
            add_to_shortlist(index)
        for index in nlargest(
            self.NEAR_DUP_TITLE_JACCARD_TOP_K,
            range(len(jaccards)),
            key=jaccards.__getitem__,
        ):
            add_to_shortlist(index)

        rerank_scores: list[float | None] = [None] * len(candidates_nodes)
        try:
            from synapse.embedding import create_reranker_engine
            from synapse.utils.documents import render_node_document

            reranker = create_reranker_engine(self.config.reranker, providers=self.config.providers)
            documents = [render_node_document(candidates_nodes[index]) for index in shortlist_indices]
            for shortlist_index, score in reranker.rerank(query_text, documents):
                if 0 <= shortlist_index < len(shortlist_indices):
                    rerank_scores[shortlist_indices[shortlist_index]] = float(score)
        except (OSError, RuntimeError, ValueError):
            pass

        best_index = max(
            range(len(candidates_nodes)),
            key=lambda index: (
                rerank_scores[index] if rerank_scores[index] is not None else -1e9,
                jaccards[index],
            ),
        )
        jaccard = jaccards[best_index]
        score = rerank_scores[best_index]
        if score is None:
            return candidates_nodes[best_index] if jaccard >= 0.5 else None
        if (score >= 3.0 and jaccard >= 0.25) or (score >= 1.5 and jaccard >= 0.5):
            return candidates_nodes[best_index]
        return None

    def _merge_item_into_node(self, item: OkfItem, existing: Node) -> str | None:
        """LLM merge of the existing node body + the new item (gap 1).

        Returns the merged OKF body for the in-place update, or None when the
        merge call fails or the response does not validate. The EXISTING
        node's okf_type governs the required sections; the merged body keeps
        the existing node's knowledge language.
        """

        import json as _json

        okf_type = existing.metadata.okf_type
        if okf_type is None or okf_type not in TYPE_REQUIRED_SECTIONS:
            # Untyped legacy node: merge only when the item type matches the
            # item's own sections (the common distiller case).
            okf_type = item.okf_type
        required = TYPE_REQUIRED_SECTIONS[okf_type]
        existing_doc = {
            "okf_type": okf_type,
            "title": existing.title,
            "takeaway": _first_section_line(existing.content, "Takeaway"),
            "sections": {
                name: _extract_section(existing.content, name) or ""
                for name in ["Takeaway", *required, "Sources"]
            },
        }
        new_doc = {
            "okf_type": item.okf_type,
            "title": item.title,
            "takeaway": item.takeaway,
            "sections": dict(item.sections),
        }
        prompt = render_merge_update_prompt(
            existing_node_json=_json.dumps(existing_doc, ensure_ascii=False),
            new_item_json=_json.dumps(new_doc, ensure_ascii=False),
        )
        try:
            payload = self._sample_json(prompt)
        except Exception as exc:  # noqa: BLE001 — merge failure falls back to plain update
            self._logger.warning(
                "Merge-update call failed; falling back to plain item body",
                extra={"node_id": existing.id, "error": str(exc)},
            )
            return None
        if not isinstance(payload, Mapping):
            return None
        sections = payload.get("sections")
        takeaway = str(payload.get("takeaway") or "").strip()
        if not isinstance(sections, Mapping) or not takeaway or len(takeaway.splitlines()) != 1:
            return None
        normalized: dict[str, str] = {}
        for key, value in sections.items():
            name = str(key).lstrip("#").strip()
            if name:
                normalized.setdefault(name, str(value))
        missing = [s for s in required if not normalized.get(s, "").strip()]
        if missing:
            return None
        lines: list[str] = ["## Takeaway", "", takeaway, ""]
        for section in required:
            lines.extend([f"## {section}", "", normalized[section].strip(), ""])
        # Evidence from the new item (already deterministically verified) is
        # appended; existing node sources stay in frontmatter via the service.
        if item.evidence:
            lines.append(render_evidence_section(item.evidence).rstrip("\n"))
            lines.append("")
        lines.append("## Sources")
        lines.append("")
        for source in existing.metadata.sources:
            lines.append(f"- {source}")
        for source in item.sources:
            if source not in existing.metadata.sources:
                lines.append(f"- {source}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _is_transcript_target_guard_error(exc: SynapseServiceError) -> bool:
        return exc.code == "INVALID_SAMPLING_RESPONSE" and (
            "never session transcripts" in exc.message
            or "not present in the candidate set" in exc.message
        )

    def _prepare_write_item(
        self,
        service: SynapseServerService,
        transcript: Node,
        item: OkfItem,
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        token = service.push_sampling_client(self._get_sampling_client())
        title_repaired = False
        title_repair_failed = False
        prepared: dict[str, Any] | None = None
        decision_endpoint: str | None = None
        guard_fallback = False
        try:
            if self._title_needs_repair(item.title):
                with self._sampling_call_context(transcript, "title_repair"):
                    repaired = self._repair_title(item)
                if repaired:
                    title_repaired = True
                    item.title = repaired
                else:
                    title_repair_failed = True
            item.sources = list(dict.fromkeys([*item.sources, transcript.id]))
            store = self._get_store()
            kept, dropped = normalize_sources(
                item.sources,
                resolve_node=lambda node_id: store.get_node(node_id) is not None,
            )
            item.sources = kept
            if not dry_run:
                try:
                    with self._sampling_call_context(transcript, "write_decider") as call_context:
                        prepared = service.prepare_distilled_write(
                            title=item.title,
                            content=item.render(),
                            node_type=NodeType.PERSISTENT,
                            sources=list(item.sources),
                            okf_type=item.okf_type,
                        )
                except SynapseServiceError as exc:
                    if not self._is_transcript_target_guard_error(exc):
                        raise
                    guard_fallback = True
                if isinstance(call_context, dict):
                    decision_endpoint = next(
                        (
                            endpoint
                            for endpoint, stats in call_context["endpoints"].items()
                            if stats.get("successes", 0)
                        ),
                        None,
                    )
            return {
                "prepared": prepared,
                "decision_endpoint": decision_endpoint,
                "title_repaired": title_repaired,
                "title_repair_failed": title_repair_failed,
                "guard_fallback": guard_fallback,
                "dropped_sources": dropped,
            }
        finally:
            service.reset_sampling_client(token)
            self._close_thread_store()

    def _write_item(
        self,
        service: SynapseServerService,
        transcript: Node,
        item: OkfItem,
        report: DistillerReport,
        *,
        prepared: dict[str, Any] | None = None,
        decision_endpoint: str | None = None,
        guard_fallback: bool = False,
        dry_run: bool,
        downgrade_supersede: bool,
    ) -> dict[str, Any]:
        if dry_run:
            return {"action": "planned", "node_ids": []}
        if prepared is None and not guard_fallback:
            try:
                with self._sampling_call_context(transcript, "write_decider") as call_context:
                    prepared = service.prepare_distilled_write(
                        title=item.title,
                        content=item.render(),
                        node_type=NodeType.PERSISTENT,
                        sources=list(item.sources),
                        okf_type=item.okf_type,
                    )
            except SynapseServiceError as exc:
                if not self._is_transcript_target_guard_error(exc):
                    raise
                guard_fallback = True
            if isinstance(call_context, dict):
                decision_endpoint = next(
                    (
                        endpoint
                        for endpoint, stats in call_context["endpoints"].items()
                        if stats.get("successes", 0)
                    ),
                    None,
                )
        with self._write_commit_lock:
            store = self._get_store()
            near_dup = self._find_near_duplicate(item, transcript, store)
            exact_takeaway_duplicate = (
                near_dup is not None
                and self._takeaway_key(item.takeaway)
                == self._takeaway_key(_extract_section(near_dup.content, "Takeaway") or "")
            )
            if exact_takeaway_duplicate:
                self._logger.info(
                    "Exact takeaway dedupe matched an active persistent node",
                    extra={"transcript_id": transcript.id, "target_node_id": near_dup.id},
                )
                candidate_content = prepared["content"] if prepared is not None else item.render()
                if candidate_content.strip() == near_dup.content.strip():
                    merged_content = near_dup.content
                else:
                    with self._sampling_call_context(transcript, "near_duplicate_merge"):
                        merged_content = self._merge_item_into_node(item, near_dup)
                    if merged_content is None:
                        raise SynapseServiceError(
                            "EXACT_TAKEAWAY_MERGE_FAILED",
                            "Exact-takeaway duplicate could not be safely merged",
                            status_code=502,
                            details={"target_node_id": near_dup.id},
                        )
                with self._sampling_call_context(transcript, "write_commit"):
                    updated = service._merge_update_node(
                        target_id=near_dup.id,
                        content=merged_content,
                        title=item.title,
                        sources=list(item.sources),
                        verified=item.verified,
                        route="distiller",
                        okf_type=item.okf_type,
                    )
                updated_node = updated.get("node")
                node_id = updated_node.get("id") if isinstance(updated_node, dict) else None
                action = str(updated.get("action", "updated"))
                self._count_action(report, action)
                if decision_endpoint is not None:
                    endpoint_actions = report.actions_by_endpoint.setdefault(decision_endpoint, {})
                    endpoint_actions[action] = endpoint_actions.get(action, 0) + 1
                return {"action": action, "node_ids": [node_id] if node_id else []}
            if guard_fallback:
                if near_dup is not None:
                    with self._sampling_call_context(transcript, "near_duplicate_merge"):
                        merged = self._merge_item_into_node(item, near_dup)
                    with self._sampling_call_context(transcript, "write_commit"):
                        updated = service._merge_update_node(
                            target_id=near_dup.id,
                            content=merged or item.render(),
                            title=item.title,
                            sources=list(item.sources),
                            verified=item.verified,
                            route="distiller",
                            okf_type=item.okf_type,
                        )
                    updated_node = updated.get("node")
                    node_id = updated_node.get("id") if isinstance(updated_node, dict) else None
                    action = str(updated.get("action", "updated"))
                    self._count_action(report, action)
                    return {"action": action, "node_ids": [node_id] if node_id else []}
                report.guard_fallback_creates += 1
                self._logger.warning(
                    "Guard fired on distilled item; falling back to create",
                    extra={"node_id": transcript.id, "item": item.title},
                )
                with self._sampling_call_context(transcript, "write_commit"):
                    fallback = service.integrate_knowledge(
                        title=item.title,
                        content=item.render(),
                        node_type=NodeType.PERSISTENT,
                        sources=list(item.sources),
                        okf_type=item.okf_type,
                        okf_version=1,
                        project=self._transcript_project(transcript),
                        validate_okf=True,
                    )
                node_payload = fallback.get("node") if isinstance(fallback, dict) else None
                node_id = node_payload.get("id") if isinstance(node_payload, dict) else None
                return {"action": "guard_fallback_create", "node_ids": [node_id] if node_id else []}
            if prepared is None:
                raise RuntimeError("distilled write has no prepared decision")
            content = prepared["content"]
            if near_dup is not None:
                with self._sampling_call_context(transcript, "near_duplicate_merge"):
                    merged = self._merge_item_into_node(item, near_dup)
                if merged is not None:
                    content = merged

            forced_node_id: str | None = None
            if not slugify_english_title(item.title):
                date_part = datetime.now(UTC).strftime("%Y%m%d")
                digest = hashlib.sha256(f"{item.title}{item.takeaway}".encode("utf-8")).hexdigest()[:8]
                type_part = item.okf_type if item.okf_type in ("decision", "fact", "procedure", "pitfall") else "node"
                forced_node_id = f"mem_{date_part}_{type_part}_{digest}"
            prepared_write = dict(prepared)
            prepared_write["content"] = content
            try:
                with self._sampling_call_context(transcript, "write_commit"):
                    result = service.commit_distilled_write(
                        prepared_write,
                        okf_version=1,
                        project=self._transcript_project(transcript),
                        node_id=forced_node_id,
                        downgrade_supersede=downgrade_supersede,
                        route="distiller",
                        verified=item.verified,
                        merge_update_target=near_dup.id if near_dup is not None else None,
                    )
            except SynapseServiceError as exc:
                if not self._is_transcript_target_guard_error(exc):
                    raise
                report.guard_fallback_creates += 1
                self._logger.warning(
                    "Guard fired on distilled item; falling back to create",
                    extra={"node_id": transcript.id, "item": item.title},
                )
                with self._sampling_call_context(transcript, "write_commit"):
                    fallback = service.integrate_knowledge(
                        title=item.title,
                        content=item.render(),
                        node_type=NodeType.PERSISTENT,
                        sources=list(item.sources),
                        okf_type=item.okf_type,
                        okf_version=1,
                        project=self._transcript_project(transcript),
                        validate_okf=True,
                    )
                node_payload = fallback.get("node") if isinstance(fallback, dict) else None
                node_id = node_payload.get("id") if isinstance(node_payload, dict) else None
                return {"action": "guard_fallback_create", "node_ids": [node_id] if node_id else []}

            decision = result.get("decision", {})
            action = str(decision.get("action", ""))
            node_payload = result.get("execution", {}).get("result", {}).get("node", {})
            node_id = node_payload.get("id") if isinstance(node_payload, dict) else None
            if not action and result.get("action") in {"updated", "unchanged"}:
                action = str(result["action"])
                merged_node = result.get("node")
                node_id = merged_node.get("id") if isinstance(merged_node, dict) else None
            self._count_action(report, action)
            if decision_endpoint is not None:
                endpoint_actions = report.actions_by_endpoint.setdefault(decision_endpoint, {})
                endpoint_actions[action] = endpoint_actions.get(action, 0) + 1
            downgrade = result.get("downgrade")
            if isinstance(downgrade, dict):
                report.decider_downgrades += 1
                report.downgraded.append(
                    {
                        "transcript_id": transcript.id,
                        "item_title": item.title,
                        "reason": downgrade.get("reason", "supersede downgraded to complement (backfill safety)"),
                        "target_node_ids": downgrade.get("target_node_ids", []),
                    }
                )
            return {"action": action, "node_ids": [node_id] if node_id else []}

    @staticmethod
    def _count_action(report: DistillerReport, action: str) -> None:
        if action == "create":
            report.decider_creates += 1
        elif action == "complement":
            report.decider_complements += 1
        elif action == "supersede":
            report.decider_supersedes += 1
        elif action == "updated":
            report.merged_updates += 1

    # ------------------------------------------------------------------
    # Transcript state
    # ------------------------------------------------------------------

    def _stamp_transcript(
        self, transcript: Node, knowledge_ids: list[str], *, distilled_turn: int | None = None, mark_current: bool = True
    ) -> None:
        path = self.runtime_paths.base / transcript.file_path
        if not path.exists():
            return
        node = self._load_transcript_from_disk(transcript.id)
        if node is None:
            return
        merged_ids = list(dict.fromkeys([*node.metadata.distilled_node_ids, *knowledge_ids]))
        # Incremental watermark (P2c): only advance when this run covered
        # turns; a stamp without a watermark keeps the previous value.
        watermark = distilled_turn if distilled_turn is not None else node.metadata.distilled_turn
        metadata = node.metadata.model_copy(
            update={
                # mark_current=False (partial segment failure): leave the hash
                # unset so the next sweep re-selects the transcript and retries
                # the failed segment range from the held-back watermark.
                "distilled_hash": content_sha256(node.content) if mark_current else None,
                "distilled_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "distilled_node_ids": merged_ids,
                "distilled_turn": watermark,
            }
        )
        write_node_file(node.model_copy(update={"metadata": metadata}), base_path=self.runtime_paths.base)
        self._sync_transcript(path)

    def _mark_backoff(self, transcript: Node) -> None:
        """Skip this transcript for the next N sweeps by touching distilled_at only.

        The hash is NOT set, so the transcript remains eligible after backoff.
        """

        node = self._load_transcript_from_disk(transcript.id)
        if node is None:
            return
        metadata = node.metadata.model_copy(
            update={"distilled_at": datetime.now(UTC).isoformat().replace("+00:00", "Z")}
        )
        write_node_file(node.model_copy(update={"metadata": metadata}), base_path=self.runtime_paths.base)
        self._sync_transcript(self.runtime_paths.base / node.file_path)

    def _load_transcript_from_disk(self, node_id: str) -> Node | None:
        from synapse.storage import read_node_file

        for candidate in (self.runtime_paths.active / f"{node_id}.md",):
            if candidate.exists():
                return read_node_file(candidate)
        return None

    def _sync_transcript(self, path: Path) -> None:
        from synapse.sync import SyncManager

        manager = SyncManager(self.config, runtime_paths=self.runtime_paths, logger=self._logger, debounce_seconds=0.0)
        try:
            manager.sync_paths([str(path)])
        finally:
            manager.close()

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def _archive_expired(self, report: DistillerReport) -> int:
        """Archive transcripts distilled at their current revision, past retention_days.

        Gate is ``is_distilled_current``: a transcript that yielded zero
        knowledge items counts (the LLM judged nothing durable) and is
        archived like any other. Undistilled or stale-distilled transcripts
        are never archived (knowledge-loss guard).
        """

        store = self._get_store()
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=self.settings.retention_days)
        archived = 0
        for node in store.list_nodes({"status": NodeStatus.ACTIVE, "type": NodeType.TRANSIENT}):
            if not is_session_transcript(node) or not is_distilled_current(node):
                continue
            updated_at = self._content_updated_at(node)
            if updated_at is None or updated_at > cutoff:
                continue
            if self._archive_transcript(node):
                archived += 1
        return archived

    def _archive_transcript(self, node: Node) -> bool:
        """Archive via the same primitives the Dreamer uses.

        Move the file to the archive with ``archive_node_path``, then remove
        the index row through SyncManager's delete path (store.delete_node),
        mirroring how the Dreamer's file moves are reconciled by its delta
        ``startup_sync`` (file gone → delete event → delete_node).
        """

        source_path = self.runtime_paths.base / node.file_path
        if not source_path.exists():
            return False
        destination = archive_node_path(self.runtime_paths.archive, node.id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return False
        source_path.replace(destination)
        from synapse.sync import SyncManager

        manager = SyncManager(self.config, runtime_paths=self.runtime_paths, logger=self._logger, debounce_seconds=0.0)
        try:
            manager.queue_event("delete", source_path)
            manager.drain_pending(force=True)
        finally:
            manager.close()
        self._logger.info(
            "Archived distilled transcript",
            extra={
                "node_id": node.id,
                "reason": "distilled-retention",
                "source_path": str(source_path),
                "archive_path": str(destination),
            },
        )
        return True
