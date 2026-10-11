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
from dataclasses import dataclass, field
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from synapse.config import SynapseConfig
from synapse.models import Node, NodeStatus, NodeType
from synapse.okf import (
    TYPE_REQUIRED_SECTIONS,
    OkfItem,
    normalize_sources,
    parse_okf_items,
    parse_title_repair,
    render_distiller_prompt,
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
    guard_fallback_creates: int = 0
    title_repairs: int = 0
    title_repair_failures: int = 0
    decider_creates: int = 0
    decider_complements: int = 0
    decider_supersedes: int = 0
    decider_downgrades: int = 0
    llm_failures: int = 0
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

    # ------------------------------------------------------------------
    # Store / service access
    # ------------------------------------------------------------------

    def _get_store(self) -> SQLiteNodeStore:
        if self._store is None:
            self._store = SQLiteNodeStore(
                self.runtime_paths.base / "synapse.db",
                embedding_dimension=self.config.embedding.dimension or 0,
            )
        return self._store

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

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

        transcripts = self.select_transcripts(limit=limit or self.settings.max_transcripts_per_run)
        report.transcripts_scanned = len(transcripts)

        for transcript in transcripts:
            try:
                detail = self._distill_transcript(transcript, report, dry_run=dry_run, downgrade_supersede=downgrade)
                report.details.append(detail)
            except Exception as exc:  # noqa: BLE001 — one bad transcript must not stop the sweep
                report.llm_failures += 1
                report.transcripts_skipped += 1
                self._logger.warning(
                    "Distill failed for transcript; backing off",
                    extra={"node_id": transcript.id, "error": str(exc)},
                )
                if not dry_run:
                    self._mark_backoff(transcript)

        if not dry_run:
            report.archived_transcripts = self._archive_expired(report)

        completed = datetime.now(UTC)
        report.completed_at = completed.isoformat().replace("+00:00", "Z")
        report.duration_ms = int((completed - started).total_seconds() * 1000)
        if not dry_run:
            self._get_store().record_distiller_run(report.metrics())
        self._logger.info(
            "Distiller run complete",
            extra={
                "transcripts_distilled": report.transcripts_distilled,
                "items_extracted": report.items_extracted,
                "llm_failures": report.llm_failures,
                "archived_transcripts": report.archived_transcripts,
                "duration_ms": report.duration_ms,
            },
        )
        return report

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
            self._logger.warning("Distill failed for transcript", extra={"node_id": transcript.id, "error": str(exc)})
            if not dry_run:
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
        own_titles = self._titles_for_ids(transcript.metadata.distilled_node_ids)

        # One extraction call per segment (same call size as the old
        # single-call flow); candidates merge afterwards.
        candidates: list[OkfItem] = []
        extraction_failures = 0
        for segment_text, start_turn in segments:
            prompt = self._build_prompt(
                transcript_content=segment_text,
                start_turn=start_turn,
                own_titles=own_titles,
            )
            try:
                payload = self._sample_json(prompt)
                candidates.extend(parse_okf_items(payload))
            except Exception as exc:  # noqa: BLE001 — one bad segment must not lose the others
                extraction_failures += 1
                self._logger.warning(
                    "Segment extraction failed",
                    extra={"node_id": transcript.id, "start_turn": start_turn, "error": str(exc)},
                )
        if extraction_failures and not candidates:
            raise RuntimeError(f"All {extraction_failures} segment extraction(s) failed")

        candidates = self._reduce_candidates(transcript, candidates, own_titles, report, dry_run=dry_run)

        # Deterministic evidence verification (P2+, no LLM): quotes must
        # appear in the named turn's ELIGIBLE text only (docs/okf.md §9) —
        # user-turn bodies and tool result lines (call lines only for
        # procedure with an ok result). Assistant prose never verifies.
        for item in candidates:
            verified, unverified = verify_evidence(
                item.evidence,
                turns={turn.id: turn.eligible_evidence_text(okf_type=item.okf_type) for turn in parsed.turns},
            )
            item.evidence = [*verified, *unverified]
            item.verified = bool(item.evidence) and not unverified

        produced_ids: list[str] = []
        detail: dict[str, Any] = {"transcript_id": transcript.id, "items": [], "segments": len(segments)}

        service = self._get_service()
        token = service.push_sampling_client(self._get_sampling_client())
        try:
            for item in candidates:
                # Title rule (docs/okf.md §3): English titles so IDs stay
                # meaningful. One repair call for CJK/degenerate titles; never
                # drop the item — fallback ID + warning if repair fails.
                if self._title_needs_repair(item.title):
                    repaired = self._repair_title(item)
                    if repaired:
                        report.title_repairs += 1
                        item.title = repaired
                    else:
                        report.title_repair_failures += 1
                # The transcript id is always a server-verified source: merge it
                # into frontmatter even when the model omitted it from the body.
                item.sources = list(dict.fromkeys([*item.sources, transcript.id]))
                # Hygiene (docs/okf.md §2): drop pseudo-sources the LLM invented.
                resolve = self._get_store().get_node
                kept, dropped = normalize_sources(item.sources, resolve_node=lambda nid: resolve(nid) is not None)
                if dropped:
                    report.items_sources_dropped += len(dropped)
                    self._logger.info(
                        "Dropped non-conforming sources",
                        extra={"node_id": transcript.id, "item": item.title, "dropped": dropped},
                    )
                item.sources = kept
                action_note = self._write_item(service, transcript, item, report, dry_run=dry_run, downgrade_supersede=downgrade_supersede)
                produced_ids.extend(action_note.get("node_ids", []))
                detail["items"].append(
                    {"title": item.title, "okf_type": item.okf_type, "verified": item.verified, **action_note}
                )
        finally:
            service.reset_sampling_client(token)

        report.items_extracted += len(candidates)
        report.transcripts_distilled += 1

        if not dry_run:
            # The watermark covers everything parsed NOW (the client may have
            # appended turns while we distilled); late turns surface on the
            # next sweep via the content-hash gate.
            self._stamp_transcript(transcript, produced_ids, distilled_turn=parsed.last_turn_number)
            self._logger.info(
                "Distilled transcript",
                extra={
                    "node_id": transcript.id,
                    "segments": len(segments),
                    "items": len(candidates),
                    "types": sorted({item.okf_type for item in candidates}),
                    "knowledge_ids": produced_ids,
                    "distilled_turn": parsed.last_turn_number,
                },
            )
        return detail

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
                prompt=prompt, system_prompt="", max_tokens=400
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
            max_items=self.settings.max_items_per_transcript,
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
        """Merge/dedupe per-segment candidates down to max_items_per_transcript.

        LLM reduce call only when candidates exceed the cap or near-duplicate
        titles are detected (whitespace-collapsed prefix match); otherwise the
        deterministic path returns candidates unchanged — an LLM call cannot
        dedupe what is provably distinct, and skipping it keeps single-segment
        sweeps at exactly one extraction call.
        """

        max_items = self.settings.max_items_per_transcript
        if not candidates:
            return []

        def _norm(title: str) -> str:
            return " ".join(title.split()).casefold()

        duplicates_likely = any(
            _norm(a.title) == _norm(b.title) or _norm(a.title).startswith(_norm(b.title)) or _norm(b.title).startswith(_norm(a.title))
            for i, a in enumerate(candidates)
            for b in candidates[i + 1 :]
        )
        if len(candidates) <= max_items and not duplicates_likely:
            return candidates

        if dry_run:
            return candidates

        candidates_json = json.dumps(
            {"items": [{"okf_type": item.okf_type, "title": item.title, "takeaway": item.takeaway,
                        "sections": item.sections, "sources": item.sources,
                        "evidence": [{"turn": e.turn, "quote": e.quote} for e in item.evidence]}
                       for item in candidates]},
            ensure_ascii=False,
        )
        prompt = render_reduce_prompt(
            candidates_json=candidates_json,
            max_items=max_items,
            known_titles=self._known_titles(),
            own_distilled_titles=own_titles,
        )
        try:
            payload = self._sample_json(prompt)
            merged = parse_okf_items(payload)
        except Exception as exc:  # noqa: BLE001 — reduce failure keeps raw candidates (cap applied deterministically)
            self._logger.warning(
                "Reduce call failed; applying deterministic cap",
                extra={"node_id": transcript.id, "error": str(exc)},
            )
            report.reduce_failures += 1
            return candidates[:max_items]
        if not merged:
            report.reduce_failures += 1
            return candidates[:max_items]
        report.reduced_items += max(0, len(candidates) - len(merged))
        return merged

    def _titles_for_ids(self, node_ids: list[str]) -> list[str]:
        if not node_ids:
            return []
        nodes = self._get_store().get_nodes(node_ids)
        return [node.title for node in nodes]

    def _sample_json(self, prompt: str) -> dict[str, Any]:
        client = self._get_sampling_client()
        return client.sample_json(prompt=prompt, system_prompt="", max_tokens=self.settings.llm_max_tokens)

    # ------------------------------------------------------------------
    # Item → write path
    # ------------------------------------------------------------------

    def _write_item(
        self,
        service: SynapseServerService,
        transcript: Node,
        item: OkfItem,
        report: DistillerReport,
        *,
        dry_run: bool,
        downgrade_supersede: bool,
    ) -> dict[str, Any]:
        if dry_run:
            return {"action": "planned", "node_ids": []}
        # item.sources already includes transcript.id (merged in _distill_transcript).
        sources = list(item.sources)
        # Degenerate-title fallback (docs/okf.md §7): if the slug is still
        # meaningless after repair, use mem_<date>_<okf_type|node>_<8hex>.
        forced_node_id: str | None = None
        if not slugify_english_title(item.title):
            date_part = datetime.now(UTC).strftime("%Y%m%d")
            digest = hashlib.sha256(f"{item.title}{item.takeaway}".encode("utf-8")).hexdigest()[:8]
            type_part = item.okf_type if item.okf_type in ("decision", "fact", "procedure", "pitfall") else "node"
            forced_node_id = f"mem_{date_part}_{type_part}_{digest}"
        try:
            result = service.write_memory(
                title=item.title,
                content=item.render(),
                node_type=NodeType.PERSISTENT,
                sources=sources,
                okf_type=item.okf_type,
                okf_version=1,
                project=self._transcript_project(transcript),
                downgrade_supersede=downgrade_supersede,
                node_id=forced_node_id,
                route="distiller",
                verified=item.verified,
            )
        except SynapseServiceError as exc:
            guard_fired = (
                exc.code == "INVALID_SAMPLING_RESPONSE"
                and (
                    "never session transcripts" in exc.message
                    or "not present in the candidate set" in exc.message
                )
            )
            if guard_fired:
                # Backstop (should not happen: transcripts are excluded from
                # write-path candidates). Fall back to a deterministic create
                # so the item is NOT lost; log and count it.
                report.guard_fallback_creates += 1
                self._logger.warning(
                    "Guard fired on distilled item; falling back to create",
                    extra={"node_id": transcript.id, "item": item.title},
                )
                fallback = service.integrate_knowledge(
                    title=item.title,
                    content=item.render(),
                    node_type=NodeType.PERSISTENT,
                    sources=list(item.sources),
                    okf_type=item.okf_type,
                    okf_version=1,
                    project=self._transcript_project(transcript),
                )
                node_payload = fallback.get("node") if isinstance(fallback, dict) else None
                node_id = node_payload.get("id") if isinstance(node_payload, dict) else None
                return {"action": "guard_fallback_create", "node_ids": [node_id] if node_id else []}
            raise
        decision = result.get("decision", {})
        action = str(decision.get("action", ""))
        node_payload = result.get("execution", {}).get("result", {}).get("node", {})
        node_id = node_payload.get("id") if isinstance(node_payload, dict) else None
        self._count_action(report, action)
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

    # ------------------------------------------------------------------
    # Transcript state
    # ------------------------------------------------------------------

    def _stamp_transcript(self, transcript: Node, knowledge_ids: list[str], *, distilled_turn: int | None = None) -> None:
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
                "distilled_hash": content_sha256(node.content),
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
