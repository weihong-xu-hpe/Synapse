"""Shared transcript lifecycle primitives for the episode distiller."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from synapse.config import SynapseConfig
from synapse.models import Node, NodeStatus, NodeType
from synapse.okf import OkfItem
from synapse.okf.transcripts import (
    content_sha256,
    is_distilled_current,
    is_represented,  # noqa: F401  (re-exported for callers/tests)
    is_session_transcript,
)
from synapse.server.service import SynapseServiceError
from synapse.storage import DistillerRunMetrics, SQLiteNodeStore, archive_node_path
from synapse.storage.markdown import write_node_file
from synapse.utils.runtime import RuntimePaths

LOGGER = logging.getLogger("synapse.distiller")
_QUARANTINE_WRITE_LOCK = threading.Lock()


@dataclass(slots=True)
class DistillerReport:
    """Lifecycle report shape convertible to the existing metrics record."""

    started_at: str
    completed_at: str
    duration_ms: int = 0
    transcripts_scanned: int = 0
    transcripts_distilled: int = 0
    transcripts_skipped: int = 0
    items_extracted: int = 0
    items_invalid: int = 0
    items_quarantined: int = 0
    decider_creates: int = 0
    decider_complements: int = 0
    decider_supersedes: int = 0
    decider_downgrades: int = 0
    llm_failures: int = 0
    archived_transcripts: int = 0

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
            items_quarantined=self.items_quarantined,
            decider_creates=self.decider_creates,
            decider_complements=self.decider_complements,
            decider_supersedes=self.decider_supersedes,
            decider_downgrades=self.decider_downgrades,
            llm_failures=self.llm_failures,
            archived_transcripts=self.archived_transcripts,
        )


# Public predicate name used by transcript-state tests.
is_fully_distilled = is_distilled_current


class Distiller:
    """Transcript selection, validation quarantine, and retention primitives."""

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths
        self.settings = config.distiller
        self._logger = logger or LOGGER
        self._store_local = threading.local()

    def _get_store(self) -> SQLiteNodeStore:
        store = getattr(self._store_local, "store", None)
        if store is None:
            store = SQLiteNodeStore(
                self.runtime_paths.base / "synapse.db",
                embedding_dimension=self.config.embedding.dimension or 0,
            )
            self._store_local.store = store
        return store

    def _close_thread_store(self) -> None:
        store = getattr(self._store_local, "store", None)
        if store is not None:
            store.close()
            self._store_local.store = None

    def close(self) -> None:
        self._close_thread_store()

    def select_transcripts(self, *, limit: int | None = None) -> list[Node]:
        """Return idle, changed session transcripts, oldest update first."""

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
        candidates.sort(key=lambda node: self._content_updated_at(node) or now)
        return candidates if limit is None else candidates[:limit]

    def _content_updated_at(self, node: Node) -> datetime | None:
        """Return the transcript file mtime, or ``None`` if unavailable."""

        path = self.runtime_paths.base / node.file_path
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(mtime, tz=UTC)

    @staticmethod
    def _final_okf_validation_details(exc: Exception) -> dict[str, Any] | None:
        if not isinstance(exc, SynapseServiceError):
            return None
        if exc.code == "INVALID_OKF_FINAL_BODY":
            details = exc.details
        else:
            execution_error = exc.details.get("execution_error")
            if not isinstance(execution_error, Mapping) or execution_error.get("code") != "INVALID_OKF_FINAL_BODY":
                return None
            details = execution_error.get("details")
            if not isinstance(details, dict):
                details = {}
            original_decision = exc.details.get("decision")
            if isinstance(original_decision, Mapping):
                details = {**details, "decision": original_decision}
        if details.get("repair_error"):
            return None
        return details

    @classmethod
    def _is_final_okf_validation_error(cls, exc: Exception) -> bool:
        return cls._final_okf_validation_details(exc) is not None

    def _quarantine_invalid_item(
        self,
        transcript: Node,
        item: OkfItem,
        prepared: dict[str, Any] | None,
        exc: SynapseServiceError,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Path:
        details = self._final_okf_validation_details(exc) or {}
        prepared = prepared or {}
        context = context or {}
        decision = details.get("decision") if isinstance(details.get("decision"), Mapping) else None
        if decision is None:
            decision = ((prepared.get("decision_result") or {}).get("decision") or {})
        target_ids = list(decision.get("target_node_ids") or [])
        for target_id in context.get("target_node_ids", []):
            if target_id and target_id not in target_ids:
                target_ids.append(str(target_id))
        merge_target = context.get("merge_update_target")
        if merge_target and merge_target not in target_ids:
            target_ids.append(str(merge_target))
        final_body = details.get("final_content")
        if not isinstance(final_body, str):
            final_body = context.get("candidate_content")
        if not isinstance(final_body, str):
            final_body = prepared.get("content")
        if not isinstance(final_body, str):
            final_body = item.render()
        now = datetime.now(UTC)
        record = {
            "ts": now.isoformat().replace("+00:00", "Z"),
            "transcript_id": transcript.id,
            "title": item.title,
            "okf_type": item.okf_type,
            "intended_action": str(decision.get("action") or context.get("intended_action") or "create"),
            "target_node_ids": target_ids,
            "final_candidate_body": final_body,
            "validator_findings": details.get("warnings") or [],
            "sources": list(prepared.get("sources") or item.sources),
            "failure_reason": str(context.get("failure_reason") or exc),
            "evidence": [{"turn": entry.turn, "quote": entry.quote} for entry in item.evidence],
        }
        directory = self.runtime_paths.base / ".distiller-quarantine"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / f"{now.strftime('%Y%m%d')}.jsonl"
        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with _QUARANTINE_WRITE_LOCK:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short write while appending distiller quarantine record")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
        return path

    def _stamp_transcript(
        self,
        transcript: Node,
        knowledge_ids: list[str],
        *,
        distilled_turn: int | None = None,
        mark_current: bool = True,
    ) -> None:
        path = self.runtime_paths.base / transcript.file_path
        if not path.exists():
            return
        node = self._load_transcript_from_disk(transcript.id)
        if node is None:
            return
        merged_ids = list(dict.fromkeys([*node.metadata.distilled_node_ids, *knowledge_ids]))
        watermark = distilled_turn if distilled_turn is not None else node.metadata.distilled_turn
        metadata = node.metadata.model_copy(
            update={
                "distilled_hash": content_sha256(node.content) if mark_current else None,
                "distilled_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "distilled_node_ids": merged_ids,
                "distilled_turn": watermark,
            }
        )
        write_node_file(node.model_copy(update={"metadata": metadata}), base_path=self.runtime_paths.base)
        self._sync_transcript(path)

    def _load_transcript_from_disk(self, node_id: str) -> Node | None:
        from synapse.storage import read_node_file

        candidate = self.runtime_paths.active / f"{node_id}.md"
        return read_node_file(candidate) if candidate.exists() else None

    def _sync_transcript(self, path: Path) -> None:
        from synapse.sync import SyncManager

        manager = SyncManager(self.config, runtime_paths=self.runtime_paths, logger=self._logger, debounce_seconds=0.0)
        try:
            manager.sync_paths([str(path)])
        finally:
            manager.close()

    def _archive_expired(self, report: DistillerReport) -> int:
        """Archive current-revision transcripts after their retention window."""

        del report  # the scheduler records episode metrics separately
        store = self._get_store()
        cutoff = datetime.now(UTC) - timedelta(days=self.settings.retention_days)
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
        """Move a transcript to the archive and reconcile its index deletion."""

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
