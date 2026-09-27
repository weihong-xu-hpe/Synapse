"""Shared service layer for Synapse server operations."""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from synapse.embedding import create_embedding_engine
from synapse.indexing import collect_health_status
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType, SensitivityLevel, generate_node_id
from synapse.retrieval import RetrievalPipeline
from synapse.server.sampling import (
    MemoryWriteSamplingDecision,
    MemoryWriteSamplingRequest,
    SamplingCandidate,
    SamplingClient,
    build_memory_write_query,
    build_memory_write_sampling_prompt,
)
from synapse.storage import (
    SQLiteNodeStore,
    SearchEventMetrics,
    WriteMemoryEventMetrics,
    extract_wiki_links,
    split_frontmatter,
    write_node_file,
)
from synapse.server.write_path import IntegrateAction, normalize_reasoning
from synapse.sync import SyncBatchResult, SyncManager
from synapse.utils.runtime import RuntimePaths, bootstrap_runtime_directories


LOGGER = logging.getLogger("synapse.mcp-daemon")
_REDACTED_FIELDS = {"auth_token", "authorization", "content"}
_SUPERSEDES_REASON_PATTERN = re.compile(r"^> \*\*Supersedes\*\*: \[\[[^\]]+\]\](?: — (?P<reason>.+))?$")
_INVALID_TITLE_MESSAGE = "Node title must not be blank"
_ACTIVE_ARCHITECTURE_DOC = "docs/design/streamable-mcp-single-path-architecture.md"


def _query_cjk_ratio(text: str) -> float:
    """Share of CJK characters in a query — cheap corpus signal for search events."""

    if not text:
        return 0.0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff")
    return round(cjk / len(text), 4)


_SAMPLING_REQUIREMENTS_MESSAGE = (
    "This high-level tool requires a sampling-capable MCP host/client that can complete sampling/createMessage. "
    f"See {_ACTIVE_ARCHITECTURE_DOC}."
)


class SynapseServiceError(Exception):
    """Base service-layer error with structured API metadata."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_payload(self) -> dict[str, object]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class NodeNotFoundError(SynapseServiceError):
    def __init__(self, node_id: str) -> None:
        super().__init__(
            "NODE_NOT_FOUND",
            f"Node with id '{node_id}' does not exist",
            status_code=404,
            details={"node_id": node_id},
        )


class SyncFailedError(SynapseServiceError):
    def __init__(self, details: dict[str, object]) -> None:
        super().__init__(
            "SYNC_FAILED",
            "Markdown write completed, but the SQLite sync step failed",
            status_code=500,
            details=details,
        )


class SynapseServerService:
    """High-level operations exposed through the Synapse server layer."""
    def __init__(
        self,
        config,
        *,
        runtime_paths: RuntimePaths | None = None,
        logger: logging.Logger | None = None,
        sampling_client: SamplingClient | None = None,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths or bootstrap_runtime_directories(config)
        self.logger = logger or LOGGER
        self._sampling_client_var: contextvars.ContextVar[SamplingClient | None] = contextvars.ContextVar(
            f"synapse_sampling_client_{id(self)}",
            default=sampling_client,
        )
        # Serializes concurrent session-keyed upserts of the same key, and
        # unkeyed writes of the same (title, content-hash) pair.
        self._session_key_locks: dict[str, threading.Lock] = {}
        self._session_key_locks_guard = threading.Lock()
        self._content_write_locks: dict[str, threading.Lock] = {}

    @property
    def sampling_client(self) -> SamplingClient | None:
        return self._sampling_client_var.get()

    @sampling_client.setter
    def sampling_client(self, value: SamplingClient | None) -> None:
        self._sampling_client_var.set(value)

    def push_sampling_client(self, sampling_client: SamplingClient | None):
        return self._sampling_client_var.set(sampling_client)

    def reset_sampling_client(self, token: object) -> None:
        self._sampling_client_var.reset(token)

    def _normalizer_sampling_client(self):
        """Sampling client for write-path OKF normalization.

        Prefers the pushed MCP sampling client; falls back to the configured
        [decider] LLM (same client class the distiller uses).
        """

        if self.sampling_client is not None:
            return self.sampling_client
        from synapse.server.decider import LocalLLMDecider

        return LocalLLMDecider(self.config.decider)

    def search_memory(
        self,
        query: str,
        top_k: int = 3,
        *,
        exclude_session_key: str | None = None,
        include: str = "default",
        source: str = "api",
    ) -> dict[str, Any]:
        started = perf_counter()
        if not query.strip():
            raise SynapseServiceError("INVALID_QUERY", "Search query must not be blank")
        # Filtering happens inside the pipeline BEFORE scoring/top_k so
        # excluded nodes leave no holes (next ranked results fill the slots).
        excluded_id: str | None = None
        if exclude_session_key and exclude_session_key.strip():
            excluded_id = self._session_node_id(exclude_session_key.strip())
        response = self._run_retrieval_search(
            query,
            top_k=top_k,
            update_access=True,
            result_filter=lambda node: self._include_result(node, include) and node.id != excluded_id,
        )
        payload: dict[str, Any] = {
            "query": response.query,
            "top_k": top_k,
            "results": [self._serialize_retrieval_item(item) for item in response.results],
            "context": response.context,
        }
        self._record_search_event(
            query=query,
            source=source,
            exclude_session_key=exclude_session_key,
            top_k=top_k,
            include=include,
            latency_ms=(perf_counter() - started) * 1000.0,
            payload=payload,
        )
        self._log_tool_call("search_memory", {"query": query, "top_k": top_k, "include": include}, payload)
        return payload

    def _record_search_event(
        self,
        *,
        query: str,
        source: str,
        exclude_session_key: str | None,
        top_k: int,
        include: str,
        latency_ms: float,
        payload: dict[str, Any],
    ) -> None:
        """Persist one search usage event. Must never break the read path."""

        try:
            results = [
                {
                    "rank": rank,
                    "node_id": item.get("node_id"),
                    "type": (item.get("node") or {}).get("metadata", {}).get("type"),
                    "okf_type": (item.get("node") or {}).get("metadata", {}).get("okf_type"),
                    "rerank_logit": item.get("rerank_logit"),
                    "score": item.get("score"),
                    "inject": bool(item.get("inject")),
                }
                for rank, item in enumerate(payload.get("results", []), start=1)
            ]
            event_query = query.strip()
            with self._store() as store:
                store.record_search_event(
                    SearchEventMetrics(
                        ts=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        source=source,
                        session_hash=hashlib.sha1(exclude_session_key.strip().encode("utf-8")).hexdigest()[:12]
                        if exclude_session_key and exclude_session_key.strip()
                        else None,
                        query=event_query,
                        query_chars=len(event_query),
                        cjk_ratio=_query_cjk_ratio(event_query),
                        top_k=top_k,
                        include=include,
                        latency_ms=round(latency_ms, 3),
                        lexical_hit_count=None,
                        results=tuple(results),
                    )
                )
                store.prune_search_events(retention_days=self.config.observability.search_events_retention_days)
        except (OSError, ValueError, sqlite3.DatabaseError) as exc:  # metrics must never break reads
            self.logger.warning("Failed to record search event", exc_info=exc)

    def _include_result(self, node: Node, include: str) -> bool:
        """Default search excludes distilled-current transcripts.

        A transcript distilled at its current revision has already contributed
        whatever the LLM judged durable — represented transcripts via produced
        knowledge nodes, zero-item transcripts by explicit judgment ("nothing
        durable here"). Both compete with knowledge in default search, so both
        are excluded; only not-yet-distilled transcripts remain searchable.
        ``include=transcripts|all`` still returns everything. Legacy session
        summaries are covered by ``is_session_transcript``.
        """

        if include in {"all", "transcripts"}:
            return True
        from synapse.okf.transcripts import is_distilled_current, is_session_transcript

        if is_session_transcript(node) and is_distilled_current(node):
            return False
        return True

    def integrate_knowledge(
        self,
        title: str,
        content: str,
        node_type: NodeType | str = NodeType.TRANSIENT,
        links: list[str] | None = None,
        sensitivity: SensitivityLevel | str = SensitivityLevel.INTERNAL,
        action: IntegrateAction | str = IntegrateAction.CREATE,
        target_node_ids: list[str] | None = None,
        reasoning: str = "",
        okf_type: str | None = None,
        okf_version: int | None = None,
        sources: list[str] | None = None,
        project: str | None = None,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute an explicit memory write action decided by the higher-level orchestration layer.

        The internal canonical execution layer does not infer intent. It only
        applies the action chosen upstream after candidate retrieval, sampling,
        and validation are complete.
        """
        clean_title = title.strip()
        if not clean_title:
            raise SynapseServiceError("INVALID_TITLE", _INVALID_TITLE_MESSAGE)

        normalized_type = node_type if isinstance(node_type, NodeType) else NodeType(str(node_type))
        normalized_sensitivity = (
            sensitivity if isinstance(sensitivity, SensitivityLevel) else SensitivityLevel(str(sensitivity))
        )
        normalized_action = action if isinstance(action, IntegrateAction) else IntegrateAction(str(action))
        normalized_links = self._normalize_links(links or [])
        resolved_target_ids = self._normalize_links(target_node_ids or [])
        normalized_reasoning = normalize_reasoning(normalized_action, reasoning)
        node_id = node_id or self._allocate_node_id(clean_title)

        # Embed target IDs into content so graph edges are stored correctly by the
        # sync layer (supersession banners are stripped on read-back, so targets
        # must also appear in the body via wiki-links).
        if normalized_action is IntegrateAction.CREATE:
            all_links = normalized_links
        else:  # SUPERSEDE or COMPLEMENT — reference targets in the body too
            all_links = self._normalize_links([*normalized_links, *resolved_target_ids])
        linked_content = self._embed_links(content, all_links)

        # Validate that all referenced targets exist before writing anything.
        target_nodes = self._load_nodes(resolved_target_ids) if resolved_target_ids else []

        # Transcript guard: complement/supersede may only target persistent
        # knowledge nodes, never session transcripts (shared predicate covers
        # legacy "Session summary …" nodes too).
        if normalized_action in {IntegrateAction.SUPERSEDE, IntegrateAction.COMPLEMENT}:
            from synapse.okf.transcripts import is_session_transcript

            transcript_targets = [node.id for node in target_nodes if is_session_transcript(node)]
            if transcript_targets:
                raise SynapseServiceError(
                    "INVALID_TARGET",
                    "complement/supersede may only target persistent knowledge nodes, never session transcripts",
                    status_code=422,
                    details={"transcript_target_node_ids": transcript_targets},
                )

        metadata = NodeMetadata(
            id=node_id,
            title=clean_title,
            type=normalized_type,
            status=NodeStatus.ACTIVE,
            supersedes=resolved_target_ids if normalized_action is IntegrateAction.SUPERSEDE else [],
            sensitivity=normalized_sensitivity,
            okf_type=okf_type,
            okf_version=okf_version,
            sources=list(sources) if sources else [],
            project=project,
        )
        new_node = Node(
            metadata=metadata,
            content=linked_content,
            file_path=Path("active") / f"{node_id}.md",
        )

        updated_existing: list[Node] = []
        supersession_reasons: dict[str, str | None] = {}

        if normalized_action is IntegrateAction.SUPERSEDE:
            # Key by the NEW node's ID so write_node_file emits the annotation.
            supersession_reasons = {node_id: normalized_reasoning}
            updated_existing = [
                existing.model_copy(
                    update={
                        "metadata": existing.metadata.model_copy(
                            update={
                                "status": NodeStatus.SUPERSEDED,
                                "superseded_by": node_id,
                            }
                        )
                    }
                )
                for existing in target_nodes
            ]

        elif normalized_action is IntegrateAction.COMPLEMENT:
            updated_existing = [
                existing.model_copy(update={"content": updated_content})
                for existing in target_nodes
                for updated_content in [self._embed_links(existing.content, [node_id])]
                if updated_content != existing.content
            ]

        sync_result = self._persist_nodes(
            [new_node, *updated_existing],
            supersession_reasons=supersession_reasons,
        )
        stored_node = self._load_node(node_id)
        refreshed_updated = (
            self._load_nodes([node.id for node in updated_existing]) if updated_existing else []
        )

        payload = {
            "node": self._serialize_node(stored_node),
            "updated_nodes": [self._serialize_node(node) for node in refreshed_updated],
            "action": normalized_action.value,
            "target_node_ids": resolved_target_ids,
            "reasoning": normalized_reasoning,
            "sync": self._serialize_sync_result(sync_result),
        }
        self._log_tool_call(
            "integrate_knowledge",
            {
                "title": clean_title,
                "node_type": normalized_type.value,
                "links": normalized_links,
                "sensitivity": normalized_sensitivity.value,
                "action": normalized_action.value,
                "target_node_ids": resolved_target_ids,
                "reasoning": normalized_reasoning,
            },
            payload,
        )
        return payload

    def write_memory(
        self,
        title: str,
        content: str,
        node_type: NodeType | str | None = None,
        links: list[str] | None = None,
        sensitivity: SensitivityLevel | str = SensitivityLevel.INTERNAL,
        query_hint: str | None = None,
        similarity_threshold: float = 0.3,
        okf_type: str | None = None,
        okf_version: int | None = None,
        sources: list[str] | None = None,
        project: str | None = None,
        node_id: str | None = None,
        downgrade_supersede: bool = False,
        route: str | None = None,
    ) -> dict[str, Any]:
        clean_title = title.strip()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        # Hold the (title, content-hash) lock across guard check -> decider ->
        # integrate so two concurrent identical writes cannot both pass the
        # guard and both create a node.
        lock_key = f"{clean_title}::{content_hash}"
        lock = self._lock_for(self._content_write_locks, self._session_key_locks_guard, lock_key)
        with lock:
            guard = self._find_identical_active_node(clean_title, content)
            if guard is not None:
                payload = {
                    "node": self._serialize_node(guard),
                    "action": "unchanged",
                    "title": clean_title,
                }
                # Coverage fix (2026-09-27): the dedupe-guard early return
                # previously wrote NO write_memory_events row, silently
                # undercounting identical writes.
                self._record_route_write_event(
                    node_id=guard.id,
                    node_type=guard.metadata.type.value,
                    action="unchanged",
                    route=route,
                )
                self._log_tool_call(
                    "write_memory_dedupe_guard",
                    {"title": clean_title, "content_hash": content_hash[:12]},
                    payload,
                )
                return payload

            return self._write_memory_locked(
                title=clean_title,
                content=content,
                node_type=node_type,
                links=links,
                sensitivity=sensitivity,
                query_hint=query_hint,
                similarity_threshold=similarity_threshold,
                okf_type=okf_type,
                okf_version=okf_version,
                sources=sources,
                project=project,
                node_id=node_id,
                downgrade_supersede=downgrade_supersede,
                route=route,
            )

    def _find_identical_active_node(self, title: str, content: str) -> Node | None:
        """ACTIVE node with the same title and byte-identical content, if any."""

        with self._store() as store:
            candidates = [
                node
                for node in store.list_nodes({"status": NodeStatus.ACTIVE, "title": title})
                if node.content == content
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.metadata.created_at)

    def _write_memory_locked(
        self,
        *,
        title: str,
        content: str,
        node_type: NodeType | str | None,
        links: list[str] | None = None,
        sensitivity: SensitivityLevel | str = SensitivityLevel.INTERNAL,
        query_hint: str | None = None,
        similarity_threshold: float = 0.3,
        okf_type: str | None = None,
        okf_version: int | None = None,
        sources: list[str] | None = None,
        project: str | None = None,
        node_id: str | None = None,
        downgrade_supersede: bool = False,
        route: str | None = None,
    ) -> dict[str, Any]:
        clean_title = title.strip()
        warnings: list[dict[str, str]] = []

        # Write-path tightening (B1/B2): explicit type wins; omitted type
        # defaults to persistent when the body carries OKF structure. A
        # persistent write without okf_type is normalized (deterministic
        # template inference first, then ONE LLM call with a fidelity guard).
        node_type_str: str | None
        if node_type is None:
            node_type_str = None
        elif isinstance(node_type, NodeType):
            node_type_str = node_type.value
        else:
            node_type_str = str(node_type)

        from synapse.server.write_normalize import WritePathNormalizer, title_needs_repair

        normalizer = WritePathNormalizer(sampling_client=self._normalizer_sampling_client())
        normalized = normalizer.normalize(
            title=clean_title,
            content=content,
            node_type=node_type_str,
            okf_type=okf_type,
            sources=sources,
        )
        clean_title = normalized.title
        content = normalized.content
        okf_type = normalized.okf_type
        sources = normalized.sources if normalized.mode == "llm_normalized" else sources
        warnings.extend(normalized.warnings)
        if normalized.type_override is not None:
            node_type = normalized.type_override
        elif node_type_str is not None:
            node_type = node_type_str
        else:
            node_type = "persistent" if normalized.mode in {"template_inferred", "llm_normalized"} else "transient"

        normalized_type = node_type if isinstance(node_type, NodeType) else NodeType(str(node_type))
        section_count = sum(1 for line in content.splitlines() if line.lstrip().startswith("##"))
        if normalized_type is NodeType.PERSISTENT:
            warnings.extend(
                self._okf_write_warnings(
                    clean_title,
                    content,
                    okf_type=okf_type,
                    sources=sources,
                )
            )

        payload = self._decide_memory_write_payload(
            title=clean_title,
            content=content,
            node_type=node_type,
            links=links,
            sensitivity=sensitivity,
            query_hint=query_hint,
            similarity_threshold=similarity_threshold,
        )

        decision_payload = payload["decision"].copy()
        downgrade_note: dict[str, Any] | None = None
        # Downgrade scope (design A3): only supersede of CURATED knowledge —
        # persistent nodes without distiller provenance — is downgraded.
        # Distilled-vs-distilled supersede stays allowed so knowledge can be
        # updated by later distillations.
        curated_targets: list[str] = []
        if (
            downgrade_supersede
            and decision_payload.get("action") == "supersede"
            and decision_payload.get("target_node_ids")
        ):
            curated_targets = self._curated_supersede_targets(list(decision_payload["target_node_ids"]))
        if (
            downgrade_supersede
            and decision_payload.get("action") == "supersede"
            and curated_targets
        ):
            decision_payload["action"] = "complement"
            downgrade_note = {
                "original_action": "supersede",
                "target_node_ids": list(decision_payload["target_node_ids"]),
                "curated_target_node_ids": curated_targets,
                "reason": "supersede downgraded to complement (curated knowledge is never superseded without owner approval)",
            }
            warnings.append(
                {
                    "code": "okf_supersede_downgraded",
                    "message": f"Supersede of curated nodes {curated_targets} downgraded to complement.",
                }
            )

        try:
            integrate_result = self.integrate_knowledge(
                title=clean_title,
                content=content,
                node_type=node_type,
                links=links,
                sensitivity=sensitivity,
                action=decision_payload["action"],
                target_node_ids=decision_payload["target_node_ids"],
                reasoning=decision_payload["reasoning"],
                okf_type=okf_type,
                okf_version=okf_version,
                sources=sources,
                project=project,
                node_id=node_id,
            )
        except SynapseServiceError as exc:
            self._record_write_memory_metrics(
                node_id=None,
                node_type=normalized_type.value,
                decision_payload=decision_payload,
                evidence=payload["evidence"],
                similarity_threshold=similarity_threshold,
                warning_codes=[warning["code"] for warning in warnings],
                execution_succeeded=False,
                route=route,
            )
            raise SynapseServiceError(
                "EXECUTION_FAILED_AFTER_DECISION",
                "Sampling produced a decision, but the subsequent low-level write failed",
                status_code=exc.status_code,
                details={
                    "decision": decision_payload,
                    "evidence": payload["evidence"],
                    "execution_error": exc.to_payload()["error"],
                },
            ) from exc

        node_payload = integrate_result.get("node") if isinstance(integrate_result, dict) else None
        written_node_id = str(node_payload.get("id")) if isinstance(node_payload, dict) else None
        self._record_write_memory_metrics(
            node_id=written_node_id,
            node_type=normalized_type.value,
            decision_payload=decision_payload,
            evidence=payload["evidence"],
            similarity_threshold=similarity_threshold,
            warning_codes=[warning["code"] for warning in warnings],
            execution_succeeded=True,
            route=route,
        )

        result = {
            "decision": decision_payload,
            "evidence": payload["evidence"],
            "execution": {
                "executed": True,
                "tool": "integrate_knowledge",
                "result": integrate_result,
            },
            "warnings": warnings,
        }
        if downgrade_note is not None:
            result["downgrade"] = downgrade_note
        self._log_tool_call(
            "write_memory",
            {
                "title": title.strip(),
                "node_type": str(node_type),
                "links": links or [],
                "sensitivity": str(sensitivity),
                "query_hint": query_hint,
                "similarity_threshold": similarity_threshold,
            },
            result,
        )
        return result

    def run_dreamer(self, batch_size: int = 8) -> dict[str, Any]:
        sampling_client = self._require_sampling_client()
        from synapse.lifecycle import Dreamer

        dreamer = Dreamer(
            self.config,
            runtime_paths=self.runtime_paths,
            sampling_client=sampling_client,
            logger=self.logger,
        )
        try:
            report = dreamer.run(batch_size=batch_size)
        finally:
            dreamer.close()
        payload = report.to_dict()
        self._log_tool_call("run_dreamer", {"batch_size": batch_size}, payload)
        return payload

    def _lock_for(self, locks: dict[str, threading.Lock], guard: threading.Lock, key: str) -> threading.Lock:
        with guard:
            lock = locks.get(key)
            if lock is None:
                lock = threading.Lock()
                locks[key] = lock
            return lock

    @staticmethod
    def _session_node_id(session_key: str) -> str:
        return f"mem_session_{hashlib.sha1(session_key.encode('utf-8')).hexdigest()[:16]}"

    def upsert_session_memory(
        self,
        *,
        session_key: str,
        title: str,
        content: str,
        node_type: NodeType | str = NodeType.TRANSIENT,
        links: list[str] | None = None,
        sensitivity: SensitivityLevel | str = SensitivityLevel.INTERNAL,
    ) -> dict[str, Any]:
        """Deterministic keyed upsert (no LLM decider).

        The node id is derived from the session key, so repeated writes for the
        same session converge on one node: absent -> create, identical
        content+title -> unchanged, different -> overwrite in place (markdown +
        index + re-embed through the normal sync path).
        """

        clean_key = session_key.strip()
        if not clean_key:
            raise SynapseServiceError("INVALID_SESSION_KEY", "session_key must not be blank")
        clean_title = title.strip()
        if not clean_title:
            raise SynapseServiceError("INVALID_TITLE", _INVALID_TITLE_MESSAGE)

        node_id = self._session_node_id(clean_key)
        lock = self._lock_for(self._session_key_locks, self._session_key_locks_guard, node_id)
        with lock:
            normalized_type = node_type if isinstance(node_type, NodeType) else NodeType(str(node_type))
            normalized_sensitivity = (
                sensitivity if isinstance(sensitivity, SensitivityLevel) else SensitivityLevel(str(sensitivity))
            )
            normalized_links = self._normalize_links(links or [])

            existing_path = self.runtime_paths.active / f"{node_id}.md"
            existing: Node | None = None
            if existing_path.exists():
                existing = self._load_node(node_id)

            if existing is not None and existing.title == clean_title and existing.content == content:
                action = "unchanged"
                stored_node = existing
                sync_result: SyncBatchResult | None = None
            elif existing is not None:
                action = "updated"
                metadata = existing.metadata.model_copy(
                    update={"title": clean_title, "session_key": clean_key}
                )
                node = existing.model_copy(
                    update={"content": content, "metadata": metadata}
                )
                absolute_path = write_node_file(node, base_path=self.runtime_paths.base)
                sync_result = self._sync_paths([absolute_path])
                stored_node = self._load_node(node_id)
            else:
                action = "created"
                metadata = NodeMetadata(
                    id=node_id,
                    title=clean_title,
                    type=normalized_type,
                    sensitivity=normalized_sensitivity,
                    session_key=clean_key,
                )
                node = Node(
                    metadata=metadata,
                    content=self._embed_links(content, normalized_links),
                    file_path=Path("active") / f"{node_id}.md",
                )
                absolute_path = write_node_file(node, base_path=self.runtime_paths.base)
                sync_result = self._sync_paths([absolute_path])
                stored_node = self._load_node(node_id)

            self._record_session_upsert_metric(node_id, normalized_type.value, action)
            payload = {
                "node": self._serialize_node(stored_node),
                "action": action,
                "session_key": clean_key,
            }
            if sync_result is not None:
                payload["sync"] = self._serialize_sync_result(sync_result)
            self._log_tool_call(
                "upsert_session_memory",
                {"session_key": clean_key, "title": clean_title},
                payload,
            )
            return payload

    def _record_session_upsert_metric(self, node_id: str, node_type: str, action: str) -> None:
        self._record_route_write_event(node_id=node_id, node_type=node_type, action=action, route="session_upsert")

    def _record_route_write_event(
        self,
        *,
        node_id: str | None,
        node_type: str,
        action: str,
        route: str | None,
    ) -> None:
        """Record a lightweight write event with route attribution (no decider evidence)."""

        try:
            with self._store() as store:
                store.record_write_memory_event(
                    WriteMemoryEventMetrics(
                        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        node_id=node_id,
                        node_type=node_type,
                        action=action,
                        candidate_count=0,
                        similarity_threshold=0.0,
                        warning_codes=(),
                        sampling_provider="",
                        execution_succeeded=True,
                        route=route,
                    )
                )
        except (OSError, ValueError, sqlite3.DatabaseError) as exc:  # metrics must never break writes
            self.logger.warning("Failed to record write-route metrics", exc_info=exc)

    def write_node(
        self,
        title: str,
        content: str,
        node_type: NodeType | str = NodeType.TRANSIENT,
        links: list[str] | None = None,
        sensitivity: SensitivityLevel | str = SensitivityLevel.INTERNAL,
    ) -> dict[str, Any]:
        clean_title = title.strip()
        if not clean_title:
            raise SynapseServiceError("INVALID_TITLE", _INVALID_TITLE_MESSAGE)

        normalized_type = node_type if isinstance(node_type, NodeType) else NodeType(str(node_type))
        normalized_sensitivity = (
            sensitivity if isinstance(sensitivity, SensitivityLevel) else SensitivityLevel(str(sensitivity))
        )
        normalized_links = self._normalize_links(links or [])
        node_id = self._allocate_node_id(clean_title)
        merged_content = self._embed_links(content, normalized_links)
        metadata = NodeMetadata(
            id=node_id,
            title=clean_title,
            type=normalized_type,
            sensitivity=normalized_sensitivity,
        )
        relative_path = Path("active") / f"{node_id}.md"
        node = Node(metadata=metadata, content=merged_content, file_path=relative_path)
        absolute_path = write_node_file(node, base_path=self.runtime_paths.base)
        sync_result = self._sync_paths([absolute_path])
        stored_node = self._load_node(node_id)
        self._record_route_write_event(
            node_id=node_id,
            node_type=normalized_type.value,
            action="create",
            route="write_node",
        )
        payload = {
            "node": self._serialize_node(stored_node),
            "sync": self._serialize_sync_result(sync_result),
        }
        self._log_tool_call(
            "write_node",
            {
                "title": clean_title,
                "node_type": normalized_type.value,
                "links": normalized_links,
                "sensitivity": normalized_sensitivity.value,
                "content": merged_content,
            },
            payload,
        )
        return payload

    def search_existing_nodes(self, query: str, similarity_threshold: float = 0.3) -> dict[str, Any]:
        if not query.strip():
            raise SynapseServiceError("INVALID_QUERY", "Search query must not be blank")

        payload = self._search_existing_nodes_payload(query, similarity_threshold)
        self._log_tool_call(
            "search_existing_nodes",
            {"query": query, "similarity_threshold": similarity_threshold},
            payload,
        )
        return payload

    def update_node_status(
        self,
        node_id: str,
        status: NodeStatus | str,
        superseded_by: str | None = None,
    ) -> dict[str, Any]:
        stored_node = self._load_node(node_id)
        normalized_status = status if isinstance(status, NodeStatus) else NodeStatus(str(status))
        normalized_superseded_by = superseded_by.strip() if superseded_by and superseded_by.strip() else None
        updated_metadata = stored_node.metadata.model_copy(
            update={
                "status": normalized_status,
                "superseded_by": normalized_superseded_by if normalized_status is NodeStatus.SUPERSEDED else None,
            }
        )
        updated_node = stored_node.model_copy(update={"metadata": updated_metadata})
        sync_result = self._persist_nodes([updated_node])
        refreshed = self._load_node(node_id)
        payload = {
            "node": self._serialize_node(refreshed),
            "sync": self._serialize_sync_result(sync_result),
        }
        self._log_tool_call(
            "update_node_status",
            {"node_id": node_id, "status": normalized_status.value, "superseded_by": normalized_superseded_by},
            payload,
        )
        return payload

    def get_node(self, node_id: str) -> dict[str, Any]:
        node = self._load_node(node_id)
        payload = self._serialize_node(node)
        self._log_tool_call("get_node", {"node_id": node_id}, payload)
        return payload

    def health(self) -> dict[str, Any]:
        health = collect_health_status(self.config, runtime_paths=self.runtime_paths)
        payload = {
            "status": health.status,
            "components": health.components,
            "warnings": health.warnings,
        }
        self._log_tool_call("health", {}, payload)
        return payload

    def stats(self) -> dict[str, Any]:
        health = collect_health_status(self.config, runtime_paths=self.runtime_paths)
        payload = {
            "status": health.status,
            "components": health.components,
            "stats": health.stats,
            "lifecycle_stats": health.lifecycle_stats,
            "write_stats": health.write_stats,
            "warnings": health.warnings,
            "database_path": health.database_path.as_posix() if health.database_path is not None else None,
            "embedding_fingerprint": health.embedding_fingerprint,
        }
        self._log_tool_call("stats", {}, payload)
        return payload

    def _load_node(self, node_id: str) -> Node:
        with self._store() as store:
            node = store.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)
        return node

    def _load_nodes(self, node_ids: list[str]) -> list[Node]:
        requested_ids = self._normalize_links(node_ids)
        if not requested_ids:
            return []
        with self._store() as store:
            nodes = {node.id: node for node in store.get_nodes(requested_ids)}
        missing = [node_id for node_id in requested_ids if node_id not in nodes]
        if missing:
            raise SynapseServiceError(
                "NODE_NOT_FOUND",
                "One or more referenced nodes could not be loaded",
                status_code=404,
                details={"missing_node_ids": missing},
            )
        return [nodes[node_id] for node_id in requested_ids]

    def _store(self) -> SQLiteNodeStore:
        return SQLiteNodeStore(
            self.runtime_paths.base / "synapse.db",
            embedding_dimension=self.config.embedding.dimension or 0,
        )

    def _require_sampling_client(self) -> SamplingClient:
        if self.sampling_client is None:
            raise SynapseServiceError(
                "SAMPLING_UNAVAILABLE",
                _SAMPLING_REQUIREMENTS_MESSAGE,
                status_code=501,
                details={
                    "sampling_required": True,
                    "supported_interfaces": ["mcp"],
                    "required_capability": "sampling",
                    "architecture_doc": _ACTIVE_ARCHITECTURE_DOC,
                },
            )
        return self.sampling_client

    def _allocate_node_id(self, title: str) -> str:
        base_id = generate_node_id(title)
        candidate = base_id
        with self._store() as store:
            if store.get_node(candidate) is None and not (self.runtime_paths.active / f"{candidate}.md").exists():
                return candidate
            # Collision: deterministic short hash of title + creation date
            # (docs/okf.md ID plan) instead of _2/_3 counters; extend from 4
            # to 6 hex only if the 4-hex variant is still taken.
            now_iso = datetime.now(UTC).isoformat()
            digest = hashlib.sha1(f"{title}{now_iso}".encode("utf-8")).hexdigest()
            candidate = f"{base_id}_{digest[:4]}"
            if store.get_node(candidate) is not None or (self.runtime_paths.active / f"{candidate}.md").exists():
                candidate = f"{base_id}_{digest[:6]}"
        return candidate

    def _embed_query(self, query: str) -> list[float] | None:
        engine = create_embedding_engine(self.config.embedding, providers=self.config.providers)
        backend = getattr(engine, "backend_name", "")
        if backend == "unavailable":
            return None
        try:
            vector = engine.embed(query)
        except (OSError, RuntimeError, ValueError):
            return None
        if not vector:
            return None
        if backend == "remote_api" and not engine.is_available():
            # Degraded batch → deterministic hash vector; must not be mixed
            # with real provider vectors (see sync/manager._embed_for_persistence).
            return None
        return vector

    def _run_retrieval_search(self, query: str, *, top_k: int, update_access: bool, result_filter=None) -> Any:
        with RetrievalPipeline(self.config, runtime_paths=self.runtime_paths) as pipeline:
            return pipeline.search(query, top_k=top_k, update_access=update_access, result_filter=result_filter)

    def _search_existing_nodes_payload(self, query: str, similarity_threshold: float) -> dict[str, Any]:
        normalized_query = " ".join(str(query or "").split()).strip()
        queries = [normalized_query] if normalized_query else []
        matches = self._collect_candidate_matches(
            queries,
            similarity_threshold=similarity_threshold,
            limit=self.config.reranker.max_candidates,
        )
        return {
            "query": normalized_query,
            "queries": queries,
            "similarity_threshold": similarity_threshold,
            "matches": matches,
        }

    @staticmethod
    def _normalize_candidate_score(score: float) -> float:
        # Reranker scores are already in [0, 1]; clamp instead of compressing.
        # The previous x/(1+x) mapping collapsed [0,1] -> [0,0.5], which combined
        # with similarity_threshold=0.5 filtered out nearly every candidate.
        return round(min(1.0, max(0.0, float(score))), 6)

    def _collect_candidate_matches(
        self,
        queries: list[str],
        *,
        similarity_threshold: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        normalized_queries = self._normalize_links([" ".join(str(query).split()) for query in queries])
        if not normalized_queries:
            return []

        best_matches: dict[str, dict[str, Any]] = {}
        per_query_limit = max(limit, self.config.retrieval.top_k)

        # Write-path candidates are knowledge only: session transcripts are
        # excluded BEFORE scoring/top_k so the decider compares a draft against
        # knowledge nodes, never against the transcript the draft came from.
        from synapse.okf.transcripts import is_session_transcript

        with RetrievalPipeline(self.config, runtime_paths=self.runtime_paths) as pipeline:
            for query in normalized_queries:
                response = pipeline.search(
                    query,
                    top_k=per_query_limit,
                    update_access=False,
                    result_filter=lambda node: not is_session_transcript(node),
                )
                candidate_items = response.candidates or response.results
                for item in candidate_items:
                    normalized_score = self._normalize_candidate_score(item.score)
                    if normalized_score < similarity_threshold:
                        continue
                    existing = best_matches.get(item.node.id)
                    if existing is None or float(existing["raw_score"]) < float(item.score):
                        best_matches[item.node.id] = {
                            "node": item.node,
                            "raw_score": float(item.score),
                            "score": normalized_score,
                            "matched_queries": [query],
                        }
                        continue
                    if query not in existing["matched_queries"]:
                        existing["matched_queries"].append(query)

        ranked_matches = sorted(
            best_matches.values(),
            key=lambda item: (-float(item["raw_score"]), str(item["node"].id)),
        )[:limit]
        return [
            self._serialize_existing_match(
                item["node"],
                float(item["score"]),
                matched_queries=list(item["matched_queries"]),
            )
            for item in ranked_matches
        ]

    def _build_sampling_candidate_queries(
        self,
        *,
        title: str,
        content: str,
        query_hint: str | None,
    ) -> tuple[str, list[str]]:
        base_query = build_memory_write_query(title=title, content=content, query_hint=None)
        normalized_hint = " ".join(str(query_hint or "").split()).strip()
        queries = self._normalize_links([base_query, normalized_hint])
        return base_query, queries or [base_query]

    def _okf_write_warnings(
        self,
        title: str,
        content: str,
        *,
        okf_type: str | None = None,
        sources: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Typed OKF validation for persistent writes (warnings-only transition).

        Explicit okf fields (e.g. from the distiller or a curated client) take
        precedence; otherwise an existing node with the same title/content
        provides them; otherwise the write is untyped. Submitted sources are
        checked against the conforming shapes (node id / session:<key> / URL);
        non-conforming entries surface as ``okf_unresolvable_source``. The
        title is checked against the English-title rules (warnings only for
        agent writes — no auto-translation, no rejection).
        """

        from synapse.okf import validate_okf_node

        if okf_type is None or sources is None:
            existing = self._find_identical_active_node(title, content)
            if existing is not None:
                okf_type = okf_type or existing.metadata.okf_type
                sources = sources if sources is not None else existing.metadata.sources
        findings = validate_okf_node(node_type="persistent", content=content, okf_type=okf_type, sources=sources, title=title)
        return [{"code": w.code, "message": w.message} for w in findings]

    def _decide_memory_write_payload(
        self,
        *,
        title: str,
        content: str,
        node_type: NodeType | str,
        links: list[str] | None,
        sensitivity: SensitivityLevel | str,
        query_hint: str | None,
        similarity_threshold: float,
    ) -> dict[str, Any]:
        clean_title = title.strip()
        if not clean_title:
            raise SynapseServiceError("INVALID_TITLE", _INVALID_TITLE_MESSAGE)

        normalized_type = node_type if isinstance(node_type, NodeType) else NodeType(str(node_type))
        normalized_sensitivity = (
            sensitivity if isinstance(sensitivity, SensitivityLevel) else SensitivityLevel(str(sensitivity))
        )
        normalized_links = self._normalize_links(links or [])

        query, candidate_queries = self._build_sampling_candidate_queries(
            title=clean_title,
            content=content,
            query_hint=query_hint,
        )
        match_payload = {
            "query": query,
            "queries": candidate_queries,
            "similarity_threshold": similarity_threshold,
            "matches": self._collect_candidate_matches(
                candidate_queries,
                similarity_threshold=similarity_threshold,
                limit=self.config.reranker.max_candidates,
            ),
        }
        matches = list(match_payload["matches"])
        candidate_ids = [str(match["node_id"]) for match in matches]
        fetched_nodes = self._load_nodes(candidate_ids[:3]) if candidate_ids else []
        sampling_client = self._require_sampling_client()

        sampling_request = MemoryWriteSamplingRequest(
            prompt="",
            title=clean_title,
            content=content,
            node_type=normalized_type.value,
            sensitivity=normalized_sensitivity.value,
            query=query,
            similarity_threshold=similarity_threshold,
            links=tuple(normalized_links),
            candidates=tuple(
                SamplingCandidate(
                    node_id=str(match["node_id"]),
                    title=str(match["title"]),
                    score=float(match["score"]),
                    file_path=str(match["file_path"]),
                    status=str(match["status"]),
                    sensitivity=str(match["sensitivity"]),
                )
                for match in matches
            ),
            candidate_nodes=tuple(fetched_nodes),
        )
        sampling_request = MemoryWriteSamplingRequest(
            prompt=build_memory_write_sampling_prompt(sampling_request),
            title=sampling_request.title,
            content=sampling_request.content,
            node_type=sampling_request.node_type,
            sensitivity=sampling_request.sensitivity,
            query=sampling_request.query,
            similarity_threshold=sampling_request.similarity_threshold,
            links=sampling_request.links,
            candidates=sampling_request.candidates,
            candidate_nodes=sampling_request.candidate_nodes,
        )

        try:
            raw_decision = sampling_client.decide_memory_write(sampling_request)
        except SynapseServiceError:
            raise
        except Exception as exc:
            raise SynapseServiceError(
                "SAMPLING_FAILED",
                "Sampling-capable host/client failed to produce a decision",
                status_code=502,
                details={"sampling_provider": sampling_client.name, "reason": str(exc)},
            ) from exc

        decision = self._normalize_sampling_decision(raw_decision, candidate_ids)
        return {
            "decision": self._serialize_sampling_decision(decision),
            "evidence": {
                "query": query,
                "queries": candidate_queries,
                "similarity_threshold": similarity_threshold,
                "candidate_count": len(matches),
                "candidates": matches,
                "fetched_full_nodes": [node.id for node in fetched_nodes],
                "sampling_provider": sampling_client.name,
            },
        }

    def _curated_supersede_targets(self, target_node_ids: list[str]) -> list[str]:
        """Supersede targets that are CURATED knowledge (persistent, no distiller provenance).

        A node is distiller-produced when any ``sources`` entry points at a
        session transcript (``mem_session_*`` id or a legacy session-summary
        node id / session reference). Such distilled nodes may supersede each
        other; curated nodes may not be superseded by backfill.
        """

        with self._store() as store:
            nodes = store.get_nodes(target_node_ids)
        curated: list[str] = []
        for node in nodes:
            if node.metadata.type is not NodeType.PERSISTENT:
                continue
            if self._has_distiller_provenance(node):
                continue
            curated.append(node.id)
        return curated

    @staticmethod
    def _has_distiller_provenance(node: Node) -> bool:
        """True when the node's sources point at session transcripts.

        Matches the shared transcript id patterns — keyed ``mem_session_*``,
        legacy ``mem_<date>_session_summary_*`` — regardless of whether the
        transcript still exists on disk (it may already be archived), plus
        ``session:<key>`` references emitted by the distiller prompt.
        """

        from synapse.okf.transcripts import SESSION_ID_PREFIX, LEGACY_ID_PATTERN

        for source in node.metadata.sources:
            source_id = source.strip()
            if source_id.startswith(SESSION_ID_PREFIX):
                return True
            if LEGACY_ID_PATTERN.match(source_id):
                return True
            # "session:<key>" style references emitted by the distiller prompt.
            if source_id.casefold().startswith("session:"):
                return True
        return False

    def _transcript_targets(self, target_node_ids: tuple[str, ...]) -> list[str]:
        """Target ids that are session transcripts (never valid complement/supersede targets).

        Uses the shared predicate so legacy ``Session summary …`` transcripts
        (no ``mem_session_`` prefix, no ``session_key``) are caught too.
        """

        from synapse.okf.transcripts import is_session_transcript

        with self._store() as store:
            nodes = store.get_nodes(list(target_node_ids))
        return [node.id for node in nodes if is_session_transcript(node)]

    def _normalize_sampling_decision(
        self,
        decision: MemoryWriteSamplingDecision,
        candidate_ids: list[str],
    ) -> MemoryWriteSamplingDecision:
        try:
            action = decision.action if isinstance(decision.action, IntegrateAction) else IntegrateAction(str(decision.action))
        except ValueError as exc:
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                "Sampling returned an unsupported action",
                status_code=502,
                details={"action": getattr(decision, "action", None)},
            ) from exc

        target_node_ids = tuple(self._normalize_links(list(decision.target_node_ids)))
        invalid_targets = [node_id for node_id in target_node_ids if node_id not in candidate_ids]
        if invalid_targets:
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                "Sampling targeted nodes that were not present in the candidate set",
                status_code=502,
                details={"invalid_target_node_ids": invalid_targets, "candidate_ids": candidate_ids},
            )

        transcript_targets = self._transcript_targets(target_node_ids)
        if transcript_targets:
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                "complement/supersede may only target persistent knowledge nodes, never session transcripts",
                status_code=502,
                details={
                    "transcript_target_node_ids": transcript_targets,
                    "candidate_ids": candidate_ids,
                    "action": action.value,
                },
            )

        if action is IntegrateAction.CREATE and target_node_ids:
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                "Sampling returned target_node_ids for a create action",
                status_code=502,
                details={"target_node_ids": list(target_node_ids)},
            )
        if action in {IntegrateAction.SUPERSEDE, IntegrateAction.COMPLEMENT} and not target_node_ids:
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                "Sampling must supply target_node_ids for complement or supersede actions",
                status_code=502,
                details={"action": action.value},
            )

        confidence = decision.confidence
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                "Sampling confidence must be between 0 and 1",
                status_code=502,
                details={"confidence": confidence},
            )

        return MemoryWriteSamplingDecision(
            action=action,
            target_node_ids=target_node_ids,
            reasoning=normalize_reasoning(action, decision.reasoning),
            confidence=None if confidence is None else float(confidence),
        )

    @staticmethod
    def _serialize_sampling_decision(decision: MemoryWriteSamplingDecision) -> dict[str, Any]:
        return {
            "action": decision.action.value,
            "target_node_ids": list(decision.target_node_ids),
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
        }

    def _sync_paths(self, paths: list[Path]) -> SyncBatchResult:
        manager = SyncManager(self.config, runtime_paths=self.runtime_paths, logger=self.logger, debounce_seconds=0.0)
        try:
            result = manager.sync_paths([str(path) for path in paths])
        finally:
            manager.close()
        if result.failed:
            raise SyncFailedError(self._serialize_sync_result(result))
        return result

    def _persist_nodes(
        self,
        nodes: list[Node],
        *,
        supersession_reasons: dict[str, str | None] | None = None,
    ) -> SyncBatchResult:
        if not nodes:
            return SyncBatchResult(backend="polling")

        written_paths: list[Path] = []
        reason_overrides = supersession_reasons or {}
        for node in nodes:
            absolute_path = self.runtime_paths.base / node.file_path
            supersession_reason = reason_overrides.get(node.id)
            if supersession_reason is None and node.metadata.supersedes and absolute_path.exists():
                supersession_reason = self._read_existing_supersession_reason(absolute_path)
            write_node_file(node, output_path=absolute_path, supersession_reason=supersession_reason)
            written_paths.append(absolute_path)
        return self._sync_paths(written_paths)

    def _read_existing_supersession_reason(self, absolute_path: Path) -> str | None:
        try:
            markdown = absolute_path.read_text(encoding="utf-8")
        except OSError:
            return None

        _frontmatter, body = split_frontmatter(markdown)
        lines = body.replace("\r\n", "\n").splitlines()

        for line in lines:
            if not line.startswith(">"):
                if line.strip():
                    break
                continue
            match = _SUPERSEDES_REASON_PATTERN.match(line)
            if match is None:
                continue
            reason = match.group("reason")
            return reason.strip() if reason else None
        return None

    def _serialize_node(self, node: Node) -> dict[str, Any]:
        return {
            "id": node.id,
            "title": node.title,
            "content": node.content,
            "file_path": node.file_path.as_posix(),
            "links": extract_wiki_links(node.content),
            "metadata": node.metadata.model_dump(mode="json"),
        }

    def _serialize_retrieval_item(self, item) -> dict[str, Any]:
        return {
            "node_id": item.node.id,
            "title": item.node.title,
            "score": round(float(item.score), 6),
            "anchor_score": round(float(item.anchor_score), 6),
            "rerank_score": round(float(item.rerank_score), 6),
            "rerank_logit": round(float(item.rerank_score), 6),
            "is_anchor": item.is_anchor,
            # Server-side inject decision (see RetrievalSettings inject_* keys).
            # Clients inject on this; ``score`` semantics stay unchanged.
            "inject": bool(item.is_injectable),
            "file_path": item.node.file_path.as_posix(),
            "status": item.node.metadata.status.value,
            "markers": list(item.markers),
            "node": self._serialize_node(item.node),
        }

    def _serialize_existing_match(
        self,
        node: Node,
        score: float,
        *,
        matched_queries: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "node_id": node.id,
            "title": node.title,
            "score": round(float(score), 6),
            "file_path": node.file_path.as_posix(),
            "status": node.metadata.status.value,
            "sensitivity": node.metadata.sensitivity.value,
        }
        if matched_queries:
            payload["matched_queries"] = matched_queries
        return payload

    @staticmethod
    def _serialize_sync_result(result: SyncBatchResult) -> dict[str, Any]:
        return {
            "upserted": result.upserted,
            "deleted": result.deleted,
            "failed": result.failed,
            "queued": result.queued,
            "backend": result.backend,
            "details": list(result.details),
        }

    @staticmethod
    def _normalize_links(links: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for item in links:
            cleaned = str(item).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @staticmethod
    def _normalize_fts_query(query: str) -> str:
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        normalized = " ".join(tokens).strip()
        return normalized or query.strip()

    @staticmethod
    def merge_related_sections(content: str) -> str:
        """Merge every ``## Related`` section into one trailing section.

        Historical `_embed_links` appended a NEW ``## Related`` block on every
        complement, so active nodes accumulated duplicates (112 nodes, one
        transcript with 30). This normalizer keeps a single trailing section
        with deduplicated ``[[id]]`` bullets in order of first appearance;
        all other content (and its order) is preserved verbatim.
        """

        text = content.strip()
        if "## Related" not in text:
            return text
        # Remove every '## Related' section: the heading plus following bullet
        # lines (and intra-section blank lines). A non-bullet, non-blank line
        # ends the section, so interleaved body text is preserved.
        pattern = re.compile(r"(?ms)^## Related\s*\n(?:(?:[ \t]*-[ \t][^\n]*\n?|[ \t]*\n(?![ \t]*\n))*)")
        bullets: list[str] = []
        seen: set[str] = set()
        non_related = pattern.sub("", text)
        for match in pattern.finditer(text):
            for line in match.group(0).splitlines()[1:]:
                stripped = line.strip()
                if not stripped:
                    continue
                bullet = stripped if stripped.startswith("- ") else f"- {stripped}"
                if bullet not in seen:
                    seen.add(bullet)
                    bullets.append(bullet)
        cleaned = re.sub(r"\n{3,}", "\n\n", non_related).strip()
        if not bullets:
            return cleaned
        related_block = "## Related\n" + "\n".join(bullets)
        if not cleaned:
            return related_block
        return f"{cleaned}\n\n{related_block}"

    def _embed_links(self, content: str, links: list[str]) -> str:
        normalized_content = content.strip()
        if not links:
            return normalized_content

        missing_links = [link for link in links if f"[[{link}]]" not in normalized_content]
        if not missing_links:
            return normalized_content

        # Merge into an EXISTING trailing '## Related' section (dedup handled
        # here for the common single-section case) instead of appending a new
        # duplicate heading.
        if "## Related" in normalized_content:
            merged = self.merge_related_sections(normalized_content)
            return f"{merged}\n" + "\n".join(f"- [[{link}]]" for link in missing_links)
        related_block = "## Related\n" + "\n".join(f"- [[{link}]]" for link in missing_links)
        if not normalized_content:
            return related_block
        return f"{normalized_content}\n\n{related_block}"

    def _log_tool_call(self, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        self.logger.info(
            "Synapse server tool call",
            extra={
                "tool_name": name,
                "parameters": self._redact_payload(arguments),
                "response_size": len(json.dumps(result, ensure_ascii=False, default=str)),
            },
        )

    def _redact_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {
                key: ("[REDACTED]" if key.casefold() in _REDACTED_FIELDS else self._redact_payload(value))
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [self._redact_payload(item) for item in payload]
        return payload

    def _record_write_memory_metrics(
        self,
        *,
        node_id: str | None,
        node_type: str,
        decision_payload: dict[str, Any],
        evidence: dict[str, Any],
        similarity_threshold: float,
        warning_codes: list[str],
        execution_succeeded: bool,
        route: str | None = None,
    ) -> None:
        try:
            with self._store() as store:
                store.record_write_memory_event(
                    WriteMemoryEventMetrics(
                        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        node_id=node_id,
                        node_type=node_type,
                        action=str(decision_payload.get("action")) if decision_payload.get("action") else None,
                        candidate_count=int(evidence.get("candidate_count") or 0),
                        similarity_threshold=float(similarity_threshold),
                        warning_codes=tuple(warning_codes),
                        sampling_provider=str(evidence.get("sampling_provider") or ""),
                        execution_succeeded=execution_succeeded,
                        route=route,
                    )
                )
        except (OSError, ValueError, sqlite3.DatabaseError) as exc:  # pragma: no cover - metrics must never break writes
            self.logger.warning("Failed to record write_memory metrics", exc_info=exc)
