"""Dreamer lifecycle pipeline -- Synapse's background memory consolidation engine.

The Dreamer runs a 6-stage pipeline modeled on sleep neuroscience:
  1. Scan -- gather stale, superseded, disputed, and unlinked candidates
  2. Triage (NREM) -- sampling decides keep / condense / archive for stale nodes
  3. Link Weaving (REM) -- sampling discovers missing associative edges
  4. Conflict Resolution -- sampling resolves disputed node pairs
  5. Execute -- carry out all decisions against the store and filesystem
  6. Report -- emit a structured summary of the run
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Callable, Iterable

from synapse.config import SynapseConfig
from synapse.lifecycle.condensation import DeterministicArchiveCondenser
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType, generate_node_id
from synapse.security import purge_expired_archive_files
from synapse.server.sampling import (
    SamplingClient,
    build_conflict_resolution_prompt,
    build_link_weaving_prompt,
    build_triage_prompt,
    parse_sampling_json_result,
)
from synapse.storage import SQLiteNodeStore, archive_node_path, write_node_file
from synapse.sync import SyncBatchResult, SyncManager
from synapse.utils.runtime import RuntimePaths, get_runtime_paths


LOGGER = logging.getLogger("synapse.dreamer")
UTC_SUFFIX = "+00:00"

# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class DreamerWarning:
    """Structured warning emitted during a Dreamer run."""

    code: str
    message: str
    node_id: str | None = None


@dataclass(slots=True, frozen=True)
class TriageDecision:
    node_id: str
    decision: str  # "keep" | "condense" | "archive"
    reason: str


@dataclass(slots=True, frozen=True)
class LinkDecision:
    node_a_id: str
    node_b_id: str


@dataclass(slots=True, frozen=True)
class ConflictDecision:
    node_a_id: str
    node_b_id: str
    decision: str  # "supersede_a" | "supersede_b" | "both_valid"
    reason: str


@dataclass(slots=True, frozen=True)
class CondensationResult:
    source_ids: tuple[str, ...]
    new_node_id: str
    new_title: str


@dataclass(slots=True, frozen=True)
class DreamerReport:
    """Structured report returned by the Dreamer pipeline."""

    started_at: str
    completed_at: str
    scanned: dict[str, int]
    triage: tuple[TriageDecision, ...] = field(default_factory=tuple)
    links_added: tuple[LinkDecision, ...] = field(default_factory=tuple)
    conflicts_resolved: tuple[ConflictDecision, ...] = field(default_factory=tuple)
    archived: tuple[str, ...] = field(default_factory=tuple)
    condensed: tuple[CondensationResult, ...] = field(default_factory=tuple)
    deleted_archive_paths: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[DreamerWarning, ...] = field(default_factory=tuple)
    sync: SyncBatchResult = field(default_factory=SyncBatchResult)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Dreamer pipeline
# ---------------------------------------------------------------------------


_TRIAGE_SYSTEM = (
    "You are a memory-lifecycle agent inside Synapse."
    " Respond with a single JSON object, no markdown fence, no extra prose."
)

_LINK_WEAVING_SYSTEM = (
    "You are a memory-association agent inside Synapse."
    " Respond with a single JSON object, no markdown fence, no extra prose."
)

_CONFLICT_SYSTEM = (
    "You are a memory-conflict-resolution agent inside Synapse."
    " Respond with a single JSON object, no markdown fence, no extra prose."
)


class Dreamer:
    """Background memory consolidation pipeline.

    Requires a ``SamplingClient`` for stages that involve LLM judgement.
    """

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths | None = None,
        store: SQLiteNodeStore | None = None,
        sampling_client: SamplingClient,
        logger: logging.Logger | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths or get_runtime_paths(config)
        self._store = store
        self._owns_store = store is None
        self._sampling_client = sampling_client
        self._logger = logger or LOGGER
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    # -- lifecycle helpers --------------------------------------------------

    def close(self) -> None:
        if self._owns_store and self._store is not None:
            self._store.close()
            self._store = None

    def __enter__(self) -> "Dreamer":
        self._get_store()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    # -- main entry ---------------------------------------------------------

    def run(self, batch_size: int = 8) -> DreamerReport:
        started_at = self._utc_now()
        warnings: list[DreamerWarning] = []
        store = self._get_store()

        self._logger.info(
            "Dreamer pipeline started",
            extra={
                "started_at": started_at.isoformat().replace(UTC_SUFFIX, "Z"),
                "active_path": str(self.runtime_paths.active),
                "archive_path": str(self.runtime_paths.archive),
            },
        )

        # Stage 1: Scan
        stale = store.find_orphan_candidates(self.config.decay.janitor_days)
        superseded = store.find_superseded_for_archival(days_threshold=7)
        disputed = self._find_disputed_pairs(store)
        missing_link_pairs = store.find_missing_link_pairs(
            cosine_threshold=0.75,
            recency_days=self.config.decay.janitor_days,
        )
        scanned = {
            "stale": len(stale),
            "superseded": len(superseded),
            "disputed": len(disputed),
            "missing_link_pairs": len(missing_link_pairs),
        }

        # Stage 2: Triage (NREM slow-wave consolidation)
        triage_decisions = self._run_triage(stale, batch_size=batch_size, warnings=warnings)

        # Stage 3: Link Weaving (REM associative dreaming)
        link_decisions = self._run_link_weaving(missing_link_pairs, batch_size=batch_size, warnings=warnings)

        # Stage 4: Conflict Resolution (interference clearance)
        conflict_decisions = self._run_conflict_resolution(disputed, batch_size=batch_size, warnings=warnings)

        # Stage 5: Execute all decisions
        archived_ids, condensed_results = self._execute_triage(triage_decisions, store, warnings)
        self._execute_links(link_decisions, store, warnings)
        self._execute_conflicts(conflict_decisions, store, warnings)
        archived_superseded_ids = self._archive_superseded(superseded, store, warnings)

        # Purge expired archives
        deleted_archive_paths = tuple(
            path.as_posix()
            for path in purge_expired_archive_files(
                self.runtime_paths.archive,
                retention_days=self.config.decay.archive_retention_days,
                now=started_at,
            )
        )

        # Delta sync
        sync_result = SyncManager(
            self.config,
            runtime_paths=self.runtime_paths,
            store=store,
            logger=self._logger,
            debounce_seconds=0.0,
        ).startup_sync()

        # Stage 6: Report
        report = DreamerReport(
            started_at=started_at.isoformat().replace(UTC_SUFFIX, "Z"),
            completed_at=self._utc_now().isoformat().replace(UTC_SUFFIX, "Z"),
            scanned=scanned,
            triage=tuple(triage_decisions),
            links_added=tuple(link_decisions),
            conflicts_resolved=tuple(conflict_decisions),
            archived=tuple([*archived_ids, *archived_superseded_ids]),
            condensed=tuple(condensed_results),
            deleted_archive_paths=deleted_archive_paths,
            warnings=tuple(warnings),
            sync=sync_result,
        )

        for warning in warnings:
            self._logger.warning(warning.message, extra={"code": warning.code, "node_id": warning.node_id})
        self._logger.info("Dreamer pipeline completed", extra=report.to_dict())
        return report

    # -- Stage 2: Triage ----------------------------------------------------

    def _run_triage(
        self,
        stale: list[Node],
        *,
        batch_size: int,
        warnings: list[DreamerWarning],
    ) -> list[TriageDecision]:
        if not stale:
            return []

        decisions: list[TriageDecision] = []
        for batch_start in range(0, len(stale), batch_size):
            batch = stale[batch_start : batch_start + batch_size]
            try:
                result = self._sampling_client.sample_json(
                    prompt=build_triage_prompt(tuple(batch)),
                    system_prompt=_TRIAGE_SYSTEM,
                    max_tokens=1500,
                )
                payload = parse_sampling_json_result(result)
                for item in payload.get("decisions") or []:
                    decisions.append(
                        TriageDecision(
                            node_id=str(item.get("node_id") or ""),
                            decision=str(item.get("decision") or "archive"),
                            reason=str(item.get("reason") or ""),
                        )
                    )
            except Exception as exc:
                self._logger.warning("Triage sampling failed for batch, defaulting to archive", exc_info=exc)
                warnings.append(
                    DreamerWarning(
                        code="triage_sampling_failed",
                        message=f"Triage sampling failed: {exc}. Defaulting batch to archive.",
                    )
                )
                for node in batch:
                    decisions.append(TriageDecision(node_id=node.id, decision="archive", reason="sampling failed"))
        return decisions

    # -- Stage 3: Link Weaving ----------------------------------------------

    def _run_link_weaving(
        self,
        pairs: list[tuple[Node, Node]],
        *,
        batch_size: int,
        warnings: list[DreamerWarning],
    ) -> list[LinkDecision]:
        if not pairs:
            return []

        decisions: list[LinkDecision] = []
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start : batch_start + batch_size]
            try:
                result = self._sampling_client.sample_json(
                    prompt=build_link_weaving_prompt(tuple(batch)),
                    system_prompt=_LINK_WEAVING_SYSTEM,
                    max_tokens=1500,
                )
                payload = parse_sampling_json_result(result)
                for item in payload.get("decisions") or []:
                    if item.get("link"):
                        decisions.append(
                            LinkDecision(
                                node_a_id=str(item.get("node_a_id") or ""),
                                node_b_id=str(item.get("node_b_id") or ""),
                            )
                        )
            except Exception as exc:
                self._logger.warning("Link weaving sampling failed for batch, skipping", exc_info=exc)
                warnings.append(
                    DreamerWarning(
                        code="link_weaving_sampling_failed",
                        message=f"Link weaving sampling failed: {exc}. Skipping batch.",
                    )
                )
        return decisions

    # -- Stage 4: Conflict Resolution ---------------------------------------

    def _run_conflict_resolution(
        self,
        disputed: list[tuple[Node, Node]],
        *,
        batch_size: int,
        warnings: list[DreamerWarning],
    ) -> list[ConflictDecision]:
        if not disputed:
            return []

        decisions: list[ConflictDecision] = []
        for batch_start in range(0, len(disputed), batch_size):
            batch = disputed[batch_start : batch_start + batch_size]
            try:
                result = self._sampling_client.sample_json(
                    prompt=build_conflict_resolution_prompt(tuple(batch)),
                    system_prompt=_CONFLICT_SYSTEM,
                    max_tokens=1500,
                )
                payload = parse_sampling_json_result(result)
                for item in payload.get("decisions") or []:
                    decisions.append(
                        ConflictDecision(
                            node_a_id=str(item.get("node_a_id") or ""),
                            node_b_id=str(item.get("node_b_id") or ""),
                            decision=str(item.get("decision") or "both_valid"),
                            reason=str(item.get("reason") or ""),
                        )
                    )
            except Exception as exc:
                self._logger.warning("Conflict resolution sampling failed for batch, skipping", exc_info=exc)
                warnings.append(
                    DreamerWarning(
                        code="conflict_resolution_sampling_failed",
                        message=f"Conflict resolution sampling failed: {exc}. Skipping batch.",
                    )
                )
        return decisions

    # -- Stage 5: Execution -------------------------------------------------

    def _execute_triage(
        self,
        decisions: list[TriageDecision],
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> tuple[list[str], list[CondensationResult]]:
        archived_ids: list[str] = []
        to_condense: list[Node] = []

        for decision in decisions:
            node = store.get_node(decision.node_id)
            if node is None:
                warnings.append(
                    DreamerWarning(
                        code="triage_node_missing",
                        node_id=decision.node_id,
                        message=f"Triage target '{decision.node_id}' not found in store.",
                    )
                )
                continue

            if decision.decision == "keep":
                store.touch_node(decision.node_id)
            elif decision.decision == "archive":
                moved = self._archive_nodes([node], reason="triage", warnings=warnings)
                archived_ids.extend(n.id for n in moved)
            elif decision.decision == "condense":
                to_condense.append(node)

        condensed_results: list[CondensationResult] = []
        if to_condense:
            condenser = DeterministicArchiveCondenser()
            now = self._utc_now()
            draft = condenser.synthesize(to_condense, now=now)
            new_id = generate_node_id(draft.title, now)

            new_node = Node(
                metadata=NodeMetadata.model_validate(
                    {
                        "id": new_id,
                        "title": draft.title,
                        "type": NodeType.PERSISTENT.value,
                        "status": NodeStatus.ACTIVE.value,
                        "tags": list(draft.tags),
                        "created_at": now.isoformat().replace(UTC_SUFFIX, "Z"),
                        "last_accessed": now.isoformat().replace(UTC_SUFFIX, "Z"),
                        "access_count": 0,
                    }
                ),
                content=draft.content,
                file_path=self.runtime_paths.active / f"{new_id}.md",
            )
            write_node_file(new_node, base_path=self.runtime_paths.active)
            condensed_results.append(
                CondensationResult(
                    source_ids=draft.source_node_ids,
                    new_node_id=new_id,
                    new_title=draft.title,
                )
            )

            # Archive source nodes
            moved = self._archive_nodes(to_condense, reason="condensed", warnings=warnings)
            archived_ids.extend(n.id for n in moved)

        return archived_ids, condensed_results

    def _execute_links(
        self,
        decisions: list[LinkDecision],
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> None:
        for decision in decisions:
            node_a = store.get_node(decision.node_a_id)
            node_b = store.get_node(decision.node_b_id)
            if node_a is None or node_b is None:
                warnings.append(
                    DreamerWarning(
                        code="link_node_missing",
                        message=f"Cannot link '{decision.node_a_id}' <-> '{decision.node_b_id}': node not found.",
                    )
                )
                continue

            link_tag_b = f"[[{decision.node_b_id}]]"
            link_tag_a = f"[[{decision.node_a_id}]]"

            a_updated = False
            b_updated = False

            if link_tag_b not in node_a.content:
                updated_a = Node(
                    metadata=node_a.metadata,
                    content=node_a.content.rstrip() + "\n\n" + link_tag_b + "\n",
                    file_path=node_a.file_path,
                )
                write_node_file(updated_a, base_path=self.runtime_paths.active)
                a_updated = True

            if link_tag_a not in node_b.content:
                updated_b = Node(
                    metadata=node_b.metadata,
                    content=node_b.content.rstrip() + "\n\n" + link_tag_a + "\n",
                    file_path=node_b.file_path,
                )
                write_node_file(updated_b, base_path=self.runtime_paths.active)
                b_updated = True

            if a_updated or b_updated:
                self._logger.info(
                    "Linked nodes",
                    extra={"node_a": decision.node_a_id, "node_b": decision.node_b_id},
                )

    def _execute_conflicts(
        self,
        decisions: list[ConflictDecision],
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> None:
        for decision in decisions:
            node_a = store.get_node(decision.node_a_id)
            node_b = store.get_node(decision.node_b_id)
            if node_a is None or node_b is None:
                warnings.append(
                    DreamerWarning(
                        code="conflict_node_missing",
                        message=(
                            f"Cannot resolve conflict '{decision.node_a_id}' <-> "
                            f"'{decision.node_b_id}': node not found."
                        ),
                    )
                )
                continue

            if decision.decision == "supersede_a":
                self._mark_superseded(node_a, node_b, store)
            elif decision.decision == "supersede_b":
                self._mark_superseded(node_b, node_a, store)
            elif decision.decision == "both_valid":
                self._clear_disputed(node_a, store)
                self._clear_disputed(node_b, store)

    def _mark_superseded(self, loser: Node, winner: Node, store: SQLiteNodeStore) -> None:
        """Mark *loser* as superseded by *winner*."""
        updated_meta = loser.metadata.model_copy(
            update={"status": NodeStatus.SUPERSEDED, "superseded_by": winner.id}
        )
        updated = Node(metadata=updated_meta, content=loser.content, file_path=loser.file_path)
        store.upsert_node(updated)
        write_node_file(updated, base_path=self.runtime_paths.active)

    def _clear_disputed(self, node: Node, store: SQLiteNodeStore) -> None:
        """Reset a disputed node back to active."""
        updated_meta = node.metadata.model_copy(update={"status": NodeStatus.ACTIVE})
        updated = Node(metadata=updated_meta, content=node.content, file_path=node.file_path)
        store.upsert_node(updated)
        write_node_file(updated, base_path=self.runtime_paths.active)

    def _archive_superseded(
        self,
        superseded: list[Node],
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> list[str]:
        """Validate and archive superseded nodes (same logic as old NightlyJanitor)."""
        valid: list[Node] = []
        for node in superseded:
            superseder_id = node.metadata.superseded_by
            if not superseder_id:
                warnings.append(
                    DreamerWarning(
                        code="invalid_superseder",
                        node_id=node.id,
                        message=f"Cannot archive superseded node '{node.id}': superseded_by is missing.",
                    )
                )
                continue

            superseder = store.get_node(superseder_id)
            superseder_path = self.runtime_paths.base / superseder.file_path if superseder is not None else None
            if (
                superseder is None
                or superseder.metadata.status is not NodeStatus.ACTIVE
                or superseder_path is None
                or not superseder_path.exists()
            ):
                warnings.append(
                    DreamerWarning(
                        code="invalid_superseder",
                        node_id=node.id,
                        message=(
                            f"Cannot archive superseded node '{node.id}': superseder '{superseder_id}' "
                            "is missing or not active."
                        ),
                    )
                )
                continue
            valid.append(node)

        archived = self._archive_nodes(valid, reason="superseded", warnings=warnings)
        return [n.id for n in archived]

    # -- Disputed pair discovery --------------------------------------------

    def _find_disputed_pairs(self, store: SQLiteNodeStore) -> list[tuple[Node, Node]]:
        """Find pairs of disputed nodes that reference each other."""
        disputed = store.find_by_status(NodeStatus.DISPUTED)
        pairs: list[tuple[Node, Node]] = []
        seen: set[str] = set()
        for node in disputed:
            if node.id in seen:
                continue
            if node.metadata.superseded_by:
                partner = store.get_node(node.metadata.superseded_by)
                if partner and partner.metadata.status == NodeStatus.DISPUTED and partner.id not in seen:
                    pairs.append((node, partner))
                    seen.add(node.id)
                    seen.add(partner.id)
        return pairs

    # -- Shared helpers (preserved from NightlyJanitor) ---------------------

    def _archive_nodes(
        self,
        nodes: Iterable[Node],
        *,
        reason: str,
        warnings: list[DreamerWarning],
    ) -> list[Node]:
        archived: list[Node] = []
        for node in nodes:
            source_path = self.runtime_paths.base / node.file_path
            destination = archive_node_path(self.runtime_paths.archive, node.id)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if not source_path.exists():
                warnings.append(
                    DreamerWarning(
                        code="missing_source_file",
                        node_id=node.id,
                        message=f"Cannot archive node '{node.id}': source file '{source_path}' does not exist.",
                    )
                )
                continue
            if destination.exists():
                warnings.append(
                    DreamerWarning(
                        code="archive_collision",
                        node_id=node.id,
                        message=f"Cannot archive node '{node.id}': archive file '{destination}' already exists.",
                    )
                )
                continue

            source_path.replace(destination)
            archived.append(node)
            self._logger.info(
                "Archived node",
                extra={
                    "node_id": node.id,
                    "reason": reason,
                    "source_path": str(source_path),
                    "archive_path": str(destination),
                },
            )
        return archived

    def _dedupe_nodes(self, nodes: Iterable[Node]) -> list[Node]:
        unique: dict[str, Node] = {}
        for node in nodes:
            unique.setdefault(node.id, node)
        return list(unique.values())

    def _get_store(self) -> SQLiteNodeStore:
        if self._store is None:
            self._store = SQLiteNodeStore(
                self.runtime_paths.base / "synapse.db",
                embedding_dimension=self.config.embedding.dimension or 0,
            )
        return self._store

    def _utc_now(self) -> datetime:
        current = self._now_provider()
        if current.tzinfo is None:
            return current.replace(tzinfo=UTC)
        return current.astimezone(UTC)
