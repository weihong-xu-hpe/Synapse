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

import hashlib
import json
import logging
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping

from synapse.config import SynapseConfig, ensure_background_model_allowed
from synapse.decision import DecisionClient
from synapse.lifecycle.condensation import CondensationDraft, DeterministicArchiveCondenser
from synapse.models import (
    Node,
    NodeMetadata,
    NodeStatus,
    NodeType,
    format_utc_datetime,
    generate_node_id,
    slugify_english_title,
)
from synapse.okf import TYPE_REQUIRED_SECTIONS, _split_sections, validate_okf_node
from synapse.security import purge_expired_archive_files
from synapse.server.sampling import (
    SamplingClient,
    build_compaction_prompt,
    build_conflict_resolution_prompt,
    build_link_weaving_prompt,
    build_semantic_merge_prompt,
    build_synthesis_prompt,
    build_triage_prompt,
)
from synapse.storage import (
    SQLiteNodeStore,
    archive_node_path,
    locate_active_node_file,
    write_node_file,
)
from synapse.storage.markdown import WIKI_LINK_PATTERN, atomic_write_text, extract_wiki_links, read_node_file
from synapse.okf.transcripts import content_sha256
from synapse.storage.sqlite import DreamerRunMetrics
from synapse.sync import SyncBatchResult, SyncManager
from synapse.utils.runtime import RuntimePaths, get_runtime_paths


LOGGER = logging.getLogger("synapse.dreamer")


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
class SynthesisResult:
    """One hub node created by the P3 synthesis (clustering) stage."""

    member_ids: tuple[str, ...]
    hub_node_id: str
    hub_title: str


@dataclass(slots=True, frozen=True)
class MergeResult:
    """One consolidated node produced by the Dreamer 2.0 semantic merge stage."""

    member_ids: tuple[str, ...]
    winner_node_id: str
    winner_title: str
    superseded_ids: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class CompactionResult:
    """One oversized node rewritten in place by the compaction stage."""

    node_id: str
    bytes_before: int
    bytes_after: int


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
    synthesized: tuple[SynthesisResult, ...] = field(default_factory=tuple)
    merged: tuple[MergeResult, ...] = field(default_factory=tuple)
    compacted: tuple[CompactionResult, ...] = field(default_factory=tuple)
    repointed_links: int = 0
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

_SYNTHESIS_SYSTEM = (
    "You are a memory-synthesis agent inside Synapse."
    " Respond with a single JSON object, no markdown fence, no extra prose."
)

_SEMANTIC_MERGE_SYSTEM = (
    "You are a memory-consolidation agent inside Synapse."
    " Respond with a single JSON object, no markdown fence, no extra prose."
)

_COMPACTION_SYSTEM = (
    "You are a memory-compaction agent inside Synapse."
    " Respond with a single JSON object, no markdown fence, no extra prose."
)

# OKF validation codes that make a sleep-consolidation product unusable.
_FATAL_OKF_PREFIXES = ("okf_missing_section:", "okf_duplicate_section:")
_FATAL_OKF_CODES = ("okf_takeaway_invalid",)
# Fraction of a member's anchor tokens that may vanish in a nighttime merge
# before the merge is rejected outright (§3.4(1) anchor-token verification).
_ANCHOR_LOSS_TOLERANCE = 0.2
# Oversized-node dedupe (§3.4(2)): cosine says "same topic", token overlap must
# confirm "same text" before an oversized node is retired into another node.
_DEDUPE_TOKEN_JACCARD = 0.6
_PROVENANCE_SECTION_NAMES = ("sources", "related", "evidence")


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
        gatekeeper_client: SamplingClient | None = None,
        decision_client: DecisionClient | None = None,
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
        # Dreamer 2.0 final-write LLM: the Tier B gatekeeper (claude-haiku-5.5)
        # whenever it is available, falling back to the Tier A sampling client.
        # An injected gatekeeper is used as-is and never closed by us.
        self._gatekeeper_client: object | None = gatekeeper_client
        self._owns_gatekeeper_client = False
        self._decision_client = (
            decision_client if decision_client is not None else DecisionClient(config.decision_model)
        )
        # Nodes archived during the current run. Stage 5 executes decisions
        # captured before Stage 4c/4d ran, so it must know which nodes have
        # already left active/ (their files are gone; the write index lags
        # until the closing startup sync).
        self._archived_this_run: set[str] = set()

    # -- lifecycle helpers --------------------------------------------------

    def close(self) -> None:
        if self._owns_gatekeeper_client:
            close_client = getattr(self._gatekeeper_client, "close", None)
            if callable(close_client):
                close_client()
            self._gatekeeper_client = None
            self._owns_gatekeeper_client = False
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
        self._archived_this_run = set()
        store = self._get_store()

        self._logger.info(
            "Dreamer pipeline started",
            extra={
                "started_at": format_utc_datetime(started_at),
                "active_path": str(self.runtime_paths.active),
                "archive_path": str(self.runtime_paths.archive),
            },
        )

        # Stage 1: Scan
        thresholds = self.config.dreamer.thresholds
        stale_orphan_days = thresholds.stale_orphan_days or self.config.decay.janitor_days
        link_weaving_recency_days = thresholds.link_weaving_recency_days or self.config.decay.janitor_days
        stale = store.find_orphan_candidates(stale_orphan_days)
        superseded = store.find_superseded_for_archival(days_threshold=thresholds.superseded_archive_days)
        disputed = self._find_disputed_pairs(store)
        missing_link_pairs = store.find_missing_link_pairs(
            cosine_threshold=thresholds.missing_link_cosine,
            recency_days=link_weaving_recency_days,
        )
        raw_missing_link_pair_count = len(missing_link_pairs)
        if raw_missing_link_pair_count > thresholds.max_missing_link_pairs_per_run:
            missing_link_pairs = missing_link_pairs[: thresholds.max_missing_link_pairs_per_run]
            warnings.append(
                DreamerWarning(
                    code="missing_link_pairs_capped",
                    message=(
                        f"Missing-link candidates capped at {thresholds.max_missing_link_pairs_per_run} "
                        f"of {raw_missing_link_pair_count}. Raise missing_link_cosine or lower the cap if needed."
                    ),
                )
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

        # Stage 4b: Synthesis (hub building over clustered typed knowledge)
        synthesis_results = self._run_synthesis(store, warnings)

        # Stage 4c: In-place semantic merge (Dreamer 2.0 sleep consolidation)
        merge_results = self._run_semantic_merge(store, warnings)

        # Stage 4d: Oversized node compaction (dedupe bloated > 4 KB nodes)
        compaction_results = self._run_oversized_compaction(store, warnings)

        # Stage 5: Execute all decisions
        archived_ids, condensed_results = self._execute_triage(triage_decisions, store, warnings)
        self._execute_links(link_decisions, store, warnings)
        self._execute_conflicts(conflict_decisions, store, warnings)
        repointed_links = self._repoint_superseded_links(store)
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
        completed_at = self._utc_now()
        all_archived_ids = list(
            dict.fromkeys(
                [
                    *archived_ids,
                    *archived_superseded_ids,
                    *(member_id for result in merge_results for member_id in result.superseded_ids),
                    *(
                        result.node_id
                        for result in compaction_results
                        if result.node_id in self._archived_this_run
                    ),
                ]
            )
        )
        report = DreamerReport(
            started_at=format_utc_datetime(started_at),
            completed_at=format_utc_datetime(completed_at),
            scanned=scanned,
            triage=tuple(triage_decisions),
            links_added=tuple(link_decisions),
            conflicts_resolved=tuple(conflict_decisions),
            archived=tuple(all_archived_ids),
            condensed=tuple(condensed_results),
            synthesized=tuple(synthesis_results),
            merged=tuple(merge_results),
            compacted=tuple(compaction_results),
            repointed_links=repointed_links,
            deleted_archive_paths=deleted_archive_paths,
            warnings=tuple(warnings),
            sync=sync_result,
        )
        self._record_metrics(report, store, batch_size=batch_size, duration_ms=_duration_ms(started_at, completed_at))

        for warning in warnings:
            self._logger.warning(warning.message, extra={"code": warning.code, "node_id": warning.node_id})
        self._logger.info("Dreamer pipeline completed", extra=report.to_dict())
        self._rebuild_handbook()
        self._export_skill_promotion_proposals(store)
        return report

    def _rebuild_handbook(self) -> None:
        """Best-effort handbook rebuild after a sleep-consolidation cycle (§5.4 step 5).

        The handbook is a compiled artefact of the OKF graph, so it is
        regenerated (idempotently) whenever the graph changes. Never fails a run.
        """

        try:
            from synapse.handbook import build_handbook

            build_handbook(self.config, self.runtime_paths)
        except Exception as exc:  # noqa: BLE001 — handbook is a compile artefact, never a blocker
            self._logger.warning("Handbook rebuild after Dreamer run failed: %s", exc)

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
                    prompt=build_triage_prompt(
                        tuple(batch),
                        low_structure_chars=self.config.dreamer.thresholds.low_structure_chars,
                    ),
                    system_prompt=_TRIAGE_SYSTEM,
                    max_tokens=self.config.decider.max_tokens,
                )
                payload = result
                for item in payload.get("decisions") or []:
                    decisions.append(
                        TriageDecision(
                            node_id=str(item.get("node_id") or ""),
                            decision=str(item.get("decision") or "archive"),
                            reason=str(item.get("reason") or ""),
                        )
                    )
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                self._logger.warning("Triage sampling failed for batch, skipping batch", exc_info=exc)
                warnings.append(
                    DreamerWarning(
                        code="triage_sampling_failed",
                        message=f"Triage sampling failed: {exc}. Skipping batch (no decisions emitted).",
                    )
                )
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
                    max_tokens=self.config.decider.max_tokens,
                )
                payload = result
                for item in payload.get("decisions") or []:
                    if item.get("link"):
                        decisions.append(
                            LinkDecision(
                                node_a_id=str(item.get("node_a_id") or ""),
                                node_b_id=str(item.get("node_b_id") or ""),
                            )
                        )
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
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
                    max_tokens=self.config.decider.max_tokens,
                )
                payload = result
                for item in payload.get("decisions") or []:
                    decisions.append(
                        ConflictDecision(
                            node_a_id=str(item.get("node_a_id") or ""),
                            node_b_id=str(item.get("node_b_id") or ""),
                            decision=str(item.get("decision") or "both_valid"),
                            reason=str(item.get("reason") or ""),
                        )
                    )
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                self._logger.warning("Conflict resolution sampling failed for batch, skipping", exc_info=exc)
                warnings.append(
                    DreamerWarning(
                        code="conflict_resolution_sampling_failed",
                        message=f"Conflict resolution sampling failed: {exc}. Skipping batch.",
                    )
                )
        return decisions

    # -- Stage 4b: Synthesis (P3) --------------------------------------------

    def _run_synthesis(self, store: SQLiteNodeStore, warnings: list[DreamerWarning]) -> list[SynthesisResult]:
        """Cluster typed active nodes per project and write one fact hub per cluster.

        Clusters already covered by an existing hub are skipped (hub detection:
        an active typed node whose Sources list contains every cluster member —
        i.e. a node whose sources are the member ids). Hubs themselves never
        re-cluster as members because their Sources point at member ids and
        their own ids never appear as a member's source... members exclude any
        node whose id appears in another candidate node's sources.
        """

        thresholds = self.config.dreamer.thresholds
        clusters = store.find_synthesis_clusters(
            cosine_threshold=thresholds.synthesis_cosine,
            min_size=thresholds.min_synthesis_cluster,
            max_size=thresholds.max_synthesis_cluster,
            exclude_hubs=True,
        )
        if not clusters:
            return []

        results: list[SynthesisResult] = []
        hub_ids_created: set[str] = set()
        for cluster in clusters[: thresholds.max_synthesis_hubs_per_run]:
            member_ids = {node.id for node in cluster}
            # Skip clusters already covered by a hub: an ACTIVE typed node
            # whose sources ⊇ member ids (and which is not itself a member).
            if self._cluster_has_hub(store, cluster, member_ids):
                continue
            # Hub nodes created in this run must not become members of later
            # clusters: drop any node referenced as a source by a prior hub.
            cluster = [node for node in cluster if node.id not in hub_ids_created]
            if len(cluster) < thresholds.min_synthesis_cluster:
                continue
            try:
                result = self._sampling_client.sample_json(
                    prompt=build_synthesis_prompt(tuple(cluster)),
                    system_prompt=_SYNTHESIS_SYSTEM,
                    max_tokens=self.config.decider.max_tokens,
                )
                payload = result
                title = str(payload.get("title") or "").strip()
                takeaway = str(payload.get("takeaway") or "").strip()
                details = str(payload.get("details") or "").strip()
                sources = [str(s).strip() for s in (payload.get("sources") or []) if str(s).strip()]
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                self._logger.warning("Synthesis sampling failed for cluster, skipping", exc_info=exc)
                warnings.append(
                    DreamerWarning(
                        code="synthesis_sampling_failed",
                        message=f"Synthesis sampling failed: {exc}. Skipping cluster.",
                    )
                )
                continue
            if not title or not takeaway or not details:
                warnings.append(
                    DreamerWarning(
                        code="synthesis_invalid_payload",
                        message="Synthesis payload missing title/takeaway/details; skipping cluster.",
                    )
                )
                continue
            projects = {((n.metadata.project or "").strip() or None) for n in cluster}
            hub_project = next(iter(projects)) if len(projects) == 1 else None
            hub = self._write_hub_node(
                store,
                title,
                takeaway,
                details,
                member_ids,
                warnings=warnings,
                project=hub_project,
            )
            if hub is None:
                continue
            hub_ids_created.add(hub.id)
            results.append(SynthesisResult(member_ids=tuple(sorted(member_ids)), hub_node_id=hub.id, hub_title=hub.title))
            self._logger.info(
                "Synthesized hub node",
                extra={"hub_node_id": hub.id, "members": sorted(member_ids)},
            )
        return results

    @staticmethod
    def _cluster_has_hub(store: SQLiteNodeStore, cluster: list[Node], member_ids: set[str]) -> bool:
        for candidate in store.list_nodes({"status": NodeStatus.ACTIVE, "type": NodeType.PERSISTENT}):
            if candidate.id in member_ids:
                continue
            if not candidate.metadata.okf_type:
                continue
            sources = set(candidate.metadata.sources)
            if sources and member_ids:
                overlap = len(sources & member_ids) / len(member_ids)
                if overlap >= 0.7:
                    return True
        return False

    def _write_hub_node(
        self,
        store: SQLiteNodeStore,
        title: str,
        takeaway: str,
        details: str,
        member_ids: set[str],
        *,
        warnings: list[DreamerWarning] | None = None,
        project: str | None = None,
    ) -> Node | None:
        from synapse.models import slugify_english_title

        if not slugify_english_title(title):
            if warnings is not None:
                warnings.append(
                    DreamerWarning(
                        code="synthesis_title_unusable",
                        message=f"Synthesis produced an unusable English title ({title!r}); skipping cluster.",
                    )
                )
            return None
        now = self._utc_now()
        hub_id = generate_node_id(title, now)
        if store.get_node(hub_id) is not None:
            if warnings is not None:
                warnings.append(
                    DreamerWarning(
                        code="synthesis_id_collision",
                        message=f"Synthesis hub id '{hub_id}' already exists; skipping cluster.",
                    )
                )
            return None
        content = "\n".join(
            [
                "## Takeaway",
                "",
                takeaway,
                "",
                "## Details",
                "",
                details,
                "",
                "## Sources",
                "",
                *(f"- {member_id}" for member_id in sorted(member_ids)),
                "",
            ]
        )
        metadata = self._knowledge_metadata(
            node_id=hub_id,
            title=title,
            okf_type="fact",
            okf_version=1,
            sources=sorted(member_ids),
            tags=["dreamer-hub"],
            now_iso=format_utc_datetime(now),
            project=project,
        )
        hub = Node(metadata=metadata, content=content)
        return self._persist(store, hub)

    # -- Stage 4c/4d: Dreamer 2.0 sleep consolidation ------------------------

    def _background_client(self) -> object:
        """Tier B gatekeeper (claude-haiku-5.5) for final-write LLM work.

        Falls back to the Tier A sampling client when no gatekeeper is
        available or its configured model violates the §3.2 model budget.
        """

        if self._gatekeeper_client is None:
            endpoint = self.config.distiller.episode.gatekeeper_endpoint
            endpoint_model = endpoint.model if endpoint is not None else "claude-haiku-5.5"
            try:
                ensure_background_model_allowed(
                    endpoint_model, purpose="Dreamer sleep consolidation gatekeeper"
                )
            except ValueError as exc:
                self._logger.warning("Dreamer gatekeeper rejected by model budget: %s", exc)
                self._gatekeeper_client = self._sampling_client
                return self._gatekeeper_client
            try:
                from synapse.lifecycle.episode_distiller import EpisodeDistiller

                client = EpisodeDistiller.make_gatekeeper_client(self.config)
            except Exception as exc:  # noqa: BLE001 — Tier A client still works
                self._logger.warning("Dreamer gatekeeper unavailable; using sampling client: %s", exc)
                client = None
            if client is not None:
                self._owns_gatekeeper_client = True
            self._gatekeeper_client = client or self._sampling_client
        return self._gatekeeper_client

    def _run_semantic_merge(
        self,
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> list[MergeResult]:
        """Stage 4c: merge near-duplicate nodes in place (§3.4(1)).

        Clusters active typed knowledge per project by cosine similarity; each
        cluster of 2-6 members collapses into ONE consolidated OKF node whose
        anchor tokens are verified against every member (§3.3 nighttime gate).
        Losers are marked ``superseded`` with a ``t_invalid`` stamp and moved to
        ``.archive/`` so the merge stays rollback-safe.
        """

        thresholds = self.config.dreamer.thresholds
        clusters = store.find_complete_linkage_clusters(
            cosine_threshold=thresholds.semantic_merge_cosine,
            floor_cosine=thresholds.semantic_merge_floor_cosine,
            min_size=thresholds.min_semantic_merge_cluster,
            max_size=thresholds.max_semantic_merge_cluster,
            project_mode=thresholds.semantic_merge_project_mode,
            exclude_hubs=True,
        )
        if not clusters:
            return []

        client = self._background_client()
        results: list[MergeResult] = []
        for cluster in clusters[: thresholds.max_merges_per_run]:
            # Idempotence / crash safety: only merge clusters whose members are
            # still active right now (a prior run may have retired some).
            live_members = [
                node
                for node in (store.get_node(member.id) for member in cluster)
                if node is not None and node.metadata.status is NodeStatus.ACTIVE
            ]
            if len(live_members) < thresholds.min_semantic_merge_cluster:
                continue
            if self._decision_client.enabled:
                prescreen = self._prescreen_merge_cluster(live_members)
                if prescreen is not None:
                    choice, confidence = prescreen
                    min_confidence = float(self.config.decision_model.min_confidence)
                    if choice == "distinct_constraints" and confidence >= max(0.80, min_confidence):
                        continue
                    if choice == "conflict" and confidence >= min_confidence:
                        warnings.append(
                            DreamerWarning(
                                code="decision_model_conflict",
                                node_id=live_members[0].id,
                                message=(
                                    f"Decision model flagged conflict in merge cluster "
                                    f"({', '.join(member.id for member in live_members)}); skipping merge."
                                ),
                            )
                        )
                        continue
            try:
                payload = client.sample_json(
                    prompt=build_semantic_merge_prompt(tuple(live_members)),
                    system_prompt=_SEMANTIC_MERGE_SYSTEM,
                    max_tokens=self.config.decider.max_tokens,
                )
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                self._logger.warning("Semantic merge sampling failed for cluster, skipping", exc_info=exc)
                warnings.append(
                    DreamerWarning(
                        code="semantic_merge_sampling_failed",
                        message=f"Semantic merge sampling failed: {exc}. Skipping cluster.",
                    )
                )
                continue
            result = self._materialize_merge(live_members, payload, store, warnings)
            if result is not None:
                results.append(result)
        return results

    def _materialize_merge(
        self,
        members: list[Node],
        payload: object,
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> MergeResult | None:
        """Validate + persist one merged node and retire its members."""

        if not isinstance(payload, Mapping):
            warnings.append(
                DreamerWarning(
                    code="semantic_merge_invalid_payload",
                    message="Semantic merge payload was not a JSON object; skipping cluster.",
                )
            )
            return None
        action = str(payload.get("action") or "merge").strip().casefold()
        merge_relation = str(payload.get("merge_relation") or "").strip().casefold()
        if action == "skip" or merge_relation in {"distinct_constraints", "conflict"}:
            warnings.append(
                DreamerWarning(
                    code="semantic_merge_skipped_distinct",
                    node_id=members[0].id,
                    message=f"Semantic merge skipped cluster ({merge_relation or action}).",
                )
            )
            return None
        title = str(payload.get("title") or "").strip()
        okf_type = str(payload.get("okf_type") or "").strip()
        takeaway = str(payload.get("takeaway") or "").strip()
        body = self._strip_provenance_sections(str(payload.get("body") or "").strip())
        member_ids = sorted(member.id for member in members)
        if not title or not slugify_english_title(title):
            warnings.append(
                DreamerWarning(
                    code="semantic_merge_title_unusable",
                    message=f"Semantic merge produced an unusable English title ({title!r}); skipping cluster.",
                )
            )
            return None
        if okf_type not in TYPE_REQUIRED_SECTIONS:
            warnings.append(
                DreamerWarning(
                    code="semantic_merge_unknown_type",
                    message=f"Semantic merge produced unknown okf_type {okf_type!r}; skipping cluster.",
                )
            )
            return None
        if not takeaway or "\n" in takeaway or len(takeaway) > 200:
            warnings.append(
                DreamerWarning(
                    code="semantic_merge_takeaway_invalid",
                    message="Semantic merge 'takeaway' must be exactly one line of <= 200 characters; skipping cluster.",
                )
            )
            return None
        if not body:
            warnings.append(
                DreamerWarning(
                    code="semantic_merge_empty_body",
                    message="Semantic merge returned an empty body; skipping cluster.",
                )
            )
            return None
        if self._missing_required_sections(body, okf_type):
            warnings.append(
                DreamerWarning(
                    code="semantic_merge_missing_sections",
                    message=(
                        f"Semantic merge body is missing required {okf_type} sections "
                        f"{self._missing_required_sections(body, okf_type)}; skipping cluster."
                    ),
                )
            )
            return None

        sources = sorted({*member_ids, *(s for member in members for s in member.metadata.sources)})
        related = sorted(
            {
                link
                for member in members
                for link in extract_wiki_links(member.content)
                if link not in member_ids
            }
        )
        content = self._render_consolidated_body(takeaway, body, sources, related)
        content = self._verify_or_reject_anchors(members, content, okf_type, sources, title, warnings)
        if content is None:
            return None

        limit = self.config.dreamer.thresholds.oversized_compaction_bytes
        member_max_bytes = max(len(member.content.encode("utf-8")) for member in members)
        merged_bytes = len(content.encode("utf-8"))
        if merged_bytes > limit and merged_bytes >= member_max_bytes:
            warnings.append(
                DreamerWarning(
                    code="semantic_merge_oversized",
                    node_id=members[0].id,
                    message=(
                        f"Semantic merge product exceeded the {limit}-byte cap without shrinking "
                        f"({merged_bytes} B >= {member_max_bytes} B); skipping cluster."
                    ),
                )
            )
            return None

        now = self._utc_now()
        winner_id = generate_node_id(title, now)
        existing = store.get_node(winner_id)
        if existing is not None:
            warnings.append(
                DreamerWarning(
                    code="semantic_merge_id_collision",
                    message=f"Semantic merge winner id '{winner_id}' already exists; skipping cluster.",
                )
            )
            return None
        now_iso = format_utc_datetime(now)
        metadata = self._knowledge_metadata(
            node_id=winner_id,
            title=title,
            okf_type=okf_type,
            okf_version=1,
            sources=sources,
            tags=sorted({tag for member in members for tag in member.metadata.tags} | {"dreamer-merged"}),
            now_iso=now_iso,
            project=members[0].metadata.project,
            t_valid=now_iso,
        )
        metadata = metadata.model_copy(
            update={
                "used_count": sum(m.metadata.used_count for m in members),
                "injected_count": sum(m.metadata.injected_count for m in members),
                "retrieved_count": max(m.metadata.retrieved_count for m in members),
                "access_count": sum(m.metadata.used_count for m in members) + sum(m.metadata.injected_count for m in members),
            }
        )
        winner = Node(
            metadata=metadata,
            content=content,
        )
        self._persist(store, winner)

        superseded: list[str] = []
        for member in members:
            fresh = store.get_node(member.id) or member
            updated = self._with_metadata(
                fresh,
                status=NodeStatus.SUPERSEDED,
                superseded_by=winner_id,
                t_invalid=now_iso,
            )
            self._persist(store, updated)
            superseded.append(member.id)
        self._archive_nodes(members, reason="semantic_merge", warnings=warnings)

        self._logger.info(
            "Merged node cluster",
            extra={"winner_node_id": winner_id, "members": member_ids},
        )
        return MergeResult(
            member_ids=tuple(member_ids),
            winner_node_id=winner_id,
            winner_title=title,
            superseded_ids=tuple(superseded),
        )

    def _run_oversized_compaction(
        self,
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> list[CompactionResult]:
        """Stage 4d: oversized (> 4 KB) nodes get compacted (§3.4(2)).

        Per node, in order of cost: (a) a near-duplicate is merged into the node
        it duplicates (deterministic, cosine >= 0.88 and token-Jaccard >= 0.6),
        (b) the gatekeeper compacts it in place under the anchor-token gate, and
        (c) if that is unavailable or cannot shrink it, the node is split by
        ``##`` section into atomic sibling nodes (A-MEM).
        """

        thresholds = self.config.dreamer.thresholds
        limit = thresholds.oversized_compaction_bytes
        candidates = [
            node
            for node in store.list_nodes({"status": NodeStatus.ACTIVE, "type": NodeType.PERSISTENT})
            if node.metadata.okf_type
            and len(node.content.encode("utf-8")) > limit
            and node.metadata.compaction_attempted_hash != content_sha256(node.content)
        ]
        if not candidates:
            return []

        client: object | None = None
        results: list[CompactionResult] = []
        for candidate in candidates[: thresholds.max_compactions_per_run]:
            node = store.get_node(candidate.id)
            if node is None or node.metadata.status is not NodeStatus.ACTIVE:
                continue
            deduped = self._dedupe_oversized(node, store, warnings)
            if deduped is not None:
                results.append(deduped)
                continue
            if client is None:
                client = self._background_client()
            stripped = self._strip_provenance_sections(node.content, strip_takeaway=False)
            try:
                payload = client.sample_json(
                    prompt=build_compaction_prompt(node, stripped_content=stripped),
                    system_prompt=_COMPACTION_SYSTEM,
                    max_tokens=self.config.decider.max_tokens,
                )
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                self._logger.warning("Oversized compaction sampling failed, skipping node", exc_info=exc)
                warnings.append(
                    DreamerWarning(
                        code="compaction_sampling_failed",
                        node_id=node.id,
                        message=f"Oversized compaction sampling failed: {exc}. Splitting instead.",
                    )
                )
                payload = None
            compacted = (
                self._materialize_compaction(node, payload, store, warnings) if payload is not None else None
            )
            if compacted is not None:
                results.append(compacted)
                continue
            split = self._split_oversized(node, store, warnings)
            if split is not None:
                results.append(split)
                continue
            marked = self._with_metadata(node, compaction_attempted_hash=content_sha256(node.content))
            self._persist(store, marked)
        return results

    def _dedupe_oversized(
        self,
        node: Node,
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> CompactionResult | None:
        """Merge an oversized node into the near-duplicate it repeats (§3.4(2)).

        Deterministic and cheap: embedding cosine answers "same topic", token
        Jaccard answers "same text". The duplicate's unique lines are appended
        to the keeper with the daytime delta merge, so the keeper only grows.
        """

        thresholds = self.config.dreamer.thresholds
        try:
            similar = store.find_near_duplicate(
                node.id, cosine_threshold=thresholds.oversized_dedupe_cosine
            )
        except Exception as exc:  # noqa: BLE001 — probe is best effort
            self._logger.warning("Near-duplicate probe failed", extra={"node_id": node.id, "error": str(exc)})
            return None

        from synapse.server.service import deterministic_section_merge

        for other_id, similarity in similar:
            keeper = store.get_node(other_id)
            if keeper is None:
                continue
            overlap = self._token_jaccard(node.content, keeper.content)
            if overlap < _DEDUPE_TOKEN_JACCARD:
                continue
            merged_body = deterministic_section_merge(keeper.content, node.content)
            verified = self._verify_or_reject_anchors(
                [node],
                merged_body,
                keeper.metadata.okf_type or "",
                list(keeper.metadata.sources),
                keeper.title,
                warnings,
            )
            if verified is None:
                continue
            bytes_before = len(node.content.encode("utf-8"))
            self._persist(store, Node(metadata=keeper.metadata, content=verified, file_path=keeper.file_path))
            now_iso = format_utc_datetime(self._utc_now())
            superseded = self._with_metadata(
                node,
                status=NodeStatus.SUPERSEDED,
                superseded_by=keeper.id,
                t_invalid=now_iso,
            )
            self._persist(store, superseded)
            self._archive_nodes([node], reason="oversized_dedupe", warnings=warnings)
            self._logger.info(
                "Deduped oversized node into keeper",
                extra={
                    "node_id": node.id,
                    "keeper_id": keeper.id,
                    "cosine": round(similarity, 4),
                    "token_jaccard": round(overlap, 3),
                },
            )
            return CompactionResult(node_id=node.id, bytes_before=bytes_before, bytes_after=0)
        return None

    @staticmethod
    def _token_jaccard(left: str, right: str) -> float:
        """Token-set Jaccard overlap of two bodies (cheap near-duplicate guard)."""

        left_tokens = set(re.findall(r"[a-z0-9]{2,}", left.casefold()))
        right_tokens = set(re.findall(r"[a-z0-9]{2,}", right.casefold()))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _split_oversized(
        self,
        node: Node,
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> CompactionResult | None:
        """Split an oversized node by ``##`` section into atomic siblings (§3.4(2)).

        Purely deterministic: knowledge sections are packed in order into
        ``<= limit`` byte chunks (with provenance kept out of standalone
        shards and sibling-link overhead reserved up front) and copied
        verbatim, so no content can be lost. Siblings inherit the source's
        project and valid OKF structure (typed ``fact`` with Takeaway +
        Details + Sources when the slice no longer satisfies the original
        type's required sections), link to each other via ``[[...]]`` +
        ``edges``, and supersede/archive the source.
        """

        limit = self.config.dreamer.thresholds.oversized_compaction_bytes
        full_knowledge = self._strip_provenance_sections(node.content, strip_takeaway=False)
        takeaway_text = self._extract_takeaway_line(full_knowledge) or node.title
        if len(takeaway_text) > 200:
            takeaway_text = takeaway_text[:199] + "…"
        # Strip ## Takeaway before packing so part 1 does not pay Takeaway twice
        # (every part gets `## Takeaway` + its sections consistently).
        knowledge_body = self._strip_provenance_sections(node.content, strip_takeaway=True)

        now = self._utc_now()
        now_iso = format_utc_datetime(now)
        sources = sorted({*node.metadata.sources, node.id})
        sources_block = "## Sources\n\n" + "\n".join(f"- {source}" for source in sources)
        sources_bytes = len(sources_block.encode("utf-8")) + 2
        takeaway_bytes = len(f"## Takeaway\n\n{takeaway_text}\n\n## Details\n\n".encode("utf-8"))
        sample_id = generate_node_id(f"{node.title} — part 1", now)
        # Sibling link format: `[[<id>_part_N]]\n` (or compact chain when many parts).
        link_bytes_per_sibling = len(sample_id.encode("utf-8")) + 12

        from synapse.models import infer_project

        resolved_project = (node.metadata.project or "").strip() or infer_project(node)

        # Compute chunk packing with headroom for Takeaway/Details wrapper and
        # sibling links (plus ## Sources when the byte cap permits).
        part_specs: list[tuple[str, str, str | None, str]] = []
        final_contents: list[str] = []
        est_links = 1
        chunks: list[str] | None = None
        for _attempt in range(3):
            link_overhead = est_links * link_bytes_per_sibling + 4
            include_sources_block = (sources_bytes + takeaway_bytes + link_overhead + 128) <= limit
            wrapper_overhead = takeaway_bytes + link_overhead + (sources_bytes if include_sources_block else 0)
            # Reserve wrapper overhead, while never shrinking below what the
            # node's own smallest multi-section chunks need if the limit is small.
            effective_limit = limit - wrapper_overhead
            if effective_limit < 64:
                chunks = None
                break
            chunks = self._pack_sections(knowledge_body, effective_limit)
            if chunks is None:
                break

            part_specs = []
            used_ids: set[str] = set()
            for index, chunk in enumerate(chunks, start=1):
                part_type = node.metadata.okf_type
                part_body = chunk
                if part_type and self._missing_required_sections(part_body, part_type):
                    # A slice rarely carries every required section of a multi-section
                    # type (e.g. pitfall Symptom/Cause/Fix); wrap it cleanly as a
                    # valid OKF `fact` so it stays typed and handbook-indexed.
                    part_type = "fact"
                    if not self._has_heading(part_body, "Details"):
                        part_body = self._wrap_as_fact_body(part_body, takeaway_text)
                elif not self._has_heading(part_body, "Takeaway"):
                    part_body = f"## Takeaway\n\n{takeaway_text}\n\n{part_body}"
                if include_sources_block:
                    part_body = f"{part_body.rstrip()}\n\n{sources_block}"
                labeled_title = f"{node.title} — part {index}"
                part_id = generate_node_id(labeled_title, now)
                # Slugs drop the part number, so every sibling needs a distinct id.
                # Strip any trailing "_part" first: with the numbered title already
                # yielding "..._part", appending blindly produced "_part_part_2".
                if part_id in used_ids or store.get_node(part_id) is not None:
                    base_part_id = re.sub(r"(?:_part)+$", "", part_id)
                    part_id = f"{base_part_id}_part_{index}"
                used_ids.add(part_id)
                part_specs.append((part_id, labeled_title, part_type, part_body))

            all_part_ids = [spec[0] for spec in part_specs]
            # Link every sibling directly when all-pairs fits cleanly; for many-part
            # splits under tight caps, fall back to a directed ring (1 link per part)
            # so every shard stays <= limit while preserving full graph connectivity.
            candidate_contents: list[str] = []
            max_final_bytes = 0
            for idx, (part_id, _title, _ptype, part_body) in enumerate(part_specs):
                all_siblings = [other_id for other_id in all_part_ids if other_id != part_id]
                linked = part_body.rstrip("\n") + "\n\n" + "\n".join(f"[[{s}]]" for s in all_siblings) + "\n"
                if len(linked.encode("utf-8")) > limit and len(all_part_ids) > 2:
                    next_sibling = all_part_ids[(idx + 1) % len(all_part_ids)]
                    linked = part_body.rstrip("\n") + f"\n\n[[{next_sibling}]]\n"
                size = len(linked.encode("utf-8"))
                max_final_bytes = max(max_final_bytes, size)
                candidate_contents.append(linked)

            if max_final_bytes <= limit:
                final_contents = candidate_contents
                break
            next_est = est_links + 1
            if next_est > max(1, len(part_specs) - 1):
                chunks = None
                break
            est_links = next_est

        if not chunks or not final_contents:
            warnings.append(
                DreamerWarning(
                    code="oversized_split_impossible",
                    node_id=node.id,
                    message=(
                        "Node exceeds the size cap but cannot be split into <= "
                        f"{limit}-byte typed parts; keeping it as is."
                    ),
                )
            )
            return None
        parts: list[Node] = []
        for (part_id, labeled_title, part_type, _part_body), linked_content in zip(part_specs, final_contents, strict=True):
            metadata = self._knowledge_metadata(
                node_id=part_id,
                title=labeled_title,
                okf_type=part_type,
                okf_version=1 if part_type else None,
                sources=sources,
                tags=sorted({*node.metadata.tags, "dreamer-split"}),
                now_iso=now_iso,
                project=resolved_project,
                t_valid=now_iso,
            )
            parts.append(self._persist(store, Node(metadata=metadata, content=linked_content)))

        bytes_after = 0
        for part in parts:
            siblings = [other.id for other in parts if other.id != part.id]
            store.upsert_edges(part.id, siblings)
            bytes_after += len(part.content.encode("utf-8"))

        superseded = self._with_metadata(
            node,
            status=NodeStatus.SUPERSEDED,
            superseded_by=parts[0].id,
            t_invalid=now_iso,
        )
        self._persist(store, superseded)
        self._archive_nodes([node], reason="oversized_split", warnings=warnings)

        bytes_before = len(node.content.encode("utf-8"))
        self._logger.info(
            "Split oversized node",
            extra={"node_id": node.id, "parts": [part.id for part in parts]},
        )
        return CompactionResult(node_id=node.id, bytes_before=bytes_before, bytes_after=bytes_after)

    @staticmethod
    def _has_heading(content: str, heading_name: str) -> bool:
        return heading_name.casefold() in {name.casefold() for name in _split_sections(content)}

    @staticmethod
    def _extract_takeaway_line(content: str) -> str:
        sections = _split_sections(content)
        for name, body in sections.items():
            if name.casefold() == "takeaway":
                for line in body.splitlines():
                    stripped = line.strip()
                    if stripped:
                        return stripped
        return ""

    @staticmethod
    def _wrap_as_fact_body(chunk: str, takeaway_text: str) -> str:
        """Ensure a split shard is a valid OKF ``fact`` (Takeaway + Details) verbatim."""

        lines = chunk.splitlines()
        takeaway_lines: list[str] = []
        detail_lines: list[str] = []
        in_takeaway = False
        fence: str | None = None
        for line in lines:
            stripped = line.strip()
            marker = stripped[:3]
            if fence is None and marker in {"```", "~~~"}:
                fence = marker
            elif fence is not None and stripped.startswith(fence):
                fence = None
            elif fence is None and stripped.startswith("## "):
                heading = stripped[3:].strip()
                if heading.casefold() == "takeaway":
                    in_takeaway = True
                    continue
                in_takeaway = False
                detail_lines.append(line)
                continue
            if in_takeaway:
                takeaway_lines.append(line)
            else:
                detail_lines.append(line)
        resolved_takeaway = next((ln.strip() for ln in takeaway_lines if ln.strip()), "") or takeaway_text
        details_body = "\n".join(detail_lines).strip()
        return f"## Takeaway\n\n{resolved_takeaway}\n\n## Details\n\n{details_body}"

    @staticmethod
    def _pack_sections(content: str, limit: int) -> list[str] | None:
        """Pack knowledge ``##`` sections into <= *limit* byte chunks, or None when impossible.

        Never emits a standalone provenance-only shard (``## Sources`` /
        ``## Related`` / ``## Evidence``) or a Takeaway-only shard.
        """

        blocks: list[tuple[str | None, list[str]]] = []
        current_heading: str | None = None
        current: list[str] = []
        fence: str | None = None
        for line in content.splitlines():
            stripped = line.strip()
            marker = stripped[:3]
            if fence is None and marker in {"```", "~~~"}:
                fence = marker
            elif fence is not None and stripped.startswith(fence):
                fence = None
            elif fence is None and stripped.startswith("## "):
                if any(item.strip() for item in current):
                    blocks.append((current_heading, current))
                current_heading = stripped[3:].strip().casefold()
                current = [line]
                continue
            current.append(line)
        if any(item.strip() for item in current):
            blocks.append((current_heading, current))
        # Drop provenance sections so they can never become standalone shards;
        # the caller attaches ## Sources cleanly to each part.
        knowledge_blocks = [
            (heading, block)
            for heading, block in blocks
            if heading not in _PROVENANCE_SECTION_NAMES
        ]
        # Require at least 2 substantive non-Takeaway knowledge blocks so
        # Takeaway alone never splits away from a single body section.
        substantive_blocks = [
            block for heading, block in knowledge_blocks if heading != "takeaway"
        ]
        if len(substantive_blocks) < 2:
            return None

        chunks: list[str] = []
        pending: list[str] = []
        pending_has_substantive = False
        size = 0
        for heading, block in knowledge_blocks:
            text = "\n".join(block).strip("\n")
            block_bytes = len(text.encode("utf-8")) + 2
            if block_bytes > limit:
                return None  # an unsplittable single section
            if pending and pending_has_substantive and size + block_bytes > limit:
                chunks.append("\n\n".join(pending))
                pending = []
                pending_has_substantive = False
                size = 0
            elif pending and not pending_has_substantive and size + block_bytes > limit:
                # Takeaway + first substantive section alone exceeds the cap.
                return None
            pending.append(text)
            if heading != "takeaway":
                pending_has_substantive = True
            size += block_bytes
        if pending:
            if not pending_has_substantive:
                return None
            chunks.append("\n\n".join(pending))
        return chunks if len(chunks) > 1 else None
    def _materialize_compaction(
        self,
        node: Node,
        payload: object,
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> CompactionResult | None:
        """Rewrite one oversized node in place when it shrinks and keeps anchors."""

        okf_type = node.metadata.okf_type or ""
        if not isinstance(payload, Mapping):
            warnings.append(
                DreamerWarning(
                    code="compaction_invalid_payload",
                    node_id=node.id,
                    message="Compaction payload was not a JSON object; skipping node.",
                )
            )
            return None
        takeaway = str(payload.get("takeaway") or "").strip()
        body = self._strip_provenance_sections(str(payload.get("body") or "").strip())
        title = str(payload.get("title") or "").strip() or node.title
        if not takeaway or "\n" in takeaway or len(takeaway) > 200 or not body:
            warnings.append(
                DreamerWarning(
                    code="compaction_payload_invalid",
                    node_id=node.id,
                    message="Compaction payload needs a single-line takeaway (<= 200 chars) and a body; skipping node.",
                )
            )
            return None
        if okf_type in TYPE_REQUIRED_SECTIONS and self._missing_required_sections(body, okf_type):
            warnings.append(
                DreamerWarning(
                    code="compaction_missing_sections",
                    node_id=node.id,
                    message=f"Compaction body is missing required {okf_type} sections; skipping node.",
                )
            )
            return None
        if not slugify_english_title(title):
            title = node.title

        provenance = self._provenance_block(node.content)
        content = "\n\n".join(part for part in (f"## Takeaway\n\n{takeaway}", body, provenance) if part.strip())
        content = self._verify_or_reject_anchors(
            [node], content, okf_type, list(node.metadata.sources), title, warnings
        )
        if content is None:
            return None

        limit = self.config.dreamer.thresholds.oversized_compaction_bytes
        bytes_before = len(node.content.encode("utf-8"))
        bytes_after = len(content.encode("utf-8"))
        if bytes_after >= bytes_before:
            warnings.append(
                DreamerWarning(
                    code="compaction_did_not_shrink",
                    node_id=node.id,
                    message="Compaction produced no size reduction; keeping the original node.",
                )
            )
            return None
        if bytes_after > limit:
            # Still over the cap: an in-place rewrite that stays oversized would
            # simply be re-picked next run. Fall through to the deterministic
            # split so the node ends up <= the cap.
            warnings.append(
                DreamerWarning(
                    code="compaction_still_oversized",
                    node_id=node.id,
                    message=(
                        f"Compaction stayed above the {limit}-byte cap "
                        f"({bytes_after} B); splitting instead."
                    ),
                )
            )
            return None

        updated = Node(
            metadata=node.metadata.model_copy(update={"title": title}), content=content, file_path=node.file_path
        )
        self._persist(store, updated)
        self._logger.info(
            "Compacted oversized node",
            extra={"node_id": node.id, "bytes_before": bytes_before, "bytes_after": bytes_after},
        )
        return CompactionResult(
            node_id=node.id,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
        )

    # -- Dreamer 2.0 validation helpers --------------------------------------

    @staticmethod
    def _missing_required_sections(body: str, okf_type: str) -> list[str]:
        """Required OKF section names absent from *body* (case-insensitive)."""

        present = {name.casefold() for name in _split_sections(body)}
        return [name for name in TYPE_REQUIRED_SECTIONS.get(okf_type, ()) if name.casefold() not in present]

    @staticmethod
    def _fatal_okf_findings(content: str, okf_type: str, sources: list[str], title: str) -> list[str]:
        """OKF findings that must block a consolidation product.

        Provenance sections are deliberately optional (the write path treats
        ``## Sources`` / ``## Related`` / ``## Evidence`` as non-knowledge), so
        only type-required sections and invalid takeaways block a write.
        """

        findings = validate_okf_node(
            node_type=NodeType.PERSISTENT.value,
            content=content,
            okf_type=okf_type,
            sources=sources,
            title=title,
        )
        fatal: list[str] = []
        for finding in findings:
            code = finding.code
            if code not in _FATAL_OKF_CODES and not code.startswith(_FATAL_OKF_PREFIXES):
                continue
            if code.startswith("okf_missing_section:"):
                name = code.split(":", 1)[1].strip().casefold()
                if name in _PROVENANCE_SECTION_NAMES:
                    continue
            fatal.append(code)
        return fatal

    def _verify_or_reject_anchors(
        self,
        members: list[Node],
        content: str,
        okf_type: str,
        sources: list[str],
        title: str,
        warnings: list[DreamerWarning],
    ) -> str | None:
        """Nighttime anchor-token verification (§3.3).

        Merges/compactions may tighten wording and de-duplicate, but concrete
        literals (error codes, paths, versions, numbers, durations) must survive
        within the tolerance budget. Products exceeding the loss budget are
        rejected; within-tolerance products are kept clean with no raw-token
        backfill line appended to the body.
        """

        from synapse.server.service import _missing_supersede_anchors, _supersede_anchor_tokens

        missing: list[str] = []
        all_anchors: set[str] = set()
        for member in members:
            # Evidence quotes are verbatim transcript excerpts that a merge never
            # re-renders (the winner keeps the member ids as Sources). Judge the
            # merge on the member's knowledge body only, leaving the write-path
            # supersede gate (which treats Evidence numbers as facts) untouched.
            member_body = self._strip_provenance_sections(member.content, strip_takeaway=False)
            try:
                all_anchors.update(_supersede_anchor_tokens(member_body))
            except Exception:  # noqa: BLE001 — anchor extraction is best effort
                continue
            missing.extend(_missing_supersede_anchors(member_body, content))
        total = len(all_anchors)
        missing = list(dict.fromkeys(missing))
        if total and len(missing) > max(1, int(total * _ANCHOR_LOSS_TOLERANCE)):
            warnings.append(
                DreamerWarning(
                    code="anchor_token_loss",
                    node_id=members[0].id,
                    message=(
                        f"Rejected consolidation product: {len(missing)} of {total} anchor tokens went missing "
                        f"(> {int(_ANCHOR_LOSS_TOLERANCE * 100)}%) — {missing[:10]}"
                    ),
                )
            )
            return None
        fatal = self._fatal_okf_findings(content, okf_type, sources, title)
        if fatal:
            warnings.append(
                DreamerWarning(
                    code="consolidation_okf_invalid",
                    node_id=members[0].id,
                    message=f"Rejected consolidation product; OKF validation failed: {fatal}",
                )
            )
            return None
        return content

    @staticmethod
    def _render_consolidated_body(
        takeaway: str,
        body: str,
        sources: list[str],
        related: list[str],
    ) -> str:
        """Assemble a consolidated OKF body: Takeaway + type sections + links."""

        parts: list[str] = ["## Takeaway", "", takeaway, "", body.strip()]
        if related:
            parts += ["", "## Related", "", *(f"- [[{link}]]" for link in related)]
        parts += ["", "## Sources", "", *(f"- {source}" for source in sources), ""]
        return "\n".join(parts).strip()

    @staticmethod
    def _provenance_block(content: str) -> str:
        """Extract the ``## Sources`` / ``## Related`` / ``## Evidence`` blocks verbatim."""

        blocks: list[str] = []
        current: list[str] = []
        in_provenance = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                if in_provenance:
                    blocks.append("\n".join(current).strip("\n"))
                    current = []
                    in_provenance = False
                name = stripped[3:].strip().casefold()
                if name in _PROVENANCE_SECTION_NAMES:
                    in_provenance = True
                    current = [line]
                continue
            if in_provenance:
                current.append(line)
        if in_provenance and current:
            blocks.append("\n".join(current).strip("\n"))
        return "\n\n".join(block for block in blocks if block.strip())

    @staticmethod
    def _strip_provenance_sections(content: str, *, strip_takeaway: bool = True) -> str:
        """Strip ## Sources / ## Related / ## Evidence (and optionally ## Takeaway) from content."""

        skip_names = set(_PROVENANCE_SECTION_NAMES)
        if strip_takeaway:
            skip_names.add("takeaway")
        kept: list[str] = []
        skip = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                name = stripped[3:].strip().casefold()
                skip = name in skip_names
            if not skip:
                kept.append(line)
        return "\n".join(kept).strip()

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
            # A node retired by Stage 4c/4d (superseded + archived) must not be
            # resurrected: Stage 2 decisions predate those stages.
            if not self._is_live_active(node):
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
            now = self._utc_now()
            now_iso = format_utc_datetime(now)
            # Dreamer 2.0 (§3.4): synthesize the condensed set with the LLM
            # first; the deterministic condenser is only the fallback (it used
            # to be the only path and produced zero-information ': User' bodies).
            synthesized = self._synthesize_condensation(to_condense, warnings)
            if synthesized is not None:
                draft, okf_type = synthesized
            else:
                draft = DeterministicArchiveCondenser().synthesize(to_condense, now=now)
                okf_type = "decision"
            new_id = generate_node_id(draft.title, now)

            new_node = Node(
                metadata=self._knowledge_metadata(
                    node_id=new_id,
                    title=draft.title,
                    okf_type=okf_type,
                    okf_version=1,
                    sources=list(draft.source_node_ids),
                    tags=list(draft.tags),
                    now_iso=now_iso,
                    t_valid=now_iso,
                ),
                content=draft.content,
            )
            self._persist(store, new_node)
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

    def _synthesize_condensation(
        self,
        nodes: list[Node],
        warnings: list[DreamerWarning],
    ) -> tuple[CondensationDraft, str] | None:
        """LLM condensation into a real OKF ``fact`` node (§3.4).

        Returns ``None`` (after a warning) whenever the LLM path is unavailable
        or its product fails validation, so the caller can fall back to the
        deterministic condenser.
        """

        try:
            client = self._background_client()
            payload = client.sample_json(
                prompt=build_synthesis_prompt(tuple(nodes)),
                system_prompt=_SYNTHESIS_SYSTEM,
                max_tokens=self.config.decider.max_tokens,
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            self._logger.warning("Condensation sampling failed; using deterministic fallback", exc_info=exc)
            warnings.append(
                DreamerWarning(
                    code="condense_sampling_failed",
                    message=f"LLM condensation failed: {exc}. Falling back to deterministic condensation.",
                )
            )
            return None
        if not isinstance(payload, Mapping):
            warnings.append(
                DreamerWarning(
                    code="condense_invalid_payload",
                    message="LLM condensation payload was not a JSON object; using deterministic fallback.",
                )
            )
            return None
        title = str(payload.get("title") or "").strip()
        takeaway = str(payload.get("takeaway") or "").strip()
        details = str(payload.get("details") or "").strip()
        member_ids = sorted(node.id for node in nodes)
        sources = sorted({*member_ids, *(s for node in nodes for s in node.metadata.sources)})
        if not title or not slugify_english_title(title) or not takeaway or "\n" in takeaway:
            warnings.append(
                DreamerWarning(
                    code="condense_invalid_payload",
                    message="LLM condensation needs an English title and a single-line takeaway; using deterministic fallback.",
                )
            )
            return None
        if not details:
            warnings.append(
                DreamerWarning(
                    code="condense_invalid_payload",
                    message="LLM condensation returned no details; using deterministic fallback.",
                )
            )
            return None
        if len(takeaway) > 200:
            takeaway = takeaway[:199] + "…"
        content = self._render_consolidated_body(takeaway, f"## Details\n\n{details}", sources, [])
        verified = self._verify_or_reject_anchors(nodes, content, "fact", sources, title, warnings)
        if verified is None:
            return None
        return (
            CondensationDraft(
                title=title,
                content=verified,
                tags=("condensed", "archive-synthesis"),
                source_node_ids=tuple(sources),
            ),
            "fact",
        )

    def _execute_links(
        self,
        decisions: list[LinkDecision],
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> None:
        for decision in decisions:
            node_a = store.get_node(decision.node_a_id)
            node_b = store.get_node(decision.node_b_id)
            # Never resurrect a node retired by Stage 4c/4d: linking would
            # recreate its Markdown file under active/. ``_is_live_active``
            # also rejects ids that vanished from the store.
            if not self._is_live_active(node_a) or not self._is_live_active(node_b):
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
                self._store_written_path(updated_a)
                a_updated = True

            if link_tag_a not in node_b.content:
                updated_b = Node(
                    metadata=node_b.metadata,
                    content=node_b.content.rstrip() + "\n\n" + link_tag_a + "\n",
                    file_path=node_b.file_path,
                )
                self._store_written_path(updated_b)
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
            # Stage 4c/4d may have retired a disputant already; rewriting it
            # would resurrect an archived node into active/.
            if not self._is_live_active(node_a) or not self._is_live_active(node_b):
                continue

            if decision.decision == "supersede_a":
                self._mark_superseded(node_a, node_b, store)
            elif decision.decision == "supersede_b":
                self._mark_superseded(node_b, node_a, store)
            elif decision.decision == "both_valid":
                self._clear_disputed(node_a, store)
                self._clear_disputed(node_b, store)

    def _mark_superseded(self, loser: Node, winner: Node, store: SQLiteNodeStore) -> None:
        """Mark *loser* as superseded by *winner* and clear winner's disputed status."""
        self._persist(
            store, self._with_metadata(loser, status=NodeStatus.SUPERSEDED, superseded_by=winner.id)
        )

        # The winner was part of a disputed pair; restore it to active so it is
        # not re-discovered as disputed on the next Dreamer cycle.
        if winner.metadata.status == NodeStatus.DISPUTED:
            self._clear_disputed(winner, store)

    def _clear_disputed(self, node: Node, store: SQLiteNodeStore) -> None:
        """Reset a disputed node back to active."""
        self._persist(store, self._with_metadata(node, status=NodeStatus.ACTIVE))

    def _resolve_supersession_chain(
        self, node: Node, store: SQLiteNodeStore, *, max_depth: int = 64
    ) -> tuple[Node | None, str]:
        """Follow superseded_by to the terminal node of the chain (cycle-safe).

        Chained supersession (A superseded by B, B superseded by C, ...) left
        most superseded nodes un-archivable because only the direct successor
        was consulted. The terminal node is the meaningful successor.

        Returns (terminal_node, outcome) where outcome is:
          "terminal"          — a node with no superseded_by was reached
          "missing_successor" — the chain references a node absent from the index
          "cycle"             — the chain loops back on itself
        """

        current = node
        seen: set[str] = {node.id}
        for _ in range(max_depth):
            successor_id = current.metadata.superseded_by
            if not successor_id:
                return current, "terminal"
            if successor_id in seen:
                return current, "cycle"
            successor = store.get_node(successor_id)
            if successor is None:
                return current, "missing_successor"
            seen.add(successor.id)
            current = successor
        return current, "cycle"

    def _archive_superseded(
        self,
        superseded: list[Node],
        store: SQLiteNodeStore,
        warnings: list[DreamerWarning],
    ) -> list[str]:
        """Archive a superseded node when its supersession chain:
          - terminates at an ACTIVE node with an existing file; OR
          - terminates at a node missing from the index (already archived or
            deleted — the terminal is just as obsolete as the node itself); OR
          - has no superseded_by at all (chain end with no successor).
        Keep the node only when the terminal is DISPUTED (a live disagreement)
        or the chain is cyclic (logged as a warning).
        """
        valid: list[Node] = []
        for node in superseded:
            if not node.metadata.superseded_by:
                valid.append(node)
                continue

            terminal, outcome = self._resolve_supersession_chain(node, store)
            if outcome == "cycle":
                warnings.append(
                    DreamerWarning(
                        code="cyclic_supersession",
                        node_id=node.id,
                        message=(
                            f"Cannot archive superseded node '{node.id}': supersession chain is cyclic."
                        ),
                    )
                )
                continue
            if outcome == "missing_successor":
                # Terminal of the chain references a node that is no longer in
                # the index (archived or deleted): just as obsolete as the node.
                valid.append(node)
                continue
            terminal_path = self._active_source_path(terminal)
            if terminal.metadata.status is NodeStatus.DISPUTED:
                warnings.append(
                    DreamerWarning(
                        code="disputed_superseder",
                        node_id=node.id,
                        message=(
                            f"Cannot archive superseded node '{node.id}': chain terminal "
                            f"'{terminal.id}' is disputed (live disagreement)."
                        ),
                    )
                )
                continue
            if terminal.metadata.status is not NodeStatus.ACTIVE or terminal_path is None:
                warnings.append(
                    DreamerWarning(
                        code="invalid_superseder",
                        node_id=node.id,
                        message=(
                            f"Cannot archive superseded node '{node.id}': chain terminal "
                            f"'{terminal.id}' is not active."
                        ),
                    )
                )
                continue
            valid.append(node)

        archived = self._archive_nodes(valid, reason="superseded", warnings=warnings)
        return [n.id for n in archived]


    def _repoint_superseded_links(self, store: SQLiteNodeStore) -> int:
        """Deterministically repoint ``[[old_id]]`` wiki links and ``sources`` to active chain terminals.

        When Dreamer merges or retires nodes (Stage 4c/4d/5), inbound ``[[old_id]]``
        links in surviving active nodes would otherwise dangle into ``.archive/``.
        Follows ``superseded_by`` chains to live ACTIVE terminals and rewrites
        referencing nodes in place before ``startup_sync`` rebuilds ``edges``.
        """

        superseded_nodes = store.find_by_status(NodeStatus.SUPERSEDED)
        if not superseded_nodes:
            return 0

        remap: dict[str, str] = {}
        for node in superseded_nodes:
            if not node.metadata.superseded_by:
                continue
            terminal, outcome = self._resolve_supersession_chain(node, store)
            if outcome == "terminal" and terminal is not None and self._is_live_active(terminal):
                if terminal.id != node.id:
                    remap[node.id] = terminal.id
        if not remap:
            return 0

        def _replace_wiki_link(match: re.Match[str]) -> str:
            raw = match.group(1)
            target, sep, alias = raw.partition("|")
            stripped = target.strip()
            replacement = remap.get(stripped)
            if replacement is None:
                return match.group(0)
            return f"[[{replacement}|{alias}]]" if sep else f"[[{replacement}]]"

        repointed_count = 0
        for node in store.find_by_status(NodeStatus.ACTIVE):
            if not self._is_live_active(node):
                continue
            local_remap = {old: new for old, new in remap.items() if new != node.id}
            if not local_remap:
                continue

            links_in_body = extract_wiki_links(node.content)
            hits = sum(1 for link in links_in_body if link in local_remap)
            new_content = WIKI_LINK_PATTERN.sub(_replace_wiki_link, node.content) if hits else node.content

            old_sources = list(node.metadata.sources)
            new_sources = list(dict.fromkeys(local_remap.get(src, src) for src in old_sources))
            sources_changed = new_sources != old_sources

            if new_content == node.content and not sources_changed:
                continue

            updated = node.model_copy(update={"content": new_content})
            if sources_changed:
                updated = self._with_metadata(updated, sources=new_sources)
            self._persist(store, updated)
            repointed_count += max(hits, 1 if sources_changed else 0)

        if repointed_count:
            self._logger.info("Repointed superseded wiki links", extra={"repointed_links": repointed_count})
        return repointed_count
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

    def _is_live_active(self, node: Node | None) -> bool:
        """True only for an ACTIVE node whose Markdown file still exists.

        Stage 5 executes decisions captured before Stage 4c/4d ran; a node those
        stages already superseded and archived must never be resurrected into
        ``active/``. Checks status, this run's archive log, and the file itself.
        """

        if node is None or node.metadata.status is not NodeStatus.ACTIVE:
            return False
        if node.id in self._archived_this_run:
            return False
        raw = node.file_path
        if raw is None or str(raw) in {"", "."}:
            return False
        candidate = raw if raw.is_absolute() else self.runtime_paths.base / raw
        if candidate.exists():
            return True
        # The index may still point at a pre-hierarchy path; locate the file by
        # its identity (metadata.id), never by where it used to be.
        return locate_active_node_file(self.runtime_paths.base, node.id) is not None

    def _store_written_path(self, node: Node, *, base_path: Path | None = None) -> Node:
        """Write *node* to its canonical path and stamp the path it landed on.

        Every Dreamer write (hub, merge, condensation, split part, compaction,
        supersession) goes through here so ``store.upsert_node`` always records
        the real ``active/<project>/<kebab-slug>.md`` location.
        """

        written = write_node_file(node, base_path=base_path or self.runtime_paths.base)
        relative = written.relative_to(self.runtime_paths.base)
        return node.model_copy(update={"file_path": relative})

    def _persist(self, store: SQLiteNodeStore, node: Node) -> Node:
        """Write *node* to its canonical path and upsert it into the index."""

        written = self._store_written_path(node)
        store.upsert_node(written)
        return written

    @staticmethod
    def _with_metadata(node: Node, **fields: object) -> Node:
        """Copy *node* with *fields* applied to its metadata (content/path preserved)."""

        return node.model_copy(update={"metadata": node.metadata.model_copy(update=fields)})

    @staticmethod
    def _knowledge_metadata(
        *,
        node_id: str,
        title: str,
        okf_type: str | None,
        sources: list[str],
        tags: list[str],
        now_iso: str,
        project: str | None = None,
        okf_version: int | None = None,
        t_valid: str | None = None,
    ) -> NodeMetadata:
        """Frontmatter for a Dreamer-produced ACTIVE persistent knowledge node."""

        return NodeMetadata.model_validate(
            {
                "id": node_id,
                "title": title,
                "type": NodeType.PERSISTENT.value,
                "status": NodeStatus.ACTIVE.value,
                "okf_type": okf_type,
                "okf_version": okf_version,
                "sources": sources,
                "project": project,
                "tags": tags,
                "created_at": now_iso,
                "last_accessed": now_iso,
                "access_count": 0,
                "t_valid": t_valid,
            }
        )

    def _archive_nodes(
        self,
        nodes: Iterable[Node],
        *,
        reason: str,
        warnings: list[DreamerWarning],
    ) -> list[Node]:
        archived: list[Node] = []
        for node in nodes:
            source_path = self._active_source_path(node)
            destination = archive_node_path(self.runtime_paths.archive, node.id)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if source_path is None or not source_path.exists():
                warnings.append(
                    DreamerWarning(
                        code="missing_source_file",
                        node_id=node.id,
                        message=f"Cannot archive node '{node.id}': source file does not exist.",
                    )
                )
                continue
            if destination.exists():
                src_text = source_path.read_text(encoding="utf-8")
                dst_text = destination.read_text(encoding="utf-8")
                if src_text == dst_text:
                    source_path.unlink(missing_ok=True)
                else:
                    collisions_dir = self.runtime_paths.archive / ".collisions"
                    collisions_dir.mkdir(parents=True, exist_ok=True)
                    sha8 = hashlib.sha1(dst_text.encode("utf-8")).hexdigest()[:8]
                    collision_path = collisions_dir / f"{node.id}.{sha8}.md"
                    if not collision_path.exists():
                        atomic_write_text(collision_path, dst_text)
                    atomic_write_text(destination, src_text)
                    source_path.unlink(missing_ok=True)
                    self._logger.info(
                        "Quarantined divergent archive collision",
                        extra={"node_id": node.id, "collision_path": str(collision_path)},
                    )
            else:
                dst_text = source_path.read_text(encoding="utf-8")
                atomic_write_text(destination, dst_text)
                source_path.unlink(missing_ok=True)
            # The file has left active/. Record it so later stages in this same
            # run know the node is no longer live, even though the write index
            # (``nodes.file_path``) only catches up at the closing startup sync.
            self._archived_this_run.add(node.id)
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

    def _active_source_path(self, node: Node) -> Path | None:
        """On-disk active path for *node*, tolerating a stale indexed path."""

        raw = node.file_path
        if raw is not None and str(raw) not in {"", "."}:
            candidate = raw if raw.is_absolute() else self.runtime_paths.base / raw
            if candidate.exists():
                return candidate
        return locate_active_node_file(self.runtime_paths.base, node.id)

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

    def _record_metrics(
        self,
        report: DreamerReport,
        store: SQLiteNodeStore,
        *,
        batch_size: int,
        duration_ms: int,
    ) -> None:
        triage_counts = {"keep": 0, "condense": 0, "archive": 0}
        for decision in report.triage:
            if decision.decision in triage_counts:
                triage_counts[decision.decision] += 1
        conflicts_superseded = sum(
            1 for decision in report.conflicts_resolved if decision.decision in {"supersede_a", "supersede_b"}
        )
        conflicts_both_valid = sum(1 for decision in report.conflicts_resolved if decision.decision == "both_valid")
        sampling_failures = sum(1 for warning in report.warnings if warning.code.endswith("_sampling_failed"))
        try:
            store.record_dreamer_run(
                DreamerRunMetrics(
                    started_at=report.started_at,
                    completed_at=report.completed_at,
                    duration_ms=duration_ms,
                    batch_size=batch_size,
                    stale_scanned=report.scanned.get("stale", 0),
                    superseded_scanned=report.scanned.get("superseded", 0),
                    disputed_scanned=report.scanned.get("disputed", 0),
                    missing_link_pairs_scanned=report.scanned.get("missing_link_pairs", 0),
                    triage_keep=triage_counts["keep"],
                    triage_condense=triage_counts["condense"],
                    triage_archive=triage_counts["archive"],
                    links_added=len(report.links_added),
                    conflicts_superseded=conflicts_superseded,
                    conflicts_both_valid=conflicts_both_valid,
                    archived=len(report.archived),
                    condensed=len(report.condensed),
                    warnings=len(report.warnings),
                    sampling_failures=sampling_failures,
                    synthesized_hubs=len(report.synthesized),
                    semantic_merges=len(report.merged),
                    oversized_compacted=len(report.compacted),
                    repointed_links=report.repointed_links,
                )
            )
        except sqlite3.DatabaseError as exc:  # pragma: no cover - metrics must never break lifecycle maintenance
            self._logger.warning("Failed to record Dreamer metrics", exc_info=exc)

    def _prescreen_merge_cluster(self, members: list[Node]) -> tuple[str, float] | None:
        state = {
            "members": [
                {
                    "id": member.id,
                    "title": member.title,
                    "okf_type": member.metadata.okf_type,
                    "snippet": member.content[:600],
                }
                for member in members
            ]
        }
        questions = {
            "merge_relation": {
                "type": "choice",
                "instructions": "How do these memory nodes relate to each other? Pick the closest relationship.",
                "criteria": {
                    "synonymous": "They state the same fact or rule in different words; one can replace the others.",
                    "complementary": "They cover different aspects of one topic; merged, nothing is lost.",
                    "distinct_constraints": "They look alike but each carries a different constraint, scope or condition.",
                    "conflict": "They assert incompatible things about the same subject.",
                },
            }
        }
        decision = self._decision_client.decide(state, questions, offline=True)
        choice, confidence = _extract_decision_choice(decision, "merge_relation")
        if choice is None:
            return None
        return choice, confidence

    def _export_skill_promotion_proposals(self, store: SQLiteNodeStore) -> None:
        try:
            eligible = [
                node
                for node in store.list_nodes({"status": NodeStatus.ACTIVE, "type": NodeType.PERSISTENT})
                if node.metadata.okf_type in ("procedure", "pitfall")
                and len(node.metadata.sources) >= 3
                and (node.metadata.used_count >= 5 or node.metadata.injected_count >= 10)
            ]
            eligible.sort(
                key=lambda node: (
                    -len(node.metadata.sources),
                    -node.metadata.used_count,
                    -node.metadata.injected_count,
                    node.id,
                )
            )
            proposals = []
            for node in eligible[:15]:
                proposals.append(
                    {
                        "node_id": node.id,
                        "title": node.title,
                        "project": node.metadata.project,
                        "okf_type": node.metadata.okf_type,
                        "sources_count": len(node.metadata.sources),
                        "used_count": node.metadata.used_count,
                        "injected_count": node.metadata.injected_count,
                        "takeaway": _extract_takeaway(node.content),
                    }
                )
            audit_dir = self.runtime_paths.base / ".audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "generated_at": format_utc_datetime(self._utc_now()),
                "proposals": proposals,
            }
            atomic_write_text(
                audit_dir / "skill-promotion-proposals.json",
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
        except Exception as exc:  # noqa: BLE001 — best-effort audit export, never a blocker
            self._logger.warning("Skill promotion proposal export failed: %s", exc)


def _duration_ms(started_at: datetime, completed_at: datetime) -> int:
    return max(0, int((completed_at - started_at).total_seconds() * 1000))


def _extract_decision_choice(payload: Mapping[str, object] | None, key: str) -> tuple[str | None, float]:
    if not isinstance(payload, Mapping):
        return None, 0.0
    answers = payload.get("answers")
    raw_answer = answers.get(key) if isinstance(answers, Mapping) else payload.get(key)
    confidence = 0.0
    if isinstance(raw_answer, Mapping):
        choice = raw_answer.get("choice") or raw_answer.get("value") or raw_answer.get("answer")
        raw_conf = raw_answer.get("confidence", payload.get("confidence", 0.0))
    else:
        choice = raw_answer
        raw_conf = payload.get("confidence", 0.0)
    try:
        confidence = float(raw_conf or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if not isinstance(choice, str) or not choice.strip():
        return None, confidence
    return choice.strip(), confidence


def _extract_takeaway(content: str) -> str:
    match = re.search(r"(?ms)^## Takeaway\s*\n(.*?)(?=^## |\Z)", content)
    if not match:
        return ""
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
