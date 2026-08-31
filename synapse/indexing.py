"""Index rebuild, startup checks, and health helpers for Phase 3."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from synapse.config import EmbeddingSettings, SynapseConfig
from synapse.embedding import create_embedding_engine
from synapse.storage import scan_markdown_files, scan_markdown_nodes
from synapse.storage.sqlite import DatabaseIntegrityReport, SQLiteNodeStore
from synapse.sync import SyncBatchResult, SyncManager
from synapse.utils.runtime import RuntimePaths, bootstrap_runtime_directories, get_runtime_paths
from synapse.utils.documents import render_node_document


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EmbeddingProbeReport:
    """Availability summary for the configured embedding engine."""

    status: str
    backend: str
    available: bool
    dimension: int
    message: str


@dataclass(slots=True, frozen=True)
class IndexRebuildResult:
    """Summary returned after a rebuild-index operation."""

    database_path: Path
    indexed_nodes: int
    embedding_status: str
    vector_backend: str
    integrity_ok: bool
    embedding_fingerprint: str


@dataclass(slots=True, frozen=True)
class SystemHealthReport:
    """Health snapshot suitable for CLI output or future API responses."""

    status: str
    components: dict[str, str]
    stats: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    lifecycle_stats: dict[str, Any] = field(default_factory=dict)
    write_stats: dict[str, Any] = field(default_factory=dict)
    database_path: Path | None = None
    embedding_fingerprint: str | None = None
    delta_sync_hook: str = "enabled"
    startup_sync_hook: str = "enabled"


@dataclass(slots=True, frozen=True)
class StartupReport:
    """Startup/integrity summary used by `serve` and future daemon startup."""

    runtime_paths: RuntimePaths
    database_path: Path
    database_integrity: DatabaseIntegrityReport | None
    embedding: EmbeddingProbeReport
    health: SystemHealthReport
    needs_rebuild: bool
    rebuilt: bool
    rebuild_reasons: list[str] = field(default_factory=list)
    rebuild_result: IndexRebuildResult | None = None
    sync_result: SyncBatchResult | None = None


def database_path_for_config(config: SynapseConfig) -> Path:
    """Resolve the SQLite database path from Synapse config."""

    return config.resolve_path(config.memory.base_path) / "synapse.db"


def compute_embedding_fingerprint(config: SynapseConfig) -> str:
    """Compute a stable fingerprint for the configured embedding index settings."""

    provider_settings: dict[str, object] = {
        "provider": config.embedding.provider,
        "model": config.embedding.model,
        "dimension": config.embedding.dimension,
    }
    if config.embedding.provider == "ollama":
        provider_settings.update(
            {
                "base_url": config.providers.ollama.base_url,
                "endpoint": config.providers.ollama.embedding_endpoint,
            }
        )
    elif config.embedding.provider == "remote_api":
        provider_settings.update(
            {
                "base_url": config.providers.remote_api.base_url,
                "endpoint": config.providers.remote_api.embedding_endpoint,
                "api_key_env": config.providers.remote_api.api_key_env,
            }
        )
    payload = json.dumps(provider_settings, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"sha1:{digest}"


def rebuild_index(
    config: SynapseConfig,
    *,
    runtime_paths: RuntimePaths | None = None,
    progress_callback: Callable[[str], None] | None = None,
    logger: logging.Logger | None = None,
) -> IndexRebuildResult:
    """Rebuild the derived SQLite index from Markdown source-of-truth files."""

    selected_logger = logger or LOGGER
    paths = runtime_paths or get_runtime_paths(config)
    db_path = database_path_for_config(config)
    nodes = scan_markdown_nodes(paths.active, relative_to=paths.base)
    engine = create_embedding_engine(config.embedding, providers=config.providers)
    texts = [render_node_document(node) for node in nodes]
    embeddings = engine.embed_batch(texts) if texts else []
    source_mtimes: list[datetime | None] = [
        datetime.fromtimestamp((paths.base / node.file_path).stat().st_mtime, tz=UTC)
        for node in nodes
    ]
    if texts and len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding engine returned {len(embeddings)} vectors for {len(texts)} nodes during rebuild"
        )

    def emit(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    emit(f"Rebuilding index at {db_path} from {len(nodes)} Markdown node(s)...")
    selected_logger.info("Starting index rebuild", extra={"node_count": len(nodes), "database_path": str(db_path)})
    fingerprint = compute_embedding_fingerprint(config)

    with SQLiteNodeStore(db_path, embedding_dimension=config.embedding.dimension or 0) as store:
        def on_progress(progress) -> None:
            emit(f"[{progress.index}/{progress.total}] Processing {progress.file_path} ({progress.node_id})")

        indexed_nodes = store.rebuild_from_nodes(
            nodes,
            embeddings,
            embedding_fingerprint=fingerprint,
            source_mtimes=source_mtimes,
            progress_callback=on_progress,
        )
        integrity = store.check_integrity()
        vector_backend = store.vector_backend

    embedding_status = probe_embedding_engine(config).status
    emit(f"Indexed {indexed_nodes} node(s). Vector backend: {vector_backend}. Integrity: {integrity.integrity_check_result}")
    selected_logger.info(
        "Index rebuild completed",
        extra={
            "node_count": indexed_nodes,
            "database_path": str(db_path),
            "vector_backend": vector_backend,
            "integrity": integrity.integrity_check_result,
        },
    )
    return IndexRebuildResult(
        database_path=db_path,
        indexed_nodes=indexed_nodes,
        embedding_status=embedding_status,
        vector_backend=vector_backend,
        integrity_ok=integrity.ok,
        embedding_fingerprint=fingerprint,
    )


def probe_embedding_engine(config: SynapseConfig) -> EmbeddingProbeReport:
    """Probe whether the configured embedding path is fully available or degraded."""

    probe_settings = _probe_settings(config.embedding)
    engine = create_embedding_engine(probe_settings, providers=config.providers)
    backend = getattr(engine, "backend_name", probe_settings.provider)
    try:
        vector = engine.embed("synapse health probe")
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive network/runtime handling
        return EmbeddingProbeReport(
            status="unavailable",
            backend=str(backend),
            available=False,
            dimension=probe_settings.dimension or 0,
            message=str(exc),
        )

    expected_dimension = probe_settings.dimension or 0
    if len(vector) != expected_dimension:
        return EmbeddingProbeReport(
            status="unavailable",
            backend=str(backend),
            available=False,
            dimension=expected_dimension,
            message="Embedding engine returned an unexpected vector length",
        )

    remote_provider = probe_settings.provider in {"ollama", "remote_api"}
    available = bool(engine.is_available())
    if remote_provider and not available:
        return EmbeddingProbeReport(
            status="degraded",
            backend=str(backend),
            available=False,
            dimension=expected_dimension,
            message="Configured provider unavailable; deterministic fallback is active",
        )

    return EmbeddingProbeReport(
        status="ok",
        backend=str(backend),
        available=True,
        dimension=expected_dimension,
        message="Embedding engine is available",
    )


def collect_health_status(
    config: SynapseConfig,
    *,
    runtime_paths: RuntimePaths | None = None,
    embedding_report: EmbeddingProbeReport | None = None,
    sync_result: SyncBatchResult | None = None,
) -> SystemHealthReport:
    """Collect a health snapshot for status commands and future APIs."""

    paths = runtime_paths or get_runtime_paths(config)
    db_path = database_path_for_config(config)
    probe = embedding_report or probe_embedding_engine(config)
    sync_manager = SyncManager(config, runtime_paths=paths)
    sync_runtime = sync_manager.describe_runtime()
    sync_manager.close()
    warnings: list[str] = []
    thresholds = _effective_dreamer_thresholds(config)
    stats = {
        "total_nodes": 0,
        "active_nodes": 0,
        "superseded_nodes": 0,
        "disputed_nodes": 0,
        "archived_nodes": len(scan_markdown_files(paths.archive)) if paths.archive.exists() else 0,
    }
    components = {
        "sqlite": "missing",
        "wal_mode": "unknown",
        "embedding_model": probe.status,
        "vector_index": "unavailable",
        "file_watcher": sync_runtime.file_watcher,
        "startup_sync": sync_runtime.startup_sync_hook,
    }
    lifecycle_stats: dict[str, Any] = {
        "thresholds": thresholds,
        "current_candidates": {
            "stale_orphans": 0,
            "missing_link_pairs": 0,
            "disputed_pairs": 0,
            "superseded_archival_candidates": 0,
        },
        "missing_link_similarity_histogram": _empty_similarity_histogram(),
        "runs": _empty_dreamer_runs_summary(),
        "decision_totals": _empty_dreamer_decision_totals(),
    }
    write_stats: dict[str, Any] = _empty_write_stats()
    embedding_fingerprint: str | None = None

    try:
        with SQLiteNodeStore(db_path, embedding_dimension=config.embedding.dimension or 0) as store:
            integrity = store.check_integrity()
            counts = store.count_nodes_by_status()
            stats["total_nodes"] = integrity.total_nodes
            stats["active_nodes"] = counts.get("active", 0)
            stats["superseded_nodes"] = counts.get("superseded", 0)
            stats["disputed_nodes"] = counts.get("disputed", 0)
            embedding_fingerprint = store.get_embedding_fingerprint()
            lifecycle_stats = {
                "thresholds": thresholds,
                "current_candidates": {
                    "stale_orphans": len(store.find_orphan_candidates(thresholds["stale_orphan_days"])),
                    "missing_link_pairs": len(
                        store.find_missing_link_pairs(
                            cosine_threshold=thresholds["missing_link_cosine"],
                            recency_days=thresholds["link_weaving_recency_days"],
                        )
                    ),
                    "disputed_pairs": store.count_disputed_pairs(),
                    "superseded_archival_candidates": len(
                        store.find_superseded_for_archival(days_threshold=thresholds["superseded_archive_days"])
                    ),
                },
                "missing_link_similarity_histogram": store.missing_link_similarity_histogram(
                    thresholds=(0.70, 0.75, 0.80, 0.85, 0.90),
                    recency_days=thresholds["link_weaving_recency_days"],
                ),
                **store.get_dreamer_metrics_summary(),
            }
            write_stats = store.get_write_memory_metrics_summary()
            components["sqlite"] = "ok" if integrity.ok else "corrupt"
            components["wal_mode"] = "ok" if integrity.wal_mode_enabled else "disabled"
            components["vector_index"] = store.vector_backend
            if stats["disputed_nodes"] > 5:
                warnings.append("Disputed node count exceeds the warning threshold (> 5).")
    except sqlite3.DatabaseError as exc:
        components["sqlite"] = f"error: {exc}"
        components["wal_mode"] = "error"
        warnings.append("SQLite database could not be opened; run `synapse rebuild-index`.")

    status = "healthy"
    if not components["sqlite"].startswith("ok"):
        status = "unhealthy"
    elif probe.status != "ok" or warnings:
        status = "degraded"

    return SystemHealthReport(
        status=status,
        components=components,
        stats=stats,
        warnings=warnings,
        lifecycle_stats=lifecycle_stats,
        write_stats=write_stats,
        database_path=db_path,
        embedding_fingerprint=embedding_fingerprint,
        delta_sync_hook=(
            f"ok ({sync_result.changed} change(s) applied via {sync_result.backend})"
            if sync_result is not None
            else sync_runtime.delta_sync_hook
        ),
        startup_sync_hook=(
            f"ok ({sync_result.changed} change(s) applied via {sync_result.backend})"
            if sync_result is not None
            else sync_runtime.startup_sync_hook
        ),
    )


def _effective_dreamer_thresholds(config: SynapseConfig) -> dict[str, int | float]:
    thresholds = config.dreamer.thresholds
    stale_orphan_days = thresholds.stale_orphan_days or config.decay.janitor_days
    link_weaving_recency_days = thresholds.link_weaving_recency_days or config.decay.janitor_days
    return {
        "missing_link_cosine": thresholds.missing_link_cosine,
        "stale_orphan_days": stale_orphan_days,
        "link_weaving_recency_days": link_weaving_recency_days,
        "superseded_archive_days": thresholds.superseded_archive_days,
        "low_structure_chars": thresholds.low_structure_chars,
        "max_missing_link_pairs_per_run": thresholds.max_missing_link_pairs_per_run,
    }


def _empty_similarity_histogram() -> dict[str, int]:
    return {"0.70": 0, "0.75": 0, "0.80": 0, "0.85": 0, "0.90": 0}


def _empty_dreamer_runs_summary() -> dict[str, int | float]:
    return {
        "total": 0,
        "last_24h": 0,
        "last_7d": 0,
        "last_30d": 0,
        "avg_duration_ms": 0.0,
        "avg_triage_decisions": 0.0,
        "avg_links_added": 0.0,
        "avg_condensed": 0.0,
        "avg_archived": 0.0,
    }


def _empty_dreamer_decision_totals() -> dict[str, int]:
    return {
        "triage_keep": 0,
        "triage_condense": 0,
        "triage_archive": 0,
        "links_added": 0,
        "conflicts_superseded": 0,
        "conflicts_both_valid": 0,
        "warnings": 0,
        "sampling_failures": 0,
    }


def _empty_write_stats() -> dict[str, Any]:
    return {
        "requests_total": 0,
        "candidate_count_avg": 0.0,
        "candidate_count_zero_rate": 0.0,
        "decision_totals": {"create": 0, "supersede": 0, "complement": 0},
        "warnings": {},
        "execution_failures": 0,
    }


def run_startup_checks(
    config: SynapseConfig,
    *,
    runtime_paths: RuntimePaths | None = None,
    auto_rebuild: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    logger: logging.Logger | None = None,
) -> StartupReport:
    """Run startup integrity checks and optionally rebuild the derived index."""

    selected_logger = logger or LOGGER
    paths = runtime_paths or bootstrap_runtime_directories(config)
    db_path = database_path_for_config(config)
    embedding_probe = probe_embedding_engine(config)
    desired_fingerprint = compute_embedding_fingerprint(config)
    database_integrity, reasons = _evaluate_database_state(
        db_path=db_path,
        embedding_dimension=config.embedding.dimension or 0,
        desired_fingerprint=desired_fingerprint,
    )
    rebuilt = False
    rebuild_result: IndexRebuildResult | None = None
    sync_result: SyncBatchResult | None = None

    needs_rebuild = bool(reasons)
    if auto_rebuild and needs_rebuild:
        selected_logger.warning("Startup checks requested index rebuild", extra={"reasons": reasons})
        rebuild_result = rebuild_index(
            config,
            runtime_paths=paths,
            progress_callback=progress_callback,
            logger=logger,
        )
        rebuilt = True
        with SQLiteNodeStore(db_path, embedding_dimension=config.embedding.dimension or 0) as store:
            database_integrity = store.check_integrity()

    with SQLiteNodeStore(db_path, embedding_dimension=config.embedding.dimension or 0) as store:
        sync_manager = SyncManager(
            config,
            runtime_paths=paths,
            store=store,
            logger=selected_logger,
        )
        sync_result = sync_manager.startup_sync()

    health = collect_health_status(
        config,
        runtime_paths=paths,
        embedding_report=embedding_probe,
        sync_result=sync_result,
    )
    return StartupReport(
        runtime_paths=paths,
        database_path=db_path,
        database_integrity=database_integrity,
        embedding=embedding_probe,
        health=health,
        needs_rebuild=needs_rebuild,
        rebuilt=rebuilt,
        rebuild_reasons=reasons,
        rebuild_result=rebuild_result,
        sync_result=sync_result,
    )


def _probe_settings(settings: EmbeddingSettings) -> EmbeddingSettings:
    return settings.model_copy(update={"timeout_seconds": min(settings.timeout_seconds, 2)})


def _evaluate_database_state(
    *,
    db_path: Path,
    embedding_dimension: int,
    desired_fingerprint: str,
) -> tuple[DatabaseIntegrityReport | None, list[str]]:
    reasons: list[str] = []
    database_integrity: DatabaseIntegrityReport | None = None
    db_exists = db_path.exists()

    try:
        with SQLiteNodeStore(db_path, embedding_dimension=embedding_dimension) as store:
            database_integrity = store.check_integrity()
            stored_fingerprint = store.get_embedding_fingerprint()
            stored_dimension = store.get_meta("embedding_dimension")
            if not db_exists:
                reasons.append("SQLite index missing")
            if not database_integrity.ok:
                reasons.append(f"SQLite integrity check failed: {database_integrity.integrity_check_result}")
            if stored_fingerprint not in {None, desired_fingerprint}:
                reasons.append("Embedding configuration changed; vector index requires rebuild")
            if stored_dimension not in {None, str(embedding_dimension)}:
                reasons.append("Embedding dimension changed; vector index requires rebuild")
    except sqlite3.DatabaseError as exc:
        reasons.append(f"SQLite database unavailable: {exc}")
        if db_path.exists():
            try:
                db_path.unlink()
            except OSError:
                pass

    return database_integrity, reasons
