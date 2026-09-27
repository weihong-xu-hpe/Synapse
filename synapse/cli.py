"""Typer-based CLI for Synapse."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from synapse import __version__
from synapse.config import SynapseConfig, load_config
from synapse.deployment import ServiceActionResult, ServiceLogView, ServiceManager, ServiceStatus
from synapse.indexing import collect_health_status, rebuild_index as rebuild_sqlite_index, run_startup_checks
from synapse.lifecycle import Dreamer
from synapse.server.decider import LocalLLMDecider
from synapse.storage import SQLiteNodeStore
from synapse.utils import RuntimePaths, bootstrap_runtime_directories, configure_logging
from typing import Any


app = typer.Typer(help="Synapse local hybrid memory CLI.", no_args_is_help=True)
dreamer_app = typer.Typer(help="Run Dreamer memory lifecycle operations.", no_args_is_help=True)
distiller_app = typer.Typer(help="Run the session distiller (transcripts → OKF knowledge).", no_args_is_help=True)
app.add_typer(dreamer_app, name="dreamer")
app.add_typer(distiller_app, name="distill")
AUDIT_LOGGER_NAME = "synapse.audit"
DAEMON_LOGGER_NAME = "synapse.mcp-daemon"


@dataclass(slots=True)
class AppState:
    config_path: Path
    runtime_paths: RuntimePaths
    config: SynapseConfig
    loggers: dict[str, logging.Logger]


def _state_from_context(ctx: typer.Context) -> AppState:
    state = ctx.obj
    if not isinstance(state, AppState):
        raise typer.Exit(code=1)
    return state


def _build_service_manager(state: AppState) -> ServiceManager:
    return ServiceManager(state.config, runtime_paths=state.runtime_paths)


def _echo_service_action_result(result: ServiceActionResult) -> None:
    typer.echo(result.message)
    if result.service_file_path is not None:
        typer.echo(f"Service manifest path: {result.service_file_path}")
    typer.echo(f"Service platform: {result.platform}")
    typer.echo(f"Service installed: {'yes' if result.installed else 'no'}")
    for warning in result.warnings:
        typer.echo(f"Note: {warning}")


def _echo_service_status(service_status: ServiceStatus) -> None:
    typer.echo(f"Daemon platform: {service_status.platform}")
    typer.echo(f"Daemon service: {service_status.service_name}")
    if service_status.service_file_path is not None:
        typer.echo(f"Daemon manifest path: {service_status.service_file_path}")
    typer.echo(f"Daemon installed: {'yes' if service_status.installed else 'no'}")
    typer.echo(f"Daemon runtime: {service_status.runtime_state}")
    typer.echo(f"Daemon enabled: {service_status.enabled_state}")
    typer.echo(f"Daemon stdout log: {service_status.stdout_log_path}")
    typer.echo(f"Daemon stderr log: {service_status.stderr_log_path}")
    for warning in service_status.warnings:
        typer.echo(f"Daemon note: {warning}")


def _echo_log_block(title: str, path: Path, content: str, *, exists: bool) -> None:
    typer.echo(f"{title}: {path}")
    if not exists:
        typer.echo("(missing)")
        return
    typer.echo(content if content else "(empty)")


def _echo_service_logs(log_view: ServiceLogView) -> None:
    _echo_log_block("Service stdout", log_view.stdout_log_path, log_view.stdout_excerpt, exists=log_view.stdout_exists)
    _echo_log_block("Service stderr", log_view.stderr_log_path, log_view.stderr_excerpt, exists=log_view.stderr_exists)
    for warning in log_view.warnings:
        typer.echo(f"Note: {warning}")


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to a Synapse TOML config file. Defaults to SYNAPSE_CONFIG_PATH or ./config.toml.",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Load config, initialize runtime paths, and set up logging before commands run."""

    try:
        loaded_config = load_config(config_path=config)
    except (FileNotFoundError, ValidationError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    runtime_paths = bootstrap_runtime_directories(loaded_config)
    loggers = configure_logging(loaded_config, runtime_paths)
    ctx.obj = AppState(
        config_path=loaded_config.config_path,
        runtime_paths=runtime_paths,
        config=loaded_config,
        loggers=loggers,
    )


@app.command()
def serve(
    ctx: typer.Context,
    run_server: Annotated[
        bool,
        typer.Option(
            "--run-server",
            help="Start the Synapse server after startup checks. Defaults to startup checks only.",
        ),
    ] = False,
) -> None:
    """Run startup checks and optionally launch the Synapse server."""

    state = _state_from_context(ctx)
    daemon_logger = state.loggers[DAEMON_LOGGER_NAME]
    daemon_logger.info("Serve command invoked")
    state.loggers["synapse.file-watcher"].info("File watcher startup sync running")

    report = run_startup_checks(
        state.config,
        runtime_paths=state.runtime_paths,
        auto_rebuild=True,
        progress_callback=typer.echo,
        logger=daemon_logger,
    )

    typer.echo(f"Synapse serve startup: {report.health.status}")
    typer.echo(f"Server binding: {state.config.server.host}:{state.config.server.port}")
    typer.echo(f"SQLite: {report.health.components['sqlite']}")
    typer.echo(f"Embedding: {report.embedding.status} ({report.embedding.backend})")
    typer.echo(f"File watcher: {report.health.components['file_watcher']}")
    typer.echo(f"Startup sync: {report.health.startup_sync_hook}")
    if report.rebuilt:
        typer.echo("Startup checks rebuilt the derived SQLite index.")
    elif report.needs_rebuild:
        typer.echo("Startup checks detected a rebuild requirement, but no rebuild was performed.")

    if not run_server:
        return

    try:
        from synapse.server import run_streamable_server
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
        raise typer.Exit(code=1) from exc

    typer.echo(f"Starting Synapse server on http://{state.config.server.host}:{state.config.server.port}")
    daemon_logger.info("Starting server runtime")
    run_streamable_server(
        state.config,
        runtime_paths=state.runtime_paths,
        logger=daemon_logger,
    )


@dreamer_app.command("run")
def run_dreamer(
    ctx: typer.Context,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", min=1, max=20, help="Number of nodes or pairs sent to the decider per batch."),
    ] = None,
) -> None:
    """Run one Dreamer consolidation pass using the configured local decider."""

    state = _state_from_context(ctx)
    effective_batch_size = batch_size or state.config.dreamer.batch_size
    logger = state.loggers[DAEMON_LOGGER_NAME]
    logger.info("Dreamer command invoked", extra={"batch_size": effective_batch_size})

    dreamer = Dreamer(
        state.config,
        runtime_paths=state.runtime_paths,
        sampling_client=LocalLLMDecider(state.config.decider),
        logger=logger,
    )
    try:
        report = dreamer.run(batch_size=effective_batch_size)
    finally:
        dreamer.close()

    typer.echo("Dreamer run completed.")
    typer.echo(f"Scanned: {report.scanned}")
    typer.echo(f"Triage decisions: {len(report.triage)}")
    typer.echo(f"Links added: {len(report.links_added)}")
    typer.echo(f"Conflicts resolved: {len(report.conflicts_resolved)}")
    typer.echo(f"Archived: {len(report.archived)}")
    typer.echo(f"Condensed: {len(report.condensed)}")
    typer.echo(f"Warnings: {len(report.warnings)}")


@distiller_app.command("run")
def run_distiller(
    ctx: typer.Context,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan and report without writing anything.")] = False,
    apply: Annotated[bool, typer.Option("--apply", help="Execute writes (default for an explicit run).")] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Max transcripts to process this run.")] = None,
    node_id: Annotated[str | None, typer.Option("--id", help="Distill one specific transcript node id.")] = None,
    legacy_groups: Annotated[
        Path | None,
        typer.Option("--legacy-groups", help="JSON file of legacy session groups (backfill mode)."),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option("--report", help="Write the run report JSON to this path."),
    ] = None,
) -> None:
    """Run one distillation sweep (or a legacy-group backfill).

    Backfill mode (--legacy-groups) defaults to supersede-downgrade ON so
    curated knowledge is never superseded without owner approval. The live
    sweep (scheduler) runs with downgrade OFF.
    """

    state = _state_from_context(ctx)
    logger = state.loggers[DAEMON_LOGGER_NAME]
    execute = apply or not dry_run
    if dry_run and apply:
        raise typer.BadParameter("Use either --dry-run or --apply, not both.")

    from synapse.lifecycle.distiller import Distiller

    distiller = Distiller(
        state.config,
        runtime_paths=state.runtime_paths,
        sampling_client=LocalLLMDecider(state.config.decider),
        logger=logger,
    )
    try:
        if legacy_groups is not None:
            report = _run_legacy_backfill(
                distiller,
                groups_path=legacy_groups,
                dry_run=dry_run or not execute,
                limit=limit,
                node_id=node_id,
            )
        elif node_id is not None:
            report = _run_single(distiller, node_id=node_id, dry_run=dry_run or not execute)
        else:
            report = distiller.run(
                limit=limit,
                dry_run=dry_run or not execute,
                downgrade_supersede=legacy_groups is not None or execute,
            )
    finally:
        distiller.close()

    summary = report.summary()
    if report_path is not None:
        report_path.write_text(json.dumps({"summary": summary, "details": report.details}, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Report written: {report_path}")
    typer.echo("Distiller run completed.")
    typer.echo(f"Transcripts scanned/distilled/skipped: {summary['transcripts_scanned']}/{summary['transcripts_distilled']}/{summary['transcripts_skipped']}")
    typer.echo(f"Items extracted/invalid: {summary['items_extracted']}/{summary['items_invalid']}")
    typer.echo(f"Decider actions: {summary['decider']}")
    if summary["downgraded"]:
        typer.echo("Supersede downgrades (backfill safety):")
        for entry in summary["downgraded"]:
            typer.echo(f"  - {entry['item_title']} (targets: {entry.get('target_node_ids')})")
    typer.echo(f"Archived transcripts: {summary['archived_transcripts']}")
    typer.echo(f"LLM failures: {summary['llm_failures']}")


def _run_legacy_backfill(
    distiller: Any,
    *,
    groups_path: Path,
    dry_run: bool,
    limit: int | None,
    node_id: str | None,
) -> Any:
    """Backfill: distill the representative of each legacy session group."""

    from synapse.lifecycle.distiller import content_sha256, is_fully_distilled

    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    if node_id is not None:
        groups = [g for g in groups if g.get("representative_id") == node_id]
    if limit is not None:
        groups = groups[:limit]

    report = None
    for group in groups:
        representative_id = str(group.get("representative_id") or "")
        member_ids = [str(m) for m in group.get("member_ids", [])]
        transcript = distiller._load_transcript_from_disk(representative_id)
        if transcript is None:
            with distiller._get_store() as store:
                transcript = store.get_node(representative_id)
        if transcript is None:
            continue
        if is_fully_distilled(transcript):
            continue
        if dry_run:
            single = distiller.run(limit=0, dry_run=True)
            single.transcripts_scanned = 0
        else:
            single = distiller._run_one_transcript(transcript, downgrade_supersede=True)
        # Stamp non-representative members as distilled-by-representative so
        # they are never picked up by the live sweep (archive them only after
        # the group representative distilled successfully).
        if not dry_run and single.transcripts_distilled > 0:
            for member_id in member_ids:
                if member_id == representative_id:
                    continue
                member = distiller._load_transcript_from_disk(member_id)
                if member is None:
                    continue
                distiller._stamp_transcript(member, [])
        if report is None:
            report = single
        else:
            report.transcripts_scanned += single.transcripts_scanned
            report.transcripts_distilled += single.transcripts_distilled
            report.transcripts_skipped += single.transcripts_skipped
            report.items_extracted += single.items_extracted
            report.items_invalid += single.items_invalid
            report.decider_creates += single.decider_creates
            report.decider_complements += single.decider_complements
            report.decider_supersedes += single.decider_supersedes
            report.decider_downgrades += single.decider_downgrades
            report.llm_failures += single.llm_failures
            report.downgraded.extend(single.downgraded)
            report.details.extend(single.details)
    return report or _empty_report()


def _run_single(distiller: Any, *, node_id: str, dry_run: bool) -> Any:
    from synapse.lifecycle.distiller import is_fully_distilled

    transcript = distiller._load_transcript_from_disk(node_id)
    if transcript is None:
        with distiller._get_store() as store:
            transcript = store.get_node(node_id)
    if transcript is None:
        raise typer.Exit(code=1)
    if is_fully_distilled(transcript):
        typer.echo(f"Node {node_id} is already fully distilled; nothing to do.")
        return _empty_report()
    return distiller._run_one_transcript(transcript, dry_run=dry_run, downgrade_supersede=True)


def _empty_report() -> Any:
    from synapse.lifecycle.distiller import DistillerReport
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return DistillerReport(started_at=now, completed_at=now, duration_ms=0)


@distiller_app.command("status")
def distiller_status(ctx: typer.Context) -> None:
    """Show distiller configuration, queue depth, and recent run metrics."""

    state = _state_from_context(ctx)
    settings = state.config.distiller
    typer.echo(f"Distiller enabled: {settings.enabled}")
    typer.echo(f"Interval: {settings.interval_minutes} min, idle threshold: {settings.idle_minutes} min")
    typer.echo(f"Retention: {settings.retention_days} days (distilled-only gate)")
    typer.echo(f"Downgrade-supersede: {settings.downgrade_supersede}, enforce-OKF: {settings.enforce_okf}")

    from synapse.lifecycle.distiller import Distiller, is_fully_distilled

    distiller = Distiller(state.config, runtime_paths=state.runtime_paths)
    try:
        queue = distiller.select_transcripts(limit=None)
        typer.echo(f"Undistilled idle transcripts queued: {len(queue)}")
        for node in queue[:10]:
            typer.echo(f"  - {node.id} ({node.title})")
    finally:
        distiller.close()

    with SQLiteNodeStore(
        state.runtime_paths.base / "synapse.db",
        embedding_dimension=state.config.embedding.dimension or 0,
    ) as store:
        rows = store._connection.execute(
            "SELECT started_at, transcripts_distilled, items_extracted, llm_failures, archived_transcripts "
            "FROM distiller_runs ORDER BY started_at DESC LIMIT 5"
        ).fetchall()
    if rows:
        typer.echo("Recent runs:")
        for row in rows:
            typer.echo(f"  {row['started_at']} distilled={row['transcripts_distilled']} items={row['items_extracted']} failures={row['llm_failures']} archived={row['archived_transcripts']}")
    else:
        typer.echo("No distiller runs recorded yet.")


@app.command("rebuild-index")
def rebuild_index(
    ctx: typer.Context,
    brain_dir: Annotated[
        Path | None,
        typer.Option(
            "--brain-dir",
            help="Optional directory containing canonical Markdown nodes. Defaults to the active Synapse directory.",
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Rebuild the derived SQLite index from Markdown source-of-truth files."""

    state = _state_from_context(ctx)
    rebuild_logger = state.loggers["synapse.file-watcher"]
    rebuild_logger.info("Rebuild-index command invoked")

    runtime_paths = state.runtime_paths
    if brain_dir is not None:
        runtime_paths = RuntimePaths(
            base=runtime_paths.base,
            active=brain_dir.resolve(),
            archive=runtime_paths.archive,
            logs=runtime_paths.logs,
            audit=runtime_paths.audit,
        )

    result = rebuild_sqlite_index(
        state.config,
        runtime_paths=runtime_paths,
        progress_callback=typer.echo,
        logger=rebuild_logger,
    )
    typer.echo(f"SQLite DB: {result.database_path}")
    typer.echo(f"Indexed nodes: {result.indexed_nodes}")
    typer.echo(f"Embedding status: {result.embedding_status}")
    typer.echo(f"Vector backend: {result.vector_backend}")
    typer.echo("Rebuild-index completed successfully.")


@app.command()
def install(
    ctx: typer.Context,
    service: Annotated[
        bool,
        typer.Option("--service", help="Install Synapse as a user background service."),
    ] = False,
) -> None:
    """Install Synapse runtime helpers, optionally including a user service."""

    state = _state_from_context(ctx)
    state.loggers[DAEMON_LOGGER_NAME].info("Install command invoked", extra={"service": service})

    if not service:
        typer.echo("Install completed. Use --service to install a user daemon.")
        return

    result = _build_service_manager(state).install()
    _echo_service_action_result(result)


@app.command()
def uninstall(
    ctx: typer.Context,
    service: Annotated[
        bool,
        typer.Option("--service", help="Remove the generated Synapse user service."),
    ] = False,
) -> None:
    """Uninstall Synapse service artifacts."""

    state = _state_from_context(ctx)
    state.loggers[DAEMON_LOGGER_NAME].info("Uninstall command invoked", extra={"service": service})

    if not service:
        typer.echo("Uninstall completed. Use --service to remove the user daemon.")
        return

    result = _build_service_manager(state).uninstall()
    _echo_service_action_result(result)


@app.command()
def restart(ctx: typer.Context) -> None:
    """Restart the configured Synapse user daemon."""

    state = _state_from_context(ctx)
    state.loggers[DAEMON_LOGGER_NAME].info("Restart command invoked")
    result = _build_service_manager(state).restart()
    _echo_service_action_result(result)


@app.command()
def logs(
    ctx: typer.Context,
    lines: Annotated[
        int,
        typer.Option("--lines", min=1, help="Number of trailing lines to print from each service log."),
    ] = 20,
) -> None:
    """Print the current Synapse service stdout/stderr log excerpts."""

    state = _state_from_context(ctx)
    state.loggers[AUDIT_LOGGER_NAME].info("Logs command invoked", extra={"lines": lines})
    log_view = _build_service_manager(state).read_logs(lines=lines)
    _echo_service_logs(log_view)


@app.command()
def status(ctx: typer.Context) -> None:
    """Report current storage, index, and runtime health."""

    state = _state_from_context(ctx)
    state.loggers[AUDIT_LOGGER_NAME].info("Status command invoked")

    health = collect_health_status(state.config, runtime_paths=state.runtime_paths)
    typer.echo(f"Synapse status: {health.status}")
    typer.echo(f"Config path: {state.config_path}")
    typer.echo(f"Server binding: {state.config.server.host}:{state.config.server.port}")
    typer.echo(f"Active directory: {state.runtime_paths.active}")
    typer.echo(f"Archive directory: {state.runtime_paths.archive}")
    typer.echo(f"Log directory: {state.runtime_paths.logs}")
    typer.echo(f"Audit directory: {state.runtime_paths.audit}")
    if health.database_path is not None:
        typer.echo(f"SQLite DB: {health.database_path}")
    typer.echo(f"SQLite: {health.components['sqlite']}")
    typer.echo(f"WAL mode: {health.components['wal_mode']}")
    typer.echo(f"Embedding model: {health.components['embedding_model']}")
    typer.echo(f"Vector index: {health.components['vector_index']}")
    typer.echo(f"Indexed nodes: {health.stats['total_nodes']}")
    typer.echo(f"Active nodes: {health.stats['active_nodes']}")
    typer.echo(f"Superseded nodes: {health.stats['superseded_nodes']}")
    typer.echo(f"Disputed nodes: {health.stats['disputed_nodes']}")
    typer.echo(f"Archived nodes: {health.stats['archived_nodes']}")
    _echo_lifecycle_stats(health.lifecycle_stats)
    _echo_write_stats(health.write_stats)
    typer.echo(f"Delta sync hook: {health.delta_sync_hook}")
    typer.echo(f"Startup sync hook: {health.startup_sync_hook}")
    _echo_service_status(_build_service_manager(state).status())
    for warning in health.warnings:
        typer.echo(f"Warning: {warning}")


@app.command("dedupe-session-summaries")
def dedupe_session_summaries(
    ctx: typer.Context,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Execute the archival. Default is a dry-run report."),
    ] = False,
) -> None:
    """Archive duplicate/stale session-summary nodes using the Dreamer archive primitive.

    Rules over ACTIVE nodes titled 'Session summary — %':
      A. exact duplicate content (sha256) -> keep newest created_at, archive the rest;
      B. whitespace-normalized content is a strict prefix of a newer kept same-title node -> archive;
      C. superseded nodes whose supersession chain terminal is ACTIVE (or superseded_by missing) -> archive.

    Non-summary ACTIVE nodes are never touched. --apply writes a JSON manifest
    (archived ids + original paths) into the archive dir for reversibility.
    """

    import json

    from synapse.lifecycle.dreamer import Dreamer
    from synapse.models import Node, NodeStatus
    from synapse.storage import SQLiteNodeStore, archive_node_path

    state = _state_from_context(ctx)
    state.loggers[AUDIT_LOGGER_NAME].info(
        "Dedupe session summaries invoked", extra={"apply": apply}
    )
    paths = state.runtime_paths
    config = state.config

    dreamer = Dreamer(
        config,
        runtime_paths=paths,
        sampling_client=LocalLLMDecider(config.decider),
        logger=state.loggers[DAEMON_LOGGER_NAME],
    )
    try:
        store = dreamer._get_store()
        active = store.list_nodes({"status": NodeStatus.ACTIVE})
        superseded = store.find_by_status(NodeStatus.SUPERSEDED)
        summaries = [n for n in active if n.title.startswith("Session summary — ")]
        curated = [n for n in active if not n.title.startswith("Session summary — ")]

        plan: dict[str, Node] = {}  # node_id -> Node to archive

        # Rule A: exact duplicate content per title; keep newest created_at.
        by_title: dict[str, list[Node]] = {}
        for node in summaries:
            by_title.setdefault(node.title, []).append(node)
        for title, group in by_title.items():
            by_hash: dict[str, list[Node]] = {}
            for node in group:
                digest = hashlib.sha256(node.content.encode("utf-8")).hexdigest()
                by_hash.setdefault(digest, []).append(node)
            for digest, dupes in by_hash.items():
                if len(dupes) < 2:
                    continue
                dupes.sort(key=lambda n: n.metadata.created_at, reverse=True)
                for older in dupes[1:]:
                    plan[older.id] = older

        # Rule B: normalized content strictly prefixes a newer kept same-title node.
        def _normalized(text: str) -> str:
            return " ".join(text.split())

        for title, group in by_title.items():
            kept = [n for n in group if n.id not in plan]
            kept.sort(key=lambda n: n.metadata.created_at)
            for node in kept:
                node_norm = _normalized(node.content)
                for other in kept:
                    if other.id == node.id or other.id in plan:
                        continue
                    if other.metadata.created_at <= node.metadata.created_at:
                        continue
                    other_norm = _normalized(other.content)
                    if node_norm and node_norm != other_norm and other_norm.startswith(node_norm):
                        plan[node.id] = node
                        break

        # Rule C: superseded nodes archivable under the fixed dreamer rule.
        rule_c: list[str] = []
        for node in superseded:
            if node.id in plan:
                continue
            if not node.metadata.superseded_by:
                plan[node.id] = node
                rule_c.append(node.id)
                continue
            terminal, outcome = dreamer._resolve_supersession_chain(node, store)
            if outcome in {"missing_successor", "terminal"}:
                # missing_successor: terminal archived/deleted — obsolete.
                # terminal with no superseded_by: end of chain.
                # Archive unless the terminal node itself is DISPUTED.
                if terminal is not None and terminal.metadata.status is NodeStatus.DISPUTED:
                    continue
                plan[node.id] = node
                rule_c.append(node.id)
            elif outcome == "cycle":
                continue  # cyclic chain: keep, dreamer logs it

        # Safety assertion: only summaries (rule A/B) or superseded (rule C) planned.
        planned_ids = set(plan)
        curated_touched = curated and planned_ids.intersection(n.id for n in curated)
        if curated_touched:
            typer.echo(f"ABORT: non-summary active nodes would be touched: {sorted(curated_touched)[:5]}")
            raise typer.Exit(code=1)

        by_rule = {"A_exact_duplicates": 0, "B_prefix_duplicates": 0, "C_superseded_chain_active": 0}
        for node in plan.values():
            if node.metadata.status is NodeStatus.SUPERSEDED:
                by_rule["C_superseded_chain_active"] += 1
            elif node.id in rule_c:
                by_rule["C_superseded_chain_active"] += 1
            else:
                # Distinguish A vs B by re-checking prefix rule
                is_prefix = False
                for other in by_title.get(node.title, []):
                    if other.id == node.id or other.id in plan and other is not node:
                        continue
                    if other.metadata.created_at > node.metadata.created_at and _normalized(other.content).startswith(_normalized(node.content)) and _normalized(node.content) != _normalized(other.content):
                        is_prefix = True
                        break
                by_rule["B_prefix_duplicates" if is_prefix else "A_exact_duplicates"] += 1

        typer.echo(f"Active summaries: {len(summaries)} | Curated active (untouched): {len(curated)} | Superseded total: {len(superseded)}")
        typer.echo(f"Planned archives: {len(plan)} (A: {by_rule['A_exact_duplicates']}, B: {by_rule['B_prefix_duplicates']}, C: {by_rule['C_superseded_chain_active']})")
        for node in sorted(plan.values(), key=lambda n: n.id)[:10]:
            typer.echo(f"  sample: {node.id} ({node.metadata.status.value}) {node.title[:50]}")

        if not apply:
            typer.echo("Dry-run only. Re-run with --apply to execute.")
            return

        archived = dreamer._archive_nodes(list(plan.values()), reason="dedupe", warnings=[])
        # Remove archived nodes from the derived index; startup sync/rebuild refreshes the rest.
        for node in archived:
            store.delete_node(node.id)

        manifest = {
            "archived_at": datetime.now(UTC).isoformat(),
            "count": len(archived),
            "nodes": [
                {
                    "id": node.id,
                    "title": node.title,
                    "original_path": (paths.base / node.file_path).as_posix(),
                    "archive_path": archive_node_path(paths.archive, node.id).as_posix(),
                }
                for node in archived
            ],
        }
        manifest_path = paths.archive / "dedupe-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"Archived {len(archived)} node(s). Manifest: {manifest_path}")
        state.loggers[AUDIT_LOGGER_NAME].info("Dedupe session summaries applied", extra={"archived": len(archived)})
    finally:
        dreamer.close()


@app.command()
def version(ctx: typer.Context) -> None:
    """Print Synapse version information."""

    state = _state_from_context(ctx)
    state.loggers[AUDIT_LOGGER_NAME].info("Version command invoked")
    typer.echo(f"Synapse {__version__}")


def _echo_lifecycle_stats(lifecycle_stats: dict[str, object]) -> None:
    current = lifecycle_stats.get("current_candidates") if isinstance(lifecycle_stats, dict) else {}
    runs = lifecycle_stats.get("runs") if isinstance(lifecycle_stats, dict) else {}
    decisions = lifecycle_stats.get("decision_totals") if isinstance(lifecycle_stats, dict) else {}
    thresholds = lifecycle_stats.get("thresholds") if isinstance(lifecycle_stats, dict) else {}
    if not all(isinstance(section, dict) for section in (current, runs, decisions, thresholds)):
        return

    typer.echo("Lifecycle stats:")
    typer.echo(
        "  Current candidates: "
        f"stale={current.get('stale_orphans', 0)}, "
        f"missing_links={current.get('missing_link_pairs', 0)}, "
        f"disputed_pairs={current.get('disputed_pairs', 0)}, "
        f"superseded_archive={current.get('superseded_archival_candidates', 0)}"
    )
    typer.echo(
        "  Dreamer runs: "
        f"total={runs.get('total', 0)}, "
        f"last_7d={runs.get('last_7d', 0)}, "
        f"avg_duration={_format_seconds(runs.get('avg_duration_ms', 0.0))}"
    )
    typer.echo(
        "  Decisions: "
        f"keep={decisions.get('triage_keep', 0)}, "
        f"condense={decisions.get('triage_condense', 0)}, "
        f"archive={decisions.get('triage_archive', 0)}, "
        f"links_added={decisions.get('links_added', 0)}"
    )
    typer.echo(
        "  Thresholds: "
        f"missing_link_cosine={thresholds.get('missing_link_cosine', 0.75)}, "
        f"link_recency_days={thresholds.get('link_weaving_recency_days', 30)}, "
        f"low_structure_chars={thresholds.get('low_structure_chars', 100)}, "
        f"max_missing_link_pairs={thresholds.get('max_missing_link_pairs_per_run', 100)}"
    )


def _echo_write_stats(write_stats: dict[str, object]) -> None:
    decisions = write_stats.get("decision_totals") if isinstance(write_stats, dict) else {}
    if not isinstance(decisions, dict):
        return
    typer.echo("Write stats:")
    typer.echo(
        "  Requests: "
        f"{write_stats.get('requests_total', 0)}, "
        f"candidates avg={write_stats.get('candidate_count_avg', 0.0)}, "
        f"zero-candidate rate={_format_percent(write_stats.get('candidate_count_zero_rate', 0.0))}"
    )
    typer.echo(
        "  Decisions: "
        f"create={decisions.get('create', 0)}, "
        f"supersede={decisions.get('supersede', 0)}, "
        f"complement={decisions.get('complement', 0)}"
    )


def _format_seconds(milliseconds: object) -> str:
    try:
        seconds = float(milliseconds) / 1000.0
    except (TypeError, ValueError):
        seconds = 0.0
    return f"{seconds:.1f}s"


def _format_percent(value: object) -> str:
    try:
        percent = float(value) * 100
    except (TypeError, ValueError):
        percent = 0.0
    return f"{percent:.0f}%"


def main() -> None:
    """Console script entry point."""

    app(prog_name="synapse")
