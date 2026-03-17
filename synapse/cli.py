"""Typer-based CLI for Synapse."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from synapse import __version__
from synapse.config import SynapseConfig, load_config
from synapse.deployment import ServiceActionResult, ServiceLogView, ServiceManager, ServiceStatus
from synapse.indexing import collect_health_status, rebuild_index as rebuild_sqlite_index, run_startup_checks
from synapse.utils import RuntimePaths, bootstrap_runtime_directories, configure_logging


app = typer.Typer(help="Synapse local hybrid memory CLI.", no_args_is_help=True)
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

    invoked = ctx.invoked_subcommand
    if invoked == "mcp-proxy":
        return

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
    typer.echo(f"Delta sync hook: {health.delta_sync_hook}")
    typer.echo(f"Startup sync hook: {health.startup_sync_hook}")
    _echo_service_status(_build_service_manager(state).status())
    for warning in health.warnings:
        typer.echo(f"Warning: {warning}")


@app.command("mcp-proxy")
def mcp_proxy(
    server: Annotated[
        str,
        typer.Argument(
            help="Server address: full URL, host:port, or just host. Examples: 10.0.1.5:8765, my-server, http://host:9000/mcp",
        ),
    ] = "127.0.0.1:8765",
) -> None:
    """Run a stdio-to-HTTP proxy for MCP sampling support in VS Code.

    This command works standalone — it does not require config.toml.
    It only needs the running Synapse HTTP server to be reachable.

    Examples:
        synapse mcp-proxy                      # localhost:8765
        synapse mcp-proxy 10.0.1.5:8765        # remote host
        synapse mcp-proxy my-server            # default port 8765
        synapse mcp-proxy http://h:9000/mcp    # full URL
    """

    if server.startswith("http://") or server.startswith("https://"):
        url = server
    else:
        if ":" not in server:
            server = f"{server}:8765"
        url = f"http://{server}/mcp"

    from synapse.server.stdio_proxy import run_stdio_proxy

    run_stdio_proxy(server_url=url)


@app.command()
def version(ctx: typer.Context) -> None:
    """Print Synapse version information."""

    state = _state_from_context(ctx)
    state.loggers[AUDIT_LOGGER_NAME].info("Version command invoked")
    typer.echo(f"Synapse {__version__}")


def main() -> None:
    """Console script entry point."""

    app(prog_name="synapse")
