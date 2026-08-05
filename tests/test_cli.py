from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import synapse.cli as cli_module
import synapse.server as server_module
from synapse.cli import app
from synapse.deployment import ServiceActionResult, ServiceLogView, ServiceStatus
from synapse.models import Node, NodeMetadata, NodeType, SensitivityLevel
from synapse.storage import write_node_file


runner = CliRunner()
FIXED_TIME = datetime(2026, 3, 7, 10, 0, tzinfo=UTC)


class FakeServiceManager:
    def __init__(self, config, *, runtime_paths=None) -> None:
        self._service_file = config.project_root / ".synapse" / "fake-service-manifest"
        self._stdout_log = (runtime_paths.logs if runtime_paths is not None else config.project_root / ".synapse/.logs") / "service.log"
        self._stderr_log = (runtime_paths.logs if runtime_paths is not None else config.project_root / ".synapse/.logs") / "service-error.log"
        self._config_path = config.config_path
        self._working_directory = config.project_root

    def install(self) -> ServiceActionResult:
        return ServiceActionResult(
            action="install",
            platform="linux",
            supported=True,
            service_file_path=self._service_file,
            installed=True,
            changed=True,
            message="Synapse service installed.",
        )

    def uninstall(self) -> ServiceActionResult:
        return ServiceActionResult(
            action="uninstall",
            platform="linux",
            supported=True,
            service_file_path=self._service_file,
            installed=False,
            changed=True,
            message="Synapse service uninstalled.",
        )

    def restart(self) -> ServiceActionResult:
        return ServiceActionResult(
            action="restart",
            platform="linux",
            supported=True,
            service_file_path=self._service_file,
            installed=True,
            changed=True,
            message="Synapse service restart requested.",
        )

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            platform="linux",
            supported=True,
            service_name="synapse-memory.service",
            service_file_path=self._service_file,
            installed=True,
            runtime_state="active",
            enabled_state="enabled",
            control_available=True,
            python_path=Path("/usr/bin/python3"),
            working_directory=self._working_directory,
            config_path=self._config_path,
            stdout_log_path=self._stdout_log,
            stderr_log_path=self._stderr_log,
        )

    def read_logs(self, *, lines: int = 20) -> ServiceLogView:
        return ServiceLogView(
            stdout_log_path=self._stdout_log,
            stderr_log_path=self._stderr_log,
            stdout_exists=True,
            stderr_exists=True,
            stdout_excerpt=f"stdout line count={lines}",
            stderr_excerpt="stderr line",
        )


def write_config(base_dir: Path) -> Path:
    config_path = base_dir / "config.toml"
    config_path.write_text(
        """
[server]
host = "127.0.0.1"
port = 8765

[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "builtin"
model = "bge-m3"
dimension = 1024
timeout_seconds = 1

[reranker]
provider = "builtin"
model = "bge-reranker-v2-m3"
max_candidates = 9
timeout_seconds = 1

[logging]
log_dir = "./.synapse/.logs"
""".strip(),
        encoding="utf-8",
    )
    return config_path


def seed_markdown_nodes(base_dir: Path) -> None:
    gateway = Node(
        metadata=NodeMetadata(
            id="mem_20260307_gateway_design",
            title="Gateway Design",
            created_at=FIXED_TIME,
            last_accessed=FIXED_TIME,
            type=NodeType.PERSISTENT,
            tags=["gateway"],
            sensitivity=SensitivityLevel.INTERNAL,
        ),
        content="# Gateway Design\n\nSee [[Rate Limiting Strategy]].",
        file_path=Path("active/mem_20260307_gateway_design.md"),
    )
    rate_limit = Node(
        metadata=NodeMetadata(
            id="mem_20260307_rate_limiting_strategy",
            title="Rate Limiting Strategy",
            created_at=FIXED_TIME,
            last_accessed=FIXED_TIME,
            type=NodeType.PERSISTENT,
            tags=["rate-limit"],
            sensitivity=SensitivityLevel.INTERNAL,
        ),
        content="# Rate Limiting Strategy\n\nToken bucket limits.",
        file_path=Path("active/mem_20260307_rate_limiting_strategy.md"),
    )
    write_node_file(gateway, base_path=base_dir / ".synapse")
    write_node_file(rate_limit, base_path=base_dir / ".synapse")


def test_cli_phase3_commands_create_runtime_logs_and_report_health(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "ServiceManager", FakeServiceManager)
    config_path = write_config(tmp_path)
    seed_markdown_nodes(tmp_path)

    commands = [
        (["--config", str(config_path), "version"], "Synapse 0.1.0"),
        (["--config", str(config_path), "rebuild-index"], "Rebuild-index completed successfully."),
        (["--config", str(config_path), "status"], "SQLite: ok"),
        (["--config", str(config_path), "serve"], "Synapse serve startup:"),
        (["--config", str(config_path), "install", "--service"], "Synapse service installed."),
    ]

    for args, expected in commands:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert expected in result.output

    rebuild_output = runner.invoke(app, ["--config", str(config_path), "rebuild-index"]).output
    assert "Indexed nodes: 2" in rebuild_output
    assert "Vector backend: python-fallback" in rebuild_output

    status_output = runner.invoke(app, ["--config", str(config_path), "status"]).output
    assert "Synapse status: healthy" in status_output
    assert "Indexed nodes: 2" in status_output
    assert "Delta sync hook: enabled" in status_output
    assert "Daemon runtime: active" in status_output

    runtime_base = tmp_path / ".synapse"
    assert (runtime_base / "active").exists()
    assert (runtime_base / ".archive").exists()
    assert (runtime_base / ".audit").exists()
    assert (runtime_base / ".logs" / "mcp-daemon.log").exists()
    assert (runtime_base / ".logs" / "file-watcher.log").exists()
    assert (runtime_base / ".logs" / "janitor.log").exists()
    assert (runtime_base / ".logs" / "audit.log").exists()


def test_cli_service_commands_surface_install_restart_logs_and_uninstall(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "ServiceManager", FakeServiceManager)
    config_path = write_config(tmp_path)

    install_result = runner.invoke(app, ["--config", str(config_path), "install", "--service"])
    assert install_result.exit_code == 0, install_result.output
    assert "Synapse service installed." in install_result.output
    assert "Service manifest path:" in install_result.output

    restart_result = runner.invoke(app, ["--config", str(config_path), "restart"])
    assert restart_result.exit_code == 0, restart_result.output
    assert "Synapse service restart requested." in restart_result.output

    logs_result = runner.invoke(app, ["--config", str(config_path), "logs", "--lines", "5"])
    assert logs_result.exit_code == 0, logs_result.output
    assert "Service stdout:" in logs_result.output
    assert "stdout line count=5" in logs_result.output
    assert "Service stderr:" in logs_result.output

    uninstall_result = runner.invoke(app, ["--config", str(config_path), "uninstall", "--service"])
    assert uninstall_result.exit_code == 0, uninstall_result.output
    assert "Synapse service uninstalled." in uninstall_result.output


def test_serve_command_no_longer_accepts_transport_option(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    result = runner.invoke(app, ["--config", str(config_path), "serve", "--transport", "stdio"])

    assert result.exit_code != 0
    assert "No such option: --transport" in result.output


def test_serve_command_run_server_uses_streamable_runtime_entrypoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "ServiceManager", FakeServiceManager)
    config_path = write_config(tmp_path)
    seed_markdown_nodes(tmp_path)

    captured: dict[str, object] = {}

    def fake_run_streamable_server(config, *, runtime_paths=None, logger=None, **kwargs) -> None:
        captured["config"] = config
        captured["runtime_paths"] = runtime_paths
        captured["logger_name"] = None if logger is None else logger.name
        captured["kwargs"] = kwargs

    monkeypatch.setattr(server_module, "run_streamable_server", fake_run_streamable_server)

    result = runner.invoke(app, ["--config", str(config_path), "serve", "--run-server"])

    assert result.exit_code == 0, result.output
    assert "Starting Synapse server on http://127.0.0.1:8765" in result.output
    assert captured["runtime_paths"] is not None
    assert captured["logger_name"] == cli_module.DAEMON_LOGGER_NAME
    assert captured["kwargs"] == {}


def test_dreamer_run_command_uses_configured_batch_size(tmp_path: Path, monkeypatch) -> None:
    config_path = write_config(tmp_path)
    captured: dict[str, object] = {}

    class FakeDecider:
        def __init__(self, settings) -> None:
            captured["decider_settings"] = settings

    class FakeDreamer:
        def __init__(self, config, *, runtime_paths, sampling_client, logger) -> None:
            captured["config"] = config
            captured["runtime_paths"] = runtime_paths
            captured["sampling_client"] = sampling_client
            captured["logger"] = logger

        def run(self, *, batch_size: int):
            captured["batch_size"] = batch_size
            return SimpleNamespace(
                scanned={"stale": 1},
                triage=(object(),),
                links_added=(),
                conflicts_resolved=(),
                archived=(),
                condensed=(),
                warnings=(),
            )

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli_module, "LocalLLMDecider", FakeDecider)
    monkeypatch.setattr(cli_module, "Dreamer", FakeDreamer)

    result = runner.invoke(app, ["--config", str(config_path), "dreamer", "run"])

    assert result.exit_code == 0, result.output
    assert captured["batch_size"] == 8
    assert captured["closed"] is True
    assert "Dreamer run completed." in result.output
    assert "Warnings: 0" in result.output
