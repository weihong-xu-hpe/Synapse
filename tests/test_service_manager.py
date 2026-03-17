from __future__ import annotations

import plistlib
from pathlib import Path

from synapse.config import load_config
from synapse.deployment import ServiceCommandResult, ServiceManager
from synapse.utils.runtime import bootstrap_runtime_directories


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


def always_available(_executable: str) -> str:
    return "/usr/bin/fake"


def make_runner(command_log: list[tuple[str, ...]]):
    def runner(command) -> ServiceCommandResult:
        command_tuple = tuple(command)
        command_log.append(command_tuple)
        stdout = ""
        if command_tuple[-2:] == ("is-active", "synapse-memory.service"):
            stdout = "active"
        elif command_tuple[-2:] == ("is-enabled", "synapse-memory.service"):
            stdout = "enabled"
        return ServiceCommandResult(
            command=command_tuple,
            available=True,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    return runner


def test_render_launchd_plist_contains_expected_fields(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    manager = ServiceManager(
        config,
        runtime_paths=runtime_paths,
        platform_name="Darwin",
        home_directory=home_dir,
        python_executable=tmp_path / ".venv/bin/python",
        user_id=501,
    )

    payload = plistlib.loads(manager.render_service_file().encode("utf-8"))
    assert manager.service_file_path == home_dir / "Library" / "LaunchAgents" / "com.synapse.memory.plist"
    assert payload["Label"] == "com.synapse.memory"
    assert payload["ProgramArguments"] == [
        str((tmp_path / ".venv/bin/python").resolve()),
        "-m",
        "synapse",
        "serve",
        "--run-server",
    ]
    assert payload["WorkingDirectory"] == str(tmp_path.resolve())
    assert payload["KeepAlive"] is True
    assert payload["RunAtLoad"] is True
    assert payload["EnvironmentVariables"]["SYNAPSE_CONFIG_PATH"] == str((tmp_path / "config.toml").resolve())
    assert payload["StandardOutPath"] == str(runtime_paths.logs / "service.log")
    assert payload["StandardErrorPath"] == str(runtime_paths.logs / "service-error.log")


def test_render_systemd_service_contains_expected_fields(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    manager = ServiceManager(
        config,
        runtime_paths=runtime_paths,
        platform_name="Linux",
        home_directory=home_dir,
        python_executable="/opt/synapse/.venv/bin/python",
    )

    unit = manager.render_service_file()
    assert manager.service_file_path == home_dir / ".config" / "systemd" / "user" / "synapse-memory.service"
    assert "[Unit]" in unit
    assert "Description=Synapse Memory Agent" in unit
    assert "ExecStart=/opt/synapse/.venv/bin/python -m synapse serve --run-server" in unit
    assert f'WorkingDirectory="{tmp_path.resolve()}"' in unit
    assert f'Environment="SYNAPSE_CONFIG_PATH={(tmp_path / "config.toml").resolve()}"' in unit
    assert "Restart=on-failure" in unit
    assert "RestartSec=5" in unit
    assert f"StandardOutput=append:{runtime_paths.logs / 'service.log'}" in unit
    assert f"StandardError=append:{runtime_paths.logs / 'service-error.log'}" in unit


def test_install_and_uninstall_are_idempotent_for_linux_user_service(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    command_log: list[tuple[str, ...]] = []
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    manager = ServiceManager(
        config,
        runtime_paths=runtime_paths,
        platform_name="Linux",
        home_directory=home_dir,
        python_executable="/usr/bin/python3",
        which=always_available,
        command_runner=make_runner(command_log),
    )

    first_install = manager.install()
    second_install = manager.install()
    status = manager.status()
    first_uninstall = manager.uninstall()
    second_uninstall = manager.uninstall()

    assert first_install.installed is True
    assert first_install.changed is True
    assert second_install.installed is True
    assert second_install.changed is False
    assert manager.service_file_path is not None and not manager.service_file_path.exists()

    assert status.installed is True
    assert status.runtime_state == "active"
    assert status.enabled_state == "enabled"

    assert first_uninstall.installed is False
    assert first_uninstall.changed is True
    assert second_uninstall.installed is False
    assert second_uninstall.changed is False

    assert ("systemctl", "--user", "daemon-reload") in command_log
    assert ("systemctl", "--user", "enable", "--now", "synapse-memory.service") in command_log
    assert ("systemctl", "--user", "disable", "--now", "synapse-memory.service") in command_log
