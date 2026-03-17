"""Cross-platform user-service helpers for Synapse daemon management."""

from __future__ import annotations

import os
import plistlib
import platform as platform_module
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from synapse.config import SynapseConfig
from synapse.utils.runtime import RuntimePaths, get_runtime_paths


LAUNCHD_LABEL = "com.synapse.memory"
SYSTEMD_SERVICE_NAME = "synapse-memory.service"
NOT_INSTALLED_STATE = "not installed"


@dataclass(slots=True, frozen=True)
class ServiceCommandResult:
    """Summary of a best-effort service control command."""

    command: tuple[str, ...]
    available: bool
    returncode: int | None
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.available and self.returncode == 0


@dataclass(slots=True, frozen=True)
class ServiceActionResult:
    """Result for install, uninstall, or restart actions."""

    action: str
    platform: str
    supported: bool
    service_file_path: Path | None
    installed: bool
    changed: bool
    message: str
    warnings: tuple[str, ...] = ()
    command_results: tuple[ServiceCommandResult, ...] = ()


@dataclass(slots=True, frozen=True)
class ServiceStatus:
    """Best-effort service installation and runtime status."""

    platform: str
    supported: bool
    service_name: str
    service_file_path: Path | None
    installed: bool
    runtime_state: str
    enabled_state: str
    control_available: bool
    python_path: Path
    working_directory: Path
    config_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    warnings: tuple[str, ...] = ()
    command_results: tuple[ServiceCommandResult, ...] = ()


@dataclass(slots=True, frozen=True)
class ServiceLogView:
    """Excerpt of the generated service stdout/stderr log files."""

    stdout_log_path: Path
    stderr_log_path: Path
    stdout_exists: bool
    stderr_exists: bool
    stdout_excerpt: str
    stderr_excerpt: str
    warnings: tuple[str, ...] = ()


CommandRunner = Callable[[Sequence[str]], ServiceCommandResult]
CommandLocator = Callable[[str], str | None]


class ServiceManager:
    """Generate and manage Synapse user services for macOS and Linux."""

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths | None = None,
        platform_name: str | None = None,
        python_executable: str | Path | None = None,
        home_directory: str | Path | None = None,
        user_id: int | None = None,
        which: CommandLocator | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths or get_runtime_paths(config)
        self.platform = self.detect_os(platform_name)
        self.python_path = Path(python_executable or sys.executable).expanduser().resolve()
        self.working_directory = config.project_root.resolve()
        self.config_path = config.config_path.resolve()
        self.home_directory = Path(home_directory or Path.home()).expanduser().resolve()
        self.user_id = user_id if user_id is not None else getattr(os, "getuid", lambda: None)()
        self._which = which or shutil.which
        self._command_runner = command_runner or self._default_command_runner

    @staticmethod
    def detect_os(platform_name: str | None = None) -> str:
        """Map the host platform into the supported service backends."""

        normalized = (platform_name or platform_module.system()).strip().lower()
        if normalized == "darwin":
            return "macos"
        if normalized == "linux":
            return "linux"
        return "unsupported"

    @property
    def supported(self) -> bool:
        return self.platform in {"macos", "linux"}

    @property
    def service_name(self) -> str:
        return LAUNCHD_LABEL if self.platform == "macos" else SYSTEMD_SERVICE_NAME

    @property
    def service_file_path(self) -> Path | None:
        if self.platform == "macos":
            return self.home_directory / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
        if self.platform == "linux":
            return self.home_directory / ".config" / "systemd" / "user" / SYSTEMD_SERVICE_NAME
        return None

    @property
    def stdout_log_path(self) -> Path:
        return self.runtime_paths.logs / "service.log"

    @property
    def stderr_log_path(self) -> Path:
        return self.runtime_paths.logs / "service-error.log"

    @property
    def program_arguments(self) -> list[str]:
        return [str(self.python_path), "-m", "synapse", "serve", "--run-server"]

    def render_service_file(self) -> str:
        """Render the service manifest for the current OS."""

        if self.platform == "macos":
            return self.render_launchd_plist()
        if self.platform == "linux":
            return self.render_systemd_service()
        raise RuntimeError("Service generation is not supported on this platform")

    def render_launchd_plist(self) -> str:
        """Render a launchd plist suitable for `~/Library/LaunchAgents`."""

        payload = {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": self.program_arguments,
            "WorkingDirectory": str(self.working_directory),
            "KeepAlive": True,
            "RunAtLoad": True,
            "StandardOutPath": str(self.stdout_log_path),
            "StandardErrorPath": str(self.stderr_log_path),
            "EnvironmentVariables": {
                "SYNAPSE_CONFIG_PATH": str(self.config_path),
            },
        }
        return plistlib.dumps(payload, sort_keys=False).decode("utf-8")

    def render_systemd_service(self) -> str:
        """Render a systemd user service unit."""

        exec_start = self._join_systemd_arguments(self.program_arguments)
        working_directory = self._quote_systemd_value(str(self.working_directory))
        config_environment = self._quote_systemd_value(f"SYNAPSE_CONFIG_PATH={self.config_path}")
        return "\n".join(
            [
                "[Unit]",
                "Description=Synapse Memory Agent",
                "After=network.target",
                "",
                "[Service]",
                "Type=simple",
                f"ExecStart={exec_start}",
                f"WorkingDirectory={working_directory}",
                f"Environment={config_environment}",
                "Restart=on-failure",
                "RestartSec=5",
                f"StandardOutput=append:{self.stdout_log_path}",
                f"StandardError=append:{self.stderr_log_path}",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )

    def install(self) -> ServiceActionResult:
        """Write the service file and best-effort enable/start it."""

        if not self.supported:
            return self._unsupported_result("install")

        service_file = self._require_service_file_path()
        service_file.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_paths.logs.mkdir(parents=True, exist_ok=True)

        previous_content = service_file.read_text(encoding="utf-8") if service_file.exists() else None
        content = self.render_service_file()
        service_file.write_text(content, encoding="utf-8")
        try:
            service_file.chmod(0o644)
        except OSError:
            pass

        command_results = self._install_commands(service_file)
        warnings = self._control_warnings(command_results)
        if self.platform == "linux":
            warnings.append(
                "For headless Linux sessions, run `loginctl enable-linger $USER` if you want the user service to survive logout."
            )

        return ServiceActionResult(
            action="install",
            platform=self.platform,
            supported=True,
            service_file_path=service_file,
            installed=True,
            changed=previous_content != content,
            message="Synapse service installed.",
            warnings=tuple(warnings),
            command_results=tuple(command_results),
        )

    def uninstall(self) -> ServiceActionResult:
        """Stop the service if possible and remove the generated service file."""

        if not self.supported:
            return self._unsupported_result("uninstall")

        service_file = self._require_service_file_path()
        existed = service_file.exists()
        command_results = self._uninstall_commands(service_file)
        if existed:
            service_file.unlink()

        warnings = self._control_warnings(command_results)
        return ServiceActionResult(
            action="uninstall",
            platform=self.platform,
            supported=True,
            service_file_path=service_file,
            installed=False,
            changed=existed,
            message="Synapse service uninstalled." if existed else "Synapse service was not installed.",
            warnings=tuple(warnings),
            command_results=tuple(command_results),
        )

    def restart(self) -> ServiceActionResult:
        """Best-effort restart of the generated service."""

        if not self.supported:
            return self._unsupported_result("restart")

        service_file = self._require_service_file_path()
        installed = service_file.exists()
        if not installed:
            return ServiceActionResult(
                action="restart",
                platform=self.platform,
                supported=True,
                service_file_path=service_file,
                installed=False,
                changed=False,
                message="Synapse service is not installed.",
            )

        command_results = self._restart_commands()
        warnings = self._control_warnings(command_results)
        return ServiceActionResult(
            action="restart",
            platform=self.platform,
            supported=True,
            service_file_path=service_file,
            installed=True,
            changed=any(result.ok for result in command_results),
            message="Synapse service restart requested.",
            warnings=tuple(warnings),
            command_results=tuple(command_results),
        )

    def status(self) -> ServiceStatus:
        """Best-effort daemon installation and runtime status."""

        service_file = self.service_file_path
        installed = bool(service_file and service_file.exists())
        if not self.supported:
            return ServiceStatus(
                platform=self.platform,
                supported=False,
                service_name=self.service_name,
                service_file_path=service_file,
                installed=installed,
                runtime_state="unsupported",
                enabled_state="unsupported",
                control_available=False,
                python_path=self.python_path,
                working_directory=self.working_directory,
                config_path=self.config_path,
                stdout_log_path=self.stdout_log_path,
                stderr_log_path=self.stderr_log_path,
                warnings=("OS daemonization is only supported on macOS and Linux.",),
            )

        if self.platform == "macos":
            command_result = self._run_command(["launchctl", "list", LAUNCHD_LABEL])
            control_available = command_result.available
            if not installed:
                runtime_state = NOT_INSTALLED_STATE
                enabled_state = NOT_INSTALLED_STATE
            elif not command_result.available:
                runtime_state = "unknown"
                enabled_state = "unknown"
            elif command_result.returncode == 0:
                runtime_state = "running"
                enabled_state = "loaded"
            else:
                runtime_state = "stopped"
                enabled_state = "not loaded"
            warnings = tuple(self._control_warnings([command_result]))
            command_results = (command_result,)
        else:
            active_result = self._run_command(["systemctl", "--user", "is-active", SYSTEMD_SERVICE_NAME])
            enabled_result = self._run_command(["systemctl", "--user", "is-enabled", SYSTEMD_SERVICE_NAME])
            control_available = active_result.available or enabled_result.available
            if not installed:
                runtime_state = NOT_INSTALLED_STATE
                enabled_state = NOT_INSTALLED_STATE
            else:
                runtime_state = self._normalize_systemd_state(active_result, fallback="unknown")
                enabled_state = self._normalize_systemd_state(enabled_result, fallback="unknown")
            warnings = tuple(self._control_warnings([active_result, enabled_result]))
            command_results = (active_result, enabled_result)

        return ServiceStatus(
            platform=self.platform,
            supported=self.supported,
            service_name=self.service_name,
            service_file_path=service_file,
            installed=installed,
            runtime_state=runtime_state,
            enabled_state=enabled_state,
            control_available=control_available,
            python_path=self.python_path,
            working_directory=self.working_directory,
            config_path=self.config_path,
            stdout_log_path=self.stdout_log_path,
            stderr_log_path=self.stderr_log_path,
            warnings=warnings,
            command_results=command_results,
        )

    def read_logs(self, *, lines: int = 20) -> ServiceLogView:
        """Return the last N lines of the service stdout/stderr log files."""

        line_count = max(1, int(lines))
        stdout_excerpt = self._tail_file(self.stdout_log_path, line_count)
        stderr_excerpt = self._tail_file(self.stderr_log_path, line_count)
        warnings: list[str] = []
        if not self.stdout_log_path.exists():
            warnings.append(f"Service stdout log does not exist yet: {self.stdout_log_path}")
        if not self.stderr_log_path.exists():
            warnings.append(f"Service stderr log does not exist yet: {self.stderr_log_path}")
        return ServiceLogView(
            stdout_log_path=self.stdout_log_path,
            stderr_log_path=self.stderr_log_path,
            stdout_exists=self.stdout_log_path.exists(),
            stderr_exists=self.stderr_log_path.exists(),
            stdout_excerpt=stdout_excerpt,
            stderr_excerpt=stderr_excerpt,
            warnings=tuple(warnings),
        )

    def _unsupported_result(self, action: str) -> ServiceActionResult:
        return ServiceActionResult(
            action=action,
            platform=self.platform,
            supported=False,
            service_file_path=self.service_file_path,
            installed=False,
            changed=False,
            message="OS daemonization is only supported on macOS and Linux.",
        )

    def _require_service_file_path(self) -> Path:
        service_file = self.service_file_path
        if service_file is None:
            raise RuntimeError("No service file path is available for this platform")
        return service_file

    def _install_commands(self, service_file: Path) -> list[ServiceCommandResult]:
        if self.platform == "macos":
            domain = self._launchd_domain
            return [
                self._run_command(["launchctl", "bootout", domain, str(service_file)]),
                self._run_command(["launchctl", "bootstrap", domain, str(service_file)]),
                self._run_command(["launchctl", "kickstart", "-k", f"{domain}/{LAUNCHD_LABEL}"]),
            ]
        return [
            self._run_command(["systemctl", "--user", "daemon-reload"]),
            self._run_command(["systemctl", "--user", "enable", "--now", SYSTEMD_SERVICE_NAME]),
        ]

    def _uninstall_commands(self, service_file: Path) -> list[ServiceCommandResult]:
        if self.platform == "macos":
            return [self._run_command(["launchctl", "bootout", self._launchd_domain, str(service_file)])]
        return [
            self._run_command(["systemctl", "--user", "disable", "--now", SYSTEMD_SERVICE_NAME]),
            self._run_command(["systemctl", "--user", "daemon-reload"]),
        ]

    def _restart_commands(self) -> list[ServiceCommandResult]:
        if self.platform == "macos":
            return [self._run_command(["launchctl", "kickstart", "-k", f"{self._launchd_domain}/{LAUNCHD_LABEL}"])]
        return [self._run_command(["systemctl", "--user", "restart", SYSTEMD_SERVICE_NAME])]

    @property
    def _launchd_domain(self) -> str:
        uid = self.user_id if self.user_id is not None else "$(id -u)"
        return f"gui/{uid}"

    def _run_command(self, command: Sequence[str]) -> ServiceCommandResult:
        executable = command[0]
        if self._which(executable) is None:
            return ServiceCommandResult(
                command=tuple(command),
                available=False,
                returncode=None,
                stderr=f"{executable} is not available on PATH",
            )
        return self._command_runner(command)

    @staticmethod
    def _default_command_runner(command: Sequence[str]) -> ServiceCommandResult:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
        return ServiceCommandResult(
            command=tuple(command),
            available=True,
            returncode=completed.returncode,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
        )

    @staticmethod
    def _normalize_systemd_state(result: ServiceCommandResult, *, fallback: str) -> str:
        if not result.available:
            return fallback
        output = (result.stdout or result.stderr).strip()
        if output:
            return output
        return "ok" if result.returncode == 0 else fallback

    @staticmethod
    def _control_warnings(command_results: Sequence[ServiceCommandResult]) -> list[str]:
        warnings: list[str] = []
        available_results = [result for result in command_results if result.available]
        if command_results and not available_results:
            warnings.append("Service control commands are unavailable; file changes were applied without OS registration.")
        for result in available_results:
            if result.returncode not in {0, None}:
                stderr = result.stderr or result.stdout or "command returned a non-zero exit status"
                warnings.append(f"Command {' '.join(result.command)}: {stderr}")
        return warnings

    @staticmethod
    def _quote_systemd_value(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    @classmethod
    def _join_systemd_arguments(cls, arguments: Sequence[str]) -> str:
        parts: list[str] = []
        for argument in arguments:
            if argument and all(character.isalnum() or character in "/._-:=+" for character in argument):
                parts.append(argument)
            else:
                parts.append(cls._quote_systemd_value(argument))
        return " ".join(parts)

    @staticmethod
    def _tail_file(path: Path, lines: int) -> str:
        if not path.exists():
            return ""
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
