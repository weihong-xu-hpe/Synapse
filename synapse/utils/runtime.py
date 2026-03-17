"""Runtime path helpers for Synapse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synapse.config import SynapseConfig


@dataclass(slots=True, frozen=True)
class RuntimePaths:
    """Resolved runtime directory layout."""

    base: Path
    active: Path
    archive: Path
    logs: Path
    audit: Path


def get_runtime_paths(config: SynapseConfig) -> RuntimePaths:
    """Resolve runtime paths from the loaded config."""

    base = config.resolve_path(config.memory.base_path)
    archive = config.resolve_path(config.memory.archive_path)
    logs = config.resolve_path(config.logging.log_dir)
    audit = base / ".audit"
    active = base / "active"
    return RuntimePaths(
        base=base,
        active=active.resolve(),
        archive=archive,
        logs=logs,
        audit=audit.resolve(),
    )


def bootstrap_runtime_directories(config: SynapseConfig) -> RuntimePaths:
    """Create the runtime directory structure required by Phase 1."""

    paths = get_runtime_paths(config)
    for path in (paths.base, paths.active, paths.archive, paths.logs, paths.audit):
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError:
            # Best-effort only. Some filesystems or environments may not support chmod.
            pass
    return paths
