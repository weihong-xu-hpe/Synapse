"""Append-only JSONL history of eval runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_PATH = Path("~/.synapse/eval/history.jsonl")


def git_sha(repo_dir: str | Path | None = None) -> str | None:
    """Short commit sha of the repo working tree, or None when unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_history_line(
    history_path: str | Path,
    report: dict[str, Any],
    *,
    golden_path: str | Path,
    repo_dir: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append one JSON line summarizing an eval run; returns the line written."""

    ts = (now or datetime.now(UTC)).isoformat()
    line: dict[str, Any] = {
        "ts": ts,
        "git_sha": git_sha(repo_dir),
        "golden_sha256": file_sha256(golden_path),
        "n_queries": report.get("overall", {}).get("n", 0),
        "overall": report.get("overall", {}),
        "slices": report.get("slices", {}),
        "stale_labels": report.get("stale_labels_total", 0),
        "queries_without_labels": report.get("queries_without_labels", []),
    }
    path = Path(history_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    return line
