"""Cross-process coordination for transcript processing and file mutations."""

from __future__ import annotations

import fcntl
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal


@contextmanager
def transcript_lock(base_path: Path, node_id: str, *, scope: Literal["processing", "mutation"]) -> Iterator[None]:
    """Lock one transcript without holding its mutation lock during model calls.

    Locks are keyed by node ID rather than the incoming session reference, so
    close requests, sweeps and CLI jobs coordinate even when their inputs differ.
    Lock files must remain in place: unlinking one while it is held would let
    another process lock a different inode for the same transcript.
    """
    lock_dir = base_path / ".transcript-locks"
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = hashlib.sha256(node_id.encode("utf-8")).hexdigest()
    descriptor = os.open(lock_dir / f"{name}.{scope}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
