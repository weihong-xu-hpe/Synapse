"""Periodic scheduler for the server-side session distiller.

Mirrors DreamerScheduler (synapse/lifecycle/scheduler.py) but runs on a short
interval (default 10 min), where macOS sleep drift is harmless: a stretched or
skipped tick loses nothing because the next sweep re-selects the same idle
transcripts. Single-flight via flock so CLI backfill runs never overlap the
live sweep.
"""

from __future__ import annotations

import fcntl
import threading
from typing import Any, Callable

from synapse.config import SynapseConfig
from synapse.lifecycle.distiller import Distiller
from synapse.utils.runtime import RuntimePaths


class DistillerScheduler:
    """In-process periodic sweep of session transcripts."""

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths,
        logger: Any = None,
        sampling_client: Any | None = None,
        timer_factory: Callable[..., Any] = threading.Timer,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths
        self._logger = logger or __import__("logging").getLogger("synapse.distiller")
        self._sampling_client = sampling_client
        self._timer_factory = timer_factory
        self._state_lock = threading.RLock()
        self._timer: Any | None = None
        self._running = False

    @property
    def interval_seconds(self) -> int:
        return self.config.distiller.interval_minutes * 60

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    def start(self) -> None:
        with self._state_lock:
            if self._running:
                return
            self._running = True
            self._schedule_next_locked()
        self._logger.info("Distiller scheduler started", extra={"interval_seconds": self.interval_seconds})

    def stop(self) -> None:
        with self._state_lock:
            self._running = False
            timer = self._timer
            self._timer = None
            if timer is not None:
                timer.cancel()
        self._logger.info("Distiller scheduler stopped")

    def _schedule_next_locked(self) -> None:
        if not self._running:
            return
        timer = self._timer_factory(self.interval_seconds, self._run_once)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _run_once(self) -> None:
        with self._state_lock:
            if not self._running:
                return
        try:
            self._run_sweep()
        except Exception as exc:  # noqa: BLE001 — scheduler must survive one failed run
            self._logger.warning("Distiller sweep failed", extra={"error": str(exc)})
        finally:
            with self._state_lock:
                self._schedule_next_locked()

    def _run_sweep(self) -> None:
        lock_path = self.runtime_paths.logs / "distiller.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("w")
        try:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._logger.info("Distiller sweep skipped: another run holds the lock")
                return
            distiller = Distiller(
                self.config,
                runtime_paths=self.runtime_paths,
                sampling_client=self._sampling_client,
                logger=self._logger,
            )
            try:
                distiller.run()
            finally:
                distiller.close()
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
