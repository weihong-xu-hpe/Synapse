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
import time
from datetime import UTC, datetime
from typing import Any, Callable
from synapse.config import SynapseConfig
from synapse.lifecycle.distiller import Distiller, DistillerReport
from synapse.models.node import Node
from synapse.utils.runtime import RuntimePaths


class DistillerScheduler:
    """In-process periodic sweep of session transcripts."""

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths,
        logger: Any = None,
        timer_factory: Callable[..., Any] = threading.Timer,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths
        self._logger = logger or __import__("logging").getLogger("synapse.distiller")
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
            started = datetime.now(UTC)
            started_at = started.isoformat().replace("+00:00", "Z")
            started_clock = time.perf_counter()
            selector = Distiller(self.config, runtime_paths=self.runtime_paths, logger=self._logger)
            from synapse.lifecycle.episode_distiller import EpisodeDistiller

            scanner_client = None
            gatekeeper_client = None
            runner = None
            try:
                scanner_client = EpisodeDistiller.make_scanner_client(self.config)
                gatekeeper_client = EpisodeDistiller.make_gatekeeper_client(self.config)
                runner = EpisodeDistiller(
                    self.config,
                    runtime_paths=self.runtime_paths,
                    sampling_client=scanner_client,
                    gatekeeper_client=gatekeeper_client,
                )
                transcripts = selector.select_transcripts(limit=self.config.distiller.max_transcripts_per_run)
                runs, llm_failures = self._distill(transcripts, runner)
                archive_started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                archived = selector._archive_expired(
                    DistillerReport(started_at=archive_started_at, completed_at=archive_started_at)
                )
                completed = datetime.now(UTC)
                runner.record_run_metrics(
                    runs,
                    started_at=started_at,
                    completed_at=completed.isoformat().replace("+00:00", "Z"),
                    duration_ms=round((time.perf_counter() - started_clock) * 1000),
                    transcripts_scanned=len(transcripts),
                    llm_failures=llm_failures,
                    archived_transcripts=archived,
                )
            finally:
                selector.close()
                try:
                    if runner is not None:
                        runner.close()
                finally:
                    for client in (scanner_client, gatekeeper_client):
                        close_client = getattr(client, "close", None)
                        if callable(close_client):
                            close_client()
            # Observability snapshot piggybacks on the sweep cadence (no
            # separate timer): today's row if missing, plus yesterday when the
            # machine slept through a day boundary. Never breaks the sweep.
            self._write_metrics_snapshots()
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    def _distill(self, transcripts: list[Node], runner: Any) -> tuple[list[Any], int]:
        runs: list[Any] = []
        failures = 0
        for transcript in transcripts:
            try:
                runs.append(runner.run(transcript.id))
            except Exception:
                failures += 1
                self._logger.exception("Episode sweep failed", extra={"transcript_id": transcript.id})
        return runs, failures

    def _write_metrics_snapshots(self) -> None:
        try:
            from synapse.observability import write_missing_snapshots

            written = write_missing_snapshots(self.config, self.runtime_paths)
            if written:
                self._logger.info("Metrics snapshots written", extra={"dates": written})
        except Exception as exc:  # noqa: BLE001 — snapshots must never break the sweep
            self._logger.warning("Metrics snapshot write failed", extra={"error": str(exc)})
