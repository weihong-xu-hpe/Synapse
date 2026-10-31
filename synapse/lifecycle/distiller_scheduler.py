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


# schema_meta counter of knowledge nodes distilled since the last Dreamer run
# (design doc §3.4(3): every N new nodes trigger an extra sleep cycle).
_DREAMER_PENDING_KEY = "dreamer_pending_knowledge_nodes"


class DistillerScheduler:
    """In-process periodic sweep of session transcripts."""

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths,
        logger: Any = None,
        timer_factory: Callable[..., Any] = threading.Timer,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths
        self._logger = logger or __import__("logging").getLogger("synapse.distiller")
        self._timer_factory = timer_factory
        self._now_fn: Callable[[], datetime] = now_fn or (lambda: datetime.now().astimezone())
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
                runs, attempts = self._distill(transcripts, runner)
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
                    failures=sum(attempt["outcome"] == "failed" for attempt in attempts),
                    archived_transcripts=archived,
                    trigger="sweep",
                    attempts_json=attempts,
                )
                # Event-driven sleep consolidation (§3.4(3)): fire a Dreamer
                # run in the background once enough new knowledge accumulated.
                self._maybe_trigger_dreamer(runs)
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

    def _maybe_trigger_dreamer(self, runs: list[Any]) -> None:
        """Event-driven and sleep/wake Dreamer triggers (§3.4(3)).

        Two independent reasons to fire an off-schedule Dreamer cycle:

        * **Event accumulation** — newly created/updated knowledge nodes
          accumulate in ``schema_meta`` across sweeps; once
          ``dreamer.event_trigger_new_nodes`` is reached the counter resets and
          a background run is fired.
        * **Missed wall-clock schedule** — the MacBook lid may be closed across
          the ``schedule_hour:schedule_minute`` window, so both ``launchd`` and
          the in-process ``threading.Timer`` silently miss the run. Because this
          scheduler ticks every few minutes while the machine is awake, it
          checks whether today's scheduled run already happened and, if not,
          catches up within one interval of the lid opening.

        Best effort — accounting failures must never break the sweep.
        """

        if not getattr(self.config.dreamer, "enabled", True):
            return
        new_nodes = sum(
            1
            for run in runs
            for op in (getattr(run, "ops", None) or [])
            if str(op.get("action")) in {"created", "updated"}
        )
        threshold = self.config.dreamer.event_trigger_new_nodes
        pending = 0
        event_due = False
        catch_up_due = False
        try:
            from synapse.storage import SQLiteNodeStore

            with SQLiteNodeStore(
                self.runtime_paths.base / "synapse.db",
                embedding_dimension=self.config.embedding.dimension or 0,
            ) as store:
                pending = int(store.get_meta(_DREAMER_PENDING_KEY) or 0) + new_nodes
                if pending >= threshold:
                    store.set_meta(_DREAMER_PENDING_KEY, "0")
                    event_due = True
                elif new_nodes:
                    store.set_meta(_DREAMER_PENDING_KEY, str(pending))
                catch_up_due = self._dreamer_schedule_catch_up_due(store)
        except Exception as exc:  # noqa: BLE001 — accounting must never break the sweep
            self._logger.warning("Dreamer event-trigger accounting failed", extra={"error": str(exc)})
            return

        if not (event_due or catch_up_due):
            return
        if catch_up_due and not event_due:
            self._logger.info(
                "Scheduled Dreamer run missed across sleep/wake; catching up",
                extra={"schedule_hour": self.config.dreamer.schedule_hour},
            )
        self._logger.info(
            "Triggering Dreamer run",
            extra={
                "pending_knowledge_nodes": pending,
                "threshold": threshold,
                "event_due": event_due,
                "catch_up_due": catch_up_due,
            },
        )
        self._trigger_dreamer_thread()

    def _dreamer_schedule_catch_up_due(self, store: Any) -> bool:
        """True when today's wall-clock Dreamer slot has passed without a run.

        Compares local wall-clock time against ``dreamer.schedule_hour/minute``
        and the newest ``dreamer_runs.started_at``.  Only meaningful in
        wall-clock mode (``schedule_hour`` set); returns False otherwise or when
        the machine is still before today's slot.
        """

        dreamer = self.config.dreamer
        schedule_hour = getattr(dreamer, "schedule_hour", None)
        if schedule_hour is None:
            return False
        now = self._now_fn()
        scheduled_today = now.replace(
            hour=schedule_hour,
            minute=dreamer.schedule_minute,
            second=0,
            microsecond=0,
        )
        if now < scheduled_today:
            return False

        row = store.connection.execute("SELECT MAX(started_at) AS value FROM dreamer_runs").fetchone()
        latest_started = row["value"] if row is not None else None
        if not latest_started:
            return True
        try:
            parsed = datetime.fromisoformat(str(latest_started).replace("Z", "+00:00"))
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=now.tzinfo)
        return parsed.astimezone(now.tzinfo) < scheduled_today

    def _trigger_dreamer_thread(self) -> None:
        """Spawn the background Dreamer cycle (seam for tests)."""

        threading.Thread(target=self._run_triggered_dreamer, name="dreamer-event-trigger", daemon=True).start()

    def _run_triggered_dreamer(self) -> None:
        """One immediate Dreamer cycle, under the Dreamer's own process lock."""

        try:
            from synapse.lifecycle.scheduler import DreamerScheduler

            DreamerScheduler(
                self.config,
                runtime_paths=self.runtime_paths,
                logger=self._logger,
            ).run_once_now()
        except Exception as exc:  # noqa: BLE001 — triggered run must not crash the scheduler
            self._logger.warning("Event-driven Dreamer trigger failed", extra={"error": str(exc)})

    def _distill(self, transcripts: list[Node], runner: Any) -> tuple[list[Any], list[dict[str, Any]]]:
        from synapse.lifecycle.episode_distiller import attempt_record

        runs: list[Any] = []
        attempts: list[dict[str, Any]] = []
        for transcript in transcripts:
            try:
                if getattr(getattr(transcript, "metadata", None), "force_retry", False):
                    run = runner.run(transcript.id, force=True)
                else:
                    run = runner.run(transcript.id)
            except Exception as exc:
                stage = getattr(exc, "distiller_stage", None) or "unknown"
                attempts.append(attempt_record(transcript.id, None, stage=stage))
                self._logger.exception("Episode sweep failed", extra={"transcript_id": transcript.id, "stage": stage})
            else:
                runs.append(run)
                attempts.append(attempt_record(transcript.id, run))
        return runs, attempts

    def _write_metrics_snapshots(self) -> None:
        try:
            from synapse.observability import write_missing_snapshots

            written = write_missing_snapshots(self.config, self.runtime_paths)
            if written:
                self._logger.info("Metrics snapshots written", extra={"dates": written})
        except Exception as exc:  # noqa: BLE001 — snapshots must never break the sweep
            self._logger.warning("Metrics snapshot write failed", extra={"error": str(exc)})
