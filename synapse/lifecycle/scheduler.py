"""In-process scheduling for periodic Dreamer consolidation runs."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from threading import Timer
from typing import TYPE_CHECKING, Any, Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Synapse currently targets Unix runtimes
    fcntl = None  # type: ignore[assignment]

from synapse.config import SynapseConfig
from synapse.utils.runtime import RuntimePaths, get_runtime_paths

if TYPE_CHECKING:
    from synapse.lifecycle.dreamer import Dreamer


LOGGER = logging.getLogger("synapse.dreamer.scheduler")
TimerFactory = Callable[[float, Callable[[], None]], Any]
DreamerFactory = Callable[..., Any]


@contextmanager
def dreamer_process_lock(lock_path: Path) -> Iterator[bool]:
    """Cross-process single-flight lock for one Dreamer cycle.

    Shared by :class:`DreamerScheduler` (timer + event triggers) and the
    ``synapse dreamer run`` CLI command so an on-demand run can never overlap a
    scheduled or event-driven one. Yields ``True`` when this process owns the
    lock and ``False`` when another process already holds it.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        if fcntl is None:  # pragma: no cover - defensive portability fallback
            yield True
            return

        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return

        try:
            yield True
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class DreamerScheduler:
    """Run Dreamer periodically while preventing overlapping processes.

    The scheduler owns only the timer and the lock file.  A fresh Dreamer is
    created for each run so its SQLite connection cannot be retained between
    scheduled executions.
    """

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths | None = None,
        logger: logging.Logger | None = None,
        sampling_client: Any | None = None,
        lock_path: Path | None = None,
        timer_factory: TimerFactory | None = None,
        dreamer_factory: DreamerFactory | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths or get_runtime_paths(config)
        self._logger = logger or LOGGER
        if sampling_client is None:
            from synapse.server.decider import LocalLLMDecider

            # The Dreamer's LLM stages run against the same long reasoning
            # prompts as the distiller (measured 150-160 s/call on
            # glm-5.3-flash), so it reuses the distiller's dedicated timeout
            # instead of the write-path [decider] timeout_seconds (120 s live),
            # which would abort every stage mid-call.
            settings = self.config.decider.model_copy(
                update={"timeout_seconds": self.config.distiller.llm_timeout_seconds}
            )
            sampling_client = LocalLLMDecider(settings)
        self._sampling_client = sampling_client
        self._lock_path = lock_path or self.runtime_paths.logs / "dreamer.lock"
        self._timer_factory = timer_factory or Timer
        if dreamer_factory is None:
            from synapse.lifecycle.dreamer import Dreamer

            dreamer_factory = Dreamer
        self._dreamer_factory = dreamer_factory
        self._state_lock = threading.RLock()
        self._timer: Any | None = None
        self._running = False
        self._now_fn: Callable[[], datetime] = now_fn or (lambda: datetime.now().astimezone())

    @property
    def interval_seconds(self) -> int:
        return self.config.dreamer.interval_hours * 60 * 60

    def next_delay_seconds(self) -> float:
        """Seconds until the next scheduled run.

        Wall-clock mode (``schedule_hour`` set): the next local occurrence of
        ``schedule_hour:schedule_minute`` — today when it is still more than a
        second away, otherwise tomorrow.  A service restarted at 14:00 or
        20:00 therefore fires the *next* day at noon instead of accumulating
        drift into the 02:00-03:00 window where the MacBook is typically
        asleep and the run would be lost (§3.4(3)).
        """

        schedule_hour = self.config.dreamer.schedule_hour
        if schedule_hour is None:
            return float(self.interval_seconds)
        now = self._now_fn()
        target = now.replace(
            hour=schedule_hour,
            minute=self.config.dreamer.schedule_minute,
            second=0,
            microsecond=0,
        )
        if (target - now).total_seconds() <= 1:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    @property
    def timer(self) -> Any | None:
        with self._state_lock:
            return self._timer

    def start(self) -> None:
        """Start periodic scheduling; repeated calls are idempotent."""

        with self._state_lock:
            if self._running:
                return
            self._running = True
            self._schedule_next_locked()
        self._logger.info("Dreamer scheduler started", extra={"interval_seconds": self.interval_seconds})

    def stop(self) -> None:
        """Cancel the pending timer and prevent any subsequent scheduling."""

        with self._state_lock:
            self._running = False
            timer = self._timer
            self._timer = None
            if timer is not None:
                timer.cancel()
        self._logger.info("Dreamer scheduler stopped")

    def _schedule_next_locked(self) -> None:
        if not self._running:
            return
        timer = self._timer_factory(self.next_delay_seconds(), self._run_once)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _run_once(self) -> None:
        with self._state_lock:
            if not self._running:
                return

        try:
            with self._process_lock() as acquired:
                if acquired:
                    self._run_dreamer()
        except Exception as exc:  # pylint: disable=broad-exception-caught  # scheduler must survive one failed run
            self._logger.warning("Scheduled Dreamer run failed; continuing schedule", exc_info=exc)
        finally:
            with self._state_lock:
                self._schedule_next_locked()

    def run_once_now(self) -> bool:
        """Run one Dreamer cycle immediately, without touching the timer (§3.4(3)).

        Used by event-driven triggers (the distiller scheduler fires it once
        enough new knowledge has accumulated). Returns False when another
        process already owns the Dreamer lock.
        """

        try:
            with self._process_lock() as acquired:
                if not acquired:
                    return False
                self._run_dreamer()
        except Exception as exc:  # pylint: disable=broad-exception-caught  # a triggered run must not crash the caller
            self._logger.warning("Event-triggered Dreamer run failed", exc_info=exc)
            return False
        return True

    def _run_dreamer(self) -> None:
        dreamer = self._dreamer_factory(
            self.config,
            runtime_paths=self.runtime_paths,
            sampling_client=self._sampling_client,
            logger=self._logger,
        )
        try:
            report = dreamer.run(batch_size=self.config.dreamer.batch_size)
        finally:
            dreamer.close()
        self._logger.info("Scheduled Dreamer run completed", extra={"report": report.to_dict()})

    @contextmanager
    def _process_lock(self) -> Iterator[bool]:
        """Acquire the Unix process lock, yielding false when another run owns it."""

        with dreamer_process_lock(self._lock_path) as acquired:
            if not acquired:
                self._logger.warning("Dreamer lock unavailable; skipping scheduled run")
            yield acquired
