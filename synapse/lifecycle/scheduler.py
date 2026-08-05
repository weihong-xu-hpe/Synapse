"""In-process scheduling for periodic Dreamer consolidation runs."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
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
    from synapse.server.sampling import SamplingClient


LOGGER = logging.getLogger("synapse.dreamer.scheduler")
TimerFactory = Callable[[float, Callable[[], None]], Any]
DreamerFactory = Callable[..., Any]


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
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths or get_runtime_paths(config)
        self._logger = logger or LOGGER
        if sampling_client is None:
            from synapse.server.decider import LocalLLMDecider

            sampling_client = LocalLLMDecider(config.decider)
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

    @property
    def interval_seconds(self) -> int:
        return self.config.dreamer.interval_hours * 60 * 60

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
        timer = self._timer_factory(self.interval_seconds, self._run_once)
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

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+") as lock_file:
            if fcntl is None:  # pragma: no cover - defensive portability fallback
                yield True
                return

            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self._logger.warning("Dreamer lock unavailable; skipping scheduled run: %s", exc)
                yield False
                return

            try:
                yield True
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
