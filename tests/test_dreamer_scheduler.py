from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from synapse.config import SynapseConfig
from synapse.lifecycle.scheduler import DreamerScheduler
from synapse.utils.runtime import bootstrap_runtime_directories


class FakeTimer:
    instances: list["FakeTimer"] = []

    def __init__(self, interval: float, callback) -> None:
        self.interval = interval
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.callback()


class FakeDreamer:
    instances: list["FakeDreamer"] = []
    should_fail = False

    def __init__(self, config, *, runtime_paths, sampling_client, logger) -> None:
        self.config = config
        self.runtime_paths = runtime_paths
        self.sampling_client = sampling_client
        self.logger = logger
        self.closed = False
        self.batch_sizes: list[int] = []
        self.instances.append(self)

    def run(self, *, batch_size: int):
        self.batch_sizes.append(batch_size)
        if self.should_fail:
            raise RuntimeError("dreamer failed")
        return SimpleNamespace(to_dict=lambda: {"ok": True})

    def close(self) -> None:
        self.closed = True


def make_scheduler(tmp_path: Path, *, dreamer_factory=FakeDreamer) -> DreamerScheduler:
    FakeTimer.instances.clear()
    FakeDreamer.instances.clear()
    FakeDreamer.should_fail = False
    config = SynapseConfig.with_defaults(tmp_path)
    config.dreamer.interval_hours = 2
    config.dreamer.batch_size = 5
    runtime_paths = bootstrap_runtime_directories(config)
    return DreamerScheduler(
        config,
        runtime_paths=runtime_paths,
        sampling_client=object(),
        timer_factory=FakeTimer,
        dreamer_factory=dreamer_factory,
    )


class FailingDreamer(FakeDreamer):
    def run(self, *, batch_size: int):
        self.batch_sizes.append(batch_size)
        raise RuntimeError("dreamer failed")


def test_scheduler_start_and_stop_are_idempotent(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path)

    scheduler.start()
    scheduler.start()

    assert scheduler.is_running is True
    assert len(FakeTimer.instances) == 1
    assert FakeTimer.instances[0].interval == 7200
    assert FakeTimer.instances[0].started is True

    scheduler.stop()
    scheduler.stop()

    assert scheduler.is_running is False
    assert FakeTimer.instances[0].cancelled is True


def test_scheduler_timer_runs_dreamer_and_schedules_next_round(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path)
    scheduler.start()

    FakeTimer.instances[0].fire()

    assert len(FakeDreamer.instances) == 1
    assert FakeDreamer.instances[0].batch_sizes == [5]
    assert FakeDreamer.instances[0].closed is True
    assert len(FakeTimer.instances) == 2
    assert FakeTimer.instances[1].started is True

    scheduler.stop()


def test_scheduler_survives_dreamer_failure_and_schedules_next_round(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, dreamer_factory=FailingDreamer)
    scheduler.start()

    FakeTimer.instances[0].fire()

    assert scheduler.is_running is True
    assert FakeDreamer.instances[0].closed is True
    assert len(FakeTimer.instances) == 2
    assert FakeTimer.instances[1].started is True

    scheduler.stop()