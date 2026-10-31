from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from synapse.config import SynapseConfig
from synapse.lifecycle.scheduler import DreamerScheduler, dreamer_process_lock
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
    config.dreamer.schedule_hour = None  # exercise the fixed-interval fallback
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


def make_wall_clock_scheduler(
    tmp_path: Path, now: datetime, *, dreamer_factory=FakeDreamer
) -> DreamerScheduler:
    """Scheduler pinned to a fixed `now` with the default noon wall-clock cadence."""

    FakeTimer.instances.clear()
    FakeDreamer.instances.clear()
    config = SynapseConfig.with_defaults(tmp_path)
    assert config.dreamer.schedule_hour == 12  # shipped default
    assert config.dreamer.schedule_minute == 0
    runtime_paths = bootstrap_runtime_directories(config)
    return DreamerScheduler(
        config,
        runtime_paths=runtime_paths,
        sampling_client=object(),
        timer_factory=FakeTimer,
        dreamer_factory=dreamer_factory,
        now_fn=lambda: now,
    )


def test_next_delay_targets_today_noon(tmp_path: Path) -> None:
    scheduler = make_wall_clock_scheduler(
        tmp_path, datetime(2026, 10, 31, 9, 30, 0).astimezone()
    )
    # 09:30 -> 12:00 same day = 2.5 h
    assert scheduler.next_delay_seconds() == 2.5 * 3600


def test_next_delay_rolls_over_to_tomorrow_noon(tmp_path: Path) -> None:
    # A service restarted at 14:00 or 20:00 must NOT drift into the 2/3 AM
    # sleep window; it targets tomorrow's noon instead.
    restarted_at_2pm = make_wall_clock_scheduler(
        tmp_path, datetime(2026, 10, 31, 14, 0, 0).astimezone()
    )
    assert restarted_at_2pm.next_delay_seconds() == (24 - 2) * 3600

    restarted_at_8pm = make_wall_clock_scheduler(
        tmp_path, datetime(2026, 10, 31, 20, 0, 0).astimezone()
    )
    assert restarted_at_8pm.next_delay_seconds() == (24 - 8) * 3600


def test_next_delay_at_noon_targets_next_day(tmp_path: Path) -> None:
    scheduler = make_wall_clock_scheduler(
        tmp_path, datetime(2026, 10, 31, 12, 0, 0).astimezone()
    )
    assert scheduler.next_delay_seconds() == 24 * 3600


def test_wall_clock_mode_drives_timer_interval(tmp_path: Path) -> None:
    clock = {"now": datetime(2026, 10, 31, 10, 0, 0).astimezone()}
    FakeTimer.instances.clear()
    FakeDreamer.instances.clear()
    config = SynapseConfig.with_defaults(tmp_path)
    runtime_paths = bootstrap_runtime_directories(config)
    scheduler = DreamerScheduler(
        config,
        runtime_paths=runtime_paths,
        sampling_client=object(),
        timer_factory=FakeTimer,
        dreamer_factory=FakeDreamer,
        now_fn=lambda: clock["now"],
    )
    scheduler.start()
    assert FakeTimer.instances[0].interval == 2 * 3600  # 10:00 -> today 12:00
    clock["now"] = datetime(2026, 10, 31, 12, 0, 0).astimezone()  # noon run fires
    FakeTimer.instances[0].fire()
    # After the noon run, the next round rolls to tomorrow noon: 24 h later.
    assert FakeTimer.instances[1].interval == 24 * 3600
    scheduler.stop()


def test_schedule_hour_none_falls_back_to_interval(tmp_path: Path) -> None:
    scheduler = make_wall_clock_scheduler(
        tmp_path, datetime(2026, 10, 31, 14, 0, 0).astimezone()
    )
    scheduler.config.dreamer.schedule_hour = None
    scheduler.config.dreamer.interval_hours = 24
    assert scheduler.next_delay_seconds() == 24 * 3600


def test_dreamer_process_lock_is_single_flight(tmp_path: Path) -> None:
    # The CLI `synapse dreamer run` and DreamerScheduler both go through this
    # helper, so an on-demand run can never overlap a scheduled one.
    lock_path = tmp_path / "dreamer.lock"
    with dreamer_process_lock(lock_path) as first:
        assert first is True
        with dreamer_process_lock(lock_path) as second:
            assert second is False
    with dreamer_process_lock(lock_path) as third:
        assert third is True