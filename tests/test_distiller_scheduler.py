from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from synapse.config import SynapseConfig
from synapse.lifecycle import distiller as distiller_module
from synapse.lifecycle import episode_distiller as episode_module
from synapse.lifecycle.distiller_scheduler import DistillerScheduler
from synapse.utils.runtime import bootstrap_runtime_directories


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, fail_ids: set[str]) -> dict[str, Any]:
    state: dict[str, Any] = {"ran": [], "clients": [], "archive_calls": 0, "limits": [], "metrics": []}

    def fake_select(self: Any, *, limit: int | None = None) -> list[Any]:
        state["limits"].append(limit)
        return [SimpleNamespace(id="mem_session_a"), SimpleNamespace(id="mem_session_b")]

    def fake_archive(self: Any, report: Any) -> int:
        state["archive_calls"] += 1
        return 0

    class FakeEpisodeDistiller:
        def __init__(self, config: Any, *, runtime_paths: Any, sampling_client: Any, gatekeeper_client: Any = None) -> None:
            pass

        @staticmethod
        def make_scanner_client(config: Any) -> _FakeClient:
            client = _FakeClient()
            state["clients"].append(client)
            return client

        @staticmethod
        def make_gatekeeper_client(config: Any) -> _FakeClient:
            client = _FakeClient()
            state["clients"].append(client)
            return client

        def run(self, session: str) -> Any:
            if session in fail_ids:
                raise RuntimeError("boom")
            state["ran"].append(session)
            return SimpleNamespace(stamped=True, lessons=[{}], ops=[{"action": "created"}])

        def record_run_metrics(self, runs: list[Any], **kwargs: Any) -> None:
            state["metrics"].append((runs, kwargs))

        def close(self) -> None:
            pass

    monkeypatch.setattr(distiller_module.Distiller, "select_transcripts", fake_select)
    monkeypatch.setattr(distiller_module.Distiller, "_archive_expired", fake_archive)
    monkeypatch.setattr(episode_module, "EpisodeDistiller", FakeEpisodeDistiller)
    monkeypatch.setattr(DistillerScheduler, "_write_metrics_snapshots", lambda self: None)
    return state


def _config(tmp_path: Any) -> tuple[SynapseConfig, Any]:
    config = SynapseConfig.with_defaults(tmp_path)
    config.embedding.provider = "builtin"
    config.reranker.provider = "builtin"
    # Sweep-mechanics tests are not about the wall-clock Dreamer catch-up; pin
    # the interval fallback so they never spawn a background Dreamer thread.
    config.dreamer.schedule_hour = None
    return config, bootstrap_runtime_directories(config)


def test_sweep_runs_episode_distiller_on_selected_transcripts(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config, runtime_paths = _config(tmp_path)
    state = _install_fakes(monkeypatch, fail_ids=set())

    DistillerScheduler(config, runtime_paths=runtime_paths)._run_sweep()

    assert state["limits"] == [config.distiller.max_transcripts_per_run]
    assert state["ran"] == ["mem_session_a", "mem_session_b"]
    assert state["archive_calls"] == 1
    assert state["clients"] and all(client.closed for client in state["clients"])
    assert state["metrics"][0][1]["transcripts_scanned"] == 2
    assert state["metrics"][0][1]["failures"] == 0
    assert [item["outcome"] for item in state["metrics"][0][1]["attempts_json"]] == ["distilled", "distilled"]


def test_sweep_continues_after_one_transcript_fails(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config, runtime_paths = _config(tmp_path)
    state = _install_fakes(monkeypatch, fail_ids={"mem_session_a"})

    DistillerScheduler(config, runtime_paths=runtime_paths)._run_sweep()

    assert state["ran"] == ["mem_session_b"]
    assert state["archive_calls"] == 1
    assert state["metrics"][0][1]["transcripts_scanned"] == 2
    assert state["metrics"][0][1]["failures"] == 1
    assert state["metrics"][0][1]["attempts_json"][0]["outcome"] == "failed"


def test_empty_sweep_still_records_run_metrics(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config, runtime_paths = _config(tmp_path)
    state = _install_fakes(monkeypatch, fail_ids=set())
    monkeypatch.setattr(distiller_module.Distiller, "select_transcripts", lambda self, *, limit=None: [])

    DistillerScheduler(config, runtime_paths=runtime_paths)._run_sweep()

    assert state["archive_calls"] == 1
    assert len(state["metrics"]) == 1
    assert state["metrics"][0][0] == []
    assert state["metrics"][0][1]["transcripts_scanned"] == 0


# --- sleep/wake wall-clock Dreamer catch-up ---------------------------------


def _seed_dreamer_run(runtime_paths: Any, started_at: str) -> None:
    from synapse.storage import DreamerRunMetrics, SQLiteNodeStore

    zero = dict(
        stale_scanned=0,
        superseded_scanned=0,
        disputed_scanned=0,
        missing_link_pairs_scanned=0,
        triage_keep=0,
        triage_condense=0,
        triage_archive=0,
        links_added=0,
        conflicts_superseded=0,
        conflicts_both_valid=0,
        archived=0,
        condensed=0,
        warnings=0,
        sampling_failures=0,
    )
    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=0) as store:
        store.record_dreamer_run(
            DreamerRunMetrics(
                started_at=started_at,
                completed_at=started_at,
                duration_ms=0,
                batch_size=8,
                **zero,
            )
        )


def _store(runtime_paths: Any) -> Any:
    from synapse.storage import SQLiteNodeStore

    return SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=0)


def test_catch_up_not_due_before_noon(tmp_path: Any) -> None:
    config, runtime_paths = _config(tmp_path)
    config.dreamer.schedule_hour = 12
    scheduler = DistillerScheduler(
        config, runtime_paths=runtime_paths, now_fn=lambda: datetime(2026, 10, 31, 9, 30).astimezone()
    )

    with _store(runtime_paths) as store:
        assert scheduler._dreamer_schedule_catch_up_due(store) is False


def test_catch_up_due_when_noon_run_never_happened(tmp_path: Any) -> None:
    config, runtime_paths = _config(tmp_path)
    config.dreamer.schedule_hour = 12
    scheduler = DistillerScheduler(
        config, runtime_paths=runtime_paths, now_fn=lambda: datetime(2026, 10, 31, 14, 0).astimezone()
    )

    with _store(runtime_paths) as store:
        assert scheduler._dreamer_schedule_catch_up_due(store) is True


def test_catch_up_not_due_when_today_run_already_happened(tmp_path: Any) -> None:
    config, runtime_paths = _config(tmp_path)
    config.dreamer.schedule_hour = 12
    # Run landed today at 12:05 local (after the 12:00 slot).
    _seed_dreamer_run(runtime_paths, datetime(2026, 10, 31, 12, 5).astimezone().isoformat())
    scheduler = DistillerScheduler(
        config, runtime_paths=runtime_paths, now_fn=lambda: datetime(2026, 10, 31, 14, 0).astimezone()
    )

    with _store(runtime_paths) as store:
        assert scheduler._dreamer_schedule_catch_up_due(store) is False


def test_catch_up_due_when_only_yesterdays_run_exists(tmp_path: Any) -> None:
    config, runtime_paths = _config(tmp_path)
    config.dreamer.schedule_hour = 12
    _seed_dreamer_run(runtime_paths, datetime(2026, 10, 30, 12, 5).astimezone().isoformat())
    scheduler = DistillerScheduler(
        config, runtime_paths=runtime_paths, now_fn=lambda: datetime(2026, 10, 31, 14, 0).astimezone()
    )

    with _store(runtime_paths) as store:
        assert scheduler._dreamer_schedule_catch_up_due(store) is True


def test_catch_up_disabled_when_interval_mode(tmp_path: Any) -> None:
    config, runtime_paths = _config(tmp_path)
    config.dreamer.schedule_hour = None
    scheduler = DistillerScheduler(
        config, runtime_paths=runtime_paths, now_fn=lambda: datetime(2026, 10, 31, 23, 0).astimezone()
    )

    with _store(runtime_paths) as store:
        assert scheduler._dreamer_schedule_catch_up_due(store) is False


def test_missed_noon_slot_triggers_dreamer_thread(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config, runtime_paths = _config(tmp_path)
    config.dreamer.schedule_hour = 12
    calls: list[int] = []
    monkeypatch.setattr(DistillerScheduler, "_trigger_dreamer_thread", lambda self: calls.append(1))

    scheduler = DistillerScheduler(
        config, runtime_paths=runtime_paths, now_fn=lambda: datetime(2026, 10, 31, 14, 0).astimezone()
    )
    scheduler._maybe_trigger_dreamer([])

    assert calls == [1]


def test_event_threshold_triggers_dreamer_thread(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config, runtime_paths = _config(tmp_path)
    config.dreamer.event_trigger_new_nodes = 2
    calls: list[int] = []
    monkeypatch.setattr(DistillerScheduler, "_trigger_dreamer_thread", lambda self: calls.append(1))

    scheduler = DistillerScheduler(config, runtime_paths=runtime_paths)
    make_run = lambda: SimpleNamespace(ops=[{"action": "created"}])  # noqa: E731
    scheduler._maybe_trigger_dreamer([make_run(), make_run()])

    assert calls == [1]
