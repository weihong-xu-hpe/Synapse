from __future__ import annotations

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
    assert state["metrics"][0][1]["llm_failures"] == 0


def test_sweep_continues_after_one_transcript_fails(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config, runtime_paths = _config(tmp_path)
    state = _install_fakes(monkeypatch, fail_ids={"mem_session_a"})

    DistillerScheduler(config, runtime_paths=runtime_paths)._run_sweep()

    assert state["ran"] == ["mem_session_b"]
    assert state["archive_calls"] == 1
    assert state["metrics"][0][1]["transcripts_scanned"] == 2
    assert state["metrics"][0][1]["llm_failures"] == 1


def test_empty_sweep_still_records_run_metrics(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config, runtime_paths = _config(tmp_path)
    state = _install_fakes(monkeypatch, fail_ids=set())
    monkeypatch.setattr(distiller_module.Distiller, "select_transcripts", lambda self, *, limit=None: [])

    DistillerScheduler(config, runtime_paths=runtime_paths)._run_sweep()

    assert state["archive_calls"] == 1
    assert len(state["metrics"]) == 1
    assert state["metrics"][0][0] == []
    assert state["metrics"][0][1]["transcripts_scanned"] == 0
