"""Shared fixtures and live-test skip logic for Synapse e2e gate tests."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from synapse.config import SynapseConfig
from synapse.server.decider import LocalLLMDecider
from synapse.utils.runtime import bootstrap_runtime_directories


LIVE_PRIMARY_URL = os.environ.get("SYNAPSE_LIVE_PRIMARY_URL", "")
LIVE_FALLBACK_URL = os.environ.get("SYNAPSE_LIVE_FALLBACK_URL", "")


def _endpoint_reachable(url: str, *, timeout: float = 2.0) -> bool:
    """Probe whether an OpenAI-compatible endpoint is reachable.

    Any HTTP response with status < 500 counts as reachable (even 404),
    because it proves the server is up and answering. Only connection
    failures or server errors count as unreachable.
    """
    try:
        response = httpx.get(url, timeout=timeout)
        return response.status_code < 500
    except (httpx.HTTPError, OSError):
        return False


@pytest.fixture
def live_primary_reachable() -> bool:
    return _endpoint_reachable(LIVE_PRIMARY_URL)


@pytest.fixture
def live_fallback_reachable() -> bool:
    return _endpoint_reachable(LIVE_FALLBACK_URL)


@pytest.fixture
def live_decider() -> LocalLLMDecider:
    """A LocalLLMDecider pointed at the real internal LLM endpoints (default config)."""
    config = SynapseConfig.with_defaults(Path("/tmp/synapse-live-test"))
    return LocalLLMDecider(config.decider)


@pytest.fixture
def live_dreamer_env(tmp_path: Path):
    """Bootstrap a full runtime environment (config + runtime_paths + decider) for live Dreamer tests."""
    config = SynapseConfig.with_defaults(tmp_path)
    runtime_paths = bootstrap_runtime_directories(config)
    decider = LocalLLMDecider(config.decider)
    return config, runtime_paths, decider
