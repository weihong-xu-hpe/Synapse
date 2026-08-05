"""E2E gate tests for LocalLLMDecider against the real internal LLM endpoint.

These tests are marked ``live`` and skipped by default (see pyproject.toml
``addopts = "-q -m 'not live'"``). Run them manually with::

    pytest -m live

They probe the real OpenAI-compatible endpoints (deepseek-v4-pro primary,
glm-5.2-fp8 fallback) to guard against protocol drift, JSON-shape regressions,
and fallback-chain breakage. When an endpoint is unreachable the tests skip
rather than fail, so CI flakiness from network blips is avoided. But when the
endpoint *is* reachable, any protocol/parse error fails the gate.
"""

from __future__ import annotations

import os

import pytest

from synapse.config import DeciderSettings
from synapse.server.decider import LocalLLMDecider
from synapse.server.sampling import (
    MemoryWriteSamplingDecision,
    MemoryWriteSamplingRequest,
    SamplingCandidate,
)
from synapse.models import Node, NodeMetadata, NodeStatus

from tests.conftest import LIVE_FALLBACK_URL, LIVE_PRIMARY_URL


pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def _create_request() -> MemoryWriteSamplingRequest:
    """A write request with no candidates -- LLM should decide 'create'."""
    return MemoryWriteSamplingRequest(
        prompt="Decide how to write this memory.",
        title="Gateway rate limits",
        content="Rate limiting complements the gateway design and protects upstream services.",
        node_type="transient",
        sensitivity="internal",
        query="gateway rate limits",
        similarity_threshold=0.3,
        links=(),
        candidates=(),
        candidate_nodes=(),
    )


def _complement_request() -> MemoryWriteSamplingRequest:
    """A write request with one similar candidate -- LLM should decide 'complement'."""
    candidate_node = Node(
        metadata=NodeMetadata(
            id="gateway-design",
            title="Gateway design",
            status=NodeStatus.ACTIVE,
        ),
        content="The API gateway routes traffic and applies auth policies.",
        file_path="active/gateway-design.md",
    )
    candidate = SamplingCandidate(
        node_id="gateway-design",
        title="Gateway design",
        score=0.82,
        status="active",
        sensitivity="internal",
        file_path="active/gateway-design.md",
    )
    return MemoryWriteSamplingRequest(
        prompt="Decide how to write this memory.",
        title="Gateway rate limits",
        content="Rate limiting complements the gateway design and protects upstream services.",
        node_type="transient",
        sensitivity="internal",
        query="gateway rate limits",
        similarity_threshold=0.3,
        links=(),
        candidates=(candidate,),
        candidate_nodes=(candidate_node,),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_live_decider_decide_memory_write_create(live_decider, live_primary_reachable) -> None:
    """No candidates -> real LLM returns a valid 'create' decision."""
    if not live_primary_reachable:
        pytest.skip(f"primary LLM endpoint unreachable: {LIVE_PRIMARY_URL}")

    decision = live_decider.decide_memory_write(_create_request())

    assert isinstance(decision, MemoryWriteSamplingDecision)
    assert decision.action == "create"
    assert decision.target_node_ids == ()
    assert decision.reasoning  # non-empty


def test_live_decider_decide_memory_write_complement(live_decider, live_primary_reachable) -> None:
    """A similar candidate exists -> real LLM returns a 'complement' decision targeting it."""
    if not live_primary_reachable:
        pytest.skip(f"primary LLM endpoint unreachable: {LIVE_PRIMARY_URL}")

    decision = live_decider.decide_memory_write(_complement_request())

    assert isinstance(decision, MemoryWriteSamplingDecision)
    assert decision.action == "complement"
    assert "gateway-design" in decision.target_node_ids
    assert decision.reasoning


def test_live_decider_fallback_endpoint_works(live_fallback_reachable) -> None:
    """Primary endpoint misconfigured -> fallback to glm-5.2-fp8 succeeds.

    Requires the OPENAI_COMPATIBLE_API_KEY env var (sk- prefixed) and a
    reachable fallback endpoint. Skips otherwise.
    """
    api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_COMPATIBLE_API_KEY not set; cannot test fallback endpoint auth")
    if not live_fallback_reachable:
        pytest.skip(f"fallback LLM endpoint unreachable: {LIVE_FALLBACK_URL}")

    # Primary endpoint deliberately broken so the decider must fall back.
    settings = DeciderSettings(
        provider="local_llm",
        base_url="http://10.235.33.60:80/invalid-path",
        model="deepseek-v4-pro",
        api_key_env="",
        fallback_base_url=LIVE_FALLBACK_URL,
        fallback_model="glm-5.2-fp8",
        fallback_api_key_env="OPENAI_COMPATIBLE_API_KEY",
        timeout_seconds=30,
        max_tokens=600,
        temperature=0.1,
    )
    decider = LocalLLMDecider(settings)

    decision = decider.decide_memory_write(_create_request())

    assert isinstance(decision, MemoryWriteSamplingDecision)
    assert decision.action == "create"
    assert decision.target_node_ids == ()
    assert decision.reasoning


def test_live_decider_sample_json_handles_prose_wrapped_json(live_decider, live_primary_reachable) -> None:
    """Real LLM may wrap JSON in prose/markdown fences -- _extract_json_payload must cope."""
    if not live_primary_reachable:
        pytest.skip(f"primary LLM endpoint unreachable: {LIVE_PRIMARY_URL}")

    result = live_decider.sample_json(
        prompt="Return exactly one JSON object with a key 'keep' and boolean value true. "
        "Do not include any other keys.",
        system_prompt="You are a JSON assistant.",
        max_tokens=200,
    )

    assert isinstance(result, dict)
    assert "keep" in result
    assert result["keep"] is True
