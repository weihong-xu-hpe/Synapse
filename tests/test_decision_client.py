"""Unit tests for System One DecisionClient and DecisionModelSettings."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from synapse.config import DecisionModelEndpointSettings, DecisionModelSettings, SynapseConfig
from synapse.decision import DecisionClient
from synapse.decision.client import validate_questions

_QUESTION = [{"id": "q1", "type": "noul", "instructions": "Is this relevant?"}]


def test_validate_questions_rejects_shapes_the_server_answers_422_or_ignores() -> None:
    ok_choice = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": None}}
    assert validate_questions({"q": ok_choice}) is None
    assert validate_questions({"q": {"type": "noul", "instructions": "Yes?"}}) is None
    # legacy shape that was shipped and 422'd for two days: 'prompt' + 'choices'
    assert validate_questions({"q": {"type": "choice", "prompt": "Which?", "choices": ["a", "b"]}}) is not None
    # noul with 'prompt' is accepted by the server but ignored (generic question) -> must be rejected here
    assert validate_questions({"q": {"type": "noul", "prompt": "Yes?"}}) is not None
    assert validate_questions({"q": {"type": "choice", "instructions": "Which?"}}) is not None
    assert validate_questions({"q": {"type": "score", "instructions": "How?", "criteria": ["only-one"]}}) is not None
    assert validate_questions({}) is not None


def test_decision_client_does_not_send_malformed_questions() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"answers": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = DecisionModelSettings(
        enabled=True,
        endpoints=[DecisionModelEndpointSettings(name="primary", base_url="http://127.0.0.1:8088", model="startlux")],
    )
    try:
        assert DecisionClient(settings, http_client=client).decide("s", {"q": {"type": "noul", "prompt": "Yes?"}}) is None
        assert called is False
    finally:
        client.close()


def test_decision_model_settings_disabled_by_default(tmp_path: Path) -> None:
    config = SynapseConfig.with_defaults(tmp_path)
    assert config.decision_model.enabled is False
    assert config.decision_model.timeout_ms == 1200
    assert config.decision_model.offline_timeout_ms == 10000
    assert config.decision_model.min_confidence == 0.70
    assert config.decision_model.endpoints == []


def test_decision_client_returns_none_immediately_when_disabled() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = DecisionModelSettings(
        enabled=False,
        endpoints=[DecisionModelEndpointSettings(name="primary", base_url="http://127.0.0.1:8088", model="startlux")],
    )
    decision_client = DecisionClient(settings, http_client=client)
    try:
        assert decision_client.decide({"query": "x"}, _QUESTION) is None
        assert called is False
    finally:
        client.close()


def test_decision_client_formats_systemone_payload_and_auth_header(monkeypatch) -> None:
    monkeypatch.setenv("STARTLUX_TEST_KEY", "secret-token")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"answers": {"q1": True}, "confidence": 0.92})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = DecisionModelSettings(
        enabled=True,
        timeout_ms=1200,
        offline_timeout_ms=10000,
        endpoints=[
            DecisionModelEndpointSettings(
                name="startlux-local",
                base_url="http://127.0.0.1:8088",
                model="startlux-decision-4b",
                api_key_env="STARTLUX_TEST_KEY",
            )
        ],
    )
    decision_client = DecisionClient(settings, http_client=client)
    try:
        result = decision_client.decide(
            {"query": "authz timeout"},
            _QUESTION,
        )
    finally:
        client.close()

    assert result == {"answers": {"q1": True}, "confidence": 0.92}
    assert captured["url"] == "http://127.0.0.1:8088/v1/systemone"
    assert captured["auth"] == "Bearer secret-token"
    assert captured["body"] == {
        "state": {"query": "authz timeout"},
        "questions": _QUESTION,
        "model": "startlux-decision-4b",
    }


def test_decision_client_fails_over_across_ordered_endpoints() -> None:
    attempts: list[tuple[str, float | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeout = request.extensions.get("timeout", {})
        read_timeout = timeout.get("read") if isinstance(timeout, dict) else None
        attempts.append((str(request.url), read_timeout))
        if "8088" in str(request.url):
            raise httpx.ReadTimeout("primary timed out", request=request)
        return httpx.Response(200, json={"answers": {"q1": "allow"}, "confidence": 0.85})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = DecisionModelSettings(
        enabled=True,
        timeout_ms=500,
        offline_timeout_ms=9000,
        endpoints=[
            DecisionModelEndpointSettings(name="primary", base_url="http://127.0.0.1:8088", model="startlux"),
            DecisionModelEndpointSettings(name="secondary", base_url="http://127.0.0.1:8089/v1", model="karpo"),
        ],
    )
    decision_client = DecisionClient(settings, http_client=client)
    try:
        result = decision_client.decide("state", _QUESTION, offline=True)
    finally:
        client.close()

    assert result == {"answers": {"q1": "allow"}, "confidence": 0.85}
    assert attempts == [
        ("http://127.0.0.1:8088/v1/systemone", 9.0),
        ("http://127.0.0.1:8089/v1/systemone", 9.0),
    ]


def test_decision_client_returns_none_when_all_endpoints_fail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = DecisionModelSettings(
        enabled=True,
        endpoints=[
            DecisionModelEndpointSettings(name="primary", base_url="http://127.0.0.1:8088", model="startlux"),
            DecisionModelEndpointSettings(name="secondary", base_url="http://127.0.0.1:8089", model="karpo"),
        ],
    )
    decision_client = DecisionClient(settings, http_client=client)
    try:
        assert decision_client.decide("state", _QUESTION) is None
    finally:
        client.close()


def test_decision_client_supports_mapping_questions() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"answers": {"m0": {"type": "noul", "noul": 0.91}}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = DecisionModelSettings(
        enabled=True,
        endpoints=[
            DecisionModelEndpointSettings(
                name="startlux-local",
                base_url="http://127.0.0.1:8088",
                model="startlux-decision-4b",
            )
        ],
    )
    decision_client = DecisionClient(settings, http_client=client)
    try:
        result = decision_client.decide(
            "User query:\nauthz timeout\n\nCandidate memories:\n[m0] AuthZ\nCheck Istio",
            {"m0": {"type": "noul", "instructions": "Does [m0] help?"}},
        )
    finally:
        client.close()

    assert result == {"answers": {"m0": {"type": "noul", "noul": 0.91}}}
    assert captured["body"] == {
        "state": "User query:\nauthz timeout\n\nCandidate memories:\n[m0] AuthZ\nCheck Istio",
        "questions": {"m0": {"type": "noul", "instructions": "Does [m0] help?"}},
        "model": "startlux-decision-4b",
    }
