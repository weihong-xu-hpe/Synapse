from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from synapse.config import DeciderSettings
from synapse.server import decider as decider_module
from synapse.server.decider import LocalLLMDecider
from synapse.server.sampling import MemoryWriteSamplingRequest


def _request() -> MemoryWriteSamplingRequest:
    return MemoryWriteSamplingRequest(
        prompt="Decide how to write this memory.",
        title="Gateway rate limits",
        content="Rate limiting complements the gateway design.",
        node_type="transient",
        sensitivity="internal",
        query="gateway rate limits",
        similarity_threshold=0.3,
        links=(),
        candidates=(),
        candidate_nodes=(),
    )


def _completion_response(content: str, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        },
    )


def test_decide_memory_write_uses_openai_compatible_completion_payload() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content)
        return _completion_response(
            '{"action":"complement","target_node_ids":["gateway-design"],'
            '"reasoning":"The draft adds a related policy.","confidence":0.91}'
        )

    settings = DeciderSettings(
        endpoints=[
            {"base_url": "http://primary.example/v1", "model": "primary-model"},
            {"base_url": "http://fallback.example/v1", "model": "fallback-model"},
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decision = LocalLLMDecider(settings, client=client).decide_memory_write(_request())

    assert decision.action == "complement"
    assert decision.target_node_ids == ("gateway-design",)
    assert decision.confidence == 0.91
    assert captured["url"] == "http://primary.example/v1/chat/completions"
    assert captured["json"]["model"] == "primary-model"
    assert captured["json"]["messages"][0]["role"] == "user"
    assert captured["json"]["messages"][0]["content"].startswith(
        "You are deciding how Synapse should write a new memory draft."
    )
    assert captured["json"]["max_tokens"] == 200_000
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert "authorization" not in captured["headers"]


def test_endpoint_extra_bodies_are_isolated_to_selected_endpoint() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((str(request.url), json.loads(request.content)))
        return _completion_response('{"ok":true}')

    settings = DeciderSettings(
        endpoints=[
            {
                "base_url": "http://primary.example/v1",
                "model": "primary-model",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            {
                "base_url": "http://fallback.example/v1",
                "model": "fallback-model",
                "extra_body": {"top_p": 0.5},
            },
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decider = LocalLLMDecider(settings, client=client)
        decider.sample_json(prompt="{}", system_prompt="")
        decider.sample_json(prompt="{}", system_prompt="")

    assert [call[0] for call in calls] == [
        "http://primary.example/v1/chat/completions",
        "http://fallback.example/v1/chat/completions",
    ]
    assert calls[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "top_p" not in calls[0][1]
    assert calls[1][1]["top_p"] == 0.5
    assert "chat_template_kwargs" not in calls[1][1]



def test_decider_retries_endpoint_before_failing_over_and_api_key(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any], str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((str(request.url), body, request.headers.get("authorization")))
        if len(calls) <= 3:
            return _completion_response("primary unavailable", status_code=503)
        return _completion_response('{"action":"create","target_node_ids":[],"reasoning":"New memory."}')

    monkeypatch.setenv("FALLBACK_KEY", "secret-key")
    monkeypatch.setattr("synapse.server.decider.sleep", lambda _delay: None)
    monkeypatch.setattr("synapse.server.decider._retry_delay_seconds", lambda _retry: 0.0)
    settings = DeciderSettings(
        endpoints=[
            {"base_url": "http://primary.example/v1", "model": "primary-model", "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            {
                "base_url": "http://fallback.example/v1",
                "model": "fallback-model",
                "api_key_env": "FALLBACK_KEY",
            },
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decider = LocalLLMDecider(settings, client=client)
        decision = decider.decide_memory_write(_request())
        metrics = decider.call_metrics()

    assert decision.action == "create"
    primary_url = "http://primary.example/v1/chat/completions"
    fallback_url = "http://fallback.example/v1/chat/completions"
    assert [call[0] for call in calls] == [primary_url, primary_url, primary_url, fallback_url]
    assert calls[0][1]["model"] == calls[1][1]["model"] == calls[2][1]["model"] == "primary-model"
    assert calls[3][1]["model"] == "fallback-model"
    assert all(call[1]["chat_template_kwargs"] == {"enable_thinking": False} for call in calls[:3])
    assert "chat_template_kwargs" not in calls[3][1]
    assert all(call[2] is None for call in calls[:3])
    assert calls[3][2] == "Bearer secret-key"
    primary = metrics["primary-model@http://primary.example/v1"]
    secondary = metrics["fallback-model@http://fallback.example/v1"]
    assert primary["calls"] == 3 and primary["failures"] == 3 and primary["successes"] == 0
    assert secondary["calls"] == 1 and secondary["successes"] == 1 and secondary["failures"] == 0


def test_retry_delay_uses_jittered_five_and_twenty_second_bases(monkeypatch) -> None:
    bounds: list[tuple[float, float]] = []

    def uniform(lower: float, upper: float) -> float:
        bounds.append((lower, upper))
        return lower

    monkeypatch.setattr(decider_module.random, "uniform", uniform)

    assert decider_module._retry_delay_seconds(1) == 4.0
    assert decider_module._retry_delay_seconds(2) == 16.0
    assert bounds == [(4.0, 6.0), (16.0, 24.0)]


@pytest.mark.parametrize(("status_code", "should_retry"), [(429, True), (503, True), (400, False)])
def test_http_status_retry_classification(monkeypatch, status_code: int, should_retry: bool) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _completion_response("temporary failure", status_code=status_code)
        return _completion_response('{"ok":true}')

    monkeypatch.setattr("synapse.server.decider.sleep", lambda _delay: None)
    monkeypatch.setattr("synapse.server.decider._retry_delay_seconds", lambda _retry: 0.0)
    settings = DeciderSettings(endpoints=[{"base_url": "http://llm.example/v1", "model": "test-model"}])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decider = LocalLLMDecider(settings, client=client)
        if should_retry:
            assert decider.sample_json(prompt="{}", system_prompt="") == {"ok": True}
        else:
            with pytest.raises(httpx.HTTPStatusError):
                decider.sample_json(prompt="{}", system_prompt="")
        metrics = decider.call_metrics()["test-model@http://llm.example/v1"]

    assert calls == (2 if should_retry else 1)
    assert metrics["calls"] == calls
    assert metrics["failures"] == 1
    assert metrics["successes"] == (1 if should_retry else 0)


def test_invalid_json_retries_and_records_stage_metrics_and_log(monkeypatch) -> None:
    calls = 0
    retry_numbers: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, content=b"not JSON")
        if calls == 2:
            return _completion_response("not JSON")
        return _completion_response('{"ok":true}')

    def retry_delay(retry_number: int) -> float:
        retry_numbers.append(retry_number)
        return 0.0

    import synapse.server.decider as decider_module

    retry_logs: list[str] = []
    monkeypatch.setattr(decider_module.LOGGER, "warning", lambda message, *args: retry_logs.append(message % args))

    monkeypatch.setattr("synapse.server.decider.sleep", lambda _delay: None)
    monkeypatch.setattr("synapse.server.decider._retry_delay_seconds", retry_delay)
    settings = DeciderSettings(endpoints=[{"base_url": "http://llm.example/v1", "model": "test-model"}])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decider = LocalLLMDecider(settings, client=client)
        with decider.call_context(transcript_id="mem_session_retry", stage="extract"):
            assert decider.sample_json(prompt="{}", system_prompt="") == {"ok": True}
        metrics = decider.call_metrics(transcript_id="mem_session_retry")

    endpoint_metrics = metrics["test-model@http://llm.example/v1"]
    stage_metrics = endpoint_metrics["stages"]["extract"]
    assert calls == 3
    assert retry_numbers == [1, 2]
    assert endpoint_metrics["calls"] == 3
    assert endpoint_metrics["failures"] == 2 and endpoint_metrics["successes"] == 1
    assert stage_metrics["calls"] == 3
    assert stage_metrics["failures"] == 2 and stage_metrics["successes"] == 1
    assert len(retry_logs) == 2
    assert all("http://llm.example/v1" in message for message in retry_logs)
    assert all("stage=extract" in message for message in retry_logs)
    assert "attempt=2/3" in retry_logs[0]
    assert "error_class=_InvalidCompletionResponse" in retry_logs[0]
    assert "attempt=3/3" in retry_logs[1]
    assert "error_class=ValueError" in retry_logs[1]


def test_transport_timeout_retries_and_configuration_error_does_not(monkeypatch) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("request timed out", request=request)
        return _completion_response('{"ok":true}')

    monkeypatch.setattr("synapse.server.decider.sleep", lambda _delay: None)
    monkeypatch.setattr("synapse.server.decider._retry_delay_seconds", lambda _retry: 0.0)
    settings = DeciderSettings(endpoints=[{"base_url": "http://llm.example/v1", "model": "test-model"}])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decider = LocalLLMDecider(settings, client=client)
        assert decider.sample_json(prompt="{}", system_prompt="") == {"ok": True}
        metrics = decider.call_metrics()["test-model@http://llm.example/v1"]

    assert calls == 2
    assert metrics["calls"] == 2 and metrics["failures"] == 1 and metrics["successes"] == 1

    with pytest.raises(ValueError, match="extra_body cannot override completion fields"):
        DeciderSettings(
            endpoints=[
                {"base_url": "http://llm.example/v1", "model": "test-model", "extra_body": {"model": "override"}}
            ]
        )


def test_sample_json_extracts_prose_and_markdown_fenced_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _completion_response('Here is the result:\n```json\n{"keep":true}\n```')

    settings = DeciderSettings(
        endpoints=[
            {"base_url": "http://primary.example/v1", "model": "primary-model"},
            {"base_url": "http://fallback.example/v1", "model": "fallback-model"},
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = LocalLLMDecider(settings, client=client).sample_json(
            prompt="Return a JSON object.",
            system_prompt="You are a JSON assistant.",
            max_tokens=42,
        )

    assert result == {"keep": True}

def test_endpoint_pool_caps_each_backend_and_attributes_metrics() -> None:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    endpoints = ("http://one.example/v1", "http://two.example/v1")
    lock = threading.Lock()
    active = {endpoint: 0 for endpoint in endpoints}
    maximum = {endpoint: 0 for endpoint in endpoints}
    calls = {endpoint: 0 for endpoint in endpoints}

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint = f"{request.url.scheme}://{request.url.host}{request.url.path.split('/chat/completions')[0]}"
        with lock:
            active[endpoint] += 1
            calls[endpoint] += 1
            maximum[endpoint] = max(maximum[endpoint], active[endpoint])
        time.sleep(0.04)
        with lock:
            active[endpoint] -= 1
        return _completion_response('{"ok":true}')

    settings = DeciderSettings(
        endpoints=[
            {"base_url": endpoints[0], "model": "one"},
            {"base_url": endpoints[1], "model": "two"},
        ],
        endpoint_max_concurrency=2,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decider = LocalLLMDecider(settings, client=client)

        def call(_: int) -> dict[str, Any]:
            with decider.call_context(transcript_id="mem_session_parallel", stage="extract"):
                return decider.sample_json(prompt="{}", system_prompt="", max_tokens=20)

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert all(result == {"ok": True} for result in pool.map(call, range(8)))

        metrics = decider.call_metrics(transcript_id="mem_session_parallel")

    assert calls[endpoints[0]] > 0 and calls[endpoints[1]] > 0
    assert maximum[endpoints[0]] <= 2
    assert maximum[endpoints[1]] <= 2
    assert sum(stats["calls"] for stats in metrics.values()) == 8
    assert all(stats["latency_total_ms"] > 0 for stats in metrics.values())


def test_call_context_overrides_request_timeout_by_stage(monkeypatch) -> None:
    settings = DeciderSettings(
        endpoints=[{"base_url": "http://llm.example/v1", "model": "test-model"}],
        timeout_seconds=300,
    )
    decider = LocalLLMDecider(settings)
    timeouts: list[int] = []

    def fake_post(**kwargs):
        timeouts.append(kwargs["timeout_seconds"])
        return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    monkeypatch.setattr(decider, "_post_completion", fake_post)
    with decider.call_context(transcript_id="mem_session_timeout", stage="extract", timeout_seconds=1800):
        assert decider.sample_json(prompt="{}", system_prompt="") == {"ok": True}
    with decider.call_context(transcript_id="mem_session_timeout", stage="write_decider", timeout_seconds=300):
        assert decider.sample_json(prompt="{}", system_prompt="") == {"ok": True}

    assert timeouts == [1800, 300]
