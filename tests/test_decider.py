from __future__ import annotations

import json
from typing import Any

import httpx

from synapse.config import DeciderSettings
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


def test_decider_fails_over_to_second_endpoint_and_api_key(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any], str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((str(request.url), body, request.headers.get("authorization")))
        if len(calls) == 1:
            return _completion_response("primary unavailable", status_code=503)
        return _completion_response('{"action":"create","target_node_ids":[],"reasoning":"New memory."}')

    monkeypatch.setenv("FALLBACK_KEY", "secret-key")
    settings = DeciderSettings(
        endpoints=[
            {"base_url": "http://primary.example/v1", "model": "primary-model"},
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
    assert [call[0] for call in calls] == [
        "http://primary.example/v1/chat/completions",
        "http://fallback.example/v1/chat/completions",
    ]
    assert calls[0][1]["model"] == "primary-model"
    assert calls[1][1]["model"] == "fallback-model"
    assert calls[0][2] is None
    assert calls[1][2] == "Bearer secret-key"
    primary = metrics["primary-model@http://primary.example/v1"]
    secondary = metrics["fallback-model@http://fallback.example/v1"]
    assert primary["failures"] == 1 and primary["successes"] == 0
    assert secondary["successes"] == 1 and secondary["failures"] == 0


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
