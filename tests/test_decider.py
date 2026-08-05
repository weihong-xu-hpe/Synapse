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
        base_url="http://primary.example/v1",
        model="primary-model",
        fallback_base_url="http://fallback.example/v1",
        fallback_model="fallback-model",
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
    assert captured["json"]["max_tokens"] == 600
    assert captured["json"]["temperature"] == 0.1
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert "authorization" not in captured["headers"]


def test_decider_falls_back_to_secondary_endpoint_and_api_key(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any], str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((str(request.url), body, request.headers.get("authorization")))
        if len(calls) == 1:
            return _completion_response("primary unavailable", status_code=503)
        return _completion_response('{"action":"create","target_node_ids":[],"reasoning":"New memory."}')

    monkeypatch.setenv("FALLBACK_KEY", "secret-key")
    settings = DeciderSettings(
        base_url="http://primary.example/v1",
        model="primary-model",
        fallback_base_url="http://fallback.example/v1",
        fallback_model="fallback-model",
        fallback_api_key_env="FALLBACK_KEY",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decision = LocalLLMDecider(settings, client=client).decide_memory_write(_request())

    assert decision.action == "create"
    assert [call[0] for call in calls] == [
        "http://primary.example/v1/chat/completions",
        "http://fallback.example/v1/chat/completions",
    ]
    assert calls[0][1]["model"] == "primary-model"
    assert calls[1][1]["model"] == "fallback-model"
    assert calls[0][2] is None
    assert calls[1][2] == "Bearer secret-key"


def test_sample_json_extracts_prose_and_markdown_fenced_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _completion_response('Here is the result:\n```json\n{"keep":true}\n```')

    settings = DeciderSettings(
        base_url="http://primary.example/v1",
        fallback_base_url="http://fallback.example/v1",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = LocalLLMDecider(settings, client=client).sample_json(
            prompt="Return a JSON object.",
            system_prompt="You are a JSON assistant.",
            max_tokens=42,
        )

    assert result == {"keep": True}
