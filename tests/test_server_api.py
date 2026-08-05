from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import socket
from threading import Thread
from time import sleep
from typing import Any, cast

import httpx
import uvicorn
from fastapi.testclient import TestClient

import synapse.server.streamable as streamable_module
from synapse.config import load_config
from synapse.server import (
    STREAMABLE_ARCHITECTURE_DOC,
    STREAMABLE_RUNTIME_MODE,
    StreamableRuntime,
    create_app,
    create_streamable_app,
    create_streamable_runtime,
    run_streamable_server,
)
from synapse.server.service import SynapseServerService
from synapse.server.sampling import MemoryWriteSamplingDecision
from synapse.utils.runtime import bootstrap_runtime_directories


AUTH_TOKEN = "super-secret-token"
SESSION_HEADER = "mcp-session-id"


class FakeSamplingClient:
    name = "fake-sampler"

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
        _ = (system_prompt, max_tokens, model_hints)
        raise AssertionError(f"Unexpected sampling prompt: {prompt}")

    def decide_memory_write(self, request):
        candidate_ids = [candidate.node_id for candidate in request.candidates]
        if candidate_ids:
            return MemoryWriteSamplingDecision(
                action="complement",
                target_node_ids=(candidate_ids[0],),
                reasoning="The draft complements the closest existing node.",
                confidence=0.93,
            )
        return MemoryWriteSamplingDecision(
            action="create",
            target_node_ids=(),
            reasoning="No overlapping node was found.",
            confidence=0.93,
        )


def write_config(base_dir: Path, *, auth_token: str = "") -> Path:
    config_path = base_dir / "config.toml"
    config_path.write_text(
        f"""
[server]
host = "127.0.0.1"
port = 8765
cors_allowed_origins = ["http://localhost:3000"]
auth_token = "{auth_token}"

[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "builtin"
model = "bge-m3"
dimension = 1024
timeout_seconds = 1

[reranker]
provider = "builtin"
model = "bge-reranker-v2-m3"
max_candidates = 9
timeout_seconds = 1

[decider]
provider = "mcp_sampling"

[logging]
log_dir = "./.synapse/.logs"
""".strip(),
        encoding="utf-8",
    )
    return config_path


def _next_sse_json(lines) -> dict[str, Any]:
    for line in lines:
        if not line or not line.startswith("data: "):
            continue
        return json.loads(line.removeprefix("data: "))
    raise AssertionError("Expected an SSE data event")


def _extract_tool_payload(response_payload: dict[str, Any]) -> dict[str, Any]:
    result = response_payload["result"]
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured

    for item in result.get("content", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "json" and isinstance(item.get("json"), dict):
            return item["json"]
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            return json.loads(item["text"])

    raise AssertionError("Expected structured tool payload")


def _initialize_session(
    client: Any,
    *,
    headers: dict[str, str] | None = None,
    supports_sampling: bool = True,
) -> str:
    request_headers = dict(headers or {})
    capabilities = {"sampling": {}} if supports_sampling else {}
    initialize_response = client.post(
        "/mcp",
        headers=request_headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": capabilities,
                "clientInfo": {"name": "streamable-test-client", "version": "1.0.0"},
            },
        },
    )
    assert initialize_response.status_code == 200
    session_id = initialize_response.headers[SESSION_HEADER]
    initialized_response = client.post(
        "/mcp",
        headers={**request_headers, SESSION_HEADER: session_id},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert initialized_response.status_code == 202
    return session_id


@contextmanager
def _run_live_server(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )
    thread = Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    last_error: Exception | None = None
    for _ in range(100):
        try:
            response = httpx.get(f"{base_url}/", timeout=0.2)
            if response.status_code == 200:
                break
        except (httpx.HTTPError, OSError) as exc:  # pragma: no cover - startup race protection
            last_error = exc
        sleep(0.05)
    else:  # pragma: no cover - defensive startup guard
        server.should_exit = True
        thread.join(timeout=5)
        raise AssertionError(f"Live Streamable test server did not start: {last_error}")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_streamable_runtime_factory_creates_native_http_runtime(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    runtime = create_streamable_runtime(config, runtime_paths=runtime_paths)

    assert isinstance(runtime, StreamableRuntime)

    app = create_streamable_app(config, runtime_paths=runtime_paths)
    assert app.state.streamable_runtime_mode == STREAMABLE_RUNTIME_MODE
    assert app.state.streamable_architecture_doc == STREAMABLE_ARCHITECTURE_DOC
    assert isinstance(app.state.streamable_runtime, StreamableRuntime)

    with TestClient(app) as client:
        root_response = client.get("/")
        assert root_response.status_code == 200
        payload = root_response.json()
        assert payload["mcp"] == "/mcp"
        assert payload["transport"] == "streamable-http"
        assert payload["session_header"] == SESSION_HEADER


def test_streamable_runtime_lifespan_starts_and_stops_dreamer_scheduler(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []

    class FakeDreamerScheduler:
        def __init__(self, config, *, runtime_paths, logger, sampling_client) -> None:
            events.append("created")
            self.config = config
            self.runtime_paths = runtime_paths
            self.logger = logger
            self.sampling_client = sampling_client

        def start(self) -> None:
            events.append("started")

        def stop(self) -> None:
            events.append("stopped")

    monkeypatch.setattr(streamable_module, "DreamerScheduler", FakeDreamerScheduler)
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_streamable_app(config, runtime_paths=runtime_paths)

    assert events == ["created"]
    with TestClient(app):
        assert events == ["created", "started"]
    assert events == ["created", "started", "stopped"]


def test_run_streamable_server_uses_runtime_factory_and_runner(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    captured: dict[str, object] = {}

    def fake_runner(app, *, host: str, port: int, log_level: str) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    run_streamable_server(
        config,
        runtime_paths=runtime_paths,
        log_level="warning",
        uvicorn_runner=fake_runner,
    )

    app = cast(Any, captured["app"])
    assert app.state.streamable_runtime_mode == STREAMABLE_RUNTIME_MODE
    assert app.state.streamable_architecture_doc == STREAMABLE_ARCHITECTURE_DOC
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["log_level"] == "warning"


def test_streamable_runtime_exposes_native_session_manager_and_orchestrator(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    runtime = create_streamable_runtime(config, runtime_paths=runtime_paths)

    session_manager = runtime.create_session_manager()
    orchestrator = runtime.create_orchestrator()

    assert runtime.execution_layer is orchestrator.service
    assert runtime.create_session_manager() is session_manager
    assert runtime.create_orchestrator() is orchestrator

    app = create_streamable_app(config, runtime_paths=runtime_paths)
    assert app.state.streamable_session_manager is not None
    assert app.state.streamable_orchestrator is not None


def test_streamable_http_auth_token_is_enforced_for_mcp_only(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, auth_token=AUTH_TOKEN))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with TestClient(app) as client:
        root_response = client.get("/")
        assert root_response.status_code == 200

        unauthorized = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["code"] == "UNAUTHORIZED"

        session_id = _initialize_session(client, headers={"Authorization": f"Bearer {AUTH_TOKEN}"})
        authorized = client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}", SESSION_HEADER: session_id},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        assert authorized.status_code == 200
        tool_names = {tool["name"] for tool in authorized.json()["result"]["tools"]}
        assert tool_names == {
            "search_memory",
            "write_memory",
        }


def test_streamable_non_initialize_requests_require_session_header(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with TestClient(app) as client:
        response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "MCP_SESSION_REQUIRED"


def test_streamable_initialize_returns_session_header_and_keeps_state_isolated(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with TestClient(app) as client:
        session_a = _initialize_session(client)

        second_initialize = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "plain-client", "version": "1.0.0"},
                },
            },
        )
        session_b = second_initialize.headers[SESSION_HEADER]
        assert session_a != session_b

        manager = app.state.streamable_session_manager
        assert manager.get_session(session_a).supports_sampling is True
        assert manager.get_session(session_a).initialized is True
        assert manager.get_session(session_b).supports_sampling is False
        assert manager.get_session(session_b).initialized is False


def test_streamable_search_memory_handles_date_like_query_tokens(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths)
    app = create_app(config, runtime_paths=runtime_paths)

    created = service.integrate_knowledge(
        title="HTTP MCP Full Test",
        content="Agent perspective MCP full test sentinel 2026-03-15.",
        action="create",
    )
    created_id = created["node"]["id"]

    with TestClient(app) as client:
        session_id = _initialize_session(client)
        response = client.post(
            "/mcp",
            headers={SESSION_HEADER: session_id},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search_memory",
                    "arguments": {
                        "query": "agent perspective mcp full test sentinel 2026-03-15",
                        "top_k": 3,
                    },
                },
            },
        )

        assert response.status_code == 200
        payload = _extract_tool_payload(response.json())
        results = payload["results"]
        assert [item["node_id"] for item in results] == [created_id]


def test_search_existing_nodes_reuses_retrieval_candidate_core(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths)

    created = service.integrate_knowledge(
        title="Session Correlation Rule",
        content="Streamable MCP sampling requires session correlation and timeout handling.",
        action="create",
    )
    created_id = created["node"]["id"]

    query = "streamable sampling session correlation timeout handling"
    search_payload = service.search_memory(query, top_k=3)
    existing_payload = service.search_existing_nodes(query, similarity_threshold=0.3)

    assert [item["node_id"] for item in search_payload["results"]] == [created_id]
    assert [item["node_id"] for item in existing_payload["matches"]] == [created_id]


def test_normalize_candidate_score_does_not_compress_reranker_scores() -> None:
    """Regression: x/(1+x) mapping collapsed [0,1] -> [0,0.5], filtering nearly all candidates."""
    normalize = SynapseServerService._normalize_candidate_score
    # Reranker outputs in [0,1] must pass through unchanged (within rounding).
    assert normalize(0.0) == 0.0
    assert normalize(0.3) == 0.3
    assert normalize(0.5) == 0.5
    assert normalize(0.9) == 0.9
    assert normalize(1.0) == 1.0
    # Out-of-range values are clamped, not compressed.
    assert normalize(1.5) == 1.0
    assert normalize(-0.2) == 0.0


def test_low_reranker_score_candidate_passes_default_threshold(tmp_path: Path) -> None:
    """A reranker score of 0.4 must survive the default 0.3 threshold (previously 0.5 filtered it)."""
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths)

    created = service.integrate_knowledge(
        title="Asyncio Gather Pattern",
        content="Use asyncio.gather to run coroutines concurrently and collect results.",
        action="create",
    )
    created_id = created["node"]["id"]

    # Default threshold is now 0.3; a partial-overlap query should still surface the node.
    payload = service.search_existing_nodes("run coroutines concurrently")
    matches = [item["node_id"] for item in payload["matches"]]
    assert created_id in matches


def test_write_memory_warns_for_unstructured_persistent_content(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(
        config,
        runtime_paths=runtime_paths,
        sampling_client=FakeSamplingClient(),
    )

    result = service.write_memory(
        title="Persistent Note",
        content="A short persistent note without structured sections.",
        node_type="persistent",
    )

    assert result["warnings"] == [
        {
            "code": "low_structure",
            "message": "Persistent memory has no ## sections; consider OKF format.",
        }
    ]


def test_write_memory_does_not_warn_for_structured_persistent_content(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(
        config,
        runtime_paths=runtime_paths,
        sampling_client=FakeSamplingClient(),
    )

    result = service.write_memory(
        title="Structured Persistent Note",
        content="## Context\nA durable context.\n\n## Decision\nKeep this policy.",
        node_type="persistent",
    )

    assert result["warnings"] == []


def test_write_memory_does_not_warn_for_unstructured_transient_content(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(
        config,
        runtime_paths=runtime_paths,
        sampling_client=FakeSamplingClient(),
    )

    result = service.write_memory(
        title="Transient Note",
        content="A short transient note without structured sections.",
        node_type="transient",
    )

    assert result["warnings"] == []


def test_streamable_sampling_unavailable_errors_point_to_active_architecture_doc(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with TestClient(app) as client:
        session_id = _initialize_session(client, supports_sampling=False)
        response = client.post(
            "/mcp",
            headers={SESSION_HEADER: session_id},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "write_memory",
                    "arguments": {
                        "title": "Gateway Rate Limits",
                        "content": "Rate limiting complements the gateway design.",
                    },
                },
            },
        )
        assert response.status_code == 200
        error = response.json()["error"]
        assert error["data"]["code"] == "SAMPLING_UNAVAILABLE"
        assert "sampling-capable MCP host/client" in error["message"]
        assert error["data"]["details"]["required_capability"] == "sampling"
        assert error["data"]["details"]["architecture_doc"] == STREAMABLE_ARCHITECTURE_DOC


def test_streamable_sampling_tools_execute_when_sampling_client_is_injected(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    service = SynapseServerService(config, runtime_paths=runtime_paths)

    base = service.integrate_knowledge(
        title="Gateway Design",
        content="Gateway design references rate limiting.",
        action="create",
    )
    base_id = base["node"]["id"]

    with TestClient(app) as client:
        session_id = _initialize_session(client)
        response = client.post(
            "/mcp",
            headers={SESSION_HEADER: session_id},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "write_memory",
                    "arguments": {
                        "title": "Gateway Rate Limits",
                        "content": "Rate limiting complements the gateway design.",
                        "query_hint": "Gateway design rate limiting",
                    },
                },
            },
        )

        assert response.status_code == 200
        payload = _extract_tool_payload(response.json())
        assert payload["decision"]["action"] == "complement"
        assert payload["decision"]["target_node_ids"] == [base_id]
        assert payload["execution"]["executed"] is True
        assert payload["execution"]["tool"] == "integrate_knowledge"


def test_streamable_low_level_canonical_tools_are_not_exposed_over_public_mcp(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with TestClient(app) as client:
        session_id = _initialize_session(client)

        tools_response = client.post(
            "/mcp",
            headers={SESSION_HEADER: session_id},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        assert tools_response.status_code == 200
        tool_names = {tool["name"] for tool in tools_response.json()["result"]["tools"]}
        assert "integrate_knowledge" not in tool_names
        assert "search_existing_nodes" not in tool_names
        assert "update_node_status" not in tool_names

        call_response = client.post(
            "/mcp",
            headers={SESSION_HEADER: session_id},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "integrate_knowledge",
                    "arguments": {
                        "title": "Should Not Be Exposed",
                        "content": "This low-level tool must remain internal.",
                    },
                },
            },
        )
        assert call_response.status_code == 200
        error = call_response.json()["error"]
        assert error["data"]["code"] == "TOOL_NOT_FOUND"


def test_streamable_sampling_query_hint_enhances_candidate_lookup_without_overriding_base_query(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    service = SynapseServerService(config, runtime_paths=runtime_paths)

    base = service.integrate_knowledge(
        title="Streamable Session Correlation",
        content="Streamable MCP sampling requires session correlation and timeout handling.",
        action="create",
    )
    base_id = base["node"]["id"]

    with TestClient(app) as client:
        session_id = _initialize_session(client)
        response = client.post(
            "/mcp",
            headers={SESSION_HEADER: session_id},
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "write_memory",
                    "arguments": {
                        "title": "Duplicate Response Guardrails",
                        "content": (
                            "Duplicate responses complement streamable MCP session correlation "
                            "and timeout handling."
                        ),
                        "query_hint": "duplicate response handling",
                    },
                },
            },
        )

        assert response.status_code == 200
        payload = _extract_tool_payload(response.json())
        assert payload["decision"]["action"] == "complement"
        assert payload["decision"]["target_node_ids"] == [base_id]
        assert payload["evidence"]["candidate_count"] >= 1
        assert "duplicate response handling" in payload["evidence"]["queries"]
        assert payload["execution"]["executed"] is True


def test_streamable_sampling_round_trip_completes_over_post_sse_stream(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with _run_live_server(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as control_client:
            session_id = _initialize_session(control_client)
            with httpx.Client(base_url=base_url, timeout=35.0) as stream_client:
                with stream_client.stream(
                    "POST",
                    "/mcp",
                    headers={SESSION_HEADER: session_id, "Accept": "text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "write_memory",
                            "arguments": {
                                "title": "Gateway Streaming Policy",
                                "content": "Streamable HTTP sampling should use one coherent session.",
                            },
                        },
                    },
                ) as response:
                    assert response.status_code == 200
                    assert response.headers["content-type"].startswith("text/event-stream")

                    lines = response.iter_lines()
                    sampling_request = _next_sse_json(lines)
                    assert sampling_request["method"] == "sampling/createMessage"
                    assert sampling_request["params"]["messages"][0]["content"]["text"].startswith(
                        "You are deciding how Synapse should write a new memory draft."
                    )

                    sampling_response = control_client.post(
                        "/mcp",
                        headers={SESSION_HEADER: session_id},
                        json={
                            "jsonrpc": "2.0",
                            "id": sampling_request["id"],
                            "result": {
                                "role": "assistant",
                                "content": {
                                    "type": "text",
                                    "text": '{"action":"create","target_node_ids":[],"reasoning":"No durable overlap exists yet.","confidence":0.93}',
                                },
                                "model": "test-model",
                                "stopReason": "endTurn",
                            },
                        },
                    )
                    assert sampling_response.status_code == 202

                    final_response = _next_sse_json(lines)
                    assert final_response["id"] == 2
                    payload = _extract_tool_payload(final_response)
                    assert payload["decision"]["action"] == "create"
                    assert payload["execution"]["executed"] is True
                    assert payload["execution"]["tool"] == "integrate_knowledge"


def test_streamable_sampling_round_trip_completes_over_session_event_stream(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with _run_live_server(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as control_client:
            session_id = _initialize_session(control_client)
            with httpx.Client(base_url=base_url, timeout=35.0) as stream_client:
                with stream_client.stream("GET", "/mcp", headers={SESSION_HEADER: session_id}) as event_stream_response:
                    assert event_stream_response.status_code == 200
                    assert event_stream_response.headers["content-type"].startswith("text/event-stream")

                    tool_response = control_client.post(
                        "/mcp",
                        headers={SESSION_HEADER: session_id},
                        json={
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": "write_memory",
                                "arguments": {
                                    "title": "Gateway Session Streaming Policy",
                                    "content": "Session event streams should support server-driven sampling.",
                                },
                            },
                        },
                    )
                    assert tool_response.status_code == 202

                    lines = event_stream_response.iter_lines()
                    sampling_request = _next_sse_json(lines)
                    assert sampling_request["method"] == "sampling/createMessage"

                    sampling_response = control_client.post(
                        "/mcp",
                        headers={SESSION_HEADER: session_id},
                        json={
                            "jsonrpc": "2.0",
                            "id": sampling_request["id"],
                            "result": {
                                "role": "assistant",
                                "content": {
                                    "type": "text",
                                    "text": '{"action":"create","target_node_ids":[],"reasoning":"No durable overlap exists yet.","confidence":0.91}',
                                },
                                "model": "test-model",
                                "stopReason": "endTurn",
                            },
                        },
                    )
                    assert sampling_response.status_code == 202

                    final_response = _next_sse_json(lines)
                    assert final_response["id"] == 2
                    payload = _extract_tool_payload(final_response)
                    assert payload["decision"]["action"] == "create"
                    assert payload["execution"]["executed"] is True

                    session = app.state.streamable_session_manager.get_session(session_id)
                    session.publish_server_message(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/message",
                            "params": {"text": "session stream remains open"},
                        }
                    )
                    keepalive_event = _next_sse_json(lines)
                    assert keepalive_event["method"] == "notifications/message"
                    assert keepalive_event["params"]["text"] == "session stream remains open"

                    close_response = control_client.delete("/mcp", headers={SESSION_HEADER: session_id})
                    assert close_response.status_code == 200


def test_streamable_initialize_rejects_existing_session_header(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={SESSION_HEADER: "streamable-existing"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"sampling": {}},
                    "clientInfo": {"name": "bad-client", "version": "1.0.0"},
                },
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_streamable_duplicate_sampling_response_is_rejected(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with _run_live_server(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as control_client:
            session_id = _initialize_session(control_client)
            with httpx.Client(base_url=base_url, timeout=35.0) as stream_client:
                with stream_client.stream(
                    "POST",
                    "/mcp",
                    headers={SESSION_HEADER: session_id, "Accept": "text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "write_memory",
                            "arguments": {
                                "title": "Gateway Cache Policy",
                                "content": "Cache invalidation belongs with gateway policy notes.",
                            },
                        },
                    },
                ) as response:
                    sampling_request = _next_sse_json(response.iter_lines())
                    payload = {
                        "jsonrpc": "2.0",
                        "id": sampling_request["id"],
                        "result": {
                            "role": "assistant",
                            "content": {
                                "type": "text",
                                "text": '{"action":"create","target_node_ids":[],"reasoning":"Fresh note.","confidence":0.9}',
                            },
                        },
                    }
                    accepted = control_client.post("/mcp", headers={SESSION_HEADER: session_id}, json=payload)
                    duplicate = control_client.post("/mcp", headers={SESSION_HEADER: session_id}, json=payload)

                    assert accepted.status_code == 202
                    assert duplicate.status_code == 409
                    assert duplicate.json()["error"]["code"] == "MCP_DUPLICATE_RESPONSE"


def test_streamable_get_endpoint_streams_server_messages(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with _run_live_server(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            missing = client.get("/mcp")
            assert missing.status_code == 400
            assert missing.json()["error"]["code"] == "MCP_SESSION_REQUIRED"

            session_id = _initialize_session(client)
            session = app.state.streamable_session_manager.get_session(session_id)
            with client.stream("GET", "/mcp", headers={SESSION_HEADER: session_id}) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                session.publish_server_message(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/message",
                        "params": {"text": "hello from Synapse"},
                    }
                )
                event = _next_sse_json(response.iter_lines())
                assert event["method"] == "notifications/message"
                assert event["params"]["text"] == "hello from Synapse"
                closed = client.delete("/mcp", headers={SESSION_HEADER: session_id})
                assert closed.status_code == 200


def test_streamable_session_close_requires_header_and_closes_active_session(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    app = create_app(config, runtime_paths=runtime_paths)

    with TestClient(app) as client:
        missing = client.delete("/mcp")
        assert missing.status_code == 400
        assert missing.json()["error"]["code"] == "MCP_SESSION_REQUIRED"

        session_id = _initialize_session(client)
        closed = client.delete("/mcp", headers={SESSION_HEADER: session_id})
        assert closed.status_code == 200
        assert closed.json()["closed"] is True
        assert closed.json()["session_id"] == session_id
