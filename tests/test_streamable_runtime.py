from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synapse.config import load_config
from synapse.server import (
    SynapseServiceError,
    StreamableSessionManager,
    StreamableToolOrchestrator,
    create_streamable_orchestrator,
    create_streamable_runtime,
    create_streamable_session_manager,
)
from synapse.utils.runtime import bootstrap_runtime_directories


def _extract_tool_payload(result: dict[str, Any]) -> dict[str, Any]:
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured

    for item in result.get("content", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "json" and isinstance(item.get("json"), dict):
            return item["json"]

    raise AssertionError("Expected structured tool payload")


def write_config(base_dir: Path) -> Path:
    config_path = base_dir / "config.toml"
    config_path.write_text(
        """
[server]
host = "127.0.0.1"
port = 8765
cors_allowed_origins = ["http://localhost:3000"]
auth_token = ""

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

[logging]
log_dir = "./.synapse/.logs"
""".strip(),
        encoding="utf-8",
    )
    return config_path


def test_streamable_session_manager_creates_tracks_and_closes_sessions() -> None:
    manager = create_streamable_session_manager()

    session = manager.create_session(client_id="test-client")

    assert session.session_id.startswith("streamable-")
    assert session.client_id == "test-client"
    assert manager.get_session(session.session_id) is session
    assert manager.list_active_sessions() == [session.session_id]

    closed = manager.close_session(session.session_id)

    assert closed["closed"] is True
    assert closed["cancelled_requests"] == 0
    assert manager.list_active_sessions() == []


def test_streamable_session_manager_tracks_capabilities_and_initialization() -> None:
    manager = StreamableSessionManager()
    session = manager.create_session()

    manager.negotiate_capabilities(
        session.session_id,
        capabilities={"sampling": {}},
        protocol_version="2025-11-25",
    )
    manager.mark_initialized(session.session_id)

    refreshed = manager.get_session(session.session_id)
    assert refreshed.protocol_version == "2025-11-25"
    assert refreshed.supports_sampling is True
    assert refreshed.initialized is True


def test_streamable_orchestrator_lists_public_tools(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    orchestrator = create_streamable_orchestrator(config, runtime_paths=runtime_paths)

    names = [tool["name"] for tool in orchestrator.list_tools()]
    assert names == [
        "run_dreamer",
        "search_memory",
        "write_memory",
    ]


def test_streamable_orchestrator_routes_retrieval_tool_calls(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    orchestrator = create_streamable_orchestrator(config, runtime_paths=runtime_paths)
    session = create_streamable_session_manager().create_session()

    result = orchestrator.invoke_tool(session, "search_memory", {"query": "gateway", "top_k": 3})

    assert "content" in result
    assert result["content"][0]["type"] == "text"
    assert isinstance(result.get("structuredContent"), dict)
    payload = _extract_tool_payload(result)
    assert payload["query"] == "gateway"
    assert payload["results"] == []


def test_streamable_orchestrator_returns_sampling_unavailable_error_without_host_support(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    orchestrator = create_streamable_orchestrator(config, runtime_paths=runtime_paths)
    session = create_streamable_session_manager().create_session()

    result = orchestrator.invoke_tool(
        session,
        "write_memory",
        {
            "title": "Gateway Design",
            "content": "Rate limiting complements the gateway design.",
        },
    )

    assert result["error"]["code"] == "SAMPLING_UNAVAILABLE"
    assert result["error"]["details"]["required_capability"] == "sampling"
    assert result["error"]["details"]["architecture_doc"] == "docs/design/streamable-mcp-single-path-architecture.md"


def test_streamable_runtime_factory_reuses_native_runtime_components(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    runtime = create_streamable_runtime(config, runtime_paths=runtime_paths)

    orchestrator = runtime.create_orchestrator()
    session_manager = runtime.create_session_manager()

    assert isinstance(orchestrator, StreamableToolOrchestrator)
    assert isinstance(session_manager, StreamableSessionManager)
    assert runtime.execution_layer is orchestrator.service


def test_streamable_session_wraps_native_mcp_session_state() -> None:
    manager = create_streamable_session_manager()
    session = manager.create_session()

    session.transport_session.set_capabilities({"sampling": {}})
    session.transport_session.mark_initialized()

    assert session.supports_sampling is True
    assert session.initialized is True
    assert session.client_capabilities == {"sampling": {}}


def test_streamable_session_matches_pending_sampling_responses() -> None:
    session = create_streamable_session_manager().create_session()
    request_id = session.create_pending_sampling_request({"messages": []}, event_sink=lambda _message: None)

    accepted = session.accept_sampling_response(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"role": "assistant", "content": {"type": "text", "text": "{}"}},
        }
    )

    assert accepted == {"accepted": True, "request_id": request_id}
    response = session.wait_for_sampling_response(request_id, timeout_seconds=0.01)
    assert response["id"] == request_id


def test_streamable_session_rejects_duplicate_sampling_responses() -> None:
    session = create_streamable_session_manager().create_session()
    request_id = session.create_pending_sampling_request({"messages": []}, event_sink=lambda _message: None)
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"role": "assistant", "content": {"type": "text", "text": "{}"}},
    }

    session.accept_sampling_response(payload)

    with pytest.raises(SynapseServiceError, match="already received"):
        session.accept_sampling_response(payload)


def test_streamable_session_times_out_pending_sampling_requests() -> None:
    session = create_streamable_session_manager().create_session()
    request_id = session.create_pending_sampling_request({"messages": []}, event_sink=lambda _message: None)

    with pytest.raises(SynapseServiceError, match="Timed out waiting") as exc_info:
        session.wait_for_sampling_response(request_id, timeout_seconds=0.01)

    assert exc_info.value.code == "SAMPLING_TIMEOUT"


def test_streamable_session_close_cancels_pending_sampling_requests() -> None:
    manager = create_streamable_session_manager()
    session = manager.create_session()
    session.create_pending_sampling_request({"messages": []}, event_sink=lambda _message: None)

    closed = manager.close_session(session.session_id)

    assert closed["cancelled_requests"] == 1
