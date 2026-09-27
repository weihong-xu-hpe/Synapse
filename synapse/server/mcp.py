"""Minimal MCP-compatible tool registry for Synapse's Streamable path."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import count
import json
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from synapse import __version__
from synapse.server.sampling import SamplingClient
from synapse.server.schemas import (
    SearchMemoryToolRequest,
    WriteMemoryRequest,
)
from synapse.server.service import SynapseServerService, SynapseServiceError


_SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-03-26")
_DEFAULT_PROTOCOL_VERSION = _SUPPORTED_PROTOCOL_VERSIONS[0]


@dataclass(slots=True)
class MCPSession:
    """Transport-agnostic session state for a single MCP client connection."""

    session_id: str
    client_capabilities: dict[str, Any] = field(default_factory=dict)
    client_initialized: bool = False
    protocol_version: str = _DEFAULT_PROTOCOL_VERSION
    _request_id_counter: count = field(default_factory=lambda: count(10_000))

    def allocate_request_id(self) -> int:
        return next(self._request_id_counter)

    def set_capabilities(self, capabilities: dict[str, Any] | None) -> None:
        self.client_capabilities = capabilities if isinstance(capabilities, dict) else {}

    def mark_initialized(self) -> None:
        self.client_initialized = True

    @property
    def client_supports_sampling(self) -> bool:
        sampling = self.client_capabilities.get("sampling")
        return isinstance(sampling, dict)


@dataclass(slots=True, frozen=True)
class MCPToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., dict[str, object]]

    def to_payload(self) -> dict[str, object]:
        schema = self.input_model.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": schema,
            "parameters": schema.get("properties", {}),
        }


class SynapseMCPServer:
    """Small MCP-like server for the Streamable MCP runtime."""

    def __init__(
        self,
        config,
        *,
        runtime_paths=None,
        logger=None,
        service: SynapseServerService | None = None,
        sampling_client: SamplingClient | None = None,
    ) -> None:
        self.service = service or SynapseServerService(
            config,
            runtime_paths=runtime_paths,
            logger=logger,
            sampling_client=sampling_client,
        )
        self._default_session = MCPSession(session_id="default")
        self._tools = self._build_tool_registry()

    def _build_tool_registry(self) -> dict[str, MCPToolDefinition]:
        return {
            tool.name: tool
            for tool in (
                MCPToolDefinition(
                    name="search_memory",
                    description=(
                        "Search the knowledge graph for relevant context. Returns ranked results with an "
                        "``inject`` flag (true = trustworthy enough to inject into agent context; decided by a "
                        "calibrated rerank-logit floor + relative margin). Inject only results with inject=true; "
                        "``score`` is a display/logit score and may be negative for correctly-ranked results. "
                        "``include``: ``default`` excludes distilled session transcripts (only not-yet-distilled "
                        "ones remain searchable), ``transcripts``/``all`` include everything."
                    ),
                    input_model=SearchMemoryToolRequest,
                    handler=self.service.search_memory,
                ),
                MCPToolDefinition(
                    name="write_memory",
                    description=(
                        "Write durable knowledge to memory. For knowledge (how a system works, decisions with "
                        "rationale, procedures, pitfalls): set type=persistent, follow the OKF template "
                        "(docs/okf.md) — okf_type decision|fact|procedure|pitfall, fixed English ## sections "
                        "(e.g. Symptom/Cause/Fix for pitfalls), a single-line ## Takeaway (<=200 chars), ## Sources "
                        "listing where the knowledge came from, and an ENGLISH title (<= 90 chars, specific claim, "
                        "no type prefix). Omitting ``type`` defaults to persistent when the body already carries "
                        "OKF structure (okf_type or matching ## sections) and transient otherwise; persistent "
                        "writes without okf_type are auto-normalized (original text preserved under "
                        "'## Original note'). Session chatter and status updates are transient."
                    ),
                    input_model=WriteMemoryRequest,
                    handler=self.service.write_memory,
                ),
            )
        }

    def list_tools(self) -> list[dict[str, object]]:
        return [self._tools[name].to_payload() for name in sorted(self._tools)]

    def allocate_client_request_id(self, session: MCPSession | None = None) -> int:
        return self._get_session(session).allocate_request_id()

    def call_tool(
        self,
        name: str,
        arguments: dict[str, object] | None = None,
        *,
        sampling_client: SamplingClient | None = None,
    ) -> dict[str, object]:
        tool = self._tools.get(name)
        if tool is None:
            raise SynapseServiceError(
                "TOOL_NOT_FOUND",
                f"Tool '{name}' is not registered",
                status_code=404,
                details={"tool_name": name},
            )
        try:
            parsed = tool.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            raise SynapseServiceError(
                "INVALID_ARGUMENTS",
                "Tool arguments failed validation",
                status_code=400,
                details={"errors": exc.errors()},
            ) from exc
        arguments = parsed.model_dump()
        if "type" in arguments:
            arguments["node_type"] = arguments.pop("type")
        with self._use_sampling_client(sampling_client):
            result = tool.handler(**arguments)
        return self._build_tool_result(result)

    def _build_tool_result(self, result: dict[str, object]) -> dict[str, object]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2, default=str),
                }
            ],
            "structuredContent": result,
        }

    def handle_request(
        self,
        payload: dict[str, Any],
        *,
        session: MCPSession | None = None,
        sampling_client: SamplingClient | None = None,
    ) -> dict[str, object] | None:
        active_session = self._get_session(session)
        request_id = payload.get("id")
        method = str(payload.get("method") or "")
        params = payload.get("params") or {}
        try:
            handled_notification, notification_result = self._handle_notification(method, active_session)
            if handled_notification:
                return notification_result

            if method == "initialize":
                result = self._handle_initialize(params, active_session)
            elif method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                result = self._handle_tool_call(params, sampling_client=sampling_client)
            elif method == "ping":
                result = {"pong": True}
            elif method.startswith("notifications/"):
                return None
            else:
                raise SynapseServiceError(
                    "METHOD_NOT_FOUND",
                    f"Unsupported MCP method '{method}'",
                    status_code=404,
                    details={"method": method},
                )
        except SynapseServiceError as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": exc.status_code,
                    "message": exc.message,
                    "data": exc.to_payload()["error"],
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _handle_initialize(self, params: dict[str, Any], session: MCPSession) -> dict[str, object]:
        requested_version = str(params.get("protocolVersion") or "").strip()
        session.protocol_version = (
            requested_version if requested_version in _SUPPORTED_PROTOCOL_VERSIONS else _DEFAULT_PROTOCOL_VERSION
        )
        capabilities = params.get("capabilities") or {}
        session.set_capabilities(capabilities if isinstance(capabilities, dict) else {})
        session.client_initialized = False
        return {
            "protocolVersion": session.protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "synapse", "version": __version__},
        }

    def _handle_tool_call(
        self,
        params: dict[str, Any],
        *,
        sampling_client: SamplingClient | None = None,
    ) -> dict[str, object]:
        return self.call_tool(
            str(params.get("name") or ""),
            params.get("arguments") if isinstance(params, dict) else None,
            sampling_client=sampling_client,
        )

    def _handle_notification(self, method: str, session: MCPSession) -> tuple[bool, dict[str, object] | None]:
        if method == "notifications/initialized":
            session.mark_initialized()
            return True, None
        if method.startswith("notifications/"):
            return True, None
        return False, None

    def _get_session(self, session: MCPSession | None) -> MCPSession:
        return session or self._default_session

    @contextmanager
    def _use_sampling_client(self, sampling_client: SamplingClient | None):
        if sampling_client is None:
            yield
            return

        token = self.service.push_sampling_client(sampling_client)
        try:
            yield
        finally:
            self.service.reset_sampling_client(token)

