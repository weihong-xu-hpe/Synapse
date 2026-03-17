"""Native Streamable MCP runtime primitives for Synapse."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Lock
from threading import Event
from time import time
from typing import Any, Callable, Protocol
from uuid import uuid4

from synapse.server.mcp import MCPSession, SynapseMCPServer
from synapse.server.sampling import (
    MemoryWriteSamplingDecision,
    MemoryWriteSamplingRequest,
    SamplingClient,
    parse_memory_write_sampling_result,
    parse_sampling_json_result,
)
from synapse.server.service import SynapseServerService, SynapseServiceError


_STREAM_CLOSE_SENTINEL = object()
_DEFAULT_SAMPLING_TIMEOUT_SECONDS = 30.0
_SAMPLING_REQUEST_METHOD = "sampling/createMessage"
_NON_OBJECT_SAMPLING_RESULT_MESSAGE = "Sampling-capable MCP client returned a non-object result"


class StreamableEventStream:
    """Thread-safe queue used to bridge server messages onto SSE streams."""

    def __init__(self) -> None:
        self._queue: Queue[object] = Queue()
        self._closed = False
        self._lock = Lock()

    def publish(self, message: dict[str, object]) -> None:
        with self._lock:
            if self._closed:
                raise SynapseServiceError(
                    "MCP_STREAM_CLOSED",
                    "Streamable MCP response stream is closed",
                    status_code=410,
                )
            self._queue.put(message)

    def read(self, timeout_seconds: float) -> dict[str, object] | None:
        try:
            item = self._queue.get(timeout=timeout_seconds)
        except Empty:
            return None
        if item is _STREAM_CLOSE_SENTINEL:
            return None
        return item if isinstance(item, dict) else None

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STREAM_CLOSE_SENTINEL)


@dataclass(slots=True)
class PendingSamplingRequest:
    """One outstanding sampling/createMessage request awaiting the client's response."""

    request_id: int
    request_payload: dict[str, object]
    created_at: float = field(default_factory=time)
    _done: Event = field(default_factory=Event, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)
    response_payload: dict[str, object] | None = None
    failure: SynapseServiceError | None = None

    def resolve(self, payload: dict[str, object]) -> None:
        with self._lock:
            if self._done.is_set():
                raise SynapseServiceError(
                    "MCP_DUPLICATE_RESPONSE",
                    f"Sampling response for request {self.request_id} was already received",
                    status_code=409,
                    details={"request_id": self.request_id},
                )
            self.response_payload = payload
            self._done.set()

    def fail(self, error: SynapseServiceError) -> None:
        with self._lock:
            if self._done.is_set():
                return
            self.failure = error
            self._done.set()

    def wait(self, timeout_seconds: float) -> dict[str, object]:
        if not self._done.wait(timeout_seconds):
            raise TimeoutError(f"Timed out waiting for sampling response {self.request_id}")
        if self.failure is not None:
            raise self.failure
        if self.response_payload is None:
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                "Sampling request completed without a response payload",
                status_code=502,
                details={"request_id": self.request_id},
            )
        return self.response_payload


class StreamableSamplingClient:
    """Sampling client backed by Streamable HTTP server-to-client requests."""

    name = "streamable-http"

    def __init__(
        self,
        session: StreamableSession,
        *,
        event_sink: Callable[[dict[str, object]], None] | None = None,
        timeout_seconds: float = _DEFAULT_SAMPLING_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session
        self._event_sink = event_sink
        self._timeout_seconds = timeout_seconds

    def sample_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 600,
        model_hints: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        response = self._request_sampling(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            model_hints=model_hints,
        )
        result_payload = response.get("result")
        if not isinstance(result_payload, dict):
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                _NON_OBJECT_SAMPLING_RESULT_MESSAGE,
                status_code=502,
            )
        try:
            return parse_sampling_json_result(result_payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                "Sampling-capable MCP client returned an unreadable JSON payload",
                status_code=502,
                details={"reason": str(exc)},
            ) from exc

    def decide_memory_write(self, request: MemoryWriteSamplingRequest) -> MemoryWriteSamplingDecision:
        response = self._request_sampling(
            prompt=request.prompt,
            system_prompt=(
                "You are a Synapse memory write planner. Return exactly one JSON object that decides create, "
                "supersede, or complement using only the supplied candidate set."
            ),
            max_tokens=600,
            model_hints=(),
        )
        result_payload = response.get("result")
        if not isinstance(result_payload, dict):
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                _NON_OBJECT_SAMPLING_RESULT_MESSAGE,
                status_code=502,
            )
        try:
            return parse_memory_write_sampling_result(result_payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                "Sampling-capable MCP client returned an unreadable memory write decision",
                status_code=502,
                details={"reason": str(exc)},
            ) from exc

    def _request_sampling(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        model_hints: tuple[str, ...],
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": prompt},
                }
            ],
            "maxTokens": max_tokens,
        }
        if system_prompt.strip():
            params["systemPrompt"] = system_prompt
        if model_hints:
            params["modelPreferences"] = {
                "hints": [{"name": hint} for hint in model_hints],
            }

        request_id = self._session.create_pending_sampling_request(params, event_sink=self._event_sink)
        response = self._session.wait_for_sampling_response(request_id, timeout_seconds=self._timeout_seconds)
        if "error" in response:
            error_payload = response.get("error")
            raise SynapseServiceError(
                "SAMPLING_FAILED",
                "Sampling-capable MCP client returned an error response",
                status_code=502,
                details={"response": error_payload, "request_id": request_id},
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise SynapseServiceError(
                "INVALID_SAMPLING_RESPONSE",
                _NON_OBJECT_SAMPLING_RESULT_MESSAGE,
                status_code=502,
                details={"request_id": request_id},
            )
        return response


@dataclass(slots=True)
class StreamableSession:
    """Transport-agnostic Streamable MCP session state."""

    session_id: str
    client_id: str = ""
    created_at: float = field(default_factory=time)
    transport_session: MCPSession = field(init=False)
    server_event_stream: StreamableEventStream = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _pending_sampling_requests: dict[int, PendingSamplingRequest] = field(default_factory=dict, init=False, repr=False)
    _completed_sampling_requests: dict[int, PendingSamplingRequest] = field(default_factory=dict, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.transport_session = MCPSession(session_id=self.session_id)
        self.server_event_stream = StreamableEventStream()

    @property
    def protocol_version(self) -> str:
        return self.transport_session.protocol_version

    @property
    def client_capabilities(self) -> dict[str, Any]:
        return self.transport_session.client_capabilities

    @property
    def initialized(self) -> bool:
        return self.transport_session.client_initialized

    @property
    def supports_sampling(self) -> bool:
        return self.transport_session.client_supports_sampling

    def publish_server_message(self, message: dict[str, object]) -> None:
        self.server_event_stream.publish(message)

    def create_sampling_client(
        self,
        *,
        event_sink: Callable[[dict[str, object]], None] | None = None,
        timeout_seconds: float = _DEFAULT_SAMPLING_TIMEOUT_SECONDS,
    ) -> StreamableSamplingClient:
        return StreamableSamplingClient(
            self,
            event_sink=event_sink,
            timeout_seconds=timeout_seconds,
        )

    def create_pending_sampling_request(
        self,
        params: dict[str, object],
        *,
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> int:
        with self._lock:
            if self._closed:
                raise SynapseServiceError(
                    "MCP_SESSION_CLOSED",
                    f"Streamable MCP session '{self.session_id}' is already closed",
                    status_code=404,
                    details={"session_id": self.session_id},
                )
            request_id = self.transport_session.allocate_request_id()
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": _SAMPLING_REQUEST_METHOD,
                "params": params,
            }
            self._pending_sampling_requests[request_id] = PendingSamplingRequest(
                request_id=request_id,
                request_payload=payload,
            )
        publisher = event_sink or self.publish_server_message
        publisher(payload)
        return request_id

    def wait_for_sampling_response(self, request_id: int, *, timeout_seconds: float) -> dict[str, object]:
        with self._lock:
            pending = self._pending_sampling_requests.get(request_id) or self._completed_sampling_requests.get(request_id)
        if pending is None:
            raise SynapseServiceError(
                "MCP_REQUEST_NOT_FOUND",
                f"Pending sampling request {request_id} does not exist",
                status_code=404,
                details={"request_id": request_id},
            )
        try:
            return pending.wait(timeout_seconds)
        except TimeoutError as exc:
            timeout_error = SynapseServiceError(
                "SAMPLING_TIMEOUT",
                "Timed out waiting for sampling/createMessage response from the MCP client",
                status_code=504,
                details={"request_id": request_id, "timeout_seconds": timeout_seconds},
            )
            self.fail_sampling_request(request_id, timeout_error)
            raise timeout_error from exc

    def accept_sampling_response(self, payload: dict[str, object]) -> dict[str, object]:
        raw_request_id = payload.get("id")
        if not isinstance(raw_request_id, int):
            raise SynapseServiceError(
                "INVALID_REQUEST",
                "Sampling response must include an integer id",
                status_code=400,
                details={"payload": payload},
            )
        with self._lock:
            if raw_request_id in self._completed_sampling_requests:
                raise SynapseServiceError(
                    "MCP_DUPLICATE_RESPONSE",
                    f"Sampling response for request {raw_request_id} was already received",
                    status_code=409,
                    details={"request_id": raw_request_id},
                )
            pending = self._pending_sampling_requests.pop(raw_request_id, None)
            if pending is None:
                raise SynapseServiceError(
                    "MCP_REQUEST_NOT_FOUND",
                    f"Pending sampling request {raw_request_id} does not exist",
                    status_code=404,
                    details={"request_id": raw_request_id},
                )
            self._completed_sampling_requests[raw_request_id] = pending
        pending.resolve(payload)
        return {"accepted": True, "request_id": raw_request_id}

    def fail_sampling_request(self, request_id: int, error: SynapseServiceError) -> None:
        with self._lock:
            pending = self._pending_sampling_requests.pop(request_id, None)
            if pending is None:
                return
            self._completed_sampling_requests[request_id] = pending
        pending.fail(error)

    def close(self) -> int:
        with self._lock:
            if self._closed:
                return 0
            self._closed = True
            pending = list(self._pending_sampling_requests.items())
            self._pending_sampling_requests.clear()
            for request_id, item in pending:
                self._completed_sampling_requests[request_id] = item
        self.server_event_stream.close()
        for request_id, item in pending:
            item.fail(
                SynapseServiceError(
                    "MCP_SESSION_CLOSED",
                    "Streamable MCP session closed while a sampling request was still pending",
                    status_code=404,
                    details={"session_id": self.session_id, "request_id": request_id},
                )
            )
        return len(pending)


class StreamableSessionManager:
    """Track Streamable sessions without coupling to HTTP/SSE primitives."""

    def __init__(self) -> None:
        self._sessions: dict[str, StreamableSession] = {}
        self._lock = Lock()

    def create_session(self, *, client_id: str = "") -> StreamableSession:
        session = StreamableSession(session_id=f"streamable-{uuid4().hex}", client_id=client_id)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> StreamableSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SynapseServiceError(
                "MCP_SESSION_NOT_FOUND",
                f"Streamable MCP session '{session_id}' does not exist",
                status_code=404,
                details={"session_id": session_id},
            )
        return session

    def negotiate_capabilities(
        self,
        session_id: str,
        *,
        capabilities: dict[str, Any] | None,
        protocol_version: str | None = None,
    ) -> StreamableSession:
        session = self.get_session(session_id)
        session.transport_session.set_capabilities(capabilities)
        if protocol_version:
            session.transport_session.protocol_version = protocol_version
        session.transport_session.client_initialized = False
        return session

    def mark_initialized(self, session_id: str) -> StreamableSession:
        session = self.get_session(session_id)
        session.transport_session.mark_initialized()
        return session

    def close_session(self, session_id: str) -> dict[str, object]:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise SynapseServiceError(
                "MCP_SESSION_NOT_FOUND",
                f"Streamable MCP session '{session_id}' does not exist",
                status_code=404,
                details={"session_id": session_id},
            )
        return {
            "closed": True,
            "cancelled_requests": session.close(),
            "session_id": session.session_id,
        }

    def list_active_sessions(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)


class StreamableTransportRuntime(Protocol):
    """Protocol for future Streamable transport implementations."""

    def read_message(self) -> dict[str, object]:
        """Return the next inbound MCP message."""
        raise NotImplementedError

    def write_message(self, message: dict[str, object]) -> None:
        """Send an outbound MCP message to the client."""
        raise NotImplementedError

    def emit_sampling_request(self, session_id: str, request: dict[str, object]) -> int:
        """Emit a sampling request and return its correlation id."""
        raise NotImplementedError

    def wait_sampling_response(
        self,
        session_id: str,
        request_id: int,
        timeout_seconds: float,
    ) -> dict[str, object]:
        """Block until the matching sampling response arrives."""
        raise NotImplementedError


class StreamableToolOrchestrator:
    """Route Streamable-facing tool calls onto the canonical Synapse service layer."""

    def __init__(
        self,
        service: SynapseServerService,
        *,
        logger: Any = None,
        sampling_client: SamplingClient | None = None,
    ) -> None:
        self.service = service
        self.logger = logger or service.logger
        self.sampling_client = sampling_client
        self._public_server = SynapseMCPServer(
            service.config,
            runtime_paths=service.runtime_paths,
            logger=self.logger,
            service=service,
            sampling_client=sampling_client,
        )

    def list_tools(self) -> list[dict[str, object]]:
        return self._public_server.list_tools()

    def invoke_tool(
        self,
        session: StreamableSession,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        _ = session
        try:
            return self._public_server.call_tool(tool_name, arguments, sampling_client=self.sampling_client)
        except SynapseServiceError as exc:
            return {
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "status_code": exc.status_code,
                }
            }


def create_streamable_session_manager() -> StreamableSessionManager:
    """Create a transport-agnostic Streamable session manager."""

    return StreamableSessionManager()


def create_streamable_orchestrator(
    config,
    *,
    runtime_paths=None,
    logger=None,
    sampling_client: SamplingClient | None = None,
) -> StreamableToolOrchestrator:
    """Create a Streamable tool orchestrator backed by the canonical service layer."""

    service = SynapseServerService(
        config,
        runtime_paths=runtime_paths,
        logger=logger,
        sampling_client=sampling_client,
    )
    return StreamableToolOrchestrator(
        service,
        logger=logger,
        sampling_client=sampling_client,
    )
