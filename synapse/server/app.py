"""FastAPI application for Synapse's Streamable MCP server runtime."""

from __future__ import annotations

import json
from threading import Thread
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from synapse import __version__
from synapse.server.decider import LocalLLMDecider
from synapse.server.mcp import SynapseMCPServer
from synapse.server.service import SynapseServerService, SynapseServiceError
from synapse.server.streamable_runtime import StreamableEventStream, StreamableToolOrchestrator, create_streamable_session_manager


PROTECTED_PREFIXES = ("/mcp",)
STREAMABLE_SESSION_HEADER = "mcp-session-id"
ACTIVE_ARCHITECTURE_DOC = "docs/design/streamable-mcp-single-path-architecture.md"
STREAMABLE_TRANSPORT_NAME = "streamable-http"
STREAMABLE_SSE_MEDIA_TYPE = "text/event-stream"
_SSE_POLL_INTERVAL_SECONDS = 0.25
_SSE_KEEPALIVE_SECONDS = 5.0


def _path_is_protected(path: str) -> bool:
    return path.startswith(PROTECTED_PREFIXES)


def _unauthorized_response() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Bearer authentication failed",
                "details": {},
            }
        },
    )


def _streamable_session_id_from_request(request: Request) -> str:
    return request.headers.get(STREAMABLE_SESSION_HEADER, "").strip()


def _jsonrpc_request_id(payload: dict[str, Any]) -> Any:
    return payload.get("id")


def _is_jsonrpc_request(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("method"), str) and bool(str(payload.get("method") or "").strip())


def _is_jsonrpc_notification(payload: dict[str, Any]) -> bool:
    return _is_jsonrpc_request(payload) and "id" not in payload


def _is_jsonrpc_response(payload: dict[str, Any]) -> bool:
    return not _is_jsonrpc_request(payload) and "id" in payload and (
        "result" in payload or "error" in payload
    )


def _require_streamable_session_id(request: Request) -> str:
    session_id = _streamable_session_id_from_request(request)
    if session_id:
        return session_id
    raise SynapseServiceError(
        "MCP_SESSION_REQUIRED",
        "Streamable MCP session header is required for all non-initialize requests",
        status_code=400,
        details={"header": STREAMABLE_SESSION_HEADER},
    )


def _tool_call_requires_sampling(payload: dict[str, Any], *, sampling_enabled: bool) -> bool:
    if not sampling_enabled:
        return False
    if str(payload.get("method") or "") != "tools/call":
        return False
    params = payload.get("params")
    if not isinstance(params, dict):
        return False
    return str(params.get("name") or "").strip() == "write_memory"


def _jsonrpc_error_response(
    payload: dict[str, Any],
    *,
    code: str,
    message: str,
    status_code: int,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": _jsonrpc_request_id(payload),
        "error": {
            "code": status_code,
            "message": message,
            "data": {
                "code": code,
                "message": message,
                "details": details or {},
            },
        },
    }


def _encode_sse_message(message: dict[str, object]) -> str:
    return f"data: {json.dumps(message, ensure_ascii=False)}\n\n"


def _request_prefers_event_stream(request: Request) -> bool:
    accept_header = request.headers.get("accept", "")
    return STREAMABLE_SSE_MEDIA_TYPE in accept_header.casefold()


def _run_streamable_request(
    *,
    mcp_server: SynapseMCPServer,
    payload: dict[str, Any],
    session,
    response_stream: StreamableEventStream,
    close_stream_when_done: bool = True,
) -> None:
    try:
        sampling_client = session.create_sampling_client(event_sink=response_stream.publish)
        response_payload = mcp_server.handle_request(
            payload,
            session=session.transport_session,
            sampling_client=sampling_client,
        )
        if response_payload is not None:
            response_stream.publish(response_payload)
    except SynapseServiceError as exc:
        response_stream.publish(
            _jsonrpc_error_response(
                payload,
                code=exc.code,
                message=exc.message,
                status_code=exc.status_code,
                details=exc.details,
            )
        )
    except (LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - defensive JSON-RPC boundary
        response_stream.publish(
            _jsonrpc_error_response(
                payload,
                code="INTERNAL_ERROR",
                message="Unhandled Streamable MCP server error",
                status_code=500,
                details={"reason": str(exc)},
            )
        )
    finally:
        if close_stream_when_done:
            response_stream.close()


async def _iterate_sse_messages(request: Request, event_stream: StreamableEventStream):
    last_event_at = perf_counter()
    while True:
        if await request.is_disconnected():
            break
        message = await run_in_threadpool(event_stream.read, _SSE_POLL_INTERVAL_SECONDS)
        if message is None:
            if event_stream.is_closed():
                break
            if perf_counter() - last_event_at >= _SSE_KEEPALIVE_SECONDS:
                last_event_at = perf_counter()
                yield ": keepalive\n\n"
            continue
        last_event_at = perf_counter()
        yield _encode_sse_message(message)


def _resolve_session_for_post(request: Request, payload: dict[str, Any], session_manager):
    method = str(payload.get("method") or "")
    session_id = _streamable_session_id_from_request(request)
    if method == "initialize":
        if session_id:
            raise SynapseServiceError(
                "INVALID_REQUEST",
                "Initialize requests must not include an existing Streamable MCP session header",
                status_code=400,
                details={"header": STREAMABLE_SESSION_HEADER},
            )
        return session_manager.create_session(client_id=_client_name_from_payload(payload))
    return session_manager.get_session(_require_streamable_session_id(request))


async def _handle_jsonrpc_request(
    *,
    request: Request,
    payload: dict[str, Any],
    session,
    mcp_server: SynapseMCPServer,
    sampling_client,
    sampling_enabled: bool,
) -> Response:
    response_headers = {STREAMABLE_SESSION_HEADER: session.session_id}
    requires_transport_sampling = (
        sampling_client is None
        and sampling_enabled
        and session.supports_sampling
        and _tool_call_requires_sampling(payload, sampling_enabled=sampling_enabled)
    )
    if requires_transport_sampling and _request_prefers_event_stream(request):
        response_stream = StreamableEventStream()
        worker = Thread(
            target=_run_streamable_request,
            kwargs={
                "mcp_server": mcp_server,
                "payload": payload,
                "session": session,
                "response_stream": response_stream,
                "close_stream_when_done": True,
            },
            daemon=True,
        )
        worker.start()
        return StreamingResponse(
            _iterate_sse_messages(request, response_stream),
            media_type=STREAMABLE_SSE_MEDIA_TYPE,
            headers=response_headers,
        )

    if requires_transport_sampling:
        worker = Thread(
            target=_run_streamable_request,
            kwargs={
                "mcp_server": mcp_server,
                "payload": payload,
                "session": session,
                "response_stream": session.server_event_stream,
                "close_stream_when_done": False,
            },
            daemon=True,
        )
        worker.start()
        return Response(status_code=202, headers=response_headers)

    response_payload = await run_in_threadpool(
        mcp_server.handle_request,
        payload,
        session=session.transport_session,
        sampling_client=sampling_client,
    )
    if response_payload is None:
        return Response(status_code=202, headers=response_headers)
    return JSONResponse(content=response_payload, headers=response_headers)


def _create_mcp_post_handler(mcp_server: SynapseMCPServer, session_manager, sampling_client, sampling_enabled: bool):
    async def mcp_streamable_api(request: Request) -> Response:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise SynapseServiceError(
                "INVALID_REQUEST",
                "Streamable MCP endpoint accepts exactly one JSON-RPC message per POST body",
                status_code=400,
            )

        session = _resolve_session_for_post(request, payload, session_manager)
        response_headers = {STREAMABLE_SESSION_HEADER: session.session_id}

        if _is_jsonrpc_response(payload):
            session.accept_sampling_response(payload)
            return Response(status_code=202, headers=response_headers)

        if _is_jsonrpc_notification(payload):
            await run_in_threadpool(
                mcp_server.handle_request,
                payload,
                session=session.transport_session,
                sampling_client=sampling_client,
            )
            return Response(status_code=202, headers=response_headers)

        if not _is_jsonrpc_request(payload):
            raise SynapseServiceError(
                "INVALID_REQUEST",
                "POST /mcp requires a JSON-RPC request, notification, or response",
                status_code=400,
            )

        return await _handle_jsonrpc_request(
            request=request,
            payload=payload,
            session=session,
            mcp_server=mcp_server,
            sampling_client=sampling_client,
            sampling_enabled=sampling_enabled,
        )

    return mcp_streamable_api


def _create_mcp_get_handler(session_manager):
    def mcp_streamable_events(request: Request) -> StreamingResponse:
        session = session_manager.get_session(_require_streamable_session_id(request))
        return StreamingResponse(
            _iterate_sse_messages(request, session.server_event_stream),
            media_type=STREAMABLE_SSE_MEDIA_TYPE,
            headers={STREAMABLE_SESSION_HEADER: session.session_id},
        )

    return mcp_streamable_events


def _create_mcp_delete_handler(session_manager):
    def mcp_close_session_api(request: Request) -> dict[str, object]:
        return session_manager.close_session(_require_streamable_session_id(request))

    return mcp_close_session_api


def _create_root_handler():
    def root() -> dict[str, object]:
        return {
            "name": "synapse",
            "version": __version__,
            "mcp": "/mcp",
            "rest": {"search": "/api/search", "write": "/api/write"},
            "transport": STREAMABLE_TRANSPORT_NAME,
            "session_header": STREAMABLE_SESSION_HEADER,
            "architecture_doc": ACTIVE_ARCHITECTURE_DOC,
        }

    return root


def _create_rest_search_handler(service: SynapseServerService):
    """Thin REST wrapper around service.search_memory for non-MCP clients (e.g. omp hooks)."""

    async def search_memory(request: Request) -> JSONResponse:
        body = await request.json()
        query = str(body.get("query", "")).strip()
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "INVALID_QUERY", "message": "Search query must not be blank", "details": {}}},
            )
        top_k = int(body.get("top_k", 3))
        result = await run_in_threadpool(service.search_memory, query, top_k)
        return JSONResponse(status_code=200, content=result)

    return search_memory


def _create_rest_write_handler(service: SynapseServerService):
    """Thin REST wrapper around service.write_memory for non-MCP clients (e.g. omp hooks)."""

    async def write_memory(request: Request) -> JSONResponse:
        body = await request.json()
        title = str(body.get("title", "")).strip()
        if not title:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "INVALID_TITLE", "message": "Title must not be blank", "details": {}}},
            )
        result = await run_in_threadpool(
            service.write_memory,
            title,
            str(body.get("content", "")),
            body.get("type", "transient"),
            body.get("links"),
            body.get("sensitivity", "internal"),
            body.get("query_hint"),
            float(body.get("similarity_threshold", 0.3)),
        )
        return JSONResponse(status_code=200, content=result)

    return write_memory

def _create_auth_logging_middleware(config, service: SynapseServerService, logger):
    async def auth_and_logging(request: Request, call_next):
        start = perf_counter()
        body = await request.body()
        response = None
        auth_failed = False

        if request.method.upper() != "OPTIONS" and _path_is_protected(request.url.path):
            configured_token = config.server.auth_token.strip()
            if configured_token:
                header_value = request.headers.get("authorization", "")
                expected = f"Bearer {configured_token}"
                if header_value != expected:
                    auth_failed = True
                    response = _unauthorized_response()

        if response is None:
            response = await call_next(request)

        latency_ms = round((perf_counter() - start) * 1000, 3)
        response_size = response.headers.get("content-length")
        if response_size is None and hasattr(response, "body"):
            response_size = str(len(getattr(response, "body") or b""))

        logger_to_use = logger or service.logger
        logger_to_use.info(
            "Synapse HTTP request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query": dict(request.query_params),
                "request_size": len(body),
                "response_size": int(response_size or 0),
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "auth_failed": auth_failed,
            },
        )
        return response

    return auth_and_logging


def _client_name_from_payload(payload: dict[str, Any]) -> str:
    params = payload.get("params")
    if not isinstance(params, dict):
        return ""
    client_info = params.get("clientInfo")
    if not isinstance(client_info, dict):
        return ""
    return str(client_info.get("name") or "").strip()


def create_app(config, *, runtime_paths=None, logger=None, sampling_client=None, lifespan=None) -> FastAPI:
    """Create the Streamable-oriented FastAPI app with a single MCP endpoint."""

    sampling_enabled = config.decider.provider == "mcp_sampling"
    effective_sampling_client = sampling_client
    if config.decider.provider == "local_llm" and effective_sampling_client is None:
        effective_sampling_client = LocalLLMDecider(config.decider)

    service = SynapseServerService(
        config,
        runtime_paths=runtime_paths,
        logger=logger,
        sampling_client=effective_sampling_client,
    )
    mcp_server = SynapseMCPServer(
        config,
        runtime_paths=service.runtime_paths,
        logger=logger,
        service=service,
        sampling_client=effective_sampling_client,
    )
    session_manager = create_streamable_session_manager()
    orchestrator = StreamableToolOrchestrator(
        service,
        logger=logger,
        sampling_client=effective_sampling_client,
    )

    app = FastAPI(title="Synapse", version=__version__, lifespan=lifespan)
    app.state.mcp_server = mcp_server
    app.state.streamable_session_manager = session_manager
    app.state.streamable_orchestrator = orchestrator
    app.state.streamable_transport = STREAMABLE_TRANSPORT_NAME
    app.state.sampling_client = effective_sampling_client

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[STREAMABLE_SESSION_HEADER],
    )

    @app.exception_handler(SynapseServiceError)
    async def handle_service_error(_request: Request, exc: SynapseServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    app.middleware("http")(_create_auth_logging_middleware(config, service, logger))

    app.post("/mcp")(
        _create_mcp_post_handler(
            mcp_server,
            session_manager,
            effective_sampling_client,
            sampling_enabled,
        )
    )
    app.get("/mcp")(_create_mcp_get_handler(session_manager))
    app.delete("/mcp")(_create_mcp_delete_handler(session_manager))
    app.get("/")(_create_root_handler())
    app.post("/api/search")(_create_rest_search_handler(service))
    app.post("/api/write")(_create_rest_write_handler(service))

    return app
