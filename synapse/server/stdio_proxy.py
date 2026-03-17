"""Stdio-to-Streamable-HTTP proxy for Synapse MCP.

Bridges VS Code's stdio MCP transport to an existing Synapse HTTP server,
enabling full sampling support (which Streamable HTTP clients may not handle).

Architecture:
    - Main thread: reads stdin, dispatches messages
    - SSE thread: reads SSE stream from POST (sampling tools) or GET (session),
      writes sampling/createMessage requests to stdout
    - Sampling responses from stdin are forwarded back to the HTTP server

Usage:
    python -m synapse mcp-proxy [<server>]

VS Code mcp.json:
    {
      "synapse": {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "python", "-m", "synapse", "mcp-proxy"]
      }
    }
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from typing import Any

import httpx

logger = logging.getLogger("synapse.stdio-proxy")

_DEFAULT_SERVER_URL = "http://127.0.0.1:8765/mcp"
_SSE_DATA_PREFIX = "data: "
_SSE_MEDIA_TYPE = "text/event-stream"
_SESSION_HEADER = "mcp-session-id"
_SAMPLING_TOOL_NAMES = frozenset({
    "decide_memory_write",
    "integrate_memory_with_sampling",
    "run_dreamer",
})

_write_lock = threading.Lock()


def _write_message(message: dict[str, Any]) -> None:
    """Write a JSON-RPC message to stdout (newline-delimited). Thread-safe."""
    raw = json.dumps(message, ensure_ascii=False)
    with _write_lock:
        sys.stdout.write(raw + "\n")
        sys.stdout.flush()


def _read_message() -> dict[str, Any] | None:
    """Read a single JSON-RPC message from stdin (newline-delimited)."""
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return None
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Ignoring non-JSON line from stdin: %s", line[:120])
        return None


def _is_jsonrpc_request(msg: dict[str, Any]) -> bool:
    return isinstance(msg.get("method"), str) and "id" in msg


def _is_jsonrpc_notification(msg: dict[str, Any]) -> bool:
    return isinstance(msg.get("method"), str) and "id" not in msg


def _is_jsonrpc_response(msg: dict[str, Any]) -> bool:
    return "id" in msg and ("result" in msg or "error" in msg) and "method" not in msg


def _needs_sse_sampling(msg: dict[str, Any]) -> bool:
    """Check if this tools/call requires sampling."""
    if str(msg.get("method", "")) != "tools/call":
        return False
    params = msg.get("params")
    if not isinstance(params, dict):
        return False
    return str(params.get("name", "")).strip() in _SAMPLING_TOOL_NAMES


def _parse_sse_events(raw_buffer: str) -> tuple[list[str], str]:
    """Extract complete SSE data payloads from a buffer, return (payloads, remaining)."""
    payloads: list[str] = []
    while "\n\n" in raw_buffer:
        event_block, raw_buffer = raw_buffer.split("\n\n", 1)
        for line in event_block.split("\n"):
            if line.startswith(_SSE_DATA_PREFIX):
                payloads.append(line[len(_SSE_DATA_PREFIX):])
    return payloads, raw_buffer


class StdioProxy:
    """Bridges stdin/stdout JSON-RPC to a Synapse Streamable HTTP server.

    Sampling flow:
    1. Tool call arrives on stdin → proxy POSTs with Accept: text/event-stream
    2. SSE reader thread reads the response stream in background
    3. When server pushes sampling/createMessage → written to stdout
    4. VS Code invokes LLM, sends response on stdin → main loop reads it
    5. Main loop forwards sampling response via POST to server
    6. Server completes tool call → SSE stream pushes final result → written to stdout
    """

    def __init__(self, server_url: str = _DEFAULT_SERVER_URL) -> None:
        self._server_url = server_url
        self._session_id: str | None = None
        self._http = httpx.Client(timeout=300.0)
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Main loop: read stdin, forward to HTTP, write responses to stdout."""
        logger.info("Stdio proxy started, server=%s", self._server_url)
        try:
            while not self._stop_event.is_set():
                msg = _read_message()
                if msg is None:
                    break
                self._handle_stdin_message(msg)
        except (BrokenPipeError, KeyboardInterrupt):
            pass
        finally:
            self._stop_event.set()
            self._http.close()
            logger.info("Stdio proxy stopped")

    def _handle_stdin_message(self, msg: dict[str, Any]) -> None:
        if _is_jsonrpc_response(msg):
            self._forward_sampling_response(msg)
            return

        if _is_jsonrpc_notification(msg):
            self._post_json(msg)
            return

        if _is_jsonrpc_request(msg):
            method = msg.get("method", "")
            if method == "initialize":
                self._handle_initialize(msg)
            elif _needs_sse_sampling(msg):
                self._handle_sampling_tool_call(msg)
            else:
                self._handle_simple_request(msg)
            return

        logger.warning("Unknown message shape: %s", json.dumps(msg)[:200])

    def _handle_initialize(self, msg: dict[str, Any]) -> None:
        resp = self._http.post(
            self._server_url,
            json=msg,
            headers={"Content-Type": "application/json"},
        )
        self._session_id = resp.headers.get(_SESSION_HEADER)
        body = resp.json()
        _write_message(body)

    def _handle_simple_request(self, msg: dict[str, Any]) -> None:
        resp = self._post_json(msg)
        if resp is None:
            return
        if resp.status_code == 202:
            return
        try:
            body = resp.json()
            _write_message(body)
        except (json.JSONDecodeError, httpx.DecodingError):
            pass

    def _handle_sampling_tool_call(self, msg: dict[str, Any]) -> None:
        """Forward a sampling tool call, read SSE in background so stdin stays free.

        The SSE stream is read in a daemon thread. When the server pushes
        sampling/createMessage, it's written to stdout. The main loop continues
        reading stdin and can pick up the sampling response from VS Code.
        """
        headers = self._session_headers()
        headers["Accept"] = _SSE_MEDIA_TYPE

        request_id = msg.get("id")

        # Use a separate httpx client for the streaming connection so
        # the main client stays free for forwarding sampling responses.
        stream_http = httpx.Client(timeout=300.0)

        def _read_sse_and_relay():
            try:
                with stream_http.stream(
                    "POST",
                    self._server_url,
                    json=msg,
                    headers=headers,
                ) as resp:
                    buffer = ""
                    for chunk in resp.iter_text():
                        if self._stop_event.is_set():
                            break
                        buffer += chunk
                        payloads, buffer = _parse_sse_events(buffer)
                        for payload in payloads:
                            try:
                                event_msg = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            _write_message(event_msg)
            except httpx.HTTPError as exc:
                _write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": f"HTTP proxy SSE error: {exc}",
                    },
                })
            finally:
                stream_http.close()

        worker = threading.Thread(target=_read_sse_and_relay, daemon=True)
        worker.start()
        # Do NOT join — return immediately so main loop can read stdin
        # for sampling responses.

    def _forward_sampling_response(self, msg: dict[str, Any]) -> None:
        self._post_json(msg)

    def _session_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._session_id:
            headers[_SESSION_HEADER] = self._session_id
        return headers

    def _post_json(self, msg: dict[str, Any]) -> httpx.Response | None:
        try:
            return self._http.post(
                self._server_url,
                json=msg,
                headers=self._session_headers(),
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP post failed: %s", exc)
            return None


def run_stdio_proxy(server_url: str = _DEFAULT_SERVER_URL) -> None:
    """Entry point for the stdio proxy."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    proxy = StdioProxy(server_url=server_url)
    proxy.run()
