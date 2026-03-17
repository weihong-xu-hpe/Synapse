"""Stdio-to-Streamable-HTTP proxy for Synapse MCP.

Bridges VS Code's stdio MCP transport to an existing Synapse HTTP server,
enabling full sampling support (which Streamable HTTP clients may not handle).

Usage:
    python -m synapse mcp-proxy [--url http://host:port/mcp]

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


def _write_message(message: dict[str, Any]) -> None:
    """Write a JSON-RPC message to stdout (newline-delimited)."""
    raw = json.dumps(message, ensure_ascii=False)
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


class StdioProxy:
    """Bridges stdin/stdout JSON-RPC to a Synapse Streamable HTTP server."""

    def __init__(self, server_url: str = _DEFAULT_SERVER_URL) -> None:
        self._server_url = server_url
        self._session_id: str | None = None
        self._http = httpx.Client(timeout=120.0)
        self._pending_sampling: dict[int, bool] = {}
        self._sse_thread: threading.Thread | None = None
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
            # This is a sampling response from VS Code — forward to HTTP server
            self._forward_sampling_response(msg)
            return

        if _is_jsonrpc_notification(msg):
            self._post_json(msg)
            return

        if _is_jsonrpc_request(msg):
            method = msg.get("method", "")
            if method == "initialize":
                self._handle_initialize(msg)
            elif self._needs_sse_sampling(msg):
                self._handle_sampling_tool_call(msg)
            else:
                self._handle_simple_request(msg)
            return

        logger.warning("Unknown message shape: %s", json.dumps(msg)[:200])

    def _handle_initialize(self, msg: dict[str, Any]) -> None:
        """Forward initialize, capture session ID."""
        resp = self._http.post(
            self._server_url,
            json=msg,
            headers={"Content-Type": "application/json"},
        )
        self._session_id = resp.headers.get(_SESSION_HEADER)
        body = resp.json()
        _write_message(body)

        # Start GET SSE listener for this session (for async sampling pushes)
        if self._session_id and self._sse_thread is None:
            self._sse_thread = threading.Thread(
                target=self._listen_sse,
                daemon=True,
            )
            self._sse_thread.start()

    def _handle_simple_request(self, msg: dict[str, Any]) -> None:
        """Forward a non-sampling request synchronously."""
        resp = self._post_json(msg)
        if resp is None:
            return
        if resp.status_code == 202:
            # Notification accepted, no body
            return
        try:
            body = resp.json()
            _write_message(body)
        except (json.JSONDecodeError, httpx.DecodingError):
            pass

    def _handle_sampling_tool_call(self, msg: dict[str, Any]) -> None:
        """Forward a sampling tool call with Accept: text/event-stream, relay SSE events."""
        headers = self._session_headers()
        headers["Accept"] = _SSE_MEDIA_TYPE

        try:
            with self._http.stream(
                "POST",
                self._server_url,
                json=msg,
                headers=headers,
            ) as resp:
                self._process_sse_stream(resp)
        except httpx.HTTPError as exc:
            _write_message({
                "jsonrpc": "2.0",
                "id": msg.get("id"),
                "error": {
                    "code": -32000,
                    "message": f"HTTP proxy error: {exc}",
                },
            })

    def _process_sse_stream(self, resp: httpx.Response) -> None:
        """Read SSE events from response, dispatch sampling requests and tool results."""
        buffer = ""
        for chunk in resp.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                event_block, buffer = buffer.split("\n\n", 1)
                for line in event_block.split("\n"):
                    if line.startswith(_SSE_DATA_PREFIX):
                        data = line[len(_SSE_DATA_PREFIX):]
                        self._dispatch_sse_data(data)

    def _dispatch_sse_data(self, raw: str) -> None:
        """Parse an SSE data payload and route it."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        if isinstance(msg.get("method"), str) and msg["method"] == "sampling/createMessage":
            # Server wants host-side sampling — push to VS Code via stdout
            _write_message(msg)
        elif "result" in msg or "error" in msg:
            # This is the final tool result — push to VS Code
            _write_message(msg)
        else:
            # Other notifications — forward
            _write_message(msg)

    def _forward_sampling_response(self, msg: dict[str, Any]) -> None:
        """Forward a sampling response from VS Code back to the HTTP server."""
        self._post_json(msg)

    def _listen_sse(self) -> None:
        """Background GET SSE listener for session-level async events."""
        if not self._session_id:
            return
        headers = {
            _SESSION_HEADER: self._session_id,
            "Accept": _SSE_MEDIA_TYPE,
        }
        try:
            with self._http.stream("GET", self._server_url, headers=headers) as resp:
                buffer = ""
                for chunk in resp.iter_text():
                    if self._stop_event.is_set():
                        break
                    buffer += chunk
                    while "\n\n" in buffer:
                        event_block, buffer = buffer.split("\n\n", 1)
                        for line in event_block.split("\n"):
                            if line.startswith(_SSE_DATA_PREFIX):
                                data = line[len(_SSE_DATA_PREFIX):]
                                self._dispatch_sse_data(data)
        except (httpx.HTTPError, httpx.StreamError):
            if not self._stop_event.is_set():
                logger.warning("SSE listener disconnected, sampling via GET SSE unavailable")

    def _needs_sse_sampling(self, msg: dict[str, Any]) -> bool:
        """Check if this tools/call requires sampling (SSE transport)."""
        if str(msg.get("method", "")) != "tools/call":
            return False
        params = msg.get("params")
        if not isinstance(params, dict):
            return False
        from synapse.server.mcp import is_sampling_tool_name
        return is_sampling_tool_name(str(params.get("name", "")))

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
