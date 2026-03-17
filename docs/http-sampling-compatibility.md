# HTTP Sampling Compatibility

[Back to README](../README.md)

> **Historical document — retired guidance.**
>
> This file only records that Synapse previously validated an HTTP/SSE sampling bridge during the transition toward a single sampling-first MCP surface. It is **not** the current product recommendation.

## What this file still means

Historically, Synapse used this document to discuss an HTTP/SSE bridge that could:

- open HTTP-scoped MCP sessions
- emit server-initiated sampling requests
- accept sampling responses over HTTP
- resume the original tool call after validation

That work was useful for proving transport semantics, but it is no longer the architecture source of truth.

## What to use instead

For current guidance, use these files in order:

1. `docs/design/streamable-mcp-single-path-architecture.md`
2. `docs/design/streamable-mcp-implementation-plan.md`
3. `docs/design/TODO-sampling-only-cutover.md`
4. `docs/usage.md`

Those documents describe the current public story:

- one public MCP surface
- high-level sampling-backed tools as the formal write/lifecycle entry
- low-level canonical helpers remaining internal-only
- repository skill files retained only as policy/reference assets

## What remains historically useful

The retired HTTP/SSE work still provided useful evidence for:

- session lifecycle semantics
- duplicate-response handling
- timeout / expiry behavior
- close / cancellation behavior

If you need to inspect those behaviors today, rely on the current regression suites rather than this retired note:

- `tests/test_server_api.py`
- `tests/test_streamable_runtime.py`
- `tests/test_write_path.py`

## Current rule of thumb

Do **not** use this file to decide:

- which transport to recommend
- whether a host is officially supported
- whether skill files are part of the runtime integration path

Use it only as historical background for the retired bridge experiments.