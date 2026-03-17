# Synapse HTTP MCP Full Test Playbook

> **Historical document — retired manual guide.**
>
> This file used to describe a manual HTTP-only validation flow for the transitional MCP/SSE sampling bridge. It is no longer the acceptance plan for Synapse.

## Why this playbook is retired

The active Synapse story is now:

- one public MCP surface
- high-level sampling-backed tools as the formal write/lifecycle entry
- low-level canonical helpers kept internal-only
- design and execution guidance centered under `docs/design/`

Because of that, a standalone HTTP-only playbook is no longer the source of truth for product acceptance.

## What to use now

If you need the current acceptance and execution picture, start with:

1. `docs/design/streamable-mcp-single-path-architecture.md`
2. `docs/design/streamable-mcp-implementation-plan.md`
3. `docs/design/TODO-sampling-only-cutover.md`
4. `docs/usage.md`

For concrete verification, rely on the repository regression suites:

- `tests/test_server_api.py`
- `tests/test_streamable_runtime.py`
- `tests/test_write_path.py`

Those tests now cover the public tool surface, sampling-backed write flow, lifecycle flow, failure semantics, and the fact that low-level canonical helpers are not publicly exposed.

## Historical value that remains

This retired playbook is still useful as a reminder that the old HTTP bridge work helped validate:

- session setup semantics
- read/write tool visibility expectations
- sampling response lifecycle behavior
- duplicate / timeout / close-path error handling

But it should be read only as historical context, not as a recommendation for how to integrate with Synapse today.

