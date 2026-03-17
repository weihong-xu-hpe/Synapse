# Synapse Overview

[Back to README](../README.md)

## What Synapse is

Synapse is a local hybrid memory system for AI agents and human-in-the-loop workflows.

It combines:

- **Markdown as the canonical knowledge store**
- **SQLite as a rebuildable derived index**
- **an MCP interface** for tools, editors, and automation
- **Lifecycle management** so memory can decay, archive, and condense instead of growing forever

The design goal is to preserve three things at the same time:

1. **local control**
2. **fast retrieval**
3. **human inspectability**

## Product philosophy

### Markdown is the source of truth

Every memory node is stored as a Markdown file with YAML frontmatter.

This keeps memory:

- human-readable
- Git-friendly
- easy to edit in external tools
- recoverable even if the search index is lost

### The index is disposable

SQLite is used as a performance layer, not as the authority.

If the index is deleted or corrupted, Synapse can rebuild it from Markdown files.

### Inference is external

Synapse does not manage model runtimes itself.

- Local inference is delegated to already-running **llama.cpp** (`llama-server`) instances.
- Remote inference is delegated to a configurable **HTTP API**.

This keeps Synapse focused on memory orchestration rather than model hosting.

### Forgetting is intentional

Synapse treats memory as a living system.

Instead of keeping everything equally hot forever, it supports:

- retrieval-time decay
- nightly janitor eviction
- superseded-node archival
- archive cleanup
- manual condensation of archived notes into higher-level summaries

## Core capabilities

Current implementation includes:

- Markdown-backed node model with frontmatter
- atomic writes and wiki-link parsing
- provider-aware embedding and reranking
- hybrid retrieval with keyword search, vector search, graph hop, and reranking
- file watching and delta sync
- MCP server surface
- write-path orchestration with conflict handling
- lifecycle janitor and archive condensation
- macOS `launchd` and Linux `systemd --user` service integration

## High-level architecture

Synapse is built around four persistent layers:

1. **Active Markdown memory** in `.synapse/active`
2. **Archived Markdown memory** in `.synapse/.archive`
3. **Derived SQLite index** in `.synapse/synapse.db`
4. **Agent-facing MCP interface**

Typical flow:

- a write creates or updates Markdown
- sync updates SQLite
- retrieval combines lexical and semantic search
- lifecycle processes keep the active set dense and relevant

## Memory model

Each node includes metadata such as:

- `id`
- `title`
- `created_at`
- `last_accessed`
- `type`
- `status`
- `supersedes`
- `superseded_by`
- `tags`
- `sensitivity`

### Records

All nodes share a single **3 500-word limit** and a single decay factor.
There is no tier distinction — every record receives equal treatment during
retrieval, decay, and lifecycle processing.

### Status values

- `active`
- `superseded`
- `disputed`

## Retrieval model

The retrieval pipeline is hybrid by design:

1. keyword search
2. vector search
3. reciprocal rank fusion
4. one-hop graph expansion
5. reranking
6. status and decay scoring
7. final context assembly

Only nodes that survive the full pipeline receive access-refresh signals.

## Write path

The write path can do more than persist new notes.

The public write path is exposed through high-level MCP tools. Synapse prepares
deterministic context, asks the connected MCP host/client for a structured JSON
decision when sampling is required, validates that decision, then compiles it
into the same low-level write contract used internally for `create`,
`supersede`, and `complement` execution.

It can:

- find related nodes
- add links automatically
- detect high-similarity conflicts
- mark nodes as superseded
- support manual disputed-status correction when needed
- preserve provenance with banners and frontmatter

## Lifecycle model

Synapse includes real lifecycle controls:

- retrieval-time decay
- janitor-based orphan eviction
- superseded archival
- archive retention cleanup
- archive condensation into new active memory notes

## Related documentation

- [Configuration](configuration.md)
- [Usage](usage.md)
- [Deployment](deployment.md)
