# Synapse

Synapse is a **local, agent-friendly hybrid memory system** for long-lived AI workflows.

It keeps **Markdown as the source of truth**, builds a **derived local search index** for fast retrieval, and exposes memory through a **single public MCP surface**.

> give an agent durable memory without giving up privacy, inspectability, or local control.

## Why it exists

Synapse is built around a local-first model:

- **Markdown is canonical**
- **SQLite is derived**
- **retrieval stays local**
- **remote reasoning is optional**
- **forgetting is intentional**

## What it includes

- Markdown-backed memory nodes with YAML frontmatter
- local indexing and hybrid retrieval
- MCP server surface for retrieval and high-level memory actions
- write-path conflict detection
- lifecycle janitor and condensation flows
- macOS `launchd` and Linux `systemd --user` daemon support

## Read the docs

This README is intentionally short. Detailed documentation is split into focused guides:

- [Overview and philosophy](docs/overview.md)
- [Configuration guide](docs/configuration.md)
- [Usage guide](docs/usage.md)
- [Deployment guide](docs/deployment.md)

## Public integration direction

Synapse is being simplified around **one public integration path**:

- a single public MCP surface for retrieval, inspection, and semantic memory operations
- high-level sampling-backed tools for write and lifecycle decisions
- an internal canonical execution layer that stays behind the public interface
- repository skill files kept only as policy/reference assets, not as runtime entry points

In practice, this means agents should treat Synapse as an MCP-native memory system:

- use `search_memory` and `get_node` for retrieval and inspection
- use high-level tools such as `decide_memory_write`, `integrate_memory_with_sampling`, and `run_dreamer`
- rely on the connected MCP host/client to negotiate sampling when semantic decisions are needed
- let Synapse assemble evidence, validate structured decisions, and execute internal write semantics on the server side

The lower-level execution primitives still exist internally, but they are no longer part of the public agent contract.

See:

- [Usage guide](docs/usage.md)
- [Configuration guide](docs/configuration.md)
- [Overview and philosophy](docs/overview.md)
- [Active design index](docs/design/README.md)

## Repository layout

```
Synapse/
├── config.toml                  # runtime configuration
├── pyproject.toml               # package metadata & dependencies
├── synapse/                     # main Python package
│   ├── cli.py                   # Typer CLI (serve, status, rebuild-index, …)
│   ├── config.py                # configuration model & loader
│   ├── indexing.py              # full-index rebuild orchestration
│   ├── interfaces.py            # shared protocols (Embedding, Reranker)
│   ├── models/
│   │   └── node.py              # Node / NodeMetadata Pydantic models
│   ├── embedding/
│   │   └── engines.py           # embedding & reranker engine implementations
│   ├── retrieval/
│   │   └── pipeline.py          # hybrid retrieval pipeline (FTS + vector + rerank)
│   ├── storage/
│   │   ├── markdown.py          # Markdown ↔ Node serialization & file I/O
│   │   └── sqlite.py            # SQLite index store (nodes, edges, vectors, FTS)
│   ├── sync/
│   │   └── manager.py           # file-watch ↔ SQLite delta sync
│   ├── server/
│   │   ├── app.py               # Streamable HTTP transport (session, SSE)
│   │   ├── mcp.py               # MCP tool registry & dispatch
│   │   ├── sampling.py          # sampling/createMessage client
│   │   ├── schemas.py           # request/response Pydantic schemas
│   │   ├── service.py           # core service layer (search, write, lifecycle)
│   │   ├── write_path.py        # canonical write execution (create/supersede/complement)
│   │   ├── streamable.py        # Starlette app factory & server entrypoint
│   │   └── streamable_runtime.py # session manager, tool orchestrator
│   ├── lifecycle/
│   │   ├── condensation.py      # archive condensation logic
│   │   └── dreamer.py           # Dreamer lifecycle pipeline
│   ├── deployment/
│   │   └── service_manager.py   # launchd / systemd service install
│   ├── security/
│   │   └── sanitization.py      # input sanitization
│   └── utils/
│       ├── documents.py         # document helpers
│       ├── logging.py           # structured logging setup
│       └── runtime.py           # runtime directory bootstrap
├── tests/                       # pytest test suite
└── docs/                        # user-facing documentation
    ├── overview.md              # philosophy & architecture overview
    ├── configuration.md         # config reference
    ├── usage.md                 # usage guide
    ├── deployment.md            # deployment & service management
    └── design/                  # architecture design docs & archive
```

## Quick start

Requirements:

- Python 3.11+
- macOS or Linux for daemon integration
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) for local inference

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Basic commands:

```bash
python -m synapse version
python -m synapse status
python -m synapse rebuild-index
python -m synapse serve --run-server
pytest
```

### Local inference with llama.cpp

Synapse uses **llama.cpp** (`llama-server`) for both embedding and reranking.
**Ollama is not suitable** — it exposes `/api/embed` but has no rerank endpoint, so the reranker silently falls back to a deterministic lexical scorer.

You will need two model files in GGUF format:

| Role | Model | Endpoint |
|---|---|---|
| Embedding | `bge-m3` | `http://127.0.0.1:47860/v1/embeddings` |
| Reranker | `bge-reranker-v2-m3` | `http://127.0.0.1:47861/v1/rerank` |

Start both servers before running Synapse:

```bash
# embedding
llama-server -m ~/models/bge-m3.gguf \
  --embeddings --port 47860 --host 127.0.0.1 &

# reranker  (--rerank --pooling rank are required)
llama-server -m ~/models/bge-reranker-v2-m3.gguf \
  --rerank --pooling rank --port 47861 --host 127.0.0.1 &
```

On macOS you can set them to launch automatically at login by placing launchd plists in `~/Library/LaunchAgents/` and loading them with `launchctl load`.

Synapse talks to both servers as plain HTTP clients. It does not manage the llama-server processes.

## Project status

The implementation plan in `docs/design/` has been completed end to end.

Synapse now includes the full local memory pipeline, interface layer, lifecycle system, and service integration.

## Guiding principle

Synapse is designed to stay understandable even when something goes wrong.

You should always be able to inspect the Markdown files, rebuild the index, and recover working memory without mystery.
