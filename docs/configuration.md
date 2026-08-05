# Synapse Configuration

[Back to README](../README.md)

## Config resolution

By default, Synapse loads `config.toml` from the current working directory.

You can override that with either:

- `--config /path/to/config.toml`
- `SYNAPSE_CONFIG_PATH=/path/to/config.toml`

## Runtime directories

Synapse creates and uses these directories under the configured memory base path:

- `.synapse/active`
- `.synapse/.archive`
- `.synapse/.logs`
- `.synapse/.audit`

## Default config example

```toml
[server]
host = "0.0.0.0"
port = 8765
cors_allowed_origins = ["*"]
auth_token = ""

[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "remote_api"
model = "bge-m3"
dimension = 1024
timeout_seconds = 30

[reranker]
provider = "remote_api"
model = "bge-reranker-v2-m3"
max_candidates = 9
timeout_seconds = 30

# Both embedding and reranking use [providers.remote_api].
# Set embedding_base_url if embedding and reranker run on separate ports.
[providers.remote_api]
base_url = "http://127.0.0.1:47861"
embedding_base_url = "http://127.0.0.1:47860"
embedding_endpoint = "/v1/embeddings"
rerank_endpoint = "/v1/rerank"
api_key_env = ""
headers = {}
request_timeout_seconds = 30

[retrieval]
engine = "sqlite"
rrf_k = 60
top_k = 3

[decay]
factor = 0.98
janitor_days = 30
archive_retention_days = 90

[sanitization]
custom_patterns = []

[logging]
retention_days = 7
max_file_size_mb = 50
log_dir = "./.synapse/.logs"

[decider]
provider = "local_llm"
base_url = "http://localhost:8000/v1"
model = "deepseek-v4-pro"
api_key_env = ""
fallback_base_url = ""
fallback_model = ""
fallback_api_key_env = "OPENAI_COMPATIBLE_API_KEY"
timeout_seconds = 30
max_tokens = 600
temperature = 0.1

[dreamer]
enabled = true
interval_hours = 12
batch_size = 8
```

## Section guide

### MCP-first public interface

Synapse's public interface is centered on high-level MCP tools such as:

- `search_memory`
- `write_memory`
- `run_dreamer`

This high-level set is the default public exposure profile.

Behind those tools, Synapse still uses the same low-level write contract internally. The configuration difference is mostly about the **client side**, not the Synapse storage or retrieval stack.

Configuration requirements:

- configure Synapse normally for retrieval and storage
- connect through an MCP client/host that advertises `capabilities.sampling` during `initialize`
- ensure the MCP client sends `notifications/initialized`
- use a transport where the client can answer `sampling/createMessage`

Important limitation:

- sampling is not enabled by a static `config.toml` flag
- there is no separate CLI transport selector anymore; Synapse is converging on a single public server runtime

Sampling model preference defaults are speed-oriented:

- `gemini-3-flash`
- `claude-4.5-haiku`

Minimal setup checklist:

1. fill out normal Synapse runtime config in `config.toml`
2. start Synapse with `python -m synapse serve --run-server`
3. ensure the MCP client advertises `capabilities.sampling`
4. ensure the MCP client can answer `sampling/createMessage`

### `[server]`

Controls the Synapse server.

- `host`, `port` — server bind address and port
- `cors_allowed_origins` — CORS policy for HTTP clients
- `auth_token` — optional bearer token for server access

If `auth_token` is blank, the HTTP API is open for local development.

This section does **not** enable or disable MCP sampling. Sampling capability is negotiated dynamically by the MCP client.
It also does not currently define the MCP tool exposure profile in `config.toml`; the default code-level profile is the public high-level surface.

### `[memory]`

Controls local storage paths.

- `base_path` — runtime base directory
- `archive_path` — archive directory for lifecycle eviction

### `[embedding]`

Controls embedding generation.

- `provider` — `remote_api` or `builtin`
- `model` — selected embedding model
- `dimension` — embedding vector dimension
- `timeout_seconds` — request timeout

### `[reranker]`

Controls reranking.

- `provider`
- `model`
- `max_candidates`
- `timeout_seconds`

### `[providers.remote_api]`

Settings for any OpenAI-compatible HTTP inference server (llama.cpp, hosted APIs, etc.).

- `base_url` — reranker server base URL
- `embedding_base_url` — optional override for embedding server; uses `base_url` if not set
- `embedding_endpoint`
- `rerank_endpoint`
- `api_key_env` — env var name holding the bearer token; leave blank for local servers
- `headers` — additional static headers
- `request_timeout_seconds`

When running both models locally on separate ports, set `embedding_base_url` to the embedding server and `base_url` to the reranker server.

### `[retrieval]`

Controls hybrid retrieval.

- `engine` — current retrieval backend (`sqlite` by default)
- `rrf_k` — reciprocal rank fusion constant
- `top_k` — maximum result count in final context

### `[decay]`

Controls forgetting behavior.

- `factor` — retrieval-time decay multiplier applied per day since last access
- `janitor_days` — age threshold for orphan eviction
- `archive_retention_days` — how long archived nodes are retained before cleanup

### `[sanitization]`

Custom regex redaction rules for remote-bound payloads.

### `[logging]`

Controls local log retention and file-size limits.

### `[decider]`

Controls the local LLM used for sampling decisions (write-path triage, link weaving, conflict resolution). The Decider is an OpenAI-compatible chat-completions client with primary + fallback endpoints.

- `provider` — currently `local_llm` (OpenAI-compatible HTTP)
- `base_url` — primary LLM endpoint base URL (e.g. `http://localhost:8000/v1`); Synapse POSTs to `{base_url}/chat/completions`
- `model` — primary model name
- `api_key_env` — env var name holding the bearer token for the primary endpoint; leave blank for no-auth local servers
- `fallback_base_url` — fallback LLM endpoint base URL; used when the primary endpoint fails
- `fallback_model` — fallback model name
- `fallback_api_key_env` — env var name holding the bearer token for the fallback endpoint
- `timeout_seconds` — HTTP request timeout per endpoint (raise this for reasoning models that emit `reasoning_content` before the final answer)
- `max_tokens` — generation token cap sent to the LLM; must be large enough for reasoning models (e.g. glm-5.2-fp8) whose `reasoning_content` consumes tokens before `content` is populated
- `temperature` — sampling temperature

> **Reasoning-model note:** some models (e.g. `glm-5.2-fp8`) return intermediate reasoning in a `reasoning_content` field and only populate `content` after reasoning completes. If `max_tokens` is too small, `content` stays empty and the Decider raises a parse error. Size `max_tokens` to leave headroom for both reasoning and the final structured answer.

### `[dreamer]`

Controls the Dreamer — Synapse's background memory consolidation engine (modeled on sleep neuroscience: scan → NREM triage → REM link weaving → conflict resolution → execute → report).

- `enabled` (bool, default `true`) — whether the in-process scheduler auto-starts when the server runs
- `interval_hours` (int, ≥1, default `12`) — interval between automatic Dreamer runs in hours
- `batch_size` (int, 1-20, default `8`) — number of nodes or node pairs sent to the Decider per batch

#### How Dreamer triggers

Dreamer runs are scheduled **in-process** by `DreamerScheduler` (a daemon `threading.Timer`), not by launchd or cron:

- **Automatic:** when `enabled = true`, `synapse serve --run-server` starts the scheduler, which fires a Dreamer run every `interval_hours`. A `flock`-based lock (`.synapse/.logs/dreamer.lock`) prevents overlapping runs.
- **Manual:** `python -m synapse dreamer run` runs one pass immediately (optional `--batch-size N`). This is useful for testing Decider wiring or forcing consolidation outside the schedule.

To change the cadence, edit `interval_hours` and restart the service. To disable automatic runs entirely, set `enabled = false` (manual `dreamer run` still works).

## Environment variables

The project includes an `.env` file for remote inference credentials:

```dotenv
SYNAPSE_MODEL_API_KEY=replace-me
```

Use this when `providers.remote_api.api_key_env` points to `SYNAPSE_MODEL_API_KEY`.

## Practical configuration patterns

### Public MCP agent setup

Use your normal `config.toml`, but run Synapse under a sampling-capable MCP host/client.

Recommended pattern:

1. start Synapse with `python -m synapse serve --run-server`
2. let the MCP client negotiate `sampling`
3. call high-level tools such as `write_memory`

This is the best fit when you want a thin agent layer and faster host-mediated semantic decisions.

### Fully local with llama.cpp (recommended)

Run two `llama-server` instances and point both providers at them:

```toml
[embedding]
provider = "remote_api"
model = "bge-m3"

[reranker]
provider = "remote_api"
model = "bge-reranker-v2-m3"

[providers.remote_api]
base_url = "http://127.0.0.1:47861"        # reranker
embedding_base_url = "http://127.0.0.1:47860"  # embedding
embedding_endpoint = "/v1/embeddings"
rerank_endpoint = "/v1/rerank"
api_key_env = ""
```

Startup commands (see README for launchd auto-start):

```bash
llama-server -m ~/models/bge-m3.gguf --embeddings --port 47860 --host 127.0.0.1 &
llama-server -m ~/models/bge-reranker-v2-m3.gguf --rerank --pooling rank --port 47861 --host 127.0.0.1 &
```

> **Note:** Ollama is not suitable — it has no `/rerank` endpoint, so reranking would silently fall back to a deterministic lexical scorer. Use llama.cpp for both.

### Hosted API

Point both at a hosted inference provider:

```toml
[embedding]
provider = "remote_api"

[reranker]
provider = "remote_api"

[providers.remote_api]
base_url = "https://models.example.com"
embedding_endpoint = "/v1/embeddings"
rerank_endpoint = "/v1/rerank"
api_key_env = "SYNAPSE_MODEL_API_KEY"
```

### Fully deterministic test workflow

Use `builtin` providers for both embedding and reranking in test or fallback scenarios.

## Notes

- If you switch embedding provider, model, or dimension, rebuild the index.
- The SQLite index is derived state; Markdown remains canonical.
