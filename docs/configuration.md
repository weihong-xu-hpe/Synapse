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
max_doc_tokens = 2048
timeout_seconds = 30

# Both embedding and reranking use [providers.remote_api].
# Set embedding_base_url if embedding and reranker run on separate ports.
[providers.remote_api]
base_url = "http://127.0.0.1:47861"
embedding_base_url = "http://127.0.0.1:47860"
embedding_endpoint = "/v1/embeddings"
rerank_endpoint = "/v1/rerank"
tokenize_endpoint = "/tokenize"
api_key_env = ""
headers = {}
request_timeout_seconds = 30

[retrieval]
engine = "sqlite"
rrf_k = 60
top_k = 3
inject_logit_floor = 0.25
inject_relative_margin = 2.5
inject_ignore_persistent_penalty = true

[decay]
factor = 0.98
janitor_days = 30
archive_retention_days = 90

[sanitization]
custom_patterns = []

[logging]
backup_count = 7
max_file_size_mb = 50
log_dir = "./.synapse/.logs"

[decider]
provider = "local_llm"
timeout_seconds = 30
max_tokens = 200000
endpoint_max_concurrency = 30
temperature = 0.0

[[decider.endpoints]]
base_url = "http://localhost:8000/v1"
model = "deepseek-v4-pro"
api_key_env = ""
# Optional per-endpoint JSON request fields; applied only when this endpoint is called.
# extra_body = { chat_template_kwargs = { enable_thinking = false } }

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
- `max_doc_tokens` — per-document truncation for rerank requests, in **real tokens** (default 2048). Token counts come from the provider's `tokenize_endpoint` when available; a conservative character-based estimate is used otherwise. Rerank latency scales with total candidate tokens; the first few hundred tokens carry the ranking signal. Must leave room for the query in the model context (at most context size − pair margin − minimum query tokens; validated at load).

### `[providers.remote_api]`

Settings for any OpenAI-compatible HTTP inference server (llama.cpp, hosted APIs, etc.).

- `base_url` — reranker server base URL
- `embedding_base_url` — optional override for embedding server; uses `base_url` if not set
- `embedding_endpoint`
- `rerank_endpoint`
- `tokenize_endpoint` — token-count endpoint used for exact request budgeting (llama.cpp `POST /tokenize`; default `/tokenize`). The embedding engine tokenizes against `embedding_base_url`, the reranker against `base_url`. Set to `""` to disable exact counting; budgets then use a conservative estimate (1 token per CJK character + 2 characters per other token) that never under-counts real token usage
- `api_key_env` — env var name holding the bearer token; leave blank for local servers
- `headers` — additional static headers
- `request_timeout_seconds`

When running both models locally on separate ports, set `embedding_base_url` to the embedding server and `base_url` to the reranker server.

### `[retrieval]`

Controls hybrid retrieval.

- `engine` — current retrieval backend (`sqlite` by default)
- `rrf_k` — reciprocal rank fusion constant
- `top_k` — maximum result count in final context
- `inject_logit_floor` (float, default `0.25`) — raw rerank-logit floor for the server-side `inject` decision on search results; results below it are never marked injectable
- `inject_relative_margin` (float, default `2.5`) — results must also sit within this margin of the query's best logit (trims far-below-best noise on long multi-part queries)
- `inject_ignore_persistent_penalty` (bool, default `true`) — compare persistent nodes' pre-penalty logit at the injection gate instead of the penalized `score`

### `[decay]`

Controls forgetting behavior.

- `factor` — retrieval-time decay multiplier applied per day since last access
- `janitor_days` — age threshold for orphan eviction
- `archive_retention_days` — how long archived nodes are retained before cleanup

### `[sanitization]`

Custom regex redaction rules for remote-bound payloads.

### `[logging]`

Controls local log rotation and file-size limits.

- `backup_count` (int, ≥1, default `7`) — number of rotated backup files kept per log file (a **file count**, not days; each file rotates when it reaches `max_file_size_mb`). The previous name `retention_days` is no longer accepted.
- `max_file_size_mb` (int, ≥1, default `50`) — rotation threshold per log file
- `log_dir` — directory for rotated log files (`synapse.log`, `mcp-daemon.log`, `file-watcher.log`, `audit.log`, plus `uvicorn.log` and `uvicorn-access.log` written by the HTTP server)

### `[decider]`

Controls the local OpenAI-compatible LLM pool used for sampling decisions (write-path triage, link weaving, conflict resolution) and distillation.

- `provider` — currently `local_llm` (OpenAI-compatible HTTP)
- `endpoints` — one or more `[[decider.endpoints]]` entries, each with `base_url`, `model`, optional `api_key_env`, and optional `extra_body`; `extra_body` is merged into only that endpoint's completion request, including after failover. It cannot override `model`, `messages`, `max_tokens`, or `response_format`.
- `endpoint_max_concurrency` (int, 1–64, default `30`) — maximum simultaneous HTTP requests to each endpoint; the cap is independent per endpoint
- `timeout_seconds` — HTTP timeout per endpoint for write-path decisions. Dreamer calls use `[distiller].llm_timeout_seconds`; episode calls use `[distiller.episode].timeout_seconds`.
- `max_tokens` (default `200000`) — generation token budget; keep it high enough for reasoning plus the structured answer rather than imposing a tight cap
- `temperature` (default `0.0`) — sampling temperature; zero is preferred for deterministic structured decisions
- `local_llm` transient failures (transport/disconnect/reset/timeout, HTTP 429/5xx, invalid JSON) get up to 3 attempts per endpoint with jittered 5s/20s backoff before endpoint failover; other 4xx and configuration errors are not retried.

> Configure the serving model name exactly as advertised by the endpoint. Leave `api_key_env` blank for no-auth local endpoints.

### `[dreamer]`

Controls the Dreamer — Synapse's background memory consolidation engine (modeled on sleep neuroscience: scan → NREM triage → REM link weaving → conflict resolution → execute → report).

- `enabled` (bool, default `true`) — whether the in-process scheduler auto-starts when the server runs
- `interval_hours` (int, ≥1, default `12`) — interval between automatic Dreamer runs in hours
- `batch_size` (int, 1-20, default `8`) — number of nodes or node pairs sent to the Decider per batch

#### `[dreamer.thresholds]`

Controls the tunable lifecycle thresholds used by Dreamer scan and triage. Defaults preserve the original conservative behavior.

- `missing_link_cosine` (float, 0.0-1.0, default `0.75`) — cosine similarity required before an unlinked recent node pair is sent to link weaving
- `stale_orphan_days` (int, default `[decay].janitor_days`) — age threshold for unlinked active nodes to enter stale orphan triage
- `link_weaving_recency_days` (int, default `[decay].janitor_days`) — recency window for active nodes considered by missing-link discovery
- `superseded_archive_days` (int, default `7`) — age threshold before superseded nodes can be archived
- `low_structure_chars` (int, default `100`) — triage prompt threshold below which sectionless notes are treated as low-structure
- `max_missing_link_pairs_per_run` (int, default `100`) — safety cap for how many missing-link pairs a single Dreamer run sends to the Decider

Use `synapse status` to inspect the current lifecycle candidate counts, Dreamer run aggregates, write-path decision ratios, and effective threshold values before tuning these settings.

#### How Dreamer triggers

Dreamer runs are scheduled **in-process** by `DreamerScheduler` (a daemon `threading.Timer`), not by launchd or cron:

- **Automatic:** when `enabled = true`, `synapse serve --run-server` starts the scheduler, which fires a Dreamer run every `interval_hours`. A `flock`-based lock (`.synapse/.logs/dreamer.lock`) prevents overlapping runs.
- **Manual:** `python -m synapse dreamer run` runs one pass immediately (optional `--batch-size N`). This is useful for testing Decider wiring or forcing consolidation outside the schedule.

To change the cadence, edit `interval_hours` and restart the service. To disable automatic runs entirely, set `enabled = false` (manual `dreamer run` still works).

### `[distiller]`

Controls the server-side transcript pipeline that converts session transcripts into typed OKF knowledge nodes (see `docs/okf.md`). The live path is the Two-Stage Episode Distiller, triggered by `/api/episode/close` or the idle-transcript fallback scheduler.

- `enabled` (bool, default `true`) — whether the in-process sweep scheduler auto-starts when the server runs
- `interval_minutes` (int, ≥1, default `10`) — sweep interval; short enough that macOS sleep drift is harmless
- `idle_minutes` (int, ≥1, default `60`) — a transcript must be untouched this long before it is distilled. The idle clock uses the transcript file mtime; stamping transcript state updates that mtime.
- `retention_days` (int, ≥1, default `45`) — current-revision transcripts are archived this many days after their last content update; undistilled or stale transcripts are never archived.
- `max_transcripts_per_run` (int, 1–5000, default `500`) — maximum idle transcripts selected by one scheduler sweep.
- `llm_timeout_seconds` (int, ≥1, default `300`) — shared long-stage timeout for Dreamer calls outside the episode pipeline; episode calls use their own `timeout_seconds`.
- `enforce_okf` (bool, default `false`) — reject persistent writes on fatal OKF validation codes instead of warning.

#### `[distiller.episode]` two-stage pipeline

The active episode pipeline dehydrates tool-polling noise, carries a Global User Spine across incremental runs, scans with a low-effort endpoint pool, and sends lessons through the gatekeeper and integrator. It runs on `/api/episode/close` and the server idle fallback.

- `scanner_endpoints` (optional list) selects Stage 1 models; `endpoints` is also accepted for the pool, while legacy `endpoint` remains the single-endpoint fallback. Missing scanner `reasoning_effort` defaults to `low`.
- `gatekeeper_endpoint` defaults to the `claude-haiku-5.5` model at `http://127.0.0.1:47870/v1`; it may be set to `null` to use that default.
- `chunk_chars` (int, ≥1,000, default `400000`) controls turn-boundary chunking for large transcripts.
- `max_tokens` (int, default `200000`) sets the output token budget for episode-model calls.
- `evidence_context_token_budget` (int, default `24000`) bounds evidence excerpts passed to the gatekeeper.
- `timeout_seconds` (int, default `300`) is the episode LLM request timeout.
- `integrate_lessons_per_call` (int, default `12`) and `retrieval_top_k_per_lesson` (int, default `5`) bound gatekeeper batches and retrieved evidence.
- Pending episodes with fewer than two dehydrated turns or fewer than 600 dehydrated characters are stamped without LLM calls.

`POST /api/episode/close` accepts `{"session_key":"…","reason":"idle|shutdown|switch"}`. It returns `status=disabled` when `[distiller].enabled` is false; otherwise it returns `status=accepted` and schedules per-session serialized background distillation. The fallback scheduler selects transcripts idle for `idle_minutes` (default 60) on the configured `interval_minutes` cadence.

Commands: `python -m synapse distill status` and `python -m synapse distill episode` for an explicitly selected copy.

`distill status` prints the active scanner and gatekeeper models plus recent run creates, updates, and gatekeeper-filtered items.


### `[observability]`

Controls the local usage-analytics tables (schema v7) and the lookback reports. All data stays in the local SQLite DB / private JSONL files — nothing is transmitted.

- `search_events_retention_days` (int, ≥1, default `180`) — `search_events` rows older than this are pruned on each write (one cheap DELETE; indexes keep it bounded)
- `eval_history_path` — JSONL history of eval runs (default `~/.synapse/eval/history.jsonl`); relative paths resolve against the config file's directory
- `audit_history_path` — JSONL history of injection audits (default `~/.synapse/eval/audit-history.jsonl`)

What gets recorded:

- **`search_events`** (one row per `search_memory` call): timestamp, `source` (`bridge` when the REST call carries `exclude_session_key`, else `rest`; `mcp` for the MCP tool), sha1-prefix `session_hash`, query text + char count + CJK ratio, top_k, include, latency, and the returned results as JSON (rank, node_id, type, okf_type, rerank_logit, score, inject). Eval runs call `RetrievalPipeline.search` directly and are **never recorded** — eval stays out of usage statistics by construction.
- **`metrics_snapshots`** (one row per local day, written by the distiller sweep; idempotent per day with 2-day backfill for slept-through boundaries; `python -m synapse metrics snapshot` runs it manually): node counts by status/type/okf_type/project and active-persistent `verified` status (`verified_true`, `verified_false`, `verified_unset`), transcript distillation state, the day's write mix by route + decider action, distiller yield (including creates, updates, quarantine, and gatekeeper-filtered lessons), search stats (count by source, p50/p95 latency, inject rate, zero-inject share, top injected node ids), Dreamer runs, and DB/active/archive sizes.
- **`write_memory_events.route`** (new column): write attribution — `mcp`, `rest`, `session_upsert`, `distiller`, `write_node`. The dedupe-guard `unchanged` early return now also records an event (it previously wrote nothing, undercounting identical writes). Legacy rows have `route = NULL`.

Commands:

- `python -m synapse metrics snapshot [--backfill-days N]` — write/print the daily snapshot
- `python -m synapse report --since 30d [--out file.md]` — markdown lookback (weekly search/latency buckets, inject rate per source, zero-injection sessions, most/least injected knowledge, knowledge growth with new-node verification breakdown, write mix, decider actions + downgrades, the Episode Distiller funnel, Dreamer yield, storage growth, eval trend from eval history JSONL, and injection-audit history). Works with partial/empty data.
- `python -m synapse audit injections --since 30d --sample 30 [--report path]` — samples injected results (`inject=true`, sources bridge/mcp) from `search_events`, asks the configured `[decider]` LLM to judge relevance (0 irrelevant / 1 related / 2 directly useful + one-line reason), prints a Wilson-interval precision estimate (per source and per okf_type), lists the worst cases, and appends a summary line to `audit_history_path`.

Monthly schedule (macOS): the `synapse install --service` flow does not manage the audit job; generate and load it explicitly:

```sh
python - <<'PY'
from synapse.config import load_config
from synapse.deployment.service_manager import ServiceManager
config = load_config()  # SYNAPSE_CONFIG_PATH or ./config.toml
print(ServiceManager(config).install_audit_launchd())
PY
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.synapse.audit.plist
```

The agent runs `python -m synapse audit injections --since 30d --sample 30` on day 1 of each month at 10:00, logging to `~/.synapse/.synapse/.logs/audit-job.log` (relative to the configured log dir).

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
