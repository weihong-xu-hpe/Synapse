# TODO: Dreamer Stats & Configurable Thresholds

> 状态：Draft for review
> 日期：2026-08-31
> 背景：当前 Dreamer 和写入路径已有可工作的保守默认值，但缺少长期可观测数据来判断 `0.9` / `0.8` / `0.75` 等门槛是否适合本地 Mac + 当前 embedding/reranker/decider 组合。
> 关联：`TODO-local-llm-upgrade.md` Phase 1（检索门槛修复）、Phase 3（Dreamer 自动巡逻）、Phase 4（OKF 结构约束）。

## 背景

Synapse 当前的记忆生命周期已经收敛为：

- `write_memory`：每次写入前检索候选，并由 Decider 选择 `create` / `supersede` / `complement`。
- Dreamer：定时扫描 stale orphan、missing-link pairs、disputed pairs 和 superseded nodes，执行 triage、link weaving、conflict resolution、archive/condense。
- `stats` / `status`：目前只返回系统健康和节点数量快照。

现有阈值整体偏保守，适合作为安全默认值，但不适合长期写死：

| 阈值 / 门槛 | 当前位置 | 当前值 | 问题 |
|---|---|---:|---|
| 写入候选相似度 | `WriteMemoryRequest.similarity_threshold` / `search_existing_nodes` | `0.3` | 已从 `0.5` 降低，但不同 reranker 分布仍可能漂移 |
| Dreamer missing-link cosine | `Dreamer.run()` → `find_missing_link_pairs(cosine_threshold=0.75)` | `0.75` | 硬编码；不同 embedding 模型、节点规模下召回差异大 |
| stale orphan 天数 | `[decay].janitor_days` | `30` | 同时被 cleanup 和 link-weaving recency 复用，语义混杂 |
| superseded 归档天数 | `find_superseded_for_archival(days_threshold=7)` | `7` | 硬编码，不方便观察后调整 |
| low-structure 字符数 | `build_triage_prompt()` | `100` | 写在 prompt 文案中，不可观测、不可配置 |

这些默认值不是错误，但应该被长期 stats 校准。否则调门槛只能凭感觉，容易在“召回太低”和“噪音太多”之间来回摆。

## 目标

1. 暴露一个可长期观察的 lifecycle/write stats 节点，用于判断当前阈值是否合适。
2. 持久化 Dreamer 每轮运行摘要，支持 all-time / recent-window 聚合。
3. 把最关键的 Dreamer 阈值从硬编码变为配置项。
4. 不改变现有默认行为：默认值保持保守，先观测，再调参。
5. 不把观测接口扩张成新的 public MCP 主路径；优先复用现有 `stats` / CLI `status` 能力。

## 非目标

- 不在本阶段实现“每个请求自动抽取多条 memory”的新 extractor。
- 不改变 `write_memory` 一次最多创建一个 node 的现有契约。
- 不用 stats 自动调参；本阶段只提供观测和配置入口。
- 不在 metrics 中保存 memory 正文、prompt、LLM 响应全文，避免隐私和体积风险。

## 设计原则

- **先观测，再调参。** 默认阈值保持不变，避免在没有数据时盲目放宽。
- **指标要形成漏斗。** 不只记录最终写了多少，还记录每一层过滤后剩多少。
- **只存结构化计数。** 长期 stats 不存内容，只存 counts、duration、decision 类型和 warning code。
- **配置命名按业务语义。** cleanup days 和 link-weaving recency days 不应共用一个字段。
- **默认行为兼容。** 没有配置新字段时，运行结果与当前实现一致。

## Proposed Stats Surface

### Service payload

扩展 `SynapseServerService.stats()` 返回值，新增 `lifecycle_stats` 和 `write_stats`：

```json
{
  "status": "healthy",
  "components": { "sqlite": "ok" },
  "stats": {
    "total_nodes": 123,
    "active_nodes": 100,
    "superseded_nodes": 10,
    "disputed_nodes": 2,
    "archived_nodes": 11
  },
  "lifecycle_stats": {
    "thresholds": {
      "missing_link_cosine": 0.75,
      "stale_orphan_days": 30,
      "link_weaving_recency_days": 30,
      "superseded_archive_days": 7,
      "low_structure_chars": 100
    },
    "current_candidates": {
      "stale_orphans": 4,
      "missing_link_pairs": 18,
      "disputed_pairs": 1,
      "superseded_archival_candidates": 3
    },
    "runs": {
      "total": 42,
      "last_24h": 2,
      "last_7d": 14,
      "last_30d": 42,
      "avg_duration_ms": 8320,
      "avg_triage_decisions": 2.1,
      "avg_links_added": 0.8,
      "avg_condensed": 0.2,
      "avg_archived": 1.4
    },
    "decision_totals": {
      "triage_keep": 12,
      "triage_condense": 8,
      "triage_archive": 33,
      "links_added": 21,
      "conflicts_superseded": 5,
      "conflicts_both_valid": 2,
      "warnings": 3,
      "sampling_failures": 1
    }
  },
  "write_stats": {
    "requests_total": 96,
    "candidate_count_avg": 3.4,
    "candidate_count_zero_rate": 0.18,
    "decision_totals": {
      "create": 64,
      "supersede": 11,
      "complement": 21
    },
    "warnings": {
      "low_structure": 7
    },
    "sampling_failures": 2
  }
}
```

### CLI surface

`synapse status` 当前已经展示节点数量。新增一个简洁区块即可：

```text
Lifecycle stats:
  Current candidates: stale=4, missing_links=18, disputed_pairs=1, superseded_archive=3
  Dreamer runs: total=42, last_7d=14, avg_duration=8.3s
  Decisions: keep=12, condense=8, archive=33, links_added=21
  Thresholds: missing_link_cosine=0.75, link_recency_days=30, low_structure_chars=100

Write stats:
  Requests: 96, candidates avg=3.4, zero-candidate rate=18%
  Decisions: create=64, supersede=11, complement=21
```

默认 `status` 保持可读摘要；后续可加 `--json` 或 `synapse stats --json` 输出完整结构。

## Metrics Model

### `dreamer_runs`

每次 Dreamer 完成后写一条 summary。

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Run id |
| `started_at` | TEXT | UTC timestamp |
| `completed_at` | TEXT | UTC timestamp |
| `duration_ms` | INTEGER | Run duration |
| `batch_size` | INTEGER | Configured/effective batch size |
| `stale_scanned` | INTEGER | Stage 1 stale orphan count |
| `superseded_scanned` | INTEGER | Stage 1 superseded archival candidates |
| `disputed_scanned` | INTEGER | Stage 1 disputed pairs count |
| `missing_link_pairs_scanned` | INTEGER | Stage 1 missing-link pairs count |
| `triage_keep` | INTEGER | LLM triage keep decisions |
| `triage_condense` | INTEGER | LLM triage condense decisions |
| `triage_archive` | INTEGER | LLM triage archive decisions |
| `links_added` | INTEGER | Executed link additions |
| `conflicts_superseded` | INTEGER | Conflict decisions that superseded one side |
| `conflicts_both_valid` | INTEGER | Conflict decisions that cleared both sides |
| `archived` | INTEGER | Actual archived source/superseded nodes |
| `condensed` | INTEGER | New condensation summary nodes created |
| `warnings` | INTEGER | Warning count |
| `sampling_failures` | INTEGER | Warning count where code ends with `_sampling_failed` |

### `write_memory_events`

每次 `write_memory` 完成后写一条 summary。

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Event id |
| `created_at` | TEXT | UTC timestamp |
| `node_id` | TEXT | Created node id, if execution succeeded |
| `node_type` | TEXT | `transient` / `persistent` |
| `action` | TEXT | `create` / `supersede` / `complement` |
| `candidate_count` | INTEGER | Candidate matches after threshold |
| `similarity_threshold` | REAL | Threshold used for this request |
| `warning_codes` | TEXT | JSON list, e.g. `["low_structure"]` |
| `sampling_provider` | TEXT | Decider/provider name |
| `execution_succeeded` | INTEGER | 0/1 |

If write execution fails after a valid decision, record the decision and failure flag before re-raising where practical. If failure happens before a decision, increment an aggregate failure row with `action = NULL`.

## Threshold Configuration

Add a nested Dreamer threshold section:

```toml
[dreamer]
enabled = true
interval_hours = 12
batch_size = 8

[dreamer.thresholds]
missing_link_cosine = 0.75
stale_orphan_days = 30
link_weaving_recency_days = 30
superseded_archive_days = 7
low_structure_chars = 100
max_missing_link_pairs_per_run = 100
```

Compatibility rules:

- If `[dreamer.thresholds].stale_orphan_days` is absent, default to `[decay].janitor_days`.
- If `[dreamer.thresholds].link_weaving_recency_days` is absent, default to `[decay].janitor_days`.
- Keep `[decay].janitor_days` for retrieval/decay compatibility until a later cleanup decides whether it should be fully retired or renamed.
- Keep `missing_link_cosine = 0.75` as the default. The point is observability-driven tuning, not changing behavior in the same patch.
- Use `max_missing_link_pairs_per_run` as a safety valve so dense embedding distributions do not cause one Dreamer run to send tens of thousands of pair decisions to the Decider.

### Why split stale and link recency?

Today `janitor_days` controls both:

1. How old an unlinked active node must be before stale cleanup considers it.
2. How recently an active node must have been accessed before link weaving considers it.

These are opposite semantic directions. Splitting them makes tuning safer:

- Lower `stale_orphan_days` means cleanup sees old orphans sooner.
- Higher/lower `link_weaving_recency_days` changes how much recent working set is eligible for link discovery.

## Missing-Link Threshold Histogram

To tune `0.9` / `0.8` / `0.75`, final counts are not enough. Add a read-only histogram helper that evaluates current active recent node pairs at fixed buckets:

```json
"missing_link_similarity_histogram": {
  "0.70": 42,
  "0.75": 18,
  "0.80": 7,
  "0.85": 3,
  "0.90": 0
}
```

Interpretation:

- If `0.90` is always `0`, then `0.9` is too high for the current embedding distribution.
- If `0.75` has many candidates but `links_added` remains near `0`, then LLM judgment is filtering them and the threshold may be acceptable.
- If `0.75` has few candidates but manual inspection finds many related memories, the threshold or embedding model likely needs adjustment.
- If candidate count explodes, cap `max_missing_link_pairs_per_run` before lowering the threshold further.

This helper should avoid storing pair identities in long-term metrics. It can compute aggregate buckets at `stats()` time or snapshot bucket counts into `dreamer_runs`.

## Implementation Plan

### Phase 1: Persist Dreamer run summaries

| File | Change |
|---|---|
| `synapse/storage/sqlite.py` | Add schema table `dreamer_runs`; add `record_dreamer_run(...)` and aggregation query helpers. |
| `synapse/lifecycle/dreamer.py` | After report construction, persist a run summary. If metrics persistence fails, emit a warning/log but do not fail the Dreamer run. |
| `tests/test_sqlite_store.py` | Cover insert + aggregate behavior using real SQLite. |
| `tests/test_dreamer.py` | Cover that a completed Dreamer run records one metrics row. |

### Phase 2: Extend `stats()` and CLI status

| File | Change |
|---|---|
| `synapse/indexing.py` | Add lifecycle stats collection: current candidate counts, thresholds, run aggregates. |
| `synapse/server/service.py` | Include `lifecycle_stats` in `stats()` payload. |
| `synapse/cli.py` | Add human-readable lifecycle/write stats to `synapse status`. |
| `tests/test_server_api.py` / `tests/test_cli.py` | Assert new fields are present without breaking existing fields. |

### Phase 3: Persist write-path summaries

| File | Change |
|---|---|
| `synapse/storage/sqlite.py` | Add `write_memory_events` table and aggregate helpers. |
| `synapse/server/service.py` | Record `write_memory` decision summaries after successful execution; record failures where safe. |
| `tests/test_server_api.py` | Cover create/supersede/complement counters and low-structure warning counters. |

### Phase 4: Make thresholds configurable

| File | Change |
|---|---|
| `synapse/config.py` | Add `DreamerThresholdSettings` nested under `DreamerSettings`. |
| `config.toml` | Add `[dreamer.thresholds]` defaults. |
| `synapse/lifecycle/dreamer.py` | Replace hardcoded `0.75` and `7`; use separate stale and link recency days. |
| `synapse/server/sampling.py` | Parameterize low-structure prompt threshold, or add a prompt builder argument with default `100`. |
| `docs/configuration.md` | Document new threshold fields and tuning workflow. |
| Tests | Cover default compatibility and configured overrides. |

## Acceptance Criteria

- `synapse status` shows lifecycle stats without requiring the server to be running.
- `SynapseServerService.stats()` returns `lifecycle_stats.thresholds`, `current_candidates`, and run aggregates.
- Dreamer runs continue to succeed even if metrics persistence fails.
- Existing tests that assert base node counts keep passing.
- Default config produces the same Dreamer candidate behavior as today.
- Tests cover at least one configured `missing_link_cosine` override and one configured `superseded_archive_days` override.
- No persisted metrics row contains memory content, prompt text, or full LLM output.

## Tuning Workflow

1. Start with defaults.
2. Let Synapse run for several days or manually run Dreamer after representative usage.
3. Inspect:
   - `candidate_count_zero_rate` for write path recall.
   - `missing_link_similarity_histogram` for link-weaving thresholds.
   - `triage_keep` / `triage_condense` / `triage_archive` ratio for cleanup aggressiveness.
   - `links_added / missing_link_pairs_scanned` for LLM acceptance rate.
4. Tune one threshold at a time.
5. Keep the previous stats window for comparison before changing another threshold.

Recommended first tuning questions:

- Is `missing_link_pairs_scanned` usually `0`? Consider lowering `missing_link_cosine` or increasing `link_weaving_recency_days`.
- Is `missing_link_pairs_scanned` high but `links_added` near `0`? Keep the threshold; the LLM is already filtering aggressively.
- Is `candidate_count_zero_rate` high for writes? Lower `similarity_threshold`, improve `query_hint`, or revisit query generation.
- Is `triage_archive` much higher than `keep + condense`? Review the triage prompt before lowering cleanup age.

## Open Questions

1. Should stats be returned only from service/CLI, or should a read-only `get_stats` MCP tool be added later?
   - Current recommendation: no new MCP tool yet; keep public MCP surface at `search_memory` / `write_memory`.
2. Should histogram buckets be fixed (`0.70`, `0.75`, `0.80`, `0.85`, `0.90`) or configurable?
   - Current recommendation: fixed buckets first; configurable buckets only if real usage needs it.
3. Should metrics live in the main `synapse.db` or a separate observability DB?
   - Current recommendation: main DB first, because this is local-only lightweight metadata and easier to query from `status`.

## Decision Summary

The current hardcoded thresholds are acceptable as defaults, but not as permanent hidden behavior. Add a long-term stats surface first, then expose the most important Dreamer thresholds as config. This keeps Synapse safe by default while making Mac-local tuning evidence-based instead of vibe-based.