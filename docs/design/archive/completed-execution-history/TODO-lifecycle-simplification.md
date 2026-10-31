# TODO: Lifecycle Simplification — Tier Removal & Dreamer Pipeline

**Status**: Draft for review
**Date**: 2026-03-16

## Motivation

Three design problems with the current lifecycle system:

1. **Note/Memory tier is redundant with decay.** The system manually labels records at write time (note = short-lived, memory = long-lived), but the decay + janitor mechanism already handles this organically. A record that keeps getting accessed survives; one that doesn't, dies. Pre-labeling duplicates what runtime observation already does.

2. **`importance` field has no writer.** It defaults to 0.5 everywhere. No LLM or human ever sets it to a different value. Dead field.

3. **Lifecycle tools are disconnected from the janitor.** The janitor runs automatically but does only dumb archival (orphan + superseded). The smart lifecycle tools (`review_memory_cluster`, `condense_memory_cluster`, `promote_memory_candidate`) exist as standalone MCP tools, but nothing orchestrates calling them. The "sleep phase" pipeline described in the original design doesn't exist.

## Design Principles

- **All records are equal at write time.** One word limit (3500), one decay curve.
- **Longevity is emergent, not declared.** A record that survives is a memory. One that decays is a note. This is observed after the fact, never labeled before.
- **Dreamer is the single entry point for all lifecycle work.** One MCP tool call (`run_dreamer`) starts the entire sleep cycle: scan → triage → weave links → resolve conflicts → execute → report.
- **No human intervention during the pipeline.** It runs autonomously. The output is a structured report returned to the caller.
- **Sleep-inspired architecture.** Modelled on human sleep memory consolidation: synaptic downscaling (decay), NREM selective consolidation (triage), and REM associative weaving (link discovery + conflict resolution).

## Changes

### Phase 1: Remove `importance` field

The field is inert — always 0.5, never updated. Remove it from the data model, SQL schema, and all downstream references.

**Files to touch:**

| File | Change |
|------|--------|
| `synapse/models/node.py` | Remove `importance` from `NodeMetadata`. Remove any serialization/validation logic for it. |
| `synapse/storage/sqlite.py` | Remove `importance` column from `nodes` table DDL, INSERT/UPSERT, and SELECT statements. |
| `synapse/lifecycle/condensation.py` | Remove `_condensed_importance` method and any importance averaging. |
| `synapse/server/service.py` | Remove `importance=0.5` from `integrate_knowledge` and anywhere else the field is set. |
| `tests/test_lifecycle.py` | Remove importance from test node factories. |
| `tests/test_sqlite_store.py` | Remove importance from test fixtures. |
| `tests/test_markdown_node_model.py` | Remove importance assertions. |

**Migration note:** Existing `.md` files with `importance:` in frontmatter should be tolerated on read (parsed but ignored) and stripped on next write. No breaking migration needed — Pydantic's `model_config = ConfigDict(extra="ignore")` or a default-field approach handles this gracefully.

### Phase 2: Remove `NodeTier` — unify to single record type

All records get one word limit: 3500 words. The `tier` field is removed from frontmatter, model, and SQL.

**Files to touch:**

| File | Change |
|------|--------|
| `synapse/models/node.py` | Remove `NodeTier` enum. Remove `TIER_WORD_LIMITS` dict. Simplify `validate_tier_word_count` to a single 3500-word check. Keep `WordCountValidation` but remove tier-dependency. |
| `synapse/config.py` | `DecaySettings`: replace `note_factor` / `memory_factor` with single `decay_factor: float`. Replace `note_janitor_days` / `memory_janitor_days` with single `janitor_days: int`. Remove `get_factor()` tier dispatch. |
| `config.toml` | Simplify `[decay]` section to `factor = 0.98`, `janitor_days = 30`, `archive_retention_days = 90`. |
| `synapse/storage/sqlite.py` | Remove `tier` column from DDL. Remove tier index. Update all INSERT/UPSERT/SELECT. `find_orphan_candidates()` no longer takes a tier param — just uses `janitor_days`. |
| `synapse/retrieval/pipeline.py` | `apply_decay` uses `config.decay.decay_factor` directly, no tier dispatch. |
| `synapse/lifecycle/janitor.py` | Single `find_orphan_candidates(janitor_days)` call instead of per-tier. |
| `synapse/lifecycle/condensation.py` | Remove tier references in condensation draft creation. Default tier for synthesized nodes no longer needed. |
| `synapse/server/schemas.py` | Remove `tier` from all request models. Remove `NodeTier` import. Remove `target_tier` field from `CondenseMemoryClusterRequest`. |
| `synapse/server/service.py` | Remove tier normalization from `integrate_knowledge`, `write_memory`, etc. |
| `synapse/server/sampling.py` | Remove tier from `MemoryWriteSamplingRequest`, prompt builders. |
| `synapse/server/mcp.py` | Update tool input schemas (tier param removal). |
| All test files | Update factories, assertions, config fixtures. |
| `docs/overview.md` | Rewrite "Tiers" section. |
| `docs/configuration.md` | Update `[decay]` section. |

**Migration note:** Existing frontmatter with `tier: note` or `tier: memory` should be ignored on read. The sync layer already rebuilds from file content, so a `rebuild-index` (or natural delta sync) handles the SQL side.

### Phase 3: Replace lifecycle tools with Dreamer pipeline

Rename `synapse/lifecycle/janitor.py` → `synapse/lifecycle/dreamer.py`. Replace three standalone MCP tools (`review_memory_cluster`, `condense_memory_cluster`, `promote_memory_candidate`) and the old `NightlyJanitor` with a single sleep-cycle pipeline.

**Dreamer pipeline design (modelled on human sleep stages):**

```
run_dreamer()  ← single MCP tool, requires sampling
  │
  ├─ Stage 1: Scan (local, no LLM) — "waking inventory"
  │   • Stale candidates: last_accessed > janitor_days AND in_degree = 0
  │   • Superseded candidates: status = superseded AND last_accessed > 7d
  │   • Disputed nodes: status = disputed
  │   • Missing-link pairs: recent active nodes with high cosine similarity
  │     but no wiki-link edge between them
  │   Result: structured candidate lists with node summaries + content.
  │
  ├─ Stage 2: Triage (sampling) — "NREM slow-wave consolidation"
  │   Batch stale candidates (5-10 per sampling request).
  │   Send node content + context to host LLM via sampling/createMessage.
  │   Prompt asks, per candidate: keep / condense / archive.
  │   LLM returns structured JSON decisions with reasoning.
  │   Validation: reject malformed decisions, fall back to archive.
  │
  ├─ Stage 3: Link Weaving (sampling) — "REM associative dreaming"
  │   Batch missing-link pairs (5-10 per sampling request).
  │   Prompt: "These node pairs are semantically close but unlinked.
  │            For each: link (should reference each other) / independent."
  │   For 'link' decisions → insert [[wiki-link]] in both nodes' markdown.
  │   This directly improves retrieval quality: graph-hop can now reach
  │   related nodes that were previously invisible to each other.
  │
  ├─ Stage 4: Conflict Resolution (sampling) — "interference clearance"
  │   Batch disputed node pairs (5-10 per sampling request).
  │   Prompt: "These nodes contradict each other.
  │            For each pair: supersede_a / supersede_b / both_valid."
  │   Execute supersession or clear 'disputed' status.
  │   Skipped entirely if no disputed nodes exist.
  │
  ├─ Stage 5: Execute (local writes)
  │   • keep → refresh last_accessed
  │   • condense → create new summary node via write_path, archive sources
  │   • archive → move to .archive/
  │   • link → write updated markdown with new [[wiki-links]]
  │   • supersede → update frontmatter (status, superseded_by)
  │   • clear_disputed → set status back to active
  │   • Clean expired archive files (> archive_retention_days)
  │   • Trigger delta sync so SQLite stays aligned.
  │   Each write goes through the existing write_path contract.
  │
  └─ Stage 6: Report (returned to caller)
      DreamerReport:
        started_at, completed_at,
        scanned: { stale, superseded, disputed, missing_link_pairs },
        triage: [ {node_id, decision, reason} ... ],
        links_added: [ {node_a, node_b} ... ],
        conflicts_resolved: [ {pair, decision} ... ],
        archived: [ {node_id, reason} ... ],
        condensed: [ {source_ids, new_node_id, new_title} ... ],
        warnings: [ ... ]
      
      Returned directly as structured MCP tool result.
      No server-side persistence — the agent relays to the user.
```

**Sampling protocol constraint:** MCP `sampling/createMessage` is a single prompt→response exchange. The host LLM has no tool-calling ability during sampling. Therefore, the Dreamer must include full node content in the prompt — sending only IDs is not viable. Batch size of 5-10 records (each up to 3500 words ≈ 5K tokens) keeps the worst-case prompt at ~50K tokens, within any modern LLM's context window.

**Batching strategy:** If a stage has more candidates than the batch limit, the Dreamer auto-chunks into multiple sequential sampling requests. Each batch is independent — no cross-batch state needed.

**Stages are skipped when empty.** If there are no disputed nodes, Stage 4 is a no-op. If there are no missing-link pairs, Stage 3 is a no-op. Only Stage 2 (triage) is always attempted when there are stale candidates.

**Files to touch:**

| File | Change |
|------|--------|
| `synapse/lifecycle/janitor.py` | Rename to `synapse/lifecycle/dreamer.py`. Rewrite `NightlyJanitor` → `Dreamer`. Add Stages 2-4 (sampling triage, link weaving, conflict resolution). Require `SamplingClient` — no fallback mode. |
| `synapse/server/service.py` | Remove `review_memory_cluster()`, `condense_memory_cluster()`, `promote_memory_candidate()`. Add `run_dreamer()` method. Remove `_sample_lifecycle_plan` / `_execute_lifecycle_plan` / `_build_lifecycle_evidence` private methods. |
| `synapse/server/mcp.py` | Remove three tool registrations. Add `run_dreamer` tool. |
| `synapse/server/schemas.py` | Remove `ReviewMemoryClusterRequest`, `CondenseMemoryClusterRequest`, `PromoteMemoryCandidateRequest`. Add `RunDreamerRequest` (may be empty or have optional params like `batch_size`). |
| `synapse/server/sampling.py` | Remove `build_review_memory_cluster_prompt`, `build_condense_memory_cluster_prompt`, `build_promote_memory_candidate_prompt`. Add `build_triage_prompt`, `build_link_weaving_prompt`, `build_conflict_resolution_prompt`. |
| `synapse/lifecycle/condensation.py` | Keep `DeterministicArchiveCondenser` as utility for the condense action inside Stage 5. Remove `ArchiveCondensationService` as a standalone entry point. |
| `synapse/storage/sqlite.py` | Add `find_missing_link_pairs(cosine_threshold, recency_days)` query for Stage 1. |
| `synapse/cli.py` | Remove `synapse janitor` and `synapse condense` CLI subcommands. The Dreamer runs only via MCP. |
| Tests | Rewrite lifecycle tool tests. New tests for each Dreamer stage with mock sampling. |
| `docs/usage.md` | Rewrite tool documentation. |

### Phase 4: Clean up MCP tool surface

After Phases 1-3, the final MCP tool set:

| Tool | Sampling | Purpose |
|------|:---:|--------|
| `search_memory` | no | Read: semantic search (includes full node lookup) |
| `write_memory` | yes | Write: decide + execute |
| `run_dreamer` | yes | Lifecycle: full sleep-cycle pipeline |

3 tools. Read / Write / Lifecycle each has a clear entry point.

**Also remove:**
- `ArchiveCondensationService.run()` as a standalone entry point — its condenser utility folds into Dreamer's Stage 5.
- All `_sample_lifecycle_plan` / `_execute_lifecycle_plan` / `_build_lifecycle_evidence` private methods in service.py.

## Ordering & Dependencies

```
Phase 1 (importance removal)
  ↓  independent, can land first
Phase 2 (tier removal)
  ↓  depends on Phase 1 being clean
Phase 3 (Dreamer pipeline)
  ↓  depends on Phase 2 (no tier in prompts)
Phase 4 (cleanup)
     depends on Phase 3
```

Phase 1 and Phase 2 are primarily data model changes. Phase 3 is the behavioral rewrite. Phase 4 is cleanup.

## What stays unchanged

- `NodeType` (transient / persistent) — orthogonal, not part of this refactor.
- `NodeStatus` (active / superseded / disputed) — still needed for conflict detection.
- `SensitivityLevel` — still needed for sanitization.
- The entire streaming/SSE/sampling transport layer in `server/app.py`, `streamable_runtime.py`.
- `write_path.py` and the `IntegrateAction` contract (create / supersede / complement).
- The retrieval pipeline structure (FTS + vector + graph hop + rerank + decay).
- `SyncManager` and file watcher.
- Security and sanitization.

## Resolved Questions

1. **Decay factor default.** Unified to a single configurable `factor` (default 0.98, ~34 day half-life). No adaptive curve for now — one knob is enough.

2. **CLI janitor mode.** Removed. `synapse janitor` and `synapse condense` CLI commands are deleted. The Dreamer runs exclusively via MCP with sampling. No fallback archive-only mode.

3. **Batch size.** Fixed at 5-10 records per sampling request. The Dreamer auto-chunks when candidates exceed the batch limit. Full node content is included in the prompt — sending only IDs is not feasible because MCP sampling has no tool-calling capability during the prompt→response exchange.

4. **Report persistence.** No server-side file. The DreamerReport is returned as the structured MCP tool result. The agent relays it to the user.
