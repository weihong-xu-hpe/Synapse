# OKF — typed knowledge-node format (Synapse spec v1)

Status: authoritative spec for Synapse knowledge nodes. This document defines the typed node
format enforced by validation (see §6); the distiller produces OKF nodes per §3.

## 1. What OKF is

An OKF node is a Markdown file in `~/synapse/active/` whose body follows one of four **typed
templates** and whose frontmatter carries the type as first-class metadata. OKF supersedes the
previous untyped notion ("persistent node with some `##` sections", synapse/server/sampling.py:140,
synapse/lifecycle/condensation.py:62-65) and realizes the design intent recorded in
docs/design/TODO-local-llm-upgrade.md:18 ("分节 + 来源 + takeaway" — sections + sources + takeaway).

Principles:
- **Typed, small set, extensible.** Four types cover how coding-agent knowledge actually occurs:
  decisions, system facts, procedures, pitfalls. Adding a type = adding a template here + a
  validator rule; nothing else changes.
- **Fixed English section headings, bilingual bodies.** Headings are machine-checked; content may
  be written in Chinese, English, or mixed (the corpus is zh/en mixed — zh is always allowed).
- **One-line takeaway + provenance are mandatory for every type.** Every node answers "so what?"
  in one line and links where the knowledge came from.
- **Knowledge, not narrative.** A node is distilled, reusable knowledge — not a session transcript.
  Session narrative lives in transient transcript nodes and is archived after distillation.

## 2. Common rules (all types)

Required frontmatter:

| Field | Type | Rule |
|---|---|---|
| `okf_type` | string enum | `decision` \| `fact` \| `procedure` \| `pitfall` |
| `okf_version` | integer | `1` (current spec version) |
| `sources` | list of strings | ≥ 1 entry; each entry is a node id (`mem_…`), a session key, or a URI. Empty sources are invalid. |

Optional frontmatter: `project` (string, e.g. `acme/payments-svc`), `tags` (existing field), plus all
system frontmatter (id/title/created_at/type/status/… — system-managed, unchanged).

Required body sections (all types):

| Section | Rule |
|---|---|
| `## Takeaway` | Exactly the first non-empty line is the takeaway — one sentence, no bullets, ≤ 200 chars. Bilingual OK. |
| `## Sources` | ≥ 1 list item `- [[<node-id>]]` or `- <session-key or URI>`. Machine-resolvable where possible. |

Additional per-type sections: see §3. All section headings are exact, case-sensitive, level-2
(`## Name`). Extra sections beyond the template are allowed after the required ones (extensibility),
but the required ones must be present and non-empty.

## 3. Types

### 3.1 `decision` — something was decided, with rationale and consequences
- **When to use**: an architectural/implementation choice that binds future work: "we use X over Y
  because Z". Supersedes/reconciles conflicting approaches.
- **Required sections**: `## Takeaway`, `## Context`, `## Decision`, `## Consequences`, `## Sources`.
  (`Context` = situation and constraints; `Decision` = what was chosen and the alternatives rejected;
  `Consequences` = what this implies, costs, follow-ups.)
- **Example** (frontmatter excerpt + body):

```markdown
---
id: mem_20260926_okf_server_side_distillation
title: Session transcripts are distilled server-side, not in the omp bridge
type: persistent
status: active
okf_type: decision
okf_version: 1
project: Synapse
sources:
  - mem_session_0123456789abcdef
---

## Takeaway
会话蒸馏在 Synapse 服务端进行，omp bridge 只负责写原始 transcript。

## Context
The omp bridge is a thin client: it writes one keyed transcript per session via `/api/write`.
Session end (`session_shutdown`) is not reliable (crashes, kill -9), and LLM configuration
(model, endpoints, fallbacks, token budgets) already lives server-side in `[decider]`.

## Decision
Distillation runs inside Synapse as a periodic sweep (`synapse/lifecycle/distiller.py`), not in
the bridge. The bridge keeps sending raw transcripts unchanged; the server selects idle
transcripts, extracts OKF items through the decider endpoint, and archives transcripts only
after they are fully distilled.

## Consequences
- Bridge stays LLM-free and crash-safe: a killed session loses nothing.
- Server-side retries/backoff replace client-side error handling.
- Distillation quality is bounded by the server's `[distiller]` config, not client capabilities.

## Sources
- [[mem_session_0123456789abcdef]]
```

### 3.2 `fact` — how something works (durable system truth)
- **When to use**: discovered behavior of a system/API/infra that will still be true next month:
  "service X calls Y via gRPC method Z", "redis key TTL is 300s", config semantics, data-model notes.
- **Required sections**: `## Takeaway`, `## Details`, `## Sources`.
  (`Details` = the fact itself, with specifics: names, values, versions. Multiple facts about one
  topic may share a node, each as a bullet or subsection under Details.)
- **Example**:

```markdown
## Takeaway
bge-reranker logits are unbounded and mostly negative; never treat them as positive scores.

## Details
- bge-reranker-v2-m3 outputs raw logits: observed range −10.5 … 0.64 over 200 samples.
- llama-server `--rerank` returns `results[].score` = logit, not sigmoid.
- FTS5 默认 unicode61 tokenizer 不切分中文长串；需 trigram。

## Sources
- [[mem_session_fedcba9876543210]]
```

### 3.3 `procedure` — how to do something, step by step
- **When to use**: reproducible task recipes: deploy steps, debug flows, triage runbooks, local
  env setup. Must be actionable without the original session.
- **Required sections**: `## Takeaway`, `## Steps`, `## Sources`.
  (`Steps` = ordered, numbered; each step starts with an action verb. Optional extra:
  `## Verification` for how to confirm success.)
- **Example**:

```markdown
## Takeaway
Run dedupe-session-summaries safely: backup → dry-run → --apply with manifest → verify.

## Steps
1. Back up the store: `sqlite3 ~/synapse/synapse.db ".backup /tmp/synapse-backup/synapse.db"` and copy `~/synapse/active/` alongside it.
2. Dry-run: `python -m synapse dedupe-session-summaries` and review which duplicate groups would be archived.
3. Apply: `python -m synapse dedupe-session-summaries --apply` — this writes a JSON manifest of archived ids + original paths into `.archive/` for reversibility.
4. Verify: spot-check a few archived files exist under `~/synapse/.archive/` and that `synapse status` shows the reduced active-node count.

## Sources
- [[mem_20260101_session_summary_example]]
```

### 3.4 `pitfall` — a gotcha: symptom → cause → fix
- **When to use**: non-obvious failure modes you already lost time to: "if you see X, it's actually
  Y, fix with Z". The most recall-valuable type for coding agents.
- **Required sections**: `## Takeaway`, `## Symptom`, `## Cause`, `## Fix`, `## Sources`.
  (`Symptom` = the observable error/signal, quote exact messages where possible — these are what
  retrieval matches on. `Cause` = root cause. `Fix` = the concrete remedy/avoidance.)
- **Example**:

```markdown
## Takeaway
log_dir 是相对 config 文件父目录解析的，会把日志写进 ~/.synapse/.synapse/.logs。

## Symptom
launchd stdout 落在仓库 .synapse/.logs，JSON 日志却出现在 ~/.synapse/.synapse/.logs。

## Cause
LoggingSettings.log_dir 以 config.toml 所在目录为基准做相对解析（synapse/config.py）。

## Fix
在 config 里写绝对路径，或把 log_dir 改为相对 CWD 解析并更新文档。

## Sources
- [[mem_session_0123456789abcdef]]
```

## 4. Reconciliation with the current code notion

| Current artifact | Disposition under OKF v1 |
|---|---|
| sampling.py:140 rule 7 ("prefer OKF … Context, Decision, Consequences") | The Context/Decision/Consequences triple becomes the `decision` template; the prompt is replaced by typed templates per §3 and distillation handles structure (persistent write validation per §6). |
| condensation.py:62-65 ("every persistent node … Context/Decision/Consequences so the store is uniformly OKF") | Dreamer condensation products become `okf_type: decision` (or `fact` when purely descriptive) and gain `## Takeaway` + `## Sources`; the existing `## Merged From` list moves into `## Sources`. |
| `low_structure` warning (service.py:339-347) | Replaced by typed validation §6 (warning-only transition, then rejection for new writes). |
| Legacy "Session summary — <project>" nodes | Transcripts, not OKF; they are **source material** referenced from `sources`, archived after distillation. |

## 5. What is NOT OKF

- Session transcripts (## User / ## Assistant dumps) — transient source material.
- One-line notes without a takeaway or sources — a plain note, flagged `okf_untyped` on write.
- Test/scaffolding nodes (e.g. "promotion candidate gamma") — transient by type.

### 5.1 Write-path normalization (MCP / unkeyed REST writes)

Agents writing through `write_memory` with `type` omitted default to **persistent** and are
normalized into OKF by the write path (`synapse/server/write_normalize.py`):

1. A body whose `##` sections match an OKF template gets `okf_type` inferred deterministically
   (Symptom+Cause+Fix → `pitfall`, Steps → `procedure`, Context+Decision+Consequences →
   `decision`, Details → `fact`); the body is stored unchanged.
2. Otherwise ONE LLM call reshapes the note into a single OKF item; the agent's original text is
   preserved verbatim under a trailing `## Original note` section (fidelity guard).
3. On LLM failure the note is stored as submitted, downgraded to transient, with an
   `okf_normalize_llm_failed` warning.
4. Non-English titles get the same one-shot English title repair as the distiller.
5. `sources` stays optional; absent sources produce a validation warning only — no pseudo-sources
   are invented. An explicit `type` always wins over these defaults.

## 6. Machine-checkable validation rules

`okf_valid(node) → list[warning]` — deterministic, no LLM. Warning codes (transition period: all
warnings; later: codes marked ⛔ reject):

| Code | Fires when |
|---|---|
| `okf_missing_type` ⛔ | persistent node without `okf_type` in frontmatter |
| `okf_unknown_type` ⛔ | `okf_type` not in the enum §2 |
| `okf_missing_version` | `okf_version` absent |
| `okf_missing_sources` ⛔ | frontmatter `sources` empty/absent |
| `okf_unresolvable_source` | a `sources` entry is neither a resolvable node id nor a URI/session key |
| `okf_missing_section:<Name>` ⛔ | a required section for the type (§3) absent or empty |
| `okf_takeaway_invalid` | `## Takeaway` missing, multi-line, or first line > 200 chars |
| `okf_takeaway_missing` ⛔ | `## Takeaway` section absent |
| `okf_heading_case` | required heading present but case/format deviates (auto-fixable) |
| `okf_untyped` | persistent node with zero `##` sections (successor of `low_structure`) |
| `okf_title_non_english` | title contains CJK (titles must be English; bodies stay bilingual) |
| `okf_title_too_long` | title exceeds 90 chars |
| `okf_title_type_prefix` | title starts with a type prefix like "Procedure:" |

Rules are checked in order; a node conforming for its type produces zero warnings. Bilingual bodies
never fire warnings — only headings are checked (fixed English strings `## Takeaway`, `## Context`,
`## Decision`, `## Consequences`, `## Details`, `## Steps`, `## Symptom`, `## Cause`, `## Fix`,
`## Sources`).

## 7. Title rule & ID plan

### Title rule (knowledge nodes)

- **English, ≤ 90 chars**: a specific claim or noun phrase carrying the key identifiers
  (service/component/error/file). Body and Takeaway keep the source language — only the title
  is English, so IDs stay meaningful and cross-session recall is consistent.
- No type prefixes ("Procedure:", "Decision:" …) — `okf_type` carries the type. No dates unless
  the knowledge is date-bound.
- Validator warnings: `okf_title_non_english` (any CJK), `okf_title_too_long` (> 90),
  `okf_title_type_prefix`.
- Agent writes (MCP/REST): warnings only — no auto-translation, no rejection.
- Distiller: the prompt requires English titles; if an item still returns a CJK/invalid title,
  ONE short repair LLM call asks only for an English title (given takeaway + body excerpt).
  If that also fails, the node is written with the fallback ID below plus the title warnings —
  the item is never dropped. Counters: `title_repairs` / `title_repair_failures` in the run report.

### ID plan (knowledge nodes)

- Format `mem_<YYYYMMDD>_<slug>`; the slug comes from the **English title**: lowercase ASCII
  tokens joined by `_`, stopwords dropped (a/an/the/of/to/for/and/or/in/on/at/by/is/are/be/with/
  via/from/into/as/that/this …), capped at **8 tokens / 48 chars** at a word boundary.
- Collision → append `_<4 hex>` = `sha1(title + created_at)[:4]` (extend to 6 hex only if still
  colliding). No `_2/_3` counters.
- Degenerate slug (< 2 meaningful tokens or < 8 chars — e.g. CJK-only titles): on the distiller
  path trigger the title repair; if still degenerate → fallback `mem_<YYYYMMDD>_<okf_type|node>_<8 hex>`.
- IDs are **immutable after creation** — title edits never rename nodes.
- Keyed transcripts keep `mem_session_<hash>`. Existing IDs are untouched (no migration).

## 8. Type registry (extensibility)

| okf_type | Required sections (beyond Takeaway/Sources) | Typical dreamer operation later |
|---|---|---|
| `decision` | Context, Decision, Consequences | supersede on conflicting decisions; never auto-archive while active |
| `fact` | Details | merge near-duplicate facts; invalidate on superseding fact |
| `procedure` | Steps | verify-freshness (re-run recipe or mark stale after N days) |
| `pitfall` | Symptom, Cause, Fix | link pitfalls to related facts; boost on repeated symptom hits |

Adding a type: append a row here with its required sections and validator rule; bump
`okf_version` only for breaking template changes. The distiller prompt and validator read this
registry from code (`synapse/okf/__init__.py` — `OKF_TYPES`, `TYPE_REQUIRED_SECTIONS`).
