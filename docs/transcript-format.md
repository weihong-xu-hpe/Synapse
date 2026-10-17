# Session transcript format (client → Synapse)

The omp bridge submits one session transcript per session key via `POST /api/write`
(`session_key` keyed upsert, no LLM). The server stores the payload verbatim as the
transcript node body; the distiller parses it into turns.

## v2 contract

A v2 transcript MUST start with the marker line:

```
<!-- synapse-transcript v2 -->
```

Turns are numbered sequentially `T1..Tn` over user/assistant messages. Tool
calls/results attach under the assistant turn that issued them. Thinking is omitted.

```markdown
<!-- synapse-transcript v2 -->
## T1 User
<full user text>

## T2 Assistant
<full assistant text>
### Tool T2.1 <tool_name>
call: <single line, <=1000 chars>
result [ok|error]:
<multiline result, preserved up to 4000 chars (800 for glob/grep)>

## T3 User
<full user text>
```

Rules:

- Turn headings are exactly `## T<n> User` / `## T<n> Assistant` (level 2, sequential
  `n` starting at 1).
- Tool sections are level 3 under the owning assistant turn: `### Tool T<n>.<m>
  <tool_name>`; `call:` is one line of at most 1,000 characters. `result [ok|error]:`
  preserves multiline formatting up to 4,000 characters for core tools and 800 characters
  for `glob` and `grep`.
- Scheduling/polling noise is omitted at render time: `todo`, `vibe_wait`, `vibe_list`,
  `vibe_kill`, `vibe_send`, `wait`, and `hub` wait/list/status/poll/kill operations.
- The complete rendered transcript is capped at 1,500,000 characters.
- The distiller extracts evidence quotes from **User turn bodies** and **tool result
  lines** only (see docs/okf.md §9) — keep those verbatim where possible.

## Legacy v1 (accepted, no marker)

Older transcripts use plain `## User` / `## Assistant` headings with no marker line
and no numbered ids. The server numbers those blocks `T1..Tn` in order of appearance
when parsing. No tool sub-sections are expected; any level-3 blocks belong to the
enclosing turn body.

## Distillation effects

- The live Two-Stage Episode Distiller dehydrates scheduling/polling noise and empty
  Assistant turns (Stage 0), builds a chronological Global User Spine, scans pending
  turns with a low-effort Stage 1 model in chunks up to `chunk_chars = 400000`, then
  passes candidate lessons to the Stage 2 gatekeeper and integrator.
- Bridge `POST /api/episode/close` triggers episode processing; the server's idle
  scheduler is the fallback, selecting transcripts idle for 60 minutes by default.
- Distillation progress is stamped on the transcript frontmatter: `distilled_turn`
  (last turn number covered) plus the existing `distilled_hash` / `distilled_at` /
  `distilled_node_ids`. Re-distillation after new turns arrive only processes turns
  after `distilled_turn` (with one previous turn as context).
