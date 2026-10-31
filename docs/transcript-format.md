# Session transcript format (client → Synapse)

The omp bridge submits one session transcript per session key via `POST /api/write`
(`session_key` keyed upsert, no LLM). The server redacts sensitive content before
storing the transcript; the distiller parses the stored body into turns.

## v2 contract

A v2 transcript MUST start with the marker line:

```
<!-- synapse-transcript v2 -->
```

Turns have stable `T<n>` numbers over user/assistant messages. Active branches
can have gaps. Tool calls/results attach to the assistant turn; thinking is omitted.

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

- Turn headings are exactly `## T<n> User` / `## T<n> Assistant`. Numbers are
  positive and stable across updates; active branches may contain gaps.
- Tool sections are level 3 under the owning assistant turn: `### Tool T<n>.<m>
  <tool_name>`; `call:` is one line of at most 1,000 characters. `result [ok|error]:`
  preserves multiline formatting up to 4,000 characters for core tools and 800 characters
  for `glob` and `grep`.
- Scheduling/polling noise is omitted at render time: `todo`, `vibe_wait`, `vibe_list`,
  `vibe_kill`, `vibe_send`, `wait`, and `hub` wait/list/status/poll/kill operations.
- The complete rendered transcript is capped at 1,500,000 characters.
- Evidence can cite User, Assistant, and surviving tool-result text. Preserve
  the original text where possible for verification.

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
- Bridge `POST /api/episode/close` triggers asynchronous processing; `accepted`
  means queued, not completed. The server's idle scheduler is the fallback.
- Short non-empty revisions selected by the scheduler are deferred without
  advancing `distilled_turn`; a content change makes them eligible again.
  Close events and explicit CLI runs scan short revisions immediately.
  Failed close/CLI attempts retain `force_retry` so the scheduler retries even
  when the pending content is too short for an ordinary sweep.
- `distilled_turn`, `distilled_hash`, `distilled_at`, and `distilled_node_ids`
  record progress. A shared transcript lock serializes processing; the stamp
  applies only to the revision actually read. Changes during processing remain
  eligible for another pass. A changed previously processed prefix is rescanned.
