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
call: <single line, <=300 chars>
result [ok|error]: <head …elided… tail, <=600 chars>

## T3 User
<full user text>
```

Rules:

- Turn headings are exactly `## T<n> User` / `## T<n> Assistant` (level 2, sequential
  `n` starting at 1).
- Tool sections are level 3 under the owning assistant turn: `### Tool T<n>.<m>
  <tool_name>`; `call:` is a single line ≤ 300 chars; `result [ok|error]:` carries a
  head/elided/tail excerpt ≤ 600 chars.
- The distiller extracts evidence quotes from **User turn bodies** and **tool result
  lines** only (see docs/okf.md §9) — keep those verbatim where possible.

## Legacy v1 (accepted, no marker)

Older transcripts use plain `## User` / `## Assistant` headings with no marker line
and no numbered ids. The server numbers those blocks `T1..Tn` in order of appearance
when parsing. No tool sub-sections are expected; any level-3 blocks belong to the
enclosing turn body.

## Distillation effects

- The distiller splits transcripts into segments at turn boundaries (~`segment_chars`,
  default 16 000); one extraction call runs per segment; oversized single turns are
  split by characters. Nothing is head/tail-truncated away.
- Distillation progress is stamped on the transcript frontmatter: `distilled_turn`
  (last turn number covered) plus the existing `distilled_hash` / `distilled_at` /
  `distilled_node_ids`. Re-distillation after new turns arrive only processes turns
  after `distilled_turn` (with one previous turn as context).
