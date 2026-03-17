---
name: memory-write
description: "Use when deciding whether to store new information in Synapse, choosing note vs memory, checking overlap with existing nodes, and deciding create/supersede/complement before calling integrate_knowledge. Trigger phrases: remember this, save this, store this in memory, write this to Synapse, update this memory, supersede old memory."
---

# Memory Write

Use this skill when new information may need to become durable Synapse knowledge.

Do **not** use this skill for plain retrieval. If the task is only to fetch context, call `search_memory` following [Retrieval Guidelines](./references/retrieval-guidelines.md).

## Read first

Before deciding a write, consult:

- [Node Taxonomy](./references/node-taxonomy.md)
- [Write Decision Matrix](./references/write-decision-matrix.md)
- [Sensitivity Policy](./references/sensitivity-policy.md)

## Workflow

1. **Decide whether the information is worth storing**
   - Prefer writing user preferences, stable facts, architecture decisions, corrected knowledge, and reusable project constraints.
   - Default to skipping polite chatter, one-off execution traces, unconfirmed guesses, and context that will expire by the next turn.

2. **Draft the node**
   - Produce a clear `title` and concise `content`.
   - Choose `tier` using [Node Taxonomy](./references/node-taxonomy.md):
     - `note` for short-lived working context
     - `memory` for durable cross-session knowledge
   - Leave `type=transient` unless the knowledge is clearly long-term and worth preserving as `persistent`.
   - Leave `sensitivity=internal` unless [Sensitivity Policy](./references/sensitivity-policy.md) says otherwise.

3. **Check for overlap before writing**
   - Summarize the draft into a compact semantic query.
   - Call `search_existing_nodes(query=..., similarity_threshold=0.0)`.
   - Fetch full node content with `get_node` only for high-similarity, high-risk candidates.

4. **Choose the write action**
   - Use [Write Decision Matrix](./references/write-decision-matrix.md).
   - Decide exactly one of: `create`, `supersede`, `complement`.
   - If uncertain, prefer `create`.

5. **Execute the write**
   - Call `integrate_knowledge` with:
     - `title`
     - `content`
     - `tier`
     - optional `type`
     - optional `sensitivity`
     - `action`
     - `target_node_ids`
     - `reasoning`

6. **Keep reasoning crisp**
   - `reasoning` should be a short, auditable sentence.
   - Explain why the new node is independent, replacing, or complementary.

## Guardrails

- Synapse executes decisions; it does **not** infer them.
- Never `supersede` a node just because the score is high.
- Never target `disputed` nodes with `supersede` or `complement`.
- Do not split into separate `write-note` and `write-memory` workflows; tier selection is part of the same write decision.
- If a node would be too long or contains multiple independently retrievable ideas, split it before writing.

## Examples

- "Remember that the user prefers principle-oriented design docs."
- "Save this architecture decision to Synapse."
- "Update the previous rate limiting memory — sliding window replaced token bucket."
- "Store this as a short-lived note, not a long-term memory."
- "Write this insight into memory and link it to the existing gateway design node."
