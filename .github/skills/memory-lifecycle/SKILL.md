---
name: memory-lifecycle
description: "Use when reviewing stale notes, archive backlog, disputed topics, or janitor/condense candidates in Synapse, deciding what to preserve, promote, summarize, or re-express. Trigger phrases: clean up memory, lifecycle review, promote note to memory, condense archive, summarize old notes, memory hygiene, forgetting pass."
---

# Memory Lifecycle

Use this skill for semantic memory governance after Synapse has already surfaced candidate material through janitor, condense, archive review, or explicit user requests.

Do **not** use this skill for ordinary retrieval or one-off note capture. Lifecycle work starts from an existing set of nodes or a clear maintenance goal.

## Read first

Before acting, consult:

- [Lifecycle Decision Guide](./references/lifecycle-decision-guide.md)
- [Node Taxonomy](./references/node-taxonomy.md)
- [Retrieval Guidelines](./references/retrieval-guidelines.md)
- [Write Decision Matrix](./references/write-decision-matrix.md)
- [Sensitivity Policy](./references/sensitivity-policy.md)

## Workflow

1. **Start from a candidate set**
   - Examples: stale notes, archive backlog, disputed clusters, manually selected nodes, or janitor reports.
   - Synapse handles the mechanical scan; this skill handles the semantic judgement.

2. **Group by theme and choose an outcome**
   - Typical outcomes:
     - no-op / keep as-is
     - recommend archive / forget
     - promote `note` -> `memory`
     - synthesize several old nodes into one new `memory`
     - re-express an overlapping or conflicting cluster as a clearer node
     - suggest manual review for unresolved disputed topics

3. **Preserve semantic boundaries**
   - Summaries should merge related material, not unrelated scraps.
   - Prefer one coherent node per question or concept.
   - If one synthesized result would answer multiple unrelated queries, split it.

4. **Reuse the standard write contract**
   - Any new lifecycle-generated node must still go through:
     - `search_memory` (returns full node objects)
     - `write_memory`
   - Use the same `create` / `supersede` / `complement` rules as ordinary writes.

5. **Handle status correction sparingly**
   - Use `update_node_status` only when doing explicit repair or review.
   - Do not treat status mutation as the normal lifecycle output.

## Guardrails

- This skill does not replace janitor, condense, archive retention, or deletion logic.
- Do not generate summary nodes just because old nodes exist; summarize only when the result improves future retrieval or long-term coherence.
- Promote `note` to `memory` only when the content proved durable through reuse, importance, or repeated resurfacing.
- If lifecycle analysis is uncertain, preserve history and prefer a new independent node or a recommendation for manual review.

## Examples

- "Review these stale notes and decide which ones should become long-term memory."
- "Condense this archive backlog into one or two durable summary memories."
- "Do a memory hygiene pass on disputed gateway notes."
- "Promote the reusable deployment lessons from notes into memory."
- "Summarize these archived architecture fragments into a single current memory node."
