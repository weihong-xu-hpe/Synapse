# Write Decision Matrix

## Step 0: decide whether to write at all

Write when the information is reusable, durable, user-requested, or meaningfully updates existing knowledge.

Skip when it is transient chatter, one-off execution noise, or an unconfirmed guess with no clear future value.

## Step 1: search for overlap

Before writing a new node:

1. Draft `title`, `content`, and `tier`
2. Produce a compact semantic summary of the draft
3. Call `search_existing_nodes(query=..., similarity_threshold=0.0)`
4. Use `get_node` only for candidates that materially affect the decision

## Action rules

### `create`

Use `create` when:
- all candidates are low similarity
- the topic is related but still independent
- matching nodes are already `superseded`
- the situation is uncertain and history should be preserved

### `supersede`

Use `supersede` only when:
- a high-similarity `active` node exists
- the draft clearly corrects, replaces, or obsoletes that node
- you can explain the replacement in one sentence

Never use `supersede` only because scores are high.

### `complement`

Use `complement` when:
- a high-similarity `active` node exists
- both old and new nodes remain valid
- the new node covers a different aspect of the same topic

## Disputed nodes

If the closest match is `disputed`:
- do not target it with `supersede` or `complement`
- prefer `create`
- mention the uncertainty in `reasoning`
- escalate for manual review if needed

## Multiple targets

- Multi-target `supersede` is allowed when the new node replaces all targets.
- Do not mix `supersede` and `complement` targets in a single write.
- If one target should be replaced and another only complemented, split the operation.

## Reasoning style

Keep `reasoning` short and auditable.

Good examples:
- "Sliding window replaced token bucket for burst traffic handling."
- "This node complements the existing gateway design with concrete rate-limit policy."
- "No clear conflict found; storing as an independent memory."
