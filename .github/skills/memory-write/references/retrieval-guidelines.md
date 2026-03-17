# Retrieval Guidelines

## Default path

For ordinary context lookup, call:

- `search_memory(query, top_k=3)`

This is a guideline, not a standalone skill.

## Query writing

Prefer:
- the user's actual question
- the current task's core topic
- a short semantic restatement of what context is needed

Avoid:
- dumping the full conversation into the query
- using titles only when the topic is broader than the title
- overly narrow wording before the first search

## When to refine

Refine or re-run the search if:
- results are weak or obviously off-topic
- the query mixes multiple unrelated questions
- you now know a more precise term after seeing initial results

## Relationship to write flows

- Use `search_memory` for reading context.
- Use `search_memory` for write-time overlap and conflict checks.
- For lifecycle work, use `search_memory` to orient around a theme, then use normal write-time checks before creating any new node.

## Full node data in search results

`search_memory` returns full node objects (content, links, metadata) in each result. No separate fetch step is needed.
