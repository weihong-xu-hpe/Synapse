# Lifecycle Decision Guide

## Scope

Use this guide when a set of existing nodes needs semantic maintenance.

The usual inputs are:
- stale notes surfaced by janitor
- archive backlog selected for condensation
- disputed or overlapping clusters
- manually chosen nodes that need review

## What lifecycle should decide

Lifecycle work answers questions such as:
- keep or leave unchanged?
- archive / forget recommendation?
- promote `note` to `memory`?
- synthesize several nodes into a new summary `memory`?
- re-express overlapping material as a cleaner node?
- recommend manual review instead of automatic restructuring?

## Promotion heuristics

Promote `note` -> `memory` when most of the following are true:
- the content survived more than one task or session
- it keeps resurfacing in retrieval
- future work will depend on it
- it encodes a stable policy, preference, or architecture lesson

Do not promote just because a note is old.

## Summary heuristics

Create a lifecycle summary node when:
- several old nodes answer one coherent question
- a summary will improve future retrieval
- the cluster contains redundancy, fragmentation, or outdated layering

Do not summarize when:
- the nodes are unrelated
- the resulting node would be too broad to retrieve well
- the source material is still unresolved or contradictory without a clear framing

## Output pattern

When lifecycle work produces a new node:
1. draft the summary or promoted node
2. run normal overlap checks
3. choose `create`, `supersede`, or `complement`
4. call `integrate_knowledge`

If the safest outcome is no semantic rewrite, keep history intact and recommend no-op or manual review.
