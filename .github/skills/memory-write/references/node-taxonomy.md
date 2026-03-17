# Node Taxonomy

## Purpose

Use this reference to choose `tier` consistently.

## `note`

Choose `note` for information that is useful now but likely to cool off quickly.

Typical examples:
- short-lived task context
- current debugging findings
- temporary working assumptions
- intermediate conclusions that may soon be replaced

Heuristics:
- likely useful for days or a short project phase
- may be revisited soon
- acceptable to forget if it stops being referenced

## `memory`

Choose `memory` for knowledge that should survive across sessions and remain valuable over time.

Typical examples:
- user preferences
- durable architecture decisions
- reusable operational lessons
- stable project constraints
- corrected facts that future work depends on

Heuristics:
- likely useful across multiple sessions
- important enough to preserve even after the current task ends
- likely to be referenced by future notes or summaries

## Promotion rule

Do not create separate write workflows for `note` and `memory`.

Instead:
- write `note` when durability is uncertain
- write `memory` when durability is already clear
- during lifecycle review, promote a `note` to `memory` only after repeated reuse or confirmed long-term value
