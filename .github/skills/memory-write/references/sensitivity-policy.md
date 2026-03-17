# Sensitivity Policy

## Default

Use `internal` unless there is a clear reason to widen or tighten access.

## `public`

Choose `public` only when the content is safe to expose broadly and contains no private project details, credentials, personal data, or internal-only operational context.

Typical examples:
- generic public documentation knowledge
- non-sensitive technical patterns
- broadly shareable reference material

## `internal`

Choose `internal` for normal project knowledge.

Typical examples:
- architecture decisions
- implementation notes
- team conventions
- ordinary operational lessons

## `private`

Choose `private` when the content includes or strongly implies sensitive material.

Typical examples:
- secrets, tokens, credentials
- customer or personal data
- sensitive internal incidents
- details that should not be broadly transmitted or surfaced

If content is truly sensitive, ask whether it should be stored at all instead of blindly writing it.
