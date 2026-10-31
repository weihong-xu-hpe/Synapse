"""Write-path execution helpers for Synapse."""

from __future__ import annotations

from enum import Enum


class IntegrateAction(str, Enum):
    """Explicit write actions for the integrate_knowledge endpoint.

    The decision of which action to use is made by Synapse's higher-level
    orchestration layer before control reaches the internal canonical write
    path. Synapse only executes the action here — it does not infer intent.
    """

    CREATE = "create"
    SUPERSEDE = "supersede"
    COMPLEMENT = "complement"


_DEFAULT_REASONING: dict[IntegrateAction, str] = {
    IntegrateAction.CREATE: "New knowledge node created.",
    IntegrateAction.SUPERSEDE: "New knowledge replaces the matched node.",
    IntegrateAction.COMPLEMENT: "Both nodes are valid and should cross-link.",
}


def normalize_reasoning(action: IntegrateAction, reasoning: str | None) -> str:
    """Return an explicit one-line explanation for the chosen action."""

    cleaned = (reasoning or "").strip()
    if cleaned:
        return cleaned
    return _DEFAULT_REASONING[action]
