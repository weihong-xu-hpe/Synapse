"""Shared transcript/distillation predicates.

Single source of truth for "what is a session transcript" and "what is its
distillation state", used by the distiller, the service search filter, and the
markdown writer. No duplicate logic elsewhere.

Predicates (design decision A1):
- ``is_session_transcript``  — keyed (``mem_session_*`` / ``session_key``) or
  legacy (transient titled ``Session summary …``) transcript nodes.
- ``is_distilled_current``   — ``distilled_hash`` matches sha256 of the current
  content. True even when zero knowledge items were produced (the LLM judged
  nothing durable); such transcripts are never re-selected.
- ``is_represented``         — ``is_distilled_current`` AND knowledge nodes
  exist. Fully-represented transcripts are excluded from default search.
"""

from __future__ import annotations

import hashlib

from synapse.models import Node, NodeType

SESSION_ID_PREFIX = "mem_session_"
LEGACY_TITLE_PREFIX = "Session summary"


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_session_transcript(node: Node) -> bool:
    """Keyed session transcripts and legacy session-summary transcripts."""

    if node.metadata.id.startswith(SESSION_ID_PREFIX) or node.metadata.session_key:
        return True
    return node.metadata.type is NodeType.TRANSIENT and node.metadata.title.startswith(LEGACY_TITLE_PREFIX)


def is_distilled_current(node: Node) -> bool:
    """Transcript was distilled at its current content revision (zero items OK)."""

    metadata = node.metadata
    if not metadata.distilled_hash:
        return False
    return metadata.distilled_hash == content_sha256(node.content)


def is_represented(node: Node) -> bool:
    """Distilled at the current revision AND knowledge nodes were produced."""

    return is_distilled_current(node) and bool(node.metadata.distilled_node_ids)
