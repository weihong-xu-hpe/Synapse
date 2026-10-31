"""Shared transcript/distillation predicates.

Single source of truth for "what is a session transcript" and "what is its
distillation state", used by the distiller, the service search filter, and the
markdown writer. No duplicate logic elsewhere.

Predicates (design decision A1):
- ``is_session_transcript``  — keyed (``mem_session_*`` / ``session_key``) or
  legacy (transient titled ``Session summary …``) transcript nodes.
- ``is_distilled_current``   — the content hash matches and no numbered v2
  turns remain beyond the recorded watermark. Zero extracted items are valid.
- ``is_represented``         — ``is_distilled_current`` AND knowledge nodes
  exist. Fully-represented transcripts are excluded from default search.
"""

from __future__ import annotations

import hashlib
import re

from synapse.models import Node, NodeType
from synapse.okf.transcript_format import V2_MARKER

SESSION_ID_PREFIX = "mem_session_"
LEGACY_TITLE_PREFIX = "Session summary"
# Legacy session-summary node ids: mem_<YYYYMMDD>_session_summary_*
LEGACY_ID_PATTERN = re.compile(r"^mem_\d{8}_session_summary_.+$")
_V2_TURN_HEADING = re.compile(r"^## T(\d+) (?:User|Assistant)[^\S\r\n]*$", re.MULTILINE)


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
    if metadata.force_retry or not metadata.distilled_hash or metadata.distilled_hash != content_sha256(node.content):
        return False
    if metadata.distilled_turn is not None and V2_MARKER in node.content:
        return not any(int(match.group(1)) > metadata.distilled_turn for match in _V2_TURN_HEADING.finditer(node.content))
    return True


def is_represented(node: Node) -> bool:
    """Distilled at the current revision AND knowledge nodes were produced."""

    return is_distilled_current(node) and bool(node.metadata.distilled_node_ids)
