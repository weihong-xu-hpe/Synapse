"""Archive condensation primitives — protocol, deterministic condenser, and draft model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Sequence

from synapse.models import Node


UTC_SUFFIX = "+00:00"

# Markdown structure with no knowledge content (see _summarize).
_FENCE_MARKER_PATTERN = re.compile(r"^(?:`{3,}|~{3,})")
_STRUCTURAL_LINE_PATTERN = re.compile(r"^(?:#{1,6}(?:\s|$)|-{3,}$|_{3,}$|\*{3,}$|<!--)")
_BULLET_MARKER_PATTERN = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
_SPEAKER_LABEL_PATTERN = re.compile(r"^(?::\s*)?(?:user|assistant|system|tool)\s*:?\s*$", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class CondensationDraft:
    """Synthesized note draft before persistence."""

    title: str
    content: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_node_ids: tuple[str, ...] = field(default_factory=tuple)


class DeterministicArchiveCondenser:
    """Local deterministic synthesizer used as the safe default and fallback."""

    name = "deterministic"

    def synthesize(
        self,
        nodes: Sequence[Node],
        *,
        now: datetime,
    ) -> CondensationDraft:
        if not nodes:
            raise ValueError("At least one archived note is required for condensation")

        source_ids = tuple(node.id for node in nodes)
        common_tags = self._collect_common_tags(nodes)
        title = f"Archive Condensation {now.date().isoformat()}"
        source_lines = [f"- [[{node.id}]] — {node.title}" for node in nodes]
        tag_lines = [f"- {tag}" for tag in common_tags] or ["- No dominant tag cluster detected."]

        # OKF-structured content (docs/okf.md, okf_type=decision): the
        # condensation product is typed, with Takeaway + Sources sections.
        context_lines = [
            f"- **{node.title}** (`{node.id}`): {self._summarize(node.content)}"
            for node in nodes
        ]
        decision_lines = [
            f"- Merged {len(nodes)} archived note(s) into a single persistent record. "
            f"Original archive files are preserved for rollback and auditability.",
            *tag_lines,
        ]

        content = "\n".join(
            [
                f"# {title}",
                "",
                f"Synthesized on {now.isoformat().replace(UTC_SUFFIX, 'Z')} from {len(nodes)} archived note(s).",
                "",
                "## Takeaway",
                "",
                f"Merged {len(nodes)} archived note(s) ({', '.join(node.title for node in nodes[:3])}"
                f"{'…' if len(nodes) > 3 else ''}) into a single persistent record.",
                "",
                "## Context",
                *context_lines,
                "",
                "## Decision",
                *decision_lines,
                "",
                "## Consequences",
                "- Consolidated record supersedes the archived sources for active retrieval.",
                "- Original archive files remain available for audit and rollback.",
                "",
                "## Sources",
                *source_lines,
            ]
        ).strip()

        merged_tags = (
            "condensed",
            "archive-synthesis",
            *common_tags,
            *(f"merged_from:{node_id}" for node_id in source_ids),
        )
        return CondensationDraft(
            title=title,
            content=content,
            tags=merged_tags,
            source_node_ids=source_ids,
        )

    def _collect_common_tags(self, nodes: Sequence[Node]) -> list[str]:
        counts: dict[str, int] = {}
        for node in nodes:
            for tag in node.metadata.tags:
                cleaned = tag.strip()
                if cleaned:
                    counts[cleaned] = counts.get(cleaned, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [tag for tag, _count in ranked[:3]]

    def _summarize(self, content: str, *, limit: int = 180) -> str:
        """Return the first *knowledge-bearing* line of a note.

        Markdown structure carries no knowledge and must never be echoed as the
        summary: headings (``## User``, ``## T1 User``, ``## Takeaway``, tool
        sections), horizontal rules (``---``), HTML comments, and code-fence
        markers are all skipped. Without this the legacy transcripts condensed
        into zero-information bullets like ``: User`` (design doc §1.3).
        """

        fallback = "Archived note with minimal body content."
        in_fence = False
        for line in content.replace("\r\n", "\n").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _FENCE_MARKER_PATTERN.match(stripped):
                in_fence = not in_fence
                continue
            if in_fence:
                candidate = stripped
            else:
                if _STRUCTURAL_LINE_PATTERN.match(stripped):
                    continue
                candidate = _BULLET_MARKER_PATTERN.sub("", stripped).strip()
            if not candidate or _SPEAKER_LABEL_PATTERN.match(candidate):
                continue
            return candidate[: limit - 1] + "…" if len(candidate) > limit else candidate
        return fallback
