"""Archive condensation primitives — protocol, deterministic condenser, and draft model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Sequence

from synapse.models import Node


UTC_SUFFIX = "+00:00"


@dataclass(slots=True, frozen=True)
class CondensationDraft:
    """Synthesized note draft before persistence."""

    title: str
    content: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_node_ids: tuple[str, ...] = field(default_factory=tuple)


class ArchiveCondenser(Protocol):
    """Protocol for pluggable archive condensation implementations.

    Implementations must be fully deterministic.
    Synapse does not call any external LLM inside lifecycle.
    """

    name: str

    def synthesize(
        self,
        nodes: Sequence[Node],
        *,
        now: datetime,
    ) -> CondensationDraft:
        """Produce a single synthesized draft from archived nodes."""


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

        # OKF-structured content: every persistent node (including sleep
        # products) uses Context / Decision / Consequences sections so the
        # store is uniformly OKF.
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
                "## Merged From",
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
        for line in content.replace("\r\n", "\n").splitlines():
            candidate = line.strip().lstrip("#").strip()
            if candidate:
                return candidate[: limit - 1] + "…" if len(candidate) > limit else candidate
        return "Archived note with minimal body content."
