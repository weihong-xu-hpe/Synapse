"""Sampling abstractions for high-level Synapse MCP tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from typing import Protocol

from synapse.models import Node
from synapse.server.write_path import IntegrateAction


_JSON_ONLY_INSTRUCTION = "Return exactly one JSON object with no markdown fence and no extra prose."
_JSON_SHAPE_HEADER = "Return exactly this JSON shape:"
DEFAULT_FAST_SAMPLING_MODEL_HINTS = ("gemini-3-flash", "claude-4.5-haiku")


@dataclass(slots=True, frozen=True)
class SamplingCandidate:
    """Compact candidate summary used by sampling-backed decisions."""

    node_id: str
    title: str
    score: float
    status: str
    sensitivity: str
    file_path: str


@dataclass(slots=True, frozen=True)
class MemoryWriteSamplingRequest:
    """Context passed to a sampling-capable host/client."""

    prompt: str
    title: str
    content: str
    node_type: str
    sensitivity: str
    query: str
    similarity_threshold: float
    links: tuple[str, ...]
    candidates: tuple[SamplingCandidate, ...]
    candidate_nodes: tuple[Node, ...]


@dataclass(slots=True, frozen=True)
class MemoryWriteSamplingDecision:
    """Structured decision returned from sampling."""

    action: IntegrateAction | str
    target_node_ids: tuple[str, ...] = ()
    reasoning: str = ""
    confidence: float | None = None


class SamplingClient(Protocol):
    """Capability injected by transports that can ask a host LLM to sample."""

    name: str

    def sample_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 600,
        model_hints: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Return a parsed JSON object from a host-side sampling round."""
        raise NotImplementedError

    def decide_memory_write(self, request: MemoryWriteSamplingRequest) -> MemoryWriteSamplingDecision:
        """Return a structured write decision for the supplied draft and candidates."""
        raise NotImplementedError


def build_memory_write_query(*, title: str, content: str, query_hint: str | None = None, max_length: int = 240) -> str:
    """Build a deterministic semantic query for overlap lookup."""

    hinted = (query_hint or "").strip()
    if hinted:
        return hinted[:max_length].strip()

    normalized_title = " ".join(title.split())
    lines = [" ".join(line.split()) for line in content.splitlines() if line.strip()]
    summary = lines[0] if lines else ""
    if summary.casefold() == normalized_title.casefold():
        summary = ""

    query = " — ".join(part for part in (normalized_title, summary) if part)
    if not query:
        query = normalized_title
    if len(query) <= max_length:
        return query
    return query[: max_length - 1].rstrip() + "…"


def build_memory_write_sampling_prompt(request: MemoryWriteSamplingRequest) -> str:
    """Render a deterministic prompt for host-side sampling."""

    candidate_lines = []
    for candidate in request.candidates:
        candidate_lines.append(
            "- "
            f"{candidate.node_id} | title={candidate.title!r} | score={candidate.score:.6f} | "
            f"status={candidate.status} | sensitivity={candidate.sensitivity}"
        )
    if not candidate_lines:
        candidate_lines.append("- <no candidates>")

    detail_sections = []
    for node in request.candidate_nodes:
        detail_sections.append(
            "\n".join(
                [
                    f"ID: {node.id}",
                    f"Title: {node.title}",
                    f"Status: {node.metadata.status.value}",
                    "Content:",
                    node.content,
                ]
            )
        )
    if not detail_sections:
        detail_sections.append("<no candidate node details fetched>")

    return "\n".join(
        [
            "You are deciding how Synapse should write a new memory draft.",
            "Return a structured decision only. Choose exactly one action: create, supersede, or complement.",
            "Rules:",
            "1. If there are no relevant candidates or all scores are below the threshold, prefer create.",
            "2. If a high-similarity active candidate is replaced or corrected by the draft, choose supersede.",
            "3. If a high-similarity active candidate is related but still valid, choose complement.",
            "4. Never target nodes that were not provided in the candidate list.",
            "5. If unsure, choose create.",
            "6. Output MUST be a single JSON object with no markdown fence and no extra prose.",
            "",
            "Draft:",
            f"Title: {request.title}",
            f"Type: {request.node_type}",
            f"Sensitivity: {request.sensitivity}",
            f"Query: {request.query}",
            f"Similarity threshold: {request.similarity_threshold}",
            f"Links: {list(request.links)}",
            "Content:",
            request.content or "<empty>",
            "",
            "Candidate summaries:",
            *candidate_lines,
            "",
            "Candidate details:",
            "\n\n---\n\n".join(detail_sections),
            "",
            _JSON_SHAPE_HEADER,
            '{"action":"create|supersede|complement","target_node_ids":[],"reasoning":"one short sentence","confidence":0.0}',
        ]
    )

_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_PATTERN = re.compile(r"(\{.*\})", re.DOTALL)


def parse_memory_write_sampling_result(result: dict[str, Any]) -> MemoryWriteSamplingDecision:
    """Parse a sampling/createMessage result into a structured decision."""

    payload = parse_sampling_json_result(result)
    return MemoryWriteSamplingDecision(
        action=str(payload.get("action") or ""),
        target_node_ids=tuple(str(item) for item in payload.get("target_node_ids") or []),
        reasoning=str(payload.get("reasoning") or ""),
        confidence=_coerce_confidence(payload.get("confidence")),
    )


def parse_sampling_json_result(result: dict[str, Any]) -> dict[str, Any]:
    """Parse a sampling/createMessage result into a JSON object payload."""

    text = _extract_text_content(result.get("content"))
    return _extract_json_payload(text)



# ---------------------------------------------------------------------------
# Dreamer prompts
# ---------------------------------------------------------------------------


def build_triage_prompt(nodes: tuple[Node, ...]) -> str:
    """Render a triage prompt for the Dreamer's NREM consolidation stage."""

    return "\n".join(
        [
            "You are performing Synapse Dreamer triage — the NREM slow-wave consolidation stage.",
            "For each stale record below, decide: keep (still valuable, refresh access), condense (merge with others into a summary), or archive (no longer useful).",
            "Rules:",
            "1. Keep records that contain durable, reusable knowledge.",
            "2. Condense records that overlap or answer the same question — they will be merged into a summary.",
            "3. Archive records that are outdated, superseded by context, or too narrow to be useful.",
            "4. When unsure, prefer archive over keep.",
            _JSON_ONLY_INSTRUCTION,
            "",
            "Stale records:",
            _render_source_nodes(nodes),
            "",
            _JSON_SHAPE_HEADER,
            '{"decisions":[{"node_id":"...","decision":"keep|condense|archive","reason":"one short sentence"}]}',
        ]
    )


def build_link_weaving_prompt(pairs: tuple[tuple[Node, Node], ...]) -> str:
    """Render a link weaving prompt for the Dreamer's REM associative stage."""

    pair_blocks = []
    for i, (node_a, node_b) in enumerate(pairs, 1):
        pair_blocks.append(
            "\n".join(
                [
                    f"Pair {i}:",
                    f"  Node A: {node_a.id} — {node_a.title}",
                    f"    Content: {node_a.content[:500]}",
                    f"  Node B: {node_b.id} — {node_b.title}",
                    f"    Content: {node_b.content[:500]}",
                ]
            )
        )

    return "\n".join(
        [
            "You are performing Synapse Dreamer link weaving — the REM associative dreaming stage.",
            "These node pairs are semantically close but have no wiki-link edge between them.",
            "For each pair, decide: link (they should cross-reference each other) or independent (similar but separate topics).",
            "Rules:",
            "1. Link nodes that discuss the same concept, decision, or system from different angles.",
            "2. Keep independent if they happen to use similar words but address different questions.",
            _JSON_ONLY_INSTRUCTION,
            "",
            "Node pairs:",
            "\n\n".join(pair_blocks),
            "",
            _JSON_SHAPE_HEADER,
            '{"decisions":[{"node_a_id":"...","node_b_id":"...","link":true}]}',
        ]
    )


def build_conflict_resolution_prompt(pairs: tuple[tuple[Node, Node], ...]) -> str:
    """Render a conflict resolution prompt for the Dreamer's interference clearance stage."""

    pair_blocks = []
    for i, (node_a, node_b) in enumerate(pairs, 1):
        pair_blocks.append(
            "\n".join(
                [
                    f"Pair {i}:",
                    f"  Node A: {node_a.id} — {node_a.title}",
                    f"    Status: {node_a.metadata.status.value}",
                    f"    Content: {node_a.content}",
                    f"  Node B: {node_b.id} — {node_b.title}",
                    f"    Status: {node_b.metadata.status.value}",
                    f"    Content: {node_b.content}",
                ]
            )
        )

    return "\n".join(
        [
            "You are performing Synapse Dreamer conflict resolution — the interference clearance stage.",
            "These node pairs are marked as disputed (contradictory information).",
            "For each pair, decide: supersede_a (keep B, retire A), supersede_b (keep A, retire B), or both_valid (no real conflict, keep both).",
            "Rules:",
            "1. Supersede the older or less accurate version.",
            "2. If both contain valid but different perspectives, choose both_valid.",
            "3. When the conflict is unclear, prefer both_valid.",
            _JSON_ONLY_INSTRUCTION,
            "",
            "Disputed pairs:",
            "\n\n".join(pair_blocks),
            "",
            _JSON_SHAPE_HEADER,
            '{"decisions":[{"node_a_id":"...","node_b_id":"...","decision":"supersede_a|supersede_b|both_valid","reason":"one short sentence"}]}',
        ]
    )


def _render_source_nodes(nodes: tuple[Node, ...]) -> str:
    blocks: list[str] = []
    for node in nodes:
        blocks.append(
            "\n".join(
                [
                    f"- ID: {node.id}",
                    f"  Title: {node.title}",
                    f"  Type: {node.metadata.type.value}",
                    f"  Status: {node.metadata.status.value}",
                    f"  Sensitivity: {node.metadata.sensitivity.value}",
                    f"  Tags: {node.metadata.tags}",
                    "  Content:",
                    *[f"    {line}" for line in (node.content.splitlines() or ["<empty>"])],
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "<no source nodes>"


def _extract_text_content(content: Any) -> str:
    if isinstance(content, dict):
        if content.get("type") == "text":
            text = str(content.get("text") or "").strip()
            if not text:
                raise ValueError(
                    "Sampling response returned empty text — "
                    "the model provider may not support sampling/createMessage"
                )
            return text
        raise ValueError("Sampling response content must be text")

    if isinstance(content, list):
        parts = [str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        text = "\n".join(part for part in parts if part.strip()).strip()
        if text:
            return text
        raise ValueError("Sampling response content array did not contain text blocks")

    raise ValueError("Sampling response content must be a text object or list of text objects")


def _extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]

    block_match = _JSON_BLOCK_PATTERN.search(stripped)
    if block_match is not None:
        candidates.insert(0, block_match.group(1).strip())

    object_match = _JSON_OBJECT_PATTERN.search(stripped)
    if object_match is not None:
        candidates.insert(0, object_match.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Sampling response did not contain a valid JSON object")


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
