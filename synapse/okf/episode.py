"""Prompt builders for the episode-based session knowledge pipeline."""

from __future__ import annotations

import json
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_episode_summary_prompt(*, turns: str, turn_ids: list[str]) -> str:
    return "\n".join(
        [
            "Summarize this coding-agent conversation episode into durable, reusable engineering lessons.",
            "Keep decisions with rationale, system facts, reproducible procedures, corrections, and pitfalls.",
            "Drop progress/status chatter, temporary state, one-off measurements, and numbers-of-the-day.",
            "Do not infer facts not supported by the supplied turns. Use only evidence turn IDs listed below.",
            "Return JSON only: {\"lessons\":[{\"topic\":\"...\",\"claim\":\"...\",\"evidence_turns\":[\"T1\"]}]}.",
            "Return an empty lessons array when nothing is durable. Do not impose a count cap; combine same-topic observations.",
            f"Eligible turn IDs: {_json(turn_ids)}",
            "Episode turns:",
            turns,
        ]
    )


def render_episode_map_prompt(*, turns: str, turn_ids: list[str], chunk_index: int, chunk_count: int) -> str:
    return "\n".join(
        [
            "Extract durable engineering lessons from this chunk of a larger conversation episode.",
            "Keep decisions with rationale, system facts, reproducible procedures, corrections, and pitfalls.",
            "Drop progress/status chatter, temporary state, one-off measurements, and numbers-of-the-day.",
            "Do not infer facts not supported by the supplied turns. Use only evidence turn IDs listed below.",
            "This is chunk " + str(chunk_index) + " of " + str(chunk_count) + "; preserve claims that later chunks may correct.",
            "Return JSON only: {\"lessons\":[{\"topic\":\"...\",\"claim\":\"...\",\"evidence_turns\":[\"T1\"]}]}.",
            "Return an empty lessons array when nothing is durable. Do not impose a count cap.",
            f"Eligible turn IDs: {_json(turn_ids)}",
            "Chunk turns:",
            turns,
        ]
    )


def render_episode_lesson_reduce_prompt(*, chunk_lessons: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "Reduce lessons mapped from chunks of one conversation episode.",
            "Merge duplicate or overlapping lessons into the fewest complete reusable claims without losing distinct facts.",
            "Reconcile evolving claims: a later explicit correction supersedes the earlier conflicting claim; retain both only when the distinction matters.",
            "Union valid evidence_turns from the supporting lessons. Never invent claims or evidence IDs.",
            "Return JSON only: {\"lessons\":[{\"topic\":\"...\",\"claim\":\"...\",\"evidence_turns\":[\"T1\"]}]}.",
            "No fixed output count cap; an empty list is valid.",
            "Chunk lessons:",
            _json(chunk_lessons),
        ]
    )


def render_episode_integrate_prompt(
    *,
    lessons: list[dict[str, Any]],
    same_session_nodes: list[dict[str, Any]],
    retrieved_nodes: list[dict[str, Any]],
    evidence_context: dict[str, str],
) -> str:
    return "\n".join(
        [
            "Integrate the supplied episode lessons into durable Synapse knowledge.",
            "For every lesson choose update to an existing node stating the same lesson, create only if no suitable node exists, or none if unsupported/redundant/transient.",
            "Default to update; every create requires a specific justification. Do not split one coherent lesson into fragments or create siblings for the same claim.",
            "Updates may target only IDs in the supplied same-session nodes or retrieved persistent candidates. Never target a session transcript.",
            "For each update/create return one full valid OKF body with the type's required sections and Sources section; title must be English.",
            "Also return evidence as exact {turn, quote} snippets copied from the eligible evidence context; do not fabricate quotes.",
            "Return JSON only: {\"ops\":[{\"action\":\"update\",\"target_id\":\"...\",\"title\":\"...\",\"okf_type\":\"fact|decision|procedure|pitfall\",\"content\":\"...\",\"evidence\":[{\"turn\":\"T1\",\"quote\":\"...\"}]},{\"action\":\"create\",\"title\":\"...\",\"okf_type\":\"...\",\"content\":\"...\",\"justification\":\"...\",\"evidence\":[{\"turn\":\"T1\",\"quote\":\"...\"}]},{\"action\":\"none\",\"lesson_index\":0,\"reason\":\"...\"}]}.",
            "An empty ops list is valid. No fixed operation-count cap.",
            "Lessons:",
            _json(lessons),
            "Previously produced nodes for this session (compact summary):",
            _json(same_session_nodes),
            "Retrieved persistent knowledge candidates (only these IDs can be update targets):",
            _json(retrieved_nodes),
            "Eligible source evidence text keyed by turn:",
            _json(evidence_context),
        ]
    )
