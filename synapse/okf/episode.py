"""Prompt builders for the episode-based session knowledge pipeline."""

from __future__ import annotations

import json
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_episode_summary_prompt(*, turns: str, turn_eligibility: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "Summarize this coding-agent conversation episode into durable, reusable engineering lessons.",
            "Keep decisions with rationale, system facts, reproducible procedures, corrections, and pitfalls.",
            "Drop progress/status chatter, temporary state, one-off measurements, and numbers-of-the-day.",
            "Evidence may cite ONLY turn IDs marked eligible=true below: assertive User bodies and attached tool results. Never cite Assistant prose as evidence; question-only User turns are not evidence.",
            "Use the full turns as context, but attach evidence_turns only from the eligible evidence list. Return JSON only: {\"lessons\":[{\"topic\":\"...\",\"claim\":\"...\",\"evidence_turns\":[\"T1\"]}] }.",
            "Return an empty lessons array when nothing is durable. Do not impose a count cap; combine same-topic observations.",
            "Turn roles and eligibility:",
            _json(turn_eligibility),
            "Episode turns:",
            turns,
        ]
    )


def render_episode_map_prompt(
    *, turns: str, turn_eligibility: list[dict[str, Any]], chunk_index: int, chunk_count: int
) -> str:
    return "\n".join(
        [
            "Extract durable engineering lessons from this chunk of a larger conversation episode.",
            "Keep decisions with rationale, system facts, reproducible procedures, corrections, and pitfalls.",
            "Drop progress/status chatter, temporary state, one-off measurements, and numbers-of-the-day.",
            "Evidence may cite ONLY turn IDs marked eligible=true below: assertive User bodies and attached tool results. Never cite Assistant prose as evidence; question-only User turns are not evidence.",
            "Use the full turns as context, but attach evidence_turns only from the eligible evidence list.",
            "This is chunk " + str(chunk_index) + " of " + str(chunk_count) + "; preserve claims that later chunks may correct.",
            "Return JSON only: {\"lessons\":[{\"topic\":\"...\",\"claim\":\"...\",\"evidence_turns\":[\"T1\"]}] }.",
            "Return an empty lessons array when nothing is durable. Do not impose a count cap.",
            "Turn roles and eligibility:",
            _json(turn_eligibility),
            "Chunk turns:",
            turns,
        ]
    )


def render_episode_lesson_reduce_prompt(
    *, chunk_lessons: list[dict[str, Any]], eligible_turns: list[dict[str, Any]]
) -> str:
    return "\n".join(
        [
            "Reduce lessons mapped from chunks of one conversation episode.",
            "Merge duplicate or overlapping lessons into the fewest complete reusable claims without losing distinct facts.",
            "Reconcile evolving claims: a later explicit correction supersedes the earlier conflicting claim; retain both only when the distinction matters.",
            "Union evidence_turns only when each turn ID is marked eligible=true below. Never add evidence IDs or cite Assistant prose/question-only User turns.",
            "Return JSON only: {\"lessons\":[{\"topic\":\"...\",\"claim\":\"...\",\"evidence_turns\":[\"T1\"]}] }.",
            "No fixed output count cap; an empty list is valid.",
            "Eligible turn IDs and roles:",
            _json(eligible_turns),
            "Chunk lessons:",
            _json(chunk_lessons),
        ]
    )


def render_episode_integrate_prompt(
    *,
    lessons: list[dict[str, Any]],
    same_session_nodes: list[dict[str, Any]],
    retrieved_nodes: list[dict[str, Any]],
    evidence_context: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "Integrate the supplied episode lessons into durable Synapse knowledge.",
            "For every lesson choose update to an existing node stating the same lesson, create only if no suitable node exists, or none if unsupported/redundant/transient.",
            "Default to update; every create requires a specific justification. Do not split one coherent lesson into fragments or create siblings for the same claim.",
            "Updates may target only IDs in the supplied same-session nodes or retrieved persistent candidates. Never target a session transcript.",
            "For each update/create return one full valid OKF body with the type's required sections and Sources section; title must be English.",
            "Takeaway is exactly one non-empty line, target at most 160 characters and never more than 200 characters.",
            "Also return evidence as exact {turn, quote} snippets copied from that lesson's verbatim eligible evidence excerpts; do not cite assistant prose or invent quotes.",
            "Return JSON only: {\"ops\":[{\"action\":\"update\",\"target_id\":\"...\",\"title\":\"...\",\"okf_type\":\"fact|decision|procedure|pitfall\",\"content\":\"...\",\"evidence\":[{\"turn\":\"T1\",\"quote\":\"...\"}]},{\"action\":\"create\",\"title\":\"...\",\"okf_type\":\"...\",\"content\":\"...\",\"justification\":\"...\",\"evidence\":[{\"turn\":\"T1\",\"quote\":\"...\"}]},{\"action\":\"none\",\"lesson_index\":0,\"reason\":\"...\"}]}.",
            "An empty ops list is valid. No fixed operation-count cap.",
            "Lessons:",
            _json(lessons),
            "Previously produced nodes for this session (compact summary):",
            _json(same_session_nodes),
            "Retrieved persistent knowledge candidates (only these IDs can be update targets):",
            _json(retrieved_nodes),
            "Per-lesson eligible evidence excerpts (verbatim, with role/source markers):",
            _json(evidence_context),
        ]
    )

def render_episode_takeaway_repair_prompt(*, title: str, okf_type: str, takeaway: str) -> str:
    return "\n".join(
        [
            "Rewrite only the supplied OKF Takeaway line; do not rewrite or add any other section.",
            "Return JSON only: {\"takeaway\":\"...\"}.",
            "The result must be one non-empty line, target at most 160 characters, and never exceed the hard 200-character limit.",
            "Preserve the original supported meaning; do not add claims.",
            f"Title: {title}",
            f"OKF type: {okf_type}",
            f"Current Takeaway: {takeaway}",
        ]
    )
