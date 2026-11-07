"""Prompt builders for the two-stage episode knowledge pipeline."""

from __future__ import annotations

import json
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


_SCANNER_POLICY = [
    "Extract only high-value, durable system knowledge; do not summarize the conversation or its progress.",
    "Keep these categories: (1) component/config/repository ownership and topology, including which repo or chart owns a resource; (2) internal or external API quirks, error codes, and contract edge cases such as 404/400/empty-result handling; (3) infrastructure, middleware, and auth debugging pitfalls stated as exact Symptom -> Cause -> Fix; (4) confirmed architectural or engineering decisions approved by the user.",
    "FORBIDDEN: self-inflicted bugs in temporary scripts or session-written code that were fixed in the same session (including typos, regex mistakes, and indentation errors); one-off probe timings or batch progress counts; early hypotheses later refuted or corrected by the user or later turns; screen-readable code literals (e.g., static function signatures, class fields, literal constant values, or default config values directly visible by reading or grepping a single file, which go stale on code edits — keep only cross-service implicit contracts, non-obvious design rationale 'why not X', and debugging pitfalls); and workspace-local ephemeral facts (e.g., temporary /tmp artifact paths, random test UUIDs created for one smoke test, transient PID/memory readings, or a single PR's point-in-time CI state) that do not generalize across sessions.",
    "Prefer behavioral contrasts: prioritize extracting lessons where a trajectory shows a clear contrast (attempt A failed with symptom X -> diagnosed cause Y -> fix/procedure B succeeded), and use the Global User Spine to resolve corrections and evolving intent. Do NOT filter or reject lessons based on `turn_eligibility` or `assistant_prose_is_not_evidence`; every User, Assistant, and Tool turn is valid Stage 1 context.",
    "Write a detailed Chinese synthesis while preserving exact English identifiers, repository names, file paths, config keys, error strings, and code symbols. `topic` is an English title.",
    "Credentials are never knowledge: never reproduce a password, client secret, token, API key, database password, or private key value in `claim` or `topic`. Keep the config key, secret name, namespace, or rotation procedure and write the value as `<redacted>`. A lesson whose only content is a credential value has no durable value — omit it.",
]


def render_episode_summary_prompt(*, turns: str, global_user_spine: str) -> str:
    return "\n".join(
        [
            "Stage 1 scanner: inspect this complete dehydrated episode and propose only its most durable, reusable system lessons (usually 3–8; fewer when warranted).",
            *_SCANNER_POLICY,
            'Return JSON only with this schema: {"lessons":[{"okf_type":"pitfall|fact|decision|procedure","topic":"English title/topic","claim":"Detailed Chinese synthesis preserving exact technical identifiers","evidence_turns":["T12","T15"]}]}.',
            "Global User Spine (all User turns in the complete transcript, chronological):",
            global_user_spine,
            "Dehydrated episode turns:",
            turns,
        ]
    )


def render_episode_map_prompt(
    *, turns: str, global_user_spine: str, chunk_index: int, chunk_count: int
) -> str:
    return "\n".join(
        [
            f"Stage 1 scanner: inspect chunk {chunk_index} of {chunk_count} from one dehydrated conversation episode.",
            *_SCANNER_POLICY,
            "Extract only lessons supported by this chunk, but keep them provisional: later chunks and the Global User Spine can correct them.",
            'Return JSON only with this schema: {"lessons":[{"okf_type":"pitfall|fact|decision|procedure","topic":"English title/topic","claim":"Detailed Chinese synthesis preserving exact technical identifiers","evidence_turns":["T12","T15"]}]}. Return an empty lessons array when no durable lesson is present.',
            "Global User Spine (all User turns in the complete transcript, chronological):",
            global_user_spine,
            "Chunk turns:",
            turns,
        ]
    )


def render_episode_lesson_reduce_prompt(
    *, chunk_lessons: list[dict[str, Any]], global_user_spine: str
) -> str:
    return "\n".join(
        [
            "Stage 1 lesson_reduce: reconcile mapped candidates into the fewest complete, high-value durable lessons without losing distinct facts.",
            *_SCANNER_POLICY,
            "Merge overlapping lessons, preserve exact evidence turn IDs, and let later explicit user corrections in the Global User Spine supersede refuted claims. Keep the correct final state, not the abandoned hypothesis.",
            'Return JSON only with this schema: {"lessons":[{"okf_type":"pitfall|fact|decision|procedure","topic":"English title/topic","claim":"Detailed Chinese synthesis preserving exact technical identifiers","evidence_turns":["T12","T15"]}]}.',
            "Global User Spine:",
            global_user_spine,
            "Mapped lesson candidates:",
            _json(chunk_lessons),
        ]
    )


def render_episode_integrate_prompt(
    *,
    lessons: list[dict[str, Any]],
    same_session_nodes: list[dict[str, Any]],
    retrieved_nodes: list[dict[str, Any]],
    evidence_context: list[dict[str, Any]],
    global_user_spine: str,
) -> str:
    return "\n".join(
        [
            "You are both the Strict Gatekeeper and the OKF Integrator for durable Synapse knowledge.",
            "Value, refutation, and workspace-ephemerality gate: output action=none for (a) a self-inflicted temporary script/code bug fixed in this session, (b) a one-off timing or batch-progress count, (c) an early hypothesis refuted by later turns, (d) workspace-local ephemeral state (random test UUIDs, temporary /tmp paths, or single-run transient metrics) presented as a universal fact, or (e) screen-readable single-file code literals (function signatures, struct fields, or static config defaults directly readable from source without cross-component nuance). Use the Global User Spine and cited excerpts to detect corrections; for example, if a later turn clarifies that 401 was not caused by CORS, do not preserve the CORS hypothesis.",
            "Merge/update versus create: default to update when a same-session or retrieved persistent node already covers the topic; use its exact target_id. Create only when no candidate fits, and provide a specific justification. Never create a duplicate sibling for an existing claim.",
            "An `update` op REPLACES the target node body in place and no history is kept, so the body supplied for that target in the retrieved candidates must survive the rewrite. Copy every substantive line of that body into `content` with byte-identical wording — reordering sections or moving a line into another section is fine, but never rephrase, merge, summarize or drop it — and only then add the new lesson on top. A long existing body stays long.",
            "Credential gate: a secret value is never knowledge. Never put a password, client secret, token, API key, database password, connection string, or private key into `content`: keep the config key, secret name, namespace, and rotation procedure, and write the value as `<redacted>`. Return action=none with reason 'credential only' when the credential value is the lesson's only substance. Never widen a quote to capture a credential value.",
            "Evidence quotes are verified literally against the turn text, so quote a fragment that excludes the credential value (for example the command with `<redacted>` in place of the secret) rather than a fragment you rewrote.",
            "Language and OKF format: title MUST be a specific English title. Content MUST be a complete valid OKF body with `## Takeaway` as one Chinese line (target <=160 characters, hard maximum 200), the required sections for okf_type, and `## Sources`. Write Takeaway and body sections in Chinese while preserving exact English identifiers, repository names, paths, commands, config keys, and error strings. Required sections: fact=Details; decision=Decision and Consequences; procedure=Steps; pitfall=Symptom, Cause, and Fix.",
            "Verification is an annotation, never a write gate. Set verified=true when the technical claim (including repo, path, config key, error string, or root cause) is grounded in the supplied excerpts from User turns, Tool outputs, or Assistant source-code/log analysis and is not refuted later. Set verified=false only for speculative or ungrounded claims. Return verbatim evidence quotes, each <=200 characters, with its turn ID.",
            'Return JSON only: {"ops":[{"action":"update","lesson_index":0,"target_id":"...","title":"...","okf_type":"fact|decision|procedure|pitfall","content":"...","related_ids":["mem_..."],"verified":true,"evidence":[{"turn":"T12","quote":"verbatim excerpt <= 200 chars"}]},{"action":"create","lesson_index":1,"title":"...","okf_type":"...","content":"...","related_ids":["mem_..."],"justification":"...","verified":false,"evidence":[]},{"action":"none","lesson_index":2,"reason":"..."}]}. `related_ids` is optional for create/update.',
            "Return one operation per lesson. Use `none` when the lesson fails the value/refutation gate. Updates may target only supplied same-session or retrieved persistent nodes; never target a session transcript.",
            "For create/update, choose at most 3 genuinely related active persistent knowledge IDs from the supplied same-session nodes or retrieved nodes. Exclude the operation's own target_id and every `mem_session_*` transcript; omit `related_ids` or return an empty array when no candidate is genuinely related.",
            "Lessons:",
            _json(lessons),
            "Existing nodes already produced from this session:",
            _json(same_session_nodes),
            "Retrieved active persistent candidates (only these IDs may be update targets):",
            _json(retrieved_nodes),
            "Global User Spine (all User turns in the complete transcript, chronological):",
            global_user_spine,
            "Verbatim excerpts for each lesson's evidence_turns (include User, Assistant, and surviving Tool sections):",
            _json(evidence_context),
        ]
    )


def render_episode_merge_repair_prompt(
    *,
    title: str,
    okf_type: str,
    existing_body: str,
    candidate_body: str,
    omitted_lines: list[str],
) -> str:
    return "\n".join(
        [
            "Repair one automatic knowledge UPDATE whose candidate body dropped knowledge that the node it replaces already contains.",
            "The candidate body REPLACES the existing node body in place and no history is kept, so every dropped line is lost knowledge.",
            'Return JSON only: {"title":"...","content":"..."} with the FULL corrected body in `content`.',
            "Every substantive line of `existing_body` MUST appear in `content` with byte-identical wording. You may reorder sections or move a line into another section, but never rephrase, merge, summarize or drop it; integrate the candidate's new information on top of that.",
            "`omitted_lines` lists the lines the write gate found missing from the candidate; each of them must appear verbatim in the repaired `content`.",
            "Keep the OKF structure of `candidate_body`: exactly one non-empty `## Takeaway` line in Chinese of at most 200 characters, the required sections for the okf_type, and `## Sources`.",
            "Preserve exact English identifiers, repository names, file paths, commands, config keys and error strings. Do not add claims and do not repeat the same statement twice.",
            f"Title: {title}",
            f"OKF type: {okf_type}",
            _json({"existing_body": existing_body, "candidate_body": candidate_body, "omitted_lines": omitted_lines}),
        ]
    )


def render_episode_takeaway_repair_prompt(*, title: str, okf_type: str, takeaway: str) -> str:
    return "\n".join(
        [
            "Rewrite only the supplied OKF Takeaway line; do not rewrite or add any other section.",
            'Return JSON only: {"takeaway":"..."}.',
            "The result must be one non-empty Chinese line, target at most 160 characters, and never exceed the hard 200-character limit.",
            "Preserve the original supported meaning; do not add claims.",
            f"Title: {title}",
            f"OKF type: {okf_type}",
            f"Current Takeaway: {takeaway}",
        ]
    )
