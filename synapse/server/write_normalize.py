"""Write-path type defaulting and OKF normalization for MCP/unkeyed writes.

Agents writing through ``write_memory`` (MCP) or the unkeyed REST ``/api/write``
historically defaulted to ``type=transient``, so durable knowledge decayed like
session chatter. This module implements the tightened defaults:

1. ``type`` omitted → persistent when the write carries OKF structure
   (explicit ``okf_type``, or a body whose sections match a known OKF template);
   explicit ``type`` always wins.
2. Persistent writes without ``okf_type`` → deterministic template inference
   from ``##`` sections; when no template matches, ONE LLM normalization call
   reshapes the note into a single OKF item with a fidelity guard (the
   original text is preserved under ``## Original note``), falling back to
   storing the text as submitted on LLM failure.
3. Non-English titles get the same one-shot repair the distiller uses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from synapse.okf import (
    OKF_TYPES,
    SECTION_CAUSE,
    SECTION_CONTEXT,
    SECTION_DECISION,
    SECTION_CONSEQUENCES,
    SECTION_DETAILS,
    SECTION_FIX,
    SECTION_STEPS,
    SECTION_SYMPTOM,
    SECTION_TAKEAWAY,
    TYPE_DECISION,
    TYPE_FACT,
    TYPE_PITFALL,
    TYPE_PROCEDURE,
    TYPE_REQUIRED_SECTIONS,
    is_okf_type,
    parse_title_repair,
    render_title_repair_prompt,
)

LOGGER = logging.getLogger(__name__)

# Section sets that imply a specific OKF template. Order matters: the first
# full match wins (pitfall and decision are the most specific).
_TEMPLATE_MATCHERS: tuple[tuple[str, frozenset[str]], ...] = (
    (TYPE_PITFALL, frozenset({SECTION_SYMPTOM, SECTION_CAUSE, SECTION_FIX})),
    (TYPE_DECISION, frozenset({SECTION_CONTEXT, SECTION_DECISION, SECTION_CONSEQUENCES})),
    (TYPE_PROCEDURE, frozenset({SECTION_STEPS})),
    (TYPE_FACT, frozenset({SECTION_DETAILS})),
)

# Templates rendered by the LLM normalizer (all four; the LLM picks).
_TEMPLATE_RULE_LINES = [
    f'- okf_type "{okf_type}": required sections {", ".join(f"## {s}" for s in (SECTION_TAKEAWAY, *required))}'
    for okf_type, required in TYPE_REQUIRED_SECTIONS.items()
]


def split_sections(body: str) -> dict[str, str]:
    """Map ``## Heading`` -> body text (same semantics as okf._split_sections)."""

    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if current is not None:
                sections.setdefault(current, "\n".join(buffer).strip())
            current = stripped[3:].strip()
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections.setdefault(current, "\n".join(buffer).strip())
    return sections


def has_okf_structure(content: str, okf_type: str | None) -> bool:
    """True when the write carries recognizable OKF structure.

    Either an explicit ``okf_type``, or a body whose ``##`` sections fully
    cover one of the four OKF templates.
    """

    if is_okf_type(okf_type):
        return True
    sections = split_sections(content)
    return any(required <= set(sections) for _, required in ((name, frozenset(spec)) for name, spec in _TEMPLATE_MATCHERS))


def infer_okf_type(content: str) -> str | None:
    """Deterministic okf_type from ``##`` sections; None when nothing matches."""

    sections = set(split_sections(content))
    for okf_type, required in _TEMPLATE_MATCHERS:
        if required <= sections:
            return okf_type
    return None


def title_needs_repair(title: str) -> bool:
    """CJK title or a degenerate/over-long one (same rule as the distiller)."""

    if any("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" for ch in title):
        return True
    return len(title) > 90


@dataclass(slots=True)
class NormalizationResult:
    """Outcome of the write-path normalization step."""

    title: str
    content: str
    okf_type: str | None
    warnings: list[dict[str, str]] = field(default_factory=list)
    sources: list[str] | None = None
    # How the final content was produced: "as_submitted" (unchanged),
    # "template_inferred" (okf_type inferred, body untouched),
    # "llm_normalized" (LLM reshaped into one OKF item), "llm_failed".
    mode: str = "as_submitted"
    # When set, overrides the resolved node type in the service (used to
    # downgrade a defaulted-persistent write to transient when LLM
    # normalization fails — an unstructured note without a working normalizer
    # is session chatter, not curated knowledge).
    type_override: str | None = None


NORMALIZE_SYSTEM_PROMPT = (
    "You normalize a single engineering knowledge note into the Synapse OKF format "
    "(docs/okf.md). Respond with a single JSON object, no markdown fence, no extra prose."
)

NORMALIZE_PROMPT_TEMPLATE = """
Normalize the note below into ONE OKF knowledge item. Do not drop information.

OKF templates (pick one okf_type):
{rules}
Every item must have a single-line '## Takeaway' (<= 200 chars) and a '## Sources'
section listing where the knowledge came from (use "agent:note" when the note has
no identifiable source).
The item TITLE must be in ENGLISH (<= 90 chars): a specific claim or noun phrase
carrying the key identifiers (service/component/error/file). No type prefix.
Body/Takeaway keep the note's source language; section headings are the fixed
English strings above. Put anything you could not restructure into the 'Details'
section (for okf_type "fact") rather than dropping it.

Respond with a single JSON object only:
{{"okf_type": "decision|fact|procedure|pitfall", "title": "...", "takeaway": "...", "sections": {{"<SectionName>": "markdown"}}, "sources": ["..."]}}

Note title: {title}

Note body:
{body}
"""


def render_normalize_prompt(title: str, body: str) -> str:
    return NORMALIZE_PROMPT_TEMPLATE.format(
        rules="\n".join(_TEMPLATE_RULE_LINES),
        title=title,
        body=body[:8000],
    )


def parse_normalize_payload(payload: object) -> tuple[str, str, str, str, list[str]] | None:
    """Parse the normalizer JSON → (okf_type, title, takeaway, sections, sources).

    sections is the rendered markdown of the required sections only. Returns
    None when the payload is unusable.
    """

    if not isinstance(payload, dict):
        return None
    okf_type = str(payload.get("okf_type") or "").strip()
    title = str(payload.get("title") or "").strip()
    takeaway = str(payload.get("takeaway") or "").strip()
    sections_raw = payload.get("sections")
    sources_raw = payload.get("sources")
    if okf_type not in OKF_TYPES or not title or not takeaway:
        return None
    if len(takeaway) > 200 or len(takeaway.splitlines()) != 1:
        return None
    if not isinstance(sections_raw, dict):
        return None
    sections = {str(k).strip(): str(v) for k, v in sections_raw.items()}
    required = TYPE_REQUIRED_SECTIONS[okf_type]
    if any(not sections.get(s, "").strip() for s in required):
        return None
    lines = [f"## {SECTION_TAKEAWAY}", "", takeaway, ""]
    for section in required:
        lines.extend([f"## {section}", "", sections[section].strip(), ""])
    if not isinstance(sources_raw, list):
        sources_raw = []
    sources = [str(s).strip() for s in sources_raw if str(s).strip()] or ["agent:note"]
    lines.extend([f"## Sources", ""])
    for source in sources:
        lines.append(f"- {source}")
    lines.append("")
    return okf_type, title, takeaway, "\n".join(lines), sources


class WritePathNormalizer:
    """Applies the B1/B2 defaults to an MCP/unkeyed write payload."""

    def __init__(self, sampling_client: Any | None = None) -> None:
        self._sampling_client = sampling_client

    def _sample_json(self, prompt: str) -> dict[str, Any]:
        if self._sampling_client is None:
            raise RuntimeError("no sampling client available for normalization")
        return self._sampling_client.sample_json(prompt=prompt, system_prompt=NORMALIZE_SYSTEM_PROMPT, max_tokens=2000)

    def normalize(
        self,
        *,
        title: str,
        content: str,
        node_type: str | None,
        okf_type: str | None,
        sources: list[str] | None,
    ) -> NormalizationResult:
        """Apply type defaulting, template inference, and LLM normalization.

        ``node_type`` None means the caller omitted ``type`` entirely.
        """

        warnings: list[dict[str, str]] = []

        # --- type defaulting (explicit type always wins) -----------------
        # Omitted type → persistent candidate: OKF-structured bodies take the
        # deterministic template path, plain bodies go through LLM
        # normalization (the note may still be durable knowledge in free
        # form). Explicit transient never reaches the normalizer.
        resolved_type = str(node_type) if node_type is not None else "persistent"

        is_persistent = resolved_type == "persistent"
        if not is_persistent:
            return NormalizationResult(title=title, content=content, okf_type=okf_type, warnings=warnings, mode="as_submitted")

        resolved_okf = okf_type if is_okf_type(okf_type) else None

        # --- title repair for non-English titles --------------------------
        final_title = title
        if title_needs_repair(final_title):
            repaired = self._repair_title(final_title, content)
            if repaired:
                final_title = repaired
            else:
                warnings.append(
                    {
                        "code": "okf_title_repair_failed",
                        "message": "Title remained non-English after repair attempt; keeping original.",
                    }
                )

        # --- template inference when okf_type missing ---------------------
        if resolved_okf is None:
            inferred = infer_okf_type(content)
            if inferred is not None:
                return NormalizationResult(
                    title=final_title,
                    content=content,
                    okf_type=inferred,
                    warnings=warnings,
                    mode="template_inferred",
                )

        # --- LLM normalization when NO template matched --------------------
        if resolved_okf is not None:
            # Explicit okf_type: trust it, keep the body as submitted.
            return NormalizationResult(
                title=final_title,
                content=content,
                okf_type=resolved_okf,
                warnings=warnings,
                mode="as_submitted",
            )

        original_body = content
        llm_failed_type = "transient" if node_type is None else "persistent"
        try:
            payload = self._sample_json(render_normalize_prompt(title=title, body=original_body))
        except Exception as exc:  # noqa: BLE001 — normalization failure stores as-submitted
            LOGGER.warning("Write-path OKF normalization failed", extra={"error": str(exc)})
            warnings.append(
                {
                    "code": "okf_normalize_llm_failed",
                    "message": "LLM normalization failed; stored as submitted without okf_type.",
                }
            )
            result = NormalizationResult(title=title, content=original_body, okf_type=None, warnings=warnings, mode="llm_failed")
            result.mode = "llm_failed"
            result.type_override = llm_failed_type
            return result

        parsed = parse_normalize_payload(payload)
        if parsed is None:
            warnings.append(
                {
                    "code": "okf_normalize_invalid_payload",
                    "message": "LLM normalization returned an unusable payload; stored as submitted without okf_type.",
                }
            )
            result = NormalizationResult(title=title, content=original_body, okf_type=None, warnings=warnings, mode="llm_failed")
            result.type_override = llm_failed_type
            return result

        norm_type, norm_title, _takeaway, rendered, norm_sources = parsed

        # Fidelity guard: the agent's original text is never lost.
        guarded = rendered.rstrip("\n") + "\n\n## Original note\n\n" + original_body.strip() + "\n"

        # Non-English normalized titles get repaired like distiller titles.
        final_title = norm_title
        if title_needs_repair(final_title):
            repaired = self._repair_title(final_title, rendered)
            if repaired:
                final_title = repaired
            else:
                warnings.append(
                    {
                        "code": "okf_title_repair_failed",
                        "message": "Normalized title remained non-English; keeping the repaired-or-original title.",
                    }
                )

        merged_sources = list(dict.fromkeys([*(sources or []), *norm_sources]))
        warnings.append(
            {
                "code": "okf_normalized_from_note",
                "message": "Note normalized into OKF by the write path; original text preserved under '## Original note'.",
            }
        )
        return NormalizationResult(
            title=final_title,
            content=guarded,
            okf_type=norm_type,
            warnings=warnings,
            sources=merged_sources,
            mode="llm_normalized",
        )

    def _repair_title(self, title: str, body_excerpt: str) -> str | None:
        try:
            prompt = render_title_repair_prompt(title=title, takeaway=body_excerpt[:200], body_excerpt=body_excerpt)
            payload = self._sample_json(prompt)
        except Exception:  # noqa: BLE001 — repair failure falls back, never drops
            return None
        repaired = parse_title_repair(payload)
        if repaired and not title_needs_repair(repaired):
            return repaired
        return None


def decide_write_type(
    *,
    node_type: str | None,
    content: str,
    okf_type: str | None,
) -> tuple[str, str]:
    """Return ``(resolved_type, mode)`` for a write; mode documents the path.

    Explicit ``node_type`` always wins. Omitted type defaults to persistent
    when the body carries OKF structure, transient otherwise.
    """

    if node_type is not None and str(node_type).strip():
        return str(node_type), "explicit"
    if has_okf_structure(content, okf_type):
        return "persistent", "defaulted_persistent"
    return "transient", "defaulted_transient"
