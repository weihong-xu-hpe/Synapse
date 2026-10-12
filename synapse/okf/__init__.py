"""OKF — typed knowledge-node format for Synapse.

Spec: docs/okf.md. Four types cover how coding-agent knowledge occurs
(decision / fact / procedure / pitfall). Section headings are fixed English
strings so they are machine-checkable; body content may be zh/en/bilingual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

OKF_VERSION = 1

TYPE_DECISION = "decision"
TYPE_FACT = "fact"
TYPE_PROCEDURE = "procedure"
TYPE_PITFALL = "pitfall"

OKF_TYPES: tuple[str, ...] = (TYPE_DECISION, TYPE_FACT, TYPE_PROCEDURE, TYPE_PITFALL)

SECTION_TAKEAWAY = "Takeaway"
SECTION_SOURCES = "Sources"
SECTION_CONTEXT = "Context"
SECTION_DECISION = "Decision"
SECTION_CONSEQUENCES = "Consequences"
SECTION_DETAILS = "Details"
SECTION_STEPS = "Steps"
SECTION_SYMPTOM = "Symptom"
SECTION_CAUSE = "Cause"
SECTION_FIX = "Fix"

TAKEAWAY_MAX_CHARS = 200

# P2+ evidence (owner decision C, docs/okf.md §9): each distilled item carries
# evidence quotes; the server verifies them deterministically.
SECTION_EVIDENCE = "Evidence"
EVIDENCE_QUOTE_MAX_CHARS = 200


@dataclass(slots=True)
class EvidenceEntry:
    """One ``{"turn": "T12", "quote": "…"}`` pair from the extraction payload."""

    turn: str
    quote: str


def parse_evidence(raw: object) -> list[EvidenceEntry]:
    """Parse the evidence array of one item; malformed entries are dropped."""

    if not isinstance(raw, list):
        return []
    entries: list[EvidenceEntry] = []
    for raw_entry in raw:
        if not isinstance(raw_entry, Mapping):
            continue
        turn = str(raw_entry.get("turn") or "").strip()
        quote = str(raw_entry.get("quote") or "").strip()
        if not turn or not quote:
            continue
        entries.append(EvidenceEntry(turn=turn, quote=quote))
    return entries


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split()).casefold()


def verify_evidence(
    evidence: list[EvidenceEntry],
    *,
    turns: Mapping[str, str],
) -> tuple[list[EvidenceEntry], list[EvidenceEntry]]:
    """Deterministically verify evidence quotes against turn texts (NO LLM).

    A quote verifies when, whitespace-collapsed and case-insensitive, it
    appears in the named turn's body or attached tool-result lines. Returns
    ``(verified, unverified)`` in input order.
    """

    verified: list[EvidenceEntry] = []
    unverified: list[EvidenceEntry] = []
    for entry in evidence:
        turn_text = turns.get(entry.turn)
        if turn_text is not None and _collapse_whitespace(entry.quote) in _collapse_whitespace(turn_text):
            verified.append(entry)
        else:
            unverified.append(entry)
    return verified, unverified


def render_evidence_section(entries: list[EvidenceEntry]) -> str:
    """Render the optional ``## Evidence`` section lines (``- T12: "quote"``)."""

    lines = [f"## {SECTION_EVIDENCE}", ""]
    for entry in entries:
        quote = entry.quote if len(entry.quote) <= EVIDENCE_QUOTE_MAX_CHARS else entry.quote[:EVIDENCE_QUOTE_MAX_CHARS]
        lines.append(f'- {entry.turn}: "{quote}"')
    lines.append("")
    return "\n".join(lines)

# Allowed source entry shapes (docs/okf.md §2): bare node id, wiki-link-wrapped
# node id, http(s) URL, or session-key reference.
_SOURCE_URL_PREFIXES = ("http://", "https://")


def normalize_sources(raw_sources: list[str], *, resolve_node=None) -> tuple[list[str], list[str]]:
    """Normalize an LLM-emitted ``sources`` list to conforming entries.

    Keep: bare ``mem_…`` node ids, ``[[mem_…]]`` (stripped to bare), existing
    node ids, ``http(s)://`` URLs, ``session:<key>``, ``skill://<name>``. Drop
    everything else (pseudo-sources like "session transcript: …（本会话）" or
    "related: [[…]]" prose). Returns ``(normalized, dropped)``. ``resolve_node`` optionally
    validates that a ``mem_…`` id exists; non-existent ids are dropped.
    """

    import re

    normalized: list[str] = []
    dropped: list[str] = []
    for raw in raw_sources or []:
        entry = str(raw).strip()
        if not entry:
            continue
        # Wiki-link wrapped id(s) inside the entry: keep each id.
        for match in re.findall(r"\[\[([^\]]+)\]\]", entry):
            candidate = match.strip()
            if candidate and (resolve_node is None or resolve_node(candidate)):
                normalized.append(candidate)
            else:
                dropped.append(entry)
        bare = re.sub(r"\[\[[^\]]+\]\]", "", entry).strip()
        if not bare and entry.strip():
            continue  # pure wiki-link entry, already handled
        entry = bare
        if entry.startswith(_SOURCE_URL_PREFIXES):
            normalized.append(entry)
            continue
        if entry.casefold().startswith("session:") and len(entry) > len("session:"):
            normalized.append(entry)
            continue
        if entry.startswith("skill://") and len(entry) > len("skill://"):
            normalized.append(entry)
            continue
        if entry.startswith("mem_"):
            if resolve_node is None or resolve_node(entry):
                normalized.append(entry)
            else:
                dropped.append(entry)
            continue
        if entry:
            dropped.append(entry)
    # Dedupe preserving order.
    seen: set[str] = set()
    result: list[str] = []
    for entry in normalized:
        if entry not in seen:
            seen.add(entry)
            result.append(entry)
    return result, dropped


# Required body sections per OKF type (beyond the common Takeaway/Sources).
TYPE_REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    TYPE_DECISION: (SECTION_CONTEXT, SECTION_DECISION, SECTION_CONSEQUENCES),
    TYPE_FACT: (SECTION_DETAILS,),
    TYPE_PROCEDURE: (SECTION_STEPS,),
    TYPE_PITFALL: (SECTION_SYMPTOM, SECTION_CAUSE, SECTION_FIX),
}


class OkfValidationError(ValueError):
    """Raised when an OKF item or node body violates the type template."""


@dataclass(slots=True, frozen=True)
class OkfWarning:
    """One machine-checkable validation finding for a node."""

    code: str
    message: str
    section: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "section": self.section}


def is_okf_type(value: object) -> bool:
    return isinstance(value, str) and value in OKF_TYPES


def _split_sections(body: str) -> dict[str, str]:
    """Map ``## Heading`` -> section body text (first heading wins per name)."""

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


def _validate_universal_sections(content: str, *, sections: dict[str, str] | None = None) -> list[OkfWarning]:
    """Check Takeaway/Sources — required for every OKF type."""

    sections = sections if sections is not None else _split_sections(content)
    warnings: list[OkfWarning] = []

    if SECTION_TAKEAWAY not in sections:
        warnings.append(OkfWarning("okf_missing_section:Takeaway", f"Missing required section '## {SECTION_TAKEAWAY}'.", SECTION_TAKEAWAY))
    else:
        takeaway_lines = [line for line in sections[SECTION_TAKEAWAY].splitlines() if line.strip()]
        if not takeaway_lines:
            warnings.append(OkfWarning("okf_takeaway_invalid", "'## Takeaway' section is empty.", SECTION_TAKEAWAY))
        elif len(takeaway_lines) > 1:
            warnings.append(OkfWarning("okf_takeaway_invalid", "'## Takeaway' must be a single line.", SECTION_TAKEAWAY))
        elif len(takeaway_lines[0]) > TAKEAWAY_MAX_CHARS:
            warnings.append(
                OkfWarning(
                    "okf_takeaway_invalid",
                    f"'## Takeaway' first line exceeds {TAKEAWAY_MAX_CHARS} chars.",
                    SECTION_TAKEAWAY,
                )
            )

    if SECTION_SOURCES not in sections:
        warnings.append(OkfWarning("okf_missing_section:Sources", f"Missing required section '## {SECTION_SOURCES}'.", SECTION_SOURCES))
    elif not sections[SECTION_SOURCES].strip():
        warnings.append(OkfWarning("okf_missing_section:Sources", f"'## {SECTION_SOURCES}' section is empty.", SECTION_SOURCES))
    return warnings


def _validate_title(title: str | None) -> list[OkfWarning]:
    """Title rule (docs/okf.md §3): English, <= 90 chars, no type prefixes.

    Body/Takeaway keep the source language; only the title must be English so
    IDs stay meaningful and cross-session recall is consistent.
    """

    warnings: list[OkfWarning] = []
    if not title or not str(title).strip():
        return warnings
    title_text = str(title).strip()
    if any("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" for ch in title_text):
        warnings.append(OkfWarning("okf_title_non_english", f"Title must be English (found CJK): {title_text[:60]!r}"))
    if len(title_text) > 90:
        warnings.append(OkfWarning("okf_title_too_long", f"Title exceeds 90 chars ({len(title_text)})."))
    for prefix in ("procedure:", "decision:", "fact:", "pitfall:", "note:", "summary:"):
        if title_text.casefold().startswith(prefix):
            warnings.append(
                OkfWarning("okf_title_type_prefix", f"Title must not start with a type prefix ('{prefix}'); okf_type carries the type.")
            )
            break
    return warnings


def validate_okf_node(
    *,
    node_type: str,
    content: str,
    okf_type: str | None,
    sources: list[str] | None,
    title: str | None = None,
) -> list[OkfWarning]:
    """Deterministically validate a node against the OKF spec (docs/okf.md §6).

    Only warnings are produced here (transition period); the caller decides
    whether any code is fatal. Bilingual bodies never fire warnings — only
    fixed English headings are checked. ``title`` enables the title rules
    (English, <= 90 chars, no type prefixes).
    """

    warnings: list[OkfWarning] = []
    warnings.extend(_validate_title(title))
    if okf_type is None or not okf_type.strip():
        warnings.append(OkfWarning("okf_missing_type", "Persistent node has no okf_type frontmatter."))
        warnings.append(OkfWarning("okf_untyped", "Persistent node has no okf_type; treated as untyped note."))
    elif not is_okf_type(okf_type):
        warnings.append(
            OkfWarning("okf_unknown_type", f"Unknown okf_type '{okf_type}'; expected one of {', '.join(OKF_TYPES)}.")
        )

    if not sources:
        warnings.append(OkfWarning("okf_missing_sources", "Persistent node has no sources frontmatter."))
    else:
        import re

        for entry in sources:
            text = str(entry).strip()
            is_url = text.startswith(_SOURCE_URL_PREFIXES)
            is_session_key = text.casefold().startswith("session:") and len(text) > len("session:")
            is_skill_uri = bool(re.fullmatch(r"skill://[A-Za-z0-9._-]+", text))
            is_node_id = bool(re.fullmatch(r"mem_[A-Za-z0-9_\-]+", text))
            if not (is_url or is_session_key or is_skill_uri or is_node_id):
                warnings.append(
                    OkfWarning(
                        "okf_unresolvable_source",
                        f"Source entry does not conform (node id, session:<key>, or http(s) URL): {text[:80]!r}",
                    )
                )
                break  # one warning is enough to flag the node

    if okf_type is None or not is_okf_type(okf_type):
        # Untyped/unknown: only the universal sections (Takeaway, Sources)
        # are checkable — every type requires them.
        warnings.extend(_validate_universal_sections(content))
        return warnings

    sections = _split_sections(content)
    warnings.extend(_validate_universal_sections(content, sections=sections))

    for required in TYPE_REQUIRED_SECTIONS[okf_type]:
        if required not in sections:
            warnings.append(
                OkfWarning(f"okf_missing_section:{required}", f"Missing required section '## {required}' for okf_type '{okf_type}'.", required)
            )
        elif not sections[required].strip():
            warnings.append(
                OkfWarning(f"okf_missing_section:{required}", f"'## {required}' section is empty.", required)
            )

    return warnings


@dataclass(slots=True)
class OkfItem:
    """One distilled knowledge item extracted from a transcript (LLM output)."""

    okf_type: str
    title: str
    takeaway: str
    sections: dict[str, str]
    sources: list[str]
    evidence: list[EvidenceEntry] = field(default_factory=list)
    # True when every evidence entry verified against the source transcript.
    verified: bool = False

    def render(self) -> str:
        """Render the full OKF Markdown body per docs/okf.md §3 (and §9)."""

        required = TYPE_REQUIRED_SECTIONS[self.okf_type]
        lines: list[str] = [f"## {SECTION_TAKEAWAY}", "", self.takeaway.strip(), ""]
        for section in required:
            body = (self.sections.get(section) or "").strip()
            if not body:
                raise OkfValidationError(f"Item '{self.title}' is missing section '{section}'.")
            lines.extend([f"## {section}", "", body, ""])
        lines.extend([f"## {SECTION_SOURCES}", ""])
        for source in self.sources:
            lines.append(f"- {source}")
        lines.append("")
        if self.evidence:
            lines.append(render_evidence_section(self.evidence))
        return "\n".join(lines)


def _normalize_section_keys(sections_raw: Mapping) -> dict[str, str]:
    """Normalize LLM section keys: strip leading '#' markers and whitespace.

    glm-5.3-flash occasionally emits ``"## Details"`` instead of
    ``"Details"``; without normalization every item is rejected as missing
    sections. First occurrence wins per normalized name.
    """

    normalized: dict[str, str] = {}
    for key, value in sections_raw.items():
        name = str(key).lstrip("#").strip()
        if not name:
            continue
        normalized.setdefault(name, str(value))
    return normalized


def parse_okf_items(payload: object) -> list[OkfItem]:
    """Parse and validate the strict distiller JSON payload into OkfItems.

    Invalid items are skipped (reported via OkfValidationError by the caller's
    counter); valid items from a partially invalid response are returned.
    Tolerant shapes (LLM variance): section keys with leading ``#`` markers
    are normalized, and an empty top-level ``takeaway`` falls back to the
    ``Takeaway`` section body when that is a single non-empty line.
    """

    if not isinstance(payload, Mapping):
        raise OkfValidationError("Distiller payload must be a JSON object.")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise OkfValidationError("Distiller payload must contain an 'items' array.")

    items: list[OkfItem] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        okf_type = str(raw.get("okf_type") or "").strip()
        title = str(raw.get("title") or "").strip()
        takeaway = str(raw.get("takeaway") or "").strip()
        sections_raw = raw.get("sections")
        sources_raw = raw.get("sources")
        if not is_okf_type(okf_type) or not title:
            continue
        if not isinstance(sections_raw, Mapping):
            continue
        sections = _normalize_section_keys(sections_raw)
        # Tolerant fallback: the model sometimes leaves the top-level
        # 'takeaway' field empty while filling '## Takeaway' (observed live on
        # glm-5.3-flash). Use the section body when it is a usable single line.
        if not takeaway:
            section_takeaway = (sections.get(SECTION_TAKEAWAY) or "").strip()
            if section_takeaway and len(section_takeaway.splitlines()) == 1:
                takeaway = section_takeaway
        sources = [str(s).strip() for s in sources_raw if str(s).strip()] if isinstance(sources_raw, list) else []
        evidence = parse_evidence(raw.get("evidence"))
        missing = [s for s in TYPE_REQUIRED_SECTIONS[okf_type] if not sections.get(s, "").strip()]
        if missing or len(takeaway.splitlines()) != 1 or len(takeaway) > TAKEAWAY_MAX_CHARS:
            continue
        items.append(
            OkfItem(okf_type=okf_type, title=title, takeaway=takeaway, sections=sections, sources=sources, evidence=evidence)
        )
    return items


def render_title_repair_prompt(*, title: str, takeaway: str, body_excerpt: str) -> str:
    """One-shot repair prompt: produce an English title for a CJK/invalid title."""

    return "\n".join(
        [
            "The following knowledge item has an invalid title (not English, or a degenerate ID slug).",
            "Write an ENGLISH title (<= 90 chars): a specific claim or noun phrase carrying the key",
            "identifiers (service/component/error/file). No type prefix, no date unless date-bound.",
            "Do not translate the body — only the title changes.",
            "",
            f"Original title: {title}",
            f"Takeaway: {takeaway}",
            "",
            "Body excerpt:",
            body_excerpt[:1500],
            "",
            'Respond with a single JSON object only: {"title": "<english title>"}',
        ]
    )


def parse_title_repair(payload: object) -> str | None:
    """Extract the repaired English title; None when the payload is unusable."""

    if not isinstance(payload, Mapping):
        return None
    title = str(payload.get("title") or "").strip()
    if not title or len(title) > 90:
        return None
    if any("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" for ch in title):
        return None
    for prefix in ("procedure:", "decision:", "fact:", "pitfall:", "note:", "summary:"):
        if title.casefold().startswith(prefix):
            return None
    return title


def render_distiller_prompt(
    *,
    transcript: str,
    known_titles: list[str],
    own_distilled_titles: list[str],
    start_turn: str = "T1",
) -> str:
    """Build one extraction prompt for one transcript segment (design doc §3, §9).

    ``transcript`` is a pre-rendered segment (already char-budgeted by the
    caller's turn-boundary chunking — no truncation here). ``start_turn``
    tells the model the segment begins mid-conversation. No item-count cap:
    the model should extract EVERYTHING durable — one item per distinct claim.
    """

    known_block = "\n".join(f"- {t}" for t in known_titles[:50]) or "- (none)"
    own_block = "\n".join(f"- {t}" for t in own_distilled_titles[:50]) or "- (none)"

    type_rules = []
    for okf_type, required in TYPE_REQUIRED_SECTIONS.items():
        type_rules.append(
            f'- okf_type "{okf_type}": required sections {", ".join(f"## {s}" for s in (SECTION_TAKEAWAY, *required, SECTION_SOURCES))}'
        )

    return "\n".join(
        [
            "You extract durable, reusable engineering knowledge from a coding-agent session transcript segment.",
            "Extract ONLY knowledge that will still be valuable later: how a system works, decisions with rationale,",
            "step-by-step procedures, and pitfalls (symptom/cause/fix).",
            "NEVER extract: session narrative, status chatter (e.g. 'done', 'running tests'), raw tool output,",
            "or transient environment specifics (ports, pids, tmp paths) unless they are the durable lesson.",
            "Language: mirror the source language of the knowledge (Chinese, English, or mixed);",
            "section headings MUST be exactly the fixed English strings below.",
            f"This segment starts at turn {start_turn}; earlier turns may appear only as context.",
            "",
            "OKF templates (pick one okf_type per item):",
            *type_rules,
            "Every item must have a single-line '## Takeaway' (<= 200 chars) and a '## Sources' section listing",
            "where the knowledge came from (a transcript/session reference or URI).",
            "The item TITLE must be in ENGLISH (<= 90 chars): a specific claim or noun phrase carrying the key",
            "identifiers (service/component/error/file). Do NOT prefix titles with the type (no 'Procedure:',",
            "'Decision:' etc) and no dates unless the knowledge is date-bound. Body/Takeaway keep the source language.",
            "",
            "EVIDENCE (required per item): cite where the knowledge came from as",
            '"evidence": [{"turn": "<Tn>", "quote": "<verbatim excerpt from that turn, <= 200 chars>"}].',
            "The quote MUST be copied VERBATIM — an exact substring of the eligible text.",
            "NEVER paraphrase and NEVER quote the assistant's own prose.",
            "Accepted evidence sources depend on the type:",
            '- fact/procedure/pitfall: a quote from a tool result line or a user turn;',
            "- decision: a quote from a user turn or a tool result.",
            "Prefer short quotes (<= 200 chars). The quote is checked mechanically.",
            "",
            "Extract EVERYTHING durable — there is no item-count limit. Emit ONE item per",
            "distinct claim; never bundle unrelated topics into a single item.",
            "Skip anything already covered by these known knowledge nodes:",
            known_block,
            "",
            "Do not re-emit knowledge already distilled from this same transcript:",
            own_block,
            "",
            "Respond with a single JSON object, no markdown fence, no extra prose:",
            '{"items": [{"okf_type": "decision|fact|procedure|pitfall", "title": "...", "takeaway": "...",',
            '  "sections": {"<SectionName>": "markdown"}, "sources": ["..."],',
            '  "evidence": [{"turn": "T12", "quote": "..."}]}]}',
            "",
            "Transcript segment:",
            transcript,
        ]
    )


def render_reduce_prompt(
    *,
    candidates_json: str,
    known_titles: list[str],
    own_distilled_titles: list[str],
) -> str:
    """Reduce step: dedupe only — merge items stating the SAME claim, split
    composites, and return every other item unchanged. No count cap."""

    known_block = "\n".join(f"- {t}" for t in known_titles[:50]) or "- (none)"
    own_block = "\n".join(f"- {t}" for t in own_distilled_titles[:50]) or "- (none)"
    return "\n".join(
        [
            "You dedupe knowledge items extracted from segments of one session transcript.",
            "The candidates below were extracted independently per segment and may overlap or duplicate.",
            "Rules:",
            "1. Merge two items ONLY when they state the SAME claim (same fact/rule/pitfall/decision,",
            "   same subject and same assertion). If in doubt, keep them separate.",
            "2. If an item bundles several UNRELATED topics, split it into one item per topic",
            "   (same okf_type and shape, sections divided accordingly).",
            "3. Return EVERY other item unchanged — do not drop, summarize, compress, or rephrase.",
            "   There is NO item-count limit.",
            "4. Preserve each item's shape exactly: okf_type, English title (<= 90 chars), single-line",
            "   takeaway (<= 200 chars), sections, sources, and the evidence array (merge evidence",
            "   lists when merging items). Do not invent new items and do not drop evidence entries.",
            "5. Drop an item entirely ONLY if it is already covered by these known knowledge nodes:",
            known_block,
            "",
            "Do not re-emit knowledge already distilled from this same transcript:",
            own_block,
            "",
            "Respond with a single JSON object, no markdown fence, no extra prose:",
            '{"items": [{"okf_type": "decision|fact|procedure|pitfall", "title": "...", "takeaway": "...",',
            '  "sections": {"<SectionName>": "markdown"}, "sources": ["..."],',
            '  "evidence": [{"turn": "T12", "quote": "..."}]}]}',
            "",
            "Candidates:",
            candidates_json,
        ]
    )
