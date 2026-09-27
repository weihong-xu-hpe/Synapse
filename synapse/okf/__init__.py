"""OKF — typed knowledge-node format for Synapse.

Spec: docs/okf.md. Four types cover how coding-agent knowledge occurs
(decision / fact / procedure / pitfall). Section headings are fixed English
strings so they are machine-checkable; body content may be zh/en/bilingual.
"""

from __future__ import annotations

from dataclasses import dataclass
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

# Allowed source entry shapes (docs/okf.md §2): bare node id, wiki-link-wrapped
# node id, http(s) URL, or session-key reference.
_SOURCE_URL_PREFIXES = ("http://", "https://")


def normalize_sources(raw_sources: list[str], *, resolve_node=None) -> tuple[list[str], list[str]]:
    """Normalize an LLM-emitted ``sources`` list to conforming entries.

    Keep: bare ``mem_…`` node ids, ``[[mem_…]]`` (stripped to bare), existing
    node ids, ``http(s)://`` URLs, ``session:<key>``. Drop everything else
    (pseudo-sources like "session transcript: …（本会话）" or "related: [[…]]"
    prose). Returns ``(normalized, dropped)``. ``resolve_node`` optionally
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
            is_node_id = bool(re.fullmatch(r"mem_[A-Za-z0-9_\-]+", text))
            if not (is_url or is_session_key or is_node_id):
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

    def render(self) -> str:
        """Render the full OKF Markdown body per docs/okf.md §3."""

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
        return "\n".join(lines)


def parse_okf_items(payload: object) -> list[OkfItem]:
    """Parse and validate the strict distiller JSON payload into OkfItems.

    Invalid items are skipped (reported via OkfValidationError by the caller's
    counter); valid items from a partially invalid response are returned.
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
        if not is_okf_type(okf_type) or not title or not takeaway:
            continue
        if not isinstance(sections_raw, Mapping):
            continue
        sections = {str(k): str(v) for k, v in sections_raw.items()}
        sources = [str(s).strip() for s in sources_raw if str(s).strip()] if isinstance(sources_raw, list) else []
        missing = [s for s in TYPE_REQUIRED_SECTIONS[okf_type] if not sections.get(s, "").strip()]
        if missing or len(takeaway) > TAKEAWAY_MAX_CHARS or len(takeaway.splitlines()) != 1:
            continue
        items.append(OkfItem(okf_type=okf_type, title=title, takeaway=takeaway, sections=sections, sources=sources))
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
    max_chars: int,
    max_items: int,
) -> str:
    """Build the distillation prompt (design doc §3)."""

    content = transcript
    if len(content) > max_chars:
        head = max_chars // 2
        tail = max_chars - head
        content = content[:head] + "\n…\n" + content[-tail:]

    known_block = "\n".join(f"- {t}" for t in known_titles[:50]) or "- (none)"
    own_block = "\n".join(f"- {t}" for t in own_distilled_titles[:50]) or "- (none)"

    type_rules = []
    for okf_type, required in TYPE_REQUIRED_SECTIONS.items():
        type_rules.append(
            f"- okf_type \"{okf_type}\": required sections {', '.join(f'## {s}' for s in (SECTION_TAKEAWAY, *required, SECTION_SOURCES))}"
        )

    return "\n".join(
        [
            "You extract durable, reusable engineering knowledge from a coding-agent session transcript.",
            "Extract ONLY knowledge that will still be valuable later: how a system works, decisions with rationale,",
            "step-by-step procedures, and pitfalls (symptom/cause/fix).",
            "NEVER extract: session narrative, status chatter (e.g. 'done', 'running tests'), raw tool output,",
            "or transient environment specifics (ports, pids, tmp paths) unless they are the durable lesson.",
            "Language: mirror the source language of the knowledge (Chinese, English, or mixed);",
            "section headings MUST be exactly the fixed English strings below.",
            "",
            "OKF templates (pick one okf_type per item):",
            *type_rules,
            "Every item must have a single-line '## Takeaway' (<= 200 chars) and a '## Sources' section listing",
            "where the knowledge came from (a transcript/session reference or URI).",
            "The item TITLE must be in ENGLISH (<= 90 chars): a specific claim or noun phrase carrying the key",
            "identifiers (service/component/error/file). Do NOT prefix titles with the type (no 'Procedure:',",
            "'Decision:' etc) and no dates unless the knowledge is date-bound. Body/Takeaway keep the source language.",
            "",
            f"Extract at most {max_items} items. Skip anything already covered by these known knowledge nodes:",
            known_block,
            "",
            "Do not re-emit knowledge already distilled from this same transcript:",
            own_block,
            "",
            "Respond with a single JSON object, no markdown fence, no extra prose:",
            '{"items": [{"okf_type": "decision|fact|procedure|pitfall", "title": "...", "takeaway": "...",',
            '  "sections": {"<SectionName>": "markdown"}, "sources": ["..."]}]}',
            "",
            "Transcript:",
            content,
        ]
    )
