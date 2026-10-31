"""Pydantic node models and helpers for Markdown-backed Synapse memory nodes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator


class NodeType(str, Enum):
    """Lifecycle type for a node."""

    TRANSIENT = "transient"
    PERSISTENT = "persistent"


class NodeStatus(str, Enum):
    """Current truth status of a node."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"


class SensitivityLevel(str, Enum):
    """Cloud transmission sensitivity level."""

    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"


WORD_LIMIT: int = 3_500


@dataclass(slots=True, frozen=True)
class WordCountValidation:
    """Result returned by word-count validation."""

    word_count: int
    word_limit: int
    within_limit: bool

    @property
    def overage(self) -> int:
        return max(0, self.word_count - self.word_limit)

    @property
    def warning(self) -> str | None:
        if self.within_limit:
            return None
        return (
            f"Node exceeds the {self.word_limit}-word guideline "
            f"by {self.overage} words."
        )


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(UTC)


def ensure_utc_datetime(value: datetime | str | None) -> datetime | None:
    """Normalize a supported timestamp value to timezone-aware UTC."""

    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        value = datetime.fromisoformat(candidate)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def format_utc_datetime(value: datetime) -> str:
    """Format timestamps consistently for frontmatter serialization."""

    normalized = ensure_utc_datetime(value)
    if normalized is None:  # pragma: no cover - defensive only
        raise ValueError("Expected datetime value.")
    return normalized.isoformat().replace("+00:00", "Z")


class NodeMetadata(BaseModel):
    """Structured metadata stored in Markdown frontmatter."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    last_accessed: datetime | None = None
    # Legacy aggregate counter. Kept in sync with injected_count + used_count
    # (see _sync_access_count) so pre-P1B callers keep working.
    access_count: int = Field(default=0, ge=0)
    # P1B three-level usage counters (design doc §4.3(1), read/write decoupling):
    #   retrieved_count — appeared in a /api/search result set (diagnostic only)
    #   injected_count  — passed the inject gate and entered a prompt (weak prior)
    #   used_count      — referenced/used, or credited by the distiller (dominant)
    retrieved_count: int = Field(default=0, ge=0)
    injected_count: int = Field(default=0, ge=0)
    used_count: int = Field(default=0, ge=0)
    # Zep-style bi-temporal validity (design doc §3.4): set when a memory merge
    # creates (t_valid) or retires (t_invalid) a node.
    t_valid: str | None = None
    t_invalid: str | None = None
    type: NodeType = NodeType.TRANSIENT
    status: NodeStatus = NodeStatus.ACTIVE
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    # Session-keyed upsert provenance (bridge integration). Persisted in
    # frontmatter so Markdown stays canonical and rebuilds preserve it.
    session_key: str | None = None
    # OKF typed-knowledge metadata (docs/okf.md). Persisted in frontmatter.
    okf_type: str | None = None
    okf_version: int | None = None
    sources: list[str] = Field(default_factory=list)
    project: str | None = None
    # Distiller state on transcript nodes (design doc §2).
    distilled_hash: str | None = None
    distilled_at: str | None = None
    distilled_node_ids: list[str] = Field(default_factory=list)
    # P2c incremental distillation: last turn number covered by distillation.
    # None = never (or legacy) distilled; re-distills process later turns only.
    distilled_turn: int | None = None
    # Content hash deferred for short-transcript distillation; reselect when it changes.
    deferred_hash: str | None = None
    # A close/CLI attempt must be retried even when its pending revision is short.
    force_retry: bool | None = None
    # P2+ evidence (owner decision C): true when every evidence quote in the
    # node was deterministically verified against the source transcript.
    # Absent on pre-existing nodes — treated as unverified-only-if-present.
    verified: bool | None = None
    tags: list[str] = Field(default_factory=list)
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL

    @field_validator("id", "title")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value must not be blank.")
        return cleaned

    @field_validator("created_at", "last_accessed", mode="before")
    @classmethod
    def _normalize_datetime(cls, value: datetime | str | None) -> datetime | None:
        return ensure_utc_datetime(value)

    @field_validator("supersedes", mode="before")
    @classmethod
    def _normalize_supersedes(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("supersedes must be a list of strings or a single string")

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("tags must be a list of strings or a single string")

    @model_validator(mode="after")
    def _set_default_last_accessed(self) -> "NodeMetadata":
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        return self

    @model_validator(mode="after")
    def _sync_access_count(self) -> "NodeMetadata":
        """Keep the legacy ``access_count`` at least ``injected_count + used_count``.

        The split counters are the source of truth for P2 scoring; legacy
        callers read ``access_count``. Never *lower* it: pre-P1B nodes carry
        historical counts that the new counters cannot reconstruct.
        """

        floor = self.injected_count + self.used_count
        if self.access_count < floor:
            self.access_count = floor
        return self

    @field_serializer("created_at", "last_accessed")
    def _serialize_datetime(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return format_utc_datetime(value)

    @property
    def word_limit(self) -> int:
        return WORD_LIMIT


class Node(BaseModel):
    """Markdown-backed memory node with normalized metadata and body content."""

    model_config = ConfigDict(extra="forbid")

    metadata: NodeMetadata
    content: str = ""
    file_path: Path = Field(default_factory=Path)

    @field_validator("file_path", mode="before")
    @classmethod
    def _normalize_path(cls, value: str | Path | None) -> Path:
        if value in (None, ""):
            return Path()
        return Path(value)

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def title(self) -> str:
        return self.metadata.title

    def word_count_validation(self) -> WordCountValidation:
        return validate_word_count(self.content)


def count_text_words(text: str) -> int:
    """Count Latin word runs and individual CJK characters as word-like units."""

    count = 0
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal count
        if buffer:
            count += 1
            buffer.clear()

    for character in text:
        codepoint = ord(character)
        is_cjk = (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        )
        if is_cjk:
            flush_buffer()
            count += 1
            continue
        if character.isalnum() or character == "_":
            buffer.append(character)
            continue
        flush_buffer()

    flush_buffer()
    return count


def validate_word_count(text: str) -> WordCountValidation:
    """Validate a node body against the word-count guideline."""

    word_count = count_text_words(text)
    return WordCountValidation(
        word_count=word_count,
        word_limit=WORD_LIMIT,
        within_limit=word_count <= WORD_LIMIT,
    )


# Stopwords dropped from knowledge slugs (docs/okf.md title rule).
SLUG_STOPWORDS = frozenset(
    "a an the of to for and or in on at by is are be with via from into as that this "
    "it its when then than if but not no do does did can could should would will was were "
    "has have had also which who whom whose what how why where while during about after "
    "before between under over again further once here there all any both each few more "
    "most other some such only own same so too very s t just don now".split()
)
SLUG_MAX_TOKENS = 8
SLUG_MAX_CHARS = 48
SLUG_MIN_CHARS = 8
SLUG_MIN_TOKENS = 2


def slugify_english_title(title: str) -> str:
    """Slug for knowledge nodes from an English title (docs/okf.md §7 ID plan).

    Lowercase ASCII tokens joined by ``_``; stopwords dropped; capped at 8
    tokens / 48 chars at a word boundary. Returns "" when the title is
    degenerate (< 2 meaningful tokens or < 8 chars) — e.g. CJK-only titles,
    whose NFKD-ASCII projection leaves nothing.
    """

    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", ascii_text) if t]
    meaningful = [t for t in tokens if t not in SLUG_STOPWORDS and not t.isdigit()]
    if not meaningful:
        # Fall back to raw tokens if everything was classified a stopword.
        meaningful = tokens
    selected: list[str] = []
    length = 0
    for token in meaningful:
        extra = len(token) + (1 if selected else 0)
        if len(selected) >= SLUG_MAX_TOKENS or length + extra > SLUG_MAX_CHARS:
            break
        selected.append(token)
        length += extra
    slug = "_".join(selected)
    if len(selected) < SLUG_MIN_TOKENS or len(slug) < SLUG_MIN_CHARS:
        return ""
    return slug


def _resolve_node_date(current_date: date | datetime | str | None) -> date:
    if current_date is None:
        return utc_now().date()
    if isinstance(current_date, (datetime, str)):
        normalized_date = ensure_utc_datetime(current_date)
        if normalized_date is None:  # pragma: no cover - defensive only
            normalized_date = utc_now()
        return normalized_date.date()
    return current_date


def generate_node_id(title: str, current_date: date | datetime | str | None = None) -> str:
    """Generate a canonical node identifier using the Synapse `mem_YYYYMMDD_slug` convention.

    Slug prefers the English knowledge-slug (stopwords dropped, capped). A
    degenerate slug (CJK-only or too-short titles) falls back to
    ``node_<sha1(title)[:10]>`` so IDs never become meaningless fragments.
    """

    node_date = _resolve_node_date(current_date)
    slug = slugify_english_title(title)
    if not slug:
        title_hash = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
        slug = f"node_{title_hash}"
    return f"mem_{node_date.strftime('%Y%m%d')}_{slug}"


# ---------------------------------------------------------------------------
# Human-friendly names for knowledge nodes (single source of truth)
# ---------------------------------------------------------------------------
#
# A node's *identity* is ``metadata.id`` (persisted in frontmatter), never its
# filename. These helpers derive the human-facing directory/filename layout
# from that identity: ``active/<project>/<kebab-slug>.md``. Both the storage
# writer and the handbook compiler import them, so there is exactly one
# project/slug convention across the codebase.

_NODE_ID_DATE_PREFIX = re.compile(r"^mem_\d{8}_")

FALLBACK_PROJECT = "general"

# Deterministic project attribution: explicit ``metadata.project`` always wins;
# otherwise the first domain whose marker appears in ``f"{id} {title} {content}"``
# claims the node. Marker order is significant (specific tokens before looser
# ones); unmatched nodes land in ``general`` — never a guessed unknown domain.
# Built-in domains are generic examples only. Machine-local workspace domains
# live outside the repository in ``~/.synapse/project_domains.json`` (override
# the location with ``SYNAPSE_PROJECT_DOMAINS_FILE``; set the variable to an
# empty value to disable custom domains entirely). Custom entries are consulted
# first and shadow built-ins with the same slug, so a local file fully
# reproduces a private workspace layout without leaking it into the codebase.
_BUILTIN_PROJECT_DOMAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("synapse", ("synapse", "dreamer", "distiller", "okf")),
    ("omp", ("omp", "skill-evolver", "autolearn", "vibe")),
    ("payments-svc", ("payments-svc", "payments")),
    ("cluster-ops", ("cluster-ops", "cluster_ops")),
)

_PROJECT_DOMAINS_FILE_ENV = "SYNAPSE_PROJECT_DOMAINS_FILE"
_DEFAULT_PROJECT_DOMAINS_FILE = Path("~/.synapse/project_domains.json")
_CUSTOM_DOMAINS_CACHE: dict[str, tuple[int, tuple[tuple[str, tuple[str, ...]], ...]]] = {}


def _custom_project_domains() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Load machine-local domain mappings (mtime-cached); ``()`` when absent/invalid.

    File format: ``[{"project": "<slug>", "markers": ["<marker>", ...]}, ...]``.
    Unreadable or malformed files are ignored, never fatal.
    """

    env_raw = os.environ.get(_PROJECT_DOMAINS_FILE_ENV)
    if env_raw is None:
        path = _DEFAULT_PROJECT_DOMAINS_FILE.expanduser()
    elif not (path_raw := env_raw.strip()):
        return ()
    else:
        path = Path(path_raw).expanduser()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return ()
    key = str(path)
    cached = _CUSTOM_DOMAINS_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        entries = None
    parsed: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            slug = normalize_project_slug(str(entry.get("project") or ""))
            markers = entry.get("markers")
            if slug == FALLBACK_PROJECT or not isinstance(markers, list):
                continue
            normalized = tuple(dict.fromkeys(str(m).strip().lower() for m in markers if str(m).strip()))
            if normalized:
                parsed.append((slug, normalized))
    result = tuple(parsed)
    _CUSTOM_DOMAINS_CACHE[key] = (mtime, result)
    return result


def project_domains() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Active project→markers map: custom local entries first, built-ins after."""

    custom = _custom_project_domains()
    custom_slugs = {slug for slug, _ in custom}
    builtin = tuple((slug, markers) for slug, markers in _BUILTIN_PROJECT_DOMAINS if slug not in custom_slugs)
    return custom + builtin


def clean_slug(node_id: str) -> str:
    """Strip the ``mem_YYYYMMDD_`` date prefix for human-readable filenames."""

    slug = _NODE_ID_DATE_PREFIX.sub("", node_id).strip("-_")
    return slug or node_id


def kebab_slug(node_id: str) -> str:
    """Human-friendly kebab-case filename stem derived from a node id.

    ``mem_20261031_helm_hook_timeout_cleanup`` becomes
    ``helm-hook-timeout-cleanup``.
    """

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", clean_slug(node_id)).replace("_", "-").strip("-.")
    return slug or node_id


def normalize_project_slug(value: str) -> str:
    """Slug a project name for use as a directory segment."""

    slug = re.sub(r"[^a-z0-9._/-]+", "-", value.lower()).strip("-.")
    return slug or FALLBACK_PROJECT


def infer_project(node: "Node") -> str:
    """Deterministic project attribution for a knowledge node.

    An explicit ``metadata.project`` wins (lowercased/slugged); otherwise known
    workspace-domain markers (built-in examples plus any machine-local
    ``~/.synapse/project_domains.json`` entries) are matched against
    id + title + content.
    """

    explicit = (node.metadata.project or "").strip()
    if explicit:
        return normalize_project_slug(explicit)
    haystack = f"{node.id} {node.title} {node.content}".lower()
    for project, markers in project_domains():
        if any(marker in haystack for marker in markers):
            return project
    return FALLBACK_PROJECT
