"""Pydantic node models and helpers for Markdown-backed Synapse memory nodes."""

from __future__ import annotations

import hashlib
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
    access_count: int = Field(default=0, ge=0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0, exclude=True)
    type: NodeType = NodeType.TRANSIENT
    tier: str = Field(default="note", exclude=True)
    status: NodeStatus = NodeStatus.ACTIVE
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
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


def slugify_title(title: str) -> str:
    """Convert a title into an ASCII-safe underscore slug."""

    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text)
    slug = ascii_text.strip("_")
    if slug:
        return slug
    title_hash = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return f"node_{title_hash}"


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
    """Generate a canonical node identifier using the Synapse `mem_YYYYMMDD_slug` convention."""

    node_date = _resolve_node_date(current_date)
    return f"mem_{node_date.strftime('%Y%m%d')}_{slugify_title(title)}"
