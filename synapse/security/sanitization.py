"""Security-focused credential redaction and archive retention helpers."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

from synapse.config import SynapseConfig


@dataclass(slots=True, frozen=True)
class RedactionPattern:
    """Compiled regex replacement rule."""

    label: str
    pattern: re.Pattern[str]
    replacement: str


@dataclass(slots=True, frozen=True)
class RedactionResult:
    """Output of a redaction pass."""

    text: str
    labels: list[str]

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(self.labels))


DEFAULT_PATTERN_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    ("API_KEY", r"sk-[a-zA-Z0-9]{32,}", "[REDACTED_API_KEY]"),
    ("EMAIL", r"[^@\s]+@[^@\s]+\.[^@\s]+", "[REDACTED_EMAIL]"),
    ("IP", r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "[REDACTED_IP]"),
    (
        "UUID",
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "[REDACTED_UUID]",
    ),
    (
        "JWT",
        r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        "[REDACTED_JWT]",
    ),
    (
        "SECRET_ASSIGNMENT",
        r"""(?i)(\b[A-Z0-9_]*(?:TOKEN|API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE[_-]?KEY|SECRET|PASSWORD|CREDENTIAL)[A-Z0-9_]*\b(?:\\?["'])?\s*[:=]\s*(?:\\?["'])?)([A-Za-z0-9._~+/=-]{16,})""",
        r"\g<1>[REDACTED_SECRET]",
    ),
    (
        "AUTHORIZATION",
        r"(?i)(\b(?:Bearer|Basic)[ \t]+)([A-Za-z0-9._~+/=-]{16,})",
        r"\g<1>[REDACTED_SECRET]",
    ),
)

# Unpaired UTF-16 surrogate: a high unit without its low partner, or a low unit
# without its high partner. Not a redaction rule — a repair applied at ingest
# so text stays UTF-8 encodable (see strip_lone_surrogates).
_LONE_SURROGATE = re.compile(r"[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]")

# Credential-shaped values, added 2026-10-31 after a client_secret reached a
# persistent knowledge node. The older SECRET_ASSIGNMENT rule only matched
# 16+ character values, so `client_secret=<short>` and `password=<short>`
# slips through; these rules close the common shapes. `scrub_credentials`
# lowers the same rules onto knowledge writes without the UUID/IP/email rules
# that would damage legitimate identifiers.
CREDENTIAL_PATTERN_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    (
        "PRIVATE_KEY_BLOCK",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        "URL_PASSWORD",
        r"(?i)(://[^\s/:@]{1,64}:)([^\s/@]{3,})(@)",
        r"\g<1>[REDACTED_CREDENTIAL]\g<3>",
    ),
    (
        "CREDENTIAL_ASSIGNMENT",
        r"(?i)(\b(?:password|passwd|pwd|passphrase|client[_-]?secret|secret[_-]?key|app[_-]?secret|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|auth[_-]?token)\b\s*[:=]\s*[\"']?)(?!\$|\{|<|\[|\*|%)([A-Za-z0-9._~+/=!@#$%^&*\-]{4,})",
        r"\g<1>[REDACTED_CREDENTIAL]",
    ),
    (
        "CREDENTIAL_BEARER",
        r"(?i)(\b(?:Bearer|Basic)\s+)([A-Za-z0-9._~+/=-]{12,})",
        r"\g<1>[REDACTED_CREDENTIAL]",
    ),
)

DEFAULT_PATTERN_DEFINITIONS = (*DEFAULT_PATTERN_DEFINITIONS, *CREDENTIAL_PATTERN_DEFINITIONS)

# Credential rules alone, for knowledge writes (see scrub_credentials).
_CREDENTIAL_PATTERNS: tuple[RedactionPattern, ...] = tuple(
    RedactionPattern(label=label, pattern=re.compile(pattern), replacement=replacement)
    for label, pattern, replacement in CREDENTIAL_PATTERN_DEFINITIONS
)

# `stringData:`/`data:` blocks in Kubernetes Secret YAML carry values on the
# following indented lines, which no single assignment rule can see.
_YAML_SECRET_BLOCK = re.compile(r"^[ \t]*(?:stringData|data):[ \t]*$")
_YAML_PAIR = re.compile(r"^([ \t]+)([A-Za-z0-9._\-/]+):[ \t]*(\S.*)$")
_BASE64_VALUE = re.compile(r"^[A-Za-z0-9+/]{16,}={0,2}$")


class RedactionEngine:
    """Apply deterministic regex-based redaction rules to outgoing payloads."""

    def __init__(self, patterns: Sequence[RedactionPattern] | None = None) -> None:
        self.patterns = list(patterns or build_default_patterns())

    @classmethod
    def from_config(cls, config: SynapseConfig) -> "RedactionEngine":
        return cls(patterns=build_redaction_patterns(config))

    def redact(self, text: str) -> tuple[str, list[str]]:
        result = self.redact_detailed(text)
        return result.text, result.labels

    def redact_detailed(self, text: str) -> RedactionResult:
        redacted_text = text
        labels: list[str] = []
        for rule in self.patterns:
            redacted_text, count = rule.pattern.subn(rule.replacement, redacted_text)
            if count:
                labels.extend([rule.label] * count)
        return RedactionResult(text=redacted_text, labels=labels)


def build_default_patterns() -> list[RedactionPattern]:
    """Compile the built-in redaction patterns."""

    return [
        RedactionPattern(label=label, pattern=re.compile(pattern), replacement=replacement)
        for label, pattern, replacement in DEFAULT_PATTERN_DEFINITIONS
    ]



def build_redaction_patterns(config: SynapseConfig) -> list[RedactionPattern]:
    """Combine built-in redaction rules with configured custom ones."""

    patterns = build_default_patterns()
    for index, rule in enumerate(config.sanitization.custom_patterns, start=1):
        if isinstance(rule, str):
            patterns.append(
                RedactionPattern(
                    label=f"CUSTOM_{index}",
                    pattern=re.compile(rule),
                    replacement=f"[REDACTED_CUSTOM_{index}]",
                )
            )
            continue
        patterns.append(
            RedactionPattern(
                label=rule.label,
                pattern=re.compile(rule.pattern),
                replacement=rule.replacement,
            )
        )
    return patterns



def strip_lone_surrogates(text: str, *, replacement: str = "\ufffd") -> str:
    """Return ``text`` with unpaired UTF-16 surrogates replaced.

    JSON clients may legally transmit a lone ``\\udXXX`` escape (an upstream
    truncation that cut an astral character between its two code units).
    ``json.loads`` accepts it, but ``str.encode("utf-8")`` raises
    ``UnicodeEncodeError`` — which would fail the whole write mid-rename and,
    for keyed transcript upserts, drop the retained snapshot. Ingest calls this
    so the note survives with U+FFFD standing in for the broken character.
    """

    if not text:
        return text
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return _LONE_SURROGATE.sub(replacement, text)
    return text


def scrub_credentials(text: str, *, replacement: str = "[REDACTED_CREDENTIAL]") -> str:
    """Redact credential values in text that will be persisted as knowledge.

    Only credential shapes are touched — passwords, client secrets, tokens,
    private keys, URL userinfo and Kubernetes secret blocks. Unlike the cloud
    sanitizer's rules this deliberately leaves UUIDs, IPs, emails, and
    other identifiers intact, because knowledge nodes cite them legitimately.
    The rules are the built-in ones only, so behaviour does not depend on the
    caller's custom redaction configuration.
    """

    if not text:
        return text
    scrubbed = text
    for rule in _CREDENTIAL_PATTERNS:
        scrubbed = rule.pattern.sub(rule.replacement, scrubbed)
    return _scrub_yaml_secret_blocks(scrubbed, replacement)


def _scrub_yaml_secret_blocks(text: str, replacement: str) -> str:
    """Replace the values inside ``stringData:``/``data:`` YAML blocks."""

    if "stringData:" not in text and "data:" not in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    block_indent: int | None = None
    for line in lines:
        indent = len(line) - len(line.lstrip(" \t"))
        if block_indent is not None and (not line.strip() or indent > block_indent):
            pair = _YAML_PAIR.match(line)
            value = pair.group(3) if pair else ""
            looks_secret = value.startswith("<") or _BASE64_VALUE.match(value)
            out.append(f"{pair.group(1)}{pair.group(2)}: {replacement}" if pair and looks_secret else line)
            continue
        block_indent = None
        if _YAML_SECRET_BLOCK.match(line):
            block_indent = indent
        out.append(line)
    return "\n".join(out)


def purge_expired_archive_files(
    archive_directory: str | Path,
    *,
    retention_days: int = 90,
    now: datetime | None = None,
) -> list[Path]:
    """Delete archived Markdown files older than the configured retention window."""

    archive_root = Path(archive_directory)
    if not archive_root.exists():
        return []

    cutoff = (now or datetime.now(UTC)) - timedelta(days=retention_days)
    removed: list[Path] = []
    for path in archive_root.rglob("*.md"):
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if modified_at < cutoff:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed
