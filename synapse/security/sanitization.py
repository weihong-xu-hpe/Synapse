"""Security-focused redaction, sensitivity filtering, and audit helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

from synapse.config import SynapseConfig
from synapse.models.node import Node, SensitivityLevel
from synapse.storage.markdown import atomic_write_text


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


@dataclass(slots=True, frozen=True)
class SanitizedPayloadBatch:
    """Sanitized payloads prepared for cloud transmission."""

    payloads: list[str]
    node_ids_sent: list[str]
    skipped_private_node_ids: list[str]
    redaction_counts: dict[str, int]
    payload_hash: str
    audit_log_path: Path | None = None


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
)


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


class SensitivityFilter:
    """Decide which nodes may be transmitted to a cloud LLM."""

    def __init__(self, redaction_engine: RedactionEngine | None = None) -> None:
        self.redaction_engine = redaction_engine or RedactionEngine()

    def can_transmit(self, node: Node) -> bool:
        return node.metadata.sensitivity is not SensitivityLevel.PRIVATE

    def prepare_for_cloud(self, node: Node) -> RedactionResult:
        if node.metadata.sensitivity is SensitivityLevel.PRIVATE:
            raise PermissionError(f"Node '{node.metadata.id}' is private and cannot be transmitted.")
        if node.metadata.sensitivity is SensitivityLevel.PUBLIC:
            return RedactionResult(text=node.content, labels=[])
        return self.redaction_engine.redact_detailed(node.content)


class AuditLogWriter:
    """Persist transmission audit entries beneath `.synapse/.audit`."""

    def __init__(self, audit_directory: str | Path) -> None:
        self.audit_directory = Path(audit_directory)
        self.audit_directory.mkdir(parents=True, exist_ok=True)
        try:
            self.audit_directory.chmod(0o700)
        except OSError:
            pass

    def write_entry(
        self,
        *,
        operation: str,
        node_ids_sent: Sequence[str],
        payloads: Sequence[str],
        redaction_counts: dict[str, int],
        llm_response_summary: str = "",
        timestamp: datetime | None = None,
    ) -> Path:
        event_time = (timestamp or datetime.now(UTC)).astimezone(UTC)
        payload_hash = compute_payload_hash(payloads)
        payload = {
            "timestamp": event_time.isoformat().replace("+00:00", "Z"),
            "operation": operation,
            "node_ids_sent": list(node_ids_sent),
            "redacted_payload_hash": payload_hash,
            "llm_response_summary": llm_response_summary,
            "redactions_applied": [f"{label}: {count}" for label, count in sorted(redaction_counts.items())],
        }
        file_name = f"{payload['timestamp'].replace(':', '').replace('-', '')}_{operation}.json"
        destination = self.audit_directory / file_name
        atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return destination



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



def compute_payload_hash(payloads: Iterable[str]) -> str:
    """Compute a stable hash for a sanitized outbound payload batch."""

    joined = "\u241e".join(payloads)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"



def sanitize_nodes_for_cloud(
    nodes: Sequence[Node],
    *,
    config: SynapseConfig | None = None,
    audit_dir: str | Path | None = None,
    operation: str = "cloud_payload",
    llm_response_summary: str = "",
    timestamp: datetime | None = None,
) -> SanitizedPayloadBatch:
    """Prepare nodes for cloud transmission without mutating source content."""

    redaction_engine = RedactionEngine.from_config(config) if config is not None else RedactionEngine()
    filter_engine = SensitivityFilter(redaction_engine)

    payloads: list[str] = []
    node_ids_sent: list[str] = []
    skipped_private_node_ids: list[str] = []
    aggregate_counts: Counter[str] = Counter()

    for node in nodes:
        if not filter_engine.can_transmit(node):
            skipped_private_node_ids.append(node.metadata.id)
            continue
        prepared = filter_engine.prepare_for_cloud(node)
        payloads.append(prepared.text)
        node_ids_sent.append(node.metadata.id)
        aggregate_counts.update(prepared.labels)

    payload_hash = compute_payload_hash(payloads)
    audit_log_path: Path | None = None
    resolved_audit_dir = Path(audit_dir) if audit_dir is not None else None
    should_audit = resolved_audit_dir is not None and bool(node_ids_sent)
    if should_audit and resolved_audit_dir is not None:
        writer = AuditLogWriter(resolved_audit_dir)
        audit_log_path = writer.write_entry(
            operation=operation,
            node_ids_sent=node_ids_sent,
            payloads=payloads,
            redaction_counts=dict(aggregate_counts),
            llm_response_summary=llm_response_summary,
            timestamp=timestamp,
        )

    return SanitizedPayloadBatch(
        payloads=payloads,
        node_ids_sent=node_ids_sent,
        skipped_private_node_ids=skipped_private_node_ids,
        redaction_counts=dict(aggregate_counts),
        payload_hash=payload_hash,
        audit_log_path=audit_log_path,
    )



def sanitize_for_cloud(
    nodes: Sequence[Node],
    *,
    config: SynapseConfig | None = None,
    audit_dir: str | Path | None = None,
    operation: str = "cloud_payload",
    llm_response_summary: str = "",
    timestamp: datetime | None = None,
) -> list[str]:
    """High-level convenience wrapper returning only payload strings."""

    batch = sanitize_nodes_for_cloud(
        nodes,
        config=config,
        audit_dir=audit_dir,
        operation=operation,
        llm_response_summary=llm_response_summary,
        timestamp=timestamp,
    )
    return batch.payloads



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



def resolve_audit_directory(config: SynapseConfig) -> Path:
    """Resolve the audit directory from configuration."""

    audit_dir = config.resolve_path(config.memory.base_path) / ".audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    try:
        audit_dir.chmod(0o700)
    except OSError:
        pass
    return audit_dir
