from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from synapse.config import SynapseConfig
from synapse.models import Node, NodeMetadata, SensitivityLevel
from synapse.security import RedactionEngine, purge_expired_archive_files, sanitize_for_cloud, sanitize_nodes_for_cloud


FIXED_TIME = datetime(2026, 3, 1, 10, 30, tzinfo=UTC)


def make_node(node_id: str, content: str, sensitivity: SensitivityLevel) -> Node:
    return Node(
        metadata=NodeMetadata(
            id=node_id,
            title=node_id,
            created_at=FIXED_TIME,
            sensitivity=sensitivity,
        ),
        content=content,
        file_path=Path(f"active/{node_id}.md"),
    )


def test_redaction_engine_handles_default_patterns() -> None:
    text = (
        "Reach me at owner@example.com from 10.0.0.5 with key "
        "sk-1234567890abcdef1234567890abcdef and uuid "
        "123e4567-e89b-12d3-a456-426614174000 plus token "
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.signature"
    )

    redacted, labels = RedactionEngine().redact(text)

    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_IP]" in redacted
    assert "[REDACTED_API_KEY]" in redacted
    assert "[REDACTED_UUID]" in redacted
    assert "[REDACTED_JWT]" in redacted
    assert sorted(labels) == ["API_KEY", "EMAIL", "IP", "JWT", "UUID"]


def test_custom_patterns_from_config_are_applied() -> None:
    config = SynapseConfig.model_validate(
        {
            "sanitization": {
                "custom_patterns": [
                    {
                        "pattern": r"ACME-[0-9]{4}",
                        "replacement": "[REDACTED_ACME]",
                        "label": "ACME_TOKEN",
                    }
                ]
            }
        }
    )

    batch = sanitize_nodes_for_cloud(
        [make_node("mem_custom", "Secret ACME-4242 token", SensitivityLevel.INTERNAL)],
        config=config,
    )

    assert batch.payloads == ["Secret [REDACTED_ACME] token"]
    assert batch.redaction_counts == {"ACME_TOKEN": 1}


def test_sanitize_nodes_for_cloud_respects_sensitivity_and_writes_audit_log(tmp_path: Path) -> None:
    public_node = make_node("mem_public", "Public summary for release notes.", SensitivityLevel.PUBLIC)
    internal_node = make_node(
        "mem_internal",
        "Contact admin@example.com with key sk-1234567890abcdef1234567890abcdef.",
        SensitivityLevel.INTERNAL,
    )
    private_node = make_node("mem_private", "Private incident analysis", SensitivityLevel.PRIVATE)
    audit_dir = tmp_path / ".synapse" / ".audit"

    batch = sanitize_nodes_for_cloud(
        [public_node, internal_node, private_node],
        audit_dir=audit_dir,
        operation="condense_batch",
        llm_response_summary="Synthesized 2 nodes into 1",
        timestamp=FIXED_TIME,
    )

    assert batch.node_ids_sent == ["mem_public", "mem_internal"]
    assert batch.skipped_private_node_ids == ["mem_private"]
    assert batch.payloads[0] == public_node.content
    assert "[REDACTED_EMAIL]" in batch.payloads[1]
    assert "[REDACTED_API_KEY]" in batch.payloads[1]
    assert batch.redaction_counts == {"API_KEY": 1, "EMAIL": 1}
    assert internal_node.content.endswith("sk-1234567890abcdef1234567890abcdef.")
    assert batch.audit_log_path is not None
    assert batch.audit_log_path.exists()

    payload = json.loads(batch.audit_log_path.read_text(encoding="utf-8"))
    assert payload["operation"] == "condense_batch"
    assert payload["node_ids_sent"] == ["mem_public", "mem_internal"]
    assert payload["redacted_payload_hash"].startswith("sha256:")
    assert payload["redactions_applied"] == ["API_KEY: 1", "EMAIL: 1"]

    payloads_only = sanitize_for_cloud([internal_node])
    assert payloads_only == [batch.payloads[1]]


def test_purge_expired_archive_files_removes_only_old_markdown(tmp_path: Path) -> None:
    archive_root = tmp_path / ".synapse" / ".archive"
    archive_root.mkdir(parents=True)
    stale = archive_root / "stale.md"
    fresh = archive_root / "fresh.md"
    stale.write_text("old", encoding="utf-8")
    fresh.write_text("new", encoding="utf-8")

    old_timestamp = (FIXED_TIME - timedelta(days=120)).timestamp()
    new_timestamp = (FIXED_TIME - timedelta(days=5)).timestamp()
    stale.touch()
    fresh.touch()
    stale.chmod(0o600)
    fresh.chmod(0o600)
    import os

    os.utime(stale, (old_timestamp, old_timestamp))
    os.utime(fresh, (new_timestamp, new_timestamp))

    removed = purge_expired_archive_files(archive_root, retention_days=90, now=FIXED_TIME)

    assert removed == [stale]
    assert stale.exists() is False
    assert fresh.exists() is True
