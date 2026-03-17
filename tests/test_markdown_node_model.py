from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from synapse.models import Node, NodeMetadata, NodeStatus, NodeType, SensitivityLevel, count_text_words
from synapse.storage import (
    add_supersession_banners,
    extract_wiki_links,
    node_from_markdown,
    node_to_markdown,
    remove_supersession_banners,
    scan_markdown_nodes,
    write_node_file,
)


FIXED_TIME = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)


def make_node(**metadata_overrides) -> Node:
    metadata_payload = {
        "id": "mem_20260301_rate_limiting",
        "title": "Token-based Microservice Rate Limiting",
        "created_at": FIXED_TIME,
        "last_accessed": datetime(2026, 3, 5, 14, 30, tzinfo=UTC),
        "access_count": 5,
        "type": NodeType.PERSISTENT,
        "status": NodeStatus.ACTIVE,
        "supersedes": ["mem_20260228_old_rate_limits"],
        "superseded_by": None,
        "sensitivity": SensitivityLevel.INTERNAL,
    }
    metadata_payload.update(metadata_overrides)
    metadata = NodeMetadata(**metadata_payload)
    return Node(
        metadata=metadata,
        content=(
            "# 背景\n\n"
            "This design refines [[API_Gateway_Design]] and [[AuthZ_Architecture_V2|AuthZ v2]].\n"
            "我们需要更稳定的限流策略。"
        ),
        file_path=Path("active/mem_20260301_rate_limiting.md"),
    )


def test_markdown_round_trip_preserves_metadata_and_content() -> None:
    original = make_node()

    markdown_text = node_to_markdown(original, supersession_reason="Replaced rate-window heuristic")
    restored = node_from_markdown(markdown_text, file_path=original.file_path)

    assert restored.metadata.id == original.metadata.id
    assert restored.metadata.title == original.metadata.title
    assert restored.metadata.type is NodeType.PERSISTENT
    assert restored.metadata.supersedes == ["mem_20260228_old_rate_limits"]
    assert restored.metadata.sensitivity is SensitivityLevel.INTERNAL
    assert restored.content == original.content


def test_markdown_parser_supports_missing_optional_fields() -> None:
    markdown_text = """---
title: Lightweight idea
---

# Lightweight idea

A tiny note with [[Linked_Node]].
"""

    restored = node_from_markdown(markdown_text, file_path="notes/lightweight-idea.md")

    assert restored.metadata.title == "Lightweight idea"
    assert restored.metadata.status is NodeStatus.ACTIVE
    assert restored.metadata.sensitivity is SensitivityLevel.INTERNAL
    assert restored.metadata.access_count == 0
    assert restored.metadata.id.startswith("mem_")


def test_extract_wiki_links_handles_mixed_language_and_aliases() -> None:
    content = "See [[Alpha_Node]] and [[设计方案]] plus [[Gamma_Node|Friendly label]]."

    assert extract_wiki_links(content) == ["Alpha_Node", "设计方案", "Gamma_Node"]


def test_tier_word_count_validation_and_scan_helpers(tmp_path: Path) -> None:
    base_path = tmp_path / ".synapse"
    first = make_node()
    second = make_node(
        id="mem_20260301_api_gateway_design",
        title="API Gateway Design",
        supersedes=[],
        sensitivity=SensitivityLevel.PUBLIC,
    ).model_copy(
        update={
            "content": "Gateway design with bilingual text. 架构 设计 pattern review.",
            "file_path": Path("active/mem_20260301_api_gateway_design.md"),
        }
    )

    write_node_file(first, base_path=base_path, supersession_reason="Historical context only")
    write_node_file(second, base_path=base_path)

    scanned = scan_markdown_nodes(base_path / "active", relative_to=base_path)
    scanned_ids = [node.metadata.id for node in scanned]

    assert scanned_ids == sorted([first.metadata.id, second.metadata.id])
    assert all(node.file_path.parts[0] == "active" for node in scanned)
    assert count_text_words("设计 alpha beta") == 4

    base_node = make_node()
    oversized = base_node.model_copy(
        update={
            "content": "word " * 3501,
        }
    )
    validation = oversized.word_count_validation()
    assert validation.within_limit is False
    assert validation.warning is not None


def test_supersession_banner_helpers_add_and_remove_cleanly() -> None:
    body = "Current content stays intact."

    with_banners = add_supersession_banners(
        body,
        superseded_by="mem_20260315_new_node",
        supersedes=["mem_20260301_old_node"],
        reason="Merged findings",
        on_date=datetime(2026, 3, 15, tzinfo=UTC),
    )

    assert "> ⚠️ **SUPERSEDED** by [[mem_20260315_new_node]] on 2026-03-15." in with_banners
    assert "> **Supersedes**: [[mem_20260301_old_node]] — Merged findings" in with_banners
    assert remove_supersession_banners(with_banners) == body
