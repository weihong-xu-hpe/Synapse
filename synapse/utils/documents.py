"""Shared document rendering helpers for embeddings and reranking."""

from __future__ import annotations

from synapse.models import Node


def render_node_document(node: Node) -> str:
    """Render a stable text representation of a node for models."""

    tags = ", ".join(node.metadata.tags)
    sections = [node.title.strip()]
    if tags:
        sections.append(f"Tags: {tags}")
    if node.content.strip():
        sections.append(node.content.strip())
    return "\n\n".join(section for section in sections if section)