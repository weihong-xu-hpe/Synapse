"""Markdown node parsing, serialization, and filesystem helpers."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from synapse.models.node import Node, NodeMetadata, generate_node_id


logger = logging.getLogger(__name__)

FRONTMATTER_BOUNDARY = "---"
WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")

_SUPERSEDED_NOTICE_PREFIX = "> ⚠️ **SUPERSEDED** by [["
_SUPERSEDED_NOTICE_CONTEXT = "> This node is retained for historical context but its conclusions are outdated."
_SUPERSEDES_NOTICE_PREFIX = "> **Supersedes**: [["


def _skip_banner_block(lines: list[str], index: int) -> int:
    line = lines[index]
    if line.startswith(_SUPERSEDED_NOTICE_PREFIX):
        next_index = index + 1
        if next_index < len(lines) and lines[next_index] == _SUPERSEDED_NOTICE_CONTEXT:
            next_index += 1
        if next_index < len(lines) and lines[next_index] == "":
            next_index += 1
        return next_index
    if line.startswith(_SUPERSEDES_NOTICE_PREFIX):
        next_index = index + 1
        if next_index < len(lines) and lines[next_index] == "":
            next_index += 1
        return next_index
    return index


def extract_wiki_links(content: str) -> list[str]:
    """Extract outgoing wiki links from a Markdown body."""

    links: list[str] = []
    for raw_match in WIKI_LINK_PATTERN.findall(content):
        target = raw_match.split("|", 1)[0].strip()
        if target:
            links.append(target)
    return links



def split_frontmatter(markdown_text: str) -> tuple[dict[str, Any], str]:
    """Split a Markdown document into frontmatter mapping and body."""

    normalized = markdown_text.replace("\r\n", "\n")
    if not normalized.startswith(f"{FRONTMATTER_BOUNDARY}\n"):
        return {}, normalized

    closing_marker = f"\n{FRONTMATTER_BOUNDARY}\n"
    end_index = normalized.find(closing_marker, len(FRONTMATTER_BOUNDARY) + 1)
    if end_index == -1:
        closing_marker = f"\n{FRONTMATTER_BOUNDARY}"
        end_index = normalized.find(closing_marker, len(FRONTMATTER_BOUNDARY) + 1)
        if end_index == -1:
            return {}, normalized
        body_start = end_index + len(closing_marker)
        if body_start < len(normalized) and normalized[body_start] == "\n":
            body_start += 1
    else:
        body_start = end_index + len(closing_marker)

    frontmatter_text = normalized[len(FRONTMATTER_BOUNDARY) + 1 : end_index]
    body = normalized[body_start:]
    payload = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(payload, dict):
        raise ValueError("Markdown frontmatter must deserialize to a mapping.")
    return payload, body



def infer_title(markdown_body: str, file_path: str | Path | None = None) -> str:
    """Infer a node title from the first heading or file name."""

    for line in markdown_body.splitlines():
        candidate = line.strip()
        if candidate.startswith("#"):
            heading = candidate.lstrip("#").strip()
            if heading:
                return heading
    if file_path is not None:
        stem = Path(file_path).stem.replace("_", " ").replace("-", " ").strip()
        if stem:
            return stem.title()
    return "Untitled Node"



def serialize_frontmatter(metadata: NodeMetadata | Mapping[str, Any]) -> str:
    """Serialize metadata into YAML frontmatter."""

    if isinstance(metadata, NodeMetadata):
        payload = metadata.model_dump(mode="json")
    else:
        payload = dict(metadata)
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    return f"{FRONTMATTER_BOUNDARY}\n{rendered}{FRONTMATTER_BOUNDARY}\n"



def remove_supersession_banners(content: str) -> str:
    """Remove generated supersession banners from the start of a Markdown body."""

    lines = content.replace("\r\n", "\n").lstrip("\n").split("\n")
    index = 0

    while index < len(lines):
        next_index = _skip_banner_block(lines, index)
        if next_index != index:
            index = next_index
            continue
        break

    remaining = "\n".join(lines[index:])
    return remaining.lstrip("\n")



def _format_superseded_banner(new_node_id: str, on_date: date | datetime | None = None) -> str:
    banner_date: date
    if on_date is None:
        banner_date = datetime.now(UTC).date()
    elif isinstance(on_date, datetime):
        banner_date = on_date.astimezone(UTC).date() if on_date.tzinfo else on_date.date()
    else:
        banner_date = on_date
    return (
        f"> ⚠️ **SUPERSEDED** by [[{new_node_id}]] on {banner_date.isoformat()}.\n"
        f"{_SUPERSEDED_NOTICE_CONTEXT}"
    )



def _format_supersedes_banner(node_id: str, reason: str | None = None) -> str:
    suffix = f" — {reason.strip()}" if reason and reason.strip() else ""
    return f"> **Supersedes**: [[{node_id}]]{suffix}"



def add_supersession_banners(
    content: str,
    *,
    superseded_by: str | None = None,
    supersedes: str | Sequence[str] | None = None,
    reason: str | None = None,
    on_date: date | datetime | None = None,
) -> str:
    """Add generated supersession banners to a Markdown body."""

    clean_body = remove_supersession_banners(content)
    banner_lines: list[str] = []

    if superseded_by:
        banner_lines.append(_format_superseded_banner(superseded_by, on_date=on_date))

    if supersedes:
        items = [supersedes] if isinstance(supersedes, str) else [item for item in supersedes if item]
        banner_lines.extend(_format_supersedes_banner(item, reason=reason) for item in items)

    if not banner_lines:
        return clean_body

    banner_block = "\n".join(banner_lines)
    if clean_body:
        return f"{banner_block}\n\n{clean_body}"
    return banner_block



def node_from_markdown(markdown_text: str, file_path: str | Path) -> Node:
    """Parse a Markdown document into a normalized `Node`."""

    frontmatter, body = split_frontmatter(markdown_text)
    title = str(frontmatter.get("title") or infer_title(body, file_path))
    frontmatter.setdefault("title", title)
    frontmatter.setdefault("created_at", datetime.now(UTC))
    frontmatter.setdefault("id", generate_node_id(title, current_date=frontmatter.get("created_at")))
    metadata = NodeMetadata.model_validate(frontmatter)
    return Node(
        metadata=metadata,
        content=remove_supersession_banners(body).rstrip("\n"),
        file_path=Path(file_path),
    )



def node_to_markdown(node: Node, *, include_banners: bool = True, supersession_reason: str | None = None) -> str:
    """Serialize a `Node` back to Markdown frontmatter plus body content."""

    body = remove_supersession_banners(node.content)
    if include_banners:
        body = add_supersession_banners(
            body,
            superseded_by=node.metadata.superseded_by,
            supersedes=node.metadata.supersedes,
            reason=supersession_reason,
            on_date=node.metadata.last_accessed,
        )
    payload = serialize_frontmatter(node.metadata)
    normalized_body = body.rstrip("\n")
    if normalized_body:
        return f"{payload}\n{normalized_body}\n"
    return payload



def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Write text atomically by replacing the destination with a sibling temp file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return destination



def active_node_path(base_path: str | Path, node_id: str) -> Path:
    """Return the canonical active Markdown path for a node."""

    return Path(base_path) / "active" / f"{node_id}.md"



def archive_node_path(archive_path: str | Path, node_id: str) -> Path:
    """Return the canonical archive Markdown path for a node."""

    return Path(archive_path) / f"{node_id}.md"



def write_node_file(
    node: Node,
    *,
    base_path: str | Path | None = None,
    output_path: str | Path | None = None,
    include_banners: bool = True,
    supersession_reason: str | None = None,
) -> Path:
    """Serialize and atomically write a node to disk."""

    validation = node.word_count_validation()
    if validation.warning:
        logger.warning(validation.warning, extra={"node_id": node.metadata.id})

    if output_path is not None:
        destination = Path(output_path)
    elif base_path is not None:
        destination = active_node_path(base_path, node.metadata.id)
    elif node.file_path and str(node.file_path) not in {"", "."}:
        destination = Path(node.file_path)
    else:
        raise ValueError("Either output_path, base_path, or node.file_path must be provided.")

    markdown = node_to_markdown(node, include_banners=include_banners, supersession_reason=supersession_reason)
    return atomic_write_text(destination, markdown)



def read_node_file(path: str | Path) -> Node:
    """Read a Markdown file from disk and parse it into a `Node`."""

    node_path = Path(path)
    markdown_text = node_path.read_text(encoding="utf-8")
    return node_from_markdown(markdown_text, file_path=node_path)



def scan_markdown_files(directory: str | Path) -> list[Path]:
    """Recursively list Markdown files below a directory."""

    root = Path(directory)
    return sorted(path for path in root.rglob("*.md") if path.is_file())



def scan_markdown_nodes(directory: str | Path, *, relative_to: str | Path | None = None) -> list[Node]:
    """Recursively load all Markdown nodes below a directory."""

    root = Path(directory)
    base = Path(relative_to) if relative_to is not None else root
    nodes: list[Node] = []
    for path in scan_markdown_files(root):
        node = read_node_file(path)
        try:
            relative_path = path.relative_to(base)
        except ValueError:
            relative_path = path
        nodes.append(node.model_copy(update={"file_path": relative_path}))
    return nodes
