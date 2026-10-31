"""Markdown node parsing, serialization, and filesystem helpers."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from synapse.models.node import (
    Node,
    NodeMetadata,
    generate_node_id,
    infer_project,
    kebab_slug,
)


logger = logging.getLogger(__name__)

FRONTMATTER_BOUNDARY = "---"
WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")

_NODE_ID_DATE_RE = re.compile(r"^mem_(\d{8})_")

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
    """Legacy flat active path for a node id (``active/<id>.md``).

    Knowledge nodes now live at ``active/<project>/<kebab-slug>.md`` (see
    ``knowledge_relative_path``); this helper survives only for transcripts and
    for cleaning up pre-hierarchy layouts, and for callers that only know an id.
    """

    return Path(base_path) / "active" / f"{node_id}.md"


def knowledge_relative_path(node: Node) -> Path:
    """Canonical human-friendly path for a knowledge node, relative to base.

    ``active/<project>/<kebab-slug>.md`` — the node's identity stays in
    ``metadata.id`` (frontmatter), so nothing depends on the file name.
    """

    return Path("active") / infer_project(node) / f"{kebab_slug(node.metadata.id)}.md"


def active_node_file_path(base_path: str | Path, node: Node) -> Path:
    """Absolute canonical active path for a knowledge node."""

    return Path(base_path) / knowledge_relative_path(node)


def locate_active_node_file(base_path: str | Path, node_id: str) -> Path | None:
    """Find the Markdown file for *node_id* anywhere under ``active/``.

    Layout-agnostic: checks the flat legacy path, the canonical
    ``active/<project>/<kebab-slug>.md`` names, then falls back to a
    frontmatter scan so a node is always findable by its identity.
    """

    active_root = Path(base_path) / "active"
    if not active_root.exists():
        return None
    slug = kebab_slug(node_id)
    candidates = [
        active_root / f"{node_id}.md",
        *sorted(active_root.rglob(f"{slug}.md")),
        *sorted(active_root.rglob(f"{slug}-*.md")),
    ]
    for candidate in candidates:
        if candidate.is_file() and _markdown_frontmatter_id(candidate) == node_id:
            return candidate
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for candidate in sorted(active_root.rglob("*.md")):
        if _markdown_frontmatter_id(candidate) == node_id:
            return candidate
    return None


def transcript_node_path(base_path: str | Path, node_id: str) -> Path:
    """Return the canonical Markdown path for a raw session transcript.

    Session transcripts (``mem_session_*`` and the legacy session summaries)
    are storage-only raw material for the distiller; they live outside
    ``active/`` so the knowledge tree stays small and reviewable.
    """

    return Path(base_path) / "transcripts" / f"{node_id}.md"



def archive_node_path(archive_path: str | Path, node_id: str) -> Path:
    """Return the canonical archive Markdown path for a node."""

    return Path(archive_path) / f"{node_id}.md"



def _is_session_transcript(node: Node) -> bool:
    from synapse.okf.transcripts import is_session_transcript

    return is_session_transcript(node)


def _markdown_frontmatter_id(path: Path) -> str | None:
    """Read the ``id`` field from a Markdown file's frontmatter (None when absent)."""

    try:
        payload, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    value = payload.get("id")
    return str(value) if value else None


def _node_date_part(node_id: str) -> str:
    """``YYYYMMDD`` from a ``mem_<date>_`` id, else a short id hash."""

    match = _NODE_ID_DATE_RE.match(node_id)
    if match is not None:
        return match.group(1)
    return hashlib.sha1(node_id.encode("utf-8")).hexdigest()[:6]


def _with_inferred_project(node: Node) -> Node:
    """Make every knowledge node's ``metadata.project`` explicit before writing."""

    if (node.metadata.project or "").strip():
        return node
    project = infer_project(node)
    return node.model_copy(update={"metadata": node.metadata.model_copy(update={"project": project})})


def _path_owned_by_other(path: Path, node_id: str) -> bool:
    """True when *path* exists and belongs to a different node (or has no id)."""

    if not path.exists():
        return False
    return _markdown_frontmatter_id(path) != node_id


def _knowledge_destination(base_path: Path, node: Node) -> Path:
    """Canonical ``active/<project>/<kebab-slug>.md`` path, collision-safe.

    Two nodes whose ids differ only by date share a kebab slug; the first keeps
    the bare slug, later ones append their date (then a short id hash) so
    distinct nodes can never overwrite one another.
    """

    directory = base_path / "active" / infer_project(node)
    slug = kebab_slug(node.metadata.id)
    candidate = directory / f"{slug}.md"
    if _path_owned_by_other(candidate, node.metadata.id):
        candidate = directory / f"{slug}-{_node_date_part(node.metadata.id)}.md"
        if _path_owned_by_other(candidate, node.metadata.id):
            digest = hashlib.sha1(node.metadata.id.encode("utf-8")).hexdigest()[:8]
            candidate = directory / f"{slug}-{digest}.md"
    return candidate


def _stale_active_file(base_path: Path, node: Node, destination: Path) -> Path | None:
    """A pre-hierarchy copy of *node* that the new destination supersedes.

    The node moved from a flat ``active/<id>.md`` (or another project folder) to
    its canonical path; leaving the old file behind would re-index a duplicate.
    Only files inside ``active/`` that still carry this node's id are removed.
    """

    raw = node.file_path
    if raw is None or str(raw) in {"", "."}:
        return None
    old_path = raw if raw.is_absolute() else base_path / raw
    if not old_path.exists():
        return None
    try:
        if old_path.resolve() == destination.resolve():
            return None
    except OSError:  # pragma: no cover - defensive
        return None
    active_root = (base_path / "active").resolve()
    try:
        old_path.resolve().relative_to(active_root)
    except ValueError:
        return None  # outside active/ — not ours to move
    if _markdown_frontmatter_id(old_path) != node.metadata.id:
        return None
    return old_path


def write_node_file(
    node: Node,
    *,
    base_path: str | Path | None = None,
    output_path: str | Path | None = None,
    include_banners: bool = True,
    supersession_reason: str | None = None,
) -> Path:
    """Serialize and atomically write a node to disk.

    Knowledge nodes written with ``base_path`` land at their canonical
    ``active/<project>/<kebab-slug>.md`` location (the id lives in the
    frontmatter, not the filename); session transcripts stay in
    ``transcripts/<id>.md``.
    """

    validation = node.word_count_validation()
    if validation.warning and not _is_session_transcript(node):
        # Session transcripts legitimately exceed the knowledge-node word
        # guideline (the bridge sends up to 16k chars); stay silent for them.
        logger.warning(validation.warning, extra={"node_id": node.metadata.id})

    stale_path: Path | None = None
    if output_path is not None:
        destination = Path(output_path)
    elif base_path is not None:
        base = Path(base_path)
        # Session transcripts are raw distiller input, not knowledge: keep
        # them out of active/ so the knowledge tree stays small.
        if _is_session_transcript(node):
            destination = transcript_node_path(base, node.metadata.id)
            # A legacy copy can still sit in active/ (pre-transcripts layout).
            # Leaving it behind would re-index as a duplicate node, so the
            # write completes the move.
            legacy = active_node_path(base, node.metadata.id)
            if legacy != destination:
                stale_path = legacy
        else:
            node = _with_inferred_project(node)
            destination = _knowledge_destination(base, node)
            stale_path = _stale_active_file(base, node, destination)
    elif node.file_path and str(node.file_path) not in {"", "."}:
        destination = Path(node.file_path)
    else:
        raise ValueError("Either output_path, base_path, or node.file_path must be provided.")

    markdown = node_to_markdown(node, include_banners=include_banners, supersession_reason=supersession_reason)
    written = atomic_write_text(destination, markdown)
    if stale_path is not None:
        try:
            stale_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to remove superseded active file after write",
                extra={"node_id": node.metadata.id, "path": str(stale_path), "error": str(exc)},
            )
    return written



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
