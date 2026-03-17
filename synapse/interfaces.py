"""Core protocols and lightweight shared data structures for future phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(slots=True, frozen=True)
class MemoryNode:
    """Minimal node representation for storage and retrieval abstractions."""

    id: str
    title: str
    path: Path
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SearchQuery:
    """Normalized search request used by search abstractions."""

    text: str
    top_k: int = 3


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Search result contract shared by retrieval implementations."""

    node_id: str
    score: float
    snippet: str = ""


class EmbeddingEngine(Protocol):
    """Abstraction for embedding backends."""

    model_name: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        """Convert a single text into a vector embedding."""

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert multiple texts into vector embeddings."""

    def is_available(self) -> bool:
        """Return whether the engine can currently serve requests."""


class RerankerEngine(Protocol):
    """Abstraction for reranking candidate documents."""

    model_name: str

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        limit: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return ranked `(document_index, score)` tuples."""

    def is_available(self) -> bool:
        """Return whether the engine can currently serve requests."""


class SearchEngine(Protocol):
    """Abstraction for hybrid retrieval engines."""

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for relevant nodes."""


class NodeStore(Protocol):
    """Abstraction for persistent node storage."""

    def upsert(self, node: MemoryNode) -> None:
        """Create or update a node."""

    def get(self, node_id: str) -> MemoryNode | None:
        """Fetch a node by ID if present."""

    def list(self) -> list[MemoryNode]:
        """List all currently known nodes."""

    def delete(self, node_id: str) -> bool:
        """Delete a node by ID, returning whether it existed."""
