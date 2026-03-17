"""Core protocols and shared data structures for Synapse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(slots=True, frozen=True)
class SearchQuery:
    """Normalized search request used by search abstractions."""

    text: str
    top_k: int = 3


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
