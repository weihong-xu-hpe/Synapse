"""Embedding package for Synapse."""

from synapse.embedding.engines import (
	DEFAULT_REGISTRY,
	EMBEDDING_MODEL_SPECS,
	HTTPJSONClient,
	ProviderError,
	RemoteAPIEmbeddingEngine,
	RemoteAPIRerankerEngine,
	RERANKER_MODEL_SPECS,
	DeterministicEmbeddingEngine,
	DeterministicRerankerEngine,
	EmbeddingModelSpec,
	EngineRegistry,
	RerankerModelSpec,
	UnavailableEmbeddingEngine,
	UnavailableRerankerEngine,
	create_embedding_engine,
	create_reranker_engine,
)

__all__ = [
	"DEFAULT_REGISTRY",
	"EMBEDDING_MODEL_SPECS",
	"HTTPJSONClient",
	"RERANKER_MODEL_SPECS",
	"DeterministicEmbeddingEngine",
	"DeterministicRerankerEngine",
	"EmbeddingModelSpec",
	"EngineRegistry",
	"ProviderError",
	"RerankerModelSpec",
	"RemoteAPIEmbeddingEngine",
	"RemoteAPIRerankerEngine",
	"UnavailableEmbeddingEngine",
	"UnavailableRerankerEngine",
	"create_embedding_engine",
	"create_reranker_engine",
]
