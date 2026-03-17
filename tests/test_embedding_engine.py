from __future__ import annotations

import json
from urllib.error import URLError

import synapse.embedding.engines as engine_module
from synapse.config import (
    EmbeddingSettings,
    ProviderSettings,
    RemoteAPIProviderSettings,
    RerankerSettings,
)
from synapse.embedding import (
    RemoteAPIEmbeddingEngine,
    RemoteAPIRerankerEngine,
    create_embedding_engine,
    create_reranker_engine,
)


class FakeHTTPResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


def _vector(dimension: int, value: float = 0.1) -> list[float]:
    return [value] * dimension


def test_embedding_dimension_and_determinism_for_bge_m3() -> None:
    engine = create_embedding_engine(EmbeddingSettings(provider="builtin", model="bge-m3"))

    first = engine.embed("Rate limiting for API gateways. 混合语言检索。")
    second = engine.embed("Rate limiting for API gateways. 混合语言检索。")

    assert engine.is_available() is True
    assert len(first) == 1024
    assert first == second


def test_model_switching_changes_vector_space_and_dimension() -> None:
    bge_engine = create_embedding_engine(EmbeddingSettings(provider="builtin", model="bge-m3"))
    jina_engine = create_embedding_engine(EmbeddingSettings(provider="builtin", model="jina-v3"))
    gte_engine = create_embedding_engine(EmbeddingSettings(provider="builtin", model="gte-qwen2", dimension=1536))

    text = "authentication design decisions"

    assert len(jina_engine.embed(text)) == 1024
    assert len(gte_engine.embed(text)) == 1536
    assert bge_engine.embed(text) != jina_engine.embed(text)


def test_reranker_orders_documents_by_relevance() -> None:
    reranker = create_reranker_engine(RerankerSettings(provider="builtin", model="bge-reranker-v2-m3"))
    query = "api gateway rate limiting"
    documents = [
        "A glossary entry about logging backends.",
        "API gateway rate limiting design with token buckets and quotas.",
        "Gateway patterns mention retries but not rate limits.",
    ]

    ranked = reranker.rerank(query, documents)

    assert reranker.is_available() is True
    assert [index for index, _ in ranked] == [1, 2, 0]
    assert ranked[0][1] > ranked[1][1] > ranked[2][1]


def test_remote_api_provider_applies_headers_and_auth(monkeypatch) -> None:
    captured_requests: list[tuple[str, dict[str, str], dict[str, object]]] = []
    monkeypatch.setenv("SYNAPSE_MODEL_API_KEY", "secret-token")

    def fake_urlopen(request, timeout):
        del timeout
        headers = {key.casefold(): value for key, value in request.header_items()}
        payload = json.loads(request.data.decode("utf-8"))
        captured_requests.append((request.full_url, headers, payload))
        if request.full_url.endswith("/v1/embeddings"):
            return FakeHTTPResponse({"data": [{"embedding": _vector(1024, value=0.25)}]})
        if request.full_url.endswith("/v1/rerank"):
            return FakeHTTPResponse(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.33},
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)

    providers = ProviderSettings(
        remote_api=RemoteAPIProviderSettings(
            base_url="https://models.example.com",
            embedding_endpoint="/v1/embeddings",
            rerank_endpoint="/v1/rerank",
            api_key_env="SYNAPSE_MODEL_API_KEY",
            headers={"X-Tenant": "dev", "X-API-Key": "{api_key}"},
            request_timeout_seconds=15,
        )
    )

    embedding_engine = create_embedding_engine(
        EmbeddingSettings(provider="remote_api", model="bge-m3"),
        providers=providers,
    )
    reranker_engine = create_reranker_engine(
        RerankerSettings(provider="remote_api", model="bge-reranker-v2-m3"),
        providers=providers,
    )

    assert isinstance(embedding_engine, RemoteAPIEmbeddingEngine)
    assert isinstance(reranker_engine, RemoteAPIRerankerEngine)
    assert len(embedding_engine.embed("remote embedding")) == 1024
    assert reranker_engine.rerank("query", ["first", "second"])[0][0] == 1

    assert len(captured_requests) == 2
    for _, headers, _ in captured_requests:
        assert headers["authorization"] == "Bearer secret-token"
        assert headers["x-api-key"] == "secret-token"
        assert headers["x-tenant"] == "dev"


def test_unavailable_provider_degrades_gracefully(monkeypatch) -> None:
    def failing_urlopen(request, timeout):
        del request, timeout
        raise URLError("connection refused")

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", failing_urlopen)

    providers = ProviderSettings(
        remote_api=RemoteAPIProviderSettings(
            base_url="https://models.example.com",
            embedding_endpoint="/v1/embeddings",
            rerank_endpoint="/v1/rerank",
        )
    )
    embedding = create_embedding_engine(
        EmbeddingSettings(provider="remote_api", model="bge-m3"),
        providers=providers,
    )
    reranker = create_reranker_engine(
        RerankerSettings(provider="remote_api", model="jina-reranker-v2"),
        providers=providers,
    )

    assert embedding.is_available() is False
    assert len(embedding.embed("ignored")) == 1024
    assert reranker.is_available() is False
    assert reranker.rerank("query", ["one query", "two"], limit=1)[0][0] == 0


def test_degradation_behavior_when_fallback_is_disabled() -> None:
    embedding = create_embedding_engine(
        EmbeddingSettings(provider="builtin", model="bge-m3"),
        allow_fallback=False,
    )
    reranker = create_reranker_engine(
        RerankerSettings(provider="builtin", model="jina-reranker-v2"),
        allow_fallback=False,
    )

    assert embedding.is_available() is False
    assert embedding.embed("ignored") == []
    assert reranker.is_available() is False
    assert reranker.rerank("query", ["one", "two"], limit=1) == [(0, 0.0)]



