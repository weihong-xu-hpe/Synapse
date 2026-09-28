from __future__ import annotations

import json
import logging
from urllib.error import URLError

import pytest

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
        if request.full_url.endswith("/tokenize"):
            return FakeHTTPResponse({"tokens": list(range(len(payload["content"])))})
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


def test_remote_embedding_batches_large_requests_without_falling_back_whole_batch(monkeypatch) -> None:
    captured_inputs: list[list[str]] = []

    def fake_urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/tokenize"):
            return FakeHTTPResponse({"tokens": list(range(len(payload["content"])))})
        inputs = payload["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        captured_inputs.append(inputs)
        return FakeHTTPResponse({"data": [{"embedding": _vector(1024, value=0.25)} for _ in inputs]})

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)

    providers = ProviderSettings(
        remote_api=RemoteAPIProviderSettings(
            base_url="https://models.example.com",
            embedding_endpoint="/v1/embeddings",
        )
    )
    embedding = create_embedding_engine(
        EmbeddingSettings(provider="remote_api", model="bge-m3"),
        providers=providers,
    )

    texts = [f"document-{index}" for index in range(33)]
    vectors = embedding.embed_batch(texts)

    assert [len(inputs) for inputs in captured_inputs] == [32, 1]
    assert len(vectors) == len(texts)
    assert embedding.is_available() is True


def test_remote_embedding_reports_partial_batch_fallback(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/tokenize"):
            return FakeHTTPResponse({"tokens": list(range(len(payload["content"])))})
        calls += 1
        if calls == 2:
            raise URLError("temporary provider failure")
        inputs = payload["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        return FakeHTTPResponse({"data": [{"embedding": _vector(1024, value=0.25)} for _ in inputs]})

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)

    providers = ProviderSettings(
        remote_api=RemoteAPIProviderSettings(
            base_url="https://models.example.com",
            embedding_endpoint="/v1/embeddings",
        )
    )
    embedding = create_embedding_engine(
        EmbeddingSettings(provider="remote_api", model="bge-m3"),
        providers=providers,
    )
    with pytest.raises(engine_module.ProviderError):
        embedding.embed_batch([f"document-{index}" for index in range(65)])

    assert calls == 3
    assert embedding.is_available() is False


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
    with pytest.raises(engine_module.ProviderError):
        embedding.embed("ignored")
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



def test_remote_embedding_batch_fallback_never_produces_hash_vectors(monkeypatch) -> None:
    """A degraded remote batch must not leak deterministic hash vectors into the
    caller's results — they would be persisted into the real vector index."""

    def fake_urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/tokenize"):
            return FakeHTTPResponse({"tokens": list(range(len(payload["content"])))})
        inputs = payload["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        return FakeHTTPResponse({"data": [{"embedding": _vector(1024, value=0.25)} for _ in inputs]})

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)

    providers = ProviderSettings(
        remote_api=RemoteAPIProviderSettings(
            base_url="https://models.example.com",
            embedding_endpoint="/v1/embeddings",
        )
    )
    embedding = create_embedding_engine(
        EmbeddingSettings(provider="remote_api", model="bge-m3"),
        providers=providers,
    )
    good = embedding.embed("fine document")
    assert embedding.is_available() is True

    # Now the provider starts failing: embed must raise (not return hash vectors).
    def failing_urlopen(request, timeout):
        del request, timeout
        raise URLError("connection refused")

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", failing_urlopen)
    try:
        embedding.embed("broken document")
        raised = False
    except (URLError, OSError, RuntimeError):
        raised = True
    assert raised, "degraded remote embedding must not return fallback vectors"


def test_remote_embedding_token_budget_splits_long_documents(monkeypatch) -> None:
    """Batching must respect a token budget, not only doc count.

    The fake tokenizer counts one token per character so token budgets are
    deterministic in the test without a real tokenizer endpoint.
    """

    captured_inputs: list[list[str]] = []
    tokenizer_calls = 0

    def fake_urlopen(request, timeout):
        nonlocal tokenizer_calls
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/tokenize"):
            tokenizer_calls += 1
            return FakeHTTPResponse({"tokens": list(range(len(payload["content"])))})
        inputs = payload["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        captured_inputs.append(inputs)
        return FakeHTTPResponse({"data": [{"embedding": _vector(1024, value=0.25)} for _ in inputs]})

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)

    providers = ProviderSettings(
        remote_api=RemoteAPIProviderSettings(
            base_url="https://models.example.com",
            embedding_endpoint="/v1/embeddings",
            tokenize_endpoint="/tokenize",
        )
    )
    embedding = create_embedding_engine(
        EmbeddingSettings(provider="remote_api", model="bge-m3"),
        providers=providers,
    )
    # 4 docs of 9000 fake tokens each: the per-request cap (8192 incl. 2
    # specials) forces one document per request.
    texts = ["x" * 9000 for _ in range(4)]
    vectors = embedding.embed_batch(texts)
    assert len(vectors) == 4
    max_request_tokens = 8192
    for inputs in captured_inputs:
        assert sum(len(t) + 2 for t in inputs) <= max_request_tokens
        assert len(inputs) <= 32
        # Oversize documents are truncated to prefixes of the original.
        for sent in inputs:
            assert sent.startswith("x")
            assert len(sent) <= 8190

    # And the single oversize doc case: truncated, sent alone, still one vector.
    captured_inputs.clear()
    vector = embedding.embed("y" * 30000)
    assert len(vector) == 1024
    assert len(captured_inputs) == 1
    assert captured_inputs[0][0].startswith("y")
    assert len(captured_inputs[0][0]) <= 8190
    assert tokenizer_calls > 0


def test_remote_reranker_truncates_documents_to_max_doc_tokens(monkeypatch) -> None:
    """Documents longer than reranker.max_doc_tokens must be truncated in the
    request payload — rerank latency scales with total candidate tokens."""

    captured_payloads: list[dict[str, object]] = []
    tokenize_calls = 0

    def fake_urlopen(request, timeout):
        nonlocal tokenize_calls
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/tokenize"):
            tokenize_calls += 1
            return FakeHTTPResponse({"tokens": list(range(len(payload["content"])))})
        captured_payloads.append(payload)
        return FakeHTTPResponse(
            {"results": [{"index": i, "relevance_score": 0.5} for i in range(len(payload["documents"]))]}
        )

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)

    providers = ProviderSettings(
        remote_api=RemoteAPIProviderSettings(
            base_url="https://models.example.com",
            rerank_endpoint="/v1/rerank",
            tokenize_endpoint="/tokenize",
        )
    )
    reranker = create_reranker_engine(
        RerankerSettings(provider="remote_api", model="bge-reranker-v2-m3", max_doc_tokens=512),
        providers=providers,
    )

    long_doc = "x" * 5000  # 5000 fake tokens ≫ the 512-token document budget
    short_doc = "tiny document"
    ranked = reranker.rerank("query", [long_doc, short_doc])

    assert [index for index, _ in ranked] == [0, 1]
    sent = captured_payloads[0]["documents"]
    assert sent[0] == "x" * 512  # truncated to a prefix of exactly the budget
    assert sent[1] == short_doc  # short docs pass through untouched
    assert len(captured_payloads[0]["query"]) <= 8192 - 512 - 8
    assert tokenize_calls > 0


def test_under_budget_text_skips_tokenize_calls(monkeypatch) -> None:
    """len(text) <= budget needs no tokenize call: XLM-R yields <= 1 token/char."""

    def failing_urlopen(request, timeout):
        del request, timeout
        raise URLError("tokenize must not be called for short text")

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", failing_urlopen)

    tokenizer = engine_module.RemoteTokenizer(
        client=engine_module.HTTPJSONClient(base_url="https://models.example.com"),
        endpoint="/tokenize",
        timeout_seconds=5,
    )
    short_text = "短文本"  # 3 chars, 3-token budget
    truncated, count = tokenizer.truncate(short_text, 3)
    assert truncated == short_text
    assert count == 3


def test_over_budget_text_truncates_to_verified_prefix(monkeypatch) -> None:
    """Over-budget text is cut to a prefix of the original whose verified
    token count fits the budget, landing close to it."""

    def fake_urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        # Fake tokenizer: each character is one token.
        return FakeHTTPResponse({"tokens": list(range(len(payload["content"])))})

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)

    tokenizer = engine_module.RemoteTokenizer(
        client=engine_module.HTTPJSONClient(base_url="https://models.example.com"),
        endpoint="/tokenize",
        timeout_seconds=5,
    )
    text = "abcdefghij" * 100  # 1000 chars/tokens, budget 700
    truncated, count = tokenizer.truncate(text, 700)

    assert truncated.startswith("abcdefghij")  # prefix of the original
    assert text.startswith(truncated)
    assert count <= 700
    assert count >= 650  # close to the budget, not far below


def test_cjk_heavy_text_truncation_keeps_budget(monkeypatch) -> None:
    """CJK-heavy text truncation stays within budget with exact counting."""

    def fake_urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        # XLM-R-like: ~2 chars per token for CJK.
        return FakeHTTPResponse({"tokens": list(range((len(payload["content"]) + 1) // 2))})

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)

    tokenizer = engine_module.RemoteTokenizer(
        client=engine_module.HTTPJSONClient(base_url="https://models.example.com"),
        endpoint="/tokenize",
        timeout_seconds=5,
    )
    text = "混合语言检索测试" * 300  # 2400 CJK chars ≈ 1200 tokens, budget 600
    truncated, count = tokenizer.truncate(text, 600)

    assert text.startswith(truncated)
    assert truncated.startswith("混合语言")
    assert count <= 600
    assert count >= 550


def test_tokenize_failure_falls_back_to_estimate(monkeypatch, caplog) -> None:
    """Tokenize failures use the conservative estimate and warn once, and the
    estimated truncation still respects the budget."""

    call_count = 0

    def failing_urlopen(request, timeout):
        nonlocal call_count
        del request, timeout
        call_count += 1
        raise URLError("tokenize endpoint down")

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", failing_urlopen)

    # Other tests may configure Synapse logging (propagate=False on the
    # "synapse" parent); re-enable propagation so caplog can capture records.
    parent_logger = logging.getLogger("synapse")
    monkeypatch.setattr(parent_logger, "propagate", True)

    tokenizer = engine_module.RemoteTokenizer(
        client=engine_module.HTTPJSONClient(base_url="https://models.example.com"),
        endpoint="/tokenize",
        timeout_seconds=5,
    )
    with caplog.at_level("WARNING", logger=engine_module.LOGGER.name):
        # 2000 CJK chars: estimate 2000 tokens, budget 800 -> cut to ~800 chars.
        truncated, count = tokenizer.truncate("检" * 2000, 800)
        truncated2, count2 = tokenizer.truncate("检" * 3000, 800)

    assert len(truncated) <= 800
    assert text_is_prefix("检" * 2000, truncated)
    assert count <= 800
    assert len(truncated2) <= 800
    assert count2 <= 800
    warnings = [
        r for r in caplog.records if r.levelname == "WARNING" and "tokenization" in r.getMessage()
    ]
    assert len(warnings) == 1  # warn once per failure, not per document
    assert call_count == 2  # first truncation path only (both hit remote first)


def text_is_prefix(full: str, part: str) -> bool:
    return full.startswith(part)


def test_tokenize_disabled_uses_estimate_only(monkeypatch) -> None:
    """Empty endpoint disables remote tokenization: estimates only, no HTTP."""

    def failing_urlopen(request, timeout):
        del request, timeout
        raise URLError("no HTTP expected with tokenization disabled")

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", failing_urlopen)

    tokenizer = engine_module.RemoteTokenizer(
        client=engine_module.HTTPJSONClient(base_url="https://models.example.com"),
        endpoint="",
        timeout_seconds=5,
    )
    assert tokenizer.available() is False

    # CJK estimate = 1 token/char: 1000-char CJK text, budget 400 -> 400 chars.
    truncated, count = tokenizer.truncate("检" * 1000, 400)
    assert len(truncated) == 400
    assert count == 400

    # Mixed text: 1/CJK + ceil(other/2).
    mixed = "检" * 100 + "a" * 100
    count = tokenizer.count(mixed)
    assert count == 100 + 50


def test_embed_single_oversize_doc_sent_alone(monkeypatch) -> None:
    """A request whose docs would exceed the token cap splits; an oversize doc
    always gets its own request and still returns one vector."""

    captured_inputs: list[list[str]] = []

    def fake_urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/tokenize"):
            # Each char = 1 token.
            return FakeHTTPResponse({"tokens": list(range(len(payload["content"])))})
        inputs = payload["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        captured_inputs.append(inputs)
        return FakeHTTPResponse({"data": [{"embedding": _vector(1024, value=0.25)} for _ in inputs]})

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)

    providers = ProviderSettings(
        remote_api=RemoteAPIProviderSettings(
            base_url="https://models.example.com",
            embedding_endpoint="/v1/embeddings",
            tokenize_endpoint="/tokenize",
        )
    )
    embedding = create_embedding_engine(
        EmbeddingSettings(provider="remote_api", model="bge-m3"),
        providers=providers,
    )
    big = "x" * 6000  # truncated to 8190? no: 6000 tokens + 2 = 6002 <= 8192, fits alone
    small = "y" * 100
    medium = "z" * 2000
    # big(6002) + medium(2002) + small(102) = 8106 <= 8192: one request.
    vectors = embedding.embed_batch([big, medium, small])
    assert len(vectors) == 3
    assert len(captured_inputs) == 1
    assert len(captured_inputs[0]) == 3

    # big2 (7000 tokens + 2) + medium (2002) = 9004 > 8192: split, big2 alone.
    captured_inputs.clear()
    big2 = "w" * 7000
    vectors = embedding.embed_batch([big2, medium, small])
    assert len(vectors) == 3
    assert [len(batch) for batch in captured_inputs] == [1, 2]


def _tokenizer_with_counting_tokenize(monkeypatch, calls: list[int]):
    """Fake tokenizer endpoint where each char = one token; records call count."""

    def fake_urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/tokenize"):
            calls.append(1)
            return FakeHTTPResponse({"tokens": list(range(len(payload["content"])))})
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)
    return engine_module.RemoteTokenizer(
        client=engine_module.HTTPJSONClient(base_url="https://models.example.com"),
        endpoint="/tokenize",
        timeout_seconds=5,
    )


def test_truncate_cache_avoids_repeat_tokenize_calls(monkeypatch) -> None:
    """A repeated truncate of the same over-budget text makes zero extra
    tokenize calls and returns the identical prefix."""

    calls: list[int] = []
    tokenizer = _tokenizer_with_counting_tokenize(monkeypatch, calls)
    text = "x" * 5000

    first_text, first_count = tokenizer.truncate(text, 1000)
    calls_after_first = len(calls)
    assert calls_after_first > 0

    second_text, second_count = tokenizer.truncate(text, 1000)
    assert len(calls) == calls_after_first  # zero extra tokenize calls
    assert (second_text, second_count) == (first_text, first_count)
    assert text.startswith(second_text)

    # A different budget is a different cache key: hits the endpoint again.
    tokenizer.truncate(text, 500)
    assert len(calls) > calls_after_first


def test_truncate_cache_eviction_bound_holds(monkeypatch) -> None:
    """The LRU never grows past the configured entry cap."""

    calls: list[int] = []
    tokenizer = _tokenizer_with_counting_tokenize(monkeypatch, calls)
    for index in range(engine_module._TRUNCATION_CACHE_MAX_ENTRIES + 100):
        tokenizer.truncate("y" * (3000 + index), 1000)

    assert len(tokenizer._cache) == engine_module._TRUNCATION_CACHE_MAX_ENTRIES


def test_truncate_cache_skips_estimate_fallback_results(monkeypatch) -> None:
    """Results produced by the estimate fallback (tokenize failure) are never
    cached; after recovery the next call tokenizes exactly again."""

    mode = {"fail": True}

    def fake_urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/tokenize"):
            if mode["fail"]:
                raise URLError("tokenize endpoint down")
            # XLM-R-like: ~1 token per 2 CJK chars.
            return FakeHTTPResponse({"tokens": list(range((len(payload["content"]) + 1) // 2))})
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", fake_urlopen)
    tokenizer = engine_module.RemoteTokenizer(
        client=engine_module.HTTPJSONClient(base_url="https://models.example.com"),
        endpoint="/tokenize",
        timeout_seconds=5,
    )

    text = "检" * 2000
    failed_text, failed_count = tokenizer.truncate(text, 800)
    assert len(failed_text) <= 800
    assert tokenizer._cache == {}  # fallback result not cached

    mode["fail"] = False
    exact_text, exact_count = tokenizer.truncate(text, 800)
    assert text.startswith(exact_text)
    assert exact_count <= 800
    assert len(tokenizer._cache) == 1  # exact result cached
    # The exact path lands closer to the budget than the 1-token-per-CJK-char
    # estimate (XLM-R emits ~1 token per 2 CJK chars), so the warm cached
    # prefix is longer than the failed-estimate prefix.
    assert len(exact_text) > len(failed_text)
    # And the warm call now returns the exact (longer) prefix from the cache.
    warm_text, warm_count = tokenizer.truncate(text, 800)
    assert (warm_text, warm_count) == (exact_text, exact_count)
