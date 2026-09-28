"""Provider-aware embedding and reranker engines with deterministic fallback behavior."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from synapse.config import (
    RERANKER_MODEL_CONTEXT_TOKENS,
    RERANKER_PAIR_SPECIAL_TOKEN_MARGIN,
    EmbeddingSettings,
    ProviderSettings,
    RerankerSettings,
)
from synapse.interfaces import EmbeddingEngine, RerankerEngine

JSON_MIME_TYPE = "application/json"
API_KEY_PLACEHOLDER = "{api_key}"


LOGGER = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"\w+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_REMOTE_EMBEDDING_REQUEST_BATCH_SIZE = 32
# Special tokens the embedding server adds per input document (BOS/EOS); the
# per-request token cap covers document tokens plus this overhead.
_EMBEDDING_SPECIAL_TOKEN_COUNT = 2
# Rerank pair specials come from RERANKER_PAIR_SPECIAL_TOKEN_MARGIN (synapse.config),
# the single definition shared with config validation.
# Maximum proportional re-tokenize passes before falling back to halving.
_TRUNCATION_MAX_PASSES = 8
# Bounded LRU size for RemoteTokenizer's exact-truncation cache. Candidate
# documents repeat heavily across searches, so caching (budget, digest) ->
# (prefix length, token count) removes repeated tokenize round-trips on the
# search hot path. Lengths are stored, never text.
_TRUNCATION_CACHE_MAX_ENTRIES = 4096

# Character ranges counted as one token each by the conservative estimator
# (CJK ideographs, kana, hangul, fullwidth/CJK punctuation). Verified against
# the XLM-R tokenizer on the live corpus: the estimate never under-counts.
_CJK_TOKEN_CHAR_RANGES: tuple[tuple[int, int], ...] = (
    (0x2E80, 0x2EFF),
    (0x3000, 0x303F),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0x3130, 0x318F),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
    (0x20000, 0x2A6DF),
)


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate: 1 per CJK char + ceil(other chars / 2).

    Measured on the active corpus against the bge tokenizer (XLM-R): this
    never under-counts real tokens, so requests budgeted with it stay safe.
    """

    cjk_chars = sum(
        1 for char in text if any(lo <= ord(char) <= hi for lo, hi in _CJK_TOKEN_CHAR_RANGES)
    )
    return cjk_chars + math.ceil((len(text) - cjk_chars) / 2)


def _truncate_by_estimate(text: str, budget: int) -> str:
    """Cut text to a prefix whose conservative token estimate fits the budget."""

    count = _estimate_tokens(text)
    if count <= budget:
        return text
    candidate = text[: max(1, (len(text) * budget) // count)]
    while _estimate_tokens(candidate) > budget:
        candidate_count = _estimate_tokens(candidate)
        candidate = candidate[: max(1, (len(candidate) * budget) // candidate_count)]
    return candidate


def _split_by_token_budget(
    token_counts: Sequence[int], *, max_batch_docs: int, max_request_tokens: int
) -> Iterator[Sequence[int]]:
    """Group documents into request batches under a doc-count and token budget.

    ``token_counts`` holds per-document token counts (upper bounds allowed).
    Each document costs ``count + _EMBEDDING_SPECIAL_TOKEN_COUNT`` against the
    request budget. A single document always yields its own batch when needed —
    one vector per input document stays the contract. Yields index sequences.
    """

    batch: list[int] = []
    batch_tokens = 0
    for index, count in enumerate(token_counts):
        cost = count + _EMBEDDING_SPECIAL_TOKEN_COUNT
        if batch and (len(batch) >= max_batch_docs or batch_tokens + cost > max_request_tokens):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(index)
        batch_tokens += cost
    if batch:
        yield batch


@dataclass(slots=True)
class RemoteTokenizer:
    """Counts and truncates text against a llama.cpp-style ``POST /tokenize`` endpoint.

    Exact counts come from the provider; when the endpoint is disabled or the
    call fails, a conservative character-based estimate is used instead
    (verified on the live corpus to never under-count). Counting never raises.

    Exact truncation results are memoized in a bounded thread-safe LRU keyed
    by ``(budget, text digest)``; candidate documents repeat heavily across
    searches, so a warm cache removes the tokenize round-trips from the search
    hot path. Only prefix lengths and token counts are stored, never text, and
    estimate-fallback results are never cached.
    """

    client: HTTPJSONClient
    endpoint: str
    timeout_seconds: int
    _warned_failures: set[str] = field(default_factory=set)
    _cache: OrderedDict[tuple[int, str], tuple[int, int]] = field(
        default_factory=OrderedDict, repr=False, compare=False
    )
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def available(self) -> bool:
        return bool(self.endpoint)

    def count(self, text: str) -> int:
        """Token count for ``text``; exact via the provider when possible."""

        if not text:
            return 0
        if self.endpoint:
            try:
                return self._count_remote(text)
            except ProviderError as exc:
                self._warn_once(exc)
        return _estimate_tokens(text)

    def truncate(self, text: str, budget: int) -> tuple[str, int]:
        """Return ``(prefix_of_text, token_count)`` with ``token_count <= budget``.

        The result is always a prefix of the original text (never detokenized
        output) and lands close to the budget. Under-budget texts are returned
        without a tokenize call: XLM-R sentencepiece emits at most one token per
        character, so ``len(text) <= budget`` guarantees the count fits.
        """

        if len(text) <= budget:
            return text, len(text)
        if not self.endpoint:
            truncated = _truncate_by_estimate(text, budget)
            return truncated, _estimate_tokens(truncated)

        cache_key = (budget, hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest())
        cached = self._cache_get(cache_key)
        if cached is not None:
            prefix_length, token_count = cached
            return text[:prefix_length], token_count

        try:
            count = self._count_remote(text)
        except ProviderError as exc:
            self._warn_once(exc)
            truncated = _truncate_by_estimate(text, budget)
            return truncated, _estimate_tokens(truncated)
        if count <= budget:
            self._cache_put(cache_key, (len(text), count))
            return text, count

        candidate = text[: max(1, (len(text) * budget) // count)]
        for _ in range(_TRUNCATION_MAX_PASSES):
            try:
                candidate_count = self._count_remote(candidate)
            except ProviderError as exc:
                self._warn_once(exc)
                truncated = _truncate_by_estimate(candidate, budget)
                return truncated, _estimate_tokens(truncated)
            if candidate_count <= budget:
                self._cache_put(cache_key, (len(candidate), candidate_count))
                return candidate, candidate_count
            candidate = candidate[: max(1, (len(candidate) * budget) // candidate_count)]
        # Proportional shrinking did not converge within the pass cap; halve
        # until a tokenize call verifies the prefix fits (terminates: fewer
        # characters can only produce fewer or equal tokens).
        while True:
            candidate = candidate[: max(1, len(candidate) // 2)]
            try:
                candidate_count = self._count_remote(candidate)
            except ProviderError as exc:
                self._warn_once(exc)
                truncated = _truncate_by_estimate(candidate, budget)
                return truncated, _estimate_tokens(truncated)
            if candidate_count <= budget:
                self._cache_put(cache_key, (len(candidate), candidate_count))
                return candidate, candidate_count

    def _cache_get(self, cache_key: tuple[int, str]) -> tuple[int, int] | None:
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
            return cached

    def _cache_put(self, cache_key: tuple[int, str], value: tuple[int, int]) -> None:
        with self._cache_lock:
            self._cache[cache_key] = value
            self._cache.move_to_end(cache_key)
            while len(self._cache) > _TRUNCATION_CACHE_MAX_ENTRIES:
                self._cache.popitem(last=False)

    def _count_remote(self, text: str) -> int:
        response = self.client.post_json(
            self.endpoint,
            {"content": text},
            timeout_seconds=self.timeout_seconds,
        )
        if isinstance(response, Mapping):
            tokens = response.get("tokens")
            if _is_numeric_sequence(tokens):
                return len(tokens)
        raise ProviderError(f"Tokenizer endpoint {self.endpoint} returned a malformed response")

    def _warn_once(self, exc: ProviderError) -> None:
        signature = str(exc)
        if signature not in self._warned_failures:
            self._warned_failures.add(signature)
            LOGGER.warning(
                "Remote tokenization failed (%s); using conservative token estimates", exc
            )


@dataclass(slots=True, frozen=True)
class EmbeddingModelSpec:
    """Static metadata for a supported embedding model name."""

    name: str
    dimension: int
    max_tokens: int
    family: str


@dataclass(slots=True, frozen=True)
class RerankerModelSpec:
    """Static metadata for a supported reranker model name."""

    name: str
    family: str
    max_tokens: int


EMBEDDING_MODEL_SPECS: dict[str, EmbeddingModelSpec] = {
    "bge-m3": EmbeddingModelSpec(name="bge-m3", dimension=1024, max_tokens=8192, family="bge"),
    "jina-v3": EmbeddingModelSpec(name="jina-v3", dimension=1024, max_tokens=8192, family="jina"),
    "gte-qwen2": EmbeddingModelSpec(name="gte-qwen2", dimension=1536, max_tokens=32768, family="gte"),
    "gemma-300m": EmbeddingModelSpec(name="gemma-300m", dimension=1024, max_tokens=8192, family="gemma"),
}

RERANKER_MODEL_SPECS: dict[str, RerankerModelSpec] = {
    name: RerankerModelSpec(name=name, family=family, max_tokens=RERANKER_MODEL_CONTEXT_TOKENS[name])
    for name, family in (
        ("bge-reranker-v2-m3", "bge"),
        ("qllama/bge-reranker-v2-m3", "bge"),
        ("jina-reranker-v2", "jina"),
    )
}


class ProviderError(RuntimeError):
    """Raised when a configured inference provider cannot satisfy a request."""


@dataclass(slots=True)
class DeterministicEmbeddingEngine:
    """Local deterministic embedding backend used as a lightweight fallback."""

    model_name: str
    dimension: int
    backend_name: str = "deterministic"
    degraded_from: str | None = None

    def is_available(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        if not tokens:
            tokens = ["<empty>"]
        vector = [0.0] * self.dimension

        for token in tokens:
            self._accumulate_token(vector, token)

        self._accumulate_document_signature(vector, text)
        return _normalize_vector(vector)

    def _accumulate_token(self, vector: list[float], token: str) -> None:
        digest = hashlib.blake2b(f"{self.model_name}:{token}".encode("utf-8"), digest_size=24).digest()
        weight = 1.0 + (digest[0] / 255.0)
        for offset in (1, 7, 13):
            index = int.from_bytes(digest[offset : offset + 4], "big") % self.dimension
            sign = -1.0 if digest[offset + 4] % 2 else 1.0
            vector[index] += sign * weight

    def _accumulate_document_signature(self, vector: list[float], text: str) -> None:
        digest = hashlib.blake2b(f"{self.model_name}|doc|{text}".encode("utf-8"), digest_size=32).digest()
        for index, value in enumerate(digest):
            vector[index % self.dimension] += ((value / 255.0) * 2.0) - 1.0


@dataclass(slots=True)
class UnavailableEmbeddingEngine:
    """Sentinel backend returned when fallback is disabled."""

    model_name: str
    dimension: int
    reason: str = "Model backend unavailable"
    backend_name: str = "unavailable"

    def is_available(self) -> bool:
        return False

    def embed(self, text: str) -> list[float]:
        del text
        return []

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        del texts
        return []


@dataclass(slots=True)
class DeterministicRerankerEngine:
    """Deterministic lexical reranker suitable for repeatable local tests."""

    model_name: str
    backend_name: str = "deterministic"
    degraded_from: str | None = None

    def is_available(self) -> bool:
        return True

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        limit: int | None = None,
    ) -> list[tuple[int, float]]:
        scored = [(index, self._score(query, document)) for index, document in enumerate(documents)]
        scored.sort(key=lambda item: (-item[1], item[0]))
        if limit is not None:
            return scored[:limit]
        return scored

    def _score(self, query: str, document: str) -> float:
        query_tokens = _tokenize(query)
        document_tokens = _tokenize(document)
        if not query_tokens:
            return 0.0

        query_set = set(query_tokens)
        document_set = set(document_tokens)
        overlap = len(query_set & document_set)
        coverage = overlap / max(1, len(query_set))
        density = overlap / max(1, len(document_set))
        phrase_bonus = 1.0 if query.strip() and query.casefold() in document.casefold() else 0.0
        ordered_bonus = _ordered_token_bonus(query_tokens, document_tokens)
        model_bias = _model_bias(self.model_name, query, document)
        return round((coverage * 3.0) + density + phrase_bonus + ordered_bonus + model_bias, 6)


@dataclass(slots=True)
class UnavailableRerankerEngine:
    """Sentinel backend returned when fallback is disabled."""

    model_name: str
    reason: str = "Model backend unavailable"
    backend_name: str = "unavailable"

    def is_available(self) -> bool:
        return False

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        limit: int | None = None,
    ) -> list[tuple[int, float]]:
        del query
        if not documents:
            return []
        fallback = [(index, 0.0) for index, _ in enumerate(documents)]
        if limit is not None:
            return fallback[:limit]
        return fallback


@dataclass(slots=True)
class HTTPJSONClient:
    """Tiny JSON HTTP client built on the standard library for easy portability."""

    base_url: str
    default_headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30

    def get_json(self, endpoint: str, *, timeout_seconds: int | None = None) -> Any:
        request = urllib_request.Request(
            _join_url(self.base_url, endpoint),
            headers={"Accept": JSON_MIME_TYPE, **dict(self.default_headers)},
            method="GET",
        )
        return _open_json_request(request, timeout_seconds or self.timeout_seconds)

    def post_json(self, endpoint: str, payload: Mapping[str, Any], *, timeout_seconds: int | None = None) -> Any:
        request = urllib_request.Request(
            _join_url(self.base_url, endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": JSON_MIME_TYPE,
                "Content-Type": JSON_MIME_TYPE,
                **dict(self.default_headers),
            },
            method="POST",
        )
        return _open_json_request(request, timeout_seconds or self.timeout_seconds)


@dataclass(slots=True)
class RemoteAPIEmbeddingEngine:
    """Generic HTTP embedding client for remote inference APIs."""

    model_name: str
    dimension: int
    client: HTTPJSONClient
    endpoint: str
    timeout_seconds: int
    fallback_engine: EmbeddingEngine
    backend_name: str = "remote_api"
    last_known_available: bool | None = None
    max_batch_tokens: int = 8192
    tokenizer: RemoteTokenizer | None = None

    def is_available(self) -> bool:
        return bool(self.last_known_available)

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        # Real-token budgeting against the model context: per document at most
        # max_batch_tokens - specials (a prefix of the original text is sent),
        # and per request at most 32 docs with total tokens + specials within
        # the context. One vector per input document is the contract. Every
        # document is tokenized: skipping short docs and using len(text) as an
        # upper bound inflates real sync batches ~2.4x (measured on the live
        # corpus), so the cheap len(text) <= budget fast path is only used by
        # reranker truncation where per-call latency matters more.
        vectors: list[list[float]] = []
        all_batches_succeeded = True
        per_doc_budget = self.max_batch_tokens - _EMBEDDING_SPECIAL_TOKEN_COUNT
        tokenizer = self.tokenizer or RemoteTokenizer(
            client=self.client, endpoint="", timeout_seconds=self.timeout_seconds
        )
        bounded_texts: list[str] = []
        token_counts: list[int] = []
        for text in texts:
            if len(text) > per_doc_budget:
                truncated, count = tokenizer.truncate(text, per_doc_budget)
            else:
                truncated = text
                count = tokenizer.count(text)
            bounded_texts.append(truncated)
            token_counts.append(count)
        failure: ProviderError | None = None
        for indexes in _split_by_token_budget(
            token_counts,
            max_batch_docs=_REMOTE_EMBEDDING_REQUEST_BATCH_SIZE,
            max_request_tokens=self.max_batch_tokens,
        ):
            batch = [bounded_texts[index] for index in indexes]
            try:
                batch_vectors = _request_embedding_vectors(
                    client=self.client,
                    endpoint=self.endpoint,
                    model_name=self.model_name,
                    texts=batch,
                    timeout_seconds=self.timeout_seconds,
                    expected_dimension=self.dimension,
                )
            except ProviderError as exc:
                # Never substitute deterministic hash vectors: callers must
                # store/retrieve nothing rather than poison the vector space.
                all_batches_succeeded = False
                LOGGER.warning("Remote embedding provider unavailable: %s", exc)
                if failure is None:
                    failure = exc
            else:
                vectors.extend(batch_vectors)
        self.last_known_available = all_batches_succeeded
        if failure is not None and not all_batches_succeeded:
            raise failure
        return vectors




@dataclass(slots=True)
class RemoteAPIRerankerEngine:
    """Generic HTTP reranker client for remote inference APIs."""

    model_name: str
    client: HTTPJSONClient
    endpoint: str
    timeout_seconds: int
    fallback_engine: RerankerEngine
    backend_name: str = "remote_api"
    last_known_available: bool | None = None
    max_batch_tokens: int = 8192
    # Per-document truncation in REAL tokens (measured via the provider's
    # tokenize endpoint): rerank latency scales with total candidate tokens,
    # and the first few hundred tokens carry the ranking signal.
    max_doc_tokens: int = 2048
    tokenizer: RemoteTokenizer | None = None

    def is_available(self) -> bool:
        return bool(self.last_known_available)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        limit: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        tokenizer = self.tokenizer or RemoteTokenizer(
            client=self.client, endpoint="", timeout_seconds=self.timeout_seconds
        )
        query_budget = self.max_batch_tokens - self.max_doc_tokens - RERANKER_PAIR_SPECIAL_TOKEN_MARGIN
        try:
            bounded_query, _ = tokenizer.truncate(query, query_budget)
            bounded_documents: list[str] = []
            for document in documents:
                bounded_document, _ = tokenizer.truncate(document, self.max_doc_tokens)
                bounded_documents.append(bounded_document)
            ranked = _request_rerank_scores(
                client=self.client,
                endpoint=self.endpoint,
                model_name=self.model_name,
                query=bounded_query,
                documents=bounded_documents,
                timeout_seconds=self.timeout_seconds,
            )
        except ProviderError as exc:
            self.last_known_available = False
            LOGGER.warning("Remote reranker provider unavailable: %s", exc)
            return self.fallback_engine.rerank(query, documents, limit=limit)
        self.last_known_available = True
        if limit is not None:
            return ranked[:limit]
        return ranked


EmbeddingFactory = Callable[[EmbeddingSettings, EmbeddingModelSpec], EmbeddingEngine | None]
RerankerFactory = Callable[[RerankerSettings, RerankerModelSpec], RerankerEngine | None]

@dataclass(slots=True)
class EngineRegistry:
    """Registry of optional runtime factories used before provider defaults."""

    embedding_factories: list[EmbeddingFactory] = field(default_factory=list)
    reranker_factories: list[RerankerFactory] = field(default_factory=list)

    def register_embedding(self, factory: EmbeddingFactory) -> None:
        self.embedding_factories.append(factory)

    def register_reranker(self, factory: RerankerFactory) -> None:
        self.reranker_factories.append(factory)


DEFAULT_REGISTRY = EngineRegistry()


def create_embedding_engine(
    settings: EmbeddingSettings,
    *,
    providers: ProviderSettings | None = None,
    registry: EngineRegistry | None = None,
    allow_fallback: bool = True,
) -> EmbeddingEngine:
    """Create an embedding engine for the configured provider and model."""

    spec = EMBEDDING_MODEL_SPECS[settings.model]
    selected_registry = registry or DEFAULT_REGISTRY
    resolved_dimension = settings.dimension or spec.dimension
    selected_providers = providers or ProviderSettings()

    for factory in selected_registry.embedding_factories:
        engine = factory(settings, spec)
        if engine is not None:
            return engine

    if settings.provider == "builtin":
        return _create_fallback_embedding_engine(settings.model, resolved_dimension, allow_fallback)

    fallback_engine = _create_fallback_embedding_engine(
        settings.model,
        resolved_dimension,
        allow_fallback,
        degraded_from=settings.provider,
    )

    remote_api = selected_providers.remote_api
    if not remote_api.embedding_endpoint:
        return fallback_engine
    embed_base_url = remote_api.embedding_base_url or remote_api.base_url
    embed_client = HTTPJSONClient(
        base_url=embed_base_url,
        default_headers=_build_provider_headers(remote_api.headers, remote_api.api_key_env),
        timeout_seconds=remote_api.request_timeout_seconds,
    )
    return RemoteAPIEmbeddingEngine(
        model_name=settings.model,
        dimension=resolved_dimension,
        client=embed_client,
        endpoint=remote_api.embedding_endpoint,
        timeout_seconds=settings.timeout_seconds,
        fallback_engine=fallback_engine,
        max_batch_tokens=spec.max_tokens,
        tokenizer=RemoteTokenizer(
            client=embed_client,
            endpoint=remote_api.tokenize_endpoint,
            timeout_seconds=remote_api.request_timeout_seconds,
        ),
    )


def create_reranker_engine(
    settings: RerankerSettings,
    *,
    providers: ProviderSettings | None = None,
    registry: EngineRegistry | None = None,
    allow_fallback: bool = True,
) -> RerankerEngine:
    """Create a reranker engine for the configured provider and model."""

    spec = RERANKER_MODEL_SPECS[settings.model]
    selected_registry = registry or DEFAULT_REGISTRY
    selected_providers = providers or ProviderSettings()

    for factory in selected_registry.reranker_factories:
        engine = factory(settings, spec)
        if engine is not None:
            return engine

    if settings.provider == "builtin":
        return _create_fallback_reranker_engine(settings.model, allow_fallback)

    fallback_engine = _create_fallback_reranker_engine(
        settings.model,
        allow_fallback,
        degraded_from=settings.provider,
    )

    remote_api = selected_providers.remote_api
    if not remote_api.rerank_endpoint:
        return fallback_engine
    rerank_client = HTTPJSONClient(
        base_url=remote_api.base_url,
        default_headers=_build_provider_headers(remote_api.headers, remote_api.api_key_env),
        timeout_seconds=remote_api.request_timeout_seconds,
    )
    return RemoteAPIRerankerEngine(
        model_name=settings.model,
        client=rerank_client,
        endpoint=remote_api.rerank_endpoint,
        timeout_seconds=settings.timeout_seconds,
        max_batch_tokens=spec.max_tokens,
        max_doc_tokens=settings.max_doc_tokens,
        fallback_engine=fallback_engine,
        tokenizer=RemoteTokenizer(
            client=rerank_client,
            endpoint=remote_api.tokenize_endpoint,
            timeout_seconds=remote_api.request_timeout_seconds,
        ),
    )


def _create_fallback_embedding_engine(
    model_name: str,
    dimension: int,
    allow_fallback: bool,
    degraded_from: str | None = None,
) -> EmbeddingEngine:
    if allow_fallback:
        return DeterministicEmbeddingEngine(
            model_name=model_name,
            dimension=dimension,
            degraded_from=degraded_from,
        )
    return UnavailableEmbeddingEngine(model_name=model_name, dimension=dimension)


def _create_fallback_reranker_engine(
    model_name: str,
    allow_fallback: bool,
    degraded_from: str | None = None,
) -> RerankerEngine:
    if allow_fallback:
        return DeterministicRerankerEngine(model_name=model_name, degraded_from=degraded_from)
    return UnavailableRerankerEngine(model_name=model_name)


def _request_embedding_vectors(
    *,
    client: HTTPJSONClient,
    endpoint: str,
    model_name: str,
    texts: Sequence[str],
    timeout_seconds: int,
    expected_dimension: int,
) -> list[list[float]]:
    payload = {"model": model_name, "input": list(texts) if len(texts) != 1 else texts[0]}
    response = client.post_json(endpoint, payload, timeout_seconds=timeout_seconds)
    vectors = _extract_embeddings(response)
    if len(vectors) == len(texts):
        return _validate_embedding_vectors(vectors, expected_dimension)
    if len(vectors) == 1 and len(texts) > 1:
        single_vectors: list[list[float]] = []
        for text in texts:
            single_response = client.post_json(
                endpoint,
                {"model": model_name, "input": text},
                timeout_seconds=timeout_seconds,
            )
            extracted = _extract_embeddings(single_response)
            if len(extracted) != 1:
                raise ProviderError("Embedding provider returned an unexpected single-document response shape")
            single_vectors.extend(_validate_embedding_vectors(extracted, expected_dimension))
        return single_vectors
    raise ProviderError(
        f"Embedding provider returned {len(vectors)} vector(s) for {len(texts)} document(s)"
    )


def _request_rerank_scores(
    *,
    client: HTTPJSONClient,
    endpoint: str,
    model_name: str,
    query: str,
    documents: Sequence[str],
    timeout_seconds: int,
) -> list[tuple[int, float]]:
    response = client.post_json(
        endpoint,
        {"model": model_name, "query": query, "documents": list(documents)},
        timeout_seconds=timeout_seconds,
    )
    ranked = _extract_rerank_scores(response, document_count=len(documents))
    if not ranked:
        raise ProviderError("Reranker provider returned no ranked results")
    return ranked


def _extract_embeddings(payload: Any) -> list[list[float]]:
    if isinstance(payload, Mapping):
        direct_embeddings = _extract_mapping_embeddings(payload)
        if direct_embeddings:
            return direct_embeddings

    if _is_numeric_sequence(payload):
        return [_coerce_float_list(payload)]

    nested_embeddings = _extract_nested_embeddings(payload)
    if nested_embeddings:
        return nested_embeddings

    raise ProviderError("Unable to extract embeddings from provider response")


def _extract_mapping_embeddings(payload: Mapping[str, Any]) -> list[list[float]]:
    direct_embedding = payload.get("embedding")
    if _is_numeric_sequence(direct_embedding):
        return [_coerce_float_list(direct_embedding)]

    grouped_embeddings = payload.get("embeddings")
    if _is_numeric_sequence(grouped_embeddings):
        return [_coerce_float_list(grouped_embeddings)]

    nested_embeddings = _extract_nested_embeddings(grouped_embeddings)
    if nested_embeddings:
        return nested_embeddings

    for key in ("data", "results"):
        nested_embeddings = _extract_nested_embeddings(payload.get(key), field_name="embedding")
        if nested_embeddings:
            return nested_embeddings

    return []


def _extract_nested_embeddings(values: Any, *, field_name: str | None = None) -> list[list[float]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []

    nested_embeddings: list[list[float]] = []
    for item in values:
        candidate = item.get(field_name) if field_name and isinstance(item, Mapping) else item
        if _is_numeric_sequence(candidate):
            nested_embeddings.append(_coerce_float_list(candidate))
    return nested_embeddings


def _extract_rerank_scores(payload: Any, *, document_count: int) -> list[tuple[int, float]]:
    parsed = _parse_ranked_items(payload)
    if parsed:
        parsed.sort(key=lambda item: (-item[1], item[0]))
        return parsed

    if isinstance(payload, Mapping):
        scores = payload.get("scores")
        if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
            ranked_scores = [(index, float(score)) for index, score in enumerate(scores[:document_count]) if isinstance(score, (int, float))]
            ranked_scores.sort(key=lambda item: (-item[1], item[0]))
            if ranked_scores:
                return ranked_scores

    raise ProviderError("Unable to extract rerank scores from provider response")


def _parse_ranked_items(payload: Any) -> list[tuple[int, float]]:
    if isinstance(payload, Mapping):
        for key in ("results", "data", "rankings"):
            items = payload.get(key)
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                ranked = _parse_ranked_sequence(items)
                if ranked:
                    return ranked
        return []

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return _parse_ranked_sequence(payload)

    return []


def _parse_ranked_sequence(items: Sequence[Any]) -> list[tuple[int, float]]:
    ranked: list[tuple[int, float]] = []
    for fallback_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        index = item.get("index", item.get("document_index", item.get("input_index", fallback_index)))
        score = item.get("relevance_score", item.get("score", item.get("similarity", item.get("logit"))))
        if isinstance(index, int) and isinstance(score, (int, float)):
            ranked.append((index, float(score)))
    ranked.sort(key=lambda entry: (-entry[1], entry[0]))
    return ranked


def _validate_embedding_vectors(vectors: Sequence[Sequence[float]], expected_dimension: int) -> list[list[float]]:
    validated: list[list[float]] = []
    for vector in vectors:
        materialized = _coerce_float_list(vector)
        if len(materialized) != expected_dimension:
            raise ProviderError(
                f"Embedding provider returned dimension {len(materialized)} but config expects {expected_dimension}"
            )
        validated.append(materialized)
    return validated


def _build_provider_headers(headers: Mapping[str, str], api_key_env: str | None = None) -> dict[str, str]:
    resolved_headers = dict(headers)
    api_key = os.getenv(api_key_env) if api_key_env else None

    if api_key:
        for key, value in list(resolved_headers.items()):
            if API_KEY_PLACEHOLDER in value:
                resolved_headers[key] = value.replace(API_KEY_PLACEHOLDER, api_key)
        if "authorization" not in {key.casefold() for key in resolved_headers}:
            resolved_headers["Authorization"] = f"Bearer {api_key}"
    else:
        for key, value in list(resolved_headers.items()):
            if API_KEY_PLACEHOLDER in value:
                resolved_headers[key] = value.replace(API_KEY_PLACEHOLDER, "")

    return resolved_headers


def _candidate_model_names(model_name: str) -> list[str]:
    base_model = model_name.split(":", 1)[0]
    candidates = [model_name]
    if base_model != model_name:
        candidates.append(base_model)
    else:
        candidates.append(f"{base_model}:latest")

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _open_json_request(request: urllib_request.Request, timeout_seconds: int) -> Any:
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read()
    except urllib_error.HTTPError as exc:
        details = _read_error_body(exc)
        reason = details or exc.reason
        raise ProviderError(f"HTTP {exc.code} calling {request.full_url}: {reason}") from exc
    except urllib_error.URLError as exc:
        raise ProviderError(f"Request to {request.full_url} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderError(f"Request to {request.full_url} timed out") from exc

    if not raw_body:
        return {}

    try:
        return json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Provider response from {request.full_url} was not valid JSON") from exc


def _read_error_body(exc: urllib_error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8").strip()
    except (AttributeError, OSError, ValueError):
        return ""


def _join_url(base_url: str, endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def _is_numeric_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, (int, float)) for item in value
    )


def _coerce_float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


def _normalize_vector(vector: Iterable[float]) -> list[float]:
    materialized = list(vector)
    norm = math.sqrt(sum(value * value for value in materialized))
    if norm == 0:
        return [0.0 for _ in materialized]
    return [round(value / norm, 8) for value in materialized]


def _ordered_token_bonus(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    joined_query = " ".join(query_tokens)
    joined_document = " ".join(document_tokens)
    if joined_query in joined_document:
        return 0.75
    consecutive_hits = 0
    for first, second in zip(query_tokens, query_tokens[1:]):
        if f"{first} {second}" in joined_document:
            consecutive_hits += 1
    return min(0.5, consecutive_hits * 0.1)


def _model_bias(model_name: str, query: str, document: str) -> float:
    digest = hashlib.blake2b(f"{model_name}|{query}|{document}".encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") / 2**32 / 1000.0
