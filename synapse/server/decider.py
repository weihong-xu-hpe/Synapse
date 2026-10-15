"""Local OpenAI-compatible LLM decision client."""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Callable, Iterator

import httpx


from synapse.config import DeciderSettings
from synapse.server.sampling import (
    MemoryWriteSamplingDecision,
    MemoryWriteSamplingRequest,
    _extract_json_payload,
    build_memory_write_sampling_prompt,
    parse_memory_write_sampling_result,
)


LOGGER = logging.getLogger("synapse.local-llm-decider")
_CHAT_COMPLETIONS_PATH = "/chat/completions"


class LocalLLMDecider:
    """Synchronous SamplingClient backed by a bounded, load-balanced LLM pool."""

    name = "local-llm"

    def __init__(self, settings: DeciderSettings, *, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.timeout_seconds)
        endpoints: dict[tuple[str, str, str, str], tuple[str, str, str, dict[str, Any]]] = {}
        for endpoint in settings.endpoints:
            base_url = endpoint.base_url.rstrip("/")
            key = (base_url, endpoint.model, endpoint.api_key_env, json.dumps(endpoint.extra_body, sort_keys=True))
            endpoints.setdefault(key, (base_url, endpoint.model, endpoint.api_key_env, endpoint.extra_body))
        self._endpoints = tuple(endpoints.values())
        self._endpoint_limit = settings.endpoint_max_concurrency
        self._routing = threading.Condition()
        self._active = [0] * len(self._endpoints)
        self._next_endpoint = 0
        self._metrics_lock = threading.Lock()
        self._call_metrics: dict[str, dict[str, Any]] = {}
        self._transcript_metrics: dict[str, dict[str, dict[str, Any]]] = {}
        self._call_context: ContextVar[dict[str, Any] | None] = ContextVar(
            f"synapse_llm_call_context_{id(self)}", default=None
        )

    def sample_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 600,
        model_hints: tuple[str, ...] = (),
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        del system_prompt, model_hints
        return self._complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            parser=lambda payload: _extract_json_payload(self._extract_content(payload)),
        )

    def decide_memory_write(self, request: MemoryWriteSamplingRequest) -> MemoryWriteSamplingDecision:
        return self._complete(
            messages=[{"role": "user", "content": build_memory_write_sampling_prompt(request)}],
            max_tokens=self.settings.max_tokens,
            parser=self._parse_memory_write_response,
        )

    @contextmanager
    def call_context(
        self,
        *,
        transcript_id: str,
        stage: str,
        timeout_seconds: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Attribute requests and optional timeout overrides to a transcript stage."""

        context: dict[str, Any] = {
            "transcript_id": transcript_id,
            "stage": stage,
            "timeout_seconds": timeout_seconds,
            "endpoints": {},
        }
        token = self._call_context.set(context)
        try:
            yield context
        finally:
            self._call_context.reset(token)

    def call_metrics(
        self, *, reset: bool = False, transcript_id: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return endpoint request counts, failures, and latency totals/averages."""

        with self._metrics_lock:
            metrics = (
                self._transcript_metrics.get(transcript_id, {})
                if transcript_id is not None
                else self._call_metrics
            )
            result: dict[str, dict[str, Any]] = {}
            for endpoint, stats in metrics.items():
                snapshot = dict(stats)
                snapshot["stages"] = {
                    stage: dict(stage_stats)
                    for stage, stage_stats in stats.get("stages", {}).items()
                }
                calls = int(snapshot["calls"])
                snapshot["avg_latency_ms"] = (
                    float(snapshot["latency_total_ms"]) / calls if calls else 0.0
                )
                snapshot["wall_time_ms"] = float(snapshot["latency_total_ms"])
                result[endpoint] = snapshot
            if reset:
                if transcript_id is None:
                    self._call_metrics.clear()
                    self._transcript_metrics.clear()
                else:
                    self._transcript_metrics.pop(transcript_id, None)
            return result

    @staticmethod
    def _parse_memory_write_response(payload: dict[str, Any]) -> MemoryWriteSamplingDecision:
        parsed_payload = _extract_json_payload(LocalLLMDecider._extract_content(payload))
        return parse_memory_write_sampling_result(
            {
                "content": {
                    "type": "text",
                    "text": json.dumps(parsed_payload, ensure_ascii=False),
                }
            }
        )

    def _acquire_endpoint(self, excluded: set[int]) -> int:
        with self._routing:
            while True:
                available = [
                    index for index, active in enumerate(self._active)
                    if index not in excluded and active < self._endpoint_limit
                ]
                if available:
                    least_loaded = min(self._active[index] for index in available)
                    tied = [index for index in available if self._active[index] == least_loaded]
                    index = next(
                        (candidate for offset in range(len(self._endpoints))
                         if (candidate := (self._next_endpoint + offset) % len(self._endpoints)) in tied),
                        tied[0],
                    )
                    self._active[index] += 1
                    self._next_endpoint = (index + 1) % len(self._endpoints)
                    return index
                self._routing.wait()

    def _release_endpoint(self, index: int) -> None:
        with self._routing:
            self._active[index] -= 1
            self._routing.notify_all()

    def _record_endpoint_call(self, index: int, *, elapsed_ms: float, succeeded: bool) -> None:
        base_url, model, _api_key_env, _extra_body = self._endpoints[index]
        key = f"{model}@{base_url}"
        context = self._call_context.get()
        with self._metrics_lock:
            stats = self._call_metrics.setdefault(key, self._new_endpoint_metrics(base_url, model))
            self._update_endpoint_metrics(stats, elapsed_ms=elapsed_ms, succeeded=succeeded)
            if context is not None:
                transcript_id = context["transcript_id"]
                stage = context["stage"]
                by_endpoint = self._transcript_metrics.setdefault(transcript_id, {})
                transcript_stats = by_endpoint.setdefault(key, self._new_endpoint_metrics(base_url, model))
                self._update_endpoint_metrics(transcript_stats, elapsed_ms=elapsed_ms, succeeded=succeeded)
                stage_stats = transcript_stats["stages"].setdefault(stage, self._new_stage_metrics())
                self._update_endpoint_metrics(stage_stats, elapsed_ms=elapsed_ms, succeeded=succeeded)
                context_stats = context["endpoints"].setdefault(key, self._new_endpoint_metrics(base_url, model))
                self._update_endpoint_metrics(context_stats, elapsed_ms=elapsed_ms, succeeded=succeeded)

    @staticmethod
    def _new_endpoint_metrics(base_url: str, model: str) -> dict[str, Any]:
        return {
            "base_url": base_url,
            "model": model,
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "latency_total_ms": 0.0,
            "latency_max_ms": 0.0,
            "stages": {},
        }

    @staticmethod
    def _new_stage_metrics() -> dict[str, Any]:
        return {"calls": 0, "successes": 0, "failures": 0, "latency_total_ms": 0.0, "latency_max_ms": 0.0}

    @staticmethod
    def _update_endpoint_metrics(stats: dict[str, Any], *, elapsed_ms: float, succeeded: bool) -> None:
        stats["calls"] += 1
        stats["successes" if succeeded else "failures"] += 1
        stats["latency_total_ms"] += elapsed_ms
        stats["latency_max_ms"] = max(stats["latency_max_ms"], elapsed_ms)

    def _complete(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        parser: Callable[[dict[str, Any]], Any],
        reasoning_effort: str | None = None,
    ) -> Any:
        if not self._endpoints:
            raise RuntimeError("no local LLM endpoint configured")
        attempted: set[int] = set()
        while len(attempted) < len(self._endpoints):
            index = self._acquire_endpoint(attempted)
            attempted.add(index)
            base_url, model, api_key_env, extra_body = self._endpoints[index]
            started = perf_counter()
            call_context = self._call_context.get()
            timeout_seconds = (
                call_context.get("timeout_seconds")
                if call_context is not None and call_context.get("timeout_seconds") is not None
                else self.settings.timeout_seconds
            )
            try:
                response_payload = self._post_completion(
                    base_url=base_url,
                    model=model,
                    api_key_env=api_key_env,
                    messages=messages,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    timeout_seconds=timeout_seconds,
                    extra_body=extra_body,
                )
                result = parser(response_payload)
            except Exception as exc:  # count every endpoint attempt, then fail over only on transport/5xx
                elapsed_ms = (perf_counter() - started) * 1000
                self._record_endpoint_call(index, elapsed_ms=elapsed_ms, succeeded=False)
                retryable = isinstance(exc, httpx.RequestError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and 500 <= exc.response.status_code < 600
                )
                if not retryable or len(attempted) >= len(self._endpoints):
                    raise
                LOGGER.warning("Local LLM endpoint %s failed; trying another endpoint: %s", base_url, exc)
            else:
                elapsed_ms = (perf_counter() - started) * 1000
                self._record_endpoint_call(index, elapsed_ms=elapsed_ms, succeeded=True)
                return result
            finally:
                self._release_endpoint(index)
        raise RuntimeError("all local LLM endpoints failed")

    def _post_completion(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        timeout_seconds: int,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.settings.temperature,
            "response_format": {"type": "json_object"},
        }
        # OpenAI-style reasoning-effort control (owner 2026-10-12): passed only
        # when explicitly configured ([distiller].reasoning_effort).
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if extra_body:
            protected = {"model", "messages", "max_tokens", "response_format"}
            conflicts = protected.intersection(extra_body)
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise ValueError(f"extra_body cannot override completion fields: {names}")
            body.update(extra_body)
        response = self._client.post(
            f"{base_url.rstrip('/')}{_CHAT_COMPLETIONS_PATH}",
            headers=headers,
            json=body,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Local LLM response must be a JSON object")
        return payload

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Local LLM response did not contain choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Local LLM response content must be a non-empty string")
        return content