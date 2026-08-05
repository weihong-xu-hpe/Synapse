"""Local OpenAI-compatible LLM decision client."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

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
    """Synchronous SamplingClient implementation backed by an HTTP LLM."""

    name = "local-llm"

    def __init__(self, settings: DeciderSettings, *, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.timeout_seconds)

    def sample_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 600,
        model_hints: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        del system_prompt, model_hints
        return self._complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            parser=lambda payload: _extract_json_payload(self._extract_content(payload)),
        )

    def decide_memory_write(self, request: MemoryWriteSamplingRequest) -> MemoryWriteSamplingDecision:
        return self._complete(
            messages=[{"role": "user", "content": build_memory_write_sampling_prompt(request)}],
            max_tokens=self.settings.max_tokens,
            parser=self._parse_memory_write_response,
        )

    @staticmethod
    def _parse_memory_write_response(payload: dict[str, Any]) -> MemoryWriteSamplingDecision:
        parsed_payload = _extract_json_payload(LocalLLMDecider._extract_content(payload))
        # parse_memory_write_sampling_result owns the decision-shape conversion and
        # validation contract used by the MCP sampling implementation. Feed it the
        # already extracted payload through its standard text-content envelope.
        return parse_memory_write_sampling_result(
            {
                "content": {
                    "type": "text",
                    "text": json.dumps(parsed_payload, ensure_ascii=False),
                }
            }
        )

    def _complete(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        parser: Callable[[dict[str, Any]], Any],
    ) -> Any:
        last_error: Exception | None = None
        endpoints = (
            (self.settings.base_url, self.settings.model, self.settings.api_key_env),
            (
                self.settings.fallback_base_url,
                self.settings.fallback_model,
                self.settings.fallback_api_key_env,
            ),
        )
        for base_url, model, api_key_env in endpoints:
            try:
                response_payload = self._post_completion(
                    base_url=base_url,
                    model=model,
                    api_key_env=api_key_env,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return parser(response_payload)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                LOGGER.warning("Local LLM endpoint %s failed: %s", base_url, exc)

        assert last_error is not None
        raise last_error

    def _post_completion(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        response = self._client.post(
            f"{base_url.rstrip('/')}{_CHAT_COMPLETIONS_PATH}",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": self.settings.temperature,
                "response_format": {"type": "json_object"},
            },
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