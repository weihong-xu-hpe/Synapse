"""HTTP client for TypeSafe / StartLux / Karpo System One endpoints."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Sequence

import httpx

from synapse.config import DecisionModelEndpointSettings, DecisionModelSettings

LOGGER = logging.getLogger("synapse.decision")


class DecisionClient:
    """Fail-open System One client with ordered multi-endpoint failover."""

    def __init__(
        self,
        settings: DecisionModelSettings,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self._http_client = http_client

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled and self.settings.endpoints)

    def _build_headers(self, endpoint: DecisionModelEndpointSettings) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        env_name = (endpoint.api_key_env or "").strip()
        if env_name:
            token = os.environ.get(env_name, "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def _endpoint_url(self, endpoint: DecisionModelEndpointSettings) -> str:
        base = endpoint.base_url.rstrip("/")
        if base.endswith("/v1/systemone"):
            return base
        if base.endswith("/v1"):
            return f"{base}/systemone"
        return f"{base}/v1/systemone"

    def decide(
        self,
        state: Mapping[str, Any] | str,
        questions: Sequence[Mapping[str, Any]],
        *,
        offline: bool = False,
    ) -> dict[str, Any] | None:
        """Query System One endpoints in order; return parsed JSON or ``None``."""

        if not self.enabled:
            return None

        timeout_ms = self.settings.offline_timeout_ms if offline else self.settings.timeout_ms
        timeout_seconds = max(0.001, float(timeout_ms) / 1000.0)
        base_payload: dict[str, Any] = {
            "state": state,
            "questions": [dict(question) for question in questions],
        }

        owns_client = self._http_client is None
        client = self._http_client or httpx.Client(timeout=timeout_seconds)
        try:
            for endpoint in self.settings.endpoints:
                url = self._endpoint_url(endpoint)
                headers = self._build_headers(endpoint)
                payload = dict(base_payload)
                if endpoint.model:
                    payload["model"] = endpoint.model
                try:
                    response = client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=timeout_seconds,
                    )
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, dict):
                        return data
                    LOGGER.warning(
                        "System One endpoint %s returned non-object payload",
                        endpoint.name,
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    LOGGER.warning(
                        "System One endpoint %s failed (%s); failing over",
                        endpoint.name,
                        exc,
                    )
            return None
        finally:
            if owns_client:
                client.close()
