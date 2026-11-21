"""HTTP client for TypeSafe / StartLux / Karpo System One endpoints."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Sequence

import httpx

from synapse.config import DecisionModelEndpointSettings, DecisionModelSettings

LOGGER = logging.getLogger("synapse.decision")

_QUESTION_TYPES = frozenset({"choice", "noul", "bool", "score"})


def validate_questions(questions: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> str | None:
    """Return why the server would reject ``questions`` with a 422, or ``None`` when they are well formed.

    Mirrors the System One contract (startlux-decision ``docs/api.md``): the question text is ``instructions`` (a
    ``prompt`` key is silently ignored and the model then answers a generic "which answer fits the evidence?"),
    ``choice`` needs ``criteria`` (a list of option ids or ``{option: description}``), ``score`` needs 2-10 levels.
    """

    items = list(questions.items()) if isinstance(questions, Mapping) else [
        (str(q.get("id", index)), q) for index, q in enumerate(questions)
    ]
    if not items:
        return "no questions"
    for name, question in items:
        if not isinstance(question, Mapping):
            return f"question {name!r} is not an object"
        qtype = question.get("type")
        if qtype not in _QUESTION_TYPES:
            return f"question {name!r} has type {qtype!r}; expected one of {sorted(_QUESTION_TYPES)}"
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            return f"question {name!r} needs non-empty 'instructions' (a 'prompt' key is ignored by the server)"
        criteria = question.get("criteria")
        if qtype == "choice" and (not criteria or (isinstance(criteria, (list, tuple)) and len(criteria) < 2)):
            return f"choice question {name!r} needs 'criteria': a list of options or {{option: description}}"
        if qtype == "score" and (not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10):
            return f"score question {name!r} needs 'criteria': a list of 2 to 10 levels"
    return None


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
        questions: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
        *,
        offline: bool = False,
    ) -> dict[str, Any] | None:
        """Query System One endpoints in order; return parsed JSON or ``None``."""

        if not self.enabled:
            return None

        timeout_ms = self.settings.offline_timeout_ms if offline else self.settings.timeout_ms
        timeout_seconds = max(0.001, float(timeout_ms) / 1000.0)
        problem = validate_questions(questions)
        if problem is not None:
            # A request we know the server rejects (422) is a caller bug, not an endpoint outage: no failover.
            LOGGER.error("System One request not sent: %s", problem)
            return None
        serialized_questions: dict[str, dict[str, Any]] | list[dict[str, Any]]
        if isinstance(questions, Mapping):
            serialized_questions = {str(key): dict(value) for key, value in questions.items()}
        else:
            serialized_questions = [dict(question) for question in questions]
        base_payload: dict[str, Any] = {
            "state": state,
            "questions": serialized_questions,
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
