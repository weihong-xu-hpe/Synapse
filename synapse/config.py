"""Configuration loading and validation for Synapse."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ uses tomllib
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG_FILE_NAME = "config.toml"

EmbeddingModel = Literal["bge-m3", "jina-v3", "gte-qwen2", "gemma-300m"]
RerankerModel = Literal["bge-reranker-v2-m3", "jina-reranker-v2", "qllama/bge-reranker-v2-m3"]
InferenceProvider = Literal["remote_api", "builtin"]
RetrievalEngine = Literal["sqlite", "lancedb"]


DEFAULT_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "bge-m3": 1024,
    "jina-v3": 1024,
    "gte-qwen2": 1536,
    "gemma-300m": 1024,
}


class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=1, le=65535)
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["*"])
    auth_token: str = ""


class MemorySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_path: Path = Path("./.synapse")
    archive_path: Path = Path("./.synapse/.archive")


class RemoteAPIProviderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "https://api.example.com"
    embedding_base_url: str = ""  # if set, overrides base_url for embedding requests
    embedding_endpoint: str = "/v1/embeddings"
    rerank_endpoint: str = "/v1/rerank"
    api_key_env: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    request_timeout_seconds: int = Field(default=30, ge=1)


class ProviderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    remote_api: RemoteAPIProviderSettings = Field(default_factory=RemoteAPIProviderSettings)


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: InferenceProvider = "remote_api"
    model: EmbeddingModel = "bge-m3"
    dimension: int | None = Field(default=None, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)

    @model_validator(mode="after")
    def populate_default_dimension(self) -> "EmbeddingSettings":
        if self.dimension is None:
            self.dimension = DEFAULT_EMBEDDING_DIMENSIONS[self.model]
        return self


class RerankerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: InferenceProvider = "remote_api"
    model: RerankerModel = "bge-reranker-v2-m3"
    max_candidates: int = Field(default=9, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: RetrievalEngine = "sqlite"
    rrf_k: int = Field(default=60, ge=1)
    top_k: int = Field(default=3, ge=1)

    def anchor_limit(self) -> int:
        return min(3, self.top_k)


class DecaySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor: float = Field(default=0.98, gt=0.0, le=1.0)
    janitor_days: int = Field(default=30, ge=1)
    archive_retention_days: int = Field(default=90, ge=1)


class CustomSanitizationPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str
    replacement: str = "[REDACTED_CUSTOM]"
    label: str = "CUSTOM"


class SanitizationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    custom_patterns: list[str | CustomSanitizationPattern] = Field(default_factory=list)


class LoggingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retention_days: int = Field(default=7, ge=1)
    max_file_size_mb: int = Field(default=50, ge=1)
    log_dir: Path = Path("./.synapse/.logs")


class DeciderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "local_llm"
    base_url: str = "http://localhost:8000/v1"
    model: str = "deepseek-v4-pro"
    api_key_env: str = ""
    fallback_base_url: str = ""
    fallback_model: str = ""
    fallback_api_key_env: str = "OPENAI_COMPATIBLE_API_KEY"
    timeout_seconds: int = 30
    max_tokens: int = 600
    temperature: float = 0.1


class DreamerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    interval_hours: int = Field(default=12, ge=1)
    batch_size: int = Field(default=8, ge=1, le=20)


class SynapseConfig(BaseModel):
    """Validated Synapse runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    server: ServerSettings = Field(default_factory=ServerSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    decay: DecaySettings = Field(default_factory=DecaySettings)
    sanitization: SanitizationSettings = Field(default_factory=SanitizationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    decider: DeciderSettings = Field(default_factory=DeciderSettings)
    dreamer: DreamerSettings = Field(default_factory=DreamerSettings)

    _config_path: Path = PrivateAttr(default=Path(DEFAULT_CONFIG_FILE_NAME))
    _project_root: Path = PrivateAttr(default=Path.cwd())

    def attach_config_path(self, config_path: Path) -> None:
        resolved = config_path.resolve()
        self._config_path = resolved
        self._project_root = resolved.parent

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def project_root(self) -> Path:
        return self._project_root

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.project_root / path).resolve()

    @classmethod
    def with_defaults(cls, project_root: Path) -> "SynapseConfig":
        config = cls()
        config.attach_config_path(project_root / DEFAULT_CONFIG_FILE_NAME)
        return config


def resolve_config_path(config_path: str | Path | None = None, cwd: str | Path | None = None) -> tuple[Path, bool]:
    """Resolve the config location.

    Resolution order:
    1. Explicit `config_path`
    2. `SYNAPSE_CONFIG_PATH`
    3. `./config.toml` relative to `cwd`
    """

    working_directory = Path(cwd or Path.cwd()).expanduser().resolve()

    explicit_source = config_path
    if explicit_source is None:
        explicit_source = os.getenv("SYNAPSE_CONFIG_PATH")

    if explicit_source is not None:
        candidate = Path(explicit_source).expanduser()
        if not candidate.is_absolute():
            candidate = working_directory / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Synapse config not found: {candidate}")
        return candidate, True

    default_candidate = (working_directory / DEFAULT_CONFIG_FILE_NAME).resolve()
    return default_candidate, default_candidate.exists()


def load_config(config_path: str | Path | None = None, cwd: str | Path | None = None) -> SynapseConfig:
    """Load a Synapse config file and apply defaults for missing sections.

    If no config file exists at the default location, a fully defaulted configuration
    is returned and anchored to the inferred project root.
    """

    resolved_path, exists = resolve_config_path(config_path=config_path, cwd=cwd)

    if not exists:
        return SynapseConfig.with_defaults(resolved_path.parent)

    with resolved_path.open("rb") as handle:
        raw_data = tomllib.load(handle)

    config = SynapseConfig.model_validate(raw_data)
    config.attach_config_path(resolved_path)
    return config
