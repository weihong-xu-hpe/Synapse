from __future__ import annotations

from pathlib import Path

import pytest

from synapse.config import load_config
from synapse.utils.runtime import bootstrap_runtime_directories


def test_load_config_uses_defaults_for_missing_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[server]
port = 9999

[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9999
    assert config.embedding.model == "bge-m3"
    assert config.embedding.provider == "remote_api"
    assert config.embedding.dimension == 1024
    assert config.reranker.provider == "remote_api"
    assert config.logging.retention_days == 7
    assert config.decay.janitor_days == 30
    assert config.decay.archive_retention_days == 90
    assert config.resolve_path(config.memory.base_path) == (tmp_path / ".synapse").resolve()


def test_load_config_honors_env_override_and_bootstraps_runtime(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        """
[server]
host = "127.0.0.1"
port = 9100

[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[logging]
log_dir = "./.synapse/.logs"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("SYNAPSE_CONFIG_PATH", str(config_path))

    config = load_config(cwd=tmp_path)
    runtime_paths = bootstrap_runtime_directories(config)

    assert config.server.host == "127.0.0.1"
    assert config.server.port == 9100
    assert config.config_path == config_path.resolve()
    assert runtime_paths.active.exists()
    assert runtime_paths.archive.exists()
    assert runtime_paths.logs.exists()
    assert runtime_paths.audit.exists()


def test_load_config_parses_provider_blocks(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[embedding]
provider = "remote_api"
model = "gte-qwen2"
dimension = 1536
timeout_seconds = 45

[reranker]
provider = "remote_api"
model = "jina-reranker-v2"
max_candidates = 12
timeout_seconds = 20

[providers.remote_api]
base_url = "https://models.example.com"
embedding_base_url = "https://embed.example.com"
embedding_endpoint = "/v1/embeddings"
rerank_endpoint = "/v1/rerank"
api_key_env = "SYNAPSE_MODEL_API_KEY"
headers = { X-Tenant = "dev", Authorization = "Bearer {api_key}" }
request_timeout_seconds = 15
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.embedding.provider == "remote_api"
    assert config.embedding.model == "gte-qwen2"
    assert config.embedding.dimension == 1536
    assert config.embedding.timeout_seconds == 45
    assert config.reranker.provider == "remote_api"
    assert config.reranker.model == "jina-reranker-v2"
    assert config.reranker.max_candidates == 12
    assert config.providers.remote_api.base_url == "https://models.example.com"
    assert config.providers.remote_api.embedding_base_url == "https://embed.example.com"
    assert config.providers.remote_api.api_key_env == "SYNAPSE_MODEL_API_KEY"
    assert config.providers.remote_api.headers["X-Tenant"] == "dev"
    assert config.providers.remote_api.headers["Authorization"] == "Bearer {api_key}"
    assert config.providers.remote_api.request_timeout_seconds == 15
