from __future__ import annotations

from pathlib import Path

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
    assert config.decider.provider == "local_llm"
    assert config.decider.base_url == "http://localhost:8000/v1"
    assert config.decider.model == "deepseek-v4-pro"
    assert config.decider.timeout_seconds == 30
    assert config.decider.max_tokens == 600
    assert config.decider.temperature == 0.1
    assert config.dreamer.enabled is True
    assert config.dreamer.interval_hours == 12
    assert config.dreamer.batch_size == 8
    assert config.dreamer.thresholds.missing_link_cosine == 0.75
    assert config.dreamer.thresholds.stale_orphan_days is None
    assert config.dreamer.thresholds.link_weaving_recency_days is None
    assert config.dreamer.thresholds.superseded_archive_days == 7
    assert config.dreamer.thresholds.low_structure_chars == 100
    assert config.dreamer.thresholds.max_missing_link_pairs_per_run == 100
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


def test_load_config_parses_decider_block(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[decider]
provider = "mcp_sampling"
base_url = "http://primary.example/v1"
model = "primary-model"
api_key_env = "PRIMARY_KEY"
fallback_base_url = "http://fallback.example/v1"
fallback_model = "fallback-model"
fallback_api_key_env = "FALLBACK_KEY"
timeout_seconds = 17
max_tokens = 321
temperature = 0.25
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.decider.provider == "mcp_sampling"
    assert config.decider.base_url == "http://primary.example/v1"
    assert config.decider.model == "primary-model"
    assert config.decider.api_key_env == "PRIMARY_KEY"
    assert config.decider.fallback_base_url == "http://fallback.example/v1"
    assert config.decider.fallback_model == "fallback-model"
    assert config.decider.fallback_api_key_env == "FALLBACK_KEY"
    assert config.decider.timeout_seconds == 17
    assert config.decider.max_tokens == 321
    assert config.decider.temperature == 0.25


def test_load_config_parses_dreamer_block(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[dreamer]
enabled = false
interval_hours = 6
batch_size = 12

[dreamer.thresholds]
missing_link_cosine = 0.82
stale_orphan_days = 14
link_weaving_recency_days = 21
superseded_archive_days = 3
low_structure_chars = 80
max_missing_link_pairs_per_run = 25
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dreamer.enabled is False
    assert config.dreamer.interval_hours == 6
    assert config.dreamer.batch_size == 12
    assert config.dreamer.thresholds.missing_link_cosine == 0.82
    assert config.dreamer.thresholds.stale_orphan_days == 14
    assert config.dreamer.thresholds.link_weaving_recency_days == 21
    assert config.dreamer.thresholds.superseded_archive_days == 3
    assert config.dreamer.thresholds.low_structure_chars == 80
    assert config.dreamer.thresholds.max_missing_link_pairs_per_run == 25
