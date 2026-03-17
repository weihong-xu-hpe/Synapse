from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError

import synapse.embedding.engines as engine_module
from synapse.config import load_config
from synapse.indexing import collect_health_status, rebuild_index, run_startup_checks
from synapse.models import Node, NodeMetadata, NodeType, SensitivityLevel
from synapse.storage import SQLiteNodeStore, write_node_file
from synapse.utils.runtime import bootstrap_runtime_directories


FIXED_TIME = datetime(2026, 3, 7, 10, 0, tzinfo=UTC)


def write_config(base_dir: Path, *, embedding_provider: str = "builtin") -> Path:
    config_path = base_dir / "config.toml"
    provider_block = """
[providers.remote_api]
base_url = "https://models.example.com"
embedding_endpoint = "/v1/embeddings"
rerank_endpoint = "/v1/rerank"
api_key_env = "SYNAPSE_MODEL_API_KEY"
headers = {}
request_timeout_seconds = 1
""".strip()
    config_path.write_text(
        f"""
[server]
host = "127.0.0.1"
port = 8765

[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "{embedding_provider}"
model = "bge-m3"
dimension = 1024
timeout_seconds = 1

[reranker]
provider = "builtin"
model = "bge-reranker-v2-m3"
max_candidates = 9
timeout_seconds = 1

{provider_block}

[logging]
log_dir = "./.synapse/.logs"
""".strip(),
        encoding="utf-8",
    )
    return config_path


def write_sample_nodes(runtime_base: Path) -> None:
    alpha = Node(
        metadata=NodeMetadata(
            id="mem_20260307_api_gateway_design",
            title="API Gateway Design",
            created_at=FIXED_TIME,
            last_accessed=FIXED_TIME,
            type=NodeType.PERSISTENT,
            supersedes=[],
            tags=["gateway", "design"],
            sensitivity=SensitivityLevel.INTERNAL,
        ),
        content="""
# API Gateway Design

This note references [[Rate Limiting Strategy]].
""".strip(),
        file_path=Path("active/mem_20260307_api_gateway_design.md"),
    )
    beta = Node(
        metadata=NodeMetadata(
            id="mem_20260307_rate_limiting_strategy",
            title="Rate Limiting Strategy",
            created_at=FIXED_TIME,
            last_accessed=FIXED_TIME,
            type=NodeType.PERSISTENT,
            supersedes=[],
            tags=["gateway", "rate-limit"],
            sensitivity=SensitivityLevel.INTERNAL,
        ),
        content="""
# Rate Limiting Strategy

Token bucket rate limiting for API gateways.
""".strip(),
        file_path=Path("active/mem_20260307_rate_limiting_strategy.md"),
    )
    write_node_file(alpha, base_path=runtime_base)
    write_node_file(beta, base_path=runtime_base)


def test_rebuild_index_is_idempotent_and_populates_sqlite(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    write_sample_nodes(runtime_paths.base)
    progress_messages: list[str] = []

    first = rebuild_index(config, runtime_paths=runtime_paths, progress_callback=progress_messages.append)
    second = rebuild_index(config, runtime_paths=runtime_paths)

    assert first.indexed_nodes == 2
    assert second.indexed_nodes == 2
    assert first.embedding_fingerprint == second.embedding_fingerprint
    assert any("[1/2] Processing" in message for message in progress_messages)
    assert first.integrity_ok is True
    assert first.vector_backend == "python-fallback"

    with SQLiteNodeStore(first.database_path, embedding_dimension=config.embedding.dimension or 0) as store:
        assert store.count_nodes() == 2
        assert store.get_edges("mem_20260307_api_gateway_design") == ["mem_20260307_rate_limiting_strategy"]
        assert store.get_embedding_fingerprint() == first.embedding_fingerprint


def test_run_startup_checks_recovers_from_corrupt_db(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    write_sample_nodes(runtime_paths.base)
    db_path = runtime_paths.base / "synapse.db"
    db_path.write_bytes(b"not a sqlite database")
    progress_messages: list[str] = []

    report = run_startup_checks(
        config,
        runtime_paths=runtime_paths,
        auto_rebuild=True,
        progress_callback=progress_messages.append,
    )

    assert report.rebuilt is True
    assert report.database_integrity is not None
    assert report.database_integrity.ok is True
    assert report.health.components["sqlite"] == "ok"
    assert any("Rebuilding index at" in message for message in progress_messages)

    with SQLiteNodeStore(db_path, embedding_dimension=config.embedding.dimension or 0) as store:
        assert store.count_nodes() == 2


def test_collect_health_status_reports_degraded_embedding_when_provider_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def failing_urlopen(request, timeout):
        del request, timeout
        raise URLError("connection refused")

    monkeypatch.setattr(engine_module.urllib_request, "urlopen", failing_urlopen)

    config = load_config(write_config(tmp_path, embedding_provider="remote_api"))
    runtime_paths = bootstrap_runtime_directories(config)
    write_sample_nodes(runtime_paths.base)
    rebuild_index(config, runtime_paths=runtime_paths)

    health = collect_health_status(config, runtime_paths=runtime_paths)

    assert health.components["sqlite"] == "ok"
    assert health.components["embedding_model"] == "degraded"
    assert health.components["vector_index"] == "python-fallback"
    assert health.stats["total_nodes"] == 2
    assert health.delta_sync_hook == "enabled"
    assert health.startup_sync_hook == "enabled"
    assert health.status == "degraded"
