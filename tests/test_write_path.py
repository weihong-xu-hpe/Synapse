from __future__ import annotations

from pathlib import Path

from synapse.config import SynapseConfig, load_config
from synapse.server.service import SynapseServerService
from synapse.storage import SQLiteNodeStore
from synapse.utils.runtime import RuntimePaths, bootstrap_runtime_directories


def write_config(base_dir: Path) -> Path:
    config_path = base_dir / "config.toml"
    config_path.write_text(
        """
[server]
host = "127.0.0.1"
port = 8765

[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "builtin"
model = "bge-m3"
dimension = 1024
timeout_seconds = 1

[reranker]
provider = "builtin"
model = "bge-reranker-v2-m3"
max_candidates = 9
timeout_seconds = 1

[retrieval]
engine = "sqlite"
rrf_k = 60
top_k = 3

[logging]
log_dir = "./.synapse/.logs"
""".strip(),
        encoding="utf-8",
    )
    return config_path


def make_service(tmp_path: Path) -> tuple[SynapseServerService, SynapseConfig, RuntimePaths]:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths)
    return service, config, runtime_paths


def test_integrate_knowledge_supersedes_existing_node_and_syncs_db(tmp_path: Path) -> None:
    service, config, runtime_paths = make_service(tmp_path)
    original = service.write_node(
        title="Rate Limiting Strategy",
        content="Token bucket is the current rate limiting strategy.",
        node_type="persistent",
    )
    original_id = original["node"]["id"]

    result = service.integrate_knowledge(
        title="Sliding Window Rate Limiting",
        content="Sliding window replaces token bucket for burst traffic handling.",
        node_type="persistent",
        action="supersede",
        target_node_ids=[original_id],
        reasoning="Sliding window replaced token bucket for burst traffic.",
    )

    new_id = result["node"]["id"]
    assert result["action"] == "supersede"
    assert result["target_node_ids"] == [original_id]
    assert result["node"]["metadata"]["status"] == "active"
    assert result["node"]["metadata"]["supersedes"] == [original_id]
    assert result["updated_nodes"][0]["metadata"]["status"] == "superseded"
    assert result["updated_nodes"][0]["metadata"]["superseded_by"] == new_id

    new_markdown = (runtime_paths.active / f"{new_id}.md").read_text(encoding="utf-8")
    original_markdown = (runtime_paths.active / f"{original_id}.md").read_text(encoding="utf-8")
    assert f"> **Supersedes**: [[{original_id}]] — Sliding window replaced token bucket for burst traffic." in new_markdown
    assert f"> ⚠️ **SUPERSEDED** by [[{new_id}]]" in original_markdown

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        stored_new = store.get_node(new_id)
        stored_original = store.get_node(original_id)
        assert stored_new is not None
        assert stored_original is not None
        assert stored_new.metadata.supersedes == [original_id]
        assert stored_original.metadata.status.value == "superseded"
        assert stored_original.metadata.superseded_by == new_id
        assert store.get_edges(new_id) == [original_id]


def test_integrate_knowledge_complements_existing_nodes_with_reciprocal_links(tmp_path: Path) -> None:
    service, config, runtime_paths = make_service(tmp_path)
    original = service.write_node(
        title="API Gateway Design",
        content="Gateway design covers auth and request routing.",
    )
    original_id = original["node"]["id"]

    result = service.integrate_knowledge(
        title="Gateway Rate Limits",
        content="Gateway rate limits complement the core gateway design.",
        action="complement",
        target_node_ids=[original_id],
        reasoning="Architecture and rate limiting are complementary views.",
    )

    new_id = result["node"]["id"]
    original_markdown = (runtime_paths.active / f"{original_id}.md").read_text(encoding="utf-8")
    new_markdown = (runtime_paths.active / f"{new_id}.md").read_text(encoding="utf-8")
    assert result["action"] == "complement"
    assert f"[[{original_id}]]" in new_markdown
    assert f"[[{new_id}]]" in original_markdown

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        assert store.get_edges(new_id) == [original_id]
        assert store.get_edges(original_id) == [new_id]


def test_integrate_knowledge_create_writes_new_node_without_touching_existing(tmp_path: Path) -> None:
    service, config, runtime_paths = make_service(tmp_path)
    existing = service.write_node(
        title="Deployment Strategy",
        content="Rolling updates are the preferred deployment strategy.",
    )
    existing_id = existing["node"]["id"]

    result = service.integrate_knowledge(
        title="Blue Green Deployment Notes",
        content="Blue green deployment is an alternative approach.",
        action="create",
    )

    new_id = result["node"]["id"]
    assert result["action"] == "create"
    assert result["target_node_ids"] == []
    assert result["updated_nodes"] == []
    assert result["node"]["metadata"]["status"] == "active"
    assert service.get_node(existing_id)["metadata"]["status"] == "active"

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        stored_new = store.get_node(new_id)
        stored_existing = store.get_node(existing_id)
        assert stored_new is not None
        assert stored_existing is not None
        assert stored_existing.metadata.status.value == "active"


def test_integrate_knowledge_supports_chain_supersession_without_losing_history(tmp_path: Path) -> None:
    service, _config, runtime_paths = make_service(tmp_path)
    alpha = service.write_node(
        title="Gateway Strategy V1",
        content="Token bucket is the initial gateway strategy.",
    )
    alpha_id = alpha["node"]["id"]

    beta = service.integrate_knowledge(
        title="Gateway Strategy V2",
        content="Sliding window replaces token bucket in the second version.",
        action="supersede",
        target_node_ids=[alpha_id],
        reasoning="Version 2 replaced version 1.",
    )
    beta_id = beta["node"]["id"]

    gamma = service.integrate_knowledge(
        title="Gateway Strategy V3",
        content="Adaptive rate limiting replaces sliding window in the third version.",
        action="supersede",
        target_node_ids=[beta_id],
        reasoning="Version 3 replaced version 2.",
    )
    gamma_id = gamma["node"]["id"]

    alpha_node = service.get_node(alpha_id)
    beta_node = service.get_node(beta_id)
    gamma_node = service.get_node(gamma_id)
    beta_markdown = (runtime_paths.active / f"{beta_id}.md").read_text(encoding="utf-8")

    assert alpha_node["metadata"]["status"] == "superseded"
    assert alpha_node["metadata"]["superseded_by"] == beta_id
    assert beta_node["metadata"]["status"] == "superseded"
    assert beta_node["metadata"]["superseded_by"] == gamma_id
    assert beta_node["metadata"]["supersedes"] == [alpha_id]
    assert gamma_node["metadata"]["status"] == "active"
    assert gamma_node["metadata"]["supersedes"] == [beta_id]
    assert f"> **Supersedes**: [[{alpha_id}]] — Version 2 replaced version 1." in beta_markdown
    assert f"> ⚠️ **SUPERSEDED** by [[{gamma_id}]]" in beta_markdown

