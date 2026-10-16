from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from synapse.config import SynapseConfig, load_config
from synapse.models import NodeStatus, NodeType
from synapse.server.sampling import MemoryWriteSamplingDecision
from synapse.okf import validate_okf_node
from synapse.server.service import SynapseServiceError, SynapseServerService
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



def test_distiller_write_repairs_final_body_and_records_final_warnings(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    class RepairClient:
        name = "repair-test"

        def __init__(self) -> None:
            self.repairs = 0

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, **_kwargs):
            assert "Return the FULL corrected body" in prompt
            assert "exactly one non-empty `## Takeaway` line of at most 200 characters" in prompt
            assert "Each required heading must occur exactly once and have a non-empty body" in prompt
            assert "Required headings: ## Takeaway, ## Details, ## Sources" in prompt
            self.repairs += 1
            return {
                "content": (
                    "## Takeaway\nKeep the final persisted body valid.\n\n"
                    "## Details\nThe repaired body retains the complete fact.\n\n"
                    "## Sources\n- mem_session_repair"
                )
            }

        def decide_memory_write(self, _request):
            return MemoryWriteSamplingDecision(
                action="create", target_node_ids=(), reasoning="new fact", confidence=0.9
            )

    client = RepairClient()
    service = SynapseServerService(config, runtime_paths=runtime_paths, sampling_client=client)
    result = service.write_memory(
        title="Final OKF validation repairs incomplete facts",
        content="## Takeaway\nKeep the final persisted body valid.\n\n## Sources\n- mem_session_repair",
        node_type=NodeType.PERSISTENT,
        okf_type="fact",
        sources=["mem_session_repair"],
        route="distiller",
    )
    node_payload = result["execution"]["result"]["node"]
    assert client.repairs == 1
    assert result["warnings"] == []

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        node = store.get_node(node_payload["id"])
        event = store.connection.execute(
            "SELECT warning_codes FROM write_memory_events WHERE node_id = ? AND route = 'distiller' ORDER BY id DESC LIMIT 1",
            (node_payload["id"],),
        ).fetchone()
    assert node is not None and event is not None
    assert not validate_okf_node(
        node_type="persistent",
        content=node.content,
        okf_type=node.metadata.okf_type,
        sources=node.metadata.sources,
        title=node.title,
    )
    assert json.loads(event["warning_codes"]) == []


def test_distiller_final_repair_can_change_okf_type(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    class TypeRepairClient:
        name = "type-repair-test"

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, **_kwargs):
            if "invalid title" in prompt.casefold() or "english title" in prompt.casefold():
                return {"title": "Procedure preserves the required steps"}
            assert "You may also correct `okf_type`" in prompt
            return {
                "title": "Procedure preserves the required steps",
                "okf_type": "procedure",
                "content": (
                    "## Takeaway\nKeep the final persisted body valid.\n\n"
                    "## Steps\n1. Apply the supported procedure.\n\n"
                    "## Sources\n- mem_session_type_repair"
                ),
            }

        def decide_memory_write(self, _request):
            return MemoryWriteSamplingDecision(
                action="create", target_node_ids=(), reasoning="new procedure", confidence=0.9
            )

    service = SynapseServerService(config, runtime_paths=runtime_paths, sampling_client=TypeRepairClient())
    result = service.write_memory(
        title="Final OKF repair can correct the type and should shorten an overlong title with supported identifiers and details",
        content="## Takeaway\nKeep the final persisted body valid.\n\n## Steps\n1. Apply the supported procedure.\n\n## Sources\n- mem_session_type_repair",
        node_type=NodeType.PERSISTENT,
        okf_type="fact",
        sources=["mem_session_type_repair"],
        route="distiller",
    )
    node_payload = result["execution"]["result"]["node"]
    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        node = store.get_node(node_payload["id"])
    assert node is not None
    assert node.metadata.okf_type == "procedure"
    assert len(node.title) <= 90
    assert not validate_okf_node(
        node_type="persistent",
        content=node.content,
        okf_type=node.metadata.okf_type,
        sources=node.metadata.sources,
        title=node.title,
    )


@pytest.mark.parametrize(
    ("okf_type", "action", "draft", "repaired"),
    [
        (
            "fact",
            "complement",
            "## Takeaway\nKeep the final fact complete.\n\n## Sources\n- mem_session_gate",
            "## Takeaway\nKeep the final fact complete.\n\n## Details\nThe final fact retains its supported detail.\n\n## Sources\n- mem_session_gate",
        ),
        (
            "procedure",
            "supersede",
            "## Takeaway\nKeep the final procedure complete.\n\n## Sources\n- mem_session_gate",
            "## Takeaway\nKeep the final procedure complete.\n\n## Steps\n1. Apply the supported procedure.\n\n## Sources\n- mem_session_gate",
        ),
        (
            "decision",
            "complement",
            "## Takeaway\nPreserve the decision consequences.\n\n## Context\nThe system needs a stable policy.\n\n## Decision\nUse the approved policy.\n\n## Consequences\n\n## Consequences\nDuplicate heading.\n\n## Sources\n- mem_session_gate",
            "## Takeaway\nPreserve the decision consequences.\n\n## Context\nThe system needs a stable policy.\n\n## Decision\nUse the approved policy.\n\n## Consequences\nThe policy remains stable.\n\n## Sources\n- mem_session_gate",
        ),
        (
            "fact",
            "supersede",
            f"## Takeaway\n{'x' * 201}\n\n## Details\nKeep the complete fact.\n\n## Sources\n- mem_session_gate",
            "## Takeaway\nKeep the final fact under the character limit.\n\n## Details\nKeep the complete fact.\n\n## Sources\n- mem_session_gate",
        ),
    ],
)
def test_distiller_repairs_final_complement_and_supersede_bodies(
    tmp_path: Path, okf_type: str, action: str, draft: str, repaired: str
) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    target_type = "fact"
    service = SynapseServerService(config, runtime_paths=runtime_paths)
    target = service.integrate_knowledge(
        title="Existing knowledge retained by gate",
        content="## Takeaway\nExisting knowledge remains available.\n\n## Details\nKeep the old node active until a valid write.\n\n## Sources\n- mem_session_old",
        node_type=NodeType.PERSISTENT,
        action="create",
        okf_type=target_type,
        sources=["mem_session_old"],
    )
    target_id = target["node"]["id"]

    class RepairClient:
        name = "final-body-repair-test"

        def __init__(self) -> None:
            self.repairs = 0

        def sample_json(self, *, prompt: str, **_kwargs):
            assert "Repair this persistent OKF body" in prompt
            self.repairs += 1
            return {"content": repaired}

        def decide_memory_write(self, _request):
            return MemoryWriteSamplingDecision(
                action=action,
                target_node_ids=(target_id,),
                reasoning="retain complete lineage",
                confidence=0.9,
            )

    client = RepairClient()
    service.sampling_client = client
    result = service.write_memory(
        title=f"Final {okf_type} body repair",
        content=draft,
        node_type=NodeType.PERSISTENT,
        okf_type=okf_type,
        sources=["mem_session_gate"],
        route="distiller",
    )
    node_payload = result["execution"]["result"]["node"]
    assert client.repairs == 1
    assert result["decision"]["action"] == action

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        node = store.get_node(node_payload["id"])
        event = store.connection.execute(
            "SELECT warning_codes FROM write_memory_events WHERE node_id = ? AND route = 'distiller' ORDER BY id DESC LIMIT 1",
            (node_payload["id"],),
        ).fetchone()
    assert node is not None and event is not None
    findings = validate_okf_node(
        node_type="persistent",
        content=node.content,
        okf_type=node.metadata.okf_type,
        sources=node.metadata.sources,
        title=node.title,
    )
    assert not findings
    assert json.loads(event["warning_codes"]) == [finding.code for finding in findings]
    assert service.get_node(target_id)["metadata"]["status"] == (
        "superseded" if action == "supersede" else "active"
    )


@pytest.mark.parametrize("omitted_anchor", ["0.5–2s", "GLM-5.2-fp8", "120s", "/etc/authz/probe.yaml"])
def test_distiller_supersede_downgrades_when_target_anchors_are_lost(
    tmp_path: Path, monkeypatch, omitted_anchor: str
) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths)
    old = service.integrate_knowledge(
        title="Gateway probe timeout contract",
        content=(
            "## Takeaway\nKeep the explicit gateway timeout and model.\n\n"
            "## Details\nThe probe window is 0.5–2s, timeout is 120s, model is `GLM-5.2-fp8`, "
            "and the file is `/etc/authz/probe.yaml`.\n\n## Sources\n- mem_session_old"
        ),
        node_type=NodeType.PERSISTENT,
        action="create",
        okf_type="fact",
        sources=["mem_session_old"],
    )
    old_id = old["node"]["id"]

    anchors = {
        "0.5–2s": "The probe window is 0.5–2s.",
        "GLM-5.2-fp8": "The model is `GLM-5.2-fp8`.",
        "120s": "The timeout is 120s.",
        "/etc/authz/probe.yaml": "The file is `/etc/authz/probe.yaml`.",
    }
    retained_details = " ".join(
        description for anchor, description in anchors.items() if anchor != omitted_anchor
    )
    monkeypatch.setattr(
        service,
        "_decide_memory_write_payload",
        lambda **_kwargs: {
            "decision": {
                "action": "supersede",
                "target_node_ids": [old_id],
                "reasoning": "replace with the updated policy",
            },
            "evidence": {"candidate_count": 1, "sampling_provider": "fake"},
        },
    )
    result = service.write_memory(
        title="Updated gateway probe timeout policy",
        content=(
            "## Takeaway\nUse the shared gateway probe policy.\n\n"
            f"## Details\n{retained_details}\n\n## Sources\n- mem_session_new"
        ),
        node_type=NodeType.PERSISTENT,
        okf_type="fact",
        sources=["mem_session_new"],
        route="distiller",
        downgrade_supersede=False,
    )

    assert result["decision"]["action"] == "complement"
    assert result["downgrade"]["reason"].startswith("supersede downgraded to complement")
    old_after = service.get_node(old_id)["metadata"]
    assert old_after["status"] == "active"
    assert old_after["superseded_by"] is None


def test_distiller_supersede_preserves_evidence_section_anchors(tmp_path: Path, monkeypatch) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths)
    old = service.integrate_knowledge(
        title="Exported node and edge count",
        content=(
            "## Takeaway\nThe graph export size is a retained fact.\n\n"
            "## Details\nThe snapshot report records the graph size.\n\n"
            "## Sources\n- mem_session_old\n\n"
            "## Evidence\n- T1: \"Exported 2272 nodes, 532 edges\""
        ),
        node_type=NodeType.PERSISTENT,
        action="create",
        okf_type="fact",
        sources=["mem_session_old"],
    )
    old_id = old["node"]["id"]
    monkeypatch.setattr(
        service,
        "_decide_memory_write_payload",
        lambda **_kwargs: {
            "decision": {
                "action": "supersede",
                "target_node_ids": [old_id],
                "reasoning": "replace with a newer export summary",
            },
            "evidence": {"candidate_count": 1, "sampling_provider": "fake"},
        },
    )

    result = service.write_memory(
        title="Updated graph export size",
        content=(
            "## Takeaway\nThe graph export size is summarized by the new report.\n\n"
            "## Details\nThe new report describes the current graph export.\n\n"
            "## Sources\n- mem_session_new"
        ),
        node_type=NodeType.PERSISTENT,
        okf_type="fact",
        sources=["mem_session_new"],
        route="distiller",
        downgrade_supersede=False,
    )

    assert result["decision"]["action"] == "complement"
    assert result["downgrade"]["reason"].startswith("supersede downgraded to complement")
    assert "2272" in result["downgrade"]["reason"]
    assert "532" in result["downgrade"]["reason"]
    old_after = service.get_node(old_id)["metadata"]
    assert old_after["status"] == "active"
    assert old_after["superseded_by"] is None


def test_concurrent_supersedes_preserve_one_lineage_pointer(tmp_path: Path, monkeypatch) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths)
    old = service.integrate_knowledge(
        title="Gateway probe timeout contract",
        content=(
            "## Takeaway\nKeep the explicit gateway timeout and model.\n\n"
            "## Details\nThe probe timeout is 120 seconds, the model is `GLM-5.2-fp8`, "
            "and the file is `/etc/authz/probe.yaml`.\n\n## Sources\n- mem_session_old"
        ),
        node_type=NodeType.PERSISTENT,
        action="create",
        okf_type="fact",
        sources=["mem_session_old"],
    )
    old_id = old["node"]["id"]
    barrier = threading.Barrier(2)

    def decide(**_kwargs):
        barrier.wait(timeout=10)
        return {
            "decision": {
                "action": "supersede",
                "target_node_ids": [old_id],
                "reasoning": "replace with the updated policy",
            },
            "evidence": {"candidate_count": 1, "sampling_provider": "fake"},
        }

    monkeypatch.setattr(service, "_decide_memory_write_payload", decide)

    def write_variant(variant: str):
        return service.write_memory(
            title=f"Updated gateway probe timeout policy {variant}",
            content=(
                "## Takeaway\nUse the shared gateway probe policy.\n\n"
                "## Details\nThe probe timeout is 120 seconds, the model is `GLM-5.2-fp8`, "
                "and the file is `/etc/authz/probe.yaml`.\n\n"
                f"Variant: {variant}.\n\n## Sources\n- mem_session_{variant}"
            ),
            node_type=NodeType.PERSISTENT,
            okf_type="fact",
            sources=[f"mem_session_{variant}"],
            route="distiller",
            downgrade_supersede=False,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write_variant, ("alpha", "beta")))

    decisions = [result["decision"]["action"] for result in results]
    assert sorted(decisions) == ["complement", "supersede"]
    superseding = [
        result["execution"]["result"]["node"]["id"]
        for result in results
        if result["decision"]["action"] == "supersede"
    ]
    old_metadata = service.get_node(old_id)["metadata"]
    assert old_metadata["status"] == "superseded"
    assert old_metadata["superseded_by"] == superseding[0]
    assert len(superseding) == 1


def test_distiller_final_body_is_not_persisted_after_failed_repair(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    class InvalidRepairClient:
        name = "invalid-repair-test"

        def __init__(self) -> None:
            self.repairs = 0

        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, **_kwargs):
            assert "Repair this persistent OKF body" in prompt
            self.repairs += 1
            return {"content": "## Takeaway\nStill invalid.\n\n## Sources\n- mem_session_invalid"}

        def decide_memory_write(self, _request):
            return MemoryWriteSamplingDecision(
                action="create", target_node_ids=(), reasoning="new fact", confidence=0.9
            )

    client = InvalidRepairClient()
    service = SynapseServerService(config, runtime_paths=runtime_paths, sampling_client=client)
    with pytest.raises(SynapseServiceError) as exc_info:
        service.write_memory(
            title="Invalid final OKF body is rejected",
            content="## Takeaway\nInvalid final OKF body is rejected.\n\n## Sources\n- mem_session_invalid",
            node_type=NodeType.PERSISTENT,
            okf_type="fact",
            sources=["mem_session_invalid"],
            route="distiller",
        )
    assert client.repairs == 1
    assert exc_info.value.code == "EXECUTION_FAILED_AFTER_DECISION"
    assert exc_info.value.details["execution_error"]["code"] == "INVALID_OKF_FINAL_BODY"
    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        assert not store.list_nodes({"status": NodeStatus.ACTIVE, "title": "Invalid final OKF body is rejected"})
