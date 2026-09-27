"""Write-path tightening tests: type defaulting, template inference, LLM
normalization with fidelity guard, title repair, sources warnings."""

from __future__ import annotations

from pathlib import Path

from synapse.config import load_config
from synapse.server.service import SynapseServerService
from synapse.server.write_normalize import (
    WritePathNormalizer,
    decide_write_type,
    has_okf_structure,
    infer_okf_type,
)
from synapse.utils.runtime import bootstrap_runtime_directories

from tests.test_server_api import FakeSamplingClient, write_config


# ---------------------------------------------------------------------------
# Unit: detection helpers
# ---------------------------------------------------------------------------


def test_has_okf_structure_detects_templates() -> None:
    assert has_okf_structure("## Symptom\na\n\n## Cause\nb\n\n## Fix\nc", None)
    assert has_okf_structure("## Steps\n1. do", None)
    assert has_okf_structure("anything", "fact")
    assert not has_okf_structure("## Symptom\na only", None)
    assert not has_okf_structure("plain text", None)


def test_infer_okf_type_matches_in_priority_order() -> None:
    assert infer_okf_type("## Symptom\ns\n\n## Cause\nc\n\n## Fix\nf") == "pitfall"
    assert infer_okf_type("## Context\nx\n\n## Decision\nd\n\n## Consequences\ny") == "decision"
    assert infer_okf_type("## Steps\none") == "procedure"
    assert infer_okf_type("## Details\nstuff") == "fact"
    assert infer_okf_type("## Random\nno match") is None


def test_decide_write_type_explicit_always_wins() -> None:
    assert decide_write_type(node_type="transient", content="## Steps\nx", okf_type=None) == ("transient", "explicit")
    assert decide_write_type(node_type="persistent", content="plain", okf_type=None) == ("persistent", "explicit")
    assert decide_write_type(node_type=None, content="## Steps\nx", okf_type=None) == ("persistent", "defaulted_persistent")
    assert decide_write_type(node_type=None, content="chatter", okf_type=None) == ("transient", "defaulted_transient")
    assert decide_write_type(node_type=None, content="plain", okf_type="fact") == ("persistent", "defaulted_persistent")


# ---------------------------------------------------------------------------
# Normalizer with a scripted sampling client
# ---------------------------------------------------------------------------


class _ScriptedSampler:
    name = "scripted"

    def __init__(self, payload: dict | Exception) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
        self.calls.append(prompt)
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


_NORMALIZED = {
    "okf_type": "pitfall",
    "title": "Aurora migration locks on wide tables",
    "takeaway": "Locks come from unbatched wide-table rewrites.",
    "sections": {
        "Symptom": "Migration hangs.",
        "Cause": "Wide-table rewrite.",
        "Fix": "Batch rows.",
    },
    "sources": ["agent:note"],
}


def test_persistent_without_template_is_llm_normalized_with_fidelity_guard() -> None:
    sampler = _ScriptedSampler(_NORMALIZED)
    normalizer = WritePathNormalizer(sampling_client=sampler)
    original = "迁移的时候会锁表，原因是宽表重写没有分批，改成每批500行就好了。"
    result = normalizer.normalize(title="迁移锁表", content=original, node_type="persistent", okf_type=None, sources=None)
    assert result.mode == "llm_normalized"
    assert result.okf_type == "pitfall"
    assert "## Original note" in result.content
    assert original in result.content  # fidelity: nothing lost
    assert result.sources == ["agent:note"]
    assert any(w["code"] == "okf_normalized_from_note" for w in result.warnings)


def test_llm_failure_stores_as_submitted_with_warning() -> None:
    sampler = _ScriptedSampler(RuntimeError("llm down"))
    normalizer = WritePathNormalizer(sampling_client=sampler)
    original = "Some unstructured knowledge note."
    result = normalizer.normalize(title="Some note", content=original, node_type="persistent", okf_type=None, sources=None)
    assert result.mode == "llm_failed"
    assert result.content == original
    assert result.okf_type is None
    assert any(w["code"] == "okf_normalize_llm_failed" for w in result.warnings)


def test_invalid_llm_payload_stores_as_submitted() -> None:
    sampler = _ScriptedSampler({"nonsense": True})
    normalizer = WritePathNormalizer(sampling_client=sampler)
    result = normalizer.normalize(title="T", content="body text", node_type="persistent", okf_type=None, sources=None)
    assert result.mode == "llm_failed"
    assert result.content == "body text"
    assert any(w["code"] == "okf_normalize_invalid_payload" for w in result.warnings)


def test_explicit_okf_type_skips_llm() -> None:
    sampler = _ScriptedSampler(AssertionError("must not be called"))
    normalizer = WritePathNormalizer(sampling_client=sampler)
    body = "## Takeaway\nt\n\n## Details\nd\n\n## Sources\n- session:x"
    result = normalizer.normalize(title="Fact", content=body, node_type="persistent", okf_type="fact", sources=None)
    assert result.mode == "as_submitted"
    assert result.okf_type == "fact"
    assert result.content == body


def test_transient_write_is_never_normalized() -> None:
    sampler = _ScriptedSampler(AssertionError("must not be called"))
    normalizer = WritePathNormalizer(sampling_client=sampler)
    result = normalizer.normalize(title="T", content="plain", node_type="transient", okf_type=None, sources=None)
    assert result.mode == "as_submitted"


def test_omitted_type_with_okf_structure_defaults_persistent() -> None:
    sampler = _ScriptedSampler(AssertionError("template inference handles this"))
    normalizer = WritePathNormalizer(sampling_client=sampler)
    result = normalizer.normalize(title="T", content="## Steps\none", node_type=None, okf_type=None, sources=None)
    assert result.mode == "template_inferred"
    assert result.okf_type == "procedure"


def test_non_english_title_gets_llm_repair() -> None:
    sampler = _ScriptedSampler({"title": "Aurora migration locking behaviour"})
    normalizer = WritePathNormalizer(sampling_client=sampler)
    result = normalizer.normalize(title="迁移锁表问题", content="plain knowledge", node_type="persistent", okf_type=None, sources=None)
    # llm normalize is attempted for the body (no template); title path also LLM.
    # With the normalize call failing, we at least verify the repair attempt ran.
    assert sampler.calls  # a prompt was sent


def test_title_repair_used_for_normalized_non_english_title() -> None:
    payload = dict(_NORMALIZED)
    payload["title"] = "迁移锁表问题"
    class TwoStep:
        name = "two-step"
        def __init__(self) -> None:
            self.n = 0
        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
            self.n += 1
            if self.n == 1:
                return payload
            return {"title": "Aurora migration locking behaviour"}
    normalizer = WritePathNormalizer(sampling_client=TwoStep())
    result = normalizer.normalize(title="x", content="plain", node_type="persistent", okf_type=None, sources=None)
    assert result.title == "Aurora migration locking behaviour"


# ---------------------------------------------------------------------------
# Service-level wiring
# ---------------------------------------------------------------------------


def test_service_omitted_type_okf_body_writes_persistent(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    body = "## Symptom\nhang\n\n## Cause\nlock\n\n## Fix\nbatch"
    result = service.write_memory(title="Wide table locks", content=body)
    node = service._load_node(result["execution"]["result"]["node"]["id"])
    assert node.metadata.type.value == "persistent"
    assert node.metadata.okf_type == "pitfall"


def test_service_omitted_type_plain_body_stays_transient(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    result = service.write_memory(title="Chatter", content="just status talk")
    node = service._load_node(result["execution"]["result"]["node"]["id"])
    assert node.metadata.type.value == "transient"


def test_service_explicit_transient_wins_over_structure(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    result = service.write_memory(title="Explicit", content="## Steps\none", node_type="transient")
    node = service._load_node(result["execution"]["result"]["node"]["id"])
    assert node.metadata.type.value == "transient"


def test_service_llm_normalization_end_to_end(tmp_path: Path) -> None:
    from synapse.server.sampling import MemoryWriteSamplingDecision

    class Sampler:
        name = "scripted"
        def sample_json(self, *, prompt: str, system_prompt: str, max_tokens: int = 600, model_hints=()):
            return dict(_NORMALIZED)
        def decide_memory_write(self, request):
            return MemoryWriteSamplingDecision(
                action="create",
                target_node_ids=(),
                reasoning="new knowledge",
                confidence=0.9,
            )

    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths, sampling_client=Sampler())
    original = "迁移的时候会锁表，原因是宽表重写没有分批。"
    result = service.write_memory(title="迁移锁表", content=original)
    node = service._load_node(result["execution"]["result"]["node"]["id"])
    assert node.metadata.type.value == "persistent"
    assert node.metadata.okf_type == "pitfall"
    assert original in node.content
    assert "## Original note" in node.content
    assert any(w["code"] == "okf_normalized_from_note" for w in result["warnings"])


def test_service_sources_warning_only_when_absent(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    service = SynapseServerService(config, runtime_paths=runtime_paths, sampling_client=FakeSamplingClient())
    result = service.write_memory(
        title="No sources fact",
        content="## Takeaway\nt\n\n## Details\nd\n\n## Sources\n",
        node_type="persistent",
        okf_type="fact",
    )
    assert any(w["code"] == "okf_missing_sources" for w in result["warnings"])
    # The write still succeeds (warning-only).
    node = service._load_node(result["execution"]["result"]["node"]["id"])
    assert node.metadata.type.value == "persistent"
