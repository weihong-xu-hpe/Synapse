from pathlib import Path

from synapse.models import Node, NodeMetadata
from synapse.server.sampling import (
    MemoryWriteSamplingRequest,
    build_memory_write_sampling_prompt,
    build_triage_prompt,
)


def _memory_write_request() -> MemoryWriteSamplingRequest:
    return MemoryWriteSamplingRequest(
        prompt="",
        title="Persistent policy",
        content="A durable policy draft.",
        node_type="persistent",
        sensitivity="internal",
        query="Persistent policy",
        similarity_threshold=0.3,
        links=(),
        candidates=(),
        candidate_nodes=(),
    )


def _node(node_id: str, content: str) -> Node:
    return Node(
        metadata=NodeMetadata(id=node_id, title=f"Node {node_id}"),
        content=content,
        file_path=Path("active") / f"{node_id}.md",
    )


def test_memory_write_sampling_prompt_requests_okf_for_persistent_memories() -> None:
    prompt = build_memory_write_sampling_prompt(_memory_write_request())

    assert "OKF structure with ## sections" in prompt
    assert "Context, Decision, Consequences" in prompt
    assert "Transient memories may be free-form" in prompt


def test_triage_prompt_prioritizes_cleanup_of_low_structure_notes() -> None:
    prompt = build_triage_prompt((_node("short-note", "A brief note."),))

    assert "content under 100 characters and no ## sections" in prompt
    assert "condensed or archived, not kept" in prompt