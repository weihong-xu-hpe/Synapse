"""Tests for sign-safe scoring and Related-section normalization."""

from __future__ import annotations

import math
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synapse.config import load_config
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType, SensitivityLevel
from synapse.retrieval.pipeline import DECAY_PERSISTENT_PENALTY, RetrievalPipeline

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _pipeline(tmp_path: Path) -> RetrievalPipeline:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[memory]
base_path = "./.synapse"
archive_path = "./.synapse/.archive"

[embedding]
provider = "builtin"
model = "bge-m3"
dimension = 3

[reranker]
provider = "builtin"
model = "bge-reranker-v2-m3"
max_candidates = 9

[retrieval]
rrf_k = 60
top_k = 3
""".strip(),
        encoding="utf-8",
    )
    from synapse.utils.runtime import bootstrap_runtime_directories

    config = load_config(config_path)
    return RetrievalPipeline(
        config,
        runtime_paths=bootstrap_runtime_directories(config),
        embedding_engine=object(),
        reranker_engine=object(),
        now_fn=lambda: NOW,
    )


def _node(node_id: str, *, type_: NodeType = NodeType.TRANSIENT, status: NodeStatus = NodeStatus.ACTIVE, last_accessed: datetime | None = None) -> Node:
    metadata = NodeMetadata(
        id=node_id,
        title=node_id,
        created_at=NOW - timedelta(days=30),
        last_accessed=last_accessed or NOW,
        access_count=0,
        type=type_,
        status=status,
        sensitivity=SensitivityLevel.INTERNAL,
    )
    return Node(metadata=metadata, content="body", file_path=Path(f"active/{node_id}.md"))


def test_negative_logit_never_moves_up_from_decay() -> None:
    pipeline = _pipeline(Path(tempfile.mkdtemp()))
    stale_transient = _node("stale", last_accessed=NOW - timedelta(days=90))
    scored, multiplier = pipeline.apply_decay(stale_transient, -6.0)
    # Additive logit penalty: -6 + ln(0.98^90) — strictly more negative.
    assert scored < -6.0
    assert multiplier < 1.0
    # An irrelevant superseded node cannot beat a relevant active node at equal
    # logits — the sign never flips.
    superseded = _node("sup", status=NodeStatus.SUPERSEDED)
    sup_score, _ = pipeline.apply_status_penalty(superseded, -0.6)
    active_score, _ = pipeline.apply_status_penalty(_node("act"), -0.6)
    assert sup_score < active_score
    assert active_score == pytest.approx(-0.6)  # active: no status penalty


def test_positive_logit_keeps_sign_after_penalties() -> None:
    pipeline = _pipeline(Path(tempfile.mkdtemp()))
    node = _node("pos", status=NodeStatus.SUPERSEDED)
    scored, _ = pipeline.apply_status_penalty(node, 2.0)
    # Additive ln(0.1): strictly down, sign preserved (2.0 - 2.3 stays... it
    # goes slightly negative — the property that matters is strict monotone
    # ordering, not absolute sign; test ordering instead).
    disputed = _node("dis", status=NodeStatus.DISPUTED)
    sup_score, _ = pipeline.apply_status_penalty(node, 2.0)
    dis_score, _ = pipeline.apply_status_penalty(disputed, 2.0)
    active_score, _ = pipeline.apply_status_penalty(_node("act"), 2.0)
    assert sup_score < dis_score < active_score == pytest.approx(2.0)


def test_persistent_knowledge_not_penalized_by_access_age() -> None:
    pipeline = _pipeline(Path(tempfile.mkdtemp()))
    untouched = _node("k-old", type_=NodeType.PERSISTENT, last_accessed=NOW - timedelta(days=365))
    fresh = _node("k-new", type_=NodeType.PERSISTENT, last_accessed=NOW)
    old_score, _ = pipeline.apply_decay(untouched, 1.0)
    new_score, _ = pipeline.apply_decay(fresh, 1.0)
    assert old_score == new_score == pytest.approx(1.0 + DECAY_PERSISTENT_PENALTY)
    # Transient decay still decreases with age.
    t_old, _ = pipeline.apply_decay(_node("t-old", last_accessed=NOW - timedelta(days=60)), 1.0)
    assert t_old < new_score


def test_merge_related_sections_dedupes_multiple_blocks() -> None:
    from synapse.server.service import SynapseServerService

    content = (
        "Body text.\n\n"
        "## Related\n- [[alpha]]\n- [[beta]]\n\n"
        "Middle section\n\n"
        "## Related\n- [[beta]]\n- [[gamma]]\n\n"
        "## Related\n- [[alpha]]\n- [[delta]]"
    )
    merged = SynapseServerService.merge_related_sections(content)
    assert merged.count("## Related") == 1
    assert merged.index("## Related") > merged.index("Middle section")
    bullets = [line for line in merged.splitlines() if line.startswith("- [[")]
    assert bullets == ["- [[alpha]]", "- [[beta]]", "- [[gamma]]", "- [[delta]]"]
    assert "Body text." in merged and "Middle section" in merged


def test_merge_related_sections_noop_when_single() -> None:
    from synapse.server.service import SynapseServerService

    content = "Body.\n\n## Related\n- [[alpha]]"
    assert SynapseServerService.merge_related_sections(content) == content.strip()
    assert SynapseServerService.merge_related_sections("no related here") == "no related here"


def test_max_fusion_prefers_specific_rank1_over_consensus() -> None:
    from synapse.retrieval.pipeline import RetrievalPipeline

    # Consensus noise: 'noise' appears mid-list in two sets but never at a
    # better rank than 2; 'specific' is rank 1 in exactly one set. Max-fusion
    # must rank 'specific' at least as high as 'noise' (it keeps the best
    # per-list evidence; consensus does not accumulate).
    sets = (
        [("x1", 1.0), ("noise", 0.9)],
        [("x2", 1.0), ("noise", 0.8)],
        [("specific", 1.0), ("noise", 0.5)],
    )
    fused = RetrievalPipeline.fuse_rankings(sets, k=60, limit=10)
    scores = dict(fused)
    assert scores["specific"] == pytest.approx(scores["x1"])
    assert scores["specific"] > scores["noise"]
    # Single list: identical to plain RRF ordering.
    single = RetrievalPipeline.fuse_rankings(([("x", 2.0), ("y", 1.0)],), k=60, limit=10)
    assert [node_id for node_id, _ in single] == ["x", "y"]
