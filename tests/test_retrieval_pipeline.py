from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synapse.config import load_config
from synapse.models import Node, NodeMetadata, NodeStatus, NodeType, SensitivityLevel
from synapse.retrieval import RetrievalPipeline
from synapse.storage import SQLiteNodeStore
from synapse.utils.runtime import bootstrap_runtime_directories


NOW = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)


class FakeEmbeddingEngine:
    model_name = "fake-embedding"
    dimension = 3

    def embed(self, text: str) -> list[float]:
        if "gateway" in text.casefold():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def is_available(self) -> bool:
        return True


class FakeRerankerEngine:
    model_name = "fake-reranker"

    def rerank(self, query: str, documents: list[str], limit: int | None = None) -> list[tuple[int, float]]:
        del query
        scores: list[tuple[int, float]] = []
        for index, document in enumerate(documents):
            if "API Gateway Design" in document:
                score = 0.90
            elif "Retry Budgets" in document:
                score = 0.89
            elif "Disputed Gateway Note" in document:
                score = 0.88
            elif "Old Gateway Policy" in document:
                score = 0.99
            else:
                score = 0.10
            scores.append((index, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:limit] if limit is not None else scores

    def is_available(self) -> bool:
        return True


def write_config(base_dir: Path) -> Path:
    config_path = base_dir / "config.toml"
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
    return config_path


def make_node(
    *,
    node_id: str,
    title: str,
    content: str,
    file_name: str,
    status: NodeStatus = NodeStatus.ACTIVE,
    last_accessed: datetime | None = None,
) -> Node:
    metadata = NodeMetadata(
        id=node_id,
        title=title,
        created_at=NOW - timedelta(days=30),
        last_accessed=last_accessed or NOW,
        access_count=0,
        type=NodeType.PERSISTENT,
        status=status,
        tags=[title.casefold().replace(" ", "-")],
        sensitivity=SensitivityLevel.INTERNAL,
    )
    return Node(metadata=metadata, content=content, file_path=Path(f"active/{file_name}"))


def test_rrf_formula_combines_rankings_correctly() -> None:
    scores = RetrievalPipeline.compute_rrf(
        (
            [("alpha", 2.0), ("beta", 1.0)],
            [("beta", 0.1), ("alpha", 0.2), ("gamma", 0.3)],
        ),
        k=60,
    )

    assert scores["alpha"] == pytest.approx((1 / 61) + (1 / 62))
    assert scores["beta"] == pytest.approx((1 / 62) + (1 / 61))
    assert scores["gamma"] == pytest.approx(1 / 63)


def test_graph_hop_returns_one_degree_neighbors_from_both_directions(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    alpha = make_node(node_id="alpha", title="Alpha", content="Gateway anchor", file_name="alpha.md")
    beta = make_node(node_id="beta", title="Beta", content="Outgoing", file_name="beta.md")
    gamma = make_node(node_id="gamma", title="Gamma", content="Incoming", file_name="gamma.md")
    delta = make_node(node_id="delta", title="Delta", content="Outgoing", file_name="delta.md")

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        for node in (alpha, beta, gamma, delta):
            store.upsert_node(node, embedding=[1.0, 0.0, 0.0])
        store.upsert_edges(alpha.id, [beta.id, delta.id])
        store.upsert_edges(gamma.id, [alpha.id])

        pipeline = RetrievalPipeline(
            config,
            store=store,
            runtime_paths=runtime_paths,
            embedding_engine=FakeEmbeddingEngine(),
            reranker_engine=FakeRerankerEngine(),
            now_fn=lambda: NOW,
        )

        neighbors = pipeline.graph_hop([alpha.id], max_neighbors=6)

    assert set(neighbors) == {beta.id, gamma.id, delta.id}


def test_rerank_decay_and_status_penalties_are_applied(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    fresh = make_node(
        node_id="fresh",
        title="API Gateway Design",
        content="Gateway query anchor.",
        file_name="fresh.md",
        last_accessed=NOW,
    )
    old_concept = make_node(
        node_id="old",
        title="Retry Budgets",
        content="Neighbor node.",
        file_name="old.md",
        last_accessed=NOW - timedelta(days=10),
    )
    disputed = make_node(
        node_id="disputed",
        title="Disputed Gateway Note",
        content="Disputed gateway note.",
        file_name="disputed.md",
        status=NodeStatus.DISPUTED,
    )
    superseded = make_node(
        node_id="superseded",
        title="Old Gateway Policy",
        content="Historical note.",
        file_name="superseded.md",
        status=NodeStatus.SUPERSEDED,
    )

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        for node in (fresh, old_concept, disputed, superseded):
            store.upsert_node(node, embedding=[1.0, 0.0, 0.0])
        pipeline = RetrievalPipeline(
            config,
            store=store,
            runtime_paths=runtime_paths,
            embedding_engine=FakeEmbeddingEngine(),
            reranker_engine=FakeRerankerEngine(),
            now_fn=lambda: NOW,
        )

        reranked = pipeline.rerank_candidates("gateway", [old_concept, disputed, fresh])
        decayed_score, decay_multiplier = pipeline.apply_decay(old_concept, 1.0)
        disputed_score, disputed_multiplier = pipeline.apply_status_penalty(disputed, 1.0)
        superseded_score, superseded_multiplier = pipeline.apply_status_penalty(superseded, 1.0)

    assert [node.id for node, _ in reranked] == [fresh.id, old_concept.id, disputed.id]
    assert decay_multiplier == pytest.approx(0.98**10)
    assert decayed_score == pytest.approx(0.98**10)
    assert disputed_multiplier == pytest.approx(0.5)
    assert disputed_score == pytest.approx(0.5)
    assert superseded_multiplier == pytest.approx(0.1)
    assert superseded_score == pytest.approx(0.1)


def test_retrieval_pipeline_returns_top_k_marks_disputed_and_updates_access_only_for_results(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    alpha = make_node(
        node_id="alpha",
        title="API Gateway Design",
        content="Gateway architecture and rate limits.",
        file_name="alpha.md",
        last_accessed=NOW,
    )
    beta = make_node(
        node_id="beta",
        title="Disputed Gateway Note",
        content="Gateway note under active dispute.",
        file_name="beta.md",
        status=NodeStatus.DISPUTED,
        last_accessed=NOW,
    )
    gamma = make_node(
        node_id="gamma",
        title="Old Gateway Policy",
        content="Gateway policy retained for history.",
        file_name="gamma.md",
        status=NodeStatus.SUPERSEDED,
        last_accessed=NOW,
    )
    delta = make_node(
        node_id="delta",
        title="Retry Budgets",
        content="Client retry budgets linked from the upstream service design.",
        file_name="delta.md",
        last_accessed=NOW - timedelta(days=10),
    )

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        store.upsert_node(alpha, embedding=[1.0, 0.0, 0.0])
        store.upsert_node(beta, embedding=[0.9, 0.1, 0.0])
        store.upsert_node(gamma, embedding=[1.0, 0.0, 0.0])
        store.upsert_node(delta, embedding=[0.0, 1.0, 0.0])
        store.upsert_edges(alpha.id, [delta.id])

        pipeline = RetrievalPipeline(
            config,
            store=store,
            runtime_paths=runtime_paths,
            embedding_engine=FakeEmbeddingEngine(),
            reranker_engine=FakeRerankerEngine(),
            now_fn=lambda: NOW,
        )

        response = pipeline.search("gateway")

        result_ids = [item.node.id for item in response.results]
        refreshed_alpha = store.get_node(alpha.id)
        refreshed_beta = store.get_node(beta.id)
        refreshed_gamma = store.get_node(gamma.id)
        refreshed_delta = store.get_node(delta.id)

    assert result_ids == [alpha.id, delta.id, beta.id]
    assert any(item.node.id == delta.id for item in response.candidates)
    assert gamma.id not in result_ids
    assert "[DISPUTED] Disputed Gateway Note" in response.context
    assert refreshed_alpha is not None and refreshed_alpha.metadata.access_count == 1
    assert refreshed_beta is not None and refreshed_beta.metadata.access_count == 1
    assert refreshed_delta is not None and refreshed_delta.metadata.access_count == 1
    assert refreshed_gamma is not None and refreshed_gamma.metadata.access_count == 0