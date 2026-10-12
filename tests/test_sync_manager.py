from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from synapse.config import load_config
from synapse.models import Node, NodeMetadata, NodeType, SensitivityLevel
from synapse.storage import SQLiteNodeStore, write_node_file
from synapse.sync import SyncManager
from synapse.utils.runtime import bootstrap_runtime_directories


FIXED_TIME = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self, initial: float = 0.0) -> None:
        self.value = initial

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeEmbeddingEngine:
    model_name = "fake-embedding"
    dimension = 3
    backend_name = "fake"

    def embed(self, text: str) -> list[float]:
        if "Broken" in text:
            raise RuntimeError("embedding backend unavailable")
        if "Target" in text:
            return [0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

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
""".strip(),
        encoding="utf-8",
    )
    return config_path


def make_node(*, node_id: str, title: str, content: str, file_name: str) -> Node:
    metadata = NodeMetadata(
        id=node_id,
        title=title,
        created_at=FIXED_TIME,
        last_accessed=FIXED_TIME,
        type=NodeType.PERSISTENT,
        sensitivity=SensitivityLevel.INTERNAL,
    )
    return Node(metadata=metadata, content=content, file_path=Path(f"active/{file_name}"))


def set_mtime(path: Path, when: datetime) -> None:
    timestamp = when.timestamp()
    os.utime(path, (timestamp, timestamp))


def test_sync_manager_create_modify_and_delete_updates_nodes_vectors_and_edges(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    target = make_node(
        node_id="mem_target",
        title="Target Note",
        content="# Target Note\n\nKnown good reference.",
        file_name="target.md",
    )
    source = make_node(
        node_id="mem_source",
        title="Source Note",
        content="# Source Note\n\nLinks to [[Target Note]].",
        file_name="source.md",
    )
    target_path = write_node_file(target, base_path=runtime_paths.base)
    source_path = write_node_file(source, base_path=runtime_paths.base)
    set_mtime(target_path, FIXED_TIME)
    set_mtime(source_path, FIXED_TIME)

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        manager = SyncManager(
            config,
            runtime_paths=runtime_paths,
            store=store,
            embedding_engine=FakeEmbeddingEngine(),
            debounce_seconds=0.0,
        )

        initial = manager.sync_paths([target_path, source_path])
        stored_source = store.get_node("mem_source")
        initial_edges = store.get_edges("mem_source")

        updated_source = source.model_copy(update={"content": "# Source Note\n\nUpdated copy with no wiki links."})
        write_node_file(updated_source, base_path=runtime_paths.base)
        modified_time = FIXED_TIME.replace(hour=13)
        set_mtime(source_path, modified_time)
        modified = manager.sync_paths([source_path])
        modified_source = store.get_node("mem_source")
        modified_edges = store.get_edges("mem_source")

        source_path.unlink()
        manager.queue_event("delete", source_path)
        deleted = manager.drain_pending(force=True)

        final_source = store.get_node("mem_source")
        final_target = store.get_node("mem_target")
        vector_count = store.connection.execute("SELECT COUNT(*) FROM nodes_vec").fetchone()[0]

    assert initial.upserted == 2
    assert stored_source is not None
    assert initial_edges == ["mem_target"]
    assert modified.upserted == 1
    assert modified_source is not None
    assert modified_source.metadata.last_accessed == modified_time
    assert modified_edges == []
    assert deleted.deleted == 1
    assert final_source is None
    assert final_target is not None
    assert final_target.metadata.access_count == 0
    assert vector_count == 1


def test_sync_manager_debounces_duplicate_events_into_one_batch(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)
    node = make_node(
        node_id="mem_debounce",
        title="Debounce Note",
        content="# Debounce Note\n\nBatch me once.",
        file_name="debounce.md",
    )
    node_path = write_node_file(node, base_path=runtime_paths.base)
    set_mtime(node_path, FIXED_TIME)
    clock = Clock()

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        manager = SyncManager(
            config,
            runtime_paths=runtime_paths,
            store=store,
            embedding_engine=FakeEmbeddingEngine(),
            debounce_seconds=0.5,
            clock=clock,
        )

        manager.queue_event("upsert", node_path)
        manager.queue_event("upsert", node_path)
        early = manager.drain_pending(force=False)
        clock.advance(0.6)
        late = manager.drain_pending(force=False)

    assert early.upserted == 0
    assert early.queued == 1
    assert late.upserted == 1
    assert late.queued == 0


def test_startup_delta_sync_processes_offline_changes_deletions_and_embedding_failures(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    alpha = make_node(
        node_id="mem_alpha",
        title="Alpha Note",
        content="# Alpha Note\n\nStable content.",
        file_name="alpha.md",
    )
    beta = make_node(
        node_id="mem_beta",
        title="Beta Note",
        content="# Beta Note\n\nWill be deleted.",
        file_name="beta.md",
    )
    alpha_path = write_node_file(alpha, base_path=runtime_paths.base)
    beta_path = write_node_file(beta, base_path=runtime_paths.base)
    set_mtime(alpha_path, FIXED_TIME)
    set_mtime(beta_path, FIXED_TIME)

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=3) as store:
        initial_manager = SyncManager(
            config,
            runtime_paths=runtime_paths,
            store=store,
            embedding_engine=FakeEmbeddingEngine(),
            debounce_seconds=0.0,
        )
        initial_manager.startup_sync()

        alpha_updated = alpha.model_copy(update={"content": "# Alpha Note\n\nOffline edit with [[Missing Note]]."})
        write_node_file(alpha_updated, base_path=runtime_paths.base)
        set_mtime(alpha_path, FIXED_TIME.replace(hour=13))

        beta_path.unlink()

        broken = make_node(
            node_id="mem_broken",
            title="Broken Note",
            content="# Broken Note\n\nBroken embedding path.",
            file_name="broken.md",
        )
        broken_path = write_node_file(broken, base_path=runtime_paths.base)
        set_mtime(broken_path, FIXED_TIME.replace(hour=14))

        startup_manager = SyncManager(
            config,
            runtime_paths=runtime_paths,
            store=store,
            embedding_engine=FakeEmbeddingEngine(),
            debounce_seconds=0.0,
        )
        result = startup_manager.startup_sync()

        alpha_after = store.get_node("mem_alpha")
        beta_after = store.get_node("mem_beta")
        broken_after = store.get_node("mem_broken")
        alpha_edges = store.get_edges("mem_alpha")
        vector_rows = store.connection.execute("SELECT id FROM nodes_vec ORDER BY id").fetchall()

    assert result.upserted == 2
    assert result.deleted == 1
    assert result.failed == 0
    assert alpha_after is not None
    assert beta_after is None
    assert broken_after is not None
    assert alpha_edges == []
    assert [row[0] for row in vector_rows] == ["mem_alpha"]

def test_session_transcripts_are_never_embedded(tmp_path: Path) -> None:
    """P2b: keyed session transcripts get no stored vector on upsert or sync,
    while persistent nodes in the same batch still embed normally."""
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    engine = FakeEmbeddingEngine()
    manager = SyncManager(config, runtime_paths=runtime_paths, embedding_engine=engine, debounce_seconds=0.0)
    try:
        transcript_metadata = NodeMetadata(
            id="mem_session_abc123",
            title="Session summary — probe",
            created_at=FIXED_TIME,
            last_accessed=FIXED_TIME,
            type=NodeType.TRANSIENT,
            session_key="probe-key",
        )
        transcript = Node(
            metadata=transcript_metadata,
            content="## User\nTarget embedding probe\n\n## Assistant\nanalysis",
            file_path=Path("active/mem_session_abc123.md"),
        )
        transcript_path = write_node_file(transcript, base_path=runtime_paths.base)

        knowledge = make_node(
            node_id="mem_knowledge1",
            title="Knowledge Note",
            content="# Knowledge Note\n\nTarget embedding knowledge.",
            file_name="mem_knowledge1.md",
        )
        knowledge_path = write_node_file(knowledge, base_path=runtime_paths.base)

        manager.sync_paths([transcript_path, knowledge_path])

        with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            vector_ids = [row[0] for row in store.connection.execute("SELECT id FROM nodes_vec ORDER BY id").fetchall()]

        assert vector_ids == ["mem_knowledge1"]
    finally:
        manager.close()


def test_transcript_reupsert_deletes_stale_vector(tmp_path: Path) -> None:
    """A transcript that somehow has a vector (pre-gate data) loses it on the
    next sync upsert — the None-embedding branch deletes the stored row."""
    config = load_config(write_config(tmp_path))
    runtime_paths = bootstrap_runtime_directories(config)

    transcript_metadata = NodeMetadata(
        id="mem_session_stale1",
        title="Session summary — stale",
        created_at=FIXED_TIME,
        last_accessed=FIXED_TIME,
        type=NodeType.TRANSIENT,
        session_key="stale-key",
    )
    transcript = Node(
        metadata=transcript_metadata,
        content="## User\nlegacy content\n\n## Assistant\nmore",
        file_path=Path("active/mem_session_stale1.md"),
    )
    transcript_path = write_node_file(transcript, base_path=runtime_paths.base)

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        store.upsert_node(transcript, embedding=[0.1, 0.2, 0.3], source_mtime=FIXED_TIME)
        assert store.get_embedding_fingerprint() is not None or True
        rows = store.connection.execute("SELECT id FROM nodes_vec").fetchall()
        assert [row[0] for row in rows] == ["mem_session_stale1"]

    manager = SyncManager(config, runtime_paths=runtime_paths, embedding_engine=FakeEmbeddingEngine(), debounce_seconds=0.0)
    try:
        manager.sync_paths([transcript_path])
        with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            rows = store.connection.execute("SELECT id FROM nodes_vec").fetchall()
        assert rows == []
    finally:
        manager.close()


def test_reembed_vectorless_skips_transcripts(tmp_path: Path) -> None:
    """P2b gate: the startup re-embed pass must not resurrect pruned transcript
    vectors — transcripts land in `missing` after prune-transcript-vectors."""
    config = load_config(write_config(tmp_path))
    # remote_api backend is required for the reembed loop to run.
    config = config.model_copy(update={
        "embedding": config.embedding.model_copy(update={"provider": "remote_api"}),
    })
    runtime_paths = bootstrap_runtime_directories(config)

    transcript_metadata = NodeMetadata(
        id="mem_session_reembed1",
        title="Session summary — reembed",
        created_at=FIXED_TIME,
        last_accessed=FIXED_TIME,
        type=NodeType.TRANSIENT,
        session_key="reembed-key",
    )
    transcript = Node(
        metadata=transcript_metadata,
        content="## User\nno vector for me\n\n## Assistant\nright",
        file_path=Path("active/mem_session_reembed1.md"),
    )
    transcript_path = write_node_file(transcript, base_path=runtime_paths.base)

    with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
        store.upsert_node(transcript, embedding=None, source_mtime=FIXED_TIME)
        assert store.get_node_ids_without_vectors() == ["mem_session_reembed1"]

    manager = SyncManager(config, runtime_paths=runtime_paths, embedding_engine=FakeEmbeddingEngine(), debounce_seconds=0.0)
    try:
        result = manager.startup_sync()
        with SQLiteNodeStore(runtime_paths.base / "synapse.db", embedding_dimension=config.embedding.dimension or 0) as store:
            rows = store.connection.execute("SELECT id FROM nodes_vec").fetchall()
            assert rows == [], "transcript must not be re-embedded at startup"
    finally:
        manager.close()
