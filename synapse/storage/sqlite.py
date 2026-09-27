"""SQLite-backed derived index for Synapse Markdown nodes."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence

from synapse.models import Node, NodeMetadata, NodeStatus
from synapse.storage.markdown import extract_wiki_links


SCHEMA_VERSION = 7
FALLBACK_VECTOR_BACKEND = "python-fallback"
SQLITE_VEC_BACKEND = "sqlite-vec"
UTC_SUFFIX = "+00:00"


class _ProcessVectorCache:
    """Process-level cache of parsed node vectors, shared across store instances.

    The server constructs a fresh SQLiteNodeStore per request, so an
    instance-level cache never warms up. This cache is keyed by resolved db
    path + embedding dimension and guarded by a lock (requests run in the
    threadpool).

    Invalidation invariant: every mutation of nodes_vec bumps the DB-stored
    generation counter ``schema_meta.vector_generation`` via triggers (FK
    cascade actions fire child triggers too). Readers compare the cached
    generation against one cheap ``SELECT`` — authoritative across processes,
    connections, and cascade deletes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, int], tuple[int, dict[str, array]]] = {}

    @staticmethod
    def _key(db_path: Path, dimension: int) -> tuple[str, int]:
        return (str(db_path.resolve()), dimension)

    def get(self, db_path: Path, dimension: int, generation: int) -> dict[str, array] | None:
        with self._lock:
            entry = self._entries.get(self._key(db_path, dimension))
        if entry is not None and entry[0] == generation:
            return entry[1]
        return None

    def put(self, db_path: Path, dimension: int, generation: int, vectors: dict[str, array]) -> None:
        with self._lock:
            self._entries[self._key(db_path, dimension)] = (generation, vectors)

    def drop(self, db_path: Path, dimension: int) -> None:
        with self._lock:
            self._entries.pop(self._key(db_path, dimension), None)


_VECTOR_CACHE = _ProcessVectorCache()


@dataclass(slots=True, frozen=True)
class DatabaseIntegrityReport:
    """Summary of SQLite integrity and runtime state."""

    ok: bool
    integrity_check_result: str
    wal_mode_enabled: bool
    schema_version: int
    vector_backend: str
    total_nodes: int


@dataclass(slots=True, frozen=True)
class RebuildProgress:
    """Progress update emitted during rebuilds."""

    index: int
    total: int
    node_id: str
    file_path: str


@dataclass(slots=True, frozen=True)
class IndexedFileState:
    """Stored file state used for startup delta synchronization."""

    node_id: str
    file_path: str
    source_mtime: datetime | None


@dataclass(slots=True, frozen=True)
class DreamerRunMetrics:
    """Persisted summary of one Dreamer lifecycle run."""

    started_at: str
    completed_at: str
    duration_ms: int
    batch_size: int
    stale_scanned: int
    superseded_scanned: int
    disputed_scanned: int
    missing_link_pairs_scanned: int
    triage_keep: int
    triage_condense: int
    triage_archive: int
    links_added: int
    conflicts_superseded: int
    conflicts_both_valid: int
    archived: int
    condensed: int
    warnings: int
    sampling_failures: int


@dataclass(slots=True, frozen=True)
class WriteMemoryEventMetrics:
    """Persisted summary of one write_memory request."""
    created_at: str
    node_id: str | None
    node_type: str
    action: str | None
    candidate_count: int
    similarity_threshold: float
    warning_codes: tuple[str, ...]
    sampling_provider: str
    execution_succeeded: bool
    # Attribution: "mcp" (agent tool call), "rest" (REST /api/write unkeyed),
    # "session_upsert" (REST keyed), "distiller", "eval_write", None = legacy.
    route: str | None = None


@dataclass(slots=True, frozen=True)
class DistillerRunMetrics:
    """Persisted summary of one distiller sweep."""

    started_at: str
    completed_at: str
    duration_ms: int
    transcripts_scanned: int
    transcripts_distilled: int
    transcripts_skipped: int
    items_extracted: int
    items_invalid: int
    decider_creates: int
    decider_complements: int
    decider_supersedes: int
    decider_downgrades: int
    llm_failures: int
    archived_transcripts: int


@dataclass(slots=True, frozen=True)
class SearchEventMetrics:
    """Persisted summary of one search request (service-layer usage signal)."""

    ts: str
    source: str
    session_hash: str | None
    query: str
    query_chars: int
    cjk_ratio: float
    top_k: int
    include: str
    latency_ms: float
    lexical_hit_count: int | None
    results: tuple[dict[str, Any], ...]


OKF_META_FIELDS = ("okf_type", "okf_version", "sources", "project", "distilled_hash", "distilled_at", "distilled_node_ids")


def _okf_meta_json(metadata: Any) -> str:
    payload = {name: getattr(metadata, name) for name in OKF_META_FIELDS}
    if not any(payload.values()):
        return "{}"
    return json.dumps({k: v for k, v in payload.items() if v is not None}, ensure_ascii=False)


def _okf_meta_from_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class SQLiteNodeStore:
    """SQLite node store and search primitives for the derived Synapse index."""

    def __init__(self, db_path: str | Path, *, embedding_dimension: int) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dimension = embedding_dimension
        self._connection = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._vector_backend = FALLBACK_VECTOR_BACKEND
        self._ensure_schema()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteNodeStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def vector_backend(self) -> str:
        return self._vector_backend

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        cursor = self._connection.cursor()
        try:
            cursor.execute("BEGIN")
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()
        finally:
            cursor.close()

    def _ensure_schema(self) -> None:
        # Probe without a write transaction first: opening an existing,
        # healthy database must perform zero writes (every request constructs
        # a store, and a write here would hold the single WAL writer lock).
        needs_setup = self._schema_needs_setup()
        if not needs_setup:
            return
        with self.transaction() as connection:
            self._create_schema(connection)
            self._ensure_node_columns(connection)
            self._ensure_observability_columns(connection)
            self._ensure_vector_table(connection)
            self._ensure_fts_triggers(connection)
            self._ensure_vector_generation_triggers(connection)
            stored_version = self._get_meta(connection, "schema_version")
            stored_dimension = self._get_meta(connection, "embedding_dimension")
            stored_backend = self._get_meta(connection, "vector_backend")
            fts_new = self._get_meta(connection, "fts_ready") != str(SCHEMA_VERSION)
            self._set_meta(connection, "schema_version", str(SCHEMA_VERSION))
            self._set_meta(connection, "vector_backend", self._vector_backend)
            self._set_meta(connection, "embedding_dimension", str(self.embedding_dimension))
            self._set_meta(connection, "fts_ready", str(SCHEMA_VERSION))
            # Rebuild the external-content FTS index only when it may be out
            # of sync: fresh database, schema change, dimension/backend change,
            # or the FTS tables were created just now without content.
            dimension_changed = stored_dimension not in {None, str(self.embedding_dimension)}
            backend_changed = stored_backend not in {None, self._vector_backend}
            if fts_new or dimension_changed or backend_changed or stored_version != str(SCHEMA_VERSION):
                connection.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")
                self._rebuild_bigram_index(connection)

    def _rebuild_bigram_index(self, connection: sqlite3.Connection) -> None:
        """Rebuild the CJK-bigram shadow content and its FTS index from nodes."""

        from synapse.storage.lexical import build_bigram_text

        connection.execute("DELETE FROM nodes_bi_content")
        for row in connection.execute("SELECT rowid, title, content, tags FROM nodes").fetchall():
            bi_title, bi_content, bi_tags = build_bigram_text(
                str(row["title"] or ""), str(row["content"] or ""), str(row["tags"] or "[]")
            )
            connection.execute(
                "INSERT INTO nodes_bi_content (rowid, title, content, tags) VALUES (?, ?, ?, ?)",
                (row["rowid"], bi_title, bi_content, bi_tags),
            )
        connection.execute("INSERT INTO nodes_fts_bigram(nodes_fts_bigram) VALUES ('rebuild')")

    def _get_meta(self, connection: sqlite3.Connection, key: str) -> str | None:
        row = connection.execute("SELECT value FROM schema_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def _schema_needs_setup(self) -> bool:
        """Return True when the database needs DDL, meta writes, or an FTS rebuild."""

        try:
            tables = {
                str(row["name"])
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
                ).fetchall()
            }
        except sqlite3.DatabaseError:
            return True
        required = {
            "schema_meta",
            "nodes",
            "nodes_fts",
            "nodes_fts_bigram",
            "nodes_bi_content",
            "edges",
            "dreamer_runs",
            "distiller_runs",
            "write_memory_events",
            "search_events",
            "metrics_snapshots",
            "nodes_vec",
            "nodes_ai",
            "nodes_ad",
            "nodes_au",
            "nodes_vec_ai",
            "nodes_vec_au",
            "nodes_vec_ad",
        }
        if not required.issubset(tables):
            return True
        if "schema_meta" not in tables:
            return True
        stored_version = self.get_meta("schema_version")
        stored_dimension = self.get_meta("embedding_dimension")
        stored_backend = self.get_meta("vector_backend")
        fts_ready = self.get_meta("fts_ready")
        return (
            stored_version != str(SCHEMA_VERSION)
            or stored_dimension != str(self.embedding_dimension)
            or stored_backend != self._vector_backend
            or fts_ready != str(SCHEMA_VERSION)
        )

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                source_mtime TEXT,
                content TEXT NOT NULL,
                type TEXT DEFAULT 'transient',
                status TEXT DEFAULT 'active',
                sensitivity TEXT DEFAULT 'internal',
                supersedes TEXT,
                superseded_by TEXT,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                tags TEXT DEFAULT '[]',
                okf_meta TEXT DEFAULT '{}'
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status)")
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                title,
                content,
                tags,
                content='nodes',
                content_rowid='rowid'
            )
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts_bigram USING fts5(
                title,
                content,
                tags,
                content='nodes_bi_content',
                content_rowid='rowid'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes_bi_content (
                rowid INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS dreamer_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                stale_scanned INTEGER NOT NULL,
                superseded_scanned INTEGER NOT NULL,
                disputed_scanned INTEGER NOT NULL,
                missing_link_pairs_scanned INTEGER NOT NULL,
                triage_keep INTEGER NOT NULL,
                triage_condense INTEGER NOT NULL,
                triage_archive INTEGER NOT NULL,
                links_added INTEGER NOT NULL,
                conflicts_superseded INTEGER NOT NULL,
                conflicts_both_valid INTEGER NOT NULL,
                archived INTEGER NOT NULL,
                condensed INTEGER NOT NULL,
                warnings INTEGER NOT NULL,
                sampling_failures INTEGER NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_dreamer_runs_started_at ON dreamer_runs(started_at)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS write_memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                node_id TEXT,
                node_type TEXT NOT NULL,
                action TEXT,
                candidate_count INTEGER NOT NULL,
                similarity_threshold REAL NOT NULL,
                warning_codes TEXT NOT NULL,
                sampling_provider TEXT NOT NULL,
                execution_succeeded INTEGER NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_write_memory_events_created_at ON write_memory_events(created_at)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS distiller_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                transcripts_scanned INTEGER NOT NULL,
                transcripts_distilled INTEGER NOT NULL,
                transcripts_skipped INTEGER NOT NULL,
                items_extracted INTEGER NOT NULL,
                items_invalid INTEGER NOT NULL,
                decider_creates INTEGER NOT NULL,
                decider_complements INTEGER NOT NULL,
                decider_supersedes INTEGER NOT NULL,
                decider_downgrades INTEGER NOT NULL,
                llm_failures INTEGER NOT NULL,
                archived_transcripts INTEGER NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_distiller_runs_started_at ON distiller_runs(started_at)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS search_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT NOT NULL,
                session_hash TEXT,
                query TEXT NOT NULL,
                query_chars INTEGER NOT NULL,
                cjk_ratio REAL NOT NULL,
                top_k INTEGER NOT NULL,
                include TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                lexical_hit_count INTEGER,
                results TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_search_events_ts ON search_events(ts)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_search_events_source ON search_events(source)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics_snapshots (
                date TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )

    def _ensure_observability_columns(self, connection: sqlite3.Connection) -> None:
        """Migration helpers for observability tables (schema v7 additions)."""

        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(write_memory_events)").fetchall()
        }
        if "route" not in columns:
            # Existing rows predate route attribution; NULL = legacy/unknown.
            connection.execute("ALTER TABLE write_memory_events ADD COLUMN route TEXT")

    def _ensure_node_columns(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "source_mtime" not in columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN source_mtime TEXT")
        if "sensitivity" not in columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN sensitivity TEXT DEFAULT 'internal'")
        if "okf_meta" not in columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN okf_meta TEXT DEFAULT '{}'")

    def _ensure_vector_table(self, connection: sqlite3.Connection) -> None:
        # sqlite-vec is intentionally not required for Phase 3. The current session
        # uses a safe Python cosine fallback backed by a normal SQLite table.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes_vec (
                id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                FOREIGN KEY (id) REFERENCES nodes(id) ON DELETE CASCADE
            )
            """
        )
        self._vector_backend = FALLBACK_VECTOR_BACKEND

    def _ensure_fts_triggers(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, title, content, tags)
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, title, content, tags)
                VALUES ('delete', old.rowid, old.title, old.content, old.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, title, content, tags)
                VALUES ('delete', old.rowid, old.title, old.content, old.tags);
                INSERT INTO nodes_fts(rowid, title, content, tags)
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;
            """
        )
        # CJK-bigram shadow index mirrors nodes_fts with bigram-expanded text
        # (synapse/storage/lexical.py). The external-content FTS5 table reads
        # from nodes_bi_content, which the store's Python write paths
        # (upsert_node / delete_node) maintain with bigram-transformed text;
        # these triggers apply the shadow content to the FTS index on every
        # mutation of nodes_bi_content.
        connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS nodes_bi_ai AFTER INSERT ON nodes_bi_content BEGIN
                INSERT INTO nodes_fts_bigram(rowid, title, content, tags)
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_bi_ad AFTER DELETE ON nodes_bi_content BEGIN
                INSERT INTO nodes_fts_bigram(nodes_fts_bigram, rowid, title, content, tags)
                VALUES ('delete', old.rowid, old.title, old.content, old.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_bi_au AFTER UPDATE ON nodes_bi_content BEGIN
                INSERT INTO nodes_fts_bigram(nodes_fts_bigram, rowid, title, content, tags)
                VALUES ('delete', old.rowid, old.title, old.content, old.tags);
                INSERT INTO nodes_fts_bigram(rowid, title, content, tags)
                VALUES (new.rowid, new.title, new.content, new.tags);
            END;
            """
        )

    def _ensure_vector_generation_triggers(self, connection: sqlite3.Connection) -> None:
        """Install triggers that bump schema_meta.vector_generation on any
        nodes_vec mutation (including FK ON DELETE CASCADE deletes from nodes,
        which fire these child triggers). The process-level vector cache
        validates its entries against this counter.
        """

        connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS nodes_vec_ai AFTER INSERT ON nodes_vec BEGIN
                INSERT INTO schema_meta(key, value) VALUES ('vector_generation', '1')
                ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1;
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_vec_au AFTER UPDATE ON nodes_vec BEGIN
                INSERT INTO schema_meta(key, value) VALUES ('vector_generation', '1')
                ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1;
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_vec_ad AFTER DELETE ON nodes_vec BEGIN
                INSERT INTO schema_meta(key, value) VALUES ('vector_generation', '1')
                ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1;
            END;
            """
        )

    def _set_meta(self, connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            """
            INSERT INTO schema_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM schema_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def set_embedding_fingerprint(self, fingerprint: str) -> None:
        with self.transaction() as connection:
            self._set_meta(connection, "embedding_fingerprint", fingerprint)

    def get_embedding_fingerprint(self) -> str | None:
        return self.get_meta("embedding_fingerprint")

    def get_wal_mode(self) -> str:
        row = self._connection.execute("PRAGMA journal_mode").fetchone()
        if row is None:
            return "unknown"
        return str(row[0]).lower()

    def is_wal_enabled(self) -> bool:
        return self.get_wal_mode() == "wal"

    def upsert_node(
        self,
        node: Node,
        embedding: list[float] | None = None,
        *,
        source_mtime: datetime | str | None = None,
    ) -> None:
        from synapse.storage.lexical import build_bigram_text

        payload = self._node_to_db_values(node, source_mtime=source_mtime)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO nodes (
                    id, title, file_path, source_mtime, content, type, status,
                    sensitivity, supersedes, superseded_by, created_at, last_accessed, access_count, tags, okf_meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    file_path = excluded.file_path,
                    source_mtime = excluded.source_mtime,
                    content = excluded.content,
                    type = excluded.type,
                    status = excluded.status,
                    sensitivity = excluded.sensitivity,
                    supersedes = excluded.supersedes,
                    superseded_by = excluded.superseded_by,
                    created_at = excluded.created_at,
                    last_accessed = excluded.last_accessed,
                    access_count = excluded.access_count,
                    tags = excluded.tags,
                    okf_meta = excluded.okf_meta
                """,
                payload,
            )
            # Keep the CJK-bigram shadow content in sync (bigram-expanded text;
            # the nodes_bi_* triggers copy it into the FTS5 shadow index).
            bi_title, bi_content, bi_tags = build_bigram_text(str(payload[1]), str(payload[4]), str(payload[13]))
            connection.execute(
                "DELETE FROM nodes_bi_content WHERE rowid = (SELECT rowid FROM nodes WHERE id = ?)",
                (node.id,),
            )
            connection.execute(
                """
                INSERT INTO nodes_bi_content (rowid, title, content, tags)
                SELECT rowid, ?, ?, ? FROM nodes WHERE id = ?
                """,
                (bi_title, bi_content, bi_tags, node.id),
            )
            if embedding is not None:
                self._upsert_embedding(connection, node.id, embedding)

    def upsert_embedding(self, node_id: str, embedding: list[float]) -> None:
        with self.transaction() as connection:
            self._upsert_embedding(connection, node_id, embedding)

    def _upsert_embedding(self, connection: sqlite3.Connection, node_id: str, embedding: list[float]) -> None:
        self._invalidate_vector_cache()
        normalized = _normalize_embedding(embedding, self.embedding_dimension)
        connection.execute(
            """
            INSERT INTO nodes_vec(id, embedding, dimension)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                embedding = excluded.embedding,
                dimension = excluded.dimension
            """,
            (node_id, json.dumps(normalized), len(normalized)),
        )

    def get_node(self, node_id: str) -> Node | None:
        row = self._connection.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def get_nodes(self, node_ids: Iterable[str]) -> list[Node]:
        requested_ids = list(dict.fromkeys(node_ids))
        if not requested_ids:
            return []
        placeholders = ", ".join("?" for _ in requested_ids)
        rows = self._connection.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})",
            requested_ids,
        ).fetchall()
        nodes = {node.id: node for node in (self._row_to_node(row) for row in rows)}
        return [nodes[node_id] for node_id in requested_ids if node_id in nodes]

    def get_node_by_file_path(self, file_path: str | Path) -> Node | None:
        normalized_path = Path(file_path).as_posix()
        row = self._connection.execute(
            "SELECT * FROM nodes WHERE file_path = ?",
            (normalized_path,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def delete_node(self, node_id: str) -> None:
        with self.transaction() as connection:
            row = connection.execute("SELECT rowid FROM nodes WHERE id = ?", (node_id,)).fetchone()
            connection.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            if row is not None:
                connection.execute("DELETE FROM nodes_bi_content WHERE rowid = ?", (row["rowid"],))

    def delete_embedding(self, node_id: str) -> None:
        with self.transaction() as connection:
            self._invalidate_vector_cache()
            connection.execute("DELETE FROM nodes_vec WHERE id = ?", (node_id,))

    def update_access(self, node_ids: list[str]) -> None:
        if not node_ids:
            return
        timestamp = _utc_now_isoformat()
        with self.transaction() as connection:
            connection.executemany(
                """
                UPDATE nodes
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                [(timestamp, node_id) for node_id in node_ids],
            )

    def update_status(
        self,
        node_id: str,
        status: NodeStatus | str,
        superseded_by: str | None = None,
    ) -> None:
        normalized_status = status.value if isinstance(status, NodeStatus) else str(status)
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE nodes
                SET status = ?, superseded_by = ?
                WHERE id = ?
                """,
                (normalized_status, superseded_by, node_id),
            )

    def upsert_edges(self, source_id: str, target_ids: list[str]) -> None:
        filtered_targets = sorted({target_id for target_id in target_ids if target_id and target_id != source_id})
        with self.transaction() as connection:
            connection.execute("DELETE FROM edges WHERE source_id = ?", (source_id,))
            connection.executemany(
                "INSERT OR IGNORE INTO edges(source_id, target_id) VALUES (?, ?)",
                [(source_id, target_id) for target_id in filtered_targets],
            )

    def get_edges(self, node_id: str, direction: Literal["outgoing", "incoming"] = "outgoing") -> list[str]:
        if direction == "outgoing":
            rows = self._connection.execute(
                "SELECT target_id FROM edges WHERE source_id = ? ORDER BY target_id",
                (node_id,),
            ).fetchall()
            return [str(row["target_id"]) for row in rows]
        if direction == "incoming":
            rows = self._connection.execute(
                "SELECT source_id FROM edges WHERE target_id = ? ORDER BY source_id",
                (node_id,),
            ).fetchall()
            return [str(row["source_id"]) for row in rows]
        raise ValueError("direction must be 'outgoing' or 'incoming'")

    def vector_search(self, embedding: list[float], limit: int = 10) -> list[tuple[str, float]]:
        if not embedding:
            return []
        normalized_query = _normalize_embedding(embedding, self.embedding_dimension)
        vectors = self._load_vectors_cached()
        scored: list[tuple[str, float]] = []
        for node_id, candidate_vector in vectors.items():
            distance = _cosine_distance(normalized_query, list(candidate_vector))
            scored.append((node_id, distance))
        scored.sort(key=lambda item: (item[1], item[0]))
        return scored[:limit]

    def _current_vector_generation(self) -> int:
        row = self._connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'vector_generation'"
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return 0

    def _load_vectors_cached(self) -> dict[str, array]:
        """Return {node_id: packed vector} shared process-wide.

        Invalidation invariant: every nodes_vec mutation (insert/update/delete,
        including FK ON DELETE CASCADE deletes from nodes — cascade actions fire
        child triggers) bumps schema_meta.vector_generation via triggers. Each
        read validates its cached generation with one cheap SELECT, which is
        authoritative across processes and connections. Vectors are stored as
        pre-normalized array('d') (8 bytes/float → ~16 MB for 2k×1024).
        """

        generation = self._current_vector_generation()
        cached = _VECTOR_CACHE.get(self.db_path, self.embedding_dimension, generation)
        if cached is not None:
            return cached
        rows = self._connection.execute(
            "SELECT id, embedding FROM nodes_vec WHERE dimension = ?",
            (self.embedding_dimension,),
        ).fetchall()
        vectors: dict[str, array] = {}
        for row in rows:
            try:
                vectors[str(row["id"])] = array("d", json.loads(str(row["embedding"])))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        _VECTOR_CACHE.put(self.db_path, self.embedding_dimension, generation, vectors)
        return vectors

    def _invalidate_vector_cache(self) -> None:
        # The DB-stored generation is the source of truth; dropping the process
        # entry only avoids a stale hit if a same-tick write hasn't committed.
        _VECTOR_CACHE.drop(self.db_path, self.embedding_dimension)


    def list_nodes(self, filters: dict[str, Any] | None = None) -> list[Node]:
        filters = filters or {}
        clauses: list[str] = []
        values: list[Any] = []
        for key in ("tier", "status", "type", "title"):
            if key in filters and filters[key] is not None:
                clauses.append(f"{key} = ?")
                values.append(filters[key].value if hasattr(filters[key], "value") else filters[key])

        query = "SELECT * FROM nodes"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC, id ASC"
        rows = self._connection.execute(query, values).fetchall()
        return [self._row_to_node(row) for row in rows]

    def count_nodes(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM nodes").fetchone()
        return int(row["count"]) if row is not None else 0

    def count_nodes_by_status(self) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT status, COUNT(*) AS count FROM nodes GROUP BY status ORDER BY status"
        ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        counts.setdefault(NodeStatus.ACTIVE.value, 0)
        counts.setdefault(NodeStatus.SUPERSEDED.value, 0)
        counts.setdefault(NodeStatus.DISPUTED.value, 0)
        return counts

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Escape a raw query string for safe use in an FTS5 MATCH expression.

        Wraps each whitespace-delimited token in double-quotes so that
        special FTS5 operators (/, *, -, etc.) are treated as literals.
        Returns an empty string when no usable tokens remain.
        """
        import re
        # Strip characters that are problematic even inside FTS5 double-quotes
        tokens = query.split()
        safe_tokens: list[str] = []
        for tok in tokens:
            # Remove embedded double-quotes to prevent breaking out of quoting
            tok = tok.replace('"', "")
            # Keep only tokens that have at least one alphanumeric char
            if re.search(r"\w", tok):
                safe_tokens.append(f'"{tok}"')
        return " ".join(safe_tokens)

    def fts_search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Lexical leg: OR semantics over the CJK-bigram shadow index.

        The primary ``nodes_fts`` table keeps unicode61 tokens (English words,
        code identifiers); ``nodes_fts_bigram`` mirrors every node with CJK
        runs expanded to overlapping bigrams so Chinese queries segment
        correctly. Both legs are queried with OR semantics (AND kills
        multi-word queries) and ranks merged per-leg by bm25.
        """

        from synapse.storage.lexical import build_or_query

        sanitized = build_or_query(query)
        if not sanitized:
            return []
        rows = self._connection.execute(
            """
            SELECT nodes.id AS node_id, -bm25(nodes_fts) AS score
            FROM nodes_fts
            JOIN nodes ON nodes.rowid = nodes_fts.rowid
            WHERE nodes_fts MATCH ?
            ORDER BY score DESC, nodes.id ASC
            LIMIT ?
            """,
            (sanitized, limit),
        ).fetchall()
        bigram_rows = self._connection.execute(
            """
            SELECT nodes.id AS node_id, -bm25(nodes_fts_bigram) AS score
            FROM nodes_fts_bigram
            JOIN nodes ON nodes.rowid = nodes_fts_bigram.rowid
            WHERE nodes_fts_bigram MATCH ?
            ORDER BY score DESC, nodes.id ASC
            LIMIT ?
            """,
            (sanitized, limit),
        ).fetchall()
        # bm25 scores are not comparable across tables; fuse by rank (RRF-style)
        # per leg, then merge with the better rank kept for ordering.
        merged: dict[str, float] = {}
        for rows_leg, weight in ((rows, 1.0), (bigram_rows, 1.0)):
            for rank, row in enumerate(rows_leg, start=1):
                contribution = weight / (60.0 + rank)
                merged[row["node_id"]] = merged.get(row["node_id"], 0.0) + contribution
        ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:limit]
        # Return pseudo-bm25 positive scores so callers keep a monotone signal.
        return [(node_id, score * 100.0) for node_id, score in ranked]


    def get_neighbors(self, node_ids: list[str], depth: int = 1) -> list[str]:
        if depth < 1 or not node_ids:
            return []
        frontier = set(node_ids)
        visited = set(node_ids)
        neighbors: set[str] = set()
        for _ in range(depth):
            if not frontier:
                break
            placeholders = ", ".join("?" for _ in frontier)
            rows = self._connection.execute(
                f"SELECT DISTINCT target_id FROM edges WHERE source_id IN ({placeholders})",
                tuple(frontier),
            ).fetchall()
            next_frontier = {str(row["target_id"]) for row in rows} - visited
            neighbors.update(next_frontier)
            visited.update(next_frontier)
            frontier = next_frontier
        return sorted(neighbors)

    def get_linked_neighbors(self, node_ids: list[str], *, limit: int = 6) -> list[str]:
        if not node_ids or limit <= 0:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        rows = self._connection.execute(
            f"""
            SELECT neighbor_id, COUNT(*) AS weight
            FROM (
                SELECT target_id AS neighbor_id
                FROM edges
                WHERE source_id IN ({placeholders})
                UNION ALL
                SELECT source_id AS neighbor_id
                FROM edges
                WHERE target_id IN ({placeholders})
            )
            WHERE neighbor_id NOT IN ({placeholders})
            GROUP BY neighbor_id
            ORDER BY weight DESC, neighbor_id ASC
            LIMIT ?
            """,
            (*node_ids, *node_ids, *node_ids, limit),
        ).fetchall()
        return [str(row["neighbor_id"]) for row in rows]

    def get_indexed_file_states(self) -> dict[str, IndexedFileState]:
        rows = self._connection.execute(
            "SELECT id, file_path, source_mtime FROM nodes ORDER BY file_path ASC"
        ).fetchall()
        return {
            str(row["file_path"]): IndexedFileState(
                node_id=str(row["id"]),
                file_path=str(row["file_path"]),
                source_mtime=_parse_iso_datetime(row["source_mtime"]),
            )
            for row in rows
        }

    def get_node_ids_without_vectors(self) -> list[str]:
        """Return ids of nodes that have no persisted embedding (degraded writes)."""

        rows = self._connection.execute(
            """
            SELECT n.id
            FROM nodes AS n
            LEFT JOIN nodes_vec AS v ON v.id = n.id
            WHERE v.id IS NULL
            ORDER BY n.id ASC
            """
        ).fetchall()
        return [str(row["id"]) for row in rows]

    def find_orphan_candidates(
        self,
        days_threshold: int,
    ) -> list[Node]:
        cutoff = _utc_cutoff(days_threshold)
        query = """
            SELECT n.*
            FROM nodes AS n
            LEFT JOIN edges AS incoming ON incoming.target_id = n.id
            LEFT JOIN edges AS outgoing ON outgoing.source_id = n.id
            WHERE n.status = ? AND n.last_accessed <= ?
            GROUP BY n.id
            HAVING COUNT(DISTINCT incoming.source_id) = 0
               AND COUNT(DISTINCT outgoing.target_id) = 0
            ORDER BY n.last_accessed ASC, n.id ASC
        """
        rows = self._connection.execute(query, [NodeStatus.ACTIVE.value, cutoff]).fetchall()
        return [self._row_to_node(row) for row in rows]

    def find_missing_link_pairs(
        self,
        cosine_threshold: float = 0.75,
        recency_days: int = 30,
    ) -> list[tuple[Node, Node]]:
        """Find active node pairs with high cosine similarity but no edge between them."""
        cutoff = _utc_cutoff(recency_days)
        # Get active nodes accessed since cutoff that have embeddings
        rows = self._connection.execute(
            """
            SELECT n.id
            FROM nodes AS n
            JOIN nodes_vec AS v ON v.id = n.id
            WHERE n.status = ? AND n.last_accessed > ?
            ORDER BY n.id ASC
            """,
            [NodeStatus.ACTIVE.value, cutoff],
        ).fetchall()
        candidate_ids = [str(row["id"]) for row in rows]

        if len(candidate_ids) < 2:
            return []

        # Load embeddings for all candidates
        embeddings: dict[str, list[float]] = {}
        for cid in candidate_ids:
            vec_row = self._connection.execute(
                "SELECT embedding FROM nodes_vec WHERE id = ?", [cid]
            ).fetchone()
            if vec_row is not None:
                try:
                    embeddings[cid] = json.loads(str(vec_row["embedding"]))
                except (json.JSONDecodeError, TypeError):
                    continue

        embed_ids = list(embeddings.keys())
        if len(embed_ids) < 2:
            return []

        pairs: list[tuple[Node, Node]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, id_a in enumerate(embed_ids):
            vec_a = embeddings[id_a]
            for id_b in embed_ids[i + 1 :]:
                vec_b = embeddings[id_b]
                try:
                    distance = _cosine_distance(vec_a, vec_b)
                except ValueError:
                    continue
                if distance > (1.0 - cosine_threshold):
                    continue

                pair_key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                if pair_key in seen_pairs:
                    continue

                # Check if edge already exists between them
                edge_row = self._connection.execute(
                    """
                    SELECT COUNT(*) AS cnt FROM edges
                    WHERE (source_id = ? AND target_id = ?)
                       OR (source_id = ? AND target_id = ?)
                    """,
                    [id_a, id_b, id_b, id_a],
                ).fetchone()
                if edge_row and int(edge_row["cnt"]) > 0:
                    continue

                node_a = self.get_node(id_a)
                node_b = self.get_node(id_b)
                if node_a is None or node_b is None:
                    continue

                seen_pairs.add(pair_key)
                pairs.append((node_a, node_b))

        return pairs

    def find_superseded_for_archival(self, days_threshold: int = 7) -> list[Node]:
        cutoff = _utc_cutoff(days_threshold)
        rows = self._connection.execute(
            """
            SELECT *
            FROM nodes
            WHERE status = ?
              AND last_accessed <= ?
            ORDER BY last_accessed ASC, id ASC
            """,
            (NodeStatus.SUPERSEDED.value, cutoff),
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def count_disputed_nodes(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM nodes WHERE status = ?",
            (NodeStatus.DISPUTED.value,),
        ).fetchone()
        return int(row["count"]) if row is not None else 0

    def count_disputed_pairs(self) -> int:
        """Count disputed node pairs using the same pairing semantics as Dreamer."""
        disputed = self.find_by_status(NodeStatus.DISPUTED)
        disputed_ids = {node.id for node in disputed}
        seen: set[str] = set()
        count = 0
        for node in disputed:
            if node.id in seen or not node.metadata.superseded_by:
                continue
            partner_id = node.metadata.superseded_by
            if partner_id in disputed_ids and partner_id not in seen:
                count += 1
                seen.add(node.id)
                seen.add(partner_id)
        return count

    def find_by_status(self, status: NodeStatus) -> list[Node]:
        """Return all nodes matching *status*, ordered by last_accessed then id."""
        rows = self._connection.execute(
            "SELECT * FROM nodes WHERE status = ? ORDER BY last_accessed ASC, id ASC",
            [status.value],
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def touch_node(self, node_id: str, *, now: datetime | None = None) -> bool:
        """Update last_accessed for a node. Returns True if the node was found."""
        accessed = (now or datetime.now(UTC)).isoformat().replace(UTC_SUFFIX, "Z")
        cursor = self._connection.execute(
            "UPDATE nodes SET last_accessed = ? WHERE id = ?",
            [accessed, node_id],
        )
        self._connection.commit()
        return cursor.rowcount > 0

    def missing_link_similarity_histogram(
        self,
        *,
        thresholds: Iterable[float],
        recency_days: int,
    ) -> dict[str, int]:
        """Count unlinked recent active node pairs above fixed cosine-similarity buckets."""
        buckets = sorted({round(float(threshold), 2) for threshold in thresholds})
        counts = {f"{threshold:.2f}": 0 for threshold in buckets}
        if not buckets:
            return counts

        embeddings = self._recent_active_embeddings(recency_days)
        ids = list(embeddings)
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1 :]:
                if self._edge_exists_between(id_a, id_b):
                    continue
                similarity = _safe_cosine_similarity(embeddings[id_a], embeddings[id_b])
                if similarity is None:
                    continue
                _increment_similarity_buckets(counts, buckets, similarity)
        return counts

    def _recent_active_embeddings(self, recency_days: int) -> dict[str, list[float]]:
        cutoff = _utc_cutoff(recency_days)
        rows = self._connection.execute(
            """
            SELECT n.id, v.embedding
            FROM nodes AS n
            JOIN nodes_vec AS v ON v.id = n.id
            WHERE n.status = ? AND n.last_accessed > ?
            ORDER BY n.id ASC
            """,
            [NodeStatus.ACTIVE.value, cutoff],
        ).fetchall()
        embeddings: dict[str, list[float]] = {}
        for row in rows:
            try:
                embeddings[str(row["id"])] = json.loads(str(row["embedding"]))
            except (json.JSONDecodeError, TypeError):
                continue
        return embeddings

    def _edge_exists_between(self, id_a: str, id_b: str) -> bool:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS cnt FROM edges
            WHERE (source_id = ? AND target_id = ?)
               OR (source_id = ? AND target_id = ?)
            """,
            [id_a, id_b, id_b, id_a],
        ).fetchone()
        return bool(row and int(row["cnt"]) > 0)

    def record_dreamer_run(self, metrics: DreamerRunMetrics) -> None:
        """Persist a low-cardinality Dreamer run summary for long-term stats."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO dreamer_runs (
                    started_at, completed_at, duration_ms, batch_size,
                    stale_scanned, superseded_scanned, disputed_scanned, missing_link_pairs_scanned,
                    triage_keep, triage_condense, triage_archive, links_added,
                    conflicts_superseded, conflicts_both_valid, archived, condensed,
                    warnings, sampling_failures
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.started_at,
                    metrics.completed_at,
                    metrics.duration_ms,
                    metrics.batch_size,
                    metrics.stale_scanned,
                    metrics.superseded_scanned,
                    metrics.disputed_scanned,
                    metrics.missing_link_pairs_scanned,
                    metrics.triage_keep,
                    metrics.triage_condense,
                    metrics.triage_archive,
                    metrics.links_added,
                    metrics.conflicts_superseded,
                    metrics.conflicts_both_valid,
                    metrics.archived,
                    metrics.condensed,
                    metrics.warnings,
                    metrics.sampling_failures,
                ),
            )

    def record_distiller_run(self, metrics: DistillerRunMetrics) -> None:
        """Persist a low-cardinality distiller sweep summary for long-term stats."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO distiller_runs (
                    started_at, completed_at, duration_ms,
                    transcripts_scanned, transcripts_distilled, transcripts_skipped,
                    items_extracted, items_invalid,
                    decider_creates, decider_complements, decider_supersedes, decider_downgrades,
                    llm_failures, archived_transcripts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.started_at,
                    metrics.completed_at,
                    metrics.duration_ms,
                    metrics.transcripts_scanned,
                    metrics.transcripts_distilled,
                    metrics.transcripts_skipped,
                    metrics.items_extracted,
                    metrics.items_invalid,
                    metrics.decider_creates,
                    metrics.decider_complements,
                    metrics.decider_supersedes,
                    metrics.decider_downgrades,
                    metrics.llm_failures,
                    metrics.archived_transcripts,
                ),
            )

    def record_write_memory_event(self, metrics: WriteMemoryEventMetrics) -> None:
        """Persist a low-cardinality write_memory summary for long-term stats."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO write_memory_events (
                    created_at, node_id, node_type, action, candidate_count,
                    similarity_threshold, warning_codes, sampling_provider, execution_succeeded, route
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.created_at,
                    metrics.node_id,
                    metrics.node_type,
                    metrics.action,
                    metrics.candidate_count,
                    metrics.similarity_threshold,
                    json.dumps(list(metrics.warning_codes)),
                    metrics.sampling_provider,
                    int(metrics.execution_succeeded),
                    metrics.route,
                ),
            )

    def record_search_event(self, metrics: SearchEventMetrics) -> None:
        """Persist one search request for usage analytics (local private DB)."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO search_events (
                    ts, source, session_hash, query, query_chars, cjk_ratio,
                    top_k, include, latency_ms, lexical_hit_count, results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.ts,
                    metrics.source,
                    metrics.session_hash,
                    metrics.query,
                    metrics.query_chars,
                    metrics.cjk_ratio,
                    metrics.top_k,
                    metrics.include,
                    metrics.latency_ms,
                    metrics.lexical_hit_count,
                    json.dumps(list(metrics.results), ensure_ascii=False),
                ),
            )

    def prune_search_events(self, *, retention_days: int) -> int:
        """Delete search_events older than the retention window; return deleted count."""
        cutoff = _utc_cutoff(retention_days)
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM search_events WHERE ts < ?", (cutoff,))
            return cursor.rowcount

    def record_metrics_snapshot(self, *, date: str, created_at: str, payload: dict[str, Any]) -> bool:
        """Insert today's snapshot row. Returns False when the date row already exists."""
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO metrics_snapshots (date, created_at, payload) VALUES (?, ?, ?)
                """,
                (date, created_at, json.dumps(payload, ensure_ascii=False)),
            )
            return cursor.rowcount > 0

    def get_metrics_snapshots(self, *, since_date: str | None = None) -> list[dict[str, Any]]:
        """Return snapshot rows ({date, created_at, payload}) oldest first."""
        if since_date is None:
            rows = self._connection.execute(
                "SELECT date, created_at, payload FROM metrics_snapshots ORDER BY date"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT date, created_at, payload FROM metrics_snapshots WHERE date >= ? ORDER BY date",
                (since_date,),
            ).fetchall()
        snapshots = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload"]))
            except json.JSONDecodeError:
                continue
            snapshots.append({"date": str(row["date"]), "created_at": str(row["created_at"]), "payload": payload})
        return snapshots

    def search_events_stats(self, *, since: str) -> dict[str, Any]:
        """Aggregate search stats since an ISO cutoff, grouped by source."""
        row = self._connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN json_array_length(results) > 0 THEN 1 ELSE 0 END) AS with_results,
                AVG(latency_ms) AS avg_latency
            FROM search_events WHERE ts >= ?
            """,
            (since,),
        ).fetchone()
        by_source: dict[str, dict[str, Any]] = {}
        source_rows = self._connection.execute(
            """
            SELECT source, COUNT(*) AS count, AVG(latency_ms) AS avg_latency,
                   SUM(CASE WHEN json_array_length(results) = 0 THEN 1 ELSE 0 END) AS zero_result
            FROM search_events WHERE ts >= ? GROUP BY source
            """,
            (since,),
        ).fetchall()
        for src in source_rows:
            by_source[str(src["source"])] = {
                "count": int(src["count"] or 0),
                "avg_latency_ms": _round_optional(src["avg_latency"]),
                "zero_result_count": int(src["zero_result"] or 0),
            }
        inject_row = self._connection.execute(
            """
            SELECT SUM(value) AS injected FROM search_events, json_each(search_events.results)
            WHERE search_events.ts >= ? AND json_extract(json_each.value, '$.inject') = 1
            """,
            (since,),
        ).fetchone()
        total = int(row["total"] or 0) if row is not None else 0
        with_results = int(row["with_results"] or 0) if row is not None else 0
        injected = int(inject_row["injected"] or 0) if inject_row is not None else 0
        return {
            "total": total,
            "with_results": with_results,
            "zero_result": total - with_results,
            "injected_results_total": injected,
            "avg_latency_ms": _round_optional(row["avg_latency"]) if row is not None else 0.0,
            "by_source": by_source,
        }

    def get_dreamer_metrics_summary(self) -> dict[str, Any]:
        """Return aggregate Dreamer metrics for service stats payloads."""
        aggregate = self._dreamer_aggregate_row()
        windows = {
            "last_24h": _utc_now_isoformat_offset(days=1),
            "last_7d": _utc_now_isoformat_offset(days=7),
            "last_30d": _utc_now_isoformat_offset(days=30),
        }
        window_counts = {
            name: self._count_rows_since("dreamer_runs", "started_at", cutoff)
            for name, cutoff in windows.items()
        }
        return {
            "runs": {
                "total": int(aggregate["total"] or 0),
                **window_counts,
                "avg_duration_ms": _round_optional(aggregate["avg_duration_ms"]),
                "avg_triage_decisions": _round_optional(aggregate["avg_triage_decisions"]),
                "avg_links_added": _round_optional(aggregate["avg_links_added"]),
                "avg_condensed": _round_optional(aggregate["avg_condensed"]),
                "avg_archived": _round_optional(aggregate["avg_archived"]),
            },
            "decision_totals": {
                "triage_keep": int(aggregate["triage_keep"] or 0),
                "triage_condense": int(aggregate["triage_condense"] or 0),
                "triage_archive": int(aggregate["triage_archive"] or 0),
                "links_added": int(aggregate["links_added"] or 0),
                "conflicts_superseded": int(aggregate["conflicts_superseded"] or 0),
                "conflicts_both_valid": int(aggregate["conflicts_both_valid"] or 0),
                "warnings": int(aggregate["warnings"] or 0),
                "sampling_failures": int(aggregate["sampling_failures"] or 0),
            },
        }

    def get_write_memory_metrics_summary(self) -> dict[str, Any]:
        """Return aggregate write_memory metrics for service stats payloads."""
        aggregate = self._connection.execute(
            """
            SELECT
                COUNT(*) AS requests_total,
                AVG(candidate_count) AS candidate_count_avg,
                SUM(CASE WHEN candidate_count = 0 THEN 1 ELSE 0 END) AS zero_candidate_count,
                SUM(CASE WHEN action = 'create' THEN 1 ELSE 0 END) AS create_count,
                SUM(CASE WHEN action = 'supersede' THEN 1 ELSE 0 END) AS supersede_count,
                SUM(CASE WHEN action = 'complement' THEN 1 ELSE 0 END) AS complement_count,
                SUM(CASE WHEN execution_succeeded = 0 THEN 1 ELSE 0 END) AS execution_failures
            FROM write_memory_events
            """
        ).fetchone()
        requests_total = int(aggregate["requests_total"] or 0)
        warning_counts: dict[str, int] = {}
        rows = self._connection.execute("SELECT warning_codes FROM write_memory_events").fetchall()
        for row in rows:
            try:
                codes = json.loads(str(row["warning_codes"] or "[]"))
            except json.JSONDecodeError:
                continue
            for code in codes:
                warning_counts[str(code)] = warning_counts.get(str(code), 0) + 1

        zero_count = int(aggregate["zero_candidate_count"] or 0)
        zero_rate = round(zero_count / requests_total, 4) if requests_total else 0.0
        return {
            "requests_total": requests_total,
            "candidate_count_avg": _round_optional(aggregate["candidate_count_avg"]),
            "candidate_count_zero_rate": zero_rate,
            "decision_totals": {
                "create": int(aggregate["create_count"] or 0),
                "supersede": int(aggregate["supersede_count"] or 0),
                "complement": int(aggregate["complement_count"] or 0),
            },
            "warnings": warning_counts,
            "execution_failures": int(aggregate["execution_failures"] or 0),
        }

    def _dreamer_aggregate_row(self) -> sqlite3.Row:
        return self._connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                AVG(duration_ms) AS avg_duration_ms,
                AVG(triage_keep + triage_condense + triage_archive) AS avg_triage_decisions,
                AVG(links_added) AS avg_links_added,
                AVG(condensed) AS avg_condensed,
                AVG(archived) AS avg_archived,
                SUM(triage_keep) AS triage_keep,
                SUM(triage_condense) AS triage_condense,
                SUM(triage_archive) AS triage_archive,
                SUM(links_added) AS links_added,
                SUM(conflicts_superseded) AS conflicts_superseded,
                SUM(conflicts_both_valid) AS conflicts_both_valid,
                SUM(warnings) AS warnings,
                SUM(sampling_failures) AS sampling_failures
            FROM dreamer_runs
            """
        ).fetchone()

    def _count_rows_since(self, table: str, column: str, cutoff: str) -> int:
        row = self._connection.execute(
            f"SELECT COUNT(*) AS count FROM {table} WHERE {column} >= ?",
            (cutoff,),
        ).fetchone()
        return int(row["count"] or 0) if row is not None else 0

    def check_integrity(self) -> DatabaseIntegrityReport:
        integrity_row = self._connection.execute("PRAGMA integrity_check").fetchone()
        integrity_result = str(integrity_row[0]) if integrity_row is not None else "unknown"
        return DatabaseIntegrityReport(
            ok=integrity_result == "ok",
            integrity_check_result=integrity_result,
            wal_mode_enabled=self.is_wal_enabled(),
            schema_version=int(self.get_meta("schema_version") or SCHEMA_VERSION),
            vector_backend=self.vector_backend,
            total_nodes=self.count_nodes(),
        )

    def reset_index(self, *, preserve_meta_keys: Iterable[str] | None = None) -> None:
        preserve_keys = set(preserve_meta_keys or {"schema_version", "vector_backend", "embedding_dimension"})
        preserved = {
            key: value
            for key, value in self._connection.execute("SELECT key, value FROM schema_meta").fetchall()
            if str(key) in preserve_keys
        }
        with self.transaction() as connection:
            self._invalidate_vector_cache()
            connection.execute("DELETE FROM edges")
            connection.execute("DELETE FROM nodes_vec")
            connection.execute("DELETE FROM nodes")
            connection.execute("DELETE FROM nodes_bi_content")
            connection.execute("DELETE FROM schema_meta")
            for key, value in preserved.items():
                self._set_meta(connection, str(key), str(value))
            self._set_meta(connection, "schema_version", str(SCHEMA_VERSION))
            self._set_meta(connection, "vector_backend", self.vector_backend)
            self._set_meta(connection, "embedding_dimension", str(self.embedding_dimension))
            connection.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")
            connection.execute("INSERT INTO nodes_fts_bigram(nodes_fts_bigram) VALUES ('rebuild')")

    def rebuild_from_nodes(
        self,
        nodes: list[Node],
        embeddings: Sequence[list[float] | None],
        *,
        embedding_fingerprint: str,
        source_mtimes: Sequence[datetime | None] | None = None,
        progress_callback: Callable[[RebuildProgress], None] | None = None,
    ) -> int:
        if len(nodes) != len(embeddings):
            raise ValueError("nodes and embeddings must have the same length")
        if source_mtimes is not None and len(source_mtimes) != len(nodes):
            raise ValueError("source_mtimes and nodes must have the same length")

        self.reset_index()
        alias_map = build_node_alias_map(nodes)
        for index, (node, embedding) in enumerate(zip(nodes, embeddings, strict=True), start=1):
            source_mtime = source_mtimes[index - 1] if source_mtimes is not None else node.metadata.last_accessed
            self.upsert_node(node, embedding=embedding, source_mtime=source_mtime)
            if progress_callback is not None:
                progress_callback(
                    RebuildProgress(
                        index=index,
                        total=len(nodes),
                        node_id=node.id,
                        file_path=node.file_path.as_posix(),
                    )
                )
        for node in nodes:
            self.upsert_edges(node.id, resolve_links_to_node_ids(extract_wiki_links(node.content), alias_map))
        self.set_embedding_fingerprint(embedding_fingerprint)
        return len(nodes)

    def _node_to_db_values(self, node: Node, *, source_mtime: datetime | str | None = None) -> tuple[Any, ...]:
        metadata = node.metadata
        return (
            node.id,
            node.title,
            node.file_path.as_posix(),
            _ensure_isoformat(source_mtime or metadata.last_accessed),
            node.content,
            metadata.type.value,
            metadata.status.value,
            metadata.sensitivity.value,
            json.dumps(metadata.supersedes),
            metadata.superseded_by,
            _ensure_isoformat(metadata.created_at),
            _ensure_isoformat(metadata.last_accessed),
            metadata.access_count,
            json.dumps(metadata.tags),
            _okf_meta_json(metadata),
        )

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        try:
            okf_meta = _okf_meta_from_json(row["okf_meta"])
        except (IndexError, KeyError):
            okf_meta = {}
        metadata = NodeMetadata.model_validate(
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "last_accessed": row["last_accessed"],
                "access_count": row["access_count"],
                "type": row["type"],
                "status": row["status"],
                "sensitivity": row["sensitivity"] or "internal",
                "supersedes": json.loads(row["supersedes"] or "[]"),
                "superseded_by": row["superseded_by"],
                "tags": json.loads(row["tags"] or "[]"),
                **okf_meta,
            }
        )
        return Node(metadata=metadata, content=str(row["content"]), file_path=Path(str(row["file_path"])))


def build_node_alias_map(nodes: Iterable[Node]) -> dict[str, str]:
    """Build a lookup from common wiki-link aliases to canonical node IDs."""

    alias_map: dict[str, str] = {}
    for node in nodes:
        candidates = {
            node.id,
            node.id.casefold(),
            node.title,
            node.title.casefold(),
            node.file_path.stem,
            node.file_path.stem.casefold(),
        }
        for tag in node.metadata.tags:
            candidates.add(tag)
            candidates.add(tag.casefold())
        for candidate in candidates:
            cleaned = candidate.strip()
            if cleaned and cleaned not in alias_map:
                alias_map[cleaned] = node.id
    return alias_map


def resolve_links_to_node_ids(links: Iterable[str], alias_map: dict[str, str]) -> list[str]:
    """Resolve extracted wiki links to canonical node IDs where possible."""

    resolved: list[str] = []
    for link in links:
        cleaned = link.strip()
        if not cleaned:
            continue
        target_id = alias_map.get(cleaned) or alias_map.get(cleaned.casefold())
        if target_id is not None:
            resolved.append(target_id)
    return resolved


def _ensure_isoformat(value: Any) -> str:
    if isinstance(value, datetime):
        normalized = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return normalized.isoformat().replace(UTC_SUFFIX, "Z")
    return str(value)


def _parse_iso_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    candidate = str(value)
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    parsed = datetime.fromisoformat(candidate)
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _utc_now_isoformat() -> str:
    return datetime.now(UTC).isoformat().replace(UTC_SUFFIX, "Z")


def _utc_now_isoformat_offset(*, days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat().replace(UTC_SUFFIX, "Z")


def _utc_cutoff(days_threshold: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_threshold)).isoformat().replace(UTC_SUFFIX, "Z")


def _normalize_embedding(embedding: list[float], expected_dimension: int) -> list[float]:
    if len(embedding) != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimension}, received {len(embedding)}"
        )
    magnitude = math.sqrt(sum(value * value for value in embedding))
    if magnitude == 0:
        return [0.0 for _ in embedding]
    return [float(value) / magnitude for value in embedding]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cosine distance requires equal-length vectors")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    similarity = max(-1.0, min(1.0, dot))
    return round(1.0 - similarity, 8)


def _safe_cosine_similarity(left: list[float], right: list[float]) -> float | None:
    try:
        return 1.0 - _cosine_distance(left, right)
    except ValueError:
        return None


def _increment_similarity_buckets(counts: dict[str, int], buckets: list[float], similarity: float) -> None:
    for threshold in buckets:
        if similarity >= threshold:
            counts[f"{threshold:.2f}"] += 1


def _round_optional(value: Any) -> float:
    if value is None:
        return 0.0
    return round(float(value), 4)
