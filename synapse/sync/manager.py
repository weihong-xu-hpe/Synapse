"""File watcher abstraction and delta sync manager for Phase 4."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

from synapse.config import SynapseConfig
from synapse.embedding import create_embedding_engine
from synapse.models import Node
from synapse.storage import SQLiteNodeStore, build_node_alias_map, extract_wiki_links, read_node_file, resolve_links_to_node_ids
from synapse.utils.documents import render_node_document
from synapse.utils.runtime import RuntimePaths, get_runtime_paths


try:  # pragma: no cover - optional dependency
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # pragma: no cover - exercised through fallback behavior instead
    FileSystemEventHandler = object  # type: ignore[assignment]
    Observer = None


LOGGER = logging.getLogger(__name__)
SyncAction = Literal["upsert", "delete"]


@dataclass(slots=True, frozen=True)
class SyncBatchResult:
    """Summary of a processed sync batch."""

    upserted: int = 0
    deleted: int = 0
    failed: int = 0
    queued: int = 0
    backend: str = "polling"
    details: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> int:
        return self.upserted + self.deleted


@dataclass(slots=True, frozen=True)
class SyncRuntimeStatus:
    """Describes the runtime sync backend exposed through health checks."""

    file_watcher: str
    delta_sync_hook: str
    startup_sync_hook: str


@dataclass(slots=True)
class QueuedSyncEvent:
    """Debounced event queued for later batch processing."""

    action: SyncAction
    path: Path
    queued_at: float


class PollingFileWatcher:
    """Snapshot-diff polling watcher used when watchdog is unavailable."""

    def __init__(self, manager: "SyncManager") -> None:
        self.manager = manager
        self._snapshot = self.manager.snapshot_active_files()

    def poll_once(self, *, drain: bool = True) -> SyncBatchResult:
        current = self.manager.snapshot_active_files()
        previous_paths = set(self._snapshot)
        current_paths = set(current)

        for path in current_paths - previous_paths:
            self.manager.queue_event("upsert", self.manager.runtime_paths.base / path)
        for path in previous_paths - current_paths:
            self.manager.queue_event("delete", self.manager.runtime_paths.base / path)
        for path in previous_paths & current_paths:
            if current[path] != self._snapshot[path]:
                self.manager.queue_event("upsert", self.manager.runtime_paths.base / path)

        self._snapshot = current
        if drain:
            return self.manager.drain_pending(force=True)
        return SyncBatchResult(queued=len(self.manager.pending_events), backend="polling")


class _WatchdogEventHandler(FileSystemEventHandler):  # pragma: no cover - optional dependency shim
    def __init__(self, manager: "SyncManager") -> None:
        self.manager = manager

    def _queue_upsert(self, event) -> None:
        if not getattr(event, "is_directory", False):
            self.manager.queue_event("upsert", Path(event.src_path))

    def on_created(self, event) -> None:
        self._queue_upsert(event)

    def on_modified(self, event) -> None:
        self._queue_upsert(event)

    def on_deleted(self, event) -> None:
        if not getattr(event, "is_directory", False):
            self.manager.queue_event("delete", Path(event.src_path))

    def on_moved(self, event) -> None:
        if not getattr(event, "is_directory", False):
            self.manager.queue_event("delete", Path(event.src_path))
            self.manager.queue_event("upsert", Path(event.dest_path))


class WatchdogFileWatcher:
    """Thin watchdog wrapper when the optional dependency is installed."""

    def __init__(self, manager: "SyncManager") -> None:
        if Observer is None:  # pragma: no cover - defensive
            raise RuntimeError("watchdog is not installed")
        self.manager = manager
        self._observer = Observer()
        self._handler = _WatchdogEventHandler(manager)

    def start(self) -> None:
        self._observer.schedule(self._handler, str(self.manager.runtime_paths.active), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=2)


class SyncManager:
    """Debounced sync manager for Markdown-to-SQLite derived index updates."""

    def __init__(
        self,
        config: SynapseConfig,
        *,
        runtime_paths: RuntimePaths | None = None,
        store: SQLiteNodeStore | None = None,
        embedding_engine=None,
        debounce_seconds: float = 0.5,
        clock: Callable[[], float] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.runtime_paths = runtime_paths or get_runtime_paths(config)
        self._store = store
        self._owns_store = store is None
        self._embedding_engine = embedding_engine or create_embedding_engine(config.embedding, providers=config.providers)
        self._debounce_seconds = debounce_seconds
        self._clock = clock or time.monotonic
        self._logger = logger or LOGGER
        self._pending: dict[Path, QueuedSyncEvent] = {}

    def close(self) -> None:
        if self._owns_store and self._store is not None:
            self._store.close()
            self._store = None

    def __enter__(self) -> "SyncManager":
        self._get_store()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    @property
    def pending_events(self) -> tuple[QueuedSyncEvent, ...]:
        return tuple(self._pending.values())

    @property
    def backend_name(self) -> str:
        return "watchdog" if Observer is not None else "polling"

    def describe_runtime(self) -> SyncRuntimeStatus:
        watcher_mode = "watchdog" if Observer is not None else "polling-fallback"
        return SyncRuntimeStatus(
            file_watcher=f"ready ({watcher_mode})",
            delta_sync_hook="enabled",
            startup_sync_hook="enabled",
        )

    def create_watcher(self):
        if Observer is not None:
            return WatchdogFileWatcher(self)
        return PollingFileWatcher(self)

    def queue_event(self, action: SyncAction, path: str | Path) -> None:
        normalized_path = Path(path)
        if not self._is_markdown_path(normalized_path):
            return
        self._pending[normalized_path] = QueuedSyncEvent(
            action=action,
            path=normalized_path,
            queued_at=self._clock(),
        )

    def drain_pending(self, *, force: bool = False) -> SyncBatchResult:
        now = self._clock()
        ready_paths = [
            path
            for path, event in self._pending.items()
            if force or (now - event.queued_at) >= self._debounce_seconds
        ]
        if not ready_paths:
            return SyncBatchResult(queued=len(self._pending), backend=self.backend_name)

        upserted = deleted = failed = 0
        details: list[str] = []
        events = [self._pending.pop(path) for path in ready_paths]
        events.sort(key=lambda item: (item.action != "delete", item.path.as_posix()))
        synced_nodes: list[Node] = []

        for event in events:
            try:
                if event.action == "delete":
                    deleted += self._sync_delete(event.path)
                    continue
                if not event.path.exists():
                    deleted += self._sync_delete(event.path)
                    continue
                synced_nodes.append(self._sync_upsert(event.path))
                upserted += 1
            except (OSError, ValueError, RuntimeError, TypeError, sqlite3.DatabaseError) as exc:
                failed += 1
                message = f"{event.action}:{event.path} -> {exc}"
                details.append(message)
                self._logger.warning("Sync event failed: %s", message)

        if synced_nodes:
            store = self._get_store()
            alias_map = build_node_alias_map(store.list_nodes())
            for node in synced_nodes:
                linked_ids = resolve_links_to_node_ids(extract_wiki_links(node.content), alias_map)
                store.upsert_edges(node.id, linked_ids)

        return SyncBatchResult(
            upserted=upserted,
            deleted=deleted,
            failed=failed,
            queued=len(self._pending),
            backend=self.backend_name,
            details=tuple(details),
        )

    def startup_sync(self) -> SyncBatchResult:
        store = self._get_store()
        disk_snapshot = self.snapshot_active_files()
        indexed = store.get_indexed_file_states()

        for relative_path, mtime in disk_snapshot.items():
            state = indexed.get(relative_path.as_posix())
            if state is None or state.source_mtime is None or mtime > state.source_mtime:
                self.queue_event("upsert", self.runtime_paths.base / relative_path)

        for indexed_path in indexed:
            if Path(indexed_path) not in disk_snapshot:
                self.queue_event("delete", self.runtime_paths.base / indexed_path)

        return self.drain_pending(force=True)

    def sync_paths(self, paths: list[str | Path]) -> SyncBatchResult:
        for path in paths:
            self.queue_event("upsert", path)
        return self.drain_pending(force=True)

    def snapshot_active_files(self) -> dict[Path, datetime]:
        snapshot: dict[Path, datetime] = {}
        if not self.runtime_paths.active.exists():
            return snapshot
        for path in sorted(self.runtime_paths.active.rglob("*.md")):
            if not path.is_file():
                continue
            snapshot[self._relative_to_base(path)] = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return snapshot

    def _sync_upsert(self, path: Path) -> Node:
        node = self._load_node_for_sync(path)
        file_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        embedding = None
        try:
            embedding = self._embedding_engine.embed(render_node_document(node))
        except (OSError, RuntimeError, ValueError) as exc:
            self._logger.warning("Embedding failed for %s: %s", path, exc)

        store = self._get_store()
        store.upsert_node(node, embedding=embedding, source_mtime=file_mtime)
        if embedding is None:
            store.delete_embedding(node.id)
        return node

    def _sync_delete(self, path: Path) -> int:
        relative_path = self._relative_to_base(path)
        node = self._get_store().get_node_by_file_path(relative_path)
        if node is None:
            return 0
        self._get_store().delete_node(node.id)
        return 1

    def _load_node_for_sync(self, path: Path) -> Node:
        parsed = read_node_file(path)
        relative_path = self._relative_to_base(path)
        file_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        metadata = parsed.metadata.model_copy(update={"last_accessed": file_mtime})
        return parsed.model_copy(update={"metadata": metadata, "file_path": relative_path})

    def _relative_to_base(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self.runtime_paths.base)
        except ValueError:
            return path

    def _is_markdown_path(self, path: Path) -> bool:
        return path.suffix.casefold() == ".md"

    def _get_store(self) -> SQLiteNodeStore:
        if self._store is None:
            self._store = SQLiteNodeStore(
                self.runtime_paths.base / "synapse.db",
                embedding_dimension=self.config.embedding.dimension or 0,
            )
        return self._store