"""Observability: daily metrics snapshots, lookback reports, injection audits.

Local-only, private-DB analytics written for the owner's 1–2 month upgrade
lookback. Everything here tolerates partial data (fresh tables, missing eval
history) and never raises into the paths it piggybacks on.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from synapse.config import SynapseConfig
from synapse.storage import SQLiteNodeStore
from synapse.storage.sqlite import _utc_cutoff
from synapse.utils.runtime import RuntimePaths, get_runtime_paths

_SINCE_PATTERN = re.compile(r"^(\d+)\s*([dwm]?)$", re.IGNORECASE)
_AUDIT_SOURCE_PREFIXES = ("bridge", "mcp")
_WILSON_Z = 1.96


class SQLiteStoreLike(Protocol):
    """Structural subset of SQLiteNodeStore used by report rendering."""

    connection: Any

    def get_metrics_snapshots(self, *, since_date: str | None = None) -> list[dict[str, Any]]: ...


def parse_since_days(value: str) -> int:
    """Parse ``30d`` / ``12w`` / ``6m`` (or a bare day count) into days."""

    match = _SINCE_PATTERN.match(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid --since value {value!r}; use e.g. 30d, 12w, 6m")
    amount = int(match.group(1))
    unit = (match.group(2) or "d").lower()
    if amount <= 0:
        raise ValueError("--since must be positive")
    if unit == "w":
        return amount * 7
    if unit == "m":
        return amount * 30
    return amount


def cjk_ratio(text: str) -> float:
    """Share of CJK characters in a query (0.0 for empty)."""

    if not text:
        return 0.0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff")
    return round(cjk / len(text), 4)


def session_hash(session_key: str | None) -> str | None:
    """sha1 of the bridge exclude_session_key, truncated — no raw keys stored."""

    if not session_key or not session_key.strip():
        return None
    return hashlib.sha1(session_key.strip().encode("utf-8")).hexdigest()[:12]


def local_day_window(day: date) -> tuple[str, str]:
    """UTC ISO cutoffs [start, end) covering one local-calendar day.

    Event timestamps are stored UTC; snapshot dates are the owner's local
    calendar, so the window converts local midnight to UTC.
    """

    start_local = datetime(day.year, day.month, day.day).astimezone()
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(UTC).isoformat().replace("+00:00", "Z")
    end_utc = end_local.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return start_utc, end_utc


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return round(ordered[max(0, index)], 1)


def _dir_size(path: Path) -> int:
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0


def _open_store(config: SynapseConfig, runtime_paths: RuntimePaths) -> SQLiteNodeStore:
    return SQLiteNodeStore(
        runtime_paths.base / "synapse.db",
        embedding_dimension=config.embedding.dimension or 0,
    )


def collect_daily_snapshot(
    config: SynapseConfig,
    runtime_paths: RuntimePaths,
    *,
    day: date,
    store: SQLiteNodeStore | None = None,
) -> dict[str, Any]:
    """Build one day's metrics payload. Read-only; never mutates store state."""

    start_utc, end_utc = local_day_window(day)
    owns_store = store is None
    if store is None:
        store = _open_store(config, runtime_paths)
    try:
        payload: dict[str, Any] = {
            "nodes": _node_counts(store),
            "transcripts": _transcript_counts(store),
            "writes": _day_write_mix(store, start_utc, end_utc),
            "distiller": _day_distiller_yield(store, start_utc, end_utc),
            "searches": _day_search_stats(store, start_utc, end_utc),
            "dreamer": _day_dreamer_runs(store, start_utc, end_utc),
            "storage": {
                "db_bytes": (runtime_paths.base / "synapse.db").stat().st_size
                if (runtime_paths.base / "synapse.db").exists()
                else 0,
                "active_bytes": _dir_size(runtime_paths.active),
                "archive_bytes": _dir_size(runtime_paths.archive),
            },
        }
        return payload
    finally:
        if owns_store:
            store.close()


def _node_counts(store: SQLiteNodeStore) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_okf_type: dict[str, int] = {}
    by_project: dict[str, int] = {}
    rows = store.connection.execute(
        """
        SELECT status, type, okf_meta, COUNT(*) AS count FROM nodes
        GROUP BY status, type, okf_meta
        """
    ).fetchall()
    total = 0
    for row in rows:
        count = int(row["count"])
        total += count
        status = str(row["status"] or "unknown")
        node_type = str(row["type"] or "unknown")
        try:
            okf_meta = json.loads(str(row["okf_meta"] or "{}"))
        except json.JSONDecodeError:
            okf_meta = {}
        okf_type = str(okf_meta.get("okf_type") or "untyped")
        project = str(okf_meta.get("project") or "unassigned")
        by_status[status] = by_status.get(status, 0) + count
        by_type[node_type] = by_type.get(node_type, 0) + count
        by_okf_type[okf_type] = by_okf_type.get(okf_type, 0) + count
        by_project[project] = by_project.get(project, 0) + count
    return {"total": total, "by_status": by_status, "by_type": by_type, "by_okf_type": by_okf_type, "by_project": by_project}


def _transcript_counts(store: SQLiteNodeStore) -> dict[str, int]:
    """Classify session transcripts without loading Node models.

    Keyed transcripts carry ``mem_session_*`` ids; legacy ones are transient
    nodes titled ``Session summary …`` (shared predicate semantics from
    synapse.okf.transcripts). Distillation state lives in okf_meta.
    """

    counts = {"represented": 0, "zero_item": 0, "undistilled": 0}
    rows = store.connection.execute(
        """
        SELECT id, content, okf_meta FROM nodes
        WHERE id LIKE 'mem\\_session\\_%' ESCAPE '\\'
           OR (type = 'transient' AND title LIKE 'Session summary%')
        """
    ).fetchall()
    for row in rows:
        try:
            okf_meta = json.loads(str(row["okf_meta"] or "{}"))
        except json.JSONDecodeError:
            okf_meta = {}
        distilled_hash = okf_meta.get("distilled_hash")
        if not distilled_hash or distilled_hash != hashlib.sha256(str(row["content"] or "").encode("utf-8")).hexdigest():
            counts["undistilled"] += 1
        elif okf_meta.get("distilled_node_ids"):
            counts["represented"] += 1
        else:
            counts["zero_item"] += 1
    return counts


def _day_write_mix(store: SQLiteNodeStore, start_utc: str, end_utc: str) -> dict[str, Any]:
    row = store.connection.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN execution_succeeded = 0 THEN 1 ELSE 0 END) AS failures
        FROM write_memory_events WHERE created_at >= ? AND created_at < ?
        """,
        (start_utc, end_utc),
    ).fetchone()
    by_route: dict[str, int] = {}
    for route_row in store.connection.execute(
        """
        SELECT COALESCE(route, COALESCE(sampling_provider, 'unknown')) AS route, COUNT(*) AS count
        FROM write_memory_events WHERE created_at >= ? AND created_at < ? GROUP BY route
        """,
        (start_utc, end_utc),
    ).fetchall():
        by_route[str(route_row["route"])] = int(route_row["count"])
    by_action: dict[str, int] = {}
    for action_row in store.connection.execute(
        """
        SELECT COALESCE(action, 'none') AS action, COUNT(*) AS count
        FROM write_memory_events WHERE created_at >= ? AND created_at < ? GROUP BY action
        """,
        (start_utc, end_utc),
    ).fetchall():
        by_action[str(action_row["action"])] = int(action_row["count"])
    warning_counts: dict[str, int] = {}
    for warning_row in store.connection.execute(
        "SELECT warning_codes FROM write_memory_events WHERE created_at >= ? AND created_at < ?",
        (start_utc, end_utc),
    ).fetchall():
        try:
            codes = json.loads(str(warning_row["warning_codes"] or "[]"))
        except json.JSONDecodeError:
            continue
        for code in codes:
            warning_counts[str(code)] = warning_counts.get(str(code), 0) + 1
    return {
        "total": int(row["total"] or 0) if row is not None else 0,
        "execution_failures": int(row["failures"] or 0) if row is not None else 0,
        "by_route": by_route,
        "by_action": by_action,
        "warnings": warning_counts,
    }


def _day_distiller_yield(store: SQLiteNodeStore, start_utc: str, end_utc: str) -> dict[str, Any]:
    row = store.connection.execute(
        """
        SELECT COUNT(*) AS runs,
               COALESCE(SUM(transcripts_scanned), 0) AS transcripts_scanned,
               COALESCE(SUM(transcripts_distilled), 0) AS transcripts_distilled,
               COALESCE(SUM(transcripts_skipped), 0) AS transcripts_skipped,
               COALESCE(SUM(items_extracted), 0) AS items_extracted,
               COALESCE(SUM(items_invalid), 0) AS items_invalid,
               COALESCE(SUM(llm_failures), 0) AS llm_failures,
               COALESCE(SUM(decider_downgrades), 0) AS decider_downgrades,
               COALESCE(SUM(archived_transcripts), 0) AS archived_transcripts
        FROM distiller_runs WHERE started_at >= ? AND started_at < ?
        """,
        (start_utc, end_utc),
    ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()}


def _day_dreamer_runs(store: SQLiteNodeStore, start_utc: str, end_utc: str) -> dict[str, Any]:
    row = store.connection.execute(
        """
        SELECT COUNT(*) AS runs,
               COALESCE(SUM(archived), 0) AS archived,
               COALESCE(SUM(condensed), 0) AS condensed,
               COALESCE(SUM(links_added), 0) AS links_added,
               COALESCE(SUM(sampling_failures), 0) AS sampling_failures
        FROM dreamer_runs WHERE started_at >= ? AND started_at < ?
        """,
        (start_utc, end_utc),
    ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()}


def _day_search_stats(store: SQLiteNodeStore, start_utc: str, end_utc: str) -> dict[str, Any]:
    rows = store.connection.execute(
        "SELECT source, latency_ms, results FROM search_events WHERE ts >= ? AND ts < ?",
        (start_utc, end_utc),
    ).fetchall()
    total = len(rows)
    by_source: dict[str, int] = {}
    latencies: list[float] = []
    injected = 0
    zero_inject_searches = 0
    inject_by_node: dict[str, int] = {}
    for row in rows:
        source = str(row["source"])
        by_source[source] = by_source.get(source, 0) + 1
        latencies.append(float(row["latency_ms"]))
        try:
            results = json.loads(str(row["results"] or "[]"))
        except json.JSONDecodeError:
            results = []
        row_injected = 0
        for item in results:
            if isinstance(item, dict) and item.get("inject"):
                row_injected += 1
                node_id = str(item.get("node_id") or "")
                inject_by_node[node_id] = inject_by_node.get(node_id, 0) + 1
        injected += row_injected
        if row_injected == 0:
            zero_inject_searches += 1
    top_injected = sorted(inject_by_node.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "count": total,
        "by_source": by_source,
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "injected_results_total": injected,
        "inject_rate": round(injected / total, 4) if total else 0.0,
        "zero_inject_share": round(zero_inject_searches / total, 4) if total else 0.0,
        "top_injected_node_ids": [{"node_id": node_id, "count": count} for node_id, count in top_injected],
    }


def _earliest_event_date(store: SQLiteNodeStore) -> date | None:
    earliest: str | None = None
    for row in [
        store.connection.execute("SELECT MIN(created_at) AS value FROM write_memory_events").fetchone(),
        store.connection.execute("SELECT MIN(started_at) AS value FROM distiller_runs").fetchone(),
        store.connection.execute("SELECT MIN(started_at) AS value FROM dreamer_runs").fetchone(),
        store.connection.execute("SELECT MIN(ts) AS value FROM search_events").fetchone(),
        store.connection.execute("SELECT MIN(created_at) AS value FROM nodes").fetchone(),
    ]:
        value = str(row["value"]) if row is not None and row["value"] else None
        if value and (earliest is None or value < earliest):
            earliest = value
    if earliest is None:
        return None
    try:
        return datetime.fromisoformat(earliest.replace("Z", "+00:00")).astimezone().date()
    except ValueError:
        return None


def write_missing_snapshots(
    config: SynapseConfig,
    runtime_paths: RuntimePaths,
    *,
    backfill_days: int = 2,
) -> list[str]:
    """Write today's snapshot row and backfill recent missing days.

    Idempotent per local date (primary-key guarded). Days before the earliest
    recorded event are skipped so fresh installs don't accumulate zero rows.
    Returns the list of dates written.
    """

    written: list[str] = []
    now = datetime.now(UTC)
    store = _open_store(config, runtime_paths)
    try:
        earliest = _earliest_event_date(store)
        today_local = now.astimezone().date()
        days = [today_local - timedelta(days=offset) for offset in range(backfill_days, -1, -1)]
        for day in days:
            if earliest is not None and day < earliest:
                continue
            date_key = day.isoformat()
            existing = store.connection.execute(
                "SELECT 1 FROM metrics_snapshots WHERE date = ?", (date_key,)
            ).fetchone()
            if existing is not None:
                continue
            payload = collect_daily_snapshot(config, runtime_paths, day=day, store=store)
            created = store.record_metrics_snapshot(
                date=date_key,
                created_at=now.isoformat().replace("+00:00", "Z"),
                payload=payload,
            )
            if created:
                written.append(date_key)
    finally:
        store.close()
    return written


def _load_jsonl_window(path: Path, *, since_days: int, now: datetime | None = None) -> list[dict[str, Any]]:
    """Read a JSONL history file, returning entries inside the lookback window."""

    cutoff = _utc_cutoff(since_days)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    del now
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = str(entry.get("ts") or "")
            if ts and ts < cutoff:
                continue
            entries.append(entry)
    except OSError:
        return []
    return entries


def _week_key(iso_ts: str) -> str:
    try:
        moment = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    local = moment.astimezone()
    year, week, _ = local.isocalendar()
    return f"{year}-W{week:02d}"


def render_report(
    config: SynapseConfig,
    runtime_paths: RuntimePaths,
    *,
    since_days: int,
    out_path: str | Path | None = None,
) -> str:
    """Render the lookback markdown report. Tolerates empty/partial data."""

    cutoff = _utc_cutoff(since_days)
    store = _open_store(config, runtime_paths)
    try:
        sections = _report_sections(config, store, runtime_paths, since_days=since_days, cutoff=cutoff)
    finally:
        store.close()

    title = f"# Synapse lookback report ({since_days}d)"
    lines = [title, "", f"_Generated {datetime.now(UTC).isoformat()} — window since {cutoff}_", ""]
    lines.extend(sections)
    content = "\n".join(lines).rstrip() + "\n"
    if out_path is not None:
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
    return content


def _report_sections(
    config: SynapseConfig,
    store: SQLiteStoreLike,
    runtime_paths: RuntimePaths,
    *,
    since_days: int,
    cutoff: str,
) -> list[str]:
    lines: list[str] = []

    # --- Search volume + latency trend (weekly buckets) ---
    rows = store.connection.execute(
        "SELECT ts, source, latency_ms, session_hash, results FROM search_events WHERE ts >= ? ORDER BY ts",
        (cutoff,),
    ).fetchall()
    lines.append("## Search volume & latency")
    if not rows:
        lines.append("_No search events recorded in the window._")
    else:
        weeks: dict[str, list[Any]] = {}
        for row in rows:
            weeks.setdefault(_week_key(str(row["ts"])), []).append(row)
        lines.append("| Week | Searches | p50 ms | p95 ms | Inject rate |")
        lines.append("|---|---|---|---|---|")
        for week in sorted(weeks):
            week_rows = weeks[week]
            latencies = [float(row["latency_ms"]) for row in week_rows]
            injected = sum(
                1
                for row in week_rows
                for item in _parse_results(row["results"])
                if item.get("inject")
            )
            total_results = sum(len(_parse_results(row["results"])) for row in week_rows)
            lines.append(
                f"| {week} | {len(week_rows)} | {_percentile(latencies, 50)} | {_percentile(latencies, 95)} | "
                f"{round(injected / total_results, 3) if total_results else 0.0} |"
            )
        # Inject rate per source
        by_source: dict[str, dict[str, int]] = {}
        for row in rows:
            source = str(row["source"])
            bucket = by_source.setdefault(source, {"searches": 0, "injected": 0, "results": 0})
            bucket["searches"] += 1
            results = _parse_results(row["results"])
            bucket["results"] += len(results)
            bucket["injected"] += sum(1 for item in results if item.get("inject"))
        lines.append("")
        lines.append("**Inject rate per source** (share of returned results flagged inject=true):")
        for source in sorted(by_source):
            bucket = by_source[source]
            rate = round(bucket["injected"] / bucket["results"], 3) if bucket["results"] else 0.0
            zero_inject = sum(
                1
                for row in rows
                if str(row["source"]) == source and not any(item.get("inject") for item in _parse_results(row["results"]))
            )
            lines.append(f"- `{source}`: {rate} ({bucket['searches']} searches, {zero_inject} with zero injections)")
        # Sessions with zero injections (bridge searches carry a session_hash)
        session_totals: dict[str, list[bool]] = {}
        for row in rows:
            session = str(row["session_hash"] or "")
            if not session:
                continue
            session_totals.setdefault(session, []).append(
                any(item.get("inject") for item in _parse_results(row["results"]))
            )
        zero_sessions = [session for session, hits in session_totals.items() if not any(hits)]
        lines.append("")
        lines.append(f"**Sessions with zero injections**: {len(zero_sessions)} of {len(session_totals)} tracked sessions")
        for session in zero_sessions[:10]:
            lines.append(f"- `{session}`")
    lines.append("")

    # --- Most/least injected knowledge ---
    inject_by_node: dict[str, int] = {}
    for row in rows:
        for item in _parse_results(row["results"]):
            if item.get("inject"):
                node_id = str(item.get("node_id") or "")
                inject_by_node[node_id] = inject_by_node.get(node_id, 0) + 1
    lines.append("## Injection distribution")
    if inject_by_node:
        titles = _node_titles(store, list(inject_by_node))
        ranked = sorted(inject_by_node.items(), key=lambda item: (-item[1], item[0]))
        lines.append("**Most injected:**")
        for node_id, count in ranked[:5]:
            lines.append(f"- `{node_id}` ({titles.get(node_id, 'title unavailable')}): {count}")
        lines.append("**Least injected (bottom of injected set):**")
        for node_id, count in ranked[-5:]:
            lines.append(f"- `{node_id}` ({titles.get(node_id, 'title unavailable')}): {count}")
    else:
        lines.append("_No injected results in the window._")
    lines.append("")

    # --- Knowledge growth ---
    lines.append("## Knowledge growth")
    growth = store.connection.execute(
        """
        SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count,
               json_extract(okf_meta, '$.okf_type') AS okf_type,
               json_extract(okf_meta, '$.project') AS project
        FROM nodes WHERE created_at >= ? GROUP BY day, okf_type, project ORDER BY day
        """,
        (cutoff,),
    ).fetchall()
    if not growth:
        lines.append("_No new nodes in the window._")
    else:
        by_day: dict[str, int] = {}
        by_okf: dict[str, int] = {}
        by_project: dict[str, int] = {}
        for row in growth:
            count = int(row["count"])
            by_day[str(row["day"])] = by_day.get(str(row["day"]), 0) + count
            okf_type = str(row["okf_type"] or "untyped")
            project = str(row["project"] or "unassigned")
            by_okf[okf_type] = by_okf.get(okf_type, 0) + count
            by_project[project] = by_project.get(project, 0) + count
        lines.append(f"New nodes: {sum(by_day.values())} over {len(by_day)} days")
        lines.append(f"- By okf_type: {json.dumps(by_okf, ensure_ascii=False)}")
        lines.append(f"- By project: {json.dumps(dict(sorted(by_project.items(), key=lambda kv: -kv[1])), ensure_ascii=False)}")
    lines.append("")

    # --- Write mix + decider ---
    write_row = store.connection.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN execution_succeeded = 0 THEN 1 ELSE 0 END) AS failures
        FROM write_memory_events WHERE created_at >= ?
        """,
        (cutoff,),
    ).fetchone()
    by_action: dict[str, int] = {}
    for row in store.connection.execute(
        "SELECT COALESCE(action, 'none') AS action, COUNT(*) AS count FROM write_memory_events WHERE created_at >= ? GROUP BY action",
        (cutoff,),
    ).fetchall():
        by_action[str(row["action"])] = int(row["count"])
    by_route: dict[str, int] = {}
    for row in store.connection.execute(
        "SELECT COALESCE(route, COALESCE(sampling_provider, 'unknown')) AS route, COUNT(*) AS count FROM write_memory_events WHERE created_at >= ? GROUP BY route",
        (cutoff,),
    ).fetchall():
        by_route[str(row["route"])] = int(row["count"])
    downgrade_row = store.connection.execute(
        "SELECT COALESCE(SUM(decider_downgrades), 0) AS downgrades FROM distiller_runs WHERE started_at >= ?",
        (cutoff,),
    ).fetchone()
    lines.append("## Writes & decider")
    lines.append(f"- Write events: {int(write_row['total'] or 0)} (execution failures: {int(write_row['failures'] or 0)})")
    lines.append(f"- By route: {json.dumps(by_route, ensure_ascii=False)}")
    lines.append(f"- Decider actions: {json.dumps(by_action, ensure_ascii=False)}")
    lines.append(f"- Supersede downgrades (distiller): {int(downgrade_row['downgrades'] or 0)}")
    lines.append("")

    # --- Distiller & Dreamer ---
    distiller_row = store.connection.execute(
        """
        SELECT COUNT(*) AS runs,
               COALESCE(SUM(transcripts_distilled), 0) AS distilled,
               COALESCE(SUM(items_extracted), 0) AS items,
               COALESCE(SUM(llm_failures), 0) AS failures,
               COALESCE(SUM(archived_transcripts), 0) AS archived
        FROM distiller_runs WHERE started_at >= ?
        """,
        (cutoff,),
    ).fetchone()
    dreamer_row = store.connection.execute(
        """
        SELECT COUNT(*) AS runs,
               COALESCE(SUM(archived), 0) AS archived,
               COALESCE(SUM(condensed), 0) AS condensed
        FROM dreamer_runs WHERE started_at >= ?
        """,
        (cutoff,),
    ).fetchone()
    lines.append("## Lifecycle")
    lines.append(
        f"- Distiller: {int(distiller_row['runs'])} runs, {int(distiller_row['distilled'])} transcripts distilled, "
        f"{int(distiller_row['items'])} items, {int(distiller_row['failures'])} LLM failures, {int(distiller_row['archived'])} archived"
    )
    lines.append(
        f"- Dreamer: {int(dreamer_row['runs'])} runs, {int(dreamer_row['archived'])} archived, {int(dreamer_row['condensed'])} condensed"
    )
    lines.append("")

    # --- Storage growth ---
    lines.append("## Storage")
    snapshots = store.get_metrics_snapshots()
    if not snapshots:
        db_path = runtime_paths.base / "synapse.db"
        lines.append(f"_No snapshots yet. Current DB size: {db_path.stat().st_size if db_path.exists() else 0} bytes._")
    else:
        lines.append("| Date | DB bytes | Active bytes | Archive bytes |")
        lines.append("|---|---|---|---|")
        for snapshot in snapshots:
            storage = snapshot["payload"].get("storage", {})
            lines.append(
                f"| {snapshot['date']} | {storage.get('db_bytes', 0)} | {storage.get('active_bytes', 0)} | {storage.get('archive_bytes', 0)} |"
            )
    lines.append("")

    # --- Eval trend ---
    lines.append("## Eval trend")
    eval_entries = _load_jsonl_window(config.observability.resolved_eval_history_path(config), since_days=since_days)
    if not eval_entries:
        lines.append(f"_No eval runs recorded at {config.observability.resolved_eval_history_path(config)}._")
    else:
        lines.append("| ts | git sha | recall@5 | recall@5_inject | mrr@10 | stale labels |")
        lines.append("|---|---|---|---|---|---|")
        for entry in eval_entries[-20:]:
            overall = entry.get("overall", {})
            lines.append(
                f"| {entry.get('ts', '')} | {entry.get('git_sha') or '-'} | {overall.get('recall@5', '-')} | "
                f"{overall.get('recall@5_inject', '-')} | {overall.get('mrr@10', '-')} | {entry.get('stale_labels', '-')} |"
            )
    lines.append("")

    # --- Injection audit history ---
    lines.append("## Injection audit history")
    audit_entries = _load_jsonl_window(config.observability.resolved_audit_history_path(config), since_days=since_days)
    if not audit_entries:
        lines.append(f"_No audits recorded at {config.observability.resolved_audit_history_path(config)}._")
    else:
        for entry in audit_entries[-10:]:
            lines.append(
                f"- {entry.get('ts', '')}: precision(useful)={entry.get('precision_useful')} "
                f"precision(related)={entry.get('precision_related')} over {entry.get('judged', 0)} judged samples"
            )
    return lines


def _parse_results(raw: Any) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _node_titles(store: Any, node_ids: list[str]) -> dict[str, str]:
    if not node_ids:
        return {}
    placeholders = ", ".join("?" for _ in node_ids)
    rows = store.connection.execute(
        f"SELECT id, title FROM nodes WHERE id IN ({placeholders})",
        node_ids,
    ).fetchall()
    return {str(row["id"]): str(row["title"]) for row in rows}


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + _WILSON_Z**2 / total
    center = (p + _WILSON_Z**2 / (2 * total)) / denominator
    spread = _WILSON_Z * math.sqrt(p * (1 - p) / total + _WILSON_Z**2 / (4 * total**2)) / denominator
    return (round(max(0.0, center - spread), 4), round(min(1.0, center + spread), 4))


AUDIT_SYSTEM_PROMPT = (
    "You judge whether a retrieved memory node is relevant to the search query."
)
# LocalLLMDecider.sample_json sends only the user message (system prompt is
# dropped), so the JSON contract must live in the user prompt itself.
_AUDIT_JSON_CONTRACT = (
    'Return exactly one JSON object: {"relevance": <0|1|2>, "reason": "<one line>"} where '
    "0 = irrelevant, 1 = related but not directly useful, 2 = directly useful for the query."
)


def _audit_prompt(query: str, node_title: str, node_content: str) -> str:
    content_excerpt = node_content.strip()[:2000]
    return (
        f"{_AUDIT_JSON_CONTRACT}\n\n"
        f"Search query:\n{query.strip()}\n\n"
        f"Retrieved node title:\n{node_title.strip()}\n\n"
        f"Retrieved node content (excerpt):\n{content_excerpt}\n\n"
        "Judge the relevance of the node to the query. Respond with the JSON object only."
    )


def _parse_audit_judgement(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Accept either the raw HTTP envelope or the already-extracted JSON object
    (LocalLLMDecider.sample_json returns the latter)."""

    candidate: Any = payload
    if "relevance" not in payload:
        try:
            from synapse.server.decider import LocalLLMDecider
            from synapse.server.sampling import _extract_json_payload

            candidate = _extract_json_payload(LocalLLMDecider._extract_content(payload))
        except (ValueError, KeyError, IndexError, TypeError):
            return None
    if not isinstance(candidate, dict):
        return None
    relevance = candidate.get("relevance")
    try:
        relevance = int(relevance)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if relevance not in (0, 1, 2):
        return None
    reason = str(candidate.get("reason") or "").strip()
    return {"relevance": relevance, "reason": reason}


def run_injection_audit(
    config: SynapseConfig,
    runtime_paths: RuntimePaths,
    *,
    since_days: int,
    sample_size: int = 30,
    report_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Sample injected results and LLM-judge their relevance (0/1/2).

    Appends a summary line to the audit history JSONL and returns the summary.
    """

    from synapse.server.decider import LocalLLMDecider

    cutoff = _utc_cutoff(since_days)
    store = _open_store(config, runtime_paths)
    try:
        rows = store.connection.execute(
            "SELECT id, ts, source, query, results FROM search_events WHERE ts >= ? AND source IN (?, ?) ORDER BY ts",
            (cutoff, *_AUDIT_SOURCE_PREFIXES[:2]),
        ).fetchall()
    finally:
        store.close()

    candidates: list[dict[str, Any]] = []
    for row in rows:
        for item in _parse_results(row["results"]):
            if item.get("inject"):
                candidates.append(
                    {
                        "event_id": int(row["id"]),
                        "ts": str(row["ts"]),
                        "source": str(row["source"]),
                        "query": str(row["query"]),
                        "node_id": str(item.get("node_id") or ""),
                        "rerank_logit": item.get("rerank_logit"),
                        "score": item.get("score"),
                    }
                )
    rng = random.Random(seed)
    if len(candidates) > sample_size:
        sampled = rng.sample(candidates, sample_size)
    else:
        sampled = candidates

    decider = LocalLLMDecider(config.decider)
    judgements: list[dict[str, Any]] = []
    node_cache: dict[str, dict[str, Any] | None] = {}
    store = _open_store(config, runtime_paths)
    try:
        for candidate in sampled:
            node_id = candidate["node_id"]
            if node_id not in node_cache:
                row = store.connection.execute(
                    "SELECT title, content, okf_meta FROM nodes WHERE id = ?",
                    (node_id,),
                ).fetchone()
                if row is None:
                    node_cache[node_id] = None
                else:
                    try:
                        okf_meta = json.loads(str(row["okf_meta"] or "{}"))
                    except json.JSONDecodeError:
                        okf_meta = {}
                    node_cache[node_id] = {
                        "title": str(row["title"]),
                        "content": str(row["content"]),
                        "okf_type": str(okf_meta.get("okf_type") or "untyped"),
                    }
            node = node_cache[node_id]
            if node is None:
                continue
            try:
                payload = decider.sample_json(
                    prompt=_audit_prompt(candidate["query"], str(node["title"]), str(node["content"])),
                    system_prompt=AUDIT_SYSTEM_PROMPT,
                    # Reasoning models spend thinking tokens from the same
                    # budget: a judge prompt measured ~100 reasoning tokens
                    # before any content at max_tokens=100 (finish_reason
                    # "length", empty content).
                    max_tokens=2000,
                )
            except Exception:
                continue
            parsed = _parse_audit_judgement(payload)
            if parsed is None:
                continue
            judgements.append({**candidate, **parsed, "node_title": str(node["title"]), "okf_type": str(node["okf_type"])})
    finally:
        store.close()

    judged = len(judgements)
    related = sum(1 for item in judgements if item["relevance"] >= 1)
    useful = sum(1 for item in judgements if item["relevance"] == 2)
    related_low, related_high = _wilson_interval(related, judged)
    useful_low, useful_high = _wilson_interval(useful, judged)

    by_source: dict[str, dict[str, Any]] = {}
    by_okf_type: dict[str, dict[str, Any]] = {}
    for item in judgements:
        for bucket, key in ((by_source, item["source"]), (by_okf_type, str(item.get("okf_type") or "unknown"))):
            stats = bucket.setdefault(key, {"judged": 0, "related": 0, "useful": 0})
            stats["judged"] += 1
            if item["relevance"] >= 1:
                stats["related"] += 1
            if item["relevance"] == 2:
                stats["useful"] += 1

    worst = sorted(judgements, key=lambda item: (item["relevance"], item["node_id"]))[:10]
    summary = {
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "since_days": since_days,
        "sample_requested": sample_size,
        "candidates_available": len(candidates),
        "judged": judged,
        "precision_related": round(related / judged, 4) if judged else None,
        "precision_useful": round(useful / judged, 4) if judged else None,
        "precision_related_wilson95": [related_low, related_high] if judged else None,
        "precision_useful_wilson95": [useful_low, useful_high] if judged else None,
        "by_source": by_source,
        "by_okf_type": by_okf_type,
    }

    history_path = config.observability.resolved_audit_history_path(config)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    if report_path is not None:
        out = Path(report_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_audit_report(summary, worst), encoding="utf-8")
    return summary


def _render_audit_report(summary: dict[str, Any], worst: list[dict[str, Any]]) -> str:
    lines = [
        f"# Injection audit — {summary['ts']}",
        "",
        f"- Window: last {summary['since_days']} days",
        f"- Candidates available: {summary['candidates_available']}, judged: {summary['judged']}",
        f"- Precision (related, relevance>=1): {summary['precision_related']} "
        f"(95% Wilson {summary['precision_related_wilson95']})",
        f"- Precision (directly useful, relevance=2): {summary['precision_useful']} "
        f"(95% Wilson {summary['precision_useful_wilson95']})",
        "",
        "## By source",
        "",
    ]
    for source, stats in sorted(summary["by_source"].items()):
        lines.append(f"- `{source}`: {stats}")
    lines.extend(["", "## By okf_type", ""])
    for okf_type, stats in sorted(summary["by_okf_type"].items()):
        lines.append(f"- `{okf_type}`: {stats}")
    lines.extend(["", "## Worst cases (lowest relevance)", ""])
    for item in worst:
        lines.append(
            f"- relevance={item['relevance']} [{item['source']}] `{item['node_id']}` — {item['reason']} (query: {item['query'][:120]})"
        )
    return "\n".join(lines) + "\n"
