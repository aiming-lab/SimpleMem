"""
PostgreSQL metadata backend for cross-session memory.

Drop-in replacement for SQLiteStorage (storage_sqlite.py) and IRISSQLStorage
(storage_iris_sql.py) backed by standard PostgreSQL.

IRIS + pgwire: point PG_DSN at the pgwire endpoint.

Requires: psycopg[binary]
  pip install "psycopg[binary]"
  or: pip install simplemem[pgvector]
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from simplemem.core.settings import settings as config
from .types import (
    ConsolidationRun,
    CrossObservation,
    EventKind,
    MemoryLink,
    ObservationType,
    RedactionLevel,
    SessionEvent,
    SessionRecord,
    SessionStatus,
    SessionSummary,
)

logger = logging.getLogger(__name__)

_local = threading.local()


def _get_conn(dsn: str):
    if not getattr(_local, "conn", None) or _local.conn.closed:
        try:
            import psycopg
        except ImportError as e:
            raise ImportError(
                "psycopg is required for the PostgreSQL cross-session backend. "
                "Install with: pip install 'psycopg[binary]'"
            ) from e
        _local.conn = psycopg.connect(dsn, autocommit=False)
    return _local.conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ev(value, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    return value.value if hasattr(value, "value") else str(value)


def _coerce(enum_cls, value):
    if value is None or isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except Exception:
        return value


def _build(model_cls, data: dict):
    if hasattr(model_cls, "model_fields"):
        allowed = set(model_cls.model_fields.keys())
    elif hasattr(model_cls, "__fields__"):
        allowed = set(model_cls.__fields__.keys())
    else:
        allowed = set(model_cls.__annotations__.keys())
    return model_cls(**{k: v for k, v in data.items() if k in allowed})


def _row_to_session(d: dict) -> SessionRecord:
    d["status"] = _coerce(SessionStatus, d.get("status"))
    return _build(SessionRecord, d)


def _row_to_event(d: dict) -> SessionEvent:
    d["kind"] = _coerce(EventKind, d.get("kind"))
    d["redaction_level"] = _coerce(RedactionLevel, d.get("redaction_level"))
    return _build(SessionEvent, d)


def _row_to_observation(d: dict) -> CrossObservation:
    d["type"] = _coerce(ObservationType, d.get("type"))
    return _build(CrossObservation, d)


def _row_to_summary(d: dict) -> SessionSummary:
    return _build(SessionSummary, d)


def _row_to_link(d: dict) -> MemoryLink:
    return _build(MemoryLink, d)


def _row_to_consolidation_run(d: dict) -> ConsolidationRun:
    return _build(ConsolidationRun, d)


def _fetchall_dicts(cur) -> list[dict]:
    cols = [desc.name for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _fetchone_dict(cur) -> Optional[dict]:
    cols = [desc.name for desc in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


class PGSQLStorage:
    """
    PostgreSQL metadata backend for cross-session memory.

    Identical public interface to SQLiteStorage and IRISSQLStorage.
    All tables are prefixed with ``{table_prefix}_`` to avoid collisions.

    Parameters
    ----------
    dsn:
        libpq connection string. Defaults to ``PG_DSN`` setting.
    table_prefix:
        Prefix for all table names. Defaults to ``"cross_mem"``.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        table_prefix: str = "cross_mem",
    ):
        self._dsn = dsn or config.PG_DSN
        self._p = table_prefix
        self._ensure_schema()
        logger.info("PGSQLStorage initialised (PostgreSQL backend, prefix=%s)", self._p)

    def __enter__(self) -> "PGSQLStorage":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        conn = getattr(_local, "conn", None)
        if conn and not conn.closed:
            try:
                conn.close()
            except Exception:
                pass
        _local.conn = None

    # ------------------------------------------------------------------ schema

    def _ensure_schema(self) -> None:
        p = self._p
        conn = _get_conn(self._dsn)
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {p}_sessions (
                    id                  SERIAL PRIMARY KEY,
                    tenant_id           TEXT NOT NULL DEFAULT 'default',
                    content_session_id  TEXT UNIQUE NOT NULL,
                    memory_session_id   TEXT UNIQUE NOT NULL,
                    project             TEXT NOT NULL,
                    user_prompt         TEXT,
                    started_at          TEXT NOT NULL,
                    ended_at            TEXT,
                    status              TEXT DEFAULT 'active',
                    metadata_json       TEXT
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {p}_session_events (
                    event_id          SERIAL PRIMARY KEY,
                    memory_session_id TEXT NOT NULL,
                    timestamp         TEXT NOT NULL,
                    kind              TEXT NOT NULL,
                    title             TEXT,
                    payload_json      TEXT,
                    redaction_level   TEXT DEFAULT 'none'
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {p}_observations (
                    obs_id            SERIAL PRIMARY KEY,
                    memory_session_id TEXT NOT NULL,
                    timestamp         TEXT NOT NULL,
                    type              TEXT NOT NULL,
                    title             TEXT NOT NULL,
                    subtitle          TEXT,
                    facts_json        TEXT,
                    narrative         TEXT,
                    concepts_json     TEXT,
                    files_json        TEXT,
                    vector_ref        TEXT
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {p}_session_summaries (
                    summary_id        SERIAL PRIMARY KEY,
                    memory_session_id TEXT NOT NULL,
                    timestamp         TEXT NOT NULL,
                    request           TEXT,
                    investigated      TEXT,
                    learned           TEXT,
                    completed         TEXT,
                    next_steps        TEXT,
                    vector_ref        TEXT
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {p}_memory_links (
                    link_id           SERIAL PRIMARY KEY,
                    memory_entry_id   TEXT NOT NULL,
                    source_kind       TEXT NOT NULL,
                    source_id         INTEGER NOT NULL,
                    score             DOUBLE PRECISION DEFAULT 0.0,
                    timestamp         TEXT NOT NULL
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {p}_consolidation_runs (
                    run_id      SERIAL PRIMARY KEY,
                    tenant_id   TEXT NOT NULL DEFAULT 'default',
                    timestamp   TEXT NOT NULL,
                    policy_json TEXT,
                    stats_json  TEXT
                )
            """)
            # Indexes
            for stmt in [
                f"CREATE INDEX IF NOT EXISTS {p}_idx_sess_tenant  ON {p}_sessions(tenant_id)",
                f"CREATE INDEX IF NOT EXISTS {p}_idx_sess_project ON {p}_sessions(project)",
                f"CREATE INDEX IF NOT EXISTS {p}_idx_sess_status  ON {p}_sessions(status)",
                f"CREATE INDEX IF NOT EXISTS {p}_idx_ev_session   ON {p}_session_events(memory_session_id)",
                f"CREATE INDEX IF NOT EXISTS {p}_idx_ev_kind      ON {p}_session_events(kind)",
                f"CREATE INDEX IF NOT EXISTS {p}_idx_obs_session  ON {p}_observations(memory_session_id)",
                f"CREATE INDEX IF NOT EXISTS {p}_idx_obs_type     ON {p}_observations(type)",
                f"CREATE INDEX IF NOT EXISTS {p}_idx_sum_session  ON {p}_session_summaries(memory_session_id)",
                f"CREATE INDEX IF NOT EXISTS {p}_idx_link_entry   ON {p}_memory_links(memory_entry_id)",
                f"CREATE INDEX IF NOT EXISTS {p}_idx_link_source  ON {p}_memory_links(source_kind, source_id)",
            ]:
                cur.execute(stmt)
        conn.commit()

    def _conn(self):
        return _get_conn(self._dsn)

    def _cur(self):
        return self._conn().cursor()

    def _commit(self) -> None:
        self._conn().commit()

    def _rollback(self) -> None:
        try:
            self._conn().rollback()
        except Exception:
            pass

    # ---------------------------------------------------------------- sessions

    def create_session(
        self,
        tenant_id: str,
        content_session_id: str,
        project: str,
        user_prompt: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SessionRecord:
        memory_session_id = str(uuid4())
        started_at = _now_iso()
        metadata_json = json.dumps(metadata) if metadata is not None else None
        try:
            with self._cur() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._p}_sessions
                        (tenant_id, content_session_id, memory_session_id, project,
                         user_prompt, started_at, status, metadata_json)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (content_session_id) DO NOTHING
                    """,
                    [tenant_id, content_session_id, memory_session_id, project,
                     user_prompt, started_at, "active", metadata_json],
                )
            self._commit()
        except Exception:
            logger.exception("create_session failed")
            self._rollback()
            raise
        session = self.get_session_by_content_id(content_session_id)
        if session is None:
            raise RuntimeError("Failed to retrieve session after insert")
        return session

    def get_session_by_content_id(self, content_session_id: str) -> Optional[SessionRecord]:
        return self._fetch_session(
            f"SELECT * FROM {self._p}_sessions WHERE content_session_id = %s",
            [content_session_id],
        )

    def get_session_by_memory_id(self, memory_session_id: str) -> Optional[SessionRecord]:
        return self._fetch_session(
            f"SELECT * FROM {self._p}_sessions WHERE memory_session_id = %s",
            [memory_session_id],
        )

    def get_session_by_id(self, session_id: int) -> Optional[SessionRecord]:
        return self._fetch_session(
            f"SELECT * FROM {self._p}_sessions WHERE id = %s",
            [session_id],
        )

    def update_session_status(
        self,
        memory_session_id: str,
        status: SessionStatus,
        ended_at: Optional[str] = None,
    ) -> None:
        status_val = _ev(status)
        if ended_at is None and status_val in {"completed", "failed"}:
            ended_at = _now_iso()
        try:
            with self._cur() as cur:
                cur.execute(
                    f"UPDATE {self._p}_sessions SET status=%s, ended_at=%s WHERE memory_session_id=%s",
                    [status_val, ended_at, memory_session_id],
                )
            self._commit()
        except Exception:
            logger.exception("update_session_status failed")
            self._rollback()
            raise

    def list_sessions(
        self,
        tenant_id: Optional[str] = None,
        project: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionRecord]:
        clauses, params = [], []
        if tenant_id:
            clauses.append("tenant_id = %s"); params.append(tenant_id)
        if project:
            clauses.append("project = %s"); params.append(project)
        if status:
            clauses.append("status = %s"); params.append(_ev(status))
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT * FROM {self._p}_sessions {where} "
                    "ORDER BY started_at DESC LIMIT %s OFFSET %s",
                    params,
                )
                return [_row_to_session(d) for d in _fetchall_dicts(cur)]
        except Exception:
            logger.exception("list_sessions failed")
            raise

    # ------------------------------------------------------------------ events

    def add_event(
        self,
        memory_session_id: str,
        kind: EventKind,
        title: Optional[str] = None,
        payload_json: Optional[dict] = None,
        redaction_level: Optional[RedactionLevel] = None,
    ) -> int:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._p}_session_events
                        (memory_session_id, timestamp, kind, title, payload_json, redaction_level)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    RETURNING event_id
                    """,
                    [memory_session_id, _now_iso(), _ev(kind), title,
                     json.dumps(payload_json) if payload_json is not None else None,
                     _ev(redaction_level, "none")],
                )
                row = cur.fetchone()
            self._commit()
            return row[0]
        except Exception:
            logger.exception("add_event failed")
            self._rollback()
            raise

    def get_events_for_session(
        self,
        memory_session_id: str,
        kinds: Optional[Sequence[EventKind]] = None,
    ) -> list[SessionEvent]:
        params: list = [memory_session_id]
        kind_clause = ""
        if kinds:
            placeholders = ",".join(["%s"] * len(kinds))
            kind_clause = f" AND kind IN ({placeholders})"
            params.extend([k.value if hasattr(k, "value") else str(k) for k in kinds])
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT * FROM {self._p}_session_events "
                    f"WHERE memory_session_id=%s{kind_clause} ORDER BY timestamp ASC",
                    params,
                )
                return [_row_to_event(d) for d in _fetchall_dicts(cur)]
        except Exception:
            logger.exception("get_events_for_session failed")
            raise

    # ----------------------------------------------------------- observations

    def store_observation(
        self,
        memory_session_id: str,
        type: ObservationType,
        title: str,
        subtitle: Optional[str] = None,
        facts_json: Optional[dict] = None,
        narrative: Optional[str] = None,
        concepts_json: Optional[Iterable[str]] = None,
        files_json: Optional[Iterable[str]] = None,
        vector_ref: Optional[str] = None,
    ) -> int:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._p}_observations
                        (memory_session_id, timestamp, type, title, subtitle,
                         facts_json, narrative, concepts_json, files_json, vector_ref)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING obs_id
                    """,
                    [memory_session_id, _now_iso(),
                     type.value if hasattr(type, "value") else str(type),
                     title, subtitle,
                     json.dumps(facts_json) if facts_json is not None else None,
                     narrative,
                     json.dumps(list(concepts_json)) if concepts_json is not None else None,
                     json.dumps(list(files_json)) if files_json is not None else None,
                     vector_ref],
                )
                row = cur.fetchone()
            self._commit()
            return row[0]
        except Exception:
            logger.exception("store_observation failed")
            self._rollback()
            raise

    def get_observations_for_session(self, memory_session_id: str) -> list[CrossObservation]:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT * FROM {self._p}_observations "
                    "WHERE memory_session_id=%s ORDER BY timestamp ASC",
                    [memory_session_id],
                )
                return [_row_to_observation(d) for d in _fetchall_dicts(cur)]
        except Exception:
            logger.exception("get_observations_for_session failed")
            raise

    def get_recent_observations(
        self,
        project: str,
        limit: int = 50,
        types: Optional[Sequence[ObservationType]] = None,
    ) -> list[CrossObservation]:
        params: list = [project]
        type_clause = ""
        if types:
            placeholders = ",".join(["%s"] * len(types))
            type_clause = f" AND o.type IN ({placeholders})"
            params.extend([t.value if hasattr(t, "value") else str(t) for t in types])
        params.append(limit)
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT o.* FROM {self._p}_observations o "
                    f"JOIN {self._p}_sessions s ON s.memory_session_id=o.memory_session_id "
                    f"WHERE s.project=%s{type_clause} ORDER BY o.timestamp DESC LIMIT %s",
                    params,
                )
                return [_row_to_observation(d) for d in _fetchall_dicts(cur)]
        except Exception:
            logger.exception("get_recent_observations failed")
            raise

    def get_observations_by_ids(self, obs_ids: list[int]) -> list[CrossObservation]:
        if not obs_ids:
            return []
        placeholders = ",".join(["%s"] * len(obs_ids))
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT * FROM {self._p}_observations WHERE obs_id IN ({placeholders})",
                    obs_ids,
                )
                return [_row_to_observation(d) for d in _fetchall_dicts(cur)]
        except Exception:
            logger.exception("get_observations_by_ids failed")
            raise

    # ------------------------------------------------------------- summaries

    def store_summary(
        self,
        memory_session_id: str,
        request: Optional[str] = None,
        investigated: Optional[str] = None,
        learned: Optional[str] = None,
        completed: Optional[str] = None,
        next_steps: Optional[str] = None,
        vector_ref: Optional[str] = None,
    ) -> int:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._p}_session_summaries
                        (memory_session_id, timestamp, request, investigated,
                         learned, completed, next_steps, vector_ref)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING summary_id
                    """,
                    [memory_session_id, _now_iso(), request, investigated,
                     learned, completed, next_steps, vector_ref],
                )
                row = cur.fetchone()
            self._commit()
            return row[0]
        except Exception:
            logger.exception("store_summary failed")
            self._rollback()
            raise

    def get_summary_for_session(self, memory_session_id: str) -> Optional[SessionSummary]:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT * FROM {self._p}_session_summaries "
                    "WHERE memory_session_id=%s ORDER BY timestamp DESC LIMIT 1",
                    [memory_session_id],
                )
                d = _fetchone_dict(cur)
            return _row_to_summary(d) if d else None
        except Exception:
            logger.exception("get_summary_for_session failed")
            raise

    def get_recent_summaries(self, project: str, limit: int = 10) -> list[SessionSummary]:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT ss.* FROM {self._p}_session_summaries ss "
                    f"JOIN {self._p}_sessions s ON s.memory_session_id=ss.memory_session_id "
                    "WHERE s.project=%s ORDER BY ss.timestamp DESC LIMIT %s",
                    [project, limit],
                )
                return [_row_to_summary(d) for d in _fetchall_dicts(cur)]
        except Exception:
            logger.exception("get_recent_summaries failed")
            raise

    # ------------------------------------------------------------------ links

    def create_link(
        self,
        memory_entry_id: str,
        source_kind: str,
        source_id: int,
        score: float = 0.0,
    ) -> int:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._p}_memory_links
                        (memory_entry_id, source_kind, source_id, score, timestamp)
                    VALUES (%s,%s,%s,%s,%s)
                    RETURNING link_id
                    """,
                    [memory_entry_id, source_kind, source_id, score, _now_iso()],
                )
                row = cur.fetchone()
            self._commit()
            return row[0]
        except Exception:
            logger.exception("create_link failed")
            self._rollback()
            raise

    def get_links_for_entry(self, memory_entry_id: str) -> list[MemoryLink]:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT * FROM {self._p}_memory_links "
                    "WHERE memory_entry_id=%s ORDER BY score DESC, timestamp DESC",
                    [memory_entry_id],
                )
                return [_row_to_link(d) for d in _fetchall_dicts(cur)]
        except Exception:
            logger.exception("get_links_for_entry failed")
            raise

    def get_links_for_source(self, source_kind: str, source_id: int) -> list[MemoryLink]:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT * FROM {self._p}_memory_links "
                    "WHERE source_kind=%s AND source_id=%s ORDER BY score DESC",
                    [source_kind, source_id],
                )
                return [_row_to_link(d) for d in _fetchall_dicts(cur)]
        except Exception:
            logger.exception("get_links_for_source failed")
            raise

    # --------------------------------------------------------- consolidation

    def record_consolidation_run(
        self,
        tenant_id: str,
        policy_json: Optional[dict] = None,
        stats_json: Optional[dict] = None,
    ) -> int:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._p}_consolidation_runs
                        (tenant_id, timestamp, policy_json, stats_json)
                    VALUES (%s,%s,%s,%s)
                    RETURNING run_id
                    """,
                    [tenant_id, _now_iso(),
                     json.dumps(policy_json) if policy_json is not None else None,
                     json.dumps(stats_json) if stats_json is not None else None],
                )
                row = cur.fetchone()
            self._commit()
            return row[0]
        except Exception:
            logger.exception("record_consolidation_run failed")
            self._rollback()
            raise

    def get_recent_consolidation_runs(self, tenant_id: str, limit: int = 10) -> list[ConsolidationRun]:
        try:
            with self._cur() as cur:
                cur.execute(
                    f"SELECT * FROM {self._p}_consolidation_runs "
                    "WHERE tenant_id=%s ORDER BY timestamp DESC LIMIT %s",
                    [tenant_id, limit],
                )
                return [_row_to_consolidation_run(d) for d in _fetchall_dicts(cur)]
        except Exception:
            logger.exception("get_recent_consolidation_runs failed")
            raise

    # ------------------------------------------------------------------- stats

    def get_stats(
        self, tenant_id: Optional[str] = None, project: Optional[str] = None
    ) -> dict[str, int]:
        return {
            "sessions":     self._count_sessions(tenant_id, project),
            "events":       self._count_joined(f"{self._p}_session_events", tenant_id, project),
            "observations": self._count_joined(f"{self._p}_observations", tenant_id, project),
            "summaries":    self._count_joined(f"{self._p}_session_summaries", tenant_id, project),
        }

    def _count_sessions(self, tenant_id: Optional[str], project: Optional[str]) -> int:
        clauses, params = [], []
        if tenant_id:
            clauses.append("tenant_id = %s"); params.append(tenant_id)
        if project:
            clauses.append("project = %s"); params.append(project)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._cur() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._p}_sessions {where}", params)
            return int(cur.fetchone()[0])

    def _count_joined(self, table: str, tenant_id: Optional[str], project: Optional[str]) -> int:
        clauses, params = [], []
        if tenant_id:
            clauses.append(f"{self._p}_sessions.tenant_id = %s"); params.append(tenant_id)
        if project:
            clauses.append(f"{self._p}_sessions.project = %s"); params.append(project)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._cur() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {table} "
                f"JOIN {self._p}_sessions "
                f"ON {self._p}_sessions.memory_session_id = {table}.memory_session_id {where}",
                params,
            )
            return int(cur.fetchone()[0])

    # ---------------------------------------------------------------- helpers

    def _fetch_session(self, query: str, params: list) -> Optional[SessionRecord]:
        try:
            with self._cur() as cur:
                cur.execute(query, params)
                d = _fetchone_dict(cur)
            return _row_to_session(d) if d else None
        except Exception:
            logger.exception("_fetch_session failed")
            raise

    def _purge_all_test_data(self) -> None:
        """Drop all tables for this prefix. Used only in tests."""
        p = self._p
        conn = _get_conn(self._dsn)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                for tbl in [
                    f"{p}_session_events",
                    f"{p}_observations",
                    f"{p}_session_summaries",
                    f"{p}_memory_links",
                    f"{p}_consolidation_runs",
                    f"{p}_sessions",
                ]:
                    cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        finally:
            conn.autocommit = False
