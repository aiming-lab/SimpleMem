# pyright: reportMissingImports=false
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import iris.dbapi as dbapi

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
import config

logger = logging.getLogger(__name__)

_local = threading.local()


def _thread_conn() -> dbapi.Connection:
    if not getattr(_local, "conn", None):
        _local.conn = dbapi.connect(
            config.IRIS_HOSTNAME,
            config.IRIS_PORT,
            config.IRIS_NAMESPACE,
            config.IRIS_USERNAME,
            config.IRIS_PASSWORD,
        )
    return _local.conn


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS CrossMem_sessions (
        id                  INTEGER IDENTITY PRIMARY KEY,
        tenant_id           VARCHAR(256) NOT NULL DEFAULT 'default',
        content_session_id  VARCHAR(256) NOT NULL,
        memory_session_id   VARCHAR(256) NOT NULL,
        project             VARCHAR(512) NOT NULL,
        user_prompt         VARCHAR(32000),
        started_at          VARCHAR(64) NOT NULL,
        ended_at            VARCHAR(64),
        status              VARCHAR(32) DEFAULT 'active',
        metadata_json       VARCHAR(32000)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS CrossMem_session_events (
        event_id          INTEGER IDENTITY PRIMARY KEY,
        memory_session_id VARCHAR(256) NOT NULL,
        timestamp         VARCHAR(64) NOT NULL,
        kind              VARCHAR(32) NOT NULL,
        title             VARCHAR(1024),
        payload_json      VARCHAR(32000),
        redaction_level   VARCHAR(32) DEFAULT 'none'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS CrossMem_observations (
        obs_id            INTEGER IDENTITY PRIMARY KEY,
        memory_session_id VARCHAR(256) NOT NULL,
        timestamp         VARCHAR(64) NOT NULL,
        type              VARCHAR(32) NOT NULL,
        title             VARCHAR(1024) NOT NULL,
        subtitle          VARCHAR(1024),
        facts_json        VARCHAR(32000),
        narrative         VARCHAR(32000),
        concepts_json     VARCHAR(4000),
        files_json        VARCHAR(4000),
        vector_ref        VARCHAR(256)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS CrossMem_session_summaries (
        summary_id        INTEGER IDENTITY PRIMARY KEY,
        memory_session_id VARCHAR(256) NOT NULL,
        timestamp         VARCHAR(64) NOT NULL,
        request           VARCHAR(32000),
        investigated      VARCHAR(32000),
        learned           VARCHAR(32000),
        completed         VARCHAR(32000),
        next_steps        VARCHAR(32000),
        vector_ref        VARCHAR(256)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS CrossMem_memory_links (
        link_id           INTEGER IDENTITY PRIMARY KEY,
        memory_entry_id   VARCHAR(256) NOT NULL,
        source_kind       VARCHAR(64) NOT NULL,
        source_id         INTEGER NOT NULL,
        score             DOUBLE DEFAULT 0.0,
        timestamp         VARCHAR(64) NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS CrossMem_consolidation_runs (
        run_id      INTEGER IDENTITY PRIMARY KEY,
        tenant_id   VARCHAR(256) NOT NULL DEFAULT 'default',
        timestamp   VARCHAR(64) NOT NULL,
        policy_json VARCHAR(32000),
        stats_json  VARCHAR(32000)
    )
    """,
]

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_cmsess_tenant ON CrossMem_sessions(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_cmsess_content ON CrossMem_sessions(content_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_cmsess_memory  ON CrossMem_sessions(memory_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_cmsess_project ON CrossMem_sessions(project)",
    "CREATE INDEX IF NOT EXISTS idx_cmsess_status  ON CrossMem_sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_cmev_session   ON CrossMem_session_events(memory_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_cmev_kind      ON CrossMem_session_events(kind)",
    "CREATE INDEX IF NOT EXISTS idx_cmobs_session  ON CrossMem_observations(memory_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_cmobs_type     ON CrossMem_observations(type)",
    "CREATE INDEX IF NOT EXISTS idx_cmsum_session  ON CrossMem_session_summaries(memory_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_cmlink_entry   ON CrossMem_memory_links(memory_entry_id)",
    "CREATE INDEX IF NOT EXISTS idx_cmlink_source  ON CrossMem_memory_links(source_kind, source_id)",
]


class IRISSQLStorage:
    """
    IRIS SQL backend for cross-session memory metadata.

    Drop-in replacement for SQLiteStorage — identical public interface.
    Uses thread-local IRIS connections (one per thread).
    Tables are prefixed CrossMem_ to avoid collisions with user schemas.
    """

    def __init__(self, table_prefix: str = "CrossMem"):
        self.table_prefix = table_prefix
        self._ensure_schema()
        logger.info("IRISSQLStorage initialised (IRIS backend)")

    def __enter__(self) -> "IRISSQLStorage":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        conn = getattr(_local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            _local.conn = None

    # ------------------------------------------------------------------ schema

    def _ensure_schema(self) -> None:
        cur = _thread_conn().cursor()
        try:
            for stmt in _SCHEMA:
                try:
                    cur.execute(stmt)
                    _thread_conn().commit()
                except Exception:
                    pass
            for stmt in _INDEXES:
                try:
                    cur.execute(stmt)
                    _thread_conn().commit()
                except Exception:
                    pass
        finally:
            cur.close()

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
        cur = _thread_conn().cursor()
        try:
            # Check if content_session_id already exists (INSERT OR IGNORE equiv)
            cur.execute(
                "SELECT COUNT(*) FROM CrossMem_sessions WHERE content_session_id = ?",
                [content_session_id],
            )
            if cur.fetchone()[0] == 0:
                cur.execute(
                    """
                    INSERT INTO CrossMem_sessions
                        (tenant_id, content_session_id, memory_session_id, project,
                         user_prompt, started_at, status, metadata_json)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    [tenant_id, content_session_id, memory_session_id, project,
                     user_prompt, started_at, "active", metadata_json],
                )
                _thread_conn().commit()
        except Exception:
            logger.exception("Failed to create session")
            _thread_conn().commit()  # ensure no dangling tx
            raise
        finally:
            cur.close()
        session = self.get_session_by_content_id(content_session_id)
        if session is None:
            raise RuntimeError("Failed to retrieve session after insert")
        return session

    def get_session_by_content_id(self, content_session_id: str) -> Optional[SessionRecord]:
        return self._fetch_session(
            "SELECT * FROM CrossMem_sessions WHERE content_session_id = ?",
            [content_session_id],
        )

    def get_session_by_memory_id(self, memory_session_id: str) -> Optional[SessionRecord]:
        return self._fetch_session(
            "SELECT * FROM CrossMem_sessions WHERE memory_session_id = ?",
            [memory_session_id],
        )

    def get_session_by_id(self, session_id: int) -> Optional[SessionRecord]:
        return self._fetch_session(
            "SELECT * FROM CrossMem_sessions WHERE id = ?",
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
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                "UPDATE CrossMem_sessions SET status=?, ended_at=? WHERE memory_session_id=?",
                [status_val, ended_at, memory_session_id],
            )
            _thread_conn().commit()
        except Exception:
            logger.exception("Failed to update session status")
            raise
        finally:
            cur.close()

    def list_sessions(
        self,
        tenant_id: Optional[str] = None,
        project: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionRecord]:
        where, params = _build_where(
            tenant_id=("tenant_id", tenant_id),
            project=("project", project),
            status=("status", _ev(status) if status else None),
        )
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                f"SELECT TOP {limit} * FROM CrossMem_sessions {where} ORDER BY started_at DESC",
                params,
            )
            return [_row_to_session(dict(zip([d[0] for d in cur.description], r)))
                    for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to list sessions")
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------ events

    def add_event(
        self,
        memory_session_id: str,
        kind: EventKind,
        title: Optional[str] = None,
        payload_json: Optional[dict] = None,
        redaction_level: Optional[RedactionLevel] = None,
    ) -> int:
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                """
                INSERT INTO CrossMem_session_events
                    (memory_session_id, timestamp, kind, title, payload_json, redaction_level)
                VALUES (?,?,?,?,?,?)
                """,
                [memory_session_id, _now_iso(), _ev(kind),
                 title,
                 json.dumps(payload_json) if payload_json is not None else None,
                 _ev(redaction_level, "none")],
            )
            _thread_conn().commit()
            return cur.lastrowid
        except Exception:
            logger.exception("Failed to add event")
            raise
        finally:
            cur.close()

    def get_events_for_session(
        self,
        memory_session_id: str,
        kinds: Optional[Sequence[EventKind]] = None,
    ) -> list[SessionEvent]:
        params: list = [memory_session_id]
        kind_clause = ""
        if kinds:
            placeholders = ",".join(["?"] * len(kinds))
            kind_clause = f" AND kind IN ({placeholders})"
            params.extend([k.value if hasattr(k, "value") else str(k) for k in kinds])
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                f"SELECT * FROM CrossMem_session_events WHERE memory_session_id=?"
                f"{kind_clause} ORDER BY timestamp ASC",
                params,
            )
            return [_row_to_event(dict(zip([d[0] for d in cur.description], r)))
                    for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch events")
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------- observations

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
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                """
                INSERT INTO CrossMem_observations
                    (memory_session_id, timestamp, type, title, subtitle,
                     facts_json, narrative, concepts_json, files_json, vector_ref)
                VALUES (?,?,?,?,?,?,?,?,?,?)
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
            _thread_conn().commit()
            return cur.lastrowid
        except Exception:
            logger.exception("Failed to store observation")
            raise
        finally:
            cur.close()

    def get_observations_for_session(self, memory_session_id: str) -> list[CrossObservation]:
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                "SELECT * FROM CrossMem_observations WHERE memory_session_id=? ORDER BY timestamp ASC",
                [memory_session_id],
            )
            return [_row_to_observation(dict(zip([d[0] for d in cur.description], r)))
                    for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch observations for session")
            raise
        finally:
            cur.close()

    def get_recent_observations(
        self,
        project: str,
        limit: int = 50,
        types: Optional[Sequence[ObservationType]] = None,
    ) -> list[CrossObservation]:
        params: list = [project]
        type_clause = ""
        if types:
            placeholders = ",".join(["?"] * len(types))
            type_clause = f" AND CrossMem_observations.type IN ({placeholders})"
            params.extend([t.value if hasattr(t, "value") else str(t) for t in types])
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                f"""
                SELECT TOP {limit} CrossMem_observations.* FROM CrossMem_observations
                JOIN CrossMem_sessions
                  ON CrossMem_sessions.memory_session_id = CrossMem_observations.memory_session_id
                WHERE CrossMem_sessions.project = ?{type_clause}
                ORDER BY CrossMem_observations.timestamp DESC
                """,
                params,
            )
            return [_row_to_observation(dict(zip([d[0] for d in cur.description], r)))
                    for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch recent observations")
            raise
        finally:
            cur.close()

    def get_observations_by_ids(self, obs_ids: list[int]) -> list[CrossObservation]:
        if not obs_ids:
            return []
        placeholders = ",".join(["?"] * len(obs_ids))
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                f"SELECT * FROM CrossMem_observations WHERE obs_id IN ({placeholders})",
                obs_ids,
            )
            return [_row_to_observation(dict(zip([d[0] for d in cur.description], r)))
                    for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch observations by ids")
            raise
        finally:
            cur.close()

    # --------------------------------------------------------------- summaries

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
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                """
                INSERT INTO CrossMem_session_summaries
                    (memory_session_id, timestamp, request, investigated,
                     learned, completed, next_steps, vector_ref)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                [memory_session_id, _now_iso(), request, investigated,
                 learned, completed, next_steps, vector_ref],
            )
            _thread_conn().commit()
            return cur.lastrowid
        except Exception:
            logger.exception("Failed to store summary")
            raise
        finally:
            cur.close()

    def get_summary_for_session(self, memory_session_id: str) -> Optional[SessionSummary]:
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                """
                SELECT TOP 1 * FROM CrossMem_session_summaries
                WHERE memory_session_id=? ORDER BY timestamp DESC
                """,
                [memory_session_id],
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_summary(dict(zip([d[0] for d in cur.description], row)))
        except Exception:
            logger.exception("Failed to fetch session summary")
            raise
        finally:
            cur.close()

    def get_recent_summaries(self, project: str, limit: int = 10) -> list[SessionSummary]:
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                f"""
                SELECT TOP {limit} CrossMem_session_summaries.* FROM CrossMem_session_summaries
                JOIN CrossMem_sessions
                  ON CrossMem_sessions.memory_session_id = CrossMem_session_summaries.memory_session_id
                WHERE CrossMem_sessions.project = ?
                ORDER BY CrossMem_session_summaries.timestamp DESC
                """,
                [project],
            )
            return [_row_to_summary(dict(zip([d[0] for d in cur.description], r)))
                    for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch recent summaries")
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------- links

    def create_link(
        self,
        memory_entry_id: str,
        source_kind: str,
        source_id: int,
        score: float = 0.0,
    ) -> int:
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                """
                INSERT INTO CrossMem_memory_links
                    (memory_entry_id, source_kind, source_id, score, timestamp)
                VALUES (?,?,?,?,?)
                """,
                [memory_entry_id, source_kind, source_id, score, _now_iso()],
            )
            _thread_conn().commit()
            return cur.lastrowid
        except Exception:
            logger.exception("Failed to create link")
            raise
        finally:
            cur.close()

    def get_links_for_entry(self, memory_entry_id: str) -> list[MemoryLink]:
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                "SELECT * FROM CrossMem_memory_links WHERE memory_entry_id=? ORDER BY score DESC, timestamp DESC",
                [memory_entry_id],
            )
            return [_row_to_link(dict(zip([d[0] for d in cur.description], r)))
                    for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch links for entry")
            raise
        finally:
            cur.close()

    def get_links_for_source(self, source_kind: str, source_id: int) -> list[MemoryLink]:
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                "SELECT * FROM CrossMem_memory_links WHERE source_kind=? AND source_id=? ORDER BY score DESC",
                [source_kind, source_id],
            )
            return [_row_to_link(dict(zip([d[0] for d in cur.description], r)))
                    for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch links for source")
            raise
        finally:
            cur.close()

    # ---------------------------------------------------------- consolidation

    def record_consolidation_run(
        self,
        tenant_id: str,
        policy_json: Optional[dict] = None,
        stats_json: Optional[dict] = None,
    ) -> int:
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                """
                INSERT INTO CrossMem_consolidation_runs
                    (tenant_id, timestamp, policy_json, stats_json)
                VALUES (?,?,?,?)
                """,
                [tenant_id, _now_iso(),
                 json.dumps(policy_json) if policy_json is not None else None,
                 json.dumps(stats_json) if stats_json is not None else None],
            )
            _thread_conn().commit()
            return cur.lastrowid
        except Exception:
            logger.exception("Failed to record consolidation run")
            raise
        finally:
            cur.close()

    def get_recent_consolidation_runs(self, tenant_id: str, limit: int = 10) -> list[ConsolidationRun]:
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                f"SELECT TOP {limit} * FROM CrossMem_consolidation_runs WHERE tenant_id=? ORDER BY timestamp DESC",
                [tenant_id],
            )
            return [_row_to_consolidation_run(dict(zip([d[0] for d in cur.description], r)))
                    for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch consolidation runs")
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------- stats

    def get_stats(
        self, tenant_id: Optional[str] = None, project: Optional[str] = None
    ) -> dict[str, int]:
        return {
            "sessions":     self._count("CrossMem_sessions", tenant_id, project),
            "events":       self._count_joined("CrossMem_session_events", tenant_id, project),
            "observations": self._count_joined("CrossMem_observations", tenant_id, project),
            "summaries":    self._count_joined("CrossMem_session_summaries", tenant_id, project),
        }

    def _count(self, table: str, tenant_id: Optional[str], project: Optional[str]) -> int:
        where, params = _build_where(
            tenant_id=("tenant_id", tenant_id),
            project=("project", project),
        )
        cur = _thread_conn().cursor()
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table} {where}", params)
            return int(cur.fetchone()[0])
        finally:
            cur.close()

    def _count_joined(self, table: str, tenant_id: Optional[str], project: Optional[str]) -> int:
        where_clauses, params = [], []
        if tenant_id:
            where_clauses.append("CrossMem_sessions.tenant_id = ?"); params.append(tenant_id)
        if project:
            where_clauses.append("CrossMem_sessions.project = ?"); params.append(project)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        cur = _thread_conn().cursor()
        try:
            cur.execute(
                f"SELECT COUNT(*) FROM {table} "
                f"JOIN CrossMem_sessions ON CrossMem_sessions.memory_session_id = {table}.memory_session_id "
                f"{where}",
                params,
            )
            return int(cur.fetchone()[0])
        finally:
            cur.close()

    # --------------------------------------------------------- private helpers

    def _fetch_session(self, query: str, params: list) -> Optional[SessionRecord]:
        cur = _thread_conn().cursor()
        try:
            cur.execute(query, params)
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_session(dict(zip([d[0] for d in cur.description], row)))
        except Exception:
            logger.exception("Failed to fetch session")
            raise
        finally:
            cur.close()


# ----------------------------------------------------------------- row helpers

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ev(value, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    return value.value if hasattr(value, "value") else str(value)


def _build_where(**kwargs) -> tuple[str, list]:
    clauses, params = [], []
    for _, (col, val) in kwargs.items():
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    return (("WHERE " + " AND ".join(clauses)) if clauses else ""), params


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
