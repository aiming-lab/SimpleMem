"""
PostgreSQL/pgvector cross-session vector store.

Drop-in replacement for CrossSessionVectorStore (storage_iris.py) backed by
PostgreSQL + pgvector instead of IRIS.  Same public interface.

IRIS + pgwire: point PG_DSN at the pgwire endpoint — no IRIS-specific code needed.

Requires: psycopg[binary] pgvector
  pip install "psycopg[binary]" pgvector
  or: pip install simplemem[pgvector]
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import List, Optional

from simplemem.core.models.memory_entry import MemoryEntry
from simplemem.core.utils.embedding import EmbeddingModel
from simplemem.core.settings import settings as config
from cross.types import CrossMemoryEntry

logger = logging.getLogger(__name__)

_local = threading.local()


def _get_conn(dsn: str):
    if not getattr(_local, "conn", None) or _local.conn.closed:
        try:
            import psycopg
        except ImportError as e:
            raise ImportError(
                "psycopg is required for the pgvector cross-session backend. "
                "Install with: pip install 'psycopg[binary]' pgvector"
            ) from e
        _local.conn = psycopg.connect(dsn, autocommit=False)
    return _local.conn


_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    entry_id          TEXT        NOT NULL,
    text              TEXT        NOT NULL,
    keywords          TEXT,
    timestamp         TEXT,
    location          TEXT,
    persons           TEXT,
    entities          TEXT,
    topic             TEXT,
    tenant_id         TEXT        NOT NULL DEFAULT '',
    memory_session_id TEXT        NOT NULL DEFAULT '',
    source_kind       TEXT        NOT NULL DEFAULT '',
    source_id         INTEGER,
    importance        DOUBLE PRECISION DEFAULT 0.5,
    valid_from        TEXT,
    valid_to          TEXT,
    superseded_by     TEXT,
    vec               vector({dim}),
    tsv               tsvector
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
)
"""

_CREATE_VEC_IDX = """
CREATE INDEX IF NOT EXISTS {table}_vec_idx
    ON {table} USING hnsw (vec vector_cosine_ops)
"""

_CREATE_TSV_IDX = "CREATE INDEX IF NOT EXISTS {table}_tsv_idx ON {table} USING gin (tsv)"

_INSERT = """
INSERT INTO {table}
    (entry_id, text, keywords, timestamp, location, persons, entities, topic,
     tenant_id, memory_session_id, source_kind, source_id, importance,
     valid_from, valid_to, superseded_by, vec)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
ON CONFLICT DO NOTHING
"""

_SEMANTIC = """
SELECT entry_id, text, keywords, timestamp, location, persons, entities, topic,
       tenant_id, memory_session_id, source_kind, source_id, importance,
       valid_from, valid_to, superseded_by
FROM {table}
{where}
ORDER BY vec <-> %s::vector
LIMIT %s
"""

_KEYWORD = """
SELECT entry_id, text, keywords, timestamp, location, persons, entities, topic,
       tenant_id, memory_session_id, source_kind, source_id, importance,
       valid_from, valid_to, superseded_by
FROM {table}
WHERE tsv @@ plainto_tsquery('english', %s)
{extra_where}
LIMIT %s
"""

_UPDATE_SUPER = """
UPDATE {table} SET superseded_by = %s, valid_to = %s WHERE entry_id = %s
"""

_UPDATE_IMP = "UPDATE {table} SET importance = %s WHERE entry_id = %s"


def _enc(lst: list) -> str:
    return json.dumps(lst or [])


def _dec(s) -> list:
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


def _parse_dt(s) -> Optional[datetime]:
    if isinstance(s, datetime):
        return s
    if isinstance(s, str) and s:
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


def _fmt_dt(dt: Optional[datetime]) -> str:
    return dt.isoformat() if dt else ""


def _row_to_cross(row) -> CrossMemoryEntry:
    return CrossMemoryEntry(
        entry_id=row[0],
        lossless_restatement=row[1],
        keywords=_dec(row[2]),
        timestamp=row[3] or None,
        location=row[4] or None,
        persons=_dec(row[5]),
        entities=_dec(row[6]),
        topic=row[7] or None,
        tenant_id=row[8] or "",
        memory_session_id=row[9] or "",
        source_kind=row[10] or "",
        source_id=row[11],
        importance=float(row[12]) if row[12] is not None else 0.5,
        valid_from=_parse_dt(row[13]),
        valid_to=_parse_dt(row[14]),
        superseded_by=row[15] or None,
    )


class PGCrossVectorStore:
    """
    PostgreSQL + pgvector implementation of the cross-session vector store.

    Duck-typed to match CrossSessionVectorStore (storage_iris.py) so
    consolidation.py, context_injector.py, and orchestrator.py work
    without modification.

    Parameters
    ----------
    dsn:
        libpq connection string. Defaults to ``PG_DSN`` setting.
    embedding_model:
        EmbeddingModel instance. Defaults to a fresh EmbeddingModel().
    table_name:
        Table for cross-session memory vectors.
        Defaults to ``"cross_memory_entries"``.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        table_name: str = "cross_memory_entries",
    ):
        self._dsn = dsn or config.PG_DSN
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name
        self._dim = self.embedding_model.dimension
        self._ensure_schema()

    # ------------------------------------------------------------------ schema

    def _ensure_schema(self) -> None:
        conn = _get_conn(self._dsn)
        with conn.cursor() as cur:
            cur.execute(_CREATE_EXTENSION)
            cur.execute(_CREATE_TABLE.format(table=self.table_name, dim=self._dim))
            cur.execute(_CREATE_VEC_IDX.format(table=self.table_name))
            cur.execute(_CREATE_TSV_IDX.format(table=self.table_name))
        conn.commit()

    def _cur(self):
        return _get_conn(self._dsn).cursor()

    def _commit(self) -> None:
        _get_conn(self._dsn).commit()

    def _t(self, sql: str) -> str:
        return sql.format(table=self.table_name, dim=self._dim)

    def _build_where(
        self,
        tenant_id: Optional[str] = None,
        memory_session_id: Optional[str] = None,
    ) -> tuple[str, list]:
        conds, params = [], []
        if tenant_id:
            conds.append("tenant_id = %s")
            params.append(tenant_id)
        if memory_session_id:
            conds.append("memory_session_id = %s")
            params.append(memory_session_id)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        return where, params

    # ------------------------------------------------------------------- write

    def add_entries(
        self,
        entries: List[MemoryEntry],
        tenant_id: str,
        memory_session_id: str,
        source_kind: str,
        source_id: int = 0,
        importance: float = 0.5,
    ) -> None:
        if not entries:
            return
        try:
            vecs = self.embedding_model.encode_documents(
                [e.lossless_restatement for e in entries]
            )
            now = datetime.utcnow().isoformat()
            rows = [
                (
                    e.entry_id,
                    e.lossless_restatement,
                    _enc(e.keywords or []),
                    e.timestamp or "",
                    e.location or "",
                    _enc(e.persons or []),
                    _enc(e.entities or []),
                    e.topic or "",
                    tenant_id,
                    memory_session_id,
                    source_kind,
                    source_id,
                    float(importance),
                    now,
                    "",
                    "",
                    list(float(x) for x in vecs[i]),
                )
                for i, e in enumerate(entries)
            ]
            with self._cur() as cur:
                cur.executemany(self._t(_INSERT), rows)
            self._commit()
            logger.debug("Inserted %d cross entries", len(entries))
        except Exception:
            logger.exception("add_entries failed")

    def add_cross_entries(self, cross_entries: List[CrossMemoryEntry]) -> None:
        if not cross_entries:
            return
        try:
            vecs = self.embedding_model.encode_documents(
                [e.lossless_restatement for e in cross_entries]
            )
            rows = [
                (
                    e.entry_id,
                    e.lossless_restatement,
                    _enc(e.keywords or []),
                    e.timestamp or "",
                    e.location or "",
                    _enc(e.persons or []),
                    _enc(e.entities or []),
                    e.topic or "",
                    e.tenant_id,
                    e.memory_session_id,
                    e.source_kind,
                    e.source_id or 0,
                    float(e.importance),
                    _fmt_dt(e.valid_from),
                    _fmt_dt(e.valid_to),
                    e.superseded_by or "",
                    list(float(x) for x in vecs[i]),
                )
                for i, e in enumerate(cross_entries)
            ]
            with self._cur() as cur:
                cur.executemany(self._t(_INSERT), rows)
            self._commit()
            logger.debug("Inserted %d cross entries", len(cross_entries))
        except Exception:
            logger.exception("add_cross_entries failed")

    # ------------------------------------------------------------------ search

    def semantic_search(
        self,
        query: str,
        top_k: int = 25,
        tenant_id: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[CrossMemoryEntry]:
        try:
            where, params = self._build_where(tenant_id=tenant_id)
            qvec = list(float(x) for x in
                        self.embedding_model.encode_single(query, is_query=True))
            sql = self._t(_SEMANTIC).format(where=where)
            with self._cur() as cur:
                cur.execute(sql, params + [qvec, top_k])
                return [_row_to_cross(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("semantic_search failed")
            return []

    def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 5,
        tenant_id: Optional[str] = None,
    ) -> List[CrossMemoryEntry]:
        if not keywords:
            return []
        try:
            extra_where, extra_params = "", []
            if tenant_id:
                extra_where = "AND tenant_id = %s"
                extra_params = [tenant_id]
            query_text = " ".join(keywords)
            sql = self._t(_KEYWORD).format(extra_where=extra_where)
            with self._cur() as cur:
                cur.execute(sql, [query_text] + extra_params + [top_k])
                return [_row_to_cross(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("keyword_search failed")
            return []

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[CrossMemoryEntry]:
        if not any([persons, timestamp_range, location, entities, tenant_id]):
            return []
        try:
            conds, params = [], []
            if tenant_id:
                conds.append("tenant_id = %s")
                params.append(tenant_id)
            if persons:
                for p in persons:
                    conds.append("persons::jsonb @> %s::jsonb")
                    params.append(json.dumps([p]))
            if location:
                conds.append("location ILIKE %s")
                params.append(f"%{location}%")
            if entities:
                for e in entities:
                    conds.append("entities::jsonb @> %s::jsonb")
                    params.append(json.dumps([e]))
            if timestamp_range:
                start, end = timestamp_range
                conds.append("timestamp >= %s AND timestamp <= %s")
                params.extend([str(start), str(end)])

            where = " AND ".join(f"({c})" for c in conds)
            cols = ("entry_id, text, keywords, timestamp, location, persons, entities, topic,"
                    " tenant_id, memory_session_id, source_kind, source_id, importance,"
                    " valid_from, valid_to, superseded_by")
            sql = f"SELECT {cols} FROM {self.table_name} WHERE {where} LIMIT {int(top_k)}"
            with self._cur() as cur:
                cur.execute(sql, params)
                return [_row_to_cross(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("structured_search failed")
            return []

    # ----------------------------------------------------------------- session

    def get_entries_for_session(
        self, memory_session_id: str
    ) -> List[CrossMemoryEntry]:
        try:
            cols = ("entry_id, text, keywords, timestamp, location, persons, entities, topic,"
                    " tenant_id, memory_session_id, source_kind, source_id, importance,"
                    " valid_from, valid_to, superseded_by")
            with self._cur() as cur:
                cur.execute(
                    f"SELECT {cols} FROM {self.table_name} WHERE memory_session_id = %s",
                    [memory_session_id],
                )
                return [_row_to_cross(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("get_entries_for_session failed")
            return []

    def get_all_entries(
        self, tenant_id: Optional[str] = None
    ) -> List[CrossMemoryEntry]:
        try:
            where, params = self._build_where(tenant_id=tenant_id)
            cols = ("entry_id, text, keywords, timestamp, location, persons, entities, topic,"
                    " tenant_id, memory_session_id, source_kind, source_id, importance,"
                    " valid_from, valid_to, superseded_by")
            with self._cur() as cur:
                cur.execute(
                    f"SELECT {cols} FROM {self.table_name} {where}", params
                )
                return [_row_to_cross(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("get_all_entries failed")
            return []

    def count_entries(
        self,
        tenant_id: Optional[str] = None,
        memory_session_id: Optional[str] = None,
    ) -> int:
        try:
            where, params = self._build_where(tenant_id, memory_session_id)
            with self._cur() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {self.table_name} {where}", params
                )
                return int(cur.fetchone()[0])
        except Exception:
            logger.exception("count_entries failed")
            return 0

    def mark_superseded(self, old_entry_id: str, new_entry_id: str) -> None:
        try:
            with self._cur() as cur:
                cur.execute(
                    self._t(_UPDATE_SUPER),
                    [new_entry_id, datetime.utcnow().isoformat(), old_entry_id],
                )
            self._commit()
        except Exception:
            logger.exception("mark_superseded failed")

    def update_importance(self, entry_id: str, new_importance: float) -> None:
        try:
            with self._cur() as cur:
                cur.execute(self._t(_UPDATE_IMP), [float(new_importance), entry_id])
            self._commit()
        except Exception:
            logger.exception("update_importance failed")

    def optimize(self) -> None:
        conn = _get_conn(self._dsn)
        old_ac = conn.autocommit
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f"VACUUM ANALYZE {self.table_name}")
        finally:
            conn.autocommit = old_ac

    def clear(self, tenant_id: Optional[str] = None) -> None:
        try:
            if tenant_id:
                with self._cur() as cur:
                    cur.execute(
                        f"DELETE FROM {self.table_name} WHERE tenant_id = %s",
                        [tenant_id],
                    )
                self._commit()
            else:
                with self._cur() as cur:
                    cur.execute(f"TRUNCATE TABLE {self.table_name}")
                self._commit()
        except Exception:
            logger.exception("clear failed")

    def close(self) -> None:
        conn = getattr(_local, "conn", None)
        if conn and not conn.closed:
            conn.close()
        _local.conn = None
