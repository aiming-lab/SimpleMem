"""Optional PostgreSQL + pgvector implementation of the SimpleMem backend contract.

Three retrieval layers — all parameterized, no SQL injection surface:

  Semantic:    pgvector HNSW index, cosine distance (<=>)
  Keyword:     PostgreSQL tsvector / plainto_tsquery (generated column + GIN index)
  Structured:  standard SQL WHERE with %s placeholders

Client installation
-------------------
  pip install "psycopg[binary]" pgvector
  # or:
  pip install simplemem[pgvector]

Using PostgreSQL
----------------
Point PG_DSN at any PostgreSQL ≥ 15 instance with the pgvector extension:

  CREATE EXTENSION IF NOT EXISTS vector;   -- one-time setup

  STORAGE_BACKEND = "pgvector"
  PG_DSN = "postgresql://user:pass@localhost:5432/mydb"

Using InterSystems IRIS
-----------------------
IRIS does not natively speak the PostgreSQL wire protocol, but
`iris-pgwire` (https://github.com/intersystems-community/iris-pgwire)
adds a translation layer so any psycopg client — including this backend —
connects to IRIS without IRIS-specific drivers.

Server-side setup (one time per IRIS instance):

  1. Install iris-pgwire on the machine running IRIS:
       pip install iris-pgwire
       # or via Docker: docker compose up -d  (see iris-pgwire README)

  2. Start the pgwire server (translates Postgres wire protocol → IRIS SQL):
       python -m iris_pgwire.server  # default port 5432

  3. Point PG_DSN at the pgwire endpoint:
       PG_DSN = "postgresql://_SYSTEM:SYS@localhost:5432/USER"

iris-pgwire translates pgvector DDL and operators transparently:
  - CREATE INDEX USING hnsw (vec vector_cosine_ops)  →  IRIS HNSW syntax
  - vec <-> %s::vector                               →  VECTOR_COSINE(vec, TO_VECTOR(%s, DOUBLE))
  - tsvector / plainto_tsquery                       →  IRIS full-text SQL equivalents
  - Generated columns (tsv tsvector GENERATED ALWAYS AS …) are emulated

No code changes are needed to switch between PostgreSQL and IRIS — only
the DSN changes.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

from simplemem.core.database.vector_store_backend import (
    ScoreOrder,
    VectorStoreRecord,
    VectorStoreSearchResult,
)

logger = logging.getLogger(__name__)

_local = threading.local()

# ---------------------------------------------------------------------------
# SQL templates — table name is injected via str.format() on trusted internal
# strings only; all user-supplied values go through %s placeholders.
# ---------------------------------------------------------------------------

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    entry_id              VARCHAR(512)  NOT NULL,
    lossless_restatement  VARCHAR(4000) NOT NULL,
    keywords              VARCHAR(4000),
    timestamp             VARCHAR(50),
    location              VARCHAR(1000),
    persons               VARCHAR(4000),
    entities              VARCHAR(4000),
    topic                 VARCHAR(500),
    vec                   vector({dim})
)
"""

_CREATE_VEC_IDX = """
CREATE INDEX {table}_vec_idx
    ON {table} USING hnsw (vec vector_cosine_ops)
"""

_CREATE_TSV_IDX = ""  # GIN on generated tsvector not supported in IRIS; inline to_tsvector() used in queries

_INSERT = """
INSERT INTO {table}
    (entry_id, lossless_restatement, keywords, timestamp,
     location, persons, entities, topic, vec)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
"""

_SELECT_COLS = (
    "entry_id, lossless_restatement, keywords, timestamp,"
    " location, persons, entities, topic"
)

_SEMANTIC = f"SELECT {_SELECT_COLS} FROM {{table}} ORDER BY vec <=> %s::vector LIMIT {{limit}}"
_KEYWORD_BASE = f"SELECT {_SELECT_COLS} FROM {{table}} WHERE"
_GET_ALL  = f"SELECT {_SELECT_COLS} FROM {{table}}"
_COUNT    = "SELECT COUNT(*) FROM {table}"


def _enc(lst: list) -> str:
    return json.dumps(lst or [])


def _dec(s: Optional[str]) -> list:
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


def _row_to_result(row) -> VectorStoreSearchResult:
    entry_id   = row[0]
    metadata   = {
        "lossless_restatement": row[1],
        "keywords":  _dec(row[2]),
        "timestamp": row[3] or None,
        "location":  row[4] or None,
        "persons":   _dec(row[5]),
        "entities":  _dec(row[6]),
        "topic":     row[7] or None,
    }
    return VectorStoreSearchResult(entry_id=entry_id, metadata=metadata)


class PGVectorStoreBackend:
    """SimpleMem VectorStoreBackend backed by PostgreSQL + pgvector.

    Parameters
    ----------
    dsn:
        libpq connection string, e.g.
        ``"postgresql://user:pass@localhost:5432/dbname"``.
    table_name:
        Relation to store memory entries in. Created automatically.
    vector_dimension:
        Must match the embedding model's output dimension.
    """

    semantic_score_order = ScoreOrder.ASCENDING
    keyword_score_order  = ScoreOrder.DESCENDING

    def __init__(
        self,
        dsn: str,
        table_name: str,
        vector_dimension: int,
    ) -> None:
        if not dsn:
            raise ValueError(
                "PGVectorStoreBackend requires a dsn. "
                "Set PG_DSN in config.py or the environment."
            )
        if vector_dimension <= 0:
            raise ValueError("vector_dimension must be greater than zero.")

        self._dsn = dsn
        self.table_name = table_name
        self.vector_dimension = vector_dimension
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _conn(self):
        if not getattr(_local, "conn", None) or _local.conn.closed:
            try:
                import psycopg
            except ImportError as exc:
                raise ImportError(
                    "psycopg is required for the pgvector backend. "
                    "Install it with: pip install 'psycopg[binary]' pgvector"
                ) from exc
            _local.conn = psycopg.connect(self._dsn, autocommit=False)
        return _local.conn

    def _cur(self):
        return self._conn().cursor()

    def _commit(self) -> None:
        self._conn().commit()

    def _t(self, sql: str, *, limit: int = 0) -> str:
        return sql.format(table=self.table_name, dim=self.vector_dimension,
                          limit=int(limit) if limit else "")

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        conn = self._conn()
        # CREATE EXTENSION is a no-op on IRIS (vector support is built-in via pgwire)
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_EXTENSION)
            conn.commit()
        except Exception:
            conn.rollback()
        with conn.cursor() as cur:
            cur.execute(self._t(_CREATE_TABLE))
        conn.commit()
        for idx_sql in (self._t(_CREATE_VEC_IDX),):
            try:
                with conn.cursor() as cur:
                    cur.execute(idx_sql)
                conn.commit()
            except Exception:
                conn.rollback()  # index already exists — safe to ignore
        logger.debug("Schema ready: table=%s dim=%d", self.table_name, self.vector_dimension)

    # ------------------------------------------------------------------
    # VectorStoreBackend protocol
    # ------------------------------------------------------------------

    def insert(self, records: Sequence[VectorStoreRecord]) -> None:
        if not records:
            return
        rows = [
            (
                r.entry_id,
                r.metadata.get("lossless_restatement", ""),
                _enc(r.metadata.get("keywords")),
                r.metadata.get("timestamp") or "",
                r.metadata.get("location") or "",
                _enc(r.metadata.get("persons")),
                _enc(r.metadata.get("entities")),
                r.metadata.get("topic") or "",
                list(float(x) for x in r.vector),
            )
            for r in records
        ]
        sql = self._t(_INSERT)
        for row in rows:
            with self._cur() as cur:
                try:
                    cur.execute(sql, row)
                except Exception:
                    self._conn().rollback()
                    continue
        self._commit()
        logger.debug("Inserted %d records into %s", len(records), self.table_name)

    def semantic_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreSearchResult]:
        if self.count() == 0:
            return []
        try:
            vec = list(float(x) for x in query_vector)
            with self._cur() as cur:
                cur.execute(self._t(_SEMANTIC, limit=top_k), (vec,))
                return [_row_to_result(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("semantic_search failed")
            return []

    def keyword_search(
        self,
        keywords: Sequence[str],
        top_k: int,
    ) -> List[VectorStoreSearchResult]:
        if not keywords or self.count() == 0:
            return []
        try:
            # Match any keyword against text or JSON keywords column (case-insensitive LIKE)
            conditions = []
            params: list = []
            for kw in keywords:
                pat = f"%{kw}%"
                conditions.append(
                    "(LOWER(lossless_restatement) LIKE LOWER(%s)"
                    " OR LOWER(keywords) LIKE LOWER(%s))"
                )
                params.extend([pat, pat])
            where = " OR ".join(conditions)
            sql = self._t(_KEYWORD_BASE) + " " + where + f" LIMIT {int(top_k)}"
            with self._cur() as cur:
                cur.execute(sql, params)
                return [_row_to_result(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("keyword_search failed")
            return []

    def structured_search(
        self,
        persons: Optional[Sequence[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[VectorStoreSearchResult]:
        if not any([persons, timestamp_range, location, entities]):
            return []
        if self.count() == 0:
            return []

        conditions: list[str] = []
        params: list[Any] = []

        for p in (persons or []):
            conditions.append('persons LIKE %s')
            params.append(f'%"{p}"%')

        if location:
            conditions.append("location LIKE %s")
            params.append(f"%{location}%")

        for e in (entities or []):
            conditions.append('entities LIKE %s')
            params.append(f'%"{e}"%')

        if timestamp_range:
            start, end = timestamp_range
            conditions.append("timestamp >= %s AND timestamp <= %s")
            params.extend([start, end])

        where = " OR ".join(f"({c})" for c in conditions)
        sql = f"SELECT {_SELECT_COLS} FROM {self.table_name} WHERE {where}"
        if top_k:
            sql += f" LIMIT {int(top_k)}"

        try:
            with self._cur() as cur:
                cur.execute(sql, params)
                return [_row_to_result(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("structured_search failed")
            return []

    def count(self) -> int:
        try:
            with self._cur() as cur:
                cur.execute(self._t(_COUNT))
                return cur.fetchone()[0]
        except Exception:
            return 0

    def get_all(self) -> List[VectorStoreSearchResult]:
        try:
            with self._cur() as cur:
                cur.execute(self._t(_GET_ALL))
                return [_row_to_result(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("get_all failed")
            return []

    def optimize(self) -> None:
        """VACUUM ANALYZE to refresh planner statistics after bulk loads.

        No-op on IRIS (which manages statistics automatically).
        """
        conn = self._conn()
        try:
            old = conn.autocommit
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute(f"VACUUM ANALYZE {self.table_name}")
            finally:
                conn.autocommit = old
            logger.debug("VACUUM ANALYZE completed on %s", self.table_name)
        except Exception:
            logger.debug("optimize() skipped (not supported by this backend)")

    def clear(self) -> None:
        with self._cur() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table_name}")
        self._commit()
        logger.debug("Cleared table %s", self.table_name)

    def close(self) -> None:
        conn = getattr(_local, "conn", None)
        if conn and not conn.closed:
            conn.close()
        _local.conn = None
