"""
PostgreSQL/pgvector Vector Store - Multi-View Indexing (Section 3.1)

Drop-in replacement for the LanceDB VectorStore backed by PostgreSQL + pgvector.
Implements the same three-layer indexing interface:
  - Semantic layer:    <-> cosine distance via pgvector HNSW index
  - Lexical layer:     plainto_tsquery full-text search on lossless_restatement
  - Symbolic layer:    standard SQL WHERE on persons, entities, location, timestamp

IRIS + pgwire compatibility:
  IRIS with iris-pgwire satisfies this interface directly — point PG_DSN at the
  pgwire endpoint. pgwire's vector_optimizer.py handles the IRIS requirement that
  vector expressions in ORDER BY must be literals rather than parameters by
  inlining them server-side before execution.

Requires: psycopg[binary] pgvector
  pip install "psycopg[binary]" pgvector
  or: pip install simplemem[pgvector]
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from simplemem.core.models.memory_entry import MemoryEntry
from simplemem.core.utils.embedding import EmbeddingModel
from simplemem.core.settings import settings as config

logger = logging.getLogger(__name__)

_local = threading.local()


def _get_conn(dsn: str):
    if not getattr(_local, "conn", None) or _local.conn.closed:
        try:
            import psycopg
        except ImportError as e:
            raise ImportError(
                "psycopg is required for the pgvector backend. "
                "Install it with: pip install 'psycopg[binary]' pgvector"
            ) from e
        _local.conn = psycopg.connect(dsn, autocommit=False)
    return _local.conn


_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    entry_id              TEXT        NOT NULL,
    lossless_restatement  TEXT        NOT NULL,
    keywords              TEXT,
    timestamp             TEXT,
    location              TEXT,
    persons               TEXT,
    entities              TEXT,
    topic                 TEXT,
    vec                   vector({dim}),
    tsv                   tsvector
        GENERATED ALWAYS AS (to_tsvector('english', lossless_restatement)) STORED
)
"""

_CREATE_VEC_INDEX = """
CREATE INDEX IF NOT EXISTS {table}_vec_idx
    ON {table} USING hnsw (vec vector_cosine_ops)
"""

_CREATE_TSV_INDEX = """
CREATE INDEX IF NOT EXISTS {table}_tsv_idx
    ON {table} USING gin (tsv)
"""

_INSERT = """
INSERT INTO {table}
    (entry_id, lossless_restatement, keywords, timestamp,
     location, persons, entities, topic, vec)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT DO NOTHING
"""

_SEMANTIC = """
SELECT entry_id, lossless_restatement, keywords, timestamp,
       location, persons, entities, topic
FROM {table}
ORDER BY vec <-> %s::vector
LIMIT %s
"""

_KEYWORD = """
SELECT entry_id, lossless_restatement, keywords, timestamp,
       location, persons, entities, topic
FROM {table}
WHERE tsv @@ plainto_tsquery('english', %s)
LIMIT %s
"""


def _enc(lst: list) -> str:
    return json.dumps(lst or [])


def _dec(s: Optional[str]) -> list:
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


def _row_to_entry(row) -> MemoryEntry:
    return MemoryEntry(
        entry_id=row[0],
        lossless_restatement=row[1],
        keywords=_dec(row[2]),
        timestamp=row[3] or None,
        location=row[4] or None,
        persons=_dec(row[5]),
        entities=_dec(row[6]),
        topic=row[7] or None,
    )


class PGVectorStore:
    """
    PostgreSQL + pgvector implementation of the SimpleMem VectorStore interface.

    Duck-typed to match simplemem.core.database.vector_store.VectorStore so
    HybridRetriever and MemoryBuilder work without modification.

    Parameters
    ----------
    dsn:
        libpq connection string, e.g.
        ``"postgresql://user:pass@localhost:5432/dbname"``
        Defaults to the ``PG_DSN`` setting / env var.
    embedding_model:
        EmbeddingModel instance. Defaults to a fresh EmbeddingModel().
    table_name:
        Table name for memory entries. Defaults to ``MEMORY_TABLE_NAME`` setting.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        table_name: Optional[str] = None,
        # Accept (and ignore) db_path / storage_options so callers can swap
        # this in without changing kwargs.
        db_path: Optional[str] = None,
        storage_options: Optional[Dict[str, Any]] = None,
    ):
        self._dsn = dsn or config.PG_DSN
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self._dim = self.embedding_model.dimension
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        conn = _get_conn(self._dsn)
        with conn.cursor() as cur:
            cur.execute(_CREATE_EXTENSION)
            cur.execute(
                _CREATE_TABLE.format(table=self.table_name, dim=self._dim)
            )
            cur.execute(_CREATE_VEC_INDEX.format(table=self.table_name))
            cur.execute(_CREATE_TSV_INDEX.format(table=self.table_name))
        conn.commit()
        logger.debug("Schema ready: table=%s dim=%d", self.table_name, self._dim)

    def _cur(self):
        return _get_conn(self._dsn).cursor()

    def _commit(self) -> None:
        _get_conn(self._dsn).commit()

    def _t(self, sql: str) -> str:
        return sql.format(table=self.table_name, dim=self._dim)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_entries(self, entries: List[MemoryEntry]) -> None:
        """Batch-embed and insert memory entries."""
        if not entries:
            return

        vectors = self.embedding_model.encode_documents(
            [e.lossless_restatement for e in entries]
        )

        rows = [
            (
                e.entry_id,
                e.lossless_restatement,
                _enc(e.keywords),
                e.timestamp or "",
                e.location or "",
                _enc(e.persons),
                _enc(e.entities),
                e.topic or "",
                # pgvector accepts a Python list as the vector value
                list(float(x) for x in vec),
            )
            for e, vec in zip(entries, vectors)
        ]

        with self._cur() as cur:
            cur.executemany(self._t(_INSERT), rows)
        self._commit()
        logger.debug("Inserted %d entries into %s", len(entries), self.table_name)

    # ------------------------------------------------------------------
    # Read — semantic
    # ------------------------------------------------------------------

    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Dense vector similarity search via pgvector <-> cosine operator."""
        qvec = list(float(x) for x in
                    self.embedding_model.encode_single(query, is_query=True))
        try:
            with self._cur() as cur:
                cur.execute(self._t(_SEMANTIC), (qvec, top_k))
                return [_row_to_entry(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("semantic_search failed")
            return []

    # ------------------------------------------------------------------
    # Read — keyword
    # ------------------------------------------------------------------

    def keyword_search(
        self, keywords: List[str], top_k: int = 3
    ) -> List[MemoryEntry]:
        """Full-text search via PostgreSQL tsvector / plainto_tsquery."""
        if not keywords:
            return []
        query_text = " ".join(keywords)
        try:
            with self._cur() as cur:
                cur.execute(self._t(_KEYWORD), (query_text, top_k))
                return [_row_to_entry(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("keyword_search failed")
            return []

    # ------------------------------------------------------------------
    # Read — structured
    # ------------------------------------------------------------------

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Symbolic/metadata search via SQL WHERE with parameterized values."""
        if not any([persons, timestamp_range, location, entities]):
            return []

        conditions: list[str] = []
        params: list[Any] = []

        if persons:
            # persons stored as JSON array; match any element via JSON containment
            for p in persons:
                conditions.append("persons::jsonb @> %s::jsonb")
                params.append(json.dumps([p]))

        if location:
            conditions.append("location ILIKE %s")
            params.append(f"%{location}%")

        if entities:
            for e in entities:
                conditions.append("entities::jsonb @> %s::jsonb")
                params.append(json.dumps([e]))

        if timestamp_range:
            start, end = timestamp_range
            conditions.append("timestamp >= %s AND timestamp <= %s")
            params.extend([start, end])

        where = " OR ".join(f"({c})" for c in conditions)
        sql = f"SELECT entry_id, lossless_restatement, keywords, timestamp, location, persons, entities, topic FROM {self.table_name} WHERE {where}"
        if top_k:
            sql += f" LIMIT {int(top_k)}"

        try:
            with self._cur() as cur:
                cur.execute(sql, params)
                return [_row_to_entry(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("structured_search failed")
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_all_entries(self) -> List[MemoryEntry]:
        with self._cur() as cur:
            cur.execute(
                f"SELECT entry_id, lossless_restatement, keywords, timestamp,"
                f" location, persons, entities, topic FROM {self.table_name}"
            )
            return [_row_to_entry(r) for r in cur.fetchall()]

    def optimize(self) -> None:
        """VACUUM ANALYZE to update planner statistics after bulk loads."""
        conn = _get_conn(self._dsn)
        old_autocommit = conn.autocommit
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f"VACUUM ANALYZE {self.table_name}")
        finally:
            conn.autocommit = old_autocommit
        logger.debug("VACUUM ANALYZE completed on %s", self.table_name)

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
