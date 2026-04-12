from __future__ import annotations

import json
import threading
from typing import List, Optional, Dict, Any

import iris.dbapi as dbapi

from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
import config


def _connect() -> dbapi.Connection:
    return dbapi.connect(
        config.IRIS_HOSTNAME,
        config.IRIS_PORT,
        config.IRIS_NAMESPACE,
        config.IRIS_USERNAME,
        config.IRIS_PASSWORD,
    )


_local = threading.local()


def _thread_conn() -> dbapi.Connection:
    if not getattr(_local, "conn", None):
        _local.conn = _connect()
    return _local.conn


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    entry_id   VARCHAR(64)    NOT NULL,
    text       VARCHAR(32000) NOT NULL,
    keywords   VARCHAR(4000),
    timestamp  VARCHAR(64),
    location   VARCHAR(512),
    persons    VARCHAR(4000),
    entities   VARCHAR(4000),
    topic      VARCHAR(512),
    vec        VECTOR(DOUBLE, {dim})
)
"""

_INSERT = """
INSERT INTO {table}
    (entry_id, text, keywords, timestamp, location, persons, entities, topic, vec)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, TO_VECTOR(?, DOUBLE, {dim}))
"""

_SEMANTIC_SEARCH = """
SELECT TOP {top_k} entry_id, text, keywords, timestamp, location, persons, entities, topic,
       VECTOR_COSINE(vec, TO_VECTOR(?, DOUBLE, {dim})) AS score
FROM {table}
ORDER BY score DESC
"""

_DELETE_ALL = "DELETE FROM {table}"
_DELETE_ONE = "DELETE FROM {table} WHERE entry_id = ?"
_COUNT      = "SELECT COUNT(*) FROM {table}"
_SELECT_ALL = "SELECT entry_id, text, keywords, timestamp, location, persons, entities, topic FROM {table}"


def _enc(lst: list) -> str:
    return json.dumps(lst)

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


class VectorStore:
    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        table_name: str = None,
        db_path: Optional[str] = None,
        storage_options: Optional[Dict[str, Any]] = None,
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self._dim = self.embedding_model.dimension
        self._ensure_table()
        print(f"Connected to IRIS table: {self.table_name}")

    def _cur(self):
        return _thread_conn().cursor()

    def _commit(self):
        _thread_conn().commit()

    def _ensure_table(self):
        cur = self._cur()
        try:
            cur.execute(_CREATE_TABLE.format(table=self.table_name, dim=self._dim))
            self._commit()
        except Exception:
            pass
        try:
            cur.execute(
                f"CREATE INDEX HNSWIdx ON TABLE {self.table_name} (vec)"
                f" AS HNSW(Distance='Cosine', M=16, efConstruction=64)"
            )
            self._commit()
        except Exception:
            pass
        finally:
            cur.close()

    def _t(self, sql: str) -> str:
        return sql.replace("{table}", self.table_name).replace("{dim}", str(self._dim))

    def add_entries(self, entries: List[MemoryEntry]):
        if not entries:
            return
        texts = [e.lossless_restatement for e in entries]
        vecs  = self.embedding_model.encode_documents(texts)
        cur   = self._cur()
        try:
            for entry, vec in zip(entries, vecs):
                cur.execute(self._t(_INSERT), [
                    entry.entry_id,
                    entry.lossless_restatement,
                    _enc(entry.keywords or []),
                    entry.timestamp or "",
                    entry.location  or "",
                    _enc(entry.persons  or []),
                    _enc(entry.entities or []),
                    entry.topic or "",
                    json.dumps([float(v) for v in vec]),
                ])
            self._commit()
            print(f"Added {len(entries)} memory entries")
        except Exception as e:
            print(f"Error adding entries: {e}")
        finally:
            cur.close()

    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        qvec = self.embedding_model.encode_single(query, is_query=True)
        cur  = self._cur()
        try:
            sql = _SEMANTIC_SEARCH.replace("{table}", self.table_name).replace("{dim}", str(self._dim)).replace("{top_k}", str(top_k))
            cur.execute(sql, [json.dumps([float(v) for v in qvec])])
            return [_row_to_entry(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []
        finally:
            cur.close()

    def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[MemoryEntry]:
        if not keywords:
            return []
        cur = self._cur()
        try:
            where_clause = " OR ".join("$FIND(text, ?) > 0" for _ in keywords)
            score_expr   = " + ".join(
                "CASE WHEN $FIND(text, ?) > 0 THEN 1 ELSE 0 END"
                for _ in keywords
            )
            sql = f"""
                SELECT TOP {top_k} entry_id, text, keywords, timestamp, location, persons, entities, topic
                FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY ({score_expr}) DESC
            """
            cur.execute(sql, list(keywords) + list(keywords))
            return [_row_to_entry(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"Error during keyword search: {e}")
            return []
        finally:
            cur.close()

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryEntry]:
        if not any([persons, timestamp_range, location, entities]):
            return []
        cur = self._cur()
        try:
            conditions, params = [], []

            if persons:
                conditions.append(f"({' OR '.join('$FIND(persons, ?) > 0' for _ in persons)})")
                params.extend(persons)
            if location:
                conditions.append("$FIND(location, ?) > 0")
                params.append(location)
            if entities:
                conditions.append(f"({' OR '.join('$FIND(entities, ?) > 0' for _ in entities)})")
                params.extend(entities)
            if timestamp_range:
                start, end = timestamp_range
                conditions.append("timestamp >= ? AND timestamp <= ?")
                params.extend([str(start), str(end)])

            limit = f"TOP {top_k}" if top_k else ""
            sql = f"""
                SELECT {limit} entry_id, text, keywords, timestamp, location, persons, entities, topic
                FROM {self.table_name}
                WHERE {" AND ".join(conditions)}
            """
            cur.execute(sql, params)
            return [_row_to_entry(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"Error during structured search: {e}")
            return []
        finally:
            cur.close()

    def get_all_entries(self) -> List[MemoryEntry]:
        cur = self._cur()
        try:
            cur.execute(self._t(_SELECT_ALL))
            return [_row_to_entry(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"Error fetching all entries: {e}")
            return []
        finally:
            cur.close()

    def optimize(self):
        pass

    def clear(self):
        cur = self._cur()
        try:
            cur.execute(self._t(_DELETE_ALL))
            self._commit()
            print("Database cleared")
        except Exception as e:
            print(f"Error clearing database: {e}")
        finally:
            cur.close()

    def close(self):
        conn = getattr(_local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            _local.conn = None
