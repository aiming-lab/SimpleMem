from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import List, Optional

import iris.dbapi as dbapi

from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
from cross.types import CrossMemoryEntry
import config


def _connect() -> dbapi.Connection:
    return dbapi.connect(
        config.IRIS_HOSTNAME,
        config.IRIS_PORT,
        config.IRIS_NAMESPACE,
        config.IRIS_USERNAME,
        config.IRIS_PASSWORD,
    )


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    entry_id          VARCHAR(64)    NOT NULL,
    text              VARCHAR(32000) NOT NULL,
    keywords          VARCHAR(4000),
    timestamp         VARCHAR(64),
    location          VARCHAR(512),
    persons           VARCHAR(4000),
    entities          VARCHAR(4000),
    topic             VARCHAR(512),
    tenant_id         VARCHAR(256),
    memory_session_id VARCHAR(256),
    source_kind       VARCHAR(64),
    source_id         INTEGER,
    importance        DOUBLE,
    valid_from        VARCHAR(64),
    valid_to          VARCHAR(64),
    superseded_by     VARCHAR(64),
    vec               VECTOR(DOUBLE, {dim})
)
"""

_INSERT = """
INSERT INTO {table}
    (entry_id, text, keywords, timestamp, location, persons, entities, topic,
     tenant_id, memory_session_id, source_kind, source_id, importance,
     valid_from, valid_to, superseded_by, vec)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, TO_VECTOR(?, DOUBLE, {dim}))
"""

_SEMANTIC_SQL = """
SELECT TOP {top_k} entry_id, text, keywords, timestamp, location, persons, entities, topic,
       tenant_id, memory_session_id, source_kind, source_id, importance,
       valid_from, valid_to, superseded_by,
       VECTOR_COSINE(vec, TO_VECTOR(?, DOUBLE, {dim})) AS score
FROM {table}
{where}
ORDER BY score DESC
"""

_COUNT_SQL     = "SELECT COUNT(*) FROM {table}"
_DELETE_WHERE  = "DELETE FROM {table} WHERE entry_id = ?"
_UPDATE_SUPER  = "UPDATE {table} SET superseded_by = ?, valid_to = ? WHERE entry_id = ?"
_UPDATE_IMP    = "UPDATE {table} SET importance = ? WHERE entry_id = ?"


def _enc(lst: list) -> str:
    return json.dumps(lst)

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


class CrossSessionVectorStore:
    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        table_name: str = "cross_memory_entries",
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name
        self._dim = self.embedding_model.dimension
        self._lock = threading.RLock()
        self._conn = _connect()
        self._ensure_table()
        print(f"Connected to IRIS cross-session table: {self.table_name}")

    def _cur(self):
        return self._conn.cursor()

    def _ensure_table(self):
        cur = self._cur()
        try:
            cur.execute(
                _CREATE_TABLE.format(table=self.table_name, dim=self._dim)
            )
            self._conn.commit()
        except Exception:
            pass
        try:
            cur.execute(
                f"CREATE INDEX HNSWIdx ON TABLE {self.table_name} (vec)"
                f" AS HNSW(Distance='Cosine', M=16, efConstruction=64)"
            )
            self._conn.commit()
        except Exception:
            pass
        finally:
            cur.close()

    def _t(self, sql: str) -> str:
        return sql.replace("{table}", self.table_name).replace("{dim}", str(self._dim))

    def _build_where(
        self,
        tenant_id: Optional[str] = None,
        memory_session_id: Optional[str] = None,
    ) -> tuple[str, list]:
        conds, params = [], []
        if tenant_id:
            conds.append("tenant_id = ?")
            params.append(tenant_id)
        if memory_session_id:
            conds.append("memory_session_id = ?")
            params.append(memory_session_id)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        return where, params

    def _insert_rows(self, rows: list[dict]):
        cur = self._cur()
        try:
            sql = _INSERT.format(table=self.table_name, dim=self._dim)
            for r in rows:
                cur.execute(sql, [
                    r["entry_id"], r["text"], r["keywords"], r["timestamp"],
                    r["location"], r["persons"], r["entities"], r["topic"],
                    r["tenant_id"], r["memory_session_id"], r["source_kind"],
                    r["source_id"], r["importance"], r["valid_from"],
                    r["valid_to"], r["superseded_by"],
                    json.dumps([float(v) for v in r["vec"]]),
                ])
            self._conn.commit()
        finally:
            cur.close()

    def add_entries(
        self,
        entries: list[MemoryEntry],
        tenant_id: str,
        memory_session_id: str,
        source_kind: str,
        source_id: int = 0,
        importance: float = 0.5,
    ):
        if not entries:
            return
        with self._lock:
            try:
                vecs = self.embedding_model.encode_documents(
                    [e.lossless_restatement for e in entries]
                )
                now = datetime.utcnow().isoformat()
                rows = [
                    {
                        "entry_id": e.entry_id,
                        "text": e.lossless_restatement,
                        "keywords": _enc(e.keywords or []),
                        "timestamp": e.timestamp or "",
                        "location": e.location or "",
                        "persons": _enc(e.persons or []),
                        "entities": _enc(e.entities or []),
                        "topic": e.topic or "",
                        "tenant_id": tenant_id,
                        "memory_session_id": memory_session_id,
                        "source_kind": source_kind,
                        "source_id": source_id,
                        "importance": float(importance),
                        "valid_from": now,
                        "valid_to": "",
                        "superseded_by": "",
                        "vec": vecs[i].tolist(),
                    }
                    for i, e in enumerate(entries)
                ]
                self._insert_rows(rows)
                print(f"Added {len(entries)} cross-session memory entries")
            except Exception as e:
                print(f"Error adding cross-session entries: {e}")

    def add_cross_entries(self, cross_entries: list[CrossMemoryEntry]):
        if not cross_entries:
            return
        with self._lock:
            try:
                vecs = self.embedding_model.encode_documents(
                    [e.lossless_restatement for e in cross_entries]
                )
                rows = [
                    {
                        "entry_id": e.entry_id,
                        "text": e.lossless_restatement,
                        "keywords": _enc(e.keywords or []),
                        "timestamp": e.timestamp or "",
                        "location": e.location or "",
                        "persons": _enc(e.persons or []),
                        "entities": _enc(e.entities or []),
                        "topic": e.topic or "",
                        "tenant_id": e.tenant_id,
                        "memory_session_id": e.memory_session_id,
                        "source_kind": e.source_kind,
                        "source_id": e.source_id or 0,
                        "importance": float(e.importance),
                        "valid_from": _fmt_dt(e.valid_from),
                        "valid_to": _fmt_dt(e.valid_to),
                        "superseded_by": e.superseded_by or "",
                        "vec": vecs[i].tolist(),
                    }
                    for i, e in enumerate(cross_entries)
                ]
                self._insert_rows(rows)
                print(f"Added {len(cross_entries)} cross-session memory entries")
            except Exception as e:
                print(f"Error adding cross-session entries: {e}")

    def _semantic(
        self,
        query: str,
        top_k: int,
        where: str = "",
        where_params: list = None,
    ) -> list[CrossMemoryEntry]:
        qvec = self.embedding_model.encode_single(query, is_query=True)
        sql  = _SEMANTIC_SQL.format(
            table=self.table_name, dim=self._dim, where=where, top_k=top_k
        )
        cur  = self._cur()
        try:
            cur.execute(sql, [json.dumps([float(v) for v in qvec])] + (where_params or []))
            return [_row_to_cross(r) for r in cur.fetchall()]
        finally:
            cur.close()

    def semantic_search(
        self,
        query: str,
        top_k: int = 25,
        tenant_id: Optional[str] = None,
        project: Optional[str] = None,
    ) -> list[CrossMemoryEntry]:
        with self._lock:
            try:
                where, params = self._build_where(tenant_id=tenant_id)
                return self._semantic(query, top_k, where, params)
            except Exception as e:
                print(f"Error during semantic search: {e}")
                return []

    def keyword_search(
        self,
        keywords: list[str],
        top_k: int = 5,
        tenant_id: Optional[str] = None,
    ) -> list[CrossMemoryEntry]:
        if not keywords:
            return []
        with self._lock:
            cur = self._cur()
            try:
                kw_filter = " OR ".join("$FIND(text, ?) > 0" for _ in keywords)
                score_expr = " + ".join(
                    "CASE WHEN $FIND(text, ?) > 0 THEN 1 ELSE 0 END"
                    for _ in keywords
                )
                tenant_clause, tenant_params = self._build_where(tenant_id=tenant_id)
                and_kw = "AND" if tenant_clause else "WHERE"
                sql = f"""
                    SELECT TOP {top_k} entry_id, text, keywords, timestamp, location,
                           persons, entities, topic, tenant_id, memory_session_id,
                           source_kind, source_id, importance, valid_from, valid_to, superseded_by
                    FROM {self.table_name}
                    {tenant_clause} {and_kw} ({kw_filter})
                    ORDER BY ({score_expr}) DESC
                """
                params = (
                    tenant_params
                    + list(keywords)
                    + list(keywords)
                )
                cur.execute(sql, params)
                return [_row_to_cross(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"Error during keyword search: {e}")
                return []
            finally:
                cur.close()

    def structured_search(
        self,
        persons: Optional[list[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[list[str]] = None,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[CrossMemoryEntry]:
        if not any([persons, timestamp_range, location, entities, tenant_id]):
            return []
        with self._lock:
            cur = self._cur()
            try:
                conds, params = [], []
                if tenant_id:
                    conds.append("tenant_id = ?"); params.append(tenant_id)
                if persons:
                    conds.append("(" + " OR ".join("$FIND(persons, ?) > 0" for _ in persons) + ")")
                    params.extend(persons)
                if location:
                    conds.append("$FIND(location, ?) > 0"); params.append(location)
                if entities:
                    conds.append("(" + " OR ".join("$FIND(entities, ?) > 0" for _ in entities) + ")")
                    params.extend(entities)
                if timestamp_range:
                    start, end = timestamp_range
                    conds.append("timestamp >= ? AND timestamp <= ?")
                    params.extend([str(start), str(end)])

                sql = f"""
                    SELECT TOP {top_k} entry_id, text, keywords, timestamp, location,
                           persons, entities, topic, tenant_id, memory_session_id,
                           source_kind, source_id, importance, valid_from, valid_to, superseded_by
                    FROM {self.table_name}
                    WHERE {" AND ".join(conds)}
                """
                cur.execute(sql, params)
                return [_row_to_cross(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"Error during structured search: {e}")
                return []
            finally:
                cur.close()

    def get_entries_for_session(self, memory_session_id: str) -> list[CrossMemoryEntry]:
        with self._lock:
            cur = self._cur()
            try:
                cur.execute(
                    f"SELECT entry_id, text, keywords, timestamp, location, persons, entities, topic,"
                    f" tenant_id, memory_session_id, source_kind, source_id, importance, valid_from, valid_to, superseded_by"
                    f" FROM {self.table_name} WHERE memory_session_id = ?",
                    [memory_session_id],
                )
                return [_row_to_cross(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"Error fetching session entries: {e}")
                return []
            finally:
                cur.close()

    def get_all_entries(self, tenant_id: Optional[str] = None) -> list[CrossMemoryEntry]:
        with self._lock:
            cur = self._cur()
            try:
                where, params = self._build_where(tenant_id=tenant_id)
                cur.execute(
                    f"SELECT entry_id, text, keywords, timestamp, location, persons, entities, topic,"
                    f" tenant_id, memory_session_id, source_kind, source_id, importance, valid_from, valid_to, superseded_by"
                    f" FROM {self.table_name} {where}",
                    params,
                )
                return [_row_to_cross(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"Error fetching entries: {e}")
                return []
            finally:
                cur.close()

    def mark_superseded(self, old_entry_id: str, new_entry_id: str):
        with self._lock:
            cur = self._cur()
            try:
                cur.execute(
                    self._t(_UPDATE_SUPER),
                    [new_entry_id, datetime.utcnow().isoformat(), old_entry_id],
                )
                self._conn.commit()
            except Exception as e:
                print(f"Error marking entry superseded: {e}")
            finally:
                cur.close()

    def update_importance(self, entry_id: str, new_importance: float):
        with self._lock:
            cur = self._cur()
            try:
                cur.execute(self._t(_UPDATE_IMP), [float(new_importance), entry_id])
                self._conn.commit()
            except Exception as e:
                print(f"Error updating importance: {e}")
            finally:
                cur.close()

    def count_entries(
        self,
        tenant_id: Optional[str] = None,
        memory_session_id: Optional[str] = None,
    ) -> int:
        with self._lock:
            cur = self._cur()
            try:
                where, params = self._build_where(tenant_id, memory_session_id)
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name} {where}", params)
                return cur.fetchone()[0]
            except Exception as e:
                print(f"Error counting entries: {e}")
                return 0
            finally:
                cur.close()

    def clear(self, tenant_id: Optional[str] = None):
        with self._lock:
            cur = self._cur()
            try:
                if tenant_id:
                    cur.execute(
                        f"DELETE FROM {self.table_name} WHERE tenant_id = ?",
                        [tenant_id],
                    )
                    print(f"Cleared entries for tenant {tenant_id}")
                else:
                    cur.execute(f"DELETE FROM {self.table_name}")
                    print("Database cleared")
                self._conn.commit()
            except Exception as e:
                print(f"Error clearing entries: {e}")
            finally:
                cur.close()

    def optimize(self):
        pass

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
