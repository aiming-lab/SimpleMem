# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
# pyright: reportDeprecated=false
"""
Cross-session vector storage using LanceDB.
"""

from typing import List, Optional, Protocol
from datetime import datetime
import os
import threading

import lancedb  # type: ignore[import-not-found]
import pyarrow as pa  # type: ignore[import-not-found]

from simplemem.core.models.memory_entry import MemoryEntry
from simplemem.core.utils.embedding import EmbeddingModel
from cross.types import CrossMemoryEntry


class ArrowTable(Protocol):
    def to_pylist(self) -> list[dict[str, object]]: ...


class ArrowSchema(Protocol):
    @property
    def names(self) -> list[str]: ...


class LanceQuery(Protocol):
    def where(self, where: str, prefilter: bool = ...) -> "LanceQuery": ...

    def limit(self, k: int) -> "LanceQuery": ...

    def to_list(self) -> list[dict[str, object]]: ...


class LanceTable(Protocol):
    def create_fts_index(
        self,
        field: str,
        use_tantivy: bool,
        tokenizer_name: Optional[str] = None,
        replace: bool = False,
    ) -> None: ...

    def add(self, data: list[dict[str, object]]) -> None: ...

    def search(self, query: object = None) -> LanceQuery: ...

    def count_rows(self) -> int: ...

    def to_arrow(self) -> ArrowTable: ...

    def update(self, where: str, values: dict[str, object]) -> None: ...

    def delete(self, where: str) -> None: ...

    def optimize(self) -> None: ...

    @property
    def schema(self) -> ArrowSchema: ...


class LanceDBConnection(Protocol):
    def table_names(self) -> list[str]: ...

    def create_table(self, name: str, schema: object) -> LanceTable: ...

    def open_table(self, name: str) -> LanceTable: ...

    def drop_table(self, name: str) -> None: ...


class CrossSessionVectorStore:
    """
    Cross-session vector storage layer.

    Uses a separate LanceDB table to store memory entries with provenance,
    extending SimpleMem's MemoryEntry format with session tracking fields.
    Reuses SimpleMem's EmbeddingModel for vector generation.
    """

    db_path: str
    embedding_model: EmbeddingModel
    table_name: str
    db: LanceDBConnection
    table: LanceTable
    _fts_initialized: bool
    _lock: threading.RLock
    _is_cloud_storage: bool
    _schema_fields: set[str]

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        table_name: str = "cross_memory_entries",
    ):
        default_path = os.path.join(
            os.path.expanduser("~"), ".simplemem-cross", "lancedb_cross"
        )
        self.db_path = db_path or default_path
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name
        self._fts_initialized = False
        self._lock = threading.RLock()

        # Detect if using cloud storage (GCS, S3, Azure)
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        # Connect to database
        if self._is_cloud_storage:
            self.db = lancedb.connect(self.db_path)
        else:
            self.db_path = os.path.expanduser(self.db_path)
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_table()

    def _init_table(self):
        """Initialize table schema and FTS index."""
        schema = pa.schema(
            [
                pa.field("entry_id", pa.string()),
                pa.field("lossless_restatement", pa.string()),
                pa.field("keywords", pa.list_(pa.string())),
                pa.field("timestamp", pa.string()),
                pa.field("location", pa.string()),
                pa.field("persons", pa.list_(pa.string())),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("topic", pa.string()),
                pa.field(
                    "vector", pa.list_(pa.float32(), self.embedding_model.dimension)
                ),
                pa.field("tenant_id", pa.string()),
                pa.field("memory_session_id", pa.string()),
                pa.field("source_kind", pa.string()),
                pa.field("source_id", pa.int64()),
                pa.field("importance", pa.float32()),
                pa.field("valid_from", pa.string()),
                pa.field("valid_to", pa.string()),
                pa.field("superseded_by", pa.string()),
            ]
        )

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing table: {self.table_name}")

        try:
            self._schema_fields = set(self.table.schema.names)
        except Exception:
            self._schema_fields = set()

    def _init_fts_index(self):
        """Initialize Full-Text Search index on lossless_restatement column."""
        if self._fts_initialized:
            return

        try:
            if self._is_cloud_storage:
                self.table.create_fts_index(
                    "lossless_restatement", use_tantivy=False, replace=True
                )
                print("FTS index created (native mode for cloud storage)")
            else:
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=True,
                    tokenizer_name="en_stem",
                    replace=True,
                )
                print("FTS index created (Tantivy mode)")
            self._fts_initialized = True
        except Exception as e:
            print(f"FTS index creation skipped: {e}")

    def _results_to_cross_entries(
        self, results: list[dict[str, object]]
    ) -> list[CrossMemoryEntry]:
        """Convert LanceDB results to CrossMemoryEntry objects."""
        entries: list[CrossMemoryEntry] = []
        for r in results:
            try:
                entry_id = self._coerce_str(r.get("entry_id"))
                lossless_restatement = self._coerce_str(r.get("lossless_restatement"))
                keywords = self._coerce_list_str(r.get("keywords"))
                timestamp = self._coerce_optional_str(r.get("timestamp"))
                location = self._coerce_optional_str(r.get("location"))
                persons = self._coerce_list_str(r.get("persons"))
                entities = self._coerce_list_str(r.get("entities"))
                topic = self._coerce_optional_str(r.get("topic"))
                tenant_id = self._coerce_str(r.get("tenant_id"))
                memory_session_id = self._coerce_str(r.get("memory_session_id"))
                source_kind = self._coerce_str(r.get("source_kind"))
                source_id = self._coerce_optional_int(r.get("source_id"))
                importance = self._coerce_float(r.get("importance"), default=0.5)
                valid_from = self._parse_optional_datetime(r.get("valid_from"))
                valid_to = self._parse_optional_datetime(r.get("valid_to"))
                superseded_by = self._coerce_optional_str(r.get("superseded_by"))

                entries.append(
                    CrossMemoryEntry(
                        entry_id=entry_id,
                        lossless_restatement=lossless_restatement,
                        keywords=keywords,
                        timestamp=timestamp,
                        location=location,
                        persons=persons,
                        entities=entities,
                        topic=topic,
                        tenant_id=tenant_id,
                        memory_session_id=memory_session_id,
                        source_kind=source_kind,
                        source_id=source_id,
                        importance=importance,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        superseded_by=superseded_by,
                    )
                )
            except Exception as e:
                print(f"Warning: Failed to parse result: {e}")
                continue
        return entries

    def _serialize_datetime(self, value: Optional[datetime]) -> str:
        if not value:
            return ""
        return value.isoformat()

    def _coerce_str(self, value: object) -> str:
        if value is None:
            return ""
        return str(value)

    def _coerce_optional_str(self, value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        return text or None

    def _coerce_list_str(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    def _coerce_optional_int(self, value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _coerce_float(self, value: object, default: float = 0.5) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _parse_optional_datetime(self, value: object) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            if not value:
                return None
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _escape_sql_string(self, value: str) -> str:
        escaped = value.replace("'", "''")
        return escaped

    def _build_where_clause(
        self,
        tenant_id: Optional[str] = None,
        memory_session_id: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Optional[str]:
        conditions = []

        if tenant_id:
            safe_tenant = self._escape_sql_string(tenant_id)
            conditions.append(f"tenant_id = '{safe_tenant}'")

        if memory_session_id:
            safe_session = self._escape_sql_string(memory_session_id)
            conditions.append(f"memory_session_id = '{safe_session}'")

        if project and "project" in self._schema_fields:
            safe_project = self._escape_sql_string(project)
            conditions.append(f"project = '{safe_project}'")

        if not conditions:
            return None

        return " AND ".join(conditions)

    def add_entries(
        self,
        entries: list[MemoryEntry],
        tenant_id: str,
        memory_session_id: str,
        source_kind: str,
        source_id: int = 0,
        importance: float = 0.5,
    ):
        """Batch add memory entries with provenance fields."""
        if not entries:
            return

        with self._lock:
            try:
                restatements = [entry.lossless_restatement for entry in entries]
                vectors = self.embedding_model.encode_documents(restatements)
                now = datetime.utcnow().isoformat()

                data = []
                for entry, vector in zip(entries, vectors):
                    data.append(
                        {
                            "entry_id": entry.entry_id,
                            "lossless_restatement": entry.lossless_restatement,
                            "keywords": entry.keywords,
                            "timestamp": entry.timestamp or "",
                            "location": entry.location or "",
                            "persons": entry.persons,
                            "entities": entry.entities,
                            "topic": entry.topic or "",
                            "vector": vector.tolist(),
                            "tenant_id": tenant_id,
                            "memory_session_id": memory_session_id,
                            "source_kind": source_kind,
                            "source_id": source_id,
                            "importance": float(importance),
                            "valid_from": now,
                            "valid_to": "",
                            "superseded_by": "",
                        }
                    )

                self.table.add(data)
                print(f"Added {len(entries)} cross-session memory entries")

                if not self._fts_initialized:
                    self._init_fts_index()
            except Exception as e:
                print(f"Error adding cross-session entries: {e}")

    def add_cross_entries(self, cross_entries: list[CrossMemoryEntry]):
        """Batch add CrossMemoryEntry records."""
        if not cross_entries:
            return

        with self._lock:
            try:
                restatements = [entry.lossless_restatement for entry in cross_entries]
                vectors = self.embedding_model.encode_documents(restatements)

                data = []
                for entry, vector in zip(cross_entries, vectors):
                    data.append(
                        {
                            "entry_id": entry.entry_id,
                            "lossless_restatement": entry.lossless_restatement,
                            "keywords": entry.keywords,
                            "timestamp": entry.timestamp or "",
                            "location": entry.location or "",
                            "persons": entry.persons,
                            "entities": entry.entities,
                            "topic": entry.topic or "",
                            "vector": vector.tolist(),
                            "tenant_id": entry.tenant_id,
                            "memory_session_id": entry.memory_session_id,
                            "source_kind": entry.source_kind,
                            "source_id": entry.source_id or 0,
                            "importance": float(entry.importance),
                            "valid_from": self._serialize_datetime(entry.valid_from),
                            "valid_to": self._serialize_datetime(entry.valid_to),
                            "superseded_by": entry.superseded_by or "",
                        }
                    )

                self.table.add(data)
                print(f"Added {len(cross_entries)} cross-session memory entries")

                if not self._fts_initialized:
                    self._init_fts_index()
            except Exception as e:
                print(f"Error adding cross-session entries: {e}")

    def semantic_search(
        self,
        query: str,
        top_k: int = 25,
        tenant_id: Optional[str] = None,
        project: Optional[str] = None,
    ) -> list[CrossMemoryEntry]:
        """Semantic search with optional tenant or project filtering."""
        with self._lock:
            try:
                if self.table.count_rows() == 0:
                    return []

                query_vector = self.embedding_model.encode_single(query, is_query=True)
                search_query = self.table.search(query_vector.tolist())

                where_clause = self._build_where_clause(
                    tenant_id=tenant_id, project=project
                )
                if where_clause:
                    search_query = search_query.where(where_clause, prefilter=True)

                results = search_query.limit(top_k).to_list()
                return self._results_to_cross_entries(results)
            except Exception as e:
                print(f"Error during semantic search: {e}")
                return []

    def keyword_search(
        self, keywords: list[str], top_k: int = 5, tenant_id: Optional[str] = None
    ) -> list[CrossMemoryEntry]:
        """Lexical search using BM25 full-text search."""
        with self._lock:
            try:
                if not keywords or self.table.count_rows() == 0:
                    return []

                query = " ".join(keywords)
                search_query = self.table.search(query)

                where_clause = self._build_where_clause(tenant_id=tenant_id)
                if where_clause:
                    search_query = search_query.where(where_clause, prefilter=True)

                results = search_query.limit(top_k).to_list()
                return self._results_to_cross_entries(results)
            except Exception as e:
                print(f"Error during keyword search: {e}")
                return []

    def structured_search(
        self,
        persons: Optional[list[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[list[str]] = None,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[CrossMemoryEntry]:
        """Metadata filtering search with optional tenant constraint."""
        with self._lock:
            try:
                if self.table.count_rows() == 0:
                    return []

                if not any([persons, timestamp_range, location, entities, tenant_id]):
                    return []

                conditions = []

                if persons:
                    escaped_persons = [self._escape_sql_string(p) for p in persons]
                    values = ", ".join([f"'{p}'" for p in escaped_persons])
                    conditions.append(f"array_has_any(persons, make_array({values}))")

                if location:
                    safe_location = self._escape_sql_string(location)
                    safe_location = safe_location.replace("%", "\\%").replace(
                        "_", "\\_"
                    )
                    conditions.append(f"location LIKE '%{safe_location}%'")

                if entities:
                    escaped_entities = [self._escape_sql_string(e) for e in entities]
                    values = ", ".join([f"'{e}'" for e in escaped_entities])
                    conditions.append(f"array_has_any(entities, make_array({values}))")

                if timestamp_range:
                    start_time, end_time = timestamp_range
                    safe_start = self._escape_sql_string(str(start_time))
                    safe_end = self._escape_sql_string(str(end_time))
                    conditions.append(
                        f"timestamp >= '{safe_start}' AND timestamp <= '{safe_end}'"
                    )

                if tenant_id:
                    safe_tenant = self._escape_sql_string(tenant_id)
                    conditions.append(f"tenant_id = '{safe_tenant}'")

                where_clause = " AND ".join(conditions)
                query = self.table.search().where(where_clause, prefilter=True)

                if top_k:
                    query = query.limit(top_k)

                results = query.to_list()
                return self._results_to_cross_entries(results)
            except Exception as e:
                print(f"Error during structured search: {e}")
                return []

    def get_entries_for_session(self, memory_session_id: str) -> list[CrossMemoryEntry]:
        """Get all entries for a specific memory session."""
        with self._lock:
            try:
                where_clause = self._build_where_clause(
                    memory_session_id=memory_session_id
                )
                if not where_clause:
                    return []
                results = (
                    self.table.search().where(where_clause, prefilter=True).to_list()
                )
                return self._results_to_cross_entries(results)
            except Exception as e:
                print(f"Error fetching session entries: {e}")
                return []

    def get_all_entries(
        self, tenant_id: Optional[str] = None
    ) -> list[CrossMemoryEntry]:
        """Get all entries, optionally filtered by tenant."""
        with self._lock:
            try:
                where_clause = self._build_where_clause(tenant_id=tenant_id)
                if where_clause:
                    results = (
                        self.table.search()
                        .where(where_clause, prefilter=True)
                        .to_list()
                    )
                else:
                    results = self.table.to_arrow().to_pylist()
                return self._results_to_cross_entries(results)
            except Exception as e:
                print(f"Error fetching entries: {e}")
                return []

    def mark_superseded(self, old_entry_id: str, new_entry_id: str):
        """Mark an entry as superseded by another entry."""
        with self._lock:
            try:
                safe_old = self._escape_sql_string(old_entry_id)
                now = datetime.utcnow().isoformat()
                self.table.update(
                    where=f"entry_id = '{safe_old}'",
                    values={"superseded_by": new_entry_id, "valid_to": now},
                )
                print(f"Marked entry {old_entry_id} as superseded")
            except Exception as e:
                print(f"Error marking entry superseded: {e}")

    def update_importance(self, entry_id: str, new_importance: float):
        """Update importance score for an entry."""
        with self._lock:
            try:
                safe_entry = self._escape_sql_string(entry_id)
                self.table.update(
                    where=f"entry_id = '{safe_entry}'",
                    values={"importance": float(new_importance)},
                )
                print(f"Updated importance for entry {entry_id}")
            except Exception as e:
                print(f"Error updating importance: {e}")

    def clear(self, tenant_id: Optional[str] = None):
        """Clear all data or data for a specific tenant."""
        with self._lock:
            try:
                if tenant_id:
                    safe_tenant = tenant_id.replace("'", "''")
                    self.table.delete(where=f"tenant_id = '{safe_tenant}'")
                    print(f"Cleared entries for tenant {tenant_id}")
                else:
                    self.db.drop_table(self.table_name)
                    self._fts_initialized = False
                    self._init_table()
                    print("Database cleared")
            except Exception as e:
                print(f"Error clearing entries: {e}")

    def optimize(self):
        """Optimize table after bulk insertions for better query performance."""
        with self._lock:
            try:
                self.table.optimize()
                print("Table optimized")
            except Exception as e:
                print(f"Error optimizing table: {e}")

    def count_entries(
        self, tenant_id: Optional[str] = None, memory_session_id: Optional[str] = None
    ) -> int:
        """Count entries with optional tenant or session filters."""
        with self._lock:
            try:
                where_clause = self._build_where_clause(
                    tenant_id=tenant_id, memory_session_id=memory_session_id
                )
                if not where_clause:
                    return self.table.count_rows()
                results = (
                    self.table.search().where(where_clause, prefilter=True).to_list()
                )
                return len(results)
            except Exception as e:
                print(f"Error counting entries: {e}")
                return 0

    def close(self) -> None:
        """Close the vector store and release resources."""
        pass
