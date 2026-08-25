"""Vector store backend contracts and the default LanceDB implementation."""

from dataclasses import dataclass
from enum import Enum
import os
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence

import lancedb
import pyarrow as pa


class ScoreOrder(str, Enum):
    """Ordering semantics for scores returned by a retrieval path."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(frozen=True)
class VectorStoreRecord:
    """A vector and its backend-neutral memory metadata."""

    entry_id: str
    vector: Sequence[float]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class VectorStoreSearchResult:
    """A backend-neutral retrieval result."""

    entry_id: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class VectorStoreBackend(Protocol):
    """Storage contract for semantic, lexical, and structured retrieval."""

    semantic_score_order: ScoreOrder
    keyword_score_order: ScoreOrder

    def insert(self, records: Sequence[VectorStoreRecord]) -> None:
        """Insert memory records and their dense vectors."""
        ...

    def semantic_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreSearchResult]:
        """Return dense-vector results ranked from best to worst."""
        ...

    def keyword_search(
        self,
        keywords: Sequence[str],
        top_k: int,
    ) -> List[VectorStoreSearchResult]:
        """Return full-text results ranked from best to worst."""
        ...

    def structured_search(
        self,
        persons: Optional[Sequence[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[VectorStoreSearchResult]:
        """Return records matching structured metadata constraints."""
        ...

    def count(self) -> int:
        """Return the number of stored records."""
        ...

    def get_all(self) -> List[VectorStoreSearchResult]:
        """Return every stored record."""
        ...

    def optimize(self) -> None:
        """Optimize backend indexes after bulk insertion."""
        ...

    def clear(self) -> None:
        """Remove all stored records and recreate backend state."""
        ...


class LanceDBVectorStoreBackend:
    """Default vector store backend using LanceDB and Tantivy."""

    semantic_score_order = ScoreOrder.ASCENDING
    keyword_score_order = ScoreOrder.DESCENDING
    _field_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        db_path: str,
        table_name: str,
        vector_dimension: int,
        storage_options: Optional[Dict[str, Any]] = None,
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.vector_dimension = vector_dimension
        self._fts_initialized = False
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        if self._is_cloud_storage:
            self.db = lancedb.connect(
                self.db_path,
                storage_options=storage_options,
            )
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_table()

    def _init_table(self) -> None:
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
                    "vector",
                    pa.list_(pa.float32(), self.vector_dimension),
                ),
            ]
        )

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing table: {self.table_name}")

    def _init_fts_index(self) -> None:
        if self._fts_initialized:
            return

        try:
            if self._is_cloud_storage:
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=False,
                    replace=True,
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
        except Exception as error:
            print(f"FTS index creation skipped: {error}")

    def insert(self, records: Sequence[VectorStoreRecord]) -> None:
        if not records:
            return

        rows = [
            {
                "entry_id": record.entry_id,
                **record.metadata,
                "vector": list(record.vector),
            }
            for record in records
        ]
        self.table.add(rows)
        self._init_fts_index()

    def semantic_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreSearchResult]:
        if self.count() == 0:
            return []

        query = self.table.search(list(query_vector))
        if filters:
            query = query.where(self._build_filter_expression(filters), prefilter=True)

        results = self._rows_to_results(
            query.limit(top_k).to_list(),
            score_field="_distance",
        )
        results.sort(key=lambda result: result.score)
        return results

    def keyword_search(
        self,
        keywords: Sequence[str],
        top_k: int,
    ) -> List[VectorStoreSearchResult]:
        if not keywords or self.count() == 0:
            return []

        query = " ".join(keywords)
        results = self._rows_to_results(
            self.table.search(query).limit(top_k).to_list(),
            score_field="_score",
        )
        results.sort(key=lambda result: result.score, reverse=True)
        return results

    def structured_search(
        self,
        persons: Optional[Sequence[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[VectorStoreSearchResult]:
        if self.count() == 0:
            return []
        if not any([persons, timestamp_range, location, entities]):
            return []

        conditions = []
        if persons:
            values = ", ".join(self._quote(value) for value in persons)
            conditions.append(f"array_has_any(persons, make_array({values}))")
        if location:
            conditions.append(f"location LIKE '%{self._escape(location)}%'")
        if entities:
            values = ", ".join(self._quote(value) for value in entities)
            conditions.append(f"array_has_any(entities, make_array({values}))")
        if timestamp_range:
            start_time, end_time = timestamp_range
            conditions.append(
                f"timestamp >= {self._quote(start_time)} "
                f"AND timestamp <= {self._quote(end_time)}"
            )

        query = self.table.search().where(" AND ".join(conditions), prefilter=True)
        if top_k:
            query = query.limit(top_k)
        return self._rows_to_results(query.to_list())

    def count(self) -> int:
        return self.table.count_rows()

    def get_all(self) -> List[VectorStoreSearchResult]:
        return self._rows_to_results(self.table.to_arrow().to_pylist())

    def optimize(self) -> None:
        self.table.optimize()

    def clear(self) -> None:
        self.db.drop_table(self.table_name)
        self._fts_initialized = False
        self._init_table()

    @classmethod
    def _build_filter_expression(cls, filters: Dict[str, Any]) -> str:
        conditions = []
        for field, value in filters.items():
            if not cls._field_pattern.fullmatch(field):
                raise ValueError(f"Invalid semantic filter field: {field!r}")
            conditions.append(f"{field} = {cls._format_filter_value(value)}")
        return " AND ".join(conditions)

    @classmethod
    def _format_filter_value(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return cls._quote(value)
        raise TypeError(
            "Semantic filters support only string, boolean, integer, and float values"
        )

    @classmethod
    def _quote(cls, value: Any) -> str:
        return "'" + cls._escape(value) + "'"

    @staticmethod
    def _escape(value: Any) -> str:
        return str(value).replace("'", "''")

    @staticmethod
    def _rows_to_results(
        rows: Sequence[Dict[str, Any]],
        score_field: Optional[str] = None,
    ) -> List[VectorStoreSearchResult]:
        results = []
        for row in rows:
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"entry_id", "vector", "_distance", "_score"}
            }
            score = None
            if score_field is not None and row.get(score_field) is not None:
                score = float(row[score_field])
            results.append(
                VectorStoreSearchResult(
                    entry_id=row["entry_id"],
                    metadata=metadata,
                    score=score,
                )
            )
        return results
