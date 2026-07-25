"""Optional Milvus implementation of the SimpleMem vector store contract."""

import json
import math
import os
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Sequence

from simplemem.core.database.vector_store_backend import (
    ScoreOrder,
    VectorStoreRecord,
    VectorStoreSearchResult,
)


class MilvusVectorStoreBackend:
    """Store SimpleMem records in Milvus Lite, Milvus, or Zilliz Cloud."""

    semantic_score_order = ScoreOrder.ASCENDING
    keyword_score_order = ScoreOrder.DESCENDING

    _ENTRY_ID_MAX_LENGTH = 4096
    _TEXT_MAX_LENGTH = 65535
    _METADATA_MAX_LENGTH = 8192
    _ARRAY_MAX_CAPACITY = 1024
    _ARRAY_ITEM_MAX_LENGTH = 4096
    _SCALAR_FILTER_FIELDS = {
        "entry_id",
        "lossless_restatement",
        "timestamp",
        "location",
        "topic",
    }
    _ARRAY_FILTER_FIELDS = {"keywords", "persons", "entities"}
    _METADATA_FIELDS = (
        "lossless_restatement",
        "keywords",
        "timestamp",
        "location",
        "persons",
        "entities",
        "topic",
    )
    _OUTPUT_FIELDS = ["entry_id", *_METADATA_FIELDS]
    _REMOTE_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
    _WINDOWS_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")

    def __init__(
        self,
        collection_name: Optional[str] = None,
        vector_dimension: int = 0,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        db_name: Optional[str] = None,
        consistency_level: Optional[str] = None,
    ):
        if vector_dimension <= 0:
            raise ValueError("vector_dimension must be greater than zero")

        MilvusClient, DataType, Function, FunctionType = self._load_pymilvus()
        self._data_type = DataType
        self._function = Function
        self._function_type = FunctionType
        self.collection_name = collection_name or os.getenv(
            "MILVUS_COLLECTION_NAME", "memory_entries"
        )
        self.vector_dimension = vector_dimension
        self.uri = uri if uri is not None else os.getenv("MILVUS_URI", "./milvus.db")
        self.token = token if token is not None else os.getenv("MILVUS_TOKEN", "")
        self.db_name = (
            db_name if db_name is not None else os.getenv("MILVUS_DB_NAME", "")
        )
        self.consistency_level = consistency_level or os.getenv(
            "MILVUS_CONSISTENCY_LEVEL", "Session"
        )
        self._use_raw_cosine_distance = self._is_lite_3_0_cosine_distance(
            self.uri,
            self._package_version("milvus-lite"),
        )

        client_options = {"uri": self.uri}
        if self.token:
            client_options["token"] = self.token
        if self.db_name:
            client_options["db_name"] = self.db_name
        self.client = MilvusClient(**client_options)
        try:
            self._init_collection()
        except Exception:
            self.client.close()
            raise

    @staticmethod
    def _load_pymilvus():
        try:
            from pymilvus import DataType, Function, FunctionType, MilvusClient
        except ImportError as error:
            raise ImportError(
                "Milvus support requires the optional dependency. "
                'Install SimpleMem with `pip install -e ".[milvus]"`.'
            ) from error
        return MilvusClient, DataType, Function, FunctionType

    @staticmethod
    def _package_version(package_name: str) -> Optional[str]:
        try:
            return version(package_name)
        except PackageNotFoundError:
            return None

    @classmethod
    def _is_lite_3_0_cosine_distance(
        cls,
        uri: str,
        milvus_lite_version: Optional[str],
    ) -> bool:
        return cls._is_local_path_uri(uri) and milvus_lite_version in {"3.0", "3.0.0"}

    @classmethod
    def _is_local_path_uri(cls, uri: str) -> bool:
        if cls._REMOTE_URI_PATTERN.match(uri):
            return False
        if cls._WINDOWS_PATH_PATTERN.match(uri):
            return True
        return ":" not in uri

    def _init_collection(self) -> None:
        if self.client.has_collection(collection_name=self.collection_name):
            self._validate_collection_schema()
            return
        self._create_collection()

    def _create_collection(self) -> None:
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="entry_id",
            datatype=self._data_type.VARCHAR,
            is_primary=True,
            max_length=self._ENTRY_ID_MAX_LENGTH,
        )
        schema.add_field(
            field_name="lossless_restatement",
            datatype=self._data_type.VARCHAR,
            max_length=self._TEXT_MAX_LENGTH,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params={"type": "standard"},
        )
        for field_name in ("keywords", "persons", "entities"):
            schema.add_field(
                field_name=field_name,
                datatype=self._data_type.ARRAY,
                element_type=self._data_type.VARCHAR,
                max_capacity=self._ARRAY_MAX_CAPACITY,
                max_length=self._ARRAY_ITEM_MAX_LENGTH,
            )
        for field_name in ("timestamp", "location", "topic"):
            schema.add_field(
                field_name=field_name,
                datatype=self._data_type.VARCHAR,
                max_length=self._METADATA_MAX_LENGTH,
            )
        schema.add_field(
            field_name="vector",
            datatype=self._data_type.FLOAT_VECTOR,
            dim=self.vector_dimension,
        )
        schema.add_field(
            field_name="sparse",
            datatype=self._data_type.SPARSE_FLOAT_VECTOR,
        )
        schema.add_function(
            self._function(
                name="lossless_restatement_bm25",
                function_type=self._function_type.BM25,
                input_field_names=["lossless_restatement"],
                output_field_names=["sparse"],
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse",
            index_type="AUTOINDEX",
            metric_type="BM25",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self.consistency_level,
        )

    def _validate_collection_schema(self) -> None:
        description = self.client.describe_collection(
            collection_name=self.collection_name
        )
        fields = {field["name"]: field for field in description.get("fields", [])}
        expected_types = {
            "entry_id": self._data_type.VARCHAR,
            "lossless_restatement": self._data_type.VARCHAR,
            "keywords": self._data_type.ARRAY,
            "timestamp": self._data_type.VARCHAR,
            "location": self._data_type.VARCHAR,
            "persons": self._data_type.ARRAY,
            "entities": self._data_type.ARRAY,
            "topic": self._data_type.VARCHAR,
            "vector": self._data_type.FLOAT_VECTOR,
            "sparse": self._data_type.SPARSE_FLOAT_VECTOR,
        }
        for field_name, expected_type in expected_types.items():
            field = fields.get(field_name)
            if field is None:
                raise ValueError(
                    f"Milvus collection {self.collection_name!r} is missing required "
                    f"field {field_name!r}"
                )
            if field.get("type") != expected_type:
                raise ValueError(
                    f"Milvus collection {self.collection_name!r} field "
                    f"{field_name!r} has type {field.get('type')!r}; "
                    f"expected {expected_type!r}"
                )

        primary_field = fields["entry_id"]
        if not primary_field.get("is_primary"):
            raise ValueError(
                f"Milvus collection {self.collection_name!r} must use entry_id as "
                "its primary key"
            )
        vector_dimension = int(fields["vector"].get("params", {}).get("dim", 0))
        if vector_dimension != self.vector_dimension:
            raise ValueError(
                f"Milvus collection {self.collection_name!r} has vector dimension "
                f"{vector_dimension}; expected {self.vector_dimension}"
            )
        for field_name in self._ARRAY_FILTER_FIELDS:
            if fields[field_name].get("element_type") != self._data_type.VARCHAR:
                raise ValueError(
                    f"Milvus collection {self.collection_name!r} field "
                    f"{field_name!r} must contain VARCHAR values"
                )

        functions = description.get("functions", [])
        has_bm25_function = any(
            function.get("type") == self._function_type.BM25
            and function.get("input_field_names") == ["lossless_restatement"]
            and function.get("output_field_names") == ["sparse"]
            for function in functions
        )
        if not has_bm25_function:
            raise ValueError(
                f"Milvus collection {self.collection_name!r} is missing the "
                "lossless_restatement BM25 function"
            )

    def insert(self, records: Sequence[VectorStoreRecord]) -> None:
        if not records:
            return
        rows = [self._record_to_row(record) for record in records]
        self.client.insert(collection_name=self.collection_name, data=rows)

    def _record_to_row(self, record: VectorStoreRecord) -> Dict[str, Any]:
        if not isinstance(record.entry_id, str):
            raise TypeError("Milvus entry_id values must be strings")
        self._validate_string_length(
            "entry_id", record.entry_id, self._ENTRY_ID_MAX_LENGTH
        )
        vector = [float(value) for value in record.vector]
        if len(vector) != self.vector_dimension:
            raise ValueError(
                f"Record {record.entry_id!r} has vector dimension {len(vector)}; "
                f"expected {self.vector_dimension}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Record {record.entry_id!r} contains a non-finite vector")

        unknown_fields = set(record.metadata) - set(self._METADATA_FIELDS)
        if unknown_fields:
            raise ValueError(
                "Milvus records contain unsupported metadata fields: "
                + ", ".join(sorted(unknown_fields))
            )
        row = {
            "entry_id": record.entry_id,
            "lossless_restatement": self._metadata_string(
                record.metadata, "lossless_restatement", self._TEXT_MAX_LENGTH
            ),
            "keywords": self._metadata_string_array(record.metadata, "keywords"),
            "timestamp": self._metadata_string(
                record.metadata, "timestamp", self._METADATA_MAX_LENGTH
            ),
            "location": self._metadata_string(
                record.metadata, "location", self._METADATA_MAX_LENGTH
            ),
            "persons": self._metadata_string_array(record.metadata, "persons"),
            "entities": self._metadata_string_array(record.metadata, "entities"),
            "topic": self._metadata_string(
                record.metadata, "topic", self._METADATA_MAX_LENGTH
            ),
            "vector": vector,
        }
        return row

    @classmethod
    def _metadata_string(
        cls,
        metadata: Dict[str, Any],
        field_name: str,
        max_length: int,
    ) -> str:
        value = metadata.get(field_name, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise TypeError(f"Milvus metadata field {field_name!r} must be a string")
        cls._validate_string_length(field_name, value, max_length)
        return value

    @classmethod
    def _metadata_string_array(
        cls,
        metadata: Dict[str, Any],
        field_name: str,
    ) -> List[str]:
        value = metadata.get(field_name, [])
        if value is None:
            value = []
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(
                f"Milvus metadata field {field_name!r} must be a sequence of strings"
            )
        values = list(value)
        if len(values) > cls._ARRAY_MAX_CAPACITY:
            raise ValueError(
                f"Milvus metadata field {field_name!r} exceeds the maximum of "
                f"{cls._ARRAY_MAX_CAPACITY} values"
            )
        for item in values:
            if not isinstance(item, str):
                raise TypeError(
                    f"Milvus metadata field {field_name!r} must contain only strings"
                )
            cls._validate_string_length(field_name, item, cls._ARRAY_ITEM_MAX_LENGTH)
        return values

    @staticmethod
    def _validate_string_length(
        field_name: str,
        value: str,
        max_length: int,
    ) -> None:
        if len(value.encode("utf-8")) > max_length:
            raise ValueError(
                f"Milvus field {field_name!r} exceeds its {max_length}-byte limit"
            )

    def semantic_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreSearchResult]:
        if top_k <= 0 or self.count() == 0:
            return []
        vector = [float(value) for value in query_vector]
        if len(vector) != self.vector_dimension:
            raise ValueError(
                f"Query vector has dimension {len(vector)}; "
                f"expected {self.vector_dimension}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Query vector contains a non-finite value")

        expression = self._build_filter_expression(filters or {})
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="vector",
            filter=expression,
            limit=top_k,
            output_fields=self._OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE", "params": {}},
            consistency_level=self.consistency_level,
        )[0]
        results = [
            self._hit_to_result(
                hit,
                score=self._semantic_distance(float(hit["distance"])),
            )
            for hit in hits
        ]
        results.sort(key=lambda result: result.score)
        return results

    def _semantic_distance(self, raw_score: float) -> float:
        if self._use_raw_cosine_distance:
            return raw_score
        return 1.0 - raw_score

    def keyword_search(
        self,
        keywords: Sequence[str],
        top_k: int,
    ) -> List[VectorStoreSearchResult]:
        if top_k <= 0 or not keywords or self.count() == 0:
            return []
        if isinstance(keywords, (str, bytes)) or not all(
            isinstance(keyword, str) for keyword in keywords
        ):
            raise TypeError("Milvus keyword queries must be a sequence of strings")
        query = " ".join(keyword for keyword in keywords if keyword.strip()).strip()
        if not query:
            return []

        hits = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="sparse",
            limit=top_k,
            output_fields=self._OUTPUT_FIELDS,
            search_params={"metric_type": "BM25", "params": {}},
            consistency_level=self.consistency_level,
        )[0]
        results = [
            self._hit_to_result(
                hit,
                score=self._bm25_relevance(float(hit["distance"])),
            )
            for hit in hits
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results

    @staticmethod
    def _bm25_relevance(raw_score: float) -> float:
        return -raw_score if raw_score < 0 else raw_score

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
        if top_k is not None and top_k <= 0:
            return []
        if not any([persons, timestamp_range, location, entities]):
            return []

        conditions = []
        if persons:
            conditions.append(
                "ARRAY_CONTAINS_ANY(persons, "
                f"{self._format_string_list(persons, 'persons')})"
            )
        if location:
            if not isinstance(location, str):
                raise TypeError("Milvus location filters must be strings")
            if "%" in location or "_" in location:
                raise ValueError(
                    "Milvus location filters do not accept LIKE wildcard characters"
                )
            conditions.append(f"location like {self._quote(f'%{location}%')}")
        if entities:
            conditions.append(
                "ARRAY_CONTAINS_ANY(entities, "
                f"{self._format_string_list(entities, 'entities')})"
            )
        if timestamp_range:
            if not isinstance(timestamp_range, tuple) or len(timestamp_range) != 2:
                raise TypeError("timestamp_range must be a two-item tuple")
            start_time, end_time = timestamp_range
            if not isinstance(start_time, str) or not isinstance(end_time, str):
                raise TypeError("Milvus timestamp bounds must be strings")
            conditions.append(
                f"timestamp >= {self._quote(start_time)} "
                f"and timestamp <= {self._quote(end_time)}"
            )

        expression = " and ".join(conditions)
        if top_k is not None:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter=expression,
                output_fields=self._OUTPUT_FIELDS,
                limit=top_k,
                consistency_level=self.consistency_level,
            )
        else:
            rows = self._query_all(filter_expression=expression)
        return [self._row_to_result(row) for row in rows]

    @classmethod
    def _build_filter_expression(cls, filters: Dict[str, Any]) -> str:
        conditions = []
        for field_name, value in filters.items():
            if field_name in cls._SCALAR_FILTER_FIELDS:
                if cls._is_filter_sequence(value):
                    conditions.append(
                        f"{field_name} in {cls._format_string_list(value, field_name)}"
                    )
                elif isinstance(value, str):
                    conditions.append(f"{field_name} == {cls._quote(value)}")
                else:
                    raise TypeError(
                        f"Milvus scalar filter {field_name!r} supports only strings "
                        "or sequences of strings"
                    )
            elif field_name in cls._ARRAY_FILTER_FIELDS:
                if cls._is_filter_sequence(value):
                    values = cls._format_string_list(value, field_name)
                    conditions.append(f"ARRAY_CONTAINS_ANY({field_name}, {values})")
                elif isinstance(value, str):
                    conditions.append(
                        f"ARRAY_CONTAINS({field_name}, {cls._quote(value)})"
                    )
                else:
                    raise TypeError(
                        f"Milvus array filter {field_name!r} supports only strings "
                        "or sequences of strings"
                    )
            else:
                raise ValueError(f"Invalid semantic filter field: {field_name!r}")
        return " and ".join(conditions)

    @staticmethod
    def _is_filter_sequence(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    @classmethod
    def _format_string_list(
        cls,
        values: Sequence[Any],
        field_name: str,
    ) -> str:
        if isinstance(values, (str, bytes)):
            raise TypeError(
                f"Milvus filter {field_name!r} must be a sequence of strings"
            )
        values = list(values)
        if not values:
            raise ValueError(f"Milvus filter {field_name!r} cannot be empty")
        if not all(isinstance(value, str) for value in values):
            raise TypeError(f"Milvus filter {field_name!r} must contain only strings")
        return "[" + ", ".join(cls._quote(value) for value in values) + "]"

    @staticmethod
    def _quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    def count(self) -> int:
        rows = self.client.query(
            collection_name=self.collection_name,
            output_fields=["count(*)"],
            consistency_level=self.consistency_level,
        )
        return int(rows[0]["count(*)"]) if rows else 0

    def get_all(self) -> List[VectorStoreSearchResult]:
        return [self._row_to_result(row) for row in self._query_all()]

    def _query_all(self, filter_expression: str = "") -> List[Dict[str, Any]]:
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            filter=filter_expression,
            output_fields=self._OUTPUT_FIELDS,
            batch_size=1000,
            consistency_level=self.consistency_level,
        )
        rows_by_id = {}
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    rows_by_id[row["entry_id"]] = row
        finally:
            iterator.close()
        return list(rows_by_id.values())

    def optimize(self) -> None:
        # Milvus maintains AUTOINDEX indexes automatically. Explicit compaction is
        # asynchronous and is not required for the backend contract.
        return None

    def clear(self) -> None:
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        self._create_collection()

    def close(self) -> None:
        """Close the underlying Milvus client connection."""
        self.client.close()

    def _hit_to_result(
        self,
        hit: Dict[str, Any],
        score: float,
    ) -> VectorStoreSearchResult:
        entity = dict(hit.get("entity") or {})
        entry_id = str(entity.pop("entry_id", hit.get("id", "")))
        return VectorStoreSearchResult(
            entry_id=entry_id,
            metadata={field: entity.get(field) for field in self._METADATA_FIELDS},
            score=score,
        )

    def _row_to_result(self, row: Dict[str, Any]) -> VectorStoreSearchResult:
        return VectorStoreSearchResult(
            entry_id=str(row["entry_id"]),
            metadata={field: row.get(field) for field in self._METADATA_FIELDS},
        )
