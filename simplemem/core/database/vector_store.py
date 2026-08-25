"""Provider-neutral facade for SimpleMem's three retrieval paths."""

from typing import Any, Callable, Dict, List, Optional

from simplemem.core.database.vector_store_backend import (
    LanceDBVectorStoreBackend,
    VectorStoreBackend,
    VectorStoreRecord,
    VectorStoreSearchResult,
)
from simplemem.core.models.memory_entry import MemoryEntry
from simplemem.core.settings import settings as config
from simplemem.core.utils.embedding import EmbeddingModel


BackendFactory = Callable[[int], VectorStoreBackend]


class VectorStore:
    """Coordinate embeddings with a pluggable multi-view storage backend."""

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        table_name: str = None,
        storage_options: Optional[Dict[str, Any]] = None,
        backend_factory: Optional[BackendFactory] = None,
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self.storage_options = storage_options

        if backend_factory is None:
            self.backend = LanceDBVectorStoreBackend(
                db_path=self.db_path,
                table_name=self.table_name,
                vector_dimension=self.embedding_model.dimension,
                storage_options=self.storage_options,
            )
        else:
            self.backend = backend_factory(self.embedding_model.dimension)

    @property
    def db(self) -> Any:
        """Expose the default backend's database handle for compatibility."""
        return getattr(self.backend, "db", None)

    @property
    def table(self) -> Any:
        """Expose the default backend's table handle for compatibility."""
        return getattr(self.backend, "table", None)

    def add_entries(self, entries: List[MemoryEntry]) -> None:
        """Embed and insert a batch of memory entries."""
        if not entries:
            return

        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)
        records = [
            VectorStoreRecord(
                entry_id=entry.entry_id,
                vector=vector.tolist(),
                metadata={
                    "lossless_restatement": entry.lossless_restatement,
                    "keywords": entry.keywords,
                    "timestamp": entry.timestamp or "",
                    "location": entry.location or "",
                    "persons": entry.persons,
                    "entities": entry.entities,
                    "topic": entry.topic or "",
                },
            )
            for entry, vector in zip(entries, vectors)
        ]

        self.backend.insert(records)
        print(f"Added {len(entries)} memory entries")

    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Search the semantic retrieval path."""
        try:
            if self.backend.count() == 0:
                return []

            query_vector = self.embedding_model.encode_single(query, is_query=True)
            results = self.backend.semantic_search(
                query_vector.tolist(),
                top_k=top_k,
            )
            return self._results_to_entries(results)
        except Exception as error:
            print(f"Error during semantic search: {error}")
            return []

    def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 3,
    ) -> List[MemoryEntry]:
        """Search the lexical full-text retrieval path."""
        try:
            if not keywords or self.backend.count() == 0:
                return []
            return self._results_to_entries(
                self.backend.keyword_search(keywords, top_k=top_k)
            )
        except Exception as error:
            print(f"Error during keyword search: {error}")
            return []

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Search the structured metadata retrieval path."""
        try:
            if self.backend.count() == 0:
                return []
            if not any([persons, timestamp_range, location, entities]):
                return []

            return self._results_to_entries(
                self.backend.structured_search(
                    persons=persons,
                    timestamp_range=timestamp_range,
                    location=location,
                    entities=entities,
                    top_k=top_k,
                )
            )
        except Exception as error:
            print(f"Error during structured search: {error}")
            return []

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all memory entries."""
        return self._results_to_entries(self.backend.get_all())

    def optimize(self) -> None:
        """Optimize backend indexes after bulk insertions."""
        self.backend.optimize()
        print("Table optimized")

    def clear(self) -> None:
        """Clear all backend data."""
        self.backend.clear()
        print("Database cleared")

    @staticmethod
    def _results_to_entries(
        results: List[VectorStoreSearchResult],
    ) -> List[MemoryEntry]:
        entries = []
        for result in results:
            try:
                metadata = result.metadata
                entries.append(
                    MemoryEntry(
                        entry_id=result.entry_id,
                        lossless_restatement=metadata["lossless_restatement"],
                        keywords=list(metadata.get("keywords") or []),
                        timestamp=metadata.get("timestamp") or None,
                        location=metadata.get("location") or None,
                        persons=list(metadata.get("persons") or []),
                        entities=list(metadata.get("entities") or []),
                        topic=metadata.get("topic") or None,
                    )
                )
            except Exception as error:
                print(f"Warning: Failed to parse result: {error}")
        return entries
