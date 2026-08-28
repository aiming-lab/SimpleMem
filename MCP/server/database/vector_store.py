from typing import List, Optional, Dict, Any
from datetime import datetime

import iris.dbapi
from langchain_intersystems import IRISVectorStore, SimilarityMetric, Predicate
from langchain_core.embeddings import Embeddings

from ..auth.models import MemoryEntry


def _connect_args(settings) -> tuple:
    return (
        settings.IRIS_HOSTNAME,
        settings.IRIS_PORT,
        settings.IRIS_NAMESPACE,
        settings.IRIS_USERNAME,
        settings.IRIS_PASSWORD,
    )


def _doc_to_entry(doc) -> Optional[MemoryEntry]:
    m = doc.metadata
    try:
        return MemoryEntry(
            entry_id=m.get("entry_id", ""),
            lossless_restatement=doc.page_content,
            keywords=m.get("keywords") or [],
            timestamp=m.get("timestamp") or None,
            location=m.get("location") or None,
            persons=m.get("persons") or [],
            entities=m.get("entities") or [],
            topic=m.get("topic") or None,
        )
    except Exception as e:
        print(f"Warning: Failed to parse result: {e}")
        return None


class MultiTenantVectorStore:
    def __init__(
        self,
        embedding_function: Embeddings,
        settings=None,
        db_path: str = None,
        embedding_dimension: int = 2560,
    ):
        self._embedding_function = embedding_function
        self._settings = settings
        self.embedding_dimension = embedding_dimension
        self._stores: Dict[str, IRISVectorStore] = {}

    def _get_store(self, table_name: str) -> IRISVectorStore:
        if table_name not in self._stores:
            self._stores[table_name] = IRISVectorStore(
                embedding_function=self._embedding_function,
                connect_args=_connect_args(self._settings),
                collection_name=table_name,
                similarity_metric=SimilarityMetric.COSINE,
            )
        return self._stores[table_name]

    async def add_entries(
        self,
        table_name: str,
        entries: List[MemoryEntry],
        embeddings: List[List[float]],
    ) -> int:
        if len(entries) != len(embeddings):
            raise ValueError("Number of entries must match number of embeddings")
        if not entries:
            return 0

        store = self._get_store(table_name)
        created_at = datetime.utcnow().isoformat()
        texts = [e.lossless_restatement for e in entries]
        metadatas = [
            {
                "entry_id": e.entry_id,
                "keywords": e.keywords or [],
                "timestamp": e.timestamp or "",
                "location": e.location or "",
                "persons": e.persons or [],
                "entities": e.entities or [],
                "topic": e.topic or "",
                "created_at": created_at,
            }
            for e in entries
        ]
        store.add_texts(texts=texts, metadatas=metadatas, ids=[e.entry_id for e in entries])
        return len(entries)

    async def semantic_search(
        self,
        table_name: str,
        query_embedding: List[float],
        top_k: int = 25,
    ) -> List[MemoryEntry]:
        store = self._get_store(table_name)
        try:
            docs = store.similarity_search_by_vector(query_embedding, k=top_k)
            return [e for e in (_doc_to_entry(d) for d in docs) if e is not None]
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    async def keyword_search(
        self,
        table_name: str,
        keywords: List[str],
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        store = self._get_store(table_name)
        try:
            docs = store.similarity_search(" ".join(keywords), k=top_k)
            return [e for e in (_doc_to_entry(d) for d in docs) if e is not None]
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    async def structured_search(
        self,
        table_name: str,
        persons: Optional[List[str]] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        timestamp_start: Optional[str] = None,
        timestamp_end: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        store = self._get_store(table_name)
        try:
            conditions = []
            if persons:
                conditions.append({Predicate.IN: {"persons": persons}})
            if location:
                conditions.append({Predicate.CONTAINS: {"location": location}})
            if entities:
                conditions.append({Predicate.IN: {"entities": entities}})
            if timestamp_start and timestamp_end:
                conditions.append({Predicate.BETWEEN: {"timestamp": [timestamp_start, timestamp_end]}})

            if not conditions:
                return []

            filt = {Predicate.AND: conditions} if len(conditions) > 1 else conditions[0]
            query_text = " ".join(persons or entities or ["memory"])
            docs = store.similarity_search(query_text, k=top_k, filter=filt)
            return [e for e in (_doc_to_entry(d) for d in docs) if e is not None]
        except Exception as e:
            print(f"Structured search error: {e}")
            return []

    async def get_all_entries(self, table_name: str) -> List[MemoryEntry]:
        store = self._get_store(table_name)
        try:
            docs_scores = store.similarity_search_with_score("memory", k=10000)
            return [e for e in (_doc_to_entry(d) for d, _ in docs_scores) if e is not None]
        except Exception as e:
            print(f"Get all entries error: {e}")
            return []

    async def count_entries(self, table_name: str) -> int:
        store = self._get_store(table_name)
        try:
            return len(store.similarity_search_with_score("memory", k=10000))
        except Exception:
            return 0

    async def clear_table(self, table_name: str) -> bool:
        try:
            if table_name in self._stores:
                self._stores[table_name].delete_collection()
                del self._stores[table_name]
            return True
        except Exception as e:
            print(f"Clear table error: {e}")
            return False

    async def delete_table(self, table_name: str) -> bool:
        return await self.clear_table(table_name)

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        try:
            store = self._get_store(table_name)
            count = len(store.similarity_search_with_score("memory", k=10000))
            return {"table_name": table_name, "entry_count": count, "embedding_dimension": self.embedding_dimension}
        except Exception as e:
            return {"table_name": table_name, "entry_count": 0, "embedding_dimension": self.embedding_dimension, "error": str(e)}
