"""
Multi-tenant vector store for SimpleMem MCP Server
Uses LanceDB for vector storage with per-user table isolation
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime

import lancedb
import pyarrow as pa

from ..auth.models import MemoryEntry


# LanceDB schema for memory entries
def get_memory_schema(embedding_dimension: int = 2560) -> pa.Schema:
    """Get PyArrow schema for memory entries"""
    return pa.schema([
        pa.field("entry_id", pa.string()),
        pa.field("lossless_restatement", pa.string()),
        pa.field("keywords", pa.list_(pa.string())),
        pa.field("timestamp", pa.string()),
        pa.field("location", pa.string()),
        pa.field("persons", pa.list_(pa.string())),
        pa.field("entities", pa.list_(pa.string())),
        pa.field("topic", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), embedding_dimension)),
        pa.field("created_at", pa.string()),
    ])


class MultiTenantVectorStore:
    """
    Multi-tenant vector storage with per-user table isolation.
    Each user gets their own LanceDB table for complete data isolation.
    """

    def __init__(
        self,
        db_path: str = "./data/lancedb",
        embedding_dimension: int = 2560,
    ):
        self.db_path = db_path
        self.embedding_dimension = embedding_dimension
        os.makedirs(db_path, exist_ok=True)

        # Connect to LanceDB
        self.db = lancedb.connect(db_path)

        # Cache for opened tables
        self._tables: Dict[str, Any] = {}

    def _get_table(self, table_name: str) -> Any:
        """Get or create a user's table"""
        if table_name not in self._tables:
            if table_name in self.db.table_names():
                self._tables[table_name] = self.db.open_table(table_name)
            else:
                # Create new table with schema
                schema = get_memory_schema(self.embedding_dimension)
                self._tables[table_name] = self.db.create_table(
                    table_name,
                    schema=schema,
                )
        return self._tables[table_name]

    async def add_entries(
        self,
        table_name: str,
        entries: List[MemoryEntry],
        embeddings: List[List[float]],
    ) -> int:
        """
        Add memory entries to a user's table

        Args:
            table_name: User's table name
            entries: List of MemoryEntry objects
            embeddings: List of embedding vectors

        Returns:
            Number of entries added
        """
        if len(entries) != len(embeddings):
            raise ValueError("Number of entries must match number of embeddings")

        if not entries:
            return 0

        table = self._get_table(table_name)
        created_at = datetime.utcnow().isoformat()

        # Build records
        records = []
        for entry, embedding in zip(entries, embeddings):
            records.append({
                "entry_id": entry.entry_id,
                "lossless_restatement": entry.lossless_restatement,
                "keywords": entry.keywords or [],
                "timestamp": entry.timestamp or "",
                "location": entry.location or "",
                "persons": entry.persons or [],
                "entities": entry.entities or [],
                "topic": entry.topic or "",
                "vector": embedding,
                "created_at": created_at,
            })

        # Add to table
        table.add(records)
        return len(records)

    async def semantic_search(
        self,
        table_name: str,
        query_embedding: List[float],
        top_k: int = 25,
    ) -> List[MemoryEntry]:
        """
        Perform semantic search using vector similarity

        Args:
            table_name: User's table name
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of matching MemoryEntry objects
        """
        table = self._get_table(table_name)

        try:
            # Check if table has data
            if table.count_rows() == 0:
                return []

            results = (
                table.search(query_embedding)
                .limit(top_k)
                .to_pandas()
            )

            entries = []
            for _, row in results.iterrows():
                entries.append(MemoryEntry(
                    entry_id=row["entry_id"],
                    lossless_restatement=row["lossless_restatement"],
                    keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                    timestamp=row["timestamp"] if row["timestamp"] else None,
                    location=row["location"] if row["location"] else None,
                    persons=list(row["persons"]) if row["persons"] is not None else [],
                    entities=list(row["entities"]) if row["entities"] is not None else [],
                    topic=row["topic"] if row["topic"] else None,
                ))

            return entries

        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    async def keyword_search(
        self,
        table_name: str,
        keywords: List[str],
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        """
        Perform keyword-based search (BM25-style matching)

        Args:
            table_name: User's table name
            keywords: List of keywords to match
            top_k: Number of results to return

        Returns:
            List of matching MemoryEntry objects
        """
        table = self._get_table(table_name)

        try:
            if table.count_rows() == 0:
                return []

            # Load all entries for keyword matching
            df = table.to_pandas()

            # Score each entry
            scores = []
            for idx, row in df.iterrows():
                score = 0
                entry_keywords = set(k.lower() for k in (row["keywords"] or []))
                entry_text = row["lossless_restatement"].lower()

                for kw in keywords:
                    kw_lower = kw.lower()
                    # Keyword list match: 2 points
                    if kw_lower in entry_keywords:
                        score += 2
                    # Text match: 1 point
                    if kw_lower in entry_text:
                        score += 1

                scores.append((idx, score))

            # Sort by score and get top-k
            scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, score in scores[:top_k] if score > 0]

            entries = []
            for idx in top_indices:
                row = df.iloc[idx]
                entries.append(MemoryEntry(
                    entry_id=row["entry_id"],
                    lossless_restatement=row["lossless_restatement"],
                    keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                    timestamp=row["timestamp"] if row["timestamp"] else None,
                    location=row["location"] if row["location"] else None,
                    persons=list(row["persons"]) if row["persons"] is not None else [],
                    entities=list(row["entities"]) if row["entities"] is not None else [],
                    topic=row["topic"] if row["topic"] else None,
                ))

            return entries

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
        """
        Perform structured/metadata-based search

        Args:
            table_name: User's table name
            persons: Filter by person names
            location: Filter by location
            entities: Filter by entities
            timestamp_start: Start of timestamp range
            timestamp_end: End of timestamp range
            top_k: Number of results to return

        Returns:
            List of matching MemoryEntry objects
        """
        table = self._get_table(table_name)

        try:
            if table.count_rows() == 0:
                return []

            df = table.to_pandas()

            # Apply filters
            mask = [True] * len(df)

            if persons:
                persons_lower = set(p.lower() for p in persons)
                for i, row in df.iterrows():
                    row_persons = set(p.lower() for p in (row["persons"] or []))
                    if not persons_lower.intersection(row_persons):
                        mask[i] = False

            if location:
                location_lower = location.lower()
                for i, row in df.iterrows():
                    if mask[i] and row["location"]:
                        if location_lower not in row["location"].lower():
                            mask[i] = False
                    elif mask[i]:
                        mask[i] = False

            if entities:
                entities_lower = set(e.lower() for e in entities)
                for i, row in df.iterrows():
                    if mask[i]:
                        row_entities = set(e.lower() for e in (row["entities"] or []))
                        if not entities_lower.intersection(row_entities):
                            mask[i] = False

            if timestamp_start:
                for i, row in df.iterrows():
                    if mask[i] and row["timestamp"]:
                        if row["timestamp"] < timestamp_start:
                            mask[i] = False

            if timestamp_end:
                for i, row in df.iterrows():
                    if mask[i] and row["timestamp"]:
                        if row["timestamp"] > timestamp_end:
                            mask[i] = False

            # Get filtered results
            filtered_df = df[[m for m in mask]][:top_k]

            entries = []
            for _, row in filtered_df.iterrows():
                entries.append(MemoryEntry(
                    entry_id=row["entry_id"],
                    lossless_restatement=row["lossless_restatement"],
                    keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                    timestamp=row["timestamp"] if row["timestamp"] else None,
                    location=row["location"] if row["location"] else None,
                    persons=list(row["persons"]) if row["persons"] is not None else [],
                    entities=list(row["entities"]) if row["entities"] is not None else [],
                    topic=row["topic"] if row["topic"] else None,
                ))

            return entries

        except Exception as e:
            print(f"Structured search error: {e}")
            return []

    async def get_all_entries(self, table_name: str) -> List[MemoryEntry]:
        """Get all entries from a user's table"""
        table = self._get_table(table_name)

        try:
            if table.count_rows() == 0:
                return []

            df = table.to_pandas()
            entries = []

            for _, row in df.iterrows():
                entries.append(MemoryEntry(
                    entry_id=row["entry_id"],
                    lossless_restatement=row["lossless_restatement"],
                    keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                    timestamp=row["timestamp"] if row["timestamp"] else None,
                    location=row["location"] if row["location"] else None,
                    persons=list(row["persons"]) if row["persons"] is not None else [],
                    entities=list(row["entities"]) if row["entities"] is not None else [],
                    topic=row["topic"] if row["topic"] else None,
                ))

            return entries

        except Exception as e:
            print(f"Get all entries error: {e}")
            return []

    async def count_entries(self, table_name: str) -> int:
        """Count entries in a user's table"""
        table = self._get_table(table_name)
        try:
            return table.count_rows()
        except Exception:
            return 0

    async def clear_table(self, table_name: str) -> bool:
        """Clear all entries from a user's table"""
        try:
            if table_name in self._tables:
                del self._tables[table_name]

            if table_name in self.db.table_names():
                self.db.drop_table(table_name)

            # Recreate empty table
            self._get_table(table_name)
            return True

        except Exception as e:
            print(f"Clear table error: {e}")
            return False

    async def delete_table(self, table_name: str) -> bool:
        """Completely delete a user's table"""
        try:
            if table_name in self._tables:
                del self._tables[table_name]

            if table_name in self.db.table_names():
                self.db.drop_table(table_name)

            return True

        except Exception as e:
            print(f"Delete table error: {e}")
            return False

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """Get statistics for a user's table"""
        try:
            table = self._get_table(table_name)
            count = table.count_rows()

            return {
                "table_name": table_name,
                "entry_count": count,
                "embedding_dimension": self.embedding_dimension,
            }
        except Exception as e:
            return {
                "table_name": table_name,
                "entry_count": 0,
                "embedding_dimension": self.embedding_dimension,
                "error": str(e),
            }
