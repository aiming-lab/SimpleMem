"""
Vector Store - Structured Multi-View Indexing Implementation (Section 3.2)

Paper Reference: Section 3.2 - Structured Indexing
Implements the three structured indexing dimensions:
- Semantic Layer: Dense vectors v_k ∈ ℝ^d (embedding-based similarity)
- Lexical Layer: Sparse vectors h_k ∈ ℝ^|V| (BM25/keyword matching)
- Symbolic Layer: Metadata R_k = {(key, val)} (structured filtering by time, entities, etc.)
"""
from typing import List, Optional, Dict, Any
import lancedb
import pyarrow as pa
import numpy as np
from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
import config
import os


class VectorStore:
    """
    Structured Multi-View Indexing - Storage and retrieval for Atomic Entries

    Paper Reference: Section 3.2 - Structured Indexing
    Implements M(m_k) with three structured layers:
    1. Semantic Layer: Dense embedding vectors for conceptual similarity
    2. Lexical Layer: Sparse keyword vectors for precise term matching
    3. Symbolic Layer: Structured metadata for deterministic filtering
    """
    def __init__(self, db_path: str = None, embedding_model: EmbeddingModel = None, table_name: str = None):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()

        # Connect to database
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self.table = None

        self._init_table()

    def _init_table(self):
        """
        Initialize table schema
        """
        # Define schema
        schema = pa.schema([
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        # Create table if it doesn't exist
        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing table: {self.table_name}")

    def add_entries(self, entries: List[MemoryEntry]):
        """
        Batch add memory entries
        """
        if not entries:
            return

        # Generate vectors (encode documents without query prompt)
        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)

        # Build data
        data = []
        for entry, vector in zip(entries, vectors):
            data.append({
                "entry_id": entry.entry_id,
                "lossless_restatement": entry.lossless_restatement,
                "keywords": entry.keywords,
                "timestamp": entry.timestamp or "",
                "location": entry.location or "",
                "persons": entry.persons,
                "entities": entry.entities,
                "topic": entry.topic or "",
                "vector": vector.tolist()
            })

        # Add to table
        self.table.add(data)
        print(f"Added {len(entries)} memory entries")

    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """
        Semantic Layer Search - Dense vector similarity

        Paper Reference: Section 3.1
        Retrieves based on v_k = E_dense(S_k) where S_k is the lossless restatement
        """
        try:
            # Check if table is empty
            if self.table.count_rows() == 0:
                return []

            # Generate query vector (use query prompt optimization for Qwen3)
            query_vector = self.embedding_model.encode_single(query, is_query=True)

            # Execute vector search
            results = self.table.search(query_vector.tolist()).limit(top_k).to_list()

            # Convert to MemoryEntry objects
            entries = []
            for result in results:
                try:
                    entry = MemoryEntry(
                        entry_id=result["entry_id"],
                        lossless_restatement=result["lossless_restatement"],
                        keywords=list(result["keywords"]) if result["keywords"] is not None else [],
                        timestamp=result["timestamp"] if result["timestamp"] else None,
                        location=result["location"] if result["location"] else None,
                        persons=list(result["persons"]) if result["persons"] is not None else [],
                        entities=list(result["entities"]) if result["entities"] is not None else [],
                        topic=result["topic"] if result["topic"] else None
                    )
                    entries.append(entry)
                except Exception as e:
                    print(f"Warning: Failed to parse search result: {e}")
                    continue

            return entries

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

    def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[MemoryEntry]:
        """
        Lexical Layer Search - Sparse keyword matching

        Paper Reference: Section 3.1
        Retrieves based on h_k = Sparse(S_k) for precise term and entity matching
        Uses inclusion-based scoring (approximates BM25)
        """
        try:
            # Get all entries (should use more efficient method in production)
            all_entries = self.table.to_pandas()

            # Handle empty table
            if len(all_entries) == 0:
                return []

            # Handle empty keywords
            if not keywords:
                return []

            # Calculate matching scores
            scored_entries = []
            for _, row in all_entries.iterrows():
                try:
                    score = 0
                    # Convert to list to avoid array truth value ambiguity
                    row_keywords = list(row["keywords"]) if row["keywords"] is not None else []
                    row_text = str(row["lossless_restatement"]).lower()

                    for kw in keywords:
                        kw_lower = str(kw).lower()
                        # Keyword list matching
                        if len(row_keywords) > 0 and any(kw_lower in str(rk).lower() for rk in row_keywords):
                            score += 2
                        # Text matching
                        if kw_lower in row_text:
                            score += 1

                    if score > 0:
                        # Convert arrays to lists for MemoryEntry
                        entry = MemoryEntry(
                            entry_id=row["entry_id"],
                            lossless_restatement=row["lossless_restatement"],
                            keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                            timestamp=row["timestamp"] if row["timestamp"] else None,
                            location=row["location"] if row["location"] else None,
                            persons=list(row["persons"]) if row["persons"] is not None else [],
                            entities=list(row["entities"]) if row["entities"] is not None else [],
                            topic=row["topic"] if row["topic"] else None
                        )
                        scored_entries.append((score, entry))
                except Exception as e:
                    print(f"Warning: Failed to process row in keyword search: {e}")
                    continue

            # Sort by score and return top_k
            scored_entries.sort(reverse=True, key=lambda x: x[0])
            return [entry for _, entry in scored_entries[:top_k]]

        except Exception as e:
            print(f"Error during keyword search: {e}")
            return []

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Symbolic Layer Search - Metadata-based deterministic filtering

        Paper Reference: Section 3.1
        Retrieves based on R_k = {(key, val)} for structured constraints
        Enables precise filtering by time, entities, persons, and locations

        Args:
            persons: Filter by person names
            timestamp_range: Filter by time range (start, end)
            location: Filter by location
            entities: Filter by entities
            top_k: Maximum number of results to return (default: no limit)
        """
        try:
            df = self.table.to_pandas()

            # Handle empty dataframe
            if len(df) == 0:
                return []

            # If no filters provided, return empty
            if not any([persons, timestamp_range, location, entities]):
                return []

            # Apply filters using numpy array for proper pandas boolean indexing
            mask = np.ones(len(df), dtype=bool)

            if persons:
                person_mask = np.array([
                    any(p in list(row["persons"]) for p in persons) if row["persons"] is not None else False
                    for _, row in df.iterrows()
                ])
                mask = mask & person_mask

            if location:
                location_mask = np.array([
                    location.lower() in str(row["location"]).lower() if row["location"] is not None else False
                    for _, row in df.iterrows()
                ])
                mask = mask & location_mask

            if entities:
                entity_mask = np.array([
                    any(e in list(row["entities"]) for e in entities) if row["entities"] is not None else False
                    for _, row in df.iterrows()
                ])
                mask = mask & entity_mask

            if timestamp_range:
                start_time, end_time = timestamp_range
                timestamp_mask = np.array([
                    bool(row["timestamp"] and start_time <= row["timestamp"] <= end_time)
                    for _, row in df.iterrows()
                ])
                mask = mask & timestamp_mask

            # Build results - use numpy boolean array for filtering
            filtered_df = df[mask]

            # Limit results if top_k is specified
            if top_k is not None and len(filtered_df) > top_k:
                filtered_df = filtered_df.head(top_k)

            entries = []
            for _, row in filtered_df.iterrows():
                try:
                    entry = MemoryEntry(
                        entry_id=row["entry_id"],
                        lossless_restatement=row["lossless_restatement"],
                        keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                        timestamp=row["timestamp"] if row["timestamp"] else None,
                        location=row["location"] if row["location"] else None,
                        persons=list(row["persons"]) if row["persons"] is not None else [],
                        entities=list(row["entities"]) if row["entities"] is not None else [],
                        topic=row["topic"] if row["topic"] else None
                    )
                    entries.append(entry)
                except Exception as e:
                    print(f"Warning: Failed to parse filtered row: {e}")
                    continue

            return entries

        except Exception as e:
            print(f"Error during structured search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_all_entries(self) -> List[MemoryEntry]:
        """
        Get all memory entries
        """
        df = self.table.to_pandas()
        entries = []
        for _, row in df.iterrows():
            entry = MemoryEntry(
                entry_id=row["entry_id"],
                lossless_restatement=row["lossless_restatement"],
                keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                timestamp=row["timestamp"] if row["timestamp"] else None,
                location=row["location"] if row["location"] else None,
                persons=list(row["persons"]) if row["persons"] is not None else [],
                entities=list(row["entities"]) if row["entities"] is not None else [],
                topic=row["topic"] if row["topic"] else None
            )
            entries.append(entry)
        return entries

    def clear(self):
        """
        Clear all data
        """
        self.db.drop_table(self.table_name)
        self._init_table()
        print("Database cleared")
