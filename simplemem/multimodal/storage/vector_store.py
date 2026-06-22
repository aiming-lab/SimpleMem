"""
Vector Store for Omni-Memory.

Provides efficient vector similarity search for MAU embeddings
using FAISS or other backends.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import json
import threading

from simplemem.multimodal.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector Store for efficient similarity search.

    Supports multiple backends:
    - FAISS (default, high performance)
    - Numpy (fallback, simple but slower)
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
        use_faiss: bool = True,
    ):
        self.config = config or EmbeddingConfig()
        self.embedding_dim = self.config.embedding_dim
        self.storage_path = Path(storage_path) if storage_path else Path("./omni_memory_data/vectors")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._use_faiss = use_faiss and self._check_faiss()
        self._index = None
        self._id_mapping: List[str] = []  # index position -> mau_id
        self._lock = threading.RLock()

        self._initialize_index()

    def _check_faiss(self) -> bool:
        """Check if FAISS is available."""
        try:
            import faiss
            return True
        except ImportError:
            logger.warning("FAISS not installed, using numpy fallback")
            return False

    def _load_id_mapping(self, mapping_file: Path) -> List[str]:
        """Load ID mapping from JSON file. Never raises; returns [] on empty/invalid file."""
        if not mapping_file.exists():
            return []
        try:
            raw = mapping_file.read_text(encoding="utf-8").strip()
            if not raw:
                return []
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse id_mapping.json at %s: %s. Using empty mapping.",
                mapping_file, e,
            )
            return []
        except Exception as e:
            logger.warning(
                "Error reading id_mapping.json at %s: %s. Using empty mapping.",
                mapping_file, e,
            )
            return []

    def _initialize_index(self):
        """Initialize the vector index."""
        index_file = self.storage_path / "vectors.index"
        mapping_file = self.storage_path / "id_mapping.json"

        if self._use_faiss:
            import faiss

            if index_file.exists():
                self._index = faiss.read_index(str(index_file))
                self._id_mapping = self._load_id_mapping(mapping_file)
                ntotal = self._index.ntotal
                logger.info(f"Loaded FAISS index with {ntotal} vectors")
                if ntotal == 0:
                    logger.warning(
                        "Vector index file exists but contains 0 vectors. "
                        "Retrieval will return no results. To fix: delete the index directory and re-run without --reuse-data to re-ingest."
                    )
            else:
                # Create new index (Inner Product for cosine similarity with normalized vectors)
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                logger.info(
                    "Created new FAISS index (file not found: %s); save after ingest to persist.",
                    index_file,
                )
        else:
            # Numpy fallback
            npy_file = index_file.with_suffix('.npy')
            if npy_file.exists():
                self._index = np.load(str(npy_file))
                self._id_mapping = self._load_id_mapping(mapping_file)
            else:
                self._index = np.zeros((0, self.embedding_dim), dtype=np.float32)

    def add(self, mau_id: str, embedding: List[float]) -> None:
        """
        Add an embedding to the index.

        Args:
            mau_id: The MAU ID
            embedding: The embedding vector
        """
        with self._lock:
            vec = np.array(embedding, dtype=np.float32).reshape(1, -1)

            # Normalize for cosine similarity
            vec = vec / (np.linalg.norm(vec) + 1e-8)

            if self._use_faiss:
                self._index.add(vec)
            else:
                self._index = np.vstack([self._index, vec]) if self._index.size else vec

            self._id_mapping.append(mau_id)
            
            # Auto-save periodically (every 100 additions)
            if len(self._id_mapping) % 100 == 0:
                try:
                    self.save()
                except Exception as e:
                    logger.warning(f"Failed to auto-save vector index: {e}")

    def add_batch(self, items: List[Tuple[str, List[float]]]) -> None:
        """
        Add multiple embeddings at once.

        Args:
            items: List of (mau_id, embedding) tuples
        """
        with self._lock:
            vectors = []
            for mau_id, embedding in items:
                vec = np.array(embedding, dtype=np.float32)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                vectors.append(vec)
                self._id_mapping.append(mau_id)

            vectors = np.array(vectors, dtype=np.float32)

            if self._use_faiss:
                self._index.add(vectors)
            else:
                self._index = np.vstack([self._index, vectors]) if self._index.size else vectors

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (mau_id, similarity_score) tuples
        """
        if len(self._id_mapping) == 0:
            logger.debug(f"Vector store is empty, returning no results")
            return []

        with self._lock:
            query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            query = query / (np.linalg.norm(query) + 1e-8)

            if self._use_faiss:
                scores, indices = self._index.search(query, min(top_k, len(self._id_mapping)))
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self._id_mapping) and score >= threshold:
                        results.append((self._id_mapping[idx], float(score)))
                logger.debug(f"FAISS search: {len(results)} results (threshold={threshold}, max_score={max(scores[0]) if len(scores[0]) > 0 else 0:.3f})")
            else:
                # Numpy fallback
                if self._index.size == 0:
                    logger.debug(f"Numpy index is empty, returning no results")
                    return []
                # Query is already normalized, index should be normalized too (done in add())
                # Compute cosine similarity: dot product of normalized vectors
                scores = np.dot(self._index, query.T).flatten()
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = []
                for idx in top_indices:
                    if idx < len(self._id_mapping):
                        score = float(scores[idx])
                        if score >= threshold:
                            results.append((self._id_mapping[idx], score))
                max_score = max(scores) if len(scores) > 0 else 0.0
                min_score = min(scores) if len(scores) > 0 else 0.0
                logger.info(f"Numpy search: {len(results)}/{len(self._id_mapping)} results (threshold={threshold}, max_score={max_score:.3f}, min_score={min_score:.3f})")
                if len(results) == 0 and len(self._id_mapping) > 0:
                    logger.warning(f"No results above threshold {threshold}, but max_score={max_score:.3f}. Lowering threshold to accept all results.")
                    # Return top results regardless of threshold
                    for idx in top_indices[:top_k]:
                        if idx < len(self._id_mapping):
                            results.append((self._id_mapping[idx], float(scores[idx])))
                    logger.info(f"Returning {len(results)} results without threshold filter")

        return results

    def size(self) -> int:
        """Get the number of vectors in the store."""
        with self._lock:
            return len(self._id_mapping)

    def search_batch(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search for multiple queries.

        Args:
            query_embeddings: List of query vectors
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        results = []
        for query in query_embeddings:
            results.append(self.search(query, top_k))
        return results

    def delete(self, mau_id: str) -> bool:
        """
        Delete an embedding by MAU ID.

        Note: FAISS doesn't support efficient deletion. This marks for rebuild.
        """
        with self._lock:
            if mau_id in self._id_mapping:
                idx = self._id_mapping.index(mau_id)
                self._id_mapping[idx] = None  # Mark as deleted
                return True
        return False

    def rebuild_index(self, items: List[Tuple[str, List[float]]]) -> None:
        """
        Rebuild the entire index from scratch.

        Args:
            items: List of (mau_id, embedding) tuples
        """
        with self._lock:
            self._id_mapping = []

            if self._use_faiss:
                import faiss
                self._index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self._index = np.zeros((0, self.embedding_dim), dtype=np.float32)

            if items:
                self.add_batch(items)

    def save(self) -> None:
        """Save index to disk."""
        with self._lock:
            index_file = self.storage_path / "vectors.index"
            mapping_file = self.storage_path / "id_mapping.json"

            if self._use_faiss:
                import faiss
                faiss.write_index(self._index, str(index_file))
            else:
                np.save(str(index_file.with_suffix('.npy')), self._index)

            mapping_file.write_text(json.dumps(self._id_mapping))

        logger.info(f"Saved vector index with {self.count()} vectors")

    def count(self) -> int:
        """Get number of vectors in index."""
        with self._lock:
            if self._use_faiss:
                return self._index.ntotal
            return len(self._index)

    def get_embedding(self, mau_id: str) -> Optional[np.ndarray]:
        """Get embedding by MAU ID."""
        with self._lock:
            if mau_id not in self._id_mapping:
                return None
            idx = self._id_mapping.index(mau_id)

            if self._use_faiss:
                return self._index.reconstruct(idx)
            else:
                return self._index[idx]


class HybridVectorStore(VectorStore):
    """
    Hybrid Vector Store supporting both text and visual embeddings.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        text_dim: int = 1536,
        visual_dim: int = 512,
        **kwargs
    ):
        self.text_dim = text_dim
        self.visual_dim = visual_dim

        # Create separate stores for each modality
        base_path = Path(storage_path) if storage_path else Path("./omni_memory_data/vectors")

        # Create EmbeddingConfig with correct dimensions
        text_embedding_config = EmbeddingConfig(embedding_dim=text_dim)
        self._text_store = VectorStore(
            storage_path=str(base_path / "text"),
            config=text_embedding_config,
            **kwargs
        )
        self._visual_store = VectorStore(
            storage_path=str(base_path / "visual"),
            config=EmbeddingConfig(embedding_dim=visual_dim),
            **kwargs
        )

    def add(self, mau_id: str, embedding: List[float], modality: Optional[str] = None) -> None:
        """
        Add embedding, automatically detecting modality by dimension.
        
        Args:
            mau_id: The MAU ID
            embedding: The embedding vector
            modality: Optional modality type ('text' or 'visual'). 
                     If None, auto-detects by dimension.
        """
        if modality == "text" or (modality is None and len(embedding) == self.text_dim):
            self._text_store.add(mau_id, embedding)
        elif modality == "visual" or (modality is None and len(embedding) == self.visual_dim):
            self._visual_store.add(mau_id, embedding)
        else:
            raise ValueError(f"Unknown embedding dimension: {len(embedding)}. Expected {self.text_dim} (text) or {self.visual_dim} (visual)")

    def add_text(self, mau_id: str, embedding: List[float]) -> None:
        """Add text embedding."""
        self._text_store.add(mau_id, embedding)

    def add_visual(self, mau_id: str, embedding: List[float]) -> None:
        """Add visual embedding."""
        self._visual_store.add(mau_id, embedding)
    
    def size(self) -> int:
        """Get total number of vectors (text + visual)."""
        return self._text_store.size() + self._visual_store.size()
    
    def search_text(self, query: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search text embeddings."""
        return self._text_store.search(query, top_k)

    def search_visual(self, query: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search visual embeddings."""
        return self._visual_store.search(query, top_k)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Unified search interface that automatically selects text or visual search.
        
        For text queries (3072 dim for text-embedding-3-large), searches text embeddings.
        For visual queries (512 dim), searches visual embeddings.
        For hybrid queries, searches both and combines results.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (mau_id, similarity_score) tuples
        """
        # Auto-detect query type by dimension
        if len(query_embedding) == self.text_dim:
            # Text query - search text embeddings
            return self._text_store.search(query_embedding, top_k, threshold)
        elif len(query_embedding) == self.visual_dim:
            # Visual query - search visual embeddings
            return self._visual_store.search(query_embedding, top_k, threshold)
        else:
            # Try both and combine (for hybrid queries)
            text_results = {}
            visual_results = {}
            
            # Try text search if query dimension matches
            try:
                for mau_id, score in self._text_store.search(query_embedding, top_k * 2, threshold):
                    text_results[mau_id] = score
            except:
                pass
            
            # Try visual search if query dimension matches
            try:
                for mau_id, score in self._visual_store.search(query_embedding, top_k * 2, threshold):
                    visual_results[mau_id] = score
            except:
                pass
            
            # Combine results
            all_ids = set(text_results.keys()) | set(visual_results.keys())
            combined = []
            
            for mau_id in all_ids:
                text_score = text_results.get(mau_id, 0)
                visual_score = visual_results.get(mau_id, 0)
                # Use max score if both exist, otherwise use the available one
                combined_score = max(text_score, visual_score) if (text_score > 0 and visual_score > 0) else (text_score + visual_score)
                if combined_score >= threshold:
                    combined.append((mau_id, combined_score))
            
            combined.sort(key=lambda x: x[1], reverse=True)
            return combined[:top_k]

    def search_hybrid(
        self,
        text_query: Optional[List[float]] = None,
        visual_query: Optional[List[float]] = None,
        top_k: int = 10,
        text_weight: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search combining text and visual.

        Args:
            text_query: Text embedding query
            visual_query: Visual embedding query
            top_k: Number of results
            text_weight: Weight for text results (visual = 1 - text_weight)
        """
        text_results = {}
        visual_results = {}

        if text_query:
            for mau_id, score in self._text_store.search(text_query, top_k * 2):
                text_results[mau_id] = score

        if visual_query:
            for mau_id, score in self._visual_store.search(visual_query, top_k * 2):
                visual_results[mau_id] = score

        # Combine scores
        all_ids = set(text_results.keys()) | set(visual_results.keys())
        combined = []

        for mau_id in all_ids:
            text_score = text_results.get(mau_id, 0)
            visual_score = visual_results.get(mau_id, 0)
            combined_score = text_weight * text_score + (1 - text_weight) * visual_score
            combined.append((mau_id, combined_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def save(self) -> None:
        """Save both stores."""
        self._text_store.save()
        self._visual_store.save()
    
    def count(self) -> int:
        """Get total number of vectors in both stores."""
        return self._text_store.count() + self._visual_store.count()