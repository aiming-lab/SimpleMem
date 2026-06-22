"""
BM25 Keyword Search Store for Omni-Memory.

Provides lexical/keyword-based retrieval alongside dense vector search.
Uses rank_bm25 for tokenization and scoring.
"""

import logging
import re
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency
_BM25 = None


def _get_bm25_class():
    global _BM25
    if _BM25 is None:
        try:
            from rank_bm25 import BM25Okapi

            _BM25 = BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 not installed. BM25 search disabled. Install with: pip install rank_bm25"
            )
            return None
    return _BM25


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    # Split on non-alphanumeric, keep tokens of length >= 2
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2]


class BM25Store:
    """
    BM25 keyword search index over MAU summaries.

    Builds an in-memory BM25 index from MAU summaries for lexical retrieval.
    Designed to complement dense vector search for hybrid retrieval.
    """

    def __init__(self):
        self._index = None  # BM25Okapi instance
        self._mau_ids: List[str] = []  # Parallel array: index position -> mau_id
        self._corpus: List[List[str]] = []  # Tokenized documents
        self._built = False

    def build_from_mau_store(self, mau_store) -> int:
        """
        Build BM25 index from all MAUs in the store.

        Args:
            mau_store: MAUStore instance with iter_all() method

        Returns:
            Number of documents indexed
        """
        BM25Class = _get_bm25_class()
        if BM25Class is None:
            logger.warning("BM25 unavailable, skipping index build")
            return 0

        self._mau_ids = []
        self._corpus = []

        for mau in mau_store.iter_all():
            # Skip archived MAUs
            if getattr(mau, "status", "ACTIVE") == "ARCHIVED":
                continue

            text = mau.summary or ""
            # Also include details text if available
            if mau.details and isinstance(mau.details, dict):
                for key in ("transcript", "text", "full_text"):
                    if key in mau.details and mau.details[key]:
                        text += " " + str(mau.details[key])[:500]
                        break
            # Include enriched metadata fields for better keyword matching
            if hasattr(mau, 'metadata') and mau.metadata:
                meta = mau.metadata
                for attr in ('persons', 'entities', 'keywords'):
                    vals = getattr(meta, attr, None)
                    if vals and isinstance(vals, list):
                        text += ' ' + ' '.join(str(v) for v in vals)
                for attr in ('location', 'topic'):
                    val = getattr(meta, attr, None)
                    if val:
                        text += ' ' + str(val)

            tokens = _tokenize(text)
            if tokens:
                self._mau_ids.append(mau.id)
                self._corpus.append(tokens)

        if not self._corpus:
            logger.warning("BM25: no documents to index")
            return 0

        self._index = BM25Class(self._corpus)
        self._built = True
        logger.info("BM25 index built with %d documents", len(self._mau_ids))
        return len(self._mau_ids)

    def search(
        self,
        query: str,
        top_k: int = 10,
        tags_filter: Optional[List[str]] = None,
        mau_store=None,
    ) -> List[Tuple[str, float]]:
        """
        Search BM25 index for relevant MAUs.

        Args:
            query: Search query string
            top_k: Number of results to return
            tags_filter: Optional tag filter (requires mau_store for lookup)
            mau_store: MAUStore for tag filtering

        Returns:
            List of (mau_id, bm25_score) tuples, sorted by score descending
        """
        if not self._built or self._index is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)

        # Get top_k * 2 candidates (to allow for filtering)
        import numpy as np

        n_candidates = min(top_k * 2, len(scores))
        top_indices = np.argsort(scores)[::-1][:n_candidates]

        results = []
        for idx in top_indices:
            idx = int(idx)
            score = float(scores[idx])
            if score <= 0:
                break  # BM25 scores of 0 mean no match

            mau_id = self._mau_ids[idx]

            # Apply tag filter if provided
            if tags_filter and mau_store:
                mau = mau_store.get(mau_id)
                if mau and not any(
                    tag in (mau.metadata.tags or []) for tag in tags_filter
                ):
                    continue

            results.append((mau_id, score))
            if len(results) >= top_k:
                break

        return results

    @property
    def is_available(self) -> bool:
        """Check if BM25 index is built and ready."""
        return self._built and self._index is not None

    @property
    def doc_count(self) -> int:
        """Number of documents in the index."""
        return len(self._mau_ids) if self._built else 0
