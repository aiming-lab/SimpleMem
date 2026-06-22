from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for memory embedders."""

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Encode a single text into a vector."""

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts. Default implementation calls encode() in a loop."""
        return [self.encode(t) for t in texts]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of produced vectors."""


class HashingEmbedder(BaseEmbedder):
    """Deterministic lightweight embedder for phase-1 optional semantic retrieval."""

    def __init__(self, dimensions: int = 64):
        self._dimensions = max(int(dimensions), 8)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _tokenize(text):
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self._dimensions
            vector[index] += 1.0
        return _l2_normalize(vector)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Semantic embedder using sentence-transformers models.

    Requires the ``sentence-transformers`` package (``pip install sentence-transformers``).
    Falls back to HashingEmbedder if the package is not installed.

    Default model: ``all-MiniLM-L6-v2`` (384-dimensional, fast, good general quality).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimensions_cache: int | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            # Cache dimensions from a probe encode
            probe = self._model.encode("probe", convert_to_numpy=True)
            self._dimensions_cache = len(probe)
            logger.info(
                "SentenceTransformerEmbedder loaded model=%s dims=%d",
                self.model_name,
                self._dimensions_cache,
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; "
                "SentenceTransformerEmbedder will raise on encode(). "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
            self._dimensions_cache = 384  # default for all-MiniLM-L6-v2
        except Exception as exc:
            logger.warning(
                "Failed to load sentence-transformers model %s: %s",
                self.model_name,
                exc,
            )
            self._model = None
            self._dimensions_cache = 384

    @property
    def dimensions(self) -> int:
        return self._dimensions_cache or 384

    @property
    def is_available(self) -> bool:
        """Whether the underlying model loaded successfully."""
        return self._model is not None

    def encode(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError(
                f"SentenceTransformerEmbedder model '{self.model_name}' not available. "
                "Install sentence-transformers: pip install sentence-transformers"
            )
        embedding = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError(
                f"SentenceTransformerEmbedder model '{self.model_name}' not available."
            )
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32,
        )
        return [e.tolist() for e in embeddings]


def create_embedder(
    mode: str = "hashing",
    model_name: str = "all-MiniLM-L6-v2",
    dimensions: int = 64,
    fallback: bool = True,
) -> BaseEmbedder:
    """Factory function to create an embedder.

    Args:
        mode: "hashing" for HashingEmbedder, "semantic" for SentenceTransformerEmbedder.
        model_name: Model name for semantic mode.
        dimensions: Vector dimensions for hashing mode.
        fallback: If True and semantic mode fails to load, fall back to HashingEmbedder.

    Returns:
        An embedder instance.
    """
    if mode == "semantic":
        embedder = SentenceTransformerEmbedder(model_name=model_name)
        if embedder.is_available:
            return embedder
        if fallback:
            logger.warning(
                "Semantic embedder not available, falling back to HashingEmbedder"
            )
            return HashingEmbedder(dimensions=dimensions)
        raise RuntimeError(
            "Semantic embedder not available and fallback disabled. "
            "Install sentence-transformers: pip install sentence-transformers"
        )
    return HashingEmbedder(dimensions=dimensions)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return vector
    return [v / norm for v in vector]


def _tokenize(text: str) -> list[str]:
    token = []
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
            continue
        if token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out
