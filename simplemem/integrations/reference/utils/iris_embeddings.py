from __future__ import annotations
from typing import List
from utils.embedding import EmbeddingModel


class IRISEmbeddingsAdapter:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self._model = embedding_model or EmbeddingModel()
        self.dimensions: int = self._model.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.encode_documents(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode_single(text, is_query=True).tolist()
