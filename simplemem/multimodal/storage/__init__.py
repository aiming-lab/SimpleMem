"""
Storage management for Omni-Memory.

Implements cold storage for heavy multimodal data and hot storage for
lightweight metadata and embeddings.
"""

from simplemem.multimodal.storage.cold_storage import ColdStorageManager
from simplemem.multimodal.storage.mau_store import MAUStore
from simplemem.multimodal.storage.vector_store import VectorStore

__all__ = [
    "ColdStorageManager",
    "MAUStore",
    "VectorStore",
]
