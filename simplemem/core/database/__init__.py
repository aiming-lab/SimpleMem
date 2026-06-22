"""
Storage backend factory.

Default: LanceDB (upstream default, zero extra deps).
Opt-in:  pgvector — set STORAGE_BACKEND=pgvector and PG_DSN in config.py or env.

IRIS + pgwire users: set STORAGE_BACKEND=pgvector and point PG_DSN at the
pgwire endpoint. No IRIS-specific code required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from simplemem.core.settings import settings

if TYPE_CHECKING:
    from simplemem.core.database.vector_store import VectorStore as _LanceStore
    from simplemem.core.database.pg_vector_store import PGVectorStore


def get_vector_store(**kwargs):
    """Return a VectorStore instance for the configured backend.

    Keyword arguments are forwarded to the store constructor, allowing
    callers to override dsn/db_path, table_name, embedding_model, etc.
    """
    backend = (kwargs.pop("backend", None) or settings.STORAGE_BACKEND).lower()
    if backend == "pgvector":
        from simplemem.core.database.pg_vector_store import PGVectorStore
        return PGVectorStore(**kwargs)
    # Default: LanceDB
    from simplemem.core.database.vector_store import VectorStore
    return VectorStore(**kwargs)


# Preserve direct import compatibility
from simplemem.core.database.vector_store import VectorStore  # noqa: E402

__all__ = ["VectorStore", "get_vector_store"]
