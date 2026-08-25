from simplemem.core.database.vector_store import VectorStore
from simplemem.core.database.vector_store_backend import (
    LanceDBVectorStoreBackend,
    ScoreOrder,
    VectorStoreBackend,
    VectorStoreRecord,
    VectorStoreSearchResult,
)

__all__ = [
    "LanceDBVectorStoreBackend",
    "PGVectorStoreBackend",
    "ScoreOrder",
    "VectorStore",
    "VectorStoreBackend",
    "VectorStoreRecord",
    "VectorStoreSearchResult",
    "get_vector_store",
]


def get_vector_store(**kwargs):
    """Return a VectorStore configured for the active STORAGE_BACKEND.

    When STORAGE_BACKEND=pgvector (or backend='pgvector' is passed),
    VectorStore is initialised with a PGVectorStoreBackend factory so
    every call goes through PostgreSQL / pgvector instead of LanceDB.

    All other keyword arguments (embedding_model, table_name, db_path, …)
    are forwarded to the VectorStore constructor.
    """
    from simplemem.core.settings import settings

    backend = (kwargs.pop("backend", None) or settings.STORAGE_BACKEND).lower()

    if backend == "pgvector":
        from simplemem.core.database.pg_vector_store_backend import PGVectorStoreBackend

        dsn = kwargs.pop("dsn", None) or settings.PG_DSN

        def _pg_factory(vector_dimension: int):
            table_name = kwargs.get("table_name") or settings.MEMORY_TABLE_NAME
            return PGVectorStoreBackend(
                dsn=dsn,
                table_name=table_name,
                vector_dimension=vector_dimension,
            )

        kwargs["backend_factory"] = _pg_factory

    return VectorStore(**kwargs)


# Lazy export so `from simplemem.core.database import PGVectorStoreBackend` works
# without making psycopg a hard dependency.
def __getattr__(name: str):
    if name == "PGVectorStoreBackend":
        from simplemem.core.database.pg_vector_store_backend import PGVectorStoreBackend
        return PGVectorStoreBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
