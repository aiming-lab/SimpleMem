from __future__ import annotations

from typing import Union

from cross.storage_sqlite import SQLiteStorage

SqlStorage = Union[SQLiteStorage, object]


def create_sql_storage(
    use_iris: bool = False,
    use_pg: bool = False,
    db_path: str = "~/.simplemem-cross/cross_memory.db",
    iris_table_prefix: str = "CrossMem",
    pg_dsn: str = "",
    pg_table_prefix: str = "cross_mem",
) -> SqlStorage:
    """Return a SQL metadata storage backend.

    Priority: use_pg > use_iris > SQLite (default).
    """
    if use_pg:
        from cross.storage_pg_sql import PGSQLStorage
        return PGSQLStorage(dsn=pg_dsn or None, table_prefix=pg_table_prefix)
    if use_iris:
        from cross.storage_iris_sql import IRISSQLStorage
        return IRISSQLStorage(table_prefix=iris_table_prefix)
    return SQLiteStorage(db_path=db_path)
