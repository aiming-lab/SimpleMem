from __future__ import annotations

from typing import Union

from cross.storage_sqlite import SQLiteStorage
from cross.storage_iris_sql import IRISSQLStorage

SqlStorage = Union[SQLiteStorage, IRISSQLStorage]


def create_sql_storage(
    use_iris: bool = False,
    db_path: str = "~/.simplemem-cross/cross_memory.db",
    iris_table_prefix: str = "CrossMem",
) -> SqlStorage:
    if use_iris:
        return IRISSQLStorage(table_prefix=iris_table_prefix)
    return SQLiteStorage(db_path=db_path)
