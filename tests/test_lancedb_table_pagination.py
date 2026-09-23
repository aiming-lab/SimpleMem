"""Regression tests for LanceDB table existence check with >10 tables.

When a LanceDB database contains more than 10 tables, ``table_names()``
returns only the first page by default.  The table-existence check must
use ``Connection.__contains__`` (which paginates fully) instead of
membership against ``table_names()``.

See https://github.com/aiming-lab/SimpleMem/issues/85
"""
from unittest.mock import patch

import pytest

from simplemem.core.database.vector_store_backend import LanceDBVectorStoreBackend


class _MockTable:
    def __init__(self, name):
        self.name = name


class _MockDB:
    """Simulates a LanceDB connection whose ``table_names()`` is capped at 10.

    ``__contains__`` sees every table (full pagination), so it can tell
    apart a table that exists but is beyond the first page from one that
    is genuinely new.
    """

    def __init__(self, all_tables):
        self._all_tables = list(all_tables)
        self.create_table_called = False
        self.open_table_called = False
        self.opened_name = None
        self.created_name = None

    def table_names(self):
        # Mirrors the deprecated default limit of 10 in recent LanceDB.
        return self._all_tables[:10]

    def __contains__(self, name):
        return name in self._all_tables

    def create_table(self, name, schema=None):
        self.create_table_called = True
        self.created_name = name
        return _MockTable(name)

    def open_table(self, name):
        self.open_table_called = True
        self.opened_name = name
        return _MockTable(name)


@pytest.fixture
def many_tables():
    return [f"table_{i}" for i in range(15)]


def _make_backend(db, tmp_path, table_name):
    with patch(
        "simplemem.core.database.vector_store_backend.lancedb.connect",
        return_value=db,
    ):
        return LanceDBVectorStoreBackend(
            db_path=str(tmp_path / "test-db"),
            table_name=table_name,
            vector_dimension=3,
        )


def test_existing_table_beyond_first_page_is_opened(many_tables, tmp_path):
    """An existing table not in table_names() first page must be opened, not created."""
    db = _MockDB(many_tables)
    _make_backend(db, tmp_path, "table_12")

    assert db.open_table_called, (
        "Expected open_table() for an existing table beyond the first page"
    )
    assert not db.create_table_called, (
        "create_table() should not be called for an existing table"
    )
    assert db.opened_name == "table_12"


def test_new_table_is_created(many_tables, tmp_path):
    """A table that truly does not exist should be created."""
    db = _MockDB(many_tables)
    _make_backend(db, tmp_path, "brand_new_table")

    assert db.create_table_called, (
        "Expected create_table() for a table that does not exist"
    )
    assert not db.open_table_called, (
        "open_table() should not be called for a new table"
    )
    assert db.created_name == "brand_new_table"


def test_table_within_first_page_is_opened(many_tables, tmp_path):
    """An existing table within the first 10 results is also opened correctly."""
    db = _MockDB(many_tables)
    _make_backend(db, tmp_path, "table_3")

    assert db.open_table_called
    assert not db.create_table_called
    assert db.opened_name == "table_3"
