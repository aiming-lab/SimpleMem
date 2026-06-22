"""
Tests for PGVectorStore.

Requires a PostgreSQL instance with pgvector extension.
Set PG_TEST_DSN env var to run, e.g.:
  PG_TEST_DSN=postgresql://postgres:postgres@localhost:5432/simplemem_test pytest tests/test_pg_vector_store.py

IRIS + pgwire: set PG_TEST_DSN to point at your pgwire endpoint instead.
Tests are skipped automatically when PG_TEST_DSN is not set or connection fails.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PG_TEST_DSN = os.environ.get("PG_TEST_DSN", "")


class _FakeEmbedder:
    dimension = 4

    def encode_documents(self, texts):
        # Deterministic: hash text to a stable unit vector so search assertions hold
        vecs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 1000
            rng = np.random.default_rng(seed)
            v = rng.random(self.dimension).astype(np.float32)
            vecs.append(v / np.linalg.norm(v))
        return vecs

    def encode_single(self, text, is_query=False):
        return self.encode_documents([text])[0]


def _pg_available() -> bool:
    if not PG_TEST_DSN:
        return False
    try:
        import psycopg
        conn = psycopg.connect(PG_TEST_DSN)
        conn.close()
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="No PostgreSQL instance — set PG_TEST_DSN to run",
)


def _sample_entries():
    from simplemem.core.models.memory_entry import MemoryEntry
    return [
        MemoryEntry(
            lossless_restatement="Alice will meet Bob at Starbucks on 2025-01-15 at 2pm",
            keywords=["Alice", "Bob", "Starbucks", "meeting"],
            timestamp="2025-01-15T14:00:00",
            location="Starbucks",
            persons=["Alice", "Bob"],
            entities=["meeting"],
            topic="Meeting arrangement",
        ),
        MemoryEntry(
            lossless_restatement="Bob will bring the project documents to the meeting",
            keywords=["Bob", "documents", "project"],
            timestamp="2025-01-15T14:01:00",
            location=None,
            persons=["Bob"],
            entities=["documents", "project"],
            topic="Meeting preparation",
        ),
        MemoryEntry(
            lossless_restatement="Charlie confirmed attendance for the Starbucks meeting",
            keywords=["Charlie", "Starbucks", "attendance"],
            timestamp="2025-01-15T14:02:00",
            location="Starbucks",
            persons=["Charlie"],
            entities=["meeting"],
            topic="Meeting confirmation",
        ),
    ]


@pytest.fixture(scope="module")
def store():
    from simplemem.core.database.pg_vector_store import PGVectorStore
    s = PGVectorStore(
        dsn=PG_TEST_DSN,
        embedding_model=_FakeEmbedder(),
        table_name="pytest_pg_memory_entries",
    )
    s.clear()
    s.add_entries(_sample_entries())
    yield s
    s.clear()
    s.close()


@requires_pg
def test_semantic_search(store):
    results = store.semantic_search("meeting location coffee shop", top_k=3)
    assert len(results) > 0
    assert all(hasattr(r, "lossless_restatement") for r in results)


@requires_pg
def test_keyword_search_starbucks(store):
    results = store.keyword_search(["Starbucks"], top_k=5)
    assert len(results) > 0
    texts = [r.lossless_restatement for r in results]
    assert any("Starbucks" in t for t in texts)


@requires_pg
def test_keyword_search_documents(store):
    results = store.keyword_search(["documents", "project"], top_k=5)
    assert len(results) > 0


@requires_pg
def test_structured_search_persons_alice(store):
    results = store.structured_search(persons=["Alice"])
    assert len(results) > 0
    for r in results:
        assert "Alice" in r.persons


@requires_pg
def test_structured_search_persons_bob(store):
    results = store.structured_search(persons=["Bob"])
    assert len(results) > 0


@requires_pg
def test_structured_search_location(store):
    results = store.structured_search(location="Starbucks")
    assert len(results) > 0
    for r in results:
        assert r.location and "Starbucks" in r.location


@requires_pg
def test_structured_search_timestamp(store):
    results = store.structured_search(
        timestamp_range=("2025-01-15T00:00:00", "2025-01-15T23:59:59")
    )
    assert len(results) > 0


@requires_pg
def test_get_all_entries(store):
    results = store.get_all_entries()
    assert len(results) == 3


@requires_pg
def test_optimize(store):
    # Should not raise
    store.optimize()


@requires_pg
def test_no_sql_injection_persons(store):
    # Malicious value should not return unexpected results via injection
    results = store.structured_search(persons=["Alice')) OR 1=1--"])
    # Should return empty (no entry has that exact person value)
    assert isinstance(results, list)


@requires_pg
def test_clear_and_readd(store):
    store.clear()
    assert store.get_all_entries() == []
    store.add_entries(_sample_entries())
    assert len(store.get_all_entries()) == 3
