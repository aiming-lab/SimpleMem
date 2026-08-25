"""Tests for PGVectorStoreBackend.

Requires a PostgreSQL instance with the pgvector extension.
Set PG_TEST_DSN to run:

  PG_TEST_DSN=postgresql://postgres:postgres@localhost:5432/simplemem_test \\
      pytest tests/test_pg_vector_store_backend.py -v

IRIS + pgwire: point PG_TEST_DSN at the pgwire endpoint.
All tests are skipped automatically when PG_TEST_DSN is unset or unreachable.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

PG_TEST_DSN = os.environ.get("PG_TEST_DSN", "")
TABLE = "pytest_pg_backend_entries"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    dimension = 4

    def encode_documents(self, texts):
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
        psycopg.connect(PG_TEST_DSN).close()
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="No PostgreSQL — set PG_TEST_DSN to run",
)


def _records():
    from simplemem.core.database.vector_store_backend import VectorStoreRecord
    return [
        VectorStoreRecord(
            entry_id="e1",
            vector=[0.9, 0.1, 0.1, 0.1],
            metadata={
                "lossless_restatement": "Alice will meet Bob at Starbucks on 2025-01-15",
                "keywords": ["Alice", "Bob", "Starbucks", "meeting"],
                "timestamp": "2025-01-15T14:00:00",
                "location": "Starbucks",
                "persons": ["Alice", "Bob"],
                "entities": ["meeting"],
                "topic": "Meeting arrangement",
            },
        ),
        VectorStoreRecord(
            entry_id="e2",
            vector=[0.1, 0.9, 0.1, 0.1],
            metadata={
                "lossless_restatement": "Bob will bring the project documents to the meeting",
                "keywords": ["Bob", "documents", "project"],
                "timestamp": "2025-01-15T14:01:00",
                "location": None,
                "persons": ["Bob"],
                "entities": ["documents", "project"],
                "topic": "Meeting preparation",
            },
        ),
        VectorStoreRecord(
            entry_id="e3",
            vector=[0.1, 0.1, 0.9, 0.1],
            metadata={
                "lossless_restatement": "Charlie confirmed attendance for the Starbucks meeting",
                "keywords": ["Charlie", "Starbucks", "attendance"],
                "timestamp": "2025-01-15T14:02:00",
                "location": "Starbucks",
                "persons": ["Charlie"],
                "entities": ["meeting"],
                "topic": "Meeting confirmation",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend():
    from simplemem.core.database.pg_vector_store_backend import PGVectorStoreBackend
    b = PGVectorStoreBackend(dsn=PG_TEST_DSN, table_name=TABLE, vector_dimension=4)
    b.clear()
    b.insert(_records())
    yield b
    b.clear()
    b.close()


@pytest.fixture(scope="module")
def store(backend):
    """VectorStore wired to the pg backend via get_vector_store()."""
    from simplemem.core.database import get_vector_store
    from simplemem.core.models.memory_entry import MemoryEntry

    s = get_vector_store(
        backend="pgvector",
        dsn=PG_TEST_DSN,
        embedding_model=_FakeEmbedder(),
        table_name=TABLE + "_vs",
    )
    s.clear()
    entries = [
        MemoryEntry(
            lossless_restatement=r.metadata["lossless_restatement"],
            keywords=r.metadata.get("keywords") or [],
            timestamp=r.metadata.get("timestamp"),
            location=r.metadata.get("location"),
            persons=r.metadata.get("persons") or [],
            entities=r.metadata.get("entities") or [],
            topic=r.metadata.get("topic"),
        )
        for r in _records()
    ]
    s.add_entries(entries)
    yield s
    s.clear()


# ---------------------------------------------------------------------------
# Backend-direct tests
# ---------------------------------------------------------------------------

@requires_pg
def test_count(backend):
    assert backend.count() == 3


@requires_pg
def test_semantic_search_returns_results(backend):
    results = backend.semantic_search([0.9, 0.1, 0.1, 0.1], top_k=3)
    assert len(results) > 0
    assert results[0].entry_id == "e1"
    assert results[0].score is not None


@requires_pg
def test_semantic_search_score_order(backend):
    results = backend.semantic_search([0.9, 0.1, 0.1, 0.1], top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores)  # ASCENDING — smaller distance is better


@requires_pg
def test_keyword_search_starbucks(backend):
    results = backend.keyword_search(["Starbucks"], top_k=5)
    assert len(results) > 0
    texts = [r.metadata["lossless_restatement"] for r in results]
    assert any("Starbucks" in t for t in texts)


@requires_pg
def test_keyword_search_documents(backend):
    results = backend.keyword_search(["documents", "project"], top_k=5)
    assert len(results) > 0


@requires_pg
def test_keyword_empty_returns_empty(backend):
    assert backend.keyword_search([], top_k=5) == []


@requires_pg
def test_structured_search_persons_alice(backend):
    results = backend.structured_search(persons=["Alice"])
    assert len(results) > 0
    for r in results:
        assert "Alice" in r.metadata["persons"]


@requires_pg
def test_structured_search_persons_bob(backend):
    results = backend.structured_search(persons=["Bob"])
    assert len(results) > 0


@requires_pg
def test_structured_search_location(backend):
    results = backend.structured_search(location="Starbucks")
    assert len(results) > 0
    for r in results:
        assert r.metadata["location"] and "Starbucks" in r.metadata["location"]


@requires_pg
def test_structured_search_timestamp_range(backend):
    results = backend.structured_search(
        timestamp_range=("2025-01-15T00:00:00", "2025-01-15T23:59:59")
    )
    assert len(results) > 0


@requires_pg
def test_structured_search_no_filters_returns_empty(backend):
    assert backend.structured_search() == []


@requires_pg
def test_get_all(backend):
    results = backend.get_all()
    assert len(results) == 3
    ids = {r.entry_id for r in results}
    assert ids == {"e1", "e2", "e3"}


@requires_pg
def test_optimize_does_not_raise(backend):
    backend.optimize()


@requires_pg
def test_no_sql_injection_persons(backend):
    results = backend.structured_search(persons=["Alice')) OR 1=1--"])
    assert isinstance(results, list)


@requires_pg
def test_clear_and_reinsert(backend):
    backend.clear()
    assert backend.count() == 0
    backend.insert(_records())
    assert backend.count() == 3


# ---------------------------------------------------------------------------
# VectorStore integration (via get_vector_store)
# ---------------------------------------------------------------------------

@requires_pg
def test_vectorstore_semantic_search(store):
    results = store.semantic_search("meeting coffee shop Starbucks", top_k=3)
    assert len(results) > 0
    assert hasattr(results[0], "lossless_restatement")


@requires_pg
def test_vectorstore_keyword_search(store):
    results = store.keyword_search(["Starbucks"], top_k=5)
    assert any("Starbucks" in r.lossless_restatement for r in results)


@requires_pg
def test_vectorstore_structured_search(store):
    results = store.structured_search(persons=["Alice"])
    assert len(results) > 0
    assert all("Alice" in r.persons for r in results)


@requires_pg
def test_vectorstore_get_all(store):
    assert len(store.get_all_entries()) == 3
