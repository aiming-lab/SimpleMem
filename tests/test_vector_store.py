"""
Tests for VectorStore optimizations.
Tests FTS, SQL filters, and semantic search.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplemem.core.database.vector_store import VectorStore
from simplemem.core.models.memory_entry import MemoryEntry


def create_test_entries():
    return [
        MemoryEntry(
            lossless_restatement="Alice suggested meeting at Starbucks on 2025-01-15 at 2pm",
            keywords=["Alice", "Starbucks", "meeting"],
            timestamp="2025-01-15T14:00:00",
            location="Starbucks",
            persons=["Alice", "Bob"],
            entities=["meeting"],
            topic="Meeting arrangement"
        ),
        MemoryEntry(
            lossless_restatement="Bob will bring the project documents to the meeting",
            keywords=["Bob", "documents", "project"],
            timestamp="2025-01-15T14:01:00",
            location=None,
            persons=["Bob"],
            entities=["documents", "project"],
            topic="Meeting preparation"
        ),
        MemoryEntry(
            lossless_restatement="Charlie confirmed attendance for the Starbucks meeting",
            keywords=["Charlie", "Starbucks", "attendance"],
            timestamp="2025-01-15T14:02:00",
            location="Starbucks",
            persons=["Charlie"],
            entities=["meeting"],
            topic="Meeting confirmation"
        )
    ]


def test_semantic_search(store):
    print("\n[TEST] Semantic search...")
    results = store.semantic_search("meeting location", top_k=5)
    assert len(results) > 0, "Semantic search should return results"
    print(f"  PASS: Found {len(results)} results")


def test_keyword_search(store):
    print("\n[TEST] FTS keyword search...")
    results = store.keyword_search(["Starbucks"])
    assert len(results) > 0, "Keyword search should return results for 'Starbucks'"
    print(f"  PASS: Found {len(results)} results for 'Starbucks'")

    results = store.keyword_search(["documents"])
    assert len(results) > 0, "Keyword search should return results for 'documents'"
    print(f"  PASS: Found {len(results)} results for 'documents'")


def test_structured_search_persons(store):
    print("\n[TEST] Structured search by persons...")
    results = store.structured_search(persons=["Alice"])
    assert len(results) > 0, "Should find entries with Alice"
    print(f"  PASS: Found {len(results)} results for persons=['Alice']")

    results = store.structured_search(persons=["Bob"])
    assert len(results) > 0, "Should find entries with Bob"
    print(f"  PASS: Found {len(results)} results for persons=['Bob']")


def test_structured_search_location(store):
    print("\n[TEST] Structured search by location...")
    results = store.structured_search(location="Starbucks")
    assert len(results) > 0, "Should find entries at Starbucks"
    print(f"  PASS: Found {len(results)} results for location='Starbucks'")


def test_structured_search_timestamp(store):
    print("\n[TEST] Structured search by timestamp range...")
    results = store.structured_search(
        timestamp_range=("2025-01-15T00:00:00", "2025-01-15T23:59:59")
    )
    assert len(results) > 0, "Should find entries in timestamp range"
    print(f"  PASS: Found {len(results)} results in timestamp range")


def test_optimize(store):
    store.optimize()


def test_get_all_entries(store):
    results = store.get_all_entries()
    assert len(results) == 3, f"Should have 3 entries, got {len(results)}"
