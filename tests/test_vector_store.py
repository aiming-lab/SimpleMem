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
    return True


def test_keyword_search(store):
    print("\n[TEST] FTS keyword search...")
    results = store.keyword_search(["Starbucks"])
    assert len(results) > 0, "Keyword search should return results for 'Starbucks'"
    print(f"  PASS: Found {len(results)} results for 'Starbucks'")

    results = store.keyword_search(["documents"])
    assert len(results) > 0, "Keyword search should return results for 'documents'"
    print(f"  PASS: Found {len(results)} results for 'documents'")
    return True


def test_structured_search_persons(store):
    print("\n[TEST] Structured search by persons...")
    results = store.structured_search(persons=["Alice"])
    assert len(results) > 0, "Should find entries with Alice"
    print(f"  PASS: Found {len(results)} results for persons=['Alice']")

    results = store.structured_search(persons=["Bob"])
    assert len(results) > 0, "Should find entries with Bob"
    print(f"  PASS: Found {len(results)} results for persons=['Bob']")
    return True


def test_structured_search_location(store):
    print("\n[TEST] Structured search by location...")
    results = store.structured_search(location="Starbucks")
    assert len(results) > 0, "Should find entries at Starbucks"
    print(f"  PASS: Found {len(results)} results for location='Starbucks'")
    return True


def test_structured_search_timestamp(store):
    print("\n[TEST] Structured search by timestamp range...")
    results = store.structured_search(
        timestamp_range=("2025-01-15T00:00:00", "2025-01-15T23:59:59")
    )
    assert len(results) > 0, "Should find entries in timestamp range"
    print(f"  PASS: Found {len(results)} results in timestamp range")
    return True


def test_optimize(store):
    print("\n[TEST] Table optimize...")
    store.optimize()
    print("  PASS: Optimize completed")
    return True


def test_get_all_entries(store):
    print("\n[TEST] Get all entries...")
    results = store.get_all_entries()
    assert len(results) == 3, f"Should have 3 entries, got {len(results)}"
    print(f"  PASS: Retrieved {len(results)} entries")
    return True


def test_gcs_connection(bucket_path, service_account_path=None):
    """
    Test GCS backend with native FTS.

    Usage:
        python tests/test_vector_store.py --gcs gs://your-bucket/lancedb --sa /path/to/service-account.json
    """
    print("\n" + "=" * 60)
    print("GCS Connection Test (Native FTS)")
    print("=" * 60)

    storage_options = None
    if service_account_path:
        storage_options = {"service_account": service_account_path}

    print(f"\nConnecting to {bucket_path}...")
    store = VectorStore(
        db_path=bucket_path,
        table_name="gcs_test_entries",
        storage_options=storage_options
    )
    store.clear()

    print("\nAdding test entries...")
    entries = create_test_entries()
    store.add_entries(entries)
    print(f"  Added {len(entries)} entries")

    passed = 0
    failed = 0

    # Test semantic search
    print("\n[TEST] Semantic search on GCS...")
    try:
        results = store.semantic_search("meeting location")
        assert len(results) > 0, "Should find results"
        print(f"  PASS: Found {len(results)} results")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # Test FTS keyword search (native mode)
    print("\n[TEST] FTS keyword search on GCS (native mode)...")
    try:
        results = store.keyword_search(["Starbucks"])
        assert len(results) > 0, "Should find Starbucks"
        print(f"  PASS: Found {len(results)} results for 'Starbucks'")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # Test structured search
    print("\n[TEST] Structured search on GCS...")
    try:
        results = store.structured_search(persons=["Alice"])
        assert len(results) > 0, "Should find Alice"
        print(f"  PASS: Found {len(results)} results for persons=['Alice']")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print("\nCleaning up...")
    store.clear()

    print("\n" + "=" * 60)
    print(f"GCS Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs", help="GCS bucket path (gs://bucket/path)")
    parser.add_argument("--sa", help="Service account JSON path")
    args = parser.parse_args()

    if args.gcs:
        return test_gcs_connection(args.gcs, args.sa)

    print("=" * 60)
    print("VectorStore Optimization Tests (Local)")
    print("=" * 60)

    test_db_path = "./tests/test_lancedb"

    print(f"\nInitializing VectorStore at {test_db_path}...")
    store = VectorStore(db_path=test_db_path, table_name="test_entries")
    store.clear()

    print("\nAdding test entries...")
    entries = create_test_entries()
    store.add_entries(entries)
    print(f"Added {len(entries)} entries")

    tests = [
        test_semantic_search,
        test_keyword_search,
        test_structured_search_persons,
        test_structured_search_location,
        test_structured_search_timestamp,
        test_optimize,
        test_get_all_entries,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test(store):
                passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
