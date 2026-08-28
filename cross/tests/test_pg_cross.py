"""
Tests for PostgreSQL cross-session backends:
  - PGCrossVectorStore  (cross/storage_pg.py)
  - PGSQLStorage        (cross/storage_pg_sql.py)

Set PG_TEST_DSN to run, e.g.:
  PG_TEST_DSN=postgresql://postgres:postgres@localhost:5432/simplemem_test \
    pytest cross/tests/test_pg_cross.py -v

Skipped automatically when PG_TEST_DSN is absent or the DB is unreachable.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PG_TEST_DSN = os.environ.get("PG_TEST_DSN", "")


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
        conn = psycopg.connect(PG_TEST_DSN)
        conn.close()
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="No PostgreSQL instance — set PG_TEST_DSN to run",
)


# =========================================================================
# PGCrossVectorStore tests
# =========================================================================

@pytest.fixture(scope="module")
def vec_store():
    from cross.storage_pg import PGCrossVectorStore
    from simplemem.core.models.memory_entry import MemoryEntry

    store = PGCrossVectorStore(
        dsn=PG_TEST_DSN,
        embedding_model=_FakeEmbedder(),
        table_name="pytest_cross_memory_entries",
    )
    store.clear()

    entries = [
        MemoryEntry(
            lossless_restatement="Alice completed the authentication module refactor",
            keywords=["Alice", "auth", "refactor"],
            timestamp="2025-03-01T10:00:00",
            location="office",
            persons=["Alice"],
            entities=["auth", "refactor"],
            topic="Engineering",
        ),
        MemoryEntry(
            lossless_restatement="Bob fixed the database connection pool leak",
            keywords=["Bob", "database", "fix"],
            timestamp="2025-03-02T11:00:00",
            location=None,
            persons=["Bob"],
            entities=["database", "connection pool"],
            topic="Engineering",
        ),
        MemoryEntry(
            lossless_restatement="Sprint planning scheduled for Monday at 9am",
            keywords=["sprint", "planning", "meeting"],
            timestamp="2025-03-03T09:00:00",
            location="conference room",
            persons=[],
            entities=["sprint", "planning"],
            topic="Process",
        ),
    ]
    store.add_entries(
        entries,
        tenant_id="test_tenant",
        memory_session_id="sess_001",
        source_kind="session",
        source_id=1,
    )
    yield store
    store.clear()
    store.close()


@requires_pg
def test_vec_semantic_search(vec_store):
    results = vec_store.semantic_search("code refactoring work", top_k=3)
    assert len(results) > 0
    assert all(hasattr(r, "lossless_restatement") for r in results)


@requires_pg
def test_vec_semantic_search_with_tenant(vec_store):
    results = vec_store.semantic_search("meeting", top_k=5, tenant_id="test_tenant")
    assert isinstance(results, list)


@requires_pg
def test_vec_keyword_search(vec_store):
    results = vec_store.keyword_search(["database"], top_k=5)
    assert len(results) > 0
    assert any("database" in r.lossless_restatement.lower() for r in results)


@requires_pg
def test_vec_keyword_search_with_tenant(vec_store):
    results = vec_store.keyword_search(["auth"], top_k=5, tenant_id="test_tenant")
    assert isinstance(results, list)


@requires_pg
def test_vec_structured_search_persons(vec_store):
    results = vec_store.structured_search(persons=["Alice"])
    assert len(results) > 0
    for r in results:
        assert "Alice" in r.persons


@requires_pg
def test_vec_structured_search_location(vec_store):
    results = vec_store.structured_search(location="office")
    assert len(results) > 0


@requires_pg
def test_vec_structured_search_timestamp(vec_store):
    results = vec_store.structured_search(
        timestamp_range=("2025-03-01T00:00:00", "2025-03-02T23:59:59")
    )
    assert len(results) > 0


@requires_pg
def test_vec_get_entries_for_session(vec_store):
    results = vec_store.get_entries_for_session("sess_001")
    assert len(results) == 3


@requires_pg
def test_vec_get_all_entries_with_tenant(vec_store):
    results = vec_store.get_all_entries(tenant_id="test_tenant")
    assert len(results) == 3


@requires_pg
def test_vec_count_entries(vec_store):
    count = vec_store.count_entries(tenant_id="test_tenant")
    assert count == 3


@requires_pg
def test_vec_count_by_session(vec_store):
    count = vec_store.count_entries(memory_session_id="sess_001")
    assert count == 3


@requires_pg
def test_vec_mark_superseded(vec_store):
    from simplemem.core.models.memory_entry import MemoryEntry
    entry = MemoryEntry(
        lossless_restatement="Temporary entry to be superseded",
        keywords=[],
        timestamp=None,
        location=None,
        persons=[],
        entities=[],
        topic=None,
    )
    vec_store.add_entries(
        [entry],
        tenant_id="test_tenant",
        memory_session_id="sess_001",
        source_kind="session",
        source_id=1,
    )
    vec_store.mark_superseded(entry.entry_id, "new_entry_999")
    # Should not raise; basic lifecycle call


@requires_pg
def test_vec_update_importance(vec_store):
    results = vec_store.get_all_entries(tenant_id="test_tenant")
    assert results
    vec_store.update_importance(results[0].entry_id, 0.9)


@requires_pg
def test_vec_no_sql_injection(vec_store):
    results = vec_store.structured_search(persons=["Alice')) OR 1=1--"])
    assert isinstance(results, list)


@requires_pg
def test_vec_optimize(vec_store):
    vec_store.optimize()


# =========================================================================
# PGSQLStorage tests
# =========================================================================

@pytest.fixture(scope="module")
def sql_store():
    from cross.storage_pg_sql import PGSQLStorage
    store = PGSQLStorage(
        dsn=PG_TEST_DSN,
        table_prefix="pytest_cross",
    )
    # Wipe state from any previous run
    store._purge_all_test_data()
    yield store
    store._purge_all_test_data()
    store.close()


@requires_pg
def test_sql_create_session(sql_store):
    session = sql_store.create_session(
        tenant_id="t1",
        content_session_id="cs_001",
        project="test-project",
        user_prompt="test prompt",
    )
    assert session.memory_session_id
    assert session.project == "test-project"
    assert session.status.value == "active"


@requires_pg
def test_sql_get_session_by_content_id(sql_store):
    s = sql_store.get_session_by_content_id("cs_001")
    assert s is not None
    assert s.content_session_id == "cs_001"


@requires_pg
def test_sql_get_session_by_memory_id(sql_store):
    s1 = sql_store.get_session_by_content_id("cs_001")
    s2 = sql_store.get_session_by_memory_id(s1.memory_session_id)
    assert s2 is not None
    assert s2.memory_session_id == s1.memory_session_id


@requires_pg
def test_sql_idempotent_create(sql_store):
    # Second call with same content_session_id must not raise or duplicate
    session = sql_store.create_session(
        tenant_id="t1",
        content_session_id="cs_001",
        project="test-project",
    )
    assert session.content_session_id == "cs_001"
    sessions = sql_store.list_sessions(project="test-project")
    assert sum(1 for s in sessions if s.content_session_id == "cs_001") == 1


@requires_pg
def test_sql_update_session_status(sql_store):
    from cross.types import SessionStatus
    s = sql_store.get_session_by_content_id("cs_001")
    sql_store.update_session_status(s.memory_session_id, SessionStatus.completed)
    updated = sql_store.get_session_by_memory_id(s.memory_session_id)
    assert updated.status == SessionStatus.completed


@requires_pg
def test_sql_list_sessions(sql_store):
    sessions = sql_store.list_sessions(project="test-project")
    assert len(sessions) >= 1


@requires_pg
def test_sql_add_event(sql_store):
    from cross.types import EventKind
    s = sql_store.get_session_by_content_id("cs_001")
    event_id = sql_store.add_event(
        s.memory_session_id,
        kind=EventKind.message,
        title="Test message",
        payload_json={"text": "hello"},
    )
    assert isinstance(event_id, int)


@requires_pg
def test_sql_get_events_for_session(sql_store):
    from cross.types import EventKind
    s = sql_store.get_session_by_content_id("cs_001")
    events = sql_store.get_events_for_session(s.memory_session_id)
    assert len(events) >= 1
    assert events[0].kind == EventKind.message


@requires_pg
def test_sql_store_observation(sql_store):
    from cross.types import ObservationType
    s = sql_store.get_session_by_content_id("cs_001")
    obs_id = sql_store.store_observation(
        s.memory_session_id,
        type=ObservationType.bugfix,
        title="Fixed null pointer",
        narrative="The null pointer was due to missing guard.",
        facts_json={"file": "main.py"},
    )
    assert isinstance(obs_id, int)


@requires_pg
def test_sql_get_observations_for_session(sql_store):
    from cross.types import ObservationType
    s = sql_store.get_session_by_content_id("cs_001")
    obs = sql_store.get_observations_for_session(s.memory_session_id)
    assert len(obs) >= 1
    assert obs[0].type == ObservationType.bugfix


@requires_pg
def test_sql_get_recent_observations(sql_store):
    obs = sql_store.get_recent_observations("test-project", limit=10)
    assert len(obs) >= 1


@requires_pg
def test_sql_store_summary(sql_store):
    s = sql_store.get_session_by_content_id("cs_001")
    summary_id = sql_store.store_summary(
        s.memory_session_id,
        request="Build the feature",
        learned="Tests must run first",
        completed="Feature implemented",
    )
    assert isinstance(summary_id, int)


@requires_pg
def test_sql_get_summary_for_session(sql_store):
    s = sql_store.get_session_by_content_id("cs_001")
    summary = sql_store.get_summary_for_session(s.memory_session_id)
    assert summary is not None
    assert summary.request == "Build the feature"


@requires_pg
def test_sql_get_recent_summaries(sql_store):
    summaries = sql_store.get_recent_summaries("test-project", limit=5)
    assert len(summaries) >= 1


@requires_pg
def test_sql_create_link(sql_store):
    link_id = sql_store.create_link(
        memory_entry_id="entry_abc",
        source_kind="session",
        source_id=1,
        score=0.85,
    )
    assert isinstance(link_id, int)


@requires_pg
def test_sql_get_links_for_entry(sql_store):
    links = sql_store.get_links_for_entry("entry_abc")
    assert len(links) >= 1
    assert links[0].score == pytest.approx(0.85, abs=1e-4)


@requires_pg
def test_sql_get_links_for_source(sql_store):
    links = sql_store.get_links_for_source("session", 1)
    assert len(links) >= 1


@requires_pg
def test_sql_record_consolidation_run(sql_store):
    run_id = sql_store.record_consolidation_run(
        tenant_id="t1",
        policy_json={"decay": 0.9},
        stats_json={"merged": 3},
    )
    assert isinstance(run_id, int)


@requires_pg
def test_sql_get_recent_consolidation_runs(sql_store):
    runs = sql_store.get_recent_consolidation_runs("t1", limit=5)
    assert len(runs) >= 1


@requires_pg
def test_sql_get_stats(sql_store):
    stats = sql_store.get_stats(project="test-project")
    assert "sessions" in stats
    assert stats["sessions"] >= 1
    assert "events" in stats
    assert "observations" in stats
    assert "summaries" in stats
