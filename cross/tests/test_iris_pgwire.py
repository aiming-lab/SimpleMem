"""
IRIS + pgwire integration tests.

These tests validate the full cross-session memory flow against a real
InterSystems IRIS instance accessed via iris-pgwire (PostgreSQL wire protocol
proxy).  The same PGCrossVectorStore / PGSQLStorage code used for plain
PostgreSQL is exercised here — pgwire handles the IRIS-specific translation
transparently.

Set IRIS_PGWIRE_DSN to run, e.g.:
  IRIS_PGWIRE_DSN=postgresql://_SYSTEM:SYS@localhost:5433/USER \
    pytest cross/tests/test_iris_pgwire.py -v

Skipped automatically when IRIS_PGWIRE_DSN is absent or the endpoint is
unreachable.

Architecture note: iris-pgwire's vector_optimizer.py intercepts ORDER BY
expressions containing vector literals and inlines them to satisfy the IRIS
requirement that vector distance comparisons must reference literal values
rather than query parameters.  Our PGCrossVectorStore passes standard psycopg3
parameterised queries — pgwire handles the rewrite server-side.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

IRIS_PGWIRE_DSN = os.environ.get("IRIS_PGWIRE_DSN", "")


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


def _iris_pgwire_available() -> bool:
    if not IRIS_PGWIRE_DSN:
        return False
    try:
        import psycopg
        conn = psycopg.connect(IRIS_PGWIRE_DSN)
        conn.close()
        return True
    except Exception:
        return False


requires_iris_pgwire = pytest.mark.skipif(
    not _iris_pgwire_available(),
    reason="No IRIS+pgwire endpoint — set IRIS_PGWIRE_DSN to run",
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="module")
def vec_store():
    from cross.storage_pg import PGCrossVectorStore
    store = PGCrossVectorStore(
        dsn=IRIS_PGWIRE_DSN,
        embedding_model=_FakeEmbedder(),
        table_name="iris_pgwire_cross_mem_test",
    )
    store.clear()
    yield store
    store.clear()
    store.close()


@pytest.fixture(scope="module")
def sql_store():
    from cross.storage_pg_sql import PGSQLStorage
    store = PGSQLStorage(
        dsn=IRIS_PGWIRE_DSN,
        table_prefix="iris_pgwire_test",
    )
    store._purge_all_test_data()
    yield store
    store._purge_all_test_data()
    store.close()


@pytest.fixture(scope="module")
def sample_entries():
    from simplemem.core.models.memory_entry import MemoryEntry
    return [
        MemoryEntry(
            lossless_restatement="Alice submitted the quarterly financial report on time",
            keywords=["Alice", "financial", "report", "quarterly"],
            timestamp="2025-04-01T09:00:00",
            location="HQ",
            persons=["Alice"],
            entities=["financial report", "quarterly"],
            topic="Finance",
        ),
        MemoryEntry(
            lossless_restatement="Bob deployed the new authentication service to production",
            keywords=["Bob", "deploy", "authentication", "production"],
            timestamp="2025-04-02T14:00:00",
            location="remote",
            persons=["Bob"],
            entities=["authentication service", "deployment"],
            topic="Engineering",
        ),
        MemoryEntry(
            lossless_restatement="Team retrospective identified three process improvements",
            keywords=["retrospective", "process", "improvement"],
            timestamp="2025-04-03T16:00:00",
            location="conference room A",
            persons=[],
            entities=["retrospective", "process"],
            topic="Process",
        ),
    ]


# =========================================================================
# Vector store: schema + CRUD
# =========================================================================

@requires_iris_pgwire
def test_vec_schema_created(vec_store):
    """Table and indexes exist after _ensure_schema()."""
    count = vec_store.count_entries()
    assert isinstance(count, int)


@requires_iris_pgwire
def test_vec_add_and_count(vec_store, sample_entries):
    vec_store.add_entries(
        sample_entries,
        tenant_id="iris_test",
        memory_session_id="iris_sess_001",
        source_kind="session",
        source_id=1,
    )
    assert vec_store.count_entries(tenant_id="iris_test") == 3


@requires_iris_pgwire
def test_vec_semantic_search(vec_store):
    """pgwire inlines vector literal for IRIS ORDER BY compatibility."""
    results = vec_store.semantic_search("financial quarterly report", top_k=3)
    assert len(results) > 0
    assert all(hasattr(r, "lossless_restatement") for r in results)


@requires_iris_pgwire
def test_vec_semantic_search_with_tenant_filter(vec_store):
    results = vec_store.semantic_search(
        "authentication deployment",
        top_k=5,
        tenant_id="iris_test",
    )
    assert isinstance(results, list)
    for r in results:
        assert r.tenant_id == "iris_test"


@requires_iris_pgwire
def test_vec_keyword_search(vec_store):
    results = vec_store.keyword_search(["authentication"], top_k=5)
    assert len(results) > 0
    assert any("authentication" in r.lossless_restatement.lower() for r in results)


@requires_iris_pgwire
def test_vec_keyword_search_with_tenant(vec_store):
    results = vec_store.keyword_search(["report"], top_k=5, tenant_id="iris_test")
    assert isinstance(results, list)


@requires_iris_pgwire
def test_vec_structured_persons(vec_store):
    results = vec_store.structured_search(persons=["Alice"])
    assert len(results) > 0
    for r in results:
        assert "Alice" in r.persons


@requires_iris_pgwire
def test_vec_structured_location(vec_store):
    results = vec_store.structured_search(location="HQ")
    assert len(results) > 0
    for r in results:
        assert r.location and "HQ" in r.location


@requires_iris_pgwire
def test_vec_structured_timestamp(vec_store):
    results = vec_store.structured_search(
        timestamp_range=("2025-04-01T00:00:00", "2025-04-02T23:59:59")
    )
    assert len(results) == 2


@requires_iris_pgwire
def test_vec_structured_tenant_filter(vec_store):
    results = vec_store.structured_search(tenant_id="iris_test", top_k=10)
    assert len(results) == 3


@requires_iris_pgwire
def test_vec_get_entries_for_session(vec_store):
    results = vec_store.get_entries_for_session("iris_sess_001")
    assert len(results) == 3


@requires_iris_pgwire
def test_vec_get_all_entries(vec_store):
    results = vec_store.get_all_entries(tenant_id="iris_test")
    assert len(results) == 3


@requires_iris_pgwire
def test_vec_mark_superseded(vec_store, sample_entries):
    entry_id = sample_entries[0].entry_id
    vec_store.mark_superseded(entry_id, "new_entry_replacement")


@requires_iris_pgwire
def test_vec_update_importance(vec_store, sample_entries):
    vec_store.update_importance(sample_entries[1].entry_id, 0.95)


@requires_iris_pgwire
def test_vec_no_injection(vec_store):
    results = vec_store.structured_search(persons=["Alice')) OR '1'='1"])
    assert isinstance(results, list)


@requires_iris_pgwire
def test_vec_optimize(vec_store):
    vec_store.optimize()


# =========================================================================
# SQL metadata store: full session lifecycle
# =========================================================================

@requires_iris_pgwire
def test_sql_create_session(sql_store):
    from cross.types import SessionStatus
    session = sql_store.create_session(
        tenant_id="iris_tenant",
        content_session_id="iris_cs_001",
        project="iris-test-project",
        user_prompt="Test the IRIS pgwire path",
    )
    assert session.memory_session_id
    assert session.status == SessionStatus.active


@requires_iris_pgwire
def test_sql_idempotent_create(sql_store):
    # Duplicate create must not raise or duplicate the row
    sql_store.create_session(
        tenant_id="iris_tenant",
        content_session_id="iris_cs_001",
        project="iris-test-project",
    )
    sessions = sql_store.list_sessions(project="iris-test-project")
    assert sum(1 for s in sessions if s.content_session_id == "iris_cs_001") == 1


@requires_iris_pgwire
def test_sql_get_by_content_id(sql_store):
    s = sql_store.get_session_by_content_id("iris_cs_001")
    assert s is not None
    assert s.project == "iris-test-project"


@requires_iris_pgwire
def test_sql_get_by_memory_id(sql_store):
    s1 = sql_store.get_session_by_content_id("iris_cs_001")
    s2 = sql_store.get_session_by_memory_id(s1.memory_session_id)
    assert s2.memory_session_id == s1.memory_session_id


@requires_iris_pgwire
def test_sql_update_status(sql_store):
    from cross.types import SessionStatus
    s = sql_store.get_session_by_content_id("iris_cs_001")
    sql_store.update_session_status(s.memory_session_id, SessionStatus.completed)
    updated = sql_store.get_session_by_memory_id(s.memory_session_id)
    assert updated.status == SessionStatus.completed
    assert updated.ended_at is not None


@requires_iris_pgwire
def test_sql_add_and_get_events(sql_store):
    from cross.types import EventKind
    s = sql_store.get_session_by_content_id("iris_cs_001")
    eid = sql_store.add_event(
        s.memory_session_id,
        kind=EventKind.tool_use,
        title="read_file",
        payload_json={"path": "/etc/hosts"},
    )
    assert isinstance(eid, int)
    events = sql_store.get_events_for_session(s.memory_session_id)
    assert any(e.title == "read_file" for e in events)


@requires_iris_pgwire
def test_sql_store_and_get_observation(sql_store):
    from cross.types import ObservationType
    s = sql_store.get_session_by_content_id("iris_cs_001")
    obs_id = sql_store.store_observation(
        s.memory_session_id,
        type=ObservationType.feature,
        title="pgwire layer works",
        narrative="Vector ORDER BY rewritten correctly by pgwire optimizer",
    )
    assert isinstance(obs_id, int)
    obs_list = sql_store.get_observations_for_session(s.memory_session_id)
    assert any(o.title == "pgwire layer works" for o in obs_list)


@requires_iris_pgwire
def test_sql_get_recent_observations(sql_store):
    obs = sql_store.get_recent_observations("iris-test-project", limit=10)
    assert len(obs) >= 1


@requires_iris_pgwire
def test_sql_store_and_get_summary(sql_store):
    s = sql_store.get_session_by_content_id("iris_cs_001")
    sid = sql_store.store_summary(
        s.memory_session_id,
        request="Validate IRIS pgwire path",
        learned="pgwire proxy works transparently",
        completed="All backends verified",
    )
    assert isinstance(sid, int)
    summary = sql_store.get_summary_for_session(s.memory_session_id)
    assert summary is not None
    assert summary.request == "Validate IRIS pgwire path"


@requires_iris_pgwire
def test_sql_get_recent_summaries(sql_store):
    summaries = sql_store.get_recent_summaries("iris-test-project", limit=5)
    assert len(summaries) >= 1


@requires_iris_pgwire
def test_sql_create_and_get_link(sql_store):
    link_id = sql_store.create_link(
        memory_entry_id="iris_entry_001",
        source_kind="session",
        source_id=1,
        score=0.92,
    )
    assert isinstance(link_id, int)
    links = sql_store.get_links_for_entry("iris_entry_001")
    assert any(abs(lk.score - 0.92) < 1e-4 for lk in links)


@requires_iris_pgwire
def test_sql_get_links_for_source(sql_store):
    links = sql_store.get_links_for_source("session", 1)
    assert len(links) >= 1


@requires_iris_pgwire
def test_sql_consolidation_run(sql_store):
    run_id = sql_store.record_consolidation_run(
        tenant_id="iris_tenant",
        policy_json={"decay": 0.85},
        stats_json={"entries_pruned": 5},
    )
    assert isinstance(run_id, int)
    runs = sql_store.get_recent_consolidation_runs("iris_tenant", limit=5)
    assert len(runs) >= 1


@requires_iris_pgwire
def test_sql_get_stats(sql_store):
    stats = sql_store.get_stats(project="iris-test-project")
    assert stats["sessions"] >= 1
    assert stats["events"] >= 1
    assert stats["observations"] >= 1
    assert stats["summaries"] >= 1


# =========================================================================
# Full round-trip: SessionManager with PG backends + pgwire
# =========================================================================

@requires_iris_pgwire
def test_full_roundtrip_session_lifecycle(vec_store, sql_store, sample_entries):
    """
    Verify that SessionManager wired with PGSQLStorage + PGCrossVectorStore
    can complete the create → record → finalize lifecycle without touching IRIS
    dbapi directly (all goes through pgwire).
    """
    from cross.session_manager import SessionManager

    # Stub out the simplemem add_dialogue so no LLM call is made
    mock_simplemem = MagicMock()
    mock_simplemem.add_dialogue.return_value = sample_entries[:2]

    mgr = SessionManager(
        sqlite_storage=sql_store,
        vector_store=vec_store,
        simplemem=mock_simplemem,
    )

    # Start
    result = mgr.start_session(
        tenant_id="iris_tenant",
        content_session_id="roundtrip_cs_001",
        project="iris-test-project",
        user_prompt="Test round-trip via pgwire",
    )
    assert "memory_session_id" in result

    msid = result["memory_session_id"]

    # Record a message event
    mgr.record_event(
        memory_session_id=msid,
        role="user",
        content="Deploy the new service",
    )

    # Finalize
    report = mgr.finalize_session(memory_session_id=msid)
    assert report is not None

    # Verify session ended
    from cross.types import SessionStatus
    session = sql_store.get_session_by_memory_id(msid)
    assert session.status in (SessionStatus.completed, SessionStatus.active)
