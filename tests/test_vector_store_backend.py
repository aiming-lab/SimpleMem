import json

import numpy as np
import pytest

from simplemem.core.database import (
    LanceDBVectorStoreBackend,
    ScoreOrder,
    VectorStore,
    VectorStoreSearchResult,
)
from simplemem.core.hybrid_retriever import HybridRetriever
from simplemem.core.models.memory_entry import MemoryEntry


class DeterministicEmbedder:
    dimension = 3

    def encode_documents(self, texts):
        return np.stack([self._encode(text) for text in texts])

    def encode_single(self, text, is_query=False):
        return self._encode(text)

    @staticmethod
    def _encode(text):
        text = text.lower()
        if "coffee" in text or "espresso" in text:
            vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif "apollo" in text or "budget" in text:
            vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            vector = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        return vector


class DeterministicLLM:
    def chat_completion(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "extract key information" in prompt:
            return json.dumps(
                {
                    "keywords": ["Apollo"],
                    "persons": ["Carol"],
                    "time_expression": None,
                    "location": None,
                    "entities": [],
                }
            )
        if "information requirements analysis" in prompt:
            return json.dumps(
                {
                    "reasoning": "Use one semantic query.",
                    "queries": ["coffee status"],
                }
            )
        if "determine what specific information is required" in prompt:
            return json.dumps(
                {
                    "question_type": "factual",
                    "key_entities": ["coffee", "Apollo", "Carol"],
                    "required_info": [
                        {
                            "info_type": "facts",
                            "description": "Retrieve all three facts",
                            "priority": "high",
                        }
                    ],
                    "relationships": [],
                    "minimal_queries_needed": 1,
                }
            )
        raise AssertionError(f"Unexpected LLM prompt: {prompt[:120]}")

    @staticmethod
    def extract_json(response):
        return json.loads(response)


class InMemoryVectorStoreBackend:
    semantic_score_order = ScoreOrder.ASCENDING
    keyword_score_order = ScoreOrder.DESCENDING

    def __init__(self):
        self.records = []
        self.optimized = False

    def insert(self, records):
        self.records.extend(records)

    def semantic_search(self, query_vector, top_k, filters=None):
        records = self.records
        if filters:
            records = [
                record
                for record in records
                if all(
                    record.metadata.get(key) == value for key, value in filters.items()
                )
            ]
        ranked = sorted(
            records,
            key=lambda record: sum(
                (left - right) ** 2 for left, right in zip(record.vector, query_vector)
            ),
        )
        return [
            self._result(
                record,
                score=sum(
                    (left - right) ** 2
                    for left, right in zip(record.vector, query_vector)
                ),
            )
            for record in ranked[:top_k]
        ]

    def keyword_search(self, keywords, top_k):
        lowered_keywords = [keyword.lower() for keyword in keywords]

        def keyword_score(record):
            text = " ".join(
                [
                    record.metadata["lossless_restatement"],
                    *record.metadata["keywords"],
                ]
            ).lower()
            return sum(text.count(keyword) for keyword in lowered_keywords)

        ranked = [
            (record, keyword_score(record))
            for record in self.records
            if keyword_score(record) > 0
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [self._result(record, score=score) for record, score in ranked[:top_k]]

    def structured_search(
        self,
        persons=None,
        timestamp_range=None,
        location=None,
        entities=None,
        top_k=None,
    ):
        matches = []
        for record in self.records:
            metadata = record.metadata
            if persons and not set(persons).intersection(metadata["persons"]):
                continue
            if location and location not in metadata["location"]:
                continue
            if entities and not set(entities).intersection(metadata["entities"]):
                continue
            if timestamp_range:
                start_time, end_time = timestamp_range
                if not start_time <= metadata["timestamp"] <= end_time:
                    continue
            matches.append(self._result(record))
        return matches[:top_k] if top_k else matches

    def count(self):
        return len(self.records)

    def get_all(self):
        return [self._result(record) for record in self.records]

    def optimize(self):
        self.optimized = True

    def clear(self):
        self.records.clear()

    @staticmethod
    def _result(record, score=None):
        return VectorStoreSearchResult(
            entry_id=record.entry_id,
            metadata=record.metadata,
            score=score,
        )


class TrackingVectorStoreBackend:
    def __init__(self, delegate):
        self.delegate = delegate
        self.semantic_score_order = delegate.semantic_score_order
        self.keyword_score_order = delegate.keyword_score_order
        self.calls = {
            "insert": 0,
            "semantic_search": 0,
            "keyword_search": 0,
            "structured_search": 0,
            "get_all": 0,
            "optimize": 0,
            "clear": 0,
        }
        self.last_records = []

    def insert(self, records):
        self.calls["insert"] += 1
        self.last_records = list(records)
        self.delegate.insert(records)

    def semantic_search(self, query_vector, top_k, filters=None):
        self.calls["semantic_search"] += 1
        return self.delegate.semantic_search(query_vector, top_k, filters)

    def keyword_search(self, keywords, top_k):
        self.calls["keyword_search"] += 1
        return self.delegate.keyword_search(keywords, top_k)

    def structured_search(self, **kwargs):
        self.calls["structured_search"] += 1
        return self.delegate.structured_search(**kwargs)

    def count(self):
        return self.delegate.count()

    def get_all(self):
        self.calls["get_all"] += 1
        return self.delegate.get_all()

    def optimize(self):
        self.calls["optimize"] += 1
        self.delegate.optimize()

    def clear(self):
        self.calls["clear"] += 1
        self.delegate.clear()


@pytest.fixture
def entries():
    return [
        MemoryEntry(
            entry_id="coffee",
            lossless_restatement="Alice drinks espresso at the neighborhood cafe.",
            keywords=["coffee", "espresso"],
            persons=["Alice"],
            topic="coffee",
        ),
        MemoryEntry(
            entry_id="apollo",
            lossless_restatement="The Project Apollo budget was approved.",
            keywords=["Apollo", "budget"],
            persons=["Bob"],
            topic="finance",
        ),
        MemoryEntry(
            entry_id="carol",
            lossless_restatement="Carol planned a trip to Paris.",
            keywords=["travel", "Paris"],
            persons=["Carol"],
            location="Paris",
            topic="travel",
        ),
    ]


@pytest.fixture
def lancedb_store(tmp_path, entries):
    store = VectorStore(
        db_path=str(tmp_path / "lancedb"),
        table_name="entries",
        embedding_model=DeterministicEmbedder(),
    )
    store.add_entries(entries)
    return store


@pytest.fixture
def custom_store(tmp_path, entries):
    created = []
    unused_default_path = tmp_path / "unused-lancedb"

    def factory(vector_dimension):
        assert vector_dimension == DeterministicEmbedder.dimension
        backend = TrackingVectorStoreBackend(InMemoryVectorStoreBackend())
        created.append(backend)
        return backend

    store = VectorStore(
        db_path=str(unused_default_path),
        table_name="entries",
        embedding_model=DeterministicEmbedder(),
        backend_factory=factory,
    )
    store.add_entries(entries)

    assert not unused_default_path.exists()
    assert store.backend is created[0]
    return store


def test_lancedb_backend_preserves_semantic_and_keyword_score_order(lancedb_store):
    backend = lancedb_store.backend
    assert isinstance(backend, LanceDBVectorStoreBackend)

    semantic_results = backend.semantic_search([1.0, 0.0, 0.0], top_k=3)
    keyword_results = backend.keyword_search(["Apollo"], top_k=3)

    assert backend.semantic_score_order == ScoreOrder.ASCENDING
    assert backend.keyword_score_order == ScoreOrder.DESCENDING
    assert [result.entry_id for result in semantic_results] == [
        "coffee",
        "apollo",
        "carol",
    ]
    assert [result.score for result in semantic_results] == sorted(
        result.score for result in semantic_results
    )
    assert [result.entry_id for result in keyword_results] == ["apollo"]
    assert keyword_results[0].score is not None


def test_custom_backend_owns_all_three_retrieval_paths(custom_store):
    assert (
        custom_store.semantic_search("coffee status", top_k=1)[0].entry_id == "coffee"
    )
    assert custom_store.keyword_search(["Apollo"], top_k=1)[0].entry_id == "apollo"
    assert (
        custom_store.structured_search(persons=["Carol"], top_k=1)[0].entry_id
        == "carol"
    )

    assert custom_store.backend.calls["insert"] == 1
    assert custom_store.backend.calls["semantic_search"] == 1
    assert custom_store.backend.calls["keyword_search"] == 1
    assert custom_store.backend.calls["structured_search"] == 1
    assert custom_store.backend.last_records[1].metadata["topic"] == "finance"


def test_lancedb_backend_applies_scalar_semantic_filters(lancedb_store):
    results = lancedb_store.backend.semantic_search(
        [1.0, 0.0, 0.0],
        top_k=3,
        filters={"topic": "finance"},
    )

    assert [result.entry_id for result in results] == ["apollo"]


def test_lancedb_backend_rejects_unsafe_filter_fields(lancedb_store):
    with pytest.raises(ValueError, match="Invalid semantic filter field"):
        lancedb_store.backend.semantic_search(
            [1.0, 0.0, 0.0],
            top_k=3,
            filters={"topic OR TRUE": "finance"},
        )


def test_lancedb_backend_escapes_filter_values(lancedb_store):
    results = lancedb_store.backend.semantic_search(
        [1.0, 0.0, 0.0],
        top_k=3,
        filters={"topic": "finance' OR TRUE"},
    )

    assert results == []


def test_lancedb_backend_preserves_all_three_public_retrieval_paths(lancedb_store):
    assert (
        lancedb_store.semantic_search("coffee status", top_k=1)[0].entry_id == "coffee"
    )
    assert lancedb_store.keyword_search(["Apollo"], top_k=1)[0].entry_id == "apollo"
    assert (
        lancedb_store.structured_search(persons=["Carol"], top_k=1)[0].entry_id
        == "carol"
    )


def _add_entry(store, entry_id, text):
    store.add_entries(
        [MemoryEntry(entry_id=entry_id, lossless_restatement=text, topic=entry_id)]
    )


def test_lancedb_keyword_search_sees_entries_from_later_inserts(lancedb_store):
    assert [e.entry_id for e in lancedb_store.keyword_search(["Apollo"], 5)] == [
        "apollo"
    ]

    _add_entry(lancedb_store, "zebra", "Dana photographed a zebra in Kenya.")
    _add_entry(lancedb_store, "tulip", "Erin planted tulip bulbs in the garden.")

    assert lancedb_store.backend.count() == 5
    assert [e.entry_id for e in lancedb_store.keyword_search(["zebra"], 5)] == [
        "zebra"
    ]
    assert [e.entry_id for e in lancedb_store.keyword_search(["tulip"], 5)] == [
        "tulip"
    ]
    # Entries from the first batch stay searchable.
    assert [e.entry_id for e in lancedb_store.keyword_search(["Apollo"], 5)] == [
        "apollo"
    ]


def test_lancedb_keyword_search_after_reopening_a_table(lancedb_store):
    _add_entry(lancedb_store, "zebra", "Dana photographed a zebra in Kenya.")

    reopened = VectorStore(
        db_path=lancedb_store.backend.db_path,
        table_name="entries",
        embedding_model=DeterministicEmbedder(),
    )

    assert [e.entry_id for e in reopened.keyword_search(["zebra"], 5)] == ["zebra"]


def test_lancedb_keyword_search_after_clear(lancedb_store):
    assert lancedb_store.keyword_search(["Apollo"], 5)

    lancedb_store.clear()
    _add_entry(lancedb_store, "zebra", "Dana photographed a zebra in Kenya.")

    assert [e.entry_id for e in lancedb_store.keyword_search(["zebra"], 5)] == [
        "zebra"
    ]
    assert lancedb_store.keyword_search(["Apollo"], 5) == []


def test_lancedb_fts_index_is_rebuilt_lazily(lancedb_store):
    backend = lancedb_store.backend
    builds = []
    original = backend.table.create_fts_index

    def tracking_create_fts_index(*args, **kwargs):
        builds.append(kwargs.get("use_tantivy"))
        return original(*args, **kwargs)

    backend.table.create_fts_index = tracking_create_fts_index

    # Several inserts in a row do not rebuild the index until it is queried.
    _add_entry(lancedb_store, "zebra", "Dana photographed a zebra in Kenya.")
    _add_entry(lancedb_store, "tulip", "Erin planted tulip bulbs in the garden.")
    assert builds == []

    lancedb_store.keyword_search(["zebra"], 5)
    lancedb_store.keyword_search(["tulip"], 5)
    assert builds == [True]

    _add_entry(lancedb_store, "otter", "Frank saw an otter by the river.")
    lancedb_store.keyword_search(["otter"], 5)
    assert builds == [True, True]


def test_custom_backend_owns_lifecycle_operations(custom_store):
    assert len(custom_store.get_all_entries()) == 3
    custom_store.optimize()
    custom_store.clear()

    assert custom_store.backend.calls["get_all"] == 1
    assert custom_store.backend.calls["optimize"] == 1
    assert custom_store.backend.calls["clear"] == 1
    assert custom_store.get_all_entries() == []


def test_hybrid_retrieval_uses_custom_backend_for_all_paths(custom_store):
    retriever = HybridRetriever(
        llm_client=DeterministicLLM(),
        vector_store=custom_store,
        semantic_top_k=1,
        keyword_top_k=1,
        structured_top_k=1,
        enable_planning=True,
        enable_reflection=False,
        enable_parallel_retrieval=False,
    )

    results = retriever.retrieve("coffee Apollo Carol")

    assert [entry.entry_id for entry in results] == ["coffee", "apollo", "carol"]
