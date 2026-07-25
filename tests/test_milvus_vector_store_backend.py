import json

import numpy as np
import pytest

pytest.importorskip("pymilvus")
from pymilvus import DataType, Function, FunctionType, MilvusClient

from simplemem.core.database import (
    MilvusVectorStoreBackend,
    ScoreOrder,
    VectorStore,
    VectorStoreRecord,
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
            vector = [1.0, 0.0, 0.0]
        elif "apollo" in text or "budget" in text:
            vector = [0.0, 1.0, 0.0]
        else:
            vector = [-1.0, 0.0, 0.0]
        return np.array(vector, dtype=np.float32)


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


def _record(
    entry_id,
    vector,
    text,
    keywords,
    timestamp,
    location,
    persons,
    entities,
    topic,
):
    return VectorStoreRecord(
        entry_id=entry_id,
        vector=vector,
        metadata={
            "lossless_restatement": text,
            "keywords": keywords,
            "timestamp": timestamp,
            "location": location,
            "persons": persons,
            "entities": entities,
            "topic": topic,
        },
    )


@pytest.fixture
def records():
    return [
        _record(
            "coffee",
            [1.0, 0.0, 0.0],
            "Alice drinks espresso at the neighborhood cafe.",
            ["coffee", "espresso"],
            "2026-01-10T09:00:00",
            "New York",
            ["Alice"],
            ["Neighborhood Cafe"],
            "coffee",
        ),
        _record(
            "apollo",
            [0.0, 1.0, 0.0],
            "The Project Apollo budget budget was approved.",
            ["Apollo", "budget"],
            "2026-02-10T09:00:00",
            "Paris",
            ["Bob"],
            ["Project Apollo"],
            "finance",
        ),
        _record(
            "carol",
            [-1.0, 0.0, 0.0],
            "Carol planned a trip to Paris.",
            ["travel", "Paris"],
            "2026-03-10T09:00:00",
            "Paris",
            ["Carol"],
            ["Rail Europe"],
            "travel",
        ),
        _record(
            "apollo-brief",
            [0.0, 0.0, 1.0],
            "Apollo launch status was discussed.",
            ["Apollo", "launch"],
            "2026-04-10T09:00:00",
            "Houston",
            ["Dana"],
            ["Project Apollo"],
            "space",
        ),
    ]


@pytest.fixture
def milvus_backend(tmp_path, records):
    backend = MilvusVectorStoreBackend(
        collection_name="memory_entries",
        vector_dimension=3,
        uri=str(tmp_path / "milvus.db"),
    )
    backend.insert(records)
    try:
        yield backend
    finally:
        backend.close()


def test_milvus_lite_preserves_semantic_and_keyword_score_order(milvus_backend):
    semantic_results = milvus_backend.semantic_search([1.0, 0.0, 0.0], top_k=4)
    keyword_results = milvus_backend.keyword_search(["Apollo", "budget"], top_k=4)

    assert milvus_backend.semantic_score_order == ScoreOrder.ASCENDING
    assert milvus_backend.keyword_score_order == ScoreOrder.DESCENDING
    assert [result.entry_id for result in semantic_results] == [
        "coffee",
        "apollo",
        "apollo-brief",
        "carol",
    ]
    assert [result.score for result in semantic_results] == sorted(
        result.score for result in semantic_results
    )
    assert keyword_results[0].entry_id == "apollo"
    assert {result.entry_id for result in keyword_results} == {
        "apollo",
        "apollo-brief",
    }
    assert [result.score for result in keyword_results] == sorted(
        (result.score for result in keyword_results), reverse=True
    )
    assert all(result.score >= 0 for result in keyword_results)


def test_milvus_lite_applies_safe_semantic_filters(milvus_backend):
    scalar_results = milvus_backend.semantic_search(
        [1.0, 0.0, 0.0],
        top_k=4,
        filters={"topic": ["finance", "travel"]},
    )
    array_results = milvus_backend.semantic_search(
        [1.0, 0.0, 0.0],
        top_k=4,
        filters={"persons": ["Alice", "Carol"]},
    )
    escaped_results = milvus_backend.semantic_search(
        [1.0, 0.0, 0.0],
        top_k=4,
        filters={"topic": 'finance" or true'},
    )

    assert [result.entry_id for result in scalar_results] == ["apollo", "carol"]
    assert [result.entry_id for result in array_results] == ["coffee", "carol"]
    assert escaped_results == []

    with pytest.raises(ValueError, match="Invalid semantic filter field"):
        milvus_backend.semantic_search(
            [1.0, 0.0, 0.0],
            top_k=4,
            filters={"topic or true": "finance"},
        )
    with pytest.raises(TypeError, match="scalar filter.*supports only strings"):
        milvus_backend.semantic_search(
            [1.0, 0.0, 0.0],
            top_k=4,
            filters={"topic": {"nested": "value"}},
        )


def test_milvus_lite_supports_structured_search(milvus_backend):
    assert {
        result.entry_id
        for result in milvus_backend.structured_search(persons=["Alice", "Carol"])
    } == {"coffee", "carol"}
    assert {
        result.entry_id for result in milvus_backend.structured_search(location="Paris")
    } == {"apollo", "carol"}
    assert {
        result.entry_id
        for result in milvus_backend.structured_search(entities=["Project Apollo"])
    } == {"apollo", "apollo-brief"}
    assert [
        result.entry_id
        for result in milvus_backend.structured_search(
            timestamp_range=("2026-02-01", "2026-02-28"),
        )
    ] == ["apollo"]
    assert len(milvus_backend.structured_search(location="Paris", top_k=1)) == 1

    with pytest.raises(ValueError, match="wildcard"):
        milvus_backend.structured_search(location="Paris%")


def test_milvus_lite_preserves_metadata_and_lifecycle_operations(milvus_backend):
    assert milvus_backend.count() == 4
    results = {result.entry_id: result for result in milvus_backend.get_all()}
    assert set(results) == {"coffee", "apollo", "carol", "apollo-brief"}
    assert results["apollo"].metadata == {
        "lossless_restatement": "The Project Apollo budget budget was approved.",
        "keywords": ["Apollo", "budget"],
        "timestamp": "2026-02-10T09:00:00",
        "location": "Paris",
        "persons": ["Bob"],
        "entities": ["Project Apollo"],
        "topic": "finance",
    }

    milvus_backend.optimize()
    milvus_backend.clear()

    assert milvus_backend.count() == 0
    assert milvus_backend.get_all() == []
    assert milvus_backend.semantic_search([1.0, 0.0, 0.0], top_k=3) == []


def test_milvus_lite_validates_reused_collection_dimension(tmp_path, records):
    uri = str(tmp_path / "reused.db")
    backend = MilvusVectorStoreBackend("memory_entries", 3, uri=uri)
    backend.insert(records[:1])
    backend.close()

    reopened = MilvusVectorStoreBackend("memory_entries", 3, uri=uri)
    assert reopened.count() == 1
    reopened.close()

    with pytest.raises(ValueError, match="vector dimension 3; expected 4"):
        MilvusVectorStoreBackend("memory_entries", 4, uri=uri)


def test_reused_collection_forwards_configured_consistency_level(
    tmp_path, records, monkeypatch
):
    uri = str(tmp_path / "consistency.db")
    initial = MilvusVectorStoreBackend(
        "memory_entries",
        3,
        uri=uri,
        consistency_level="Bounded",
    )
    initial.insert(records)
    initial.close()

    class RecordingMilvusClient(MilvusClient):
        def __init__(self, *args, **kwargs):
            self.read_calls = []
            super().__init__(*args, **kwargs)

        def search(self, *args, **kwargs):
            self.read_calls.append(("search", kwargs.get("consistency_level")))
            return super().search(*args, **kwargs)

        def query(self, *args, **kwargs):
            self.read_calls.append(("query", kwargs.get("consistency_level")))
            return super().query(*args, **kwargs)

        def query_iterator(self, *args, **kwargs):
            self.read_calls.append(("query_iterator", kwargs.get("consistency_level")))
            return super().query_iterator(*args, **kwargs)

    monkeypatch.setattr(
        MilvusVectorStoreBackend,
        "_load_pymilvus",
        staticmethod(lambda: (RecordingMilvusClient, DataType, Function, FunctionType)),
    )
    backend = MilvusVectorStoreBackend(
        "memory_entries",
        3,
        uri=uri,
        consistency_level="Session",
    )
    try:
        assert backend.count() == len(records)
        backend.semantic_search([1.0, 0.0, 0.0], top_k=1)
        backend.keyword_search(["Apollo"], top_k=1)
        backend.structured_search(persons=["Alice"], top_k=1)
        backend.get_all()

        methods = [method for method, _ in backend.client.read_calls]
        assert methods.count("search") == 2
        assert methods.count("query") >= 4
        assert methods.count("query_iterator") == 1
        assert all(
            consistency_level == "Session"
            for _, consistency_level in backend.client.read_calls
        )
    finally:
        backend.close()


def test_milvus_lite_3_0_cosine_workaround_is_narrow():
    assert MilvusVectorStoreBackend._is_lite_3_0_cosine_distance("./milvus.db", "3.0.0")
    assert not MilvusVectorStoreBackend._is_lite_3_0_cosine_distance(
        "./milvus.db", "3.1.0"
    )
    assert not MilvusVectorStoreBackend._is_lite_3_0_cosine_distance(
        "https://example.api.zillizcloud.com", "3.0.0"
    )
    assert not MilvusVectorStoreBackend._is_lite_3_0_cosine_distance(
        "localhost:19530", "3.0.0"
    )


def test_hybrid_retrieval_uses_milvus_for_all_paths(tmp_path):
    entries = [
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
    store = VectorStore(
        db_path=str(tmp_path / "unused-lancedb"),
        table_name="memory_entries",
        embedding_model=DeterministicEmbedder(),
        backend_factory=lambda dimension: MilvusVectorStoreBackend(
            collection_name="memory_entries",
            vector_dimension=dimension,
            uri=str(tmp_path / "hybrid.db"),
        ),
    )
    store.add_entries(entries)
    retriever = HybridRetriever(
        llm_client=DeterministicLLM(),
        vector_store=store,
        semantic_top_k=1,
        keyword_top_k=1,
        structured_top_k=1,
        enable_planning=True,
        enable_reflection=False,
        enable_parallel_retrieval=False,
    )

    try:
        results = retriever.retrieve("coffee Apollo Carol")
        assert [entry.entry_id for entry in results] == [
            "coffee",
            "apollo",
            "carol",
        ]
        assert not (tmp_path / "unused-lancedb").exists()
    finally:
        store.backend.close()
