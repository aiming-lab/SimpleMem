import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("rank_bm25")


@pytest.fixture(params=["simplemem.multimodal", "omni_memory"])
def bm25_module(request, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "OmniSimpleMem"))
    return importlib.import_module(f"{request.param}.retrieval.bm25_store")


def memory(identifier, text, tags=(), status="ACTIVE"):
    return SimpleNamespace(
        id=identifier,
        summary=text,
        details=None,
        status=status,
        metadata=SimpleNamespace(tags=list(tags)),
    )


class MemoryStore:
    def __init__(self, memories):
        self.memories = {item.id: item for item in memories}

    def iter_all(self):
        return iter(self.memories.values())

    def get(self, identifier):
        return self.memories.get(identifier)


@pytest.fixture
def indexed_store(bm25_module):
    # Relevant tagged documents rank behind four stronger, untagged matches.
    # Unrelated documents keep the query term's BM25 IDF positive.
    memories = [memory(f"other-{i}", "quasar " * (8 - i)) for i in range(4)]
    memories += [
        memory("tagged-first", "quasar quasar details", ["astronomy"]),
        memory("tagged-second", "quasar observation details", ["astronomy"]),
    ]
    memories += [memory(f"filler-{i}", f"unrelated background topic {i}") for i in range(10)]
    store = MemoryStore(memories)
    index = bm25_module.BM25Store()
    assert index.build_from_mau_store(store) == len(memories)
    return index, store


def test_filter_searches_past_unrelated_top_candidates(indexed_store):
    index, store = indexed_store
    unfiltered = index.search("quasar", top_k=6)
    assert [identifier for identifier, _ in unfiltered[:4]] == [f"other-{i}" for i in range(4)]

    results = index.search("quasar", top_k=2, tags_filter=["astronomy"], mau_store=store)

    assert results == unfiltered[4:]
    assert [identifier for identifier, _ in results] == ["tagged-first", "tagged-second"]


def test_missing_memory_cannot_satisfy_a_tag_filter(indexed_store):
    index, store = indexed_store
    del store.memories["other-0"]

    results = index.search("quasar", top_k=2, tags_filter=["astronomy"], mau_store=store)

    assert [identifier for identifier, _ in results] == ["tagged-first", "tagged-second"]


@pytest.mark.parametrize("empty_kind", ["removed", "archived", "untokenizable"])
def test_empty_rebuild_discards_previous_index(indexed_store, empty_kind):
    index, store = indexed_store
    assert index.search("quasar")
    if empty_kind == "removed":
        store.memories.clear()
    elif empty_kind == "archived":
        for item in store.memories.values():
            item.status = "ARCHIVED"
    else:
        for item in store.memories.values():
            item.summary = "!"

    assert index.build_from_mau_store(store) == 0
    assert index.search("quasar") == []
    assert not index.is_available
    assert index.doc_count == 0

    store.memories["new"] = memory("new", "fresh astronomy context")
    assert index.build_from_mau_store(store) == 1
    assert index.is_available
    assert index.doc_count == 1


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_limit_returns_no_results(indexed_store, top_k):
    index, _ = indexed_store
    assert index.search("quasar", top_k=top_k) == []
