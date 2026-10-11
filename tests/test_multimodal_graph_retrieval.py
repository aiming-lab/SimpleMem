import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(params=["simplemem.multimodal", "omni_memory"])
def graph_module(request, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "OmniSimpleMem"))
    return importlib.import_module(f"{request.param}.knowledge.graph_retriever")


def retriever(module, graph, names):
    extractor = Mock()
    extractor.extract.return_value = SimpleNamespace(
        entities=[SimpleNamespace(name=name) for name in names]
    )
    return module.GraphRetriever(graph, entity_extractor=extractor)


def add_entity(module, graph, name, aliases=()):
    entity = module.Entity(name=name, aliases=list(aliases), source_mau_ids=[f"memory-{name}"])
    graph.add_entity(entity)
    return entity


def connect(module, graph, source, target):
    graph.add_relation(module.Relation(
        subject_id=source.entity_id, object_id=target.entity_id, predicate="knows"
    ))


@pytest.mark.parametrize("fallback", [False, True])
def test_duplicate_matches_do_not_consume_distinct_seed_slots(graph_module, tmp_path, fallback):
    graph = graph_module.KnowledgeGraph(storage_path=str(tmp_path))
    alice = add_entity(graph_module, graph, "Alice", aliases=["Alicia"])
    carol = add_entity(graph_module, graph, "Carol")
    dave = add_entity(graph_module, graph, "Dave")
    connect(graph_module, graph, carol, dave)
    names = [] if fallback else ["Alice", "Alicia", "Carol"]

    result = retriever(graph_module, graph, names).retrieve("Alice Alicia Carol", max_entities=2)

    assert [entity.entity_id for entity in result.entities] == [alice.entity_id, carol.entity_id]
    assert [(entity.entity_id, hops) for entity, hops, _ in result.related_entities] == [(dave.entity_id, 1)]
    assert "memory-Dave" in result.mau_ids


@pytest.mark.parametrize("names", [["Alice", "Bob"], ["Bob", "Alice"]])
def test_related_distance_uses_nearest_seed(graph_module, tmp_path, names):
    graph = graph_module.KnowledgeGraph(storage_path=str(tmp_path))
    alice, bridge, target, bob = [
        add_entity(graph_module, graph, name) for name in ["Alice", "Bridge", "Target", "Bob"]
    ]
    connect(graph_module, graph, alice, bridge)
    connect(graph_module, graph, bridge, target)
    connect(graph_module, graph, bob, target)

    result = retriever(graph_module, graph, names).retrieve("Alice and Bob", max_entities=2)

    related = {entity.name: (hops, path) for entity, hops, path in result.related_entities}
    assert related == {"Bridge": (1, "knows"), "Target": (1, "knows")}
    assert {entity.entity_id for entity in result.entities}.isdisjoint(
        entity.entity_id for entity, _, _ in result.related_entities
    )


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_seed_limit_preserves_first_distinct_match_order(graph_module, tmp_path, limit):
    graph = graph_module.KnowledgeGraph(storage_path=str(tmp_path))
    for name in ["Alice", "Bob", "Carol"]:
        add_entity(graph_module, graph, name)

    result = retriever(graph_module, graph, ["Alice", "Alice", "Bob", "Carol"]).retrieve(
        "query", max_entities=limit
    )

    assert [entity.name for entity in result.entities] == ["Alice", "Bob", "Carol"][:limit]
