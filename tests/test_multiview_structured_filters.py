import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(params=["simplemem.evolver", "evolvemem"])
def index(request, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "EvolveMem"))
    module = importlib.import_module(f"{request.param}.multi_retriever")

    class FixedEmbedder:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 2))

    return module.MultiViewIndex(
        [
            {
                "content": "Alice had coffee in Paris.",
                "persons": ["Alice"],
                "location": "Paris",
                "entities": ["coffee"],
            },
            {
                "content": "Bob had tea in London.",
                "persons": ["Bob"],
                "location": "London",
                "entities": ["tea"],
            },
            {
                "content": "Alice had coffee in London.",
                "persons": ["Alice"],
                "location": "London",
                "entities": ["coffee"],
            },
        ],
        embedder=FixedEmbedder(),
    )


@pytest.mark.parametrize(
    "filters",
    [
        {"persons": ["Alice"], "location": "Berlin"},
        {"persons": ["Alice"], "entities": ["cake"]},
        {"location": "Berlin", "entities": ["coffee"]},
        {"location": "Paris", "entities": ["cake"]},
    ],
)
def test_unmatched_constraint_does_not_return_partial_matches(index, filters):
    assert index.structured_search(**filters) == []


def test_matching_filters_intersect_case_insensitively(index):
    assert index.structured_search(
        persons=["ALICE"], location="par", entities=["COFFEE"]
    ) == [(0, 1.0)]


def test_values_within_one_filter_are_alternatives(index):
    assert index.structured_search(
        persons=["Alice", "Bob"], location="London", entities=["tea"]
    ) == [(1, 1.0)]
