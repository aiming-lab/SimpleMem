import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(params=["simplemem.evolver", "evolvemem"])
def retriever_module(request, monkeypatch):
    # Exercise both the installed package and the standalone EvolveMem entry point.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "EvolveMem"))
    return importlib.import_module(f"{request.param}.multi_retriever")


class FixedEmbedder:
    def encode(self, texts, **kwargs):
        vectors = {
            "Alice prefers email.": [0.9, np.sqrt(1 - 0.9**2)],
            "Alice prefers phone calls.": [0.8, 0.6],
            "preferred contact method": [1.0, 0.0],
        }
        return np.array([vectors[text] for text in texts])


@pytest.mark.parametrize("fusion_mode", ["first_found", "rrf", "semantic_only", "weighted_sum"])
@pytest.mark.parametrize("half_life_days", [None, 7.0])
def test_time_decay_changes_which_memory_fits_in_context(
    retriever_module, fusion_mode, half_life_days
):
    memories = [
        {"content": "Alice prefers email.", "timestamp": "2026-01-01"},
        {"content": "Alice prefers phone calls.", "timestamp": "2026-01-31"},
    ]
    index = retriever_module.MultiViewIndex(memories, embedder=FixedEmbedder())
    config = retriever_module.RetrievalConfig(
        semantic_top_k=2,
        keyword_top_k=0,
        structured_top_k=0,
        max_context=1,
        fusion_mode=fusion_mode,
        time_decay_half_life_days=half_life_days,
        enable_entity_swap=False,
    )

    results = retriever_module.retrieve_multiview(
        index, "preferred contact method", config, reference_date="2026-01-31"
    )

    expected = memories[1 if half_life_days else 0]["content"]
    assert [result.content for result in results] == [expected]


def test_time_decay_preserves_ties_and_memories_without_dates(retriever_module):
    memories = [
        {"timestamp": "2026-01-01"},
        {"timestamp": "2026-01-31"},
        {},
    ]

    results = retriever_module._apply_time_decay(
        [(0, 1.0), (1, 0.8), (2, 0.8)], memories, 7.0, "2026-01-31"
    )

    assert [index for index, _ in results] == [1, 2, 0]
    assert results[0][1] == results[1][1] == 0.8
