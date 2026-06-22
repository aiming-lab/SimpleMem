"""Benchmark adapters — map heterogeneous benchmark data into a unified
`BenchmarkSample` format consumed by `EvolutionEngine`.

Each adapter provides:
  - load(...) -> list[BenchmarkSample]
  - scoring_fn(prediction, reference, qa_meta) -> float  (bounded [0,1])
  - answer_prompt(question, context, qa_meta) -> str     (format-specific)

The engine calls these via the `BenchmarkAdapter` protocol; adding a new
benchmark means adding a file here, not touching the core engine.
"""

from .base import (
    BenchmarkAdapter,
    BenchmarkSample,
    QuestionMeta,
    register_adapter,
    get_adapter,
)
from .locomo import LoCoMoAdapter
from .longmemeval import LongMemEvalAdapter
from .membench import MemBenchAdapter

# Registry — populated on import
register_adapter("locomo", LoCoMoAdapter)
register_adapter("longmemeval", LongMemEvalAdapter)
register_adapter("membench", MemBenchAdapter)

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkSample",
    "QuestionMeta",
    "LoCoMoAdapter",
    "LongMemEvalAdapter",
    "MemBenchAdapter",
    "get_adapter",
    "register_adapter",
]
