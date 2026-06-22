"""
Memory routing module for benchmark-safe self-evolution.

Provides intelligent routing between Episodic (original MAU) and Semantic
(derived/distilled) memory stores.
"""

from simplemem.multimodal.routing.router import MemoryRouter, RouteResult
from simplemem.multimodal.routing.policy import (
    BenchmarkSafeGuard,
    RetentionPolicy,
    CircuitBreaker,
)
from simplemem.multimodal.routing.features import compute_route_features

__all__ = [
    "MemoryRouter",
    "RouteResult",
    "BenchmarkSafeGuard",
    "RetentionPolicy",
    "CircuitBreaker",
    "compute_route_features",
]
