"""
EvolveMem — Self-Evolving Memory Architecture for LLM Agents.

A four-layer memory system with typed knowledge representation, adaptive
retrieval policy, replay-based offline evaluation, and promotion-gated
self-evolution — all backed by SQLite + FTS5.
"""

from .candidate import generate_policy_candidates
from .consolidator import MemoryConsolidator
from .diagnosis import DiagnosisReport, MemoryDiagnostics, QAResult
from .evolution import EvolutionConfig, EvolutionEngine, EvolutionResult
from .extractor import ExtractionConfig, MemoryExtractor
from .manager import MemoryManager
from .models import MemoryQuery, MemoryStatus, MemoryType, MemoryUnit
from .multi_retriever import (
    MultiViewIndex,
    RetrievalConfig,
    RetrievedMemory,
    format_context,
    retrieve_multiview,
)
from .promotion import MemoryPromotionCriteria, should_promote
from .replay import (
    MemoryReplayEvaluator,
    MemoryReplaySample,
    load_replay_samples,
    run_policy_candidate_replay,
)
from .scope import derive_memory_scope
from .self_upgrade import MemorySelfUpgradeOrchestrator
from .store import MemoryStore
from .telemetry import MemoryTelemetryStore
from .upgrade_worker import MemoryUpgradeWorker

__all__ = [
    # Core
    "MemoryManager",
    "MemoryStore",
    "MemoryConsolidator",
    "MemoryQuery",
    "MemoryStatus",
    "MemoryType",
    "MemoryUnit",
    "MemoryTelemetryStore",
    # Self-Evolution (NEW)
    "EvolutionEngine",
    "EvolutionConfig",
    "EvolutionResult",
    "MemoryExtractor",
    "ExtractionConfig",
    "MemoryDiagnostics",
    "DiagnosisReport",
    "QAResult",
    # Multi-View Retrieval (NEW)
    "MultiViewIndex",
    "RetrievalConfig",
    "RetrievedMemory",
    "retrieve_multiview",
    "format_context",
    # Policy Evolution
    "MemoryPromotionCriteria",
    "should_promote",
    "MemoryReplayEvaluator",
    "MemoryReplaySample",
    "load_replay_samples",
    "run_policy_candidate_replay",
    "MemorySelfUpgradeOrchestrator",
    "MemoryUpgradeWorker",
    "derive_memory_scope",
    "generate_policy_candidates",
]
