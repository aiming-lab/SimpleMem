"""
Memory Consolidator for Omni-Memory.

Implements adaptive memory consolidation inspired by biological memory systems:
- Importance-based retention
- Temporal decay and forgetting
- Cross-modal memory strengthening
- Sleep-like consolidation cycles
"""

import logging
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationMetrics:
    """Metrics tracked during consolidation."""

    total_memories: int = 0
    retained_memories: int = 0
    archived_memories: int = 0
    strengthened_memories: int = 0
    merged_memories: int = 0
    consolidation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_memories": self.total_memories,
            "retained_memories": self.retained_memories,
            "archived_memories": self.archived_memories,
            "strengthened_memories": self.strengthened_memories,
            "merged_memories": self.merged_memories,
            "consolidation_time_ms": self.consolidation_time_ms,
        }


@dataclass
class MemoryImportance:
    """Importance scoring for a memory."""

    mau_id: str
    base_score: float = 1.0

    # Factors
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    cross_modal_links: int = 0
    entity_frequency: int = 0

    # Computed importance
    _cached_importance: Optional[float] = None
    _cache_time: float = 0.0

    def compute_importance(
        self,
        recency_weight: float = 0.3,
        access_weight: float = 0.3,
        cross_modal_weight: float = 0.2,
        entity_weight: float = 0.2,
        recency_decay: float = 0.95,
    ) -> float:
        """
        Compute importance score for the memory.

        Factors:
        - Recency: Recent memories are more important
        - Access frequency: Frequently accessed memories matter
        - Cross-modal links: Multi-modal memories are richer
        - Entity connections: Memories about known entities
        """
        now = time.time()

        # Cache for 1 minute
        if self._cached_importance is not None and (now - self._cache_time) < 60:
            return self._cached_importance

        # Recency score (exponential decay)
        days_since_creation = (now - self.creation_time) / 86400
        recency_score = recency_decay**days_since_creation

        days_since_access = (now - self.last_access_time) / 86400
        access_recency = recency_decay**days_since_access

        # Access frequency score (logarithmic)
        import math

        access_score = math.log(1 + self.access_count) / 5  # Normalize
        access_score = min(access_score, 1.0)

        # Cross-modal bonus
        cross_modal_score = min(self.cross_modal_links / 3, 1.0)

        # Entity frequency score
        entity_score = min(self.entity_frequency / 10, 1.0)

        # Weighted combination
        importance = (
            self.base_score * 0.2
            + recency_score * recency_weight * 0.5
            + access_recency * recency_weight * 0.5
            + access_score * access_weight
            + cross_modal_score * cross_modal_weight
            + entity_score * entity_weight
        )

        self._cached_importance = importance
        self._cache_time = now

        return importance

    def record_access(self) -> None:
        """Record an access to this memory."""
        self.access_count += 1
        self.last_access_time = time.time()
        self._cached_importance = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mau_id": self.mau_id,
            "base_score": self.base_score,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "creation_time": self.creation_time,
            "cross_modal_links": self.cross_modal_links,
            "entity_frequency": self.entity_frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryImportance":
        return cls(**data)


class MemoryConsolidator:
    """
    Memory Consolidator - Adaptive memory management.

    Key Features:
    1. Importance-Based Retention: Keep valuable memories
    2. Adaptive Forgetting: Remove low-value memories
    3. Cross-Modal Strengthening: Boost multimodal memories
    4. Periodic Consolidation: Sleep-like memory processing

    This implements biologically-inspired memory consolidation:
    - Short-term memories with high importance → Long-term retention
    - Infrequently accessed, old memories → Graceful forgetting
    - Cross-modal associations → Strengthened connections
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        self.storage_path = Path(storage_path or "./omni_memory_data/consolidation")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.config = config

        # Importance tracking
        self._importance_map: Dict[str, MemoryImportance] = {}
        self._importance_file = self.storage_path / "importance.jsonl"

        # Consolidation thresholds
        self.forgetting_threshold = 0.1  # Below this, consider forgetting
        self.strengthening_threshold = 0.8  # Above this, strengthen
        self.max_memory_age_days = 365  # Maximum age before forced review

        # Consolidation schedule
        self.consolidation_interval_hours = 24.0
        self._last_consolidation: float = 0

        # Load existing importance data
        self._load_importance()

    def register_memory(
        self,
        mau_id: str,
        base_score: float = 1.0,
        cross_modal_links: int = 0,
        entity_frequency: int = 0,
    ) -> None:
        """
        Register a new memory for importance tracking.

        Args:
            mau_id: Memory ID
            base_score: Initial importance
            cross_modal_links: Number of cross-modal connections
            entity_frequency: Entity mention frequency
        """
        if mau_id in self._importance_map:
            # Update existing
            existing = self._importance_map[mau_id]
            existing.cross_modal_links = max(
                existing.cross_modal_links, cross_modal_links
            )
            existing.entity_frequency = max(existing.entity_frequency, entity_frequency)
        else:
            self._importance_map[mau_id] = MemoryImportance(
                mau_id=mau_id,
                base_score=base_score,
                cross_modal_links=cross_modal_links,
                entity_frequency=entity_frequency,
            )

    def record_access(self, mau_id: str) -> None:
        """Record that a memory was accessed."""
        if mau_id in self._importance_map:
            self._importance_map[mau_id].record_access()
        else:
            # Auto-register with default values
            self.register_memory(mau_id)
            self._importance_map[mau_id].record_access()

    def get_importance(self, mau_id: str) -> float:
        """Get current importance score for a memory."""
        if mau_id in self._importance_map:
            return self._importance_map[mau_id].compute_importance()
        return 0.5  # Default for unknown

    def get_importance_batch(self, mau_ids: List[str]) -> Dict[str, float]:
        """Get importance scores for multiple memories."""
        return {mau_id: self.get_importance(mau_id) for mau_id in mau_ids}

    def should_consolidate(self) -> bool:
        """Check if consolidation should run."""
        hours_since_last = (time.time() - self._last_consolidation) / 3600
        return hours_since_last >= self.consolidation_interval_hours

    def consolidate(
        self, mau_store=None, knowledge_graph=None, force=False, run_id=None
    ) -> ConsolidationMetrics:
        """
        Run memory consolidation.

        This is the "sleep" cycle for the memory system:
        1. Identify low-importance memories for forgetting
        2. Strengthen high-importance memories
        3. Update importance scores
        4. Clean up stale data

        Args:
            mau_store: Optional MAU store for actual deletion
            force: Force consolidation even if schedule not met

        Returns:
            ConsolidationMetrics with results
        """
        if not force and not self.should_consolidate():
            return ConsolidationMetrics()

        if run_id is None:
            run_id = f"consolidation_{int(time.time())}"

        start_time = time.time()
        metrics = ConsolidationMetrics()

        metrics.total_memories = len(self._importance_map)

        to_forget: List[str] = []
        to_strengthen: List[str] = []

        for mau_id, importance_data in self._importance_map.items():
            score = importance_data.compute_importance()

            if score < self.forgetting_threshold:
                to_forget.append(mau_id)
            elif score > self.strengthening_threshold:
                to_strengthen.append(mau_id)

        # Process forgetting -> archive (non-destructive)
        for mau_id in to_forget:
            # Keep in importance tracking for future reference (do NOT delete)
            metrics.archived_memories += 1

            # Archive in MAU store (soft delete, never physical delete)
            if mau_store:
                try:
                    mau_store.archive(
                        mau_id,
                        reason="consolidation",
                        run_id=run_id,
                    )
                    logger.info(f"Archived MAU {mau_id} (consolidation run={run_id})")
                except Exception as e:
                    logger.warning(f"Failed to archive MAU {mau_id}: {e}")

        # Process strengthening (boost base score)
        for mau_id in to_strengthen:
            self._importance_map[mau_id].base_score = min(
                self._importance_map[mau_id].base_score * 1.1,
                2.0,  # Cap at 2x base
            )
            metrics.strengthened_memories += 1

        metrics.retained_memories = len(self._importance_map)
        metrics.consolidation_time_ms = (time.time() - start_time) * 1000

        # Update last consolidation time
        self._last_consolidation = time.time()

        # Save importance data
        self._save_importance()

        logger.info(
            f"Consolidation complete: {metrics.retained_memories} retained, "
            f"{metrics.archived_memories} archived, {metrics.strengthened_memories} strengthened"
        )

        return metrics

    def get_candidates_for_forgetting(
        self,
        limit: int = 100,
    ) -> List[str]:
        """Get memories that are candidates for forgetting."""
        candidates = []

        for mau_id, importance_data in self._importance_map.items():
            if importance_data.compute_importance() < self.forgetting_threshold:
                candidates.append(mau_id)

        return candidates[:limit]

    def get_high_importance_memories(
        self,
        limit: int = 100,
    ) -> List[str]:
        """Get highest importance memories."""
        scored = [
            (mau_id, data.compute_importance())
            for mau_id, data in self._importance_map.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [mau_id for mau_id, _ in scored[:limit]]

    def boost_cross_modal(
        self,
        mau_ids: List[str],
        boost_factor: float = 1.2,
    ) -> None:
        """
        Boost importance for cross-modal memory groups.

        When memories from different modalities are linked,
        boost their importance as they form richer representations.
        """
        for mau_id in mau_ids:
            if mau_id in self._importance_map:
                data = self._importance_map[mau_id]
                data.cross_modal_links += len(mau_ids) - 1
                data.base_score *= boost_factor

    def _save_importance(self) -> None:
        """Save importance data to disk."""
        with open(self._importance_file, "w", encoding="utf-8") as f:
            for data in self._importance_map.values():
                f.write(json.dumps(data.to_dict(), ensure_ascii=False) + "\n")

    def _load_importance(self) -> None:
        """Load importance data from disk."""
        if self._importance_file.exists():
            with open(self._importance_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        importance = MemoryImportance.from_dict(data)
                        self._importance_map[importance.mau_id] = importance

            logger.info(
                f"Loaded importance data for {len(self._importance_map)} memories"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get consolidator statistics."""
        if not self._importance_map:
            return {
                "total_tracked": 0,
                "avg_importance": 0.0,
                "low_importance_count": 0,
                "high_importance_count": 0,
            }

        scores = [d.compute_importance() for d in self._importance_map.values()]

        return {
            "total_tracked": len(self._importance_map),
            "avg_importance": sum(scores) / len(scores),
            "low_importance_count": sum(
                1 for s in scores if s < self.forgetting_threshold
            ),
            "high_importance_count": sum(
                1 for s in scores if s > self.strengthening_threshold
            ),
            "last_consolidation": self._last_consolidation,
            "should_consolidate": self.should_consolidate(),
        }
