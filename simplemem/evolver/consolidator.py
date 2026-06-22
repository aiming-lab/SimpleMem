from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from .models import MemoryType, MemoryUnit, utc_now_iso
from .store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Consolidation for duplicates, near-duplicates, stale summaries, and decay."""

    def __init__(
        self,
        store: MemoryStore,
        similarity_threshold: float = 0.80,
        decay_after_days: int = 30,
        decay_factor: float = 0.05,
        min_importance: float = 0.15,
        decay_mode: str = "linear",
    ):
        self.store = store
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self.decay_after_days = max(1, decay_after_days)
        self.decay_factor = max(0.0, min(0.5, decay_factor))
        self.min_importance = max(0.0, min(1.0, min_importance))
        self.decay_mode = decay_mode if decay_mode in ("linear", "exponential") else "linear"

    def consolidate(self, scope_id: str) -> dict:
        units = self.store.list_active(scope_id, limit=2000)
        now = utc_now_iso()
        superseded = 0

        # Keep only the newest working summary active.
        working = [u for u in units if u.memory_type == MemoryType.WORKING_SUMMARY]
        working.sort(key=lambda u: u.updated_at, reverse=True)
        for stale in working[1:]:
            self.store.supersede(stale.memory_id, working[0].memory_id, updated_at=now)
            superseded += 1

        # Exact duplicate content deduplication.
        seen: dict[tuple[str, str], str] = {}
        remaining: list[MemoryUnit] = []
        for unit in units:
            if unit.memory_type == MemoryType.WORKING_SUMMARY:
                continue
            key = (unit.memory_type.value, unit.content.strip())
            if key not in seen:
                seen[key] = unit.memory_id
                remaining.append(unit)
                continue
            self.store.supersede(unit.memory_id, seen[key], updated_at=now)
            superseded += 1

        # Near-duplicate merging via token overlap similarity.
        superseded += self._merge_near_duplicates(remaining, now)

        # Entity-based cross-type reinforcement.
        reinforced = self._reinforce_shared_entities(remaining)

        # Importance decay for old, unused memories.
        decayed = self._apply_importance_decay(units, now)

        result = {"superseded": superseded, "decayed": decayed, "reinforced": reinforced}
        if superseded or decayed or reinforced:
            logger.info(
                "[Consolidator] scope=%s superseded=%d decayed=%d reinforced=%d (pool=%d)",
                scope_id, superseded, decayed, reinforced, len(units),
            )
        return result

    def dry_run(self, scope_id: str) -> dict:
        """Preview what consolidation would do without applying changes.

        Returns counts and details of what would happen.
        """
        units = self.store.list_active(scope_id, limit=2000)
        now = utc_now_iso()

        # Stale working summaries.
        working = [u for u in units if u.memory_type == MemoryType.WORKING_SUMMARY]
        working.sort(key=lambda u: u.updated_at, reverse=True)
        stale_summaries = len(working) - 1 if len(working) > 1 else 0

        # Exact duplicates.
        seen: dict[tuple[str, str], str] = {}
        exact_dupes = 0
        remaining: list[MemoryUnit] = []
        for unit in units:
            if unit.memory_type == MemoryType.WORKING_SUMMARY:
                continue
            key = (unit.memory_type.value, unit.content.strip())
            if key in seen:
                exact_dupes += 1
            else:
                seen[key] = unit.memory_id
                remaining.append(unit)

        # Near-duplicates.
        near_dupes = self._count_near_duplicates(remaining)

        # Decay candidates.
        decay_candidates = self._count_decay_candidates(units)

        return {
            "stale_summaries": stale_summaries,
            "exact_duplicates": exact_dupes,
            "near_duplicates": near_dupes,
            "decay_candidates": decay_candidates,
            "total_actions": stale_summaries + exact_dupes + near_dupes + decay_candidates,
        }

    def _count_near_duplicates(self, units: list[MemoryUnit]) -> int:
        """Count near-duplicate pairs without merging."""
        if len(units) < 2 or self.similarity_threshold <= 0.0:
            return 0
        count = 0
        token_sets: dict[str, set[str]] = {}
        for u in units:
            token_sets[u.memory_id] = set(_tokenize(u.content.lower()))
        by_type: dict[str, list[MemoryUnit]] = {}
        for u in units:
            by_type.setdefault(u.memory_type.value, []).append(u)
        alive = set(u.memory_id for u in units)
        for group in by_type.values():
            for i in range(len(group)):
                if group[i].memory_id not in alive:
                    continue
                for j in range(i + 1, len(group)):
                    if group[j].memory_id not in alive:
                        continue
                    sim = _jaccard_similarity(
                        token_sets[group[i].memory_id],
                        token_sets[group[j].memory_id],
                    )
                    if sim >= self.similarity_threshold:
                        alive.discard(group[j].memory_id)
                        count += 1
        return count

    def _count_decay_candidates(self, units: list[MemoryUnit]) -> int:
        """Count how many memories would have importance decayed."""
        if self.decay_factor <= 0.0:
            return 0
        now = datetime.now(timezone.utc)
        count = 0
        for unit in units:
            if unit.memory_type == MemoryType.WORKING_SUMMARY:
                continue
            if unit.importance <= self.min_importance:
                continue
            reference = unit.last_accessed_at or unit.updated_at
            try:
                ref_dt = datetime.fromisoformat(reference.replace("Z", "+00:00"))
                if ref_dt.tzinfo is None:
                    ref_dt = ref_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            age_days = max((now - ref_dt).total_seconds() / 86400.0, 0.0)
            if age_days >= self.decay_after_days:
                count += 1
        return count

    def _merge_near_duplicates(self, units: list[MemoryUnit], now: str) -> int:
        if len(units) < 2 or self.similarity_threshold <= 0.0:
            return 0

        superseded = 0
        alive = set(u.memory_id for u in units)
        token_sets: dict[str, set[str]] = {}
        for u in units:
            token_sets[u.memory_id] = set(_tokenize(u.content.lower()))

        # Group by memory type to only merge within the same type.
        by_type: dict[str, list[MemoryUnit]] = {}
        for u in units:
            by_type.setdefault(u.memory_type.value, []).append(u)

        for group in by_type.values():
            for i in range(len(group)):
                if group[i].memory_id not in alive:
                    continue
                for j in range(i + 1, len(group)):
                    if group[j].memory_id not in alive:
                        continue
                    sim = _jaccard_similarity(
                        token_sets[group[i].memory_id],
                        token_sets[group[j].memory_id],
                    )
                    if sim >= self.similarity_threshold:
                        # Keep the one with higher importance; if equal keep the newer one.
                        keep, drop = group[i], group[j]
                        if drop.importance > keep.importance or (
                            drop.importance == keep.importance
                            and drop.updated_at > keep.updated_at
                        ):
                            keep, drop = drop, keep
                        self.store.supersede(drop.memory_id, keep.memory_id, updated_at=now)
                        alive.discard(drop.memory_id)
                        superseded += 1
        return superseded


    def _reinforce_shared_entities(self, units: list[MemoryUnit]) -> int:
        """Boost reinforcement score for memories that share entities with other memories.

        When multiple memories mention the same entity, it indicates that entity
        is important across different contexts. This cross-references without merging.
        """
        if len(units) < 2:
            return 0

        # Build entity → memory_id index.
        entity_index: dict[str, list[str]] = {}
        unit_map: dict[str, MemoryUnit] = {}
        for u in units:
            unit_map[u.memory_id] = u
            for ent in u.entities:
                ent_lower = ent.lower()
                if len(ent_lower) < 2:
                    continue
                entity_index.setdefault(ent_lower, []).append(u.memory_id)

        # Find memories that share entities with at least one other memory.
        reinforced = 0
        seen: set[str] = set()
        now = utc_now_iso()
        for ent, mem_ids in entity_index.items():
            if len(mem_ids) < 2:
                continue
            for mid in mem_ids:
                if mid in seen:
                    continue
                unit = unit_map[mid]
                # Small reinforcement boost: 0.05 per shared entity, capped at 0.3.
                boost = min(0.05, 0.3 - unit.reinforcement_score)
                if boost > 0.001:
                    new_score = round(unit.reinforcement_score + boost, 4)
                    self.store.update_reinforcement(mid, new_score, now)
                    seen.add(mid)
                    reinforced += 1

        return reinforced

    def _apply_importance_decay(self, units: list[MemoryUnit], now_iso: str) -> int:
        """Reduce importance of old memories that haven't been accessed recently."""
        if self.decay_factor <= 0.0:
            return 0

        now = datetime.now(timezone.utc)
        decayed = 0
        for unit in units:
            if unit.memory_type == MemoryType.WORKING_SUMMARY:
                continue
            if unit.importance <= self.min_importance:
                continue

            # Use last_accessed_at if available, otherwise use updated_at.
            reference = unit.last_accessed_at or unit.updated_at
            try:
                ref_dt = datetime.fromisoformat(reference.replace("Z", "+00:00"))
                if ref_dt.tzinfo is None:
                    ref_dt = ref_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            age_days = max((now - ref_dt).total_seconds() / 86400.0, 0.0)
            if age_days < self.decay_after_days:
                continue

            # Apply decay based on configured mode.
            periods = (age_days - self.decay_after_days) / float(self.decay_after_days)
            if self.decay_mode == "exponential":
                # Exponential: importance * e^(-decay_factor * periods)
                new_importance = max(
                    self.min_importance,
                    unit.importance * math.exp(-self.decay_factor * min(periods, 10.0)),
                )
            else:
                # Linear: importance * (1 - decay_factor * periods)
                new_importance = max(
                    self.min_importance,
                    unit.importance * (1.0 - self.decay_factor * min(periods, 5.0)),
                )
            if round(new_importance, 4) < round(unit.importance, 4):
                self.store.update_importance(unit.memory_id, round(new_importance, 4), now_iso)
                decayed += 1

        return decayed


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / float(union) if union > 0 else 0.0


def _tokenize(text: str) -> list[str]:
    token: list[str] = []
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
            continue
        if token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out
