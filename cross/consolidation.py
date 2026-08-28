# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
# pyright: reportDeprecated=false
# pyright: reportUnusedImport=false
"""
Memory consolidation worker for SimpleMem-Cross.

Periodically maintains memory quality by performing three operations:
  1. Decay  — reduce importance of old entries over time
  2. Merge  — combine near-duplicate entries with very high semantic similarity
  3. Prune  — soft-delete entries whose importance has fallen below a threshold

All deletions are soft: entries are marked as superseded rather than removed.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

try:
    from cross.storage_iris import CrossSessionVectorStore
except ImportError:
    CrossSessionVectorStore = None  # type: ignore[assignment,misc]
from cross.storage_sqlite import SQLiteStorage
from cross.types import ConsolidationRun

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy & result models
# ---------------------------------------------------------------------------


class ConsolidationPolicy(BaseModel):
    """Configurable knobs for a single consolidation pass."""

    max_age_days: int = 90
    """Entries older than this (based on valid_from) receive importance decay."""

    decay_factor: float = 0.9
    """Multiplier applied to importance for each decay period elapsed."""

    merge_similarity_threshold: float = 0.95
    """Cosine similarity above which two entries are considered near-duplicates."""

    min_importance: float = 0.05
    """Entries below this importance after decay are pruned (soft-deleted)."""

    max_entries_per_run: int = 1000
    """Maximum number of entries processed in one consolidation pass."""


class ConsolidationResult(BaseModel):
    """Outcome counters for a single consolidation pass."""

    decayed_count: int = 0
    merged_count: int = 0
    pruned_count: int = 0
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 when either vector has zero magnitude.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _parse_valid_from(valid_from: object) -> Optional[datetime]:
    """Best-effort parse of the valid_from field into a timezone-aware datetime."""
    if isinstance(valid_from, datetime):
        if valid_from.tzinfo is None:
            return valid_from.replace(tzinfo=timezone.utc)
        return valid_from
    if isinstance(valid_from, str) and valid_from:
        try:
            dt = datetime.fromisoformat(valid_from)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class ConsolidationWorker:
    """Executes a single consolidation pass for a given tenant.

    Instantiate with the shared storage backends and an optional policy
    override, then call :meth:`run` to execute decay / merge / prune.
    """

    def __init__(
        self,
        sqlite_storage: SQLiteStorage,
        vector_store: CrossSessionVectorStore,
        policy: Optional[ConsolidationPolicy] = None,
    ) -> None:
        self.sqlite_storage = sqlite_storage
        self.vector_store = vector_store
        self.policy = policy or ConsolidationPolicy()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, tenant_id: str) -> ConsolidationResult:
        """Execute one full consolidation pass for *tenant_id*.

        Steps:
          1. Fetch all active (non-superseded) entries for the tenant.
          2. Decay importance of entries older than ``max_age_days``.
          3. Merge entries with cosine similarity above the threshold.
          4. Prune entries whose importance has dropped below ``min_importance``.
          5. Record the run in SQLite.

        Returns a :class:`ConsolidationResult` with counters.
        """
        t0 = time.time()
        logger.info(
            "Starting consolidation for tenant=%s with policy=%s",
            tenant_id,
            self.policy.model_dump(),
        )

        try:
            # Fetch entries and filter to active (non-superseded) ones only
            all_entries = self.vector_store.get_all_entries(tenant_id=tenant_id)
            entries = [e for e in all_entries if not e.superseded_by]
            entries = entries[: self.policy.max_entries_per_run]
            logger.info(
                "Fetched %d active entries (out of %d total) for tenant=%s",
                len(entries),
                len(all_entries),
                tenant_id,
            )

            decayed = self._decay_old_entries(entries, tenant_id)
            merged = self._merge_similar_entries(entries, tenant_id)
            pruned = self._prune_low_importance(entries, tenant_id)

        except Exception:
            logger.exception("Consolidation failed for tenant=%s", tenant_id)
            decayed = merged = pruned = 0

        duration = time.time() - t0

        result = ConsolidationResult(
            decayed_count=decayed,
            merged_count=merged,
            pruned_count=pruned,
            duration_seconds=round(duration, 3),
        )

        # Persist run metadata
        try:
            self.sqlite_storage.record_consolidation_run(
                tenant_id=tenant_id,
                policy_json=self.policy.model_dump(),  # type: ignore[arg-type]
                stats_json=result.model_dump(),  # type: ignore[arg-type]
            )
        except Exception:
            logger.exception(
                "Failed to record consolidation run for tenant=%s", tenant_id
            )

        logger.info(
            "Consolidation complete for tenant=%s: %s (%.3fs)",
            tenant_id,
            result.model_dump(),
            duration,
        )
        return result

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def _decay_old_entries(self, entries: list, tenant_id: str) -> int:
        """Apply importance decay to entries older than ``max_age_days``.

        For each qualifying entry the new importance is:
            importance *= decay_factor
        and the vector store is updated in-place.

        Returns the number of entries whose importance was decayed.
        """
        now = datetime.now(timezone.utc)
        decayed = 0

        for entry in entries:
            valid_from_dt = _parse_valid_from(entry.valid_from)
            if valid_from_dt is None:
                continue

            age_days = (now - valid_from_dt).total_seconds() / 86400.0
            if age_days <= self.policy.max_age_days:
                continue

            new_importance = entry.importance * self.policy.decay_factor
            try:
                self.vector_store.update_importance(entry.entry_id, new_importance)
                # Reflect the change on the in-memory object so subsequent
                # steps (prune) see the updated value.
                entry.importance = new_importance  # type: ignore[misc]
                decayed += 1
                logger.debug(
                    "Decayed entry %s: %.4f -> %.4f (age %.1f days)",
                    entry.entry_id,
                    entry.importance,
                    new_importance,
                    age_days,
                )
            except Exception:
                logger.exception("Failed to decay entry %s", entry.entry_id)

        logger.info("Decay phase: %d entries decayed for tenant=%s", decayed, tenant_id)
        return decayed

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_similar_entries(self, entries: list, tenant_id: str) -> int:
        """Merge near-duplicate entries based on cosine similarity.

        For each pair of entries whose lossless_restatement embeddings
        exceed ``merge_similarity_threshold``, the entry with the lower
        importance is marked as superseded by the one with higher importance.

        Returns the number of entries marked as superseded (merged away).
        """
        if len(entries) < 2:
            return 0

        # Compute embeddings for all entries in a single batch
        texts = [e.lossless_restatement for e in entries]
        try:
            vectors_np = self.vector_store.embedding_model.encode_documents(texts)
            vectors: list[list[float]] = [v.tolist() for v in vectors_np]
        except Exception:
            logger.exception(
                "Failed to encode documents for merge phase (tenant=%s)",
                tenant_id,
            )
            return 0

        # Track which entries have already been merged away
        merged_ids: set[str] = set()
        merged_count = 0

        for i in range(len(entries)):
            if entries[i].entry_id in merged_ids:
                continue
            for j in range(i + 1, len(entries)):
                if entries[j].entry_id in merged_ids:
                    continue

                sim = _cosine_similarity(vectors[i], vectors[j])
                if sim < self.policy.merge_similarity_threshold:
                    continue

                # Determine winner (higher importance) and loser
                if entries[i].importance >= entries[j].importance:
                    winner, loser = entries[i], entries[j]
                else:
                    winner, loser = entries[j], entries[i]

                try:
                    self.vector_store.mark_superseded(loser.entry_id, winner.entry_id)
                    merged_ids.add(loser.entry_id)
                    merged_count += 1
                    logger.debug(
                        "Merged entry %s into %s (sim=%.4f)",
                        loser.entry_id,
                        winner.entry_id,
                        sim,
                    )
                except Exception:
                    logger.exception(
                        "Failed to merge entry %s into %s",
                        loser.entry_id,
                        winner.entry_id,
                    )

        logger.info(
            "Merge phase: %d entries merged for tenant=%s",
            merged_count,
            tenant_id,
        )
        return merged_count

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------

    def _prune_low_importance(self, entries: list, tenant_id: str) -> int:
        """Soft-delete entries whose importance is below ``min_importance``.

        Uses :meth:`mark_superseded` with ``new_entry_id`` set to
        ``"__pruned__"`` to indicate the entry was removed by consolidation
        rather than replaced by another entry.

        Returns the number of entries pruned.
        """
        pruned = 0

        for entry in entries:
            # Skip entries already merged away in this run
            if entry.superseded_by:
                continue

            if entry.importance >= self.policy.min_importance:
                continue

            try:
                self.vector_store.mark_superseded(entry.entry_id, "__pruned__")
                pruned += 1
                logger.debug(
                    "Pruned entry %s (importance=%.4f)",
                    entry.entry_id,
                    entry.importance,
                )
            except Exception:
                logger.exception("Failed to prune entry %s", entry.entry_id)

        logger.info(
            "Prune phase: %d entries pruned for tenant=%s",
            pruned,
            tenant_id,
        )
        return pruned


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_consolidation(
    sqlite_storage: SQLiteStorage,
    vector_store: CrossSessionVectorStore,
    tenant_id: str,
    policy: Optional[ConsolidationPolicy] = None,
) -> ConsolidationResult:
    """Run a single consolidation pass — convenience wrapper.

    Creates a :class:`ConsolidationWorker` with the given backends and
    optional *policy*, then executes :meth:`ConsolidationWorker.run`.

    Args:
        sqlite_storage: SQLite backend for recording consolidation metadata.
        vector_store: IRIS vector store with cross-session entries.
        tenant_id: Tenant whose entries should be consolidated.
        policy: Optional override for the default :class:`ConsolidationPolicy`.

    Returns:
        A :class:`ConsolidationResult` summarising what was done.
    """
    worker = ConsolidationWorker(
        sqlite_storage=sqlite_storage,
        vector_store=vector_store,
        policy=policy,
    )
    return worker.run(tenant_id)
