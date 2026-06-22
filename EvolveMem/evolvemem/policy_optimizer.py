from __future__ import annotations

from .metrics import summarize_memory_store
from .policy_store import MemoryPolicyState
from .store import MemoryStore
from .telemetry import MemoryTelemetryStore


class MemoryPolicyOptimizer:
    """Bounded policy tuner that adapts retrieval parameters based on store state
    and telemetry signals.

    Uses memory volume, type distribution, density, and retrieval telemetry to
    propose safe incremental policy changes. All changes stay within hard bounds.
    """

    def __init__(
        self,
        store: MemoryStore,
        telemetry_store: MemoryTelemetryStore | None = None,
    ):
        self.store = store
        self.telemetry_store = telemetry_store

    def propose(self, scope_id: str, current: MemoryPolicyState) -> MemoryPolicyState:
        stats = summarize_memory_store(self.store, scope_id)
        proposed = MemoryPolicyState(**current.__dict__)
        notes = list(current.notes)

        active = int(stats.get("active", 0))
        dominant = str(stats.get("dominant_type", "") or "")
        density = float(stats.get("memory_density", 0.0) or 0.0)
        active_by_type = stats.get("active_by_type", {})

        # Low-volume safety: don't tune weights before having enough data.
        if active < 5:
            proposed.notes = _truncate_notes(notes)
            return proposed

        # Volume-based mode switching.
        if active >= 25:
            proposed.retrieval_mode = "hybrid"
            proposed.max_injected_units = max(4, min(8, current.max_injected_units))
            notes.append("Switched to hybrid retrieval because memory volume increased.")

        if active >= 80:
            proposed.max_injected_units = min(10, max(current.max_injected_units, 8))
            proposed.max_injected_tokens = min(1200, max(current.max_injected_tokens, 900))
            notes.append("Raised injection budget because active memory volume is high.")

        # Type-distribution-based weight tuning.
        if dominant == "working_summary" and density > 0.6:
            proposed.recency_weight = min(0.6, current.recency_weight + 0.1)
            notes.append("Raised recency weight because working summaries dominate the active memory pool.")

        if dominant == "preference":
            proposed.metadata_weight = min(0.8, current.metadata_weight + 0.05)
            notes.append("Raised metadata weight because preference memories dominate and benefit from tighter matching.")

        # Diversity-aware tuning.
        semantic_count = int(active_by_type.get("semantic", 0))
        project_count = int(active_by_type.get("project_state", 0))
        factual_fraction = (semantic_count + project_count) / max(active, 1)
        if factual_fraction >= 0.4 and active >= 10:
            proposed.importance_weight = min(0.8, max(current.importance_weight, 0.55))
            if proposed.importance_weight > current.importance_weight:
                notes.append("Raised importance weight because factual memories are a large share of the pool.")

        # Episodic dominance tuning.
        episodic_count = int(active_by_type.get("episodic", 0))
        episodic_fraction = episodic_count / max(active, 1)
        if episodic_fraction >= 0.5 and active >= 15:
            proposed.keyword_weight = min(1.5, max(current.keyword_weight, 1.1))
            if proposed.keyword_weight > current.keyword_weight:
                notes.append("Raised keyword weight because episodic memories dominate and need stronger term matching.")

        # Telemetry-driven tuning: adjust injection budget based on recent retrieval patterns.
        telemetry_proposals = self._propose_from_telemetry(current, scope_id, active_by_type)
        if telemetry_proposals:
            for key, value, reason in telemetry_proposals:
                if key == "max_injected_units":
                    proposed.max_injected_units = value
                elif key == "max_injected_tokens":
                    proposed.max_injected_tokens = value
                notes.append(reason)

        proposed.notes = _truncate_notes(notes)
        return proposed

    def _propose_from_telemetry(
        self, current: MemoryPolicyState, scope_id: str, active_by_type: dict | None = None
    ) -> list[tuple[str, int, str]]:
        """Analyze recent retrieval telemetry to propose injection budget changes."""
        if self.telemetry_store is None:
            return []

        events = self.telemetry_store.read_recent(limit=50)
        retrieval_events = [
            e for e in events
            if e.get("event_type") == "memory_retrieval"
            and e.get("payload", {}).get("scope_id") == scope_id
        ]
        if len(retrieval_events) < 5:
            return []

        proposals: list[tuple[str, int, str]] = []

        # Check if retrieval consistently hits the unit limit.
        recent = retrieval_events[-10:]
        avg_retrieved = sum(
            e["payload"].get("retrieved_count", 0) for e in recent
        ) / float(len(recent))
        if avg_retrieved >= current.max_injected_units * 0.9:
            new_units = min(10, current.max_injected_units + 1)
            if new_units > current.max_injected_units:
                proposals.append((
                    "max_injected_units",
                    new_units,
                    f"Raised max units from {current.max_injected_units} to {new_units} because retrieval consistently saturates the limit.",
                ))

        # Check if injected tokens consistently approach the budget.
        avg_tokens = sum(
            e["payload"].get("injected_tokens", 0) for e in recent
        ) / float(len(recent))
        if avg_tokens >= current.max_injected_tokens * 0.85:
            new_tokens = min(1400, current.max_injected_tokens + 100)
            if new_tokens > current.max_injected_tokens:
                proposals.append((
                    "max_injected_tokens",
                    new_tokens,
                    f"Raised max tokens from {current.max_injected_tokens} to {new_tokens} because token budget is consistently near capacity.",
                ))

        # Check if retrieval is consistently returning zero results.
        zero_count = sum(
            1 for e in recent if e["payload"].get("retrieved_count", 0) == 0
        )
        if zero_count >= len(recent) * 0.6:
            new_units = max(4, current.max_injected_units - 1)
            if new_units < current.max_injected_units:
                proposals.append((
                    "max_injected_units",
                    new_units,
                    f"Reduced max units from {current.max_injected_units} to {new_units} because most retrievals return nothing.",
                ))

        # Check for type skew: if certain types exist in the pool but never appear
        # in retrieval results, it may indicate a scoring or weight issue.
        if active_by_type and len(retrieval_events) >= 10:
            retrieved_types: set[str] = set()
            for e in retrieval_events[-20:]:
                for t in e["payload"].get("types_retrieved", []):
                    retrieved_types.add(t)
            pool_types = set(active_by_type.keys())
            never_retrieved = pool_types - retrieved_types - {"working_summary"}
            if never_retrieved and len(pool_types) >= 3:
                # Bump metadata weight slightly to help underrepresented types surface.
                new_meta = min(0.8, current.metadata_weight + 0.05)
                if new_meta > current.metadata_weight:
                    proposals.append((
                        "max_injected_units",
                        min(10, current.max_injected_units + 1),
                        f"Type skew detected: {', '.join(sorted(never_retrieved))} never retrieved. Raised units to improve diversity.",
                    ))

        return proposals


def _truncate_notes(notes: list[str], limit: int = 12) -> list[str]:
    if len(notes) <= limit:
        return notes
    return notes[-limit:]
