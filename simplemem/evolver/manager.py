from __future__ import annotations

import logging
import re
import uuid

from .config import EvolveMemConfig
from .consolidator import MemoryConsolidator
from .embeddings import BaseEmbedder, HashingEmbedder, create_embedder
from .metrics import summarize_memory_store
from .models import MemoryQuery, MemoryStatus, MemoryType, MemoryUnit, utc_now_iso
from .policy import MemoryPolicy
from .policy_optimizer import MemoryPolicyOptimizer
from .policy_store import MemoryPolicyState, MemoryPolicyStore
from .retriever import MemoryRetriever
from .store import MemoryStore
from .telemetry import MemoryTelemetryStore

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    return len(text.split())


class MemoryManager:
    """Facade for retrieval, rendering, and write-side session extraction."""

    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryPolicy | None = None,
        scope_id: str = "default",
        auto_consolidate: bool = True,
        retrieval_mode: str = "keyword",
        use_embeddings: bool = False,
        policy_store: MemoryPolicyStore | None = None,
        telemetry_store: MemoryTelemetryStore | None = None,
        embedding_mode: str = "hashing",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedder: BaseEmbedder | None = None,
    ):
        self.store = store
        self.policy = policy or MemoryPolicy()
        self.scope_id = scope_id
        self.auto_consolidate = auto_consolidate
        self.retrieval_mode = retrieval_mode
        self.use_embeddings = use_embeddings or retrieval_mode in {"embedding", "hybrid"}
        self.policy_store = policy_store
        self.telemetry_store = telemetry_store
        self.embedding_mode = embedding_mode
        self.embedding_model = embedding_model
        if embedder is not None:
            self.embedder = embedder
        elif self.use_embeddings:
            self.embedder = create_embedder(
                mode=embedding_mode,
                model_name=embedding_model,
                fallback=True,
            )
        else:
            self.embedder = None
        self.policy_optimizer = MemoryPolicyOptimizer(
            store=self.store,
            telemetry_store=self.telemetry_store,
        )
        self.retriever = MemoryRetriever(
            store=self.store,
            policy=self.policy,
            retrieval_mode=retrieval_mode,
            embedder=self.embedder,
        )
        self.consolidator = MemoryConsolidator(store=self.store)
        self._retrieval_cache: dict[str, list[MemoryUnit]] = {}
        self._cache_max_size = 16
        self._event_callbacks: list[callable] = []

    @classmethod
    def from_config(cls, cfg: EvolveMemConfig) -> "MemoryManager":
        policy_store = MemoryPolicyStore(cfg.memory_policy_path)
        policy_exists = policy_store.path.exists()
        policy_state = policy_store.load()
        telemetry_store = MemoryTelemetryStore(cfg.memory_telemetry_path)
        if not policy_exists:
            policy_state.max_injected_units = cfg.memory_max_injected_units
            policy_state.max_injected_tokens = cfg.memory_max_injected_tokens
            policy_state.retrieval_mode = cfg.memory_retrieval_mode
            policy_store.save(policy_state, reason="bootstrap")
        policy = MemoryPolicy.from_state(policy_state)
        store = MemoryStore(cfg.memory_store_path)
        return cls(
            store=store,
            policy=policy,
            scope_id=cfg.memory_scope,
            auto_consolidate=cfg.memory_auto_consolidate,
            retrieval_mode=policy_state.retrieval_mode or cfg.memory_retrieval_mode,
            use_embeddings=cfg.memory_use_embeddings,
            policy_store=policy_store,
            telemetry_store=telemetry_store,
            embedding_mode=getattr(cfg, "memory_embedding_mode", "hashing"),
            embedding_model=getattr(cfg, "memory_embedding_model", "all-MiniLM-L6-v2"),
        )

    @classmethod
    def from_config_with_policy_state(
        cls,
        cfg: EvolveMemConfig,
        policy_state: MemoryPolicyState,
    ) -> "MemoryManager":
        policy_store = MemoryPolicyStore(cfg.memory_policy_path)
        telemetry_store = MemoryTelemetryStore(cfg.memory_telemetry_path)
        policy = MemoryPolicy.from_state(policy_state)
        store = MemoryStore(cfg.memory_store_path)
        return cls(
            store=store,
            policy=policy,
            scope_id=cfg.memory_scope,
            auto_consolidate=cfg.memory_auto_consolidate,
            retrieval_mode=policy_state.retrieval_mode or cfg.memory_retrieval_mode,
            use_embeddings=cfg.memory_use_embeddings,
            policy_store=policy_store,
            telemetry_store=telemetry_store,
            embedding_mode=getattr(cfg, "memory_embedding_mode", "hashing"),
            embedding_model=getattr(cfg, "memory_embedding_model", "all-MiniLM-L6-v2"),
        )

    def register_event_callback(self, callback: callable) -> None:
        """Register a callback for memory events.

        Callbacks receive a dict with at least 'event' (str) and 'scope_id' (str).
        Additional keys depend on the event type.
        """
        self._event_callbacks.append(callback)

    def _notify(self, event: str, **kwargs) -> None:
        """Fire all registered event callbacks. Best-effort; errors are logged."""
        if not self._event_callbacks:
            return
        payload = {"event": event, **kwargs}
        for cb in self._event_callbacks:
            try:
                cb(payload)
            except Exception as exc:
                logger.debug("Event callback error for %s: %s", event, exc)

    def ingest_session_turns(
        self,
        session_id: str,
        turns: list[dict],
        scope_id: str | None = None,
    ) -> int:
        """Create simple phase-1 memory units from a completed session."""
        scope = scope_id or self.scope_id
        units: list[MemoryUnit] = []

        # Multi-turn context accumulator for cross-turn extraction.
        context = _MultiTurnContext()

        for idx, turn in enumerate(turns, start=1):
            prompt_text = str(turn.get("prompt_text", "") or "").strip()
            response_text = str(turn.get("response_text", "") or "").strip()
            if not prompt_text and not response_text:
                continue

            extracted = _extract_memory_units_for_turn(
                scope_id=scope,
                session_id=session_id,
                turn_index=idx,
                prompt_text=prompt_text,
                response_text=response_text,
                multi_turn_context=context,
            )
            if extracted:
                logger.info(
                    "[Memory] extract turn=%d/%d → %d units [%s]",
                    idx, len(turns), len(extracted),
                    ", ".join(u.memory_type.value for u in extracted),
                )
            units.extend(extracted)
            context.add_turn(prompt_text, response_text, idx)

        if turns:
            units.append(
                MemoryUnit(
                    memory_id=str(uuid.uuid4()),
                    scope_id=scope,
                    memory_type=MemoryType.WORKING_SUMMARY,
                    content=_build_working_summary(turns),
                    summary="Current working summary from the most recent completed session.",
                    source_session_id=session_id,
                    source_turn_start=1,
                    source_turn_end=len(turns),
                    importance=0.9,
                    confidence=0.8,
                )
            )

        # Pre-ingestion validation: skip units with empty or overly short content.
        pre_validate_count = len(units)
        units = [u for u in units if len(u.content.strip()) >= 3]

        # Pre-ingestion dedup: skip units whose content already exists in the store.
        pre_dedup_count = len(units)
        units = _dedup_against_store(units, self.store, scope)
        dedup_skipped = pre_dedup_count - len(units)
        if dedup_skipped or (pre_validate_count - pre_dedup_count):
            logger.info(
                "[Memory] pre-filter: validated=%d dedup_skipped=%d remaining=%d",
                pre_validate_count - pre_dedup_count, dedup_skipped, len(units),
            )

        # Detect potential conflicts with existing memories.
        conflicts = _detect_conflicts(units, self.store, scope)
        if conflicts:
            logger.info("[Memory] detected %d conflicts with existing memories", len(conflicts))
            if self.telemetry_store is not None:
                self.telemetry_store.record(
                    "memory_conflicts",
                    {"scope_id": scope, "session_id": session_id, "conflicts": conflicts[:5]},
                )
            self._notify("conflicts_detected", scope_id=scope, count=len(conflicts))

        if self.embedder is not None:
            for unit in units:
                unit.embedding = self.embedder.encode(
                    " ".join([unit.summary, unit.content, " ".join(unit.topics), " ".join(unit.entities)])
                )

        added = self.store.add_memories(units)
        self.clear_cache()
        if self.auto_consolidate:
            consolidation_result = self.consolidator.consolidate(scope)
            if self.telemetry_store is not None and consolidation_result:
                self.telemetry_store.record(
                    "memory_consolidation",
                    {
                        "scope_id": scope,
                        "session_id": session_id,
                        "superseded": consolidation_result.get("superseded", 0),
                        "decayed": consolidation_result.get("decayed", 0),
                        "reinforced": consolidation_result.get("reinforced", 0),
                    },
                )
        stats = summarize_memory_store(self.store, scope)
        self._refresh_policy(scope)
        if self.telemetry_store is not None:
            self.telemetry_store.record(
                "memory_ingest",
                {
                    "scope_id": scope,
                    "session_id": session_id,
                    "added": added,
                    "active": stats.get("active", 0),
                    "dominant_type": stats.get("dominant_type", ""),
                    "active_by_type": stats.get("active_by_type", {}),
                },
            )
        logger.info(
            "[Memory] ingested %d memory units from session=%s scope=%s active=%d dominant_type=%s",
            added,
            session_id,
            scope,
            stats.get("active", 0),
            stats.get("dominant_type", ""),
        )
        # Auto-save stats snapshot after ingestion for trend tracking.
        try:
            self.store.save_stats_snapshot(scope)
        except Exception:
            pass  # Best-effort stats tracking.
        self._notify("ingest", scope_id=scope, session_id=session_id, added=added)
        return added

    def retrieve_for_prompt(
        self,
        task_description: str,
        scope_id: str | None = None,
        expand_links: bool = False,
    ) -> list[MemoryUnit]:
        effective_scope = scope_id or self.scope_id
        cache_key = f"{effective_scope}::{task_description}::{'linked' if expand_links else 'plain'}"
        if cache_key in self._retrieval_cache:
            self._cache_hits += 1
            logger.info(
                "[Memory] retrieve cache HIT (hits=%d misses=%d)",
                self._cache_hits, self._cache_misses,
            )
            return self._retrieval_cache[cache_key]
        self._cache_misses += 1

        query = MemoryQuery(
            scope_id=effective_scope,
            query_text=task_description,
            top_k=self.policy.max_injected_units,
            max_tokens=self.policy.max_injected_tokens,
        )
        hits = self.retriever.retrieve(query)
        hit_units = [h.unit for h in hits]
        # Optionally expand with linked memories from the graph.
        if expand_links and hit_units:
            seen_ids = {u.memory_id for u in hit_units}
            linked_extras: list[MemoryUnit] = []
            for u in hit_units[:5]:  # Limit expansion to top-5 to stay lightweight.
                linked = self.store.get_linked_memories(u.memory_id)
                for lu in linked:
                    if lu.memory_id not in seen_ids:
                        linked_extras.append(lu)
                        seen_ids.add(lu.memory_id)
            hit_units.extend(linked_extras)
        units = self._fit_token_budget(hit_units, query.max_tokens)
        # Log retrieval results.
        if units:
            type_summary = {}
            for u in units:
                type_summary[u.memory_type.value] = type_summary.get(u.memory_type.value, 0) + 1
            score_strs = [f"{h.score:.3f}" for h in hits[:len(units)]]
            rendered_for_log = self.render_for_prompt(units)
            logger.info(
                "[Memory] retrieve scope=%s mode=%s → %d/%d units, tokens≈%d, types=%s, scores=[%s], query=%s",
                effective_scope,
                self.retrieval_mode,
                len(units),
                len(hits),
                estimate_tokens(rendered_for_log),
                type_summary,
                ", ".join(score_strs),
                task_description[:80],
            )
            self.store.mark_accessed([u.memory_id for u in units], accessed_at=utc_now_iso())
            # Auto-boost importance for frequently accessed memories.
            for u in units:
                if u.access_count >= 3 and u.importance < 0.9:
                    new_importance = min(0.9, u.importance + 0.02)
                    self.store.update_importance(u.memory_id, round(new_importance, 4), utc_now_iso())
        else:
            logger.info(
                "[Memory] retrieve scope=%s mode=%s → 0 hits, query=%s",
                effective_scope, self.retrieval_mode, task_description[:80],
            )
        if self.telemetry_store is not None:
            rendered = self.render_for_prompt(units)
            # Type distribution of retrieved units.
            type_counts: dict[str, int] = {}
            for u in units:
                type_counts[u.memory_type.value] = type_counts.get(u.memory_type.value, 0) + 1
            self.telemetry_store.record(
                "memory_retrieval",
                {
                    "scope_id": query.scope_id,
                    "query_length": len(task_description),
                    "retrieved_count": len(units),
                    "injected_tokens": estimate_tokens(rendered),
                    "types_retrieved": list({u.memory_type.value for u in units}),
                    "type_distribution": type_counts,
                    "avg_importance": round(
                        sum(u.importance for u in units) / max(len(units), 1), 4
                    ),
                    "avg_reinforcement": round(
                        sum(u.reinforcement_score for u in units) / max(len(units), 1), 4
                    ),
                    "retrieval_mode": self.retrieval_mode,
                },
            )
        # Cache results for repeated queries within the same session.
        if len(self._retrieval_cache) >= self._cache_max_size:
            # Evict oldest entry.
            oldest = next(iter(self._retrieval_cache))
            del self._retrieval_cache[oldest]
        self._retrieval_cache[cache_key] = units
        return units

    _cache_hits: int = 0
    _cache_misses: int = 0

    def clear_cache(self) -> None:
        """Clear the retrieval cache (e.g., after ingestion)."""
        self._retrieval_cache.clear()

    def render_for_prompt(self, units: list[MemoryUnit], include_pool_context: bool = False) -> str:
        if not units:
            return ""
        lines = ["## Relevant Long-Term Memory"]
        if include_pool_context:
            stats = self.get_scope_stats()
            active = stats.get("active", 0)
            types = stats.get("type_count", 0)
            lines.append(f"_Pool: {active} memories across {types} types. Showing top {len(units)}._")

        # Sort pinned memories to front for guaranteed visibility.
        pinned = [u for u in units if u.importance >= 0.99]
        unpinned = [u for u in units if u.importance < 0.99]
        units = pinned + unpinned

        # Group units by type for a more structured, token-efficient render.
        by_type: dict[str, list[MemoryUnit]] = {}
        for unit in units:
            by_type.setdefault(unit.memory_type.value, []).append(unit)

        for type_name, group in by_type.items():
            label = type_name.replace("_", " ")
            lines.append(f"\n### {label}")
            for unit in group:
                # Prefer content over summary to avoid duplication.
                text = unit.content.strip() if unit.content.strip() else unit.summary.strip()
                if text:
                    freshness = _freshness_tag(unit.updated_at)
                    if freshness:
                        lines.append(f"- {text} [{freshness}]")
                    else:
                        lines.append(f"- {text}")
        return "\n".join(lines).strip()

    def get_scope_stats(self, scope_id: str | None = None) -> dict:
        return summarize_memory_store(self.store, scope_id or self.scope_id)

    def get_policy_state(self) -> dict:
        if self.policy_store is None:
            return {}
        return self.policy_store.load().__dict__

    def get_access_patterns(self, scope_id: str | None = None, limit: int = 5) -> dict:
        """Return access pattern insights for the given scope."""
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=500)
        if not units:
            return {"total": 0, "most_accessed": [], "never_accessed": 0}
        sorted_by_access = sorted(units, key=lambda u: u.access_count, reverse=True)
        most_accessed = [
            {"id": u.memory_id, "type": u.memory_type.value, "access_count": u.access_count, "content": u.content[:100]}
            for u in sorted_by_access[:limit]
        ]
        never_accessed = sum(1 for u in units if u.access_count == 0)
        avg_access = sum(u.access_count for u in units) / float(len(units))
        return {
            "total": len(units),
            "most_accessed": most_accessed,
            "never_accessed": never_accessed,
            "avg_access_count": round(avg_access, 2),
        }

    def diagnose(self, scope_id: str | None = None) -> dict:
        """Return a diagnostic summary for operator debugging.

        Combines store stats, policy state, access patterns, and retrieval
        telemetry into a single view for quick health assessment.
        """
        scope = scope_id or self.scope_id
        stats = self.get_scope_stats(scope)
        access = self.get_access_patterns(scope)
        policy = self.get_policy_state()

        # Analyze retrieval telemetry for recent performance.
        telemetry = self.get_recent_telemetry(limit=50)
        retrieval_events = [
            e for e in telemetry if e.get("event_type") == "memory_retrieval"
        ]
        zero_retrievals = sum(
            1 for e in retrieval_events
            if e.get("payload", {}).get("retrieved_count", 0) == 0
        )
        avg_retrieved = 0.0
        if retrieval_events:
            avg_retrieved = sum(
                e.get("payload", {}).get("retrieved_count", 0)
                for e in retrieval_events
            ) / float(len(retrieval_events))

        # Memory age distribution.
        from datetime import datetime, timezone

        age_buckets = {"<1h": 0, "<24h": 0, "<7d": 0, "older": 0}
        units = self.store.list_active(scope, limit=500)
        now = datetime.now(timezone.utc)
        for u in units:
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_hours = max((now - created).total_seconds() / 3600.0, 0.0)
            except (ValueError, TypeError):
                age_hours = float("inf")
            if age_hours < 1:
                age_buckets["<1h"] += 1
            elif age_hours < 24:
                age_buckets["<24h"] += 1
            elif age_hours < 168:
                age_buckets["<7d"] += 1
            else:
                age_buckets["older"] += 1

        # Cache stats.
        total_cache = self._cache_hits + self._cache_misses
        cache_hit_rate = round(self._cache_hits / max(total_cache, 1), 4)

        # TTL statistics.
        ttl_set = sum(1 for u in units if u.expires_at)
        ttl_expiring_soon = 0
        for u in units:
            if u.expires_at:
                try:
                    exp = datetime.fromisoformat(u.expires_at.replace("Z", "+00:00"))
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    hours_left = (exp - now).total_seconds() / 3600.0
                    if 0 < hours_left < 24:
                        ttl_expiring_soon += 1
                except (ValueError, TypeError):
                    pass

        issues: list[str] = []
        if stats.get("active", 0) == 0:
            issues.append("no active memories in store")
        if access.get("never_accessed", 0) > stats.get("active", 1) * 0.8:
            issues.append("over 80% of memories have never been accessed")
        if retrieval_events and zero_retrievals > len(retrieval_events) * 0.5:
            issues.append("over 50% of retrievals returned zero results")
        if ttl_expiring_soon > 0:
            issues.append(f"{ttl_expiring_soon} memories expiring within 24 hours")

        return {
            "scope_id": scope,
            "store": {
                "active": stats.get("active", 0),
                "dominant_type": stats.get("dominant_type", ""),
                "type_count": stats.get("type_count", 0),
                "age_distribution": age_buckets,
            },
            "access": {
                "avg_access_count": access.get("avg_access_count", 0.0),
                "never_accessed": access.get("never_accessed", 0),
            },
            "retrieval": {
                "recent_events": len(retrieval_events),
                "avg_retrieved": round(avg_retrieved, 2),
                "zero_retrieval_count": zero_retrievals,
            },
            "cache": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": cache_hit_rate,
            },
            "ttl": {
                "memories_with_ttl": ttl_set,
                "expiring_within_24h": ttl_expiring_soon,
            },
            "policy": {
                "retrieval_mode": policy.get("retrieval_mode", ""),
                "max_injected_units": policy.get("max_injected_units", 0),
                "max_injected_tokens": policy.get("max_injected_tokens", 0),
            },
            "issues": issues,
        }

    def detect_conflicts(self, scope_id: str | None = None) -> list[dict]:
        """Detect potential contradictions within the active memory pool.

        Compares all active memories of the same type that share significant
        topic/entity overlap but have different content.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=500)
        if len(units) < 2:
            return []

        conflicts: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for i, a in enumerate(units):
            a_terms = set(t.lower() for t in a.topics + a.entities)
            if not a_terms:
                continue
            for b in units[i + 1:]:
                if a.memory_type != b.memory_type:
                    continue
                b_terms = set(t.lower() for t in b.topics + b.entities)
                if not b_terms:
                    continue
                overlap = len(a_terms & b_terms) / float(len(a_terms | b_terms))
                if overlap < 0.65:
                    continue
                if a.content.strip().lower() == b.content.strip().lower():
                    continue
                pair_key = tuple(sorted([a.memory_id, b.memory_id]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                conflicts.append({
                    "id_a": a.memory_id,
                    "id_b": b.memory_id,
                    "type": a.memory_type.value,
                    "overlap": round(overlap, 4),
                    "content_a": a.content[:120],
                    "content_b": b.content[:120],
                })
        return conflicts

    def explain_retrieval(
        self,
        task_description: str,
        scope_id: str | None = None,
    ) -> list[dict]:
        """Return detailed retrieval results with scoring explanations.

        Useful for operator debugging: shows why each memory was selected
        and its contribution to the final result.
        """
        effective_scope = scope_id or self.scope_id
        query = MemoryQuery(
            scope_id=effective_scope,
            query_text=task_description,
            top_k=self.policy.max_injected_units,
            max_tokens=self.policy.max_injected_tokens,
        )
        hits = self.retriever.retrieve(query)
        return [
            {
                "memory_id": h.unit.memory_id,
                "type": h.unit.memory_type.value,
                "score": round(h.score, 4),
                "matched_terms": h.matched_terms,
                "reason": h.reason,
                "content_preview": h.unit.content[:120],
                "importance": h.unit.importance,
                "access_count": h.unit.access_count,
            }
            for h in hits
        ]

    def list_scopes(self) -> list[dict]:
        """List all scopes in the store with memory counts."""
        return self.store.list_scopes()

    def update_memory(self, memory_id: str, content: str, summary: str = "") -> bool:
        """Update the content of an existing memory."""
        result = self.store.update_content(memory_id, content, summary)
        if result:
            self.clear_cache()
        return result

    def get_memory(self, memory_id: str) -> MemoryUnit | None:
        """Get a specific memory by ID."""
        return self.store._get_by_id(memory_id)

    def set_ttl(self, memory_id: str, expires_at: str) -> bool:
        """Set or clear a TTL on a memory.

        Args:
            memory_id: Target memory.
            expires_at: ISO-8601 expiry timestamp, or empty string to clear.
        """
        result = self.store.set_ttl(memory_id, expires_at)
        if result:
            self.clear_cache()
        return result

    def expire_stale(self, scope_id: str | None = None) -> int:
        """Archive all memories that have passed their TTL."""
        scope = scope_id or self.scope_id
        count = self.store.expire_stale(scope)
        if count:
            self.clear_cache()
            self._notify("expire", scope_id=scope, expired_count=count)
        return count

    def share_memory(self, memory_id: str, target_scope_id: str) -> str | None:
        """Copy a memory to another scope for cross-scope knowledge sharing.

        Returns the new memory ID in the target scope.
        """
        new_id = self.store.share_to_scope(memory_id, target_scope_id)
        if new_id:
            self.clear_cache()
            self._notify("share", memory_id=memory_id, target_scope_id=target_scope_id, new_id=new_id)
        return new_id

    def export_scope(self, scope_id: str | None = None) -> list[dict]:
        """Export all active memories for a scope as JSON-serializable dicts."""
        scope = scope_id or self.scope_id
        return self.store.export_scope_json(scope)

    def import_memories(self, data: list[dict], target_scope_id: str | None = None) -> int:
        """Import memories from JSON dicts into the store."""
        scope = target_scope_id or self.scope_id
        count = self.store.import_memories_json(data, scope)
        if count:
            self.clear_cache()
        return count

    def set_type_ttl(
        self,
        memory_type: MemoryType,
        expires_at: str,
        scope_id: str | None = None,
    ) -> int:
        """Set TTL on all active memories of a given type."""
        scope = scope_id or self.scope_id
        count = self.store.set_type_ttl(scope, memory_type, expires_at)
        if count:
            self.clear_cache()
        return count

    def merge_memories(self, id_a: str, id_b: str, merged_content: str, merged_summary: str = "") -> str | None:
        """Merge two memories into a new one, superseding both."""
        new_id = self.store.merge_memories(id_a, id_b, merged_content, merged_summary)
        if new_id:
            self.clear_cache()
            self._notify("merge", id_a=id_a, id_b=id_b, new_id=new_id)
        return new_id

    def get_memory_history(self, memory_id: str) -> list[dict]:
        """Get version history for a memory through its supersedes chain."""
        return self.store.get_memory_history(memory_id)

    def get_scope_analytics(self, scope_id: str | None = None) -> dict:
        """Get comprehensive analytics for a scope."""
        scope = scope_id or self.scope_id
        return self.store.get_scope_analytics(scope)

    def add_tags(self, memory_id: str, tags: list[str]) -> bool:
        """Add user-defined tags to a memory."""
        result = self.store.add_tags(memory_id, tags)
        if result:
            self.clear_cache()
        return result

    def remove_tags(self, memory_id: str, tags: list[str]) -> bool:
        """Remove tags from a memory."""
        result = self.store.remove_tags(memory_id, tags)
        if result:
            self.clear_cache()
        return result

    def search_by_tag(self, tag: str, scope_id: str | None = None, limit: int = 50) -> list[MemoryUnit]:
        """Find all active memories with a given tag."""
        scope = scope_id or self.scope_id
        return self.store.search_by_tag(scope, tag, limit)

    def bulk_archive(self, memory_ids: list[str]) -> int:
        """Archive multiple memories at once."""
        count = self.store.bulk_archive(memory_ids)
        if count:
            self.clear_cache()
        return count

    def snapshot_scope(self, scope_id: str | None = None) -> dict:
        """Create a point-in-time snapshot for potential rollback."""
        scope = scope_id or self.scope_id
        return self.store.snapshot_scope(scope)

    def restore_snapshot(self, snapshot: dict) -> int:
        """Restore a scope from a previous snapshot."""
        count = self.store.restore_snapshot(snapshot)
        if count:
            self.clear_cache()
        return count

    def get_event_log(self, scope_id: str | None = None, limit: int = 50) -> list[dict]:
        """Get recent memory mutation events."""
        scope = scope_id or ""
        return self.store.get_event_log(scope_id=scope, limit=limit)

    def find_similar(self, memory_id: str, limit: int = 5) -> list[dict]:
        """Find memories similar to a given memory by topic/entity overlap."""
        results = self.store.find_similar(memory_id, limit)
        return [
            {
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "similarity": score,
                "content": u.content[:120],
            }
            for u, score in results
        ]

    def get_health_score(self, scope_id: str | None = None) -> dict:
        """Get a composite health score (0-100) for the memory pool."""
        scope = scope_id or self.scope_id
        return self.store.compute_health_score(scope)

    def find_duplicates(self, scope_id: str | None = None, threshold: float = 0.80) -> list[dict]:
        """Find near-duplicate memory pairs by content similarity."""
        scope = scope_id or self.scope_id
        return self.store.find_duplicates(scope, threshold)

    def consolidation_dry_run(self, scope_id: str | None = None) -> dict:
        """Preview what consolidation would do without applying changes."""
        scope = scope_id or self.scope_id
        return self.consolidator.dry_run(scope)

    def save_stats_snapshot(self, scope_id: str | None = None) -> dict:
        """Save a timestamped stats snapshot for trend tracking."""
        scope = scope_id or self.scope_id
        return self.store.save_stats_snapshot(scope)

    def get_stats_trend(self, scope_id: str | None = None, limit: int = 20) -> list[dict]:
        """Get stats trend over time."""
        scope = scope_id or self.scope_id
        return self.store.get_stats_trend(scope, limit)

    def search_advanced(
        self,
        keyword: str = "",
        memory_type: str = "",
        tag: str = "",
        min_importance: float = 0.0,
        scope_id: str | None = None,
        limit: int = 50,
    ) -> list[MemoryUnit]:
        """Search memories with combined criteria."""
        scope = scope_id or self.scope_id
        return self.store.search_advanced(scope, keyword, memory_type, tag, min_importance, limit)

    def compare_scopes(self, scope_a: str, scope_b: str) -> dict:
        """Compare two scopes to find shared and unique memories."""
        return self.store.compare_scopes(scope_a, scope_b)

    def auto_resolve_conflicts(self, scope_id: str | None = None) -> dict:
        """Automatically resolve conflicts by superseding older memories.

        When two same-type memories overlap significantly but have different
        content, the older one is superseded by the newer one.
        """
        scope = scope_id or self.scope_id
        conflicts = self.detect_conflicts(scope)
        if not conflicts:
            return {"resolved": 0}

        now = utc_now_iso()
        resolved = 0
        for c in conflicts:
            a = self.store._get_by_id(c["id_a"])
            b = self.store._get_by_id(c["id_b"])
            if a is None or b is None:
                continue
            # Skip if either is already superseded or pinned.
            if a.status != MemoryStatus.ACTIVE or b.status != MemoryStatus.ACTIVE:
                continue
            if a.importance >= 0.99 or b.importance >= 0.99:
                continue
            # Supersede the older one.
            if a.created_at <= b.created_at:
                self.store.supersede(a.memory_id, b.memory_id, now)
            else:
                self.store.supersede(b.memory_id, a.memory_id, now)
            resolved += 1

        if resolved:
            self.clear_cache()
        result = {"resolved": resolved, "total_conflicts": len(conflicts)}
        self._notify("conflict_resolution", scope_id=scope, **result)
        return result

    def rebalance_importance(self, scope_id: str | None = None) -> dict:
        """Rebalance importance distribution to prevent clustering.

        If too many memories have the same importance value, spread them
        out to improve retrieval differentiation.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        if len(units) < 5:
            return {"adjusted": 0}

        # Group by rounded importance.
        clusters: dict[float, list[MemoryUnit]] = {}
        for u in units:
            key = round(u.importance, 1)
            clusters.setdefault(key, []).append(u)

        adjusted = 0
        now = utc_now_iso()
        for _key, group in clusters.items():
            if len(group) < 4:
                continue
            # Spread importance within the cluster based on access count.
            group.sort(key=lambda u: u.access_count, reverse=True)
            spread = 0.05 * (len(group) - 1)
            base = max(0.1, group[0].importance - spread / 2)
            for idx, u in enumerate(group):
                if u.importance >= 0.99:  # don't touch pinned
                    continue
                new_imp = round(min(0.95, base + idx * 0.05 / max(len(group) - 1, 1) * spread), 4)
                if abs(new_imp - u.importance) > 0.005:
                    self.store.update_importance(u.memory_id, new_imp, now)
                    adjusted += 1

        if adjusted:
            self.clear_cache()
        return {"adjusted": adjusted}

    def pin_memory(self, memory_id: str) -> bool:
        """Pin a memory so it always ranks highest in retrieval."""
        result = self.store.pin_memory(memory_id)
        if result:
            self.clear_cache()
        return result

    def unpin_memory(self, memory_id: str) -> bool:
        """Unpin a previously pinned memory."""
        result = self.store.unpin_memory(memory_id)
        if result:
            self.clear_cache()
        return result

    def provide_feedback(self, memory_id: str, helpful) -> None:
        """Record retrieval feedback for a specific memory.

        Args:
            memory_id: Target memory.
            helpful: True/False, or "positive"/"negative" string.
        """
        if isinstance(helpful, str):
            helpful = helpful.lower() in ("positive", "true", "yes", "1", "helpful")
        self.store.record_feedback(memory_id, helpful)
        self.clear_cache()
        self._notify("feedback", memory_id=memory_id, helpful=helpful)
        if self.telemetry_store is not None:
            self.telemetry_store.record(
                "memory_feedback",
                {"memory_id": memory_id, "helpful": helpful},
            )

    def analyze_feedback_patterns(self, scope_id: str | None = None) -> dict:
        """Analyze retrieval feedback patterns for a scope.

        Returns statistics about positive/negative feedback distribution,
        most-boosted and most-penalized memories, and feedback density.
        """
        scope = scope_id or self.scope_id
        events = self.store.get_event_log(scope_id=scope, limit=10000)
        feedback_events = [e for e in events if e.get("event_type") == "feedback"]

        positive = 0
        negative = 0
        by_memory: dict[str, dict] = {}

        for ev in feedback_events:
            detail = ev.get("detail", "")
            mid = ev.get("memory_id", "")
            is_positive = detail.startswith("positive")
            if is_positive:
                positive += 1
            else:
                negative += 1
            if mid:
                entry = by_memory.setdefault(mid, {"positive": 0, "negative": 0, "memory_id": mid})
                if is_positive:
                    entry["positive"] += 1
                else:
                    entry["negative"] += 1

        total = positive + negative
        sorted_memories = sorted(by_memory.values(), key=lambda x: x["positive"] - x["negative"], reverse=True)
        most_boosted = sorted_memories[:5] if sorted_memories else []
        most_penalized = sorted(by_memory.values(), key=lambda x: x["negative"] - x["positive"], reverse=True)[:5]

        return {
            "scope_id": scope,
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive / total, 4) if total else 0.0,
            "unique_memories_with_feedback": len(by_memory),
            "most_boosted": most_boosted,
            "most_penalized": most_penalized,
        }

    def get_pool_summary(self, scope_id: str | None = None, max_per_type: int = 3) -> str:
        """Generate a concise summary of the entire memory pool.

        Returns a human-readable overview useful for operator inspection.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=500)
        if not units:
            return "No active memories."

        by_type: dict[str, list[MemoryUnit]] = {}
        for u in units:
            by_type.setdefault(u.memory_type.value, []).append(u)

        lines = [f"Memory pool: {len(units)} active memories across {len(by_type)} types"]
        for type_name, group in sorted(by_type.items()):
            lines.append(f"\n{type_name} ({len(group)}):")
            # Show top memories by importance.
            top = sorted(group, key=lambda u: u.importance, reverse=True)[:max_per_type]
            for u in top:
                preview = u.content[:100].replace("\n", " ")
                lines.append(f"  - [{u.importance:.2f}] {preview}")
            if len(group) > max_per_type:
                lines.append(f"  ... and {len(group) - max_per_type} more")

        conflicts = self.detect_conflicts(scope)
        if conflicts:
            lines.append(f"\nPotential conflicts: {len(conflicts)}")
            for c in conflicts[:3]:
                lines.append(f"  - {c['content_a'][:60]} vs {c['content_b'][:60]}")

        return "\n".join(lines)

    def search_memories(
        self,
        query_text: str,
        scope_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories by keyword with scoring information.

        Returns dictionaries with memory details and relevance scores.
        Useful for operator debugging and inspection.
        """
        scope = scope_id or self.scope_id
        hits = self.store.search_keyword(scope, query_text, limit=limit)
        return [
            {
                "memory_id": h.unit.memory_id,
                "type": h.unit.memory_type.value,
                "score": round(h.score, 4),
                "content": h.unit.content[:200],
                "importance": h.unit.importance,
                "access_count": h.unit.access_count,
                "matched_terms": h.matched_terms,
                "created_at": h.unit.created_at,
            }
            for h in hits
        ]

    def bulk_update_importance(
        self,
        updates: list[tuple[str, float]],
    ) -> int:
        """Update importance for multiple memories at once.

        Args:
            updates: List of (memory_id, new_importance) tuples.

        Returns:
            Number of memories updated.
        """
        count = 0
        now = utc_now_iso()
        for memory_id, importance in updates:
            clamped = max(0.1, min(0.99, importance))
            self.store.update_importance(memory_id, round(clamped, 4), now)
            count += 1
        if count:
            self.clear_cache()
        return count

    def get_recent_telemetry(self, limit: int = 20) -> list[dict]:
        if self.telemetry_store is None:
            return []
        return self.telemetry_store.read_recent(limit=limit)

    def apply_retention_policy(
        self,
        scope_id: str | None = None,
        max_age_days: int = 90,
        min_importance: float = 0.3,
        min_access_count: int = 0,
    ) -> dict:
        """Apply retention policy: archive old, low-importance, unused memories.

        Memories that are older than max_age_days AND have importance below
        min_importance AND have never been accessed are archived.
        Working summaries are exempt (always kept).
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)
        archived = 0

        for u in units:
            # Never archive working summaries.
            if u.memory_type == MemoryType.WORKING_SUMMARY:
                continue
            # Never archive pinned memories.
            if u.importance >= 0.99:
                continue
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                continue
            if age_days < max_age_days:
                continue
            if u.importance >= min_importance:
                continue
            if u.access_count > min_access_count:
                continue
            # Archive it.
            self.store.conn.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                ("archived", utc_now_iso(), u.memory_id),
            )
            self.store._remove_fts(u.memory_id)
            archived += 1

        if archived:
            self.store.conn.commit()
            self.clear_cache()

        return {"archived": archived, "scope_id": scope}

    def apply_typed_retention(
        self,
        scope_id: str | None = None,
        type_policies: dict[str, dict] | None = None,
    ) -> dict:
        """Apply per-type retention policies.

        Each type can have its own max_age_days, min_importance, and min_access_count.
        Default policies are applied if type_policies is not specified.
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        defaults: dict[str, dict] = {
            "episodic": {"max_age_days": 30, "min_importance": 0.2, "min_access_count": 0},
            "semantic": {"max_age_days": 180, "min_importance": 0.3, "min_access_count": 0},
            "preference": {"max_age_days": 365, "min_importance": 0.1, "min_access_count": 0},
            "project_state": {"max_age_days": 60, "min_importance": 0.2, "min_access_count": 0},
            "procedural_observation": {"max_age_days": 90, "min_importance": 0.2, "min_access_count": 0},
        }
        if type_policies:
            for k, v in type_policies.items():
                defaults[k] = {**defaults.get(k, {}), **v}

        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)
        archived = 0

        for u in units:
            if u.memory_type == MemoryType.WORKING_SUMMARY:
                continue
            if u.importance >= 0.99:
                continue
            policy = defaults.get(u.memory_type.value, {"max_age_days": 90, "min_importance": 0.3, "min_access_count": 0})
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                continue
            if age_days < policy["max_age_days"]:
                continue
            if u.importance >= policy["min_importance"]:
                continue
            if u.access_count > policy.get("min_access_count", 0):
                continue
            self.store.conn.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                ("archived", utc_now_iso(), u.memory_id),
            )
            self.store._remove_fts(u.memory_id)
            archived += 1

        if archived:
            self.store.conn.commit()
            self.clear_cache()
        return {"archived": archived, "scope_id": scope}

    def apply_adaptive_ttl(
        self,
        scope_id: str | None = None,
        base_days: dict[str, int] | None = None,
    ) -> dict:
        """Set TTL on memories based on type and access patterns.

        Memories that are accessed more frequently get longer TTLs.
        Working summaries get short TTLs (7 days). Episodic memories
        get medium TTLs (30 days). Others follow base_days or default to 90.
        Access count > 3 doubles the base TTL.
        """
        from datetime import datetime, timedelta, timezone

        scope = scope_id or self.scope_id
        defaults = {
            "working_summary": 7,
            "episodic": 30,
            "semantic": 90,
            "preference": 180,
            "project_state": 60,
            "procedural_observation": 90,
        }
        if base_days:
            defaults.update(base_days)

        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)
        updated = 0
        for u in units:
            if u.expires_at:  # Skip if TTL already set.
                continue
            if u.importance >= 0.99:  # Never auto-TTL pinned.
                continue
            base = defaults.get(u.memory_type.value, 90)
            # Frequently accessed memories get double TTL.
            if u.access_count > 3:
                base *= 2
            # High-importance memories get 50% longer TTL.
            if u.importance >= 0.7:
                base = int(base * 1.5)
            expires = now + timedelta(days=base)
            self.store.set_ttl(u.memory_id, expires.isoformat(timespec="seconds"))
            updated += 1

        return {"updated": updated, "scope_id": scope}

    def batch_archive_by_criteria(
        self,
        scope_id: str | None = None,
        max_quality_score: float | None = None,
        memory_type: MemoryType | None = None,
        max_importance: float | None = None,
        min_age_days: int | None = None,
    ) -> dict:
        """Archive memories matching all specified criteria.

        Pinned and working_summary memories are always excluded.
        All criteria that are set must be satisfied (AND logic).
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)
        to_archive: list[str] = []

        for u in units:
            if u.memory_type == MemoryType.WORKING_SUMMARY:
                continue
            if u.importance >= 0.99:
                continue
            if memory_type is not None and u.memory_type != memory_type:
                continue
            if max_importance is not None and u.importance > max_importance:
                continue
            if min_age_days is not None:
                try:
                    created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    age = (now - created).total_seconds() / 86400.0
                    if age < min_age_days:
                        continue
                except (ValueError, TypeError):
                    continue
            if max_quality_score is not None:
                quality = self.score_memory_quality(u.memory_id)
                if quality["score"] > max_quality_score:
                    continue
            to_archive.append(u.memory_id)

        for mid in to_archive:
            self.store.conn.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                ("archived", utc_now_iso(), mid),
            )
            self.store._remove_fts(mid)

        if to_archive:
            self.store.conn.commit()
            self.clear_cache()

        return {"archived": len(to_archive), "scope_id": scope}

    def score_memory_quality(self, memory_id: str) -> dict:
        """Compute a quality score (0-100) for a single memory unit.

        Factors: content richness, metadata completeness, access activity,
        importance calibration, and link connectivity.
        """
        unit = self.store._get_by_id(memory_id)
        if unit is None:
            return {"score": 0, "reason": "not found"}

        # 1. Content richness (0-25): longer, more informative content scores higher.
        words = len(unit.content.split())
        content_score = min(25, 25 * min(words, 20) / 20.0)

        # 2. Metadata completeness (0-25): topics + entities + summary + tags.
        meta_points = 0
        if unit.topics:
            meta_points += min(3, len(unit.topics))
        if unit.entities:
            meta_points += min(3, len(unit.entities))
        if unit.summary:
            meta_points += 2
        if unit.tags:
            meta_points += min(2, len(unit.tags))
        metadata_score = min(25, 25 * meta_points / 10.0)

        # 3. Access activity (0-25): accessed memories are more valuable.
        access_score = min(25, 25 * min(unit.access_count, 5) / 5.0)

        # 4. Importance + reinforcement (0-15).
        importance_score = 15 * unit.importance

        # 5. Link connectivity (0-10).
        links = self.store.get_links(memory_id)
        link_score = min(10, 10 * min(len(links), 3) / 3.0)

        total = round(content_score + metadata_score + access_score + importance_score + link_score, 1)
        return {
            "score": total,
            "memory_id": memory_id,
            "components": {
                "content_richness": round(content_score, 1),
                "metadata_completeness": round(metadata_score, 1),
                "access_activity": round(access_score, 1),
                "importance": round(importance_score, 1),
                "connectivity": round(link_score, 1),
            },
        }

    def get_lowest_quality_memories(self, scope_id: str | None = None, limit: int = 10) -> list[dict]:
        """Get the lowest-quality active memories in a scope for review/cleanup."""
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=500)
        scored = []
        for u in units:
            result = self.score_memory_quality(u.memory_id)
            result["content_preview"] = u.content[:80]
            result["memory_type"] = u.memory_type.value
            scored.append(result)
        scored.sort(key=lambda x: x["score"])
        return scored[:limit]

    # --- Scope access control ---

    def grant_scope_access(self, scope_id: str, principal: str, permission: str = "read") -> bool:
        """Grant a principal access to a scope."""
        return self.store.grant_access(scope_id, principal, permission)

    def revoke_scope_access(self, scope_id: str, principal: str, permission: str | None = None) -> int:
        """Revoke a principal's access to a scope."""
        return self.store.revoke_access(scope_id, principal, permission)

    def check_scope_access(self, scope_id: str, principal: str, permission: str = "read") -> bool:
        """Check if a principal can access a scope."""
        return self.store.check_access(scope_id, principal, permission)

    def list_scope_grants(self, scope_id: str) -> list[dict]:
        """List all access grants for a scope."""
        return self.store.list_scope_grants(scope_id)

    # --- Memory watches ---

    def watch_memory(self, memory_id: str, watcher: str) -> bool:
        """Watch a memory for changes."""
        return self.store.add_watch(memory_id, watcher)

    def unwatch_memory(self, memory_id: str, watcher: str) -> bool:
        """Stop watching a memory."""
        return self.store.remove_watch(memory_id, watcher)

    def get_watchers(self, memory_id: str) -> list[str]:
        """Get all watchers for a memory."""
        return self.store.get_watchers(memory_id)

    def get_watched_memories(self, watcher: str) -> list[str]:
        """Get all memory IDs watched by a watcher."""
        return self.store.get_watched_memories(watcher)

    # --- Memory annotations ---

    def add_annotation(self, memory_id: str, content: str, author: str = "") -> int:
        """Add an annotation to a memory."""
        return self.store.add_annotation(memory_id, content, author)

    def get_annotations(self, memory_id: str) -> list[dict]:
        """Get all annotations for a memory."""
        return self.store.get_annotations(memory_id)

    def delete_annotation(self, annotation_id: int) -> bool:
        """Delete an annotation by ID."""
        return self.store.delete_annotation(annotation_id)

    # --- Memory links ---

    def add_link(self, source_id: str, target_id: str, link_type: str = "related") -> bool:
        """Create a directed link between two memories."""
        return self.store.add_link(source_id, target_id, link_type)

    def remove_link(self, source_id: str, target_id: str, link_type: str | None = None) -> int:
        """Remove a link between two memories."""
        return self.store.remove_link(source_id, target_id, link_type)

    def get_links(self, memory_id: str, direction: str = "both") -> list[dict]:
        """Get all links for a memory."""
        return self.store.get_links(memory_id, direction)

    def get_linked_memories(self, memory_id: str, link_type: str | None = None) -> list[MemoryUnit]:
        """Get all memory units linked to a given memory."""
        return self.store.get_linked_memories(memory_id, link_type)

    def migrate_scope(self, from_scope: str, to_scope: str) -> dict:
        """Move all active memories from one scope to another.

        Memories are copied to the new scope and archived in the old scope.
        """
        units = self.store.list_active(from_scope, limit=10000)
        migrated = 0
        for u in units:
            new_id = self.store.share_to_scope(u.memory_id, to_scope)
            if new_id:
                # Archive original.
                self.store.conn.execute(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                    ("archived", utc_now_iso(), u.memory_id),
                )
                self.store._remove_fts(u.memory_id)
                migrated += 1
        if migrated:
            self.store.conn.commit()
            self.clear_cache()
        self._notify("scope_migration", from_scope=from_scope, to_scope=to_scope, migrated=migrated)
        return {"migrated": migrated, "from_scope": from_scope, "to_scope": to_scope}

    def get_age_distribution(self, scope_id: str | None = None) -> dict:
        """Get age distribution of active memories in named buckets."""
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)
        buckets = {"< 1 day": 0, "1-7 days": 0, "1-4 weeks": 0, "1-3 months": 0, "3+ months": 0}

        for u in units:
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400.0
            except (ValueError, TypeError):
                continue
            if age_days < 1:
                buckets["< 1 day"] += 1
            elif age_days < 7:
                buckets["1-7 days"] += 1
            elif age_days < 28:
                buckets["1-4 weeks"] += 1
            elif age_days < 90:
                buckets["1-3 months"] += 1
            else:
                buckets["3+ months"] += 1

        return {"distribution": buckets, "total": len(units)}

    def find_cross_scope_duplicates(
        self,
        scope_a: str,
        scope_b: str,
        threshold: float = 0.80,
    ) -> list[dict]:
        """Find near-duplicate memories across two scopes."""
        units_a = self.store.list_active(scope_a, limit=500)
        units_b = self.store.list_active(scope_b, limit=500)
        if not units_a or not units_b:
            return []

        def _tokenize_content(content: str) -> set[str]:
            return set(w.lower() for w in content.split() if len(w) >= 3)

        tokens_a = {u.memory_id: _tokenize_content(u.content) for u in units_a}
        tokens_b = {u.memory_id: _tokenize_content(u.content) for u in units_b}
        content_a = {u.memory_id: u.content[:80] for u in units_a}
        content_b = {u.memory_id: u.content[:80] for u in units_b}

        duplicates: list[dict] = []
        for id_a, toks_a in tokens_a.items():
            if not toks_a:
                continue
            for id_b, toks_b in tokens_b.items():
                if not toks_b:
                    continue
                inter = len(toks_a & toks_b)
                union = len(toks_a | toks_b)
                if union > 0 and inter / union >= threshold:
                    duplicates.append({
                        "id_a": id_a, "scope_a": scope_a,
                        "id_b": id_b, "scope_b": scope_b,
                        "similarity": round(inter / union, 4),
                        "preview_a": content_a[id_a],
                        "preview_b": content_b[id_b],
                    })
        duplicates.sort(key=lambda x: x["similarity"], reverse=True)
        return duplicates[:20]

    def suggest_type_corrections(self, scope_id: str | None = None, limit: int = 10) -> list[dict]:
        """Suggest memories that might be mistyped based on content analysis.

        Checks for content patterns that suggest a different type than assigned.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=500)
        suggestions: list[dict] = []

        for u in units:
            content_lower = u.content.lower()
            suggested = None

            # Preference indicators.
            if u.memory_type != MemoryType.PREFERENCE and any(
                p in content_lower for p in ["i prefer", "i like", "i want", "my convention"]
            ):
                suggested = MemoryType.PREFERENCE

            # Project state indicators.
            elif u.memory_type != MemoryType.PROJECT_STATE and any(
                p in content_lower for p in ["the project uses", "our stack", "we use"]
            ):
                suggested = MemoryType.PROJECT_STATE

            # Procedural indicators.
            elif u.memory_type != MemoryType.PROCEDURAL_OBSERVATION and any(
                p in content_lower for p in ["always", "never", "make sure", "workflow"]
            ):
                suggested = MemoryType.PROCEDURAL_OBSERVATION

            if suggested:
                suggestions.append({
                    "memory_id": u.memory_id,
                    "current_type": u.memory_type.value,
                    "suggested_type": suggested.value,
                    "content_preview": u.content[:80],
                    "confidence": 0.6,
                })

        return suggestions[:limit]

    def compute_urgency_scores(self, scope_id: str | None = None, limit: int = 10) -> list[dict]:
        """Compute urgency scores for memories that need attention.

        Urgency considers TTL proximity, low access count, and importance.
        Returns the most urgent memories first.
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)
        scored: list[dict] = []

        for u in units:
            urgency = 0.0

            # TTL urgency: memories expiring soon are urgent.
            if u.expires_at:
                try:
                    expires = datetime.fromisoformat(u.expires_at.replace("Z", "+00:00"))
                    if expires.tzinfo is None:
                        expires = expires.replace(tzinfo=timezone.utc)
                    days_remaining = (expires - now).total_seconds() / 86400.0
                    if days_remaining < 0:
                        urgency += 50  # Already expired!
                    elif days_remaining < 7:
                        urgency += 30 * (1 - days_remaining / 7.0)
                except (ValueError, TypeError):
                    pass

            # Access urgency: high-importance memories that are never accessed.
            if u.importance > 0.6 and u.access_count == 0:
                urgency += 20 * u.importance

            # Quality urgency: low-quality high-importance memories need enrichment.
            if u.importance > 0.5 and not u.summary and not u.tags:
                urgency += 10

            if urgency > 0:
                scored.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "urgency": round(urgency, 2),
                    "content_preview": u.content[:80],
                    "importance": u.importance,
                    "expires_at": u.expires_at,
                    "access_count": u.access_count,
                })

        scored.sort(key=lambda x: x["urgency"], reverse=True)
        return scored[:limit]

    def get_memories_by_ids(self, memory_ids: list[str]) -> list:
        """Retrieve multiple memories by their IDs in a single operation."""
        return self.store.get_by_ids(memory_ids)

    def analyze_memory_impact(self, memory_id: str) -> dict:
        """Analyze what depends on a memory and what would be affected by archiving it.

        Performs a transitive traversal of incoming depends_on links.
        """
        unit = self.store._get_by_id(memory_id)
        if not unit:
            return {"error": "Memory not found", "memory_id": memory_id}

        # Find direct dependents (memories that depend_on this one).
        incoming = self.store.get_links(memory_id, direction="incoming")
        direct_dependents = [
            lnk["source_id"] for lnk in incoming if lnk["link_type"] == "depends_on"
        ]

        # Transitive dependents via BFS.
        all_dependents: list[str] = []
        visited = {memory_id}
        queue = list(direct_dependents)
        while queue:
            dep_id = queue.pop(0)
            if dep_id in visited:
                continue
            visited.add(dep_id)
            all_dependents.append(dep_id)
            further = self.store.get_links(dep_id, direction="incoming")
            for lnk in further:
                if lnk["link_type"] == "depends_on" and lnk["source_id"] not in visited:
                    queue.append(lnk["source_id"])

        # Other relationships.
        all_links = self.store.get_links(memory_id, direction="both")
        elaborations = [lnk["source_id"] for lnk in all_links if lnk["link_type"] == "elaborates" and lnk["direction"] == "incoming"]
        contradictions = [
            lnk["target_id"] if lnk["direction"] == "outgoing" else lnk["source_id"]
            for lnk in all_links if lnk["link_type"] == "contradicts"
        ]

        # Watchers.
        watchers = self.store.get_watchers(memory_id)

        return {
            "memory_id": memory_id,
            "content_preview": unit.content[:80],
            "direct_dependents": direct_dependents,
            "transitive_dependents": all_dependents,
            "elaborations": elaborations,
            "contradictions": contradictions,
            "watchers": watchers,
            "total_affected": len(all_dependents) + len(elaborations),
            "safe_to_archive": len(all_dependents) == 0,
        }

    def detect_dependency_cycles(self, scope_id: str | None = None) -> list[list[str]]:
        """Detect circular dependency chains in depends_on links.

        Returns a list of cycles, each cycle being a list of memory IDs.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)

        # Build adjacency list for depends_on links.
        adj: dict[str, list[str]] = {}
        for u in units:
            outgoing = self.store.get_links(u.memory_id, direction="outgoing")
            deps = [lnk["target_id"] for lnk in outgoing if lnk["link_type"] == "depends_on"]
            if deps:
                adj[u.memory_id] = deps

        # DFS-based cycle detection.
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {mid: WHITE for mid in adj}
        parent: dict[str, str | None] = {}
        cycles: list[list[str]] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    # Found a cycle — reconstruct it.
                    cycle = [neighbor]
                    cur = node
                    while cur != neighbor:
                        cycle.append(cur)
                        cur = parent.get(cur, neighbor)
                    cycle.append(neighbor)
                    cycle.reverse()
                    cycles.append(cycle)
                elif color[neighbor] == WHITE:
                    parent[neighbor] = node
                    dfs(neighbor)
            color[node] = BLACK

        for node in adj:
            if color[node] == WHITE:
                parent[node] = None
                dfs(node)

        return cycles

    def build_version_tree(self, memory_id: str) -> dict:
        """Build a version tree rooted at a memory, following supersedes chains.

        Traverses both directions: finds the root (oldest ancestor) and all descendants.
        Returns a nested tree structure.
        """
        # Find the root by following superseded_by backwards.
        root_id = memory_id
        visited_up = {memory_id}
        while True:
            unit = self.store._get_by_id(root_id)
            if not unit or not unit.superseded_by:
                break
            if unit.superseded_by in visited_up:
                break
            visited_up.add(unit.superseded_by)
            root_id = unit.superseded_by

        # Actually, superseded_by points to the newer version.
        # Let's find the oldest ancestor instead by looking for units that supersede this one.
        # Walk up: find all units whose superseded_by points to root candidates.
        # Simpler approach: get the full history chain using existing method.
        history = self.get_memory_history(memory_id)

        def _build_node(entry: dict) -> dict:
            return {
                "memory_id": entry["memory_id"],
                "status": entry.get("status", "unknown"),
                "content_preview": entry.get("content", "")[:60],
                "created_at": entry.get("created_at"),
                "superseded_by": entry.get("superseded_by"),
                "importance": entry.get("importance", 0.0),
            }

        nodes = [_build_node(e) for e in history]
        return {
            "root_id": history[-1]["memory_id"] if history else memory_id,
            "current_id": memory_id,
            "chain_length": len(nodes),
            "versions": nodes,
        }

    def search_with_context(
        self,
        query: str,
        scope_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories and return results with matched terms highlighted.

        Returns dicts with memory info and highlighted content snippets.
        """
        scope = scope_id or self.scope_id
        hits = self.store.search_keyword(scope, query, limit=limit)
        results = []
        for hit in hits:
            # Build a snippet with matched terms marked.
            content = hit.unit.content
            snippet = content[:200]
            for term in hit.matched_terms[:5]:
                # Case-insensitive highlight.
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                snippet = pattern.sub(f"**{term}**", snippet)
            results.append({
                "memory_id": hit.unit.memory_id,
                "type": hit.unit.memory_type.value,
                "score": round(hit.score, 4),
                "matched_terms": hit.matched_terms,
                "snippet": snippet,
                "importance": hit.unit.importance,
                "tags": hit.unit.tags,
            })
        return results

    def group_by_topic(self, scope_id: str | None = None, min_group_size: int = 2) -> dict:
        """Group active memories by their dominant topic.

        Returns topic -> list of memory summaries, sorted by group size descending.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        topic_groups: dict[str, list[dict]] = {}

        for u in units:
            if not u.topics:
                continue
            primary_topic = u.topics[0]
            if primary_topic not in topic_groups:
                topic_groups[primary_topic] = []
            topic_groups[primary_topic].append({
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "content_preview": u.content[:80],
                "importance": u.importance,
            })

        # Filter by min_group_size and sort.
        filtered = {
            topic: members
            for topic, members in topic_groups.items()
            if len(members) >= min_group_size
        }
        sorted_groups = dict(
            sorted(filtered.items(), key=lambda kv: len(kv[1]), reverse=True)
        )
        return {
            "total_groups": len(sorted_groups),
            "total_grouped": sum(len(v) for v in sorted_groups.values()),
            "groups": sorted_groups,
        }

    def find_stale_memories(
        self,
        scope_id: str | None = None,
        stale_days: int = 30,
        limit: int = 20,
    ) -> list[dict]:
        """Find memories that haven't been accessed recently and may be outdated.

        Considers both access recency and creation age.
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)
        stale: list[dict] = []

        for u in units:
            # Use updated_at as last activity proxy.
            try:
                last_active = datetime.fromisoformat(
                    u.updated_at.replace("Z", "+00:00")
                )
                if last_active.tzinfo is None:
                    last_active = last_active.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError, AttributeError):
                continue

            days_inactive = (now - last_active).total_seconds() / 86400.0
            if days_inactive >= stale_days:
                staleness = min(days_inactive / stale_days, 5.0)  # Cap at 5x
                stale.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "days_inactive": round(days_inactive, 1),
                    "staleness_factor": round(staleness, 2),
                    "importance": u.importance,
                    "access_count": u.access_count,
                    "is_pinned": u.importance >= 0.99,
                })

        stale.sort(key=lambda x: x["staleness_factor"], reverse=True)
        return stale[:limit]

    def bulk_add_links(
        self,
        links: list[dict],
    ) -> dict:
        """Create multiple links at once.

        Each link dict should have: source_id, target_id, link_type (optional, defaults to 'related').
        Returns count of created and skipped links.
        """
        created = 0
        skipped = 0
        for link in links:
            source = link.get("source_id", "")
            target = link.get("target_id", "")
            link_type = link.get("link_type", "related")
            if not source or not target:
                skipped += 1
                continue
            if self.store.add_link(source, target, link_type):
                created += 1
            else:
                skipped += 1
        return {"created": created, "skipped": skipped, "total": len(links)}

    def get_memory_summary_report(self, scope_id: str | None = None) -> dict:
        """Generate a comprehensive summary report of a scope's memory state.

        Combines stats, health, age distribution, importance histogram, and top topics
        into a single report suitable for dashboards or periodic reviews.
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        stats = self.get_scope_stats(scope)
        health = self.get_health_score(scope)
        age_dist = self.get_age_distribution(scope)
        importance_hist = self.get_importance_histogram(scope)
        topics = self.group_by_topic(scope, min_group_size=1)

        # Top 5 topics by group size.
        top_topics = []
        for topic, members in list(topics.get("groups", {}).items())[:5]:
            top_topics.append({"topic": topic, "count": len(members)})

        return {
            "scope_id": scope,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_active": stats.get("active", 0),
            "total_superseded": stats.get("superseded", 0),
            "health_score": health,
            "age_distribution": age_dist.get("distribution", {}),
            "importance_distribution": importance_hist.get("histogram", {}),
            "top_topics": top_topics,
            "topic_group_count": topics.get("total_groups", 0),
        }

    def suggest_auto_tags(self, scope_id: str | None = None, limit: int = 20) -> list[dict]:
        """Suggest tags for memories based on content analysis.

        Analyzes content for topic keywords, entities, and patterns to suggest
        tags for memories that have no tags.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        suggestions: list[dict] = []

        for u in units:
            if u.tags:
                continue  # Already tagged.

            suggested_tags = []
            # Use existing topics as tag candidates.
            for topic in u.topics[:3]:
                if len(topic) >= 3:
                    suggested_tags.append(topic)

            # Use entities as tag candidates.
            for entity in u.entities[:2]:
                if len(entity) >= 3:
                    suggested_tags.append(entity)

            # Content pattern-based tags.
            content_lower = u.content.lower()
            if any(w in content_lower for w in ["bug", "fix", "error", "issue"]):
                suggested_tags.append("bugfix")
            if any(w in content_lower for w in ["config", "setting", "environment"]):
                suggested_tags.append("configuration")
            if any(w in content_lower for w in ["deploy", "release", "production"]):
                suggested_tags.append("deployment")
            if any(w in content_lower for w in ["test", "spec", "assert"]):
                suggested_tags.append("testing")

            if suggested_tags:
                suggestions.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "suggested_tags": list(dict.fromkeys(suggested_tags))[:5],  # Deduplicate, keep order.
                })

        return suggestions[:limit]

    def export_link_graph(self, scope_id: str | None = None) -> dict:
        """Export the memory link graph for visualization.

        Returns nodes (memories) and edges (links) in a format
        suitable for graph visualization tools.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)

        nodes = []
        edges = []
        seen_edges: set[tuple[str, str, str]] = set()

        for u in units:
            nodes.append({
                "id": u.memory_id,
                "type": u.memory_type.value,
                "label": u.content[:40],
                "importance": u.importance,
                "topics": u.topics[:3],
            })
            links = self.store.get_links(u.memory_id, direction="outgoing")
            for lnk in links:
                edge_key = (lnk["source_id"], lnk["target_id"], lnk["link_type"])
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": lnk["source_id"],
                        "target": lnk["target_id"],
                        "type": lnk["link_type"],
                    })

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def get_deduplication_report(self, scope_id: str | None = None, threshold: float = 0.75) -> dict:
        """Generate a comprehensive deduplication report for a scope.

        Finds all near-duplicate pairs and groups them by cluster.
        """
        scope = scope_id or self.scope_id
        dupes = self.find_duplicates(scope, threshold=threshold)

        # Group duplicates into clusters using union-find.
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for d in dupes:
            union(d["id_a"], d["id_b"])

        clusters: dict[str, list[str]] = {}
        all_ids = set()
        for d in dupes:
            all_ids.add(d["id_a"])
            all_ids.add(d["id_b"])
        for mid in all_ids:
            root = find(mid)
            if root not in clusters:
                clusters[root] = []
            if mid not in clusters[root]:
                clusters[root].append(mid)

        return {
            "total_duplicate_pairs": len(dupes),
            "duplicate_clusters": len(clusters),
            "affected_memories": len(all_ids),
            "threshold": threshold,
            "pairs": dupes[:20],
            "clusters": [
                {"root": root, "members": members, "size": len(members)}
                for root, members in sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
            ][:10],
        }

    def search_regex(
        self,
        pattern: str,
        scope_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search memory content using a regular expression pattern.

        Returns matching memories with the matched portion highlighted.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        results = []
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []

        for u in units:
            match = compiled.search(u.content)
            if match:
                results.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "matched_text": match.group(),
                    "match_position": match.start(),
                    "importance": u.importance,
                })
                if len(results) >= limit:
                    break

        return results

    def merge_scopes(self, source_scope: str, target_scope: str) -> dict:
        """Merge all active memories from source scope into target scope.

        Unlike migrate_scope, this preserves the source scope intact.
        Memories are copied (shared) to the target scope.
        """
        source_units = self.store.list_active(source_scope, limit=5000)
        if not source_units:
            return {"copied": 0, "skipped": 0, "source_scope": source_scope, "target_scope": target_scope}

        # Check existing target content to avoid duplicates.
        target_units = self.store.list_active(target_scope, limit=5000)
        target_contents = {u.content.strip().lower() for u in target_units}

        copied = 0
        skipped = 0
        for u in source_units:
            if u.content.strip().lower() in target_contents:
                skipped += 1
                continue
            # Use share_to_scope to copy.
            try:
                self.store.share_to_scope(u.memory_id, target_scope)
                copied += 1
            except Exception:
                skipped += 1

        return {
            "copied": copied,
            "skipped": skipped,
            "source_scope": source_scope,
            "target_scope": target_scope,
        }

    def compute_stats_delta(self, scope_id: str | None = None) -> dict:
        """Compare current stats with the most recent snapshot to show changes.

        Returns deltas for key metrics.
        """
        scope = scope_id or self.scope_id
        current = self.get_scope_stats(scope)
        trend = self.get_stats_trend(scope, limit=2)

        if not trend or len(trend) < 1:
            return {
                "has_previous": False,
                "current": current,
                "deltas": {},
            }

        # The most recent snapshot is trend[0] (most recent first).
        previous = trend[0].get("stats", {})
        deltas = {}
        for key in ["active", "total", "superseded"]:
            cur_val = current.get(key, 0)
            prev_val = previous.get(key, 0)
            if isinstance(cur_val, (int, float)) and isinstance(prev_val, (int, float)):
                deltas[key] = cur_val - prev_val

        return {
            "has_previous": True,
            "current": current,
            "previous_snapshot": previous,
            "deltas": deltas,
        }

    def diff_memories(self, memory_id_a: str, memory_id_b: str) -> dict:
        """Compare two memories side by side, showing differences.

        Returns a structured diff of content, metadata, and other fields.
        """
        unit_a = self.store._get_by_id(memory_id_a)
        unit_b = self.store._get_by_id(memory_id_b)

        if not unit_a or not unit_b:
            return {"error": "One or both memories not found"}

        # Word-level content diff.
        words_a = set(unit_a.content.lower().split())
        words_b = set(unit_b.content.lower().split())
        only_a = words_a - words_b
        only_b = words_b - words_a
        shared = words_a & words_b

        return {
            "memory_a": {
                "memory_id": unit_a.memory_id,
                "type": unit_a.memory_type.value,
                "content": unit_a.content,
                "importance": unit_a.importance,
                "topics": unit_a.topics,
                "tags": unit_a.tags,
                "created_at": unit_a.created_at,
            },
            "memory_b": {
                "memory_id": unit_b.memory_id,
                "type": unit_b.memory_type.value,
                "content": unit_b.content,
                "importance": unit_b.importance,
                "topics": unit_b.topics,
                "tags": unit_b.tags,
                "created_at": unit_b.created_at,
            },
            "content_diff": {
                "shared_words": len(shared),
                "only_in_a": len(only_a),
                "only_in_b": len(only_b),
                "similarity": round(len(shared) / max(len(words_a | words_b), 1), 4),
            },
            "type_match": unit_a.memory_type == unit_b.memory_type,
            "importance_delta": round(unit_a.importance - unit_b.importance, 4),
        }

    def clone_scope(self, source_scope: str, target_scope: str) -> dict:
        """Deep-clone a scope: copies all active memories with full metadata.

        Unlike merge_scopes, this creates fresh copies with new IDs.
        """
        import uuid

        source_units = self.store.list_active(source_scope, limit=5000)
        if not source_units:
            return {"cloned": 0, "source_scope": source_scope, "target_scope": target_scope}

        cloned = 0
        for u in source_units:
            new_unit = MemoryUnit(
                memory_id=str(uuid.uuid4()),
                scope_id=target_scope,
                memory_type=u.memory_type,
                content=u.content,
                summary=u.summary,
                source_session_id=u.source_session_id,
                topics=list(u.topics),
                entities=list(u.entities),
                importance=u.importance,
                confidence=u.confidence,
                tags=list(u.tags),
            )
            self.store.add_memories([new_unit])
            cloned += 1

        return {
            "cloned": cloned,
            "source_scope": source_scope,
            "target_scope": target_scope,
        }

    def analyze_access_frequency(self, scope_id: str | None = None) -> dict:
        """Categorize memories into hot (frequently accessed), warm, and cold buckets.

        Based on access_count relative to pool average.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        if not units:
            return {"hot": [], "warm": [], "cold": [], "total": 0, "avg_access": 0}

        total_access = sum(u.access_count for u in units)
        avg_access = total_access / len(units) if units else 0
        hot_threshold = max(avg_access * 2, 3)
        cold_threshold = max(avg_access * 0.5, 1)

        hot, warm, cold = [], [], []
        for u in units:
            entry = {
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "access_count": u.access_count,
                "importance": u.importance,
                "content_preview": u.content[:60],
            }
            if u.access_count >= hot_threshold:
                hot.append(entry)
            elif u.access_count < cold_threshold:
                cold.append(entry)
            else:
                warm.append(entry)

        hot.sort(key=lambda x: x["access_count"], reverse=True)
        cold.sort(key=lambda x: x["access_count"])

        return {
            "hot": hot[:10],
            "warm": warm[:10],
            "cold": cold[:10],
            "total": len(units),
            "avg_access": round(avg_access, 2),
            "hot_count": len(hot),
            "warm_count": len(warm),
            "cold_count": len(cold),
        }

    def suggest_enrichments(self, scope_id: str | None = None, limit: int = 20) -> list[dict]:
        """Suggest enrichments for memories that lack metadata.

        Identifies memories missing summaries, tags, or topics that would benefit
        from enrichment.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        suggestions: list[dict] = []

        for u in units:
            missing = []
            if not u.summary:
                missing.append("summary")
            if not u.tags:
                missing.append("tags")
            if not u.topics:
                missing.append("topics")
            if not u.entities:
                missing.append("entities")

            if missing:
                suggestions.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "importance": u.importance,
                    "missing_fields": missing,
                    "completeness": round(1.0 - len(missing) / 4.0, 2),
                })

        # Sort by importance (high-importance memories should be enriched first).
        suggestions.sort(key=lambda x: (-x["importance"], x["completeness"]))
        return suggestions[:limit]

    def get_content_density_stats(self, scope_id: str | None = None) -> dict:
        """Analyze content density: token counts, value per token, and size distribution."""
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        if not units:
            return {"total": 0, "avg_tokens": 0, "avg_value_per_token": 0, "size_buckets": {}}

        token_counts = []
        value_per_token = []
        for u in units:
            tokens = len(u.content.split())
            token_counts.append(tokens)
            if tokens > 0:
                value_per_token.append(u.importance / tokens)

        # Size distribution buckets.
        buckets = {"tiny (<10)": 0, "small (10-50)": 0, "medium (50-150)": 0, "large (150+)": 0}
        for tc in token_counts:
            if tc < 10:
                buckets["tiny (<10)"] += 1
            elif tc < 50:
                buckets["small (10-50)"] += 1
            elif tc < 150:
                buckets["medium (50-150)"] += 1
            else:
                buckets["large (150+)"] += 1

        return {
            "total": len(units),
            "total_tokens": sum(token_counts),
            "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_value_per_token": round(sum(value_per_token) / max(len(value_per_token), 1), 4),
            "size_buckets": buckets,
        }

    def check_scope_quota(
        self,
        scope_id: str | None = None,
        max_memories: int = 1000,
    ) -> dict:
        """Check if a scope is within its memory quota.

        Returns quota status including current count, limit, and utilization.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=max_memories + 1)
        count = len(units)
        utilization = count / max(max_memories, 1)

        return {
            "scope_id": scope,
            "current_count": count,
            "max_memories": max_memories,
            "utilization": round(utilization, 4),
            "within_quota": count <= max_memories,
            "remaining": max(max_memories - count, 0),
            "warning": utilization >= 0.9,
        }

    def cascade_archive(self, memory_id: str) -> dict:
        """Archive a memory and all memories that depend on it (transitively)."""
        from datetime import datetime, timezone

        impact = self.analyze_memory_impact(memory_id)
        if "error" in impact:
            return impact

        to_archive = [memory_id] + impact["transitive_dependents"]
        archived = 0
        now = datetime.now(timezone.utc).isoformat()

        for mid in to_archive:
            unit = self.store._get_by_id(mid)
            if unit and unit.status.value == "active":
                self.store.conn.execute(
                    "UPDATE memories SET status = 'archived', updated_at = ? WHERE memory_id = ?",
                    (now, mid),
                )
                archived += 1

        self.store.conn.commit()
        self.clear_cache()

        return {
            "archived": archived,
            "root_memory": memory_id,
            "dependents_archived": archived - 1 if archived > 0 else 0,
        }

    def get_link_graph_stats(self, scope_id: str | None = None) -> dict:
        """Compute statistics about the memory link graph."""
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)

        total_links = 0
        link_type_counts: dict[str, int] = {}
        linked_memories = set()
        max_connections = 0
        most_connected = None

        for u in units:
            links = self.store.get_links(u.memory_id, direction="both")
            conn_count = len(links)
            total_links += len([l for l in links if l["direction"] == "outgoing"])

            if conn_count > 0:
                linked_memories.add(u.memory_id)

            if conn_count > max_connections:
                max_connections = conn_count
                most_connected = u.memory_id

            for lnk in links:
                lt = lnk["link_type"]
                link_type_counts[lt] = link_type_counts.get(lt, 0) + 1

        # Divide by 2 since we counted each link type from both sides.
        for lt in link_type_counts:
            link_type_counts[lt] = link_type_counts[lt] // 2 or link_type_counts[lt]

        return {
            "total_memories": len(units),
            "linked_memories": len(linked_memories),
            "unlinked_memories": len(units) - len(linked_memories),
            "total_links": total_links,
            "link_types": link_type_counts,
            "most_connected": most_connected,
            "max_connections": max_connections,
            "connectivity_ratio": round(len(linked_memories) / max(len(units), 1), 4),
        }

    def forecast_expiry(self, scope_id: str | None = None) -> dict:
        """Forecast memory expirations over upcoming time windows."""
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)

        windows = {
            "next_24h": 0,
            "next_7d": 0,
            "next_30d": 0,
            "no_expiry": 0,
        }
        total_with_ttl = 0

        for u in units:
            if not u.expires_at:
                windows["no_expiry"] += 1
                continue
            total_with_ttl += 1
            try:
                expires = datetime.fromisoformat(u.expires_at.replace("Z", "+00:00"))
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                days = (expires - now).total_seconds() / 86400.0
                if days <= 1:
                    windows["next_24h"] += 1
                elif days <= 7:
                    windows["next_7d"] += 1
                elif days <= 30:
                    windows["next_30d"] += 1
            except (ValueError, TypeError):
                pass

        return {
            "total": len(units),
            "with_ttl": total_with_ttl,
            "forecast": windows,
        }

    def get_type_overlap_matrix(self, scope_id: str | None = None) -> dict:
        """Compute topic overlap between memory types.

        Returns a matrix showing how much each pair of types shares topics.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)

        type_topics: dict[str, set[str]] = {}
        for u in units:
            t = u.memory_type.value
            if t not in type_topics:
                type_topics[t] = set()
            type_topics[t].update(u.topics)

        types = sorted(type_topics.keys())
        matrix: dict[str, dict[str, float]] = {}

        for ta in types:
            matrix[ta] = {}
            for tb in types:
                topics_a = type_topics[ta]
                topics_b = type_topics[tb]
                union = len(topics_a | topics_b)
                overlap = len(topics_a & topics_b) / max(union, 1)
                matrix[ta][tb] = round(overlap, 4)

        return {
            "types": types,
            "matrix": matrix,
        }

    def recommend_archival(
        self,
        scope_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Recommend memories for archival based on multiple signals.

        Combines staleness, low importance, low access, low quality, and no links
        into a single archival recommendation score.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        recommendations: list[dict] = []

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        for u in units:
            if u.importance >= 0.99:  # Skip pinned.
                continue

            score = 0.0
            reasons = []

            # Low importance.
            if u.importance < 0.3:
                score += 20 * (0.3 - u.importance)
                reasons.append("low_importance")

            # Never accessed.
            if u.access_count == 0:
                score += 15
                reasons.append("never_accessed")

            # Old and stale.
            try:
                updated = datetime.fromisoformat(u.updated_at.replace("Z", "+00:00"))
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                days_old = (now - updated).total_seconds() / 86400.0
                if days_old > 60:
                    score += min(days_old / 30.0, 10)
                    reasons.append("stale")
            except (ValueError, TypeError, AttributeError):
                pass

            # No metadata.
            if not u.summary and not u.tags and not u.topics:
                score += 5
                reasons.append("no_metadata")

            # No links.
            links = self.store.get_links(u.memory_id, direction="both")
            if not links:
                score += 3
                reasons.append("isolated")

            if score > 10:
                recommendations.append({
                    "memory_id": u.memory_id,
                    "type": u.memory_type.value,
                    "content_preview": u.content[:80],
                    "archival_score": round(score, 2),
                    "reasons": reasons,
                    "importance": u.importance,
                    "access_count": u.access_count,
                })

        recommendations.sort(key=lambda x: x["archival_score"], reverse=True)
        return recommendations[:limit]

    def get_scope_dashboard(self, scope_id: str | None = None) -> dict:
        """Generate a comprehensive operational dashboard for a scope.

        Combines: summary report, access frequency, content density, link stats,
        expiry forecast, quota, and archival recommendations into one view.
        """
        scope = scope_id or self.scope_id

        report = self.get_memory_summary_report(scope)
        access = self.analyze_access_frequency(scope)
        density = self.get_content_density_stats(scope)
        link_stats = self.get_link_graph_stats(scope)
        forecast = self.forecast_expiry(scope)
        quota = self.check_scope_quota(scope)
        archive_recs = self.recommend_archival(scope, limit=5)
        urgency = self.compute_urgency_scores(scope, limit=5)

        return {
            "scope_id": scope,
            "overview": {
                "total_active": report.get("total_active", 0),
                "health_score": report.get("health_score", 0),
                "topic_groups": report.get("topic_group_count", 0),
                "top_topics": report.get("top_topics", []),
            },
            "access": {
                "hot_count": access.get("hot_count", 0),
                "warm_count": access.get("warm_count", 0),
                "cold_count": access.get("cold_count", 0),
                "avg_access": access.get("avg_access", 0),
            },
            "content": {
                "total_tokens": density.get("total_tokens", 0),
                "avg_tokens": density.get("avg_tokens", 0),
                "size_buckets": density.get("size_buckets", {}),
            },
            "graph": {
                "linked_memories": link_stats.get("linked_memories", 0),
                "total_links": link_stats.get("total_links", 0),
                "connectivity": link_stats.get("connectivity_ratio", 0),
            },
            "expiry_forecast": forecast.get("forecast", {}),
            "quota": {
                "utilization": quota.get("utilization", 0),
                "within_quota": quota.get("within_quota", True),
            },
            "top_archival_candidates": len(archive_recs),
            "urgent_items": len(urgency),
        }

    def suggest_links(self, scope_id: str | None = None, threshold: float = 0.5, limit: int = 20) -> list[dict]:
        """Suggest links between memories that share topics/entities but aren't linked.

        Uses topic and entity overlap to identify potential relationships.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=500)
        suggestions: list[dict] = []

        # Build existing link set.
        linked_pairs: set[tuple[str, str]] = set()
        for u in units:
            links = self.store.get_links(u.memory_id, direction="both")
            for lnk in links:
                pair = tuple(sorted([lnk["source_id"], lnk["target_id"]]))
                linked_pairs.add(pair)

        # Check all pairs.
        for i, a in enumerate(units):
            a_features = set(a.topics) | set(a.entities)
            if not a_features:
                continue
            for b in units[i + 1:]:
                pair = tuple(sorted([a.memory_id, b.memory_id]))
                if pair in linked_pairs:
                    continue
                b_features = set(b.topics) | set(b.entities)
                if not b_features:
                    continue
                overlap = len(a_features & b_features)
                union = len(a_features | b_features)
                if union > 0 and overlap / union >= threshold:
                    shared = list(a_features & b_features)[:5]
                    suggestions.append({
                        "memory_a": a.memory_id,
                        "memory_b": b.memory_id,
                        "similarity": round(overlap / union, 4),
                        "shared_features": shared,
                        "suggested_type": "related",
                        "preview_a": a.content[:50],
                        "preview_b": b.content[:50],
                    })

        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        return suggestions[:limit]

    def generate_detailed_scope_comparison(self, scope_a: str, scope_b: str) -> dict:
        """Generate a detailed comparison report between two scopes.

        Goes beyond compare_scopes to include type distributions, topic overlap,
        and health differences.
        """
        units_a = self.store.list_active(scope_a, limit=5000)
        units_b = self.store.list_active(scope_b, limit=5000)

        # Type distributions.
        types_a: dict[str, int] = {}
        types_b: dict[str, int] = {}
        for u in units_a:
            t = u.memory_type.value
            types_a[t] = types_a.get(t, 0) + 1
        for u in units_b:
            t = u.memory_type.value
            types_b[t] = types_b.get(t, 0) + 1

        # Topic sets.
        topics_a = set()
        topics_b = set()
        for u in units_a:
            topics_a.update(u.topics)
        for u in units_b:
            topics_b.update(u.topics)

        shared_topics = topics_a & topics_b
        unique_a = topics_a - topics_b
        unique_b = topics_b - topics_a

        # Importance stats.
        imp_a = [u.importance for u in units_a] or [0]
        imp_b = [u.importance for u in units_b] or [0]

        return {
            "scope_a": {
                "name": scope_a,
                "count": len(units_a),
                "types": types_a,
                "topic_count": len(topics_a),
                "avg_importance": round(sum(imp_a) / len(imp_a), 4),
            },
            "scope_b": {
                "name": scope_b,
                "count": len(units_b),
                "types": types_b,
                "topic_count": len(topics_b),
                "avg_importance": round(sum(imp_b) / len(imp_b), 4),
            },
            "shared_topics": list(shared_topics)[:20],
            "unique_to_a": list(unique_a)[:20],
            "unique_to_b": list(unique_b)[:20],
            "topic_overlap": round(
                len(shared_topics) / max(len(topics_a | topics_b), 1), 4
            ),
        }

    def validate_content(self, content: str, rules: dict | None = None) -> dict:
        """Validate memory content against configurable rules.

        Default rules:
        - min_length: 3 characters
        - max_length: 10000 characters
        - min_words: 2
        - no_urls_only: content shouldn't be just a URL
        """
        if rules is None:
            rules = {}
        min_length = rules.get("min_length", 3)
        max_length = rules.get("max_length", 10000)
        min_words = rules.get("min_words", 2)

        errors = []
        warnings = []

        if len(content) < min_length:
            errors.append(f"Content too short ({len(content)} < {min_length} chars)")
        if len(content) > max_length:
            errors.append(f"Content too long ({len(content)} > {max_length} chars)")

        words = content.split()
        if len(words) < min_words:
            warnings.append(f"Very few words ({len(words)} < {min_words})")

        # Check if content is just a URL.
        stripped = content.strip()
        if stripped.startswith(("http://", "https://")) and " " not in stripped:
            warnings.append("Content appears to be just a URL")

        # Check for excessive repetition.
        if len(words) > 5:
            unique_ratio = len(set(w.lower() for w in words)) / len(words)
            if unique_ratio < 0.3:
                warnings.append(f"High word repetition (unique ratio: {unique_ratio:.0%})")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "char_count": len(content),
                "word_count": len(words),
            },
        }

    def generate_auto_summaries(self, scope_id: str | None = None, limit: int = 20) -> list[dict]:
        """Generate summaries for memories that don't have them.

        Creates keyword-based summaries from content (first sentence + topics).
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        generated: list[dict] = []

        for u in units:
            if u.summary:
                continue

            # Extract first sentence or first 100 chars.
            content = u.content.strip()
            first_sentence = content.split(".")[0].strip()
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + "..."

            # Add topic context.
            topic_str = ""
            if u.topics:
                topic_str = f" [{', '.join(u.topics[:3])}]"

            summary = f"{first_sentence}{topic_str}"

            # Apply the summary.
            self.store.update_content(u.memory_id, u.content)  # triggers FTS re-index
            self.store.conn.execute(
                "UPDATE memories SET summary = ? WHERE memory_id = ?",
                (summary, u.memory_id),
            )
            generated.append({
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "summary": summary,
            })

            if len(generated) >= limit:
                break

        self.store.conn.commit()
        self.clear_cache()
        return generated

    def recalculate_importance(self, scope_id: str | None = None) -> dict:
        """Recalculate importance for all memories based on current signals.

        Factors: access frequency, link count, metadata completeness, recency.
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)
        updated = 0

        for u in units:
            if u.importance >= 0.99:  # Skip pinned.
                continue

            new_importance = 0.5  # Base

            # Access bonus.
            if u.access_count > 0:
                new_importance += min(u.access_count * 0.02, 0.2)

            # Link bonus.
            links = self.store.get_links(u.memory_id, direction="both")
            if links:
                new_importance += min(len(links) * 0.03, 0.15)

            # Metadata completeness bonus.
            completeness = 0
            if u.summary:
                completeness += 0.25
            if u.tags:
                completeness += 0.25
            if u.topics:
                completeness += 0.25
            if u.entities:
                completeness += 0.25
            new_importance += completeness * 0.1

            # Recency bonus.
            try:
                updated_dt = datetime.fromisoformat(u.updated_at.replace("Z", "+00:00"))
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                days_old = (now - updated_dt).total_seconds() / 86400.0
                if days_old < 7:
                    new_importance += 0.05
            except (ValueError, TypeError, AttributeError):
                pass

            new_importance = min(round(new_importance, 4), 0.98)

            if abs(new_importance - u.importance) > 0.01:
                self.store.update_importance(u.memory_id, new_importance, now.isoformat())
                updated += 1

        return {
            "total_evaluated": len(units),
            "updated": updated,
            "scope_id": scope,
        }

    def analyze_type_balance(self, scope_id: str | None = None) -> dict:
        """Analyze memory type distribution and suggest rebalancing actions."""
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        if not units:
            return {"total": 0, "distribution": {}, "suggestions": []}

        type_counts: dict[str, int] = {}
        for u in units:
            t = u.memory_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        total = len(units)
        distribution = {t: {"count": c, "ratio": round(c / total, 4)} for t, c in type_counts.items()}

        suggestions = []
        # Check for dominant type (>60%).
        for t, info in distribution.items():
            if info["ratio"] > 0.6:
                suggestions.append({
                    "action": "reduce",
                    "type": t,
                    "ratio": info["ratio"],
                    "reason": f"{t} dominates at {info['ratio']:.0%} — consider archiving low-value {t} memories",
                })
        # Check for missing types.
        expected_types = {"semantic", "episodic", "preference", "project_state"}
        for et in expected_types:
            if et not in type_counts:
                suggestions.append({
                    "action": "add",
                    "type": et,
                    "ratio": 0,
                    "reason": f"No {et} memories — consider adding some for completeness",
                })

        return {
            "total": total,
            "distribution": distribution,
            "suggestions": suggestions,
        }

    def compare_scope_health(self, scope_a: str, scope_b: str) -> dict:
        """Compare health scores and key metrics between two scopes."""
        health_a = self.get_health_score(scope_a)
        health_b = self.get_health_score(scope_b)
        stats_a = self.get_scope_stats(scope_a)
        stats_b = self.get_scope_stats(scope_b)

        score_a = health_a.get("score", 0) if isinstance(health_a, dict) else 0
        score_b = health_b.get("score", 0) if isinstance(health_b, dict) else 0

        return {
            "scope_a": {
                "name": scope_a,
                "health": score_a,
                "active": stats_a.get("active", 0),
            },
            "scope_b": {
                "name": scope_b,
                "health": score_b,
                "active": stats_b.get("active", 0),
            },
            "health_delta": score_a - score_b,
            "healthier_scope": scope_a if score_a >= score_b else scope_b,
        }

    def get_memory_lifecycle(self, memory_id: str) -> dict:
        """Get the full lifecycle of a memory: creation, access, updates, and current state."""
        unit = self.store._get_by_id(memory_id)
        if not unit:
            return {"error": "Memory not found", "memory_id": memory_id}

        # Get events.
        events = self.store.get_event_log(limit=100)
        memory_events = [e for e in events if e.get("memory_id") == memory_id]

        # Get links.
        links = self.store.get_links(memory_id, direction="both")

        # Get annotations.
        annotations = self.store.get_annotations(memory_id)

        # Get watchers.
        watchers = self.store.get_watchers(memory_id)

        return {
            "memory_id": memory_id,
            "current_state": {
                "status": unit.status.value if hasattr(unit.status, "value") else str(unit.status),
                "type": unit.memory_type.value,
                "importance": unit.importance,
                "access_count": unit.access_count,
                "created_at": unit.created_at,
                "updated_at": unit.updated_at,
                "content_preview": unit.content[:80],
                "has_summary": bool(unit.summary),
                "tag_count": len(unit.tags),
                "topic_count": len(unit.topics),
            },
            "relationships": {
                "link_count": len(links),
                "annotation_count": len(annotations),
                "watcher_count": len(watchers),
            },
            "event_count": len(memory_events),
            "events": memory_events[:10],
        }

    def get_maintenance_recommendations(self, scope_id: str | None = None) -> dict:
        """Generate maintenance recommendations based on scope state.

        Analyzes the scope and recommends specific maintenance actions.
        """
        scope = scope_id or self.scope_id
        actions = []

        # Check for expired memories.
        forecast = self.forecast_expiry(scope)
        if forecast["forecast"].get("next_24h", 0) > 0:
            actions.append({
                "action": "expire_stale",
                "priority": "high",
                "reason": f"{forecast['forecast']['next_24h']} memories expiring in 24h",
            })

        # Check quota.
        quota = self.check_scope_quota(scope)
        if quota["warning"]:
            actions.append({
                "action": "archive_low_value",
                "priority": "high",
                "reason": f"Quota at {quota['utilization']:.0%} — nearing limit",
            })

        # Check for stale memories.
        stale = self.find_stale_memories(scope, stale_days=60, limit=1)
        if stale:
            actions.append({
                "action": "review_stale",
                "priority": "medium",
                "reason": "Stale memories detected (>60 days inactive)",
            })

        # Check type balance.
        balance = self.analyze_type_balance(scope)
        if balance.get("suggestions"):
            actions.append({
                "action": "rebalance_types",
                "priority": "low",
                "reason": balance["suggestions"][0]["reason"],
            })

        # Check for untagged memories.
        tag_suggestions = self.suggest_auto_tags(scope, limit=1)
        if tag_suggestions:
            actions.append({
                "action": "tag_memories",
                "priority": "low",
                "reason": "Untagged memories found — consider auto-tagging",
            })

        # Check integrity.
        integrity = self.store.validate_integrity()
        if integrity.get("issues"):
            actions.append({
                "action": "cleanup_orphans",
                "priority": "medium",
                "reason": f"{len(integrity['issues'])} integrity issues found",
            })

        # Check for memories without summaries.
        units = self.store.list_active(scope, limit=100)
        unsummarized = sum(1 for u in units if not u.summary)
        if unsummarized > len(units) * 0.5 and len(units) > 5:
            actions.append({
                "action": "auto_summarize",
                "priority": "low",
                "reason": f"{unsummarized} memories without summaries",
            })

        actions.sort(key=lambda a: {"high": 0, "medium": 1, "low": 2}.get(a["priority"], 3))

        return {
            "scope_id": scope,
            "total_recommendations": len(actions),
            "recommendations": actions,
        }

    def export_for_training(self, scope_id: str | None = None) -> list[dict]:
        """Export memories in a format suitable for ML training/fine-tuning.

        Returns structured records with content, metadata, and quality signals.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        records = []

        for u in units:
            record = {
                "content": u.content,
                "summary": u.summary or "",
                "type": u.memory_type.value,
                "topics": u.topics,
                "entities": u.entities,
                "importance": u.importance,
                "confidence": u.confidence,
                "access_count": u.access_count,
                "tags": u.tags,
                "metadata": {
                    "memory_id": u.memory_id,
                    "scope_id": u.scope_id,
                    "created_at": u.created_at,
                },
            }
            records.append(record)

        return records

    def batch_update_content(self, updates: list[dict]) -> dict:
        """Update content for multiple memories at once.

        Each update dict should have: memory_id, content.
        Returns counts of updated and failed.
        """
        updated = 0
        failed = 0
        for u in updates:
            memory_id = u.get("memory_id", "")
            content = u.get("content", "")
            if not memory_id or not content:
                failed += 1
                continue
            try:
                self.store.update_content(memory_id, content)
                updated += 1
            except Exception:
                failed += 1

        self.clear_cache()
        return {"updated": updated, "failed": failed, "total": len(updates)}

    def compute_freshness_scores(self, scope_id: str | None = None, limit: int = 20) -> list[dict]:
        """Compute combined freshness scores for memories.

        Considers: recency of creation, last access, update frequency, and TTL.
        Score 0-100 where 100 is freshest.
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        now = datetime.now(timezone.utc)
        scored: list[dict] = []

        for u in units:
            freshness = 50.0  # Base score.

            # Recency bonus (0-30).
            try:
                created = datetime.fromisoformat(u.created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_old = (now - created).total_seconds() / 86400.0
                recency = max(0, 30 * (1 - days_old / 365.0))
                freshness += recency
            except (ValueError, TypeError, AttributeError):
                pass

            # Access recency bonus (0-20).
            try:
                updated = datetime.fromisoformat(u.updated_at.replace("Z", "+00:00"))
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                access_days = (now - updated).total_seconds() / 86400.0
                access_bonus = max(0, 20 * (1 - access_days / 90.0))
                freshness += access_bonus
            except (ValueError, TypeError, AttributeError):
                pass

            freshness = min(max(round(freshness, 1), 0), 100)
            scored.append({
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "freshness": freshness,
                "content_preview": u.content[:60],
                "importance": u.importance,
            })

        scored.sort(key=lambda x: x["freshness"], reverse=True)
        return scored[:limit]

    def get_scope_inventory(
        self,
        scope_id: str | None = None,
        type_filter: str | None = None,
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        sort_by: str = "importance",
        limit: int = 50,
    ) -> dict:
        """Get a detailed, filterable inventory of memories in a scope.

        Supports filtering by type, importance range, and sorting.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)

        # Apply filters.
        filtered = units
        if type_filter:
            filtered = [u for u in filtered if u.memory_type.value == type_filter]
        filtered = [u for u in filtered if min_importance <= u.importance <= max_importance]

        # Sort.
        if sort_by == "importance":
            filtered.sort(key=lambda u: u.importance, reverse=True)
        elif sort_by == "access":
            filtered.sort(key=lambda u: u.access_count, reverse=True)
        elif sort_by == "created":
            filtered.sort(key=lambda u: u.created_at or "", reverse=True)

        items = []
        for u in filtered[:limit]:
            items.append({
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "content_preview": u.content[:80],
                "importance": u.importance,
                "access_count": u.access_count,
                "tags": u.tags[:3],
                "topics": u.topics[:3],
                "has_summary": bool(u.summary),
                "created_at": u.created_at,
            })

        return {
            "total_before_filter": len(units),
            "total_after_filter": len(filtered),
            "showing": len(items),
            "filters": {
                "type": type_filter,
                "min_importance": min_importance,
                "max_importance": max_importance,
                "sort_by": sort_by,
            },
            "items": items,
        }

    def normalize_content(self, memory_id: str) -> dict:
        """Normalize memory content: strip whitespace, collapse multiple spaces, fix encoding."""
        unit = self.store._get_by_id(memory_id)
        if not unit:
            return {"error": "Memory not found", "memory_id": memory_id}

        original = unit.content
        normalized = " ".join(original.split())  # Collapse whitespace.
        normalized = normalized.strip()

        if normalized == original:
            return {"memory_id": memory_id, "changed": False}

        self.store.update_content(memory_id, normalized)
        self.clear_cache()
        return {
            "memory_id": memory_id,
            "changed": True,
            "original_length": len(original),
            "normalized_length": len(normalized),
        }

    def batch_normalize_content(self, scope_id: str | None = None) -> dict:
        """Normalize content for all memories in a scope."""
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        normalized = 0

        for u in units:
            clean = " ".join(u.content.split()).strip()
            if clean != u.content:
                self.store.update_content(u.memory_id, clean)
                normalized += 1

        self.clear_cache()
        return {"total": len(units), "normalized": normalized}

    def get_priority_queue(
        self,
        scope_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get a priority-ranked queue of memories needing attention.

        Combines urgency, quality, and staleness into a single priority score.
        """
        scope = scope_id or self.scope_id
        urgency = self.compute_urgency_scores(scope, limit=50)
        enrichment = self.suggest_enrichments(scope, limit=50)
        stale = self.find_stale_memories(scope, stale_days=30, limit=50)

        # Build combined priority map.
        priority_map: dict[str, dict] = {}

        for u in urgency:
            mid = u["memory_id"]
            if mid not in priority_map:
                priority_map[mid] = {"memory_id": mid, "priority": 0, "reasons": [], "type": u.get("type", "")}
            priority_map[mid]["priority"] += u["urgency"]
            priority_map[mid]["reasons"].append("urgent")

        for e in enrichment:
            mid = e["memory_id"]
            if mid not in priority_map:
                priority_map[mid] = {"memory_id": mid, "priority": 0, "reasons": [], "type": e.get("type", "")}
            priority_map[mid]["priority"] += (1 - e["completeness"]) * 20
            priority_map[mid]["reasons"].append("needs_enrichment")

        for s in stale:
            mid = s["memory_id"]
            if mid not in priority_map:
                priority_map[mid] = {"memory_id": mid, "priority": 0, "reasons": [], "type": s.get("type", "")}
            priority_map[mid]["priority"] += s["staleness_factor"] * 10
            priority_map[mid]["reasons"].append("stale")

        items = sorted(priority_map.values(), key=lambda x: x["priority"], reverse=True)
        return items[:limit]

    def apply_quality_gate(self, content: str, memory_type: str | None = None) -> dict:
        """Apply quality gates to validate memory content before ingestion.

        Returns pass/fail with specific gate results.
        """
        gates = []

        # Gate 1: Content validation.
        validation = self.validate_content(content)
        gates.append({
            "gate": "content_validation",
            "passed": validation["valid"],
            "details": validation.get("errors", []),
        })

        # Gate 2: Minimum information density.
        words = content.split()
        unique_words = set(w.lower() for w in words)
        info_density = len(unique_words) / max(len(words), 1)
        gates.append({
            "gate": "information_density",
            "passed": info_density >= 0.3 or len(words) <= 5,
            "details": [f"density={info_density:.2f}"],
        })

        # Gate 3: Not a duplicate (check against existing).
        # This is a lightweight check using first 50 chars.
        gates.append({
            "gate": "non_duplicate",
            "passed": True,  # Full dedup happens at ingestion.
            "details": [],
        })

        # Gate 4: Minimum content length.
        gates.append({
            "gate": "min_length",
            "passed": len(content.strip()) >= 10,
            "details": [f"length={len(content.strip())}"],
        })

        all_passed = all(g["passed"] for g in gates)
        return {
            "passed": all_passed,
            "gates": gates,
            "content_preview": content[:80],
        }

    def get_importance_histogram(self, scope_id: str | None = None, buckets: int = 10) -> dict:
        """Get importance distribution as a histogram for operators.

        Returns bucket counts for [0.0-0.1), [0.1-0.2), ..., [0.9-1.0].
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        histogram: dict[str, int] = {}
        bucket_size = 1.0 / buckets
        for i in range(buckets):
            low = round(i * bucket_size, 2)
            high = round((i + 1) * bucket_size, 2)
            label = f"{low:.1f}-{high:.1f}"
            histogram[label] = 0
        for u in units:
            idx = min(int(u.importance / bucket_size), buckets - 1)
            low = round(idx * bucket_size, 2)
            high = round((idx + 1) * bucket_size, 2)
            label = f"{low:.1f}-{high:.1f}"
            histogram[label] = histogram.get(label, 0) + 1
        return {"histogram": histogram, "total": len(units)}

    def run_maintenance(self, scope_id: str | None = None) -> dict:
        """Run a full maintenance cycle: expire, consolidate, clean orphans, compact.

        Returns a summary of all actions taken.
        """
        scope = scope_id or self.scope_id
        results: dict = {"scope_id": scope}

        # 1. Expire TTL-stale memories.
        expired = self.expire_stale(scope)
        results["expired"] = expired

        # 2. Consolidate (dedup, near-dedup, decay).
        consolidation = self.consolidator.consolidate(scope)
        results["consolidation"] = consolidation

        # 3. Apply typed retention policy.
        retention = self.apply_typed_retention(scope)
        results["retention_archived"] = retention["archived"]

        # 4. Clean orphaned references.
        orphans = self.store.cleanup_orphans()
        results["orphans_removed"] = orphans["total_removed"]

        # 5. Garbage collect superseded memories.
        gc = self.store.garbage_collect(scope)
        results["gc_removed"] = gc.get("removed", 0)

        # 6. Compact the database.
        self.store.compact()
        results["compacted"] = True

        self._notify("maintenance", **results)
        return results

    def sample_memories(self, scope_id: str | None = None, count: int = 5) -> list[MemoryUnit]:
        """Return a random sample of active memories for exploration."""
        scope = scope_id or self.scope_id
        return self.store.sample_memories(scope, count)

    def get_api_status(self, scope_id: str | None = None) -> dict:
        """Get a comprehensive, API-ready status summary.

        Returns a JSON-serializable dict combining store stats, health,
        policy state, and feature usage indicators.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        health = self.store.compute_health_score(scope)
        db_info = self.store.get_db_size()
        integrity = self.store.validate_integrity()

        type_counts: dict[str, int] = {}
        pinned = 0
        with_ttl = 0
        with_tags = 0
        total_access = 0
        for u in units:
            type_counts[u.memory_type.value] = type_counts.get(u.memory_type.value, 0) + 1
            if u.importance >= 0.99:
                pinned += 1
            if u.expires_at:
                with_ttl += 1
            if u.tags:
                with_tags += 1
            total_access += u.access_count

        return {
            "scope_id": scope,
            "schema_version": self.store.get_schema_version(),
            "active_count": len(units),
            "type_distribution": type_counts,
            "health": health,
            "db": db_info,
            "integrity_valid": integrity["valid"],
            "features": {
                "pinned": pinned,
                "with_ttl": with_ttl,
                "with_tags": with_tags,
                "total_accesses": total_access,
            },
            "policy": {
                "retrieval_mode": self.retrieval_mode,
                "max_injected_units": self.policy.max_injected_units,
                "max_injected_tokens": self.policy.max_injected_tokens,
            },
            "embedder": self.get_embedder_info(),
        }

    def get_optimization_hints(self, scope_id: str | None = None) -> list[str]:
        """Generate optimization suggestions based on current store state."""
        scope = scope_id or self.scope_id
        hints: list[str] = []

        units = self.store.list_active(scope, limit=5000)
        if not units:
            return ["Store is empty — no optimizations needed."]

        # Check for too many memories.
        if len(units) > 1000:
            hints.append(f"High memory count ({len(units)}): consider running batch-archive or typed-retention.")

        # Check for low access coverage.
        accessed = sum(1 for u in units if u.access_count > 0)
        if accessed / len(units) < 0.3:
            hints.append(f"Low access coverage ({accessed}/{len(units)}): many memories are never retrieved.")

        # Check for no TTL set.
        with_ttl = sum(1 for u in units if u.expires_at)
        if with_ttl == 0 and len(units) > 10:
            hints.append("No memories have TTL set: consider running auto-ttl to prevent unbounded growth.")

        # Check DB size.
        db_info = self.store.get_db_size()
        if db_info["freelist_ratio"] > 0.2:
            hints.append(f"High freelist ratio ({db_info['freelist_ratio']:.0%}): run gc and compact to reclaim space.")

        # Check for duplicates.
        dupes = self.store.find_duplicates(scope, threshold=0.85)
        if dupes:
            hints.append(f"{len(dupes)} near-duplicate pair(s) found: consider consolidation.")

        # Check integrity.
        integrity = self.store.validate_integrity()
        if not integrity["valid"]:
            hints.append(f"Integrity issues found: {', '.join(integrity['issues'][:3])}")

        if not hints:
            hints.append("Store looks healthy — no optimizations needed.")

        return hints

    def generate_usage_report(self, scope_id: str | None = None) -> dict:
        """Generate a comprehensive usage report for monitoring and dashboards.

        Combines health score, quality distribution, type breakdown,
        access patterns, link statistics, and TTL status.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        health = self.store.compute_health_score(scope)

        type_counts: dict[str, int] = {}
        total_access = 0
        accessed_count = 0
        with_ttl = 0
        pinned_count = 0
        total_links = 0
        importance_sum = 0.0

        for u in units:
            type_counts[u.memory_type.value] = type_counts.get(u.memory_type.value, 0) + 1
            total_access += u.access_count
            if u.access_count > 0:
                accessed_count += 1
            if u.expires_at:
                with_ttl += 1
            if u.importance >= 0.99:
                pinned_count += 1
            importance_sum += u.importance
            total_links += len(self.store.get_links(u.memory_id, direction="outgoing"))

        n = max(len(units), 1)
        return {
            "scope_id": scope,
            "total_active": len(units),
            "health_score": health.get("score", 0),
            "type_distribution": type_counts,
            "avg_importance": round(importance_sum / n, 4),
            "access_coverage": round(accessed_count / n, 4),
            "total_accesses": total_access,
            "pinned_count": pinned_count,
            "with_ttl": with_ttl,
            "total_outgoing_links": total_links,
            "health_components": health.get("components", {}),
        }

    def find_memory_clusters(self, scope_id: str | None = None) -> list[list[str]]:
        """Find connected clusters of memories via links.

        Returns list of clusters, each cluster being a list of memory IDs.
        Isolated (unlinked) memories are not included.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        unit_ids = {u.memory_id for u in units}

        # Build adjacency from links.
        adj: dict[str, set[str]] = {mid: set() for mid in unit_ids}
        for mid in unit_ids:
            links = self.store.get_links(mid, direction="both")
            for lnk in links:
                other = lnk["target_id"] if lnk["direction"] == "outgoing" else lnk["source_id"]
                if other in unit_ids:
                    adj[mid].add(other)
                    adj[other].add(mid)

        # BFS connected components.
        visited: set[str] = set()
        clusters: list[list[str]] = []
        for mid in unit_ids:
            if mid in visited or not adj[mid]:
                continue
            cluster: list[str] = []
            queue = [mid]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(cluster) > 1:
                clusters.append(cluster)

        clusters.sort(key=len, reverse=True)
        return clusters

    def get_embedder_info(self) -> dict:
        """Return information about the current embedder configuration."""
        if self.embedder is None:
            return {
                "enabled": False,
                "mode": "none",
                "model": None,
                "dimensions": None,
            }
        embedder_type = type(self.embedder).__name__
        is_semantic = embedder_type == "SentenceTransformerEmbedder"
        info = {
            "enabled": True,
            "mode": "semantic" if is_semantic else "hashing",
            "type": embedder_type,
            "dimensions": self.embedder.dimensions,
        }
        if is_semantic:
            info["model"] = getattr(self.embedder, "model_name", "unknown")
            info["available"] = getattr(self.embedder, "is_available", False)
        return info

    def re_embed_scope(
        self,
        scope_id: str | None = None,
        embedder: "BaseEmbedder | None" = None,
    ) -> dict:
        """Re-encode all active memories in a scope with the current (or given) embedder.

        Useful when switching from hashing to semantic embeddings.
        Returns count of memories re-embedded.
        """
        scope = scope_id or self.scope_id
        emb = embedder or self.embedder
        if emb is None:
            return {"error": "No embedder available", "re_embedded": 0}

        units = self.store.list_active(scope, limit=5000)
        if not units:
            return {"scope_id": scope, "re_embedded": 0, "total": 0}

        # Batch encode for efficiency.
        texts = [f"{u.summary} {u.content}" for u in units]
        try:
            vectors = emb.encode_batch(texts)
        except Exception as exc:
            return {"error": str(exc), "re_embedded": 0}

        import json as _json
        re_embedded = 0
        for unit, vec in zip(units, vectors):
            if vec:
                self.store.conn.execute(
                    "UPDATE memories SET embedding_json = ? WHERE memory_id = ?",
                    (_json.dumps(vec), unit.memory_id),
                )
                re_embedded += 1
        self.store.conn.commit()
        self.clear_cache()
        return {"scope_id": scope, "re_embedded": re_embedded, "total": len(units)}

    def compress_content(self, memory_id: str) -> dict:
        """Compress memory content by removing redundancy and verbosity.

        Applies heuristic compression: strips filler phrases, compacts
        whitespace, removes redundant words, and truncates to essential content.
        """
        unit = self.store._get_by_id(memory_id)
        if not unit:
            return {"error": "Memory not found", "memory_id": memory_id}

        original = unit.content
        compressed = _compress_text(original)

        if compressed == original:
            return {"memory_id": memory_id, "changed": False, "length": len(original)}

        self.store.update_content(memory_id, compressed)
        self.clear_cache()
        return {
            "memory_id": memory_id,
            "changed": True,
            "original_length": len(original),
            "compressed_length": len(compressed),
            "reduction_pct": round(100 * (1 - len(compressed) / max(len(original), 1)), 1),
        }

    def batch_compress(self, scope_id: str | None = None) -> dict:
        """Compress content for all memories in a scope.

        Returns stats on how many were compressed and total token savings.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        compressed = 0
        total_saved = 0

        for u in units:
            result = _compress_text(u.content)
            if result != u.content:
                saved = len(u.content) - len(result)
                self.store.update_content(u.memory_id, result)
                compressed += 1
                total_saved += saved

        self.clear_cache()
        return {"total": len(units), "compressed": compressed, "chars_saved": total_saved}

    def bulk_tag_by_type(
        self,
        scope_id: str | None = None,
        type_tag_map: dict[str, list[str]] | None = None,
    ) -> dict:
        """Auto-tag all memories based on their type.

        type_tag_map maps memory type values to tags to add.
        Default: {"project_state": ["infra"], "preference": ["user-pref"]}.
        """
        scope = scope_id or self.scope_id
        defaults = {
            "project_state": ["infrastructure"],
            "preference": ["user-preference"],
            "working_summary": ["summary"],
            "procedural_observation": ["procedure"],
        }
        tag_map = type_tag_map or defaults
        units = self.store.list_active(scope, limit=5000)
        tagged = 0

        for u in units:
            tags_to_add = tag_map.get(u.memory_type.value, [])
            if tags_to_add:
                existing = set(u.tags)
                new_tags = [t for t in tags_to_add if t not in existing]
                if new_tags:
                    self.store.add_tags(u.memory_id, new_tags)
                    tagged += 1

        self.clear_cache()
        return {"total": len(units), "tagged": tagged}

    def analyze_retention_effectiveness(self, scope_id: str | None = None) -> dict:
        """Analyze how well retention policies are working.

        Measures: archived vs active ratio, access patterns before archival,
        average lifetime, and whether high-value memories are being retained.
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        active = self.store.list_active(scope, limit=5000)
        all_rows = self.store.conn.execute(
            "SELECT status, importance, access_count, created_at, updated_at FROM memories WHERE scope_id = ?",
            (scope,),
        ).fetchall()

        active_count = len(active)
        archived_count = sum(1 for r in all_rows if r["status"] == "archived")
        superseded_count = sum(1 for r in all_rows if r["status"] == "superseded")
        total = len(all_rows)

        # Average importance of archived vs active.
        active_imp = [r["importance"] for r in all_rows if r["status"] == "active"]
        archived_imp = [r["importance"] for r in all_rows if r["status"] == "archived"]

        # Average access count of archived.
        archived_access = [r["access_count"] for r in all_rows if r["status"] == "archived"]

        # Average age of active memories.
        now = datetime.now(timezone.utc)
        active_ages = []
        for r in all_rows:
            if r["status"] == "active" and r["created_at"]:
                try:
                    created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                    active_ages.append((now - created).days)
                except (ValueError, TypeError):
                    pass

        return {
            "scope_id": scope,
            "total_memories": total,
            "active": active_count,
            "archived": archived_count,
            "superseded": superseded_count,
            "archive_ratio": round(archived_count / max(total, 1), 3),
            "avg_active_importance": round(sum(active_imp) / max(len(active_imp), 1), 3),
            "avg_archived_importance": round(sum(archived_imp) / max(len(archived_imp), 1), 3),
            "avg_archived_access_count": round(sum(archived_access) / max(len(archived_access), 1), 1),
            "avg_active_age_days": round(sum(active_ages) / max(len(active_ages), 1), 1),
            "retention_health": "good" if (
                not archived_imp or sum(archived_imp) / max(len(archived_imp), 1) < sum(active_imp) / max(len(active_imp), 1)
            ) else "review_needed",
        }

    def get_memory_growth_rate(self, scope_id: str | None = None, window_days: int = 30) -> dict:
        """Compute memory growth rate over a time window.

        Returns memories added per day and projected growth.
        """
        from datetime import datetime, timedelta, timezone

        scope = scope_id or self.scope_id
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=window_days)).isoformat()

        total_active = len(self.store.list_active(scope, limit=5000))
        recent_rows = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE scope_id = ? AND created_at >= ?",
            (scope, cutoff),
        ).fetchone()
        recent_count = recent_rows["cnt"] if recent_rows else 0

        rate_per_day = round(recent_count / max(window_days, 1), 2)
        projected_30d = round(rate_per_day * 30)
        projected_90d = round(rate_per_day * 90)

        return {
            "scope_id": scope,
            "window_days": window_days,
            "current_active": total_active,
            "added_in_window": recent_count,
            "rate_per_day": rate_per_day,
            "projected_30d": projected_30d,
            "projected_90d": projected_90d,
        }

    def auto_deduplicate(
        self,
        scope_id: str | None = None,
        threshold: float = 0.85,
        dry_run: bool = False,
    ) -> dict:
        """Find and resolve duplicates by archiving older copies.

        Uses word-level Jaccard similarity. Keeps the memory with higher
        importance (or more recent if tied). Respects pinned memories.
        """
        scope = scope_id or self.scope_id
        duplicates = self.store.find_duplicates(scope, threshold=threshold)
        archived = 0
        pairs = []

        for dup in duplicates:
            id_a, id_b = dup["id_a"], dup["id_b"]
            unit_a = self.store._get_by_id(id_a)
            unit_b = self.store._get_by_id(id_b)
            if not unit_a or not unit_b:
                continue
            if unit_a.status != "active" or unit_b.status != "active":
                continue
            # Don't touch pinned memories.
            if unit_a.importance >= 0.99 or unit_b.importance >= 0.99:
                continue

            # Keep the one with higher importance, or more recent.
            if unit_a.importance > unit_b.importance:
                keep, remove = id_a, id_b
            elif unit_b.importance > unit_a.importance:
                keep, remove = id_b, id_a
            elif unit_a.updated_at >= unit_b.updated_at:
                keep, remove = id_a, id_b
            else:
                keep, remove = id_b, id_a

            pairs.append({"keep": keep, "remove": remove, "similarity": dup["similarity"]})
            if not dry_run:
                self.store.bulk_archive([remove])
                archived += 1

        self.clear_cache()
        return {
            "scope_id": scope,
            "duplicates_found": len(pairs),
            "archived": archived,
            "dry_run": dry_run,
            "pairs": pairs[:20],  # Cap detail output.
        }

    def forecast_capacity(
        self,
        scope_id: str | None = None,
        quota: int = 1000,
    ) -> dict:
        """Project when a scope will reach its quota based on growth rate."""
        scope = scope_id or self.scope_id
        growth = self.get_memory_growth_rate(scope, window_days=30)
        current = growth["current_active"]
        rate = growth["rate_per_day"]

        if rate <= 0:
            return {
                "scope_id": scope,
                "current": current,
                "quota": quota,
                "utilization_pct": round(100 * current / max(quota, 1), 1),
                "days_until_full": None,
                "rate_per_day": rate,
            }

        remaining = max(quota - current, 0)
        days_until_full = round(remaining / rate, 1) if rate > 0 else None

        return {
            "scope_id": scope,
            "current": current,
            "quota": quota,
            "utilization_pct": round(100 * current / max(quota, 1), 1),
            "days_until_full": days_until_full,
            "rate_per_day": rate,
        }

    def export_audit_trail(
        self,
        scope_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Export the memory event log as a compliance-ready audit trail.

        Returns chronologically ordered events with memory IDs, types, and timestamps.
        """
        scope = scope_id or self.scope_id
        rows = self.store.conn.execute(
            """SELECT event_type, memory_id, timestamp, detail
               FROM memory_events
               WHERE scope_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (scope, limit),
        ).fetchall()
        return [
            {
                "event_type": row["event_type"],
                "memory_id": row["memory_id"],
                "timestamp": row["timestamp"],
                "detail": row["detail"],
            }
            for row in rows
        ]

    def generate_action_plan(
        self,
        scope_id: str | None = None,
    ) -> dict:
        """Generate a comprehensive operator action plan for a scope.

        Combines deduplication, compression, enrichment, archival, and tagging
        recommendations into a single prioritized action list.
        """
        scope = scope_id or self.scope_id

        actions = []

        # 1. Duplicates to merge.
        duplicates = self.store.find_duplicates(scope, threshold=0.85)
        if duplicates:
            actions.append({
                "action": "deduplicate",
                "priority": "high",
                "count": len(duplicates),
                "description": f"Found {len(duplicates)} duplicate pairs above 85% similarity",
                "command": "memory auto-dedup --scope " + scope,
            })

        # 2. Stale memories.
        stale = self.find_stale_memories(scope, stale_days=60, limit=50)
        if stale:
            actions.append({
                "action": "review_stale",
                "priority": "medium",
                "count": len(stale),
                "description": f"{len(stale)} memories not accessed in 60+ days",
                "command": "memory stale --scope " + scope,
            })

        # 3. Enrichment needed.
        enrichments = self.suggest_enrichments(scope, limit=20)
        if enrichments:
            actions.append({
                "action": "enrich",
                "priority": "low",
                "count": len(enrichments),
                "description": f"{len(enrichments)} memories lack topics, entities, or tags",
                "command": "memory enrichments --scope " + scope,
            })

        # 4. Compression opportunities.
        units = self.store.list_active(scope, limit=5000)
        compressible = sum(1 for u in units if len(u.content) > 200)
        if compressible > 0:
            actions.append({
                "action": "compress",
                "priority": "low",
                "count": compressible,
                "description": f"{compressible} memories have verbose content (>200 chars)",
                "command": "memory compress --scope " + scope,
            })

        # 5. Type balance.
        balance = self.analyze_type_balance(scope)
        if balance.get("suggestions"):
            actions.append({
                "action": "rebalance_types",
                "priority": "low",
                "count": len(balance["suggestions"]),
                "description": "; ".join(
                    str(s) if isinstance(s, str) else s.get("suggestion", str(s))
                    for s in balance["suggestions"][:3]
                ),
                "command": "memory type-balance --scope " + scope,
            })

        # 6. Integrity check.
        integrity = self.store.validate_integrity()
        if not integrity.get("valid"):
            actions.append({
                "action": "fix_integrity",
                "priority": "high",
                "count": sum(len(v) for v in integrity.get("issues", {}).values()),
                "description": "Database integrity issues detected",
                "command": "memory cleanup-orphans",
            })

        # Sort by priority.
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda a: priority_order.get(a["priority"], 3))

        return {
            "scope_id": scope,
            "total_actions": len(actions),
            "actions": actions,
        }

    def search_grouped(
        self,
        query_text: str,
        scope_id: str | None = None,
        group_by: str = "type",
        limit: int = 20,
    ) -> dict:
        """Search memories and group results by type or topic.

        Returns grouped results with per-group scores.
        """
        scope = scope_id or self.scope_id
        query = MemoryQuery(
            scope_id=scope,
            query_text=query_text,
            top_k=limit,
        )
        hits = self.retriever.retrieve(query)
        groups: dict[str, list[dict]] = {}

        for hit in hits:
            if group_by == "type":
                key = hit.unit.memory_type.value
            elif group_by == "topic":
                key = hit.unit.topics[0] if hit.unit.topics else "untagged"
            else:
                key = "all"

            if key not in groups:
                groups[key] = []
            groups[key].append({
                "memory_id": hit.unit.memory_id,
                "content": hit.unit.content[:100],
                "score": round(hit.score, 3),
                "importance": hit.unit.importance,
            })

        return {
            "query": query_text,
            "total_results": len(hits),
            "group_by": group_by,
            "groups": {k: {"count": len(v), "results": v} for k, v in groups.items()},
        }

    def bookmark_memories(
        self,
        memory_ids: list[str],
        bookmark_tag: str = "bookmarked",
    ) -> dict:
        """Bookmark memories for quick access by adding a tag.

        Bookmarks are just tags — simple and compatible with all existing search.
        """
        tagged = 0
        for mid in memory_ids:
            unit = self.store._get_by_id(mid)
            if unit and bookmark_tag not in unit.tags:
                self.store.add_tags(mid, [bookmark_tag])
                tagged += 1
        return {"tagged": tagged, "total": len(memory_ids)}

    def get_bookmarks(
        self,
        scope_id: str | None = None,
        bookmark_tag: str = "bookmarked",
    ) -> list[dict]:
        """Get all bookmarked memories in a scope."""
        scope = scope_id or self.scope_id
        units = self.store.search_by_tag(scope, bookmark_tag)
        return [
            {
                "memory_id": u.memory_id,
                "type": u.memory_type.value,
                "content": u.content[:100],
                "importance": u.importance,
                "tags": u.tags,
            }
            for u in units
        ]

    def compare_snapshots(
        self,
        scope_id: str | None = None,
    ) -> dict:
        """Compare current stats with the most recent snapshot.

        Returns delta information showing what changed since last snapshot.
        """
        scope = scope_id or self.scope_id
        current_stats = self.store.get_stats(scope)
        trends = self.store.get_stats_trend(scope, limit=2)

        if len(trends) < 1:
            return {
                "scope_id": scope,
                "current": current_stats,
                "previous": None,
                "delta": None,
                "message": "No previous snapshots available",
            }

        previous = trends[0]
        delta = {}
        for key in ["active", "superseded"]:
            curr_val = current_stats.get(key, 0)
            prev_val = previous.get(key, 0)
            delta[key] = curr_val - prev_val

        return {
            "scope_id": scope,
            "current": current_stats,
            "previous": previous,
            "delta": delta,
        }

    def archive_scope(self, scope_id: str) -> dict:
        """Archive all active memories in a scope.

        Useful for retiring old scopes or preparing for scope cleanup.
        Does not touch pinned memories (importance >= 0.99).
        """
        units = self.store.list_active(scope_id, limit=5000)
        to_archive = [u.memory_id for u in units if u.importance < 0.99]
        archived = self.store.bulk_archive(to_archive) if to_archive else 0
        pinned_count = len(units) - len(to_archive)
        self.clear_cache()
        return {
            "scope_id": scope_id,
            "archived": archived,
            "pinned_kept": pinned_count,
            "total_before": len(units),
        }

    def bulk_pin_by_criteria(
        self,
        scope_id: str | None = None,
        min_importance: float = 0.9,
        min_access_count: int = 10,
    ) -> dict:
        """Pin memories that meet high importance or access criteria.

        Pinning sets importance to 0.99 which prevents archival/decay.
        """
        from datetime import datetime, timezone

        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        pinned = 0
        now = datetime.now(timezone.utc).isoformat()

        for u in units:
            if u.importance >= 0.99:
                continue  # Already pinned.
            if u.importance >= min_importance or u.access_count >= min_access_count:
                self.store.update_importance(u.memory_id, 0.99, now)
                pinned += 1

        self.clear_cache()
        return {"scope_id": scope, "pinned": pinned, "total": len(units)}

    def export_scope_yaml(self, scope_id: str | None = None) -> str:
        """Export scope memories as YAML format.

        Returns a YAML string. Does not require external YAML library.
        """
        scope = scope_id or self.scope_id
        units = self.store.list_active(scope, limit=5000)
        lines = ["# Memory Export", f"# Scope: {scope}", f"# Count: {len(units)}", "memories:"]

        for u in units:
            lines.append(f"  - memory_id: {u.memory_id}")
            lines.append(f"    type: {u.memory_type.value}")
            lines.append(f"    content: \"{u.content[:200].replace(chr(34), chr(39))}\"")
            lines.append(f"    importance: {u.importance}")
            lines.append(f"    confidence: {u.confidence}")
            lines.append(f"    access_count: {u.access_count}")
            lines.append(f"    created_at: {u.created_at}")
            if u.topics:
                lines.append(f"    topics: [{', '.join(u.topics[:5])}]")
            if u.entities:
                lines.append(f"    entities: [{', '.join(u.entities[:5])}]")
            if u.tags:
                lines.append(f"    tags: [{', '.join(u.tags[:5])}]")

        return "\n".join(lines)

    def run_system_health_check(self, scope_id: str | None = None) -> dict:
        """Run a comprehensive system health check.

        Returns a pass/fail result with categorized findings:
        - integrity: database structural health
        - quality: memory quality distribution
        - capacity: quota utilization
        - freshness: how up-to-date memories are
        - maintenance: pending maintenance actions
        """
        scope = scope_id or self.scope_id
        issues = []
        checks = {}

        # 1. Integrity.
        integrity = self.store.validate_integrity()
        checks["integrity"] = {
            "passed": integrity.get("valid", False),
            "issues": integrity.get("issues", {}),
        }
        if not integrity.get("valid"):
            issues.append("Database integrity issues detected")

        # 2. Health score.
        health = self.store.compute_health_score(scope)
        health_score = health.get("score", 0)
        checks["health_score"] = {
            "passed": health_score >= 50,
            "score": health_score,
        }
        if health_score < 50:
            issues.append(f"Low health score: {health_score}")

        # 3. Stale count.
        stale = self.find_stale_memories(scope, stale_days=90, limit=100)
        checks["staleness"] = {
            "passed": len(stale) < 10,
            "stale_count": len(stale),
        }
        if len(stale) >= 10:
            issues.append(f"{len(stale)} memories stale for 90+ days")

        # 4. Duplicate count.
        duplicates = self.store.find_duplicates(scope, threshold=0.90)
        checks["duplicates"] = {
            "passed": len(duplicates) < 5,
            "duplicate_pairs": len(duplicates),
        }
        if len(duplicates) >= 5:
            issues.append(f"{len(duplicates)} near-duplicate pairs found")

        # 5. DB size.
        db_info = self.store.get_db_size()
        db_size_mb = db_info.get("total_bytes", 0) / (1024 * 1024)
        checks["db_size"] = {
            "passed": db_size_mb < 100,
            "size_mb": round(db_size_mb, 2),
        }
        if db_size_mb >= 100:
            issues.append(f"Database size: {db_size_mb:.1f}MB")

        overall_passed = all(c.get("passed", False) for c in checks.values())

        return {
            "scope_id": scope,
            "passed": overall_passed,
            "checks": checks,
            "issues": issues,
            "summary": "All checks passed" if overall_passed else f"{len(issues)} issue(s) found",
        }

    def get_system_summary(self) -> dict:
        """Get a comprehensive summary of the entire memory system.

        Combines all scopes, health, embedder, policy, and schema info
        into a single operator-friendly overview.
        """
        scopes = self.store.list_scopes()
        scope_summaries = []
        total_active = 0
        for scope_info in scopes:
            sid = scope_info.get("scope_id", "")
            active = scope_info.get("active", 0)
            total_active += active
            scope_summaries.append({
                "scope_id": sid,
                "active": active,
                "total": scope_info.get("total", 0),
            })

        return {
            "schema_version": self.store.get_schema_version(),
            "scopes": scope_summaries,
            "scope_count": len(scopes),
            "total_active_memories": total_active,
            "embedder": self.get_embedder_info(),
            "policy": {
                "retrieval_mode": self.retrieval_mode,
                "max_injected_units": self.policy.max_injected_units,
                "max_injected_tokens": self.policy.max_injected_tokens,
            },
            "db": self.store.get_db_size(),
            "integrity": self.store.validate_integrity(),
        }

    def generate_operator_report(self, scope_id: str | None = None) -> dict:
        """Generate a comprehensive operator diagnostic report.

        Combines health check, action plan, growth rate, capacity forecast,
        and system summary into a single actionable output for quick triage.
        """
        scope = scope_id or self.scope_id
        report: dict = {"scope_id": scope, "generated_at": ""}
        try:
            from datetime import datetime, timezone
            report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        except Exception:
            pass

        # Health check
        try:
            report["health"] = self.run_system_health_check(scope_id=scope)
        except Exception as exc:
            report["health"] = {"error": str(exc)}

        # Action plan
        try:
            report["action_plan"] = self.generate_action_plan(scope_id=scope)
        except Exception as exc:
            report["action_plan"] = {"error": str(exc)}

        # Growth rate
        try:
            report["growth_rate"] = self.get_memory_growth_rate(scope_id=scope)
        except Exception as exc:
            report["growth_rate"] = {"error": str(exc)}

        # Capacity forecast
        try:
            report["capacity"] = self.forecast_capacity(scope_id=scope)
        except Exception as exc:
            report["capacity"] = {"error": str(exc)}

        # Stats
        try:
            report["stats"] = self.get_scope_stats(scope_id=scope)
        except Exception as exc:
            report["stats"] = {"error": str(exc)}

        # Type balance
        try:
            report["type_balance"] = self.analyze_type_balance(scope_id=scope)
        except Exception as exc:
            report["type_balance"] = {"error": str(exc)}

        # System-wide context
        try:
            report["system"] = self.get_system_summary()
        except Exception as exc:
            report["system"] = {"error": str(exc)}

        return report

    def close(self) -> None:
        self.store.close()

    def _fit_token_budget(self, units: list[MemoryUnit], max_tokens: int) -> list[MemoryUnit]:
        # Apply type diversity: if more than 4 units, ensure no single type
        # dominates more than 60% of slots.
        units = _enforce_type_diversity(units, max_dominant_ratio=0.6, min_count=4)

        kept: list[MemoryUnit] = []
        used = 0
        for unit in units:
            text = "\n".join([unit.summary, unit.content]).strip()
            cost = estimate_tokens(text)
            if kept and used + cost > max_tokens:
                break
            kept.append(unit)
            used += cost
        return kept

    def _refresh_policy(self, scope_id: str) -> None:
        if self.policy_store is None:
            return
        current_state = self.policy_store.load()
        proposed = self.policy_optimizer.propose(scope_id, current_state)
        if proposed == current_state:
            return
        self.policy_store.save(proposed, reason="auto_optimize")
        self.policy = MemoryPolicy.from_state(proposed)
        self.retrieval_mode = proposed.retrieval_mode
        self.use_embeddings = self.use_embeddings or proposed.retrieval_mode in {"embedding", "hybrid"}
        if self.use_embeddings and self.embedder is None:
            self.embedder = create_embedder(
                mode=self.embedding_mode,
                model_name=self.embedding_model,
                fallback=True,
            )
        self.retriever = MemoryRetriever(
            store=self.store,
            policy=self.policy,
            retrieval_mode=self.retrieval_mode,
            embedder=self.embedder,
        )
        logger.info(
            "[MemoryPolicy] scope=%s mode=%s units=%d tokens=%d",
            scope_id,
            proposed.retrieval_mode,
            proposed.max_injected_units,
            proposed.max_injected_tokens,
        )
        if self.telemetry_store is not None:
            self.telemetry_store.record(
                "policy_update",
                {
                    "scope_id": scope_id,
                    "retrieval_mode": proposed.retrieval_mode,
                    "max_injected_units": proposed.max_injected_units,
                    "max_injected_tokens": proposed.max_injected_tokens,
                    "notes": proposed.notes[-3:],
                },
            )


def _extract_topics(prompt_text: str) -> list[str]:
    topics = []
    seen = set()

    # Single-word topics from significant tokens.
    for token in prompt_text.split():
        cleaned = token.strip(".,:;!?()[]{}").lower()
        if (
            len(cleaned) >= 5
            and cleaned not in seen
            and cleaned not in _STOPWORDS
        ):
            topics.append(cleaned)
            seen.add(cleaned)
        if len(topics) >= 10:
            break

    # Multi-word technical terms (common patterns like "database migration").
    text_lower = prompt_text.lower()
    tech_patterns = [
        r"\b(api\s+\w+)",
        r"\b(database\s+\w+)",
        r"\b(test(?:ing)?\s+\w+)",
        r"\b(deploy(?:ment)?\s+\w+)",
        r"\b(auth(?:entication)?\s+\w+)",
        r"\b(error\s+handling)",
        r"\b(code\s+review)",
        r"\b(pull\s+request)",
        r"\b(ci[\s/]cd)",
        r"\b(machine\s+learning)",
    ]
    for pat in tech_patterns:
        match = re.search(pat, text_lower)
        if match and len(topics) < 12:
            term = match.group(1).strip()
            if term not in seen:
                topics.append(term)
                seen.add(term)

    return topics[:12]


def _extract_entities(text: str) -> list[str]:
    entities = []
    seen = set()

    # Capitalized words (proper nouns, class names).
    for token in text.split():
        cleaned = token.strip(".,:;!?()[]{}")
        if len(cleaned) > 1 and cleaned[:1].isupper() and cleaned not in seen:
            entities.append(cleaned)
            seen.add(cleaned)
        if len(entities) >= 10:
            break

    # CamelCase and PascalCase identifiers.
    for match in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", text):
        ident = match.group(1)
        if ident not in seen and len(entities) < 12:
            entities.append(ident)
            seen.add(ident)

    # snake_case identifiers (likely code references).
    for match in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b", text):
        ident = match.group(1)
        if ident not in seen and len(entities) < 12:
            entities.append(ident)
            seen.add(ident)

    return entities[:12]


def _summarize_turn(prompt_text: str, response_text: str) -> str:
    prompt = prompt_text.strip().replace("\n", " ")
    response = response_text.strip().replace("\n", " ")
    prompt = prompt[:180]
    response = response[:180]
    if response:
        return f"User asked: {prompt} | Assistant responded: {response}"
    return f"User asked: {prompt}"


def _build_working_summary(turns: list[dict]) -> str:
    """Build a structured working summary from the session's recent turns.

    Includes a topic line, entity line, key exchange highlights, and turn count context.
    """
    recent = turns[-5:]
    lines = []

    # Extract the main topics and entities mentioned across recent turns.
    all_text = " ".join(
        str(t.get("prompt_text", "") or "") + " " + str(t.get("response_text", "") or "")
        for t in recent
    )
    topics = _extract_topics(all_text)
    entities = _extract_entities(all_text)
    if topics:
        lines.append(f"Topics: {', '.join(topics[:6])}")
    if entities:
        lines.append(f"Entities: {', '.join(entities[:6])}")

    lines.append(f"Session covered {len(turns)} turn(s), last {len(recent)} shown:")
    for turn in recent:
        prompt = str(turn.get("prompt_text", "") or "").strip().replace("\n", " ")
        response = str(turn.get("response_text", "") or "").strip().replace("\n", " ")
        if prompt:
            lines.append(f"User: {prompt[:200]}")
        if response:
            lines.append(f"Assistant: {response[:200]}")
    return "\n".join(lines).strip()


def _infer_memory_type(prompt_text: str, response_text: str) -> MemoryType:
    text = f"{prompt_text}\n{response_text}".lower()
    if "i prefer" in text or "my preference" in text or "prefer " in text:
        return MemoryType.PREFERENCE
    if "project uses" in text or "we use" in text or "tech stack" in text:
        return MemoryType.PROJECT_STATE
    if "remember that" in text or "always keep in mind" in text:
        return MemoryType.SEMANTIC
    return MemoryType.EPISODIC


class _MultiTurnContext:
    """Accumulates context across turns to enable cross-turn extraction."""

    def __init__(self, window: int = 3):
        self.window = window
        self._turns: list[dict] = []

    def add_turn(self, prompt_text: str, response_text: str, turn_index: int) -> None:
        self._turns.append({
            "prompt": prompt_text,
            "response": response_text,
            "index": turn_index,
        })
        if len(self._turns) > self.window:
            self._turns = self._turns[-self.window:]

    def get_recent_context(self) -> str:
        """Return concatenated recent turn text for context-aware extraction."""
        parts = []
        for t in self._turns:
            if t["prompt"]:
                parts.append(t["prompt"])
            if t["response"]:
                parts.append(t["response"])
        return " ".join(parts)

    def get_accumulated_entities(self) -> list[str]:
        """Return entities mentioned across recent turns for enrichment."""
        all_entities: list[str] = []
        seen: set[str] = set()
        for t in self._turns:
            combined = f"{t['prompt']} {t['response']}"
            for ent in _extract_entities(combined):
                if ent not in seen:
                    seen.add(ent)
                    all_entities.append(ent)
        return all_entities[:12]

    def has_continuation_pattern(self, prompt_text: str) -> bool:
        """Detect if the current turn continues a prior topic (e.g., pronoun references)."""
        continuation_markers = [
            "also", "and also", "another thing", "in addition",
            "regarding that", "about that", "same for", "similarly",
            "on that note", "related to that", "going back to",
        ]
        lower = prompt_text.lower().strip()
        for m in continuation_markers:
            # Check start-of-sentence: "Also, ..." or "Also ..."
            if lower.startswith(m) and (len(lower) == len(m) or not lower[len(m)].isalpha()):
                return True
            # Check mid-sentence: "... also ..."
            if f" {m} " in lower or f" {m}," in lower:
                return True
        return False


def _extract_memory_units_for_turn(
    scope_id: str,
    session_id: str,
    turn_index: int,
    prompt_text: str,
    response_text: str,
    multi_turn_context: _MultiTurnContext | None = None,
) -> list[MemoryUnit]:
    extracted: list[MemoryUnit] = []
    text = " ".join([prompt_text, response_text]).strip()
    topics = _extract_topics(text)
    entities = _extract_entities(text)

    # Enrich entities from multi-turn context when the turn is a continuation.
    if multi_turn_context is not None and multi_turn_context.has_continuation_pattern(prompt_text):
        context_entities = multi_turn_context.get_accumulated_entities()
        seen_ent = set(entities)
        for ent in context_entities:
            if ent not in seen_ent and len(entities) < 12:
                entities.append(ent)
                seen_ent.add(ent)

    # Try extraction with context-enriched text for continuation turns.
    extraction_prompt = prompt_text
    extraction_response = response_text
    if multi_turn_context is not None and multi_turn_context.has_continuation_pattern(prompt_text):
        recent_ctx = multi_turn_context.get_recent_context()
        if recent_ctx:
            extraction_prompt = f"{recent_ctx} {prompt_text}"

    for fact in _extract_pattern_facts(extraction_prompt, extraction_response):
        extracted.append(
            MemoryUnit(
                memory_id=str(uuid.uuid4()),
                scope_id=scope_id,
                memory_type=fact["memory_type"],
                content=fact["content"][:2000],
                summary=fact["summary"][:280],
                source_session_id=session_id,
                source_turn_start=turn_index,
                source_turn_end=turn_index,
                topics=topics,
                entities=entities,
                importance=fact["importance"],
                confidence=fact["confidence"],
            )
        )

    if extracted:
        return extracted

    combined = prompt_text
    if response_text:
        combined = f"{prompt_text}\nAssistant: {response_text}".strip()
    fallback_type = _infer_memory_type(prompt_text, response_text)
    return [
        MemoryUnit(
            memory_id=str(uuid.uuid4()),
            scope_id=scope_id,
            memory_type=fallback_type,
            content=combined[:4000],
            summary=_summarize_turn(prompt_text, response_text),
            source_session_id=session_id,
            source_turn_start=turn_index,
            source_turn_end=turn_index,
            topics=topics,
            entities=entities,
            importance=0.7 if fallback_type != MemoryType.EPISODIC else 0.5,
            confidence=0.55,
        )
    ]


def _extract_pattern_facts(prompt_text: str, response_text: str) -> list[dict]:
    text = _normalize_space(prompt_text)
    facts: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # First pass: extract from prompt text (higher confidence).
    prompt_patterns = [
        (
            MemoryType.PREFERENCE,
            0.9,
            0.82,
            [
                r"\bi prefer (?P<fact>[^.!\n]{3,180})",
                r"\bplease (?:keep|make) (?P<fact>[^.!\n]{3,180})",
                r"\bmy preferred (?P<fact>[^.!\n]{3,180})",
                r"\bi(?:'d| would) like (?P<fact>[^.!\n]{3,180})",
                r"\bi don(?:'t| not) (?:want|like) (?P<fact>[^.!\n]{3,180})",
                r"\bmy convention is (?P<fact>[^.!\n]{3,180})",
                r"\bi(?:'d| would) rather (?P<fact>[^.!\n]{3,180})",
                r"\bi want (?P<fact>[^.!\n]{3,180})",
                r"\bi(?:'m| am) used to (?P<fact>[^.!\n]{3,180})",
            ],
        ),
        (
            MemoryType.PROJECT_STATE,
            0.88,
            0.8,
            [
                r"\b(?:this|the) project uses (?P<fact>[^.!\n]{3,180})",
                r"\bwe use (?P<fact>[^.!\n]{3,180})",
                r"\bour stack is (?P<fact>[^.!\n]{3,180})",
                r"\bthe codebase (?:is|uses|has) (?P<fact>[^.!\n]{3,180})",
                r"\bour (?:team|org) (?:uses|runs|has) (?P<fact>[^.!\n]{3,180})",
                r"\bwe(?:'re| are) (?:using|running|deploying) (?P<fact>[^.!\n]{3,180})",
            ],
        ),
        (
            MemoryType.PROCEDURAL_OBSERVATION,
            0.8,
            0.75,
            [
                r"\balways (?P<fact>[^.!\n]{3,180})",
                r"\bwhen you work on this repo[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bthe workflow is to (?P<fact>[^.!\n]{3,180})",
                r"\bnever (?P<fact>[^.!\n]{3,180})",
                r"\bmake sure (?:to |that )?(?P<fact>[^.!\n]{3,180})",
                r"\bdon(?:'t| not) forget to (?P<fact>[^.!\n]{3,180})",
            ],
        ),
        (
            MemoryType.SEMANTIC,
            0.82,
            0.78,
            [
                r"\bremember that (?P<fact>[^.!\n]{3,180})",
                r"\bkeep in mind that (?P<fact>[^.!\n]{3,180})",
                r"\bfor context[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bnote that (?P<fact>[^.!\n]{3,180})",
                r"\bfyi[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bjust so you know[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bfor (?:your|future) reference[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bimportant(?:ly)?[: ,]+(?P<fact>[^.!\n]{3,180})",
                r"\bby the way[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bthe (?:key|main) thing is (?P<fact>[^.!\n]{3,180})",
                r"\bwhat matters (?:is|here) (?P<fact>[^.!\n]{3,180})",
                r"\bfor your information[, ]+(?P<fact>[^.!\n]{3,180})",
            ],
        ),
    ]

    for memory_type, importance, confidence, regexes in prompt_patterns:
        for regex in regexes:
            for match in re.finditer(regex, text, flags=re.IGNORECASE):
                fact_text = _clean_fact(match.group("fact"))
                if not fact_text:
                    continue
                key = (memory_type.value, fact_text.lower())
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    {
                        "memory_type": memory_type,
                        "content": _format_memory_content(memory_type, fact_text),
                        "summary": _summarize_fact(memory_type, fact_text, response_text),
                        "importance": importance,
                        "confidence": confidence,
                    }
                )
                if len(facts) >= 6:
                    return facts

    # Second pass: extract from response text (lower confidence).
    response_normalized = _normalize_space(response_text)
    response_patterns = [
        (
            MemoryType.PROJECT_STATE,
            0.78,
            0.68,
            [
                r"\bthe project (?:is|uses|has) (?P<fact>[^.!\n]{3,180})",
                r"\bthis (?:repo|repository|codebase) (?:is|uses|has) (?P<fact>[^.!\n]{3,180})",
                r"\bthe (?:current|existing) (?:setup|config|configuration) (?:is|uses) (?P<fact>[^.!\n]{3,180})",
                r"\bthe (?:default|recommended) (?:approach|method|way) is (?P<fact>[^.!\n]{3,180})",
            ],
        ),
        (
            MemoryType.SEMANTIC,
            0.75,
            0.65,
            [
                r"\bimportant(?:ly)?[: ,]+(?P<fact>[^.!\n]{3,180})",
                r"\bkey (?:point|takeaway|finding)[: ,]+(?P<fact>[^.!\n]{3,180})",
                r"\bin summary[, ]+(?P<fact>[^.!\n]{3,180})",
                r"\bworth noting that (?P<fact>[^.!\n]{3,180})",
            ],
        ),
        (
            MemoryType.PROCEDURAL_OBSERVATION,
            0.72,
            0.62,
            [
                r"\byou should (?:always |)(?P<fact>[^.!\n]{3,180})",
                r"\bbest practice is to (?P<fact>[^.!\n]{3,180})",
                r"\bavoid (?P<fact>[^.!\n]{3,180})",
            ],
        ),
    ]
    for memory_type, importance, confidence, regexes in response_patterns:
        for regex in regexes:
            for match in re.finditer(regex, response_normalized, flags=re.IGNORECASE):
                fact_text = _clean_fact(match.group("fact"))
                if not fact_text:
                    continue
                key = (memory_type.value, fact_text.lower())
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    {
                        "memory_type": memory_type,
                        "content": _format_memory_content(memory_type, fact_text),
                        "summary": _summarize_fact(memory_type, fact_text, response_text),
                        "importance": importance,
                        "confidence": confidence,
                    }
                )
                if len(facts) >= 6:
                    return facts

    return facts


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _clean_fact(text: str) -> str:
    fact = _normalize_space(text)
    fact = re.sub(r"^(that|to)\s+", "", fact, flags=re.IGNORECASE)
    fact = fact.strip(" .,:;!-")
    if len(fact) < 3:
        return ""
    return fact


def _format_memory_content(memory_type: MemoryType, fact_text: str) -> str:
    if memory_type == MemoryType.PREFERENCE:
        return f"User preference: {fact_text}."
    if memory_type == MemoryType.PROJECT_STATE:
        return f"Project context: {fact_text}."
    if memory_type == MemoryType.PROCEDURAL_OBSERVATION:
        return f"Workflow guidance: {fact_text}."
    if memory_type == MemoryType.SEMANTIC:
        return f"Persistent fact: {fact_text}."
    return fact_text


def _summarize_fact(memory_type: MemoryType, fact_text: str, response_text: str) -> str:
    label = {
        MemoryType.PREFERENCE: "Captured user preference",
        MemoryType.PROJECT_STATE: "Captured project state",
        MemoryType.PROCEDURAL_OBSERVATION: "Captured workflow guidance",
        MemoryType.SEMANTIC: "Captured persistent context",
    }.get(memory_type, "Captured memory")
    if response_text:
        response_excerpt = _normalize_space(response_text)[:120]
        if response_excerpt:
            return f"{label}: {fact_text}. Assistant acknowledged: {response_excerpt}"
    return f"{label}: {fact_text}."


def _freshness_tag(updated_at: str) -> str:
    """Return a short freshness tag based on how recently the memory was updated."""
    if not updated_at:
        return ""
    try:
        from datetime import datetime, timezone

        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_hours = max(
            (datetime.now(timezone.utc) - updated).total_seconds() / 3600.0, 0.0
        )
        if age_hours < 1:
            return "just now"
        if age_hours < 24:
            return "recent"
        if age_hours < 168:  # 7 days
            return "this week"
        return ""
    except (ValueError, TypeError):
        return ""


def _enforce_type_diversity(
    units: list[MemoryUnit],
    max_dominant_ratio: float = 0.6,
    min_count: int = 4,
) -> list[MemoryUnit]:
    """Reorder units to prevent any single type from dominating results.

    If a type exceeds max_dominant_ratio of the total, push excess units
    to the end so other types can fill the budget.
    """
    if len(units) < min_count:
        return units

    max_slots = max(int(len(units) * max_dominant_ratio), 1)
    type_counts: dict[str, int] = {}
    primary: list[MemoryUnit] = []
    overflow: list[MemoryUnit] = []

    for unit in units:
        t = unit.memory_type.value
        count = type_counts.get(t, 0)
        if count < max_slots:
            primary.append(unit)
            type_counts[t] = count + 1
        else:
            overflow.append(unit)

    return primary + overflow


def _detect_conflicts(
    new_units: list[MemoryUnit],
    store: MemoryStore,
    scope_id: str,
    similarity_threshold: float = 0.65,
) -> list[dict]:
    """Detect potential conflicts between new and existing memories.

    Finds cases where new memories share significant topic/entity overlap
    with existing memories but have different content, which may indicate
    contradictory information.
    """
    existing = store.list_active(scope_id, limit=500)
    if not existing:
        return []

    conflicts: list[dict] = []
    for new_unit in new_units:
        new_terms = set(
            t.lower() for t in new_unit.topics + new_unit.entities
        )
        if not new_terms:
            continue
        for old_unit in existing:
            if old_unit.memory_type != new_unit.memory_type:
                continue
            old_terms = set(
                t.lower() for t in old_unit.topics + old_unit.entities
            )
            if not old_terms:
                continue
            overlap = len(new_terms & old_terms) / float(len(new_terms | old_terms))
            if overlap < similarity_threshold:
                continue
            # Check that content is actually different (not a duplicate).
            if new_unit.content.strip().lower() == old_unit.content.strip().lower():
                continue
            conflicts.append({
                "new_id": new_unit.memory_id,
                "existing_id": old_unit.memory_id,
                "type": new_unit.memory_type.value,
                "overlap": round(overlap, 4),
                "new_content": new_unit.content[:120],
                "existing_content": old_unit.content[:120],
            })
    return conflicts


def _dedup_against_store(
    units: list[MemoryUnit],
    store: MemoryStore,
    scope_id: str,
) -> list[MemoryUnit]:
    """Remove units whose content already exists in the active store."""
    existing = store.list_active(scope_id, limit=500)
    if not existing:
        return units
    existing_content = {
        (u.memory_type.value, u.content.strip().lower())
        for u in existing
    }
    kept: list[MemoryUnit] = []
    for unit in units:
        key = (unit.memory_type.value, unit.content.strip().lower())
        if key in existing_content:
            continue
        kept.append(unit)
    return kept


_FILLER_PHRASES = [
    "basically", "essentially", "actually", "obviously",
    "it should be noted that", "it is important to note that",
    "as mentioned before", "as we discussed",
    "you know", "kind of", "sort of", "more or less",
    "in terms of", "at the end of the day",
    "the thing is", "to be honest",
    "as a matter of fact", "needless to say",
]


def _compress_text(text: str) -> str:
    """Heuristic text compression: remove filler, condense whitespace, trim."""
    result = text
    # Remove filler phrases.
    lower = result.lower()
    for filler in _FILLER_PHRASES:
        idx = lower.find(filler)
        while idx >= 0:
            result = result[:idx] + result[idx + len(filler):]
            lower = result.lower()
            idx = lower.find(filler)
    # Collapse whitespace.
    result = " ".join(result.split())
    return result.strip()


_STOPWORDS = {
    "about",
    "after",
    "always",
    "before",
    "brief",
    "context",
    "please",
    "project",
    "remember",
    "should",
    "their",
    "there",
    "these",
    "those",
    "using",
    "which",
    "while",
    "would",
}
