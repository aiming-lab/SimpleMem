from __future__ import annotations

from dataclasses import dataclass, field

from .policy_store import MemoryPolicyState


@dataclass
class MemoryPolicy:
    max_injected_units: int = 6
    max_injected_tokens: int = 800
    recent_bonus_hours: int = 72
    keyword_weight: float = 1.0
    metadata_weight: float = 0.45
    importance_weight: float = 0.5
    recency_weight: float = 0.3
    type_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "working_summary": 1.2,
            "project_state": 1.1,
            "preference": 1.0,
            "semantic": 1.0,
            "episodic": 0.8,
            "procedural_observation": 0.9,
        }
    )

    @classmethod
    def from_state(cls, state: MemoryPolicyState) -> "MemoryPolicy":
        kwargs: dict = dict(
            max_injected_units=state.max_injected_units,
            max_injected_tokens=state.max_injected_tokens,
            recent_bonus_hours=state.recent_bonus_hours,
            keyword_weight=state.keyword_weight,
            metadata_weight=state.metadata_weight,
            importance_weight=state.importance_weight,
            recency_weight=state.recency_weight,
        )
        if state.type_boosts:
            kwargs["type_boosts"] = dict(state.type_boosts)
        return cls(**kwargs)

    @classmethod
    def from_profile(cls, profile: str) -> "MemoryPolicy":
        """Create a policy from a named profile.

        Profiles:
        - 'balanced': default weights (good for general use)
        - 'recall': more results, broader matching
        - 'precision': fewer results, stricter matching
        - 'recent': heavily weight recent memories
        """
        profiles = {
            "balanced": cls(),
            "recall": cls(
                max_injected_units=10,
                max_injected_tokens=1200,
                keyword_weight=0.8,
                metadata_weight=0.6,
                importance_weight=0.3,
                recency_weight=0.2,
            ),
            "precision": cls(
                max_injected_units=4,
                max_injected_tokens=500,
                keyword_weight=1.2,
                metadata_weight=0.3,
                importance_weight=0.7,
                recency_weight=0.1,
            ),
            "recent": cls(
                max_injected_units=6,
                max_injected_tokens=800,
                keyword_weight=0.7,
                metadata_weight=0.3,
                importance_weight=0.3,
                recency_weight=0.8,
                recent_bonus_hours=24,
            ),
        }
        return profiles.get(profile, cls())
