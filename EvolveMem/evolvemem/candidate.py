from __future__ import annotations

from dataclasses import replace

from .policy_store import MemoryPolicyState


def generate_policy_candidates(current: MemoryPolicyState) -> list[MemoryPolicyState]:
    """Generate a small bounded candidate set around the current live policy.

    Varies retrieval mode, injection budget, and retrieval weight parameters
    to produce diverse but bounded candidates.
    """
    candidates: list[MemoryPolicyState] = []
    seen: set[tuple] = set()

    # Phase 1: Vary mode, units, and tokens (original grid).
    for retrieval_mode in _candidate_modes(current.retrieval_mode):
        for units in _candidate_units(current.max_injected_units):
            for tokens in _candidate_tokens(current.max_injected_tokens):
                candidate = replace(
                    current,
                    retrieval_mode=retrieval_mode,
                    max_injected_units=units,
                    max_injected_tokens=tokens,
                    notes=list(current.notes) + ["candidate_generated"],
                )
                key = _candidate_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)

    # Phase 2: Vary weight parameters around current values.
    weight_variants = _candidate_weight_variants(current)
    for variant in weight_variants:
        key = _candidate_key(variant)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(variant)

    return candidates


def _candidate_key(candidate: MemoryPolicyState) -> tuple:
    return (
        candidate.retrieval_mode,
        candidate.max_injected_units,
        candidate.max_injected_tokens,
        round(candidate.keyword_weight, 2),
        round(candidate.metadata_weight, 2),
        round(candidate.importance_weight, 2),
        round(candidate.recency_weight, 2),
    )


def _candidate_modes(current_mode: str) -> list[str]:
    modes = [current_mode]
    if current_mode == "keyword":
        modes.append("hybrid")
    elif current_mode == "hybrid":
        modes.extend(["keyword", "embedding"])
    elif current_mode == "embedding":
        modes.append("hybrid")
    return modes


def _candidate_units(current_units: int) -> list[int]:
    values = {max(4, current_units - 2), current_units, min(10, current_units + 2)}
    return sorted(values)


def _candidate_tokens(current_tokens: int) -> list[int]:
    values = {max(400, current_tokens - 200), current_tokens, min(1400, current_tokens + 200)}
    return sorted(values)


def _candidate_weight_variants(current: MemoryPolicyState) -> list[MemoryPolicyState]:
    """Generate weight-variant candidates by perturbing retrieval weights."""
    variants: list[MemoryPolicyState] = []

    # Vary keyword weight.
    for delta in [-0.2, 0.2]:
        kw = _clamp(current.keyword_weight + delta, 0.3, 2.0)
        if round(kw, 2) != round(current.keyword_weight, 2):
            variants.append(replace(
                current,
                keyword_weight=round(kw, 2),
                notes=list(current.notes) + ["candidate_generated", "keyword_weight_variant"],
            ))

    # Vary metadata weight.
    for delta in [-0.15, 0.15]:
        mw = _clamp(current.metadata_weight + delta, 0.1, 1.0)
        if round(mw, 2) != round(current.metadata_weight, 2):
            variants.append(replace(
                current,
                metadata_weight=round(mw, 2),
                notes=list(current.notes) + ["candidate_generated", "metadata_weight_variant"],
            ))

    # Vary importance weight.
    for delta in [-0.15, 0.15]:
        iw = _clamp(current.importance_weight + delta, 0.1, 1.0)
        if round(iw, 2) != round(current.importance_weight, 2):
            variants.append(replace(
                current,
                importance_weight=round(iw, 2),
                notes=list(current.notes) + ["candidate_generated", "importance_weight_variant"],
            ))

    # Vary recency weight.
    for delta in [-0.1, 0.1]:
        rw = _clamp(current.recency_weight + delta, 0.0, 0.8)
        if round(rw, 2) != round(current.recency_weight, 2):
            variants.append(replace(
                current,
                recency_weight=round(rw, 2),
                notes=list(current.notes) + ["candidate_generated", "recency_weight_variant"],
            ))

    return variants


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
