from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryPromotionCriteria:
    min_query_overlap_delta: float = 0.0
    min_continuation_overlap_delta: float = 0.0
    min_response_overlap_delta: float = 0.0
    min_specificity_delta: float = -0.05
    min_focus_score_delta: float = 0.0
    min_value_density_delta: float = 0.0
    min_grounding_score_delta: float = 0.0
    min_coverage_score_delta: float = 0.0
    min_sample_count: int = 10
    max_zero_retrieval_increase: int = 2


def should_promote(comparison: dict, criteria: MemoryPromotionCriteria | None = None) -> bool:
    active = criteria or MemoryPromotionCriteria()
    if int(comparison.get("sample_count", 0)) < active.min_sample_count:
        return False
    if float(comparison.get("avg_query_overlap_delta", 0.0)) < active.min_query_overlap_delta:
        return False
    if float(comparison.get("avg_continuation_overlap_delta", 0.0)) < active.min_continuation_overlap_delta:
        return False
    if float(comparison.get("avg_response_overlap_delta", 0.0)) < active.min_response_overlap_delta:
        return False
    if float(comparison.get("avg_specificity_delta", 0.0)) < active.min_specificity_delta:
        return False
    if float(comparison.get("avg_focus_score_delta", 0.0)) < active.min_focus_score_delta:
        return False
    if float(comparison.get("avg_value_density_delta", 0.0)) < active.min_value_density_delta:
        return False
    if float(comparison.get("avg_grounding_score_delta", 0.0)) < active.min_grounding_score_delta:
        return False
    if float(comparison.get("avg_coverage_score_delta", 0.0)) < active.min_coverage_score_delta:
        return False
    # Block promotion if candidate causes too many more zero-retrieval samples.
    zero_delta = int(comparison.get("zero_retrieval_delta", 0))
    if zero_delta > active.max_zero_retrieval_increase:
        return False
    return bool(comparison.get("candidate_beats_baseline", False))
