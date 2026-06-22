"""Feature extraction for memory routing decisions."""

from __future__ import annotations

import math


def gini_coefficient(scores: list[float]) -> float:
    """
    Compute the Gini coefficient for retrieval score skewness.

    High Gini means scores are concentrated in a few items.
    """
    if not scores or all(score == 0 for score in scores):
        return 0.0

    ordered_scores = sorted(scores)
    count = len(ordered_scores)
    if count == 1:
        return 0.0

    total = sum(ordered_scores)
    if total == 0:
        return 0.0

    weighted_cumulative = sum(
        (index + 1) * value for index, value in enumerate(ordered_scores)
    )
    return (2.0 * weighted_cumulative) / (count * total) - (count + 1.0) / count


def score_entropy(scores: list[float]) -> float:
    """Compute Shannon entropy (bits) of score distribution."""
    if not scores:
        return 0.0

    total = sum(scores)
    if total == 0:
        return 0.0

    entropy = 0.0
    for score in scores:
        if score <= 0:
            continue
        probability = score / total
        entropy -= probability * math.log2(probability)

    return entropy


def compute_route_features(
    semantic_scores: list[float],
    episodic_scores: list[float],
) -> dict[str, float]:
    """Compute routing features from semantic and episodic score lists."""

    def _top1(scores: list[float]) -> float:
        return scores[0] if scores else 0.0

    def _gap(scores: list[float]) -> float:
        return scores[0] - scores[1] if len(scores) > 1 else 0.0

    def _mean(scores: list[float]) -> float:
        return (sum(scores) / len(scores)) if scores else 0.0

    return {
        "gini_sem": gini_coefficient(semantic_scores),
        "gini_epi": gini_coefficient(episodic_scores),
        "entropy_sem": score_entropy(semantic_scores),
        "entropy_epi": score_entropy(episodic_scores),
        "top1_sem": _top1(semantic_scores),
        "top1_epi": _top1(episodic_scores),
        "gap_sem": _gap(semantic_scores),
        "gap_epi": _gap(episodic_scores),
        "mean_sem": _mean(semantic_scores),
        "mean_epi": _mean(episodic_scores),
    }
