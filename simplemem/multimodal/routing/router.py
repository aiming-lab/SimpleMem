"""Memory routing logic for episodic and semantic retrieval paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union

from .features import compute_route_features


class TrainedRouterProtocol(Protocol):
    """Protocol for Phase 2 trained router compatibility."""

    def predict(
        self, query: str, features: dict[str, float]
    ) -> dict[str, Union[str, float]]:
        """Predict route decision and confidence."""
        ...


@dataclass
class RouteResult:
    decision: str
    episodic_results: list[float]
    semantic_results: list[float]
    features: dict[str, float]
    confidence: float


class MemoryRouter:
    """Phase 1 heuristic router with a Phase 2 trained placeholder."""

    def __init__(
        self,
        gini_threshold: float = 0.65,
        top1_threshold: float = 0.75,
        gap_threshold: float = 0.15,
        episodic_margin: float = 0.1,
        close_margin: float = 0.05,
        shadow_mode: bool = True,
    ):
        self.gini_threshold: float = gini_threshold
        self.top1_threshold: float = top1_threshold
        self.gap_threshold: float = gap_threshold
        self.episodic_margin: float = episodic_margin
        self.close_margin: float = close_margin
        self.shadow_mode: bool = shadow_mode
        self._trained_router: Optional[TrainedRouterProtocol] = None

    def route(
        self,
        query: str,
        semantic_scores: list[float],
        episodic_scores: list[float],
        benchmark_safe: bool = False,
    ) -> RouteResult:
        """Route a query between baseline episodic, semantic, and hybrid paths."""
        features = compute_route_features(semantic_scores, episodic_scores)

        if benchmark_safe:
            return RouteResult(
                decision="BASELINE_EPISODIC",
                episodic_results=episodic_scores,
                semantic_results=semantic_scores if self.shadow_mode else [],
                features=features,
                confidence=1.0,
            )

        if self._trained_router is not None:
            return self._trained_route(
                query, features, semantic_scores, episodic_scores
            )

        return self._heuristic_route(features, semantic_scores, episodic_scores)

    def _heuristic_route(
        self,
        feat: dict[str, float],
        sem_scores: list[float],
        epi_scores: list[float],
    ) -> RouteResult:
        """Phase 1 conservative heuristic routing."""
        if feat["top1_epi"] >= feat["top1_sem"] + self.episodic_margin:
            return RouteResult(
                decision="BASELINE_EPISODIC",
                episodic_results=epi_scores,
                semantic_results=sem_scores if self.shadow_mode else [],
                features=feat,
                confidence=0.8,
            )

        sem_easy = (
            feat["gini_sem"] >= self.gini_threshold
            and feat["top1_sem"] >= self.top1_threshold
            and feat["gap_sem"] >= self.gap_threshold
        )
        if not sem_easy:
            return RouteResult(
                decision="BASELINE_EPISODIC",
                episodic_results=epi_scores,
                semantic_results=sem_scores if self.shadow_mode else [],
                features=feat,
                confidence=0.6,
            )

        if feat["top1_epi"] >= feat["top1_sem"] - self.close_margin:
            return RouteResult(
                decision="HYBRID",
                episodic_results=epi_scores,
                semantic_results=sem_scores,
                features=feat,
                confidence=0.7,
            )

        return RouteResult(
            decision="SEMANTIC_ONLY",
            episodic_results=epi_scores if self.shadow_mode else [],
            semantic_results=sem_scores,
            features=feat,
            confidence=0.75,
        )

    def _trained_route(
        self,
        query: str,
        features: dict[str, float],
        sem_scores: list[float],
        epi_scores: list[float],
    ) -> RouteResult:
        """Phase 2 placeholder for trained routing."""
        if self._trained_router is None:
            raise RuntimeError("Trained router is not initialized")

        prediction = self._trained_router.predict(query, features)
        decision = prediction.get("decision", "BASELINE_EPISODIC")
        confidence_value = prediction.get("confidence", 0.5)

        if not isinstance(decision, str):
            decision = "BASELINE_EPISODIC"

        if not isinstance(confidence_value, (int, float)):
            confidence_value = 0.5

        return RouteResult(
            decision=decision,
            episodic_results=epi_scores,
            semantic_results=sem_scores,
            features=features,
            confidence=float(confidence_value),
        )
