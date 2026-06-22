"""Policy and safety controls for memory routing."""

from __future__ import annotations

import os
from typing import Optional, Protocol


class BenchmarkConfigProtocol(Protocol):
    """Config protocol for benchmark_safe support."""

    benchmark_safe: bool


class BenchmarkSafeGuard:
    """Hard guard to force baseline episodic behavior in benchmark mode."""

    def __init__(self, config: object = None):
        self._config_enabled: bool = False
        if config is not None:
            # Support both direct config.benchmark_safe and config.router.benchmark_safe
            router_cfg = getattr(config, "router", None)
            if router_cfg is not None:
                self._config_enabled = bool(
                    getattr(router_cfg, "benchmark_safe", False)
                )
            else:
                self._config_enabled = bool(getattr(config, "benchmark_safe", False))

    @property
    def enabled(self) -> bool:
        """Return whether benchmark-safe mode is enabled."""
        return (
            os.getenv("OMNI_MEMORY_BENCHMARK_SAFE", "0") == "1" or self._config_enabled
        )

    def should_force_baseline(
        self, query_params: Optional[dict[str, object]] = None
    ) -> bool:
        """Return True when baseline episodic path must be enforced."""
        if self.enabled:
            return True
        if query_params and bool(query_params.get("benchmark_safe", False)):
            return True
        return False


class RetentionPolicy:
    """Policy guard for allowed actions on episodic and semantic stores."""

    FORBIDDEN_EPISODIC: set[str] = {"delete", "overwrite", "truncate"}

    @classmethod
    def validate_action(cls, store_type: str, action: str) -> bool:
        """Validate whether a store action is allowed."""
        if store_type == "episodic" and action in cls.FORBIDDEN_EPISODIC:
            raise PermissionError(f"Action '{action}' is forbidden on episodic store.")
        return True


class CircuitBreaker:
    """Automatic fallback guard based on recent evaluation scores."""

    def __init__(
        self,
        baseline_score: float = 0.0,
        tolerance: float = 0.02,
        window_size: int = 20,
    ):
        self.baseline_score: float = baseline_score
        self.tolerance: float = tolerance
        self.window_size: int = window_size
        self.recent_scores: list[float] = []
        self.tripped: bool = False

    def record(self, score: float) -> None:
        """Record a score and trip when rolling average degrades."""
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores = self.recent_scores[-self.window_size :]

        if len(self.recent_scores) >= self.window_size:
            average = sum(self.recent_scores) / len(self.recent_scores)
            if average < self.baseline_score - self.tolerance:
                self.tripped = True

    def should_force_baseline(self) -> bool:
        """Return True if fallback to baseline is currently required."""
        return self.tripped

    def reset(self) -> None:
        """Reset breaker status and accumulated scores."""
        self.tripped = False
        self.recent_scores.clear()
