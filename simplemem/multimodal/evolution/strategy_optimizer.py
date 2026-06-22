"""
Adaptive Retrieval Strategy Optimizer for Self-Evolving Omni-Memory.

Thompson Sampling-based routing policy that learns which retrieval
strategy works best for each query type.

Inspired by:
- MemEvolve (arXiv 2512.18746): Modular memory architecture evolution
- SimpleMem (arXiv 2601.02553): Adaptive query-aware retrieval depth
- Memento (arXiv 2508.16153): Neural case-selection policy for memory

Core mechanism: Beta posteriors per (query_type, strategy) pair.
Selection via Thompson Sampling gives O(log T) regret with natural
exploration-exploitation balance.
"""

import logging
import json
import time
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BetaPosterior:
    """Beta distribution posterior for Thompson Sampling."""
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self) -> float:
        """Sample from Beta(alpha, beta) via gamma variates."""
        a = max(self.alpha, 0.01)
        b = max(self.beta, 0.01)
        x = random.gammavariate(a, 1.0)
        y = random.gammavariate(b, 1.0)
        if x + y == 0:
            return 0.5
        return x / (x + y)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def update_success(self, weight: float = 1.0) -> None:
        self.alpha += weight

    def update_failure(self, weight: float = 1.0) -> None:
        self.beta += weight

    def decay(self, factor: float = 0.99) -> None:
        """Decay toward uniform prior for non-stationary adaptation."""
        self.alpha = max(1.0, self.alpha * factor)
        self.beta = max(1.0, self.beta * factor)

    def to_dict(self) -> Dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BetaPosterior":
        return cls(alpha=data.get("alpha", 1.0), beta=data.get("beta", 1.0))


class StrategyOptimizer:
    """
    Adaptive retrieval strategy selector using Thompson Sampling.

    Strategies:
    - parametric: Fast recall from distilled memory (cheapest)
    - vector: Standard vector similarity via pyramid retriever
    - graph: Knowledge graph traversal for multi-hop queries
    - hybrid: Combined vector + graph (most thorough)
    """

    def __init__(self, config=None, storage_path: Optional[str] = None):
        from simplemem.multimodal.evolution.evolution_config import StrategyOptimizerConfig
        self.config = config or StrategyOptimizerConfig()
        self.storage_path = Path(storage_path or "./omni_memory_data/evolution/strategy")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._posteriors: Dict[str, Dict[str, BetaPosterior]] = {}

        for qtype in self.config.query_types:
            self._posteriors[qtype] = {
                strategy: BetaPosterior(
                    alpha=self.config.prior_alpha,
                    beta=self.config.prior_beta,
                )
                for strategy in self.config.strategies
            }

        self._set_informative_priors()

        self._selection_count: Dict[str, Dict[str, int]] = {
            qtype: {s: 0 for s in self.config.strategies}
            for qtype in self.config.query_types
        }

        self._state_file = self.storage_path / "strategy_optimizer_state.json"
        self._load_state()

        logger.info("StrategyOptimizer initialized")

    def _set_informative_priors(self) -> None:
        """Encode domain knowledge as informative priors."""
        priors = {
            "factual": {"parametric": (3, 1), "vector": (2, 1), "graph": (1, 1), "hybrid": (2, 1)},
            "temporal": {"parametric": (1, 2), "vector": (3, 1), "graph": (1, 1), "hybrid": (2, 1)},
            "reasoning": {"parametric": (1, 2), "vector": (1, 1), "graph": (2, 1), "hybrid": (3, 1)},
            "cross_modal": {"parametric": (1, 2), "vector": (3, 1), "graph": (1, 1), "hybrid": (2, 1)},
            "entity_centric": {"parametric": (1, 1), "vector": (1, 1), "graph": (3, 1), "hybrid": (2, 1)},
        }

        for qtype, strategies in priors.items():
            if qtype in self._posteriors:
                for strategy, (a, b) in strategies.items():
                    if strategy in self._posteriors[qtype]:
                        self._posteriors[qtype][strategy].alpha = float(a)
                        self._posteriors[qtype][strategy].beta = float(b)

    def select_strategy(self, query_type: str) -> str:
        """Select retrieval strategy via Thompson Sampling."""
        if query_type not in self._posteriors:
            query_type = "factual"

        posteriors = self._posteriors[query_type]

        if random.random() < self.config.min_exploration_rate:
            selected = random.choice(self.config.strategies)
        else:
            samples = {
                strategy: posterior.sample()
                for strategy, posterior in posteriors.items()
            }
            selected = max(samples, key=samples.get)

        if query_type in self._selection_count:
            self._selection_count[query_type][selected] = (
                self._selection_count[query_type].get(selected, 0) + 1
            )

        logger.debug(
            f"Strategy selected: {selected} for query_type={query_type} "
            f"(means: {', '.join(f'{s}={p.mean():.2f}' for s, p in posteriors.items())})"
        )

        return selected

    def get_strategy_ranking(self, query_type: str) -> List[Tuple[str, float]]:
        """Get strategies ranked by expected success rate."""
        if query_type not in self._posteriors:
            query_type = "factual"

        ranking = [
            (strategy, posterior.mean())
            for strategy, posterior in self._posteriors[query_type].items()
        ]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def record_outcome(
        self,
        query_type: str,
        strategy: str,
        success: bool,
        quality: float = 0.5,
    ) -> None:
        """Update posterior based on strategy outcome."""
        if query_type not in self._posteriors:
            query_type = "factual"

        if strategy not in self._posteriors.get(query_type, {}):
            return

        posterior = self._posteriors[query_type][strategy]

        if success:
            posterior.update_success(weight=quality)
        else:
            posterior.update_failure(weight=(1.0 - quality))

        for s, p in self._posteriors[query_type].items():
            if s != strategy:
                p.decay(self.config.posterior_decay)

        logger.debug(
            f"Strategy outcome: {strategy} for {query_type}, "
            f"success={success}, quality={quality:.2f}, "
            f"posterior=Beta({posterior.alpha:.1f}, {posterior.beta:.1f})"
        )

    def get_best_strategy(self, query_type: str) -> str:
        ranking = self.get_strategy_ranking(query_type)
        return ranking[0][0] if ranking else "hybrid"

    def get_uncertainty(self, query_type: str, strategy: str) -> float:
        if query_type in self._posteriors and strategy in self._posteriors[query_type]:
            return self._posteriors[query_type][strategy].variance()
        return 1.0

    def _save_state(self) -> None:
        state = {
            "posteriors": {
                qtype: {
                    strategy: posterior.to_dict()
                    for strategy, posterior in strategies.items()
                }
                for qtype, strategies in self._posteriors.items()
            },
            "selection_count": self._selection_count,
            "saved_at": time.time(),
        }
        with open(self._state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                posteriors_data = state.get("posteriors", {})
                for qtype, strategies in posteriors_data.items():
                    if qtype in self._posteriors:
                        for strategy, pdata in strategies.items():
                            if strategy in self._posteriors[qtype]:
                                self._posteriors[qtype][strategy] = BetaPosterior.from_dict(pdata)

                self._selection_count = state.get("selection_count", self._selection_count)
                logger.info("StrategyOptimizer state loaded")
            except Exception as e:
                logger.warning(f"Failed to load StrategyOptimizer state: {e}")

    def save(self) -> None:
        self._save_state()

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "posteriors": {},
            "selection_counts": self._selection_count,
            "best_strategies": {},
        }

        for qtype in self.config.query_types:
            stats["posteriors"][qtype] = {
                strategy: {
                    "mean": posterior.mean(),
                    "variance": posterior.variance(),
                    "alpha": posterior.alpha,
                    "beta": posterior.beta,
                }
                for strategy, posterior in self._posteriors.get(qtype, {}).items()
            }
            stats["best_strategies"][qtype] = self.get_best_strategy(qtype)

        return stats
