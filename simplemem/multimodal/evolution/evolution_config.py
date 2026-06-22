"""
Configuration for Meta-Memory Evolution (MME) framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class MetaControllerConfig:
    """Configuration for the self-evolving meta-controller."""

    trigger_learning_rate: float = 0.01
    consolidation_learning_rate: float = 0.005
    retrieval_learning_rate: float = 0.01

    target_retrieval_precision: float = 0.7
    target_memory_utilization: float = 0.8
    target_answer_quality: float = 0.7

    trigger_threshold_min: float = 0.5
    trigger_threshold_max: float = 0.95
    forgetting_threshold_min: float = 0.05
    forgetting_threshold_max: float = 0.3

    update_interval_queries: int = 10
    warmup_queries: int = 20

    ema_decay: float = 0.95
    semantic_compaction_level: float = 0.0  # 0.0 = no compaction, 1.0 = aggressive


@dataclass
class ExperienceEngineConfig:
    """Configuration for the experience accumulation engine."""

    enable_self_reflection: bool = True
    reflection_model: str = "gpt-4.1-nano"
    max_reflection_tokens: int = 200

    max_experiences: int = 10000
    experience_retention_days: int = 90

    experience_embedding_model: str = "text-embedding-3-large"
    experience_top_k: int = 3

    reflection_batch_size: int = 5
    async_reflection: bool = True


@dataclass
class StrategyOptimizerConfig:
    """Configuration for adaptive retrieval strategy optimizer."""

    strategies: List[str] = field(
        default_factory=lambda: ["parametric", "vector", "graph", "hybrid"]
    )

    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    query_types: List[str] = field(
        default_factory=lambda: [
            "factual",
            "temporal",
            "reasoning",
            "cross_modal",
            "entity_centric",
        ]
    )

    min_exploration_rate: float = 0.05
    posterior_decay: float = 0.99


@dataclass
class EvolutionConfig:
    """Master configuration for Meta-Memory Evolution (MME)."""

    meta_controller: MetaControllerConfig = field(default_factory=MetaControllerConfig)
    experience_engine: ExperienceEngineConfig = field(
        default_factory=ExperienceEngineConfig
    )
    strategy_optimizer: StrategyOptimizerConfig = field(
        default_factory=StrategyOptimizerConfig
    )

    enable_evolution: bool = True
    evolution_log_dir: str = "./omni_memory_data/evolution"

    save_interval_queries: int = 50

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
