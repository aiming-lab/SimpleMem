"""
Meta-Memory Controller for Self-Evolving Omni-Memory.

Adaptive parameter optimization based on downstream feedback signals.

Inspired by:
- MemEvolve (arXiv 2512.18746): Meta-evolution of memory architecture
- MCMA (arXiv 2601.07470): Memory management as learnable cognitive skill
- MAGELLAN (ICML 2025): Metacognitive learning progress prediction

Update rule: θ_{t+1} = clip(θ_t + α · (metric_t - target), θ_min, θ_max)
"""

import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveParameters:
    """Current adaptive parameter values maintained by the meta-controller."""

    visual_trigger_threshold_high: float = 0.9
    visual_trigger_threshold_low: float = 0.7
    audio_vad_threshold: float = 0.5

    default_top_k: int = 10
    auto_expand_threshold: float = 0.85

    forgetting_threshold: float = 0.1
    strengthening_threshold: float = 0.8

    parametric_confidence_threshold: float = 0.6
    semantic_compaction_level: float = 0.0  # 0.0 = no compaction, 1.0 = aggressive

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveParameters":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EvolutionMetrics:
    """EMA-smoothed metrics for evolution decisions."""

    retrieval_precision: float = 0.5
    retrieval_recall: float = 0.5
    answer_quality: float = 0.5

    ingestion_accept_rate: float = 0.5
    memory_utilization: float = 0.5

    parametric_hit_rate: float = 0.0
    graph_hit_rate: float = 0.0
    expansion_usefulness: float = 0.5

    total_queries: int = 0
    total_ingestions: int = 0
    total_reflections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MetaController:
    """
    Self-Evolving Memory Controller.

    Tracks retrieval effectiveness via EMA and adapts system parameters:
    - Trigger thresholds (precision/recall tradeoff)
    - Consolidation aggressiveness (memory utilization)
    - Retrieval parameters (answer quality)
    """

    def __init__(self, config=None, storage_path: Optional[str] = None):
        from .evolution_config import MetaControllerConfig

        self.config = config or MetaControllerConfig()
        self.storage_path = Path(storage_path or "./omni_memory_data/evolution")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.params = AdaptiveParameters()
        self.metrics = EvolutionMetrics()

        self._state_file = self.storage_path / "meta_controller_state.json"
        self._load_state()

        logger.info("MetaController initialized")

    def record_query_outcome(
        self,
        query: str,
        memories_retrieved: int,
        memories_useful: int,
        answer_quality: float,
        strategy_used: str,
        parametric_used: bool = False,
        graph_used: bool = False,
        expansion_used: bool = False,
        expansion_helpful: bool = False,
    ) -> None:
        """Record outcome of a query-answer cycle for metric tracking."""
        decay = self.config.ema_decay

        if memories_retrieved > 0:
            precision = memories_useful / memories_retrieved
            self.metrics.retrieval_precision = (
                decay * self.metrics.retrieval_precision + (1 - decay) * precision
            )

        found_relevant = 1.0 if memories_useful > 0 else 0.0
        self.metrics.retrieval_recall = (
            decay * self.metrics.retrieval_recall + (1 - decay) * found_relevant
        )

        self.metrics.answer_quality = (
            decay * self.metrics.answer_quality + (1 - decay) * answer_quality
        )

        if parametric_used:
            hit = 1.0 if answer_quality > 0.7 else 0.0
            self.metrics.parametric_hit_rate = (
                decay * self.metrics.parametric_hit_rate + (1 - decay) * hit
            )

        if graph_used:
            hit = 1.0 if answer_quality > 0.6 else 0.0
            self.metrics.graph_hit_rate = (
                decay * self.metrics.graph_hit_rate + (1 - decay) * hit
            )

        if expansion_used:
            useful = 1.0 if expansion_helpful else 0.0
            self.metrics.expansion_usefulness = (
                decay * self.metrics.expansion_usefulness + (1 - decay) * useful
            )

        self.metrics.total_queries += 1

        if (
            self.metrics.total_queries >= self.config.warmup_queries
            and self.metrics.total_queries % self.config.update_interval_queries == 0
        ):
            self._update_parameters()

    def record_ingestion_outcome(self, accepted: bool) -> None:
        """Record whether an ingestion was accepted or rejected."""
        decay = self.config.ema_decay
        value = 1.0 if accepted else 0.0
        self.metrics.ingestion_accept_rate = (
            decay * self.metrics.ingestion_accept_rate + (1 - decay) * value
        )
        self.metrics.total_ingestions += 1

    def _update_parameters(self) -> None:
        """Core self-evolution step: adapt parameters based on metric gaps."""
        cfg = self.config

        # Trigger threshold: precision too low → raise, recall too low → lower
        precision_gap = (
            self.metrics.retrieval_precision - cfg.target_retrieval_precision
        )
        new_high = (
            self.params.visual_trigger_threshold_high
            + cfg.trigger_learning_rate * precision_gap
        )
        self.params.visual_trigger_threshold_high = max(
            cfg.trigger_threshold_min, min(cfg.trigger_threshold_max, new_high)
        )
        self.params.visual_trigger_threshold_low = max(
            cfg.trigger_threshold_min, self.params.visual_trigger_threshold_high - 0.2
        )

        # Consolidation: high utilization → increase semantic compaction, NOT episodic deletion
        # forgetting_threshold is frozen — it only controls ARCHIVE threshold, never DELETE
        utilization_gap = (
            self.metrics.memory_utilization - cfg.target_memory_utilization
        )
        new_compaction = (
            self.params.semantic_compaction_level
            + cfg.consolidation_learning_rate * utilization_gap
        )
        self.params.semantic_compaction_level = max(0.0, min(1.0, new_compaction))

        # Auto-expand: expansions useful → lower threshold, not useful → raise
        expand_signal = self.metrics.expansion_usefulness - 0.5
        new_expand = (
            self.params.auto_expand_threshold
            - cfg.retrieval_learning_rate * expand_signal
        )
        self.params.auto_expand_threshold = max(0.5, min(0.95, new_expand))

        # Parametric confidence calibration
        if self.metrics.parametric_hit_rate > 0.1:
            confidence_signal = self.metrics.parametric_hit_rate - 0.5
            new_conf = (
                self.params.parametric_confidence_threshold - 0.005 * confidence_signal
            )
            self.params.parametric_confidence_threshold = max(0.3, min(0.9, new_conf))

        logger.info(
            f"Meta-controller update: trigger_high={self.params.visual_trigger_threshold_high:.3f}, "
            f"forget={self.params.forgetting_threshold:.3f}, "
            f"expand={self.params.auto_expand_threshold:.3f}, "
            f"parametric_conf={self.params.parametric_confidence_threshold:.3f}"
        )

        if self.metrics.total_queries % (cfg.update_interval_queries * 5) == 0:
            self._save_state()

    def get_adaptive_params(self) -> AdaptiveParameters:
        return self.params

    def get_trigger_threshold(self, trigger_type: str = "visual") -> Dict[str, float]:
        if trigger_type == "visual":
            return {
                "high": self.params.visual_trigger_threshold_high,
                "low": self.params.visual_trigger_threshold_low,
            }
        elif trigger_type == "audio":
            return {"vad_threshold": self.params.audio_vad_threshold}
        return {}

    def get_retrieval_params(self) -> Dict[str, Any]:
        return {
            "default_top_k": self.params.default_top_k,
            "auto_expand_threshold": self.params.auto_expand_threshold,
            "parametric_confidence_threshold": self.params.parametric_confidence_threshold,
        }

    def get_consolidation_params(self) -> Dict[str, float]:
        return {
            "forgetting_threshold": self.params.forgetting_threshold,
            "strengthening_threshold": self.params.strengthening_threshold,
        }

    def _save_state(self) -> None:
        state = {
            "params": self.params.to_dict(),
            "metrics": self.metrics.to_dict(),
            "saved_at": time.time(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.debug("MetaController state saved")

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.params = AdaptiveParameters.from_dict(state.get("params", {}))
                self.metrics = EvolutionMetrics.from_dict(state.get("metrics", {}))
                logger.info(
                    f"MetaController state loaded (total_queries={self.metrics.total_queries})"
                )
            except Exception as e:
                logger.warning(f"Failed to load MetaController state: {e}")

    def save(self) -> None:
        self._save_state()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "adaptive_params": self.params.to_dict(),
            "metrics": self.metrics.to_dict(),
            "is_warmed_up": self.metrics.total_queries >= self.config.warmup_queries,
        }
