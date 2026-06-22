"""
SimpleMem unified retrieval configuration.

This Config object is what simplemem.optimize() produces and what
SimpleMem(..., config=config) consumes at inference time.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Retrieval configuration for SimpleMem."""

    # Per-channel top-k
    k_sem: int = 0
    k_kw: int = 5
    k_str: int = 0

    # Context budget
    context_budget: int = 8

    # Fusion
    fusion_mode: str = "sum"  # sum | weighted | rrf
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic": 1.0,
        "keyword": 1.0,
        "structured": 1.0,
    })

    # Answer generation
    answer_style: str = "concise"  # concise | verbose | extractive

    # Optional augmentations
    enable_entity_swap: bool = False
    enable_query_decomposition: bool = False
    enable_answer_verification: bool = False

    # Per-category overrides (category_name -> partial config dict)
    category_overrides: Dict[str, Any] = field(default_factory=dict)

    # Evolution metadata
    evolved: bool = False
    evolution_rounds: int = 0
    source_benchmark: str = ""

    def save(self, path: str) -> None:
        """Save config to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load config from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_config(path: str) -> Config:
    """Load a Config from a JSON file (convenience wrapper)."""
    return Config.from_file(path)
