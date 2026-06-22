"""
EvolveMem configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvolveMemConfig:
    """Configuration for the EvolveMem self-evolution system."""

    # Base directories
    memory_dir: str = "~/.simplemem"
    record_dir: str = "~/.simplemem/records"

    # Store / policy / telemetry paths (derived from memory_dir if not set)
    memory_store_path: str = ""
    memory_policy_path: str = ""
    memory_telemetry_path: str = ""

    # Scope
    memory_scope: str = "default"

    # Retrieval
    memory_retrieval_mode: str = "keyword"  # keyword | semantic | hybrid
    memory_use_embeddings: bool = False
    memory_embedding_mode: str = "hashing"  # hashing | model
    memory_embedding_model: str = "all-MiniLM-L6-v2"

    # Injection limits
    memory_max_injected_units: int = 5
    memory_max_injected_tokens: int = 2000

    # Consolidation
    memory_auto_consolidate: bool = True

    # Auto-upgrade worker
    memory_auto_upgrade_enabled: bool = False
    memory_auto_upgrade_interval_seconds: int = 3600
    memory_auto_upgrade_require_review: bool = True
    memory_review_stale_after_hours: int = 72

    def __post_init__(self):
        memory_dir = Path(self.memory_dir).expanduser()
        if not self.memory_store_path:
            self.memory_store_path = str(memory_dir / "store.jsonl")
        if not self.memory_policy_path:
            self.memory_policy_path = str(memory_dir / "policy.json")
        if not self.memory_telemetry_path:
            self.memory_telemetry_path = str(memory_dir / "telemetry.jsonl")
