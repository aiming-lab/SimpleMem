"""EvolveMem configuration — standalone config dataclass for the paper-faithful entry point."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvolveMemConfig:
    # Storage
    memory_store_path: str = "memory_data/store/memory.db"
    memory_scope: str = "default"
    memory_dir: str = "memory_data/store"

    # Retrieval
    memory_retrieval_mode: str = "keyword"  # "keyword" | "hybrid" | "embedding"
    memory_use_embeddings: bool = False
    memory_embedding_mode: str = "hashing"  # "hashing" | "semantic"
    memory_embedding_model: str = "all-MiniLM-L6-v2"
    memory_max_injected_units: int = 6
    memory_max_injected_tokens: int = 800

    # Policy
    memory_policy_path: str = "memory_data/store/policy.json"
    memory_telemetry_path: str = "memory_data/store/telemetry.jsonl"

    # Auto-upgrade
    memory_auto_upgrade_enabled: bool = False
    memory_auto_upgrade_interval_seconds: int = 900
    memory_auto_upgrade_require_review: bool = True
    memory_review_stale_after_hours: int = 72

    # Ingestion
    memory_auto_extract: bool = True
    memory_auto_consolidate: bool = True
