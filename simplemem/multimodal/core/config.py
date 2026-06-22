"""
Configuration management for Omni-Memory system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class EntropyTriggerConfig:
    """Configuration for modal entropy triggers."""

    # Visual trigger settings
    visual_similarity_threshold_high: float = 0.9  # Above this = static, discard
    visual_similarity_threshold_low: float = 0.7   # Below this = significant change, trigger
    visual_encoder: str = "clip"  # Options: clip, siglip, dinov2
    visual_model_name: str = "UCSC-VLAA/openvision-vit-large-patch14-224"

    # Audio trigger settings
    audio_energy_threshold: float = 0.01  # Minimum energy to consider
    audio_vad_threshold: float = 0.5      # Voice activity detection threshold
    audio_min_speech_duration_ms: int = 500  # Minimum speech duration to trigger

    # General settings
    enable_visual_trigger: bool = True
    enable_audio_trigger: bool = True


@dataclass
class StorageConfig:
    """Configuration for storage management."""

    # Base directories
    base_dir: str = "./omni_memory_data"
    cold_storage_dir: str = "./omni_memory_data/cold_storage"
    index_dir: str = "./omni_memory_data/index"

    # Storage backends
    use_s3: bool = False
    s3_bucket: Optional[str] = None
    s3_prefix: str = "omni_memory/"

    # File organization
    organize_by_date: bool = True
    organize_by_modality: bool = True

    # Cleanup settings
    max_storage_gb: float = 100.0
    auto_cleanup_enabled: bool = False


@dataclass
class RetrievalConfig:
    """Configuration for pyramid retrieval system."""

    # Coarse retrieval (Step 1)
    default_top_k: int = 10
    max_summaries_in_context: int = 20

    # Fine retrieval (Step 2)
    max_expanded_items: int = 5
    max_raw_content_tokens: int = 2000

    # Token budgets
    summary_token_budget: int = 500
    details_token_budget: int = 1500
    evidence_token_budget: int = 3000

    # Retrieval modes
    enable_hybrid_search: bool = True
    enable_graph_traversal: bool = True

    # Expansion settings
    auto_expand_threshold: float = 0.85  # Auto-expand if relevance > threshold


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    model_name: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    batch_size: int = 32

    # For visual embeddings
    visual_embedding_model: str = "UCSC-VLAA/openvision-vit-large-patch14-224"
    visual_embedding_dim: int = 768

    def apply_backend_preset(self, preset: str) -> None:
        """Apply a named visual embedding preset."""
        presets = {
            "openvision": ("UCSC-VLAA/openvision-vit-base-patch16-224", 768),
            "openvision-large": ("UCSC-VLAA/openvision-vit-large-patch14-224", 768),
            "openvision-large-336": ("UCSC-VLAA/openvision-vit-large-patch14-336", 768),
            "openvision-huge": ("UCSC-VLAA/openvision-vit-huge-patch14-224", 1024),
        }
        if preset in presets:
            self.visual_embedding_model, self.visual_embedding_dim = presets[preset]


@dataclass
class LLMConfig:
    """Configuration for LLM interactions."""

    # API settings
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None

    # Model selection
    summary_model: str = "gpt-4o-mini"
    query_model: str = "gpt-4o-mini"
    caption_model: str = "gpt-4o"

    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 1000

    # Whisper settings
    whisper_model: str = "whisper-1"


@dataclass
class EventConfig:
    """Configuration for event management."""

    # Event creation
    auto_create_events: bool = True
    event_time_window_seconds: float = 300.0  # 5 minutes default
    min_maus_per_event: int = 1

    # Event summarization
    summarize_on_close: bool = True
    max_maus_for_summary: int = 20


@dataclass
class OmniMemoryConfig:
    """
    Main configuration class for Omni-Memory system.

    Combines all sub-configurations into a unified config object.
    """

    # Sub-configurations
    entropy_trigger: EntropyTriggerConfig = field(default_factory=EntropyTriggerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    event: EventConfig = field(default_factory=EventConfig)

    # Self-evolution configuration (lazy import to avoid circular deps)
    evolution: Optional[Any] = None  # EvolutionConfig, set via enable_evolution()

    # Global settings
    enable_self_evolution: bool = False
    debug_mode: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Initialize from environment variables if not set."""
        if self.llm.api_base_url is None:
            self.llm.api_base_url = os.getenv("OPENAI_API_BASE")
        if self.llm.api_key is None:
            self.llm.api_key = os.getenv("OPENAI_API_KEY")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def to_json(self) -> str:
        """Serialize config to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OmniMemoryConfig":
        """Create config from dictionary."""
        return cls(
            entropy_trigger=EntropyTriggerConfig(**data.get("entropy_trigger", {})),
            storage=StorageConfig(**data.get("storage", {})),
            retrieval=RetrievalConfig(**data.get("retrieval", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            llm=LLMConfig(**data.get("llm", {})),
            event=EventConfig(**data.get("event", {})),
            debug_mode=data.get("debug_mode", False),
            log_level=data.get("log_level", "INFO"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OmniMemoryConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, file_path: str) -> "OmniMemoryConfig":
        """Load config from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def save_to_file(self, file_path: str) -> None:
        """Save config to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def ensure_directories(self) -> None:
        """Create necessary directories."""
        Path(self.storage.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.storage.cold_storage_dir).mkdir(parents=True, exist_ok=True)
        Path(self.storage.index_dir).mkdir(parents=True, exist_ok=True)

    def enable_evolution(self, evolution_config=None) -> "OmniMemoryConfig":
        """Enable self-evolution with optional custom config."""
        if evolution_config is None:
            from simplemem.multimodal.evolution.evolution_config import EvolutionConfig
            evolution_config = EvolutionConfig()
        self.evolution = evolution_config
        self.enable_self_evolution = True
        return self

    def set_unified_model(self, model_name: str) -> None:
        """Set all LLM models to the same model name."""
        self.llm.query_model = model_name
        self.llm.summary_model = model_name
        self.llm.caption_model = model_name

    @classmethod
    def create_default(cls) -> "OmniMemoryConfig":
        """Create default configuration."""
        config = cls()
        config.ensure_directories()
        return config
