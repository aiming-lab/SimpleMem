"""
Settings configuration for SimpleMem MCP Server
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


def _load_env_file():
    """Load .env file manually if it exists"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


@dataclass
class Settings:
    """Application settings"""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # JWT Configuration
    jwt_secret_key: str = field(default_factory=lambda: os.getenv(
        "JWT_SECRET_KEY",
        "simplemem-secret-key-change-in-production"
    ))
    jwt_algorithm: str = "HS256"
    jwt_expiration_days: int = 30

    # Encryption for API Keys
    encryption_key: str = field(default_factory=lambda: os.getenv(
        "ENCRYPTION_KEY",
        "simplemem-encryption-key-32bytes!"  # Must be 32 bytes for AES-256
    ))

    # Database Paths
    data_dir: str = field(default_factory=lambda: os.getenv(
        "DATA_DIR",
        "./data"
    ))
    lancedb_path: str = field(default_factory=lambda: os.getenv(
        "LANCEDB_PATH",
        "./data/lancedb"
    ))
    user_db_path: str = field(default_factory=lambda: os.getenv(
        "USER_DB_PATH",
        "./data/users.db"
    ))

    # LLM Provider Configuration
    llm_provider: str = field(default_factory=lambda: os.getenv(
        "LLM_PROVIDER",
        "openrouter"  # Options: "openrouter", "ollama"
    ))

    # OpenRouter Configuration (used when llm_provider is "openrouter")
    openrouter_base_url: str = field(default_factory=lambda: os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    ))

    # Ollama Configuration (used when llm_provider is "ollama")
    ollama_base_url: str = field(default_factory=lambda: os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434/v1"
    ))

    # Common LLM Configuration
    llm_model: str = field(default_factory=lambda: os.getenv(
        "LLM_MODEL",
        "openai/gpt-4.1-mini" # "qwen3:4b-instruct" Default model name
    ))
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL",
        "qwen3-embedding:4b"  # Default embedding model
    ))
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv(
        "EMBEDDING_DIMENSION",
        "2560"  # Default: 2560 for qwen3-embedding:4b, 768 for nomic-embed-text
    )))

    # Memory Building Configuration
    window_size: int = 20
    overlap_size: int = 2

    # Retrieval Configuration
    semantic_top_k: int = 25
    keyword_top_k: int = 5
    enable_planning: bool = True
    enable_reflection: bool = True
    max_reflection_rounds: int = 2

    # LLM Configuration
    llm_temperature: float = 0.1
    llm_max_retries: int = 3
    use_streaming: bool = True

    def __post_init__(self):
        """Ensure directories exist; use absolute paths so cwd and permissions are predictable."""
        data_dir = os.path.abspath(os.path.expanduser(self.data_dir))
        lancedb_path = os.path.abspath(os.path.expanduser(self.lancedb_path))
        self.data_dir = data_dir
        self.lancedb_path = lancedb_path
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.lancedb_path, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create data dir(s): {e}. "
                "In Docker, use a named volume for data (see docker-compose.yml) or ensure the mounted dir is writable by the container user."
            ) from e


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
