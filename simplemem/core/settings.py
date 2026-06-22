"""
SimpleMem runtime settings.

Resolution order for each attribute:
  1. User's top-level ``config.py`` (if importable on sys.path)
  2. Environment variable of the same name
  3. Built-in default
"""
import os


_DEFAULTS = {
    "OPENAI_API_KEY": "",
    "OPENAI_BASE_URL": None,
    "OPENROUTER_API_KEY": "",
    "OPENROUTER_BASE_URL": None,
    "LLM_MODEL": "gpt-4.1-mini",
    "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
    "EMBEDDING_DIMENSION": 1024,
    "EMBEDDING_CONTEXT_LENGTH": 32768,
    "ENABLE_THINKING": False,
    "USE_STREAMING": True,
    "USE_JSON_FORMAT": False,
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 4096,
    "WINDOW_SIZE": 40,
    "OVERLAP_SIZE": 2,
    "SEMANTIC_TOP_K": 25,
    "KEYWORD_TOP_K": 5,
    "STRUCTURED_TOP_K": 5,
    "LANCEDB_PATH": "./lancedb_data",
    "MEMORY_TABLE_NAME": "memory_entries",
    "ENABLE_PARALLEL_PROCESSING": True,
    "MAX_PARALLEL_WORKERS": 16,
    "ENABLE_PARALLEL_RETRIEVAL": True,
    "MAX_RETRIEVAL_WORKERS": 8,
    "ENABLE_PLANNING": True,
    "ENABLE_REFLECTION": True,
    "MAX_REFLECTION_ROUNDS": 2,
}


def _coerce(default, raw):
    if default is None or isinstance(default, str):
        return raw
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


class Settings:
    def __init__(self):
        try:
            import config as _user_config
        except ImportError:
            _user_config = None
        self._user_config = _user_config

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup fails, so no recursion via _user_config.
        user_cfg = self.__dict__.get("_user_config")
        if user_cfg is not None and hasattr(user_cfg, name):
            return getattr(user_cfg, name)
        env_val = os.getenv(name)
        if name in _DEFAULTS:
            if env_val is not None:
                return _coerce(_DEFAULTS[name], env_val)
            return _DEFAULTS[name]
        if env_val is not None:
            return env_val
        raise AttributeError(
            f"Setting {name!r} is not defined. Set it in config.py or as env var."
        )


settings = Settings()
