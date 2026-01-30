"""External integrations for SimpleMem"""

from .openrouter import OpenRouterClient, OpenRouterClientManager
from .ollama import OllamaClient, OllamaClientManager

__all__ = [
    "OpenRouterClient",
    "OpenRouterClientManager",
    "OllamaClient",
    "OllamaClientManager",
]
