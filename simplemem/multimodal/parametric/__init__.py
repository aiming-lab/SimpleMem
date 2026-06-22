"""
Parametric Memory module for Omni-Memory.

Provides memory distillation from episodic memories into a lightweight
neural model for fast, differentiable recall.
"""

from simplemem.multimodal.parametric.memory_distiller import MemoryDistiller
from simplemem.multimodal.parametric.parametric_store import ParametricMemoryStore
from simplemem.multimodal.parametric.consolidator import MemoryConsolidator

__all__ = [
    "MemoryDistiller",
    "ParametricMemoryStore", 
    "MemoryConsolidator",
]
