"""
Core data structures for Omni-Memory system.
"""

from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.event import EventNode, EventLevel
from simplemem.multimodal.core.config import OmniMemoryConfig

__all__ = [
    "MultimodalAtomicUnit",
    "ModalityType",
    "EventNode",
    "EventLevel",
    "OmniMemoryConfig",
]
