"""
Multimodal Processors for Omni-Memory.

Processors handle ingestion of different modalities with entropy-triggered
filtering for efficient storage.
"""

from simplemem.multimodal.processors.text_processor import TextProcessor
from simplemem.multimodal.processors.image_processor import ImageProcessor
from simplemem.multimodal.processors.audio_processor import AudioProcessor
from simplemem.multimodal.processors.video_processor import VideoProcessor
from simplemem.multimodal.processors.base import BaseProcessor, ProcessingResult

__all__ = [
    "TextProcessor",
    "ImageProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "BaseProcessor",
    "ProcessingResult",
]
