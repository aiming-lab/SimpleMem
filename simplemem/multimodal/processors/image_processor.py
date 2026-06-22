"""
Image Processor for Omni-Memory.

Handles image input with visual entropy triggering.
"""

import logging
from typing import Optional, Any, List, Union
from pathlib import Path
import base64
import io
import numpy as np

from simplemem.multimodal.processors.base import BaseProcessor, ProcessingResult
from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.config import OmniMemoryConfig
from simplemem.multimodal.storage.cold_storage import ColdStorageManager
from simplemem.multimodal.triggers.visual_trigger import VisualEntropyTrigger
from simplemem.multimodal.triggers.base import TriggerDecision
from simplemem.multimodal.utils.openvision_clip import (
    extract_openvision_image_embedding,
    is_openvision_model,
    load_openvision_clip_model,
    to_pil_image,
)

logger = logging.getLogger(__name__)


class ImageProcessor(BaseProcessor):
    """
    Image Processor with Visual Entropy Triggering.

    Key Features:
    1. Visual entropy trigger filters redundant frames
    2. VLM-based captioning for accepted images
    3. CLIP embeddings for retrieval
    4. Cold storage for raw images
    """

    def __init__(
        self,
        config: Optional[OmniMemoryConfig] = None,
        cold_storage: Optional[ColdStorageManager] = None,
        trigger: Optional[VisualEntropyTrigger] = None,
    ):
        super().__init__(config, cold_storage)

        # Initialize trigger
        self.trigger = trigger or VisualEntropyTrigger(
            config=self.config.entropy_trigger
        )

        # CLIP model for embeddings
        self._clip_model = None
        self._clip_processor = None
        self._clip_backend: Optional[str] = None
        self._clip_device: str = "cpu"
    
    def _normalize_model(self, model_name: str) -> str:
        """Normalize model name to ensure correct format."""
        from simplemem.multimodal.utils.model_utils import normalize_model_name
        return normalize_model_name(model_name)

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.VISUAL

    def process(
        self,
        data: Any,
        session_id: Optional[str] = None,
        force: bool = False,
        generate_caption: bool = True,
        **kwargs
    ) -> ProcessingResult:
        """
        Process image input.

        Args:
            data: Image data (file path, PIL Image, numpy array, or bytes)
            session_id: Optional session identifier
            force: Force processing even if trigger rejects
            generate_caption: Whether to generate VLM caption

        Returns:
            ProcessingResult with MAU
        """
        # Load image
        try:
            image = self._load_image(data)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return ProcessingResult(
                success=False,
                error=f"Failed to load image: {e}",
            )

        # Run entropy trigger
        trigger_result = self.trigger.evaluate(image)

        if not force and trigger_result.decision == TriggerDecision.REJECT:
            return ProcessingResult(
                success=False,
                skipped=True,
                trigger_result=trigger_result,
            )

        # Generate caption/summary
        if generate_caption:
            summary = self.generate_summary(image)
        else:
            summary = "Image captured"

        # Generate embedding
        embedding = self.generate_embedding(image)

        # Store in cold storage
        raw_pointer = self._store_image(image, session_id)

        # Create MAU
        mau = self.create_mau(
            summary=summary,
            embedding=embedding,
            raw_pointer=raw_pointer,
            session_id=session_id,
            trigger_result=trigger_result,
        )

        # Add image-specific metadata
        mau.metadata.source = kwargs.get('source', 'image_input')

        return ProcessingResult(
            success=True,
            mau=mau,
            trigger_result=trigger_result,
            metadata={"image_size": image.size if hasattr(image, 'size') else None},
        )

    def _load_image(self, data: Any):
        """Load image from various formats."""
        from PIL import Image

        if isinstance(data, Image.Image):
            return data
        elif isinstance(data, (str, Path)):
            return Image.open(data).convert('RGB')
        elif isinstance(data, bytes):
            return Image.open(io.BytesIO(data)).convert('RGB')
        elif hasattr(data, '__array__'):
            import numpy as np
            arr = np.array(data)
            return Image.fromarray(arr).convert('RGB')
        else:
            raise ValueError(f"Unsupported image format: {type(data)}")

    def _store_image(self, image, session_id: Optional[str] = None) -> str:
        """Store image in cold storage."""
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()

        return self.cold_storage.store(
            data=image_bytes,
            modality=ModalityType.VISUAL,
            extension='.jpg',
            session_id=session_id,
        )

    def generate_summary(self, data: Any) -> str:
        """Generate caption using VLM."""
        image = self._load_image(data) if not hasattr(data, 'save') else data

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Call VLM
        client = self._get_llm_client()

        try:
            response = client.chat.completions.create(
                model=self._normalize_model(self.config.llm.caption_model),
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence. Focus on key objects, actions, and context."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"VLM captioning failed: {e}")
            return "Image captured"

    def generate_embedding(self, data: Any) -> List[float]:
        """Generate CLIP embedding for image."""
        image = self._load_image(data) if not hasattr(data, 'save') else data

        # Use trigger's encoder if available
        if hasattr(self.trigger, 'get_embedding'):
            embedding = self.trigger.get_embedding(image)
            if embedding is not None:
                return embedding.tolist()

        # Fallback to loading our own CLIP
        try:
            self._load_clip()
            if self._clip_model is None:
                # CLIP not available, return empty embedding (will use text-based retrieval)
                return []
            
            import torch

            if self._clip_backend == "openvision_open_clip":
                embedding = extract_openvision_image_embedding(
                    model=self._clip_model,
                    preprocess=self._clip_processor,
                    image=image,
                    device=self._clip_device,
                )
            else:
                image = to_pil_image(image)
                inputs = self._clip_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self._clip_device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._clip_model.get_image_features(**inputs)
                    embedding = outputs.cpu().numpy().flatten()

            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.tolist()
        except Exception as e:
            logger.debug(f"CLIP embedding failed: {e}. Using text-based retrieval.")
            return []

    def _load_clip(self):
        """Load CLIP model."""
        if self._clip_model is not None:
            return

        try:
            import torch

            model_name = self.config.entropy_trigger.visual_model_name
            self._clip_device = "cuda" if torch.cuda.is_available() else "cpu"

            if is_openvision_model(model_name):
                self._clip_model, self._clip_processor, resolved_name = load_openvision_clip_model(
                    model_name=model_name,
                    device=self._clip_device,
                )
                self._clip_backend = "openvision_open_clip"
                logger.info(f"Loaded OpenVision model: {resolved_name}")
                return

            from transformers import CLIPProcessor, CLIPModel

            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
            self._clip_model = CLIPModel.from_pretrained(model_name).to(self._clip_device)
            self._clip_model.eval()
            self._clip_backend = "transformers_clip"
        except ImportError as e:
            # Use DEBUG instead of ERROR since fallback is available
            logger.debug(f"CLIP dependencies not available ({e}). Visual embeddings will use fallback mode. Install 'torch' and 'transformers' for enhanced visual processing.")
            self._clip_model = None
        except Exception as e:
            logger.debug(f"Failed to load CLIP: {e}. Using fallback mode.")
            self._clip_model = None

    def reset_trigger(self):
        """Reset the visual entropy trigger."""
        self.trigger.reset()

    def get_detailed_caption(self, image_pointer: str) -> str:
        """Generate detailed caption for stored image (lazy expansion)."""
        # Retrieve from cold storage
        image_bytes = self.cold_storage.retrieve(image_pointer)
        if not image_bytes:
            return ""

        image = self._load_image(image_bytes)

        # Generate detailed caption
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        client = self._get_llm_client()

        try:
            response = client.chat.completions.create(
                model=self._normalize_model(self.config.llm.caption_model),
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Provide a detailed description of this image including:
1. Main subjects and their appearance
2. Actions or events taking place
3. Setting and environment
4. Notable objects and their positions
5. Any text visible in the image"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Detailed caption failed: {e}")
            return ""
