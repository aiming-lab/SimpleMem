"""
Visual Entropy Trigger for Omni-Memory.

Uses lightweight visual encoders (CLIP/SigLIP/DINOv2) to detect
meaningful visual changes between frames, filtering out redundant content.
"""

import logging
from typing import Optional, Any, Dict, Union
from pathlib import Path
import numpy as np

from simplemem.multimodal.triggers.base import BaseTrigger, TriggerResult, TriggerDecision
from simplemem.multimodal.core.config import EntropyTriggerConfig
from simplemem.multimodal.utils.openvision_clip import (
    extract_openvision_image_embedding,
    is_openvision_model,
    load_openvision_clip_model,
    to_pil_image,
)

logger = logging.getLogger(__name__)


class VisualEntropyTrigger(BaseTrigger):
    """
    Visual Entropy Trigger using lightweight visual encoders.

    Key Design:
    - Uses CLIP/SigLIP for fast feature extraction
    - Compares consecutive frames via cosine similarity
    - High similarity (>0.9) = static scene, discard
    - Low similarity (<0.7) = significant change, trigger VLM processing

    This converts continuous visual streams into sparse event streams.
    """

    def __init__(
        self,
        config: Optional[EntropyTriggerConfig] = None,
        similarity_threshold_high: float = 0.9,
        similarity_threshold_low: float = 0.7,
        encoder_type: str = "clip",
        model_name: Optional[str] = None,
    ):
        super().__init__(enabled=True)

        if config:
            self.threshold_high = config.visual_similarity_threshold_high
            self.threshold_low = config.visual_similarity_threshold_low
            self.encoder_type = config.visual_encoder
            self.model_name = config.visual_model_name
            self.enabled = config.enable_visual_trigger
        else:
            self.threshold_high = similarity_threshold_high
            self.threshold_low = similarity_threshold_low
            self.encoder_type = encoder_type
            self.model_name = model_name or "UCSC-VLAA/openvision-vit-large-patch14-224"

        # Lazy load encoder
        self._encoder = None
        self._processor = None
        self._clip_backend: Optional[str] = None
        self._device = None
        self._previous_embedding: Optional[np.ndarray] = None

    def _load_encoder(self):
        """Lazy load the visual encoder."""
        if self._encoder is not None:
            return

        # If visual trigger is disabled, skip loading encoder
        if not self.enabled:
            self._encoder = "fallback"
            logger.debug("Visual trigger disabled, using fallback mode.")
            return

        try:
            import torch
            import os

            # Configure proxy for HuggingFace downloads if available
            proxy_url = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
            if proxy_url:
                os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Disable hf_transfer for proxy compatibility

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.encoder_type == "clip":
                if is_openvision_model(self.model_name):
                    self._encoder, self._processor, resolved_name = load_openvision_clip_model(
                        model_name=self.model_name,
                        device=self._device,
                    )
                    self._clip_backend = "openvision_open_clip"
                    self.model_name = resolved_name
                else:
                    from transformers import CLIPProcessor, CLIPModel

                    logger.info(f"Loading CLIP model: {self.model_name} (this may take a while on first run)...")
                    self._processor = CLIPProcessor.from_pretrained(self.model_name)
                    self._encoder = CLIPModel.from_pretrained(self.model_name).to(self._device)
                    self._encoder.eval()
                    self._clip_backend = "transformers_clip"
            elif self.encoder_type == "siglip":
                # SigLIP support
                from transformers import SiglipProcessor, SiglipModel
                logger.info(f"Loading SigLIP model: {self.model_name} (this may take a while on first run)...")
                self._processor = SiglipProcessor.from_pretrained(self.model_name)
                self._encoder = SiglipModel.from_pretrained(self.model_name).to(self._device)
                self._encoder.eval()
            elif self.encoder_type == "dinov2":
                # DINOv2 support
                from transformers import AutoImageProcessor, AutoModel
                logger.info(f"Loading DINOv2 model: {self.model_name} (this may take a while on first run)...")
                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                self._encoder = AutoModel.from_pretrained(self.model_name).to(self._device)
                self._encoder.eval()
            else:
                raise ValueError(f"Unknown encoder type: {self.encoder_type}")

            logger.info(f"Loaded {self.encoder_type} encoder: {self.model_name}")

        except ImportError as e:
            # Use DEBUG level instead of WARNING since fallback mode is available
            # Fallback mode uses simple pixel statistics which works without torch
            if "torch" in str(e).lower() or "transformers" in str(e).lower():
                logger.debug(f"Visual encoder dependencies not available ({e}). Using fallback mode (pixel statistics). Install 'torch' and 'transformers' for enhanced visual processing.")
            else:
                logger.debug(f"Could not load visual encoder: {e}. Using fallback mode (pixel statistics).")
            self._encoder = "fallback"
        except Exception as e:
            # Handle other errors (e.g., network issues, model download failures)
            logger.warning(f"Failed to load visual encoder: {e}. Using fallback mode.")
            self._encoder = "fallback"

    def _extract_embedding(self, image: Any) -> Optional[np.ndarray]:
        """Extract visual embedding from image."""
        self._load_encoder()

        if self._encoder == "fallback":
            # Fallback: use simple pixel statistics
            return self._fallback_embedding(image)

        try:
            import torch

            if self.encoder_type == "clip" and self._clip_backend == "openvision_open_clip":
                embedding = extract_openvision_image_embedding(
                    model=self._encoder,
                    preprocess=self._processor,
                    image=image,
                    device=self._device,
                )
            else:
                image = to_pil_image(image)
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    if self.encoder_type in ["clip", "siglip"]:
                        outputs = self._encoder.get_image_features(**inputs)
                    else:  # dinov2
                        outputs = self._encoder(**inputs).last_hidden_state.mean(dim=1)

                    embedding = outputs.cpu().numpy().flatten()

            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding

        except Exception as e:
            logger.error(f"Error extracting visual embedding: {e}")
            return self._fallback_embedding(image)

    def _fallback_embedding(self, image: Any) -> np.ndarray:
        """Fallback embedding using basic image statistics."""
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        # Resize to small size for fast processing
        image = image.resize((64, 64))
        arr = np.array(image).flatten().astype(np.float32)

        # Normalize
        arr = arr / 255.0
        arr = arr / (np.linalg.norm(arr) + 1e-8)
        return arr

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2))

    def evaluate(self, data: Any) -> TriggerResult:
        """
        Evaluate visual data for entropy trigger.

        Args:
            data: Image data (PIL Image, numpy array, or file path)

        Returns:
            TriggerResult with decision based on similarity to previous frame
        """
        # Extract embedding
        current_embedding = self._extract_embedding(data)

        if current_embedding is None:
            return TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=1.0,
                reason="Failed to extract embedding, accepting by default",
            )

        # First frame always accepted
        if self._previous_embedding is None:
            self._previous_embedding = current_embedding
            return TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=1.0,
                reason="First frame in sequence, accepting",
                similarity_score=0.0,
                entropy_delta=1.0,
            )

        # Compute similarity
        similarity = self._compute_similarity(current_embedding, self._previous_embedding)
        entropy_delta = 1.0 - similarity

        # Make decision based on thresholds
        if similarity > self.threshold_high:
            # Very similar to previous - scene is static
            decision = TriggerDecision.REJECT
            reason = f"Scene static (similarity={similarity:.3f} > {self.threshold_high})"
        elif similarity < self.threshold_low:
            # Very different - significant change occurred
            decision = TriggerDecision.ACCEPT
            reason = f"Significant change detected (similarity={similarity:.3f} < {self.threshold_low})"
            # Update previous embedding only on accept
            self._previous_embedding = current_embedding
        else:
            # Middle ground - minor change
            decision = TriggerDecision.UNCERTAIN
            reason = f"Minor change (similarity={similarity:.3f}), borderline case"

        return TriggerResult(
            decision=decision,
            score=entropy_delta,
            reason=reason,
            similarity_score=similarity,
            entropy_delta=entropy_delta,
            metadata={"encoder": self.encoder_type},
        )

    def reset(self) -> None:
        """Reset trigger state (clear previous frame reference)."""
        self._previous_embedding = None

    def force_accept_next(self) -> None:
        """Force the next frame to be accepted (useful for keyframes)."""
        self._previous_embedding = None

    def get_embedding(self, image: Any) -> Optional[np.ndarray]:
        """Public method to get embedding for an image."""
        return self._extract_embedding(image)


class VisualEntropyTriggerBatch(VisualEntropyTrigger):
    """
    Batch version of visual entropy trigger for processing video frames.

    Maintains a sliding window of embeddings for more robust change detection.
    """

    def __init__(
        self,
        window_size: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self._embedding_window: list = []

    def evaluate(self, data: Any) -> TriggerResult:
        """Evaluate with sliding window smoothing."""
        current_embedding = self._extract_embedding(data)

        if current_embedding is None:
            return TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=1.0,
                reason="Failed to extract embedding",
            )

        # First frame
        if not self._embedding_window:
            self._embedding_window.append(current_embedding)
            return TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=1.0,
                reason="First frame in sequence",
            )

        # Compute average similarity to window
        similarities = [
            self._compute_similarity(current_embedding, emb)
            for emb in self._embedding_window
        ]
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)

        # Use max similarity for decision (most conservative)
        if max_similarity > self.threshold_high:
            decision = TriggerDecision.REJECT
            reason = f"Similar to recent frames (max_sim={max_similarity:.3f})"
        elif avg_similarity < self.threshold_low:
            decision = TriggerDecision.ACCEPT
            reason = f"Significant change from window (avg_sim={avg_similarity:.3f})"
        else:
            decision = TriggerDecision.UNCERTAIN
            reason = f"Minor change (avg_sim={avg_similarity:.3f})"

        # Update window on accept
        if decision == TriggerDecision.ACCEPT:
            self._embedding_window.append(current_embedding)
            if len(self._embedding_window) > self.window_size:
                self._embedding_window.pop(0)

        return TriggerResult(
            decision=decision,
            score=1.0 - avg_similarity,
            reason=reason,
            similarity_score=avg_similarity,
            entropy_delta=1.0 - avg_similarity,
            metadata={
                "window_size": len(self._embedding_window),
                "max_similarity": max_similarity,
            },
        )

    def reset(self) -> None:
        """Reset window."""
        self._embedding_window = []
