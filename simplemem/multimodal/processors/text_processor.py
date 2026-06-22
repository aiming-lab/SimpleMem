"""
Text Processor for Omni-Memory.

Handles text input with information-based filtering.
"""

import logging
from typing import Optional, Any, List

from simplemem.multimodal.processors.base import BaseProcessor, ProcessingResult
from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.config import OmniMemoryConfig
from simplemem.multimodal.storage.cold_storage import ColdStorageManager
from simplemem.multimodal.triggers.base import TriggerResult, TriggerDecision

logger = logging.getLogger(__name__)


class TextProcessor(BaseProcessor):
    """
    Text Processor for Omni-Memory.

    Unlike visual/audio, text doesn't have an entropy trigger.
    Instead, it uses information density filtering:
    - Skip very short or empty text
    - Detect redundant/repetitive text
    - Extract key facts and entities
    """

    def __init__(
        self,
        config: Optional[OmniMemoryConfig] = None,
        cold_storage: Optional[ColdStorageManager] = None,
        min_length: int = 10,
        max_length: int = 10000,
        detect_redundancy: bool = True,
    ):
        super().__init__(config, cold_storage)
        self.min_length = min_length
        self.max_length = max_length
        self.detect_redundancy = detect_redundancy

        # Track recent texts for redundancy detection
        self._recent_texts: List[str] = []
        self._max_recent = 50

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.TEXT

    def process(
        self,
        data: Any,
        session_id: Optional[str] = None,
        force: bool = False,
        store_raw: bool = False,
        **kwargs
    ) -> ProcessingResult:
        """
        Process text input.

        Args:
            data: Text string or file path
            session_id: Optional session identifier
            force: Force processing even if filtered
            store_raw: Store raw text in cold storage

        Returns:
            ProcessingResult with MAU
        """
        # Extract text
        if isinstance(data, str):
            if len(data) < 500 and data.endswith(('.txt', '.md', '.json')):
                # Likely a file path
                try:
                    from pathlib import Path
                    text = Path(data).read_text(encoding='utf-8')
                except:
                    text = data
            else:
                text = data
        else:
            text = str(data)

        # Length filtering
        if len(text.strip()) < self.min_length and not force:
            return ProcessingResult(
                success=False,
                skipped=True,
                trigger_result=TriggerResult(
                    decision=TriggerDecision.REJECT,
                    score=0.0,
                    reason=f"Text too short ({len(text)} < {self.min_length})",
                ),
            )

        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length] + "..."

        # Redundancy check
        if self.detect_redundancy and not force:
            trigger_result = self._check_redundancy(text)
            if trigger_result.decision == TriggerDecision.REJECT:
                return ProcessingResult(
                    success=False,
                    skipped=True,
                    trigger_result=trigger_result,
                )
        else:
            trigger_result = TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=1.0,
                reason="Text accepted",
            )

        # Generate summary
        summary = self.generate_summary(text)

        # Generate embedding (uses EmbeddingService: CLIP / Tongyi / Doubao)
        embedding = self.generate_embedding(text)
        if not embedding:
            logger.warning("Text embedding empty, skip storing MAU for this text")
            return ProcessingResult(
                success=False,
                error="No embedding data received",
                trigger_result=trigger_result,
            )

        # Store raw text if requested
        raw_pointer = None
        if store_raw:
            raw_pointer = self.cold_storage.store(
                data=text,
                modality=ModalityType.TEXT,
                session_id=session_id,
            )

        # Create MAU
        mau = self.create_mau(
            summary=summary,
            embedding=embedding,
            raw_pointer=raw_pointer,
            session_id=session_id,
            trigger_result=trigger_result,
        )

        # For text, we can store the full content in details
        mau.details = {"full_text": text}

        # Update recent texts
        self._recent_texts.append(text[:200])
        if len(self._recent_texts) > self._max_recent:
            self._recent_texts.pop(0)

        return ProcessingResult(
            success=True,
            mau=mau,
            trigger_result=trigger_result,
        )

    def _check_redundancy(self, text: str) -> TriggerResult:
        """Check if text is redundant compared to recent texts."""
        if not self._recent_texts:
            return TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=1.0,
                reason="First text in session",
            )

        # Simple word overlap check
        text_words = set(text.lower().split())

        max_overlap = 0.0
        for recent in self._recent_texts:
            recent_words = set(recent.lower().split())
            if not recent_words:
                continue
            overlap = len(text_words & recent_words) / len(text_words | recent_words)
            max_overlap = max(max_overlap, overlap)

        if max_overlap > 0.8:
            return TriggerResult(
                decision=TriggerDecision.REJECT,
                score=1.0 - max_overlap,
                reason=f"Highly redundant (overlap={max_overlap:.2f})",
            )
        elif max_overlap > 0.5:
            return TriggerResult(
                decision=TriggerDecision.UNCERTAIN,
                score=1.0 - max_overlap,
                reason=f"Partially redundant (overlap={max_overlap:.2f})",
            )
        else:
            return TriggerResult(
                decision=TriggerDecision.ACCEPT,
                score=1.0 - max_overlap,
                reason=f"Novel content (overlap={max_overlap:.2f})",
            )

    def generate_summary(self, text: str) -> str:
        """Generate summary of text."""
        # For short text, use as-is
        if len(text) <= 200:
            return text

        # Use LLM for summarization
        prompt = f"""Summarize the following text in one concise sentence:

{text[:2000]}

Summary:"""

        summary = self._call_llm(prompt)
        return summary if summary else text[:200]

    def generate_embedding(self, data: Any) -> List[float]:
        """Generate embedding for text (uses EmbeddingService so CLIP/Tongyi/Doubao backends work)."""
        from simplemem.multimodal.utils.embedding import EmbeddingService
        text = str(data)
        if not text or not text.strip():
            return []
        try:
            emb_svc = EmbeddingService(self.config)
            return emb_svc.embed_text(text[:8000])  # Limit for embedding models
        except Exception as e:
            logger.error("Text embedding failed: %s", e)
            return []

    def extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text."""
        prompt = f"""Extract key entities (people, places, organizations, concepts) from:

{text[:2000]}

Return as comma-separated list:"""

        result = self._call_llm(prompt)
        if result:
            return [e.strip() for e in result.split(',')]
        return []

    def reset(self):
        """Reset recent text tracking."""
        self._recent_texts = []
