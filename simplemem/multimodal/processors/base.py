"""
Base processor class for multimodal data processing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
from pathlib import Path

from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.config import OmniMemoryConfig
from simplemem.multimodal.triggers.base import TriggerResult, TriggerDecision
from simplemem.multimodal.storage.cold_storage import ColdStorageManager

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing multimodal data."""

    success: bool
    mau: Optional[MultimodalAtomicUnit] = None
    trigger_result: Optional[TriggerResult] = None
    error: Optional[str] = None
    skipped: bool = False  # True if skipped due to trigger rejection
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "mau_id": self.mau.id if self.mau else None,
            "trigger_decision": self.trigger_result.decision.value if self.trigger_result else None,
            "skipped": self.skipped,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseProcessor(ABC):
    """
    Base class for multimodal processors.

    Processors are responsible for:
    1. Running entropy triggers to filter redundant data
    2. Generating summaries and embeddings
    3. Storing raw data in cold storage
    4. Creating MAU records
    """

    def __init__(
        self,
        config: Optional[OmniMemoryConfig] = None,
        cold_storage: Optional[ColdStorageManager] = None,
    ):
        self.config = config or OmniMemoryConfig()
        self.cold_storage = cold_storage or ColdStorageManager(self.config.storage)

        # LLM client for summarization
        self._llm_client = None

    @property
    @abstractmethod
    def modality_type(self) -> ModalityType:
        """Return the modality type this processor handles."""
        pass

    @abstractmethod
    def process(
        self,
        data: Any,
        session_id: Optional[str] = None,
        force: bool = False,
        **kwargs
    ) -> ProcessingResult:
        """
        Process incoming data and create MAU.

        Args:
            data: The input data (file path, bytes, etc.)
            session_id: Optional session identifier
            force: Force processing even if trigger rejects

        Returns:
            ProcessingResult with MAU or skip reason
        """
        pass

    @abstractmethod
    def generate_summary(self, data: Any) -> str:
        """Generate a text summary of the data."""
        pass

    @abstractmethod
    def generate_embedding(self, data: Any) -> List[float]:
        """Generate embedding vector for the data."""
        pass

    def _get_llm_client(self):
        """Get or create LLM client."""
        if self._llm_client is None:
            from openai import OpenAI
            import httpx
            import os
            
            # Only pass non-None parameters to avoid compatibility issues
            client_kwargs = {}
            
            # Get API key from config or environment
            api_key = self.config.llm.api_key
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("BOSCH_API_KEY")
            
            if api_key:
                client_kwargs["api_key"] = api_key
            else:
                logger.error("No API key found in config or environment variables for LLM client")
                raise ValueError("API key is required for LLM operations")
            
            if self.config.llm.api_base_url is not None:
                client_kwargs["base_url"] = self.config.llm.api_base_url
            elif os.getenv("OPENAI_BASE_URL") or os.getenv("BOSCH_BASE_URL"):
                client_kwargs["base_url"] = os.getenv("OPENAI_BASE_URL") or os.getenv("BOSCH_BASE_URL")
            
            # Create http_client - let httpx use environment variables for proxy
            # Don't pass proxies explicitly, let httpx.Client read from environment
            http_client = httpx.Client()
            client_kwargs["http_client"] = http_client
            
            self._llm_client = OpenAI(**client_kwargs)
            logger.debug(f"Created LLM client with base_url: {client_kwargs.get('base_url', 'default')}")
        return self._llm_client

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Call LLM for text generation."""
        client = self._get_llm_client()
        from simplemem.multimodal.utils.model_utils import normalize_model_name
        model = model or self.config.llm.summary_model
        model = normalize_model_name(model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _get_embedding_service(self):
        """Get or create a shared EmbeddingService instance."""
        if not hasattr(self, "_embedding_service") or self._embedding_service is None:
            from simplemem.multimodal.utils.embedding import EmbeddingService
            self._embedding_service = EmbeddingService(self.config)
        return self._embedding_service

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding using EmbeddingService (local or OpenAI)."""
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for embedding generation")
            return []

        try:
            svc = self._get_embedding_service()
            embedding = svc.embed_text(text[:8000])
            if not embedding:
                logger.error(f"Failed to generate embedding: returned empty list")
                return []
            logger.debug(f"Generated embedding (dim: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    def create_mau(
        self,
        summary: str,
        embedding: List[float],
        raw_pointer: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        tags: Optional[List[str]] = None,
        trigger_result: Optional[TriggerResult] = None,
        **kwargs
    ) -> MultimodalAtomicUnit:
        """Create a MAU with the given data."""
        from simplemem.multimodal.core.mau import MAUMetadata, QualityMetrics

        quality = QualityMetrics()
        if trigger_result:
            quality.trigger_score = trigger_result.score
            quality.entropy_delta = trigger_result.entropy_delta or 0.0

        metadata = MAUMetadata(
            session_id=session_id,
            tags=tags or [],
            quality=quality,
        )

        mau = MultimodalAtomicUnit(
            modality_type=self.modality_type,
            summary=summary,
            embedding=embedding if embedding else None,
            raw_pointer=raw_pointer,
            metadata=metadata,
        )

        if timestamp:
            mau.timestamp = timestamp

        return mau
