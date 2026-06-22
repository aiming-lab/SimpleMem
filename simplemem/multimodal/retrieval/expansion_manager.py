"""
Expansion Manager for Omni-Memory.

Manages lazy expansion of MAU details and raw content loading.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import base64

from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.config import OmniMemoryConfig
from simplemem.multimodal.storage.cold_storage import ColdStorageManager
from simplemem.multimodal.storage.mau_store import MAUStore

logger = logging.getLogger(__name__)


@dataclass
class ExpandedContent:
    """Expanded content from a MAU."""

    mau_id: str
    modality_type: ModalityType
    summary: str

    # Expanded details
    details: Optional[Dict[str, Any]] = None

    # Raw content (loaded from cold storage)
    raw_text: Optional[str] = None
    raw_image_base64: Optional[str] = None
    raw_audio_transcript: Optional[str] = None

    # Metadata
    token_estimate: int = 0
    load_time_ms: float = 0


class ExpansionManager:
    """
    Expansion Manager for lazy content loading.

    Manages the "expand" phase of pyramid retrieval:
    - Loads details from MAU store
    - Retrieves raw content from cold storage
    - Processes content for LLM consumption
    - Tracks token usage
    """

    def __init__(
        self,
        mau_store: MAUStore,
        cold_storage: ColdStorageManager,
        config: Optional[OmniMemoryConfig] = None,
    ):
        self.mau_store = mau_store
        self.cold_storage = cold_storage
        self.config = config or OmniMemoryConfig()

        # VLM for image re-captioning
        self._llm_client = None
    
    def _normalize_model(self, model_name: str) -> str:
        """Normalize model name to ensure correct format."""
        from simplemem.multimodal.utils.model_utils import normalize_model_name
        return normalize_model_name(model_name)

        # Cache for expensive operations
        self._caption_cache: Dict[str, str] = {}

    def _get_llm_client(self):
        """Get or create LLM client."""
        if self._llm_client is None:
            from openai import OpenAI
            import httpx
            # Only pass non-None parameters to avoid compatibility issues
            client_kwargs = {}
            if self.config.llm.api_key is not None:
                client_kwargs["api_key"] = self.config.llm.api_key
            if self.config.llm.api_base_url is not None:
                client_kwargs["base_url"] = self.config.llm.api_base_url
            
            # Create http_client explicitly to avoid proxies parameter issues
            # This prevents OpenAI SDK from reading proxy settings from environment variables
            http_client = httpx.Client()
            client_kwargs["http_client"] = http_client
            
            self._llm_client = OpenAI(**client_kwargs)
        return self._llm_client

    def expand_single(
        self,
        mau_id: str,
        load_raw: bool = True,
        regenerate_caption: bool = False,
    ) -> Optional[ExpandedContent]:
        """
        Expand a single MAU with full details.

        Args:
            mau_id: The MAU ID to expand
            load_raw: Whether to load raw content from cold storage
            regenerate_caption: Whether to regenerate caption from raw data

        Returns:
            ExpandedContent or None if not found
        """
        import time
        start_time = time.time()

        mau = self.mau_store.get(mau_id)
        if not mau:
            logger.warning(f"MAU not found: {mau_id}")
            return None

        result = ExpandedContent(
            mau_id=mau.id,
            modality_type=mau.modality_type,
            summary=mau.summary,
            details=mau.details,
        )

        token_estimate = len(mau.summary) // 4

        if mau.details:
            token_estimate += len(str(mau.details)) // 4

        # Load raw content if requested
        if load_raw and mau.raw_pointer:
            raw_result = self._load_raw(mau, regenerate_caption)
            if raw_result:
                result.raw_text = raw_result.get("text")
                result.raw_image_base64 = raw_result.get("image_base64")
                result.raw_audio_transcript = raw_result.get("transcript")
                token_estimate += raw_result.get("tokens", 0)

        result.token_estimate = token_estimate
        result.load_time_ms = (time.time() - start_time) * 1000

        return result

    def expand_batch(
        self,
        mau_ids: List[str],
        load_raw: bool = True,
        token_budget: Optional[int] = None,
    ) -> List[ExpandedContent]:
        """
        Expand multiple MAUs with optional token budget.

        Args:
            mau_ids: List of MAU IDs
            load_raw: Whether to load raw content
            token_budget: Optional maximum tokens to use

        Returns:
            List of ExpandedContent
        """
        results = []
        tokens_used = 0

        for mau_id in mau_ids:
            expanded = self.expand_single(mau_id, load_raw=load_raw)
            if not expanded:
                continue

            # Check budget
            if token_budget and tokens_used + expanded.token_estimate > token_budget:
                logger.info(f"Token budget reached ({tokens_used}/{token_budget}), stopping expansion")
                break

            results.append(expanded)
            tokens_used += expanded.token_estimate

        return results

    def _load_raw(
        self,
        mau: MultimodalAtomicUnit,
        regenerate_caption: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Load raw content from cold storage."""
        if not mau.raw_pointer:
            return None

        result = {"tokens": 0}

        try:
            data = self.cold_storage.retrieve(mau.raw_pointer)
            if not data:
                return None

            if mau.modality_type == ModalityType.TEXT:
                text = data.decode('utf-8')
                result["text"] = text
                result["tokens"] = len(text) // 4

            elif mau.modality_type == ModalityType.VISUAL:
                result["image_base64"] = base64.b64encode(data).decode('utf-8')
                result["tokens"] = 500  # Rough estimate for image

                if regenerate_caption:
                    caption = self._generate_detailed_caption(result["image_base64"])
                    if caption:
                        result["detailed_caption"] = caption
                        result["tokens"] += len(caption) // 4

            elif mau.modality_type == ModalityType.AUDIO:
                # For audio, return transcript from details or re-transcribe
                if mau.details and "transcript" in mau.details:
                    result["transcript"] = mau.details["transcript"]
                    result["tokens"] = len(result["transcript"]) // 4
                else:
                    result["tokens"] = 100

            elif mau.modality_type == ModalityType.VIDEO:
                # For video, return frame information
                if mau.details:
                    result["frame_count"] = mau.details.get("frames_processed", 0)
                    result["frame_ids"] = mau.details.get("frame_mau_ids", [])
                result["tokens"] = 200

        except Exception as e:
            logger.error(f"Error loading raw content: {e}")
            return None

        return result

    def _generate_detailed_caption(self, image_base64: str) -> Optional[str]:
        """Generate detailed caption for an image."""
        # Check cache
        cache_key = image_base64[:100]  # Use prefix as cache key
        if cache_key in self._caption_cache:
            return self._caption_cache[cache_key]

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
4. Notable objects
5. Any text visible"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }],
                max_tokens=500,
            )
            caption = response.choices[0].message.content.strip()
            self._caption_cache[cache_key] = caption
            return caption
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return None

    def format_expanded_for_llm(
        self,
        expanded_list: List[ExpandedContent],
        include_images: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Format expanded content for LLM messages.

        Returns content in OpenAI message format for multimodal input.
        """
        content_parts = []

        for expanded in expanded_list:
            # Add text description
            parts = [f"[{expanded.modality_type.value}] {expanded.summary}"]

            if expanded.details:
                parts.append(f"Details: {expanded.details}")

            if expanded.raw_text:
                parts.append(f"Content: {expanded.raw_text[:2000]}")

            if expanded.raw_audio_transcript:
                parts.append(f"Transcript: {expanded.raw_audio_transcript}")

            content_parts.append({
                "type": "text",
                "text": "\n".join(parts)
            })

            # Add image if available
            if include_images and expanded.raw_image_base64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{expanded.raw_image_base64}"
                    }
                })

        return content_parts

    def get_expansion_cost_estimate(self, mau_ids: List[str]) -> Dict[str, Any]:
        """
        Estimate the cost of expanding given MAUs.

        Returns token and API call estimates without actually loading.
        """
        total_tokens = 0
        image_count = 0
        audio_count = 0

        for mau_id in mau_ids:
            mau = self.mau_store.get(mau_id)
            if not mau:
                continue

            total_tokens += len(mau.summary) // 4
            if mau.details:
                total_tokens += len(str(mau.details)) // 4

            if mau.raw_pointer:
                if mau.modality_type == ModalityType.VISUAL:
                    image_count += 1
                    total_tokens += 500
                elif mau.modality_type == ModalityType.AUDIO:
                    audio_count += 1
                    total_tokens += 200
                elif mau.modality_type == ModalityType.TEXT:
                    total_tokens += 500  # Estimate

        return {
            "estimated_tokens": total_tokens,
            "image_count": image_count,
            "audio_count": audio_count,
            "mau_count": len(mau_ids),
        }

    def clear_cache(self):
        """Clear the caption cache."""
        self._caption_cache.clear()
