"""
Pyramid Retriever - Token-efficient retrieval through preview-then-expand.

Core design:
1. Coarse Preview: Return only summaries (minimal tokens)
2. Selective Expansion: Load raw content only for selected items
3. Hierarchical Pruning: Access event hierarchy progressively

Merged from:
- safe-evolution: rich format_for_llm (full text, timestamps, metadata), embed_text_clip
- main: dimension checking, hybrid search, OpenVision CLIP support
"""

import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.event import EventNode, EventLevel
from simplemem.multimodal.core.config import OmniMemoryConfig, RetrievalConfig
from simplemem.multimodal.storage.mau_store import MAUStore
from simplemem.multimodal.storage.vector_store import VectorStore
from simplemem.multimodal.storage.cold_storage import ColdStorageManager

logger = logging.getLogger(__name__)


class RetrievalLevel(str, Enum):
    """Levels of retrieval detail."""

    SUMMARY = "summary"  # Only text summaries (cheapest)
    METADATA = "metadata"  # Summaries + metadata
    DETAILS = "details"  # Full MAU details (no raw content)
    EVIDENCE = "evidence"  # Full content including raw data (most expensive)


@dataclass
class RetrievalResult:
    """Result from pyramid retrieval."""

    query: str
    level: RetrievalLevel
    items: List[Dict[str, Any]]
    total_candidates: int
    tokens_used_estimate: int

    # For expansion decisions
    can_expand: bool = True
    expansion_candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "level": self.level.value,
            "items": self.items,
            "total_candidates": self.total_candidates,
            "tokens_used_estimate": self.tokens_used_estimate,
            "can_expand": self.can_expand,
            "expansion_candidates": self.expansion_candidates,
        }


@dataclass
class ExpansionRequest:
    """Request to expand specific items."""

    mau_ids: List[str]
    level: RetrievalLevel = RetrievalLevel.EVIDENCE


class PyramidRetriever:
    """
    Pyramid Retriever - Token-efficient multimodal retrieval.

    Design Philosophy:
    - Like a file manager: show thumbnails first, full image on request
    - 90% of queries answered from summaries alone
    - Raw multimodal content only loaded when explicitly needed

    Two-Stage Process:
    1. Coarse Preview: Vector search -> return summaries
    2. Selective Expansion: User/LLM requests -> load raw content
    """

    def __init__(
        self,
        mau_store: MAUStore,
        vector_store: VectorStore,
        cold_storage: ColdStorageManager,
        config: Optional[OmniMemoryConfig] = None,
    ):
        self.mau_store = mau_store
        self.vector_store = vector_store
        self.cold_storage = cold_storage
        self.config = config or OmniMemoryConfig()
        self.retrieval_config = self.config.retrieval
        self._embedding_service = None
        self._embedding_lock = threading.Lock()

    def _get_embedding_service(self):
        """Lazy-init shared EmbeddingService so CLIP/embedding is loaded once, not per query."""
        if self._embedding_service is not None:
            return self._embedding_service
        with self._embedding_lock:
            if self._embedding_service is not None:
                return self._embedding_service
            from simplemem.multimodal.utils.embedding import EmbeddingService

            self._embedding_service = EmbeddingService(self.config)
        return self._embedding_service

    def retrieve_preview(
        self,
        query: str,
        top_k: Optional[int] = None,
        modality_filter: Optional[ModalityType] = None,
        time_range: Optional[Tuple[float, float]] = None,
        tags_filter: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """
        Stage 1: Coarse Preview Retrieval.

        Returns summaries with metadata/details for rich LLM context.
        LLM/user can then decide which items to expand.

        Args:
            query: Search query
            top_k: Number of results (default from config)
            modality_filter: Filter by modality type
            time_range: Filter by time (start, end)
            tags_filter: Filter by tags

        Returns:
            RetrievalResult with summaries + metadata + details
        """
        top_k = top_k or self.retrieval_config.default_top_k

        # Get query embedding using the same backend that built the stored vectors.
        # IMPORTANT: Must use embed_text() (not embed_text_clip()) to match stored vector dimensions.
        # - LoCoMo stores: text-embedding-3-large (3072-dim) → query must be 3072-dim
        # - Mem-Gallery stores: all-MiniLM-L6-v2 (384-dim) → query must be 384-dim
        # embed_text() auto-detects the correct backend from config.
        try:
            emb_svc = self._get_embedding_service()
            query_embedding = emb_svc.embed_text(query)
        except Exception as e:
            logger.error("Query embedding failed: %s", e)
            query_embedding = []
        if not query_embedding:
            return RetrievalResult(
                query=query,
                level=RetrievalLevel.SUMMARY,
                items=[],
                total_candidates=0,
                tokens_used_estimate=0,
            )

        # Dimension check (useful for debugging)
        expected_dim = self.config.embedding.embedding_dim
        if len(query_embedding) != expected_dim:
            logger.warning(
                f"Query embedding dim={len(query_embedding)} != config expected={expected_dim}. "
                f"This may cause retrieval failure if stored vectors have different dim."
            )

        # Vector search
        search_results = self.vector_store.search(query_embedding, top_k=top_k * 2)

        # Get MAU summaries with rich metadata
        items = []
        mau_ids = [mau_id for mau_id, score in search_results]

        for mau_id, score in search_results:
            mau = self.mau_store.get(mau_id)
            if not mau:
                continue

            # Apply filters
            if modality_filter and mau.modality_type != modality_filter:
                continue

            if time_range:
                if mau.timestamp < time_range[0] or mau.timestamp > time_range[1]:
                    continue

            if tags_filter:
                if not any(tag in mau.metadata.tags for tag in tags_filter):
                    continue

            # Add summary item WITH metadata for rich LLM context (from safe-evolution)
            item_dict = {
                "id": mau.id,
                "summary": mau.summary,
                "modality_type": mau.modality_type.value,
                "timestamp": mau.timestamp,
                "score": score,
                "tags": mau.metadata.tags,
                "has_raw_data": mau.raw_pointer is not None,
                "metadata": mau.metadata.to_dict(),
            }
            # Include details for richer context in format_for_llm
            if mau.details and isinstance(mau.details, dict):
                item_dict["details"] = mau.details
            items.append(item_dict)

            if len(items) >= top_k:
                break

        # Estimate tokens (rough: ~1 token per 4 characters)
        tokens_estimate = sum(len(item["summary"]) // 4 for item in items)

        return RetrievalResult(
            query=query,
            level=RetrievalLevel.SUMMARY,
            items=items,
            total_candidates=len(search_results),
            tokens_used_estimate=tokens_estimate,
            expansion_candidates=[item["id"] for item in items if item["has_raw_data"]],
        )

    def expand(
        self,
        request: ExpansionRequest,
    ) -> RetrievalResult:
        """
        Stage 2: Selective Expansion.

        Load detailed content for selected items only.
        This is where expensive multimodal tokens are used.

        Args:
            request: ExpansionRequest specifying which items to expand

        Returns:
            RetrievalResult with full details/evidence
        """
        items = []
        tokens_estimate = 0

        for mau_id in request.mau_ids[: self.retrieval_config.max_expanded_items]:
            mau = self.mau_store.get(mau_id)
            if not mau:
                continue

            item = {
                "id": mau.id,
                "summary": mau.summary,
                "modality_type": mau.modality_type.value,
                "timestamp": mau.timestamp,
                "metadata": mau.metadata.to_dict(),
            }

            # Add details if available
            if mau.details:
                item["details"] = mau.details
                tokens_estimate += len(str(mau.details)) // 4

            # Load raw content for evidence level
            if request.level == RetrievalLevel.EVIDENCE and mau.raw_pointer:
                raw_content = self._load_raw_content(mau)
                if raw_content:
                    item["raw_content"] = raw_content
                    tokens_estimate += raw_content.get("token_estimate", 0)

            items.append(item)
            tokens_estimate += len(mau.summary) // 4

        return RetrievalResult(
            query="",
            level=request.level,
            items=items,
            total_candidates=len(request.mau_ids),
            tokens_used_estimate=tokens_estimate,
            can_expand=False,
        )

    def _load_raw_content(self, mau: MultimodalAtomicUnit) -> Optional[Dict[str, Any]]:
        """Load raw content from cold storage for a MAU."""
        if not mau.raw_pointer:
            return None

        result = {"pointer": mau.raw_pointer}

        # On-demand image: TEXT MAU but raw_pointer points to image (caption-only embedding)
        if mau.modality_type == ModalityType.TEXT and "vision_on_demand" in (
            mau.metadata.tags or []
        ):
            result["type"] = "image"
            result["token_estimate"] = 500
            data = self.cold_storage.retrieve(mau.raw_pointer)
            if data:
                import base64

                result["base64"] = base64.b64encode(data).decode("utf-8")
            return result

        if mau.modality_type == ModalityType.TEXT:
            # Load text content
            data = self.cold_storage.retrieve(mau.raw_pointer)
            if data:
                text = data.decode("utf-8")
                result["text"] = text
                result["token_estimate"] = len(text) // 4

        elif mau.modality_type == ModalityType.VISUAL:
            # For images, return pointer and optionally base64
            result["type"] = "image"
            result["token_estimate"] = 500  # Rough estimate for image tokens

            # Optionally load as base64 for VLM
            data = self.cold_storage.retrieve(mau.raw_pointer)
            if data:
                import base64

                result["base64"] = base64.b64encode(data).decode("utf-8")

        elif mau.modality_type == ModalityType.AUDIO:
            # For audio, return transcript if available
            result["type"] = "audio"
            if mau.details and "transcript" in mau.details:
                result["transcript"] = mau.details["transcript"]
                result["token_estimate"] = len(result["transcript"]) // 4
            else:
                result["token_estimate"] = 100

        elif mau.modality_type == ModalityType.VIDEO:
            # For video, return frame summaries
            result["type"] = "video"
            if mau.details:
                result["frame_count"] = mau.details.get("frames_processed", 0)
                result["frame_mau_ids"] = mau.details.get("frame_mau_ids", [])
            result["token_estimate"] = 200

        return result

    def retrieve_with_budget(
        self,
        query: str,
        token_budget: int,
        prefer_modality: Optional[ModalityType] = None,
        time_range: Optional[Tuple[float, float]] = None,
        tags_filter: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """
        Retrieve with explicit token budget constraint.

        Automatically adjusts detail level to fit budget.
        """
        # Start with preview
        preview = self.retrieve_preview(
            query,
            modality_filter=prefer_modality,
            time_range=time_range,
            tags_filter=tags_filter,
        )

        if preview.tokens_used_estimate <= token_budget:
            # Budget allows expansion
            remaining_budget = token_budget - preview.tokens_used_estimate
            items_to_expand = []

            for item in preview.items:
                # Estimate expansion cost
                estimated_cost = 100 if item["modality_type"] == "text" else 500
                if estimated_cost <= remaining_budget:
                    items_to_expand.append(item["id"])
                    remaining_budget -= estimated_cost

            if items_to_expand:
                expansion = self.expand(
                    ExpansionRequest(
                        mau_ids=items_to_expand,
                        level=RetrievalLevel.DETAILS,
                    )
                )
                # Merge results
                expanded_ids = {item["id"] for item in expansion.items}
                preview.items = [
                    next((e for e in expansion.items if e["id"] == item["id"]), item)
                    if item["id"] in expanded_ids
                    else item
                    for item in preview.items
                ]
                preview.tokens_used_estimate += expansion.tokens_used_estimate

        return preview

    def format_for_llm(
        self,
        result: RetrievalResult,
        include_instructions: bool = True,
    ) -> str:
        """
        Format retrieval results for LLM context.

        Returns a prompt-ready string with rich structured context including
        full text, timestamps, speaker info, entities, and other metadata.
        (from safe-evolution — critical for LoCoMo F1 performance)
        """
        parts = []

        if include_instructions:
            parts.append(
                "I found the following relevant memories. "
                "If you need more details about any item, ask me to expand it by ID."
            )

        parts.append(f"\n[{len(result.items)} memories found]\n")

        for i, item in enumerate(result.items, 1):
            # Start with context index and main content
            line_parts = [f"[Context {i}]"]
            line_parts.append(f"Content: {item['summary']}")
            # Add full text without aggressive truncation
            if item.get("details"):
                details = item["details"]
                if isinstance(details, dict):
                    for key in ("transcript", "text", "full_text"):
                        if key in details and details[key]:
                            full_text = str(details[key])[:2000]
                            line_parts.append(f"Full text: {full_text}")
                            break

            # Add timestamp in DD Month YYYY format
            # Timestamps have been corrected to actual conversation dates (2022-2024)
            if item.get("timestamp") and item["timestamp"] > 0:
                from datetime import datetime
                try:
                    dt = datetime.fromtimestamp(item["timestamp"])
                    if dt.year < 2026:  # Show all real conversation dates
                        line_parts.append(f"Time: {dt.strftime('%d %B %Y')}")
                except (ValueError, OSError):
                    pass
            # Add relevant tags (exclude internal locomo_ tags)
            if item.get("tags"):
                user_tags = [t for t in item["tags"] if not t.startswith("locomo_")]
                if user_tags:
                    line_parts.append(f"Tags: {', '.join(user_tags)}")
            # Add metadata: speaker, persons, entities, location, topic
            metadata = item.get("metadata", {})
            if isinstance(metadata, dict):
                if metadata.get("speaker_id"):
                    line_parts.append(f"Speaker: {metadata['speaker_id']}")
                if metadata.get("persons"):
                    persons = metadata["persons"]
                    if isinstance(persons, list):
                        line_parts.append(f"Persons: {', '.join(str(p) for p in persons)}")
                if metadata.get("entities"):
                    entities = metadata["entities"]
                    if isinstance(entities, list):
                        line_parts.append(f"Entities: {', '.join(str(e) for e in entities)}")
                if metadata.get("location"):
                    line_parts.append(f"Location: {metadata['location']}")
                if metadata.get("topic"):
                    line_parts.append(f"Topic: {metadata['topic']}")
            parts.append("\n".join(line_parts))

        return "\n\n".join(parts)

    def interactive_retrieve(
        self,
        query: str,
        llm_callback=None,
    ) -> Tuple[RetrievalResult, Optional[str]]:
        """
        Interactive retrieval with LLM decision-making.

        1. Get preview
        2. Ask LLM if expansion needed
        3. Expand if requested

        Args:
            query: Search query
            llm_callback: Function to call LLM for expansion decision

        Returns:
            Tuple of (final result, LLM response)
        """
        # Stage 1: Preview
        preview = self.retrieve_preview(query)

        if not preview.items:
            return preview, "No relevant memories found."

        # Format for LLM
        context = self.format_for_llm(preview)

        if llm_callback is None:
            return preview, context

        # Ask LLM for expansion decision
        expansion_prompt = f"""Based on the query "{query}" and these memory summaries:

{context}

Do you need to see the full content of any items to answer the query?
If yes, list the IDs you want to expand (comma-separated).
If no, respond with "NONE".

Your response:"""

        llm_response = llm_callback(expansion_prompt)

        # Parse expansion request
        if llm_response and "NONE" not in llm_response.upper():
            # Extract IDs from response
            import re

            ids = re.findall(r"mau_\w+", llm_response)
            if ids:
                expansion = self.expand(
                    ExpansionRequest(
                        mau_ids=ids,
                        level=RetrievalLevel.EVIDENCE,
                    )
                )
                return expansion, self.format_for_llm(
                    expansion, include_instructions=False
                )

        return preview, context
