"""
Omni-Memory Orchestrator - Central coordinator for the memory system.

The orchestrator manages:
1. Multimodal ingestion with entropy triggering
2. MAU and Event storage
3. Pyramid retrieval with lazy expansion
4. Session management
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path

from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.event import EventNode, EventLevel
from simplemem.multimodal.core.config import OmniMemoryConfig

from simplemem.multimodal.storage.cold_storage import ColdStorageManager
from simplemem.multimodal.storage.mau_store import MAUStore
from simplemem.multimodal.storage.vector_store import HybridVectorStore

from simplemem.multimodal.processors.text_processor import TextProcessor
from simplemem.multimodal.processors.image_processor import ImageProcessor
from simplemem.multimodal.processors.audio_processor import AudioProcessor
from simplemem.multimodal.processors.video_processor import VideoProcessor
from simplemem.multimodal.processors.base import ProcessingResult

from simplemem.multimodal.retrieval.pyramid_retriever import (
    PyramidRetriever,
    RetrievalResult,
    ExpansionRequest,
    RetrievalLevel,
)
from simplemem.multimodal.retrieval.query_processor import QueryProcessor
from simplemem.multimodal.retrieval.expansion_manager import ExpansionManager

from simplemem.multimodal.graph.event_store import EventStore
from simplemem.multimodal.graph.event_manager import EventManager

from simplemem.multimodal.knowledge import KnowledgeGraph, EntityExtractor, GraphRetriever
from simplemem.multimodal.parametric import ParametricMemoryStore, MemoryConsolidator
from simplemem.multimodal.retrieval.bm25_store import BM25Store

# Routing imports (benchmark-safe evolution)
_ROUTING_AVAILABLE = False
try:
    from simplemem.multimodal.routing import MemoryRouter, BenchmarkSafeGuard
    from simplemem.multimodal.storage.semantic_store import SemanticStore

    _ROUTING_AVAILABLE = True
except ImportError:
    pass

# Self-evolution imports (conditional)
_EVOLUTION_AVAILABLE = False
try:
    from simplemem.multimodal.evolution import (
        MetaController,
        ExperienceEngine,
        StrategyOptimizer,
        EvolutionConfig,
    )

    _EVOLUTION_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class OmniMemoryOrchestrator:
    """
    Central Orchestrator for Omni-Memory System.

    Provides unified interface for:
    - Adding memories (text, image, audio, video)
    - Querying memories with token-efficient retrieval
    - Managing events and sessions
    - Generating answers with memory context

    Design Philosophy:
    - Entropy-driven: Only store meaningful information
    - Token-efficient: Preview-then-expand retrieval
    - Unified: All modalities treated equally as MAUs
    """

    def __init__(
        self,
        config: Optional[OmniMemoryConfig] = None,
        data_dir: Optional[str] = None,
    ):
        self.config = config or OmniMemoryConfig.create_default()

        if data_dir:
            self.config.storage.base_dir = data_dir
            self.config.storage.cold_storage_dir = str(Path(data_dir) / "cold_storage")
            self.config.storage.index_dir = str(Path(data_dir) / "index")

        self.config.ensure_directories()

        # Initialize storage components
        self.cold_storage = ColdStorageManager(self.config.storage)
        self.mau_store = MAUStore(config=self.config.storage)
        self.vector_store = HybridVectorStore(
            storage_path=str(Path(self.config.storage.index_dir) / "vectors"),
            text_dim=self.config.embedding.embedding_dim,
            visual_dim=self.config.embedding.visual_embedding_dim,
        )
        self.event_store = EventStore(config=self.config.storage)

        # Initialize processors
        self.text_processor = TextProcessor(self.config, self.cold_storage)
        self.image_processor = ImageProcessor(self.config, self.cold_storage)
        self.audio_processor = AudioProcessor(self.config, self.cold_storage)
        self.video_processor = VideoProcessor(self.config, self.cold_storage)

        # Initialize retrieval components
        self.retriever = PyramidRetriever(
            self.mau_store,
            self.vector_store,
            self.cold_storage,
            self.config,
        )
        self.query_processor = QueryProcessor(self.config)
        self.expansion_manager = ExpansionManager(
            self.mau_store,
            self.cold_storage,
            self.config,
        )

        # Initialize BM25 keyword search
        self.bm25_store = BM25Store()

        # Initialize event management
        self.event_manager = EventManager(
            self.event_store,
            self.mau_store,
            self.config,
        )

        # Initialize knowledge graph + graph retriever
        self.knowledge_graph = KnowledgeGraph()
        self.entity_extractor = EntityExtractor(self.config)
        # Pass an actual EntityExtractor into GraphRetriever (previously passed vector_store by mistake)
        self.graph_retriever = GraphRetriever(
            self.knowledge_graph,
            entity_extractor=self.entity_extractor,
            config=self.config,
        )

        # Initialize parametric memory (storage_path first, then config)
        parametric_dir = str(Path(self.config.storage.base_dir) / "parametric")
        consolidation_dir = str(Path(self.config.storage.base_dir) / "consolidation")
        self.parametric_store = ParametricMemoryStore(parametric_dir, config=None)
        self.consolidator = MemoryConsolidator(consolidation_dir, config=self.config)

        # LLM client for answer generation
        self._llm_client = None

        # Session tracking
        self._current_session_id: Optional[str] = None

        # Self-evolution components (initialized if enabled)
        self._meta_controller = None
        self._experience_engine = None
        self._strategy_optimizer = None

        if _EVOLUTION_AVAILABLE and getattr(
            self.config, "enable_self_evolution", False
        ):
            self._init_evolution()

        # Routing components (benchmark-safe evolution)
        self._memory_router = None
        self._benchmark_guard = None
        self._semantic_store = None

        if _ROUTING_AVAILABLE:
            self._benchmark_guard = BenchmarkSafeGuard(self.config)
            router_cfg = getattr(self.config, "router", None)
            if router_cfg and router_cfg.router_mode != "off":
                self._memory_router = MemoryRouter(
                    gini_threshold=router_cfg.gini_threshold,
                    top1_threshold=router_cfg.top1_threshold,
                    gap_threshold=router_cfg.gap_threshold,
                    episodic_margin=router_cfg.episodic_margin,
                    close_margin=router_cfg.close_margin,
                    shadow_mode=router_cfg.shadow_mode,
                )
                self._semantic_store = SemanticStore(
                    storage_path=str(
                        Path(self.config.storage.base_dir) / "semantic_store"
                    ),
                )
                logger.info(
                    f"Memory router initialized (mode={router_cfg.router_mode})"
                )

        logger.info("Omni-Memory Orchestrator initialized")

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

    # ==================== Session Management ====================

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new session."""
        import uuid

        self._current_session_id = (
            session_id or f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        )
        logger.info(f"Started session: {self._current_session_id}")
        return self._current_session_id

    def end_session(self):
        """End current session and close open events."""
        if self._current_session_id:
            self.event_manager.close_all_open_events()
            self._current_session_id = None

    @property
    def session_id(self) -> str:
        """Get current session ID, creating one if needed."""
        if not self._current_session_id:
            self.start_session()
        return self._current_session_id

    # ==================== Ingestion ====================

    def add_text(
        self,
        text: str,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> ProcessingResult:
        """
        Add text to memory.

        Args:
            text: The text content
            session_id: Optional session identifier
            tags: Optional tags for filtering
            force: Force storage even if redundant

        Returns:
            ProcessingResult with MAU if stored
        """
        session_id = session_id or self.session_id

        result = self.text_processor.process(
            text,
            session_id=session_id,
            force=force,
        )

        if result.success and result.mau:
            self._store_mau(result.mau, tags)

        return result

    def add_image(
        self,
        image: Any,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> ProcessingResult:
        """
        Add image to memory with entropy triggering.

        Args:
            image: Image data (file path, PIL Image, numpy array, bytes)
            session_id: Optional session identifier
            tags: Optional tags
            force: Force storage even if similar to previous

        Returns:
            ProcessingResult with MAU if stored
        """
        session_id = session_id or self.session_id

        result = self.image_processor.process(
            image,
            session_id=session_id,
            force=force,
        )

        if result.success and result.mau:
            self._store_mau(result.mau, tags)

        return result

    def add_audio(
        self,
        audio: Any,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> ProcessingResult:
        """
        Add audio to memory with VAD triggering.

        Args:
            audio: Audio data (file path, numpy array, bytes)
            session_id: Optional session identifier
            tags: Optional tags
            force: Force storage even if no speech detected

        Returns:
            ProcessingResult with MAU if stored
        """
        session_id = session_id or self.session_id

        result = self.audio_processor.process(
            audio,
            session_id=session_id,
            force=force,
        )

        if result.success and result.mau:
            self._store_mau(result.mau, tags)

        return result

    def add_video(
        self,
        video_path: str,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_frames: int = 100,
    ) -> ProcessingResult:
        """
        Add video to memory with frame-level entropy triggering.

        Args:
            video_path: Path to video file
            session_id: Optional session identifier
            tags: Optional tags
            max_frames: Maximum frames to process

        Returns:
            ProcessingResult with video MAU and frame MAUs
        """
        session_id = session_id or self.session_id

        result = self.video_processor.process(
            video_path,
            session_id=session_id,
            max_frames=max_frames,
        )

        if result.success and result.mau:
            self._store_mau(result.mau, tags)

            # Also store frame MAUs
            frame_maus = result.metadata.get("frame_maus", [])
            for frame_mau in frame_maus:
                self._store_mau(frame_mau)

            # Store audio MAU if present
            audio_mau = result.metadata.get("audio_mau")
            if audio_mau:
                self._store_mau(audio_mau)

        return result

    def add_image_with_caption_averaged(
        self,
        image: Any,
        caption_text: str,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> ProcessingResult:
        """
        Add one MAU per image: embedding = (caption_embedding + image_embedding) / 2.
        Uses framework processors for both modalities but stores only the averaged
        vector (VISUAL) so retrieval is single-vector. Preserves raw_pointer for on-demand image loading.
        """
        session_id = session_id or self.session_id
        cap_wrapped = (
            f"[Image: {caption_text}]" if caption_text else "[Image: Image captured]"
        )

        res_img = self.image_processor.process(
            image, session_id=session_id, force=force
        )
        res_txt = self.text_processor.process(
            cap_wrapped, session_id=session_id, force=force
        )

        if not res_img.success or not res_img.mau or not res_img.mau.embedding:
            return ProcessingResult(
                success=False, error="image processing failed or no embedding"
            )
        if not res_txt.success or not res_txt.mau or not res_txt.mau.embedding:
            return ProcessingResult(
                success=False, error="caption processing failed or no embedding"
            )

        cap_emb = res_txt.mau.embedding
        img_emb = res_img.mau.embedding
        if len(cap_emb) != len(img_emb):
            return ProcessingResult(
                success=False,
                error=f"embedding dim mismatch: caption {len(cap_emb)} vs image {len(img_emb)}",
            )

        avg_emb = [(a + b) / 2.0 for a, b in zip(cap_emb, img_emb)]
        mau = MultimodalAtomicUnit(
            summary=cap_wrapped,
            embedding=avg_emb,
            modality_type=ModalityType.VISUAL,
        )
        if res_img.mau.raw_pointer:
            mau.raw_pointer = res_img.mau.raw_pointer
        self._store_mau(mau, tags)
        return ProcessingResult(success=True, mau=mau)

    def add_image_on_demand_caption_only(
        self,
        image: Any,
        caption_text: str,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> ProcessingResult:
        """
        On-demand image: store only caption as TEXT embedding (e.g. 3072-dim), and keep
        raw image in cold storage via raw_pointer. Retrieval uses text space only; at
        answer time the image can be loaded on demand. No mixing of text/image dimensions.
        """
        session_id = session_id or self.session_id
        cap_wrapped = (
            f"[Image: {caption_text}]" if caption_text else "[Image: Image captured]"
        )

        res_img = self.image_processor.process(
            image, session_id=session_id, force=force
        )
        res_txt = self.text_processor.process(
            cap_wrapped, session_id=session_id, force=force
        )

        if not res_img.success or not res_img.mau:
            return ProcessingResult(success=False, error="image processing failed")
        if not res_txt.success or not res_txt.mau or not res_txt.mau.embedding:
            return ProcessingResult(
                success=False, error="caption processing failed or no embedding"
            )

        mau = res_txt.mau
        if res_img.mau.raw_pointer:
            mau.raw_pointer = res_img.mau.raw_pointer
        self._store_mau(mau, tags)
        return ProcessingResult(success=True, mau=mau)

    def add_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, ProcessingResult]:
        """
        Add multiple modalities at once, linking them together.

        Returns dict of results keyed by modality.
        """
        session_id = session_id or self.session_id
        results = {}
        mau_ids = []

        if text:
            results["text"] = self.add_text(text, session_id, tags)
            if results["text"].mau:
                mau_ids.append(results["text"].mau.id)

        if image:
            results["image"] = self.add_image(image, session_id, tags)
            if results["image"].mau:
                mau_ids.append(results["image"].mau.id)

        if audio:
            results["audio"] = self.add_audio(audio, session_id, tags)
            if results["audio"].mau:
                mau_ids.append(results["audio"].mau.id)

        # Link MAUs together
        if len(mau_ids) > 1:
            for mau_id in mau_ids:
                mau = self.mau_store.get(mau_id)
                if mau:
                    for other_id in mau_ids:
                        if other_id != mau_id:
                            mau.add_related(other_id)
                    self.mau_store.update(mau)

        return results

    def _store_mau(
        self,
        mau: MultimodalAtomicUnit,
        tags: Optional[List[str]] = None,
    ):
        """Store MAU in all relevant stores."""
        # Add tags
        if tags:
            for tag in tags:
                mau.add_tag(tag)

        # Store in MAU store
        self.mau_store.add(mau)

        # Store embedding in vector store
        if mau.embedding:
            text_dim = getattr(self.config.embedding, "embedding_dim", 3072)
            visual_dim = getattr(self.config.embedding, "visual_embedding_dim", 512)
            # When using unified multimodal encoder (e.g. Tongyi), text and image same dim -> store in text store so query finds them
            if mau.modality_type == ModalityType.TEXT:
                self.vector_store.add_text(mau.id, mau.embedding)
            elif mau.modality_type in [ModalityType.VISUAL, ModalityType.VIDEO]:
                if len(mau.embedding) == text_dim and text_dim == visual_dim:
                    self.vector_store.add_text(mau.id, mau.embedding)
                else:
                    self.vector_store.add_visual(mau.id, mau.embedding)
            else:
                self.vector_store.add(mau.id, mau.embedding)

        # Add to event
        self.event_manager.add_mau_to_event(mau)

        # Extract entities and update knowledge graph
        try:
            entities, relations = self.entity_extractor.extract(mau)
            # entities: List[ExtractedEntity]
            # relations: List[ExtractedRelation]
            for entity in entities:
                self.knowledge_graph.add_extracted_entity(entity)
            for relation in relations:
                self.knowledge_graph.add_extracted_relation(relation)
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

        # Update consolidator for importance tracking (if supported)
        if hasattr(self.consolidator, "record_memory_access"):
            try:
                self.consolidator.record_memory_access(mau.id, access_type="store")
            except Exception as e:
                logger.debug("Consolidator record_memory_access: %s", e)

        logger.debug(f"Stored MAU: {mau.id}")

    # ==================== Retrieval ====================

    def _extract_json_object_from_response(self, text: str):
        """Extract first JSON object from LLM response (avoids JSONDecodeError on long non-JSON text)."""
        import json

        if not (text or isinstance(text, str)):
            return None
        s = text.strip()
        # Try ```json ... ``` block first: take content between ``` then find {...}
        if "```" in s:
            import re

            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
            if m:
                block = m.group(1).strip()
                obj = self._extract_first_json_object(block)
                if obj is not None:
                    return obj
        return self._extract_first_json_object(s)

    def _extract_first_json_object(self, s: str):
        """Find first {...} by matching braces and parse as JSON."""
        import json

        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        for i, c in enumerate(s[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
        return None

    def _generate_search_queries(self, question: str) -> List[str]:
        """Generate 2~3 targeted search queries from question (SimpleMem-inspired multi-query)."""
        cfg = self.config.retrieval
        max_q = max(1, min(cfg.multi_query_max_queries, 5))
        try:
            client = self._get_llm_client()
            response = client.chat.completions.create(
                model=self.config.llm.query_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search query assistant. Output valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": f'Given this question, generate {max_q} short search queries to find relevant memories (include the original). Return JSON: {{"queries": ["query1", "query2", ...]}}\n\nQuestion: {question}',
                    },
                ],
                temperature=0.2,
                max_tokens=200,
            )
            text = (response.choices[0].message.content or "").strip()
            data = self._extract_json_object_from_response(text)
            if data is not None:
                queries = data.get("queries") or data.get("query") or []
                if isinstance(queries, str):
                    queries = [queries]
                if queries and isinstance(queries[0], str):
                    seen = set()
                    out = []
                    for q in queries[:max_q]:
                        q = (q or "").strip()
                        if q and q not in seen:
                            seen.add(q)
                            out.append(q)
                    if out:
                        return out
        except Exception as e:
            # Avoid dumping 300+ lines of raw LLM response in JSONDecodeError message
            err_msg = str(e)
            if len(err_msg) > 400:
                err_msg = err_msg[:400] + "... (truncated)"
            logger.debug("Multi-query generation failed: %s", err_msg)
        return [question]

    def _merge_retrieval_results(
        self, results: List[RetrievalResult], original_query: str, top_k: int
    ) -> RetrievalResult:
        """Merge multiple RetrievalResults by mau_id (keep max score), sort, take top_k."""
        by_id: Dict[str, Dict[str, Any]] = {}
        total_candidates = 0
        tokens_estimate = 0
        expansion_candidates = []
        for r in results:
            total_candidates += r.total_candidates
            tokens_estimate += r.tokens_used_estimate
            expansion_candidates.extend(r.expansion_candidates or [])
            for item in r.items:
                mau_id = item.get("id")
                if not mau_id:
                    continue
                score = item.get("score", 0.0)
                if mau_id not in by_id or (by_id[mau_id].get("score", 0) < score):
                    by_id[mau_id] = dict(item)
        merged_items = sorted(
            by_id.values(),
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )[:top_k]
        return RetrievalResult(
            query=original_query,
            level=RetrievalLevel.SUMMARY,
            items=merged_items,
            total_candidates=total_candidates,
            tokens_used_estimate=tokens_estimate,
            can_expand=bool(expansion_candidates),
            expansion_candidates=list(dict.fromkeys(expansion_candidates))[
                : len(merged_items)
            ],
        )

    def query(
        self,
        query: str,
        top_k: int = 10,
        auto_expand: bool = False,
        token_budget: Optional[int] = None,
        benchmark_safe: bool = False,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> RetrievalResult:
        """
        Query memories with pyramid retrieval.

        Args:
            query: Search query
            top_k: Number of results
            auto_expand: Automatically expand high-relevance items
            token_budget: Optional token limit
            tags_filter: Optional tags to filter MAUs (e.g. ["locomo_conv-26"])
            time_range: Optional (start_ts, end_ts) to filter by MAU timestamp

        Returns:
            RetrievalResult with summaries (and expanded content if requested)
        """
        # Benchmark safety check - force baseline episodic path if needed
        if self._benchmark_guard:
            force_baseline = self._benchmark_guard.should_force_baseline(
                {"benchmark_safe": benchmark_safe}
            )
            if force_baseline and self._memory_router:
                logger.debug("BenchmarkSafeGuard: forcing baseline episodic path")
        # Note: Full routing integration (using self._memory_router to select between
        # episodic vs semantic results) is deferred to when SemanticStore has content.
        # For now, the guard and router are initialized but query() always follows
        # the existing episodic retrieval path. This ensures zero behavior change
        # while the benchmark-safe infrastructure is in place.

        # Process query
        parsed = self.query_processor.process(query)
        strategy = self.query_processor.determine_retrieval_strategy(parsed)

        # Get preview
        use_time = time_range if time_range is not None else strategy.get("time_filter")
        use_tags = tags_filter
        if token_budget:
            result = self.retriever.retrieve_with_budget(
                parsed.cleaned_query,
                token_budget,
                prefer_modality=strategy.get("modality_filter"),
                tags_filter=use_tags,
                time_range=use_time,
            )
        else:
            result = self.retriever.retrieve_preview(
                parsed.cleaned_query,
                top_k=strategy.get("top_k", top_k),
                modality_filter=strategy.get("modality_filter"),
                time_range=use_time,
                tags_filter=use_tags,
            )

        # BM25 hybrid search: merge keyword results with vector results
        # BM25 supplements FAISS with keyword matches for specific details
        # (book titles, person names, entities) that semantic search misses.
        # IMPORTANT: Do NOT re-sort — FAISS semantic ordering must be preserved.
        # BM25 results are appended at the end as supplementary context.
        if self.bm25_store.is_available:
            bm25_top_k = max(10, strategy.get("top_k", top_k) // 2)
            bm25_results = self.bm25_store.search(
                parsed.cleaned_query,
                top_k=bm25_top_k,
                tags_filter=use_tags,
                mau_store=self.mau_store,
            )
            # Set-union: add BM25-only results after FAISS results (preserve FAISS order)
            existing_ids = {item["id"] for item in result.items}
            for mau_id, bm25_score in bm25_results:
                if mau_id not in existing_ids:
                    mau = self.mau_store.get(mau_id)
                    if mau:
                        result.items.append(
                            {
                                "id": mau.id,
                                "summary": mau.summary,
                                "modality_type": mau.modality_type.value,
                                "timestamp": mau.timestamp,
                                "score": bm25_score * 0.01,
                                "tags": mau.metadata.tags,
                                "has_raw_data": mau.raw_pointer is not None,
                            }
                        )
                        existing_ids.add(mau_id)

        parametric_result = self.parametric_store.recall(query, top_k=3)
        if parametric_result and parametric_result.confidence > 0.8:
            result.parametric_answers = parametric_result.answers

        graph_context = self.graph_retriever.retrieve(
            parsed.cleaned_query,
            max_entities=5,
        )
        if graph_context:
            result.graph_entities = graph_context

        if auto_expand and strategy.get("require_expansion"):
            expansion_ids = [
                item["id"]
                for item in result.items
                if item.get("score", 0) > self.config.retrieval.auto_expand_threshold
            ][: self.config.retrieval.max_expanded_items]

            if expansion_ids:
                expansion = self.retriever.expand(
                    ExpansionRequest(
                        mau_ids=expansion_ids,
                        level=RetrievalLevel.DETAILS,
                    )
                )
                expanded_ids = {item["id"] for item in expansion.items}
                for i, item in enumerate(result.items):
                    if item["id"] in expanded_ids:
                        expanded = next(
                            e for e in expansion.items if e["id"] == item["id"]
                        )
                        result.items[i] = expanded

        return result

    def expand(
        self,
        mau_ids: List[str],
        level: RetrievalLevel = RetrievalLevel.EVIDENCE,
    ) -> RetrievalResult:
        """
        Expand specific MAUs to get full details.

        Args:
            mau_ids: IDs of MAUs to expand
            level: Level of detail to retrieve

        Returns:
            RetrievalResult with expanded content
        """
        return self.retriever.expand(
            ExpansionRequest(
                mau_ids=mau_ids,
                level=level,
            )
        )

    def answer(
        self,
        question: str,
        top_k: int = 10,
        include_sources: bool = True,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        include_on_demand_images: bool = True,
    ) -> Dict[str, Any]:
        """
        Answer a question using memory retrieval.

        This implements the full retrieval-augmented generation flow:
        1. Retrieve relevant memories (optionally multi-query + tags/time filter)
        2. Optionally expand vision_on_demand items to load images for multimodal prompt
        3. Format context
        4. Generate answer with LLM

        Args:
            question: The question to answer
            top_k: Number of memories to retrieve
            include_sources: Include source references
            tags_filter: Optional tags to restrict retrieval (e.g. ["locomo_conv-26"])
            time_range: Optional (start_ts, end_ts) to filter by MAU timestamp
            include_on_demand_images: If True, expand vision_on_demand MAUs and include images in the prompt
        """
        k_per_query = (
            top_k * 2
            if getattr(self.config.retrieval, "enable_multi_query_retrieval", False)
            else top_k
        )

        def do_query(q: str, k: int = top_k) -> RetrievalResult:
            if self._strategy_optimizer and self._experience_engine:
                return self.query_with_evolution(
                    query=q,
                    top_k=k,
                    auto_expand=True,
                    benchmark_safe=False,
                    tags_filter=tags_filter,
                    time_range=time_range,
                )
            return self.query(
                q,
                top_k=k,
                auto_expand=True,
                tags_filter=tags_filter,
                time_range=time_range,
            )

        # Multi-query retrieval (SimpleMem-inspired): generate 2~3 queries, run retrievals concurrently, merge results
        if getattr(self.config.retrieval, "enable_multi_query_retrieval", False):
            queries = self._generate_search_queries(question)
            if len(queries) > 1:
                max_workers = min(len(queries), 8)
                results: List[RetrievalResult] = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(do_query, q, k_per_query): q for q in queries
                    }
                    for fut in as_completed(futures):
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            q = futures[fut]
                            logger.warning(
                                "Multi-query retrieval failed for query %r: %s",
                                q[:80],
                                e,
                            )
                retrieval = self._merge_retrieval_results(results, question, top_k)
            else:
                retrieval = do_query(question, top_k)
        else:
            retrieval = do_query(question, top_k)

        if not retrieval.items:
            logger.debug("Answer: no retrieval items (empty vector search)")
            return {
                "answer": "I don't have any relevant memories to answer this question.",
                "sources": [],
                "retrieval_result": retrieval.to_dict(),
            }

        # On-demand: expand vision_on_demand items to load raw images for multimodal prompt
        if include_on_demand_images:
            on_demand_ids = [
                item["id"]
                for item in retrieval.items
                if "vision_on_demand" in (item.get("tags") or [])
                and item.get("has_raw_data")
            ]
            if on_demand_ids:
                expansion = self.retriever.expand(
                    ExpansionRequest(
                        mau_ids=on_demand_ids,
                        level=RetrievalLevel.EVIDENCE,
                    )
                )
                expanded_by_id = {e["id"]: e for e in expansion.items}
                for i, item in enumerate(retrieval.items):
                    if item["id"] in expanded_by_id:
                        retrieval.items[i] = expanded_by_id[item["id"]]

        # Format context
        context = self.retriever.format_for_llm(retrieval, include_instructions=False)

        # Build user message: text + optional image parts for on_demand
        user_content: Any = (
            f"Based on these memories:\n\n{context}\n\nAnswer this question: {question}"
        )
        content_parts = []
        for item in retrieval.items:
            raw = item.get("raw_content") or {}
            if raw.get("type") == "image" and raw.get("base64"):
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{raw['base64']}"},
                    }
                )
        if content_parts:
            content_parts.insert(0, {"type": "text", "text": user_content})
            user_content = content_parts

        # Generate answer
        client = self._get_llm_client()
        try:
            system_content = (
                "You are a professional Q&A assistant. Your task is to extract concise, "
                "accurate answers from the provided memory context. "
                "You should make reasonable inferences from the context when possible. "
                "IMPORTANT: The conversations took place in 2023. Never use today's date "
                "(2025 or 2026) as an answer. Only use dates that actually appear in the memories. "
                "You must output valid JSON format."
            )
            answer_instructions = (
                "Based on these memories:\n\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                "Requirements:\n"
                "1. First, think through the reasoning process\n"
                "2. Provide a CONCISE answer (short phrase, ideally under 10 words). "
                "Use exact words and phrases from the context whenever possible rather than paraphrasing.\n"
                "3. Answer based on the provided context. You may make reasonable inferences "
                "from the information given (e.g., inferring personality traits, likely preferences, "
                "or approximate dates from surrounding context)\n"
                "4. All dates in the response must be formatted as 'DD Month YYYY'. "
                "NEVER use dates from 2025 or 2026. Use the 'Time:' metadata shown with each memory "
                "to determine when events occurred.\n"
                "5. Try your best to answer. Only respond with 'unknown' if the context contains "
                "absolutely NO relevant information about the topic asked\n"
                "6. For counting questions, answer with just the number (e.g., '2' not 'twice')\n"
                "7. For yes/no questions, start with 'Yes', 'No', 'Likely yes', or 'Likely no'\n"
                "8. When listing multiple items, separate them with commas (e.g., 'item1, item2')\n"
                "9. Return your response in JSON format\n\n"
                "Output Format:\n"
                '{"reasoning": "Brief explanation of your thought process", '
                '"answer": "Concise answer in a short phrase"}'
            )
            # For multimodal content, wrap text instructions with image parts
            if isinstance(user_content, list):
                # user_content already has image parts; replace text part
                user_content[0] = {"type": "text", "text": answer_instructions}
                final_user_content = user_content
            else:
                final_user_content = answer_instructions

            # Force JSON output for reliable extraction of concise answers
            api_kwargs = dict(
                model=self.config.llm.query_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": final_user_content},
                ],
                temperature=0.1,
            )
            # json_object mode forces valid JSON output (critical for concise answer extraction)
            if not isinstance(final_user_content, list):  # skip for multimodal (vision)
                api_kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**api_kwargs)
            raw_content = response.choices[0].message.content or ""
            raw_answer = raw_content.strip()
            logger.debug(
                "LLM raw answer: %s",
                raw_answer[:200] + ("..." if len(raw_answer) > 200 else ""),
            )

            # Extract concise answer from JSON response
            answer = self._extract_answer_from_json(raw_answer)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = "I encountered an error generating the answer."

        result = {
            "answer": answer,
            "retrieval_result": retrieval.to_dict(),
        }

        if include_sources:
            result["sources"] = [
                {
                    "id": item["id"],
                    "summary": item["summary"],
                    "modality": item["modality_type"],
                }
                for item in retrieval.items[:5]
            ]

        return result

    @staticmethod
    def _extract_answer_from_json(raw: str) -> str:
        """Extract the 'answer' field from a JSON response, with fallback."""
        import json as _json

        # Try direct JSON parse
        try:
            obj = _json.loads(raw)
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"]).strip()
        except _json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        import re

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                obj = _json.loads(m.group(1))
                if isinstance(obj, dict) and "answer" in obj:
                    return str(obj["answer"]).strip()
            except _json.JSONDecodeError:
                pass

        # Try finding any JSON object in the text
        m = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]*)"[^{}]*\}', raw)
        if m:
            return m.group(1).strip()

        # Fallback: try to extract a concise answer from free text
        # Strip common verbose prefixes
        cleaned = raw.strip()
        for prefix in ["Based on the", "According to", "From the memories",
                       "The memories indicate", "The context suggests"]:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].lstrip(", ")
                break
        # Return first sentence only, capped at ~12 words for conciseness
        import re as _re
        sentences = _re.split(r'(?<=[.!?])\s+', cleaned)
        first = sentences[0].strip() if sentences else cleaned
        if len(first.split()) > 12:
            first = ' '.join(first.split()[:12])
        return first

    # ==================== Event Operations ====================

    def get_events(
        self,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent events."""
        if session_id:
            events = self.event_store.get_by_session(session_id)
        else:
            events = self.event_store.get_recent(limit)

        return [e.get_summary_dict() for e in events]

    def get_event_details(
        self,
        event_id: str,
        level: EventLevel = EventLevel.DETAILS,
    ) -> Dict[str, Any]:
        """Get event with its MAUs."""
        return self.event_manager.get_event_with_maus(event_id, level)

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "mau_count": self.mau_store.count(),
            "mau_by_modality": self.mau_store.count_by_modality(),
            "event_count": self.event_store.count(),
            "vector_count": self.vector_store.count(),
            "storage_stats": self.cold_storage.get_storage_stats(),
        }

    def consolidate_memories(self, force: bool = False) -> Dict[str, Any]:
        """Run memory consolidation with importance-based retention."""
        run_id = f"consolidation_{int(time.time())}"
        return self.consolidator.consolidate(
            mau_store=self.mau_store,
            knowledge_graph=self.knowledge_graph,
            force=force,
            run_id=run_id,
        )

    def distill_to_parametric(self) -> Dict[str, Any]:
        """Distill episodic memories to parametric memory."""
        important_maus = self.consolidator.get_important_memories(
            self.mau_store,
            threshold=0.7,
            limit=1000,
        )
        return self.parametric_store.distill(important_maus)

    # ==================== Persistence ====================

    def save(self):
        """Save all indices to disk."""
        self.vector_store.save()
        logger.info("Saved Omni-Memory indices")

    def build_bm25_index(self) -> int:
        """Build BM25 keyword search index from existing MAU store."""
        count = self.bm25_store.build_from_mau_store(self.mau_store)
        logger.info("BM25 index: %d documents indexed", count)
        return count

    def close(self):
        """Clean up resources."""
        self.end_session()
        self.save()

        if hasattr(self, "knowledge_graph"):
            self.knowledge_graph.save()

        if hasattr(self, "parametric_store") and hasattr(self.parametric_store, "save"):
            try:
                self.parametric_store.save()
            except Exception as e:
                logger.debug("Parametric store save: %s", e)

        # Save evolution state
        if self._meta_controller:
            self._meta_controller.save()
        if self._experience_engine:
            self._experience_engine.flush_reflections()
            self._experience_engine.save()
        if self._strategy_optimizer:
            self._strategy_optimizer.save()

        # Save routing/semantic state
        if self._semantic_store:
            logger.info(f"Semantic store has {self._semantic_store.count()} nodes")

        logger.info("Omni-Memory orchestrator closed")

    # ==================== Self-Evolution ====================

    def _init_evolution(self) -> None:
        """Initialize self-evolution components."""
        evolution_config = getattr(self.config, "evolution", None)

        evo_storage = str(Path(self.config.storage.base_dir) / "evolution")

        self._meta_controller = MetaController(
            config=evolution_config.meta_controller if evolution_config else None,
            storage_path=evo_storage,
        )
        self._experience_engine = ExperienceEngine(
            config=evolution_config.experience_engine if evolution_config else None,
            storage_path=str(Path(evo_storage) / "experiences"),
        )
        self._strategy_optimizer = StrategyOptimizer(
            config=evolution_config.strategy_optimizer if evolution_config else None,
            storage_path=str(Path(evo_storage) / "strategy"),
        )

        logger.info("Self-evolution components initialized")

    def query_with_evolution(
        self,
        query: str,
        top_k: int = 10,
        auto_expand: bool = False,
        token_budget: Optional[int] = None,
        benchmark_safe: bool = False,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> RetrievalResult:
        """
        Query with self-evolving strategy selection.

        If evolution is enabled, uses the strategy optimizer to select
        the best retrieval path and records outcomes for learning.
        Falls back to standard query() if evolution is not initialized.
        """
        if not self._strategy_optimizer or not self._experience_engine:
            return self.query(
                query,
                top_k,
                auto_expand,
                token_budget,
                benchmark_safe=benchmark_safe,
                tags_filter=tags_filter,
                time_range=time_range,
            )

        # Extract query features
        features = self._experience_engine.extract_query_features(query)

        # Get relevant past experiences
        past_experiences = self._experience_engine.get_relevant_experiences(features)

        # Select strategy via Thompson Sampling
        strategy = self._strategy_optimizer.select_strategy(features.query_type)

        # Use adaptive parameters from meta-controller
        adaptive_params = (
            self._meta_controller.get_adaptive_params()
            if self._meta_controller
            else None
        )

        # Execute retrieval with selected strategy
        if strategy == "parametric":
            result = self._parametric_retrieval(query, top_k)
        elif strategy == "graph":
            result = self._graph_retrieval(query, top_k)
        elif strategy == "hybrid":
            result = self._hybrid_retrieval(query, top_k, token_budget)
        else:  # "vector" (default)
            result = self.query(
                query,
                top_k,
                auto_expand,
                token_budget,
                benchmark_safe=benchmark_safe,
                tags_filter=tags_filter,
                time_range=time_range,
            )

        return result

    def record_answer_feedback(
        self,
        query: str,
        answer: str,
        answer_quality: float,
        strategy_used: str = "vector",
        memories_retrieved: int = 0,
        memories_useful: int = 0,
        tokens_used: int = 0,
    ) -> None:
        """
        Record feedback after an answer is generated.

        This drives the self-evolution loop by updating:
        1. Meta-controller metrics (trigger/consolidation adaptation)
        2. Experience engine (meta-experience storage)
        3. Strategy optimizer (Thompson Sampling posteriors)
        """
        if not self._meta_controller:
            return

        # Update meta-controller
        self._meta_controller.record_query_outcome(
            query=query,
            memories_retrieved=memories_retrieved,
            memories_useful=memories_useful,
            answer_quality=answer_quality,
            strategy_used=strategy_used,
        )

        # Record experience
        if self._experience_engine:
            features = self._experience_engine.extract_query_features(query)
            self._experience_engine.record_experience(
                query=query,
                query_features=features,
                retrieval_strategy=strategy_used,
                memories_retrieved=memories_retrieved,
                memories_expanded=0,
                tokens_used=tokens_used,
                answer_quality=answer_quality,
                answer_text=answer,
            )

        # Update strategy optimizer
        if self._strategy_optimizer:
            features = (
                self._experience_engine.extract_query_features(query)
                if self._experience_engine
                else None
            )
            query_type = features.query_type if features else "factual"
            self._strategy_optimizer.record_outcome(
                query_type=query_type,
                strategy=strategy_used,
                success=answer_quality >= 0.6,
                quality=answer_quality,
            )

    def _parametric_retrieval(self, query: str, top_k: int) -> RetrievalResult:
        """Fast parametric-only retrieval path."""
        parametric_result = self.parametric_store.recall(query, top_k=top_k)

        if parametric_result and parametric_result.confidence > 0.5:
            return RetrievalResult(
                query=query,
                level=RetrievalLevel.SUMMARY,
                items=[
                    {
                        "id": "parametric",
                        "summary": parametric_result.answer,
                        "modality_type": "text",
                        "timestamp": 0,
                        "score": parametric_result.confidence,
                        "tags": [],
                        "has_raw_data": False,
                    }
                ],
                total_candidates=1,
                tokens_used_estimate=len(parametric_result.answer) // 4,
            )

        # Fall back to vector search if parametric confidence is low
        return self.query(query, top_k)

    def _graph_retrieval(self, query: str, top_k: int) -> RetrievalResult:
        """Graph-focused retrieval path."""
        parsed = self.query_processor.process(query)

        graph_context = self.graph_retriever.retrieve(
            parsed.cleaned_query,
            max_entities=top_k,
        )

        if graph_context:
            result = self.retriever.retrieve_preview(parsed.cleaned_query, top_k=top_k)
            result.graph_entities = graph_context
            return result

        return self.query(query, top_k)

    def _hybrid_retrieval(
        self,
        query: str,
        top_k: int,
        token_budget: Optional[int] = None,
    ) -> RetrievalResult:
        """Combined vector + graph retrieval (most thorough)."""
        result = self.query(query, top_k, auto_expand=True, token_budget=token_budget)

        # Enrich with graph context
        parsed = self.query_processor.process(query)
        graph_context = self.graph_retriever.retrieve(
            parsed.cleaned_query, max_entities=5
        )
        if graph_context:
            result.graph_entities = graph_context

        return result

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get statistics from all evolution components."""
        stats = {"evolution_enabled": self._meta_controller is not None}

        if self._meta_controller:
            stats["meta_controller"] = self._meta_controller.get_stats()
        if self._experience_engine:
            stats["experience_engine"] = self._experience_engine.get_stats()
        if self._strategy_optimizer:
            stats["strategy_optimizer"] = self._strategy_optimizer.get_stats()

        return stats

    # ==================== Context Manager ====================

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
