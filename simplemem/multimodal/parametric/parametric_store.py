"""
Parametric Memory Store for Omni-Memory.

Provides a unified interface for storing and retrieving from
parametric (distilled) memory alongside episodic memory.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from simplemem.multimodal.parametric.memory_distiller import MemoryDistiller, DistillationConfig

logger = logging.getLogger(__name__)


@dataclass
class ParametricQueryResult:
    """Result from parametric memory query."""
    query: str
    answer: str
    confidence: float = 0.0
    source: str = "parametric"  # "parametric" or "fallback"
    latency_ms: float = 0.0

    @property
    def answers(self) -> List[str]:
        """List of answers (orchestrator expects .answers)."""
        return [self.answer] if self.answer else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "source": self.source,
            "latency_ms": self.latency_ms,
        }


class ParametricMemoryStore:
    """
    Parametric Memory Store - Fast recall pathway.
    
    Provides:
    1. Fast parametric recall (no retrieval overhead)
    2. Confidence scoring for parametric answers
    3. Fallback to retrieval when confidence is low
    4. Integration with distillation pipeline
    
    This implements the "fast thinking" pathway from dual-process theory,
    complementing the "slow thinking" retrieval pathway.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        config: Optional[DistillationConfig] = None,
        distiller: Optional[MemoryDistiller] = None,
    ):
        self.storage_path = Path(storage_path or "./omni_memory_data/parametric")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.config = config or DistillationConfig()
        self.distiller = distiller or MemoryDistiller(
            config=self.config,
            storage_path=str(self.storage_path),
        )
        
        # Confidence threshold for using parametric answer
        self.confidence_threshold = 0.6
        
        # Query cache for efficiency
        self._cache: Dict[str, ParametricQueryResult] = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 3600  # 1 hour

    def recall(self, query: str, top_k: int = 3) -> Optional[ParametricQueryResult]:
        """
        Recall from parametric memory (orchestrator interface).
        Returns a single result; top_k is ignored (store returns one answer per query).
        """
        try:
            return self.query(query, use_cache=True)
        except Exception as e:
            logger.debug("Parametric recall failed: %s", e)
            return None

    def query(
        self,
        question: str,
        use_cache: bool = True,
    ) -> ParametricQueryResult:
        """
        Query parametric memory.
        
        Args:
            question: Question to answer
            use_cache: Whether to use cached results
            
        Returns:
            ParametricQueryResult with answer and confidence
        """
        # Check cache
        cache_key = question.lower().strip()
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Check TTL
            if time.time() - cached.latency_ms < self._cache_ttl_seconds:
                return cached
        
        # Query parametric model
        start_time = time.time()
        
        try:
            answer = self.distiller.query(question)
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate confidence based on answer quality
            confidence = self._estimate_confidence(question, answer)
            
            result = ParametricQueryResult(
                query=question,
                answer=answer,
                confidence=confidence,
                source="parametric",
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"Parametric query failed: {e}")
            result = ParametricQueryResult(
                query=question,
                answer="",
                confidence=0.0,
                source="error",
                latency_ms=(time.time() - start_time) * 1000,
            )
        
        # Cache result
        if use_cache:
            self._add_to_cache(cache_key, result)
        
        return result
    
    def _estimate_confidence(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        Estimate confidence in parametric answer.
        
        Heuristics:
        - Empty/short answers: low confidence
        - "I don't know" type answers: low confidence
        - Longer, specific answers: higher confidence
        """
        if not answer or len(answer.strip()) < 10:
            return 0.1
        
        # Check for uncertainty markers
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "i cannot",
            "no information",
            "not trained",
            "not yet trained",
        ]
        
        answer_lower = answer.lower()
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                return 0.2
        
        # Length-based confidence (longer = more confident, up to a point)
        length_score = min(len(answer) / 200, 1.0) * 0.4
        
        # Specificity score (contains numbers, names, etc.)
        specificity_score = 0.0
        if any(char.isdigit() for char in answer):
            specificity_score += 0.2
        if any(word[0].isupper() for word in answer.split() if word):
            specificity_score += 0.2
        
        return min(0.5 + length_score + specificity_score, 1.0)
    
    def _add_to_cache(
        self,
        key: str,
        result: ParametricQueryResult,
    ) -> None:
        """Add result to cache with eviction."""
        if len(self._cache) >= self._cache_max_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = result
    
    def is_confident(
        self,
        result: ParametricQueryResult,
    ) -> bool:
        """Check if parametric result is confident enough to use."""
        return result.confidence >= self.confidence_threshold
    
    def add_memory(
        self,
        mau_summary: str,
        mau_details: Optional[str] = None,
        mau_id: Optional[str] = None,
        modality: str = "text",
        entity_frequency: int = 1,
    ) -> None:
        """
        Add memory to distillation pipeline.
        
        Creates QA pairs for future distillation.
        """
        self.distiller.add_qa_from_mau(
            mau_summary=mau_summary,
            mau_details=mau_details,
            mau_id=mau_id,
            modality=modality,
            entity_frequency=entity_frequency,
        )
    
    def trigger_distillation(
        self,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Trigger distillation if conditions are met."""
        return self.distiller.distill(force=force)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parametric memory statistics."""
        distiller_stats = self.distiller.get_stats()
        return {
            "cache_size": len(self._cache),
            "confidence_threshold": self.confidence_threshold,
            **distiller_stats,
        }
    
    def clear_cache(self) -> None:
        """Clear query cache."""
        self._cache.clear()
