"""
Experience Engine for Self-Evolving Omni-Memory.

Accumulates meta-experiences from query-answer cycles and enables
self-reflection to learn from successes and failures.

Inspired by:
- ReasoningBank (arXiv 2509.25140): Self-judged success/failure distillation
- Self-Evolving GPT (ACL 2024): Experience accumulation and transfer
- A-MEM (NeurIPS 2025): Dynamic memory linking
"""

import logging
import json
import threading
import time
import uuid
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QueryFeatures:
    """Extracted features of a query for strategy matching."""
    query_type: str = "factual"
    entity_count: int = 0
    has_temporal_signal: bool = False
    estimated_complexity: float = 0.5
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryFeatures":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MetaExperience:
    """A single meta-experience from a query-answer cycle."""
    experience_id: str = field(default_factory=lambda: f"exp_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}")
    timestamp: float = field(default_factory=time.time)

    query_text: str = ""
    query_features: QueryFeatures = field(default_factory=QueryFeatures)

    retrieval_strategy: str = "hybrid"
    memories_retrieved: int = 0
    memories_expanded: int = 0
    tokens_used: int = 0

    answer_quality: float = 0.5
    success: bool = False

    lessons_learned: str = ""
    strategy_suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "timestamp": self.timestamp,
            "query_text": self.query_text,
            "query_features": self.query_features.to_dict(),
            "retrieval_strategy": self.retrieval_strategy,
            "memories_retrieved": self.memories_retrieved,
            "memories_expanded": self.memories_expanded,
            "tokens_used": self.tokens_used,
            "answer_quality": self.answer_quality,
            "success": self.success,
            "lessons_learned": self.lessons_learned,
            "strategy_suggestion": self.strategy_suggestion,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaExperience":
        data = dict(data)
        features_data = data.pop("query_features", {})
        features = QueryFeatures.from_dict(features_data) if features_data else QueryFeatures()
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(query_features=features, **filtered)


class ExperienceEngine:
    """
    Experience accumulation and self-reflection engine.

    Closed-loop cycle:
    1. After query → self-evaluate answer quality
    2. Store meta-experience (query features + strategy + outcome)
    3. Before query → retrieve similar past experiences
    4. Use past experiences to inform strategy selection
    """

    def __init__(self, config=None, storage_path: Optional[str] = None):
        from simplemem.multimodal.evolution.evolution_config import ExperienceEngineConfig
        self.config = config or ExperienceEngineConfig()
        self.storage_path = Path(storage_path or "./omni_memory_data/evolution/experiences")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._experiences: List[MetaExperience] = []
        self._type_index: Dict[str, List[int]] = defaultdict(list)
        self._experiences_file = self.storage_path / "experiences.jsonl"

        self._pending_reflections: List[Dict[str, Any]] = []
        self._llm_client = None
        self._lock = threading.Lock()

        self._load_experiences()
        logger.info(f"ExperienceEngine initialized ({len(self._experiences)} experiences loaded)")

    def extract_query_features(self, query: str) -> QueryFeatures:
        """Extract features from query using lightweight heuristics."""
        query_lower = query.lower()
        tokens = query.split()
        token_count = len(tokens)

        temporal_keywords = [
            "when", "before", "after", "during", "yesterday", "last week",
            "recently", "earlier", "later", "time", "date", "year", "month"
        ]
        reasoning_keywords = [
            "why", "how", "explain", "because", "reason", "cause",
            "compare", "difference", "relationship", "implication"
        ]
        cross_modal_keywords = [
            "show me", "image", "picture", "video", "audio", "sound",
            "visual", "look like", "hear", "watch"
        ]
        entity_keywords = ["who", "where", "which person", "which place"]

        has_temporal = any(kw in query_lower for kw in temporal_keywords)
        has_reasoning = any(kw in query_lower for kw in reasoning_keywords)
        has_cross_modal = any(kw in query_lower for kw in cross_modal_keywords)
        has_entity_signal = any(kw in query_lower for kw in entity_keywords)

        if has_cross_modal:
            query_type = "cross_modal"
        elif has_temporal:
            query_type = "temporal"
        elif has_reasoning:
            query_type = "reasoning"
        elif has_entity_signal:
            query_type = "entity_centric"
        else:
            query_type = "factual"

        entity_count = sum(
            1 for i, word in enumerate(tokens)
            if i > 0 and word and word[0].isupper() and len(word) > 1
        )

        complexity = min(1.0, (
            0.2 * (token_count / 30) +
            0.3 * (1.0 if has_reasoning else 0.0) +
            0.2 * (entity_count / 5) +
            0.15 * (1.0 if has_temporal else 0.0) +
            0.15 * (1.0 if has_cross_modal else 0.0)
        ))

        return QueryFeatures(
            query_type=query_type,
            entity_count=entity_count,
            has_temporal_signal=has_temporal,
            estimated_complexity=complexity,
            token_count=token_count,
        )

    def record_experience(
        self,
        query: str,
        query_features: QueryFeatures,
        retrieval_strategy: str,
        memories_retrieved: int,
        memories_expanded: int,
        tokens_used: int,
        answer_quality: float,
        answer_text: str = "",
        context_text: str = "",
    ) -> MetaExperience:
        """Record a meta-experience from a completed query-answer cycle (thread-safe)."""
        to_reflect = None
        with self._lock:
            experience = MetaExperience(
                query_text=query,
                query_features=query_features,
                retrieval_strategy=retrieval_strategy,
                memories_retrieved=memories_retrieved,
                memories_expanded=memories_expanded,
                tokens_used=tokens_used,
                answer_quality=answer_quality,
                success=answer_quality >= 0.6,
            )

            idx = len(self._experiences)
            self._experiences.append(experience)
            self._type_index[query_features.query_type].append(idx)

            with open(self._experiences_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(experience.to_dict(), ensure_ascii=False) + '\n')

            if self.config.enable_self_reflection and answer_text:
                self._pending_reflections.append({
                    "experience_idx": idx,
                    "query": query,
                    "answer": answer_text,
                    "context": context_text[:500],
                    "strategy": retrieval_strategy,
                })

                if len(self._pending_reflections) >= self.config.reflection_batch_size:
                    batch = self._pending_reflections[:self.config.reflection_batch_size]
                    self._pending_reflections = self._pending_reflections[self.config.reflection_batch_size:]
                    to_reflect = batch
                else:
                    to_reflect = None
        if to_reflect:
            self._run_batch_reflection(to_reflect)

        logger.debug(
            f"Recorded experience: type={query_features.query_type}, "
            f"strategy={retrieval_strategy}, quality={answer_quality:.2f}"
        )

        return experience

    def get_relevant_experiences(
        self,
        query_features: QueryFeatures,
        top_k: Optional[int] = None,
    ) -> List[MetaExperience]:
        """Retrieve past experiences similar to the current query."""
        top_k = top_k or self.config.experience_top_k

        if not self._experiences:
            return []

        type_indices = self._type_index.get(query_features.query_type, [])
        if not type_indices:
            type_indices = list(range(len(self._experiences)))

        scored = []
        now = time.time()

        for idx in type_indices:
            exp = self._experiences[idx]
            days_old = (now - exp.timestamp) / 86400
            recency_score = math.exp(-0.1 * days_old)
            complexity_sim = 1.0 - abs(
                exp.query_features.estimated_complexity - query_features.estimated_complexity
            )
            score = 0.5 * recency_score + 0.3 * complexity_sim + 0.2 * exp.answer_quality
            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._experiences[idx] for idx, _ in scored[:top_k]]

    def get_strategy_success_rates(
        self,
        query_type: str,
        lookback: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Get success rates for each strategy given a query type."""
        type_indices = self._type_index.get(query_type, [])
        recent_indices = type_indices[-lookback:] if len(type_indices) > lookback else type_indices

        stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"successes": 0, "total": 0, "quality_sum": 0.0}
        )

        for idx in recent_indices:
            exp = self._experiences[idx]
            s = stats[exp.retrieval_strategy]
            s["total"] += 1
            s["quality_sum"] += exp.answer_quality
            if exp.success:
                s["successes"] += 1

        result = {}
        for strategy, s in stats.items():
            if s["total"] > 0:
                result[strategy] = {
                    "success_rate": s["successes"] / s["total"],
                    "avg_quality": s["quality_sum"] / s["total"],
                    "count": s["total"],
                }
        return result

    def _run_batch_reflection(self, batch: Optional[List[Dict[str, Any]]] = None) -> None:  # noqa: E501
        """Run self-reflection on a batch of experiences via LLM (batch may be passed in or popped under lock)."""
        if batch is None:
            with self._lock:
                if not self._pending_reflections:
                    return
                batch = self._pending_reflections[:self.config.reflection_batch_size]
                self._pending_reflections = self._pending_reflections[self.config.reflection_batch_size:]

        try:
            client = self._get_llm_client()

            for item in batch:
                idx = item["experience_idx"]
                if idx >= len(self._experiences):
                    continue

                prompt = (
                    f"Evaluate this Q&A interaction:\n"
                    f"Query: {item['query']}\n"
                    f"Strategy: {item['strategy']}\n"
                    f"Context (truncated): {item['context'][:300]}\n"
                    f"Answer: {item['answer'][:300]}\n\n"
                    f"Rate answer quality 0-1 and briefly state what worked/didn't "
                    f"and what strategy would be better. "
                    f"Format: QUALITY: <score>\\nLESSON: <text>\\nSUGGESTION: <text>"
                )

                response = client.chat.completions.create(
                    model=self.config.reflection_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=self.config.max_reflection_tokens,
                )

                reflection_text = response.choices[0].message.content.strip()
                quality, lesson, suggestion = self._parse_reflection(reflection_text)

                exp = self._experiences[idx]
                if quality is not None:
                    exp.answer_quality = quality
                    exp.success = quality >= 0.6
                exp.lessons_learned = lesson
                exp.strategy_suggestion = suggestion

        except Exception as e:
            logger.warning(f"Batch reflection failed: {e}")

    def _parse_reflection(self, text: str) -> Tuple[Optional[float], str, str]:
        """Parse structured reflection output from LLM."""
        quality = None
        lesson = ""
        suggestion = ""

        for line in text.split('\n'):
            line = line.strip()
            if line.upper().startswith("QUALITY:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    score_str = score_str.replace("%", "").replace("/1", "").strip()
                    quality = float(score_str)
                    if quality > 1.0:
                        quality = quality / 100.0
                    quality = max(0.0, min(1.0, quality))
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("LESSON:"):
                lesson = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.upper().startswith("SUGGESTION:"):
                suggestion = line.split(":", 1)[1].strip() if ":" in line else ""

        return quality, lesson, suggestion

    def _get_llm_client(self):
        if self._llm_client is None:
            from openai import OpenAI
            import httpx
            self._llm_client = OpenAI(http_client=httpx.Client())
        return self._llm_client

    def _load_experiences(self) -> None:
        if self._experiences_file.exists():
            with open(self._experiences_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            exp = MetaExperience.from_dict(data)
                            idx = len(self._experiences)
                            self._experiences.append(exp)
                            self._type_index[exp.query_features.query_type].append(idx)
                        except (json.JSONDecodeError, Exception) as e:
                            logger.warning(f"Failed to load experience: {e}")

    def flush_reflections(self) -> None:
        """Force process all pending reflections (thread-safe)."""
        while True:
            with self._lock:
                if not self._pending_reflections:
                    break
                batch = self._pending_reflections[:self.config.reflection_batch_size]
                self._pending_reflections = self._pending_reflections[self.config.reflection_batch_size:]
            self._run_batch_reflection(batch)

    def save(self) -> None:
        with open(self._experiences_file, 'w', encoding='utf-8') as f:
            for exp in self._experiences:
                f.write(json.dumps(exp.to_dict(), ensure_ascii=False) + '\n')
        logger.debug(f"Saved {len(self._experiences)} experiences")

    def get_stats(self) -> Dict[str, Any]:
        type_counts = {t: len(indices) for t, indices in self._type_index.items()}

        strategy_counts: Dict[str, int] = defaultdict(int)
        total_quality = 0.0
        success_count = 0

        for exp in self._experiences:
            strategy_counts[exp.retrieval_strategy] += 1
            total_quality += exp.answer_quality
            if exp.success:
                success_count += 1

        n = len(self._experiences) or 1

        return {
            "total_experiences": len(self._experiences),
            "by_query_type": type_counts,
            "by_strategy": dict(strategy_counts),
            "avg_quality": total_quality / n,
            "success_rate": success_count / n,
            "pending_reflections": len(self._pending_reflections),
        }
