from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from .embeddings import cosine_similarity
from .models import MemoryQuery, MemorySearchHit
from .policy import MemoryPolicy
from .store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retrieves memory hits from store using the current policy."""

    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryPolicy | None = None,
        retrieval_mode: str = "keyword",
        embedder=None,
    ):
        self.store = store
        self.policy = policy or MemoryPolicy()
        self.retrieval_mode = retrieval_mode
        self.embedder = embedder

    def retrieve(self, query: MemoryQuery) -> list[MemorySearchHit]:
        mode = self.retrieval_mode
        if mode == "auto":
            mode = self._auto_select_mode(query)
            logger.info("[Retriever] auto-selected mode=%s for query(%d chars)", mode, len(query.query_text))
        if mode == "embedding":
            hits = self._retrieve_embedding(query)
            logger.info("[Retriever] mode=embedding scope=%s hits=%d", query.scope_id, len(hits))
            return hits
        if mode == "hybrid":
            hits = self._retrieve_hybrid(query)
            logger.info("[Retriever] mode=hybrid scope=%s hits=%d", query.scope_id, len(hits))
            return hits
        # Try expanded query if direct keyword search yields too few results.
        limit = min(query.top_k, self.policy.max_injected_units)
        hits = self.store.search_keyword(
            scope_id=query.scope_id,
            query_text=query.query_text,
            limit=limit,
        )
        expanded_flag = False
        if len(hits) < max(2, limit // 2):
            expanded = _expand_query(query.query_text)
            if expanded != query.query_text:
                expanded_flag = True
                extra = self.store.search_keyword(
                    scope_id=query.scope_id,
                    query_text=expanded,
                    limit=limit,
                )
                seen = {h.unit.memory_id for h in hits}
                for h in extra:
                    if h.unit.memory_id not in seen:
                        # Slight score penalty for expansion-only matches.
                        hits.append(MemorySearchHit(
                            unit=h.unit,
                            score=h.score * 0.85,
                            matched_terms=h.matched_terms,
                        ))
                        seen.add(h.unit.memory_id)
                hits.sort(key=lambda h: (h.score, h.unit.updated_at), reverse=True)
                hits = hits[:limit]
        # Apply tag-based boosting if context tags are provided.
        if query.context_tags:
            hits = _apply_tag_boost(hits, query.context_tags)
        logger.info(
            "[Retriever] mode=keyword scope=%s hits=%d expanded=%s",
            query.scope_id, len(hits), expanded_flag,
        )
        return hits

    def _auto_select_mode(self, query: MemoryQuery) -> str:
        """Auto-select retrieval mode based on query characteristics.

        - Short queries (< 4 words): keyword is most reliable
        - Medium queries with embedder available: hybrid for best recall
        - Long queries without embedder: keyword
        """
        terms = _tokenize(query.query_text)
        if self.embedder is not None and len(terms) >= 4:
            return "hybrid"
        return "keyword"

    def _retrieve_hybrid(self, query: MemoryQuery) -> list[MemorySearchHit]:
        query_terms = _tokenize(query.query_text)
        if not query_terms:
            return []

        units = self.store.list_active(query.scope_id, limit=500)
        if not units:
            return []

        # Build IDF weights across the active corpus.
        doc_freq: dict[str, int] = {}
        unit_content_terms: list[set[str]] = []
        unit_metadata_terms: list[set[str]] = []
        for unit in units:
            content = set(_tokenize(" ".join([unit.summary, unit.content])))
            metadata = set(_tokenize(" ".join(unit.topics + unit.entities)))
            unit_content_terms.append(content)
            unit_metadata_terms.append(metadata)
            all_terms = content | metadata
            for term in set(query_terms):
                if term in all_terms:
                    doc_freq[term] = doc_freq.get(term, 0) + 1

        num_docs = float(len(units))
        query_embedding = self.embedder.encode(query.query_text) if self.embedder else []
        hits: list[MemorySearchHit] = []
        for idx, unit in enumerate(units):
            if query.include_types and unit.memory_type not in query.include_types:
                continue

            content_terms = unit_content_terms[idx]
            metadata_terms = unit_metadata_terms[idx]
            matched = sorted(
                term for term in query_terms
                if term in content_terms or term in metadata_terms
            )
            if not matched:
                continue

            # IDF-weighted keyword and metadata overlap.
            keyword_idf = sum(
                _log2(num_docs / float(doc_freq.get(term, 1)))
                for term in query_terms if term in content_terms
            )
            metadata_idf = sum(
                _log2(num_docs / float(doc_freq.get(term, 1)))
                for term in query_terms if term in metadata_terms
            )
            embedding_score = (
                cosine_similarity(query_embedding, unit.embedding)
                if query_embedding and unit.embedding
                else 0.0
            )
            recency_bonus = _estimate_recency_bonus(unit.updated_at, self.policy.recent_bonus_hours)
            type_boost = self.policy.type_boosts.get(unit.memory_type.value, 1.0)
            # Confidence factor: memories with higher confidence score slightly better.
            confidence_factor = 0.8 + 0.2 * unit.confidence
            score = (
                self.policy.keyword_weight * keyword_idf
                + self.policy.metadata_weight * metadata_idf
                + embedding_score
                + self.policy.importance_weight * unit.importance
                + self.policy.recency_weight * recency_bonus
                + unit.reinforcement_score
            ) * type_boost * confidence_factor
            reason_parts = [f"matched: {', '.join(matched[:5])}"]
            if recency_bonus > 0.3:
                reason_parts.append("recent")
            if unit.importance >= 0.8:
                reason_parts.append("high importance")
            if unit.reinforcement_score > 0.1:
                reason_parts.append("reinforced")
            hits.append(MemorySearchHit(
                unit=unit, score=score, matched_terms=matched,
                reason="; ".join(reason_parts),
            ))

        hits.sort(key=lambda hit: (hit.score, hit.unit.updated_at), reverse=True)
        hits = hits[: min(query.top_k, self.policy.max_injected_units)]
        if query.context_tags:
            hits = _apply_tag_boost(hits, query.context_tags)
        return hits

    def _retrieve_embedding(self, query: MemoryQuery) -> list[MemorySearchHit]:
        if self.embedder is None:
            return []
        query_embedding = self.embedder.encode(query.query_text)
        if not query_embedding:
            return []

        hits: list[MemorySearchHit] = []
        for unit in self.store.list_active(query.scope_id, limit=500):
            if query.include_types and unit.memory_type not in query.include_types:
                continue
            if not unit.embedding:
                continue
            similarity = cosine_similarity(query_embedding, unit.embedding)
            if similarity <= 0.0:
                continue
            type_boost = self.policy.type_boosts.get(unit.memory_type.value, 1.0)
            confidence_factor = 0.8 + 0.2 * unit.confidence
            score = (
                similarity
                + self.policy.importance_weight * unit.importance
                + unit.reinforcement_score
            ) * type_boost * confidence_factor
            hits.append(MemorySearchHit(unit=unit, score=score, matched_terms=[]))

        hits.sort(key=lambda hit: (hit.score, hit.unit.updated_at), reverse=True)
        hits = hits[: min(query.top_k, self.policy.max_injected_units)]
        if query.context_tags:
            hits = _apply_tag_boost(hits, query.context_tags)
        return hits


def _apply_tag_boost(hits: list[MemorySearchHit], context_tags: list[str]) -> list[MemorySearchHit]:
    """Boost scores for memories whose tags overlap with the query's context tags.

    Each matching tag adds a 15% boost, capped at 50% total. Re-sorts after boosting.
    """
    if not context_tags:
        return hits
    tag_set = set(t.lower() for t in context_tags)
    for hit in hits:
        unit_tags = set(t.lower() for t in hit.unit.tags)
        overlap = len(tag_set & unit_tags)
        if overlap:
            boost = min(0.15 * overlap, 0.5)
            hit.score *= 1.0 + boost
    hits.sort(key=lambda h: (h.score, h.unit.updated_at), reverse=True)
    return hits


def _log2(x: float) -> float:
    return math.log2(max(x, 1.0))


def _tokenize(text: str) -> list[str]:
    token: list[str] = []
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
            continue
        if token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out


_QUERY_EXPANSIONS: dict[str, list[str]] = {
    "db": ["database"],
    "database": ["db"],
    "auth": ["authentication", "authorization"],
    "authentication": ["auth"],
    "authorization": ["auth"],
    "config": ["configuration", "settings"],
    "configuration": ["config", "settings"],
    "settings": ["config", "configuration"],
    "api": ["endpoint", "rest"],
    "endpoint": ["api"],
    "test": ["testing", "tests"],
    "testing": ["test", "tests"],
    "deploy": ["deployment"],
    "deployment": ["deploy"],
    "js": ["javascript"],
    "javascript": ["js"],
    "ts": ["typescript"],
    "typescript": ["ts"],
    "py": ["python"],
    "python": ["py"],
    "repo": ["repository"],
    "repository": ["repo"],
    "ci": ["continuous integration", "pipeline"],
    "cd": ["continuous deployment"],
    "k8s": ["kubernetes"],
    "kubernetes": ["k8s"],
}


def _expand_query(query_text: str) -> str:
    """Expand query terms with common synonyms/abbreviations for better recall."""
    terms = _tokenize(query_text)
    expanded: list[str] = list(terms)
    for term in terms:
        synonyms = _QUERY_EXPANSIONS.get(term, [])
        for syn in synonyms:
            if syn not in expanded:
                expanded.append(syn)
    result = " ".join(expanded)
    return result


def _estimate_recency_bonus(updated_at: str, recent_bonus_hours: int) -> float:
    if not updated_at:
        return 0.0
    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_seconds = max((datetime.now(timezone.utc) - updated).total_seconds(), 0.0)
    age_hours = age_seconds / 3600.0
    if recent_bonus_hours <= 0:
        return 0.0
    return max(0.0, 1.0 - (age_hours / float(recent_bonus_hours)))
