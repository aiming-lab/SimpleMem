"""
Query Processor for Omni-Memory.

Handles query understanding, filtering, and routing.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import re
from datetime import datetime, timedelta

from simplemem.multimodal.core.mau import ModalityType
from simplemem.multimodal.core.config import OmniMemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Parsed and analyzed query."""

    original_query: str
    cleaned_query: str

    # Extracted filters
    time_range: Optional[Tuple[float, float]] = None
    modality_hints: List[ModalityType] = None
    tags: List[str] = None

    # Query characteristics
    is_factual: bool = False      # "What is X?"
    is_temporal: bool = False     # "When did X happen?"
    is_visual: bool = False       # "What did X look like?"
    is_evidence_seeking: bool = False  # "Show me proof of X"

    # Expansion hints
    likely_needs_expansion: bool = False
    suggested_depth: str = "summary"  # summary, details, evidence

    # LLM-based analysis fields
    intent_type: str = "unknown"  # factual, temporal, comparative, exploratory, verification
    entities: List[str] = field(default_factory=list)  # Extracted named entities
    reformulated_query: Optional[str] = None  # Optimized query for retrieval
    confidence: float = 1.0  # Confidence in parsing


class QueryProcessor:
    """
    Query Processor for intelligent query understanding.

    Analyzes queries to:
    1. Extract temporal filters
    2. Detect modality preferences
    3. Determine retrieval depth needed
    4. Clean and optimize query text
    """

    def __init__(self, config: Optional[OmniMemoryConfig] = None):
        self.config = config or OmniMemoryConfig()
        self._llm_client = None

        # Temporal keywords
        self._temporal_patterns = {
            r'\byesterday\b': lambda: self._get_day_range(-1),
            r'\btoday\b': lambda: self._get_day_range(0),
            r'\blast week\b': lambda: self._get_week_range(-1),
            r'\bthis week\b': lambda: self._get_week_range(0),
            r'\blast month\b': lambda: self._get_month_range(-1),
            r'\bthis month\b': lambda: self._get_month_range(0),
            r'\b(\d{4}-\d{2}-\d{2})\b': lambda m: self._parse_date(m.group(1)),
        }

        # Modality hints
        self._modality_keywords = {
            ModalityType.VISUAL: ['image', 'picture', 'photo', 'look like', 'see', 'show', 'visual'],
            ModalityType.AUDIO: ['audio', 'sound', 'hear', 'said', 'voice', 'speech', 'listen'],
            ModalityType.VIDEO: ['video', 'clip', 'recording', 'footage'],
            ModalityType.TEXT: ['text', 'written', 'document', 'note', 'message'],
        }

        # Evidence-seeking patterns
        self._evidence_patterns = [
            r'\bshow me\b',
            r'\bprove\b',
            r'\beverify\b',
            r'\bexactly\b',
            r'\bspecifically\b',
            r'\bdetails?\b',
        ]

    def _get_day_range(self, offset: int) -> Tuple[float, float]:
        """Get timestamp range for a day offset from today."""
        target = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        target += timedelta(days=offset)
        start = target.timestamp()
        end = (target + timedelta(days=1)).timestamp()
        return (start, end)

    def _get_week_range(self, offset: int) -> Tuple[float, float]:
        """Get timestamp range for a week offset."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_week = today - timedelta(days=today.weekday())
        start_of_week += timedelta(weeks=offset)
        start = start_of_week.timestamp()
        end = (start_of_week + timedelta(weeks=1)).timestamp()
        return (start, end)

    def _get_month_range(self, offset: int) -> Tuple[float, float]:
        """Get timestamp range for a month offset."""
        today = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Simple approximation
        target = today + timedelta(days=offset * 30)
        start = target.timestamp()
        end = (target + timedelta(days=30)).timestamp()
        return (start, end)

    def _parse_date(self, date_str: str) -> Tuple[float, float]:
        """Parse YYYY-MM-DD date string."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            start = dt.timestamp()
            end = (dt + timedelta(days=1)).timestamp()
            return (start, end)
        except ValueError:
            return None

    def process(self, query: str) -> ParsedQuery:
        """
        Process and analyze a query.

        Args:
            query: Raw query string

        Returns:
            ParsedQuery with extracted information
        """
        result = ParsedQuery(
            original_query=query,
            cleaned_query=query.strip(),
            modality_hints=[],
            tags=[],
        )

        query_lower = query.lower()

        # Extract time range
        for pattern, range_func in self._temporal_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                if callable(range_func):
                    try:
                        result.time_range = range_func(match) if match.groups() else range_func()
                        result.is_temporal = True
                    except:
                        pass
                break

        # Detect modality hints
        for modality, keywords in self._modality_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if modality not in result.modality_hints:
                        result.modality_hints.append(modality)
                    if modality == ModalityType.VISUAL:
                        result.is_visual = True

        # Detect evidence-seeking queries
        for pattern in self._evidence_patterns:
            if re.search(pattern, query_lower):
                result.is_evidence_seeking = True
                result.likely_needs_expansion = True
                result.suggested_depth = "evidence"
                break

        # Detect factual queries
        if re.match(r'^(what|who|where|when|how|why)\b', query_lower):
            result.is_factual = True

        # Determine suggested depth
        if result.is_evidence_seeking:
            result.suggested_depth = "evidence"
        elif result.is_visual or result.modality_hints:
            result.suggested_depth = "details"
            result.likely_needs_expansion = True
        else:
            result.suggested_depth = "summary"

        # Clean query (remove temporal markers for embedding)
        cleaned = query
        for pattern in self._temporal_patterns.keys():
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        result.cleaned_query = ' '.join(cleaned.split()).strip()

        return result

    def extract_tags(self, query: str) -> List[str]:
        """Extract potential tags from query."""
        # Simple extraction: words that look like tags
        words = query.lower().split()
        tags = []

        # Common tag patterns
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 2 and word not in ['the', 'and', 'for', 'with', 'that']:
                tags.append(word)

        return tags[:5]  # Limit to 5 tags

    def generate_search_variants(self, query: str) -> List[str]:
        """Generate query variants for better recall."""
        variants = [query]

        # Add without question words
        for prefix in ['what', 'who', 'where', 'when', 'how', 'why', 'show me', 'find']:
            if query.lower().startswith(prefix):
                variant = query[len(prefix):].strip()
                if variant:
                    variants.append(variant)

        # Remove common words
        cleaned = re.sub(r'\b(the|a|an|is|are|was|were|do|does|did)\b', '', query.lower())
        cleaned = ' '.join(cleaned.split())
        if cleaned and cleaned != query.lower():
            variants.append(cleaned)

        return list(set(variants))

    def determine_retrieval_strategy(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """
        Determine optimal retrieval strategy based on parsed query.

        Returns:
            Dictionary with retrieval parameters
        """
        strategy = {
            "top_k": 10,
            "use_hybrid": True,
            "modality_filter": None,
            "time_filter": parsed.time_range,
            "require_expansion": parsed.likely_needs_expansion,
            "depth": parsed.suggested_depth,
        }

        # Adjust top_k based on query type
        if parsed.is_evidence_seeking:
            strategy["top_k"] = 5  # Fewer but more detailed
        elif parsed.is_temporal:
            strategy["top_k"] = 20  # More results for time-based queries

        # Set modality filter if strong hint
        if len(parsed.modality_hints) == 1:
            strategy["modality_filter"] = parsed.modality_hints[0]

        return strategy

    # ==================== LLM-based Intent Analysis ====================

    def _get_llm_client(self):
        """Get or create LLM client for intent analysis."""
        if self._llm_client is None:
            try:
                from openai import OpenAI
                import httpx
                
                client_kwargs = {}
                if hasattr(self.config, 'llm') and self.config.llm.api_key:
                    client_kwargs["api_key"] = self.config.llm.api_key
                if hasattr(self.config, 'llm') and self.config.llm.api_base_url:
                    client_kwargs["base_url"] = self.config.llm.api_base_url
                
                http_client = httpx.Client()
                client_kwargs["http_client"] = http_client
                self._llm_client = OpenAI(**client_kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                return None
        return self._llm_client

    def analyze_intent_llm(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze query intent for better retrieval.

        Args:
            query: The user's query string

        Returns:
            Dict with:
            - intent_type: str (factual, temporal, comparative, exploratory, verification)
            - entities: List[str] (extracted entities)
            - time_references: List[str] (temporal expressions)
            - modality_hints: List[str] (visual, audio, text)
            - reformulated_query: str (optimized for retrieval)
            
            Returns None if LLM analysis fails.
        """
        try:
            client = self._get_llm_client()
            if client is None:
                return None

            prompt = '''Analyze this memory retrieval query and extract:
1. intent_type: one of [factual, temporal, comparative, exploratory, verification]
2. entities: list of named entities (people, places, things)
3. time_references: any temporal expressions (yesterday, last week, etc.)
4. modality_hints: likely content types [visual, audio, text, video]
5. reformulated_query: optimized search query

Query: {query}

Respond in JSON format only, no explanation.'''

            model = "gpt-4o-mini"
            if hasattr(self.config, 'llm') and hasattr(self.config.llm, 'query_model'):
                model = self.config.llm.query_model

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a query analysis assistant. Output valid JSON only."},
                    {"role": "user", "content": prompt.format(query=query)}
                ],
                temperature=0,
                max_tokens=500,
            )

            import json
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            return result

        except Exception as e:
            logger.warning(f"LLM intent analysis failed: {e}, falling back to regex")
            return None

    def process_with_llm(self, query: str) -> ParsedQuery:
        """
        Process query with LLM-enhanced intent analysis.

        This is a more accurate but slower alternative to the regex-based process().
        Falls back to regex analysis if LLM fails.

        Args:
            query: Raw query string

        Returns:
            ParsedQuery with LLM-enhanced extraction
        """
        result = self.process(query)
        llm_analysis = self.analyze_intent_llm(query)

        if llm_analysis:
            result.intent_type = llm_analysis.get("intent_type", result.intent_type)
            result.entities = llm_analysis.get("entities", [])
            result.reformulated_query = llm_analysis.get("reformulated_query", query)
            
            llm_modalities = llm_analysis.get("modality_hints", [])
            for mod_str in llm_modalities:
                mod_str_lower = mod_str.lower()
                if mod_str_lower == "visual" and ModalityType.VISUAL not in result.modality_hints:
                    result.modality_hints.append(ModalityType.VISUAL)
                elif mod_str_lower == "audio" and ModalityType.AUDIO not in result.modality_hints:
                    result.modality_hints.append(ModalityType.AUDIO)
                elif mod_str_lower == "video" and ModalityType.VIDEO not in result.modality_hints:
                    result.modality_hints.append(ModalityType.VIDEO)
                elif mod_str_lower == "text" and ModalityType.TEXT not in result.modality_hints:
                    result.modality_hints.append(ModalityType.TEXT)

            result.confidence = 0.9
        else:
            result.confidence = 0.7

        return result
