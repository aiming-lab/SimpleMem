"""
Memory Extractor --- Automated LLM-based memory extraction with quality control.

Implements the extraction pipeline proven in V8 evaluation:
- Sliding window segmentation (configurable window size and overlap)
- Structured metadata extraction (timestamp, location, persons, entities, topic)
- Retry with chunk splitting for failed extractions
- Coverage verification and gap detection
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import MemoryType, MemoryUnit

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a professional information extraction assistant.
Extract ALL valuable information from the following dialogue into structured memory entries.

{context}

[Dialogue from {date}]
{dialogue_text}

[Requirements]
1. Complete Coverage: Generate entries for ALL facts, events, opinions, plans, feelings.
2. Force Disambiguation: PROHIBIT pronouns (he/she/it/they). Use actual names and absolute dates.
3. Lossless Restatement: Each entry must be complete, independent, self-contained.
4. Extract EVERY specific detail -- no paraphrasing fine-grained facts:
   - Named entities: book/movie/song/game titles (keep quotation marks), brand names,
     places, pet names, nicknames, colors, specific activities, specific numbers.
   - Quantities: exact counts, frequencies ("twice", "three times"), durations
     ("for 3 years", "since 2019").
   - Lists: if someone mentions multiple items (books, hobbies, gifts, instruments,
     activities), create ONE entry that lists them ALL (e.g. "Joanna's writings include
     screenplays, books, online blog posts, and journals"), not a single item.
   - Gifts and possessions: for each gift/possession mentioned, create an explicit entry
     (who gave what to whom, when, e.g. "Nate gave Joanna a stuffed toy pup named Tilly
     on 25 May 2022").
   - Facts stated implicitly through dialogue: capture them as direct statements
     (e.g. "Nate plays video games on a Nintendo" if a Switch-exclusive game is mentioned).
5. Cover names, places, objects, opinions, plans, feelings, events, dates, gifts,
   hobbies, relationships, pets, travel, food, books, art, music, work, family, health.

[Output Format]
Return a JSON array:
[
  {{
    "lossless_restatement": "Complete sentence with all subjects, objects, time, location",
    "keywords": ["keyword1", "keyword2"],
    "timestamp": "YYYY-MM-DD or null",
    "location": "location or null",
    "persons": ["name1", "name2"],
    "entities": ["entity1"],
    "topic": "topic phrase"
  }}
]

Return ONLY the JSON array. Extract at least 15 entries (more if the window contains
multiple distinct facts). Prioritise completeness over brevity."""


@dataclass
class ExtractionConfig:
    """Configuration for memory extraction."""
    window_size: int = 40
    overlap: int = 2
    max_retries: int = 5
    chunk_size_on_failure: int = 15
    min_restatement_words: int = 5
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1


@dataclass
class ExtractionResult:
    """Result of memory extraction for one session."""
    session_id: str
    memories: list[dict] = field(default_factory=list)
    turn_count: int = 0
    success: bool = True
    error: str = ""


def _robust_parse_json(raw: str) -> list | None:
    """Parse JSON from LLM response with multiple fallback strategies."""
    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Regex: extract outermost [...]
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            # Try repairing truncated array
            text = m.group().rstrip()
            last_brace = text.rfind("}")
            if last_brace > 0:
                try:
                    return json.loads(text[:last_brace + 1].rstrip().rstrip(",") + "\n]")
                except json.JSONDecodeError:
                    pass

    return None


def _ensure_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(v).strip() for v in val if v]
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    return []


class MemoryExtractor:
    """Extracts structured memories from conversation turns using LLM.

    Args:
        llm_call: Callable that takes (messages: list[dict], max_tokens: int, temperature: float)
                  and returns the LLM response string. This decouples extraction from any
                  specific LLM provider.
        config: ExtractionConfig with window size, retry, and quality parameters.
    """

    def __init__(
        self,
        llm_call: Callable[[list[dict], int, float], str],
        config: ExtractionConfig | None = None,
    ):
        self.llm_call = llm_call
        self.config = config or ExtractionConfig()

    def extract_session(
        self,
        turns: list[dict],
        session_id: str,
        date_str: str = "",
        prev_context: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract memories from a single session's conversation turns.

        Args:
            turns: List of turn dicts with 'speaker' and 'text' fields.
            session_id: Identifier for this session.
            date_str: Human-readable date/time string for the session.
            prev_context: Previous session's last few memory restatements (for dedup context).

        Returns:
            ExtractionResult with extracted memories.
        """
        if not turns:
            return ExtractionResult(session_id=session_id, turn_count=0)

        windows = self._segment_windows(turns)
        all_memories = []
        context_entries = list(prev_context or [])

        for wi, window in enumerate(windows):
            memories = self._extract_window(
                window, session_id, date_str, context_entries
            )
            if memories:
                all_memories.extend(memories)
                context_entries = [m["content"] for m in memories[-5:]]
            else:
                # Fallback: try smaller chunks
                logger.info(
                    "Window %d failed, trying chunk-based extraction", wi
                )
                chunk_memories = self._extract_chunked(
                    window, session_id, date_str
                )
                all_memories.extend(chunk_memories)
                if chunk_memories:
                    context_entries = [m["content"] for m in chunk_memories[-5:]]

        return ExtractionResult(
            session_id=session_id,
            memories=all_memories,
            turn_count=len(turns),
            success=len(all_memories) > 0,
        )

    def extract_sessions(
        self,
        sessions: list[tuple[str, str, list[dict]]],
        cache_dir: str | None = None,
    ) -> tuple[list[dict], list[ExtractionResult]]:
        """Extract memories from multiple sessions sequentially.

        Supports incremental caching: each session's result is appended to
        ``<cache_dir>/extracted_memories_incremental.jsonl`` immediately after
        extraction.  If a prior partial run left that file, already-extracted
        sessions are skipped automatically so work is never repeated.

        Args:
            sessions: List of (session_id, date_str, turns) tuples.
            cache_dir: Directory for incremental cache files.  When *None*,
                       incremental caching is disabled (legacy behaviour).

        Returns:
            (all_memories, per_session_results)
        """
        import os

        incr_path = (
            os.path.join(cache_dir, "extracted_memories_incremental.jsonl")
            if cache_dir
            else None
        )

        done_sessions: dict[str, list[dict]] = {}
        if incr_path and os.path.exists(incr_path):
            with open(incr_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    done_sessions[rec["session_id"]] = rec["memories"]
            logger.info(
                "Incremental cache: %d sessions already extracted, resuming",
                len(done_sessions),
            )

        all_memories = []
        results = []
        prev_context = None

        for idx, (session_id, date_str, turns) in enumerate(sessions):
            if session_id in done_sessions:
                cached_mems = done_sessions[session_id]
                all_memories.extend(cached_mems)
                if cached_mems:
                    prev_context = [m["content"] for m in cached_mems[-5:]]
                logger.info(
                    "Session %s: %d memories (from cache) [%d/%d]",
                    session_id, len(cached_mems), idx + 1, len(sessions),
                )
                continue

            result = self.extract_session(
                turns, session_id, date_str, prev_context
            )
            results.append(result)
            all_memories.extend(result.memories)

            if result.memories:
                prev_context = [m["content"] for m in result.memories[-5:]]

            if incr_path:
                with open(incr_path, "a") as f:
                    f.write(
                        json.dumps(
                            {"session_id": session_id, "memories": result.memories},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            logger.info(
                "Session %s: %d memories from %d turns [%d/%d]",
                session_id, len(result.memories), result.turn_count,
                idx + 1, len(sessions),
            )

        return all_memories, results

    def verify_coverage(
        self,
        memories: list[dict],
        keywords: list[str],
    ) -> dict[str, bool]:
        """Check if extracted memories cover expected keywords.

        Returns dict mapping keyword -> found_in_memories.
        """
        contents = " ".join(m.get("content", "").lower() for m in memories)
        return {kw: kw.lower() in contents for kw in keywords}

    def memories_to_units(
        self,
        memories: list[dict],
        scope_id: str,
    ) -> list[MemoryUnit]:
        """Convert raw extracted memory dicts to MemoryUnit objects."""
        units = []
        for mem in memories:
            content = mem.get("content", "")
            if not content or len(content.split()) < self.config.min_restatement_words:
                continue

            mtype = MemoryType.EPISODIC  # Default
            topic = mem.get("topic", "")
            if topic and any(w in topic.lower() for w in ["preference", "like", "favorite"]):
                mtype = MemoryType.PREFERENCE
            elif topic and any(w in topic.lower() for w in ["fact", "knowledge", "identity"]):
                mtype = MemoryType.SEMANTIC

            unit = MemoryUnit(
                memory_id=str(uuid.uuid4()),
                scope_id=scope_id,
                memory_type=mtype,
                content=content,
                summary=content[:100],
                entities=mem.get("persons", []) + mem.get("entities", []),
                topics=mem.get("keywords", []),
                importance=0.5,
            )
            units.append(unit)
        return units

    # ---- Internal methods ----

    def _segment_windows(self, turns: list[dict]) -> list[list[dict]]:
        """Sliding window segmentation with overlap."""
        ws = self.config.window_size
        step = max(1, ws - self.config.overlap)
        windows = []
        i = 0
        while i < len(turns):
            window = turns[i:i + ws]
            if window:
                windows.append(window)
            i += step
        return windows

    def _extract_window(
        self,
        turns: list[dict],
        session_id: str,
        date_str: str,
        context_entries: list[str],
    ) -> list[dict]:
        """Extract memories from a single window with retries."""
        lines = [
            f"[{date_str}] {t.get('speaker', '?')}: {t.get('text', '')}"
            for t in turns
        ]
        dialogue_text = "\n".join(lines)
        if len(dialogue_text.strip()) < 30:
            return []

        context = ""
        if context_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for e in context_entries[:3]:
                context += f"- {e}\n"

        prompt = EXTRACTION_PROMPT.format(
            context=context, date=date_str, dialogue_text=dialogue_text
        )
        msgs = [
            {"role": "system", "content": "You are a professional information extraction assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(self.config.max_retries):
            try:
                response = self.llm_call(
                    msgs, self.config.llm_max_tokens, self.config.llm_temperature
                )
                if not response or len(response.strip()) < 10:
                    logger.debug("Attempt %d: empty response", attempt + 1)
                    time.sleep(2)
                    continue

                data = _robust_parse_json(response)
                if data is None:
                    logger.debug("Attempt %d: parse failed", attempt + 1)
                    time.sleep(2)
                    continue

                if isinstance(data, dict):
                    for key in ("entries", "memory_entries", "results", "data"):
                        if key in data and isinstance(data[key], list):
                            data = data[key]
                            break
                    else:
                        if "lossless_restatement" in data:
                            data = [data]
                        else:
                            continue

                if not isinstance(data, list) or len(data) == 0:
                    time.sleep(2)
                    continue

                memories = self._parse_items(data, session_id)
                if memories:
                    return memories

            except Exception as e:
                logger.warning("Extraction attempt %d error: %s", attempt + 1, e)
                time.sleep(3 * (attempt + 1))

        return []

    def _extract_chunked(
        self,
        turns: list[dict],
        session_id: str,
        date_str: str,
    ) -> list[dict]:
        """Fallback: extract in smaller chunks when full window fails."""
        chunk_size = self.config.chunk_size_on_failure
        all_memories = []

        for ci in range(0, len(turns), chunk_size):
            chunk = turns[ci:ci + chunk_size]
            memories = self._extract_window(chunk, session_id, date_str, [])
            if memories:
                all_memories.extend(memories)
                logger.info(
                    "Chunk %d/%d: %d memories",
                    ci // chunk_size, len(turns) // chunk_size, len(memories),
                )

        return all_memories

    def _parse_items(self, data: list, session_id: str) -> list[dict]:
        """Parse and validate extracted items."""
        memories = []
        for item in data:
            if not isinstance(item, dict):
                continue
            restatement = item.get("lossless_restatement", "")
            if not restatement or len(restatement.split()) < self.config.min_restatement_words:
                continue
            memories.append({
                "content": restatement,
                "keywords": _ensure_list(item.get("keywords", [])),
                "timestamp": item.get("timestamp"),
                "location": item.get("location"),
                "persons": _ensure_list(item.get("persons", [])),
                "entities": _ensure_list(item.get("entities", [])),
                "topic": item.get("topic"),
                "session_id": session_id,
            })
        return memories
