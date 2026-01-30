"""
Memory Builder - Stage 1: Semantic Structured Compression

Converts raw dialogues into atomic memory entries through:
- Coreference resolution (remove pronouns)
- Temporal anchoring (absolute timestamps)
- Information extraction (keywords, persons, entities, etc.)

Simplified: Direct processing without buffering.
"""

from typing import List, Optional
from datetime import datetime
import uuid

from ..auth.models import MemoryEntry, Dialogue
from ..database.vector_store import MultiTenantVectorStore

# Type alias for LLM client (supports both OpenRouter and Ollama)
LLMClient = object  # Duck-typed: can be OpenRouterClient or OllamaClient


class MemoryBuilder:
    """
    Builds atomic memory entries from dialogues.
    Direct processing - no buffering required.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: MultiTenantVectorStore,
        table_name: str,
        window_size: int = 40,  # Max dialogues per LLM call
        overlap_size: int = 2,
        temperature: float = 0.1,
    ):
        self.client = llm_client
        self.vector_store = vector_store
        self.table_name = table_name
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.temperature = temperature

        # Context from previous processing for deduplication
        self._previous_context: str = ""
        self._total_processed = 0

    async def add_dialogue(
        self,
        speaker: str,
        content: str,
        timestamp: Optional[str] = None,
        auto_process: bool = True,
    ) -> dict:
        """
        Add a single dialogue and immediately process it into memory.

        Args:
            speaker: Speaker name
            content: Dialogue content
            timestamp: Optional ISO 8601 timestamp
            auto_process: Ignored (always processes immediately)

        Returns:
            Status dict with processing results
        """
        dialogue = Dialogue(
            dialogue_id=self._total_processed + 1,
            speaker=speaker,
            content=content,
            timestamp=timestamp or datetime.utcnow().isoformat(),
        )

        # Process immediately
        entries = await self._generate_memory_entries([dialogue])

        if not entries:
            return {
                "added": True,
                "processed": True,
                "entries_created": 0,
                "message": "No extractable information found",
            }

        # Generate embeddings
        texts = [entry.lossless_restatement for entry in entries]
        embeddings = await self.client.create_embedding(texts)

        # Store entries
        count = await self.vector_store.add_entries(
            self.table_name,
            entries,
            embeddings,
        )

        # Update context for next call
        self._previous_context = self._build_context_summary(entries)
        self._total_processed += 1

        return {
            "added": True,
            "processed": True,
            "entries_created": count,
        }

    async def add_dialogues(
        self,
        dialogues: List[dict],
        auto_process: bool = True,
    ) -> dict:
        """
        Add multiple dialogues and process them immediately.

        Args:
            dialogues: List of dialogue dicts with speaker, content, timestamp
            auto_process: Ignored (always processes immediately)

        Returns:
            Status dict with processing info
        """
        if not dialogues:
            return {
                "added": 0,
                "entries_created": 0,
                "message": "No dialogues provided",
            }

        # Convert to Dialogue objects
        dialogue_objects = []
        for i, dlg in enumerate(dialogues):
            dialogue_objects.append(Dialogue(
                dialogue_id=self._total_processed + i + 1,
                speaker=dlg.get("speaker", ""),
                content=dlg.get("content", ""),
                timestamp=dlg.get("timestamp") or datetime.utcnow().isoformat(),
            ))

        total_entries = 0

        # Process in windows if too many dialogues
        for i in range(0, len(dialogue_objects), self.window_size):
            window = dialogue_objects[i:i + self.window_size]

            entries = await self._generate_memory_entries(window)

            if entries:
                # Generate embeddings
                texts = [entry.lossless_restatement for entry in entries]
                embeddings = await self.client.create_embedding(texts)

                # Store entries
                count = await self.vector_store.add_entries(
                    self.table_name,
                    entries,
                    embeddings,
                )
                total_entries += count

                # Update context for next window
                self._previous_context = self._build_context_summary(entries)

        self._total_processed += len(dialogues)

        return {
            "added": len(dialogues),
            "entries_created": total_entries,
        }

    async def _generate_memory_entries(
        self,
        dialogues: List[Dialogue],
    ) -> List[MemoryEntry]:
        """
        Generate atomic memory entries from dialogues using LLM

        Args:
            dialogues: List of Dialogue objects

        Returns:
            List of MemoryEntry objects
        """
        # Build dialogue text
        dialogue_text = self._format_dialogues(dialogues)

        # Build prompt
        prompt = self._build_extraction_prompt(dialogue_text)

        # Call LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional information extraction assistant. "
                    "Extract atomic, self-contained facts from dialogues. "
                    "Each fact must be independently understandable without context. "
                    "Always resolve pronouns to actual names and convert relative times to absolute timestamps."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )

                # Parse response
                data = self.client.extract_json(response)
                if data is None:
                    continue

                # Handle both list and dict with "entries" key
                entries_data = data if isinstance(data, list) else data.get("entries", [])

                entries = []
                for item in entries_data:
                    entry = MemoryEntry(
                        entry_id=str(uuid.uuid4()),
                        lossless_restatement=item.get("lossless_restatement", ""),
                        keywords=item.get("keywords", []),
                        timestamp=item.get("timestamp"),
                        location=item.get("location"),
                        persons=item.get("persons", []),
                        entities=item.get("entities", []),
                        topic=item.get("topic"),
                    )
                    if entry.lossless_restatement:
                        entries.append(entry)

                return entries

            except Exception as e:
                print(f"Memory extraction attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return []

        return []

    def _format_dialogues(self, dialogues: List[Dialogue]) -> str:
        """Format dialogues into readable text"""
        lines = []
        for dlg in dialogues:
            ts = f" [{dlg.timestamp}]" if dlg.timestamp else ""
            lines.append(f"{dlg.speaker}{ts}: {dlg.content}")
        return "\n".join(lines)

    def _build_extraction_prompt(self, dialogue_text: str) -> str:
        """Build the extraction prompt"""
        context_section = ""
        if self._previous_context:
            context_section = f"""
## Context from Previous Processing (for reference, avoid duplication):
{self._previous_context}

---
"""

        return f"""{context_section}## Dialogues to Process:
{dialogue_text}

---

## Extraction Requirements:

1. **Complete Coverage**: Capture ALL valuable information from the dialogues.

2. **Self-Contained Facts**: Each entry must be independently understandable.
   - BAD: "He will meet Bob tomorrow" (Who is "he"? When is "tomorrow"?)
   - GOOD: "Alice will meet Bob at Starbucks on 2025-01-15 at 14:00"

3. **Coreference Resolution**: Replace ALL pronouns with actual names.
   - Replace: he, she, it, they, him, her, them, his, hers, their
   - With: The actual person's name or entity

4. **Temporal Anchoring**: Convert ALL relative times to absolute ISO 8601 format.
   - "tomorrow" → Calculate actual date
   - "next week" → Calculate actual date range
   - "in 2 hours" → Calculate actual time

5. **Information Extraction**: For each entry, extract:
   - `lossless_restatement`: Complete, unambiguous fact
   - `keywords`: Core terms for search (3-7 keywords)
   - `timestamp`: ISO 8601 format if mentioned
   - `location`: Specific location name
   - `persons`: All person names involved
   - `entities`: Companies, products, organizations, etc.
   - `topic`: Brief topic phrase (2-5 words)

## Output Format (JSON only, no other text):
{{
  "entries": [
    {{
      "lossless_restatement": "Complete self-contained fact...",
      "keywords": ["keyword1", "keyword2", ...],
      "timestamp": "2025-01-15T14:00:00" or null,
      "location": "Starbucks, Downtown" or null,
      "persons": ["Alice", "Bob"],
      "entities": ["Company XYZ"] or [],
      "topic": "Meeting arrangement"
    }}
  ]
}}

Return ONLY valid JSON. No explanations or other text."""

    def _build_context_summary(self, entries: List[MemoryEntry]) -> str:
        """Build context summary from entries for next processing"""
        if not entries:
            return ""

        summaries = []
        for entry in entries[-5:]:  # Keep last 5 entries as context
            summaries.append(f"- {entry.lossless_restatement}")

        return "\n".join(summaries)

    def get_stats(self) -> dict:
        """Get processing statistics"""
        return {
            "total_dialogues_processed": self._total_processed,
        }
