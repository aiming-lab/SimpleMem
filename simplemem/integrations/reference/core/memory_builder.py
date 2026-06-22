"""
Memory Builder - Stage 1: Semantic Structured Compression (Section 3.1)

Implements Semantic Structured Compression:
- Entropy-based non-linear filter: Φ_gate (conceptual - filters low-density dialogue)
- De-linearization transformation: F_θ (converts dialogue to atomic entries)
- Generates self-contained Atomic Entries {m_k} via coreference resolution and temporal anchoring
"""
from typing import List, Optional
from models.memory_entry import MemoryEntry, Dialogue
from utils.llm_client import LLMClient
from database.vector_store import VectorStore
import config
import json
import asyncio
import concurrent.futures
from functools import partial


class MemoryBuilder:
    """
    Memory Builder - Stage 1: Semantic Structured Compression

    Paper Reference: Section 3.1 - Semantic Structured Compression

    Core Functions:
    1. Entropy-based filtering (implicit via window processing)
    2. De-linearization transformation F_θ: Dialogue → Atomic Entries
    3. Coreference resolution Φ_coref (no pronouns)
    4. Temporal anchoring Φ_time (absolute timestamps)
    5. Generate self-contained Atomic Entries {m_k}
    """
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        window_size: int = None,
        enable_parallel_processing: bool = True,
        max_parallel_workers: int = 3
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.window_size = window_size or config.WINDOW_SIZE
        
        # Use config values as default if not explicitly provided
        self.enable_parallel_processing = enable_parallel_processing if enable_parallel_processing is not None else getattr(config, 'ENABLE_PARALLEL_PROCESSING', True)
        self.max_parallel_workers = max_parallel_workers if max_parallel_workers is not None else getattr(config, 'MAX_PARALLEL_WORKERS', 4)

        # Dialogue buffer
        self.dialogue_buffer: List[Dialogue] = []
        self.processed_count = 0

        # Previous window entries (for context)
        self.previous_entries: List[MemoryEntry] = []

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True):
        """
        Add a dialogue to the buffer
        """
        self.dialogue_buffer.append(dialogue)

        # Auto process
        if auto_process and len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def add_dialogues(self, dialogues: List[Dialogue], auto_process: bool = True):
        """
        Batch add dialogues with optional parallel processing
        """
        if self.enable_parallel_processing and len(dialogues) > self.window_size * 2:
            # Use parallel processing for large batches
            self.add_dialogues_parallel(dialogues)
        else:
            # Use sequential processing for smaller batches
            for dialogue in dialogues:
                self.add_dialogue(dialogue, auto_process=False)

            # Process complete windows
            if auto_process:
                while len(self.dialogue_buffer) >= self.window_size:
                    self.process_window()
    
    def add_dialogues_parallel(self, dialogues: List[Dialogue]):
        """
        Add dialogues using parallel processing for better performance
        """
        try:
            # Add all dialogues to buffer first
            self.dialogue_buffer.extend(dialogues)
            
            # Group into windows for parallel processing (including remaining dialogues)
            windows_to_process = []
            while len(self.dialogue_buffer) >= self.window_size:
                window = self.dialogue_buffer[:self.window_size]
                self.dialogue_buffer = self.dialogue_buffer[self.window_size:]
                windows_to_process.append(window)
            
            # Add remaining dialogues as a smaller batch (no need to process separately)
            if self.dialogue_buffer:
                windows_to_process.append(self.dialogue_buffer)
                self.dialogue_buffer = []  # Clear buffer since we're processing all
            
            if windows_to_process:
                print(f"\n[Parallel Processing] Processing {len(windows_to_process)} batches in parallel with {self.max_parallel_workers} workers")
                print(f"Batch sizes: {[len(w) for w in windows_to_process]}")
                
                # Process all windows/batches in parallel (including remaining dialogues)
                self._process_windows_parallel(windows_to_process)
                
        except Exception as e:
            print(f"[Parallel Processing] Failed: {e}. Falling back to sequential processing...")
            # Fallback to sequential processing
            for window in windows_to_process:
                self.dialogue_buffer = window + self.dialogue_buffer
                self.process_window()

    def process_window(self):
        """
        Process current window dialogues - Core logic
        """
        if not self.dialogue_buffer:
            return

        # Extract window
        window = self.dialogue_buffer[:self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.window_size:]

        print(f"\nProcessing window: {len(window)} dialogues (processed {self.processed_count} so far)")

        # Call LLM to generate memory entries
        entries = self._generate_memory_entries(window)

        # Store to database
        if entries:
            self.vector_store.add_entries(entries)
            self.previous_entries = entries  # Save as context
            self.processed_count += len(window)

        print(f"Generated {len(entries)} memory entries")

    def process_remaining(self):
        """
        Process remaining dialogues (fallback method, normally handled in parallel)
        """
        if self.dialogue_buffer:
            print(f"\nProcessing remaining dialogues: {len(self.dialogue_buffer)} (fallback mode)")
            entries = self._generate_memory_entries(self.dialogue_buffer)
            if entries:
                self.vector_store.add_entries(entries)
                self.processed_count += len(self.dialogue_buffer)
            self.dialogue_buffer = []
            print(f"Generated {len(entries)} memory entries")

    def _generate_memory_entries(self, dialogues: List[Dialogue]) -> List[MemoryEntry]:
        """
        De-linearization Transformation F_θ: W_t → {m_k}

        Paper Reference: Section 3.1 - Eq. (3)
        Applies composite transformation: F_θ = Φ_time ∘ Φ_coref ∘ Φ_extract

        Key requirements:
        1. Generate multiple Atomic Entries to cover all information
        2. Φ_coref: Force coreference resolution (no pronouns)
        3. Φ_time: Temporal anchoring (convert relative to absolute time)
        4. Reference previous window entries to avoid duplication
        """
        # Build dialogue text
        dialogue_text = "\n".join([str(d) for d in dialogues])
        dialogue_ids = [d.dialogue_id for d in dialogues]

        # Build context
        context = ""
        if self.previous_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for entry in self.previous_entries[:3]:  # Only show first 3
                context += f"- {entry.lossless_restatement}\n"

        # Build prompt
        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, context)

        # Call LLM
        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times if parsing fails
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )

                # Parse response
                entries = self._parse_llm_response(response, dialogue_ids)
                return entries

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed to parse LLM response: {e}")
                    print(f"Retrying...")
                else:
                    print(f"All {max_retries} attempts failed to parse LLM response: {e}")
                    print(f"Raw response: {response[:500] if 'response' in locals() else 'No response'}")
                    return []

    def _build_extraction_prompt(
        self,
        dialogue_text: str,
        dialogue_ids: List[int],
        context: str
    ) -> str:
        """
        Build LLM extraction prompt
        """
        return f"""
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

{context}

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information in the dialogues is captured
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow)
3. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence
4. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time in ISO 8601 format (if explicit time mentioned in dialogue)
   - location: Specific location name (if mentioned)
   - persons: All person names mentioned
   - entities: Companies, products, organizations, etc.
   - topic: The topic of this information

[Output Format]
Return a JSON array, each element is a memory entry:

```json
[
  {{
    "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)",
    "keywords": ["keyword1", "keyword2", ...],
    "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
    "location": "location name or null",
    "persons": ["name1", "name2", ...],
    "entities": ["entity1", "entity2", ...],
    "topic": "topic phrase"
  }},
  ...
]
```

[Example]
Dialogues:
[2025-11-15T14:30:00] Alice: Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product
[2025-11-15T14:31:00] Bob: Okay, I'll prepare the materials

Output:
```json
[
  {{
    "lossless_restatement": "Alice suggested at 2025-11-15T14:30:00 to meet with Bob at Starbucks on 2025-11-16T14:00:00 to discuss the new product.",
    "keywords": ["Alice", "Bob", "Starbucks", "new product", "meeting"],
    "timestamp": "2025-11-16T14:00:00",
    "location": "Starbucks",
    "persons": ["Alice", "Bob"],
    "entities": ["new product"],
    "topic": "Product discussion meeting arrangement"
  }},
  {{
    "lossless_restatement": "Bob agreed to attend the meeting and committed to prepare relevant materials.",
    "keywords": ["Bob", "prepare materials", "agree"],
    "timestamp": null,
    "location": null,
    "persons": ["Bob"],
    "entities": [],
    "topic": "Meeting preparation confirmation"
  }}
]
```

Now process the above dialogues. Return ONLY the JSON array, no other explanations.
"""

    def _parse_llm_response(
        self,
        response: str,
        dialogue_ids: List[int]
    ) -> List[MemoryEntry]:
        """
        Parse LLM response to MemoryEntry list
        """
        # Extract JSON
        data = self.llm_client.extract_json(response)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array but got: {type(data)}")

        entries = []
        for item in data:
            # Create MemoryEntry
            entry = MemoryEntry(
                lossless_restatement=item["lossless_restatement"],
                keywords=item.get("keywords", []),
                timestamp=item.get("timestamp"),
                location=item.get("location"),
                persons=item.get("persons", []),
                entities=item.get("entities", []),
                topic=item.get("topic")
            )
            entries.append(entry)

        return entries
    
    def _process_windows_parallel(self, windows: List[List[Dialogue]]):
        """
        Process multiple windows in parallel using ThreadPoolExecutor
        """
        all_entries = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all window processing tasks
            future_to_window = {}
            for i, window in enumerate(windows):
                dialogue_ids = [d.dialogue_id for d in window]
                future = executor.submit(self._generate_memory_entries_worker, window, dialogue_ids, i+1)
                future_to_window[future] = (window, i+1)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_window):
                window, window_num = future_to_window[future]
                try:
                    entries = future.result()
                    all_entries.extend(entries)
                    print(f"[Parallel Processing] Window {window_num} completed: {len(entries)} entries")
                except Exception as e:
                    print(f"[Parallel Processing] Window {window_num} failed: {e}")
        
        # Store all entries to database in batch
        if all_entries:
            print(f"\n[Parallel Processing] Storing {len(all_entries)} entries to database...")
            self.vector_store.add_entries(all_entries)
            self.processed_count += sum(len(window) for window in windows)
            
            # Update previous entries (use last window's entries for context)
            if all_entries:
                self.previous_entries = all_entries[-10:]  # Keep last 10 entries for context
        
        print(f"[Parallel Processing] Completed processing {len(windows)} windows")
    
    def _generate_memory_entries_worker(self, window: List[Dialogue], dialogue_ids: List[int], window_num: int) -> List[MemoryEntry]:
        """
        Worker function for parallel processing of a single batch (full window or remaining dialogues)
        """
        batch_size = len(window)
        batch_type = "full window" if batch_size == self.window_size else f"remaining batch"
        print(f"[Worker {window_num}] Processing {batch_type} with {batch_size} dialogues")
        
        # Build dialogue text
        dialogue_text = "\n".join([str(d) for d in window])
        
        # Build context (shared across all workers - this is fine for parallel processing)
        context = ""
        if self.previous_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for entry in self.previous_entries[:3]:  # Only show first 3
                context += f"- {entry.lossless_restatement}\n"

        # Build prompt
        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, context)

        # Call LLM
        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times if parsing fails
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )

                # Parse response
                entries = self._parse_llm_response(response, dialogue_ids)
                print(f"[Worker {window_num}] Generated {len(entries)} entries")
                return entries

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Worker {window_num}] Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"[Worker {window_num}] All {max_retries} attempts failed: {e}")
                    return []
