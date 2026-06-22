"""
Answer Generator - Final synthesis from retrieved atomic contexts

Paper Reference: Section 3.3 - Reconstructive Synthesis (Read Path)
Generates answers from the final context C_final synthesized by query-aware retrieval
"""
from typing import List
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
import config


class AnswerGenerator:
    """
    Answer Generator - Reconstructive Synthesis from Atomic Contexts

    Paper Reference: Section 3.3 - Eq. (10)
    Synthesizes final answer from pruned, query-specific context:
    C_final = ⊕_{m ∈ Top-k_dyn(S)} [t_m: Content(m)]

    Features:
    1. Receive query and retrieved atomic entries
    2. Generate answers from disambiguated, self-contained facts
    3. Ensure accuracy through atomic context independence
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_answer(self, query: str, contexts: List[MemoryEntry]) -> str:
        """
        Generate answer

        Args:
        - query: User question
        - contexts: List of retrieved relevant MemoryEntry

        Returns:
        - Generated answer (concise phrase)
        """
        if not contexts:
            return "No relevant information found"

        # Build context string
        context_str = self._format_contexts(contexts)

        # Build prompt
        prompt = self._build_answer_prompt(query, context_str)

        # Call LLM to generate answer
        messages = [
            {
                "role": "system",
                "content": "You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times
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

                # Parse JSON response
                result = self.llm_client.extract_json(response)
                # Return the answer from JSON
                return result.get("answer", response.strip())

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Answer generation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Warning: Failed to parse JSON response after {max_retries} attempts: {e}")
                    # Fallback to raw response
                    if 'response' in locals():
                        return response.strip()
                    else:
                        return "Failed to generate answer"

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        """
        Format contexts to readable text
        """
        formatted = []
        for i, entry in enumerate(contexts, 1):
            parts = [f"[Context {i}]"]
            parts.append(f"Content: {entry.lossless_restatement}")

            if entry.timestamp:
                parts.append(f"Time: {entry.timestamp}")

            if entry.location:
                parts.append(f"Location: {entry.location}")

            if entry.persons:
                parts.append(f"Persons: {', '.join(entry.persons)}")

            if entry.entities:
                parts.append(f"Related Entities: {', '.join(entry.entities)}")

            if entry.topic:
                parts.append(f"Topic: {entry.topic}")

            formatted.append("\n".join(parts))

        return "\n\n".join(formatted)

    def _build_answer_prompt(self, query: str, context_str: str) -> str:
        """
        Build answer generation prompt
        """
        return f"""
Answer the user's question based on the provided context.

User Question: {query}

Relevant Context:
{context_str}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format

Output Format:
```json
{{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}}
```

Example:
Question: "When will they meet?"
Context: "Alice suggested meeting Bob at 2025-11-16T14:00:00..."

Output:
```json
{{
  "reasoning": "The context explicitly states the meeting time as 2025-11-16T14:00:00",
  "answer": "16 November 2025 at 2:00 PM"
}}
```

Now answer the question. Return ONLY the JSON, no other text.
"""
