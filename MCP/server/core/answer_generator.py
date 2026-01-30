"""
Answer Generator - Final Synthesis Module

Synthesizes answers from retrieved memory contexts using LLM.
"""

from typing import List, Optional

from ..auth.models import MemoryEntry

# Type alias for LLM client (supports both OpenRouter and Ollama)
LLMClient = object  # Duck-typed: can be OpenRouterClient or OllamaClient


class AnswerGenerator:
    """
    Generates answers from retrieved memory contexts.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.1,
    ):
        self.client = llm_client
        self.temperature = temperature

    async def generate_answer(
        self,
        query: str,
        contexts: List[MemoryEntry],
    ) -> dict:
        """
        Generate an answer from retrieved contexts

        Args:
            query: User's question
            contexts: Retrieved MemoryEntry objects

        Returns:
            Dict with answer and reasoning
        """
        if not contexts:
            return {
                "answer": "I don't have any relevant memories to answer this question.",
                "reasoning": "No relevant context was found in the memory store.",
                "confidence": "low",
            }

        # Format contexts
        context_str = self._format_contexts(contexts)

        # Build prompt
        prompt = self._build_answer_prompt(query, context_str)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on provided context. "
                    "Always base your answers on the given context. "
                    "If the context doesn't contain enough information, say so."
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

                data = self.client.extract_json(response)
                if data:
                    return {
                        "answer": data.get("answer", "Unable to generate answer."),
                        "reasoning": data.get("reasoning", ""),
                        "confidence": data.get("confidence", "medium"),
                    }

            except Exception as e:
                print(f"Answer generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        "answer": "An error occurred while generating the answer.",
                        "reasoning": f"Error: {str(e)}",
                        "confidence": "low",
                    }

        return {
            "answer": "Unable to generate answer after multiple attempts.",
            "reasoning": "JSON parsing failed.",
            "confidence": "low",
        }

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        """Format memory entries into readable context"""
        formatted = []

        for i, entry in enumerate(contexts[:30], 1):  # Limit to 30 entries
            parts = [f"[{i}] {entry.lossless_restatement}"]

            metadata = []
            if entry.timestamp:
                metadata.append(f"Time: {entry.timestamp}")
            if entry.location:
                metadata.append(f"Location: {entry.location}")
            if entry.persons:
                metadata.append(f"Persons: {', '.join(entry.persons)}")
            if entry.entities:
                metadata.append(f"Entities: {', '.join(entry.entities)}")
            if entry.topic:
                metadata.append(f"Topic: {entry.topic}")

            if metadata:
                parts.append(f"   ({'; '.join(metadata)})")

            formatted.append("\n".join(parts))

        return "\n\n".join(formatted)

    def _build_answer_prompt(self, query: str, context_str: str) -> str:
        """Build the answer generation prompt"""
        return f"""Answer the user's question based on the provided context.

## User Question:
{query}

## Relevant Context:
{context_str}

## Requirements:
1. Think through the reasoning process step by step
2. Base your answer ONLY on the provided context
3. Provide a CONCISE answer (short phrase or 1-2 sentences)
4. Format dates as 'DD Month YYYY' (e.g., "15 January 2025")
5. If context is insufficient, clearly state that

## Confidence Levels:
- "high": Context directly answers the question
- "medium": Context provides partial or indirect information
- "low": Context is insufficient or answer requires inference

## Output Format (JSON only):
{{
  "reasoning": "Brief explanation of how you derived the answer",
  "answer": "Concise answer to the question",
  "confidence": "high/medium/low"
}}

Return ONLY valid JSON. No other text."""

    async def generate_summary(
        self,
        entries: List[MemoryEntry],
        topic: Optional[str] = None,
    ) -> str:
        """
        Generate a summary of memory entries

        Args:
            entries: MemoryEntry objects to summarize
            topic: Optional topic focus

        Returns:
            Summary text
        """
        if not entries:
            return "No memories to summarize."

        # Format entries
        entries_text = "\n".join([
            f"- {entry.lossless_restatement}"
            for entry in entries[:50]
        ])

        topic_str = f" about {topic}" if topic else ""

        prompt = f"""Summarize the following memories{topic_str}:

{entries_text}

Provide a concise summary (2-4 sentences) that captures the key information.

Return ONLY the summary text, no JSON or formatting."""

        messages = [
            {"role": "system", "content": "You are a helpful summarization assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
            )
            return response.strip()
        except Exception as e:
            return f"Error generating summary: {e}"
