"""
LLM Client - Handles all LLM interactions via OpenRouter
"""
import json
from typing import List, Dict, Any, Optional
from .openrouter import OpenRouterClient
import config


class LLMClient:
    """
    LLM client using OpenRouter API
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize LLM client with OpenRouter
        
        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY)
            model: Model name (defaults to config.LLM_MODEL)
            temperature: Sampling temperature (defaults to config.TEMPERATURE)
            max_tokens: Max tokens in response (defaults to config.MAX_TOKENS)
        """
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.model = model or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.max_tokens = max_tokens or config.MAX_TOKENS

        # Initialize OpenRouter client
        self.client = OpenRouterClient(
            api_key=self.api_key,
            base_url=config.OPENROUTER_BASE_URL,
            llm_model=self.model,
            app_name="SimpleMem Skill",
        )
        
        print(f"LLM Client initialized with model: {self.model}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> str:
        """
        Standard chat completion with retry mechanism
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            response_format: Optional response format (e.g., {"type": "json_object"})
            max_retries: Number of retry attempts on failure
            
        Returns:
            Generated text content
        """
        temp = temperature if temperature is not None else self.temperature

        # Retry mechanism
        last_exception = None
        for attempt in range(max_retries):
            try:
                return self.client.chat_completion(
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                    response_format=response_format,
                )
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    import time
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    print(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"LLM API call failed after {max_retries} attempts")
                    raise last_exception

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response text
        
        Args:
            text: Raw text that may contain JSON
            
        Returns:
            Parsed JSON object
        """
        return self.client.extract_json(text)

    def close(self):
        """Close the client connection"""
        self.client.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
