"""
Embedding utilities - Generate vector embeddings via OpenRouter
"""
from typing import List
import numpy as np
from .openrouter import OpenRouterClient
import config


class EmbeddingModel:
    """
    Embedding model using OpenRouter API
    """
    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        dimension: int = None,
    ):
        """
        Initialize embedding model with OpenRouter
        
        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY)
            model_name: Embedding model name (defaults to config.EMBEDDING_MODEL)
            dimension: Embedding dimension (defaults to config.EMBEDDING_DIMENSION)
        """
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.dimension = dimension or config.EMBEDDING_DIMENSION
        
        # Initialize OpenRouter client
        self.client = OpenRouterClient(
            api_key=self.api_key,
            base_url=config.OPENROUTER_BASE_URL,
            embedding_model=self.model_name,
            app_name="SimpleMem Skill",
        )
        
        print(f"Embedding model initialized: {self.model_name} (dim={self.dimension})")

    def encode(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: int = 100,
        max_retries: int = 3,
    ) -> np.ndarray:
        """
        Encode list of texts to vectors
        
        Args:
            texts: List of texts to embed
            is_query: Whether these are queries (vs documents) - not used for OpenRouter
            batch_size: Number of texts to process at once
            max_retries: Number of retry attempts on failure
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Retry mechanism for each batch
            last_exception = None
            for attempt in range(max_retries):
                try:
                    embeddings = self.client.create_embedding(batch)
                    all_embeddings.extend(embeddings)
                    break
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        import time
                        wait_time = (2 ** attempt)
                        print(f"Embedding API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Embedding API call failed after {max_retries} attempts")
                        raise last_exception
        
        return np.array(all_embeddings, dtype=np.float32)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode a single text to a vector
        
        Args:
            text: Text to embed
            is_query: Whether this is a query (not used for OpenRouter)
            
        Returns:
            numpy array of embedding (embedding_dim,)
        """
        embeddings = self.encode([text], is_query=is_query)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """
        Encode documents (same as encode with is_query=False)
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        return self.encode(texts, is_query=False)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """
        Encode queries (same as encode with is_query=True)
        
        Args:
            texts: List of query texts to embed
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        return self.encode(texts, is_query=True)

    def close(self):
        """Close the client connection"""
        self.client.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
