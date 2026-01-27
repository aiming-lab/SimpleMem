"""
Embedding utilities - Generate vector embeddings using SentenceTransformers
Supports Qwen3 Embedding models through SentenceTransformers interface
"""
from typing import List, Optional, Dict, Any
import numpy as np
import config
import os


class EmbeddingModel:
    """
    Embedding model using SentenceTransformers (supports Qwen3 and other models)
    """
    def __init__(
        self,
        model_name: str = None,
        use_optimization: bool = True,
        use_remote: Optional[bool] = None,
    ):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_optimization = use_optimization
        self.use_remote = (
            use_remote
            if use_remote is not None
            else getattr(config, "USE_REMOTE_EMBEDDING", False)
        )

        # Known/expected embedding dimension (can be updated after first remote call)
        self.dimension = getattr(config, "EMBEDDING_DIMENSION", None)

        self.model_type = None
        self.supports_query_prompt = False

        print(f"Loading embedding model: {self.model_name} (remote={self.use_remote})")

        if self.use_remote:
            self._init_remote_embedding()
        else:
            # Check if it's a Qwen3 model (through SentenceTransformers)
            if self.model_name.startswith("qwen3"):
                self._init_qwen3_sentence_transformer()
            else:
                self._init_standard_sentence_transformer()

        # Ensure dimension is set for downstream schema building
        if not self.dimension:
            raise ValueError("Embedding dimension is not set; check configuration or model.")

    def _init_remote_embedding(self):
        """Initialize remote embedding client (OpenAI compatible API)."""
        from openai import OpenAI

        api_key = getattr(config, "EMBEDDING_API_KEY", None) or getattr(
            config, "OPENAI_API_KEY", None
        )
        base_url = getattr(config, "EMBEDDING_BASE_URL", None) or getattr(
            config, "OPENAI_BASE_URL", None
        )

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            print(f"Using remote embedding base URL: {base_url}")

        if not api_key:
            raise ValueError("EMBEDDING_API_KEY/OPENAI_API_KEY is not configured for remote embeddings.")

        self.client = OpenAI(**client_kwargs)
        self.model_type = "remote"

    def _init_qwen3_sentence_transformer(self):
        """Initialize Qwen3 model using SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Map model names to actual model paths
            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B", 
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
            }
            
            model_path = qwen3_models.get(self.model_name, self.model_name)
            print(f"Loading Qwen3 model via SentenceTransformers: {model_path}")
            
            # Initialize with optimization settings
            if self.use_optimization:
                try:
                    # Try to use flash_attention_2 and left padding for better performance
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2", 
                            "device_map": "auto"
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True
                    )
                    print("Qwen3 loaded with flash_attention_2 optimization")
                except Exception as e:
                    print(f"Flash attention failed ({e}), using standard loading...")
                    self.model = SentenceTransformer(model_path, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_path, trust_remote_code=True)
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3_sentence_transformer"
            
            # Check if Qwen3 supports query prompts
            self.supports_query_prompt = hasattr(self.model, 'prompts') and 'query' in getattr(self.model, 'prompts', {})
            
            print(f"Qwen3 model loaded successfully with dimension: {self.dimension}")
            if self.supports_query_prompt:
                print("Query prompt support detected")
                
        except Exception as e:
            print(f"Failed to load Qwen3 model: {e}")
            print("Falling back to default SentenceTransformers model...")
            self._fallback_to_sentence_transformer()

    def _init_standard_sentence_transformer(self):
        """Initialize standard SentenceTransformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "sentence_transformer"
            self.supports_query_prompt = False
            print(f"SentenceTransformer model loaded with dimension: {self.dimension}")
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _fallback_to_sentence_transformer(self):
        """Fallback to default SentenceTransformer model"""
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using fallback model: {fallback_model}")
        self.model_name = fallback_model
        self._init_standard_sentence_transformer()

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to vectors
        
        Args:
        - texts: List of texts to encode
        - is_query: Whether these are query texts (for Qwen3 prompt optimization)
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.model_type == "remote":
            return self._encode_remote(texts)

        # Use query prompt for Qwen3 models when encoding queries
        if self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self._encode_with_query_prompt(texts)
        else:
            return self._encode_standard(texts)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode single text
        
        Args:
        - text: Text to encode
        - is_query: Whether this is a query text (for Qwen3 prompt optimization)
        """
        return self.encode([text], is_query=is_query)[0]
    
    def encode_query(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries with optimal settings for Qwen3
        """
        return self.encode(queries, is_query=True)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents (no query prompt)
        """
        return self.encode(documents, is_query=False)

    def _encode_remote(self, texts: List[str]) -> np.ndarray:
        """Encode texts using remote embedding API."""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )

            # Ensure order by index
            data = sorted(response.data, key=lambda d: d.index)
            vectors = [item.embedding for item in data]
            embeddings = np.array(vectors, dtype=np.float32)

            # Dimension bookkeeping and validation
            if embeddings.size == 0:
                raise ValueError("Remote embedding API returned empty embeddings.")

            detected_dim = embeddings.shape[1]
            if self.dimension and detected_dim != self.dimension:
                print(
                    f"Warning: embedding dimension mismatch (config={self.dimension}, remote={detected_dim}); "
                    "updating to remote dimension."
                )
                self.dimension = detected_dim
            elif not self.dimension:
                self.dimension = detected_dim

            # Normalize to align with SentenceTransformers behavior
            return self._normalize_embeddings(embeddings)

        except Exception as e:
            raise RuntimeError(f"Remote embedding request failed: {e}") from e
    
    def _encode_with_query_prompt(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3 query prompt"""
        try:
            embeddings = self.model.encode(
                texts, 
                prompt_name="query",  # Use Qwen3's query prompt
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            print(f"Query prompt encoding failed: {e}, falling back to standard encoding")
            return self._encode_standard(texts)
    
    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        """Encode texts using standard method"""
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embedding vectors to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
