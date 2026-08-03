"""
OrcaRouter integration for LLM and Embedding services.

OrcaRouter is an OpenAI-compatible gateway; it uses the same wire protocol as
Requesty/OpenRouter, so the LLM transport is inherited from the Requesty
integration and only the endpoint, defaults, and API-key validation differ.
"""

from typing import Dict, Optional

from .requesty import RequestyClient


class OrcaRouterClient(RequestyClient):
    """
    OrcaRouter API client for LLM and Embedding operations.
    Each instance is bound to a specific user's API key.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.orcarouter.ai/v1",
        llm_model: str = "openai/gpt-5.6-sol",
        embedding_model: str = "openai/text-embedding-3-small",
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            llm_model=llm_model,
            embedding_model=embedding_model,
        )

    async def verify_api_key(self) -> tuple[bool, Optional[str]]:
        """
        Verify that the API key is a valid OrcaRouter key.

        OrcaRouter keys start with ``sk-orca-``. The key is validated against
        ``GET /v1/models`` with the bearer token (this call does not consume
        any tokens).
        """
        if not self.api_key or not self.api_key.startswith("sk-orca-"):
            return False, "Invalid key format. OrcaRouter API keys start with 'sk-orca-'. Get yours at orcarouter.ai"

        try:
            client = self._get_client()
            # Use /models endpoint to verify the key
            response = await client.get("/models")
            if response.status_code == 200:
                return True, None
            elif response.status_code == 401:
                return False, "Invalid or expired API key"
            elif response.status_code == 403:
                return False, "API key access denied"
            else:
                return False, f"API error: {response.status_code}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"


class OrcaRouterClientManager:
    """
    Manages OrcaRouter client instances for multiple users.
    Provides client pooling and lifecycle management.
    """

    def __init__(
        self,
        base_url: str = "https://api.orcarouter.ai/v1",
        llm_model: str = "openai/gpt-5.6-sol",
        embedding_model: str = "openai/text-embedding-3-small",
    ):
        self.base_url = base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._clients: Dict[str, OrcaRouterClient] = {}

    def get_client(self, api_key: str) -> OrcaRouterClient:
        """
        Get or create an OrcaRouter client for the given API key

        Args:
            api_key: User's OrcaRouter API key

        Returns:
            OrcaRouterClient instance
        """
        # Use hash of API key as cache key for security
        import hashlib
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        if key_hash not in self._clients:
            self._clients[key_hash] = OrcaRouterClient(
                api_key=api_key,
                base_url=self.base_url,
                llm_model=self.llm_model,
                embedding_model=self.embedding_model,
            )

        return self._clients[key_hash]

    async def close_all(self):
        """Close all client connections"""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    async def remove_client(self, api_key: str):
        """Remove and close a specific client"""
        import hashlib
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        if key_hash in self._clients:
            await self._clients[key_hash].close()
            del self._clients[key_hash]
