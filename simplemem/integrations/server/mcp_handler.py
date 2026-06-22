"""
MCP Protocol Handler - JSON-RPC 2.0 over SSE

Implements the Model Context Protocol for remote clients like Claude Desktop.
"""

import json
import asyncio
from typing import Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict

from .auth.models import User, MemoryEntry
from .database.vector_store import MultiTenantVectorStore
from .core.memory_builder import MemoryBuilder
from .core.retriever import Retriever
from .core.answer_generator import AnswerGenerator

# Type alias for client manager (supports both OpenRouter and Ollama)
ClientManager = object  # Duck-typed: can be OpenRouterClientManager or OllamaClientManager


@dataclass
class JsonRpcRequest:
    jsonrpc: str
    method: str
    id: Optional[int | str]
    params: Optional[dict] = None


@dataclass
class JsonRpcResponse:
    jsonrpc: str = "2.0"
    id: Optional[int | str] = None
    result: Optional[Any] = None
    error: Optional[dict] = None

    def to_dict(self):
        d = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


# MCP Protocol Constants
MCP_VERSION = "2025-03-26"  # Streamable HTTP transport
SERVER_NAME = "simplemem"
SERVER_VERSION = "1.0.0"


class MCPHandler:
    """
    Handles MCP protocol messages for a specific user session.
    """

    def __init__(
        self,
        user: User,
        api_key: str,
        vector_store: MultiTenantVectorStore,
        client_manager: ClientManager,
        settings: Any,
    ):
        self.user = user
        self.api_key = api_key
        self.vector_store = vector_store
        self.client_manager = client_manager
        self.settings = settings
        self.initialized = False

        # Lazy-loaded components
        self._memory_builder: Optional[MemoryBuilder] = None
        self._retriever: Optional[Retriever] = None
        self._answer_generator: Optional[AnswerGenerator] = None

    def _get_client(self):
        return self.client_manager.get_client(self.api_key)

    def _get_memory_builder(self) -> MemoryBuilder:
        if not self._memory_builder:
            self._memory_builder = MemoryBuilder(
                llm_client=self._get_client(),
                vector_store=self.vector_store,
                table_name=self.user.table_name,
                window_size=self.settings.window_size,
                overlap_size=self.settings.overlap_size,
                temperature=self.settings.llm_temperature,
            )
        return self._memory_builder

    def _get_retriever(self) -> Retriever:
        if not self._retriever:
            self._retriever = Retriever(
                llm_client=self._get_client(),
                vector_store=self.vector_store,
                table_name=self.user.table_name,
                semantic_top_k=self.settings.semantic_top_k,
                keyword_top_k=self.settings.keyword_top_k,
                enable_planning=self.settings.enable_planning,
                enable_reflection=self.settings.enable_reflection,
                max_reflection_rounds=self.settings.max_reflection_rounds,
                temperature=self.settings.llm_temperature,
            )
        return self._retriever

    def _get_answer_generator(self) -> AnswerGenerator:
        if not self._answer_generator:
            self._answer_generator = AnswerGenerator(
                llm_client=self._get_client(),
                temperature=self.settings.llm_temperature,
            )
        return self._answer_generator

    async def handle_message(self, message: str) -> str:
        """Handle a JSON-RPC message and return response"""
        try:
            data = json.loads(message)
            request = JsonRpcRequest(
                jsonrpc=data.get("jsonrpc", "2.0"),
                method=data.get("method", ""),
                id=data.get("id"),
                params=data.get("params", {}),
            )
            response = await self._dispatch(request)
            return json.dumps(response.to_dict(), ensure_ascii=False)
        except json.JSONDecodeError as e:
            return json.dumps(JsonRpcResponse(
                error={"code": -32700, "message": f"Parse error: {e}"}
            ).to_dict())
        except Exception as e:
            return json.dumps(JsonRpcResponse(
                error={"code": -32603, "message": f"Internal error: {e}"}
            ).to_dict())

    async def _dispatch(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Dispatch request to appropriate handler"""
        method = request.method
        params = request.params or {}

        handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
        }

        handler = handlers.get(method)
        if not handler:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32601, "message": f"Method not found: {method}"}
            )

        try:
            result = await handler(params)
            return JsonRpcResponse(id=request.id, result=result)
        except Exception as e:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)}
            )

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request"""
        self.initialized = True
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "description": "SimpleMem - Advanced Lifelong Memory System for LLM Agents. "
                              "Features: Semantic lossless compression, coreference resolution, "
                              "temporal anchoring, hybrid retrieval (semantic + lexical + symbolic), "
                              "and intelligent query planning with reflection.",
            },
            "instructions": """SimpleMem is your long-term memory system. Use it to:

1. STORE conversations: Use memory_add or memory_add_batch to save dialogues.
   The system automatically extracts facts, resolves pronouns, and anchors timestamps.
   Memories are stored immediately - no manual flush needed.

2. RECALL information: Use memory_query to ask questions about past conversations.
   The system retrieves relevant memories and synthesizes answers.

3. BROWSE memories: Use memory_retrieve to see raw stored facts with metadata.

4. MANAGE: Use memory_stats to check status, memory_clear to reset (careful!).

Tips:
- Use memory_query for natural questions, memory_retrieve for browsing
- Each memory_add call processes and stores data immediately""",
        }

    async def _handle_initialized(self, params: dict) -> dict:
        """Handle initialized notification"""
        return {}

    async def _handle_ping(self, params: dict) -> dict:
        """Handle ping request"""
        return {}

    async def _handle_tools_list(self, params: dict) -> dict:
        """Handle tools/list request"""
        return {
            "tools": [
                {
                    "name": "memory_add",
                    "description": """Add a dialogue to SimpleMem long-term memory system.

SimpleMem is an advanced lifelong memory system that:
- Stores conversations as atomic, self-contained facts (no pronouns, absolute timestamps)
- Uses semantic compression to extract key information (persons, locations, entities, topics)
- Supports hybrid retrieval (semantic + keyword + metadata filtering)

The dialogue is processed immediately by LLM and stored. No manual flush needed.

Example: memory_add(speaker="Alice", content="I'll meet Bob at Starbucks tomorrow at 2pm")
→ Stored as: "Alice will meet Bob at Starbucks on 2025-01-14 at 14:00"
   with metadata: persons=["Alice","Bob"], location="Starbucks", topic="Meeting arrangement\"""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "speaker": {
                                "type": "string",
                                "description": "Name of the speaker (will be used for coreference resolution)",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content of the dialogue (pronouns will be resolved, relative times will be converted to absolute)",
                            },
                            "timestamp": {
                                "type": "string",
                                "description": "ISO 8601 timestamp of when this was said (used for temporal anchoring). Defaults to now.",
                            },
                        },
                        "required": ["speaker", "content"],
                    },
                },
                {
                    "name": "memory_add_batch",
                    "description": """Add multiple dialogues to SimpleMem at once.

Efficient for importing conversation history. Each dialogue is processed with:
- Coreference resolution (he/she → actual names)
- Temporal anchoring (tomorrow → actual date)
- Entity extraction (persons, locations, organizations)

All dialogues are processed immediately and stored. No manual flush needed.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "dialogues": {
                                "type": "array",
                                "description": "List of dialogues to add",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "speaker": {"type": "string", "description": "Speaker name"},
                                        "content": {"type": "string", "description": "Dialogue content"},
                                        "timestamp": {"type": "string", "description": "ISO 8601 timestamp"},
                                    },
                                    "required": ["speaker", "content"],
                                },
                            },
                        },
                        "required": ["dialogues"],
                    },
                },
                {
                    "name": "memory_query",
                    "description": """Query SimpleMem and get an AI-generated answer based on stored memories.

This is the primary way to retrieve information from long-term memory. The system:
1. Analyzes query complexity (simple fact vs multi-hop reasoning)
2. Generates targeted search queries
3. Performs hybrid retrieval (semantic similarity + keyword matching + metadata filtering)
4. Optionally reflects to find missing information for complex queries
5. Synthesizes a concise answer from retrieved contexts

Best for: "When did Alice and Bob plan to meet?", "What does Alice think about the project?", "Summarize recent events with Bob"

Returns: answer, reasoning, confidence level, and number of memory entries used.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Natural language question about stored memories",
                            },
                            "enable_reflection": {
                                "type": "boolean",
                                "description": "Enable iterative refinement for complex multi-hop queries. Default: true. Disable for simple factual lookups to save tokens.",
                            },
                        },
                        "required": ["question"],
                    },
                },
                {
                    "name": "memory_retrieve",
                    "description": """Retrieve relevant memory entries without generating an answer.

Returns raw memory entries with full metadata. Use this when you need:
- Direct access to stored facts
- To process/analyze memories yourself
- To show the user what's stored about a topic

Each entry contains: content (self-contained fact), timestamp, location, persons, entities, topic.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (semantic search + keyword matching)",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of entries to return. Default: 10",
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "memory_clear",
                    "description": """Clear ALL memories for this user. This action CANNOT be undone.

Use with caution. This removes all stored memory entries from the vector database.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "memory_stats",
                    "description": """Get statistics about the memory store.

Returns:
- Total number of stored memory entries
- User ID and table info

Use to check if memories are being stored correctly.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
            ]
        }

    async def _handle_tools_call(self, params: dict) -> dict:
        """Handle tools/call request"""
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool_handlers = {
            "memory_add": self._tool_memory_add,
            "memory_add_batch": self._tool_memory_add_batch,
            "memory_query": self._tool_memory_query,
            "memory_retrieve": self._tool_memory_retrieve,
            "memory_clear": self._tool_memory_clear,
            "memory_stats": self._tool_memory_stats,
        }

        handler = tool_handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")

        result = await handler(arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                }
            ]
        }

    async def _tool_memory_add(self, args: dict) -> dict:
        builder = self._get_memory_builder()
        return await builder.add_dialogue(
            speaker=args["speaker"],
            content=args["content"],
            timestamp=args.get("timestamp"),
        )

    async def _tool_memory_add_batch(self, args: dict) -> dict:
        builder = self._get_memory_builder()
        return await builder.add_dialogues(
            dialogues=args["dialogues"],
        )

    async def _tool_memory_query(self, args: dict) -> dict:
        retriever = self._get_retriever()
        generator = self._get_answer_generator()

        contexts = await retriever.retrieve(
            query=args["question"],
            enable_reflection=args.get("enable_reflection", True),
        )

        answer_result = await generator.generate_answer(
            query=args["question"],
            contexts=contexts,
        )

        return {
            "question": args["question"],
            "answer": answer_result["answer"],
            "reasoning": answer_result["reasoning"],
            "confidence": answer_result["confidence"],
            "contexts_used": len(contexts),
        }

    async def _tool_memory_retrieve(self, args: dict) -> dict:
        retriever = self._get_retriever()
        top_k = args.get("top_k", 10)

        entries = await retriever.retrieve(
            query=args["query"],
            enable_reflection=False,
        )

        return {
            "query": args["query"],
            "results": [
                {
                    "content": e.lossless_restatement,
                    "timestamp": e.timestamp,
                    "location": e.location,
                    "persons": e.persons,
                    "entities": e.entities,
                    "topic": e.topic,
                }
                for e in entries[:top_k]
            ],
            "total": len(entries),
        }

    async def _tool_memory_clear(self, args: dict) -> dict:
        success = await self.vector_store.clear_table(self.user.table_name)
        return {
            "success": success,
            "message": "All memories cleared" if success else "Failed",
        }

    async def _tool_memory_stats(self, args: dict) -> dict:
        stats = self.vector_store.get_stats(self.user.table_name)
        builder = self._get_memory_builder()
        builder_stats = builder.get_stats()
        return {
            "user_id": self.user.user_id,
            "entry_count": stats.get("entry_count", 0),
            "total_dialogues_processed": builder_stats.get("total_dialogues_processed", 0),
        }

    async def _handle_resources_list(self, params: dict) -> dict:
        """Handle resources/list request"""
        return {
            "resources": [
                {
                    "uri": f"memory://{self.user.user_id}/stats",
                    "name": "Memory Statistics",
                    "description": "Statistics about your memory store",
                    "mimeType": "application/json",
                },
                {
                    "uri": f"memory://{self.user.user_id}/all",
                    "name": "All Memories",
                    "description": "All stored memory entries",
                    "mimeType": "application/json",
                },
            ]
        }

    async def _handle_resources_read(self, params: dict) -> dict:
        """Handle resources/read request"""
        uri = params.get("uri", "")

        if uri.endswith("/stats"):
            stats = self.vector_store.get_stats(self.user.table_name)
            content = json.dumps(stats, ensure_ascii=False)
        elif uri.endswith("/all"):
            entries = await self.vector_store.get_all_entries(self.user.table_name)
            content = json.dumps({
                "entries": [e.to_dict() for e in entries],
                "total": len(entries),
            }, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown resource: {uri}")

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content,
                }
            ]
        }
