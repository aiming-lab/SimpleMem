# SimpleMem MCP Server

**Production-Ready Memory Service for LLM Agents via Model Context Protocol (MCP)**

SimpleMem MCP Server is a cloud-hosted long-term memory service for LLM agents, implementing the **Streamable HTTP** transport (MCP 2025-03-26 spec). It enables AI assistants like Claude, Cursor, and other MCP-compatible clients to store, retrieve, and query conversational memories with ease.

## Features

- **Semantic Lossless Compression**: Converts dialogues into atomic, self-contained facts
- **Coreference Resolution**: Automatically replaces pronouns (he/she/it) with actual names
- **Temporal Anchoring**: Converts relative times (tomorrow, next week) to absolute timestamps
- **Hybrid Retrieval**: Semantic search + keyword matching + metadata filtering
- **Intelligent Planning**: Automatic query decomposition and reflection for complex queries
- **Multi-tenant Isolation**: Per-user data tables with token authentication
- **OpenRouter Integration**: Powered by OpenRouter's LLM and Embedding services
- **OrcaRouter Integration**: OrcaRouter is supported as a first-class provider (an OpenAI-compatible gateway), alongside Requesty and Ollama, via the `LLM_PROVIDER` setting
- **Production Optimized**: Faster response times compared to the academic reference implementation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleMem MCP Server                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              HTTP Server (FastAPI)                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │  │
│  │  │  Web UI    │  │  REST API  │  │  MCP Streamable    │  │  │
│  │  │  (/)       │  │  (/api/*)  │  │  HTTP (/mcp)       │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Token Authentication                     │  │
│  │            (JWT + AES-256 Encrypted API Keys)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐     │
│  │  User A    │       │  User B    │       │  User C    │     │
│  │  Table     │       │  Table     │       │  Table     │     │
│  └────────────┘       └────────────┘       └────────────┘     │
│  └─────────────────── LanceDB ──────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               OpenRouter API Integration                  │  │
│  │  LLM: openai/gpt-4.1-mini    Embed: qwen/qwen3-embed-4b  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Using the Cloud Service

The easiest way to use SimpleMem is via our hosted service at **https://mcp.simplemem.cloud**

1. Visit `https://mcp.simplemem.cloud`
2. Enter your OpenRouter API Key
3. Get your authentication token
4. Configure your MCP client (see below)

### Self-Hosting

#### 1. Install Dependencies

```bash
cd MCP
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Configure Environment Variables (Optional)

```bash
# Production environment recommended settings
export JWT_SECRET_KEY="your-secure-random-secret-key"
export ENCRYPTION_KEY="your-32-byte-encryption-key!!"
```

#### 3. Start the Server

```bash
python run.py
```

Output:
```
============================================================
  SimpleMem MCP Server
  Multi-tenant Memory Service for LLM Agents
============================================================

  Web UI:     http://localhost:8000/
  REST API:   http://localhost:8000/api/
  MCP:        http://localhost:8000/mcp

------------------------------------------------------------
```

## MCP Protocol

### Protocol Information

| Item | Value |
|------|-------|
| Protocol Version | 2025-03-26 |
| Transport | Streamable HTTP |
| Message Format | JSON-RPC 2.0 |
| Authentication | Bearer Token |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp` | POST | Send JSON-RPC messages (requests, notifications) |
| `/mcp` | GET | Server-to-client SSE stream |
| `/mcp` | DELETE | Terminate session |

### Authentication

All MCP requests require a Bearer token in the Authorization header:

```
Authorization: Bearer <your-token>
```

After initialization, include the session ID header:

```
Mcp-Session-Id: <session-id>
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_add` | Add a single dialogue to memory (auto-extracts facts, resolves pronouns, anchors timestamps) |
| `memory_add_batch` | Add multiple dialogues at once |
| `memory_query` | Query memories and generate AI-synthesized answers (with planning + hybrid retrieval + reflection) |
| `memory_retrieve` | Retrieve relevant memory entries (returns raw data) |
| `memory_stats` | Get memory statistics |
| `memory_clear` | Clear all memories (irreversible) |

## Client Configuration

Add to your MCP JSON settings:

```json
{
  "mcpServers": {
    "simplemem": {
      "url": "https://mcp.simplemem.cloud/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```


## How It Works

### Write Flow (Dialogue -> Memory)

```
Dialogue Input                  Processing                     Memory Storage
───────────────────────────────────────────────────────────────────────────

"I'll meet Bob            ┌─────────────────┐
 at Starbucks             │ LLM Processing  │
 tomorrow at 3pm"      ──▶│                 │ ──────────────▶  Atomic Fact
                          └─────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────────────┐
                                              │ Atomic Fact:            │
                                              │ "User will meet Bob at  │
                                              │  Starbucks on           │
                                              │  2025-01-15 at 15:00"   │
                                              │                         │
                                              │ persons: [User, Bob]    │
                                              │ location: Starbucks     │
                                              │ timestamp: 2025-01-15   │
                                              │ topic: Meeting          │
                                              └───────────┬─────────────┘
                                                          │
                                                          ▼
                                              ┌─────────────────────────┐
                                              │      Embedding          │
                                              │   (qwen3-embed-4b)      │
                                              └───────────┬─────────────┘
                                                          │
                                                          ▼
                                              ┌─────────────────────────┐
                                              │   LanceDB Vector Store  │
                                              └─────────────────────────┘
```

### Read Flow (Query -> Answer)

```
User Question: "When am I meeting Bob?"
                │
                ▼
┌───────────────────────────────┐
│  1. Query Complexity Analysis │
│     - Type: Temporal query    │
│     - Entity: Bob             │
│     - Complexity: 0.3 (simple)│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  2. Generate Search Queries   │
│     → "Bob meeting time"      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  3. Hybrid Retrieval          │
│     - Semantic (vector)       │
│     - Keyword (BM25)          │
│     - Metadata (persons)      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  4. Answer Generation         │
│     Context + Question → LLM  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Response:                    │
│  {                            │
│    "answer": "15 January 2025 │
│              at 3:00 PM at    │
│              Starbucks",      │
│    "confidence": "high",      │
│    "contexts_used": 1         │
│  }                            │
└───────────────────────────────┘
```

## Configuration Options

The LLM/embedding backend is selected with the `LLM_PROVIDER` environment variable (default `openrouter`):

| `LLM_PROVIDER` | Base URL env var | Default |
|----------------|------------------|---------|
| `openrouter` | `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `requesty` | `REQUESTY_BASE_URL` | `https://router.requesty.ai/v1` |
| `ollama` | `OLLAMA_BASE_URL` | `http://localhost:11434/v1` |
| `orcarouter` | `ORCAROUTER_BASE_URL` | `https://api.orcarouter.ai/v1` |

[OrcaRouter](https://www.orcarouter.ai) is an OpenAI-compatible gateway that provides unified access to many LLM vendors (OpenAI, Anthropic, Google, DeepSeek, Moonshot, and more) through a single endpoint. It also runs gateway-level, zero-trust security for AI agents on the same endpoint — screening every prompt/response and governing every tool call on a default-deny basis, with no application code changes. Keys start with `sk-orca-` and can be created at https://www.orcarouter.ai.

| Option | Default | Description |
|--------|---------|-------------|
| `window_size` | 20 | Number of dialogues per processing batch |
| `semantic_top_k` | 25 | Semantic search result count |
| `keyword_top_k` | 5 | Keyword search result count |
| `enable_planning` | true | Enable query planning |
| `enable_reflection` | true | Enable reflection iteration |
| `max_reflection_rounds` | 2 | Maximum reflection rounds |
| `llm_model` | openai/gpt-4.1-mini | LLM model |
| `embedding_model` | qwen/qwen3-embedding-4b | Embedding model |

## Development

```bash
# Development mode (auto-reload)
python run.py --reload

# Specify port
python run.py --port 3000

# View help
python run.py --help
```

## License

MIT License

## Note

Built upon SimpleMem research implementation, refactored and optimized for production deployment with multi-tenant support, faster processing, and comprehensive user isolation.
