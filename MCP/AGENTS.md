# MCP/server/ — MCP Cloud Server

## OVERVIEW

Production MCP server for cloud-hosted SimpleMem at `mcp.simplemem.cloud`. FastAPI + Streamable HTTP (MCP 2025-03-26). Multi-tenant with token auth. Deployable via Docker.

## WHERE TO LOOK

| Task | Location |
|------|----------|
| MCP tool handlers | `mcp_handler.py` |
| HTTP server / app factory | `http_server.py` |
| Core memory logic (server-side) | `core/memory_builder.py`, `core/retriever.py`, `core/answer_generator.py` |
| User token management | `auth/token_manager.py` |
| Auth models | `auth/models.py` |
| Per-user LanceDB storage | `database/vector_store.py`, `database/user_store.py` |
| Ollama integration | `integrations/ollama.py` |
| OpenRouter integration | `integrations/openrouter.py` |
| Server config/settings | `../config/settings.py` |
| Entry point | `../run.py` |
| Frontend (web UI) | `../frontend/` — static HTML/JS/CSS |

## ARCHITECTURE

`run.py` → `http_server.py` (FastAPI app) → `mcp_handler.py` (JSON-RPC dispatch) → per-user `VectorStore` instances. Auth via Bearer token → `token_manager.py`. Each user gets isolated LanceDB table.

## ANTI-PATTERNS

- `MCP/reference/` is a **frozen snapshot** of the original SimpleMem — never edit it, never treat as active code
- Server-side `core/` is an independent copy from root `core/` — changes must be made separately (not synced automatically)
- Do NOT hardcode model names — use `../config/settings.py`

## DOCKER

```bash
# From repo root
docker compose up -d
# Web UI: http://localhost:8000/
# MCP endpoint: http://localhost:8000/mcp/sse?token=<TOKEN>
```
