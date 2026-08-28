# cross/ — Cross-Session Persistent Memory

## OVERVIEW

Persistent memory across conversations — session lifecycle, vector storage, context injection, consolidation. Outperforms Claude-Mem by 64% on LoCoMo. Async-first design.

## WHERE TO LOOK

| Task | Location |
|------|----------|
| All cross-session ops (facade) | `orchestrator.py` → `CrossMemOrchestrator` + `create_orchestrator()` |
| Session lifecycle (start/record/stop/end) | `session_manager.py` |
| Vector storage (semantic search) | `storage_iris.py` |
| Session metadata (SQLite) | `storage_sqlite.py` — 6 tables |
| Token-budgeted context injection | `context_injector.py` |
| Event collection + redaction | `collectors.py` — 3-tier `RedactionFilter` |
| Memory decay/merge/prune | `consolidation.py` — `ConsolidationWorker` |
| Lifecycle hooks (abstract) | `hooks.py` — `SessionHooks` base class |
| REST API (FastAPI) | `api_http.py` — 8 endpoints under `/cross/` |
| MCP tools | `api_mcp.py` — `MCPToolRegistry` with 8 tools |
| Pydantic models / enums | `types.py` |
| Tests | `tests/` — 127 tests, no GPU/API required |

## ARCHITECTURE

`create_orchestrator(project=...)` → `CrossMemOrchestrator` (facade) → `SessionManager` (SQLite) + `ContextInjector` (LanceDB) + `ConsolidationWorker`.

Storage: SQLite at `~/.simplemem-cross/cross_memory.db`, IRIS vector store (table: `cross_memory_entries`).

## CONSTRAINTS

- **Never import from `core/`** — only wraps SimpleMem via composition (duck typing, not subclass)
- Async-first: all public orchestrator methods are `async`
- Multi-tenant via `tenant_id` param — default is `"default"`
- Max context tokens default: 2000 (configurable per `create_orchestrator`)

## ANTI-PATTERNS

- Do NOT call `storage_sqlite.py` or `storage_lancedb.py` directly from outside — go through `orchestrator.py`
- Do NOT add Chinese to any source, comments, or strings
- Tests use real SQLite temp DBs + mocked LanceDB — do NOT add live API calls to tests

## TEST COMMANDS

```bash
pytest cross/tests/ -v
pytest cross/tests/test_e2e.py -v      # End-to-end lifecycle
pytest cross/tests/test_storage.py -v  # Storage backends
```
