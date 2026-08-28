# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code
in this repository.

## Commands

```bash
# Setup (required before first run)
cp config.py.example config.py   # edit with API key and IRIS credentials
pip install -r requirements.txt

# Tests
pytest cross/tests/ -v            # cross-session memory tests (127 tests)
pytest tests/ -v                  # root package unit tests
pytest OmniSimpleMem/tests/ -v    # multimodal tests (needs GPU deps for some)
pytest cross/tests/test_e2e.py -v # end-to-end cross-session test

# Single test
pytest cross/tests/test_storage.py::TestSQLiteStorage::test_create_session -v

# Benchmarks
python test_locomo10.py
python test_locomo10.py --num-samples 5 --result-file results.json

# MCP server
cd MCP && python run.py           # bare metal
docker compose up -d              # Docker (web UI at localhost:8000)
```

## Architecture

After the v0.3.0 upstream refactor, the repo has two coexisting layouts:

```text
simplemem/                ← installable package (upstream v0.3.0+)
    core/                 LLM pipeline — memory_builder, hybrid_retriever,
                          answer_generator, settings
    core/database/        vector_store.py (LanceDB — upstream default)
    core/models/          MemoryEntry, Dialogue
    core/utils/           embedding, llm_client
    text/system.py        SimpleMemSystem (package entry point)
    multimodal/           Omni-SimpleMem (text+image+audio+video)
    evolver/              EvolveMem — self-evolving memory (v0.3.0 new)
    integrations/         MCP server, SKILL, reference copy

database/vector_store.py  ← IRIS fork: IRIS SQL + HNSW vector store
                            (replaces simplemem/core/database/vector_store.py
                            for this fork; imports from simplemem.core.*)
simplemem_router.py       unified entry point; auto-detects text/omni backend
main.py                   SimpleMemSystem (flat-path entry point, still works)

cross/                    cross-session persistent memory
    orchestrator.py       top-level facade; use create_orchestrator()
    session_manager.py    session lifecycle (start/record/stop/end)
    context_injector.py   token-budgeted context injection at session start
    consolidation.py      memory decay/merge/prune worker
    collectors.py         event recording with 3-tier redaction
    storage_sqlite.py     session/event/observation metadata (default)
    storage_iris_sql.py   IRIS SQL metadata backend (use_iris_sql=True)
    storage_iris.py       IRIS vector store for cross_memory_entries
    storage_factory.py    create_sql_storage(use_iris=False)
    api_http.py           FastAPI REST API
    api_mcp.py            MCP tool definitions (8 tools)
```

### Key data flow (IRIS fork)

`add_dialogue()` → `MemoryBuilder` extracts `MemoryEntry` objects via LLM
(not chunking) → `database/vector_store.py` (IRIS) → `memory_entries` + HNSW.

`ask()` → `HybridRetriever` → parallel `VECTOR_COSINE` + `$FIND` + `WHERE`
SQL → `AnswerGenerator`.

## Hard constraints

- **`simplemem/core/`** is the upstream package — treat as read-only. Our IRIS
  vector store lives at `database/vector_store.py` alongside it, not inside it.
- `cross/` imports from `simplemem.core.*` (not flat `models.*`/`utils.*`).
- `simplemem/integrations/reference/` is a frozen snapshot — not active code.
- `SKILL/simplemem-skill/` is a standalone distribution copy (LanceDB) — no updates.
- `simplemem/multimodal/` (OmniSimpleMem) uses its own vector stores; IRIS backend
  in `database/` does not affect `mode="omni"`.
- Python 3.10 required. `config.py` is gitignored — never commit it.
- `tests/` IRIS vector store tests skip automatically when no container is reachable.
  Run a container first: `idt container up --name simplemem-iris --port <port>`

## IRIS backend notes

- Thread-local connections (`threading.local()`): each thread opens its own connection.
  `enable_parallel_retrieval=True` gives true DB parallelism.
- `TOP N` must be a literal integer — IRIS rejects parameterized `TOP ?`.
- HNSW index only activates with both `TOP N` and `ORDER BY ... DESC` on the distance
  function.
- After an IRIS upgrade, a stale HNSW version error requires:
  `DROP INDEX HNSWIdx ON TABLE memory_entries` then restart (recreated automatically).
- Changing `EMBEDDING_MODEL` to a different output dimension requires dropping and
  recreating the table (all stored memories lost).

## Cross-session storage selection

```python
# SQLite metadata + IRIS vectors (default, good for local/single-machine)
orch = create_orchestrator("my-project")

# Full IRIS: metadata AND vectors in one namespace; enables cross-table SQL JOINs
orch = create_orchestrator("my-project", use_iris_sql=True)
```
