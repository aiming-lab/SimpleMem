# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-10
**Commit:** 94ef7d7
**Branch:** main

## OVERVIEW

SimpleMem is a research codebase for efficient lifelong LLM agent memory — text (SimpleMem) and multimodal (Omni-SimpleMem) — published on arXiv. Python 3.10, LanceDB vectors, OpenAI-compatible APIs.

## STRUCTURE

```
SimpleMem/
├── main.py                  # SimpleMemSystem class — core text memory (3-stage pipeline)
├── simplemem_router.py      # Unified entry point; registry-based factory for text/omni modes
├── config.py.example        # Template — copy to config.py before use
├── core/                    # Memory pipeline primitives (memory_builder, hybrid_retriever, answer_generator)
├── models/                  # Pydantic models: MemoryEntry, Dialogue
├── utils/                   # LLM client, embedding model
├── database/                # LanceDB vector store wrapper
├── cross/                   # Cross-session persistent memory (separate sub-system)
├── OmniSimpleMem/           # Multimodal memory extension (text+image+audio+video)
├── MCP/                     # Cloud MCP server + web UI (Docker-deployable)
├── SKILL/simplemem-skill/   # Claude Skills integration package
├── test_locomo10.py         # Primary benchmark runner (LoCoMo-10)
├── test_ref/                # Reference test utilities (load_dataset, test_advanced)
├── tests/                   # Unit tests for root package
└── docs/i18n/               # Translated READMEs (12 languages, docs only)
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add/fix text memory logic | `core/memory_builder.py`, `core/hybrid_retriever.py` |
| Change retrieval behavior | `core/hybrid_retriever.py`, `core/answer_generator.py` |
| Add new router backend | `simplemem_router.py` — use `register()` |
| Cross-session memory | `cross/orchestrator.py` → facade for all cross ops |
| Multimodal memory | `OmniSimpleMem/omni_memory/orchestrator.py` |
| MCP server | `MCP/server/mcp_handler.py`, `MCP/server/http_server.py` |
| Benchmark eval | `test_locomo10.py` (text), `OmniSimpleMem/benchmarks/` |
| Config | `config.py` (gitignored — copy from `config.py.example`) |
| Docker deployment | `docker-compose.yml`, `MCP/run.sh` |

## ARCHITECTURE — TEXT PIPELINE

Three-stage: `add_dialogue()` → **Stage 1** MemoryBuilder (semantic compression) → **Stage 2** online synthesis → VectorStore; `ask()` → **Stage 3** HybridRetriever (planning) → AnswerGenerator.

All stages in `core/`. Entry via `SimpleMemSystem` (`main.py`) or router (`simplemem_router.py`).

## CONSTRAINTS (CRITICAL)

- **Original SimpleMem (`main.py`, `core/`, `models/`, `utils/`, `database/`) is byte-identical to the published paper** — never modify these files
- Cross-session extensions live in `cross/` using composition, not subclassing
- All code in English — no Chinese in source
- Python 3.10 required (type hints and match syntax)
- `config.py` is gitignored — never commit API keys

## ANTI-PATTERNS

- Do NOT modify `main.py` or `core/` — use composition/wrapping
- Do NOT import `cross/` from `core/` — one-way dependency only
- Do NOT hardcode model names — use `config.py` values
- `MCP/reference/` is a frozen snapshot — do not treat as active code
- Do NOT run tests requiring GPU without `requirements-gpu.txt` installed

## COMMANDS

```bash
# Setup
cp config.py.example config.py
pip install -r requirements.txt

# Benchmark
python test_locomo10.py
python test_locomo10.py --num-samples 5 --result-file results.json

# Cross-session tests
pytest cross/tests/ -v

# Root tests
pytest tests/ -v

# OmniSimpleMem tests
pytest OmniSimpleMem/tests/ -v

# MCP server (Docker)
docker compose up -d
# MCP server (bare metal)
cd MCP && python run.py

# GPU deps
pip install -r requirements-gpu.txt
```

## NOTES

- `simplemem_router.py` auto-detects backend on first API call: `add_dialogue()` → text, `add_image()/add_text()/add_audio()/add_video()` → omni
- IRIS stores vectors in SQL tables: `memory_entries` (text pipeline), `cross_memory_entries` (cross-session). Tables and HNSW index created automatically.
- Embedding default: `Qwen/Qwen3-Embedding-0.6B` (1024-d); configurable in `config.py`
- Parallel mode: `enable_parallel_processing=True` for batch ops
- MCP cloud: `mcp.simplemem.cloud` — self-host via Docker
