# OmniSimpleMem/ — Multimodal Memory Extension

## OVERVIEW

Extends SimpleMem to text + image + audio + video memory. SOTA on LoCoMo (F1=0.613, +47%) and Mem-Gallery (F1=0.810, +51%). Separate package with own setup.py and requirements.txt.

## STRUCTURE

```
OmniSimpleMem/
├── omni_memory/           # Core package
│   ├── orchestrator.py    # Top-level facade (entry point)
│   ├── app.py             # FastAPI app
│   ├── core/              # Event + MAU primitives
│   ├── processors/        # Per-modality ingest (text/image/audio/video + base)
│   ├── retrieval/         # Pyramid retriever, BM25, query expansion
│   ├── storage/           # MAU store, semantic store, cold storage, vector store
│   ├── routing/           # Modality routing policy + features
│   ├── knowledge/         # Knowledge graph (entity extractor, graph retriever)
│   ├── evolution/         # Meta-controller, experience engine, strategy optimizer
│   ├── triggers/          # Audio + visual trigger detection
│   ├── parametric/        # Parametric memory consolidation + distillation
│   ├── evaluation/        # Benchmark runners + metrics
│   └── graph/             # Event store + event manager
├── benchmarks/            # LoCoMo + Mem-Gallery benchmark runners
├── examples/              # quickstart.py, multimodal_memory.py, api_server.py
├── configs/               # YAML configs for benchmarks
└── tests/                 # 8 test files (unit, no GPU by default)
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| All omni ops (entry point) | `omni_memory/orchestrator.py` |
| Add/fix modality processor | `omni_memory/processors/<modality>_processor.py` |
| Retrieval behavior | `omni_memory/retrieval/pyramid_retriever.py` |
| Knowledge graph augmentation | `omni_memory/knowledge/knowledge_graph.py` |
| Routing policy | `omni_memory/routing/policy.py` |
| Evolution / meta-learning | `omni_memory/evolution/meta_controller.py` |
| Run LoCoMo benchmark | `benchmarks/locomo/run_locomo.py` |
| Run Mem-Gallery benchmark | `benchmarks/memgallery/adapter.py` |
| Quick usage examples | `examples/quickstart.py` |

## KEY DESIGN

Three principles: **Selective Ingestion** (entropy-driven filtering per modality) → **Progressive Retrieval** (FAISS + BM25, pyramid token-budget) → **Knowledge Graph Augmentation** (multi-hop cross-modal reasoning).

Router auto-selects omni backend when `add_image()`, `add_audio()`, or `add_video()` is called first via `simplemem_router.py`.

## ANTI-PATTERNS

- Do NOT modify root `core/` or `main.py` — omni uses its own internal core
- GPU-dependent tests require `requirements-gpu.txt` — standard `pytest` skips them
- Do NOT import omni_memory from root `core/` — one-way dependency

## COMMANDS

```bash
pip install -e OmniSimpleMem/   # Install as editable package
pytest OmniSimpleMem/tests/ -v
python OmniSimpleMem/examples/quickstart.py
```
