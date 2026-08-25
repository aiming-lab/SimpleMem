

<h2 align="center"><b>Omni-SimpleMem: Unified Multimodal Memory for Lifelong AI Agents</b></h2>

<p align="center">
  <b><i>Extending <a href="https://github.com/aiming-lab/SimpleMem">SimpleMem</a> to multimodal — store, organize, and recall text, image, audio & video experiences with state-of-the-art accuracy.</i></b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="#-testing"><img src="https://img.shields.io/badge/Tests-126%20passed-brightgreen?logo=pytest&logoColor=white" alt="126 Tests Passed"></a>
  <a href="#-architecture"><img src="https://img.shields.io/badge/Modules-13%20packages-blueviolet?logo=buffer&logoColor=white" alt="13 Packages"></a>
  <a href="#-results"><img src="https://img.shields.io/badge/SOTA-LoCoMo%20%7C%20Mem--Gallery-ff6f00?logo=target&logoColor=white" alt="SOTA"></a>
  <a href="https://github.com/aiming-lab/AutoResearchClaw"><img src="https://img.shields.io/badge/Discovered_by-AutoResearchClaw-d73a49?logo=github&logoColor=white" alt="AutoResearchClaw"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#-architecture">Architecture</a> &nbsp;·&nbsp;
  <a href="#-results">Results</a> &nbsp;·&nbsp;
  <a href="#-benchmarks">Benchmarks</a> &nbsp;·&nbsp;
  <a href="#%EF%B8%8F-configuration">Config</a> &nbsp;·&nbsp;
  <a href="#-citation">Citation</a>
</p>

<p align="center">
  <img src="assets/framework.png" width="90%" alt="Omni-SimpleMem Framework Overview">
</p>

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem/OmniSimpleMem
pip install -e .
```

<details>
<summary>📦 Optional dependency groups</summary>

```bash
pip install -e ".[all]"      # Everything
pip install -e ".[visual]"   # Image/video (torch, transformers, CLIP)
pip install -e ".[audio]"    # Audio (soundfile, librosa)
pip install -e ".[vector]"   # FAISS vector search
pip install -e ".[server]"   # FastAPI REST server
pip install -e ".[dev]"      # Development (pytest)
```

</details>

### Basic Usage

```python
from omni_memory import OmniMemoryOrchestrator, OmniMemoryConfig

config = OmniMemoryConfig()
config.embedding.model_name = "all-MiniLM-L6-v2"
config.embedding.embedding_dim = 384

orchestrator = OmniMemoryOrchestrator(config=config, data_dir="./my_memory")

# Store
orchestrator.add_text(
    "User loves hiking in the Rocky Mountains every summer.",
    tags=["session_id:D1", "timestamp:2024-06-15"],
)

# Query
result = orchestrator.query("What does the user enjoy?", top_k=5)
for item in result.items:
    print(item["summary"])

orchestrator.close()
```

<details>
<summary>🌐 REST API Server</summary>

```bash
pip install -e ".[server]"
export OPENAI_API_KEY=your_key_here
python examples/api_server.py
# Visit http://localhost:8000/docs
```

</details>

<details>
<summary>🔌 MCP Server (use Omni-SimpleMem from any MCP client)</summary>

Expose multimodal memory to Claude Desktop or any MCP client over stdio:

```bash
python -m omni_mcp --data-dir ~/.omni_simplemem/mcp
```

Provides 12 tools — `omni_add_text` / `omni_add_image` / `omni_add_audio` /
`omni_add_video` / `omni_add_document`, `omni_query` / `omni_answer`, plus
namespace management. Media arguments accept local paths, `http(s)://`, Google
Drive share links, `s3://` (S3 or MinIO) and `gs://`. Every tool takes an
optional `namespace` giving each agent a fully isolated memory cluster.

See [`omni_mcp/README.md`](omni_mcp/README.md) for setup, tool reference and
configuration.

</details>

<details>
<summary>📝 More examples</summary>

- [`examples/quickstart.py`](examples/quickstart.py) — Basic text memory
- [`examples/multimodal_memory.py`](examples/multimodal_memory.py) — Multimodal content
- [`examples/api_server.py`](examples/api_server.py) — FastAPI REST server
- [`omni_mcp/`](omni_mcp/) — MCP server (multimodal memory over stdio)

</details>

---

## Highlights

<table>
<tr>
<td align="center" width="140">📈 <b>+411%</b><br><sub>LoCoMo F1</sub></td>
<td align="center" width="140">📈 <b>+214%</b><br><sub>Mem-Gallery F1</sub></td>
<td align="center" width="140">⚡ <b>5.81 q/s</b><br><sub>3.5x faster</sub></td>
<td align="center" width="140">🧠 <b>4 modalities</b><br><sub>Text · Image · Audio · Video</sub></td>
<td align="center" width="140">🔬 <b>~50 experiments</b><br><sub>Fully autonomous</sub></td>
</tr>
</table>

| Benchmark | Previous SOTA | Omni-SimpleMem (GPT-5.1) | Improvement |
|:--|:--:|:--:|:--:|
| **LoCoMo** (F1) | 0.432 (SimpleMem) | **0.613** | +41.9% |
| **Mem-Gallery** (F1) | 0.697 (MuRAG) | **0.810** | +16.2% |

> Omni-SimpleMem achieves SOTA across **all five LLM backbones** (GPT-4o, GPT-4o-mini, GPT-4.1-nano, GPT-5.1, GPT-5-nano), outperforming six baselines including MemVerse, Mem0, Claude-Mem, A-MEM, MemGPT, and SimpleMem. Architecture autonomously discovered by [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) (~50 experiments, ~72 hours, zero human intervention).



---

## 🏗️ Architecture

Omni-SimpleMem builds on three architectural principles:

### 1. Selective Ingestion

| | Modality | Method | Effect |
|:--:|:--|:--|:--|
| 🎞️ | **Visual** | CLIP cosine-similarity scene-change detection | ~70% storage reduction |
| 🎙️ | **Audio** | VAD silence/noise filtering | ~40% reduction |
| 📝 | **Text** | Jaccard overlap redundancy filter | Dedup |

All modalities unify into **Multimodal Atomic Units (MAUs)**: `<summary, embedding, cold_pointer, timestamp, modality, links>`. Lightweight metadata lives in **hot storage** (RAM); raw media in **cold storage** (disk/S3), loaded on demand.

### 2. Progressive Retrieval with Hybrid Search

**Hybrid search**: FAISS (dense) + BM25 (sparse) merged via **set-union** (autonomously discovered). Results expand progressively under a token budget:

| Level | Content | Cost |
|:--|:--|:--|
| 🔍 Preview | Summaries (~10 tokens) | Low |
| 📄 Details | Full text + metadata | Medium |
| 📦 Evidence | Raw content from cold storage | High |

### 3. Knowledge Graph Augmentation

A graph with **7 entity types** and **12 relation types** enables cross-modal multi-hop reasoning via seed identification → h-hop expansion → distance-decayed scoring.

<details>
<summary>📐 Full system diagram</summary>

```
Input (Text / Image / Audio / Video)
  │
  ▼
┌─────────────────────────────┐
│   Entropy-Driven Filtering  │  ← CLIP / VAD / Jaccard
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   Two-Tier Storage          │
│   Hot: summaries, embeddings│  ← FAISS + BM25 index
│   Cold: raw media (disk/S3) │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   Hybrid Search             │
│   Dense (FAISS) ∪ Sparse    │  ← Set-union merging
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   Pyramid Retrieval         │  ← Token-budget aware
│   Preview → Details →       │
│   Evidence                  │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   Knowledge Graph Traversal │  ← h-hop expansion
└─────────────┬───────────────┘
              ▼
         LLM Answer
```

</details>

---

## 📊 Results

<details open>
<summary><b>LoCoMo — Overall F1</b></summary>

| Method | GPT-4o | GPT-4o-mini | GPT-4.1-nano | GPT-5.1 | GPT-5-nano |
|:--|:--:|:--:|:--:|:--:|:--:|
| Mem0 | 0.397 | 0.364 | 0.310 | 0.390 | 0.352 |
| A-MEM | 0.394 | 0.357 | 0.216 | 0.385 | 0.348 |
| MemGPT | 0.404 | 0.364 | 0.316 | 0.385 | 0.355 |
| SimpleMem | 0.432 | 0.404 | 0.342 | 0.418 | 0.388 |
| **Omni-SimpleMem** | **0.598** | **0.519** | **0.492** | **0.613** | **0.522** |

</details>

<details open>
<summary><b>Mem-Gallery — Overall F1</b></summary>

| Method | GPT-4o | GPT-4o-mini | GPT-4.1-nano | GPT-5.1 | GPT-5-nano |
|:--|:--:|:--:|:--:|:--:|:--:|
| Mem0 | 0.298 | 0.291 | 0.268 | 0.270 | 0.283 |
| A-MEM | 0.370 | 0.330 | 0.365 | 0.408 | 0.505 |
| MemGPT | 0.435 | 0.398 | 0.360 | 0.425 | 0.388 |
| SimpleMem | 0.535 | 0.498 | 0.518 | 0.538 | 0.522 |
| **Omni-SimpleMem** | **0.797** | **0.749** | **0.780** | **0.810** | **0.787** |

</details>

### Efficiency

<p align="center">
  <img src="assets/efficiency.png" width="80%" alt="Efficiency Comparison">
</p>

---

## 🔬 Benchmarks

Omni-SimpleMem is evaluated on two benchmarks:
- [**LoCoMo**](https://github.com/snap-research/locomo) — 1,986 QA pairs across 10 conversations and 5 categories (Multi-hop, Single-hop, Temporal, Open-domain, Adversarial)
- [**Mem-Gallery**](https://github.com/Mem-Gallery/Mem-Gallery) — 1,711 QA pairs from 240 dialogues across 9 multimodal categories

### Reproducing LoCoMo Results

```bash
# 1. Download dataset
git clone https://github.com/snap-research/locomo.git

# 2. Set API key
export OPENAI_API_KEY="your-openai-api-key"

# 3. Run benchmark
python benchmarks/locomo/run_locomo.py \
    --data-path /path/to/locomo/data/locomo10.json \
    --model gpt-4o -o ./locomo_results

# Quick test (1 conversation, 20 QA pairs)
python benchmarks/locomo/run_locomo.py \
    --data-path /path/to/locomo/data/locomo10.json \
    --max-conversations 1 --max-qa 20
```

---

## ⚙️ Configuration

<details>
<summary>Full configuration reference</summary>

```python
from omni_memory import OmniMemoryConfig

config = OmniMemoryConfig()

# Embedding
config.embedding.model_name = "all-MiniLM-L6-v2"  # Local (384d)
config.embedding.embedding_dim = 384

# LLM
config.llm.summary_model = "gpt-4o-mini"
config.llm.query_model = "gpt-4o"
config.llm.temperature = 0.0

# Retrieval
config.retrieval.default_top_k = 20
config.retrieval.enable_hybrid_search = True
config.retrieval.enable_graph_traversal = True

# Storage
config.storage.base_dir = "./my_memory_data"

# Unified model shortcut
config.set_unified_model("gpt-4o")
```

</details>

See [`configs/`](configs/) for benchmark-specific YAML configurations.

---

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📁 Package Structure

<details>
<summary>Click to expand</summary>

```
OmniSimpleMem/
├── omni_memory/               # Core package
│   ├── orchestrator.py        # Central coordinator
│   ├── app.py                 # FastAPI REST server
│   ├── core/                  # MAU, config, events
│   ├── storage/               # FAISS vector store, cold storage, dedup
│   ├── retrieval/             # Pyramid retriever, BM25, query processor
│   ├── processors/            # Text, image, audio, video processors
│   ├── triggers/              # CLIP visual & VAD audio triggers
│   ├── knowledge/             # Knowledge graph & entity extraction
│   ├── graph/                 # Event management
│   ├── parametric/            # Memory consolidation
│   ├── routing/               # Query routing
│   ├── evolution/             # Self-evolution
│   └── utils/                 # Embedding, model utilities, logging
├── configs/                   # Benchmark YAML configs
├── benchmarks/                # LoCoMo & Mem-Gallery adapters
├── tests/                     # 126 unit tests
├── examples/                  # Usage examples
├── setup.py
├── requirements.txt
└── LICENSE                    # Apache 2.0
```

</details>

---

## 🔧 Implementation Details

| | Component | Details |
|:--|:--|:--|
| 🔎 | Dense retrieval | FAISS — all-MiniLM-L6-v2 (384d) or text-embedding-3-large (3072d) |
| 🔤 | Sparse retrieval | BM25 via `rank_bm25` |
| 🖼️ | Visual embeddings | OpenVision CLIP ViT-L/14 (768d) |
| 🕸️ | Knowledge graph | In-memory, 7 entity types, 12 relation types |
| ⚡ | Concurrency | Thread-safe (`RLock`), parallel workers |
| 💾 | Storage | JSONL MAU persistence, FAISS index serialization |

---

## 🙏 Acknowledgments

Omni-SimpleMem's architecture was discovered through [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw), an autonomous 23-stage research pipeline. The most impactful discoveries were bug fixes (+175%), prompt engineering (+188%), and architectural changes (+44%) — each exceeding all hyperparameter tuning combined.

---

## 📄 License

<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="Apache 2.0"></a>

---

## 📌 Citation

```bibtex
@article{omnisimplemem2026,
  title   = {{Omni-SimpleMem}: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory},
  author  = {Liu, Jiaqi and Ling, Zipeng and Qiu, Shi and Liu, Yanqing and Han, Siwei and Xia, Peng and Tu, Haoqin and Zheng, Zeyu and Xie, Cihang and Fleming, Charles and Ding, Mingyu and Yao, Huaxiu},
  journal = {arXiv preprint arXiv:2604.01007},
  year    = {2026},
}
```

---

<p align="center">
  <sub>Part of the <a href="https://github.com/aiming-lab/SimpleMem"><b>SimpleMem</b></a> family &nbsp;·&nbsp; Architecture discovered by <a href="https://github.com/aiming-lab/AutoResearchClaw">AutoResearchClaw</a></sub>
</p>
