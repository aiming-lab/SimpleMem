<div align="center">

<img alt="simplemem_logo" src="https://github.com/user-attachments/assets/6ea54ad1-e007-442c-99d7-1174b10d1fec" width="450">

<div align="center">

## Efficient Lifelong Memory for LLM Agents — Text & Multimodal

<small>Store, compress, and retrieve long-term memories with semantic lossless compression. Now with multimodal support for text, image, audio & video. Works across Claude, Cursor, LM Studio, and more.</small>

</div>

<p><b>Works with any AI platform that supports MCP or Python integration</b></p>

<table>
<tr>

<td align="center" width="100">
  <a href="https://www.anthropic.com/claude">
    <img src="https://cdn.simpleicons.org/claude/D97757" width="48" height="48" alt="Claude Desktop" />
  </a><br/>
  <sub>
    <a href="https://www.anthropic.com/claude"><b>Claude Desktop</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://cursor.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/cursor/FFFFFF">
      <img src="https://cdn.simpleicons.org/cursor/000000" width="48" height="48" alt="Cursor" />
    </picture>
  </a><br/>
  <sub>
    <a href="https://cursor.com"><b>Cursor</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://lmstudio.ai">
    <img src="https://github.com/lmstudio-ai.png?size=200" width="48" height="48" alt="LM Studio" />
  </a><br/>
  <sub>
    <a href="https://lmstudio.ai"><b>LM Studio</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://cherry-ai.com">
    <img src="https://github.com/CherryHQ.png?size=200" width="48" height="48" alt="Cherry Studio" />
  </a><br/>
  <sub>
    <a href="https://cherry-ai.com"><b>Cherry Studio</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://pypi.org/project/simplemem/">
    <img src="https://cdn.simpleicons.org/pypi/3775A9" width="48" height="48" alt="PyPI" />
  </a><br/>
  <sub>
    <a href="https://pypi.org/project/simplemem/"><b>PyPI Package</b></a>
  </sub>
</td>

<td align="center" width="100">
  <sub><b>+ Any MCP<br/>Client</b></sub>
</td>

</tr>
</table>

<div align="center">

<br/>

[🇨🇳 中文](./docs/i18n/README.zh-CN.md) •
[🇯🇵 日本語](./docs/i18n/README.ja.md) •
[🇰🇷 한국어](./docs/i18n/README.ko.md) •
[🇪🇸 Español](./docs/i18n/README.es.md) •
[🇫🇷 Français](./docs/i18n/README.fr.md) •
[🇩🇪 Deutsch](./docs/i18n/README.de.md) •
[🇧🇷 Português](./docs/i18n/README.pt-br.md)<br/>
[🇷🇺 Русский](./docs/i18n/README.ru.md) •
[🇸🇦 العربية](./docs/i18n/README.ar.md) •
[🇮🇹 Italiano](./docs/i18n/README.it.md) •
[🇻🇳 Tiếng Việt](./docs/i18n/README.vi.md) •
[🇹🇷 Türkçe](./docs/i18n/README.tr.md)

<br/>

[![Project Page](https://img.shields.io/badge/🎬_INTERACTIVE_DEMO-Visit_Our_Website-FF6B6B?style=for-the-badge&labelColor=FF6B6B&color=4ECDC4&logoColor=white)](https://aiming-lab.github.io/SimpleMem-Page)

<p align="center">
  <a href="https://arxiv.org/abs/2601.02553"><img src="https://img.shields.io/badge/arXiv-2601.02553-b31b1b?style=flat&labelColor=555" alt="arXiv"></a>
  <a href="https://github.com/aiming-lab/SimpleMem"><img src="https://img.shields.io/badge/github-SimpleMem-181717?style=flat&labelColor=555&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/aiming-lab/SimpleMem?style=flat&label=license&labelColor=555&color=2EA44F" alt="License"></a>
  <a href="https://github.com/aiming-lab/SimpleMem/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat&labelColor=555" alt="PRs Welcome"></a>
  <br/>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/v/simplemem?style=flat&label=pypi&labelColor=555&color=3775A9&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/pyversions/simplemem?style=flat&label=python&labelColor=555&color=3775A9&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://mcp.simplemem.cloud"><img src="https://img.shields.io/badge/MCP-mcp.simplemem.cloud-14B8A6?style=flat&labelColor=555" alt="MCP Server"></a>
  <a href="https://github.com/aiming-lab/SimpleMem"><img src="https://img.shields.io/badge/Claude_Skills-supported-FFB000?style=flat&labelColor=555" alt="Claude Skills"></a>
  <br/>
  <a href="https://discord.gg/KA2zC32M"><img src="https://img.shields.io/badge/Discord-Join_Chat-5865F2?style=flat&labelColor=555&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="fig/wechat_logo3.JPG"><img src="https://img.shields.io/badge/WeChat-Group-07C160?style=flat&labelColor=555&logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

<br/>

[🚀 Quick Start](#-quick-start) • [🌟 Overview](#-overview) • [📈 Results](#-results) • [🧠 Omni-SimpleMem](#-omni-simplemem-multimodal-memory) • [📦 Installation](#-installation) • [🔄 Cross-Session Memory](#-cross-session-memory-text-memory) • [🔌 MCP Server](#-mcp-server-text-memory) • [📝 Citation](#-citation)

</div>

</div>

<br/>

## 🔥 News

- **[04/02/2026]** 🧠 **Omni-SimpleMem — Multimodal Memory is Here!** SimpleMem now supports **text, image, audio & video** memory. Achieving **new SOTA on LoCoMo (F1=0.613, +47%)** and **Mem-Gallery (F1=0.810, +51%)** over previous best, Omni-SimpleMem brings state-of-the-art multimodal lifelong memory to your agents. [View Omni-SimpleMem →](OmniSimpleMem/)
- **[02/09/2026]** 🚀 **Cross-Session Memory is Here — Outperforming Claude-Mem by 64%!** SimpleMem now supports **persistent memory across conversations**. On the LoCoMo benchmark, SimpleMem achieves a **64% performance boost** over Claude-Mem. Your agents can now recall context, decisions, and learnings from previous sessions automatically. [View Cross-Session Documentation →](cross/README.md)
- **[01/20/2026]** **SimpleMem is now available on PyPI!** 📦 Install directly via `pip install simplemem`. [View Package Usage Guide →](docs/PACKAGE_USAGE.md)
- **[01/19/2026]** **Added Local Memory Storage for SimpleMem Skill!** 💾 SimpleMem Skill now supports local memory storage within Claude Skills.
- **[01/18/2026]** **SimpleMem now supports Claude Skills!** 🚀 Use SimpleMem in claude.ai for long-term memory across conversations. Register at [mcp.simplemem.cloud](https://mcp.simplemem.cloud), configure your token, and import the skill!
- **[01/14/2026]** **SimpleMem MCP Server is now LIVE and Open Source!** 🎉 Cloud-hosted memory service at [mcp.simplemem.cloud](https://mcp.simplemem.cloud). Integrates with LM Studio, Cherry Studio, Cursor, Claude Desktop via **Streamable HTTP** MCP protocol. [View MCP Documentation →](MCP/README.md)
- **[01/08/2026]** 🔥 Join our [Discord](https://discord.gg/KA2zC32M) and [WeChat Group](fig/wechat_logo3.JPG) to collaborate and exchange ideas!
- **[01/05/2026]** SimpleMem paper was released on [arXiv](https://arxiv.org/abs/2601.02553)!

---

## 📑 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🌟 Overview](#-overview)
- [📈 Results](#-results)
- [📝 SimpleMem: Text Memory](#-simplemem-text-memory)
- [🧠 Omni-SimpleMem: Multimodal Memory](#-omni-simplemem-multimodal-memory)
- [📦 Installation](#-installation)
- [🗄️ Using InterSystems IRIS as the Vector Backend](#️-using-intersystems-iris-as-the-vector-backend)
- [🐳 Docker](#-run-with-docker)
- [🔌 Router Utilities](#-router-utilities)
- [🔄 Cross-Session Memory](#-cross-session-memory-text-memory)
- [🤖 Using SimpleMem with Claude + IRIS](#-using-simplemem-with-claude--iris)
- [🔧 Adapting SimpleMem for Your IRIS Data](#-adapting-simplemem-for-your-iris-data)
- [⚠️ IRIS Backend — Known Gotchas](#️-iris-backend--known-gotchas)
- [🔌 MCP Server](#-mcp-server-text-memory)
- [🗺️ Roadmap](#️-roadmap)
- [📊 Evaluation](#-evaluation)
- [📝 Citation](#-citation)

---

## 🚀 Quick Start

### 🧠 Understanding the Basic Workflow

At a high level, SimpleMem works as a long-term memory system for LLM-based agents. The workflow consists of three simple steps:

1. **Store information** – Dialogues or facts are processed and converted into structured, atomic memories.
2. **Index memory** – Stored memories are organized using semantic embeddings and structured metadata.
3. **Retrieve relevant memory** – When a query is made, SimpleMem retrieves the most relevant stored information based on meaning rather than keywords.

This design allows LLM agents to maintain context, recall past information efficiently, and avoid repeatedly processing redundant history.

### 🎓 Basic Usage

SimpleMem provides a **unified entry point** via `simplemem_router`. The default `mode="auto"` **automatically detects** which backend to use based on what you call — no manual configuration needed:

```python
import simplemem_router as simplemem

mem = simplemem.create()  # mode="auto" — backend chosen by first call
```

The first method you call determines the backend:

| First call | Backend selected | Why |
|:--|:--|:--|
| `add_dialogue()` | **Text** (SimpleMem) | Dialogue-based API → text mode |
| `add_text()` / `add_image()` / `add_audio()` / `add_video()` | **Omni** (Omni-SimpleMem) | Multimodal API → omni mode |

<table>
<tr>
<td width="50%">

**📝 Auto → Text** (pure text input)

```python
import simplemem_router as simplemem

mem = simplemem.create()  # auto mode

# add_dialogue() → text backend auto-selected
mem.add_dialogue(
    "Alice",
    "Bob, let's meet at Starbucks tomorrow at 2pm",
    "2025-11-15T14:30:00",
)
mem.add_dialogue(
    "Bob",
    "Sure, I'll bring the market analysis report",
    "2025-11-15T14:31:00",
)
mem.finalize()

answer = mem.ask("When and where will Alice and Bob meet?")
# → "16 November 2025 at 2:00 PM at Starbucks"
```

</td>
<td width="50%">

**🧠 Auto → Omni** (multimodal input)

```python
import simplemem_router as simplemem

mem = simplemem.create()  # auto mode

# add_image() → omni backend auto-selected
mem.add_text(
    "User loves hiking in the Rocky Mountains.",
    tags=["session_id:D1"],
)
mem.add_image("photo.jpg", tags=["session_id:D1"])
mem.add_audio("voice_note.wav", tags=["session_id:D1"])

result = mem.query("What does the user enjoy?", top_k=5)
for item in result.items:
    print(item["summary"])

mem.close()
```

</td>
</tr>
</table>

> **💡 Tip**: Auto mode picks the lightest backend that fits your data. You can still use `mode="text"` or `mode="omni"` explicitly if you prefer.

---

### 🚄 Advanced: Parallel Processing

For large-scale dialogue processing, enable parallel mode:

```python
import simplemem_router as simplemem

mem = simplemem.create(
    mode="text",
    clear_db=True,
    enable_parallel_processing=True,  # ⚡ Parallel memory building
    max_parallel_workers=8,
    enable_parallel_retrieval=True,   # 🔍 Parallel query execution
    max_retrieval_workers=4
)
```

> **💡 Pro Tip**: Parallel processing significantly reduces latency for batch operations!

---

## 🌟 Overview

**SimpleMem** is a family of efficient memory frameworks — **SimpleMem** for text and **Omni-SimpleMem** for multimodal (text, image, audio, video) — based on **semantic lossless compression** that addresses the fundamental challenge of **efficient long-term memory for LLM agents**. Unlike existing systems that either passively accumulate redundant context or rely on expensive iterative reasoning loops, SimpleMem maximizes **information density** and **token utilization** through a three-stage pipeline:

<table>
<tr>
<td width="33%" align="center">

### 🔍 Stage 1
**Semantic Structured Compression**

Distills unstructured interactions into compact, multi-view indexed memory units

</td>
<td width="33%" align="center">

### 🗂️ Stage 2
**Online Semantic Synthesis**

Intra-session process that instantly integrates related context into unified abstract representations to eliminate redundancy

</td>
<td width="33%" align="center">

### 🎯 Stage 3
**Intent-Aware Retrieval Planning**

Infers search intent to dynamically determine retrieval scope and construct precise context efficiently

</td>
</tr>
</table>

> For multimodal memory, see [Omni-SimpleMem](#-omni-simplemem-multimodal-memory) below.

<div align="center">
<img src="fig/Fig_framework.png" alt="SimpleMem Framework" width="900"/>

*The SimpleMem Architecture: (1) Semantic Structured Compression filters low-utility dialogue and converts informative windows into compact, context-independent memory units. (2) Online Semantic Synthesis consolidates related fragments during writing, maintaining a compact and coherent memory topology. (3) Intent-Aware Retrieval Planning infers search intent to adapt retrieval scope and query forms, enabling parallel multi-view retrieval and token-efficient context construction.*
</div>

---

### 🏆 Performance Comparison

<div align="center">

<img src="fig/Fig_tradeoff.png" alt="Performance vs Efficiency Trade-off" width="900"/>

*SimpleMem achieves superior F1 score (43.24%) with minimal token cost (~550), occupying the ideal top-left position.*

**Speed Comparison Demo**

<video src="https://github.com/aiming-lab/SimpleMem/raw/main/fig/simplemem-new.mp4" controls width="900"></video>

*SimpleMem vs. Baseline: Real-time speed comparison demonstration*

</div>

<div align="center">

**LoCoMo-10 Benchmark Results (GPT-4.1-mini)**

| Model | ⏱️ Construction Time | 🔎 Retrieval Time | ⚡ Total Time | 🎯 Average F1 |
|:------|:--------------------:|:-----------------:|:-------------:|:-------------:|
| A-Mem | 5140.5s | 796.7s | 5937.2s | 32.58% |
| LightMem | 97.8s | 577.1s | 675.9s | 24.63% |
| Mem0 | 1350.9s | 583.4s | 1934.3s | 34.20% |
| **SimpleMem** ⭐ | **92.6s** | **388.3s** | **480.9s** | **43.24%** |

</div>

---

## 📈 Results

### 📊 Benchmark Results (LoCoMo)

<details open>
<summary><b>🏆 Cross-Session Memory Comparison</b></summary>

| System | LoCoMo Score | vs SimpleMem |
|:-------|:------------:|:------------:|
| **SimpleMem** | **48** | — |
| Claude-Mem | 29.3 | **+64%** |

</details>

<details>
<summary><b>🔬 High-Capability Models (GPT-4.1-mini)</b></summary>

| Task Type | SimpleMem F1 | Mem0 F1 | Improvement |
|:----------|:------------:|:-------:|:-----------:|
| **MultiHop** | 43.46% | 30.14% | **+43.8%** |
| **Temporal** | 58.62% | 48.91% | **+19.9%** |
| **SingleHop** | 51.12% | 41.3% | **+23.8%** |

</details>

<details>
<summary><b>⚙️ Efficient Models (Qwen2.5-1.5B)</b></summary>

| Metric | SimpleMem | Mem0 | Notes |
|:-------|:---------:|:----:|:------|
| **Average F1** | 25.23% | 23.77% | Competitive with 99× smaller model |

</details>

### 🧠 Omni-SimpleMem Results

<table>
<tr>
<td align="center" width="170">🏆 <b>0.613 F1</b><br><sub>LoCoMo (+47% over prev. SOTA)</sub></td>
<td align="center" width="170">🏆 <b>0.810 F1</b><br><sub>Mem-Gallery (+51% over prev. SOTA)</sub></td>
<td align="center" width="140">⚡ <b>3.5x faster</b><br><sub>retrieval throughput</sub></td>
<td align="center" width="140">🧠 <b>4 modalities</b><br><sub>Text · Image · Audio · Video</sub></td>
</tr>
</table>

---

## 📝 SimpleMem: Text Memory

### 1️⃣ Semantic Structured Compression

SimpleMem applies an **implicit semantic density gating** mechanism integrated into the LLM generation process to filter redundant interaction content. The system reformulates raw dialogue streams into **compact memory units** — self-contained facts with resolved coreferences and absolute timestamps. Each unit is indexed through three complementary representations for flexible retrieval:

<div align="center">

| 🔍 Layer | 📊 Type | 🎯 Purpose | 🛠️ Implementation |
|---------|---------|------------|-------------------|
| **Semantic** | Dense | Conceptual similarity | Vector embeddings (1024-d) |
| **Lexical** | Sparse | Exact term matching | BM25-style keyword index |
| **Symbolic** | Metadata | Structured filtering | Timestamps, entities, persons |

</div>

**✨ Example Transformation:**
```diff
- Input:  "He'll meet Bob tomorrow at 2pm"  [❌ relative, ambiguous]
+ Output: "Alice will meet Bob at Starbucks on 2025-11-16T14:00:00"  [✅ absolute, atomic]
```

---

### 2️⃣ Online Semantic Synthesis

Unlike traditional systems that rely on asynchronous background maintenance, SimpleMem performs synthesis **on-the-fly during the write phase**. Related memory units are synthesized into higher-level abstract representations within the current session scope, allowing repetitive or structurally similar experiences to be **denoised and compressed immediately**.

**✨ Example Synthesis:**
```diff
- Fragment 1: "User wants coffee"
- Fragment 2: "User prefers oat milk"
- Fragment 3: "User likes it hot"
+ Consolidated: "User prefers hot coffee with oat milk"
```

This proactive synthesis ensures the memory topology remains compact and free of redundant fragmentation.

---

### 3️⃣ Intent-Aware Retrieval Planning

Instead of fixed-depth retrieval, SimpleMem leverages the reasoning capabilities of the LLM to generate a **comprehensive retrieval plan**. Given a query, the planning module infers **latent search intent** to dynamically determine retrieval scope and depth:

$$\{ q_{\text{sem}}, q_{\text{lex}}, q_{\text{sym}}, d \} \sim \mathcal{P}(q, H)$$

The system then executes **parallel multi-view retrieval** across semantic, lexical, and symbolic indexes, and merges results through ID-based deduplication:

<table>
<tr>
<td width="50%">

**🔹 Simple Queries**
- Direct fact lookup via single memory unit
- Minimal retrieval depth
- Fast response time

</td>
<td width="50%">

**🔸 Complex Queries**
- Aggregation across multiple events
- Expanded retrieval depth
- Comprehensive coverage

</td>
</tr>
</table>

**📈 Result**: 43.24% F1 score with **30× fewer tokens** than full-context methods.

---

<div align="center">

# 🧠 Omni-SimpleMem: Multimodal Memory

**NEW** — SimpleMem now handles text, image, audio & video.

</div>

**Omni-SimpleMem** extends SimpleMem to **unified multimodal memory** — supporting text, image, audio, and video experiences with state-of-the-art accuracy across all five LLM backbones tested.

Built on three principles: **Selective Ingestion** (entropy-driven filtering for each modality), **Progressive Retrieval** (hybrid FAISS + BM25 search with pyramid token-budget expansion), and **Knowledge Graph Augmentation** (multi-hop cross-modal reasoning).

> 📖 Full documentation, benchmarks, and architecture details: [**Omni-SimpleMem →**](OmniSimpleMem/)

---

## 📦 Installation

### 📝 Notes for First-Time Users

- Ensure you are using **Python 3.10 in your active environment**, not just installed globally.
- An OpenAI-compatible API key must be configured **before running any memory construction or retrieval**, otherwise initialization may fail.
- When using non-OpenAI providers (e.g., Qwen or Azure OpenAI), verify both the model name and `OPENAI_BASE_URL` in `config.py`.
- For large dialogue datasets, enabling parallel processing can significantly reduce memory construction time.

### 📋 Requirements

- 🐍 Python 3.10
- 🔑 OpenAI-compatible API (OpenAI, Qwen, Azure OpenAI, etc.)

### 🛠️ Setup

```bash
# 📥 Clone repository
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem

# 📦 Install dependencies
pip install -r requirements.txt

# ⚙️ Configure API settings
cp config.py.example config.py
# Edit config.py with your API key and preferences
```

### ⚙️ Configuration Example

```python
# config.py
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = None  # or custom endpoint for Qwen/Azure

LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # State-of-the-art retrieval
```

---

## 🗄️ Using InterSystems IRIS as the Vector Backend

This fork replaces LanceDB with **[InterSystems IRIS](https://www.intersystems.com/products/intersystems-iris/)** as the vector store, using IRIS's native `VECTOR` SQL type and `HNSW` approximate nearest-neighbor index. All three retrieval layers (semantic, keyword, structured) run as pure SQL — no external vector database process required.

> **Why IRIS?** In head-to-head benchmarks on a real agent memory workload, this IRIS + HNSW implementation matches or outperforms LanceDB + IVF above ~5k entries per user, delivers ~4× faster keyword search (native `$FIND` vs Tantivy), and adds full ACID transactions, SQL joins across memory and application data, and zero additional infrastructure when IRIS is already in your stack. See [benchmark results](#benchmark-iris-vs-lancedb) below.

### Prerequisites

- InterSystems IRIS 2025.1 or later (includes `VECTOR` type and `HNSW` index)
- Python package: `intersystems-irispython`

```bash
pip install intersystems-irispython
```

**Get IRIS free:** [InterSystems IRIS Community Edition](https://www.intersystems.com/try-intersystems-iris-for-free/) — available as a Docker image or installer.

```bash
docker pull intersystemsdc/iris-community
docker run -d --name iris -p 1972:1972 -p 52773:52773 intersystemsdc/iris-community
```

### Configuration

```python
# config.py — IRIS connection settings
IRIS_HOSTNAME  = "localhost"
IRIS_PORT      = 1972          # default superserver port
IRIS_NAMESPACE = "USER"
IRIS_USERNAME  = "_SYSTEM"
IRIS_PASSWORD  = "SYS"

MEMORY_TABLE_NAME = "memory_entries"
```

For IRIS Cloud or a managed instance, set `IRIS_HOSTNAME` to your endpoint and update credentials accordingly.

### What the backend does automatically

On first use, `VectorStore` and `CrossSessionVectorStore` each run:

```sql
CREATE TABLE memory_entries (
    entry_id  VARCHAR(64),
    text      VARCHAR(32000),
    ...       -- keyword, timestamp, location, persons, entities, topic
    vec       VECTOR(DOUBLE, 1024)
)

CREATE INDEX HNSWIdx ON TABLE memory_entries (vec)
  AS HNSW(Distance='Cosine', M=16, efConstruction=64)
```

No manual schema setup needed. The HNSW index is created idempotently — safe to restart.

### Benchmark: IRIS vs LanceDB

All measurements on localhost, Apple M-series, Qwen3-Embedding-0.6B (1024-d vectors):

**Vector search latency (1024-d, TOP 10):**

| Corpus size | LanceDB plain | LanceDB + IVF | IRIS plain | IRIS + HNSW |
|-------------|--------------|---------------|------------|-------------|
| 500 entries | 2.5ms | 2.4ms | 2.3ms | 2.5ms |
| 2,000 | 3.4ms | 3.3ms | 6.5ms | **3.9ms** |
| 5,000 | 4.2ms | 4.1ms | 16ms | **4.7ms** |
| 10,000 | 8.0ms | 7.8ms | 31ms | **5.5ms** |

**Keyword and metadata search (25 entries, Recall@10):**

| Search type | LanceDB | IRIS | Winner |
|-------------|---------|------|--------|
| Semantic | 1.000 | 1.000 | Tie |
| Keyword (multi-word) | 1.000* | 1.000 | Tie |
| Structured (persons) | 0.857 | 0.857 | Tie |
| Keyword latency | ~7ms | **~1ms** | IRIS |
| Structured latency | ~3ms | **~0.8ms** | IRIS |

*LanceDB keyword search requires `pylance` installed and `create_fts_index()` called after every data load. Without this, it silently returns zero results. IRIS `$FIND` works out of the box.

**Key finding:** An ISC customer implementing an IRIS Vector Search solution found that SimpleMem's SQL-first approach — combining `VECTOR_COSINE` semantic search with `$FIND` keyword scoring and structured SQL filters in a single connection — **outperformed their standalone IRIS Vector Search implementation**, which relied on a separate vector index layer without the hybrid retrieval pipeline.

---

## 🐳 Run with Docker

The **MCP Server** can be run in Docker for a consistent, isolated environment. Data is persisted in a host volume.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

### Quick run

```bash
# From the repository root
docker compose up -d
```

- **Web UI:** http://localhost:8000/
- **REST API:** http://localhost:8000/api/
- **MCP (SSE):** http://localhost:8000/mcp/sse?token=&lt;TOKEN&gt;

Data is stored in `./data` on the host (created automatically).

### Custom configuration

1. Copy the environment template and edit it:
   ```bash
   cp .env.example .env
   # Edit .env: set JWT_SECRET_KEY, ENCRYPTION_KEY, LLM_PROVIDER, model URLs, etc.
   ```
2. Run with the env file:
   ```bash
   docker compose --env-file .env up -d
   ```

### Using Ollama on the host

When `LLM_PROVIDER=ollama` and Ollama runs on your machine (not in Docker), set in `.env`:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
```

On Linux, `host.docker.internal` is enabled automatically via the Compose file.

### Useful commands

```bash
docker compose logs -f simplemem   # Follow logs
docker compose down                 # Stop and remove containers
```

> 📖 For self-hosting the MCP server (Docker or bare metal), see [MCP Documentation](MCP/README.md).

---

## 🔌 Router Utilities

The router uses a **registry-based factory** pattern — backends are lazily loaded only when requested, and dependencies are checked before instantiation.

```python
import simplemem_router as simplemem

# List all registered modes
simplemem.list_modes()
# {'text': 'Single-modal text memory with semantic lossless compression',
#  'omni': 'Multimodal memory — text, image, audio, video (Omni-SimpleMem)'}

# Check if a mode's dependencies are satisfied
simplemem.is_available("omni")  # True / False

# Check which mode was auto-selected
mem = simplemem.create()
print(mem.mode)  # "auto" (pending), "text", or "omni"

# Register a custom backend
simplemem.register(
    mode="my_backend",
    module_path="my_package.memory",
    class_name="MyMemorySystem",
    description="Custom memory backend",
    required_deps=["my_package"],
)
mem = simplemem.create(mode="my_backend")
```

---

## ❓ Common Setup Issues & Troubleshooting

If you encounter issues while setting up or running SimpleMem for the first time, check the following common cases:

### 1️⃣ API Key Not Detected
- Ensure your API key is correctly set in `config.py`
- For OpenAI-compatible providers (Qwen, Azure, etc.), verify that `OPENAI_BASE_URL` is configured correctly
- Restart your Python environment after updating the key

### 2️⃣ Python Version Mismatch
- SimpleMem requires **Python 3.10**
- Check your version using:
  ```bash
  python --version
  ```

---

## 🔄 Cross-Session Memory *(text memory)*

**SimpleMem-Cross** extends SimpleMem with persistent cross-conversation memory capabilities. Agents can recall context, decisions, and observations from previous sessions — enabling continuity across conversations without manual context re-injection.

### Key Features

| Feature | Description |
|---------|-------------|
| **Session Lifecycle** | Full session management with start/record/stop/end lifecycle |
| **Automatic Context Injection** | Token-budgeted context from previous sessions injected at session start |
| **Event Collection** | Record messages, tool uses, file changes with automatic redaction |
| **Observation Extraction** | Heuristic extraction of decisions, discoveries, and learnings |
| **Provenance Tracking** | Every memory entry links back to source evidence |
| **Consolidation** | Decay, merge, and prune old memories to maintain quality |

### Quick Example

```python
from cross.orchestrator import create_orchestrator

async def main():
    orch = create_orchestrator(project="my-project")

    # Start session — previous context is injected automatically
    result = await orch.start_session(
        content_session_id="session-001",
        user_prompt="Continue building the REST API",
    )
    print(result["context"])  # Relevant context from previous sessions

    # Record events during the session
    await orch.record_message(result["memory_session_id"], "User asked about JWT")
    await orch.record_tool_use(
        result["memory_session_id"],
        tool_name="read_file",
        tool_input="auth/jwt.py",
        tool_output="class JWTHandler: ...",
    )

    # Finalize — extracts observations, generates summary, stores memories
    report = await orch.stop_session(result["memory_session_id"])
    print(f"Stored {report.entries_stored} memory entries")

    await orch.end_session(result["memory_session_id"])
    orch.close()
```

### Architecture

```
Agent Frameworks (Claude Code / Cursor / custom)
                    |
     +--------------+--------------+
     |                             |
Hook/Lifecycle Adapter      HTTP/MCP API (FastAPI)
     |                             |
     +--------------+--------------+
                    |
           CrossMemOrchestrator
                    |
  +-----------------+------------------+
  |                 |                  |
Session Manager  Context Injector  Consolidation
(SQLite)         (budgeted bundle) (decay/merge/prune)
  |                 |                  |
  +---------+-------+                  |
            |                          |
   Cross-Session Vector Store (LanceDB) <--+
```

### Module Reference

| Module | Description |
|--------|-------------|
| `cross/types.py` | Pydantic models, enums, records |
| `cross/storage_sqlite.py` | SQLite backend for sessions, events, observations (default) |
| `cross/storage_iris_sql.py` | IRIS SQL backend — same interface as SQLiteStorage, zero SQLite dependency |
| `cross/storage_factory.py` | `create_sql_storage(use_iris=False)` — selects backend |
| `cross/storage_iris.py` | IRIS vector store for memory entries (HNSW index) |
| `cross/hooks.py` | Lifecycle hooks (SessionStart/ToolUse/End) |
| `cross/collectors.py` | Event collection with 3-tier redaction |
| `cross/session_manager.py` | Full session lifecycle orchestration |
| `cross/context_injector.py` | Token-budgeted context builder |
| `cross/orchestrator.py` | Top-level facade and factory |
| `cross/api_http.py` | FastAPI REST endpoints |
| `cross/api_mcp.py` | MCP tool definitions |
| `cross/consolidation.py` | Memory maintenance worker |

> 📖 For detailed API documentation, see [Cross-Session README](cross/README.md)

---

## 🤖 Using SimpleMem with Claude + IRIS

This fork is particularly well-suited for Claude deployments that already run InterSystems IRIS — a common configuration in healthcare, enterprise, and ISC customer environments.

### Direct Python integration

```python
from main import SimpleMemSystem

mem = SimpleMemSystem()

# Store conversation turns
mem.add_dialogue([
    {"role": "user",    "content": "Schedule a demo with Acme Corp next Tuesday at 2pm"},
    {"role": "assistant","content": "Done — I've noted the Acme Corp demo for Tuesday at 2pm."},
])

# Retrieve on next session
answer = mem.ask("When is the Acme Corp demo?")
print(answer)  # "Tuesday at 2pm"
```

### Cross-session memory: SQLite (default) or full IRIS

By default, cross-session metadata (sessions, events, observations, summaries) is stored in a local SQLite file. For ISC customers who want **everything in IRIS** — no SQLite dependency, single namespace, single backup:

```python
from cross.orchestrator import create_orchestrator

# Default: SQLite for metadata, IRIS for vectors
orch = create_orchestrator("my-project")

# Full IRIS: metadata AND vectors both in IRIS
orch = create_orchestrator("my-project", use_iris_sql=True)
```

**Why full IRIS mode matters:** when `use_iris_sql=True`, all six metadata tables (`CrossMem_sessions`, `CrossMem_observations`, `CrossMem_session_summaries`, etc.) live in the same IRIS namespace as the vector store (`cross_memory_entries`). This enables SQL JOINs that are impossible when metadata is in SQLite:

```sql
-- Which memory entries came from sessions where the agent made a discovery?
SELECT m.text, m.timestamp, o.title AS discovery
FROM cross_memory_entries m
JOIN CrossMem_memory_links l   ON l.memory_entry_id = m.entry_id
JOIN CrossMem_observations o   ON o.obs_id = l.source_id
JOIN CrossMem_sessions s       ON s.memory_session_id = o.memory_session_id
WHERE o.type = 'discovery'
  AND s.project = 'my-project'
ORDER BY m.timestamp DESC

-- What did the agent learn in sessions that produced high-scoring memories?
SELECT s.started_at, s.user_prompt, ss.learned,
       AVG(l.score) AS avg_memory_score
FROM CrossMem_sessions s
JOIN CrossMem_session_summaries ss ON ss.memory_session_id = s.memory_session_id
JOIN CrossMem_memory_links l       ON l.memory_entry_id IN (
    SELECT entry_id FROM cross_memory_entries WHERE tenant_id = s.tenant_id
)
WHERE s.project = 'my-project'
GROUP BY s.id, s.started_at, s.user_prompt, ss.learned
ORDER BY avg_memory_score DESC
```

These queries are also joinable against **your existing application tables** — patient records, orders, tickets, or any other data in the same IRIS namespace.

### With Claude via MCP (self-hosted, IRIS backend)

1. Start the MCP server pointing at your IRIS instance:

```bash
# Set IRIS connection in .env
IRIS_HOSTNAME=your-iris-host
IRIS_PORT=1972
IRIS_USERNAME=_SYSTEM
IRIS_PASSWORD=SYS

cd MCP && python run.py
```

2. Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "simplemem": {
      "url": "http://localhost:8000/mcp",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```

3. Claude now has persistent memory across conversations, backed by IRIS.

### Why this matters for ISC customers

If you are building an AI agent solution on InterSystems IRIS, this architecture gives you:

- **Persistent agent memory** stored directly in your existing IRIS namespace — no separate vector database process
- **Hybrid retrieval** (semantic + keyword + structured) in a single SQL connection using `VECTOR_COSINE`, `$FIND`, and standard `WHERE` clauses
- **Full SQL access** to memory data alongside your clinical, operational, or transactional data — join memory entries against patient records, orders, or any IRIS table
- **Single namespace** (`use_iris_sql=True`) — vectors, session metadata, observations, and summaries all in one IRIS database, queryable together with your application data, no SQLite file to manage or back up
- **ACID transactions** — memory writes participate in your existing IRIS transactions
- **HNSW index** created automatically via `CREATE INDEX ... AS HNSW(Distance='Cosine')` — no manual schema setup

An ISC customer is currently running Claude with this SimpleMem + IRIS SQL backend and finds it outperforming a parallel IRIS Vector Search implementation they are also evaluating.

---

## 🔧 Adapting SimpleMem for Your IRIS Data

This section covers how to integrate SimpleMem with data you already have in IRIS, and how to understand the ingestion pipeline well enough to adapt it.

### How ingestion works (not chunking)

Most RAG systems split documents into fixed-size chunks and embed them directly. SimpleMem does not do this. Instead, `MemoryBuilder` sends each dialogue window to an LLM with a prompt that extracts structured `MemoryEntry` objects — each a fully self-contained, pronoun-resolved, timestamp-resolved fact. This is semantic compression, not chunking.

```
your text
   │
   ▼
MemoryBuilder (sliding window, LLM extraction)
   │
   ▼
[MemoryEntry, MemoryEntry, ...]   ← what gets embedded and stored
   │
   ▼
VectorStore (IRIS: VECTOR_COSINE + $FIND + SQL WHERE)
```

Each `MemoryEntry` has:
- `lossless_restatement` — the sentence that gets embedded (semantic layer)
- `keywords` — for `$FIND` keyword search (lexical layer)
- `persons`, `location`, `entities`, `timestamp` — for SQL filter search (symbolic layer)

### Pattern 1 — Ingest your existing IRIS table

If you have an existing table of text (notes, messages, documents), run it through the builder to populate SimpleMem's memory table:

```python
from main import SimpleMemSystem
from models.memory_entry import Dialogue
import intersystems_iris as ii

mem = SimpleMemSystem()

conn = ii.createConnection('localhost', 1972, 'USER', '_SYSTEM', 'SYS')
cur = conn.cursor()
cur.execute("SELECT id, note_text, created_at, author FROM MyApp.ClinicalNotes")

for row in cur.fetchall():
    dialogues = [Dialogue(
        dialogue_id=row[0],
        speaker=row[3] or "system",
        content=row[1],
        timestamp=row[2],
    )]
    mem.memory_builder._generate_memory_entries(dialogues)
    # or: mem.add_dialogue(dialogues) to go through the full pipeline

conn.close()
```

For large text (long documents, reports), split into passages first before creating `Dialogue` objects — the LLM context window is the limit, not a chunk size parameter. A passage of 500–1000 words per `Dialogue` is a reasonable target.

### Pattern 2 — Query your existing table alongside SimpleMem memory

Override `VectorStore` to search both your table and SimpleMem's memory table in one call. Your table needs a `VECTOR(DOUBLE, 1024)` column (add one with `ALTER TABLE`, populate it using your embedding model).

```python
from database.vector_store import VectorStore
from models.memory_entry import MemoryEntry
from typing import List, Optional
import json

class HybridCustomerStore(VectorStore):
    def __init__(self, customer_table: str, **kwargs):
        super().__init__(**kwargs)
        self._customer_table = customer_table

    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        simplemem_results = super().semantic_search(query, top_k)
        customer_results  = self._search_customer_table(query, top_k)
        seen = {e.entry_id for e in simplemem_results}
        merged = simplemem_results + [e for e in customer_results if e.entry_id not in seen]
        return merged[:top_k]

    def _search_customer_table(self, query: str, top_k: int) -> List[MemoryEntry]:
        qvec = self.embedding_model.encode_single(query, is_query=True)
        cur = self._cur()
        try:
            cur.execute(
                f"SELECT TOP {top_k} id, note_text, "
                f"VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE, {self._dim})) s "
                f"FROM {self._customer_table} ORDER BY s DESC",
                [json.dumps(qvec.tolist())]
            )
            return [
                MemoryEntry(entry_id=str(r[0]), lossless_restatement=r[1])
                for r in cur.fetchall()
            ]
        finally:
            cur.close()
```

Then wire it into the system:

```python
from main import SimpleMemSystem

mem = SimpleMemSystem()
mem.memory_builder.vector_store = HybridCustomerStore(
    customer_table="MyApp.ClinicalNotes",
    embedding_model=mem.memory_builder.vector_store.embedding_model,
    table_name="memory_entries",
)
mem.retriever.vector_store = mem.memory_builder.vector_store
```

### Pattern 3 — Use SimpleMem retrieval against a pure SQL table (no pipeline)

If you only want the hybrid retrieval layer (semantic + keyword + structured) against your own table, skip `MemoryBuilder` entirely and use `VectorStore` directly:

```python
from database.vector_store import VectorStore
from utils.embedding import EmbeddingModel

store = VectorStore(table_name="MyApp.AgentMemory")

# Add entries manually (bypassing LLM compression)
from models.memory_entry import MemoryEntry
store.add_entries([
    MemoryEntry(
        lossless_restatement="Patient John Smith has a penicillin allergy documented on 2025-03-14.",
        keywords=["penicillin", "allergy", "John Smith"],
        persons=["John Smith"],
        entities=["penicillin"],
        timestamp="2025-03-14T00:00:00",
    )
])

# Retrieve
results = store.semantic_search("does the patient have any drug allergies?")
results = store.keyword_search(["penicillin", "allergy"])
results = store.structured_search(persons=["John Smith"])
```

### Adding vector embeddings to an existing IRIS table

```sql
-- Add embedding column to your existing table
ALTER TABLE MyApp.ClinicalNotes ADD COLUMN embedding VECTOR(DOUBLE, 1024)

-- Add HNSW index for fast search
CREATE INDEX EmbeddingIdx ON TABLE MyApp.ClinicalNotes (embedding)
  AS HNSW(Distance='Cosine', M=16, efConstruction=64)
```

Then populate with Python:

```python
from utils.embedding import EmbeddingModel
import intersystems_iris as ii, json

emb = EmbeddingModel()
conn = ii.createConnection('localhost', 1972, 'USER', '_SYSTEM', 'SYS')
cur = conn.cursor()

cur.execute("SELECT id, note_text FROM MyApp.ClinicalNotes WHERE embedding IS NULL")
rows = cur.fetchall()

texts = [r[1] for r in rows]
vectors = emb.encode_documents(texts)

for (row_id, _), vec in zip(rows, vectors):
    cur.execute(
        f"UPDATE MyApp.ClinicalNotes SET embedding = TO_VECTOR(?, DOUBLE, 1024) WHERE id = ?",
        [json.dumps(vec.tolist()), row_id]
    )
conn.commit()
conn.close()
```

### Configuration reference for this fork

```python
# config.py — all settings relevant to the IRIS backend

# IRIS connection
IRIS_HOSTNAME  = "localhost"
IRIS_PORT      = 1972
IRIS_NAMESPACE = "USER"
IRIS_USERNAME  = "_SYSTEM"
IRIS_PASSWORD  = "SYS"

# Memory table (created automatically on first use)
MEMORY_TABLE_NAME = "memory_entries"

# Embedding model — must match dimension used when the table was created
# Changing this after data is loaded requires rebuilding the table
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # 1024-d

# Retrieval layer top-k values
SEMANTIC_TOP_K   = 5
KEYWORD_TOP_K    = 3
STRUCTURED_TOP_K = 5

# Memory builder sliding window
WINDOW_SIZE  = 10   # dialogues per LLM extraction call
OVERLAP_SIZE = 2    # dialogues carried forward for continuity context
```

> **Embedding dimension lock-in**: once a table is created with a given dimension, all vectors stored in it must match. If you change `EMBEDDING_MODEL` to a model with a different output dimension, drop and recreate the table. The HNSW index is recreated automatically.

---

## ⚠️ IRIS Backend — Known Gotchas

### Connection and threading

**Each thread gets its own IRIS connection** (thread-local). `VectorStore` and `CrossSessionVectorStore` both use `threading.local()` — a new connection is opened the first time any method is called from a given thread. This means:

- `enable_parallel_retrieval=True` **works and gives true DB parallelism** — each worker thread opens its own connection and queries execute concurrently. This is a significant difference from LanceDB, which used a single shared connection with a lock.
- `enable_parallel_processing=True` **works** — LLM extraction runs in parallel, then `add_entries()` is called once on the main thread. No connection contention.
- `store.close()` only closes the calling thread's connection. In a server/long-running process, connections accumulate (one per thread). For FastAPI/async use, call `store.close()` in a thread cleanup hook, or set `IRIS_MAX_CONNECTIONS` in your IRIS instance config to cap the pool.

### `TOP` clause is literal, not parameterized

IRIS SQL requires `TOP N` to be a literal integer in the query string. Parameterized `TOP ?` raises `SQLCODE -1`. This is handled internally — `top_k` values are formatted into the SQL string, not bound as parameters. **Implication**: if you write custom queries against the IRIS tables, don't try to bind `TOP ?`.

### `$FIND` is substring, not word-boundary

`$FIND(text, 'bob')` matches `"bobby"` and `"elbow"`. For conversational memory entries this is usually acceptable. If you need exact word matching, wrap the keyword search with a post-filter in Python, or use `%MATCHES '*\bbob\b*'` syntax (IRIS pattern matching supports word boundaries via `\b` in some contexts).

### HNSW index requires `TOP` + `ORDER BY DESC`

The HNSW index is **only used** when the query includes both a `TOP N` clause and `ORDER BY ... DESC` on the vector distance function. A query without `TOP` will fall back to a full scan even if the HNSW index exists. All three search methods in `VectorStore` already follow this pattern correctly.

### HNSW index versioning

IRIS documentation notes that future versions may change the internal HNSW storage format, requiring an index rebuild. If after an IRIS upgrade you see:

```
HNSW index HNSWIdx was built using an unsupported HNSW storage version
```

Drop and recreate: `DROP INDEX HNSWIdx ON TABLE memory_entries` then restart SimpleMem (the index is recreated automatically by `_ensure_table()`).

### Embedding dimension lock-in (HNSW is strict)

The HNSW index is tied to the vector dimension at creation time. Changing `EMBEDDING_MODEL` to one with a different output dimension requires:

```sql
DROP TABLE memory_entries    -- or cross_memory_entries
```

Then restart SimpleMem — the table and index are recreated with the new dimension. All previously stored memories are lost on drop.

### Parallel mode — `enable_parallel_processing` vs `enable_parallel_retrieval`

| Mode | What's parallel | DB impact |
|---|---|---|
| `enable_parallel_processing=True` | LLM extraction calls | Single `add_entries()` call after all workers finish. Zero contention. |
| `enable_parallel_retrieval=True` | Search query calls | Each thread opens its own IRIS connection. True concurrent DB reads. Works correctly. |
| Both enabled | All of the above | Works. Monitor IRIS connection count with `SELECT COUNT(*) FROM %SYS.ProcessQuery WHERE ClientName LIKE '%iris%'`. |

### `OmniSimpleMem` and `SKILL/` are unaffected

`OmniSimpleMem/` uses its own internal vector stores (FAISS + custom storage). It does not go through `database/vector_store.py`. The IRIS backend change has no effect on `mode="omni"`.

`SKILL/simplemem-skill/` is a self-contained distribution copy that still uses LanceDB. It is intentionally not updated — it ships as a standalone package for Claude Skills and manages its own dependencies.

### Cross-session memory — SQLite vs IRIS SQL backend

By default, `create_orchestrator` uses SQLite for session/event/observation metadata. To run everything in IRIS:

```python
orch = create_orchestrator("my-project", use_iris_sql=True)
```

The `use_iris_sql=True` flag switches the metadata backend to `IRISSQLStorage` — six `CrossMem_*` tables created automatically in your IRIS namespace. The vector store (`cross_memory_entries`) always uses IRIS regardless of this flag.

SQLite is the better default for local or single-machine deployments. Full IRIS mode is better when you want a single namespace, single backup, and the ability to JOIN memory data with your application tables.

### Cross-session memory — `lancedb_path` parameter is deprecated

`create_orchestrator(project=..., lancedb_path=...)` still accepts `lancedb_path` for backward compatibility but **ignores it**. Use `iris_table="my_table_name"` to override the default `cross_memory_entries` table name.

---

## 🔌 MCP Server *(text memory)*

SimpleMem is available as a **cloud-hosted memory service** via the Model Context Protocol (MCP), enabling seamless integration with AI assistants like Claude Desktop, Cursor, and other MCP-compatible clients.

**🌐 Cloud Service**: [mcp.simplemem.cloud](https://mcp.simplemem.cloud) — or self-host the MCP server locally using [Docker](#-run-with-docker).

### Key Features

| Feature | Description |
|---------|-------------|
| **Streamable HTTP** | MCP 2025-03-26 protocol with JSON-RPC 2.0 |
| **Multi-tenant Isolation** | Per-user data tables with token authentication |
| **Hybrid Retrieval** | Semantic search + keyword matching + metadata filtering |
| **Production Optimized** | Faster response times with OpenRouter integration |

### Quick Configuration

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

> 📖 For detailed setup instructions and self-hosting guide, see [MCP Documentation](MCP/README.md)

---

## 🗺️ Roadmap

**Omni-SimpleMem infrastructure** — bringing multimodal memory to all shared components:

- [ ] Omni cross-session memory (text + image + audio + video persistence)
- [ ] Omni MCP server (multimodal memory via MCP protocol)
- [ ] Omni Docker support
- [ ] Omni PyPI package (`pip install omni-simplemem`)
- [ ] Omni Claude Skills integration

**Core improvements:**

- [ ] Streaming ingestion for real-time memory updates
- [ ] Memory sharing across multiple agents
- [ ] Benchmark expansion (more multimodal benchmarks)

Contributions welcome! Open an [issue](https://github.com/aiming-lab/SimpleMem/issues) to discuss.

---

## 📊 Evaluation

### 🧪 Run Benchmark Tests

```bash
# 🎯 Full LoCoMo benchmark
python test_locomo10.py

# 📉 Subset evaluation (5 samples)
python test_locomo10.py --num-samples 5

# 💾 Custom output file
python test_locomo10.py --result-file my_results.json
```

---

### 🔬 Reproduce Paper Results

Use the exact configurations in `config.py`:
- **🚀 High-capability**: GPT-4.1-mini, Qwen3-Plus
- **⚙️ Efficient**: Qwen2.5-1.5B, Qwen2.5-3B
- **🔍 Embedding**: Qwen3-Embedding-0.6B (1024-d)

---

## 📝 Citation

If you use SimpleMem in your research, please cite:

```bibtex
@article{simplemem2025,
  title={SimpleMem: Efficient Lifelong Memory for LLM Agents},
  author={Liu, Jiaqi and Su, Yaofeng and Xia, Peng and Zhou, Yiyang and Han, Siwei and  Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2601.02553},
  year={2025},
  url={https://github.com/aiming-lab/SimpleMem}
}
```

```bibtex
@article{omnisimplemem2026,
  title   = {Omni-SimpleMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory},
  author  = {Liu, Jiaqi and Ling, Zipeng and Qiu, Shi and Liu, Yanqing and Han, Siwei and Xia, Peng and Tu, Haoqin and Zheng, Zeyu and Xie, Cihang and Fleming, Charles and Ding, Mingyu and Yao, Huaxiu},
  journal = {arXiv preprint arXiv:2604.01007},
  year    = {2026},
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We would like to thank the following projects and teams:

- 🔍 **Embedding Model**: [Qwen3-Embedding](https://github.com/QwenLM/Qwen) - State-of-the-art retrieval performance
- 🗄️ **Vector Database**: [LanceDB](https://lancedb.com/) - High-performance columnar storage
- 📊 **Benchmark**: [LoCoMo](https://github.com/snap-research/locomo) - Long-context memory evaluation framework
