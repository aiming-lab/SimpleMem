

<h2 align="center"><b>EvolveMem: Self-Evolving Memory Architecture via AutoResearch</b></h2>

<p align="center">
  <b><i>Extending <a href="https://github.com/aiming-lab/SimpleMem">SimpleMem</a> with self-evolving retrieval infrastructure. The system autonomously researches its own architecture through LLM-driven closed-loop diagnosis.</i></b>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="#-results"><img src="https://img.shields.io/badge/SOTA-LoCoMo%20%7C%20MemBench-ff6f00?logo=target&logoColor=white" alt="SOTA"></a>
  <a href="#-citation"><img src="https://img.shields.io/badge/NeurIPS-2026-blue?logo=arxiv&logoColor=white" alt="NeurIPS 2026"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#%EF%B8%8F-architecture">Architecture</a> &nbsp;·&nbsp;
  <a href="#-results">Results</a> &nbsp;·&nbsp;
  <a href="#-self-evolution-trajectory">Evolution</a> &nbsp;·&nbsp;
  <a href="#-citation">Citation</a>
</p>

---

## 💡 Key Idea

Every existing memory system evolves what it *stores* but never how it *retrieves*. EvolveMem closes this gap.

The retrieval infrastructure (fusion weights, context budgets, answer styles, per-category overrides, ...) is exposed as a **structured action space** and optimized through an autonomous closed-loop:

| Step | What happens |
|:--:|:--|
| 📊 **Evaluate** | Run held-out QA, write per-question failure logs |
| 🔍 **Diagnose** | LLM reads failure logs, identifies root causes |
| 💡 **Propose** | Targeted configuration adjustments |
| 🛡️ **Guard** | Auto-revert if performance drops |

This closed-loop self-evolution realizes an **AutoResearch** process: the system conducts the observe-hypothesize-experiment-validate cycle on its own architecture.

---

## ✨ Highlights

<table>
<tr>
<td align="center" width="160">📈 <b>+25.7%</b><br><sub>vs. strongest baseline (LoCoMo)</sub></td>
<td align="center" width="160">📈 <b>+18.9%</b><br><sub>vs. strongest baseline (MemBench)</sub></td>
<td align="center" width="160">🧬 <b>Self-expanding</b><br><sub>3 new dimensions discovered</sub></td>
<td align="center" width="160">🔄 <b>Positive transfer</b><br><sub>Cross-benchmark generalization</sub></td>
<td align="center" width="140">⚙️ <b>7 rounds</b><br><sub>Fully autonomous</sub></td>
</tr>
</table>

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem/EvolveMem
pip install -r requirements.txt
```

### Configuration

```bash
export OPENAI_API_KEY="your-key-here"
export OPENAI_API_BASE="https://api.openai.com/v1"   # or Azure endpoint
export LLM_MODEL="gpt-4o"
```

### Run Self-Evolution

```bash
# Full evolution on LoCoMo (7 rounds)
python run_evolution.py --data data/locomo10.json --max-rounds 7

# Quick 3-round evolution
python run_evolution.py --data data/locomo10.json --max-rounds 3

# Start from pre-extracted memory cache
python run_evolution.py --use-cache cache.json --max-rounds 5
```

### Run Benchmark Evaluation

```bash
# LoCoMo evaluation
python run_benchmark.py locomo --sample 0 --initial weak --max-rounds 3

# MemBench evaluation
python run_benchmark.py membench --agent FirstAgent \
    --categories simple comparative aggregative conditional \
    --initial weak --max-rounds 3
```

---

## 🏗️ Architecture

EvolveMem consists of three layers connected by a self-evolution feedback loop:

### 1. 🗄️ Structured Memory Store

| Component | Description |
|:--|:--|
| **SQLite + FTS5** | Persistent storage with full-text search |
| **LLM Extraction** | Sliding window with retry, chunk-splitting, coverage verification |
| **Consolidation** | Deduplication, importance decay, entity reinforcement |

### 2. 🔍 Multi-View Retrieval (Evolvable Action Space)

| View | Signal | Purpose |
|:--|:--|:--|
| 📝 **Lexical** | BM25 | Exact keyword matching |
| 🧠 **Semantic** | Dense embeddings | Conceptual similarity |
| 🏷️ **Structured** | Entity/location/person metadata | Structured filtering |

Fusion mode, per-view weights, context budgets, answer styles, and per-category overrides are all **evolvable parameters**.

### 3. 🧬 Self-Evolution Engine (AutoResearch)

The engine reads per-question failure logs, diagnoses root causes, and proposes targeted adjustments. Three safeguards ensure robustness:

| Safeguard | Trigger | Action |
|:--|:--|:--|
| 🛡️ **Revert** | Performance drops > threshold | Roll back to best-so-far |
| 🔀 **Explore** | Score plateaus for 2 rounds | Random perturbation |
| ⏹️ **Converge** | Improvement < epsilon | Terminate and return best |

<details>
<summary>📐 Full system diagram</summary>

```
Raw Conversations
  │
  ▼
┌─────────────────────────────┐
│   LLM-Based Extraction      │  ← Sliding window + retry + coverage verify
│   → Typed Memory Units      │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   Multi-View Retrieval      │
│   BM25 ∪ Semantic ∪ Struct  │  ← Evolvable fusion (sum/weighted/RRF)
│   + Entity-swap             │
│   + Query decomposition     │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   Answer Generation         │  ← Per-category style + verification
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   Evaluation + Diagnosis    │  ← LLM reads per-question failure logs
│   → Structured proposal     │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   Meta-Analyzer             │  ← Revert / Explore / Apply
│   → Updated config θ        │
└─────────────┬───────────────┘
              │
              └──── Loop back to Retrieval ────┘
```

</details>

---

## 📊 Results

### LoCoMo (Token-F1)

| Method | GPT-4o | GPT-5.1 |
|:--|:--:|:--:|
| MemVerse | 0.365 | 0.383 |
| Mem0 | 0.397 | 0.390 |
| A-MEM | 0.394 | 0.385 |
| MemGPT | 0.404 | 0.385 |
| SimpleMem | 0.432 | 0.418 |
| **EvolveMem** | **0.543** | **0.572** |

### MemBench (Accuracy %)

| Method | GPT-4o | GPT-5.1 |
|:--|:--:|:--:|
| RecentMemory | 57.1 | 60.7 |
| MemGPT | 57.1 | 60.7 |
| MemoryBank | 46.4 | 64.3 |
| SCMemory | 39.3 | 32.1 |
| **EvolveMem** | **67.9** | **71.4** |

---

## 🧬 Self-Evolution Trajectory

Starting from a minimal BM25-only baseline (F1 = 30.5%), the system autonomously discovers and activates retrieval mechanisms over 7 rounds:

| Round | Stage | Automated Change | F1 (%) |
|:--:|:--:|:--|:--:|
| R0 | 🟢 start | BM25-only, k=5 | 30.5 |
| R1 | ⚙️ auto | Intent planning + RRF fusion | 35.8 |
| R2 | 🔙 revert | MMR diversity (reverted) | 34.8 |
| R3 | ⚙️ auto | Entity-swap for Cat. 5 | 37.2 |
| R4 | ⚙️ auto | Per-category answer styles | 38.5 |
| R5 | ⚙️ auto | Query decomposition for Cat. 1/4 | 38.1 |
| R6 | ⚙️ auto | Cat. 3 inferential subtypes + swap expansion | 45.4 |
| R7 | ⚙️ auto | Answer verification + hyperparameter sweep | **54.3** |

Three configuration dimensions emerged from failure diagnosis that were **not in the original design**:
- 🔀 **Query decomposition** (splitting multi-hop questions into sub-queries)
- 🔄 **Adversarial entity-swap** (stripping misleading names before retrieval)
- ✅ **Answer verification** (second-pass LLM review of low-confidence outputs)

---

## 🔄 Cross-Benchmark Transfer

| Configuration | LoCoMo (F1) | MemBench (Acc) |
|:--|:--:|:--:|
| Baseline | 0.305 | / |
| C_L (LoCoMo only) | 0.543 | 0.543 |
| C_LM (LoCoMo → MemBench) | **0.593** | **0.792** |
| C_M (MemBench only) | / | 0.679 |

> Continued evolution from a LoCoMo prior **outperforms** scratch evolution on MemBench (+16.6% relative) while also **improving** LoCoMo performance. Pareto improvement on both benchmarks.

---

## 📝 Citation

```bibtex
@article{evolvemem2026,
  title={EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents},
  author={Liu, Jiaqi and Ye, Xinyu and Xia, Peng and Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2605.13941},
  year={2026},
  url={https://arxiv.org/abs/2605.13941}
}
```
