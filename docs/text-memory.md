# SimpleMem: Text Memory

How the text backend turns raw dialogue into compact, retrievable memory. For the high-level summary and where this fits the unified package, see the [main README Overview](../README.md#-overview).

## 1. Semantic Structured Compression

SimpleMem applies an **implicit semantic density gating** mechanism integrated into the LLM generation process to filter redundant interaction content. The system reformulates raw dialogue streams into **compact memory units**, self-contained facts with resolved coreferences and absolute timestamps. Each unit is indexed through three complementary representations for flexible retrieval:

| 🔍 Layer | 📊 Type | 🎯 Purpose | 🛠️ Implementation |
|---------|---------|------------|-------------------|
| **Semantic** | Dense | Conceptual similarity | Vector embeddings (1024-d) |
| **Lexical** | Sparse | Exact term matching | BM25-style keyword index |
| **Symbolic** | Metadata | Structured filtering | Timestamps, entities, persons |

**Example transformation:**

```diff
- Input:  "He'll meet Bob tomorrow at 2pm"  [relative, ambiguous]
+ Output: "Alice will meet Bob at Starbucks on 2025-11-16T14:00:00"  [absolute, atomic]
```

## 2. Online Semantic Synthesis

Unlike traditional systems that rely on asynchronous background maintenance, SimpleMem performs synthesis **on-the-fly during the write phase**. Related memory units are synthesized into higher-level abstract representations within the current session scope, allowing repetitive or structurally similar experiences to be **denoised and compressed immediately**.

**Example synthesis:**

```diff
- Fragment 1: "User wants coffee"
- Fragment 2: "User prefers oat milk"
- Fragment 3: "User likes it hot"
+ Consolidated: "User prefers hot coffee with oat milk"
```

This proactive synthesis keeps the memory topology compact and free of redundant fragmentation.

## 3. Intent-Aware Retrieval Planning

Instead of fixed-depth retrieval, SimpleMem leverages the reasoning capabilities of the LLM to generate a **comprehensive retrieval plan**. Given a query, the planning module infers **latent search intent** to dynamically determine retrieval scope and depth:

$$\{ q_{\text{sem}}, q_{\text{lex}}, q_{\text{sym}}, d \} \sim \mathcal{P}(q, H)$$

The system then executes **parallel multi-view retrieval** across semantic, lexical, and symbolic indexes, and merges results through ID-based deduplication:

| 🔹 Simple Queries | 🔸 Complex Queries |
|:--|:--|
| Direct fact lookup via single memory unit | Aggregation across multiple events |
| Minimal retrieval depth | Expanded retrieval depth |
| Fast response time | Comprehensive coverage |

**Result:** 43.24% F1 score with **30x fewer tokens** than full-context methods.
