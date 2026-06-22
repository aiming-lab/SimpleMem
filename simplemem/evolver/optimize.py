"""
simplemem.optimize() — degraded retrieval-tuner.

Runs EvolveMem's self-evolution loop on a user-supplied dev set to find a
better retrieval configuration for an existing SimpleMem instance.

This is the *degraded* mode of EvolveMem. It is intentionally NOT the
paper-faithful entry point. It feeds EvolveMem:

  - your `dev_questions` as (q, answer) pairs (no per-category labels)
  - your `mem`'s already-built memories (no fresh extraction loop)
  - `adapter=None` (no benchmark-specific prompts)

So per-category overrides, adapter prompts, and extractor evolution are all
disabled. What remains is LLM-driven search over global retrieval hyper-
parameters (top_k, max_context, fusion mode, weights, global toggles). To
reproduce paper numbers, use `EvolveMem/run_evolution.py` instead.
"""

from typing import Any, Callable, List, Tuple

from simplemem.config import Config


def run_optimization(
    mem: Any,
    dev_questions: List[Tuple[str, str]],
    max_rounds: int = 7,
    **kwargs: Any,
) -> Config:
    """Optimize retrieval config on a user dev set.

    Args:
        mem: A finalized SimpleMem instance with memories already loaded.
        dev_questions: List of (question, ground_truth_answer) tuples.
        max_rounds: Maximum evolution rounds.
        **kwargs: Forwarded to EvolutionConfig where applicable
            (e.g. `convergence_threshold`, `cache_dir`, `results_dir`,
            `benchmark_name`).

    Returns:
        Config with the best retrieval parameters found.
    """
    from simplemem.evolver.evolution import EvolutionConfig, EvolutionEngine

    backend = _resolve_backend(mem)
    llm_call = _wrap_llm_call(backend)
    embedder = _wrap_embedder(backend)
    initial_memories = _memories_from_backend(backend)
    qa_pairs = _qa_pairs_from_tuples(dev_questions)

    evo_config = EvolutionConfig(
        max_rounds=max_rounds,
        **{k: v for k, v in kwargs.items() if hasattr(EvolutionConfig, k)},
    )

    engine = EvolutionEngine(
        llm_call=llm_call,
        embedder=embedder,
        config=evo_config,
        adapter=None,
    )
    result = engine.evolve(
        sessions=[],
        qa_pairs=qa_pairs,
        initial_memories=initial_memories,
    )

    best = getattr(result, "final_config", {}) or {}
    return Config(
        k_sem=best.get("semantic_top_k", 0),
        k_kw=best.get("keyword_top_k", 5),
        k_str=best.get("structured_top_k", 0),
        context_budget=best.get("max_context", 8),
        fusion_mode=best.get("fusion_mode", "sum"),
        fusion_weights={
            "semantic": best.get("weight_semantic", 1.0),
            "keyword": best.get("weight_keyword", 1.0),
            "structured": best.get("weight_structured", 1.0),
        },
        answer_style=best.get("answer_style", "concise"),
        enable_entity_swap=best.get("enable_entity_swap", False),
        enable_query_decomposition=best.get("enable_query_decomposition", False),
        enable_answer_verification=best.get("enable_answer_verification", False),
        category_overrides=best.get("per_category_overrides", {}),
        evolved=True,
        evolution_rounds=getattr(result, "best_round", max_rounds),
        source_benchmark=kwargs.get("benchmark_name", ""),
    )


# ---------------------------------------------------------------------------
# Backend adapters (kept here so we do not touch SimpleMem's own code)
# ---------------------------------------------------------------------------


def _resolve_backend(mem: Any) -> Any:
    """Unwrap AutoMemory -> SimpleMemSystem if needed."""
    backend = getattr(mem, "_backend", mem)
    if backend is None:
        raise RuntimeError(
            "simplemem.optimize(): the given memory instance is not initialized. "
            "Call mem.add_dialogue(...) and mem.finalize() first."
        )
    if not hasattr(backend, "llm_client"):
        raise TypeError(
            "simplemem.optimize() currently only supports the text backend. "
            f"Backend {type(backend).__name__} does not expose `llm_client`."
        )
    return backend


def _wrap_llm_call(backend: Any) -> Callable[..., str]:
    """Adapt SimpleMem's LLMClient.chat_completion to EvolveMem's
    (messages, max_tokens, temperature) -> str signature.

    SimpleMem's chat_completion takes no max_tokens, so we drop it.
    """
    llm_client = backend.llm_client

    def llm_call(messages, max_tokens: int = 4096, temperature: float = 0.1) -> str:
        try:
            out = llm_client.chat_completion(messages, temperature=temperature)
            return out or ""
        except Exception:
            return ""

    return llm_call


class _SentenceTransformerEncodeAdapter:
    """Wrap SimpleMem's EmbeddingModel so EvolveMem's
    `embedder.encode(texts, normalize_embeddings=True)` works unmodified.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def encode(self, texts, normalize_embeddings: bool = False, **_: Any):
        return self._inner.encode(texts)


def _wrap_embedder(backend: Any):
    inner = getattr(backend, "embedding_model", None)
    if inner is None:
        return None
    return _SentenceTransformerEncodeAdapter(inner)


def _memories_from_backend(backend: Any) -> List[dict]:
    """Convert SimpleMem MemoryEntry objects into the dict format that
    EvolveMem's MultiViewIndex consumes (it reads `.get('content')`,
    `.get('keywords')`, `.get('persons')`, `.get('entities')`, `.get('topic')`,
    `.get('location')`, `.get('timestamp')`, `.get('session_id')`).
    """
    if not hasattr(backend, "get_all_memories"):
        raise TypeError(
            "Backend does not expose `get_all_memories()`. "
            "simplemem.optimize() needs an indexable memory store."
        )
    entries = backend.get_all_memories() or []
    return [
        {
            "content": getattr(e, "lossless_restatement", "") or "",
            "keywords": list(getattr(e, "keywords", []) or []),
            "persons": list(getattr(e, "persons", []) or []),
            "entities": list(getattr(e, "entities", []) or []),
            "topic": getattr(e, "topic", None),
            "location": getattr(e, "location", None),
            "timestamp": getattr(e, "timestamp", None),
            "session_id": "default",
        }
        for e in entries
    ]


_DEFAULT_CATEGORY = 0


def _qa_pairs_from_tuples(dev_questions: List[Tuple[str, str]]) -> List[dict]:
    """Convert (question, gt) tuples into the dict shape EvolveMem expects.

    EvolveMem coerces `category` to int internally, so degraded mode uses a
    numeric placeholder (0) that does not collide with LoCoMo (1-5) or
    MemBench category ids. This collapses all per-category cookbook recipes
    into a single global track.
    """
    out: List[dict] = []
    for item in dev_questions:
        if isinstance(item, dict):
            out.append({**item, "category": item.get("category", _DEFAULT_CATEGORY)})
            continue
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(
                "dev_questions must be a list of (question, answer) pairs."
            )
        q, a = item[0], item[1]
        out.append({"question": q, "answer": a, "category": _DEFAULT_CATEGORY})
    return out
