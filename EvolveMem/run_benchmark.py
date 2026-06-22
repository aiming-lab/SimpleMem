"""Unified benchmark runner for EvolveMem self-evolution.

Works for any registered benchmark (locomo / longmemeval / membench) via the
adapter system. Supports:
  - weak-initial vs strong-initial (for evolved-minus-static delta experiments)
  - small-sample fast iteration (--max-samples / --max-qa)
  - MemBench subset selection (--agent, --categories, --topics)
  - LongMemEval split selection (--split oracle|s|m)
  - Full multi-metric recording into evolution_results/<benchmark>/<run_id>/round_<N>/

Every round persists:
  round_<N>/raw_results.jsonl  — per-question full detail
  round_<N>/summary.json       — per-subcategory, per-metric aggregates
  round_<N>.json               — flat top-level view (back-compat)

Usage
-----
    # LoCoMo, weak-initial, 3-round evolution, sample 0
    python run_benchmark.py locomo --sample 0 --initial weak --max-rounds 3

    # LongMemEval oracle, quick small-sample iteration
    python run_benchmark.py longmemeval --split oracle --max-samples 30 \
        --initial weak --max-rounds 3

    # MemBench FirstAgent LowLevel 4 categories, 20 QA each
    python run_benchmark.py membench --agent FirstAgent \
        --categories simple comparative aggregative conditional \
        --max-per-category 20 --initial weak --max-rounds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid

import yaml
from openai import OpenAI

from evolvemem.benchmarks import (
    BenchmarkSample,
    LoCoMoAdapter,
    LongMemEvalAdapter,
    MemBenchAdapter,
    get_adapter,
)
from evolvemem.evolution import (  # noqa: F401
    evolved_config,
)
from evolvemem.evolution import (
    EvolutionConfig,
    EvolutionEngine,
    strong_initial_config,
    weak_initial_config,
)
from evolvemem.multi_retriever import RetrievalConfig

logger = logging.getLogger(__name__)


# ── LLM client ────────────────────────────────────────────────────────────

def _load_key_config(path: str = "openai_key.yaml") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY or provide openai_key.yaml")
    return {
        "api_key": api_key,
        "base_url": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "model": os.environ.get("LLM_MODEL", "gpt-4o"),
    }


def _make_llm_call(cfg: dict):
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    model = cfg.get("model", "gpt-4o")

    supports_temperature = not model.startswith("gpt-5") and not model.startswith("o")

    def llm_call(messages, max_tokens: int = 4096, temperature: float = 0.1):
        kwargs = dict(
            model=model, messages=messages,
            max_completion_tokens=max_tokens,
        )
        if supports_temperature:
            kwargs["temperature"] = temperature
        for attempt in range(3):
            try:
                r = client.chat.completions.create(**kwargs)
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.warning("LLM call failed after 3 attempts: %s", e)
                    return ""
        return ""

    return llm_call


# ── Sample loaders per benchmark ─────────────────────────────────────────

def _load_samples(args) -> tuple[list[BenchmarkSample], object]:
    if args.benchmark == "locomo":
        adapter = LoCoMoAdapter()
        path = args.data or "data/locomo10.json"
        if args.samples:
            indices = [int(x) for x in args.samples.split(",")]
        elif args.sample is not None:
            indices = [args.sample]
        else:
            indices = None
        samples = adapter.load(path, sample_indices=indices, max_qa=args.max_qa)
    elif args.benchmark == "longmemeval":
        adapter = LongMemEvalAdapter()
        split = args.split or "oracle"
        path = args.data or f"data/longmemeval/longmemeval_{split}.json"
        samples = adapter.load(
            path,
            max_samples=args.max_samples,
            qtype_filter=args.qtypes,
            stratify=args.stratify,
        )
    elif args.benchmark == "membench":
        adapter = MemBenchAdapter()
        path = args.data or "data/membench/repo/MemData"
        samples = adapter.load(
            path,
            agent=args.agent,
            categories=args.categories,
            topics=args.topics,
            max_samples_per_file=args.max_per_category,
        )
    else:
        raise ValueError(f"unknown benchmark: {args.benchmark}")
    return samples, adapter


def _merge_samples_for_evolution(
    samples: list[BenchmarkSample],
) -> tuple[list[tuple[str, str, list[dict]]], list[dict]]:
    """Flatten a list of BenchmarkSamples into a single (sessions, qa_pairs)
    tuple suitable for `EvolutionEngine.evolve`.

    Session ids get a per-sample prefix to avoid collisions.
    """
    all_sessions: list[tuple[str, str, list[dict]]] = []
    all_qa: list[dict] = []
    for s in samples:
        for (sid, date, turns) in s.sessions:
            all_sessions.append((f"{s.sample_id}::{sid}", date, turns))
        for qa in s.qa_pairs:
            qa2 = dict(qa)
            qa2["_sample_id"] = s.sample_id
            all_qa.append(qa2)
    return all_sessions, all_qa


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=("locomo", "longmemeval", "membench"))
    p.add_argument("--data", default=None, help="override default data path")
    p.add_argument("--initial", default="weak",
                   choices=("weak", "strong", "terminal", "custom"),
                   help="starting RetrievalConfig — weak for big evolution delta; "
                        "terminal for the hardcoded known-best config (skip evolution)")
    p.add_argument("--config-from", default=None,
                   help="path to a prior run's evolution_summary.json; its final_config "
                        "becomes the initial config (used for cross-benchmark transfer tests)")
    p.add_argument("--static", action="store_true",
                   help="skip evolution loop — just evaluate with the initial config once")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--maturation-round", type=int, default=None,
                   help="Round at which evolution matures to full configuration "
                        "(default: 5 for LoCoMo)")
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--embed-model", default=None,
                   help="sentence-transformers model id; default BAAI/bge-base-en-v1.5")
    p.add_argument("--run-id", default=None, help="override auto-generated run id")
    p.add_argument("--cache", default=None,
                   help="pre-extracted memories json; skips extraction when set")
    p.add_argument("--answer-model", default=None,
                   help="override model for inference/answer-gen only; "
                        "extraction uses whatever produced the --cache file")
    p.add_argument("--verbose", action="store_true")

    # LoCoMo
    p.add_argument("--sample", type=int, default=None, help="LoCoMo sample index")
    p.add_argument("--samples", default=None, help="LoCoMo comma-separated sample indices (e.g. 0,3,5,8 for joint evolution)")

    # LongMemEval
    p.add_argument("--split", default=None, choices=(None, "oracle", "s", "m"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--qtypes", nargs="*", default=None,
                   help="LongMemEval question_type filter")
    p.add_argument("--stratify", action="store_true",
                   help="LongMemEval: evenly distribute across question_types "
                        "(avoids the first-N-all-temporal bias)")

    # MemBench
    p.add_argument("--agent", default="FirstAgent", choices=("FirstAgent", "ThirdAgent"))
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--topics", nargs="*", default=None)
    p.add_argument("--max-per-category", type=int, default=None)

    # Shared QA cap
    p.add_argument("--max-qa", type=int, default=None, help="cap QA per sample (LoCoMo)")

    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("evolvemem").setLevel(logging.INFO)

    # Load samples + adapter
    samples, adapter = _load_samples(args)
    sessions, qa_pairs = _merge_samples_for_evolution(samples)
    n_sessions = len(sessions)
    n_qa = len(qa_pairs)
    print(f"Benchmark: {args.benchmark}")
    print(f"Samples: {len(samples)} | Sessions: {n_sessions} | QA: {n_qa}")
    print(f"Primary metric: {adapter.primary_metric}")

    # LLM
    key_cfg = _load_key_config()
    if args.answer_model:
        key_cfg = dict(key_cfg)
        key_cfg["model"] = args.answer_model
        print(f"Answer-model override: {args.answer_model}")
    llm_call = _make_llm_call(key_cfg)

    # Factory for per-category answer-model routing (scheme G).
    # Returns a bound llm_call for any model name; cached by EvolutionEngine.
    def llm_call_factory(model_name: str):
        per_model_cfg = dict(key_cfg)
        per_model_cfg["model"] = model_name
        return _make_llm_call(per_model_cfg)

    # Embedder (only needed if fusion uses semantic).
    # Default: BAAI/bge-base-en-v1.5 (768-d, strong open-source English
    # encoder, consistent top-3 on BEIR). Override via --embed-model.
    # Fall back to MiniLM (tiny, 384-d) when the BGE download fails.
    embedder = None
    if not args.no_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            emb_name = args.embed_model or "BAAI/bge-base-en-v1.5"
            try:
                embedder = SentenceTransformer(emb_name)
                print(f"Embedder: {emb_name}")
            except Exception as e:
                print(f"Embedder '{emb_name}' failed ({e}); falling back to MiniLM-L6")
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("sentence-transformers not installed — semantic disabled")

    # Initial config
    if args.config_from:
        with open(args.config_from) as f:
            prior = json.load(f)
        cfg_fields = prior.get("final_config") or prior.get("config") or prior
        # Filter to fields RetrievalConfig actually has
        allowed = set(RetrievalConfig.__dataclass_fields__.keys())
        cfg_clean = {k: v for k, v in cfg_fields.items() if k in allowed}
        ret_cfg = RetrievalConfig(**cfg_clean)
        print(f"Loaded config from {args.config_from}")
    elif args.initial == "weak":
        ret_cfg = weak_initial_config()
    elif args.initial == "strong":
        ret_cfg = strong_initial_config()
    elif args.initial == "terminal":
        ret_cfg = evolved_config()
    else:
        ret_cfg = RetrievalConfig()

    if args.static:
        args.max_rounds = 1
        print("Static mode: single-pass evaluation (no evolution)")

    # Run id & output dir
    run_id = args.run_id or time.strftime(
        f"{args.benchmark}_{args.initial}_%Y%m%d_%H%M%S"
    )
    cache_dir = f"evolution_cache/{args.benchmark}/{run_id}"
    results_dir = f"evolution_results/{args.benchmark}/{run_id}"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    mat_round = args.maturation_round
    active_model = key_cfg.get("model", "gpt-4o")
    if args.answer_model:
        active_model = args.answer_model
    if mat_round is None and args.benchmark == "locomo" and not args.static:
        if active_model.startswith("gpt"):
            mat_round = 5

    cfg = EvolutionConfig(
        max_rounds=max(args.max_rounds, (mat_round or 0) + 2),
        convergence_threshold=0.005,
        initial_retrieval_config=ret_cfg,
        cache_dir=cache_dir,
        results_dir=results_dir,
        maturation_round=mat_round,
    )

    # Hook LLM-judge for LongMemEval (opt-in via env var)
    if args.benchmark == "longmemeval" and os.environ.get("METAMEM_LLM_JUDGE"):
        adapter.llm_judge_call = llm_call

    engine = EvolutionEngine(
        llm_call=llm_call, embedder=embedder,
        config=cfg, adapter=adapter,
        llm_call_factory=llm_call_factory,
    )

    initial_memories = None
    if args.cache:
        with open(args.cache) as f:
            initial_memories = json.load(f)
        print(f"Cache: {len(initial_memories)} memories")

    # Run-level metadata
    with open(os.path.join(results_dir, "run_meta.json"), "w") as f:
        json.dump({
            "benchmark": args.benchmark,
            "run_id": run_id,
            "initial": args.initial,
            "initial_config": cfg.initial_retrieval_config.__dict__,
            "n_samples": len(samples),
            "n_sessions": n_sessions,
            "n_qa": n_qa,
            "primary_metric": adapter.primary_metric,
            "args": vars(args),
            "model": key_cfg.get("model"),
        }, f, indent=2, default=str)

    t0 = time.time()
    result = engine.evolve(
        sessions=sessions, qa_pairs=qa_pairs,
        initial_memories=initial_memories,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print(result.trajectory())
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save the evolution-level summary
    with open(os.path.join(results_dir, "evolution_summary.json"), "w") as f:
        json.dump({
            "run_id": run_id,
            "benchmark": args.benchmark,
            "initial": args.initial,
            "best_round": result.best_round,
            "best_primary": result.best_f1,
            "final_config": result.final_config,
            "total_duration": result.total_duration,
            "rounds": [
                {
                    "round": r.round_id,
                    "primary": r.f1,
                    "all_metrics": r.all_metrics,
                    "subcategory_scores": r.subcategory_scores,
                    "zero_count": r.zero_f1_count,
                    "total": r.total_questions,
                    "mems": r.memory_count,
                    "improvements": r.improvements_applied,
                    "config": r.retrieval_config,
                }
                for r in result.rounds
            ],
        }, f, indent=2, default=str)

    print(f"\nArtifacts: {results_dir}")


if __name__ == "__main__":
    main()
