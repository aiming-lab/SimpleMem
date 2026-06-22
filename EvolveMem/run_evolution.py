"""
EvolveMem Self-Evolution Runner
===============================
Demonstrates the self-evolution capability on LoCoMo benchmark.
The framework automatically extracts, retrieves, evaluates, diagnoses,
and adjusts its configuration through iterative rounds.

Usage:
    python run_evolution.py                          # Full evolution (5 rounds)
    python run_evolution.py --max-rounds 3           # Quick 3-round evolution
    python run_evolution.py --use-cache cache.json   # Start with pre-extracted memories
"""

import argparse
import json
import logging
import os
import time

from openai import OpenAI

# ── LLM Configuration (set via environment variables) ──
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")

if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

client = OpenAI(base_url=API_BASE, api_key=API_KEY)


def llm_call(messages: list[dict], max_tokens: int = 4096, temperature: float = 0.1) -> str:
    """Universal LLM call function passed to all EvolveMem components."""
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            content = r.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                logging.warning("LLM call failed after 3 attempts: %s", e)
                return ""


def load_locomo_sample(data_path: str, sample_idx: int = 0):
    """Load a LoCoMo sample and return (sessions, qa_pairs, metadata)."""
    with open(data_path) as f:
        data = json.load(f)

    sample = data[sample_idx]
    qa_pairs = sample["qa"]
    conversation = sample["conversation"]
    speaker_a = conversation.get("speaker_a", "A")
    speaker_b = conversation.get("speaker_b", "B")

    # Parse sessions
    session_keys = sorted(
        [k for k in conversation.keys()
         if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )

    sessions = []
    for sk in session_keys:
        dt_key = sk + "_date_time"
        date_str = conversation.get(dt_key, "")
        turns_raw = conversation[sk]
        if isinstance(turns_raw, str):
            try:
                turns = json.loads(turns_raw)
            except json.JSONDecodeError:
                turns = []
        elif isinstance(turns_raw, list):
            turns = turns_raw
        else:
            turns = []
        sessions.append((sk, date_str, turns))

    return sessions, qa_pairs, {"speaker_a": speaker_a, "speaker_b": speaker_b}


def main():
    parser = argparse.ArgumentParser(description="EvolveMem Self-Evolution Runner")
    parser.add_argument("--data", default="data/locomo10.json", help="LoCoMo data path")
    parser.add_argument("--sample", type=int, default=0, help="Sample index")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max evolution rounds")
    parser.add_argument("--use-cache", type=str, default=None, help="Pre-extracted memory cache")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable semantic search")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Always show evolution progress
    logging.getLogger("evolvemem.evolution").setLevel(logging.INFO)

    print("=" * 70)
    print("EvolveMem Self-Evolution Engine")
    print(f"Model: {LLM_MODEL}")
    print(f"Max rounds: {args.max_rounds}")
    print("=" * 70)

    # Load data
    sessions, qa_pairs, meta = load_locomo_sample(args.data, args.sample)
    print(f"Sample {args.sample}: {meta['speaker_a']} & {meta['speaker_b']}")
    print(f"Sessions: {len(sessions)}, QA pairs: {len(qa_pairs)}")

    # Load embedder
    embedder = None
    if not args.no_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("Embedder: all-MiniLM-L6-v2 loaded")
        except ImportError:
            print("sentence-transformers not installed, semantic search disabled")

    # Load cached memories if provided
    initial_memories = None
    if args.use_cache:
        with open(args.use_cache) as f:
            initial_memories = json.load(f)
        print(f"Loaded {len(initial_memories)} cached memories")

    # Configure evolution
    from evolvemem.evolution import EvolutionConfig, EvolutionEngine
    from evolvemem.multi_retriever import RetrievalConfig

    config = EvolutionConfig(
        max_rounds=args.max_rounds,
        convergence_threshold=0.005,
        initial_retrieval_config=RetrievalConfig(
            semantic_top_k=20,
            keyword_top_k=8,
            structured_top_k=5,
            max_context=25,
        ),
        cache_dir=f"evolution_cache/sample_{args.sample}",
        results_dir=f"evolution_results/sample_{args.sample}",
    )

    # Run evolution
    engine = EvolutionEngine(
        llm_call=llm_call,
        embedder=embedder,
        config=config,
    )

    print("\nStarting self-evolution...")
    print("-" * 70)

    result = engine.evolve(
        sessions=sessions,
        qa_pairs=qa_pairs,
        initial_memories=initial_memories,
    )

    # Print results
    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print(result.trajectory())
    print(f"\nTotal time: {result.total_duration:.0f}s")
    print(f"Best config: {json.dumps(result.final_config, indent=2)}")

    # Show per-round improvements
    if len(result.rounds) > 1:
        print("\nPer-round improvements:")
        for r in result.rounds:
            if r.improvements_applied:
                print(f"  Round {r.round_id}: {', '.join(r.improvements_applied)}")


if __name__ == "__main__":
    main()
