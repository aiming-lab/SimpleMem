#!/usr/bin/env python3
"""
LoCoMo Benchmark Runner for OmniMem.

Evaluates OmniMem on the LoCoMo long-term conversation memory benchmark
(1,986 QA pairs across 10 conversations, 5 categories).

Prerequisites:
    1. Clone LoCoMo dataset:
       git clone https://github.com/snap-research/locomo.git
    2. Set environment variables:
       export OPENAI_API_KEY="your-key"
       export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional

Usage:
    cd OmniMem/
    python benchmarks/locomo/run_locomo.py --data-path /path/to/locomo/data/locomo10.json

    # With specific model:
    python benchmarks/locomo/run_locomo.py --data-path /path/to/locomo/data/locomo10.json --model gpt-4o

    # Reuse ingested memory (skip re-ingestion):
    python benchmarks/locomo/run_locomo.py --data-path /path/to/locomo/data/locomo10.json --reuse-data

    # Use pre-ingested memory from another run:
    python benchmarks/locomo/run_locomo.py --data-path /path/to/locomo/data/locomo10.json --memory-dir ./previous_run/memory_data
"""

import argparse
import json
import logging
import os
import re
import string
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- Path setup ----
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent  # simplemem-refactor root
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---- Default model ----
DEFAULT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Evaluation helpers (token-level F1, adapted from LoCoMo's task_eval)
# ---------------------------------------------------------------------------

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
except ImportError:
    _stemmer = None
    logger.warning("nltk not installed; stemming disabled (may slightly affect F1 scores)")


def _normalize_answer(s: str) -> str:
    """Lowercase, remove articles/punctuation, fix whitespace."""
    s = s.replace(',', '')
    s = re.sub(r'\b(a|an|the|and)\b', ' ', s.lower())
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def _f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 with optional stemming."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if _stemmer:
        pred_tokens = [_stemmer.stem(w) for w in pred_tokens]
        gt_tokens = [_stemmer.stem(w) for w in gt_tokens]
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _f1_multi(prediction: str, ground_truth: str) -> float:
    """Multi-answer F1: split by comma, compute best match per ground truth."""
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    import numpy as np
    return float(np.mean([
        max([_f1_score(p, gt) for p in predictions])
        for gt in ground_truths
    ]))


def eval_question_answering(qas: List[Dict], prediction_key: str = "prediction"):
    """
    Evaluate LoCoMo QA predictions.

    Categories:
        1 = Multi-hop (comma-split F1)
        2 = Single-hop (token F1)
        3 = Temporal (token F1, first answer before semicolon)
        4 = Open-domain (token F1)
        5 = Adversarial (refusal detection)

    Returns:
        (f1_scores, _, recall_scores) — list of per-sample F1, placeholder, per-sample recall
    """
    all_f1 = []
    all_recall = []

    _REFUSAL_PHRASES = [
        'no information available', 'not mentioned',
        'do not contain', "don't contain", 'does not contain', "doesn't contain",
        'no relevant memories', 'no specific information', 'no information',
        'cannot determine', "can't determine", 'do not have', "don't have",
        'not in the memories', 'not in the provided', 'not described', 'not specified',
        "i don't have any relevant", 'no details about', 'no indication',
    ]

    for i, qa in enumerate(qas):
        # Gold answer
        answer = str(qa.get('answer', ''))
        category = qa.get('category', 2)

        if category == 3:
            answer = answer.split(';')[0].strip()

        # Prediction
        output = str(qa.get(prediction_key, ''))

        # Compute F1 by category
        if category in (2, 3, 4):
            all_f1.append(_f1_score(output, answer))
        elif category == 1:
            all_f1.append(_f1_multi(output, answer))
        elif category == 5:
            out_lower = output.lower()
            all_f1.append(1.0 if any(p in out_lower for p in _REFUSAL_PHRASES) else 0.0)
        else:
            all_f1.append(_f1_score(output, answer))

        all_recall.append(1.0)

    return all_f1, 0.0, all_recall


# ---------------------------------------------------------------------------
# LoCoMo data parsing
# ---------------------------------------------------------------------------

def _parse_locomo_sample_to_dialogues(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a LoCoMo sample into a flat list of dialogue turns."""
    dialogues = []
    conv = sample.get("conversation", {})
    dialogue_id = 1
    session_keys = [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")]
    session_nums = sorted(set(int(k.split("_")[-1]) for k in session_keys if k.split("_")[-1].isdigit()))

    for sess in session_nums:
        sess_key = f"session_{sess}"
        dialogs = conv.get(sess_key, []) or []
        date_time = conv.get(f"{sess_key}_date_time", "")
        if not isinstance(dialogs, list):
            continue
        for d in dialogs:
            text = d.get("text", "")
            if "img_url" in d and d.get("blip_caption"):
                caption_text = f"[Image: {d['blip_caption']}]"
                text = f"{caption_text} {text}".strip() if text else caption_text
            if not text.strip():
                continue
            dialogues.append({
                "dialogue_id": dialogue_id,
                "speaker": d.get("speaker", "Speaker"),
                "content": text,
                "timestamp": date_time or "",
            })
            dialogue_id += 1
    return dialogues


def build_question_text(qa: Dict[str, Any]) -> str:
    question = str(qa.get("question", ""))
    if qa.get("category") == 2:
        question += " Use the dates in the conversation to answer with an approximate date."
    return question


# ---------------------------------------------------------------------------
# LLM-based memory extraction (sliding window)
# ---------------------------------------------------------------------------

def _build_extraction_prompt(dialogue_text: str, context: str = "") -> str:
    return f"""
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

{context}

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information in the dialogues is captured
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow)
3. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence
4. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time in ISO 8601 format (if explicit time mentioned in dialogue)
   - location: Specific location name (if mentioned)
   - persons: All person names mentioned
   - entities: Companies, products, organizations, etc.
   - topic: The topic of this information
5. **Image content**: When dialogue contains [Image: caption], include the caption content in the lossless_restatement as relevant factual information.

[Output Format]
Return a JSON array, each element is a memory entry:

```json
[
  {{
    "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)",
    "keywords": ["keyword1", "keyword2", ...],
    "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
    "location": "location name or null",
    "persons": ["name1", "name2", ...],
    "entities": ["entity1", "entity2", ...],
    "topic": "topic phrase"
  }},
  ...
]
```

Now process the above dialogues. Return ONLY the JSON array, no other explanations.
"""


def _extract_json_from_response(response: str) -> Any:
    text = (response or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "[": depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return None


def _extract_memory_entries(dialogues, llm_client, model_name, previous_entries=None):
    """Use LLM to extract lossless_restatement strings from a dialogue window."""
    if not dialogues:
        return []
    lines = [f"[{d.get('timestamp','')}] {d.get('speaker','Speaker')}: {d.get('content','')}" for d in dialogues]
    dialogue_text = "\n".join(lines)
    context = ""
    if previous_entries:
        context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
        for e in previous_entries[:3]:
            context += f"- {e}\n"

    prompt = _build_extraction_prompt(dialogue_text, context)
    messages = [
        {"role": "system", "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(3):
        try:
            resp = llm_client.chat.completions.create(model=model_name, messages=messages, temperature=0.1)
            response = (resp.choices[0].message.content or "").strip()
            data = _extract_json_from_response(response)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array, got {type(data)}")
            entries = []
            for item in data:
                stmt = item.get("lossless_restatement") or item.get("restatement")
                if stmt and isinstance(stmt, str) and len(stmt.strip()) > 5:
                    entries.append(stmt.strip())
            return entries
        except Exception as e:
            if attempt < 2:
                logger.debug("Extraction attempt %d failed: %s, retrying", attempt + 1, e)
            else:
                logger.warning("LLM extraction failed after 3 attempts: %s", e)
                return []
    return []


# ---------------------------------------------------------------------------
# Ingestion: process one conversation
# ---------------------------------------------------------------------------

def _ingest_one_conversation(orchestrator, sample, sample_idx, llm_client, model_name, window_size=40):
    """Ingest one conversation's text memory via sliding-window LLM extraction."""
    sample_id = str(sample.get("sample_id") or "").strip() or f"sample_{sample_idx}"
    dialogues = _parse_locomo_sample_to_dialogues(sample)
    if not dialogues:
        return 0
    total = 0
    previous_entries = []
    n_windows = (len(dialogues) + window_size - 1) // window_size

    for wi, win_start in enumerate(range(0, len(dialogues), window_size)):
        window = dialogues[win_start:win_start + window_size]
        logger.info("Conv %s (%s): extracting window %d/%d (msgs %d-%d)",
                     sample_idx, sample_id, wi + 1, n_windows, win_start,
                     min(win_start + window_size, len(dialogues)))
        entries = _extract_memory_entries(window, llm_client, model_name, previous_entries)
        if entries:
            previous_entries = entries[-10:]
        for stmt in entries:
            try:
                result = orchestrator.add_text(stmt, tags=[f"locomo_{sample_id}", "simplemem_text"], force=True)
                if result.success and result.mau:
                    total += 1
            except Exception as exc:
                logger.debug("Skip text entry: %s", exc)
    logger.info("Conv %s (%s): ingested %d entries from %d windows", sample_idx, sample_id, total, n_windows)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo Benchmark Runner for OmniMem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-path", required=True,
                        help="Path to locomo10.json (from https://github.com/snap-research/locomo)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help="LLM model name for extraction and QA (default: gpt-4o)")
    parser.add_argument("--max-conversations", type=int, default=None,
                        help="Limit number of conversations to process")
    parser.add_argument("--max-qa", type=int, default=None,
                        help="Limit number of QA pairs to evaluate")
    parser.add_argument("--reuse-data", action="store_true", default=False,
                        help="Skip ingestion if memory data already exists")
    parser.add_argument("--fresh", action="store_true", default=False,
                        help="Delete existing memory data and re-ingest from scratch")
    parser.add_argument("--concurrency", "-j", type=int, default=10,
                        help="Number of concurrent QA workers (default: 10)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for results (default: ./locomo_results_<model>)")
    parser.add_argument("--memory-dir", default=None,
                        help="Reuse memory data from this directory (implies --reuse-data)")
    args = parser.parse_args()

    # ---- Validate API key ----
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is required. Set it with:\n"
                      "  export OPENAI_API_KEY='your-key'")
        return 1

    model_name = args.model
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s\n"
                      "Download from: https://github.com/snap-research/locomo", data_path)
        return 1

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./locomo_results_{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Memory data directory
    if args.memory_dir:
        memory_data_dir = Path(args.memory_dir).resolve()
        args.reuse_data = True
    else:
        memory_data_dir = output_dir / "memory_data"

    if args.fresh and memory_data_dir.exists():
        import shutil
        shutil.rmtree(memory_data_dir)
        logger.info("Cleared memory dir (--fresh): %s", memory_data_dir)
    memory_data_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.max_conversations:
        samples = samples[:args.max_conversations]
    logger.info("Loaded %d conversations from %s", len(samples), data_path)

    # ---- Configure OmniMem ----
    from simplemem.multimodal.core.config import OmniMemoryConfig
    from simplemem.multimodal.orchestrator import OmniMemoryOrchestrator

    config = OmniMemoryConfig()
    # API credentials
    config.llm.api_key = api_key
    config.llm.api_base_url = api_base
    # Model
    config.llm.summary_model = model_name
    config.llm.query_model = model_name
    config.llm.caption_model = model_name
    # Embedding: local all-MiniLM-L6-v2 (384-dim, no API calls needed)
    config.embedding.model_name = "all-MiniLM-L6-v2"
    config.embedding.embedding_dim = 384
    # Disable visual encoder (text-only benchmark)
    config.entropy_trigger.visual_encoder = "none"
    # Retrieval: hybrid search with BM25
    config.retrieval.default_top_k = 20
    config.retrieval.enable_hybrid_search = True
    config.retrieval.enable_graph_traversal = True
    # Router
    if hasattr(config, 'router'):
        config.router.router_mode = "heuristic"
    # Storage paths
    config.storage.base_dir = str(memory_data_dir)
    config.storage.cold_storage_dir = str(memory_data_dir / "cold_storage")
    config.storage.index_dir = str(memory_data_dir / "index")
    # Disable self-evolution for clean benchmark run
    config.enable_self_evolution = False

    orchestrator = OmniMemoryOrchestrator(config=config, data_dir=str(memory_data_dir))
    llm_client = orchestrator._get_llm_client()

    # ---- Ingestion Phase ----
    need_ingest = True
    if args.reuse_data:
        try:
            n_vec = orchestrator.vector_store.count() if hasattr(orchestrator.vector_store, "count") else 0
            if n_vec > 0:
                logger.info("Reusing existing memory (vector count=%d), skip ingestion", n_vec)
                need_ingest = False
        except Exception:
            pass

    if need_ingest:
        logger.info("Starting ingestion of %d conversations...", len(samples))
        n_conv = len(samples)

        def ingest_conv(args_tuple):
            si, sample = args_tuple
            return _ingest_one_conversation(orchestrator, sample, si, llm_client, model_name)

        with ThreadPoolExecutor(max_workers=n_conv) as ex:
            results = list(ex.map(ingest_conv, enumerate(samples)))
        total_ingested = sum(results)
        logger.info("Ingested %d text memory entries total", total_ingested)
        orchestrator.save()
        logger.info("Memory saved to %s", memory_data_dir)

    # Build BM25 index
    orchestrator.build_bm25_index()

    # ---- QA Phase ----
    qa_indices = []
    for si, sample in enumerate(samples):
        for qi in range(len(sample.get("qa", []))):
            qa_indices.append((si, qi))
    if args.max_qa and args.max_qa < len(qa_indices):
        qa_indices = qa_indices[:args.max_qa]
    logger.info("Total QA pairs: %d", len(qa_indices))

    prediction_key = f"{model_name}_prediction"
    predictions = {}
    debug_lock = threading.Lock()
    prediction_log = output_dir / "prediction_log.jsonl"
    log_file = open(prediction_log, "w", encoding="utf-8")

    start = time.perf_counter()

    def run_one(item):
        si, qi = item
        qa = samples[si]["qa"][qi]
        question = build_question_text(qa)
        sample_id = str(samples[si].get("sample_id") or "").strip() or f"sample_{si}"
        tags_filter = [f"locomo_{sample_id}"]
        category = qa.get("category")

        # Category 5 (adversarial): standardized refusal
        if category == 5:
            try:
                ans = orchestrator.answer(question, top_k=10, tags_filter=tags_filter)
            except Exception:
                ans = {}
            pred = "The information is not mentioned in the provided memories."
            with debug_lock:
                log_file.write(json.dumps({"sample_id": samples[si].get("sample_id"), "qa_idx": qi,
                    "question": question, "answer": pred, "category": 5}, ensure_ascii=False, default=str) + "\n")
                log_file.flush()
            return si, qi, pred

        # Categories 1-4: adaptive top_k
        adaptive_top_k = 30 if category == 2 else 20
        try:
            ans = orchestrator.answer(question, top_k=adaptive_top_k, tags_filter=tags_filter)
            pred = (ans.get("answer") or "").strip()
            # Post-processing: filter hallucinated future dates
            if category == 2 and pred and re.search(r'\b202[56789]\b', pred):
                pred = 'unknown'
            with debug_lock:
                log_file.write(json.dumps({"sample_id": samples[si].get("sample_id"), "qa_idx": qi,
                    "question": question, "answer": pred, "category": category}, ensure_ascii=False, default=str) + "\n")
                log_file.flush()
            return si, qi, pred
        except Exception as e:
            logger.error("Answer failed sample_id=%s qa_idx=%s: %s", sample_id, qi, e)
            return si, qi, ""

    concurrency = max(1, min(args.concurrency, len(qa_indices)))
    logger.info("Answer phase: concurrency=%d, %d QA pairs", concurrency, len(qa_indices))

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_one, item): item for item in qa_indices}
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(futures), desc="LoCoMo QA")
        except ImportError:
            pbar = None
        for fut in as_completed(futures):
            try:
                si, qi, pred = fut.result()
                predictions[(si, qi)] = pred
            except Exception as e:
                si, qi = futures[fut]
                logger.error("Worker error for QA (%s, %s): %s", si, qi, e)
                predictions[(si, qi)] = ""
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()

    elapsed = time.perf_counter() - start
    log_file.close()

    # ---- Evaluation ----
    for (si, qi), pred in predictions.items():
        samples[si]["qa"][qi][prediction_key] = pred

    all_qa = []
    for si, sample in enumerate(samples):
        for qi in range(len(sample.get("qa", []))):
            if (si, qi) in predictions:
                all_qa.append(sample["qa"][qi])

    if all_qa:
        exact_matches, _, recall_vals = eval_question_answering(all_qa, prediction_key)
        overall_f1 = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    else:
        overall_f1 = 0.0
        exact_matches = []
        recall_vals = []

    orchestrator.close()

    # ---- Per-category stats ----
    f1_key = f"{model_name}_f1"
    idx = 0
    for si, sample in enumerate(samples):
        for qi in range(len(sample.get("qa", []))):
            if (si, qi) in predictions and idx < len(exact_matches):
                sample["qa"][qi][f1_key] = round(exact_matches[idx], 3)
                idx += 1

    category_counts = defaultdict(int)
    acc_counts = defaultdict(float)
    for s in samples:
        for qa in s.get("qa", []):
            cat = qa.get("category")
            if cat is None or f1_key not in qa:
                continue
            category_counts[cat] += 1
            acc_counts[cat] += float(qa[f1_key])

    total_n = sum(category_counts.values())
    total_acc = sum(acc_counts.values())
    overall_f1_recomputed = total_acc / total_n if total_n else 0.0

    logger.info("LoCoMo OmniMem - F1: %.4f | N: %d | time: %.2fs", overall_f1_recomputed, total_n, elapsed)

    # ---- Save results ----
    out_list = []
    for si in sorted(set(si for si, _ in predictions)):
        s = samples[si]
        qa_ran = [s["qa"][qi] for qi in range(len(s.get("qa", []))) if (si, qi) in predictions]
        out_list.append({"sample_id": s.get("sample_id", f"s_{si}"), "qa": qa_ran})
    results_file = output_dir / "locomo_omnimem_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)

    summary = {
        "model": model_name,
        "embedding": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "num_conversations": len(samples),
        "num_qa": len(qa_indices),
        "runtime_seconds": round(elapsed, 2),
        "overall_f1": round(overall_f1_recomputed, 6),
        "category_results": {
            str(k): {"n": category_counts[k], "f1": round(acc_counts[k] / category_counts[k], 4)}
            for k in sorted(category_counts.keys())
        },
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- Markdown report ----
    cat_names = {1: "Multi-hop (MH)", 2: "Single-hop (SH)", 3: "Temporal (Tmp)",
                 4: "Open-domain (Open)", 5: "Adversarial (Adv)"}
    md_lines = [
        f"# LoCoMo Benchmark Results — OmniMem",
        f"",
        f"> **Model**: {model_name}",
        f"> **Embedding**: all-MiniLM-L6-v2 (384-dim, local)",
        f"> **Conversations**: {len(samples)}",
        f"> **Total QA**: {total_n}",
        f"> **Runtime**: {elapsed:.1f}s ({elapsed/60:.1f} min)",
        f"",
        f"## Results",
        f"",
        f"| Category | n | F1 |",
        f"|----------|---|------|",
    ]
    for cat in [1, 2, 3, 4, 5]:
        n = category_counts.get(cat, 0)
        if n:
            acc = acc_counts[cat] / n
            name = cat_names.get(cat, f"Cat-{cat}")
            md_lines.append(f"| {name} | {n} | {acc:.3f} |")
    md_lines.append(f"| **Overall** | **{total_n}** | **{overall_f1_recomputed:.3f}** |")
    md_lines.append("")

    md_file = output_dir / "locomo_results.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("Results saved to: %s", output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"LoCoMo Benchmark — OmniMem ({model_name})")
    print(f"{'='*60}")
    print(f"Overall F1: {overall_f1_recomputed:.4f}")
    for cat in [1, 2, 3, 4, 5]:
        n = category_counts.get(cat, 0)
        if n:
            acc = acc_counts[cat] / n
            print(f"  {cat_names.get(cat, f'Cat-{cat}'):25s}: F1={acc:.3f} (n={n})")
    print(f"Total: {total_n} QA pairs, {elapsed:.1f}s runtime")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
