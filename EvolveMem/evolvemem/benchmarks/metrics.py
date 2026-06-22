"""Multi-metric evaluation utilities for all benchmarks.

Each metric is optional — if the required library is missing, that metric
yields None (recorded explicitly so we know it's missing vs. zero).

Primary/heavyweight metrics (BERTScore, METEOR, SBERT) are opt-in because
they slow per-question eval by 100–1000× compared to F1.

Benchmark adapters declare which metrics they compute by overriding
`BenchmarkAdapter.all_metrics`.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Callable

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tok(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(str(text or ""))]


# ── Core text metrics (stdlib-only fallbacks) ──

def f1_token(pred: str, ref: str) -> float:
    """Token-overlap F1 (set-based, matches SimpleMem paper)."""
    p, r = set(_tok(pred)), set(_tok(ref))
    if not p or not r:
        return 0.0
    common = p & r
    if not common:
        return 0.0
    precision = len(common) / len(p)
    recall = len(common) / len(r)
    return 2 * precision * recall / (precision + recall)


def f1_multiset(pred: str, ref: str) -> float:
    """Token-overlap F1, multiset (matches older EvolveMem evolution.py)."""
    p, r = _tok(pred), _tok(ref)
    if not p or not r:
        return 0.0
    rc = list(r)
    c = 0
    for t in p:
        if t in rc:
            c += 1
            rc.remove(t)
    if c == 0:
        return 0.0
    precision = c / len(p)
    recall = c / len(r)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _tok(pred) == _tok(ref) else 0.0


def contains_match(pred: str, ref: str) -> float:
    """1.0 if reference tokens are all contained in prediction."""
    p, r = set(_tok(pred)), set(_tok(ref))
    if not r:
        return 0.0
    return 1.0 if r <= p else 0.0


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


def bleu_n(pred: str, ref: str, n: int = 1) -> float:
    """Smoothed BLEU-n without nltk dependency. Brevity penalty applied."""
    p_toks = _tok(pred)
    r_toks = _tok(ref)
    if not p_toks or not r_toks:
        return 0.0
    p_ng = _ngrams(p_toks, n)
    r_ng = _ngrams(r_toks, n)
    if not p_ng:
        return 0.0
    from collections import Counter
    rc = Counter(r_ng)
    clipped = 0
    pc = Counter()
    for g in p_ng:
        pc[g] += 1
    for g, cnt in pc.items():
        clipped += min(cnt, rc.get(g, 0))
    # smoothing: add 1/(2^k) where k is number of zero n-gram overlaps
    if clipped == 0:
        clipped = 0.5
    precision = clipped / len(p_ng)
    # brevity penalty
    bp = 1.0 if len(p_toks) >= len(r_toks) else math.exp(1 - len(r_toks) / max(1, len(p_toks)))
    return precision * bp


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F-score via LCS."""
    p, r = _tok(pred), _tok(ref)
    if not p or not r:
        return 0.0
    # LCS table
    m, n = len(p), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


# ── Optional heavyweight metrics (lazy imports) ──

@lru_cache(maxsize=1)
def _sbert():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def sbert_similarity(pred: str, ref: str) -> float | None:
    model = _sbert()
    if model is None or not pred or not ref:
        return None
    try:
        a = model.encode([pred], normalize_embeddings=True)
        b = model.encode([ref], normalize_embeddings=True)
        return float((a @ b.T)[0][0])
    except Exception:
        return None


def rougel_scorer(pred: str, ref: str) -> float | None:
    try:
        from rouge_score import rouge_scorer
        s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(ref, pred)
        return float(s["rougeL"].fmeasure)
    except Exception:
        return None


# ── MCQ metrics ──

_LETTER_RE = re.compile(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)")


def mcq_accuracy(pred: str, gt_letter: str) -> float:
    gt = str(gt_letter).strip().upper()
    if not gt:
        return 0.0
    m = _LETTER_RE.search(str(pred).upper())
    return 1.0 if (m and m.group(1) == gt) else 0.0


def mcq_accuracy_with_text(pred: str, gt_letter: str, choices: dict[str, str]) -> float:
    """MCQ accuracy that also accepts full answer text (partial credit)."""
    if mcq_accuracy(pred, gt_letter) > 0:
        return 1.0
    gt_text = choices.get(gt_letter, "") if isinstance(choices, dict) else ""
    if not gt_text:
        return 0.0
    pt = set(_tok(pred))
    gt_t = set(_tok(gt_text))
    if not gt_t:
        return 0.0
    return 1.0 if gt_t <= pt else 0.0


# ── LLM-as-judge (LongMemEval style, optional) ──

JUDGE_PROMPT = """You are a strict grader. Given a question, a gold answer, and a predicted answer, decide if the prediction is correct.

Question: {question}
Gold answer: {reference}
Predicted answer: {prediction}

Return ONLY one token: YES if the prediction conveys the same factual content as the gold answer (minor paraphrase ok), otherwise NO."""


def llm_judge(
    prediction: str,
    reference: str,
    question: str,
    llm_call: Callable | None,
) -> float | None:
    if llm_call is None:
        return None
    try:
        resp = llm_call(
            [
                {"role": "system", "content": "You are a strict grader. Answer YES or NO only."},
                {"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, reference=reference, prediction=prediction,
                )},
            ],
            8, 0.0,
        )
        if not resp:
            return None
        return 1.0 if "yes" in resp.strip().lower() else 0.0
    except Exception:
        return None


# ── Bundles by benchmark ──

def compute_text_metrics(pred: str, ref: str, want_sbert: bool = False) -> dict[str, float]:
    """Cheap-to-compute text metrics always runnable."""
    out = {
        "f1": f1_multiset(pred, ref),
        "f1_set": f1_token(pred, ref),
        "exact_match": exact_match(pred, ref),
        "contains": contains_match(pred, ref),
        "rouge_l": rouge_l(pred, ref),
        "bleu_1": bleu_n(pred, ref, n=1),
        "bleu_4": bleu_n(pred, ref, n=4),
    }
    rs = rougel_scorer(pred, ref)
    if rs is not None:
        out["rouge_l_official"] = rs
    if want_sbert:
        sb = sbert_similarity(pred, ref)
        if sb is not None:
            out["sbert_sim"] = sb
    return out


def compute_mcq_metrics(pred: str, gt_letter: str, choices: dict) -> dict[str, float]:
    return {
        "accuracy": mcq_accuracy(pred, gt_letter),
        "accuracy_text": mcq_accuracy_with_text(pred, gt_letter, choices or {}),
    }
