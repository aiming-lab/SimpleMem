"""Base interfaces for benchmark adapters.

All benchmarks speak the same language to `EvolutionEngine`:
  - sessions: list[(session_id, date_str, turns)], turn = {speaker, text, ...}
  - qa_pairs: list[dict] with 'question', 'answer', 'category' (int), plus
              adapter-specific extras under 'meta'
  - scoring_fn: (pred, ref, qa) -> float
  - answer_prompt_builder: (question, context, qa) -> list[{role, content}]
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

_REGISTRY: dict[str, type["BenchmarkAdapter"]] = {}


def register_adapter(name: str, cls: type["BenchmarkAdapter"]) -> None:
    _REGISTRY[name] = cls


def get_adapter(name: str) -> "BenchmarkAdapter":
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown benchmark '{name}'. known={list(_REGISTRY)}"
        )
    return _REGISTRY[name]()


@dataclass
class QuestionMeta:
    """Adapter-specific metadata that travels with each QA pair."""
    qid: str = ""
    qtype: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSample:
    """One evaluation unit: a set of sessions + its QA pairs.

    For LoCoMo each conversation is one sample.
    For LongMemEval each question+haystack is one sample.
    For MemBench each tid is one sample.
    """
    sample_id: str
    sessions: list[tuple[str, str, list[dict]]]
    qa_pairs: list[dict]  # each: {question, answer, category, meta:QuestionMeta}
    benchmark: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    """Adapter that normalizes one benchmark to the common format."""

    name: str = "abstract"
    primary_metric: str = "f1"  # used by evolution decisions

    # Optional label for subcategory dimension(s) to aggregate on. Most
    # benchmarks have a single dimension (category/qtype) but MemBench has
    # two (agent × category). Set to list of meta.extras keys plus "category".
    subcategory_keys: tuple[str, ...] = ("category",)

    @abstractmethod
    def load(self, path: str, **kwargs) -> list[BenchmarkSample]:
        """Load raw data from `path` into a list of BenchmarkSample."""
        raise NotImplementedError

    @abstractmethod
    def score(self, prediction: str, reference: str, qa: dict) -> float:
        """Score one prediction in [0,1] — the primary metric."""
        raise NotImplementedError

    def score_all(
        self,
        prediction: str,
        reference: str,
        qa: dict,
    ) -> dict[str, float | None]:
        """Compute the full metric bundle for this benchmark. Subclasses
        override. Default = single-entry {primary_metric: self.score(...)}.
        """
        return {self.primary_metric: self.score(prediction, reference, qa)}

    def subcategory_of(self, qa: dict) -> tuple[str, ...]:
        """Return the subcategory tuple used for per-subcategory aggregation.

        Looks up `subcategory_keys` on the QA meta; falls back to top-level
        `category` if the key isn't present.
        """
        parts: list[str] = []
        meta = qa.get("meta") or {}
        extras = meta.get("extras") if isinstance(meta, dict) else {}
        for k in self.subcategory_keys:
            if k == "category":
                parts.append(str(qa.get("category", 0)))
            elif isinstance(meta, dict) and k in meta:
                parts.append(str(meta[k]))
            elif isinstance(extras, dict) and k in extras:
                parts.append(str(extras[k]))
            else:
                parts.append("")
        return tuple(parts)

    def build_answer_prompt(
        self,
        question: str,
        context: str,
        qa: dict,
    ) -> tuple[str, str]:
        """Return (system, user) prompt for answering one question.
        Default = token-F1-friendly concise style; override per benchmark.
        """
        system = (
            "Professional Q&A assistant. Concise answers grounded in context. "
            "JSON output only."
        )
        user = (
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Rules:\n"
            "1. Answer in 1-10 words. Use exact words from context.\n"
            "2. Be specific (e.g., 'Sweden', not 'her country').\n"
            "3. Return JSON: {\"reasoning\":\"brief\",\"answer\":\"concise\"}"
        )
        return system, user

    def canonical_category(self, qa: dict) -> int:
        """Map adapter-specific qtype to a small int category for diagnostics."""
        return int(qa.get("category", 0))


# ---- Reusable scoring primitives ----

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "once": "1", "twice": "2", "thrice": "3", "single": "1", "double": "2",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
}


def _normalize_numbers(tokens: list[str]) -> list[str]:
    """Map number-words to digit form so 'twice' and '2' token-match."""
    return [_NUMBER_WORDS.get(t, t) for t in tokens]


def _tokenize(s: str) -> list[str]:
    return _normalize_numbers(
        _PUNCT_RE.sub(" ", str(s).lower()).split()
    )


def token_f1(prediction: str, reference: str) -> float:
    p, r = _tokenize(prediction), _tokenize(reference)
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
    pr = c / len(p)
    rec = c / len(r)
    return 2 * pr * rec / (pr + rec)


def multiple_choice_match(prediction: str, ground_truth_letter: str) -> float:
    """Match a predicted multiple-choice letter/answer against the gold letter.

    Accepts:
      - bare letter ("A", "B)", "(C)", "The answer is D")
      - the full answer text (match by exact substring of any choice)
    Returns 1.0 on match, 0.0 otherwise.
    """
    gt = str(ground_truth_letter).strip().upper()
    if not gt:
        return 0.0
    pred = str(prediction).strip()
    if not pred:
        return 0.0
    # Case 1: first alphabetic letter mentioned inside a word-boundary/paren
    m = re.search(r"(?:^|[^A-Za-z])([A-Z])(?:[^A-Za-z]|$)", pred.upper())
    if m and m.group(1) == gt:
        return 1.0
    return 0.0


def accuracy_with_choices(
    prediction: str,
    gt_letter: str,
    choices: dict[str, str],
) -> float:
    """MCQ accuracy that also accepts prediction of the full answer text."""
    if multiple_choice_match(prediction, gt_letter) > 0:
        return 1.0
    gt_text = choices.get(gt_letter, "")
    if gt_text:
        pred_tokens = set(_tokenize(prediction))
        gt_tokens = set(_tokenize(gt_text))
        if gt_tokens and pred_tokens >= gt_tokens:
            return 1.0
    return 0.0
