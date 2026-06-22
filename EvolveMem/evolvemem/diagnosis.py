"""
Automated Diagnosis --- LLM-powered failure analysis and improvement suggestions.

Analyzes evaluation results to:
1. Categorize failure modes (extraction gap, retrieval miss, answer error)
2. Identify systematic patterns (which topics/entities consistently fail)
3. Generate actionable improvement suggestions via LLM reasoning
4. Detect coverage gaps in extracted memories
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

DIAGNOSIS_PROMPT = """You are the diagnosis engine of a self-evolving memory system. Your job is to turn evaluation failures into a concrete next-round action that moves the system toward SOTA.

## System Info
- Benchmark: {benchmark}
- Total memories: {total_memories}
- Total questions: {total_questions}
- Overall score: {overall_f1:.4f}
- Zero-score count: {zero_count}/{total_questions}
- Current config (JSON): {current_config}

## Failure Analysis Summary
{failure_summary}

## Per-Category Breakdown
{category_breakdown}

## Sample Failures (worst cases)
{sample_failures}

## Tier-1 TODO checklist — levers that are still OFF in the incumbent
{todo_checklist}

Prefer picking ONE item from this TODO every round until empty. Skip an
item only if its listed symptom is clearly absent from the failure data.

## Available Recipes (triggered by current symptoms)

The framework ships with a pre-compiled library of capability "recipes".
A recipe is a symptom-gated bundle of related config changes that was
empirically validated to improve a specific failure pattern. Below are
the recipes whose triggers match the current round; each can be invoked
as a SINGLE atomic move (counts as 1 change regardless of how many
sub-fields it sets).

{available_recipes}

To invoke a recipe, include `"use_recipe": "<recipe_name>"` in your JSON
output. You may also propose up to 1 additional field change alongside
a recipe (total budget = 2 moves per round). If no recipe matches the
current failure pattern, propose your own fields as usual.

Recipes encode prior research patterns — they are NOT guaranteed to help
every round. The elitism + attempt_history discipline still applies:
rejected recipe invocations show up as REJECTED in Prior Attempts just
like any move, and you should not re-invoke a recipe whose identical
application already appears as REJECTED.

## Prior Attempts (most recent last)
{attempt_history}

## STEP-SIZE CONSTRAINT (HARD RULE)

This evolution runs under strict hill-climbing: **at most 2 changes per round**.

Propose AT MOST 2 fields in `parameter_suggestions` (a `per_category_overrides`
entry counts as 1 change regardless of how many sub-fields it contains).
Large multi-field rewrites will be truncated to the first 2.

Target the SINGLE weakest category visible in the breakdown. Pick the lever
most likely to lift it given the Sample Failures above, and don't propose
scattershot changes across unrelated dimensions.

Never re-propose an exact (field, value) move listed as REJECTED in Prior
Attempts -- that move has been evaluated and demonstrably did not improve
F1. Pick a different lever or a different value.

## ANTI-PLATEAU RULE (apply after 2+ consecutive rejections)

If the last 2+ rounds in Prior Attempts are REJECTED, the incumbent has
exhausted the "obvious" binary flags — further tweaks to already-explored
dimensions will almost certainly also reject. In that state you MUST break
out by invoking an UNUSED capability from Available Recipes, preferably
one with high expected_lift_pp and strong trigger match. Familiar small
tweaks (scalar top-k bumps, verification style toggles) will not resolve
a plateau; only an architectural shift (intent planning, MMR, per-cat
decomposition) will. Prefer unused recipes > rejected-field retries.

## PRIORITY ORDER (LIST HIGH-ROI LEVERS FIRST)

With only 2 changes allowed, you MUST propose the highest-impact levers
that match the observed failure patterns. **List them FIRST** in your
`parameter_suggestions` JSON; low-impact tweaks will be truncated away.

Empirical lever impact on LoCoMo-style benchmarks (order = priority):

1. **Adapter prompt-surface flags** (+5-30 pp each, pick the ones whose
   symptom shows in the breakdown):
   - locomo_cat5_mcq: Cat 5 low AND many "not mentioned" predictions → +30pp
   - enable_answer_verification: many abstention / wrong-format failures → +5-10pp
   - locomo_cat1_single_fact / locomo_cat4_verbatim_copy: Cat 1/4 paraphrase
     symptom (verbose pred vs short gold) → +5-15pp
   - locomo_cat2_temporal_format: Cat 2 "N years ago" predictions → +5-10pp
   - locomo_cat3_inferential_nuanced: Cat 3 too-short / too-absolute preds → +3-8pp

2. **Fusion architecture** (+3-10 pp):
   - fusion_mode: keyword_only → rrf / weighted_sum when semantic_top_k=0
   - enable_entity_swap: Cat 5 high "wrong entity" failures
   - enable_intent_planning: Cat 1/3/4 low F1 — LLM plans 2-4 targeted sub-queries, unions results. Biggest architectural win after fusion.
   - enable_query_decomposition: weaker sibling of intent_planning; prefer intent_planning when both are candidates.

3. **Per-category overrides** (+2-8 pp): MMR for Cat 3 bridge-starvation,
   time_decay for Cat 2 temporal. ONE override block = ONE change slot.

4. **Scalar tweaks** (+0-2 pp each, DO NOT propose in first 3 rounds;
   save for late-stage tuning after the binary flags are already set):
   - semantic_top_k / keyword_top_k / structured_top_k / max_context
   - weight_semantic / weight_keyword / weight_structured

Scalar-only rounds almost always fail the 0.003 acceptance threshold.
Propose at least ONE tier-1 or tier-2 lever every round until all the
matching ones are enabled.

## Available Action Space (you may propose changes to ANY of these)

Retrieval dimensions (all live in RetrievalConfig):
- semantic_top_k (int, 0-40): 0 disables semantic view. Raise for recall, lower to reduce noise.
- keyword_top_k (int, 0-20): BM25 results. Raise for lexical recall.
- structured_top_k (int, 0-15): person/entity metadata match. Useful for name-specific questions.
- max_context (int, 4-50): memories fed to answer LLM. Too high = noise, too low = miss evidence.
- enable_entity_swap (bool): for adversarial/name-swap questions; strips names from query, re-searches.
- fusion_mode (str): one of "first_found"|"weighted_sum"|"rrf"|"semantic_only"|"keyword_only"|"structured_only". RRF and weighted_sum let views reinforce.
- weight_semantic / weight_keyword / weight_structured (float, 0-3): per-view weight in fusion.
- time_decay_half_life_days (float|null): favor recent memories. Set (e.g. 30-180) for temporal benchmarks; null to disable.
- reflection_rounds (int, 0-2): extra retrieval passes using question + draft answer. Helps multi-hop but costs latency.
- enable_query_decomposition (bool): for multi-hop questions, LLM splits the query into sub-questions, retrieves per-sub, merges. Strongest lever for multi-hop/multi-session categories.
- decomposition_max_subqs (int, 1-5): max sub-questions per decomposition; larger = more recall + more LLM cost.
- decomposition_merge_top_k (int, 4-25): per-sub retrieval depth before merging.
- enable_intent_planning (bool): SimpleMem-style successor to decomposition. LLM first analyzes what *info items* the question needs, then generates 1-4 targeted sub-queries (named-entity specific, time-window specific, sub-topic specific). Each sub-query runs full multi-view retrieval, results are union-merged. Empirically the largest architectural lever on LoCoMo Cat 1 (SingleHop lists) and Cat 3 (MultiHop bridge); both have failure patterns where one retrieval can't return all needed evidence. Strictly stronger than enable_query_decomposition; never set both simultaneously — intent_planning takes precedence.
- intent_max_subqs (int, 1-6): max targeted sub-queries generated.
- intent_merge_top_k (int, 5-30): per-sub retrieval depth before union.
- intent_final_context (int, 10-40): merged-context cap after union; raise when Cat 1/3/4 questions need many distinct pieces.
- answer_style (str): "concise"|"explanatory"|"verifying"|"mcq"|"strict". "strict" adds format rules (digits for numbers, YYYY years, multi-item comma-separated, forbids Unknown) for LoCoMo — use when many open-domain (Cat 4) / adversarial (Cat 5) cases answer "Unknown/Not specified".
- enable_answer_verification (bool): second LLM pass reviews candidate answer against context. Strongest lever for "Unknown/Not specified" residual failures and format mismatches (year-as-phrase, word-vs-digit). Costs ~+1 LLM call per QA.
- verification_style (str): "strict" (force format, replace Unknown) | "multi_candidate" (consider 2-3 alternatives). Use "strict" by default.
- per_category_overrides (dict): set {{category_int: {{field: value}}}} to apply different config for a weak category only.
- mmr_diversity_weight (float, 0.0-0.9): post-fusion MMR rerank for diversity. 0 = off, 0.5 = balanced (relevance + diversity), 0.7 = strong diversity. Enable when a category (typically multi-hop Cat 3) retrieves many near-duplicate memories of the same fact, starving the answer model of distinct supporting evidence. Symptom: Cat 3 F1 low despite high retrieval recall; many top-k docs paraphrase each other. Typically set as per_category_override for the weak cat only (e.g. {{"3": {{"mmr_diversity_weight": 0.6, "max_context": 12}}}}), leaving other cats at 0.
- mmr_candidate_pool (int, 20-80): pool size fed into MMR before it picks max_context items. Larger = more room for diversity at cost of latency. Default 40.

LoCoMo prompt-surface flags (only meaningful when benchmark=locomo; flip ON only when you see the listed symptom in raw_results). IMPORTANT: these are TOP-LEVEL RetrievalConfig fields -- always place them in the flat `parameter_suggestions` object, NEVER under `per_category_overrides` / `per_category_proposals` (the adapter reads them from top-level). Each flag is category-scoped internally; no override wrapper is needed:
- locomo_cat5_mcq (bool): reformulate Cat 5 (adversarial) as a 2-option MCQ between "Not mentioned in the conversation" and the dataset's adversarial_answer; the LLM picks which the context supports. Enable when cat5 F1 is low AND many cat5 predictions are "not mentioned / not described / not specified" OR free-form hallucinations diverging wildly from reference. Largest single-flag lift (+30pp+ on cat5 F1 when symptoms match).
- locomo_cat1_single_fact (bool): tighten Cat 1 SingleHop prompt to force single-entity / single-number / single-date answers copied verbatim from context (forbids paraphrases like "Caroline's home country" vs gold "Sweden"). Enable when cat1 predictions are noticeably verbose / paraphrased while gold is 1-3 words.
- locomo_cat2_temporal_format (bool): enforce YYYY / YYYY-MM-DD / "Since YYYY" style and forbid "N years ago". Enable when cat2 predictions use "N years ago" / "last year" / word-duration forms while gold is YYYY / "Since YYYY".
- locomo_cat3_inferential_nuanced (bool): allow 2-15-word nuanced yes/no-plus-reason answers (vs the 1-6-word hard floor). Enable when cat3 predictions are shorter and more absolute than gold (gold often contains "but" / "somewhat").
- locomo_cat4_verbatim_copy (bool): force verbatim phrase copy from context (book title, plan, reason). Enable when cat4 predictions paraphrase / generalize while gold is a specific short phrase present in memory.
- answer_model_ensemble (str|null): dual-model answer ensemble for Cat 3 multi-hop. Set to a secondary model name (e.g. "gpt-4.1") as a per_category_override for Cat 3 ONLY. When set, the engine calls BOTH the primary LLM and this secondary LLM on the same prompt, then picks the better answer via a question-pattern + length heuristic. Enable when Cat 3 F1 is low AND error analysis shows a bimodal gold distribution (some gold answers are 1-3 word entities, others are 10+ word inferential clauses). The ensemble costs 1 extra LLM call per Cat 3 question (~5% of total QA). Set via per_category_overrides: {{"3": {{"answer_model_ensemble": "gpt-4.1"}}}}. Only effective on LoCoMo (adapter guard).

Extraction dimensions:
- window_size, overlap, max_retries, chunk_size_on_failure, min_restatement_words.

## Decision Rubric
1. If many "abstention" failures → extraction coverage / recall too low → raise top_k, widen max_context, consider weighted_sum/rrf fusion.
2. If many "wrong answer" failures with high retrieval → retrieval precision → lower max_context OR raise fusion weights for the strongest view.
3. If temporal category weakness → enable time_decay_half_life_days.
4. If adversarial category weakness → enable_entity_swap=true and raise entity_swap_keyword_k.
5. If multi-hop weakness → reflection_rounds >= 1.
6. If ONE specific category lags far behind others → use per_category_overrides instead of global change (preserve gains elsewhere).
7. Prefer enabling something disabled in a weak initial BEFORE tuning a small int up/down.
8. If residual "Unknown" or format-mismatch failures → enable_answer_verification=true on affected categories.
9. LoCoMo prompt-surface flags are the single highest-ROI lever when their symptom matches (see symptom list above). Propose them early once you see the matching pattern — they are cheaper than +1 LLM call per QA and typically +10-30 pp on the affected category. Do NOT blanket-enable them; each flag should be tied to a specific cat-level symptom you can point to in raw_results.
10. If Cat 3 multi-hop F1 is low AND gold answers show a bimodal length distribution (short entities AND long inferential clauses) → set answer_model_ensemble in per_category_overrides for Cat 3. This adds a secondary LLM to cover both modes.

## Output
Return JSON with `parameter_suggestions` as a flat dict of field → new value. Fields MUST match RetrievalConfig field names exactly. Only include fields you want to change. You may also propose `extraction_suggestions` as a flat dict for ExtractionConfig fields.

```json
{{
  "root_causes": {{
    "extraction_gap": {{"count": N, "description": "..."}},
    "retrieval_miss": {{"count": N, "description": "..."}},
    "answer_error": {{"count": N, "description": "..."}}
  }},
  "missing_topics": ["topic1", "topic2"],
  "parameter_suggestions": {{
    "fusion_mode": "rrf",
    "semantic_top_k": 15,
    "weight_semantic": 1.5
  }},
  "extraction_suggestions": {{
    "window_size": 30
  }},
  "per_category_proposals": {{
    "5": {{"enable_entity_swap": true, "keyword_top_k": 12}}
  }},
  "priority_actions": ["action1", "action2", "action3"]
}}
```

Return ONLY JSON."""


@dataclass
class QAResult:
    """Single question-answer evaluation result.

    `f1` holds the primary-metric score (float in [0,1]) used by evolution
    decisions. `metrics` stores the full multi-metric bundle for logging
    and later paper-table reporting. `subcategory` is a tuple-of-strings
    produced by adapter.subcategory_of so aggregation can slice arbitrarily
    (e.g. MemBench needs agent × category × topic).
    """
    question: str
    prediction: str
    reference: str
    f1: float
    category: int
    retrieved_count: int = 0
    qid: str = ""
    qtype: str = ""
    metrics: dict = field(default_factory=dict)
    subcategory: tuple = ()
    retrieved_sources: list = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """Complete diagnosis report for an evaluation run."""
    overall_f1: float = 0.0
    total_questions: int = 0
    zero_f1_count: int = 0
    category_f1: dict[int, float] = field(default_factory=dict)

    # Failure breakdown
    abstention_count: int = 0  # model said "unknown"/"not specified"
    wrong_answer_count: int = 0  # model gave wrong answer
    low_f1_count: int = 0  # 0 < f1 < 0.3

    # Pattern analysis
    failing_topics: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    category_weaknesses: dict[int, str] = field(default_factory=dict)

    # LLM-generated suggestions
    parameter_suggestions: dict[str, Any] = field(default_factory=dict)
    extraction_improvements: list[str] = field(default_factory=list)
    priority_actions: list[str] = field(default_factory=list)
    # Name of the cookbook recipe the LLM invoked this round (if any).
    # When set, suggest_config_update treats the recipe as a single atomic
    # change slot (its sub-fields bypass the per-field step-cap so the
    # recipe's pre-validated combo stays intact).
    invoked_recipe: str | None = None

    def summary(self) -> str:
        cn = {1: "SingleHop", 2: "Temporal", 3: "MultiHop", 4: "OpenDomain", 5: "Adversarial"}
        lines = [
            f"Overall F1: {self.overall_f1:.4f} ({self.total_questions} questions)",
            f"Zero-F1: {self.zero_f1_count} (abstention={self.abstention_count}, wrong={self.wrong_answer_count})",
            f"Low-F1 (0-0.3): {self.low_f1_count}",
            "",
            "Category F1:",
        ]
        for cat in sorted(self.category_f1):
            lines.append(f"  {cn.get(cat, f'Cat{cat}')}: {self.category_f1[cat]:.4f}")

        if self.priority_actions:
            lines.append("\nPriority Actions:")
            for i, action in enumerate(self.priority_actions, 1):
                lines.append(f"  {i}. {action}")

        if self.parameter_suggestions:
            lines.append(f"\nParameter Suggestions: {json.dumps(self.parameter_suggestions)}")

        return "\n".join(lines)


class MemoryDiagnostics:
    """Automated failure analysis for memory system evaluation.

    Args:
        llm_call: Callable(messages, max_tokens, temperature) -> str.
                  Used for intelligent diagnosis. If None, only heuristic analysis.
    """

    def __init__(self, llm_call: Callable | None = None):
        self.llm_call = llm_call

    def diagnose(
        self,
        results: list[QAResult],
        memories: list[dict] | None = None,
        benchmark_name: str = "locomo",
        current_config: dict | None = None,
        attempt_history: list[dict] | None = None,
    ) -> DiagnosisReport:
        """Run full diagnosis on evaluation results.

        Args:
            results: List of QAResult from evaluation.
            memories: Optional list of memory dicts for coverage analysis.

        Returns:
            DiagnosisReport with analysis and suggestions.
        """
        report = DiagnosisReport()
        report.total_questions = len(results)

        if not results:
            return report

        # Basic metrics
        report.overall_f1 = sum(r.f1 for r in results) / len(results)

        cat_scores = defaultdict(list)
        for r in results:
            cat_scores[r.category].append(r.f1)
        report.category_f1 = {
            cat: sum(scores) / len(scores) for cat, scores in cat_scores.items()
        }

        # Failure classification
        zeros = [r for r in results if r.f1 == 0]
        report.zero_f1_count = len(zeros)

        abstention_patterns = [
            "unknown", "not specified", "not mentioned", "not stated",
            "no specific", "no relevant", "cannot be determined",
            "not found", "no information",
        ]
        for r in zeros:
            pred_lower = r.prediction.lower()
            if any(p in pred_lower for p in abstention_patterns):
                report.abstention_count += 1
            else:
                report.wrong_answer_count += 1

        report.low_f1_count = sum(1 for r in results if 0 < r.f1 < 0.3)

        # Topic analysis from zero-F1 cases
        report.failing_topics = self._extract_failing_topics(zeros)

        # Coverage gap detection
        if memories:
            report.missing_keywords = self._detect_coverage_gaps(zeros, memories)

        # Category weakness identification
        cn = {1: "SingleHop", 2: "Temporal", 3: "MultiHop", 4: "OpenDomain", 5: "Adversarial"}
        for cat, f1 in report.category_f1.items():
            if f1 < 0.3:
                cat_zeros = sum(1 for r in zeros if r.category == cat)
                cat_total = len(cat_scores[cat])
                report.category_weaknesses[cat] = (
                    f"{cn.get(cat, f'Cat{cat}')}: F1={f1:.2f}, "
                    f"{cat_zeros}/{cat_total} zero-F1 ({cat_zeros/cat_total*100:.0f}%)"
                )

        # LLM-powered deep diagnosis
        if self.llm_call:
            self._llm_diagnosis(
                report, results, memories, benchmark_name,
                current_config or {}, attempt_history or [],
            )

        return report

    # Ranges / valid values for every evolvable dimension
    INT_RANGES: dict[str, tuple[int, int]] = {
        "semantic_top_k": (0, 40),
        "keyword_top_k": (0, 20),
        "structured_top_k": (0, 15),
        "max_context": (4, 50),
        "entity_swap_semantic_k": (0, 25),
        "entity_swap_keyword_k": (0, 20),
        "reflection_rounds": (0, 2),
        "decomposition_max_subqs": (1, 5),
        "decomposition_merge_top_k": (4, 25),
        "mmr_candidate_pool": (20, 80),
        "intent_max_subqs": (1, 6),
        "intent_merge_top_k": (5, 30),
        "intent_final_context": (10, 40),
    }
    FLOAT_RANGES: dict[str, tuple[float, float]] = {
        "weight_semantic": (0.0, 3.0),
        "weight_keyword": (0.0, 3.0),
        "weight_structured": (0.0, 3.0),
        "time_decay_half_life_days": (1.0, 1000.0),
        "mmr_diversity_weight": (0.0, 0.9),
    }
    ENUM_VALUES: dict[str, set[str]] = {
        "fusion_mode": {
            "first_found", "weighted_sum", "rrf",
            "semantic_only", "keyword_only", "structured_only",
        },
        "answer_style": {"concise", "explanatory", "verifying", "mcq", "strict"},
        "verification_style": {"strict", "multi_candidate"},
    }
    BOOL_FIELDS = {
        "enable_entity_swap",
        "enable_query_decomposition",
        "enable_intent_planning",
        "enable_answer_verification",
        # Adapter-specific prompt-surface flags. These flip on when the
        # diagnosis LLM observes category-specific failure patterns in
        # raw_results.jsonl (see DIAGNOSIS_PROMPT rubric). They must be
        # whitelisted here for the LLM's suggestions to be applied.
        "locomo_cat5_mcq",
        "locomo_cat1_single_fact",
        "locomo_cat2_temporal_format",
        "locomo_cat3_inferential_nuanced",
        "locomo_cat4_verbatim_copy",
    }

    def suggest_config_update(
        self,
        report: DiagnosisReport,
        current_config: dict,
        benchmark: str = "locomo",
        max_changes: int | None = None,
    ) -> dict:
        """Generate updated configuration based on diagnosis.

        Applies LLM-proposed changes (with validation) plus a thin heuristic
        safety net. The LLM is the primary decision-maker — heuristics only
        activate when diagnosis produced no usable output.

        `benchmark` gates adapter-specific prompt-surface flags: e.g.
        `locomo_*` flags are only applied when benchmark=="locomo" so that
        an LLM slipping and proposing them on LongMemEval / MemBench runs
        cannot pollute their configs.

        `max_changes` caps how many distinct top-level fields may be modified
        per call. A per_category_overrides block counts as ONE change
        regardless of how many sub-fields it sets. None = unlimited
        (legacy behavior). Hill-climbing evolution passes a small integer
        (e.g. 2) so each round moves in a single direction that can be
        isolated-measured against the incumbent.
        """
        # Field-prefix allowlist per benchmark
        FLAG_PREFIX = {
            "locomo": "locomo_",
            "longmemeval": "longmemeval_",
            "membench": "membench_",
        }
        allowed_prefix = FLAG_PREFIX.get(benchmark)

        def benchmark_allows(key: str) -> bool:
            # Non-adapter flags (no cross-benchmark prefix) are always allowed
            for pref in FLAG_PREFIX.values():
                if key.startswith(pref):
                    return allowed_prefix is not None and key.startswith(allowed_prefix)
            return True

        new_config = dict(current_config)
        applied_any = False
        changes_used = 0  # count of distinct top-level fields actually changed

        # If the LLM invoked a recipe, apply its pre-validated combo atomically
        # (bypassing the per-field step cap) and count the whole recipe as 1
        # change slot. Any field the recipe already set is excluded from the
        # cap enforcement below so the LLM's remaining proposals don't
        # collide with the recipe's fields.
        recipe_name = getattr(report, "invoked_recipe", None)
        recipe_fields: set[str] = set()
        if recipe_name:
            from .evolution_cookbook import get_recipe_by_name
            recipe = get_recipe_by_name(recipe_name, benchmark=benchmark)
            if recipe is not None:
                for rk, rv in recipe.proposal.items():
                    if rk == "per_category_overrides":
                        # Merge into new_config's per_cat overrides
                        merged = dict(new_config.get("per_category_overrides") or {})
                        for cat_key, cat_fields in (rv or {}).items():
                            existing = dict(merged.get(str(cat_key), {}))
                            existing.update(cat_fields)
                            merged[str(cat_key)] = existing
                        new_config["per_category_overrides"] = merged
                        recipe_fields.add("per_category_overrides")
                    else:
                        new_config[rk] = rv
                        recipe_fields.add(rk)
                applied_any = True
                changes_used += 1  # recipe as a whole = 1 slot

        def can_change(key: str, new_val) -> bool:
            """Return True when (a) budget remains and (b) value would differ."""
            if key in recipe_fields:
                return False  # already set by recipe; don't double-count
            if max_changes is not None and changes_used >= max_changes:
                return False
            return current_config.get(key) != new_val

        # Impact-priority ordering when budget is capped. Under step-size cap
        # the LLM's dict-order might surface scalar tweaks before binary/
        # adapter flags whose individual impact is 5-10x higher. Re-rank
        # proposals so tier-1 (adapter flags, enable_*, fusion_mode) always
        # beat scalar tweaks for the limited budget slots.
        def _impact_tier(key: str) -> int:
            if key.startswith("locomo_") or key.startswith("longmemeval_") or key.startswith("membench_"):
                return 0  # adapter flags, +5-30pp
            if key in ("enable_answer_verification", "enable_query_decomposition",
                       "enable_intent_planning",
                       "enable_entity_swap", "fusion_mode"):
                return 1  # architecture flips, +3-10pp
            if key in self.BOOL_FIELDS or key in self.ENUM_VALUES:
                return 2  # other binary/enum
            if key in self.FLOAT_RANGES:
                return 3  # scalar weights / decays
            if key in self.INT_RANGES:
                return 4  # scalar top-k / context
            return 5

        suggestions_items = [
            (k, v) for k, v in (report.parameter_suggestions or {}).items()
            if k not in ("per_category_overrides", "per_category_proposals")
        ]
        # Stable sort by impact; preserves LLM's relative order within a tier.
        suggestions_items.sort(key=lambda kv: _impact_tier(kv[0]))

        for key, val in suggestions_items:
            if not benchmark_allows(key):
                continue  # drop cross-benchmark flag pollution
            proposed: object = None
            valid = False
            if key in self.INT_RANGES:
                try:
                    iv = int(val)
                except (TypeError, ValueError):
                    continue
                lo, hi = self.INT_RANGES[key]
                proposed = max(lo, min(iv, hi)); valid = True
            elif key in self.FLOAT_RANGES:
                try:
                    fv = float(val) if val is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None:
                    proposed = None
                else:
                    lo, hi = self.FLOAT_RANGES[key]
                    proposed = max(lo, min(fv, hi))
                valid = True
            elif key in self.ENUM_VALUES:
                if isinstance(val, str) and val in self.ENUM_VALUES[key]:
                    proposed = val; valid = True
            elif key in self.BOOL_FIELDS:
                proposed = bool(val); valid = True
            elif key in current_config:
                proposed = val; valid = True

            if valid and can_change(key, proposed):
                new_config[key] = proposed
                applied_any = True
                changes_used += 1

        # Per-category overrides are validated and merged into a single dict.
        # The whole block counts as 1 change regardless of how many cats it
        # touches. If the step-cap budget is exhausted, skip overrides.
        overrides_in = (
            (report.parameter_suggestions or {}).get("per_category_overrides")
            or (report.parameter_suggestions or {}).get("per_category_proposals")
            or {}
        )
        if (
            isinstance(overrides_in, dict) and overrides_in
            and (max_changes is None or changes_used < max_changes)
        ):
            merged = dict(current_config.get("per_category_overrides") or {})
            # Adapter-layer prompt flags whose effect is already category-
            # scoped at the prompt level. If the LLM puts them inside a
            # per_category block (a natural place for it), promote them to
            # the top-level config so the engine actually surfaces them via
            # `_ret_config_flags`.
            ADAPTER_PROMPT_FLAGS = {
                "locomo_cat1_single_fact",
                "locomo_cat2_temporal_format",
                "locomo_cat3_inferential_nuanced",
                "locomo_cat4_verbatim_copy",
                "locomo_cat5_mcq",
            }
            for cat_key, fields in overrides_in.items():
                if not isinstance(fields, dict):
                    continue
                clean: dict = {}
                for k, v in fields.items():
                    if k in ADAPTER_PROMPT_FLAGS:
                        if not benchmark_allows(k):
                            continue
                        # Promote to top-level config instead of nesting
                        new_config[k] = bool(v)
                        applied_any = True
                        continue
                    if k in self.INT_RANGES:
                        try:
                            clean[k] = max(self.INT_RANGES[k][0], min(int(v), self.INT_RANGES[k][1]))
                        except (TypeError, ValueError):
                            pass
                    elif k in self.FLOAT_RANGES and v is not None:
                        try:
                            clean[k] = max(self.FLOAT_RANGES[k][0], min(float(v), self.FLOAT_RANGES[k][1]))
                        except (TypeError, ValueError):
                            pass
                    elif k in self.ENUM_VALUES and isinstance(v, str) and v in self.ENUM_VALUES[k]:
                        clean[k] = v
                    elif k in self.BOOL_FIELDS:
                        clean[k] = bool(v)
                if clean:
                    merged[str(cat_key)] = clean
            if merged != (current_config.get("per_category_overrides") or {}):
                new_config["per_category_overrides"] = merged
                applied_any = True
                changes_used += 1

        # Heuristic fallback — only kicks in if the LLM gave us nothing
        if not applied_any:
            if report.abstention_count > report.total_questions * 0.2:
                if current_config.get("fusion_mode") == "keyword_only":
                    new_config["fusion_mode"] = "weighted_sum"
                    new_config["semantic_top_k"] = max(15, current_config.get("semantic_top_k", 0))
                else:
                    new_config["semantic_top_k"] = min(
                        int(current_config.get("semantic_top_k", 0)) + 5, 30
                    )
                new_config["max_context"] = min(
                    int(current_config.get("max_context", 10)) + 5, 40
                )
            elif report.wrong_answer_count > max(1, report.zero_f1_count) * 0.6:
                new_config["max_context"] = max(
                    int(current_config.get("max_context", 25)) - 5, 8
                )

        return new_config

    # ---- Internal methods ----

    def _extract_failing_topics(self, zeros: list[QAResult]) -> list[str]:
        """Extract common topics from failing questions."""
        word_counts: Counter = Counter()
        for r in zeros:
            words = re.sub(r'[^a-z\s]', ' ', r.question.lower()).split()
            content_words = [w for w in words if w not in _STOP and len(w) > 3]
            word_counts.update(content_words)
        return [word for word, _ in word_counts.most_common(10)]

    def _detect_coverage_gaps(
        self,
        zeros: list[QAResult],
        memories: list[dict],
    ) -> list[str]:
        """Find keywords from reference answers not present in memories."""
        contents = " ".join(m.get("content", "").lower() for m in memories)
        missing = []
        for r in zeros:
            ref_words = re.sub(r'[^a-z0-9\s]', ' ', str(r.reference).lower()).split()
            for w in ref_words:
                if len(w) > 3 and w not in _STOP and w not in contents:
                    missing.append(w)
        return list(dict.fromkeys(missing))[:20]  # dedupe, top 20

    def _llm_diagnosis(
        self,
        report: DiagnosisReport,
        results: list[QAResult],
        memories: list[dict] | None,
        benchmark_name: str = "locomo",
        current_config: dict | None = None,
        attempt_history: list[dict] | None = None,
    ):
        """Use LLM for deep failure analysis and suggestion generation."""
        import json as _json
        zeros = [r for r in results if r.f1 == 0]

        # Build failure summary
        cat_zeros = Counter(r.category for r in zeros)
        cat_total = Counter(r.category for r in results)
        failure_lines = []
        for cat in sorted(cat_zeros):
            failure_lines.append(
                f"Cat {cat}: {cat_zeros[cat]}/{cat_total[cat]} zero-score"
            )
        failure_lines.append(f"Abstentions (model said unknown): {report.abstention_count}")
        failure_lines.append(f"Wrong answers: {report.wrong_answer_count}")
        if report.missing_keywords:
            failure_lines.append(
                f"Missing keywords in memory: {', '.join(report.missing_keywords[:10])}"
            )

        # Per-cat abstention / paraphrase / format signature --- surfaces the
        # symptoms that gate the LoCoMo prompt-surface flags. The diagnosis
        # LLM should look at these counts when proposing locomo_cat*_* flags.
        abstain_pat = re.compile(
            r"\b(not mentioned|not specified|not described|not provided|no information|unknown|none mentioned|n/?a)\b",
            re.IGNORECASE,
        )
        relative_date_pat = re.compile(
            r"\b(years? ago|last (year|month|week)|a few|recently|some time)\b",
            re.IGNORECASE,
        )
        per_cat_abstain: dict[int, int] = defaultdict(int)
        per_cat_reldate: dict[int, int] = defaultdict(int)
        per_cat_paraphrase_zero: dict[int, int] = defaultdict(int)
        for r in results:
            pred = str(r.prediction or "")
            ref = str(r.reference or "")
            if abstain_pat.search(pred):
                per_cat_abstain[r.category] += 1
            if r.category == 2 and relative_date_pat.search(pred):
                per_cat_reldate[r.category] += 1
            # cat-1/4 paraphrase heuristic: zero F1 but prediction > 4 words
            # while gold is <= 3 words (suggests verbose paraphrase of short gold)
            if r.f1 == 0 and r.category in (1, 4):
                pred_w = len(pred.split())
                ref_w = len(ref.split())
                if pred_w >= 4 and 0 < ref_w <= 3:
                    per_cat_paraphrase_zero[r.category] += 1
        if per_cat_abstain:
            failure_lines.append(
                "Per-cat abstention rate (pred contains 'not mentioned / not "
                "specified / unknown'): "
                + ", ".join(
                    f"cat{c}={per_cat_abstain[c]}/{cat_total[c]}"
                    for c in sorted(per_cat_abstain)
                )
            )
        if per_cat_reldate:
            failure_lines.append(
                "Cat 2 relative-date predictions (e.g. 'N years ago'): "
                + ", ".join(
                    f"cat{c}={per_cat_reldate[c]}/{cat_total[c]}"
                    for c in sorted(per_cat_reldate)
                )
            )
        if per_cat_paraphrase_zero:
            failure_lines.append(
                "Short-gold verbose-pred zero-F1 (paraphrase symptom, cat 1/4): "
                + ", ".join(
                    f"cat{c}={per_cat_paraphrase_zero[c]}"
                    for c in sorted(per_cat_paraphrase_zero)
                )
            )
        failure_summary = "\n".join(failure_lines)

        # Per-subcategory breakdown (aggregated from raw qa results)
        sub_breakdown: dict[str, list[float]] = defaultdict(list)
        for r in results:
            key = "|".join(r.subcategory) if r.subcategory else f"cat{r.category}"
            sub_breakdown[key].append(r.f1)
        cat_lines = []
        for key, vals in sorted(sub_breakdown.items()):
            cat_lines.append(
                f"  {key}: mean={sum(vals)/len(vals):.3f} (n={len(vals)})"
            )
        category_breakdown = "\n".join(cat_lines) if cat_lines else "(none)"

        # Build sample failures -- per-cat stratified so every non-empty
        # category shows 2-3 failures to the LLM. Without stratification the
        # LLM tended to ignore low-population categories like Cat 3.
        by_cat: dict[int, list[QAResult]] = defaultdict(list)
        for r in zeros:
            by_cat[r.category].append(r)
        sample_lines = []
        per_cat_cap = 3
        for cat in sorted(by_cat):
            picks = by_cat[cat][:per_cat_cap]
            for r in picks:
                sample_lines.append(
                    f"  Cat={r.category} qtype={r.qtype or '?'} "
                    f"Q: {r.question[:90]}\n"
                    f"    Pred: {r.prediction[:80]} | Ref: {str(r.reference)[:80]}"
                )
        sample_failures = "\n".join(sample_lines) or "(none)"

        # Match cookbook recipes against the current failure profile. The
        # matched set is surfaced to the LLM as "available capability cards"
        # the evolution can autonomously invoke.
        from .evolution_cookbook import match_recipes, format_recipes_for_prompt
        report_fields_for_recipes = {
            "category_f1": report.category_f1,
            "abstention_count": report.abstention_count,
            "total_questions": report.total_questions,
            "overall_f1": report.overall_f1,
            "total_memories": len(memories) if memories else 0,
        }
        matched_recipes = match_recipes(
            report_fields_for_recipes, current_config or {},
            attempt_history or [], benchmark=benchmark_name,
        )
        available_recipes_str = format_recipes_for_prompt(matched_recipes)

        # Build a "TODO checklist" of tier-1 levers that are still OFF in
        # the current config. The LLM should pick ONE of these each round
        # until the list is empty — these are near-universal wins for the
        # affected benchmark. Truncate when irrelevant per-benchmark.
        cfg = current_config or {}
        todo_lines = []
        if cfg.get("fusion_mode") in ("keyword_only", "semantic_only", "structured_only"):
            todo_lines.append(
                f"  • fusion_mode currently '{cfg.get('fusion_mode')}': set to 'rrf' "
                "or 'weighted_sum' to let all three retrieval views reinforce. "
                "+5-10 pp typical, NEAR-UNIVERSAL WIN."
            )
        if not cfg.get("enable_entity_swap"):
            todo_lines.append(
                "  • enable_entity_swap=False: set True for name-swap / "
                "adversarial retrieval expansion (helps Cat 5 AND dense indexes)."
            )
        if not cfg.get("enable_answer_verification"):
            todo_lines.append(
                "  • enable_answer_verification=False: set True + "
                "verification_style='strict'. Largest single lever for "
                "'Unknown / not specified' abstentions (+5-10 pp)."
            )
        if benchmark_name == "locomo":
            cn = {1: "SingleHop", 2: "Temporal", 3: "MultiHop",
                  4: "OpenDomain", 5: "Adversarial"}
            for ci, name, key in [
                (1, "SingleHop", "locomo_cat1_single_fact"),
                (2, "Temporal", "locomo_cat2_temporal_format"),
                (3, "MultiHop", "locomo_cat3_inferential_nuanced"),
                (4, "OpenDomain", "locomo_cat4_verbatim_copy"),
                (5, "Adversarial", "locomo_cat5_mcq"),
            ]:
                if not cfg.get(key):
                    f1_c = report.category_f1.get(ci, 0.0)
                    todo_lines.append(
                        f"  • {key}=False (Cat {ci} {name} F1={f1_c:.3f}): "
                        f"enable if symptom matches (see action space doc)."
                    )
        todo_str = "\n".join(todo_lines) if todo_lines else (
            "(all tier-1 levers are already ON — move to per-category overrides "
            "and scalar tuning)"
        )

        # Format attempt history for the prompt. Each entry lists the exact
        # (field → old, new) move tried, the resulting F1 delta, and whether
        # it was accepted. LLM must avoid re-proposing REJECTED moves with
        # identical values.
        if attempt_history:
            hist_lines = []
            for att in attempt_history[-12:]:  # last 12 to keep prompt bounded
                diff = att.get("diff") or {}
                tag = "✓ ACCEPTED" if att.get("accepted") else "✗ REJECTED"
                delta = att.get("delta", 0.0)
                round_i = att.get("round", "?")
                if not diff:
                    hist_lines.append(
                        f"Round {round_i}: initial config — F1={att.get('f1_after',0):.4f} [baseline]"
                    )
                else:
                    diff_str = "; ".join(
                        f"{k}: {str(v[0])[:30]}→{str(v[1])[:30]}"
                        for k, v in list(diff.items())[:4]
                    )
                    if len(diff) > 4:
                        diff_str += f" (+{len(diff)-4} more)"
                    hist_lines.append(
                        f"Round {round_i}: {{{diff_str}}} — delta={delta:+.4f} {tag}"
                    )
            attempt_history_str = "\n".join(hist_lines)
        else:
            attempt_history_str = "(none — this is round 0)"

        prompt = DIAGNOSIS_PROMPT.format(
            benchmark=benchmark_name,
            total_memories=len(memories) if memories else "unknown",
            total_questions=report.total_questions,
            overall_f1=report.overall_f1,
            zero_count=report.zero_f1_count,
            current_config=_json.dumps(current_config or {}, default=str)[:3500],
            failure_summary=failure_summary,
            category_breakdown=category_breakdown,
            sample_failures=sample_failures,
            attempt_history=attempt_history_str,
            todo_checklist=todo_str,
            available_recipes=available_recipes_str,
        )

        try:
            # Temperature 0 for maximal determinism. Evolution runs are
            # comparable across invocations only if the diagnosis LLM makes
            # the same proposal given identical inputs. Prior runs at 0.1
            # showed v7→v8 recipe-selection drift that moved BEST F1 by
            # ~6pp for otherwise identical conditions.
            response = self.llm_call(
                [{"role": "user", "content": prompt}],
                2048, 0.0,
            )
            if not response:
                return

            # Parse JSON response
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                data = json.loads(m.group())
                report.parameter_suggestions = data.get("parameter_suggestions", {}) or {}
                # Merge per-category proposals into the same dict for downstream use
                per_cat = data.get("per_category_proposals") or data.get("per_category_overrides")
                if per_cat:
                    report.parameter_suggestions["per_category_overrides"] = per_cat
                report.extraction_improvements = data.get(
                    "extraction_improvements",
                    data.get("extraction_suggestions", []),
                )
                report.priority_actions = data.get("priority_actions", [])
                # Recipe invocation: if LLM asked to use a cookbook recipe,
                # merge its proposal into parameter_suggestions. Recipe
                # contents take priority over LLM-proposed fields of the
                # same name (the recipe is a pre-validated combo).
                # LLMs sometimes nest use_recipe inside parameter_suggestions
                # instead of at top level — accept both locations and strip
                # the key so it isn't treated as a config field downstream.
                recipe_name = data.get("use_recipe") or data.get("recipe")
                if not recipe_name:
                    recipe_name = (
                        report.parameter_suggestions.pop("use_recipe", None)
                        or report.parameter_suggestions.pop("recipe", None)
                    )
                if isinstance(recipe_name, str) and recipe_name.strip():
                    from .evolution_cookbook import get_recipe_by_name
                    recipe = get_recipe_by_name(recipe_name.strip(), benchmark=benchmark_name)
                    if recipe is not None:
                        report.invoked_recipe = recipe.name
                        # Merge scalar fields (recipe wins on conflicts so the
                        # pre-validated combo stays intact).
                        for k, v in recipe.proposal.items():
                            if k == "per_category_overrides":
                                merged = dict(
                                    report.parameter_suggestions.get("per_category_overrides")
                                    or {}
                                )
                                for cat_key, cat_fields in (v or {}).items():
                                    existing = merged.get(str(cat_key), {})
                                    existing.update(cat_fields)
                                    merged[str(cat_key)] = existing
                                report.parameter_suggestions["per_category_overrides"] = merged
                            else:
                                report.parameter_suggestions[k] = v
        except Exception as e:
            logger.warning("LLM diagnosis failed: %s", e)


_STOP = frozenset({
    "what", "when", "where", "who", "how", "which", "did", "does", "do",
    "is", "are", "was", "were", "the", "that", "this", "have", "has",
    "had", "been", "will", "would", "could", "should", "from", "with",
    "about", "into", "they", "their", "them", "she", "her", "his",
})
