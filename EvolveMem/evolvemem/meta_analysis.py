"""Meta-Evolution Analysis Layer.

Per-round `MemoryDiagnostics` looks at one snapshot and proposes the next
config. The **meta layer** looks at the *history of rounds* and answers
different questions:

  - Is the evolution stagnating? If the last 2 rounds both delta<1pp, we
    may be stuck in a local optimum.
  - Which past change helped / hurt? Should we revert the most recent move
    if it lowered primary?
  - Which subcategory is the worst and has stayed the worst? Focus next
    round there via per_category_overrides, rather than global tweaks.
  - Is there an evolutionary dimension the framework lacks entirely? When
    the LLM says "none of the knobs I have can fix failures of type X",
    log a proposal for a new framework-level dimension. This is the
    "授人以渔" loop: the framework itself nominates where to grow.

The output is a structured plan that the engine applies BEFORE (or instead
of) `diagnostics.suggest_config_update`'s proposal.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


META_PROMPT = """You are the META evolution analyzer for a self-evolving memory system. You can see the entire history of evolution rounds, not just the latest one. Your job is to make decisions per-round diagnosis cannot: detect stagnation, identify which past changes helped, focus on the worst sub-category, and propose new framework dimensions when all current knobs are exhausted.

## Benchmark
{benchmark}

## Primary metric
{primary_metric}

## Evolution history so far
{history}

## Current-round worst subcategories (primary metric)
{worst_subs}

## Current-round worst failure samples
{sample_failures}

## Available action space (same as per-round diagnosis)
Retrieval fields (RetrievalConfig):
  semantic_top_k (int 0-40), keyword_top_k (int 0-20), structured_top_k (int 0-15),
  max_context (int 4-50), enable_entity_swap (bool),
  fusion_mode (enum: first_found|weighted_sum|rrf|semantic_only|keyword_only|structured_only),
  weight_semantic/weight_keyword/weight_structured (float 0-3),
  time_decay_half_life_days (float|null), reflection_rounds (int 0-2),
  answer_style (enum: concise|explanatory|verifying|mcq),
  per_category_overrides (dict category_id -> subfield dict).

## Task
Decide the next round's action. Output JSON with these fields (all optional):

```json
{{
  "decision_type": "tune"|"revert"|"focus_subcategory"|"explore"|"propose_new_dim",
  "reasoning": "one sentence explanation",
  "parameter_suggestions": {{ ...field: value... }},
  "per_category_overrides": {{ "5": {{"enable_entity_swap": true}}}},
  "revert_to_round": 2,
  "explore_field": "weight_semantic",
  "new_dimension_proposal": "add graph-neighbor retrieval: follow entity-links between memories"
}}
```

Rules:
- If latest round's primary < previous round's primary by >2pp, prefer `revert_to_round` = previous.
- If latest round's primary - first round's primary < 1pp after 3 rounds, set `decision_type=explore` and pick ONE field to randomly perturb (name it in explore_field, give a new value in parameter_suggestions).
- If ONE subcategory is >20pp below the mean, use `per_category_overrides` instead of global changes.
- If all knobs have been tried and nothing moves primary (stagnation across 3 rounds), set `decision_type=propose_new_dim` and write a concrete new capability in `new_dimension_proposal`. Examples: "entity-graph retrieval", "query-rewrite via LLM", "multi-hop chain-of-retrieval", "extract with a different prompt mode".
- Always give `reasoning`. Return ONLY the JSON object.
"""


@dataclass
class MetaPlan:
    """Structured output from the meta-analyzer.

    The engine merges `parameter_suggestions` on top of the diagnosis'
    proposal. `revert_to_round` triggers roll-back. `new_dimension_proposal`
    is logged for humans (framework-level growth).
    """
    decision_type: str = "tune"
    reasoning: str = ""
    parameter_suggestions: dict = field(default_factory=dict)
    per_category_overrides: dict = field(default_factory=dict)
    revert_to_round: int | None = None
    explore_field: str | None = None
    new_dimension_proposal: str = ""


class MetaEvolutionAnalyzer:
    """Cross-round strategic decisions.

    Args:
        llm_call: callable used for meta-level reasoning. If None, only
                  deterministic history heuristics run.
    """

    def __init__(self, llm_call: Callable | None = None):
        self.llm_call = llm_call

    # ── Heuristic helpers ──

    @staticmethod
    def _is_regression(history: list[dict]) -> bool:
        if len(history) < 2:
            return False
        return (history[-1]["primary"] - history[-2]["primary"]) < -0.02

    @staticmethod
    def _is_stagnant(history: list[dict], window: int = 3, threshold: float = 0.01) -> bool:
        if len(history) < window:
            return False
        tail = history[-window:]
        return max(r["primary"] for r in tail) - min(r["primary"] for r in tail) < threshold

    @staticmethod
    def _stuck_subcategories(history: list[dict], window: int = 3, cap: float = 0.20) -> list[str]:
        """Subcategories whose primary metric stayed below `cap` across the last
        `window` rounds — candidates for new-dimension proposals.
        """
        if len(history) < window:
            return []
        tail = history[-window:]
        keys: set[str] = set()
        for r in tail:
            keys.update((r.get("subcategory_scores") or {}).keys())

        def _score(entry):
            if not isinstance(entry, dict):
                return None
            for m, v in entry.items():
                if m != "n":
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
            return None

        stuck = []
        for k in keys:
            vals = []
            for r in tail:
                s = _score((r.get("subcategory_scores") or {}).get(k))
                if s is not None:
                    vals.append(s)
            if len(vals) == window and max(vals) < cap:
                stuck.append(k)
        return stuck

    @staticmethod
    def _worst_subcategories(
        subcategory_scores: dict,
        primary_mean: float,
        limit: int = 3,
    ) -> list[tuple[str, float]]:
        out = []
        for key, entry in subcategory_scores.items():
            if not isinstance(entry, dict):
                continue
            # pick first metric that isn't "n"
            for m, v in entry.items():
                if m == "n":
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                out.append((key, fv))
                break
        out.sort(key=lambda kv: kv[1])
        return out[:limit]

    # ── Main entry point ──

    def analyze(
        self,
        benchmark: str,
        primary_metric: str,
        round_history: list[dict],
        current_qa_failures: list[dict],
        current_config: dict,
    ) -> MetaPlan:
        """Produce a MetaPlan for the next round.

        `round_history`: list of dicts with keys
            round, primary, all_metrics, subcategory_scores, config, improvements.
        `current_qa_failures`: list of {question, prediction, reference, category, qtype}
            for the worst failures in the latest round.
        """
        plan = MetaPlan()
        if not round_history:
            return plan

        latest = round_history[-1]
        # Deterministic triggers
        if self._is_regression(round_history):
            # Recommend revert to previous best round (not just last)
            best_r = max(round_history, key=lambda r: r["primary"])
            if best_r["round"] != latest["round"]:
                plan.decision_type = "revert"
                plan.revert_to_round = best_r["round"]
                plan.reasoning = (
                    f"Latest round regressed by "
                    f"{latest['primary']-round_history[-2]['primary']:+.3f}; "
                    f"roll back to round {best_r['round']} (best so far "
                    f"primary={best_r['primary']:.4f})."
                )
                return plan

        # Stuck subcategories across 3+ rounds -> structural capability missing
        stuck_subs = self._stuck_subcategories(round_history, window=3, cap=0.20)
        if stuck_subs:
            plan.decision_type = "propose_new_dim"
            plan.reasoning = (
                f"Subcategor{'y' if len(stuck_subs)==1 else 'ies'} "
                f"{stuck_subs} stuck below 20% for 3 consecutive rounds; "
                f"the current action space cannot fix them."
            )
            # Heuristic default proposal — LLM call below may override with a
            # richer text. This captures multi-hop-style failures most often.
            plan.new_dimension_proposal = (
                "Add switchable retrieval strategy `enable_query_decomposition`: "
                "when the question targets these subcategories, call the LLM to "
                "split it into 2-3 single-hop sub-questions, retrieve per sub, "
                "merge, then answer the original question against the combined "
                "context. Exposes a new evolvable binary dim plus "
                "`decomposition_max_subqs` int."
            )
            # Fall through to LLM for a richer message (optional)

        # Stagnation over 3 rounds -> exploration
        if plan.decision_type not in ("revert", "propose_new_dim") and \
           self._is_stagnant(round_history, window=3, threshold=0.01):
            plan.decision_type = "explore"
            # Pick a promising field to perturb based on what's already high-value
            # — default to fusion_mode rotation if haven't tried all
            tried_fusion = {r["config"].get("fusion_mode") for r in round_history if isinstance(r.get("config"), dict)}
            unexplored = [
                m for m in ("rrf", "weighted_sum", "first_found")
                if m not in tried_fusion
            ]
            if unexplored:
                plan.explore_field = "fusion_mode"
                plan.parameter_suggestions = {"fusion_mode": unexplored[0]}
                plan.reasoning = (
                    f"Stagnation over last 3 rounds (span<1pp); "
                    f"try untried fusion_mode={unexplored[0]}."
                )
                return plan

        # Worst-subcategory focus
        worst = self._worst_subcategories(
            latest.get("subcategory_scores") or {},
            latest.get("primary") or 0.0,
            limit=1,
        )
        if worst and worst[0][1] < max(0.05, (latest.get("primary") or 0.0) - 0.20):
            # One sub is way below the global mean -> per-category override recommended
            sub_key, sub_val = worst[0]
            plan.decision_type = "focus_subcategory"
            plan.reasoning = (
                f"Subcategory '{sub_key}' at {sub_val:.3f} is >=20pp below "
                f"mean ({latest['primary']:.3f}); apply per_category_overrides."
            )

        # LLM meta call for structured suggestions (fills in per_category_overrides etc.)
        if self.llm_call is not None:
            llm_plan = self._llm_meta_call(
                benchmark, primary_metric,
                round_history, current_qa_failures, current_config,
            )
            if llm_plan:
                # Merge LLM's decision into plan, LLM takes precedence on field values
                # but deterministic triggers above already may have set a firm decision
                if plan.decision_type in ("tune", "focus_subcategory") or not plan.parameter_suggestions:
                    plan = llm_plan

        return plan

    # ── LLM call ──

    def _llm_meta_call(
        self,
        benchmark: str,
        primary_metric: str,
        round_history: list[dict],
        current_qa_failures: list[dict],
        current_config: dict,
    ) -> MetaPlan | None:
        # Build readable history
        hist_lines = []
        for r in round_history:
            cfg_brief = {
                k: r["config"].get(k) for k in (
                    "fusion_mode", "semantic_top_k", "keyword_top_k",
                    "structured_top_k", "max_context", "enable_entity_swap",
                    "time_decay_half_life_days", "reflection_rounds",
                ) if isinstance(r.get("config"), dict)
            }
            subs_brief = {
                k: (v.get(primary_metric) if isinstance(v, dict) else v)
                for k, v in (r.get("subcategory_scores") or {}).items()
            }
            hist_lines.append(
                f"Round {r['round']}: primary={r['primary']:.4f} "
                f"config={json.dumps(cfg_brief, default=str)} "
                f"subs={json.dumps(subs_brief, default=str)}"
            )
        history_str = "\n".join(hist_lines) or "(none)"

        latest = round_history[-1]
        worst = self._worst_subcategories(
            latest.get("subcategory_scores") or {},
            latest.get("primary") or 0.0,
            limit=4,
        )
        worst_str = "\n".join(f"  {k}: {v:.3f}" for k, v in worst) or "(none)"

        fail_lines = []
        for f in current_qa_failures[:10]:
            fail_lines.append(
                f"  cat={f.get('category')} qtype={f.get('qtype','')} "
                f"Q: {str(f.get('question',''))[:90]}\n"
                f"    Pred: {str(f.get('prediction',''))[:60]} | "
                f"Ref: {str(f.get('reference',''))[:60]}"
            )
        samples_str = "\n".join(fail_lines) or "(none)"

        prompt = META_PROMPT.format(
            benchmark=benchmark,
            primary_metric=primary_metric,
            history=history_str,
            worst_subs=worst_str,
            sample_failures=samples_str,
        )

        try:
            resp = self.llm_call(
                [{"role": "user", "content": prompt}],
                2048, 0.2,
            )
            if not resp:
                return None
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group())
            plan = MetaPlan(
                decision_type=data.get("decision_type", "tune") or "tune",
                reasoning=data.get("reasoning", ""),
                parameter_suggestions=data.get("parameter_suggestions") or {},
                per_category_overrides=data.get("per_category_overrides") or {},
                revert_to_round=data.get("revert_to_round"),
                explore_field=data.get("explore_field"),
                new_dimension_proposal=data.get("new_dimension_proposal", "") or "",
            )
            return plan
        except Exception as e:
            logger.warning("Meta-analysis LLM call failed: %s", e)
            return None
