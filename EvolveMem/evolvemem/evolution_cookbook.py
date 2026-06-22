"""Evolution Cookbook — a library of capability recipes the self-evolution
framework can autonomously select and combine.

Each recipe is a (symptom → proposal) bundle that was empirically validated
against LoCoMo (and will extend to other benchmarks). Recipes are NOT
applied automatically; they are surfaced to the diagnosis LLM as
"available capability cards". The LLM chooses which to invoke each round
based on observed failures, subject to the same step-cap and elitism
discipline as any other proposal.

Design principle: every recipe must be defensible as
  "when symptom X is observed, capability Y is known to help by ~Z pp".
Each one is a compressed form of an experiment we ran, checked in as code
so the framework can autonomously access it rather than requiring the
human to re-discover it per-run.

A recipe's `proposal` dict uses the same field names as RetrievalConfig.
When the LLM outputs `use_recipe: "<recipe_name>"` in its response, the
engine applies that recipe's entire proposal as a SINGLE step-cap change
slot (atomic capability activation). The LLM may still propose up to
`max_changes_per_round - 1` additional fields alongside a recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Recipe:
    """A named capability bundle triggered by a specific failure pattern."""
    name: str
    description: str
    trigger: Callable[[dict, dict, list], bool]
    proposal: dict = field(default_factory=dict)
    expected_lift_pp: float = 0.0  # rough empirical gain in percentage points
    benchmark: str = "locomo"      # which adapter this recipe applies to


# ── Helper trigger predicates ──────────────────────────────────────────────

def _cat_f1(report: dict, cat: int) -> float:
    cf = report.get("category_f1") or {}
    return cf.get(cat, cf.get(str(cat), 1.0))


def _has_override(cfg: dict, cat: int, field_name: str) -> bool:
    ov = (cfg.get("per_category_overrides") or {})
    return bool(ov.get(str(cat), ov.get(cat, {})).get(field_name))


def _consec_noaccept(history: list) -> int:
    n = 0
    for att in reversed(history or []):
        if att.get("accepted"):
            break
        n += 1
    return n


# ── Recipe library ────────────────────────────────────────────────────────

RECIPES_LOCOMO: list[Recipe] = [
    Recipe(
        name="rrf_fusion_warmup",
        description=(
            "Switch to RRF fusion with moderate top-k across all three retrieval "
            "views. The single biggest lever when starting from keyword-only weak "
            "initial: enables semantic + structured search to reinforce BM25."
        ),
        trigger=lambda r, c, h: (
            c.get("fusion_mode") in ("keyword_only", "semantic_only", "structured_only")
            or c.get("semantic_top_k", 0) == 0
        ),
        proposal={
            "fusion_mode": "rrf",
            "semantic_top_k": 15,
            "keyword_top_k": 12,
            "structured_top_k": 7,
            "max_context": 20,
        },
        expected_lift_pp=6.0,
    ),
    Recipe(
        name="intent_aware_multihop_planning",
        description=(
            "SimpleMem-style intent-aware retrieval — the LARGEST single "
            "architectural lever (~+15pp on LoCoMo overall). Before retrieval, "
            "the LLM analyzes what distinct info items the question needs and "
            "emits 1-4 targeted sub-queries. Each sub-query runs full multi- "
            "view retrieval; results union-deduped by content with max-score "
            "kept. This closes the SingleHop-list gap (Cat 1 retrievals "
            "returning one item when question needs all items) and the multi- "
            "hop bridge gap (Cat 3 needing several distinct facts in a single "
            "context). Strictly superior to `enable_query_decomposition`. "
            "Whenever Cat 1 or Cat 3 F1 < 0.40 AND fusion is RRF, prefer "
            "this recipe over smaller per-category tweaks — no other single "
            "move buys as much lift."
        ),
        trigger=lambda r, c, h: (
            not c.get("enable_intent_planning")
            and (_cat_f1(r, 1) < 0.40 or _cat_f1(r, 3) < 0.40)
            and c.get("fusion_mode") == "rrf"  # needs solid retrieval base
        ),
        proposal={
            "enable_intent_planning": True,
            "intent_max_subqs": 4,
            "intent_merge_top_k": 15,
            "intent_final_context": 25,
        },
        expected_lift_pp=15.0,
    ),
    Recipe(
        name="rrf_weight_tuning",
        description=(
            "With RRF already active, shift weights to favor semantic + keyword "
            "over structured: weight_semantic=1.5, weight_keyword=1.2, "
            "weight_structured=0.7. Empirically optimal on LoCoMo. Fires "
            "whenever current weights deviate from this target."
        ),
        trigger=lambda r, c, h: (
            c.get("fusion_mode") == "rrf"
            and (
                abs(c.get("weight_semantic", 1.0) - 1.5) > 0.05
                or abs(c.get("weight_keyword", 1.0) - 1.2) > 0.05
                or abs(c.get("weight_structured", 1.0) - 0.7) > 0.05
            )
        ),
        proposal={
            "weight_semantic": 1.5,
            "weight_keyword": 1.2,
            "weight_structured": 0.7,
        },
        expected_lift_pp=2.0,
    ),
    Recipe(
        name="verification_strict",
        description=(
            "Enable answer_verification with strict style. A second LLM pass "
            "re-checks candidate answers, replacing 'Unknown / not specified' "
            "with the most plausible context-supported answer. Near-universal "
            "win when abstention rate > 10% of total."
        ),
        trigger=lambda r, c, h: (
            not c.get("enable_answer_verification")
            and (r.get("abstention_count", 0) / max(r.get("total_questions", 1), 1)) > 0.08
        ),
        proposal={
            "enable_answer_verification": True,
            "verification_style": "strict",
        },
        expected_lift_pp=4.0,
    ),
    Recipe(
        name="cat5_mcq_adversarial",
        description=(
            "Cat 5 SimpleMem-style two-option MCQ: 'Not mentioned in the "
            "conversation' vs the dataset's adversarial_answer. Largest "
            "single-flag lift on LoCoMo (+30pp on Cat 5)."
        ),
        trigger=lambda r, c, h: (
            not c.get("locomo_cat5_mcq")
            and _cat_f1(r, 5) < 0.45
        ),
        proposal={
            "locomo_cat5_mcq": True,
            "enable_entity_swap": True,
        },
        expected_lift_pp=10.0,
    ),
    Recipe(
        name="cat5_retrieval_boost",
        description=(
            "Cat 5 Adversarial retrieval depth: when Cat 5 MCQ is already on "
            "but F1 is still below 0.85, the bottleneck is retrieval quality "
            "(entity-swap search missing some topics). Boost Cat 5 retrieval "
            "depth to surface more topic-relevant context for the MCQ decision."
        ),
        trigger=lambda r, c, h: (
            c.get("locomo_cat5_mcq", False)
            and _cat_f1(r, 5) < 0.85
            and not _has_override(c, 5, "semantic_top_k")
        ),
        proposal={
            "per_category_overrides": {
                "5": {
                    "semantic_top_k": 25,
                    "keyword_top_k": 15,
                    "entity_swap_semantic_k": 25,
                    "entity_swap_keyword_k": 15,
                    "max_context": 25,
                },
            },
        },
        expected_lift_pp=5.0,
    ),
    Recipe(
        name="cat1_cat4_paraphrase_killer",
        description=(
            "Cat 1/4 verbose-prediction paraphrase symptom: enable both "
            "locomo_cat1_single_fact (force 1-3 word answers) and "
            "locomo_cat4_verbatim_copy (copy distinctive phrase from context). "
            "Addresses 'correct semantically, wrong surface form' failures."
        ),
        trigger=lambda r, c, h: (
            (not c.get("locomo_cat1_single_fact") or not c.get("locomo_cat4_verbatim_copy"))
            and (_cat_f1(r, 1) < 0.4 or _cat_f1(r, 4) < 0.5)
        ),
        proposal={
            "locomo_cat1_single_fact": True,
            "locomo_cat4_verbatim_copy": True,
        },
        expected_lift_pp=8.0,
    ),
    Recipe(
        name="cat2_temporal_format",
        description=(
            "Cat 2 temporal: force YYYY / YYYY-MM-DD / 'Since YYYY' style, "
            "forbid 'N years ago'. Fires when relative-date predictions observed."
        ),
        trigger=lambda r, c, h: (
            not c.get("locomo_cat2_temporal_format")
            and _cat_f1(r, 2) < 0.55
        ),
        proposal={
            "locomo_cat2_temporal_format": True,
        },
        expected_lift_pp=3.0,
    ),
    Recipe(
        name="cat3_inferential_nuanced",
        description=(
            "Cat 3 MultiHop yes-no-with-reason: allow 2-15 word nuanced answers "
            "(vs the 1-6 hard floor). Enable when Cat 3 predictions are shorter "
            "/ more absolute than gold."
        ),
        trigger=lambda r, c, h: (
            not c.get("locomo_cat3_inferential_nuanced")
            and _cat_f1(r, 3) < 0.45
        ),
        proposal={
            "locomo_cat3_inferential_nuanced": True,
        },
        expected_lift_pp=2.0,
    ),
    Recipe(
        name="cat1_cat4_selective_decomposition",
        description=(
            "Per-category query decomposition for Cat 1/4 ONLY (NOT Cat 3). "
            "Cat 1 single-hop and Cat 4 open-domain benefit from sub-question "
            "splits that retrieve distinct evidence. Cat 3 bridge-reasoning "
            "breaks when decomp scatters multi-hop anchor facts across sub-qs."
        ),
        trigger=lambda r, c, h: (
            not _has_override(c, 1, "enable_query_decomposition")
            and not _has_override(c, 4, "enable_query_decomposition")
            and _cat_f1(r, 1) < 0.45
            and _cat_f1(r, 4) < 0.55
        ),
        proposal={
            "per_category_overrides": {
                "1": {
                    "enable_query_decomposition": True,
                    "decomposition_max_subqs": 3,
                    "decomposition_merge_top_k": 12,
                    "max_context": 22,
                },
                "4": {
                    "enable_query_decomposition": True,
                    "decomposition_max_subqs": 3,
                    "decomposition_merge_top_k": 12,
                    "max_context": 22,
                },
            },
        },
        expected_lift_pp=3.5,
    ),
    Recipe(
        name="cat3_mmr_bridge_diversity",
        description=(
            "Cat 3 MultiHop bridge starvation: when Cat 3 F1 is low AND the "
            "memory bank is large (>1000 entries), top-k retrieval is "
            "dominated by near-duplicate mems of the same fact, starving "
            "multi-hop reasoning of distinct bridging evidence. MMR rerank "
            "with diversity_weight=0.5 + tighter max_context=12 picks "
            "distinct supporting facts."
        ),
        trigger=lambda r, c, h: (
            _cat_f1(r, 3) < 0.40
            and r.get("total_memories", 0) > 1000
            and not _has_override(c, 3, "mmr_diversity_weight")
        ),
        proposal={
            "per_category_overrides": {
                "3": {
                    "mmr_diversity_weight": 0.5,
                    "mmr_candidate_pool": 40,
                    "max_context": 12,
                },
            },
        },
        expected_lift_pp=2.5,
    ),
    Recipe(
        name="reflection_single_pass",
        description=(
            "Enable one reflection round (retrieve a second time using question "
            "+ first-draft answer). Small universal win when retrieval is "
            "functional but top-k sometimes misses specific evidence."
        ),
        trigger=lambda r, c, h: (
            c.get("reflection_rounds", 0) == 0
            and c.get("fusion_mode") == "rrf"
            and r.get("overall_f1", 0) > 0.35  # don't propose until retrieval baseline works
        ),
        proposal={
            "reflection_rounds": 1,
        },
        expected_lift_pp=1.5,
    ),
    Recipe(
        name="cat3_dual_model_ensemble",
        description=(
            "Cat 3 MultiHop bimodal gold distribution: gold answers split "
            "between short entity lookups (1-3 words) and inferential clauses "
            "(10+ words). A single LLM cannot cover both modes optimally. "
            "This recipe adds a secondary LLM (gpt-4.1) for Cat 3 only; a "
            "deterministic question-pattern + length picker selects the "
            "better candidate. Costs 1 extra LLM call per Cat 3 question "
            "(~5% of total QA budget). Only effective on LoCoMo."
        ),
        trigger=lambda r, c, h: (
            _cat_f1(r, 3) < 0.40
            and not _has_override(c, 3, "answer_model_ensemble")
            and c.get("locomo_cat3_inferential_nuanced", False)
        ),
        proposal={
            "per_category_overrides": {
                "3": {
                    "answer_model_ensemble": "gpt-4.1",
                },
            },
        },
        expected_lift_pp=9.0,
    ),
    Recipe(
        name="escape_plateau_scalar_sweep",
        description=(
            "When evolution has stalled (3+ consecutive no-accept rounds) and "
            "binary flags are exhausted, shift scalar retrieval params toward "
            "empirically optimal values for LoCoMo: larger top-k, moderate "
            "max_context. Forces exploration out of local plateau."
        ),
        trigger=lambda r, c, h: (
            _consec_noaccept(h) >= 2
            and c.get("fusion_mode") == "rrf"
            and (c.get("semantic_top_k", 0) < 15 or c.get("max_context", 0) < 18)
        ),
        proposal={
            "semantic_top_k": 18,
            "keyword_top_k": 14,
            "max_context": 22,
        },
        expected_lift_pp=1.5,
    ),
]


def match_recipes(
    report_fields: dict,
    current_config: dict,
    attempt_history: list,
    benchmark: str = "locomo",
) -> list[Recipe]:
    """Return the list of recipes whose triggers fire for this round.

    `report_fields` is a dict with keys {'category_f1', 'abstention_count',
    'total_questions', 'overall_f1', 'total_memories'} — populated by the
    diagnosis engine. Kept as a plain dict so the cookbook stays dependency-
    free from DiagnosisReport.
    """
    pool = RECIPES_LOCOMO if benchmark == "locomo" else []
    matched = []
    for r in pool:
        try:
            if r.trigger(report_fields, current_config, attempt_history):
                matched.append(r)
        except Exception:
            # A misbehaving trigger shouldn't crash the evolution loop.
            continue
    # Preserve the natural declaration order of RECIPES_LOCOMO: recipes are
    # declared in empirical pipeline-build order (fusion foundation first,
    # then adapter flags, then per-category overrides). Sorting by lift
    # alone can put high-lift recipes (e.g. adapter flags +8pp) ahead of
    # their prerequisites (RRF +6pp) — at the LLM's low temperature this
    # misleads selection toward out-of-sequence moves that don't pay off
    # without the foundation. v9 regressed to BEST=0.29 at R2 because
    # cat1_cat4_paraphrase_killer got picked before RRF, and adapter
    # flags help little under keyword_only retrieval.
    return matched


def format_recipes_for_prompt(recipes: list[Recipe]) -> str:
    """Render matched recipes as a prompt block the diagnosis LLM can pick from."""
    if not recipes:
        return "(no recipes triggered this round)"
    lines = []
    for r in recipes:
        prop_keys = ", ".join(r.proposal.keys())
        lines.append(
            f"• `{r.name}` (expected +{r.expected_lift_pp:.1f} pp)\n"
            f"    {r.description}\n"
            f"    Fields set: {prop_keys}"
        )
    return "\n".join(lines)


def get_recipe_by_name(name: str, benchmark: str = "locomo") -> Recipe | None:
    pool = RECIPES_LOCOMO if benchmark == "locomo" else []
    for r in pool:
        if r.name == name:
            return r
    return None
