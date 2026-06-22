from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import EvolveMemConfig
from .manager import MemoryManager
from .models import MemoryUnit
from .policy_store import MemoryPolicyStore


@dataclass
class MemoryReplaySample:
    session_id: str
    turn: int
    scope_id: str
    query_text: str
    response_text: str
    next_state_text: str


@dataclass
class MemoryReplayResult:
    sample_count: int
    avg_retrieved: float
    avg_query_overlap: float
    avg_continuation_overlap: float
    avg_response_overlap: float
    avg_specificity: float
    avg_focus_score: float
    avg_value_density: float
    avg_grounding_score: float = 0.0
    avg_coverage_score: float = 0.0
    zero_retrieval_count: int = 0

    @property
    def composite_score(self) -> float:
        """Single composite quality score for quick operator comparison.

        Weighted combination of overlap, focus, grounding, coverage, and density
        signals, penalized by zero-retrieval rate.
        """
        if self.sample_count == 0:
            return 0.0
        raw = (
            0.20 * self.avg_query_overlap
            + 0.15 * self.avg_continuation_overlap
            + 0.15 * self.avg_response_overlap
            + 0.15 * self.avg_focus_score
            + 0.10 * self.avg_grounding_score
            + 0.10 * self.avg_coverage_score
            + 0.10 * self.avg_value_density
            + 0.05 * self.avg_specificity
        )
        # Penalize zero-retrieval rate.
        zero_rate = self.zero_retrieval_count / float(self.sample_count)
        return round(raw * (1.0 - 0.5 * zero_rate), 4)


def load_replay_samples(
    path: str,
    default_scope: str = "default",
    max_samples: int = 0,
    telemetry_path: str = "",
) -> list[MemoryReplaySample]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return []

    samples: list[MemoryReplaySample] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        query_text = str(record.get("instruction_text") or record.get("prompt_text") or "").strip()
        response_text = str(record.get("response_text") or "").strip()
        next_state = record.get("next_state") or {}
        next_state_text = ""
        if isinstance(next_state, dict):
            next_state_text = _flatten_content(next_state.get("content", ""))
        if not query_text:
            continue
        samples.append(
            MemoryReplaySample(
                session_id=str(record.get("session_id", "")),
                turn=int(record.get("turn", 0) or 0),
                scope_id=str(record.get("memory_scope") or default_scope),
                query_text=query_text,
                response_text=response_text,
                next_state_text=next_state_text,
            )
        )
    if max_samples > 0 and len(samples) > max_samples:
        # Telemetry-weighted sampling: prefer sessions with richer retrieval history.
        session_weights = _load_session_weights(telemetry_path) if telemetry_path else {}

        by_session: dict[str, list[MemoryReplaySample]] = {}
        for s in samples:
            by_session.setdefault(s.session_id, []).append(s)

        if session_weights:
            # Sort sessions by weight descending so higher-signal sessions are drawn first.
            session_ids = sorted(
                by_session.keys(),
                key=lambda sid: session_weights.get(sid, 1.0),
                reverse=True,
            )
        else:
            session_ids = list(by_session.keys())

        result: list[MemoryReplaySample] = []
        idx = 0
        while len(result) < max_samples and session_ids:
            sid = session_ids[idx % len(session_ids)]
            group = by_session[sid]
            if group:
                result.append(group.pop(0))
            else:
                session_ids.remove(sid)
            idx += 1
        return result
    return samples


class MemoryReplayEvaluator:
    """Offline comparison helper for baseline and candidate memory policies."""

    def evaluate(
        self,
        manager: MemoryManager,
        samples: list[MemoryReplaySample],
    ) -> MemoryReplayResult:
        if not samples:
            return MemoryReplayResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        retrieved_total = 0.0
        query_overlap_total = 0.0
        continuation_overlap_total = 0.0
        response_overlap_total = 0.0
        specificity_total = 0.0
        focus_total = 0.0
        value_density_total = 0.0
        grounding_total = 0.0
        coverage_total = 0.0
        zero_retrieval_count = 0

        for sample in samples:
            units = manager.retrieve_for_prompt(sample.query_text, scope_id=sample.scope_id)
            rendered = manager.render_for_prompt(units)
            query_overlap = _term_overlap(rendered, sample.query_text)
            continuation_overlap = _term_overlap(rendered, sample.next_state_text)
            response_overlap = _term_overlap(rendered, sample.response_text)
            focus_score = _focus_score(
                rendered,
                " ".join(
                    [
                        sample.query_text,
                        sample.response_text,
                        sample.next_state_text,
                    ]
                ),
            )
            grounding = _grounding_score(units, sample.query_text, sample.response_text)
            coverage = _coverage_score(rendered, sample.response_text)
            retrieved_total += float(len(units))
            if not units:
                zero_retrieval_count += 1
            query_overlap_total += query_overlap
            continuation_overlap_total += continuation_overlap
            response_overlap_total += response_overlap
            specificity_total += _specificity_score(rendered)
            focus_total += focus_score
            value_density_total += _value_density_score(
                len(units),
                query_overlap,
                continuation_overlap,
                response_overlap,
                focus_score,
            )
            grounding_total += grounding
            coverage_total += coverage

        count = float(len(samples))
        return MemoryReplayResult(
            sample_count=len(samples),
            avg_retrieved=retrieved_total / count,
            avg_query_overlap=query_overlap_total / count,
            avg_continuation_overlap=continuation_overlap_total / count,
            avg_response_overlap=response_overlap_total / count,
            avg_specificity=specificity_total / count,
            avg_focus_score=focus_total / count,
            avg_value_density=value_density_total / count,
            avg_grounding_score=grounding_total / count,
            avg_coverage_score=coverage_total / count,
            zero_retrieval_count=zero_retrieval_count,
        )

    def compare(
        self,
        baseline: MemoryReplayResult,
        candidate: MemoryReplayResult,
    ) -> dict:
        return {
            "sample_count": candidate.sample_count,
            "avg_retrieved_delta": round(candidate.avg_retrieved - baseline.avg_retrieved, 4),
            "avg_query_overlap_delta": round(candidate.avg_query_overlap - baseline.avg_query_overlap, 4),
            "avg_continuation_overlap_delta": round(
                candidate.avg_continuation_overlap - baseline.avg_continuation_overlap,
                4,
            ),
            "avg_response_overlap_delta": round(
                candidate.avg_response_overlap - baseline.avg_response_overlap,
                4,
            ),
            "avg_specificity_delta": round(
                candidate.avg_specificity - baseline.avg_specificity,
                4,
            ),
            "avg_focus_score_delta": round(
                candidate.avg_focus_score - baseline.avg_focus_score,
                4,
            ),
            "avg_value_density_delta": round(
                candidate.avg_value_density - baseline.avg_value_density,
                4,
            ),
            "avg_grounding_score_delta": round(
                candidate.avg_grounding_score - baseline.avg_grounding_score,
                4,
            ),
            "avg_coverage_score_delta": round(
                candidate.avg_coverage_score - baseline.avg_coverage_score,
                4,
            ),
            "zero_retrieval_delta": candidate.zero_retrieval_count - baseline.zero_retrieval_count,
            "composite_score_delta": round(
                candidate.composite_score - baseline.composite_score, 4
            ),
            "candidate_beats_baseline": (
                candidate.avg_query_overlap >= baseline.avg_query_overlap
                and candidate.avg_continuation_overlap >= baseline.avg_continuation_overlap
                and candidate.avg_response_overlap >= baseline.avg_response_overlap
                and candidate.avg_focus_score >= baseline.avg_focus_score
                and candidate.avg_value_density >= baseline.avg_value_density
                and candidate.avg_grounding_score >= baseline.avg_grounding_score
                and candidate.avg_coverage_score >= baseline.avg_coverage_score
            ),
        }


def write_replay_report(path: str, baseline: MemoryReplayResult, candidate: MemoryReplayResult, comparison: dict) -> None:
    report_path = Path(path).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": {
            "sample_count": baseline.sample_count,
            "avg_retrieved": baseline.avg_retrieved,
            "avg_query_overlap": baseline.avg_query_overlap,
            "avg_continuation_overlap": baseline.avg_continuation_overlap,
            "avg_response_overlap": baseline.avg_response_overlap,
            "avg_specificity": baseline.avg_specificity,
            "avg_focus_score": baseline.avg_focus_score,
            "avg_value_density": baseline.avg_value_density,
            "avg_grounding_score": baseline.avg_grounding_score,
            "avg_coverage_score": baseline.avg_coverage_score,
            "composite_score": baseline.composite_score,
        },
        "candidate": {
            "sample_count": candidate.sample_count,
            "avg_retrieved": candidate.avg_retrieved,
            "avg_query_overlap": candidate.avg_query_overlap,
            "avg_continuation_overlap": candidate.avg_continuation_overlap,
            "avg_response_overlap": candidate.avg_response_overlap,
            "avg_specificity": candidate.avg_specificity,
            "avg_focus_score": candidate.avg_focus_score,
            "avg_value_density": candidate.avg_value_density,
            "avg_grounding_score": candidate.avg_grounding_score,
            "avg_coverage_score": candidate.avg_coverage_score,
            "composite_score": candidate.composite_score,
        },
        "comparison": comparison,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def run_policy_candidate_replay(
    cfg: EvolveMemConfig,
    samples: list[MemoryReplaySample],
    candidate_policy_path: str,
) -> tuple[MemoryReplayResult, MemoryReplayResult, dict]:
    evaluator = MemoryReplayEvaluator()
    baseline_manager = MemoryManager.from_config(cfg)
    candidate_store = MemoryPolicyStore(candidate_policy_path)
    candidate_state = candidate_store.load()
    candidate_manager = MemoryManager.from_config_with_policy_state(cfg, candidate_state)
    try:
        baseline = evaluator.evaluate(baseline_manager, samples)
        candidate = evaluator.evaluate(candidate_manager, samples)
        comparison = evaluator.compare(baseline, candidate)
    finally:
        baseline_manager.close()
        candidate_manager.close()
    return baseline, candidate, comparison


def _load_session_weights(telemetry_path: str) -> dict[str, float]:
    """Build per-session weights from retrieval telemetry.

    Sessions with more retrieval events and higher average importance
    get higher weight, meaning they are sampled first during stratified
    replay sampling.
    """
    tpath = Path(telemetry_path).expanduser()
    if not tpath.exists():
        return {}

    session_events: dict[str, list[dict]] = {}
    try:
        for line in tpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("event_type") != "memory_retrieval":
                continue
            payload = event.get("payload", {})
            scope_id = str(payload.get("scope_id", ""))
            if scope_id:
                session_events.setdefault(scope_id, []).append(payload)
    except Exception:
        return {}

    if not session_events:
        return {}

    weights: dict[str, float] = {}
    for sid, events in session_events.items():
        count = len(events)
        avg_imp = sum(e.get("avg_importance", 0.5) for e in events) / max(count, 1)
        avg_retrieved = sum(e.get("retrieved_count", 0) for e in events) / max(count, 1)
        # Weight = event_count * (1 + avg_importance) * (1 + log_retrieved)
        import math
        weights[sid] = count * (1.0 + avg_imp) * (1.0 + math.log1p(avg_retrieved))

    # Normalize to [0.5, 2.0] range.
    if weights:
        max_w = max(weights.values())
        min_w = min(weights.values())
        spread = max_w - min_w
        if spread > 0:
            weights = {
                sid: round(0.5 + 1.5 * (w - min_w) / spread, 4)
                for sid, w in weights.items()
            }
        else:
            weights = {sid: 1.0 for sid in weights}

    return weights


class MemoryReplayJudge:
    """Optional LLM-based quality judge for replay evaluation.

    This is an interface for adding LLM-based quality signals to replay
    evaluation when an LLM is available offline. Subclass this and
    implement score_memory_relevance() to provide model-graded signals.
    """

    def score_memory_relevance(
        self,
        memory_text: str,
        query_text: str,
        response_text: str,
    ) -> float:
        """Score how relevant retrieved memory is to the query and response.

        Returns a float in [0.0, 1.0] where 1.0 is perfectly relevant.
        Default implementation returns 0.0 (no LLM available).
        """
        return 0.0

    def is_available(self) -> bool:
        """Check if the judge has a working LLM connection."""
        return False


class HeuristicReplayJudge(MemoryReplayJudge):
    """Heuristic-based judge for testing without a real LLM endpoint.

    Uses term overlap and keyword matching to approximate relevance scoring.
    Useful for local testing, CI, and development environments.
    """

    def score_memory_relevance(
        self,
        memory_text: str,
        query_text: str,
        response_text: str,
    ) -> float:
        memory_terms = set(_tokenize(memory_text.lower()))
        query_terms = set(_tokenize(query_text.lower()))
        response_terms = set(_tokenize(response_text.lower()))

        if not memory_terms:
            return 0.0

        # Query overlap: how well memory matches the query
        query_overlap = len(memory_terms & query_terms) / max(len(query_terms), 1)

        # Response overlap: how well memory matches the response
        response_overlap = len(memory_terms & response_terms) / max(len(response_terms), 1)

        # Combined score with query weighted higher
        score = 0.6 * query_overlap + 0.4 * response_overlap

        # Bonus for longer, more informative matches
        if len(memory_terms & query_terms) >= 3:
            score = min(1.0, score + 0.1)

        return round(min(1.0, score), 4)

    def is_available(self) -> bool:
        return True


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(parts)
    return str(content or "")


def _term_overlap(left: str, right: str) -> float:
    left_terms = set(_tokenize(left))
    right_terms = set(_tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    shared = len(left_terms & right_terms)
    return shared / float(len(right_terms))


def _specificity_score(text: str) -> float:
    terms = _tokenize(text)
    if not terms:
        return 0.0
    informative = [term for term in terms if len(term) >= 6]
    unique_ratio = len(set(terms)) / float(len(terms))
    informative_ratio = len(informative) / float(len(terms))
    return round((unique_ratio + informative_ratio) / 2.0, 4)


def _focus_score(memory_text: str, reference_text: str) -> float:
    memory_terms = set(_tokenize(memory_text))
    reference_terms = set(_tokenize(reference_text))
    if not memory_terms:
        return 0.0
    if not reference_terms:
        return 0.0
    shared = len(memory_terms & reference_terms)
    return round(shared / float(len(memory_terms)), 4)


def _value_density_score(
    retrieved_units: int,
    query_overlap: float,
    continuation_overlap: float,
    response_overlap: float,
    focus_score: float,
) -> float:
    denominator = float(max(retrieved_units, 1))
    useful_signal = query_overlap + continuation_overlap + response_overlap + focus_score
    return round(useful_signal / denominator, 4)


def _grounding_score(
    units: list[MemoryUnit],
    query_text: str,
    response_text: str,
) -> float:
    """Score how well retrieved memory units are grounded in the task context.

    Uses entity and topic overlap between the retrieved units' metadata
    and the task context (query + response) as a proxy for task-relevance.
    This goes beyond raw word overlap by focusing on structured metadata.
    """
    if not units:
        return 0.0

    context_terms = set(_tokenize(query_text + " " + response_text))
    informative_context = {t for t in context_terms if len(t) >= 4}
    if not informative_context:
        return 0.0

    unit_scores: list[float] = []
    for unit in units:
        metadata_terms = set(
            t.lower() for t in unit.entities + unit.topics
        )
        if not metadata_terms:
            # Fall back to content keywords for units without metadata.
            content_tokens = _tokenize(unit.content)
            metadata_terms = {t for t in content_tokens if len(t) >= 5}
        if not metadata_terms:
            unit_scores.append(0.0)
            continue
        shared = len(metadata_terms & informative_context)
        unit_scores.append(shared / float(len(metadata_terms)))

    return round(sum(unit_scores) / float(len(unit_scores)), 4)


def _coverage_score(memory_text: str, response_text: str) -> float:
    """Score how much of the response content was anticipated by retrieved memory.

    Measures the fraction of informative response terms that appear in the
    retrieved memory block. High coverage means memory provided useful context
    for generating the response.
    """
    if not response_text or not memory_text:
        return 0.0

    response_terms = set(_tokenize(response_text))
    informative_response = {t for t in response_terms if len(t) >= 4}
    if not informative_response:
        return 0.0

    memory_terms = set(_tokenize(memory_text))
    covered = len(informative_response & memory_terms)
    return round(covered / float(len(informative_response)), 4)


def _tokenize(text: str) -> list[str]:
    token: list[str] = []
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
            continue
        if token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out
