from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .candidate import generate_policy_candidates
from .config import EvolveMemConfig
from .policy_store import MemoryPolicyState, MemoryPolicyStore
from .promotion import MemoryPromotionCriteria, should_promote
from .replay import load_replay_samples, run_policy_candidate_replay, write_replay_report


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_within_recent_window(
    timestamp: datetime | None,
    now: datetime,
    recent_window_hours: int,
) -> bool:
    if timestamp is None or recent_window_hours <= 0:
        return False
    age_seconds = (now - timestamp).total_seconds()
    return 0.0 <= age_seconds <= float(recent_window_hours * 3600)


@dataclass
class MemoryUpgradeDecision:
    promoted: bool
    comparison: dict
    report_path: str
    candidate_policy_path: str
    reason: str


class MemorySelfUpgradeOrchestrator:
    """Controlled promotion path for memory policy candidates."""

    def __init__(self, cfg: EvolveMemConfig, history_path: str):
        self.cfg = cfg
        self.history_path = Path(history_path).expanduser()
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.review_queue_path = self.history_path.with_name("upgrade_review_queue.jsonl")
        self.review_history_path = self.history_path.with_name("review_history.jsonl")
        self.cycle_history_path = self.history_path.with_name("upgrade_cycle_history.jsonl")
        self.artifact_dir = self.history_path.parent

    def evaluate_candidate(
        self,
        candidate_policy_path: str,
        replay_records_path: str,
        report_path: str,
        criteria: MemoryPromotionCriteria | None = None,
        require_review: bool = False,
    ) -> MemoryUpgradeDecision:
        decision = self._evaluate_candidate_once(
            candidate_policy_path=candidate_policy_path,
            replay_records_path=replay_records_path,
            report_path=report_path,
            criteria=criteria,
        )
        self._append_history(decision)
        replay_passed = decision.reason in {"promoted", "pending_review"}
        if replay_passed and require_review:
            queued = MemoryUpgradeDecision(
                promoted=False,
                comparison=decision.comparison,
                report_path=decision.report_path,
                candidate_policy_path=decision.candidate_policy_path,
                reason="pending_review",
            )
            self._append_history(queued)
            self._enqueue_review(queued)
            self._append_review_event("queued", queued)
            return queued
        if decision.promoted:
            self._promote_candidate(candidate_policy_path)
        return decision

    def _evaluate_candidate_once(
        self,
        candidate_policy_path: str,
        replay_records_path: str,
        report_path: str,
        criteria: MemoryPromotionCriteria | None = None,
    ) -> MemoryUpgradeDecision:
        samples = load_replay_samples(replay_records_path, default_scope=self.cfg.memory_scope)
        baseline, candidate, comparison = run_policy_candidate_replay(
            cfg=self.cfg,
            samples=samples,
            candidate_policy_path=candidate_policy_path,
        )
        write_replay_report(report_path, baseline, candidate, comparison)
        active_criteria = criteria or MemoryPromotionCriteria()
        replay_passed = should_promote(comparison, active_criteria)
        return MemoryUpgradeDecision(
            promoted=replay_passed,
            comparison=comparison,
            report_path=str(Path(report_path).expanduser()),
            candidate_policy_path=str(Path(candidate_policy_path).expanduser()),
            reason="promoted" if replay_passed else "rejected",
        )

    def default_candidate_dir(self) -> str:
        return str(self.artifact_dir / "candidates")

    def default_reports_dir(self) -> str:
        return str(self.artifact_dir / "candidate_reports")

    def default_cycle_summary_path(self) -> str:
        return str(self.artifact_dir / "last_upgrade_cycle.json")

    def generate_candidate_files(self, output_dir: str) -> list[str]:
        live_store = MemoryPolicyStore(self.cfg.memory_policy_path)
        current = live_store.load()
        candidates = generate_policy_candidates(current)
        target_dir = Path(output_dir).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for idx, candidate in enumerate(candidates, start=1):
            path = target_dir / f"candidate_{idx:03d}.json"
            MemoryPolicyStore(str(path)).save(candidate, reason="generated_candidate")
            paths.append(str(path))
        return paths

    def evaluate_candidate_directory(
        self,
        candidate_dir: str,
        replay_records_path: str,
        reports_dir: str,
        criteria: MemoryPromotionCriteria | None = None,
        require_review: bool = False,
        cycle_summary_path: str = "",
    ) -> list[MemoryUpgradeDecision]:
        candidate_paths = sorted(Path(candidate_dir).expanduser().glob("*.json"))
        evaluated: list[MemoryUpgradeDecision] = []
        for candidate_path in candidate_paths:
            report_path = Path(reports_dir).expanduser() / f"{candidate_path.stem}_report.json"
            evaluated.append(
                self._evaluate_candidate_once(
                    candidate_policy_path=str(candidate_path),
                    replay_records_path=replay_records_path,
                    report_path=str(report_path),
                    criteria=criteria,
                )
            )
        if not evaluated:
            return []
        best = self._select_best_candidate(evaluated)
        decisions: list[MemoryUpgradeDecision] = []
        for decision in evaluated:
            if decision.candidate_policy_path != best.candidate_policy_path:
                rejected = MemoryUpgradeDecision(
                    promoted=False,
                    comparison=decision.comparison,
                    report_path=decision.report_path,
                    candidate_policy_path=decision.candidate_policy_path,
                    reason="rejected_not_best",
                )
                self._append_history(rejected)
                decisions.append(rejected)
                continue
            final = decision
            if decision.promoted and require_review:
                final = MemoryUpgradeDecision(
                    promoted=False,
                    comparison=decision.comparison,
                    report_path=decision.report_path,
                    candidate_policy_path=decision.candidate_policy_path,
                    reason="pending_review",
                )
                self._append_history(final)
                self._enqueue_review(final)
                self._append_review_event("queued", final)
            else:
                self._append_history(final)
                if final.promoted:
                    self._promote_candidate(final.candidate_policy_path)
            decisions.append(final)
        self._write_cycle_summary(
            decisions=decisions,
            cycle_summary_path=cycle_summary_path or self.default_cycle_summary_path(),
        )
        return decisions

    def run_auto_upgrade_cycle(
        self,
        replay_records_path: str,
        candidate_dir: str = "",
        reports_dir: str = "",
        criteria: MemoryPromotionCriteria | None = None,
        require_review: bool = False,
        cycle_summary_path: str = "",
    ) -> list[MemoryUpgradeDecision]:
        resolved_candidate_dir = candidate_dir or self.default_candidate_dir()
        resolved_reports_dir = reports_dir or self.default_reports_dir()
        summary_path = cycle_summary_path or self.default_cycle_summary_path()
        self.generate_candidate_files(resolved_candidate_dir)
        decisions = self.evaluate_candidate_directory(
            candidate_dir=resolved_candidate_dir,
            replay_records_path=replay_records_path,
            reports_dir=resolved_reports_dir,
            criteria=criteria,
            require_review=require_review,
            cycle_summary_path=summary_path,
        )
        cleanup = self.cleanup_artifacts(
            candidate_dir=resolved_candidate_dir,
            reports_dir=resolved_reports_dir,
        )
        self._attach_cleanup_to_cycle_summary(summary_path, cleanup)
        return decisions

    def read_history(self, limit: int = 20) -> list[dict]:
        if not self.history_path.exists():
            return []
        items: list[dict] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return items

    def summarize_history(self, recent_window_hours: int = 168) -> dict:
        items = self.read_history(limit=1000000)
        now = datetime.now(timezone.utc)
        summary = {
            "total": len(items),
            "promoted": 0,
            "pending_review": 0,
            "rejected": 0,
            "approved_review": 0,
            "rejected_review": 0,
            "recent_window_hours": recent_window_hours,
            "recent_total": 0,
            "recent_promoted": 0,
            "recent_pending_review": 0,
            "recent_rejected": 0,
            "recent_approved_review": 0,
            "recent_rejected_review": 0,
        }
        for item in items:
            reason = str(item.get("reason", ""))
            timestamp = _parse_iso_timestamp(str(item.get("timestamp", "")))
            is_recent = _is_within_recent_window(timestamp, now, recent_window_hours)
            if is_recent:
                summary["recent_total"] += 1
            if reason == "promoted":
                summary["promoted"] += 1
                if is_recent:
                    summary["recent_promoted"] += 1
            elif reason == "pending_review":
                summary["pending_review"] += 1
                if is_recent:
                    summary["recent_pending_review"] += 1
            elif reason == "approved_review":
                summary["approved_review"] += 1
                if is_recent:
                    summary["recent_approved_review"] += 1
            elif reason == "rejected_review":
                summary["rejected_review"] += 1
                if is_recent:
                    summary["recent_rejected_review"] += 1
            elif reason.startswith("rejected"):
                summary["rejected"] += 1
                if is_recent:
                    summary["recent_rejected"] += 1
        return summary

    def summarize_candidate_directory(self, candidate_dir: str) -> dict:
        paths = sorted(Path(candidate_dir).expanduser().glob("*.json"))
        return {
            "count": len(paths),
            "candidate_files": [str(path) for path in paths[:20]],
        }

    def summarize_review_queue(self, stale_after_hours: int = 72) -> dict:
        queue = self.read_review_queue()
        now = datetime.now(timezone.utc)
        ages_hours: list[float] = []
        stale_candidates: list[str] = []
        for item in queue:
            parsed = _parse_iso_timestamp(str(item.get("timestamp", "")))
            if parsed is None:
                continue
            age_hours = max((now - parsed).total_seconds() / 3600.0, 0.0)
            ages_hours.append(age_hours)
            if age_hours >= stale_after_hours:
                stale_candidates.append(str(item.get("candidate_policy_path", "")))
        return {
            "pending_count": len(queue),
            "stale_after_hours": stale_after_hours,
            "stale_count": len(stale_candidates),
            "oldest_age_hours": round(max(ages_hours), 2) if ages_hours else 0.0,
            "newest_age_hours": round(min(ages_hours), 2) if ages_hours else 0.0,
            "stale_candidates": stale_candidates[:20],
        }

    def summarize_operational_health(
        self,
        stale_after_hours: int = 72,
        recent_window_hours: int = 168,
    ) -> dict:
        queue = self.summarize_review_queue(stale_after_hours=stale_after_hours)
        review = self.summarize_review_history(recent_window_hours=recent_window_hours)
        cycles = self.summarize_cycle_history(recent_window_hours=recent_window_hours)
        upgrade = self.summarize_history(recent_window_hours=recent_window_hours)

        level = "healthy"
        reasons: list[str] = []

        if queue["stale_count"] > 0:
            level = "critical"
            reasons.append("stale review queue")
        elif queue["pending_count"] > 0:
            level = "warning"
            reasons.append("pending review queue")

        if review["backlog_pressure_hours"] >= float(stale_after_hours):
            level = "critical"
            reasons.append("review backlog pressure high")
        elif review["backlog_pressure_hours"] >= max(float(stale_after_hours) / 3.0, 12.0):
            if level == "healthy":
                level = "warning"
            reasons.append("review backlog pressure elevated")

        if cycles["recent_cycles"] > 0 and cycles["recent_pending_review_cycle_rate"] >= 0.5:
            if level == "healthy":
                level = "warning"
            reasons.append("recent cycles frequently pending review")

        if cycles["recent_cycles"] > 0 and cycles["recent_promoted_cycle_rate"] == 0.0:
            if level == "healthy" and queue["pending_count"] == 0:
                level = "warning"
            reasons.append("recent cycles produced no promotions")

        if upgrade["recent_total"] == 0 and cycles["total_cycles"] > 0 and queue["pending_count"] == 0:
            if level == "healthy":
                level = "warning"
            reasons.append("no recent upgrade decisions")

        deduped_reasons: list[str] = []
        for reason in reasons:
            if reason not in deduped_reasons:
                deduped_reasons.append(reason)

        return {
            "level": level,
            "reasons": deduped_reasons,
            "pending_review": queue["pending_count"],
            "stale_review": queue["stale_count"],
            "backlog_pressure_hours": review["backlog_pressure_hours"],
            "recent_promoted_cycle_rate": cycles["recent_promoted_cycle_rate"],
            "recent_pending_review_cycle_rate": cycles["recent_pending_review_cycle_rate"],
            "recent_upgrade_decisions": upgrade["recent_total"],
        }

    def read_cycle_summary(self, path: str = "") -> dict:
        summary_path = Path(path or self.default_cycle_summary_path()).expanduser()
        if not summary_path.exists():
            return {}
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def read_cycle_history(self, limit: int = 20) -> list[dict]:
        if not self.cycle_history_path.exists():
            return []
        items: list[dict] = []
        for line in self.cycle_history_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return items

    def summarize_cycle_history(self, recent_window_hours: int = 168) -> dict:
        items = self.read_cycle_history(limit=1000000)
        now = datetime.now(timezone.utc)
        summary = {
            "total_cycles": len(items),
            "promoted_cycles": 0,
            "pending_review_cycles": 0,
            "avg_candidates": 0.0,
            "avg_best_score": 0.0,
            "promoted_cycle_rate": 0.0,
            "pending_review_cycle_rate": 0.0,
            "recent_window_hours": recent_window_hours,
            "recent_cycles": 0,
            "recent_promoted_cycles": 0,
            "recent_pending_review_cycles": 0,
            "recent_avg_candidates": 0.0,
            "recent_avg_best_score": 0.0,
            "recent_promoted_cycle_rate": 0.0,
            "recent_pending_review_cycle_rate": 0.0,
        }
        if not items:
            return summary

        candidate_counts: list[float] = []
        best_scores: list[float] = []
        recent_candidate_counts: list[float] = []
        recent_best_scores: list[float] = []
        for item in items:
            num_promoted = int(item.get("num_promoted", 0) or 0)
            num_pending_review = int(item.get("num_pending_review", 0) or 0)
            num_candidates = float(item.get("num_candidates", 0) or 0)
            metric_summary = item.get("metric_summary", {})
            best_score = float(metric_summary.get("best_score", 0.0) or 0.0)
            timestamp = _parse_iso_timestamp(str(item.get("updated_at", "")))
            is_recent = _is_within_recent_window(timestamp, now, recent_window_hours)
            if num_promoted > 0:
                summary["promoted_cycles"] += 1
            if num_pending_review > 0:
                summary["pending_review_cycles"] += 1
            candidate_counts.append(num_candidates)
            best_scores.append(best_score)
            if is_recent:
                summary["recent_cycles"] += 1
                if num_promoted > 0:
                    summary["recent_promoted_cycles"] += 1
                if num_pending_review > 0:
                    summary["recent_pending_review_cycles"] += 1
                recent_candidate_counts.append(num_candidates)
                recent_best_scores.append(best_score)

        summary["avg_candidates"] = round(sum(candidate_counts) / float(len(candidate_counts)), 2)
        summary["avg_best_score"] = round(sum(best_scores) / float(len(best_scores)), 4)
        summary["promoted_cycle_rate"] = round(
            summary["promoted_cycles"] / float(summary["total_cycles"]),
            4,
        )
        summary["pending_review_cycle_rate"] = round(
            summary["pending_review_cycles"] / float(summary["total_cycles"]),
            4,
        )
        if recent_candidate_counts:
            summary["recent_avg_candidates"] = round(
                sum(recent_candidate_counts) / float(len(recent_candidate_counts)),
                2,
            )
        if recent_best_scores:
            summary["recent_avg_best_score"] = round(
                sum(recent_best_scores) / float(len(recent_best_scores)),
                4,
            )
        if summary["recent_cycles"] > 0:
            summary["recent_promoted_cycle_rate"] = round(
                summary["recent_promoted_cycles"] / float(summary["recent_cycles"]),
                4,
            )
            summary["recent_pending_review_cycle_rate"] = round(
                summary["recent_pending_review_cycles"] / float(summary["recent_cycles"]),
                4,
            )
        return summary

    def read_review_queue(self) -> list[dict]:
        if not self.review_queue_path.exists():
            return []
        items: list[dict] = []
        for line in self.review_queue_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return items

    def read_review_history(self, limit: int = 20) -> list[dict]:
        if not self.review_history_path.exists():
            return []
        items: list[dict] = []
        for line in self.review_history_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return items

    def summarize_review_history(self, recent_window_hours: int = 168) -> dict:
        items = self.read_review_history(limit=1000000)
        now = datetime.now(timezone.utc)
        summary = {
            "total": len(items),
            "queued": 0,
            "approved": 0,
            "rejected": 0,
            "resolved_count": 0,
            "avg_resolution_hours": 0.0,
            "max_resolution_hours": 0.0,
            "pending_estimate": 0,
            "backlog_pressure_hours": 0.0,
            "approval_rate": 0.0,
            "rejection_rate": 0.0,
            "recent_window_hours": recent_window_hours,
            "recent_total": 0,
            "recent_queued": 0,
            "recent_approved": 0,
            "recent_rejected": 0,
            "recent_resolved_count": 0,
            "recent_avg_resolution_hours": 0.0,
            "recent_approval_rate": 0.0,
            "recent_rejection_rate": 0.0,
        }
        queued_at: dict[str, datetime] = {}
        resolution_hours: list[float] = []
        recent_resolution_hours: list[float] = []
        for item in items:
            event = str(item.get("event", ""))
            candidate_path = str(item.get("candidate_policy_path", ""))
            timestamp = _parse_iso_timestamp(str(item.get("timestamp", "")))
            if _is_within_recent_window(timestamp, now, recent_window_hours):
                summary["recent_total"] += 1
            if event == "queued":
                summary["queued"] += 1
                if _is_within_recent_window(timestamp, now, recent_window_hours):
                    summary["recent_queued"] += 1
                if candidate_path and timestamp is not None:
                    queued_at[candidate_path] = timestamp
            elif event == "approved":
                summary["approved"] += 1
                if _is_within_recent_window(timestamp, now, recent_window_hours):
                    summary["recent_approved"] += 1
                if candidate_path and timestamp is not None and candidate_path in queued_at:
                    resolution = max(
                        (timestamp - queued_at[candidate_path]).total_seconds() / 3600.0,
                        0.0,
                    )
                    resolution_hours.append(resolution)
                    if _is_within_recent_window(timestamp, now, recent_window_hours):
                        recent_resolution_hours.append(resolution)
            elif event == "rejected":
                summary["rejected"] += 1
                if _is_within_recent_window(timestamp, now, recent_window_hours):
                    summary["recent_rejected"] += 1
                if candidate_path and timestamp is not None and candidate_path in queued_at:
                    resolution = max(
                        (timestamp - queued_at[candidate_path]).total_seconds() / 3600.0,
                        0.0,
                    )
                    resolution_hours.append(resolution)
                    if _is_within_recent_window(timestamp, now, recent_window_hours):
                        recent_resolution_hours.append(resolution)
        if resolution_hours:
            summary["resolved_count"] = len(resolution_hours)
            summary["avg_resolution_hours"] = round(
                sum(resolution_hours) / float(len(resolution_hours)),
                2,
            )
            summary["max_resolution_hours"] = round(max(resolution_hours), 2)
        if recent_resolution_hours:
            summary["recent_resolved_count"] = len(recent_resolution_hours)
            summary["recent_avg_resolution_hours"] = round(
                sum(recent_resolution_hours) / float(len(recent_resolution_hours)),
                2,
            )
        summary["pending_estimate"] = max(summary["queued"] - summary["resolved_count"], 0)
        summary["backlog_pressure_hours"] = round(
            summary["pending_estimate"] * summary["avg_resolution_hours"],
            2,
        )
        if summary["resolved_count"] > 0:
            summary["approval_rate"] = round(
                summary["approved"] / float(summary["resolved_count"]),
                4,
            )
            summary["rejection_rate"] = round(
                summary["rejected"] / float(summary["resolved_count"]),
                4,
            )
        if summary["recent_resolved_count"] > 0:
            summary["recent_approval_rate"] = round(
                summary["recent_approved"] / float(summary["recent_resolved_count"]),
                4,
            )
            summary["recent_rejection_rate"] = round(
                summary["recent_rejected"] / float(summary["recent_resolved_count"]),
                4,
            )
        return summary

    def approve_review_candidate(self, candidate_policy_path: str) -> bool:
        path = str(Path(candidate_policy_path).expanduser())
        queue = self.read_review_queue()
        remaining = [item for item in queue if item.get("candidate_policy_path") != path]
        matched = len(queue) != len(remaining)
        if not matched:
            return False
        self._write_review_queue(remaining)
        self._promote_candidate(path)
        self._append_history(
            MemoryUpgradeDecision(
                promoted=True,
                comparison={},
                report_path="",
                candidate_policy_path=path,
                reason="approved_review",
            )
        )
        self._append_review_event("approved", candidate_policy_path=path)
        return True

    def reject_review_candidate(self, candidate_policy_path: str) -> bool:
        path = str(Path(candidate_policy_path).expanduser())
        queue = self.read_review_queue()
        remaining = [item for item in queue if item.get("candidate_policy_path") != path]
        matched = len(queue) != len(remaining)
        if not matched:
            return False
        self._write_review_queue(remaining)
        self._append_history(
            MemoryUpgradeDecision(
                promoted=False,
                comparison={},
                report_path="",
                candidate_policy_path=path,
                reason="rejected_review",
            )
        )
        self._append_review_event("rejected", candidate_policy_path=path)
        return True

    def cleanup_artifacts(
        self,
        candidate_dir: str = "",
        reports_dir: str = "",
        keep_candidates: int = 24,
        keep_reports: int = 48,
    ) -> dict:
        candidate_root = Path(candidate_dir or self.default_candidate_dir()).expanduser()
        reports_root = Path(reports_dir or self.default_reports_dir()).expanduser()
        protected_candidates = {
            str(Path(item.get("candidate_policy_path", "")).expanduser())
            for item in self.read_review_queue()
            if item.get("candidate_policy_path")
        }
        cycle_summary = self.read_cycle_summary()
        for item in cycle_summary.get("ranking", []):
            candidate_path = item.get("candidate_policy_path")
            if candidate_path:
                protected_candidates.add(str(Path(candidate_path).expanduser()))

        removed_candidates = self._cleanup_files(
            root=candidate_root,
            pattern="*.json",
            keep=max(keep_candidates, 0),
            protected=protected_candidates,
        )
        removed_reports = self._cleanup_files(
            root=reports_root,
            pattern="*.json",
            keep=max(keep_reports, 0),
        )
        return {
            "removed_candidates": removed_candidates,
            "removed_reports": removed_reports,
            "kept_candidates": keep_candidates,
            "kept_reports": keep_reports,
        }

    def _promote_candidate(self, candidate_policy_path: str) -> None:
        candidate_store = MemoryPolicyStore(candidate_policy_path)
        target_store = MemoryPolicyStore(self.cfg.memory_policy_path)
        target_store.save(candidate_store.load(), reason="promote_candidate")

    def _select_best_candidate(self, decisions: list[MemoryUpgradeDecision]) -> MemoryUpgradeDecision:
        return max(
            decisions,
            key=lambda decision: (
                1 if decision.promoted else 0,
                self._decision_score(decision.comparison),
                decision.candidate_policy_path,
            ),
        )

    def _decision_score(self, comparison: dict) -> float:
        # Prefer composite score delta when available.
        composite_delta = comparison.get("composite_score_delta")
        if composite_delta is not None:
            return float(composite_delta)
        return (
            float(comparison.get("avg_query_overlap_delta", 0.0))
            + float(comparison.get("avg_continuation_overlap_delta", 0.0))
            + float(comparison.get("avg_response_overlap_delta", 0.0))
            + 0.35 * float(comparison.get("avg_focus_score_delta", 0.0))
            + 0.35 * float(comparison.get("avg_value_density_delta", 0.0))
            + 0.25 * float(comparison.get("avg_specificity_delta", 0.0))
            + 0.4 * float(comparison.get("avg_grounding_score_delta", 0.0))
            + 0.4 * float(comparison.get("avg_coverage_score_delta", 0.0))
        )

    def _append_history(self, decision: MemoryUpgradeDecision) -> None:
        payload = {
            "timestamp": _utc_now_iso(),
            "promoted": decision.promoted,
            "reason": decision.reason,
            "comparison": decision.comparison,
            "report_path": decision.report_path,
            "candidate_policy_path": decision.candidate_policy_path,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_cycle_summary(self, decisions: list[MemoryUpgradeDecision], cycle_summary_path: str) -> None:
        path = Path(cycle_summary_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        metric_summary = self._summarize_decision_metrics(decisions)
        ranking = sorted(
            [
                {
                    "candidate_policy_path": decision.candidate_policy_path,
                    "reason": decision.reason,
                    "score": round(self._decision_score(decision.comparison), 4),
                    "promoted": decision.promoted,
                }
                for decision in decisions
            ],
            key=lambda item: (item["score"], item["candidate_policy_path"]),
            reverse=True,
        )
        payload = {
            "updated_at": _utc_now_iso(),
            "num_candidates": len(decisions),
            "num_promoted": sum(1 for decision in decisions if decision.promoted),
            "num_pending_review": sum(1 for decision in decisions if decision.reason == "pending_review"),
            "metric_summary": metric_summary,
            "ranking": ranking,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        with self.cycle_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _enqueue_review(self, decision: MemoryUpgradeDecision) -> None:
        existing = self.read_review_queue()
        normalized_path = str(Path(decision.candidate_policy_path).expanduser())
        if any(item.get("candidate_policy_path") == normalized_path for item in existing):
            return
        payload = {
            "timestamp": _utc_now_iso(),
            "candidate_policy_path": normalized_path,
            "report_path": decision.report_path,
            "comparison": decision.comparison,
            "reason": decision.reason,
        }
        with self.review_queue_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _append_review_event(
        self,
        event: str,
        decision: MemoryUpgradeDecision | None = None,
        candidate_policy_path: str = "",
    ) -> None:
        path = candidate_policy_path or (decision.candidate_policy_path if decision is not None else "")
        payload = {
            "timestamp": _utc_now_iso(),
            "event": event,
            "candidate_policy_path": str(Path(path).expanduser()) if path else "",
        }
        if decision is not None:
            payload["reason"] = decision.reason
            payload["report_path"] = decision.report_path
            payload["comparison"] = decision.comparison
        with self.review_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_review_queue(self, items: list[dict]) -> None:
        with self.review_queue_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _attach_cleanup_to_cycle_summary(self, cycle_summary_path: str, cleanup: dict) -> None:
        path = Path(cycle_summary_path).expanduser()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        payload["cleanup"] = cleanup
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _cleanup_files(
        self,
        root: Path,
        pattern: str,
        keep: int,
        protected: set[str] | None = None,
    ) -> int:
        if not root.exists():
            return 0
        protected = protected or set()
        files = sorted(
            [path for path in root.glob(pattern) if path.is_file()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        kept = 0
        removed = 0
        for path in files:
            normalized = str(path.expanduser())
            if normalized in protected:
                kept += 1
                continue
            if kept < keep:
                kept += 1
                continue
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def _summarize_decision_metrics(self, decisions: list[MemoryUpgradeDecision]) -> dict:
        if not decisions:
            return {}
        keys = [
            "avg_query_overlap_delta",
            "avg_continuation_overlap_delta",
            "avg_response_overlap_delta",
            "avg_specificity_delta",
            "avg_focus_score_delta",
            "avg_value_density_delta",
            "avg_grounding_score_delta",
            "avg_coverage_score_delta",
            "composite_score_delta",
        ]
        summary: dict[str, float | int | str] = {
            "best_candidate": "",
            "best_reason": "",
            "best_score": 0.0,
        }
        for key in keys:
            values = [float(decision.comparison.get(key, 0.0)) for decision in decisions]
            summary[f"{key}_avg"] = round(sum(values) / float(len(values)), 4)
            summary[f"{key}_max"] = round(max(values), 4)
            summary[f"{key}_min"] = round(min(values), 4)
        best = max(decisions, key=lambda decision: self._decision_score(decision.comparison))
        summary["best_candidate"] = best.candidate_policy_path
        summary["best_reason"] = best.reason
        summary["best_score"] = round(self._decision_score(best.comparison), 4)
        return summary
