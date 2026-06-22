from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import EvolveMemConfig
from .promotion import MemoryPromotionCriteria
from .self_upgrade import MemorySelfUpgradeOrchestrator

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryUpgradeWorker:
    """Background worker that runs bounded memory self-upgrade cycles."""

    def __init__(
        self,
        config: EvolveMemConfig,
        window_check: Callable[[], bool] | None = None,
    ):
        self.config = config
        self.window_check = window_check
        self._stop = asyncio.Event()
        self._last_processed_mtime: float = 0.0
        self.state_path = Path(self.config.memory_dir).expanduser() / "upgrade_worker_state.json"
        self.alerts_path = Path(self.config.memory_dir).expanduser() / "upgrade_alerts.json"
        self.alerts_history_path = Path(self.config.memory_dir).expanduser() / "upgrade_alerts_history.jsonl"
        self.health_history_path = Path(self.config.memory_dir).expanduser() / "upgrade_health_history.jsonl"
        self._load_state()

    async def run(self) -> None:
        self._write_state("running", detail="worker started")
        logger.info(
            "[MemoryUpgradeWorker] started enabled=%s interval=%ss require_review=%s stale_after=%sh",
            self.config.memory_auto_upgrade_enabled,
            self.config.memory_auto_upgrade_interval_seconds,
            self.config.memory_auto_upgrade_require_review,
            self.config.memory_review_stale_after_hours,
        )
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                self._write_state("error", detail=str(exc))
                logger.warning("[MemoryUpgradeWorker] cycle failed: %s", exc, exc_info=True)
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=max(self.config.memory_auto_upgrade_interval_seconds, 30),
                )
            except asyncio.TimeoutError:
                continue
        self._write_state("stopped", detail="worker stopped")

    async def run_once(self) -> bool:
        if not self.config.memory_auto_upgrade_enabled:
            self._write_alerts([])
            self._write_health_snapshot({"level": "healthy", "reasons": []}, state="idle")
            self._write_state("idle", detail="auto upgrade disabled")
            return False
        if self.window_check is not None and not self.window_check():
            self._write_alerts([])
            self._write_health_snapshot({"level": "healthy", "reasons": []}, state="waiting_window")
            self._write_state("waiting_window", detail="no upgrade window open")
            logger.debug("[MemoryUpgradeWorker] skipped: no upgrade window open")
            return False

        records_path = Path(self.config.record_dir).expanduser() / "conversations.jsonl"
        if not records_path.exists() or records_path.stat().st_size <= 0:
            self._write_alerts([])
            self._write_health_snapshot({"level": "healthy", "reasons": []}, state="idle")
            self._write_state("idle", detail="no replay records")
            logger.debug("[MemoryUpgradeWorker] skipped: no replay records")
            return False

        mtime = records_path.stat().st_mtime
        if mtime <= self._last_processed_mtime:
            self._write_alerts([])
            self._write_health_snapshot({"level": "healthy", "reasons": []}, state="idle")
            self._write_state("idle", detail="records unchanged")
            logger.debug("[MemoryUpgradeWorker] skipped: records unchanged")
            return False

        orchestrator = MemorySelfUpgradeOrchestrator(
            self.config,
            history_path=str(Path(self.config.memory_dir).expanduser() / "upgrade_history.jsonl"),
        )
        review_summary = orchestrator.summarize_review_queue(
            stale_after_hours=self.config.memory_review_stale_after_hours
        )
        health = orchestrator.summarize_operational_health(
            stale_after_hours=self.config.memory_review_stale_after_hours
        )
        if review_summary["pending_count"] > 0:
            state = "waiting_review_stale" if review_summary["stale_count"] > 0 else "waiting_review"
            self._write_alerts(self._build_review_alerts(review_summary))
            self._write_health_snapshot(health, state=state)
            self._write_state(
                state,
                detail=(
                    "pending review queue is not empty "
                    f"(pending={review_summary['pending_count']} stale={review_summary['stale_count']} "
                    f"threshold_h={review_summary['stale_after_hours']})"
                ),
            )
            logger.info("[MemoryUpgradeWorker] skipped: pending review queue is not empty")
            return False
        # Run maintenance (expire TTL-stale, consolidate, cleanup) before upgrade cycle.
        try:
            from .manager import MemoryManager
            manager = MemoryManager.from_config(self.config)
            maint = manager.run_maintenance(self.config.memory_scope)
            expired = maint.get("expired", 0)
            if expired:
                logger.info("[MemoryUpgradeWorker] expired %d TTL-stale memories", expired)
            consolidated = maint.get("consolidated", {}).get("merged", 0)
            if consolidated:
                logger.info("[MemoryUpgradeWorker] consolidated %d near-duplicate memories", consolidated)
            manager.close()
        except Exception as exc:
            logger.debug("[MemoryUpgradeWorker] maintenance skipped: %s", exc)

        self._write_alerts([])
        self._write_health_snapshot(health, state="processing")
        self._write_state("processing", detail="running auto-upgrade cycle")
        decisions = orchestrator.run_auto_upgrade_cycle(
            replay_records_path=str(records_path),
            criteria=MemoryPromotionCriteria(min_sample_count=1),
            require_review=self.config.memory_auto_upgrade_require_review,
        )
        promoted = sum(1 for decision in decisions if decision.promoted)
        pending = sum(1 for decision in decisions if decision.reason == "pending_review")
        logger.info(
            "[MemoryUpgradeWorker] processed=%d promoted=%d pending_review=%d",
            len(decisions),
            promoted,
            pending,
        )
        self._last_processed_mtime = mtime
        self._write_state(
            "idle",
            detail=f"processed={len(decisions)} promoted={promoted} pending_review={pending}",
        )
        refreshed_health = orchestrator.summarize_operational_health(
            stale_after_hours=self.config.memory_review_stale_after_hours
        )
        self._write_health_snapshot(refreshed_health, state="idle")
        if pending > 0:
            refreshed_summary = orchestrator.summarize_review_queue(
                stale_after_hours=self.config.memory_review_stale_after_hours
            )
            self._write_alerts(self._build_review_alerts(refreshed_summary))
        else:
            self._write_alerts([])
        return True

    def stop(self) -> None:
        self._stop.set()

    def _write_state(self, state: str, detail: str = "") -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": _utc_now_iso(),
            "state": state,
            "detail": detail,
            "enabled": self.config.memory_auto_upgrade_enabled,
            "interval_seconds": self.config.memory_auto_upgrade_interval_seconds,
            "require_review": self.config.memory_auto_upgrade_require_review,
            "review_stale_after_hours": self.config.memory_review_stale_after_hours,
            "last_processed_mtime": self._last_processed_mtime,
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        try:
            self._last_processed_mtime = float(payload.get("last_processed_mtime", 0.0) or 0.0)
        except (TypeError, ValueError):
            self._last_processed_mtime = 0.0

    def _write_alerts(self, alerts: list[dict]) -> None:
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": _utc_now_iso(),
            "alerts": alerts,
        }
        self.alerts_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        with self.alerts_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_alert_history(self, limit: int = 20) -> list[dict]:
        if not self.alerts_history_path.exists():
            return []
        items: list[dict] = []
        for line in self.alerts_history_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return items

    def summarize_alert_history(self) -> dict:
        items = self.read_alert_history(limit=1000000)
        now = datetime.now(timezone.utc)
        recent_window_hours = 168
        summary = {
            "total_snapshots": len(items),
            "nonempty_snapshots": 0,
            "warning_count": 0,
            "critical_count": 0,
            "blocked_count": 0,
            "stale_count": 0,
            "nonempty_snapshot_rate": 0.0,
            "blocked_snapshot_rate": 0.0,
            "stale_snapshot_rate": 0.0,
            "recent_window_hours": recent_window_hours,
            "recent_total_snapshots": 0,
            "recent_nonempty_snapshots": 0,
            "recent_warning_count": 0,
            "recent_critical_count": 0,
            "recent_blocked_count": 0,
            "recent_stale_count": 0,
            "recent_nonempty_snapshot_rate": 0.0,
            "recent_blocked_snapshot_rate": 0.0,
            "recent_stale_snapshot_rate": 0.0,
        }
        for item in items:
            alerts = item.get("alerts", [])
            timestamp = item.get("updated_at", "")
            is_recent = False
            if isinstance(timestamp, str):
                normalized = timestamp.replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(normalized)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    else:
                        parsed = parsed.astimezone(timezone.utc)
                    is_recent = 0.0 <= (now - parsed).total_seconds() <= float(recent_window_hours * 3600)
                except ValueError:
                    is_recent = False
            if is_recent:
                summary["recent_total_snapshots"] += 1
            if alerts:
                summary["nonempty_snapshots"] += 1
                if is_recent:
                    summary["recent_nonempty_snapshots"] += 1
            for alert in alerts:
                level = str(alert.get("level", ""))
                code = str(alert.get("code", ""))
                if level == "warning":
                    summary["warning_count"] += 1
                    if is_recent:
                        summary["recent_warning_count"] += 1
                elif level == "critical":
                    summary["critical_count"] += 1
                    if is_recent:
                        summary["recent_critical_count"] += 1
                if code == "review_queue_blocked":
                    summary["blocked_count"] += 1
                    if is_recent:
                        summary["recent_blocked_count"] += 1
                elif code == "review_queue_stale":
                    summary["stale_count"] += 1
                    if is_recent:
                        summary["recent_stale_count"] += 1
        if summary["total_snapshots"] > 0:
            summary["nonempty_snapshot_rate"] = round(
                summary["nonempty_snapshots"] / float(summary["total_snapshots"]),
                4,
            )
            summary["blocked_snapshot_rate"] = round(
                summary["blocked_count"] / float(summary["total_snapshots"]),
                4,
            )
            summary["stale_snapshot_rate"] = round(
                summary["stale_count"] / float(summary["total_snapshots"]),
                4,
            )
        if summary["recent_total_snapshots"] > 0:
            summary["recent_nonempty_snapshot_rate"] = round(
                summary["recent_nonempty_snapshots"] / float(summary["recent_total_snapshots"]),
                4,
            )
            summary["recent_blocked_snapshot_rate"] = round(
                summary["recent_blocked_count"] / float(summary["recent_total_snapshots"]),
                4,
            )
            summary["recent_stale_snapshot_rate"] = round(
                summary["recent_stale_count"] / float(summary["recent_total_snapshots"]),
                4,
            )
        return summary

    def _write_health_snapshot(self, health: dict, state: str) -> None:
        self.health_history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": _utc_now_iso(),
            "state": state,
            "level": str(health.get("level", "unknown")),
            "reasons": list(health.get("reasons", [])),
            "pending_review": int(health.get("pending_review", 0) or 0),
            "stale_review": int(health.get("stale_review", 0) or 0),
            "backlog_pressure_hours": float(health.get("backlog_pressure_hours", 0.0) or 0.0),
            "recent_promoted_cycle_rate": float(health.get("recent_promoted_cycle_rate", 0.0) or 0.0),
            "recent_pending_review_cycle_rate": float(
                health.get("recent_pending_review_cycle_rate", 0.0) or 0.0
            ),
            "recent_upgrade_decisions": int(health.get("recent_upgrade_decisions", 0) or 0),
        }
        with self.health_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_health_history(self, limit: int = 20) -> list[dict]:
        if not self.health_history_path.exists():
            return []
        items: list[dict] = []
        for line in self.health_history_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return items

    def summarize_health_history(self, recent_window_hours: int = 168) -> dict:
        items = self.read_health_history(limit=1000000)
        now = datetime.now(timezone.utc)
        summary = {
            "total_snapshots": len(items),
            "healthy_count": 0,
            "warning_count": 0,
            "critical_count": 0,
            "recent_window_hours": recent_window_hours,
            "recent_snapshots": 0,
            "recent_healthy_count": 0,
            "recent_warning_count": 0,
            "recent_critical_count": 0,
            "healthy_rate": 0.0,
            "warning_rate": 0.0,
            "critical_rate": 0.0,
            "recent_healthy_rate": 0.0,
            "recent_warning_rate": 0.0,
            "recent_critical_rate": 0.0,
        }
        for item in items:
            level = str(item.get("level", ""))
            timestamp = item.get("updated_at", "")
            is_recent = False
            if isinstance(timestamp, str):
                normalized = timestamp.replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(normalized)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    else:
                        parsed = parsed.astimezone(timezone.utc)
                    is_recent = 0.0 <= (now - parsed).total_seconds() <= float(recent_window_hours * 3600)
                except ValueError:
                    is_recent = False
            if is_recent:
                summary["recent_snapshots"] += 1
            if level == "healthy":
                summary["healthy_count"] += 1
                if is_recent:
                    summary["recent_healthy_count"] += 1
            elif level == "warning":
                summary["warning_count"] += 1
                if is_recent:
                    summary["recent_warning_count"] += 1
            elif level == "critical":
                summary["critical_count"] += 1
                if is_recent:
                    summary["recent_critical_count"] += 1
        if summary["total_snapshots"] > 0:
            summary["healthy_rate"] = round(summary["healthy_count"] / float(summary["total_snapshots"]), 4)
            summary["warning_rate"] = round(summary["warning_count"] / float(summary["total_snapshots"]), 4)
            summary["critical_rate"] = round(summary["critical_count"] / float(summary["total_snapshots"]), 4)
        if summary["recent_snapshots"] > 0:
            summary["recent_healthy_rate"] = round(
                summary["recent_healthy_count"] / float(summary["recent_snapshots"]),
                4,
            )
            summary["recent_warning_rate"] = round(
                summary["recent_warning_count"] / float(summary["recent_snapshots"]),
                4,
            )
            summary["recent_critical_rate"] = round(
                summary["recent_critical_count"] / float(summary["recent_snapshots"]),
                4,
            )
        return summary

    def _build_review_alerts(self, review_summary: dict) -> list[dict]:
        if review_summary.get("pending_count", 0) <= 0:
            return []
        level = "warning"
        code = "review_queue_blocked"
        if review_summary.get("stale_count", 0) > 0:
            level = "critical"
            code = "review_queue_stale"
        return [
            {
                "level": level,
                "code": code,
                "pending_count": int(review_summary.get("pending_count", 0)),
                "stale_count": int(review_summary.get("stale_count", 0)),
                "stale_after_hours": int(review_summary.get("stale_after_hours", 0)),
                "oldest_age_hours": float(review_summary.get("oldest_age_hours", 0.0)),
            }
        ]
