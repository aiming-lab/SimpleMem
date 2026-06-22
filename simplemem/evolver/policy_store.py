from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MemoryPolicyState:
    version: int = 1
    retrieval_mode: str = "keyword"
    max_injected_units: int = 6
    max_injected_tokens: int = 800
    keyword_weight: float = 1.0
    metadata_weight: float = 0.45
    importance_weight: float = 0.5
    recency_weight: float = 0.3
    recent_bonus_hours: int = 72
    type_boosts: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class MemoryPolicyRevision:
    timestamp: str
    reason: str
    state: MemoryPolicyState


class MemoryPolicyStore:
    """JSON persistence and revision history for adaptive memory policy state."""

    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path = self.path.with_suffix(self.path.suffix + ".history.jsonl")

    def load(self) -> MemoryPolicyState:
        if not self.path.exists():
            return MemoryPolicyState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return MemoryPolicyState()
        return _state_from_dict(data)

    def save(self, state: MemoryPolicyState, reason: str = "update") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(state), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        revision = MemoryPolicyRevision(timestamp=_utc_now_iso(), reason=reason, state=state)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_revision_to_dict(revision), ensure_ascii=False) + "\n")

    def history(self) -> list[MemoryPolicyRevision]:
        if not self.history_path.exists():
            return []
        revisions: list[MemoryPolicyRevision] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            state_data = data.get("state", {})
            revisions.append(
                MemoryPolicyRevision(
                    timestamp=str(data.get("timestamp", "")),
                    reason=str(data.get("reason", "")),
                    state=_state_from_dict(state_data),
                )
            )
        return revisions

    def rollback(self, steps: int = 1) -> MemoryPolicyState:
        revisions = self.history()
        if steps < 1:
            steps = 1
        if len(revisions) <= steps:
            raise ValueError("not enough policy revisions to rollback")
        target = revisions[-(steps + 1)].state
        self.save(target, reason=f"rollback:{steps}")
        return target


def validate_policy_state(state: MemoryPolicyState) -> list[str]:
    """Validate a policy state and return a list of issues (empty if valid)."""
    issues: list[str] = []
    if state.retrieval_mode not in {"keyword", "hybrid", "embedding"}:
        issues.append(f"Invalid retrieval_mode: {state.retrieval_mode}")
    if state.max_injected_units < 1 or state.max_injected_units > 20:
        issues.append(f"max_injected_units out of range [1, 20]: {state.max_injected_units}")
    if state.max_injected_tokens < 100 or state.max_injected_tokens > 3000:
        issues.append(f"max_injected_tokens out of range [100, 3000]: {state.max_injected_tokens}")
    if state.keyword_weight < 0.0 or state.keyword_weight > 3.0:
        issues.append(f"keyword_weight out of range [0, 3]: {state.keyword_weight}")
    if state.metadata_weight < 0.0 or state.metadata_weight > 2.0:
        issues.append(f"metadata_weight out of range [0, 2]: {state.metadata_weight}")
    if state.importance_weight < 0.0 or state.importance_weight > 2.0:
        issues.append(f"importance_weight out of range [0, 2]: {state.importance_weight}")
    if state.recency_weight < 0.0 or state.recency_weight > 2.0:
        issues.append(f"recency_weight out of range [0, 2]: {state.recency_weight}")
    return issues


def _state_from_dict(data: dict) -> MemoryPolicyState:
    return MemoryPolicyState(
        **{
            key: value
            for key, value in data.items()
            if key in MemoryPolicyState.__dataclass_fields__
        }
    )


def _revision_to_dict(revision: MemoryPolicyRevision) -> dict:
    return {
        "timestamp": revision.timestamp,
        "reason": revision.reason,
        "state": asdict(revision.state),
    }
