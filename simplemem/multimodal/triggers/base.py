"""
Base class for entropy triggers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Dict
from enum import Enum


class TriggerDecision(str, Enum):
    """Decision made by entropy trigger."""
    ACCEPT = "accept"      # Data has significant information, process it
    REJECT = "reject"      # Data is redundant/low-info, skip it
    UNCERTAIN = "uncertain"  # Borderline case, may need further analysis


@dataclass
class TriggerResult:
    """Result from entropy trigger evaluation."""

    decision: TriggerDecision
    score: float  # Entropy/change score (0-1)
    reason: str   # Human-readable explanation

    # Optional detailed metrics
    similarity_score: Optional[float] = None  # For visual: similarity to previous
    energy_level: Optional[float] = None      # For audio: signal energy
    entropy_delta: Optional[float] = None     # Change in entropy

    # Metadata from trigger
    metadata: Optional[Dict[str, Any]] = None

    def should_process(self) -> bool:
        """Check if this data should be processed."""
        return self.decision == TriggerDecision.ACCEPT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "score": self.score,
            "reason": self.reason,
            "similarity_score": self.similarity_score,
            "energy_level": self.energy_level,
            "entropy_delta": self.entropy_delta,
            "metadata": self.metadata,
        }


class BaseTrigger(ABC):
    """
    Base class for modal entropy triggers.

    Entropy triggers are lightweight filters that run before expensive
    processing to determine if incoming data contains meaningful information.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._previous_state: Any = None

    @abstractmethod
    def evaluate(self, data: Any) -> TriggerResult:
        """
        Evaluate incoming data and decide whether to process it.

        Args:
            data: The incoming data (image, audio chunk, etc.)

        Returns:
            TriggerResult with decision and metrics
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the trigger state (e.g., clear previous frame reference)."""
        pass

    def update_state(self, new_state: Any) -> None:
        """Update the internal state after processing."""
        self._previous_state = new_state

    def get_previous_state(self) -> Any:
        """Get the previous state for comparison."""
        return self._previous_state

    def bypass(self, data: Any) -> TriggerResult:
        """Bypass the trigger and always accept."""
        return TriggerResult(
            decision=TriggerDecision.ACCEPT,
            score=1.0,
            reason="Trigger bypassed",
        )

    def __call__(self, data: Any) -> TriggerResult:
        """Convenience method to evaluate data."""
        if not self.enabled:
            return self.bypass(data)
        return self.evaluate(data)
