"""
Event Node - Hierarchical event organization for Omni-Memory.

Events group related MAUs together and provide hierarchical organization
for efficient retrieval with progressive detail expansion.
"""

import uuid
import time
from enum import IntEnum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json


class EventLevel(IntEnum):
    """
    Hierarchical levels for event organization.

    Level 1: High-level event summaries (most token-efficient)
    Level 2: Sub-events and semantic details
    Level 3: Raw multimodal evidence (most expensive)
    """
    SUMMARY = 1      # High-level summary only
    DETAILS = 2      # Sub-events and semantic details
    EVIDENCE = 3     # Raw multimodal evidence


@dataclass
class EventNode:
    """
    Event Node - Groups related MAUs into semantic events.

    Events enable:
    1. Hierarchical retrieval (summary -> details -> evidence)
    2. Cross-modal linking (same event contains text, image, audio)
    3. Temporal organization of memories
    4. Query-driven selective expansion
    """

    # Core identification
    event_id: str = field(default_factory=lambda: f"event_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}")

    # Temporal bounds
    time_start: float = field(default_factory=time.time)
    time_end: Optional[float] = None

    # Hierarchical level
    level: EventLevel = EventLevel.SUMMARY

    # Summary and description
    event_summary: str = ""  # One-sentence summary
    event_description: Optional[str] = None  # Detailed description (Level 2)

    # Embedding for event-level retrieval
    embedding: Optional[List[float]] = None

    # Child MAU references
    children_mau_ids: List[str] = field(default_factory=list)

    # Hierarchical structure
    parent_event_id: Optional[str] = None
    child_event_ids: List[str] = field(default_factory=list)

    # Metadata
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    modalities_present: List[str] = field(default_factory=list)  # Which modalities are in this event

    # Statistics
    mau_count: int = 0
    total_duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert EventNode to dictionary."""
        return {
            "event_id": self.event_id,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "level": self.level.value,
            "event_summary": self.event_summary,
            "event_description": self.event_description,
            "embedding": self.embedding,
            "children_mau_ids": self.children_mau_ids,
            "parent_event_id": self.parent_event_id,
            "child_event_ids": self.child_event_ids,
            "session_id": self.session_id,
            "tags": self.tags,
            "modalities_present": self.modalities_present,
            "mau_count": self.mau_count,
            "total_duration_ms": self.total_duration_ms,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventNode":
        """Create EventNode from dictionary."""
        level = EventLevel(data.pop("level", 1))
        return cls(level=level, **data)

    @classmethod
    def from_json(cls, json_str: str) -> "EventNode":
        """Create EventNode from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def add_mau(self, mau_id: str, modality: str) -> None:
        """Add a MAU to this event."""
        if mau_id not in self.children_mau_ids:
            self.children_mau_ids.append(mau_id)
            self.mau_count = len(self.children_mau_ids)
            if modality not in self.modalities_present:
                self.modalities_present.append(modality)

    def remove_mau(self, mau_id: str) -> None:
        """Remove a MAU from this event."""
        if mau_id in self.children_mau_ids:
            self.children_mau_ids.remove(mau_id)
            self.mau_count = len(self.children_mau_ids)

    def add_child_event(self, child_event_id: str) -> None:
        """Add a child event (sub-event)."""
        if child_event_id not in self.child_event_ids:
            self.child_event_ids.append(child_event_id)

    def set_parent(self, parent_event_id: str) -> None:
        """Set parent event."""
        self.parent_event_id = parent_event_id

    def close_event(self, end_time: Optional[float] = None) -> None:
        """Close the event with end time."""
        self.time_end = end_time or time.time()

    def add_tag(self, tag: str) -> None:
        """Add a tag for filtering."""
        if tag not in self.tags:
            self.tags.append(tag)

    def get_time_range(self) -> tuple:
        """Get time range as tuple."""
        return (self.time_start, self.time_end)

    def get_duration_seconds(self) -> Optional[float]:
        """Get event duration in seconds."""
        if self.time_end is None:
            return None
        return self.time_end - self.time_start

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get Level 1 summary only (most token-efficient)."""
        return {
            "event_id": self.event_id,
            "event_summary": self.event_summary,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "mau_count": self.mau_count,
            "modalities_present": self.modalities_present,
        }

    def get_details_dict(self) -> Dict[str, Any]:
        """Get Level 2 details (moderate token usage)."""
        result = self.get_summary_dict()
        result.update({
            "event_description": self.event_description,
            "tags": self.tags,
            "child_event_ids": self.child_event_ids,
        })
        return result

    def get_evidence_dict(self) -> Dict[str, Any]:
        """Get Level 3 with MAU references (prepare for raw data loading)."""
        result = self.get_details_dict()
        result.update({
            "children_mau_ids": self.children_mau_ids,
        })
        return result

    def __repr__(self) -> str:
        return (
            f"Event(id={self.event_id}, level={self.level.name}, "
            f"summary='{self.event_summary[:40]}...', maus={self.mau_count})"
        )


@dataclass
class EventHierarchy:
    """
    Manages hierarchical event structure for efficient traversal.
    """

    events: Dict[str, EventNode] = field(default_factory=dict)
    root_event_ids: List[str] = field(default_factory=list)

    def add_event(self, event: EventNode) -> None:
        """Add an event to the hierarchy."""
        self.events[event.event_id] = event
        if event.parent_event_id is None:
            if event.event_id not in self.root_event_ids:
                self.root_event_ids.append(event.event_id)

    def get_event(self, event_id: str) -> Optional[EventNode]:
        """Get event by ID."""
        return self.events.get(event_id)

    def get_children(self, event_id: str) -> List[EventNode]:
        """Get child events."""
        event = self.events.get(event_id)
        if not event:
            return []
        return [self.events[cid] for cid in event.child_event_ids if cid in self.events]

    def get_ancestors(self, event_id: str) -> List[EventNode]:
        """Get all ancestor events from root to parent."""
        ancestors = []
        event = self.events.get(event_id)
        while event and event.parent_event_id:
            parent = self.events.get(event.parent_event_id)
            if parent:
                ancestors.insert(0, parent)
                event = parent
            else:
                break
        return ancestors

    def get_events_at_level(self, level: EventLevel) -> List[EventNode]:
        """Get all events at a specific level."""
        return [e for e in self.events.values() if e.level == level]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hierarchy."""
        return {
            "events": {eid: e.to_dict() for eid, e in self.events.items()},
            "root_event_ids": self.root_event_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventHierarchy":
        """Deserialize hierarchy."""
        hierarchy = cls()
        hierarchy.root_event_ids = data.get("root_event_ids", [])
        for eid, edata in data.get("events", {}).items():
            hierarchy.events[eid] = EventNode.from_dict(edata)
        return hierarchy
