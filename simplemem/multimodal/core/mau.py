"""
Multimodal Atomic Unit (MAU) - The core storage unit for Omni-Memory.

MAU represents a unified atomic storage unit that treats text, image, audio,
and video equally. It stores lightweight metadata in memory while keeping
heavy raw data in cold storage with pointer references.
"""

import uuid
import time
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json


class ModalityType(str, Enum):
    """Supported modality types in Omni-Memory."""

    TEXT = "text"
    VISUAL = "visual"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


@dataclass
class QualityMetrics:
    """Quality metrics captured during ingestion."""

    trigger_score: float = 0.0  # Entropy trigger score (0-1)
    confidence: float = 0.0  # Processing confidence (0-1)
    entropy_delta: float = 0.0  # Change in entropy from previous frame/segment

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "QualityMetrics":
        return cls(**data)


@dataclass
class MAUMetadata:
    """Metadata associated with a Multimodal Atomic Unit."""

    session_id: Optional[str] = None
    source: Optional[str] = None  # e.g., "camera_front", "microphone_1", "user_input"
    tags: List[str] = field(default_factory=list)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    duration_ms: Optional[int] = None  # For audio/video segments
    frame_index: Optional[int] = None  # For video frames
    speaker_id: Optional[str] = None  # For audio with speaker detection
    # Enriched metadata fields (extracted from MAU text for better retrieval)
    persons: Optional[List[str]] = None  # Person names mentioned
    entities: Optional[List[str]] = None  # Organizations, products, etc.
    keywords: Optional[List[str]] = None  # Core keywords for BM25
    location: Optional[str] = None  # Location mentioned
    topic: Optional[str] = None  # Topic category

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "source": self.source,
            "tags": self.tags,
            "quality": self.quality.to_dict(),
            "duration_ms": self.duration_ms,
            "frame_index": self.frame_index,
            "speaker_id": self.speaker_id,
            "persons": self.persons,
            "entities": self.entities,
            "keywords": self.keywords,
            "location": self.location,
            "topic": self.topic,
        }
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MAUMetadata":
        quality_data = data.pop("quality", {})
        quality = (
            QualityMetrics.from_dict(quality_data) if quality_data else QualityMetrics()
        )
        # Filter to only known fields to avoid TypeError from extra keys
        import dataclasses
        known_fields = {f.name for f in dataclasses.fields(cls)} - {'quality'}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(quality=quality, **filtered)


@dataclass
class MAULinks:
    """Links connecting MAU to events and other MAUs."""

    event_id: Optional[str] = None  # Parent event this MAU belongs to
    prev_mau_id: Optional[str] = None  # Previous MAU in temporal sequence
    next_mau_id: Optional[str] = None  # Next MAU in temporal sequence
    related_mau_ids: List[str] = field(default_factory=list)  # Cross-modal relations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "prev": self.prev_mau_id,
            "next": self.next_mau_id,
            "related": self.related_mau_ids if self.related_mau_ids else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MAULinks":
        return cls(
            event_id=data.get("event_id"),
            prev_mau_id=data.get("prev"),
            next_mau_id=data.get("next"),
            related_mau_ids=data.get("related", []),
        )


@dataclass
class MultimodalAtomicUnit:
    """
    Multimodal Atomic Unit (MAU) - Unified storage unit for all modalities.

    Design Principles:
    1. RAM stores only lightweight fields (summary, embedding, metadata)
    2. Heavy raw data stored in cold storage via raw_pointer
    3. Details field is lazily populated only when deep query requires it

    This achieves Memory-Compute Decoupling for efficient multimodal storage.
    """

    # Core identification
    id: str = field(
        default_factory=lambda: f"mau_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    )
    timestamp: float = field(default_factory=time.time)
    modality_type: ModalityType = ModalityType.TEXT

    # Lightweight indexing fields (always in RAM)
    summary: str = ""  # Short text summary for coarse retrieval
    embedding: Optional[List[float]] = None  # Vector for dense retrieval

    # Cold storage pointer (raw data stored externally)
    raw_pointer: Optional[str] = None  # Path/URI to raw data in cold storage

    # Lazily loaded details (populated only on deep query)
    details: Optional[Dict[str, Any]] = None

    # Metadata and relations
    metadata: MAUMetadata = field(default_factory=MAUMetadata)
    links: MAULinks = field(default_factory=MAULinks)

    # Region pointers for selective loading (especially for images)
    region_pointers: Optional[List[str]] = None

    # Lifecycle management (benchmark-safe evolution)
    status: str = "ACTIVE"  # ACTIVE | ARCHIVED | PINNED
    storage_tier: str = "HOT"  # HOT | COLD
    archived_at: Optional[float] = None
    archive_reason: Optional[str] = None
    archive_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert MAU to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "modality_type": self.modality_type.value,
            "summary": self.summary,
            "embedding": self.embedding,
            "raw_pointer": self.raw_pointer,
            "details": self.details,
            "status": self.status,
            "storage_tier": self.storage_tier,
            "archived_at": self.archived_at,
            "archive_reason": self.archive_reason,
            "archive_run_id": self.archive_run_id,
            "metadata": self.metadata.to_dict(),
            "links": self.links.to_dict(),
            "region_pointers": self.region_pointers,
        }

    def to_json(self) -> str:
        """Serialize MAU to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalAtomicUnit":
        """Create MAU from dictionary."""
        data = dict(data)
        metadata = MAUMetadata.from_dict(data.pop("metadata", {}))
        links = MAULinks.from_dict(data.pop("links", {}))
        modality_type = ModalityType(data.pop("modality_type", "text"))
        valid_fields = {
            k
            for k in cls.__dataclass_fields__
            if k not in {"metadata", "links", "modality_type"}
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(
            metadata=metadata, links=links, modality_type=modality_type, **filtered_data
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MultimodalAtomicUnit":
        """Create MAU from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_lightweight_dict(self) -> Dict[str, Any]:
        """Get only lightweight fields for RAM storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "modality_type": self.modality_type.value,
            "summary": self.summary,
            "embedding": self.embedding,
            "raw_pointer": self.raw_pointer,
            "status": self.status,
            "storage_tier": self.storage_tier,
            "metadata": self.metadata.to_dict(),
            "links": self.links.to_dict(),
        }

    def has_details(self) -> bool:
        """Check if details have been loaded."""
        return self.details is not None

    def clear_details(self) -> None:
        """Clear details to free memory."""
        self.details = None

    def set_event(self, event_id: str) -> None:
        """Associate this MAU with an event."""
        self.links.event_id = event_id

    def link_previous(self, prev_mau_id: str) -> None:
        """Link to previous MAU in sequence."""
        self.links.prev_mau_id = prev_mau_id

    def link_next(self, next_mau_id: str) -> None:
        """Link to next MAU in sequence."""
        self.links.next_mau_id = next_mau_id

    def add_related(self, mau_id: str) -> None:
        """Add cross-modal relation."""
        if mau_id not in self.links.related_mau_ids:
            self.links.related_mau_ids.append(mau_id)

    def add_tag(self, tag: str) -> None:
        """Add a tag for filtering."""
        if tag not in self.metadata.tags:
            self.metadata.tags.append(tag)

    def __repr__(self) -> str:
        return (
            f"MAU(id={self.id}, type={self.modality_type.value}, "
            f"summary='{self.summary[:50]}...' if len(self.summary) > 50 else self.summary)"
        )
