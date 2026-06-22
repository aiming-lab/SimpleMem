"""
Event Store for Omni-Memory.

Persistent storage for EventNodes with efficient retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import threading

from simplemem.multimodal.core.event import EventNode, EventLevel, EventHierarchy
from simplemem.multimodal.core.config import StorageConfig

logger = logging.getLogger(__name__)


class EventStore:
    """
    Persistent store for EventNodes.

    Provides:
    - CRUD operations for events
    - Time-based retrieval
    - Hierarchical traversal
    - Session-based organization
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        config: Optional[StorageConfig] = None,
    ):
        self.config = config or StorageConfig()
        self.storage_path = Path(storage_path or self.config.index_dir) / "events"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory indices
        self._id_index: Dict[str, EventNode] = {}
        self._time_index: Dict[str, List[str]] = {}  # date -> [event_ids]
        self._session_index: Dict[str, List[str]] = {}  # session_id -> [event_ids]
        self._hierarchy: EventHierarchy = EventHierarchy()

        self._lock = threading.RLock()

        # Load existing events
        self._load_events()

    def _get_storage_file(self, date_key: str) -> Path:
        """Get storage file for a date."""
        return self.storage_path / f"events_{date_key}.jsonl"

    def _get_date_key(self, timestamp: float) -> str:
        """Get date key from timestamp."""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

    def _load_events(self):
        """Load events from storage files."""
        with self._lock:
            for file_path in self.storage_path.glob("events_*.jsonl"):
                date_key = file_path.stem.replace("events_", "")

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            event = EventNode.from_dict(data)
                            self._index_event(event, date_key)
                        except json.JSONDecodeError:
                            continue

        logger.info(f"Loaded {len(self._id_index)} events from storage")

    def _index_event(self, event: EventNode, date_key: Optional[str] = None):
        """Index an event in memory."""
        self._id_index[event.event_id] = event

        if date_key is None:
            date_key = self._get_date_key(event.time_start)

        if date_key not in self._time_index:
            self._time_index[date_key] = []
        if event.event_id not in self._time_index[date_key]:
            self._time_index[date_key].append(event.event_id)

        if event.session_id:
            if event.session_id not in self._session_index:
                self._session_index[event.session_id] = []
            if event.event_id not in self._session_index[event.session_id]:
                self._session_index[event.session_id].append(event.event_id)

        self._hierarchy.add_event(event)

    def add(self, event: EventNode) -> str:
        """
        Add an event to the store.

        Args:
            event: The EventNode to store

        Returns:
            The event ID
        """
        with self._lock:
            date_key = self._get_date_key(event.time_start)
            file_path = self._get_storage_file(date_key)

            # Write to file
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(event.to_json() + '\n')

            # Index in memory
            self._index_event(event, date_key)

        logger.debug(f"Added event: {event.event_id}")
        return event.event_id

    def get(self, event_id: str) -> Optional[EventNode]:
        """Get an event by ID."""
        with self._lock:
            return self._id_index.get(event_id)

    def update(self, event: EventNode) -> bool:
        """
        Update an existing event.

        Note: Rewrites the storage file.
        """
        with self._lock:
            if event.event_id not in self._id_index:
                return False

            old_event = self._id_index[event.event_id]
            date_key = self._get_date_key(old_event.time_start)
            file_path = self._get_storage_file(date_key)

            if not file_path.exists():
                logger.warning("Event storage file missing, skip update: %s", file_path)
                self._id_index[event.event_id] = event
                self._hierarchy.events[event.event_id] = event
                return True

            # Read all events (utf-8-sig strips BOM if present)
            events = []
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.warning("Skip invalid JSON line in %s: %s", file_path, e)
                            continue
                        if data.get('event_id') == event.event_id:
                            events.append(event.to_dict())
                        else:
                            events.append(data)
            except OSError as e:
                logger.warning("Cannot read event file %s: %s", file_path, e)
                self._id_index[event.event_id] = event
                self._hierarchy.events[event.event_id] = event
                return True

            # Rewrite file
            with open(file_path, 'w', encoding='utf-8') as f:
                for e in events:
                    f.write(json.dumps(e, ensure_ascii=False) + '\n')

            # Update index
            self._id_index[event.event_id] = event
            self._hierarchy.events[event.event_id] = event

        return True

    def delete(self, event_id: str) -> bool:
        """Delete an event."""
        with self._lock:
            if event_id not in self._id_index:
                return False

            event = self._id_index[event_id]
            date_key = self._get_date_key(event.time_start)
            file_path = self._get_storage_file(date_key)

            if not file_path.exists():
                # File already missing, just remove from in-memory indices
                del self._id_index[event_id]
                for date_events in self._time_index.values():
                    if event_id in date_events:
                        date_events.remove(event_id)
                for session_events in self._session_index.values():
                    if event_id in session_events:
                        session_events.remove(event_id)
                if event_id in self._hierarchy.events:
                    del self._hierarchy.events[event_id]
                return True

            # Read all events except the one to delete (utf-8-sig strips BOM if present)
            events = []
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.warning("Skip invalid JSON line in %s: %s", file_path, e)
                            continue
                        if data.get('event_id') != event_id:
                            events.append(data)
            except OSError as e:
                logger.warning("Cannot read event file %s: %s", file_path, e)
                del self._id_index[event_id]
                for date_events in self._time_index.values():
                    if event_id in date_events:
                        date_events.remove(event_id)
                for session_events in self._session_index.values():
                    if event_id in session_events:
                        session_events.remove(event_id)
                if event_id in self._hierarchy.events:
                    del self._hierarchy.events[event_id]
                return True

            # Rewrite file
            with open(file_path, 'w', encoding='utf-8') as f:
                for e in events:
                    f.write(json.dumps(e, ensure_ascii=False) + '\n')

            # Remove from indices
            del self._id_index[event_id]
            for date_events in self._time_index.values():
                if event_id in date_events:
                    date_events.remove(event_id)
            for session_events in self._session_index.values():
                if event_id in session_events:
                    session_events.remove(event_id)
            if event_id in self._hierarchy.events:
                del self._hierarchy.events[event_id]

        return True

    def get_by_time_range(
        self,
        start_time: float,
        end_time: float,
    ) -> List[EventNode]:
        """Get events within a time range."""
        results = []

        with self._lock:
            for event in self._id_index.values():
                if event.time_start >= start_time:
                    if event.time_end is None or event.time_start <= end_time:
                        results.append(event)

        return sorted(results, key=lambda e: e.time_start)

    def get_by_session(self, session_id: str) -> List[EventNode]:
        """Get all events in a session."""
        with self._lock:
            event_ids = self._session_index.get(session_id, [])
            return [self._id_index[eid] for eid in event_ids if eid in self._id_index]

    def get_recent(self, limit: int = 10) -> List[EventNode]:
        """Get most recent events."""
        with self._lock:
            events = list(self._id_index.values())
        events.sort(key=lambda e: e.time_start, reverse=True)
        return events[:limit]

    def get_open_events(self) -> List[EventNode]:
        """Get events that haven't been closed."""
        with self._lock:
            return [e for e in self._id_index.values() if e.time_end is None]

    def get_hierarchy(self) -> EventHierarchy:
        """Get the event hierarchy."""
        return self._hierarchy

    def get_root_events(self) -> List[EventNode]:
        """Get top-level events (no parent)."""
        with self._lock:
            return [
                self._id_index[eid]
                for eid in self._hierarchy.root_event_ids
                if eid in self._id_index
            ]

    def get_children(self, event_id: str) -> List[EventNode]:
        """Get child events."""
        return self._hierarchy.get_children(event_id)

    def get_summaries(
        self,
        event_ids: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get lightweight event summaries."""
        results = []

        with self._lock:
            if event_ids:
                events = [self._id_index[eid] for eid in event_ids if eid in self._id_index]
            else:
                events = list(self._id_index.values())[:limit]

            for event in events[:limit]:
                results.append(event.get_summary_dict())

        return results

    def count(self) -> int:
        """Get total event count."""
        with self._lock:
            return len(self._id_index)

    def count_by_session(self) -> Dict[str, int]:
        """Get event count per session."""
        with self._lock:
            return {sid: len(eids) for sid, eids in self._session_index.items()}
