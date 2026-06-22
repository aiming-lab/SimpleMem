"""
Event Manager for Omni-Memory.

Manages automatic event creation, grouping, and summarization.
"""

import logging
import time
from typing import Optional, List, Dict, Any

from simplemem.multimodal.core.mau import MultimodalAtomicUnit, ModalityType
from simplemem.multimodal.core.event import EventNode, EventLevel
from simplemem.multimodal.core.config import OmniMemoryConfig, EventConfig
from simplemem.multimodal.graph.event_store import EventStore
from simplemem.multimodal.storage.mau_store import MAUStore

logger = logging.getLogger(__name__)


class EventManager:
    """
    Event Manager for automatic event organization.

    Responsibilities:
    1. Group MAUs into events based on time/context
    2. Generate event summaries
    3. Manage event hierarchy
    4. Support hierarchical retrieval
    """

    def __init__(
        self,
        event_store: EventStore,
        mau_store: MAUStore,
        config: Optional[OmniMemoryConfig] = None,
    ):
        self.event_store = event_store
        self.mau_store = mau_store
        self.config = config or OmniMemoryConfig()
        self.event_config = self.config.event

        # Track current open event per session
        self._current_events: Dict[str, str] = {}  # session_id -> event_id

        # LLM client for summarization
        self._llm_client = None
    
    def _normalize_model(self, model_name: str) -> str:
        """Normalize model name to ensure correct format."""
        from simplemem.multimodal.utils.model_utils import normalize_model_name
        return normalize_model_name(model_name)

    def _get_llm_client(self):
        """Get or create LLM client."""
        if self._llm_client is None:
            from openai import OpenAI
            import httpx
            # Only pass non-None parameters to avoid compatibility issues
            client_kwargs = {}
            if self.config.llm.api_key is not None:
                client_kwargs["api_key"] = self.config.llm.api_key
            if self.config.llm.api_base_url is not None:
                client_kwargs["base_url"] = self.config.llm.api_base_url
            
            # Create http_client explicitly to avoid proxies parameter issues
            # This prevents OpenAI SDK from reading proxy settings from environment variables
            http_client = httpx.Client()
            client_kwargs["http_client"] = http_client
            
            self._llm_client = OpenAI(**client_kwargs)
        return self._llm_client

    def get_or_create_event(
        self,
        session_id: str,
        timestamp: Optional[float] = None,
    ) -> EventNode:
        """
        Get current event or create new one for session.

        Events are created based on time windows.
        """
        timestamp = timestamp or time.time()

        # Check for existing open event
        if session_id in self._current_events:
            event_id = self._current_events[session_id]
            event = self.event_store.get(event_id)

            if event and event.time_end is None:
                # Check if within time window
                elapsed = timestamp - event.time_start
                if elapsed < self.event_config.event_time_window_seconds:
                    return event

                # Time window expired, close current event
                self.close_event(event_id)

        # Create new event
        event = EventNode(
            session_id=session_id,
            time_start=timestamp,
            level=EventLevel.SUMMARY,
        )
        self.event_store.add(event)
        self._current_events[session_id] = event.event_id

        logger.info(f"Created new event: {event.event_id} for session {session_id}")
        return event

    def add_mau_to_event(
        self,
        mau: MultimodalAtomicUnit,
        event_id: Optional[str] = None,
    ) -> str:
        """
        Add a MAU to an event.

        If event_id not specified, uses current event for session.

        Returns:
            The event ID the MAU was added to
        """
        session_id = mau.metadata.session_id or "default"

        if event_id:
            event = self.event_store.get(event_id)
        else:
            event = self.get_or_create_event(session_id, mau.timestamp)

        if not event:
            event = self.get_or_create_event(session_id, mau.timestamp)

        # Add MAU to event
        event.add_mau(mau.id, mau.modality_type.value)

        # Link MAU to event
        mau.set_event(event.event_id)

        # Update event in store
        self.event_store.update(event)

        return event.event_id

    def close_event(
        self,
        event_id: str,
        generate_summary: bool = True,
    ) -> Optional[EventNode]:
        """
        Close an event and optionally generate summary.

        Args:
            event_id: The event to close
            generate_summary: Whether to generate summary from MAUs

        Returns:
            The closed EventNode
        """
        event = self.event_store.get(event_id)
        if not event:
            return None

        # Close the event
        event.close_event()

        # Generate summary if enabled
        if generate_summary and self.event_config.summarize_on_close:
            summary = self._generate_event_summary(event)
            if summary:
                event.event_summary = summary

        # Update store
        self.event_store.update(event)

        # Remove from current events
        for session_id, eid in list(self._current_events.items()):
            if eid == event_id:
                del self._current_events[session_id]

        logger.info(f"Closed event: {event_id}")
        return event

    def _generate_event_summary(self, event: EventNode) -> str:
        """Generate summary from event's MAUs."""
        if not event.children_mau_ids:
            return "Empty event"

        # Get MAU summaries
        maus = self.mau_store.get_batch(
            event.children_mau_ids[:self.event_config.max_maus_for_summary]
        )

        if not maus:
            return "Event with no accessible memories"

        # Collect MAU summaries
        summaries = []
        for mau in maus:
            summaries.append(f"[{mau.modality_type.value}] {mau.summary}")

        summaries_text = "\n".join(summaries)

        # Generate event summary via LLM
        client = self._get_llm_client()
        try:
            response = client.chat.completions.create(
                model=self._normalize_model(self.config.llm.summary_model),
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a concise one-sentence summary of an event based on its component memories."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this event:\n{summaries_text}"
                    }
                ],
                temperature=0,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Event summary generation failed: {e}")
            return f"Event with {len(maus)} memories"

    def create_parent_event(
        self,
        child_event_ids: List[str],
        session_id: Optional[str] = None,
    ) -> Optional[EventNode]:
        """
        Create a parent event that groups child events.

        Useful for creating higher-level abstractions.
        """
        if not child_event_ids:
            return None

        # Get child events
        children = [self.event_store.get(eid) for eid in child_event_ids]
        children = [c for c in children if c is not None]

        if not children:
            return None

        # Determine time range
        time_start = min(c.time_start for c in children)
        time_end = max(c.time_end or c.time_start for c in children)

        # Create parent event
        parent = EventNode(
            session_id=session_id or children[0].session_id,
            time_start=time_start,
            time_end=time_end,
            level=EventLevel.SUMMARY,
            child_event_ids=child_event_ids,
        )

        # Update children
        for child in children:
            child.set_parent(parent.event_id)
            self.event_store.update(child)

        # Generate summary for parent
        child_summaries = [c.event_summary for c in children if c.event_summary]
        if child_summaries:
            parent.event_summary = self._summarize_child_events(child_summaries)

        self.event_store.add(parent)
        return parent

    def _summarize_child_events(self, summaries: List[str]) -> str:
        """Summarize child event summaries."""
        client = self._get_llm_client()
        try:
            response = client.chat.completions.create(
                model=self._normalize_model(self.config.llm.summary_model),
                messages=[
                    {
                        "role": "system",
                        "content": "Create a brief high-level summary combining these events."
                    },
                    {
                        "role": "user",
                        "content": f"Events:\n" + "\n".join(f"- {s}" for s in summaries)
                    }
                ],
                temperature=0,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Child event summary failed: {e}")
            return f"Period with {len(summaries)} events"

    def get_event_with_maus(
        self,
        event_id: str,
        level: EventLevel = EventLevel.SUMMARY,
    ) -> Dict[str, Any]:
        """
        Get event with its MAUs at specified detail level.

        Level 1 (SUMMARY): Event summary only
        Level 2 (DETAILS): Event + MAU summaries
        Level 3 (EVIDENCE): Event + full MAU details
        """
        event = self.event_store.get(event_id)
        if not event:
            return {}

        result = {
            "event": event.get_summary_dict(),
        }

        if level == EventLevel.SUMMARY:
            return result

        # Level 2+: Add MAU summaries
        mau_summaries = self.mau_store.get_summaries(event.children_mau_ids)
        result["mau_summaries"] = mau_summaries

        if level == EventLevel.DETAILS:
            result["event"] = event.get_details_dict()
            return result

        # Level 3: Full MAU data
        maus = self.mau_store.get_batch(event.children_mau_ids)
        result["maus"] = [m.to_dict() for m in maus]
        result["event"] = event.get_evidence_dict()

        return result

    def find_related_events(
        self,
        event_id: str,
        limit: int = 5,
    ) -> List[EventNode]:
        """Find events related to a given event."""
        event = self.event_store.get(event_id)
        if not event:
            return []

        # Get events in same session
        related = []
        if event.session_id:
            session_events = self.event_store.get_by_session(event.session_id)
            related.extend([e for e in session_events if e.event_id != event_id])

        # Get temporally adjacent events
        time_range_events = self.event_store.get_by_time_range(
            event.time_start - 3600,  # 1 hour before
            (event.time_end or event.time_start) + 3600,  # 1 hour after
        )
        for e in time_range_events:
            if e.event_id != event_id and e not in related:
                related.append(e)

        # Sort by time proximity
        related.sort(key=lambda e: abs(e.time_start - event.time_start))
        return related[:limit]

    def close_all_open_events(self):
        """Close all open events (e.g., at session end)."""
        for event_id in list(self._current_events.values()):
            self.close_event(event_id)

    def cleanup_empty_events(self) -> int:
        """Remove events with no MAUs."""
        deleted = 0
        for event in list(self.event_store._id_index.values()):
            if not event.children_mau_ids:
                self.event_store.delete(event.event_id)
                deleted += 1
        return deleted
