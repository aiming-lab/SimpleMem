# pyright: reportMissingImports=false
"""
Progressive context injection for SimpleMem-Cross.

Builds a token-budgeted ContextBundle at session start by retrieving
relevant past memory (summaries, observations, semantic matches) and
packaging it for the agent's system prompt.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, TypeVar

from cross.storage_iris import CrossSessionVectorStore
from cross.storage_sqlite import SQLiteStorage
from cross.types import (
    ContextBundle,
    CrossMemoryEntry,
    CrossObservation,
    SessionSummary,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Token estimation helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Simple word-count-based token estimate.

    Roughly approximates sub-word tokeniser output by splitting on
    whitespace.  Cheap enough to call thousands of times without any
    external dependency.
    """
    return len(text.split())


def _text_for_summary(summary: SessionSummary) -> str:
    """Derive a representative text string from a SessionSummary."""
    parts: list[str] = []
    if summary.request:
        parts.append(f"Request: {summary.request}")
    if summary.investigated:
        parts.append(f"Investigated: {summary.investigated}")
    if summary.learned:
        parts.append(f"Learned: {summary.learned}")
    if summary.completed:
        parts.append(f"Completed: {summary.completed}")
    if summary.next_steps:
        parts.append(f"Next steps: {summary.next_steps}")
    return " | ".join(parts) if parts else "Session summary available."


def _text_for_observation(obs: CrossObservation) -> str:
    """Derive a representative text string from a CrossObservation."""
    detail = obs.subtitle or obs.narrative or ""
    if detail:
        return f"{obs.title}: {detail}"
    return obs.title


def _text_for_entry(entry: CrossMemoryEntry) -> str:
    """Derive a representative text string from a CrossMemoryEntry."""
    return entry.lossless_restatement


# ---------------------------------------------------------------------------
# Budget packing
# ---------------------------------------------------------------------------


def _budget_items(
    items: Sequence[T],
    text_fn: Any,
    remaining_tokens: int,
) -> tuple[list[T], int]:
    """Greedily pack *items* into a token budget.

    Parameters
    ----------
    items:
        Ordered sequence of candidate items (highest priority first).
    text_fn:
        Callable that extracts a text representation from an item so we
        can estimate its token cost.
    remaining_tokens:
        How many tokens are still available in the budget.

    Returns
    -------
    tuple of (accepted items list, tokens consumed).
    """
    accepted: list[T] = []
    consumed = 0
    for item in items:
        cost = _estimate_tokens(text_fn(item))
        if cost == 0:
            # Zero-cost items are free to include
            accepted.append(item)
            continue
        if consumed + cost > remaining_tokens:
            break
        accepted.append(item)
        consumed += cost
    return accepted, consumed


# ---------------------------------------------------------------------------
# ContextInjector
# ---------------------------------------------------------------------------


class ContextInjector:
    """Builds a token-budgeted :class:`ContextBundle` for new sessions.

    The bundle is filled progressively in priority order:

    1. **Session summaries** (highest priority) -- compact overviews of
       what happened in recent sessions.
    2. **Observations** -- fine-grained facts (decisions, bugfixes, etc.)
       from recent sessions.
    3. **Semantic search results** -- vector-similarity matches against the
       user's current prompt (only when a prompt is provided).

    Each tier is greedily packed until the token budget is exhausted.
    """

    def __init__(
        self,
        sqlite_storage: SQLiteStorage,
        vector_store: CrossSessionVectorStore,
        max_tokens: int = 2000,
    ) -> None:
        self.sqlite_storage = sqlite_storage
        self.vector_store = vector_store
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(
        self,
        tenant_id: str,
        project: str,
        user_prompt: Optional[str] = None,
    ) -> ContextBundle:
        """Build a :class:`ContextBundle` for the start of a new session.

        Parameters
        ----------
        tenant_id:
            Tenant identifier for multi-tenant isolation.
        project:
            Project name used to scope SQLite queries.
        user_prompt:
            Optional initial user message.  When provided, semantic search
            is performed against the vector store to surface relevant
            long-term memories.

        Returns
        -------
        A fully-packed ``ContextBundle`` whose
        ``total_tokens_estimate`` reflects the budget consumed.
        """
        budget_remaining = self.max_tokens
        total_tokens = 0

        # --- 1. Session summaries (highest priority) -------------------
        raw_summaries = self._fetch_summaries(project)
        summaries, tokens_used = _budget_items(
            raw_summaries,
            _text_for_summary,
            budget_remaining,
        )
        budget_remaining -= tokens_used
        total_tokens += tokens_used
        logger.debug(
            "Context injection: packed %d/%d summaries (%d tokens)",
            len(summaries),
            len(raw_summaries),
            tokens_used,
        )

        # --- 2. Observations -------------------------------------------
        raw_observations = self._fetch_observations(project)
        observations, tokens_used = _budget_items(
            raw_observations,
            _text_for_observation,
            budget_remaining,
        )
        budget_remaining -= tokens_used
        total_tokens += tokens_used
        logger.debug(
            "Context injection: packed %d/%d observations (%d tokens)",
            len(observations),
            len(raw_observations),
            tokens_used,
        )

        # --- 3. Semantic search (only when prompt provided) ------------
        memory_entries: list[CrossMemoryEntry] = []
        if user_prompt and budget_remaining > 0:
            raw_entries = self._fetch_semantic(
                user_prompt,
                tenant_id=tenant_id,
            )
            memory_entries, tokens_used = _budget_items(
                raw_entries,
                _text_for_entry,
                budget_remaining,
            )
            budget_remaining -= tokens_used
            total_tokens += tokens_used
            logger.debug(
                "Context injection: packed %d/%d semantic entries (%d tokens)",
                len(memory_entries),
                len(raw_entries),
                tokens_used,
            )

        bundle = ContextBundle(
            session_summaries=summaries,
            timeline_observations=observations,
            memory_entries=memory_entries,
            total_tokens_estimate=total_tokens,
        )
        logger.info(
            "Context bundle built: %d summaries, %d observations, "
            "%d memory entries, ~%d tokens",
            len(summaries),
            len(observations),
            len(memory_entries),
            total_tokens,
        )
        return bundle

    # ------------------------------------------------------------------
    # Token helpers (exposed for testing)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Simple word-count-based token estimate (``len(text.split())``)."""
        return _estimate_tokens(text)

    @staticmethod
    def _budget_entries(
        entries: Sequence[Any],
        max_tokens: int,
        text_fn: Any = str,
    ) -> list[Any]:
        """Greedily pack *entries* within a token budget.

        Parameters
        ----------
        entries:
            Items to pack.
        max_tokens:
            Maximum number of estimated tokens to allow.
        text_fn:
            Callable to extract text from an entry for token estimation.
            Defaults to ``str``.

        Returns
        -------
        List of entries that fit within the budget.
        """
        accepted, _ = _budget_items(entries, text_fn, max_tokens)
        return accepted

    # ------------------------------------------------------------------
    # Private data-fetching helpers
    # ------------------------------------------------------------------

    def _fetch_summaries(self, project: str) -> List[SessionSummary]:
        """Retrieve recent session summaries from SQLite."""
        try:
            return self.sqlite_storage.get_recent_summaries(project, limit=5)
        except Exception:
            logger.exception("Failed to fetch recent summaries for project %s", project)
            return []

    def _fetch_observations(self, project: str) -> List[CrossObservation]:
        """Retrieve recent observations from SQLite."""
        try:
            return self.sqlite_storage.get_recent_observations(project, limit=20)
        except Exception:
            logger.exception(
                "Failed to fetch recent observations for project %s",
                project,
            )
            return []

    def _fetch_semantic(
        self,
        query: str,
        tenant_id: str,
    ) -> list[CrossMemoryEntry]:
        """Run semantic search against the vector store."""
        try:
            return self.vector_store.semantic_search(
                query,
                top_k=10,
                tenant_id=tenant_id,
            )
        except Exception:
            logger.exception("Semantic search failed for tenant %s", tenant_id)
            return []


# ---------------------------------------------------------------------------
# ContextRenderer
# ---------------------------------------------------------------------------


class ContextRenderer:
    """Utility for rendering a :class:`ContextBundle` into prompt-ready text.

    All methods are static -- no instance state is required.
    """

    @staticmethod
    def render_for_system_prompt(
        bundle: ContextBundle,
        max_tokens: int = 1500,
    ) -> str:
        """Render the bundle and wrap it in a system-prompt section.

        The output is suitable for direct insertion into a system message,
        surrounded by ``<cross_session_memory>`` XML tags so the model can
        easily identify the injected context.

        Parameters
        ----------
        bundle:
            The context bundle to render.
        max_tokens:
            Token cap forwarded to ``bundle.render()``.

        Returns
        -------
        A string ready for inclusion in the system prompt, or an empty
        string if the bundle contains no data.
        """
        rendered = bundle.render(max_tokens=max_tokens)
        if not rendered:
            return ""
        lines = [
            "<cross_session_memory>",
            "The following is relevant context from previous sessions.",
            "Use it to inform your responses but do not repeat it verbatim.",
            "",
            rendered,
            "</cross_session_memory>",
        ]
        return "\n".join(lines)

    @staticmethod
    def render_summary_only(bundle: ContextBundle) -> str:
        """Render only the session-summaries section of the bundle.

        Parameters
        ----------
        bundle:
            The context bundle whose summaries should be rendered.

        Returns
        -------
        A newline-joined string containing only session summary lines,
        or an empty string if there are no summaries.
        """
        if not bundle.session_summaries:
            return ""
        lines: list[str] = ["Session summaries:"]
        for summary in bundle.session_summaries:
            text = _text_for_summary(summary)
            lines.append(f"- {text}")
        return "\n".join(lines)
