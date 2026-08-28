# pyright: reportMissingImports=false
from __future__ import annotations

"""
Cross-session memory orchestrator for SimpleMem-Cross.

This module provides :class:`CrossMemOrchestrator`, the top-level facade
that wires together SQLite storage, IRIS vector search, session
management, context injection, and lifecycle hooks into a single,
easy-to-use entry point.

Typical usage::

    async with CrossMemOrchestrator(project="my-project") as orch:
        result = await orch.start_session("sess-001", user_prompt="Fix the login bug")
        mid = result["memory_session_id"]

        await orch.record_message(mid, "Looking at auth module...")
        await orch.record_tool_use(mid, "read_file", "auth.py", "<contents>")

        report = await orch.stop_session(mid)
        await orch.end_session(mid)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from cross.context_injector import ContextInjector, ContextRenderer
from cross.hooks import DefaultHooks
from cross.session_manager import SessionManager
from cross.storage_sqlite import SQLiteStorage

try:
    from cross.storage_iris import CrossSessionVectorStore
except ImportError:
    CrossSessionVectorStore = None  # type: ignore[assignment,misc]
from cross.storage_factory import create_sql_storage
from cross.types import ContextBundle, CrossMemoryEntry, FinalizationReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = "~/.simplemem-cross/cross_memory.db"
_DEFAULT_IRIS_TABLE = "cross_memory_entries"


# ---------------------------------------------------------------------------
# CrossMemOrchestrator
# ---------------------------------------------------------------------------


class CrossMemOrchestrator:
    """Top-level entry point for the SimpleMem-Cross system.

    The orchestrator is the **main user-facing class**.  It initialises all
    internal components (SQLite storage, IRIS vector store, session
    manager, context injector, lifecycle hooks) and exposes a clean,
    high-level API for managing cross-session memory.

    Parameters
    ----------
    project:
        Project name or path used to scope memories.
    tenant_id:
        Tenant identifier for multi-tenant isolation.
    db_path:
        Path to the SQLite database file.  Defaults to
        ``~/.simplemem-cross/cross_memory.db``.
    iris_table:
        IRIS SQL table name for the vector store.  Defaults to
        ``cross_memory_entries``.  The table is created automatically
        on first use with a HNSW index.
    use_iris_sql:
        If True, use IRIS SQL tables for session/event/observation metadata
        instead of SQLite.  Requires IRIS connection settings in config.py.
        Defaults to False (SQLite).
    lancedb_path:
        Deprecated. Ignored. Kept for backward compatibility.
    max_context_tokens:
        Maximum token budget for the context bundle assembled at
        session start.
    simplemem:
        Optional reference to a ``SimpleMemSystem`` instance (duck-typed).
        When provided, session finalization will run the SimpleMem 3-stage
        pipeline to produce rich ``MemoryEntry`` objects.
    """

    def __init__(
        self,
        project: str,
        tenant_id: str = "default",
        db_path: Optional[str] = None,
        lancedb_path: Optional[str] = None,
        iris_table: Optional[str] = None,
        use_iris_sql: bool = False,
        max_context_tokens: int = 2000,
        simplemem: Optional[Any] = None,
    ) -> None:
        self.project = project
        self.tenant_id = tenant_id

        self.sqlite_storage = create_sql_storage(
            use_iris=use_iris_sql,
            db_path=db_path or _DEFAULT_DB_PATH,
        )
        self.vector_store = CrossSessionVectorStore(
            table_name=iris_table or _DEFAULT_IRIS_TABLE,
        )

        # -- session manager ------------------------------------------------
        self.session_manager = SessionManager(
            sqlite_storage=self.sqlite_storage,
            vector_store=self.vector_store,
            simplemem=simplemem,
        )

        # -- context injection ----------------------------------------------
        self.context_injector = ContextInjector(
            sqlite_storage=self.sqlite_storage,
            vector_store=self.vector_store,
            max_tokens=max_context_tokens,
        )

        # -- lifecycle hooks (exposed for framework integrations) -----------
        self.hooks = DefaultHooks(
            session_manager=self.session_manager,
            context_injector=self.context_injector,
        )

        logger.info(
            "CrossMemOrchestrator initialised: project=%s tenant=%s "
            "sql_backend=%s iris_table=%s max_tokens=%d",
            project,
            tenant_id,
            "iris" if use_iris_sql else "sqlite",
            iris_table or _DEFAULT_IRIS_TABLE,
            max_context_tokens,
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def start_session(
        self,
        content_session_id: str,
        user_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a new cross-session memory session.

        Creates a ``SessionRecord`` in SQLite, builds a token-budgeted
        :class:`ContextBundle` containing relevant memories from previous
        sessions, and renders it into a string suitable for the agent's
        system prompt.

        Parameters
        ----------
        content_session_id:
            Unique session ID from the agent framework / host.
        user_prompt:
            The initial user prompt that started the session.  When
            provided, semantic search is used to surface relevant
            long-term memories in the context bundle.

        Returns
        -------
        dict
            ``memory_session_id`` — internal session identifier (str).
            ``context`` — rendered context string for the system prompt.
            ``context_bundle`` — the raw :class:`ContextBundle` object
            (may be ``None`` if context building failed).
        """
        try:
            session_record = await asyncio.to_thread(
                self.session_manager.start_session,
                tenant_id=self.tenant_id,
                content_session_id=content_session_id,
                project=self.project,
                user_prompt=user_prompt,
            )
            memory_session_id = session_record.memory_session_id

            context_bundle = await asyncio.to_thread(
                self._build_context_safe, user_prompt
            )
            rendered_context = self._render_context_safe(context_bundle)

            logger.info(
                "Started session %s (content_id=%s, context_tokens=%d)",
                memory_session_id,
                content_session_id,
                context_bundle.total_tokens_estimate if context_bundle else 0,
            )

            return {
                "memory_session_id": memory_session_id,
                "context": rendered_context,
                "context_bundle": context_bundle,
            }

        except Exception:
            logger.exception(
                "Failed to start session for content_id=%s",
                content_session_id,
            )
            raise

    async def record_message(
        self,
        memory_session_id: str,
        content: str,
        role: str = "user",
    ) -> None:
        """Record a user or assistant message in the active session.

        Parameters
        ----------
        memory_session_id:
            Internal session identifier returned by :meth:`start_session`.
        content:
            Raw text content of the message.
        role:
            Speaker role (``"user"``, ``"assistant"``, ``"system"``).
        """
        try:
            await asyncio.to_thread(
                self.session_manager.record_message,
                memory_session_id=memory_session_id,
                content=content,
                role=role,
            )
            logger.debug(
                "Recorded %s message for session %s (%d chars)",
                role,
                memory_session_id,
                len(content),
            )
        except Exception:
            logger.exception(
                "Failed to record message for session %s",
                memory_session_id,
            )
            raise

    async def record_tool_use(
        self,
        memory_session_id: str,
        tool_name: str,
        tool_input: str,
        tool_output: str,
    ) -> None:
        """Record a tool invocation in the active session.

        Parameters
        ----------
        memory_session_id:
            Internal session identifier.
        tool_name:
            Name of the tool that was invoked.
        tool_input:
            Serialised input passed to the tool.
        tool_output:
            Serialised output returned by the tool.
        """
        try:
            await asyncio.to_thread(
                self.session_manager.record_tool_use,
                memory_session_id=memory_session_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
            )
            logger.debug(
                "Recorded tool_use '%s' for session %s",
                tool_name,
                memory_session_id,
            )
        except Exception:
            logger.exception(
                "Failed to record tool use '%s' for session %s",
                tool_name,
                memory_session_id,
            )
            raise

    async def stop_session(
        self,
        memory_session_id: str,
    ) -> FinalizationReport:
        """Stop a session, triggering memory finalization.

        Finalization extracts observations from recorded events, optionally
        runs the SimpleMem 3-stage pipeline to produce rich memory entries,
        generates a session summary, and persists everything to storage.

        Parameters
        ----------
        memory_session_id:
            Internal session identifier.

        Returns
        -------
        FinalizationReport
            Report detailing what was produced (observation count,
            entries stored, whether a summary was generated, etc.).
        """
        try:
            report = await asyncio.to_thread(
                self.session_manager.finalize_session,
                memory_session_id=memory_session_id,
            )
            logger.info(
                "Session %s finalized: observations=%d entries=%d summary=%s",
                memory_session_id,
                report.observations_count,
                report.entries_stored,
                report.summary_generated,
            )
            return report

        except Exception:
            logger.exception("Failed to finalize session %s", memory_session_id)
            return FinalizationReport(
                memory_session_id=memory_session_id,
                observations_count=0,
                summary_generated=False,
                entries_stored=0,
                consolidation_triggered=False,
            )

    async def end_session(
        self,
        memory_session_id: str,
    ) -> None:
        """Mark the session as completed and release transient resources.

        Should be called **after** :meth:`stop_session` to perform final
        cleanup.

        Parameters
        ----------
        memory_session_id:
            Internal session identifier.
        """
        try:
            await asyncio.to_thread(
                self.session_manager.end_session,
                memory_session_id=memory_session_id,
            )
            logger.info("Ended session %s", memory_session_id)
        except Exception:
            logger.exception("Failed to end session %s", memory_session_id)

    # ------------------------------------------------------------------
    # Query / retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[CrossMemoryEntry]:
        """Search across all sessions for relevant memories.

        Performs a semantic (vector-similarity) search against the
        cross-session vector store.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[CrossMemoryEntry]
            Matching memory entries sorted by relevance.
        """
        try:
            results = self.vector_store.semantic_search(
                query=query,
                top_k=top_k,
                tenant_id=self.tenant_id,
            )
            logger.debug(
                "Search returned %d results for query='%s'",
                len(results),
                query[:80],
            )
            return results
        except Exception:
            logger.exception("Search failed for query='%s'", query[:80])
            return []

    def get_context_for_prompt(
        self,
        user_prompt: Optional[str] = None,
    ) -> str:
        """Build and render cross-session context for a system prompt.

        Assembles a token-budgeted context bundle from recent session
        summaries, observations, and (when *user_prompt* is provided)
        semantically relevant memory entries, then renders it into a
        string wrapped in ``<cross_session_memory>`` tags.

        Parameters
        ----------
        user_prompt:
            Optional current user message for semantic matching.

        Returns
        -------
        str
            Rendered context string ready for insertion into a system
            prompt.  Returns an empty string if no context is available
            or an error occurs.
        """
        bundle = self._build_context_safe(user_prompt)
        return self._render_context_safe(bundle)

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics from the SQLite storage.

        Returns
        -------
        dict
            Keys include ``sessions``, ``events``, ``observations``,
            ``summaries`` with integer counts.  Returns an empty dict
            on error.
        """
        try:
            stats: Dict[str, Any] = self.sqlite_storage.get_stats(
                tenant_id=self.tenant_id,
                project=self.project,
            )
            stats["project"] = self.project
            stats["tenant_id"] = self.tenant_id
            return stats
        except Exception:
            logger.exception("Failed to get stats")
            return {}

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close storage connections and release resources.

        Safe to call multiple times.
        """
        try:
            self.sqlite_storage.close()
        except Exception:
            logger.exception("Error closing SQLite storage")
        try:
            if hasattr(self.vector_store, "close"):
                self.vector_store.close()  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Error closing vector store")
        logger.debug("CrossMemOrchestrator closed")

    async def __aenter__(self) -> CrossMemOrchestrator:
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the async context manager, closing resources."""
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context_safe(
        self,
        user_prompt: Optional[str] = None,
    ) -> Optional[ContextBundle]:
        """Build a context bundle, returning ``None`` on failure."""
        try:
            return self.context_injector.build_context(
                tenant_id=self.tenant_id,
                project=self.project,
                user_prompt=user_prompt,
            )
        except Exception:
            logger.exception("Failed to build context bundle")
            return None

    def _render_context_safe(
        self,
        bundle: Optional[ContextBundle],
    ) -> str:
        """Render a context bundle into a system-prompt string.

        Returns an empty string if *bundle* is ``None`` or rendering fails.
        """
        if bundle is None:
            return ""
        try:
            return ContextRenderer.render_for_system_prompt(bundle)
        except Exception:
            logger.exception("Failed to render context bundle")
            return ""

    def __repr__(self) -> str:
        return (
            f"CrossMemOrchestrator(project={self.project!r}, "
            f"tenant_id={self.tenant_id!r})"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_orchestrator(
    project: str,
    **kwargs: Any,
) -> CrossMemOrchestrator:
    """Convenience factory for creating a :class:`CrossMemOrchestrator`.

    Parameters
    ----------
    project:
        Project name or path used to scope memories.
    **kwargs:
        Forwarded to :class:`CrossMemOrchestrator` (``tenant_id``,
        ``db_path``, ``iris_table``, ``use_iris_sql``, ``max_context_tokens``,
        ``simplemem``).  ``lancedb_path`` is accepted but ignored.

    Returns
    -------
    CrossMemOrchestrator
        A fully-initialised orchestrator instance.

    Example
    -------
    ::

        orch = create_orchestrator("my-project", tenant_id="org-123")
    """
    return CrossMemOrchestrator(project=project, **kwargs)
