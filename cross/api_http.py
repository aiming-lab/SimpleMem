# pyright: reportMissingImports=false
"""
FastAPI HTTP endpoints for SimpleMem-Cross.

Provides a REST API layer over the cross-session memory orchestrator,
exposing session lifecycle management, event recording, search, and
health-check endpoints.

Usage::

    from cross.api_http import create_app

    app = create_app(project="my-project")

Or mount the router into an existing FastAPI application::

    from cross.api_http import create_cross_router

    router = create_cross_router(orchestrator)
    app.include_router(router, prefix="/cross")
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class StartSessionRequest(BaseModel):
    """Request body for starting a new memory session."""

    tenant_id: str = Field(
        ..., min_length=1, description="Tenant identifier for multi-tenant isolation"
    )
    content_session_id: str = Field(
        ..., min_length=1, description="External (host-side) session identifier"
    )
    project: str = Field(
        ..., min_length=1, description="Project name this session belongs to"
    )
    user_prompt: Optional[str] = Field(
        None, description="Optional initial user prompt that started the session"
    )


class StartSessionResponse(BaseModel):
    """Response returned after successfully starting a memory session."""

    memory_session_id: str = Field(..., description="Unique cross-session memory id")
    context: str = Field(
        "", description="Injected context from prior sessions (may be empty)"
    )
    context_tokens: int = Field(
        0, description="Estimated token count of the injected context"
    )


class RecordMessageRequest(BaseModel):
    """Request body for recording a chat message event."""

    memory_session_id: str = Field(
        ..., min_length=1, description="Target memory session id"
    )
    content: str = Field(..., min_length=1, description="Message body")
    role: Optional[str] = Field(
        "user",
        pattern="^(user|assistant|system)$",
        description='Speaker role: "user", "assistant", or "system"',
    )


class RecordToolUseRequest(BaseModel):
    """Request body for recording a tool invocation event."""

    memory_session_id: str = Field(
        ..., min_length=1, description="Target memory session id"
    )
    tool_name: str = Field(..., min_length=1, description="Name of the invoked tool")
    tool_input: str = Field(..., description="Serialised input passed to the tool")
    tool_output: str = Field(..., description="Serialised output returned by the tool")


class StopSessionResponse(BaseModel):
    """Response returned after finalizing (stopping) a session."""

    memory_session_id: str
    observations_count: int = Field(0, description="Number of observations extracted")
    summary_generated: bool = Field(False, description="Whether a summary was produced")
    entries_stored: int = Field(
        0, description="Number of memory entries written to the vector store"
    )


class SearchRequest(BaseModel):
    """Request body for semantic search across stored memory."""

    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: Optional[int] = Field(10, ge=1, le=100, description="Max results to return")
    tenant_id: Optional[str] = Field(
        None, min_length=1, description="Restrict search to a specific tenant"
    )


class SearchEntry(BaseModel):
    """A single search result entry."""

    text: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response for a search query."""

    entries: List[SearchEntry] = Field(default_factory=list)
    count: int = 0


class StatsResponse(BaseModel):
    """Aggregate statistics for the cross-session memory system."""

    sessions: int = 0
    events: int = 0
    observations: int = 0
    summaries: int = 0


class EventIdResponse(BaseModel):
    """Lightweight acknowledgement containing only the event id."""

    event_id: int


class ErrorDetail(BaseModel):
    """Structured error response body."""

    detail: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    service: str = "simplemem-cross"
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Module-level state for uptime tracking
# ---------------------------------------------------------------------------

_startup_time: float = time.monotonic()


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_cross_router(orchestrator: Any) -> APIRouter:
    """Create a FastAPI ``APIRouter`` wired to the given *orchestrator*.

    The *orchestrator* is duck-typed and expected to expose at least the
    following methods (matching ``SessionManager``'s public API):

    * ``start_session(tenant_id, content_session_id, project, user_prompt=None)``
    * ``record_message(memory_session_id, content, role)``
    * ``record_tool_use(memory_session_id, tool_name, tool_input, tool_output)``
    * ``finalize_session(memory_session_id)``  (mapped to *stop*)
    * ``end_session(memory_session_id)``
    * ``search(query, top_k, tenant_id)``  (optional)
    * ``get_stats()``  (optional)

    Parameters
    ----------
    orchestrator:
        An object implementing the session-manager interface.

    Returns
    -------
    APIRouter
        A router ready to be mounted on a FastAPI app.
    """

    router = APIRouter(tags=["cross"])

    # ----- session lifecycle ------------------------------------------------

    @router.post(
        "/sessions/start",
        response_model=StartSessionResponse,
        summary="Start a new memory session",
    )
    async def start_session(req: StartSessionRequest) -> StartSessionResponse:
        """Create a new cross-session memory session and optionally inject
        context from prior sessions."""
        try:
            result = orchestrator.start_session(
                tenant_id=req.tenant_id,
                content_session_id=req.content_session_id,
                project=req.project,
                user_prompt=req.user_prompt,
            )

            # The orchestrator may return a SessionRecord or a dict —
            # extract the memory_session_id flexibly.
            memory_session_id: str
            if hasattr(result, "memory_session_id"):
                memory_session_id = result.memory_session_id
            elif isinstance(result, dict):
                memory_session_id = result.get("memory_session_id", "")
            else:
                memory_session_id = str(result)

            # Context injection (optional orchestrator capability)
            context = ""
            context_tokens = 0
            if hasattr(result, "context"):
                context = getattr(result, "context", "") or ""
            if hasattr(result, "context_tokens"):
                context_tokens = getattr(result, "context_tokens", 0) or 0

            logger.info("Started session %s via HTTP", memory_session_id)
            return StartSessionResponse(
                memory_session_id=memory_session_id,
                context=context,
                context_tokens=context_tokens,
            )
        except Exception as exc:
            logger.exception("Error starting session")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ----- event recording --------------------------------------------------

    @router.post(
        "/sessions/{memory_session_id}/message",
        response_model=EventIdResponse,
        summary="Record a chat message event",
    )
    async def record_message(
        memory_session_id: str,
        req: RecordMessageRequest,
    ) -> EventIdResponse:
        """Record a user/assistant/system message for the given session."""
        try:
            event_id = orchestrator.record_message(
                memory_session_id=memory_session_id,
                content=req.content,
                role=req.role or "user",
            )
            return EventIdResponse(event_id=event_id)
        except Exception as exc:
            logger.exception(
                "Error recording message for session %s", memory_session_id
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/sessions/{memory_session_id}/tool-use",
        response_model=EventIdResponse,
        summary="Record a tool invocation event",
    )
    async def record_tool_use(
        memory_session_id: str,
        req: RecordToolUseRequest,
    ) -> EventIdResponse:
        """Record a tool call (name, input, output) for the given session."""
        try:
            event_id = orchestrator.record_tool_use(
                memory_session_id=memory_session_id,
                tool_name=req.tool_name,
                tool_input=req.tool_input,
                tool_output=req.tool_output,
            )
            return EventIdResponse(event_id=event_id)
        except Exception as exc:
            logger.exception(
                "Error recording tool use for session %s", memory_session_id
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ----- session stop / end -----------------------------------------------

    @router.post(
        "/sessions/{memory_session_id}/stop",
        response_model=StopSessionResponse,
        summary="Finalize (stop) a memory session",
    )
    async def stop_session(memory_session_id: str) -> StopSessionResponse:
        """Finalize a session: flush events, extract observations, generate
        summary, and optionally run the SimpleMem pipeline."""
        try:
            report = orchestrator.finalize_session(memory_session_id)

            # Accept FinalizationReport dataclass or a plain dict
            if hasattr(report, "observations_count"):
                return StopSessionResponse(
                    memory_session_id=memory_session_id,
                    observations_count=getattr(report, "observations_count", 0),
                    summary_generated=getattr(report, "summary_generated", False),
                    entries_stored=getattr(report, "entries_stored", 0),
                )
            elif isinstance(report, dict):
                return StopSessionResponse(
                    memory_session_id=memory_session_id,
                    observations_count=report.get("observations_count", 0),
                    summary_generated=report.get("summary_generated", False),
                    entries_stored=report.get("entries_stored", 0),
                )
            else:
                return StopSessionResponse(
                    memory_session_id=memory_session_id,
                    observations_count=0,
                    summary_generated=False,
                    entries_stored=0,
                )
        except Exception as exc:
            logger.exception("Error stopping session %s", memory_session_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/sessions/{memory_session_id}/end",
        summary="Mark a session as completed",
    )
    async def end_session(memory_session_id: str) -> Dict[str, Any]:
        """Mark the session as completed (or failed) in the backing store.

        Unlike ``/stop`` this does **not** run finalization — it simply
        updates the session status.  Call ``/stop`` first if you need
        observations and summaries to be generated.
        """
        try:
            orchestrator.end_session(memory_session_id)
            logger.info("Ended session %s via HTTP", memory_session_id)
            return {"memory_session_id": memory_session_id, "status": "completed"}
        except Exception as exc:
            logger.exception("Error ending session %s", memory_session_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ----- search -----------------------------------------------------------

    @router.post(
        "/search",
        response_model=SearchResponse,
        summary="Semantic search across stored memory",
    )
    async def search(req: SearchRequest) -> SearchResponse:
        """Run a semantic search against the cross-session memory store."""
        try:
            search_fn = getattr(orchestrator, "search", None)
            if search_fn is None:
                raise HTTPException(
                    status_code=501,
                    detail="Search is not supported by the current orchestrator",
                )

            raw_results = search_fn(
                query=req.query,
                top_k=req.top_k or 10,
                tenant_id=req.tenant_id,
            )

            # Normalise results into SearchEntry list
            entries: List[SearchEntry] = []
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, dict):
                        entries.append(
                            SearchEntry(
                                text=item.get("text", ""),
                                score=float(item.get("score", 0.0)),
                                metadata={
                                    k: v
                                    for k, v in item.items()
                                    if k not in ("text", "score")
                                },
                            )
                        )
                    elif hasattr(item, "text"):
                        entries.append(
                            SearchEntry(
                                text=getattr(item, "text", ""),
                                score=float(getattr(item, "score", 0.0)),
                                metadata=getattr(item, "metadata", {}),
                            )
                        )

            return SearchResponse(entries=entries, count=len(entries))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error performing search")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ----- stats ------------------------------------------------------------

    @router.get(
        "/stats",
        response_model=StatsResponse,
        summary="Get aggregate memory statistics",
    )
    async def get_stats() -> StatsResponse:
        """Return aggregate counts of sessions, events, observations, and
        summaries managed by the orchestrator."""
        try:
            stats_fn = getattr(orchestrator, "get_stats", None)
            if stats_fn is None:
                raise HTTPException(
                    status_code=501,
                    detail="Stats are not supported by the current orchestrator",
                )

            raw = stats_fn()

            if isinstance(raw, dict):
                return StatsResponse(
                    sessions=raw.get("sessions", 0),
                    events=raw.get("events", 0),
                    observations=raw.get("observations", 0),
                    summaries=raw.get("summaries", 0),
                )
            elif hasattr(raw, "sessions"):
                return StatsResponse(
                    sessions=getattr(raw, "sessions", 0),
                    events=getattr(raw, "events", 0),
                    observations=getattr(raw, "observations", 0),
                    summaries=getattr(raw, "summaries", 0),
                )
            else:
                return StatsResponse()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error fetching stats")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ----- health -----------------------------------------------------------

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
    )
    async def health_check() -> HealthResponse:
        """Simple liveness probe.  Returns service name and uptime."""
        return HealthResponse(
            status="ok",
            service="simplemem-cross",
            uptime_seconds=round(time.monotonic() - _startup_time, 2),
        )

    return router


# ---------------------------------------------------------------------------
# Convenience: full FastAPI app factory
# ---------------------------------------------------------------------------


def create_app(
    project: str = "default",
    *,
    orchestrator: Any = None,
    cors_origins: Optional[List[str]] = None,
    **kwargs: Any,
) -> FastAPI:
    """Create a complete FastAPI application with the cross-session router.

    If *orchestrator* is ``None`` a minimal ``SessionManager`` will be
    constructed using default SQLite and LanceDB paths derived from
    *project*.

    Parameters
    ----------
    project:
        Project name used to derive default storage paths when no
        *orchestrator* is supplied.
    orchestrator:
        Pre-configured orchestrator instance.  When provided, *project*
        is only used for metadata.
    cors_origins:
        Allowed CORS origins.  Defaults to ``["*"]`` (allow all).
    **kwargs:
        Additional keyword arguments forwarded to ``FastAPI()``.

    Returns
    -------
    FastAPI
        A fully-configured application instance.
    """

    global _startup_time
    _startup_time = time.monotonic()

    app = FastAPI(
        title="SimpleMem-Cross API",
        description="REST API for cross-session memory management",
        version="0.1.0",
        **kwargs,
    )

    allowed_origins = cors_origins if cors_origins is not None else ["*"]
    allow_credentials_only_with_explicit_origins = (
        cors_origins is not None and "*" not in cors_origins
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials_only_with_explicit_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # -- Build default orchestrator if needed --------------------------------
    if orchestrator is None:
        try:
            from cross.storage_sqlite import SQLiteStorage
            from cross.storage_iris import CrossSessionVectorStore
            from cross.session_manager import SessionManager

            sqlite_storage = SQLiteStorage()
            vector_store = CrossSessionVectorStore()
            orchestrator = SessionManager(
                sqlite_storage=sqlite_storage,
                vector_store=vector_store,
            )
            logger.info("Created default SessionManager for project '%s'", project)
        except Exception:
            logger.exception(
                "Failed to create default orchestrator; "
                "endpoints will return 500 until a valid orchestrator is provided"
            )

            class _FailingOrchestrator:
                """Placeholder that raises on every call."""

                def __getattr__(self, name: str) -> Any:
                    def _fail(*_a: Any, **_kw: Any) -> None:
                        raise RuntimeError(
                            "No orchestrator available. "
                            "Pass one explicitly via create_app(orchestrator=...)"
                        )

                    return _fail

            orchestrator = _FailingOrchestrator()

    # -- Mount router --------------------------------------------------------
    router = create_cross_router(orchestrator)
    app.include_router(router, prefix="/cross")

    logger.info("SimpleMem-Cross API ready (project=%s)", project)
    return app
