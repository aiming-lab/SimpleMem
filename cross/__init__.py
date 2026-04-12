# pyright: reportMissingImports=false
"""
SimpleMem-Cross: Cross-conversation memory extension for SimpleMem.

This package adds persistent cross-session memory capabilities to SimpleMem,
enabling agents to recall context from previous conversations.

Quick start::

    from cross.orchestrator import CrossMemOrchestrator, create_orchestrator

    orchestrator = create_orchestrator(project="my-project")
    result = await orchestrator.start_session(
        content_session_id="session-1",
        user_prompt="Continue working on the API",
    )
    print(result["context"])  # injected cross-session memory
"""

from __future__ import annotations

# -- Types (always importable) ------------------------------------------------
from cross.types import (
    ContextBundle,
    ConsolidationRun,
    CrossMemoryEntry,
    CrossObservation,
    EventKind,
    FinalizationReport,
    MemoryLink,
    ObservationType,
    RedactionLevel,
    SessionEvent,
    SessionRecord,
    SessionStatus,
    SessionSummary,
)

# -- Storage -------------------------------------------------------------------
from cross.storage_sqlite import SQLiteStorage
from cross.storage_iris import CrossSessionVectorStore

# -- Core logic ----------------------------------------------------------------
from cross.collectors import EventCollector, ObservationExtractor, RedactionFilter
from cross.session_manager import SessionManager
from cross.context_injector import ContextInjector, ContextRenderer
from cross.hooks import DefaultHooks, HookResult, NoOpHooks, SessionHooks

# -- Orchestrator (main entry point) ------------------------------------------
from cross.orchestrator import CrossMemOrchestrator, create_orchestrator

# -- API -----------------------------------------------------------------------
from cross.api_http import create_app, create_cross_router
from cross.api_mcp import MCPToolRegistry, create_mcp_tools

# -- Maintenance ---------------------------------------------------------------
from cross.consolidation import (
    ConsolidationPolicy,
    ConsolidationResult,
    ConsolidationWorker,
    run_consolidation,
)

__all__ = [
    # Types
    "ContextBundle",
    "ConsolidationRun",
    "CrossMemoryEntry",
    "CrossObservation",
    "EventKind",
    "FinalizationReport",
    "MemoryLink",
    "ObservationType",
    "RedactionLevel",
    "SessionEvent",
    "SessionRecord",
    "SessionStatus",
    "SessionSummary",
    # Storage
    "SQLiteStorage",
    "CrossSessionVectorStore",
    # Core logic
    "EventCollector",
    "ObservationExtractor",
    "RedactionFilter",
    "SessionManager",
    "ContextInjector",
    "ContextRenderer",
    "DefaultHooks",
    "HookResult",
    "NoOpHooks",
    "SessionHooks",
    # Orchestrator
    "CrossMemOrchestrator",
    "create_orchestrator",
    # API
    "create_app",
    "create_cross_router",
    "MCPToolRegistry",
    "create_mcp_tools",
    # Maintenance
    "ConsolidationPolicy",
    "ConsolidationResult",
    "ConsolidationWorker",
    "run_consolidation",
]
