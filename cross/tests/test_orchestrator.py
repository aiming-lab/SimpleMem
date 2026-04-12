# pyright: reportMissingImports=false
"""Unit tests for CrossMemOrchestrator."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure cross.* imports resolve from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cross.orchestrator import CrossMemOrchestrator, create_orchestrator
from cross.types import ContextBundle, FinalizationReport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT = "test-project"
TENANT = "default"
CONTENT_SID = "content-sess-orch-001"
USER_PROMPT = "Fix the login bug"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(
    tmp_path: Path,
    *,
    mock_vector_cls: MagicMock,
    mock_injector_cls: MagicMock,
) -> CrossMemOrchestrator:
    """Create a CrossMemOrchestrator with real SQLite but mocked vector/injector."""
    # Configure mock vector store instance
    mock_vs_instance = mock_vector_cls.return_value
    mock_vs_instance.semantic_search.return_value = []
    mock_vs_instance.add_entries.return_value = None
    mock_vs_instance.get_all_entries.return_value = []

    # Configure mock injector instance
    mock_inj_instance = mock_injector_cls.return_value
    mock_inj_instance.build_context.return_value = ContextBundle(
        total_tokens_estimate=0,
    )

    return CrossMemOrchestrator(
        project=PROJECT,
        tenant_id=TENANT,
        db_path=str(tmp_path / "test.db"),
        iris_table="pytest_cross_entries",
    )


def _patch_externals():
    """Return a pair of patchers for CrossSessionVectorStore and ContextInjector."""
    p_vs = patch("cross.orchestrator.CrossSessionVectorStore")
    p_ci = patch("cross.orchestrator.ContextInjector")
    return p_vs, p_ci


# ---------------------------------------------------------------------------
# TestCrossMemOrchestrator
# ---------------------------------------------------------------------------


class TestCrossMemOrchestrator:
    """Tests covering the CrossMemOrchestrator public API."""

    # -- init --------------------------------------------------------------

    def test_init_creates_components(self, tmp_path: Path) -> None:
        """Orchestrator creates storage, session manager, context injector, and hooks."""
        p_vs, p_ci = _patch_externals()
        with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
            orch = _make_orchestrator(
                tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
            )

            assert orch.project == PROJECT
            assert orch.tenant_id == TENANT
            assert orch.sqlite_storage is not None
            assert orch.vector_store is not None
            assert orch.session_manager is not None
            assert orch.context_injector is not None
            assert orch.hooks is not None

            orch.close()

    # -- start_session -----------------------------------------------------

    def test_start_session(self, tmp_path: Path) -> None:
        """start_session returns a dict with memory_session_id and context keys."""

        async def _run() -> None:
            p_vs, p_ci = _patch_externals()
            with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
                orch = _make_orchestrator(
                    tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
                )
                try:
                    result = await orch.start_session(
                        CONTENT_SID, user_prompt=USER_PROMPT
                    )

                    assert isinstance(result, dict)
                    assert "memory_session_id" in result
                    assert "context" in result
                    assert "context_bundle" in result
                    assert isinstance(result["memory_session_id"], str)
                    assert len(result["memory_session_id"]) > 0
                finally:
                    orch.close()

        asyncio.run(_run())

    # -- record_message ----------------------------------------------------

    def test_record_message(self, tmp_path: Path) -> None:
        """record_message succeeds after starting a session."""

        async def _run() -> None:
            p_vs, p_ci = _patch_externals()
            with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
                orch = _make_orchestrator(
                    tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
                )
                try:
                    result = await orch.start_session(
                        CONTENT_SID, user_prompt=USER_PROMPT
                    )
                    mid = result["memory_session_id"]

                    # Should not raise
                    await orch.record_message(mid, "Looking at the auth module...")
                    await orch.record_message(
                        mid, "Found the issue in login handler.", role="assistant"
                    )
                finally:
                    orch.close()

        asyncio.run(_run())

    # -- record_tool_use ---------------------------------------------------

    def test_record_tool_use(self, tmp_path: Path) -> None:
        """record_tool_use succeeds after starting a session."""

        async def _run() -> None:
            p_vs, p_ci = _patch_externals()
            with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
                orch = _make_orchestrator(
                    tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
                )
                try:
                    result = await orch.start_session(
                        CONTENT_SID, user_prompt=USER_PROMPT
                    )
                    mid = result["memory_session_id"]

                    # Should not raise
                    await orch.record_tool_use(
                        mid,
                        tool_name="read_file",
                        tool_input="auth.py",
                        tool_output="<file contents>",
                    )
                finally:
                    orch.close()

        asyncio.run(_run())

    # -- stop_session ------------------------------------------------------

    def test_stop_session(self, tmp_path: Path) -> None:
        """stop_session returns a FinalizationReport."""

        async def _run() -> None:
            p_vs, p_ci = _patch_externals()
            with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
                orch = _make_orchestrator(
                    tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
                )
                try:
                    result = await orch.start_session(
                        CONTENT_SID, user_prompt=USER_PROMPT
                    )
                    mid = result["memory_session_id"]

                    report = await orch.stop_session(mid)

                    assert isinstance(report, FinalizationReport)
                    assert report.memory_session_id == mid
                    assert isinstance(report.observations_count, int)
                    assert isinstance(report.entries_stored, int)
                    assert isinstance(report.summary_generated, bool)
                    assert isinstance(report.consolidation_triggered, bool)
                finally:
                    orch.close()

        asyncio.run(_run())

    # -- end_session -------------------------------------------------------

    def test_end_session(self, tmp_path: Path) -> None:
        """end_session after stop_session completes without error."""

        async def _run() -> None:
            p_vs, p_ci = _patch_externals()
            with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
                orch = _make_orchestrator(
                    tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
                )
                try:
                    result = await orch.start_session(
                        CONTENT_SID, user_prompt=USER_PROMPT
                    )
                    mid = result["memory_session_id"]

                    await orch.stop_session(mid)
                    # Should not raise
                    await orch.end_session(mid)
                finally:
                    orch.close()

        asyncio.run(_run())

    # -- full_lifecycle ----------------------------------------------------

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Full lifecycle: start -> record messages -> record tool use -> stop -> end."""

        async def _run() -> None:
            p_vs, p_ci = _patch_externals()
            with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
                orch = _make_orchestrator(
                    tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
                )
                try:
                    # Start
                    result = await orch.start_session(
                        CONTENT_SID, user_prompt=USER_PROMPT
                    )
                    mid = result["memory_session_id"]
                    assert isinstance(result["context"], str)

                    # Record messages
                    await orch.record_message(mid, "Investigating login module...")
                    await orch.record_message(
                        mid, "Found a null-check issue.", role="assistant"
                    )

                    # Record tool use
                    await orch.record_tool_use(
                        mid, "read_file", "login.py", "<source code>"
                    )
                    await orch.record_tool_use(
                        mid, "write_file", "login.py", "<patched>"
                    )

                    # Stop (finalize)
                    report = await orch.stop_session(mid)
                    assert isinstance(report, FinalizationReport)
                    assert report.memory_session_id == mid

                    # End
                    await orch.end_session(mid)
                finally:
                    orch.close()

        asyncio.run(_run())

    # -- search ------------------------------------------------------------

    def test_search_returns_list(self, tmp_path: Path) -> None:
        """search returns a list (empty when store has no data)."""
        p_vs, p_ci = _patch_externals()
        with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
            orch = _make_orchestrator(
                tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
            )
            try:
                results = orch.search("login bug")

                assert isinstance(results, list)
                # Verify the mock vector store was called
                mock_vs_cls.return_value.semantic_search.assert_called_once_with(
                    query="login bug",
                    top_k=10,
                    tenant_id=TENANT,
                )
            finally:
                orch.close()

    # -- get_context_for_prompt --------------------------------------------

    def test_get_context_for_prompt(self, tmp_path: Path) -> None:
        """get_context_for_prompt returns a string."""
        p_vs, p_ci = _patch_externals()
        with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
            orch = _make_orchestrator(
                tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
            )
            try:
                ctx = orch.get_context_for_prompt(
                    user_prompt="Tell me about the auth system"
                )

                assert isinstance(ctx, str)
            finally:
                orch.close()

    # -- get_stats ---------------------------------------------------------

    def test_get_stats(self, tmp_path: Path) -> None:
        """get_stats returns a dict with project and tenant_id keys."""
        p_vs, p_ci = _patch_externals()
        with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
            orch = _make_orchestrator(
                tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
            )
            try:
                stats = orch.get_stats()

                assert isinstance(stats, dict)
                assert stats.get("project") == PROJECT
                assert stats.get("tenant_id") == TENANT
            finally:
                orch.close()

    # -- close -------------------------------------------------------------

    def test_close(self, tmp_path: Path) -> None:
        """close completes without error and is safe to call multiple times."""
        p_vs, p_ci = _patch_externals()
        with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
            orch = _make_orchestrator(
                tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
            )

            # Should not raise
            orch.close()
            # Safe to call again
            orch.close()

    # -- context manager ---------------------------------------------------

    def test_context_manager(self, tmp_path: Path) -> None:
        """async with orchestrator works without errors."""

        async def _run() -> None:
            p_vs, p_ci = _patch_externals()
            with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
                orch = _make_orchestrator(
                    tmp_path, mock_vector_cls=mock_vs_cls, mock_injector_cls=mock_ci_cls
                )
                async with orch as o:
                    assert o is orch
                    assert o.project == PROJECT

                    result = await o.start_session("ctx-mgr-sess", user_prompt="test")
                    assert "memory_session_id" in result

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test create_orchestrator factory
# ---------------------------------------------------------------------------


class TestCreateOrchestrator:
    """Tests for the create_orchestrator convenience factory."""

    def test_create_orchestrator(self, tmp_path: Path) -> None:
        """create_orchestrator returns a properly configured orchestrator."""
        p_vs, p_ci = _patch_externals()
        with p_vs as mock_vs_cls, p_ci as mock_ci_cls:
            mock_vs_cls.return_value = MagicMock()
            mock_ci_cls.return_value = MagicMock()

            orch = create_orchestrator(
                PROJECT,
                tenant_id=TENANT,
                db_path=str(tmp_path / "factory.db"),
                iris_table="pytest_cross_factory_entries",
                max_context_tokens=500,
            )

            assert isinstance(orch, CrossMemOrchestrator)
            assert orch.project == PROJECT
            assert orch.tenant_id == TENANT
            orch.close()
