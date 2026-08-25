"""
Isolated memory namespaces for the Omni-SimpleMem MCP server.

Each namespace is a fully independent memory cluster: its own data directory,
its own cold storage, MAU store, vector index and event store. Two agents using
different namespaces cannot see or affect each other's memories.

Orchestrators are created lazily on first use (loading one is expensive) and
cached, with an LRU bound so a long-running server with many namespaces does
not grow without limit.
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_NAMESPACE = "default"

# A namespace becomes a directory name, so it must not be able to escape the
# base directory or collide with path syntax.
_VALID_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class NamespaceError(Exception):
    """Raised for invalid namespace names or namespace management failures."""


def validate_namespace(name: Optional[str]) -> str:
    """Validate and normalize a namespace name.

    Rejects anything that could traverse outside the base directory
    (``..``, ``/``, ``\\``, absolute paths, NUL bytes, etc.).
    """
    if name is None or str(name).strip() == "":
        return DEFAULT_NAMESPACE

    candidate = str(name).strip()

    if not _VALID_NAMESPACE.match(candidate):
        raise NamespaceError(
            f"Invalid namespace {candidate!r}. Namespaces must be 1-64 characters, "
            "start with a letter or digit, and contain only letters, digits, "
            "'.', '_' or '-'."
        )
    # Defense in depth: reject dot-only names that normalize to a parent dir.
    if candidate in (".", "..") or set(candidate) == {"."}:
        raise NamespaceError(f"Invalid namespace {candidate!r}.")
    return candidate


class NamespaceManager:
    """Owns the lifecycle of one Omni-Memory orchestrator per namespace."""

    def __init__(self, base_dir: str, max_open: int = 8):
        self.base_dir = os.path.abspath(os.path.expanduser(base_dir))
        self.max_open = max(1, int(max_open))
        self._open: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.RLock()
        os.makedirs(self.base_dir, exist_ok=True)

    # -- paths -------------------------------------------------------------

    def data_dir_for(self, namespace: str) -> str:
        namespace = validate_namespace(namespace)
        path = os.path.join(self.base_dir, namespace)
        # Final guard: the resolved path must stay inside base_dir.
        resolved = os.path.realpath(path)
        base = os.path.realpath(self.base_dir)
        if resolved != base and not resolved.startswith(base + os.sep):
            raise NamespaceError(f"Namespace path escapes the base directory: {namespace!r}")
        return path

    # -- orchestrators -----------------------------------------------------

    def get(self, namespace: Optional[str] = None):
        """Return (creating if needed) the orchestrator for a namespace."""
        namespace = validate_namespace(namespace)

        with self._lock:
            if namespace in self._open:
                self._open.move_to_end(namespace)
                return self._open[namespace]

            data_dir = self.data_dir_for(namespace)
            os.makedirs(data_dir, exist_ok=True)

            # Imported lazily: pulls in the heavy Omni-Memory stack, and we only
            # want that cost when a tool actually touches memory.
            from omni_memory import OmniMemoryOrchestrator

            orchestrator = OmniMemoryOrchestrator(data_dir=data_dir)
            self._open[namespace] = orchestrator
            self._evict_if_needed()
            return orchestrator

    def _evict_if_needed(self) -> None:
        while len(self._open) > self.max_open:
            name, orchestrator = self._open.popitem(last=False)
            self._safe_close(orchestrator, name)

    @staticmethod
    def _safe_close(orchestrator: Any, name: str) -> None:
        for method in ("save", "close"):
            try:
                getattr(orchestrator, method)()
            except Exception:
                # Never let teardown of one namespace break the server.
                pass

    # -- introspection -----------------------------------------------------

    def list_namespaces(self) -> List[Dict[str, Any]]:
        """List namespaces that exist on disk, with basic metadata."""
        results: List[Dict[str, Any]] = []
        base = Path(self.base_dir)
        if not base.is_dir():
            return results

        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            try:
                validate_namespace(entry.name)
            except NamespaceError:
                continue  # ignore stray directories
            results.append(
                {
                    "namespace": entry.name,
                    "data_dir": str(entry),
                    "loaded": entry.name in self._open,
                    "size_bytes": _dir_size(entry),
                }
            )
        return results

    def is_loaded(self, namespace: str) -> bool:
        return validate_namespace(namespace) in self._open

    # -- lifecycle ---------------------------------------------------------

    def save(self, namespace: Optional[str] = None) -> None:
        with self._lock:
            if namespace is not None:
                name = validate_namespace(namespace)
                orchestrator = self._open.get(name)
                if orchestrator is not None:
                    try:
                        orchestrator.save()
                    except Exception:
                        pass
                return
            for orchestrator in self._open.values():
                try:
                    orchestrator.save()
                except Exception:
                    pass

    def unload(self, namespace: str) -> bool:
        """Close and drop a namespace's orchestrator (data stays on disk)."""
        name = validate_namespace(namespace)
        with self._lock:
            orchestrator = self._open.pop(name, None)
            if orchestrator is None:
                return False
            self._safe_close(orchestrator, name)
            return True

    def close_all(self) -> None:
        with self._lock:
            while self._open:
                name, orchestrator = self._open.popitem(last=False)
                self._safe_close(orchestrator, name)


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total
