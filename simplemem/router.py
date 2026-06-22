"""
SimpleMem — Unified Memory Router

Registry-based factory that routes between memory system backends.
Inspired by Mem0's provider factory and HuggingFace's AutoModel pattern.

Usage:
    from simplemem import SimpleMem, create

    # Auto mode (backend chosen by first call)
    mem = SimpleMem()
    mem.add_dialogue("Alice", "Let's meet at 2pm", "2025-11-15T14:30:00")
    mem.finalize()
    answer = mem.ask("When will they meet?")

    # Explicit text mode
    mem = create(mode="text", clear_db=True)

    # Multimodal mode (text, image, audio, video)
    mem = create(mode="omni", data_dir="./my_memory")
    mem.add_text("User loves hiking.", tags=["session_id:D1"])
    result = mem.query("What does the user enjoy?", top_k=5)
    mem.close()

    # List available modes
    simplemem.list_modes()
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Backend descriptor
# ---------------------------------------------------------------------------

class _Backend:
    """Metadata and lazy-loader for a single memory backend."""

    __slots__ = ("mode", "module_path", "class_name", "description",
                 "required_deps", "setup", "init")

    def __init__(
        self,
        mode: str,
        module_path: str,
        class_name: str,
        description: str = "",
        required_deps: Optional[List[str]] = None,
        setup: Optional[Callable[[], None]] = None,
        init: Optional[Callable[..., Any]] = None,
    ):
        self.mode = mode
        self.module_path = module_path
        self.class_name = class_name
        self.description = description
        self.required_deps = required_deps or []
        self.setup = setup          # called once before first import
        self.init = init            # custom constructor; if None, use cls(**kwargs)

    def check_deps(self) -> None:
        """Verify that all required packages are importable."""
        missing = []
        for dep in self.required_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        if missing:
            raise ImportError(
                f"Mode '{self.mode}' requires packages that are not installed: "
                f"{', '.join(missing)}.\n"
                f"Install them with:  pip install {' '.join(missing)}"
            )

    def load_class(self) -> type:
        """Lazily import and return the backend class."""
        if self.setup is not None:
            self.setup()
        self.check_deps()
        module = importlib.import_module(self.module_path)
        cls = getattr(module, self.class_name, None)
        if cls is None:
            raise ImportError(
                f"Cannot find class '{self.class_name}' in module '{self.module_path}'."
            )
        return cls


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: Dict[str, _Backend] = {}


def register(
    mode: str,
    module_path: str,
    class_name: str,
    description: str = "",
    required_deps: Optional[List[str]] = None,
    setup: Optional[Callable[[], None]] = None,
    init: Optional[Callable[..., Any]] = None,
    *,
    overwrite: bool = False,
) -> None:
    """
    Register a memory backend.

    Args:
        mode:          Short name used in ``create(mode=...)``.
        module_path:   Dotted Python import path for the module.
        class_name:    Class to instantiate from that module.
        description:   Human-readable one-liner shown in ``list_modes()``.
        required_deps: Package names checked before import (e.g. ``["torch"]``).
        setup:         Callable executed once before the first import
                       (e.g. to adjust ``sys.path``).
        init:          Custom constructor ``fn(cls, **kwargs) -> instance``.
                       If *None*, the default ``cls(**kwargs)`` is used.
        overwrite:     Allow replacing an existing registration.
    """
    if mode in _registry and not overwrite:
        raise ValueError(
            f"Mode '{mode}' is already registered. "
            f"Pass overwrite=True to replace it."
        )
    _registry[mode] = _Backend(
        mode=mode,
        module_path=module_path,
        class_name=class_name,
        description=description,
        required_deps=required_deps,
        setup=setup,
        init=init,
    )
    logger.debug("Registered backend: %s -> %s.%s", mode, module_path, class_name)


def list_modes() -> Dict[str, str]:
    """Return ``{mode: description}`` for every registered backend."""
    return {m: b.description for m, b in _registry.items()}


def is_available(mode: str) -> bool:
    """Check whether *mode* is registered and its dependencies are met."""
    if mode not in _registry:
        return False
    try:
        _registry[mode].check_deps()
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create(mode: str = "auto", **kwargs: Any) -> Any:
    """
    Create a SimpleMem memory system instance.

    Args:
        mode:    ``"auto"``  — auto-detect backend from usage (default),
                 ``"text"``  — single-modal SimpleMem,
                 ``"omni"``  — multimodal Omni-SimpleMem.
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A memory system instance (type depends on *mode*).

    Raises:
        ValueError:  Unknown mode.
        ImportError:  Missing dependencies for the requested mode.

    Examples::

        # Auto mode (recommended) — backend chosen by first call
        mem = create()
        mem.add_dialogue(...)   # → text backend
        # or
        mem.add_image(...)      # → omni backend

        # Explicit text mode
        mem = create(mode="text", clear_db=True)

        # Explicit omni mode
        mem = create(mode="omni", data_dir="./data")
    """
    if mode == "auto":
        return AutoMemory(**kwargs)

    if mode not in _registry:
        available = ", ".join(f"'{m}'" for m in _registry)
        raise ValueError(
            f"Unknown mode: '{mode}'. Available modes: {available}, 'auto'."
        )

    backend = _registry[mode]
    cls = backend.load_class()

    if backend.init is not None:
        return backend.init(cls, **kwargs)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Built-in backend: text (SimpleMem)
# ---------------------------------------------------------------------------

register(
    mode="text",
    module_path="simplemem.text.system",
    class_name="SimpleMemSystem",
    description="Single-modal text memory with semantic lossless compression",
)


# ---------------------------------------------------------------------------
# Built-in backend: omni (Omni-SimpleMem)
# ---------------------------------------------------------------------------

def _init_omni(cls: type, **kwargs: Any) -> Any:
    """Custom constructor: create default config if none is provided."""
    from simplemem.multimodal import OmniMemoryConfig
    config = kwargs.pop("config", None) or OmniMemoryConfig()
    return cls(config=config, **kwargs)


register(
    mode="omni",
    module_path="simplemem.multimodal.orchestrator",
    class_name="OmniMemoryOrchestrator",
    description="Multimodal memory — text, image, audio, video (Omni-SimpleMem)",
    init=_init_omni,
)


# ---------------------------------------------------------------------------
# Auto-routing memory (mode="auto")
# ---------------------------------------------------------------------------

class AutoMemory:
    """
    Auto-routing memory system.

    Lazily initializes the appropriate backend based on the first method called:

    - **Text-only methods** (``add_dialogue``, ``finalize``, ``ask``)
      → SimpleMem text backend.
    - **Multimodal methods** (``add_text``, ``add_image``, ``add_audio``,
      ``add_video``, ``query``)
      → Omni-SimpleMem backend.

    Once a backend is selected it cannot be changed for the lifetime of the
    instance.  If you need to switch, create a new instance with an explicit
    ``mode``.

    Examples::

        mem = create()                # mode="auto" is the default
        mem.add_dialogue(...)         # → text backend initialized
        mem.ask("When?")

        mem = create()
        mem.add_image("photo.jpg")   # → omni backend initialized
        mem.query("What did I see?")
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._backend: Any = None
        self._mode: Optional[str] = None

    # -- internal helpers --------------------------------------------------

    def _init_backend(self, mode: str) -> None:
        """Initialize the backend; called exactly once."""
        if self._backend is not None:
            return
        self._mode = mode
        self._backend = create(mode=mode, **self._kwargs)
        logger.info("AutoMemory: initialized '%s' backend", mode)

    def _require(self, target_mode: str, method: str) -> None:
        """Ensure the backend matches *target_mode*, or initialize it."""
        if self._backend is None:
            self._init_backend(target_mode)
            return
        if self._mode != target_mode:
            other = "omni" if target_mode == "text" else "text"
            raise RuntimeError(
                f"Cannot call {method}() — the system is running in "
                f"'{self._mode}' mode (auto-selected by a prior call).  "
                f"'{method}' requires '{target_mode}' mode.\n"
                f"Hint: create a new instance with "
                f"simplemem.create(mode='{target_mode}') to use this method."
            )

    # -- text-mode API -----------------------------------------------------

    def add_dialogue(
        self, speaker: str, content: str, timestamp: Optional[str] = None
    ) -> Any:
        """Add a dialogue turn.  *Selects text backend on first call.*"""
        self._require("text", "add_dialogue")
        return self._backend.add_dialogue(speaker, content, timestamp)

    def add_dialogues(self, dialogues: Any) -> Any:
        """Batch-add dialogues.  *Selects text backend on first call.*"""
        self._require("text", "add_dialogues")
        return self._backend.add_dialogues(dialogues)

    def finalize(self) -> Any:
        """Finalize text memory construction."""
        self._require("text", "finalize")
        return self._backend.finalize()

    def ask(self, question: str) -> str:
        """Ask a question (text mode)."""
        self._require("text", "ask")
        return self._backend.ask(question)

    # -- omni-mode API -----------------------------------------------------

    def add_text(self, text: str, **kwargs: Any) -> Any:
        """Add text content.  *Selects omni backend on first call.*"""
        self._require("omni", "add_text")
        return self._backend.add_text(text, **kwargs)

    def add_image(self, *args: Any, **kwargs: Any) -> Any:
        """Add an image.  *Selects omni backend on first call.*"""
        self._require("omni", "add_image")
        return self._backend.add_image(*args, **kwargs)

    def add_audio(self, *args: Any, **kwargs: Any) -> Any:
        """Add an audio clip.  *Selects omni backend on first call.*"""
        self._require("omni", "add_audio")
        return self._backend.add_audio(*args, **kwargs)

    def add_video(self, *args: Any, **kwargs: Any) -> Any:
        """Add a video.  *Selects omni backend on first call.*"""
        self._require("omni", "add_video")
        return self._backend.add_video(*args, **kwargs)

    def query(self, question: str, **kwargs: Any) -> Any:
        """Query memories (omni mode)."""
        self._require("omni", "query")
        return self._backend.query(question, **kwargs)

    # -- shared ------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by the backend."""
        if self._backend is not None and hasattr(self._backend, "close"):
            self._backend.close()

    def get_all_memories(self) -> Any:
        """Return all stored memory entries (text mode)."""
        self._require("text", "get_all_memories")
        return self._backend.get_all_memories()

    @property
    def mode(self) -> str:
        """Current mode: ``'text'``, ``'omni'``, or ``'auto'`` (pending)."""
        return self._mode or "auto"

    def __repr__(self) -> str:
        cls = type(self).__name__
        if self._backend is None:
            return f"<{cls} mode=auto (pending — call add_*() to initialize)>"
        return f"<{cls} mode={self._mode} backend={type(self._backend).__name__}>"
