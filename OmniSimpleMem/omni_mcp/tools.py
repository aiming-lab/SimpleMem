"""
Tool definitions and dispatch for the Omni-SimpleMem MCP server.

Every tool accepts an optional ``namespace`` argument that selects an isolated
memory cluster, so multiple agents can share one server without seeing each
other's memories.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .media import MediaError, ensure_kind, extract_document_text, resolve_media
from .namespaces import DEFAULT_NAMESPACE, NamespaceError, NamespaceManager

_NAMESPACE_PROP = {
    "namespace": {
        "type": "string",
        "description": (
            "Isolated memory cluster to use. Each namespace has its own storage and "
            f"is invisible to other namespaces. Defaults to '{DEFAULT_NAMESPACE}'."
        ),
    }
}

_TAGS_PROP = {
    "tags": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Optional tags stored with the memory, usable as retrieval filters.",
    }
}

_MEDIA_DESCRIPTION = (
    "Local file path, or a remote reference: http(s):// URL, a Google Drive share "
    "link, s3://bucket/key (S3 or MinIO, via S3_ENDPOINT_URL), or gs://bucket/object."
)


def tool_definitions() -> List[Dict[str, Any]]:
    """MCP tool schemas advertised via tools/list."""
    return [
        {
            "name": "omni_add_text",
            "description": (
                "Store text in multimodal memory. The text is compressed into an atomic "
                "memory unit with entity extraction and temporal anchoring."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text content to remember."},
                    "session_id": {"type": "string", "description": "Optional session identifier."},
                    "force": {
                        "type": "boolean",
                        "description": "Store even if the content looks redundant.",
                    },
                    **_TAGS_PROP,
                    **_NAMESPACE_PROP,
                },
                "required": ["text"],
            },
        },
        {
            "name": "omni_add_image",
            "description": (
                "Store an image in multimodal memory. The image is captioned and embedded; "
                "entropy triggering skips frames that are near-duplicates of recent ones."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": _MEDIA_DESCRIPTION},
                    "session_id": {"type": "string", "description": "Optional session identifier."},
                    "force": {
                        "type": "boolean",
                        "description": "Store even if visually similar to the previous image.",
                    },
                    **_TAGS_PROP,
                    **_NAMESPACE_PROP,
                },
                "required": ["image"],
            },
        },
        {
            "name": "omni_add_audio",
            "description": (
                "Store audio in multimodal memory. Speech is transcribed and stored as a "
                "searchable atomic memory unit; silence is skipped by VAD triggering."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "audio": {"type": "string", "description": _MEDIA_DESCRIPTION},
                    "session_id": {"type": "string", "description": "Optional session identifier."},
                    "force": {"type": "boolean", "description": "Store even if no speech detected."},
                    **_TAGS_PROP,
                    **_NAMESPACE_PROP,
                },
                "required": ["audio"],
            },
        },
        {
            "name": "omni_add_video",
            "description": (
                "Store a video in multimodal memory. Frames are sampled with entropy "
                "triggering so only visually significant frames become memories."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "video": {"type": "string", "description": _MEDIA_DESCRIPTION},
                    "session_id": {"type": "string", "description": "Optional session identifier."},
                    "max_frames": {
                        "type": "integer",
                        "description": "Maximum frames to sample (default 100).",
                    },
                    **_TAGS_PROP,
                    **_NAMESPACE_PROP,
                },
                "required": ["video"],
            },
        },
        {
            "name": "omni_add_document",
            "description": (
                "Extract text from a document (.txt, .md, .json, .csv, .yaml, .pdf, .docx) "
                "and store it in memory."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "document": {"type": "string", "description": _MEDIA_DESCRIPTION},
                    "session_id": {"type": "string", "description": "Optional session identifier."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Truncate extracted text to this many characters (default 20000).",
                    },
                    **_TAGS_PROP,
                    **_NAMESPACE_PROP,
                },
                "required": ["document"],
            },
        },
        {
            "name": "omni_query",
            "description": (
                "Retrieve relevant memories for a query and return their summaries, "
                "without generating an answer. Use omni_answer for a synthesized answer."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {"type": "integer", "description": "Number of results (default 10)."},
                    "auto_expand": {
                        "type": "boolean",
                        "description": "Expand highly relevant items to full detail.",
                    },
                    "tags_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only consider memories carrying these tags.",
                    },
                    **_NAMESPACE_PROP,
                },
                "required": ["query"],
            },
        },
        {
            "name": "omni_answer",
            "description": (
                "Answer a question from multimodal memory using retrieval-augmented "
                "generation. Requires an LLM API key to be configured."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to answer."},
                    "top_k": {"type": "integer", "description": "Memories to retrieve (default 10)."},
                    "include_sources": {
                        "type": "boolean",
                        "description": "Include source references (default true).",
                    },
                    "tags_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only consider memories carrying these tags.",
                    },
                    **_NAMESPACE_PROP,
                },
                "required": ["question"],
            },
        },
        {
            "name": "omni_stats",
            "description": "Report memory statistics for a namespace (unit counts by modality, storage size).",
            "inputSchema": {
                "type": "object",
                "properties": {**_NAMESPACE_PROP},
            },
        },
        {
            "name": "omni_list_events",
            "description": "List event nodes (grouped memories) in a namespace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum events to return (default 20)."},
                    **_NAMESPACE_PROP,
                },
            },
        },
        {
            "name": "omni_list_namespaces",
            "description": "List all isolated memory namespaces that exist on this server.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "omni_consolidate",
            "description": "Run memory consolidation for a namespace (importance-based retention).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "force": {"type": "boolean", "description": "Consolidate even if not due."},
                    **_NAMESPACE_PROP,
                },
            },
        },
        {
            "name": "omni_delete_namespace",
            "description": (
                "PERMANENTLY delete all memories in a namespace. Destructive and "
                "irreversible: requires confirm=true."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Namespace to delete."},
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to actually delete.",
                    },
                },
                "required": ["namespace", "confirm"],
            },
        },
    ]


def _as_plain_dict(value: Any) -> Dict[str, Any]:
    """Coerce a result object (dataclass, metrics object, dict) into a dict.

    Several Omni-Memory calls return rich objects; MCP clients need JSON, and a
    bare ``repr()`` string is not usable by the model.
    """
    if isinstance(value, dict):
        return value
    for attr in ("to_dict", "_asdict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                converted = method()
                if isinstance(converted, dict):
                    return converted
            except Exception:
                pass
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict) and data:
        return {k: v for k, v in data.items() if not k.startswith("_")}
    try:
        import dataclasses

        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
    except Exception:
        pass
    return {"result": str(value)}


class ToolExecutor:
    """Executes MCP tool calls against namespaced Omni-Memory orchestrators."""

    def __init__(self, manager: NamespaceManager):
        self.manager = manager

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _mau_info(result: Any) -> Dict[str, Any]:
        mau = getattr(result, "mau", None)
        info: Dict[str, Any] = {
            "success": bool(getattr(result, "success", False)),
            "skipped": bool(getattr(result, "skipped", False)),
        }
        if getattr(result, "error", None):
            info["error"] = result.error
        if mau is not None:
            info["mau_id"] = getattr(mau, "id", None)
            modality = getattr(mau, "modality_type", None)
            info["modality"] = getattr(modality, "value", str(modality)) if modality else None
            summary = getattr(mau, "summary", None)
            if summary:
                info["summary"] = str(summary)[:500]
        return info

    def _add_media(
        self,
        arguments: Dict[str, Any],
        key: str,
        method: str,
        extra: Optional[Dict[str, Any]] = None,
        expected_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        reference = arguments.get(key)
        if not reference:
            raise ValueError(f"'{key}' is required.")

        orchestrator = self.manager.get(arguments.get("namespace"))
        resolved = resolve_media(str(reference))
        try:
            if expected_kind:
                # Guard against e.g. a .txt being ingested as a "video", which
                # would otherwise store a meaningless memory unit.
                ensure_kind(resolved.path, expected_kind, str(reference))
            kwargs: Dict[str, Any] = {
                "session_id": arguments.get("session_id"),
                "tags": arguments.get("tags"),
            }
            if extra:
                kwargs.update(extra)
            result = getattr(orchestrator, method)(resolved.path, **kwargs)
        finally:
            resolved.cleanup()

        info = self._mau_info(result)
        info["source"] = resolved.source
        return info

    # -- dispatch ----------------------------------------------------------

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return a JSON-serializable payload."""
        arguments = arguments or {}
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        return handler(arguments)

    # -- ingestion tools ---------------------------------------------------

    def _tool_omni_add_text(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        text = arguments.get("text")
        if not text or not str(text).strip():
            raise ValueError("'text' is required and must be non-empty.")

        orchestrator = self.manager.get(arguments.get("namespace"))
        result = orchestrator.add_text(
            str(text),
            session_id=arguments.get("session_id"),
            tags=arguments.get("tags"),
            force=bool(arguments.get("force", False)),
        )
        return self._mau_info(result)

    def _tool_omni_add_image(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._add_media(
            arguments,
            "image",
            "add_image",
            {"force": bool(arguments.get("force", False))},
            expected_kind="image",
        )

    def _tool_omni_add_audio(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._add_media(
            arguments,
            "audio",
            "add_audio",
            {"force": bool(arguments.get("force", False))},
            expected_kind="audio",
        )

    def _tool_omni_add_video(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        max_frames = arguments.get("max_frames")
        extra = {"max_frames": int(max_frames)} if max_frames is not None else {}
        return self._add_media(arguments, "video", "add_video", extra, expected_kind="video")

    def _tool_omni_add_document(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        reference = arguments.get("document")
        if not reference:
            raise ValueError("'document' is required.")

        max_chars = int(arguments.get("max_chars") or 20000)
        orchestrator = self.manager.get(arguments.get("namespace"))
        resolved = resolve_media(str(reference))
        try:
            text = extract_document_text(resolved.path)
        finally:
            resolved.cleanup()

        if not text.strip():
            return {
                "success": False,
                "error": "No extractable text found in document.",
                "source": resolved.source,
            }

        truncated = len(text) > max_chars
        result = orchestrator.add_text(
            text[:max_chars],
            session_id=arguments.get("session_id"),
            tags=arguments.get("tags"),
        )
        info = self._mau_info(result)
        info["source"] = resolved.source
        info["characters_stored"] = min(len(text), max_chars)
        info["truncated"] = truncated
        return info

    # -- retrieval tools ---------------------------------------------------

    def _tool_omni_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = arguments.get("query")
        if not query or not str(query).strip():
            raise ValueError("'query' is required and must be non-empty.")

        orchestrator = self.manager.get(arguments.get("namespace"))
        result = orchestrator.query(
            str(query),
            top_k=int(arguments.get("top_k") or 10),
            auto_expand=bool(arguments.get("auto_expand", False)),
            tags_filter=arguments.get("tags_filter"),
        )
        payload = result.to_dict() if hasattr(result, "to_dict") else {"items": []}
        payload["count"] = len(payload.get("items", []))
        return payload

    def _tool_omni_answer(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        question = arguments.get("question")
        if not question or not str(question).strip():
            raise ValueError("'question' is required and must be non-empty.")

        orchestrator = self.manager.get(arguments.get("namespace"))
        answer = orchestrator.answer(
            str(question),
            top_k=int(arguments.get("top_k") or 10),
            include_sources=bool(arguments.get("include_sources", True)),
            tags_filter=arguments.get("tags_filter"),
        )
        return answer if isinstance(answer, dict) else {"answer": str(answer)}

    # -- management tools --------------------------------------------------

    def _tool_omni_stats(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        namespace = arguments.get("namespace") or DEFAULT_NAMESPACE
        orchestrator = self.manager.get(namespace)
        stats = orchestrator.get_stats()
        stats["namespace"] = namespace
        return stats

    def _tool_omni_list_events(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        limit = int(arguments.get("limit") or 20)
        orchestrator = self.manager.get(arguments.get("namespace"))
        events = orchestrator.get_events()
        serialized = []
        for event in list(events)[:limit]:
            if hasattr(event, "to_dict"):
                serialized.append(event.to_dict())
            elif isinstance(event, dict):
                serialized.append(event)
            else:
                serialized.append({"event": str(event)})
        return {"count": len(serialized), "events": serialized}

    def _tool_omni_list_namespaces(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        namespaces = self.manager.list_namespaces()
        return {"count": len(namespaces), "namespaces": namespaces}

    def _tool_omni_consolidate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        orchestrator = self.manager.get(arguments.get("namespace"))
        result = orchestrator.consolidate_memories(force=bool(arguments.get("force", False)))
        return _as_plain_dict(result)

    def _tool_omni_delete_namespace(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        import shutil

        namespace = arguments.get("namespace")
        if not namespace:
            raise ValueError("'namespace' is required.")
        if not arguments.get("confirm"):
            return {
                "deleted": False,
                "error": "Refusing to delete without confirm=true.",
                "namespace": namespace,
            }

        data_dir = self.manager.data_dir_for(namespace)
        self.manager.unload(namespace)
        if not os.path.isdir(data_dir):
            return {"deleted": False, "error": "Namespace does not exist.", "namespace": namespace}

        shutil.rmtree(data_dir)
        return {"deleted": True, "namespace": namespace}


def format_result(payload: Any) -> str:
    """Render a tool payload as the text content MCP clients display."""
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(payload)


# Tools whose failure is almost always a missing LLM/embedding credential.
_LLM_HINT = (
    "This operation needs an LLM API key. Set OPENAI_API_KEY (and optionally "
    "OPENAI_API_BASE for a compatible gateway) in the environment of the MCP "
    "server process, then retry. Text and document ingestion work without a key; "
    "image/audio/video captioning and query/answer do not."
)


def humanize_error(exc: BaseException) -> str:
    """Turn a raw exception into an actionable message for the calling agent."""
    message = str(exc)
    lowered = message.lower()

    if "api_key" in lowered or "api key" in lowered:
        return f"{type(exc).__name__}: {message}\n\n{_LLM_HINT}"

    return f"{type(exc).__name__}: {message}"
