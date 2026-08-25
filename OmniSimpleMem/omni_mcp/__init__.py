"""
Omni-SimpleMem MCP server.

Exposes the Omni-Memory multimodal memory pipeline (text, image, audio, video,
documents) to any MCP client over the stdio transport, with isolated per-agent
memory namespaces.

Run it with::

    python -m omni_mcp --data-dir ~/.omni_simplemem/mcp
"""

from .media import MediaError, resolve_media
from .namespaces import DEFAULT_NAMESPACE, NamespaceError, NamespaceManager
from .server import OmniMCPServer, main
from .tools import ToolExecutor, tool_definitions

__version__ = "1.0.0"

__all__ = [
    "OmniMCPServer",
    "NamespaceManager",
    "NamespaceError",
    "DEFAULT_NAMESPACE",
    "ToolExecutor",
    "tool_definitions",
    "resolve_media",
    "MediaError",
    "main",
]
