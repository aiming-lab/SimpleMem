"""
Omni-SimpleMem MCP server (JSON-RPC 2.0 over stdio).

Implements the MCP stdio transport: newline-delimited JSON-RPC on stdin/stdout.

A note on stdout discipline: the stdio transport owns stdout exclusively, and
the Omni-Memory stack prints diagnostics (model fallbacks, progress) with plain
``print()``. Those writes would corrupt the JSON-RPC stream, so at startup we
swap ``sys.stdout`` for ``sys.stderr`` and keep a private handle to the real
stdout that only protocol frames are written to.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from typing import Any, Dict, Optional

from .namespaces import NamespaceError, NamespaceManager
from .tools import ToolExecutor, format_result, humanize_error, tool_definitions

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "omni-simplemem"
SERVER_VERSION = "1.0.0"

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

logger = logging.getLogger("omni_mcp")

INSTRUCTIONS = """Omni-SimpleMem is a multimodal long-term memory for agents.

STORE
  omni_add_text      remember text
  omni_add_image     remember an image (local path, http(s), Google Drive, s3://, gs://)
  omni_add_audio     remember audio (transcribed)
  omni_add_video     remember a video (significant frames only)
  omni_add_document  remember a .txt/.md/.json/.csv/.yaml/.pdf/.docx file

RECALL
  omni_query   retrieve relevant memory summaries
  omni_answer  synthesized answer over memory (needs an LLM API key)

MANAGE
  omni_stats, omni_list_events, omni_consolidate
  omni_list_namespaces, omni_delete_namespace

ISOLATION
  Every tool takes an optional `namespace`. Each namespace is a separate memory
  cluster with its own storage, so different agents cannot read each other's
  memories. Omit it to use the 'default' namespace.
"""


class OmniMCPServer:
    """MCP protocol server exposing Omni-Memory over stdio."""

    def __init__(self, base_dir: str, max_open_namespaces: int = 8):
        self.manager = NamespaceManager(base_dir, max_open=max_open_namespaces)
        self.executor = ToolExecutor(self.manager)
        self.initialized = False
        self._stdout = sys.stdout

    # -- protocol plumbing -------------------------------------------------

    def _write(self, payload: Dict[str, Any]) -> None:
        try:
            self._stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._stdout.flush()
        except (BrokenPipeError, ValueError):
            raise SystemExit(0)

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _result(request_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    # -- request handling --------------------------------------------------

    def handle_request(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle one JSON-RPC message. Returns None for notifications."""
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        is_notification = "id" not in message

        if not method:
            return None if is_notification else self._error(
                request_id, INVALID_REQUEST, "Missing 'method'."
            )

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method in ("notifications/initialized", "initialized"):
                self.initialized = True
                return None
            elif method in ("notifications/cancelled", "notifications/progress"):
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": tool_definitions()}
            elif method == "tools/call":
                result = self._handle_tools_call(params)
            elif method == "resources/list":
                result = {"resources": self._resources()}
            elif method == "resources/read":
                result = self._handle_resources_read(params)
            elif method == "prompts/list":
                result = {"prompts": []}
            elif method == "shutdown":
                result = {}
            else:
                if is_notification:
                    return None
                return self._error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except NamespaceError as exc:
            if is_notification:
                return None
            return self._error(request_id, INVALID_PARAMS, str(exc))
        except Exception as exc:  # never let one bad call kill the server
            logger.error("Error handling %s: %s", method, exc)
            logger.debug(traceback.format_exc())
            if is_notification:
                return None
            return self._error(request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

        if is_notification:
            return None
        return self._result(request_id, result)

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.initialized = True
        # Echo the client's protocol version when we understand it, so clients
        # that negotiate a newer revision are not rejected outright.
        client_version = params.get("protocolVersion")
        version = client_version if isinstance(client_version, str) and client_version else MCP_PROTOCOL_VERSION
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": INSTRUCTIONS,
        }

    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not name:
            raise ValueError("tools/call requires a tool 'name'.")

        try:
            payload = self.executor.call(name, arguments)
        except Exception as exc:
            # Tool-level failures are reported as tool results with isError,
            # which is what MCP clients surface to the model.
            logger.debug(traceback.format_exc())
            return {
                "content": [{"type": "text", "text": humanize_error(exc)}],
                "isError": True,
            }

        return {"content": [{"type": "text", "text": format_result(payload)}], "isError": False}

    def _resources(self):
        return [
            {
                "uri": "omni://namespaces",
                "name": "Memory namespaces",
                "description": "All isolated memory clusters on this server.",
                "mimeType": "application/json",
            },
            {
                "uri": "omni://stats",
                "name": "Memory statistics",
                "description": "Statistics for the default namespace.",
                "mimeType": "application/json",
            },
        ]

    def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("uri", "")
        if uri == "omni://namespaces":
            payload = self.executor.call("omni_list_namespaces", {})
        elif uri == "omni://stats":
            payload = self.executor.call("omni_stats", {})
        else:
            raise ValueError(f"Unknown resource: {uri}")
        return {
            "contents": [
                {"uri": uri, "mimeType": "application/json", "text": format_result(payload)}
            ]
        }

    # -- main loop ---------------------------------------------------------

    def serve_forever(self, stdin=None) -> None:
        stream = stdin or sys.stdin
        logger.info("Omni-SimpleMem MCP server ready (base_dir=%s)", self.manager.base_dir)

        try:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._write(self._error(None, PARSE_ERROR, f"Parse error: {exc}"))
                    continue

                if isinstance(message, list):
                    # Batch request: respond with the non-notification results.
                    responses = [r for r in (self.handle_request(m) for m in message) if r]
                    for response in responses:
                        self._write(response)
                    continue

                if not isinstance(message, dict):
                    self._write(self._error(None, INVALID_REQUEST, "Request must be an object."))
                    continue

                response = self.handle_request(message)
                if response is not None:
                    self._write(response)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        try:
            self.manager.close_all()
        except Exception:
            pass


def _configure_stdio() -> None:
    """Keep stdout exclusively for JSON-RPC frames; send everything else to stderr."""
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, os.getenv("OMNI_MCP_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="[omni-mcp] %(levelname)s %(message)s",
    )
    # Any library print() now lands on stderr instead of corrupting the protocol.
    sys.stdout = sys.stderr


def default_base_dir() -> str:
    return os.path.abspath(
        os.path.expanduser(
            os.getenv("OMNI_MCP_DATA_DIR", "~/.omni_simplemem/mcp")
        )
    )


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="omni-simplemem-mcp",
        description="Omni-SimpleMem MCP server (multimodal memory over stdio).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Base directory holding per-namespace memory clusters "
             "(default: $OMNI_MCP_DATA_DIR or ~/.omni_simplemem/mcp).",
    )
    parser.add_argument(
        "--max-open-namespaces",
        type=int,
        default=int(os.getenv("OMNI_MCP_MAX_OPEN_NAMESPACES", "8")),
        help="How many namespace orchestrators to keep loaded at once (default 8).",
    )
    args = parser.parse_args(argv)

    real_stdout = sys.stdout
    _configure_stdio()

    server = OmniMCPServer(
        base_dir=args.data_dir or default_base_dir(),
        max_open_namespaces=args.max_open_namespaces,
    )
    server._stdout = real_stdout
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
