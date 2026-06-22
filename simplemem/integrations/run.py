#!/usr/bin/env python3
"""
SimpleMem MCP Server - Runner Script

Starts the HTTP server which provides:
- Web UI for configuration (/)
- REST API for memory operations (/api/*)
- MCP over SSE for Claude Desktop (/mcp/*)
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="SimpleMem MCP Server - Multi-tenant Memory Service for LLM Agents"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  SimpleMem MCP Server")
    print("  Multi-tenant Memory Service for LLM Agents")
    print("=" * 60)
    print()
    print(f"  Web UI:     http://localhost:{args.port}/")
    print(f"  REST API:   http://localhost:{args.port}/api/")
    print(f"  MCP (SSE):  http://localhost:{args.port}/mcp/sse?token=<TOKEN>")
    print()
    print("-" * 60)

    import uvicorn
    uvicorn.run(
        "server.http_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
