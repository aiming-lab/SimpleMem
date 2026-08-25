"""Entry point so the server can be started with ``python -m omni_mcp``."""

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
