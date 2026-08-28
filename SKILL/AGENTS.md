# SKILL/simplemem-skill/ — Claude Skills Integration

## OVERVIEW

Packaged SimpleMem for Claude Skills (claude.ai). Self-contained copy of SimpleMem core with local storage support. Not a development target — mirrors root package for distribution.

## STRUCTURE

```
simplemem-skill/
├── SKILL.md              # Claude Skills manifest and usage instructions
├── src/                  # Self-contained SimpleMem copy
│   ├── main.py           # SimpleMemSystem (mirrors root main.py)
│   ├── core/             # Pipeline primitives (mirrors root core/)
│   ├── models/           # Pydantic models
│   ├── utils/            # LLM client, embedding, OpenRouter helper
│   └── database/         # LanceDB vector store
├── scripts/
│   └── cli_persistent_memory.py  # CLI wrapper for skill invocation
├── references/           # Architecture + usage guides for skill authors
└── requirements.txt      # Minimal deps for skill runtime
```

## ANTI-PATTERNS

- This is a **distribution copy** — do NOT develop features here; develop in root then sync
- `src/` mirrors `main.py` + `core/` + `models/` + `utils/` — must stay consistent with root
- `references/` is documentation for skill consumers, not active code

## NOTES

- `src/utils/openrouter.py` is skill-specific (OpenRouter integration not in root)
- Cloud registration at `mcp.simplemem.cloud` required for token-based skill mode
- `SimpleMem.skill` in repo root is the compiled skill artifact
