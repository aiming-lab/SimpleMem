# SimpleMem Skill

A self-contained Claude skill for managing persistent conversational memory using vector-based retrieval.

## What is this?

This directory contains the **simplemem-skill** - a production-ready skill that enables Claude to maintain long-term conversation memory across sessions. The skill uses a vector database (LanceDB) to store, retrieve, and query dialogue histories.

## Quick Install

```bash
# Copy skill to Claude's skills directory
cp -r simplemem-skill ~/.claude/skills/

# Install dependencies
cd ~/.claude/skills/simplemem-skill
pip install -r requirements.txt

# Configure API key
cp src/config.py.example src/config.py
# Edit src/config.py and add your OPENROUTER_API_KEY
```

## What's Inside

```
simplemem-skill/
├── SKILL.md              # Main skill documentation (Claude reads this)
├── requirements.txt      # Python dependencies
├── scripts/              # CLI tools for memory management
├── src/                  # Core SimpleMem implementation
├── references/           # Detailed guides (loaded on-demand)
└── data/                 # LanceDB storage (auto-created)
```

## Features

- **Persistent Memory**: Store dialogue entries with speaker, content, and timestamp
- **Vector Retrieval**: Semantic search using OpenRouter embeddings
- **Batch Import**: Import conversation histories from JSONL files
- **Reflection Mode**: Multi-step reasoning for complex queries
- **Custom Tables**: Organize different conversation contexts separately

## Usage with Claude

Once installed, Claude automatically discovers and uses this skill when you:
- Ask to "remember this conversation"
- Request to "query past memories"
- Say "add to memory" or "import conversations"
- Ask about "conversation history"

## Architecture

SimpleMem uses a three-stage pipeline:
1. **Semantic Structured Compression** - Process and compress dialogues
2. **Structured Indexing** - Store in LanceDB with vector embeddings
3. **Adaptive Query-Aware Retrieval** - Hybrid semantic + BM25 search

## API Integration

The skill uses **OpenRouter** as a unified API gateway for both LLM operations and embeddings, eliminating the need for multiple API keys or local model installations.

Supported models (configurable):
- **LLM**: Any OpenRouter model (default: `openai/gpt-4.1-mini`)
- **Embeddings**: Any OpenRouter embedding model (default: `qwen/qwen3-embedding-8b`)

## Documentation

- **Main Guide**: `simplemem-skill/SKILL.md`
- **OpenRouter Setup**: `simplemem-skill/references/openrouter-guide.md`
- **Import Guide**: `simplemem-skill/references/import-guide.md`
- **CLI Reference**: `simplemem-skill/references/cli-reference.md`
- **Architecture Details**: `simplemem-skill/references/architecture.md`

## Requirements

- Python 3.10+
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

## License

See LICENSE file in the parent directory.
