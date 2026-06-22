---
name: simplemem-skill
description: Store and retrieve conversation memories across sessions. Use when asked to 'remember this', 'save conversation', 'add to memory', 'what did we discuss about...', 'query memories', or 'import chat history'. Also use proactively to preserve important dialogue context and decisions.
---

# SimpleMem Skill

Persistent conversational memory across sessions.

## Proactive Usage

Save memories when discovering valuable dialogue:
- Important decisions or commitments made in conversation
- Complex information that may be referenced later
- Context from long discussions worth preserving
- Solutions to problems that took effort to uncover

Check memories before:
- Answering questions about past conversations
- Resuming work from previous sessions
- Building on earlier discussion topics

## Quick Start

```bash
# Add a dialogue
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py add --speaker "Alice" --content "Meet Bob tomorrow at 2pm"

# Query memories
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py query --question "When should Alice meet Bob?"
```

## Operations

### Save

Add single dialogue:

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py add --speaker "User" --content "Your message here"
```

With timestamp (ISO 8601):

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py add --speaker "Alice" --content "Message" --timestamp "2026-01-17T14:00:00Z"
```

### Query

Semantic query with answer:

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py query --question "What did Alice say about meetings?"
```

With reflection for deeper analysis:

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py query --question "Your question" --enable-reflection
```

Raw retrieval:

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py retrieve --query "Alice meetings" --top-k 5
```

### Maintain

View statistics:

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py stats
```

Clear all memories:

```bash
# Use with caution - irreversible
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py clear --yes
```

## Batch Import

For importing conversation histories from JSONL files, see [references/import-guide.md](references/import-guide.md).

## Custom Table Names

Use different tables to organize conversation contexts:

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py --table-name my_custom_table add --speaker "User" --content "Message"
```

## Data Format

All dialogues are stored with:
- `speaker`: Who said it (string)
- `content`: What was said (string)
- `timestamp`: When it was said (ISO 8601 datetime, auto-generated if omitted)

## Advanced Usage

For detailed information:
- **OpenRouter setup and model selection**: [references/openrouter-guide.md](references/openrouter-guide.md)
- **JSONL import format and batch operations**: [references/import-guide.md](references/import-guide.md)
- **CLI command reference**: [references/cli-reference.md](references/cli-reference.md)
- **System architecture and configuration**: [references/architecture.md](references/architecture.md)

## Setup

**Install dependencies**:

```bash
cd ~/.claude/skills/simplemem-skill
pip install -r requirements.txt
```

**Configure OpenRouter API**:

```bash
cp src/config.py.example src/config.py
# Edit src/config.py and set your OPENROUTER_API_KEY
```

See [references/openrouter-guide.md](references/openrouter-guide.md) for API key setup and model customization.

**Data storage**: Memories persist in `data/lancedb/` (auto-created).
