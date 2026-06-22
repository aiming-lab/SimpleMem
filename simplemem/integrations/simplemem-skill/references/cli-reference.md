# CLI Command Reference

## Global Options

- `--table-name TABLE_NAME`: Use a custom table name instead of the default `memory_entries`

## Commands

### add

Add a single dialogue entry.

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py add --speaker SPEAKER --content CONTENT [--timestamp TIMESTAMP]
```

**Arguments:**
- `--speaker`: (required) Who said it
- `--content`: (required) What was said
- `--timestamp`: (optional) ISO 8601 datetime string (e.g., "2026-01-17T14:30:00Z")

**Example:**
```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py add --speaker "Alice" --content "Project deadline is Friday"
```

### import

Import multiple dialogues from a JSONL file.

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py import --file FILE_PATH
```

**Arguments:**
- `--file`: (required) Path to JSONL file containing dialogue entries

**Example:**
```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py import --file conversations.jsonl
```

### query

Query the memory system and generate an answer.

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py query --question QUESTION [--enable-reflection] [--top-k K]
```

**Arguments:**
- `--question`: (required) Natural language question to ask
- `--enable-reflection`: (optional) Enable multi-step reflection for deeper analysis
- `--top-k`: (optional) Number of relevant entries to retrieve (default: 5)

**Examples:**
```bash
# Simple query
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py query --question "What did Alice say about the deadline?"

# With reflection enabled
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py query --question "Summarize all project updates" --enable-reflection
```

### retrieve

Retrieve raw dialogue entries matching a query.

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py retrieve --query QUERY [--top-k K]
```

**Arguments:**
- `--query`: (required) Search query string
- `--top-k`: (optional) Number of results to return (default: 5)

**Example:**
```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py retrieve --query "deadline meeting" --top-k 10
```

### stats

Display memory store statistics (total entries, table name, database path).

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py stats
```

### clear

Delete all entries from the memory store.

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py clear --yes
```

**Arguments:**
- `--yes`: (required) Confirmation flag to prevent accidental deletion

**Warning:** This operation is irreversible. All memory entries will be permanently deleted.