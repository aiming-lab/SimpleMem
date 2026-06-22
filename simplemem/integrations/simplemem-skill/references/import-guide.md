# Batch Import Guide

## JSONL Format

Each line is a JSON object with the following fields:

- `speaker`: (required) String identifying who spoke
- `content`: (required) String containing what was said
- `timestamp`: (optional) ISO 8601 datetime string (defaults to current time if omitted)

## Example JSONL File

Create a file named `dialogues.jsonl`:

```jsonl
{"speaker": "Alice", "content": "Let's meet tomorrow at 2pm", "timestamp": "2026-01-16T14:00:00Z"}
{"speaker": "Bob", "content": "Sounds good, I'll be there"}
{"speaker": "Alice", "content": "Don't forget to bring the documents"}
```

## Import Command

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py add_batch --file dialogues.jsonl
```

With custom table:

```bash
python ~/.claude/skills/simplemem-skill/scripts/cli_persistent_memory.py --table-name project_conversations add_batch --file dialogues.jsonl
```

## Notes

- If `timestamp` is omitted, the current system time is used
- Lines with parsing errors are skipped and reported
- All successfully parsed entries are added in a single batch operation
- The file path can be absolute or relative to the current working directory