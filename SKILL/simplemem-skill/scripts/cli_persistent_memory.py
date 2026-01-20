#!/usr/bin/env python3
"""
CLI for one-shot atomic operations on SimpleMemSystem.

Usage examples:
    python scripts/cli_persistent_memory.py add --speaker Alice --content "Meet Bob at 2pm"
    python scripts/cli_persistent_memory.py import --file dialogues.jsonl
    python scripts/cli_persistent_memory.py query --question "When will Alice meet Bob?" --enable-reflection
    python scripts/cli_persistent_memory.py retrieve --query "Alice"
    python scripts/cli_persistent_memory.py stats
    python scripts/cli_persistent_memory.py clear --yes

The script performs a single operation per invocation and exits. Optional global arg: `--table-name`.
"""
import argparse
import json
import os
import sys
from typing import List
from datetime import datetime

# Ensure the skill's `src/` folder is on sys.path so `from main import SimpleMemSystem` works
script_dir = os.path.dirname(os.path.abspath(__file__))
skill_root = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(skill_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from main import SimpleMemSystem
from models.memory_entry import Dialogue

# Default DB path is fixed relative to this script (no CLI override)
DEFAULT_DB = os.path.abspath(os.path.join(script_dir, "..", "data", "lancedb"))


def build_system(db_path: str = DEFAULT_DB, table_name: str = None, clear_db: bool = False) -> SimpleMemSystem:
    return SimpleMemSystem(
        db_path=db_path,
        table_name=table_name,
        clear_db=clear_db
    )


# -----------------------------
# Add / Import functions
# -----------------------------


def cmd_add(args):
    system = build_system(table_name=args.table_name, clear_db=False)
    timestamp = args.timestamp or datetime.utcnow().isoformat(timespec="seconds")
    system.add_dialogue(args.speaker, args.content, timestamp=timestamp)
    # Ensure remaining buffer processed immediately
    system.finalize()
    print("OK: dialogue added and finalized")


def cmd_import(args):
    system = build_system(table_name=args.table_name, clear_db=False)
    dialogues: List[Dialogue] = []
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Expect JSON lines with keys: speaker, content, timestamp (optional)
                obj = json.loads(line)
                dialogue_id = obj.get("dialogue_id") or None
                # If no dialogue_id provided, use incremental placeholder (-1)
                dialogue = Dialogue(
                    dialogue_id=dialogue_id or -1,
                    speaker=obj.get("speaker", "unknown"),
                    content=obj.get("content", ""),
                    timestamp=obj.get("timestamp")
                )
                dialogues.append(dialogue)
    except Exception as e:
        print(f"Failed to read file: {e}")
        sys.exit(2)

    if not dialogues:
        print("No dialogues found in file")
        return

    # Use batch add
    system.add_dialogues(dialogues)
    system.finalize()
    print(f"OK: imported {len(dialogues)} dialogues and finalized")


# -----------------------------
# Query / Retrieval functions
# -----------------------------


def cmd_query(args):
    system = build_system(table_name=args.table_name, clear_db=False)
    # Use retriever directly to control reflection
    enable_reflection = getattr(args, "enable_reflection", False)
    contexts = system.hybrid_retriever.retrieve(args.question, enable_reflection=enable_reflection)
    answer = system.answer_generator.generate_answer(args.question, contexts)
    # Plain-text output for easier consumption by LLMs / humans
    print("--- SimpleMem Answer ---")
    print(f"Question: {args.question}\n")
    print("Answer:")
    print(answer)
    print()
    print(f"Contexts used: {len(contexts)}")
    print("--- End ---")


def cmd_retrieve(args):
    system = build_system(table_name=args.table_name, clear_db=False)
    entries = system.hybrid_retriever.retrieve(args.query, enable_reflection=False)
    top_k = getattr(args, "top_k", 10)
    selected = entries[:top_k]
    print(f"--- Retrieval Results for: {args.query} (top {len(selected)}) ---")
    for i, e in enumerate(selected, 1):
        print(f"\n[{i}] {e.lossless_restatement}")
        if e.timestamp:
            print(f"Time: {e.timestamp}")
        if e.location:
            print(f"Location: {e.location}")
        if e.persons:
            print(f"Persons: {', '.join(e.persons)}")
        if e.entities:
            print(f"Entities: {', '.join(e.entities)}")
        if e.topic:
            print(f"Topic: {e.topic}")

    print(f"\nTotal matched entries: {len(entries)}")
    print("--- End ---")


# -----------------------------
# Management / Utility functions
# -----------------------------


def cmd_stats(args):
    system = build_system(table_name=args.table_name, clear_db=False)
    memories = system.get_all_memories()
    print("--- Memory Store Stats ---")
    print(f"Entry count: {len(memories)}")
    print(f"Table name: {system.vector_store.table_name}")
    print("--- End ---")


def cmd_print(args):
    system = build_system(table_name=args.table_name, clear_db=False)
    system.print_memories()


def cmd_clear(args):
    if not args.yes:
        print("To confirm clearing the DB, pass --yes")
        sys.exit(1)
    system = build_system(table_name=args.table_name, clear_db=False)
    system.vector_store.clear()
    print("OK: database cleared")


def main():
    parser = argparse.ArgumentParser(description="SimpleMem one-shot CLI")
    parser.add_argument("--table-name", help="Memory table name", default=None)

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # -----------------------------
    # Add / Import commands
    # -----------------------------
    # `add`: add a single dialogue (speaker + content + optional timestamp)
    p_add = subparsers.add_parser(
        "add",
        help="Add a single dialogue and finalize",
        description="Add a single dialogue entry to the memory store and finalize processing."
    )
    p_add.add_argument("--speaker", required=True, help="Speaker name")
    p_add.add_argument("--content", required=True, help="Dialogue content")
    p_add.add_argument("--timestamp", required=False, help="ISO 8601 timestamp (optional)")
    p_add.set_defaults(func=cmd_add)

    # `import`: import dialogues from a JSON or JSONL file (list or one-json-per-line)
    p_import = subparsers.add_parser(
        "import",
        help="Import dialogues from JSON/JSONL file",
        description="Batch import dialogues from a JSON array or JSONL file. Each item should contain speaker, content, and optional timestamp."
    )
    p_import.add_argument("--file", required=True, help="JSON or JSONL file: list or lines with {speaker,content,timestamp}")
    p_import.set_defaults(func=cmd_import)

    # -----------------------------
    # Query / Retrieval commands
    # -----------------------------
    # `query`: natural-language question -> retrieval + answer generation
    p_query = subparsers.add_parser(
        "query",
        help="Ask a question (optionally enable reflection)",
        description="Ask a natural language question. The system will retrieve relevant memories and synthesize a concise answer."
    )
    p_query.add_argument("--question", required=True, help="Natural language question about stored memories")
    p_query.add_argument("--enable-reflection", action="store_true", help="Enable reflection for complex queries (may use more tokens)")
    p_query.set_defaults(func=cmd_query)

    # `retrieve`: return raw memory entries matching a query (no answer synthesis)
    p_retrieve = subparsers.add_parser(
        "retrieve",
        help="Retrieve raw memory entries",
        description="Retrieve raw memory entries (content + metadata) for a given query."
    )
    p_retrieve.add_argument("--query", required=True, help="Search query (semantic + keyword matching)")
    p_retrieve.add_argument("--top-k", type=int, default=10, help="Maximum number of entries to return")
    p_retrieve.set_defaults(func=cmd_retrieve)

    # -----------------------------
    # Management / Utility commands
    # -----------------------------
    # `stats`: basic statistics about the memory store
    p_stats = subparsers.add_parser("stats", help="Get basic memory store statistics", description="Return total entry count and table name.")
    p_stats.set_defaults(func=cmd_stats)

    # `print`: pretty-print all memories (debugging)
    p_print = subparsers.add_parser("print", help="Print all memories", description="Print all stored memory entries to stdout for debugging.")
    p_print.set_defaults(func=cmd_print)

    # `clear`: destructive clear of the database (requires --yes confirmation)
    p_clear = subparsers.add_parser("clear", help="Clear the database (destructive)", description="Clear all stored memories. This cannot be undone.")
    p_clear.add_argument("--yes", action="store_true", help="Confirm clear")
    p_clear.set_defaults(func=cmd_clear)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
