# SimpleMem Package Usage Guide

This guide provides comprehensive documentation for using SimpleMem as a pip-installable Python package.

> **Public API.** The package exposes a small, stable surface:
> `SimpleMem`, `create`, `list_modes`, `optimize`, `Config`, and `load_config`
> (see `simplemem.__all__`). All examples below use these names.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Core API Reference](#core-api-reference)
- [Advanced Usage](#advanced-usage)
- [Data Models](#data-models)
- [Environment Variables](#environment-variables)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Basic Installation

```bash
pip install simplemem
```

### With GPU Support (CUDA)

```bash
pip install simplemem[gpu]
```

### With Development Tools

```bash
pip install simplemem[dev]
```

### With Benchmark Tools

```bash
pip install simplemem[benchmark]
```

### Full Installation (All Dependencies)

```bash
pip install simplemem[all]
```

### Requirements

- Python 3.10 or higher
- OpenAI-compatible API key (OpenAI, Qwen, Azure OpenAI, etc.)

---

## Quick Start

### Minimal Example

```python
from simplemem import SimpleMem

# Initialize the system. mode="auto" (default): the backend is chosen
# by the first method you call — add_dialogue() selects the text backend.
mem = SimpleMem()

# Add dialogues with timestamps
mem.add_dialogue("Alice", "Let's meet at Starbucks tomorrow at 2pm", "2025-01-15T14:30:00")
mem.add_dialogue("Bob", "Sure, I'll bring the report", "2025-01-15T14:31:00")

# Finalize memory encoding
mem.finalize()

# Query the memory
answer = mem.ask("When and where will Alice and Bob meet?")
print(answer)
# Output: "Alice and Bob will meet at Starbucks on January 16, 2025 at 2:00 PM"
```

Provide the API key via the `OPENAI_API_KEY` environment variable, a top-level
`config.py` (see [`config.py.example`](../config.py.example)), or by passing it
explicitly (see [Using Custom LLM Endpoints](#using-custom-llm-endpoints)).

### Using Environment Variables

```python
import os
from simplemem import SimpleMem

# Set API key via environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize (reads OPENAI_API_KEY from the environment)
mem = SimpleMem()
```

### Choosing a Backend Explicitly

`SimpleMem()` auto-selects a backend, but you can request one directly with
`create()`:

```python
from simplemem import create, list_modes

# Single-modal text memory
mem = create(mode="text", clear_db=True)

# Multimodal memory (text, image, audio, video)
mem = create(mode="omni", data_dir="./my_memory")

# Inspect the available backends
print(list_modes())
# {'text': 'Single-modal text memory ...', 'omni': 'Multimodal memory ...'}
```

`create(mode="text", ...)` returns the text memory system directly, exposing the
full text API (`add_dialogue`, `add_dialogues`, `finalize`, `ask`,
`get_all_memories`, `print_memories`).

---

## Configuration

SimpleMem resolves runtime settings in the following order (highest priority first):

1. **Constructor parameters** passed to `SimpleMem(...)` / `create(...)`
2. **A top-level `config.py`** on the Python path (copy from `config.py.example`)
3. **Environment variables** of the same name (e.g. `OPENAI_API_KEY`, `LLM_MODEL`)
4. **Built-in defaults**

### Using `config.py`

The simplest way to configure a local checkout is to copy the template and edit it:

```bash
cp config.py.example config.py
# Edit config.py with your API key, base URL, and model preferences
```

```python
# config.py
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = None            # or a custom OpenAI-compatible endpoint
LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
```

### Constructor Parameters (text backend)

When you use the text backend, the constructor accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | `$OPENAI_API_KEY` | OpenAI-compatible API key |
| `model` | str | `"gpt-4.1-mini"` | LLM model name |
| `base_url` | str | None | Custom API endpoint |
| `db_path` | str | `"./lancedb_data"` | LanceDB storage path |
| `table_name` | str | `"memory_entries"` | Memory table name |
| `clear_db` | bool | False | Clear existing database on start |
| `enable_thinking` | bool | None | Deep-thinking mode (Qwen-compatible models) |
| `use_streaming` | bool | None | Stream LLM responses |
| `enable_planning` | bool | None | Multi-query retrieval planning |
| `enable_reflection` | bool | None | Reflection-based retrieval |
| `max_reflection_rounds` | int | None | Max reflection iterations |
| `enable_parallel_processing` | bool | None | Parallel memory building |
| `max_parallel_workers` | int | None | Max workers for building |
| `enable_parallel_retrieval` | bool | None | Parallel query execution |
| `max_retrieval_workers` | int | None | Max workers for retrieval |

`None` means "fall back to the value from `config.py`, the environment, or the
built-in default."

```python
from simplemem import create

mem = create(
    mode="text",
    clear_db=True,
    model="gpt-4.1-mini",
    enable_parallel_processing=True,
    max_parallel_workers=8,
)
```

### Optimized Retrieval Config

`optimize()` produces a `Config` object that tunes retrieval hyperparameters for
your data. Save it once and reload it for deployment:

```python
import simplemem
from simplemem import SimpleMem, load_config

# mem is a finalized SimpleMem instance with memories already built
dev_questions = [
    ("When is the meeting?", "2pm tomorrow at Starbucks"),
    ("What should Bob prepare?", "the report"),
]
config = simplemem.optimize(mem, dev_questions, max_rounds=3)
config.save("my_config.json")

# Later, deploy with the optimized config
config = load_config("my_config.json")
mem = SimpleMem(config=config)
```

See [`EvolveMem/`](../EvolveMem/) for the full self-evolution loop and the
meaning of each `Config` field.

---

## Core API Reference

### `SimpleMem`

The main entry point. In the default `mode="auto"`, the first method you call
selects the backend:

| First call | Backend | API surface |
|:--|:--|:--|
| `add_dialogue()` | text | `add_dialogue`, `add_dialogues`, `finalize`, `ask`, `get_all_memories` |
| `add_text()` / `add_image()` / `add_audio()` / `add_video()` | omni | `add_text`, `add_image`, `add_audio`, `add_video`, `query`, `close` |

Once a backend is chosen it cannot be switched for that instance; create a new
instance (or use `create(mode=...)`) to change backends.

#### Text methods

##### `add_dialogue(speaker, content, timestamp=None)`

Add a single dialogue entry to the memory.

```python
mem.add_dialogue(
    speaker="Alice",
    content="I finished the quarterly report",
    timestamp="2025-01-15T10:00:00",  # ISO 8601 format
)
```

##### `add_dialogues(dialogues)`

Batch-add multiple dialogues.

```python
from simplemem.core.models.memory_entry import Dialogue

dialogues = [
    Dialogue(dialogue_id=1, speaker="Alice", content="Hello", timestamp="2025-01-15T10:00:00"),
    Dialogue(dialogue_id=2, speaker="Bob", content="Hi there!", timestamp="2025-01-15T10:01:00"),
]
mem.add_dialogues(dialogues)
```

##### `finalize()`

Process any remaining dialogues in the buffer. Always call this after adding all
dialogues and before querying.

```python
mem.finalize()
```

##### `ask(question)`

Query the memory system with a natural-language question.

```python
answer = mem.ask("What did Alice say about the report?")
```

##### `get_all_memories()`

Retrieve all stored memory entries (useful for debugging).

```python
for entry in mem.get_all_memories():
    print(entry.lossless_restatement)
```

#### Multimodal methods

When the omni backend is selected, `query()` returns a result whose `.items`
each expose a `"summary"`:

```python
mem = SimpleMem()  # auto mode
mem.add_text("User loves hiking in the Rocky Mountains.", tags=["session_id:D1"])
mem.add_image("photo.jpg", tags=["session_id:D1"])

result = mem.query("What does the user enjoy?", top_k=5)
for item in result.items:
    print(item["summary"])

mem.close()
```

### `create(mode="auto", **kwargs)`

Factory that returns a backend instance. `kwargs` are forwarded to the selected
backend's constructor.

```python
from simplemem import create

mem = create(mode="text", clear_db=True)
```

### `optimize(mem, dev_questions, max_rounds=7)`

Runs EvolveMem's offline diagnosis loop over `dev_questions`
(a list of `(question, ground_truth)` tuples) and returns an optimized `Config`.

### `Config` / `load_config(path)`

Retrieval configuration produced by `optimize()`. `Config.save(path)` writes JSON;
`load_config(path)` (or `Config.from_file(path)`) reads it back.

---

## Advanced Usage

### Parallel Processing for Large Datasets

For processing large dialogue datasets, enable parallel processing:

```python
from simplemem import create

mem = create(
    mode="text",
    clear_db=True,
    enable_parallel_processing=True,
    max_parallel_workers=16,  # Adjust based on your CPU cores
    enable_parallel_retrieval=True,
    max_retrieval_workers=8,
)
```

### Using Custom LLM Endpoints

SimpleMem talks to any OpenAI-compatible API:

```python
from simplemem import create

# Using Qwen (Alibaba DashScope)
mem = create(
    mode="text",
    api_key="your-qwen-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    clear_db=True,
)

# Using Azure OpenAI
mem = create(
    mode="text",
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    model="gpt-4.1-mini",
    clear_db=True,
)
```

### Multi-tenant Memory Tables

Use separate tables for different users or contexts:

```python
from simplemem import create

# User A's memory
system_a = create(mode="text", table_name="user_alice_memories", clear_db=False)

# User B's memory
system_b = create(mode="text", table_name="user_bob_memories", clear_db=False)
```

### Deep Thinking Mode

Enable enhanced reasoning for complex queries (supported by Qwen models):

```python
from simplemem import create

mem = create(mode="text", enable_thinking=True, clear_db=True)
```

---

## Data Models

The dialogue and memory data classes live in
`simplemem.core.models.memory_entry`.

### `MemoryEntry`

Represents an atomic, self-contained memory unit produced by the compression
pipeline (returned by `get_all_memories()`).

```python
from simplemem.core.models.memory_entry import MemoryEntry
```

### `Dialogue`

Represents a raw dialogue input for `add_dialogues()`.

```python
from simplemem.core.models.memory_entry import Dialogue

dialogue = Dialogue(
    dialogue_id=1,
    speaker="Alice",
    content="Let's discuss the new product launch",
    timestamp="2025-01-15T14:30:00",
)
```

---

## Environment Variables

Settings are read from the environment when they are not set in `config.py` or
passed to the constructor. The variable name matches the `config.py` key:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI-compatible API key | Required |
| `OPENAI_BASE_URL` | Custom API endpoint | None |
| `LLM_MODEL` | LLM model name | `"gpt-4.1-mini"` |
| `EMBEDDING_MODEL` | Embedding model | `"Qwen/Qwen3-Embedding-0.6B"` |
| `EMBEDDING_DIMENSION` | Embedding dimension | `1024` |
| `LANCEDB_PATH` | LanceDB storage path | `"./lancedb_data"` |
| `MEMORY_TABLE_NAME` | Memory table name | `"memory_entries"` |

Example `.env` file:

```bash
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
LANCEDB_PATH=./my_memory_db
```

> The MCP server (Docker) uses a separate set of variables; see
> [`.env.example`](../.env.example) and the [MCP documentation](../MCP/README.md).

---

## Examples

### Personal Assistant Memory

```python
from simplemem import create
import os

os.environ["OPENAI_API_KEY"] = "your-key"

# Create a persistent memory for a personal assistant
mem = create(
    mode="text",
    db_path="./assistant_memory",
    clear_db=False,  # Persist across sessions
)

# Add user preferences
mem.add_dialogue("User", "I prefer to wake up at 6am", "2025-01-15T08:00:00")
mem.add_dialogue("User", "I'm allergic to peanuts", "2025-01-15T08:05:00")
mem.add_dialogue("User", "My favorite restaurant is The Green Kitchen", "2025-01-15T08:10:00")
mem.finalize()

# Later, query preferences
answer = mem.ask("What are the user's dietary restrictions?")
print(answer)  # "The user is allergic to peanuts"
```

### Meeting Notes Processing

```python
from simplemem import create
from simplemem.core.models.memory_entry import Dialogue

mem = create(mode="text", clear_db=True)

# Process a meeting transcript
meeting_dialogues = [
    Dialogue(dialogue_id=1, speaker="PM", content="Let's review Q1 targets", timestamp="2025-01-15T10:00:00"),
    Dialogue(dialogue_id=2, speaker="Sales", content="We achieved 120% of our target", timestamp="2025-01-15T10:02:00"),
    Dialogue(dialogue_id=3, speaker="PM", content="Great! Q2 target is set to 50M", timestamp="2025-01-15T10:05:00"),
    Dialogue(dialogue_id=4, speaker="Finance", content="Budget approval needed by Friday", timestamp="2025-01-15T10:08:00"),
]

mem.add_dialogues(meeting_dialogues)
mem.finalize()

# Query meeting insights
print(mem.ask("What was the Q1 performance?"))
print(mem.ask("What's the deadline for budget approval?"))
```

### Multi-Session Memory

```python
from simplemem import create

# Session 1: Add information
mem = create(mode="text", db_path="./persistent_memory", clear_db=False)
mem.add_dialogue("User", "My birthday is March 15th", "2025-01-10T10:00:00")
mem.finalize()

# ... application closes ...

# Session 2: Query previously stored information
mem = create(mode="text", db_path="./persistent_memory", clear_db=False)
answer = mem.ask("When is the user's birthday?")
print(answer)  # "The user's birthday is March 15th"
```

---

## Troubleshooting

### Common Issues

#### API Key Not Found

```
Error: OpenAI API key not found
```

**Solution**: Set the API key via environment variable, `config.py`, or the constructor:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
# or
mem = create(mode="text", api_key="your-key")
```

#### Database Permission Error

```
Error: Cannot write to database path
```

**Solution**: Ensure the database path is writable:
```python
mem = create(mode="text", db_path="/path/with/write/permission")
```

#### Memory Not Found in Query

**Solution**: Ensure `finalize()` is called after adding all dialogues:
```python
mem.add_dialogue(...)
mem.finalize()  # Don't forget this!
answer = mem.ask(...)
```

#### Slow Performance with Large Datasets

**Solution**: Enable parallel processing:
```python
mem = create(
    mode="text",
    enable_parallel_processing=True,
    max_parallel_workers=16,
)
```

### Getting Help

- [GitHub Issues](https://github.com/aiming-lab/SimpleMem/issues)
- [Discord Community](https://discord.gg/KA2zC32M)
- [Paper](https://arxiv.org/abs/2601.02553)

---

## License

SimpleMem is released under the MIT License. See [LICENSE](../LICENSE) for details.
