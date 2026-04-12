# SimpleMem Package Usage Guide

This guide provides comprehensive documentation for using SimpleMem as a pip-installable Python package.

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
from simplemem import SimpleMemSystem

# Initialize the system with your API key
system = SimpleMemSystem(
    api_key="your-openai-api-key",
    clear_db=True  # Start fresh
)

# Add dialogues with timestamps
system.add_dialogue("Alice", "Let's meet at Starbucks tomorrow at 2pm", "2025-01-15T14:30:00")
system.add_dialogue("Bob", "Sure, I'll bring the report", "2025-01-15T14:31:00")

# Finalize memory encoding
system.finalize()

# Query the memory
answer = system.ask("When and where will Alice and Bob meet?")
print(answer)
# Output: "Alice and Bob will meet at Starbucks on January 16, 2025 at 2:00 PM"
```

### Using Environment Variables

```python
import os
from simplemem import SimpleMemSystem

# Set API key via environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize without explicit api_key parameter
system = SimpleMemSystem(clear_db=True)
```

---

## Configuration

SimpleMem offers flexible configuration through three priority levels:

1. **Constructor Parameters** (highest priority)
2. **Environment Variables**
3. **Default Values** (lowest priority)

### Using SimpleMemConfig

```python
from simplemem import SimpleMemConfig, set_config, SimpleMemSystem

# Create custom configuration
config = SimpleMemConfig(
    openai_api_key="your-api-key",
    llm_model="gpt-4.1-mini",
    embedding_model="Qwen/Qwen3-Embedding-0.6B",
    iris_hostname="localhost",
    iris_port=1972,
    iris_namespace="USER",
    iris_username="_SYSTEM",
    iris_password="SYS",
    enable_parallel_processing=True,
    max_parallel_workers=8,
)

# Set as global config
set_config(config)

# Create system (will use global config)
system = SimpleMemSystem(clear_db=True)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openai_api_key` | str | `$OPENAI_API_KEY` | OpenAI API key |
| `openai_base_url` | str | None | Custom API endpoint |
| `llm_model` | str | `"gpt-4.1-mini"` | LLM model name |
| `embedding_model` | str | `"Qwen/Qwen3-Embedding-0.6B"` | Embedding model |
| `iris_hostname` | str | `"localhost"` | IRIS server hostname |
| `iris_port` | int | `1972` | IRIS superserver port |
| `iris_namespace` | str | `"USER"` | IRIS namespace |
| `iris_username` | str | `"_SYSTEM"` | IRIS username |
| `iris_password` | str | `"SYS"` | IRIS password |
| `memory_table_name` | str | `"memory_entries"` | IRIS table (auto-created) |
| `enable_parallel_processing` | bool | True | Parallel memory building |
| `max_parallel_workers` | int | 16 | Max workers for building |
| `enable_parallel_retrieval` | bool | True | Parallel query execution |
| `max_retrieval_workers` | int | 8 | Max workers for retrieval |
| `enable_planning` | bool | True | Multi-query planning |
| `enable_reflection` | bool | True | Reflection-based retrieval |
| `max_reflection_rounds` | int | 2 | Max reflection iterations |

---

## Core API Reference

### SimpleMemSystem

The main class for interacting with SimpleMem.

#### Constructor

```python
SimpleMemSystem(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    db_path: Optional[str] = None,
    table_name: Optional[str] = None,
    clear_db: bool = False,
    enable_thinking: Optional[bool] = None,
    use_streaming: Optional[bool] = None,
    enable_planning: Optional[bool] = None,
    enable_reflection: Optional[bool] = None,
    max_reflection_rounds: Optional[int] = None,
    enable_parallel_processing: Optional[bool] = None,
    max_parallel_workers: Optional[int] = None,
    enable_parallel_retrieval: Optional[bool] = None,
    max_retrieval_workers: Optional[int] = None,
)
```

#### Methods

##### `add_dialogue(speaker, content, timestamp=None)`

Add a single dialogue entry to the memory.

```python
system.add_dialogue(
    speaker="Alice",
    content="I finished the quarterly report",
    timestamp="2025-01-15T10:00:00"  # ISO 8601 format
)
```

##### `add_dialogues(dialogues)`

Batch add multiple dialogues.

```python
from simplemem import Dialogue

dialogues = [
    Dialogue(dialogue_id=1, speaker="Alice", content="Hello", timestamp="2025-01-15T10:00:00"),
    Dialogue(dialogue_id=2, speaker="Bob", content="Hi there!", timestamp="2025-01-15T10:01:00"),
]
system.add_dialogues(dialogues)
```

##### `finalize()`

Process any remaining dialogues in the buffer. Always call this after adding all dialogues.

```python
system.finalize()
```

##### `ask(question)`

Query the memory system with a natural language question.

```python
answer = system.ask("What did Alice say about the report?")
```

##### `get_all_memories()`

Retrieve all stored memory entries (useful for debugging).

```python
memories = system.get_all_memories()
for mem in memories:
    print(f"Entry: {mem.lossless_restatement}")
```

##### `print_memories()`

Print all memory entries in a formatted manner.

```python
system.print_memories()
```

### create_system()

Factory function to create a SimpleMem system with simplified parameters.

```python
from simplemem import create_system

system = create_system(
    clear_db=True,
    enable_parallel_processing=True,
    max_parallel_workers=8
)
```

---

## Advanced Usage

### Parallel Processing for Large Datasets

For processing large dialogue datasets, enable parallel processing:

```python
system = SimpleMemSystem(
    api_key="your-key",
    clear_db=True,
    enable_parallel_processing=True,
    max_parallel_workers=16,  # Adjust based on your CPU cores
    enable_parallel_retrieval=True,
    max_retrieval_workers=8
)
```

### Using Custom LLM Endpoints

SimpleMem supports OpenAI-compatible APIs:

```python
# Using Qwen API
system = SimpleMemSystem(
    api_key="your-qwen-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    clear_db=True
)

# Using Azure OpenAI
system = SimpleMemSystem(
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/",
    model="gpt-4",
    clear_db=True
)
```

### Multi-tenant Memory Tables

Use separate tables for different users or contexts:

```python
# User A's memory
system_a = SimpleMemSystem(
    api_key="your-key",
    table_name="user_alice_memories",
    clear_db=False
)

# User B's memory
system_b = SimpleMemSystem(
    api_key="your-key",
    table_name="user_bob_memories",
    clear_db=False
)
```

### Deep Thinking Mode

Enable enhanced reasoning for complex queries (supported by Qwen models):

```python
system = SimpleMemSystem(
    api_key="your-key",
    enable_thinking=True,
    clear_db=True
)
```

---

## Data Models

### MemoryEntry

Represents an atomic, self-contained memory unit.

```python
from simplemem import MemoryEntry

# MemoryEntry structure
entry = MemoryEntry(
    entry_id="unique-id",
    lossless_restatement="Alice discussed the marketing strategy with Bob at Starbucks on 2025-01-15.",
    keywords=["Alice", "Bob", "marketing", "strategy"],
    timestamp="2025-01-15T14:30:00",
    location="Starbucks, Shanghai",
    persons=["Alice", "Bob"],
    entities=["marketing strategy"],
    topic="Product marketing discussion"
)
```

### Dialogue

Represents a raw dialogue input.

```python
from simplemem import Dialogue

dialogue = Dialogue(
    dialogue_id=1,
    speaker="Alice",
    content="Let's discuss the new product launch",
    timestamp="2025-01-15T14:30:00"
)
```

---

## Environment Variables

SimpleMem supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_BASE_URL` | Custom API endpoint | None |
| `SIMPLEMEM_MODEL` | LLM model name | `"gpt-4.1-mini"` |
| `SIMPLEMEM_EMBEDDING_MODEL` | Embedding model | `"Qwen/Qwen3-Embedding-0.6B"` |
| `SIMPLEMEM_IRIS_HOSTNAME` | IRIS server hostname | `"localhost"` |
| `SIMPLEMEM_IRIS_PORT` | IRIS superserver port | `1972` |

Example `.env` file:

```bash
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
SIMPLEMEM_MODEL=gpt-4.1-mini
SIMPLEMEM_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
SIMPLEMEM_DB_PATH=./my_memory_db
```

---

## Examples

### Personal Assistant Memory

```python
from simplemem import SimpleMemSystem
import os

os.environ["OPENAI_API_KEY"] = "your-key"

# Create a persistent memory for personal assistant
system = SimpleMemSystem(
    db_path="./assistant_memory",
    clear_db=False  # Persist across sessions
)

# Add user preferences
system.add_dialogue("User", "I prefer to wake up at 6am", "2025-01-15T08:00:00")
system.add_dialogue("User", "I'm allergic to peanuts", "2025-01-15T08:05:00")
system.add_dialogue("User", "My favorite restaurant is The Green Kitchen", "2025-01-15T08:10:00")
system.finalize()

# Later, query preferences
answer = system.ask("What are the user's dietary restrictions?")
print(answer)  # "The user is allergic to peanuts"
```

### Meeting Notes Processing

```python
from simplemem import SimpleMemSystem, Dialogue

system = SimpleMemSystem(api_key="your-key", clear_db=True)

# Process meeting transcript
meeting_dialogues = [
    Dialogue(dialogue_id=1, speaker="PM", content="Let's review Q1 targets", timestamp="2025-01-15T10:00:00"),
    Dialogue(dialogue_id=2, speaker="Sales", content="We achieved 120% of our target", timestamp="2025-01-15T10:02:00"),
    Dialogue(dialogue_id=3, speaker="PM", content="Great! Q2 target is set to 50M", timestamp="2025-01-15T10:05:00"),
    Dialogue(dialogue_id=4, speaker="Finance", content="Budget approval needed by Friday", timestamp="2025-01-15T10:08:00"),
]

system.add_dialogues(meeting_dialogues)
system.finalize()

# Query meeting insights
print(system.ask("What was the Q1 performance?"))
print(system.ask("What's the deadline for budget approval?"))
```

### Multi-Session Memory

```python
from simplemem import SimpleMemSystem

# Session 1: Add information
system = SimpleMemSystem(
    api_key="your-key",
    db_path="./persistent_memory",
    clear_db=False
)
system.add_dialogue("User", "My birthday is March 15th", "2025-01-10T10:00:00")
system.finalize()

# ... application closes ...

# Session 2: Query previously stored information
system = SimpleMemSystem(
    api_key="your-key",
    db_path="./persistent_memory",
    clear_db=False  # Keep existing data
)
answer = system.ask("When is the user's birthday?")
print(answer)  # "The user's birthday is March 15th"
```

---

## Troubleshooting

### Common Issues

#### API Key Not Found

```
Error: OpenAI API key not found
```

**Solution**: Set the API key via environment variable or constructor parameter:
```python
os.environ["OPENAI_API_KEY"] = "your-key"
# or
system = SimpleMemSystem(api_key="your-key")
```

#### Database Permission Error

```
Error: Cannot write to database path
```

**Solution**: Ensure the database path is writable:
```python
system = SimpleMemSystem(db_path="/path/with/write/permission")
```

#### Memory Not Found in Query

**Solution**: Ensure `finalize()` is called after adding all dialogues:
```python
system.add_dialogue(...)
system.finalize()  # Don't forget this!
answer = system.ask(...)
```

#### Slow Performance with Large Datasets

**Solution**: Enable parallel processing:
```python
system = SimpleMemSystem(
    enable_parallel_processing=True,
    max_parallel_workers=16
)
```

### Getting Help

- [GitHub Issues](https://github.com/aiming-lab/SimpleMem/issues)
- [Discord Community](https://discord.gg/KA2zC32M)
- [Paper](https://arxiv.org/abs/2601.02553)

---

## License

SimpleMem is released under the MIT License. See [LICENSE](../LICENSE) for details.
