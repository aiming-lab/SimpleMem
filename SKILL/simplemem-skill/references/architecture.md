# SimpleMem Architecture

## Overview

SimpleMem is a persistent conversational memory system that stores dialogues in a vector database (LanceDB) for semantic retrieval and question answering. It uses **OpenRouter** as a unified API gateway for both LLM and embedding services.

## Components

### API Layer - OpenRouter

- **Unified Gateway**: All LLM and embedding calls go through OpenRouter API
- **Multi-Provider**: Access models from Anthropic, OpenAI, Google, Qwen, and more
- **Single API Key**: One API key for all services
- **Cost Tracking**: Built-in usage tracking and budgeting

### Storage Layer

- **Vector Database**: LanceDB for efficient similarity search
- **Table Structure**: Each dialogue entry contains:
  - `speaker`: String
  - `content`: String
  - `timestamp`: Datetime
  - `embedding`: Vector representation of the content

### Retrieval Layer

- **Hybrid Retriever**: Combines semantic (vector) and keyword (BM25) search
- **Configurable top-k**: Control how many relevant entries to retrieve

### Generation Layer

- **Answer Generator**: Uses retrieved context to generate natural language answers
- **Reflection Mode**: Optional multi-step reasoning for complex queries

## Data Storage

By default, the system stores data in:
- Database path: `SKILL/simplemem-skill/data/lancedb/`
- Default table: `memory_entries`

## Configuration

The system requires an OpenRouter API key:
- **Get key**: [openrouter.ai/keys](https://openrouter.ai/keys)
- **Key format**: Starts with `sk-or-`
- **Models**: Configure LLM and embedding models in `config.py`

### Recommended Models

**LLM Models (via OpenRouter)**:
- `anthropic/claude-3.5-sonnet` - Powerful reasoning (recommended)
- `anthropic/claude-3-haiku` - Fast and economical
- `openai/gpt-4o-mini` - Fast OpenAI
- `qwen/qwen-2.5-72b-instruct` - Free, good quality

**Embedding Models (via OpenRouter)**:
- `openai/text-embedding-3-small` - Economical (1536 dims)
- `openai/text-embedding-3-large` - Best quality (3072 dims)

## Workflow

1. **Input**: Dialogue entry (speaker, content, timestamp)
2. **Embedding**: Content is converted to a vector via OpenRouter
3. **Storage**: Entry and embedding are stored in LanceDB
4. **Retrieval**: Queries are embedded and used to find similar entries
5. **Generation**: Retrieved entries provide context for answer generation via OpenRouter

## Performance Considerations

- **API Costs**: Both embedding and LLM calls cost credits (track on OpenRouter dashboard)
- **Vector Search**: Efficient even with large numbers of entries
- **Batch Operations**: More efficient than individual additions
- **Response Time**: Depends on top-k, reflection settings, and model choice