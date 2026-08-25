# Requesty Integration - Quick Reference

## Why Requesty?

[Requesty](https://requesty.ai) is an OpenAI-compatible router that SimpleMem can use as a unified API gateway for LLM and embedding operations:

- **Single API Key**: One key for all models
- **Cost Tracking**: Built-in dashboard to monitor usage and costs
- **Model Flexibility**: Easy to switch between providers and models
- **Simpler Setup**: No need to manage multiple API keys

## Getting Started

### 1. Get Your API Key

Visit [app.requesty.ai/api-keys](https://app.requesty.ai/api-keys) and create an account to generate an API key.

### 2. Configure the Skill

```bash
cd SKILL/simplemem-skill
cp src/config.py.example src/config.py
```

Edit `src/config.py` to point the OpenAI-compatible base URL at Requesty and set your key:

```python
OPENROUTER_BASE_URL = "https://router.requesty.ai/v1"
OPENROUTER_API_KEY = "your-requesty-key"
```

### 3. Choose Your Models

Edit `src/config.py` to select models (Requesty uses the same `provider/model` naming as OpenRouter):

```python
# LLM Model (for chat/reasoning)
LLM_MODEL = "openai/gpt-4.1-mini"

# Embedding Model (for vector search)
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Embedding dimension (must match the embedding model)
EMBEDDING_DIMENSION = 1536
```

**Important**: The `EMBEDDING_DIMENSION` must match your chosen embedding model. Check the model documentation in the [Requesty docs](https://docs.requesty.ai).

## Cost Management

Track your usage on the [Requesty dashboard](https://app.requesty.ai).

## Troubleshooting

**"Invalid or expired API key" error**:
- Get a new key from [app.requesty.ai/api-keys](https://app.requesty.ai/api-keys)

**"Model not found" error**:
- Check the model name in the [Requesty docs](https://docs.requesty.ai)
- Ensure you're using the full path (e.g., `openai/gpt-4.1-mini`, not just `gpt-4.1-mini`)

**Embedding dimension mismatch error**:
```
RuntimeError: lance error: LanceError(Arrow): Arrow error: C Data interface error:
Invalid: ListType can only be casted to FixedSizeListType if the lists are all
the expected size.
```
- This means `EMBEDDING_DIMENSION` doesn't match your embedding model
- Check your model's actual dimension in the Requesty docs and update `EMBEDDING_DIMENSION`
- If you change the embedding dimension, clear the old database: `rm -rf data/lancedb/*`

**"Connection error" or timeout**:
- Check your internet connection
- The Requesty router may be experiencing downtime
- Try again after a few minutes
