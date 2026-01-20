# OpenRouter Integration - Quick Reference

## Why OpenRouter?

SimpleMem uses **OpenRouter** as a unified API gateway for all LLM and embedding operations:

✅ **Single API Key**: One key for all models  
✅ **Cost Tracking**: Built-in dashboard to monitor usage and costs  
✅ **Model Flexibility**: Easy to switch between providers and models  
✅ **Simpler Setup**: No need to manage multiple API keys  

## Getting Started

### 1. Get Your API Key

Visit [openrouter.ai/keys](https://openrouter.ai/keys) and create an account. Your API key will start with `sk-or-`.

### 2. Configure the Skill

```bash
cd SKILL/simplemem-skill
cp src/config.py.example src/config.py
```

Edit `src/config.py`:

```python
OPENROUTER_API_KEY = "sk-or-your-actual-key-here"
```

### 3. Choose Your Models

Edit `src/config.py` to select models:

```python
# LLM Model (for chat/reasoning)
LLM_MODEL = "openai/gpt-4.1-mini"

# Embedding Model (for vector search)
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

# Embedding dimension (must match the embedding model)
EMBEDDING_DIMENSION = 4096
```

**Important**: The `EMBEDDING_DIMENSION` must match your chosen embedding model. Check the model documentation on [openrouter.ai/models](https://openrouter.ai/models).

## Cost Management

Track your usage on the [OpenRouter dashboard](https://openrouter.ai/activity).

## Troubleshooting

**"Invalid key format" error**:
- Make sure your key starts with `sk-or-`
- Get a new key from [openrouter.ai/keys](https://openrouter.ai/keys)

**"Model not found" error**:
- Check the model name at [openrouter.ai/models](https://openrouter.ai/models)
- Ensure you're using the full path (e.g., `openai/gpt-4.1-mini`, not just `gpt-4.1-mini`)

**Embedding dimension mismatch error**:
```
RuntimeError: lance error: LanceError(Arrow): Arrow error: C Data interface error: 
Invalid: ListType can only be casted to FixedSizeListType if the lists are all 
the expected size.
```
- This means `EMBEDDING_DIMENSION` doesn't match your embedding model
- Check your model's actual dimension on OpenRouter and update `EMBEDDING_DIMENSION`
- If you change the embedding dimension, clear the old database: `rm -rf data/lancedb/*`

**"Connection error" or timeout**:
- Check your internet connection
- OpenRouter may be experiencing downtime
- Try again after a few minutes
