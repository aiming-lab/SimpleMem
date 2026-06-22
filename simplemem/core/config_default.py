"""
Configuration file - System parameters and LLM settings

IMPORTANT:
1. Copy this file to config.py
2. Configure your settings below
3. Never commit config.py to version control (it contains your API key)
"""

# ============================================================================
# LLM Configuration
# ============================================================================

# OpenAI API Key (required)
# Get your key from: https://platform.openai.com/api-keys
OPENAI_API_KEY = "your-api-key-here"

# Custom OpenAI Base URL (optional)
# Set to None to use default OpenAI endpoint
# Examples:
#   - Qwen/Alibaba: "https://dashscope.aliyuncs.com/compatible-mode/v1"
#   - Azure OpenAI: "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT"
#   - Local server: "http://localhost:8000/v1"
#   - OpenAI (default): None
OPENAI_BASE_URL = None

# LLM Model name
# Examples: "gpt-4.1-mini", "gpt-4.1", "qwen3-max", "qwen-plus-2025-07-28"
LLM_MODEL = "gpt-4.1-mini"

# Embedding model (local, no API needed)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024  # For Qwen3: up to 1024, supports 32-1024
EMBEDDING_CONTEXT_LENGTH = 32768  # Qwen3 supports 32k context


# ============================================================================
# Advanced LLM Features
# ============================================================================

# Enable deep thinking mode (for Qwen and compatible models)
# Adds extra_body={"enable_thinking": True} to API calls
# Set to False for OpenAI models (they don't support this)
ENABLE_THINKING = False

# Enable streaming responses (outputs content as it's generated)
USE_STREAMING = True

# Enable JSON format mode (ensures LLM outputs valid JSON)
# Adds response_format={"type": "json_object"} to API calls
# Helps prevent parsing failures from extra text like ```json
USE_JSON_FORMAT = False


# ============================================================================
# Memory Building Parameters
# ============================================================================

# Number of dialogues per window (for locomo; for other dataset, please finetune it)
WINDOW_SIZE = 40

# Window overlap size (for context continuity)
OVERLAP_SIZE = 2


# ============================================================================
# Retrieval Parameters (can be adjusted to balance between token usage and performance)
# ============================================================================

# Max entries returned by semantic search (vector similarity)
SEMANTIC_TOP_K = 25

# Max entries returned by keyword search (BM25 matching)
KEYWORD_TOP_K = 5

# Max entries returned by structured search (metadata filtering)
STRUCTURED_TOP_K = 5


# ============================================================================
# Database Configuration
# ============================================================================

# Path to LanceDB storage
LANCEDB_PATH = "./lancedb_data"

# Memory table name
MEMORY_TABLE_NAME = "memory_entries"



# ============================================================================
# Parallel Processing Configuration
# ============================================================================

# Memory Building Parallel Processing
ENABLE_PARALLEL_PROCESSING = True
MAX_PARALLEL_WORKERS = 16  # Number of parallel workers for memory building

# Retrieval Parallel Processing  
ENABLE_PARALLEL_RETRIEVAL = True
MAX_RETRIEVAL_WORKERS = 8  # Number of parallel workers for retrieval queries

# Planning and Reflection Configuration
ENABLE_PLANNING = True
ENABLE_REFLECTION = True
MAX_REFLECTION_ROUNDS = 2


# ============================================================================
# LLM-as-Judge Configuration (not used yet)
# ============================================================================

# Judge LLM API Key (optional - if None, uses OPENAI_API_KEY)
JUDGE_API_KEY = "your api-key here"

# Judge LLM Base URL (optional - if None, uses OPENAI_BASE_URL)
# Example: Use cheaper endpoint for evaluation
JUDGE_BASE_URL = "https://api.openai.com/v1/"

# Judge LLM Model (optional - if None, uses LLM_MODEL)
JUDGE_MODEL = "gpt-4.1-mini"

# Judge specific settings
JUDGE_ENABLE_THINKING = False  # Usually false for evaluation tasks
JUDGE_USE_STREAMING = False    # Usually false for evaluation
JUDGE_TEMPERATURE = 0.3        

# Example configurations:
# 1. Use cheaper model for judge evaluation:
#    JUDGE_MODEL = "gpt-4.1-mini"
#
# 2. Use different API provider for judge:
#    JUDGE_API_KEY = "your-judge-api-key"
#    JUDGE_BASE_URL = "https://api.different-provider.com/v1"
#    JUDGE_MODEL = "different-provider-model"
#
# 3. Use Qwen for judge (if available):
#    JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#    JUDGE_MODEL = "qwen-plus-2025-09-11"

