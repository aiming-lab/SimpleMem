source .venv/bin/activate
export JWT_SECRET_KEY="your-secure-random-secret-key"
export ENCRYPTION_KEY="your-32-byte-encryption-key!!"

export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen3:4b-instruct
export EMBEDDING_MODEL=qwen3-embedding:4b
export EMBEDDING_DIMENSION=2560
env

python run.py
