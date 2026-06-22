#!/usr/bin/env python3
"""
Test script to verify Ollama connection and get API token
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from server.integrations.ollama import OllamaClient
from config.settings import get_settings


async def test_ollama():
    """Test Ollama connection"""
    settings = get_settings()

    print("=" * 60)
    print("  SimpleMem MCP - Ollama Test")
    print("=" * 60)
    print()
    print(f"  LLM Provider: {settings.llm_provider}")
    print(f"  Ollama Base URL: {settings.ollama_base_url}")
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Embedding Model: {settings.embedding_model}")
    print()
    print("-" * 60)

    # Test 1: Verify connection
    print("\n1. Testing Ollama connection...")
    client = OllamaClient(base_url=settings.ollama_base_url)
    is_valid, error = await client.verify_api_key()
    await client.close()

    if is_valid:
        print("   ✓ Ollama is accessible!")
    else:
        print(f"   ✗ Connection failed: {error}")
        return False

    # Test 2: Test embedding
    print("\n2. Testing embedding generation...")
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        embedding_model=settings.embedding_model
    )
    try:
        embedding = await client.create_single_embedding("test")
        print(f"   ✓ Embedding generated successfully! Dimension: {len(embedding)}")
        await client.close()
    except Exception as e:
        print(f"   ✗ Embedding failed: {e}")
        await client.close()
        return False

    # Test 3: Test chat completion
    print("\n3. Testing chat completion...")
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        llm_model=settings.llm_model
    )
    try:
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "Say 'Hello from Ollama!'"}
            ],
            temperature=0.1
        )
        print(f"   ✓ Chat completion successful!")
        print(f"   Response: {response[:100]}...")
        await client.close()
    except Exception as e:
        print(f"   ✗ Chat completion failed: {e}")
        await client.close()
        return False

    print("\n" + "=" * 60)
    print("  All tests passed! Ollama is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    asyncio.run(test_ollama())
