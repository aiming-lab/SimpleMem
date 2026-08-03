#!/usr/bin/env python3
"""
Test script to verify OrcaRouter connection and get API token
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from server.integrations.orcarouter import OrcaRouterClient
from config.settings import get_settings


async def test_orcarouter():
    """Test OrcaRouter connection"""
    settings = get_settings()

    print("=" * 60)
    print("  SimpleMem MCP - OrcaRouter Test")
    print("=" * 60)
    print()
    print(f"  LLM Provider: {settings.llm_provider}")
    print(f"  OrcaRouter Base URL: {settings.orcarouter_base_url}")
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Embedding Model: {settings.embedding_model}")
    print()
    print("-" * 60)

    api_key = os.getenv("ORCAROUTER_API_KEY", "")
    if not api_key:
        print("  ✗ ORCAROUTER_API_KEY not set; skipping live tests")
        return False

    # Test 1: Verify API key
    print("\n1. Testing OrcaRouter API key...")
    client = OrcaRouterClient(api_key=api_key, base_url=settings.orcarouter_base_url)
    is_valid, error = await client.verify_api_key()
    await client.close()

    if is_valid:
        print("   ✓ OrcaRouter key is valid!")
    else:
        print(f"   ✗ Key validation failed: {error}")
        return False

    # Test 2: Test embedding
    print("\n2. Testing embedding generation...")
    client = OrcaRouterClient(
        api_key=api_key,
        base_url=settings.orcarouter_base_url,
        embedding_model="openai/text-embedding-3-small",
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
    client = OrcaRouterClient(
        api_key=api_key,
        base_url=settings.orcarouter_base_url,
        llm_model=settings.llm_model,
    )
    try:
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "Say 'Hello from OrcaRouter!'"}
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
    print("  All tests passed! OrcaRouter is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    asyncio.run(test_orcarouter())
