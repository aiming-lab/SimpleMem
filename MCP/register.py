#!/usr/bin/env python3
"""
Simple registration script to get API token for SimpleMem MCP
"""
import requests
import json

API_BASE = "http://localhost:8000"

def register():
    """Register and get token"""
    print("=" * 60)
    print("  SimpleMem MCP - Registration")
    print("=" * 60)
    print()

    # For Ollama, we don't need a real API key
    api_key = "ollama-placeholder-key"

    print(f"Registering with API key: {api_key}")
    print()

    try:
        response = requests.post(
            f"{API_BASE}/api/auth/register",
            json={"openrouter_api_key": api_key},
            timeout=30
        )

        data = response.json()

        if data.get("success"):
            print("✓ Registration successful!")
            print()
            print("-" * 60)
            print("  Your Credentials:")
            print("-" * 60)
            print(f"  Token: {data['token']}")
            print(f"  User ID: {data['user_id']}")
            print(f"  MCP Endpoint: {API_BASE}/mcp")
            print()
            print("-" * 60)
            print("  Claude Desktop Config:")
            print("-" * 60)
            config = {
                "mcpServers": {
                    "simplemem": {
                        "type": "http",
                        "url": f"{API_BASE}/mcp",
                        "headers": {
                            "Authorization": f"Bearer {data['token']}"
                        }
                    }
                }
            }
            print(json.dumps(config, indent=2))
            print("=" * 60)
            return data['token']
        else:
            print(f"✗ Registration failed: {data.get('error')}")
            return None

    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure the server is running on http://localhost:8000")
        return None


if __name__ == "__main__":
    register()
