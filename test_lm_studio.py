#!/usr/bin/env python3
"""Test script to verify LM Studio integration."""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(override=True)

from rag.llm_providers import LLMManager

def test_lm_studio():
    print("Testing LM Studio integration...")

    manager = LLMManager()

    # List providers
    providers = manager.list_providers()
    print("Available providers:")
    for p in providers:
        print(f"  - {p['name']}: {'Available' if p['available'] else 'Not available'}")

    # Test generation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you respond briefly?"}
    ]

    try:
        response = manager.generate_response(messages, max_tokens=50)
        print(f"\n✅ Success! Response: {response}")
        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False

if __name__ == "__main__":
    success = test_lm_studio()
    sys.exit(0 if success else 1)