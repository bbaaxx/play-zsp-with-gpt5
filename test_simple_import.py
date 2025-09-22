#!/usr/bin/env python3

print("Testing imports...")

try:
    from business import FileManager, ChatProcessor, AnalyticsEngine
    print("✅ Business imports OK")
except Exception as e:
    print(f"❌ Business imports failed: {e}")

try:
    from rag import ChatDataFrame
    print("✅ RAG imports OK")
except Exception as e:
    print(f"❌ RAG imports failed: {e}")

try:
    analytics = AnalyticsEngine()
    df = ChatDataFrame()
    print(f"✅ Objects created. ChatDataFrame is_empty: {df.is_empty}")
except Exception as e:
    print(f"❌ Object creation failed: {e}")

print("Import test complete.")