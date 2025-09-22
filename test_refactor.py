import sys
import os
sys.path.insert(0, ".")

from business import FileManager, ChatProcessor, AnalyticsEngine
from rag import ChatDataFrame
print("✅ All imports successful")

engine = AnalyticsEngine()
df = ChatDataFrame()
result = engine.analyze_chat_basic(df, None)
print("✅ Progress callback fix works")

import app
state = app.AppState()
print("✅ AppState works")

print("\n🎉 All refactoring fixes are working correctly!")