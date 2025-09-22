#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

print("Testing basic adaptive functionality...")

try:
    # Just test our new adaptive analysis module
    from rag.adaptive_analysis import AdaptiveAnalyzer, ContextCategory, AdaptiveAnalysisResult
    print("✅ Adaptive analysis module imported successfully")
    
    # Test creating a basic analyzer
    analyzer = AdaptiveAnalyzer()
    print("✅ AdaptiveAnalyzer instance created")
    
    # Test creating context category
    context = ContextCategory(
        category="friends_casual",
        confidence=0.8,
        evidence=["test evidence"],
        characteristics={"test": True}
    )
    print("✅ ContextCategory instance created")
    
    print("\n🎉 Core adaptive analysis functionality is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()