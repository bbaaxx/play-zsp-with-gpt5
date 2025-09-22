#!/usr/bin/env python3
"""
Validation script for refactoring fixes.
"""

import os
import sys

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Validating refactored imports and functionality...")
    
    # Test 1: Basic imports
    try:
        from business import AnalyticsEngine, FileManager, ChatProcessor
        from rag import ChatDataFrame
        print("✅ Core imports successful")
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False
    
    # Test 2: Object creation
    try:
        analytics = AnalyticsEngine()
        file_mgr = FileManager()
        chat_proc = ChatProcessor()
        chat_df = ChatDataFrame()
        print("✅ Object creation successful")
    except Exception as e:
        print(f"❌ Object creation failed: {e}")
        return False
    
    # Test 3: Progress callback handling
    try:
        result = analytics.analyze_chat_basic(chat_df, None)
        print("✅ Progress callback (None) handling works")
    except Exception as e:
        print(f"❌ Progress callback handling failed: {e}")
        return False
    
    # Test 4: App imports (without running server)
    try:
        import unittest.mock
        # Mock gradio to avoid actual server startup
        with unittest.mock.patch('gradio.Blocks'), \
             unittest.mock.patch('gradio.Progress'):
            from app import AppState, analyze_chat, analyze_chat_adaptive
            state = AppState()
            print("✅ App module imports successful")
    except Exception as e:
        print(f"❌ App imports failed: {e}")
        return False
    
    print("\n🎉 All validation checks passed! Refactoring is successful.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)