#!/usr/bin/env python3
"""
Test script to verify import fixes after refactoring.
"""

import sys
import traceback

def test_imports():
    """Test that all main imports work correctly."""
    try:
        # Test business module imports
        from business import FileManager, ChatProcessor, AnalyticsEngine
        print("✅ Business module imports successful")
        
        # Test rag module imports
        from rag import ChatDataFrame, ChatAnalyzer, AdaptiveAnalyzer
        print("✅ RAG module imports successful")
        
        # Test that classes can be instantiated
        file_manager = FileManager()
        chat_processor = ChatProcessor()
        analytics_engine = AnalyticsEngine()
        print("✅ Business objects created successfully")
        
        # Test ChatDataFrame
        chat_df = ChatDataFrame()
        print(f"✅ ChatDataFrame created, is_empty: {chat_df.is_empty}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_app_imports():
    """Test that app.py imports work."""
    try:
        # Mock gradio Progress to avoid issues
        import unittest.mock
        with unittest.mock.patch('gradio.Progress'):
            from app import AppState, build_ui
            print("✅ App module imports successful")
            
            # Test AppState
            state = AppState()
            print("✅ AppState created successfully")
            
        return True
    except Exception as e:
        print(f"❌ App import test failed: {e}")
        traceback.print_exc()
        return False

def test_progress_callback_handling():
    """Test that progress callback handling works."""
    try:
        from business import AnalyticsEngine
        from rag import ChatDataFrame
        
        engine = AnalyticsEngine()
        empty_df = ChatDataFrame()
        
        # Test with None progress callback (should not crash)
        result = engine.analyze_chat_basic(empty_df, None)
        print("✅ Progress callback None handling works")
        
        # Test with mock progress callback
        class MockProgress:
            def __call__(self, value, desc=""):
                print(f"Progress: {value*100:.0f}% - {desc}")
                
        mock_progress = MockProgress()
        result = engine.analyze_chat_basic(empty_df, mock_progress)
        print("✅ Progress callback mock handling works")
        
        return True
    except Exception as e:
        print(f"❌ Progress callback test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing import fixes after refactoring...\n")
    
    success = True
    
    print("1. Testing basic imports...")
    success &= test_imports()
    print()
    
    print("2. Testing app imports...")
    success &= test_app_imports()
    print()
    
    print("3. Testing progress callback handling...")
    success &= test_progress_callback_handling()
    print()
    
    if success:
        print("🎉 All tests passed! Refactoring appears successful.")
        sys.exit(0)
    else:
        print("💥 Some tests failed. Check the errors above.")
        sys.exit(1)