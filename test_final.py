import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_basic_imports():
    try:
        from business import FileManager, ChatProcessor, AnalyticsEngine
        from rag import ChatDataFrame, ChatAnalyzer
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_app_components():
    try:
        # Import app module
        import app
        
        # Test AppState
        state = app.AppState()
        print("✅ AppState created")
        
        # Test that analytics methods exist
        assert hasattr(state.analytics_engine, 'analyze_chat_basic')
        assert hasattr(state.analytics_engine, 'analyze_chat_adaptive')
        print("✅ Analytics methods exist")
        
        return True
    except Exception as e:
        print(f"❌ App components test failed: {e}")
        return False

def test_progress_callback_fix():
    try:
        from business import AnalyticsEngine
        from rag import ChatDataFrame
        
        engine = AnalyticsEngine()
        df = ChatDataFrame()
        
        # Test with None - this should not crash anymore
        result = engine.analyze_chat_basic(df, None)
        print("✅ Progress callback None handling fixed")
        
        return True
    except Exception as e:
        print(f"❌ Progress callback test failed: {e}")
        return False

if __name__ == "__main__":
    print("Final refactor validation...")
    
    success = True
    success &= test_basic_imports()
    success &= test_app_components() 
    success &= test_progress_callback_fix()
    
    if success:
        print("\n🎉 All tests passed! Refactor is working correctly.")
    else:
        print("\n💥 Some tests failed.")