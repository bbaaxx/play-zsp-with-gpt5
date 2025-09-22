#!/usr/bin/env python3
"""Simple test for adaptive analysis functionality"""

import sys
import os
sys.path.insert(0, '.')

def test_import():
    """Test if all components can be imported"""
    try:
        # Test individual module imports
        from rag.adaptive_analysis import (
            AdaptiveAnalyzer,
            ContextDetector, 
            SpecializedAgentFactory,
            ContextCategory,
            AdaptiveAnalysisResult
        )
        print("✅ All adaptive analysis components imported")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    try:
        from rag.adaptive_analysis import ContextDetector, SpecializedAgentFactory, ContextCategory
        
        # Test context detector creation
        detector = ContextDetector(llm_manager=None)  # No LLM manager for basic test
        print("✅ ContextDetector created")
        
        # Test agent factory
        fake_context = ContextCategory(
            category="friends_casual",
            confidence=0.8,
            evidence=["test evidence"],
            characteristics={"test": "data"}
        )
        
        agents = SpecializedAgentFactory.create_agents([fake_context])
        print(f"✅ Agent factory created {len(agents)} agents")
        
        if agents:
            agent = agents[0]
            print(f"✅ Agent details: {agent.name}, Focus: {len(agent.analysis_focus)} areas")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_integration():
    """Test app integration"""
    try:
        # Check if app can import our new functions
        import app
        
        # Check if new functions exist
        has_adaptive = hasattr(app, 'analyze_chat_adaptive')
        print(f"✅ App has analyze_chat_adaptive: {has_adaptive}")
        
        # Check if AppState has new field
        state = app.AppState()
        has_adaptive_field = hasattr(state, 'last_adaptive_analysis')
        print(f"✅ AppState has last_adaptive_analysis: {has_adaptive_field}")
        
        return has_adaptive and has_adaptive_field
        
    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Adaptive Analysis Implementation ===")
    
    success = True
    success &= test_import()
    success &= test_basic_functionality()
    success &= test_app_integration()
    
    if success:
        print("\n🎉 All tests passed! Implementation looks good.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    sys.exit(0 if success else 1)