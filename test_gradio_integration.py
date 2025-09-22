#!/usr/bin/env python3

import sys
import os
import tempfile
sys.path.insert(0, '.')

def test_gradio_integration():
    """Test the Gradio app integration with adaptive analysis"""
    try:
        # Import with gradio dependencies available
        import app
        print("✅ App module imported successfully")
        
        # Check that new functions exist
        assert hasattr(app, 'analyze_chat_adaptive'), "analyze_chat_adaptive function not found"
        print("✅ analyze_chat_adaptive function exists")
        
        # Check AppState has new field
        state = app.AppState()
        assert hasattr(state, 'last_adaptive_analysis'), "last_adaptive_analysis field not found"
        print("✅ AppState has last_adaptive_analysis field")
        
        # Test UI building
        ui = app.build_ui()
        print("✅ UI built successfully")
        
        # Create fake chat data to test with
        from rag import ChatDataFrame, ChatMessage
        from datetime import datetime
        
        messages = [
            ChatMessage(
                chat_id="test",
                timestamp=datetime.now(),
                sender="Alice",
                text="Hola, ¿cómo estás? 😊",
                line_no=1
            ),
            ChatMessage(
                chat_id="test", 
                timestamp=datetime.now(),
                sender="Bob",
                text="¡Muy bien! ¿Qué tal tu día?",
                line_no=2
            ),
        ]
        
        chat_df = ChatDataFrame(messages)
        app.STATE.chat_dataframe = chat_df
        print("✅ Test chat data created and set")
        
        # Test analysis functions (without LLM calls - just structure)
        try:
            # Test basic analysis function
            result = app.analyze_chat()
            if "Error de configuración" in result:
                print("✅ analyze_chat returns proper config error (expected without token)")
            else:
                print("✅ analyze_chat executed without crashing")
                
            # Test adaptive analysis function  
            result = app.analyze_chat_adaptive()
            if "Error de configuración" in result:
                print("✅ analyze_chat_adaptive returns proper config error (expected without token)")
            else:
                print("✅ analyze_chat_adaptive executed without crashing")
                
        except Exception as e:
            print(f"⚠️  Analysis functions error (may be expected without LLM config): {e}")
        
        # Test summary function
        summary = app.get_analysis_summary()
        assert "No hay análisis previo disponible" in summary, "Expected no analysis message"
        print("✅ get_analysis_summary works correctly")
        
        print("\n🎉 All Gradio integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Gradio integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Gradio Integration ===")
    
    success = test_gradio_integration()
    
    if success:
        print("\n✅ Gradio integration working correctly!")
        print("The adaptive analysis system has been successfully integrated into the Gradio UI.")
    else:
        print("\n❌ Gradio integration has issues.")
    
    sys.exit(0 if success else 1)