#!/usr/bin/env python3
"""
Integration tests for the WhatsApp RAG system.
Tests that verify overall system functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest.mock
from datetime import datetime

import pytest

from rag import ChatDataFrame, ChatMessage
from business import FileManager, ChatProcessor, AnalyticsEngine


class TestBasicIntegration:
    """Test basic system integration without UI dependencies."""
    
    def test_business_module_imports(self):
        """Test that business module imports work correctly."""
        file_manager = FileManager()
        chat_processor = ChatProcessor()
        analytics_engine = AnalyticsEngine()
        
        assert file_manager is not None
        assert chat_processor is not None
        assert analytics_engine is not None
    
    def test_rag_module_imports(self):
        """Test that RAG module imports work correctly."""
        from rag import ChatDataFrame, ChatAnalyzer
        
        chat_df = ChatDataFrame()
        chat_analyzer = ChatAnalyzer()
        
        assert chat_df is not None
        assert chat_analyzer is not None
        assert chat_df.is_empty
    
    def test_progress_callback_handling(self):
        """Test that progress callback None handling works."""
        engine = AnalyticsEngine()
        df = ChatDataFrame()
        
        # This should not crash with None progress callback
        result = engine.analyze_chat_basic(df, None)
        assert result is not None
        
        # Test with mock progress callback
        class MockProgress:
            def __call__(self, value, desc=""):
                pass
                
        mock_progress = MockProgress()
        result = engine.analyze_chat_basic(df, mock_progress)
        assert result is not None
    
    def test_chat_processing_pipeline(self):
        """Test the complete chat processing pipeline."""
        # Create sample messages
        messages = [
            ChatMessage(
                chat_id="test",
                timestamp=datetime.now(),
                sender="Alice",
                text="¡Hola amigo! ¿Cómo estás?",
                line_no=1
            ),
            ChatMessage(
                chat_id="test", 
                timestamp=datetime.now(),
                sender="Bob",
                text="¡Muy bien! ¿Y tú?",
                line_no=2
            ),
        ]
        
        # Test ChatDataFrame creation and processing
        chat_df = ChatDataFrame(messages)
        assert not chat_df.is_empty
        assert len(chat_df) == 2
        
        # Test basic analytics without LLM calls
        engine = AnalyticsEngine()
        result = engine.analyze_chat_basic(chat_df, None)
        assert result is not None


class TestAppIntegration:
    """Test app integration components that don't require Gradio server."""
    
    @pytest.mark.skip(reason="App import requires audio dependencies not available in test environment")
    def test_app_state_creation(self):
        """Test AppState creation and basic functionality."""
        # Mock gradio components to avoid server startup
        with unittest.mock.patch('gradio.Blocks'), \
             unittest.mock.patch('gradio.Progress'):
            import app
            
            state = app.AppState()
            assert state is not None
            assert hasattr(state, 'analytics_engine')
            assert hasattr(state.analytics_engine, 'analyze_chat_basic')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])