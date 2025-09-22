#!/usr/bin/env python3
"""
Tests for core RAG functionality without UI dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime

import pytest

from rag import ChatDataFrame, ChatMessage


class TestCoreRagFunctionality:
    """Test core RAG functionality."""
    
    def test_chat_dataframe_creation(self):
        """Test ChatDataFrame creation and basic operations."""
        # Test empty dataframe
        df = ChatDataFrame()
        assert df.is_empty
        assert len(df) == 0
        
        # Test with messages
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
        
        df = ChatDataFrame(messages)
        assert not df.is_empty
        assert len(df) == 2
    
    def test_message_statistics(self):
        """Test message statistics generation."""
        messages = [
            ChatMessage(
                chat_id="test",
                timestamp=datetime(2023, 10, 12, 21, 15),
                sender="Alice",
                text="Mensaje 1",
                line_no=1
            ),
            ChatMessage(
                chat_id="test", 
                timestamp=datetime(2023, 10, 12, 21, 16),
                sender="Bob",
                text="Mensaje 2",
                line_no=2
            ),
            ChatMessage(
                chat_id="test", 
                timestamp=datetime(2023, 10, 12, 21, 17),
                sender="Alice",
                text="Mensaje 3",
                line_no=3
            ),
        ]
        
        df = ChatDataFrame(messages)
        stats = df.get_message_stats()
        
        assert stats['total_messages'] == 3
        assert stats['unique_authors'] == 2
        assert stats['authors']['Alice'] == 2
        assert stats['authors']['Bob'] == 1
        assert stats['most_active_hour'] == 21
    
    def test_filtering_operations(self):
        """Test various filtering operations."""
        messages = [
            ChatMessage("test", datetime.now(), "Alice", "¡Hola amigo!", 1),
            ChatMessage("test", datetime.now(), "Bob", "¡Hola! ¿Cómo estás?", 2),
            ChatMessage("test", datetime.now(), "Alice", "Todo bien, gracias", 3),
        ]
        
        df = ChatDataFrame(messages)
        
        # Test author filtering
        alice_msgs = df.filter_by_author("Alice")
        assert len(alice_msgs) == 2
        
        # Test content filtering
        hola_msgs = df.filter_by_content("Hola", case_sensitive=False)
        assert len(hola_msgs) == 2
        
        # Test regex content filtering
        question_msgs = df.filter_by_content(r"\?", regex=True)
        assert len(question_msgs) == 1


class TestAdaptiveFunctionality:
    """Test adaptive analysis core functionality."""
    
    def test_adaptive_analysis_imports(self):
        """Test that adaptive analysis components can be imported."""
        try:
            from rag.adaptive_analysis import (
                AdaptiveAnalyzer,
                ContextDetector, 
                SpecializedAgentFactory,
                ContextCategory,
                AdaptiveAnalysisResult
            )
            
            # Test basic instantiation
            analyzer = AdaptiveAnalyzer()
            detector = ContextDetector(llm_manager=None)
            
            assert analyzer is not None
            assert detector is not None
            
        except ImportError as e:
            pytest.skip(f"Adaptive analysis not available: {e}")
    
    def test_context_detection_without_llm(self):
        """Test context detection without LLM calls."""
        try:
            from rag.adaptive_analysis import ContextDetector
            
            messages = [
                ChatMessage("test", datetime.now(), "Alice", "¡Hola amigo! jaja", 1),
                ChatMessage("test", datetime.now(), "Bob", "¿Qué tal? genial", 2),
            ]
            
            chat_df = ChatDataFrame(messages)
            detector = ContextDetector(llm_manager=None)
            
            # Test pattern-based context detection
            contexts = detector._detect_pattern_contexts(chat_df)
            
            assert isinstance(contexts, list)
            # Pattern-based detection should find some contexts
            if contexts:
                for ctx in contexts:
                    assert hasattr(ctx, 'category')
                    assert hasattr(ctx, 'confidence')
                    assert hasattr(ctx, 'evidence')
                    
        except ImportError as e:
            pytest.skip(f"Adaptive analysis not available: {e}")
    
    def test_specialized_agent_factory(self):
        """Test specialized agent factory."""
        try:
            from rag.adaptive_analysis import SpecializedAgentFactory, ContextCategory
            
            test_context = ContextCategory(
                category="friends_casual",
                confidence=0.8,
                evidence=["amigo", "jaja", "genial"],
                characteristics={"tone": "casual"}
            )
            
            agents = SpecializedAgentFactory.create_agents([test_context])
            
            assert isinstance(agents, list)
            assert len(agents) > 0
            
            if agents:
                agent = agents[0]
                assert hasattr(agent, 'name')
                assert hasattr(agent, 'analysis_focus')
                assert isinstance(agent.analysis_focus, list)
                
        except ImportError as e:
            pytest.skip(f"Adaptive analysis not available: {e}")


class TestFileProcessing:
    """Test file processing functionality."""
    
    def test_sample_file_loading(self):
        """Test loading sample WhatsApp file."""
        from rag import parse_whatsapp_txt
        
        try:
            with open('data/sample_whatsapp.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            
            messages = parse_whatsapp_txt(content)
            assert len(messages) > 0
            
            # Test that messages have required fields
            msg = messages[0]
            assert hasattr(msg, 'sender')
            assert hasattr(msg, 'text')
            assert hasattr(msg, 'timestamp')
            assert hasattr(msg, 'chat_id')
            assert hasattr(msg, 'line_no')
            
        except FileNotFoundError:
            pytest.skip("Sample file not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])