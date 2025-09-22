#!/usr/bin/env python3
"""
Tests for the adaptive analysis functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime

import pytest

from rag import ChatDataFrame, ChatMessage


class TestAdaptiveAnalysisImports:
    """Test that adaptive analysis components can be imported."""
    
    def test_adaptive_analysis_imports(self):
        """Test that all adaptive analysis components can be imported."""
        from rag.adaptive_analysis import (
            AdaptiveAnalyzer,
            ContextDetector
        )
        
        # Test basic instantiation
        analyzer = AdaptiveAnalyzer()
        detector = ContextDetector(llm_manager=None)
        
        assert analyzer is not None
        assert detector is not None


class TestContextDetection:
    """Test context detection functionality."""
    
    def test_context_category_creation(self):
        """Test ContextCategory creation."""
        from rag.adaptive_analysis import ContextCategory
        
        context = ContextCategory(
            category="friends_casual",
            confidence=0.8,
            evidence=["amigo", "jaja", "genial"],
            characteristics={"tone": "casual", "relationship": "friends"}
        )
        
        assert context.category == "friends_casual"
        assert context.confidence == 0.8
        assert "amigo" in context.evidence
        assert context.characteristics["tone"] == "casual"
    
    def test_pattern_based_context_detection(self):
        """Test pattern-based context detection without LLM calls."""
        from rag.adaptive_analysis import ContextDetector
        
        # Create sample chat data with casual friend conversation
        messages = [
            ChatMessage(
                chat_id="test",
                timestamp=datetime.now(),
                sender="Alice",
                text="¡Hola amigo! ¿Vamos al cine esta noche? jaja sería genial",
                line_no=1
            ),
            ChatMessage(
                chat_id="test", 
                timestamp=datetime.now(),
                sender="Bob",
                text="¡Perfecto! Me encanta la idea. ¿Qué película quieres ver?",
                line_no=2
            ),
        ]
        
        chat_df = ChatDataFrame(messages)
        detector = ContextDetector(llm_manager=None)
        
        # Test pattern-based detection
        contexts = detector._detect_pattern_contexts(chat_df)
        
        assert isinstance(contexts, list)
        # Should detect some pattern-based contexts
        if contexts:
            for ctx in contexts:
                assert hasattr(ctx, 'category')
                assert hasattr(ctx, 'confidence')
                assert hasattr(ctx, 'evidence')


class TestSpecializedAgents:
    """Test specialized agent factory."""
    
    def test_agent_factory_creation(self):
        """Test that agent factory can create specialized agents."""
        from rag.adaptive_analysis import SpecializedAgentFactory, ContextCategory
        
        test_contexts = [
            ContextCategory(
                category="friends_casual",
                confidence=0.8,
                evidence=["amigo", "jaja"],
                characteristics={"tone": "casual"}
            ),
            ContextCategory(
                category="work_formal",
                confidence=0.7,
                evidence=["reunión", "proyecto"],
                characteristics={"tone": "formal"}
            )
        ]
        
        agents = SpecializedAgentFactory.create_agents(test_contexts)
        
        assert isinstance(agents, list)
        assert len(agents) > 0
        
        # Check agent properties
        agent = agents[0]
        assert hasattr(agent, 'name')
        assert hasattr(agent, 'analysis_focus')
        assert hasattr(agent, 'system_prompt')
        assert isinstance(agent.analysis_focus, list)


class TestAdaptiveAnalyzer:
    """Test the main adaptive analyzer."""
    
    def test_adaptive_analyzer_creation(self):
        """Test AdaptiveAnalyzer can be created."""
        from rag.adaptive_analysis import AdaptiveAnalyzer
        
        analyzer = AdaptiveAnalyzer()
        
        assert analyzer is not None
        assert hasattr(analyzer, 'context_detector')
    
    def test_adaptive_analysis_result_structure(self):
        """Test AdaptiveAnalysisResult structure."""
        from rag.adaptive_analysis import AdaptiveAnalysisResult, ContextCategory
        from rag.smart_analysis import AnalysisResult
        
        # Create mock analysis components
        contexts = [
            ContextCategory(
                category="test",
                confidence=0.5,
                evidence=["test"],
                characteristics={}
            )
        ]
        
        # Create a basic analysis result (use actual AnalysisResult structure)
        from rag.smart_analysis import TrendSummary
        
        basic_analysis = AnalysisResult(
            trend_summaries=[
                TrendSummary(
                    trend_type="test",
                    description="Test summary",
                    confidence_score=0.8,
                    supporting_data={},
                    time_period="test period"
                )
            ],
            anomalies=[],
            quotable_messages=[],
            analysis_metadata={}
        )
        
        adaptive_result = AdaptiveAnalysisResult(
            basic_analysis=basic_analysis,
            detected_contexts=contexts,
            specialized_analyses={},
            adaptive_insights=[],
            analysis_metadata={}
        )
        
        assert adaptive_result.basic_analysis == basic_analysis
        assert len(adaptive_result.detected_contexts) == 1
        assert adaptive_result.detected_contexts[0].category == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])