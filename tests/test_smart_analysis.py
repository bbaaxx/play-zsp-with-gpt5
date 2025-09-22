#!/usr/bin/env python3
"""
Tests for smart analysis functionality including datetime serialization fixes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from datetime import datetime, date

import pytest

from rag import ChatDataFrame, parse_whatsapp_txt, ChatAnalyzer


class TestSmartAnalysisImports:
    """Test smart analysis module imports."""
    
    def test_basic_imports(self):
        """Test that all smart analysis components can be imported."""
        from rag import ChatAnalyzer, ChatDataFrame
        
        # Test instantiation
        analyzer = ChatAnalyzer()
        df = ChatDataFrame()
        
        assert analyzer is not None
        assert df is not None
    
    def test_analysis_classes_available(self):
        """Test that analysis result classes are available."""
        # These should be available in the rag module
        from rag import ChatDataFrame
        
        # Test that we can create a dataframe and call basic methods
        df = ChatDataFrame()
        assert df.is_empty
        
        # Test with sample data
        with open('data/sample_whatsapp.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        messages = parse_whatsapp_txt(content)
        df = ChatDataFrame(messages)
        
        assert not df.is_empty
        assert len(df) > 0


class TestDatetimeSerializationFix:
    """Test the datetime serialization fix."""
    
    def test_datetime_serialization_problem(self):
        """Test that the original datetime serialization problem exists."""
        # Simulate the problematic data structure
        test_dict = {
            date(2024, 1, 1): {'messages': 10},
            date(2024, 1, 2): {'messages': 15},
            datetime.now().date(): {'messages': 20}
        }
        
        # This should fail with TypeError
        with pytest.raises(TypeError):
            json.dumps(test_dict)
    
    def test_datetime_serialization_fix(self):
        """Test the datetime serialization fix."""
        def serialize_dataframe_dict(df_dict):
            """Convert datetime keys to strings for JSON serialization."""
            if not df_dict:
                return {}
            
            serialized = {}
            for key, value in df_dict.items():
                # Convert datetime keys to string
                if hasattr(key, 'strftime'):
                    str_key = key.strftime('%Y-%m-%d') if hasattr(key, 'date') else str(key)
                else:
                    str_key = str(key)
                
                # Convert values that may be problematic
                if isinstance(value, dict):
                    value = serialize_dataframe_dict(value)
                
                serialized[str_key] = value
            
            return serialized
        
        # Test with problematic data
        test_dict = {
            date(2024, 1, 1): {'messages': 10},
            date(2024, 1, 2): {'messages': 15},
            datetime.now().date(): {'messages': 20}
        }
        
        # This should work with the fix
        fixed_dict = serialize_dataframe_dict(test_dict)
        result = json.dumps(fixed_dict, indent=2)
        
        assert result is not None
        assert '"2024-01-01"' in result
        assert '"2024-01-02"' in result
    
    def test_with_actual_data(self):
        """Test datetime serialization fix with actual chat data."""
        # Load sample data
        with open('data/sample_whatsapp.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse and create dataframe
        messages = parse_whatsapp_txt(content)
        df = ChatDataFrame(messages)
        
        if not df.is_empty:
            # Get activity data that would have problematic datetime keys
            daily_activity = df.get_daily_activity()
            
            if not daily_activity.empty:
                daily_dict = daily_activity.head(10).to_dict()
                
                # Test serialization function
                def serialize_dataframe_dict(df_dict):
                    if not df_dict:
                        return {}
                    
                    serialized = {}
                    for key, value in df_dict.items():
                        if hasattr(key, 'strftime'):
                            str_key = key.strftime('%Y-%m-%d') if hasattr(key, 'date') else str(key)
                        else:
                            str_key = str(key)
                        
                        if isinstance(value, dict):
                            value = serialize_dataframe_dict(value)
                        
                        serialized[str_key] = value
                    
                    return serialized
                
                # Test serialization of nested dict with date keys
                for col_name, col_data in daily_dict.items():
                    if col_data:  # Only test if there's data
                        serialized_col = serialize_dataframe_dict(col_data)
                        json.dumps(serialized_col)  # Should not raise exception
                        break


class TestChatAnalyzerFunctionality:
    """Test ChatAnalyzer functionality without LLM calls."""
    
    def test_analyzer_creation(self):
        """Test ChatAnalyzer can be created."""
        analyzer = ChatAnalyzer()
        assert analyzer is not None
    
    def test_basic_anomaly_detection(self):
        """Test basic anomaly detection that doesn't require LLM."""
        with open('data/sample_whatsapp.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        messages = parse_whatsapp_txt(content)
        df = ChatDataFrame(messages)
        
        if not df.is_empty:
            analyzer = ChatAnalyzer()
            
            # Test anomaly detection (should work without LLM for basic patterns)
            anomalies = analyzer.detect_unusual_behavior(df)
            assert isinstance(anomalies, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])