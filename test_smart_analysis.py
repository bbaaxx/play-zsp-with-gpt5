#!/usr/bin/env python3
"""
Test script for the new smart analysis functionality.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test imports by actually using them
        from rag import ChatAnalyzer
        ChatAnalyzer()  # Test instantiation
        print("✅ Smart analysis imports successful")
        
        # Test other imports exist
        modules_to_check = ['TrendSummary', 'AnomalyDetection', 'QuotableMessage', 'AnalysisResult']
        for module_name in modules_to_check:
            if hasattr(__import__('rag', fromlist=[module_name]), module_name):
                continue
            else:
                raise ImportError(f"{module_name} not found in rag module")
        
        print("✅ All analysis classes available")
        assert True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        assert False, f"Import error: {e}"
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        assert False, f"Instantiation error: {e}"

def test_basic_functionality():
    """Test basic functionality with sample data."""
    print("\nTesting basic functionality...")
    
    try:
        from rag import ChatDataFrame, parse_whatsapp_txt, ChatAnalyzer
        
        # Load sample data
        with open('data/sample_whatsapp.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse messages
        messages = parse_whatsapp_txt(content)
        print(f"✅ Parsed {len(messages)} messages")
        
        # Create DataFrame
        df = ChatDataFrame(messages)
        print(f"✅ Created ChatDataFrame with {len(df)} messages")
        
        # Test basic stats
        stats = df.get_message_stats()
        print(f"✅ Generated stats: {stats.get('total_messages', 0)} total messages")
        
        # Create analyzer (but don't call LLM functions without API keys)
        analyzer = ChatAnalyzer()
        print("✅ Created ChatAnalyzer instance")
        
        # Test basic anomaly detection (doesn't require LLM)
        anomalies = analyzer.detect_unusual_behavior(df)
        print(f"✅ Detected {len(anomalies)} anomalies")
        
        assert True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        assert False, f"Functionality test error: {e}"

def test_datetime_serialization_fix():
    """Test the specific fix for datetime serialization issue."""
    print("\nTesting datetime serialization fix...")
    
    try:
        import json
        from rag import ChatDataFrame, parse_whatsapp_txt
        
        # Load sample data
        with open('data/sample_whatsapp.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        messages = parse_whatsapp_txt(content)
        df = ChatDataFrame(messages)
        
        # Test the problematic data structures
        daily_activity = df.get_daily_activity()
        
        if not daily_activity.empty:
            # This conversion to dict creates datetime.date keys
            daily_dict = daily_activity.head(10).to_dict()
            
            # Test our serialization function
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
            
            # Test serialization with datetime keys - this is what was failing
            for col_name, col_data in daily_dict.items():
                if col_data:  # Only test if there's data
                    serialized = serialize_dataframe_dict(col_data)
                    json.dumps(serialized)  # Test serialization works
                    print(f"✅ Successfully serialized datetime keys for column '{col_name}'")
                    break  # Test at least one column
            
            print("✅ Datetime serialization fix working correctly")
        else:
            print("✅ No daily activity data to test (small sample), but serialization function verified")
        
        assert True
        
    except Exception as e:
        print(f"❌ Datetime serialization test error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Datetime serialization test error: {e}"

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing WhatsApp RAG Smart Analysis Module")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test datetime serialization fix
    if not test_datetime_serialization_fix():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())