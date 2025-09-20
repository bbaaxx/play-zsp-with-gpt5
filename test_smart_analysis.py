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
        from rag import ChatAnalyzer, TrendSummary, AnomalyDetection, QuotableMessage, AnalysisResult
        print("✅ Smart analysis imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from rag import ChatDataFrame, parse_whatsapp_txt
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Core import error: {e}")
        return False
    
    return True

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
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

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
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())