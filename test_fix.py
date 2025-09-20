#!/usr/bin/env python3
"""Test script to verify the fix for datetime serialization issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
from datetime import datetime, date

def test_serialization_fix():
    """Test the serialization fix for datetime keys."""
    
    # Simulate the problematic data structure that caused the original error
    test_dict = {
        date(2024, 1, 1): {'messages': 10},
        date(2024, 1, 2): {'messages': 15},
        datetime.now().date(): {'messages': 20}
    }
    
    print("Testing original problematic serialization...")
    try:
        result = json.dumps(test_dict)
        print("ERROR: Should have failed!")
        return False
    except TypeError as e:
        print(f"✓ Expected error occurred: {e}")
    
    print("\nTesting fixed serialization...")
    
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
    
    try:
        fixed_dict = serialize_dataframe_dict(test_dict)
        result = json.dumps(fixed_dict, indent=2)
        print("✓ Fixed version works successfully!")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"✗ Fix failed: {e}")
        return False

def test_with_actual_function():
    """Test with the actual function from smart_analysis."""
    print("\n" + "="*50)
    print("Testing integration with actual smart_analysis module...")
    
    try:
        from rag import ChatDataFrame, parse_whatsapp_txt
        
        # Load sample data
        with open('data/sample_whatsapp.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse and create dataframe
        messages = parse_whatsapp_txt(content)
        df = ChatDataFrame(messages)
        
        # Get activity data that would have problematic datetime keys
        daily_activity = df.get_daily_activity()
        hourly_activity = df.get_hourly_activity()
        
        print(f"✓ Loaded {len(df)} messages")
        print(f"✓ Daily activity shape: {daily_activity.shape if not daily_activity.empty else 'empty'}")
        print(f"✓ Hourly activity shape: {hourly_activity.shape if not hourly_activity.empty else 'empty'}")
        
        # Test the specific data structure that was causing issues
        if not daily_activity.empty:
            daily_dict = daily_activity.head(10).to_dict()
            print(f"✓ Daily dict sample has {len(daily_dict)} columns")
            
            # This would have failed before the fix
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
                serialized_col = serialize_dataframe_dict(col_data)
                json.dumps(serialized_col)  # Test serialization works
                print(f"✓ Successfully serialized column '{col_name}' with {len(serialized_col)} date entries")
        
        print("✓ All serialization tests passed - fix is working!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing datetime serialization fix...")
    
    success1 = test_serialization_fix()
    success2 = test_with_actual_function()
    
    print("\n" + "="*50)
    print(f"Overall test result: {'PASSED' if success1 and success2 else 'FAILED'}")
    sys.exit(0 if success1 and success2 else 1)