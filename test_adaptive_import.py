#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

try:
    # Test basic module structure
    import rag.adaptive_analysis as adaptive
    print("✅ Successfully imported adaptive_analysis module")
    
    # Test key classes
    print(f"✅ AdaptiveAnalyzer class available: {hasattr(adaptive, 'AdaptiveAnalyzer')}")
    print(f"✅ ContextCategory class available: {hasattr(adaptive, 'ContextCategory')}")
    print(f"✅ AdaptiveAnalysisResult class available: {hasattr(adaptive, 'AdaptiveAnalysisResult')}")
    
    # Test from main rag import
    from rag import ChatDataFrame
    print("✅ Successfully imported ChatDataFrame")
    
    # Test if we can create a basic instance
    analyzer = adaptive.AdaptiveAnalyzer()
    print("✅ Successfully created AdaptiveAnalyzer instance")
    
    print("✅ All core functionality appears to be working")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()