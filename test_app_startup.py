#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

try:
    # Test if app can be imported and basic UI built
    print("Testing app import...")
    import app
    print("✅ App module imported successfully")
    
    print("Testing UI build...")
    ui = app.build_ui()
    print("✅ UI built successfully")
    
    print("Testing state initialization...")
    state = app.AppState()
    print("✅ AppState initialized successfully")
    
    print("✅ All basic app functionality working")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Runtime error: {e}")
    import traceback
    traceback.print_exc()