#!/usr/bin/env python3
"""Quick syntax test for kb_search.py"""

try:
    import sys
    sys.path.insert(0, 'knowledge-base')
    
    # Test basic import
    print("Testing kb_search import...")
    import kb_search
    print("✅ kb_search imports successfully")
    
    # Test basic functionality
    print("Testing search engine creation...")
    engine = kb_search.create_search_engine()
    print("✅ Search engine created successfully")
    
    print("All tests passed!")
    
except SyntaxError as e:
    print(f"❌ Syntax Error: {e}")
    print(f"File: {e.filename}, Line: {e.lineno}")
except ImportError as e:
    print(f"❌ Import Error: {e}")
except Exception as e:
    print(f"❌ Other Error: {e}")
    import traceback
    traceback.print_exc()
