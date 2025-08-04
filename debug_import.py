#!/usr/bin/env python3
"""
Debug script to isolate import failures
"""

import sys
from pathlib import Path

print("=== IMPORT DEBUG TEST ===")

try:
    print("1. Testing basic imports...")
    import logging
    import asyncio
    import os
    print("✅ Basic imports OK")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    sys.exit(1)

try:
    print("2. Testing MCP imports...")
    from mcp.server import Server
    from mcp.types import TextContent, Tool
    print("✅ MCP imports OK")
except Exception as e:
    print(f"❌ MCP imports failed: {e}")
    sys.exit(1)

try:
    print("3. Testing Config import...")
    sys.path.insert(0, '/Users/brock/Documents/GitHub/census-mcp-server/src')
    from utils.config import Config
    print("✅ Config import OK")
except Exception as e:
    print(f"❌ Config import failed: {e}")

try:
    print("4. Testing kb_search path...")
    kb_path = Path('/Users/brock/Documents/GitHub/census-mcp-server/knowledge-base')
    if kb_path.exists():
        sys.path.insert(0, str(kb_path))
        print(f"✅ kb_search path added: {kb_path}")
    else:
        print(f"❌ kb_search path missing: {kb_path}")
except Exception as e:
    print(f"❌ kb_search path setup failed: {e}")

try:
    print("5. Testing kb_search import...")
    from kb_search import create_search_engine
    print("✅ kb_search import OK")
except Exception as e:
    print(f"❌ kb_search import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("6. Testing PythonCensusAPI import...")
    from data_retrieval.python_census_api import PythonCensusAPI
    print("✅ PythonCensusAPI import OK")
except Exception as e:
    print(f"❌ PythonCensusAPI import failed: {e}")
    import traceback
    traceback.print_exc()

print("=== IMPORT DEBUG COMPLETE ===")
