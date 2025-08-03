#!/usr/bin/env python3
"""
Debug script to test imports step by step
Run this to identify exactly where the OpenAI key is needed
"""

import os
import sys
from pathlib import Path

print("🧪 Testing imports step by step...")

# Test 1: Basic imports
try:
    print("\n1. Testing basic imports...")
    import logging
    import sqlite3
    from typing import Dict, Any, Optional, List
    print("✅ Basic imports OK")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    sys.exit(1)

# Test 2: Geographic handler (should not need OpenAI)
try:
    print("\n2. Testing geographic handler...")
    from data_retrieval.geographic_handler import GeographicHandler
    print("✅ Geographic handler import OK")
    
    # Test initialization
    gh = GeographicHandler()
    print("✅ Geographic handler initialization OK")
    
    # Test a simple lookup
    result = gh.resolve_location("Minnesota")
    print(f"✅ Geographic resolution test: {result.get('name', 'Success')}")
    
except Exception as e:
    print(f"❌ Geographic handler failed: {e}")
    print("This should NOT require OpenAI key - it's pure SQLite")

# Test 3: Census API (might need OpenAI for semantic search)
try:
    print("\n3. Testing Census API...")
    from data_retrieval.python_census_api import PythonCensusAPI
    print("✅ Census API import OK")
    
    # Test initialization
    api = PythonCensusAPI()
    print("✅ Census API initialization OK")
    
except Exception as e:
    print(f"❌ Census API failed: {e}")
    print("This might need OpenAI key if it uses semantic search")

# Test 4: Check environment variables
print("\n4. Checking environment variables...")
census_key = os.getenv('CENSUS_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')

print(f"CENSUS_API_KEY: {'✅ Set' if census_key else '❌ Missing'}")
print(f"OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Missing'}")

if not openai_key:
    print("\n💡 If you need OpenAI embeddings, set:")
    print("export OPENAI_API_KEY='your-key-here'")

print("\n🎯 Import test complete!")
