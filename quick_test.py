#!/usr/bin/env python3
"""
Quick import test to verify components are loadable
"""

import sys
import os
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
kb_path = current_dir / "knowledge-base"
src_path = current_dir / "src"

sys.path.insert(0, str(kb_path))
sys.path.insert(0, str(src_path))

print("🔧 Testing component imports...")

try:
    from llm_statistical_advisor import create_llm_statistical_advisor, LLMStatisticalAdvisor
    print("✅ LLM Statistical Advisor imports")
except Exception as e:
    print(f"❌ LLM Statistical Advisor import failed: {e}")

try:
    from kb_search import create_search_engine, CensusSearchEngine
    print("✅ Search engine imports")
except Exception as e:
    print(f"❌ Search engine import failed: {e}")

try:
    from geographic_parsing import GeographicContext, create_geographic_parser
    print("✅ Geographic parsing imports")
except Exception as e:
    print(f"❌ Geographic parsing import failed: {e}")

try:
    from variable_search import create_variables_search, VariablesSearch
    print("✅ Variable search imports")
except Exception as e:
    print(f"❌ Variable search import failed: {e}")

# Check environment
openai_key = os.getenv('OPENAI_API_KEY')
print(f"🔑 OpenAI API Key: {'Present' if openai_key else 'Missing'}")

# Check critical directories
kb_dirs = [
    kb_path / "variables-db",
    kb_path / "table-catalog", 
    kb_path / "methodology-db",
    kb_path / "geo-db" / "geography.db"
]

for dir_path in kb_dirs:
    exists = dir_path.exists()
    print(f"📁 {dir_path.name}: {'✅' if exists else '❌'}")

print("\n🎯 Component availability summary:")
print("- All critical Python modules should import successfully")
print("- OpenAI API key should be present for embeddings")
print("- Knowledge base directories should exist")
print("\nIf all show ✅, the wiring should work!")
