#!/usr/bin/env python3
"""
Debug script to isolate semantic integration issues
"""

import sys
from pathlib import Path

# Test 1: Basic imports
print("=== TESTING BASIC IMPORTS ===")
try:
    import logging
    import requests
    import json
    print("✅ Basic imports successful")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    sys.exit(1)

# Test 2: Path setup
print("\n=== TESTING PATH SETUP ===")
current_path = Path(__file__).parent
knowledge_base_path = current_path.parent / "knowledge-base"
print(f"Current path: {current_path}")
print(f"Knowledge base path: {knowledge_base_path}")
print(f"Knowledge base exists: {knowledge_base_path.exists()}")

if knowledge_base_path.exists():
    kb_search_path = knowledge_base_path / "kb_search.py"
    print(f"kb_search.py exists: {kb_search_path.exists()}")
else:
    print("❌ Knowledge base directory not found")

# Test 3: Semantic search import
print("\n=== TESTING SEMANTIC SEARCH IMPORT ===")
sys.path.append(str(knowledge_base_path))

try:
    from kb_search import search
    print("✅ Semantic search import successful")
    
    # Test a simple search
    results = search("population", k=1)
    print(f"✅ Test search returned {len(results)} results")
    
except ImportError as e:
    print(f"❌ Semantic search import failed: {e}")
except Exception as e:
    print(f"❌ Semantic search test failed: {e}")

# Test 4: Simplified PythonCensusAPI
print("\n=== TESTING SIMPLIFIED API CLASS ===")

class SimplePythonCensusAPI:
    """Minimal version for testing"""
    
    def __init__(self):
        print("Initializing SimplePythonCensusAPI...")
        
        # Skip semantic search for now
        self.semantic_search = None
        
        # Basic mappings only
        self.variable_mappings = {
            'population': 'B01003_001E',
            'median income': 'B19013_001E',
            'teacher salary': 'B24022_011E'
        }
        print("✅ SimplePythonCensusAPI initialized successfully")
    
    def resolve_variables(self, variables):
        """Test variable resolution without semantic search"""
        resolved = []
        metadata = {}
        
        for var in variables:
            var_lower = var.lower().strip()
            
            if var_lower in self.variable_mappings:
                resolved.append(self.variable_mappings[var_lower])
                metadata[var] = {
                    'variables': [self.variable_mappings[var_lower]],
                    'source': 'hardcoded',
                    'confidence': 1.0
                }
            else:
                metadata[var] = {
                    'variables': [],
                    'source': 'none',
                    'confidence': 0.0,
                    'error': 'Not in hardcoded mappings'
                }
                
        return resolved, metadata

try:
    api = SimplePythonCensusAPI()
    
    # Test variable resolution
    test_vars = ["teacher salary", "population", "unknown variable"]
    resolved, metadata = api.resolve_variables(test_vars)
    
    print(f"✅ Variable resolution test successful:")
    for var, meta in metadata.items():
        print(f"  {var} -> {meta.get('variables', [])} (confidence: {meta.get('confidence', 0)})")
        
except Exception as e:
    print(f"❌ SimplePythonCensusAPI test failed: {e}")

print("\n=== DEBUGGING COMPLETE ===")
print("Run this script to identify the specific failure point.")
