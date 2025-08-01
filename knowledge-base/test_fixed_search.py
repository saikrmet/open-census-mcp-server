#!/usr/bin/env python3
"""Test script for the fixed search system"""

import sys
from pathlib import Path

# Add knowledge-base to path
sys.path.insert(0, str(Path(__file__).parent))

from kb_search import ConceptBasedCensusSearchEngine

def test_search():
    """Test the fixed search system"""
    print("Testing fixed concept-based search system...")
    
    try:
        # Initialize search engine
        engine = ConceptBasedCensusSearchEngine()
        print("✅ Search engine initialized successfully")
        
        # Test search
        print("\n🔍 Testing search for 'median household income'...")
        results = engine.search('median household income', max_results=3)
        
        print(f"\n📊 Search returned {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.variable_id} (confidence: {result.confidence:.3f})")
            print(f"     Table: {result.table_id} - {result.title}")
            print(f"     Label: {result.label}")
            print()
        
        print("✅ Search completed successfully - no segfault!")
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search()
