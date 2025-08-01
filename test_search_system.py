#!/usr/bin/env python3
"""
Test the search system to find where B19013_001E search is failing
"""

import sys
import os
sys.path.append('knowledge-base')

def test_search_system():
    try:
        from kb_search import ConceptBasedCensusSearchEngine
        
        print("Testing search system for B19013_001E...")
        
        # Initialize the search engine
        engine = ConceptBasedCensusSearchEngine(
            catalog_dir="knowledge-base/table-catalog",
            variables_dir="knowledge-base/variables-db"
        )
        
        # Test search for median household income
        results = engine.search("median household income", max_results=5)
        
        print(f"\nSearch results for 'median household income':")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.variable_id} - {result.label} (confidence: {result.confidence:.3f})")
        
        # Check if B19013_001E is in the results
        b19013_found = any(r.variable_id == 'B19013_001E' for r in results)
        print(f"\nB19013_001E found in results: {b19013_found}")
        
        # Also test direct variable lookup
        var_info = engine.get_variable_info('B19013_001E')
        if var_info:
            print(f"\n✅ Direct lookup of B19013_001E successful:")
            print(f"Concept: {var_info.get('concept', 'N/A')}")
            print(f"Label: {var_info.get('label', 'N/A')}")
        else:
            print(f"\n❌ Direct lookup of B19013_001E failed")
            
        return True
        
    except ImportError as e:
        print(f"Cannot import search engine: {e}")
        return False
    except Exception as e:
        print(f"Error testing search: {e}")
        return False

if __name__ == "__main__":
    test_search_system()
