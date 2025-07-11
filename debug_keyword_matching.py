#!/usr/bin/env python3
"""
Debug what's actually being matched in keyword search
"""

import json
import re
import collections
from pathlib import Path

def debug_keyword_search():
    """Show exactly what tokens are being matched"""
    
    # Load the keyword search system
    import sys
    sys.path.append('.')
    from keyword_search_system import KeywordCensusSearch
    
    search_system = KeywordCensusSearch()
    search_system.build_index()
    
    # Test queries
    test_queries = [
        "average house cost",
        "latino population", 
        "household income",
        "how many elderly"
    ]
    
    tokenizer = re.compile(r"[A-Za-z0-9]+")
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("=" * 50)
        
        # Extract query tokens
        query_tokens = tokenizer.findall(query.lower())
        print(f"Query tokens: {query_tokens}")
        
        # See what each token matches
        for token in query_tokens:
            matches = search_system.keyword_index.get(token, set())
            print(f"  '{token}' → {len(matches)} variables")
            
            # Show first few matches with their labels
            for i, var_id in enumerate(list(matches)[:5]):
                if var_id in search_system.variable_metadata:
                    label = search_system.variable_metadata[var_id]['label']
                    print(f"    {var_id}: {label[:60]}...")
                if i >= 4:  # Show max 5
                    break
        
        # Get the actual search result
        results = search_system.search(query, k=1)
        if results:
            result = results[0]
            print(f"\n✅ Top result: {result['variable_id']}")
            print(f"   Label: {result['label']}")
            print(f"   Confidence: {result['confidence']}")
            
            # Check if this result actually contains query tokens
            result_text = f"{result['label']} {result['concept']}".lower()
            result_tokens = set(tokenizer.findall(result_text))
            query_token_set = set(query_tokens)
            
            overlap = query_token_set.intersection(result_tokens)
            print(f"   Query tokens in result: {overlap}")
            print(f"   Result tokens: {list(result_tokens)[:10]}...")
            
            if not overlap:
                print("   🚨 NO TOKEN OVERLAP - This is a false match!")
        else:
            print("❌ No results")

if __name__ == "__main__":
    debug_keyword_search()
