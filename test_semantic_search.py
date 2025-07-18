#!/usr/bin/env python3
"""
Test semantic search for commute/transportation queries.
Diagnoses why "commute time" isn't finding the right variables.

Usage:
    python test_semantic_search.py
"""

import sys
from pathlib import Path

# Add your knowledge-base directory to Python path
project_root = Path(__file__).parent
kb_path = project_root / "knowledge-base"

if kb_path.exists():
    sys.path.insert(0, str(kb_path))
    print(f"✅ Added knowledge-base path: {kb_path}")
else:
    print(f"❌ Knowledge-base directory not found at: {kb_path}")
    print("   Please run this script from your project root directory")
    sys.exit(1)

try:
    from kb_search import search, search_with_synonyms
    print("✅ Successfully imported semantic search functions")
except ImportError as e:
    print(f"❌ Failed to import kb_search: {e}")
    print("   Make sure kb_search.py exists in your knowledge-base directory")
    sys.exit(1)

def test_semantic_search():
    """Run comprehensive tests on semantic search for transportation queries."""
    
    print("\n" + "="*70)
    print("🔍 SEMANTIC SEARCH DIAGNOSTIC TESTS")
    print("="*70)
    
    # Test 1: Exact DP03 description
    print("\n1️⃣  TEST: Exact DP03 description")
    print("-" * 40)
    try:
        results = search("COMMUTING TO WORK", k=5)
        if results:
            print("✅ Found results for 'COMMUTING TO WORK':")
            for i, result in enumerate(results, 1):
                var_id = result.get('variable_id', 'Unknown')
                score = result.get('score', result.get('re_rank', 0))
                label = result.get('label', 'No label')
                print(f"   {i}. {var_id} (score: {score:.3f}) | {label[:60]}")
        else:
            print("❌ No results found for 'COMMUTING TO WORK'")
    except Exception as e:
        print(f"❌ Error searching 'COMMUTING TO WORK': {e}")
    
    # Test 2: Original failing query
    print("\n2️⃣  TEST: Original failing query")
    print("-" * 40)
    try:
        results = search("commute time", k=5)
        if results:
            print("✅ Found results for 'commute time':")
            for i, result in enumerate(results, 1):
                var_id = result.get('variable_id', 'Unknown')
                score = result.get('score', result.get('re_rank', 0))
                label = result.get('label', 'No label')
                print(f"   {i}. {var_id} (score: {score:.3f}) | {label[:60]}")
        else:
            print("❌ No results found for 'commute time'")
    except Exception as e:
        print(f"❌ Error searching 'commute time': {e}")
    
    # Test 3: Query variations
    print("\n3️⃣  TEST: Query variations")
    print("-" * 40)
    test_queries = [
        "commute",
        "travel time", 
        "transportation work",
        "journey work",
        "travel time to work",
        "mean travel time"
    ]
    
    for query in test_queries:
        try:
            results = search(query, k=3)
            print(f"\n'{query}':")
            if results:
                for i, result in enumerate(results, 1):
                    var_id = result.get('variable_id', 'Unknown')
                    score = result.get('score', result.get('re_rank', 0))
                    print(f"   {i}. {var_id} (score: {score:.3f})")
            else:
                print("   ❌ No results")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 4: Check for DP03 variables
    print("\n4️⃣  TEST: DP03 variables in index")
    print("-" * 40)
    try:
        results = search("DP03", k=10)
        if results:
            dp03_vars = [r for r in results if 'DP03' in r.get('variable_id', '')]
            print(f"✅ Found {len(dp03_vars)} DP03 variables:")
            for result in dp03_vars[:5]:  # Show first 5
                var_id = result.get('variable_id', 'Unknown')
                label = result.get('label', 'No label')
                print(f"   - {var_id} | {label[:50]}")
        else:
            print("❌ No DP03 variables found in index")
    except Exception as e:
        print(f"❌ Error searching for DP03: {e}")
    
    # Test 5: Check for B08303 variables  
    print("\n5️⃣  TEST: B08303 variables in index")
    print("-" * 40)
    try:
        results = search("B08303", k=5)
        if results:
            b08_vars = [r for r in results if 'B08303' in r.get('variable_id', '')]
            print(f"✅ Found {len(b08_vars)} B08303 variables:")
            for result in b08_vars:
                var_id = result.get('variable_id', 'Unknown')
                label = result.get('label', 'No label')
                print(f"   - {var_id} | {label[:50]}")
        else:
            print("❌ No B08303 variables found")
    except Exception as e:
        print(f"❌ Error searching for B08303: {e}")
    
    # Test 6: Search with synonyms
    print("\n6️⃣  TEST: Search with synonyms")
    print("-" * 40)
    try:
        results = search_with_synonyms("commute time", k=5)
        if results:
            print("✅ Results with synonym expansion:")
            for i, result in enumerate(results, 1):
                var_id = result.get('variable_id', 'Unknown')
                score = result.get('score', result.get('re_rank', 0))
                label = result.get('label', 'No label')
                print(f"   {i}. {var_id} (score: {score:.3f}) | {label[:60]}")
        else:
            print("❌ No results with synonym expansion")
    except Exception as e:
        print(f"❌ Error with synonym search: {e}")
    
    # Test 7: Transportation-related terms
    print("\n7️⃣  TEST: Transportation keywords")
    print("-" * 40)
    transport_keywords = ["transportation", "vehicle", "car", "public transit", "drove alone"]
    
    for keyword in transport_keywords:
        try:
            results = search(keyword, k=2)
            if results:
                print(f"\n'{keyword}': {len(results)} results")
                for result in results[:2]:
                    var_id = result.get('variable_id', 'Unknown')
                    table_id = var_id.split('_')[0] if '_' in var_id else 'Unknown'
                    print(f"   - {var_id} (table: {table_id})")
            else:
                print(f"\n'{keyword}': No results")
        except Exception as e:
            print(f"\n'{keyword}': Error - {e}")
    
    print("\n" + "="*70)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("="*70)
    print("Check the results above to identify:")
    print("1. Are DP03 variables missing from your index?")
    print("2. Are confidence thresholds too high?") 
    print("3. Is 'COMMUTING TO WORK' text not searchable?")
    print("4. Are B08303 variables present but not semantic enough?")
    print("5. Does synonym expansion help?")
    print("\nBased on results, we can fix the specific issue!")

if __name__ == "__main__":
    test_semantic_search()
