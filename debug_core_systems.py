#!/usr/bin/env python3
"""Debug variable mapping system"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "knowledge-base"))

def test_variable_mapping():
    """Test if variable mapping works at all"""
    
    print("🔍 Testing Variable Mapping System")
    print("=" * 50)
    
    # Test 1: Can we load kb_search?
    try:
        print("Testing kb_search import...")
        import kb_search
        print("✅ kb_search imports successfully")
        
        print("Testing search engine creation...")
        engine = kb_search.create_search_engine()
        print("✅ Search engine created")
        
    except Exception as e:
        print(f"❌ kb_search failed: {e}")
        return
    
    # Test 2: Basic variable search
    print("\n🔍 Testing Basic Variable Search")
    print("-" * 30)
    
    test_queries = [
        "population",
        "total population", 
        "B01003_001E",  # Should be exact match
        "median household income",
        "poverty rate"
    ]
    
    for query in test_queries:
        try:
            print(f"\nSearching: '{query}'")
            results = engine.search(query, max_results=3)
            
            if results:
                print(f"✅ Found {len(results)} results")
                for i, result in enumerate(results[:2]):
                    print(f"  {i+1}. {result.variable_id} - {result.label[:60]}...")
                    print(f"     Confidence: {result.confidence:.2f}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    # Test 3: Check if variable metadata loaded
    print(f"\n📊 Database Stats")
    print("-" * 20)
    print(f"Variables loaded: {len(engine.variable_search.variables_metadata)}")
    print(f"Tables loaded: {len(engine.table_search.tables)}")
    
    # Test 4: Sample a few variable records
    print(f"\n📋 Sample Variables")
    print("-" * 20)
    for i, var in enumerate(engine.variable_search.variables_metadata[:3]):
        print(f"{i+1}. {var.get('variable_id', 'NO_ID')}: {var.get('label', 'NO_LABEL')[:50]}...")

def test_geography():
    """Test geographic resolution"""
    
    print("\n\n🌍 Testing Geographic Resolution")
    print("=" * 50)
    
    try:
        from data_retrieval.geographic_handler import CompleteGeographicHandler
        
        print("Testing geographic handler...")
        handler = CompleteGeographicHandler()
        print("✅ Geographic handler created")
        
        # Test problem locations
        test_locations = [
            "United States",  # Should work
            "CA",            # Should work  
            "Austin, TX",    # Likely broken
            "Sheridan, WY",  # Definitely broken
            "Richmond, CA"   # Likely broken
        ]
        
        for location in test_locations:
            try:
                print(f"\nResolving: '{location}'")
                result = handler.resolve_location(location)
                
                if 'error' in result:
                    print(f"❌ Failed: {result['error']}")
                    if result.get('suggestions'):
                        print(f"   Suggestions: {result['suggestions'][:2]}")
                else:
                    print(f"✅ Success: {result['name']} ({result['geography_type']})")
                    if result['geography_type'] == 'place':
                        print(f"   FIPS: {result.get('state_fips')}:{result.get('place_fips')}")
                        
            except Exception as e:
                print(f"❌ Exception: {e}")
        
        handler.close()
        
    except Exception as e:
        print(f"❌ Geographic handler failed: {e}")

if __name__ == "__main__":
    test_variable_mapping()
    test_geography()
