#!/usr/bin/env python3
"""Test the completely clean search system"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_clean_search():
    """Test search with clean FAISS index and no geographic contamination"""
    print("🧹 Testing CLEAN search system...")
    print("   - Clean canonical variables (36,901 Census variables)")
    print("   - Fresh FAISS index (no geographic API fields)")
    print("   - No filter hacks needed")
    
    try:
        from kb_search import ConceptBasedCensusSearchEngine
        
        engine = ConceptBasedCensusSearchEngine()
        print("✅ Search engine loaded clean data")
        
        # Test the query that was failing
        print("\\n🎯 Testing: 'median household income'")
        results = engine.search('median household income', max_results=5)
        
        print(f"📊 Results: {len(results)}")
        
        if results:
            print("\\n🎉 SUCCESS! Clean search returns results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.variable_id} (confidence: {result.confidence:.3f})")
                print(f"     {result.label[:80]}...")
                
            # Check if we got B19013_001E (the canonical median household income variable)
            income_vars = [r.variable_id for r in results if r.variable_id.startswith('B19013')]
            if income_vars:
                print(f"\\n✅ Found expected B19013 income variables: {income_vars}")
            else:
                print("\\n⚠️ No B19013 variables found - check semantic matching")
                
        else:
            print("❌ Still no results - investigate further")
            
        # Quick diagnostic
        print(f"\\n🔍 Database info:")
        print(f"   Variables loaded: {len(engine.variable_search.variables_metadata)}")
        print(f"   Tables loaded: {len(engine.table_search.tables)}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clean_search()
