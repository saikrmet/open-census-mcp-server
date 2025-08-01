#!/usr/bin/env python3
"""Test the fixed search with geographic field filtering"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_fixed_search():
    """Test search after filtering geographic fields"""
    print("🔧 Testing search with geographic field filtering...")
    
    try:
        from kb_search import ConceptBasedCensusSearchEngine
        
        engine = ConceptBasedCensusSearchEngine()
        
        # Test the specific case that was failing
        print("\\n🎯 Testing: 'median household income'")
        results = engine.search('median household income', max_results=5)
        
        print(f"📊 Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.variable_id} (confidence: {result.confidence:.3f})")
            print(f"     Table: {result.table_id} - {result.title[:60]}...")
            print(f"     Label: {result.label[:60]}...")
            print()
        
        if results:
            print("✅ SUCCESS: Search now returns results!")
            
            # Verify we got the right kind of results
            income_vars = [r for r in results if 'income' in r.label.lower()]
            median_vars = [r for r in results if 'median' in r.label.lower()]
            
            print(f"\\n📈 Quality check:")
            print(f"   Results with 'income': {len(income_vars)}")
            print(f"   Results with 'median': {len(median_vars)}")
            
            if income_vars and median_vars:
                print("✅ Results are semantically relevant!")
            else:
                print("⚠️ Results may not be optimal - need embedding improvement")
                
        else:
            print("❌ Still no results - deeper issue exists")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_search()
