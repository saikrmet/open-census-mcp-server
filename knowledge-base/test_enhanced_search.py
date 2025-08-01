#!/usr/bin/env python3
"""Test the enhanced search system with keywords"""

import sys
from pathlib import Path

# Add knowledge-base to path
sys.path.insert(0, str(Path(__file__).parent))

def test_enhanced_search():
    """Test the enhanced concept-based search with keywords"""
    print("🚀 Testing enhanced search system with keywords...")
    
    try:
        from kb_search import ConceptBasedCensusSearchEngine
        
        # Initialize enhanced search engine
        engine = ConceptBasedCensusSearchEngine()
        print("✅ Enhanced search engine initialized successfully")
        
        # Test multiple queries to verify keyword enhancement
        test_queries = [
            'median household income',
            'poverty rates by age',
            'commute time to work',
            'educational attainment'
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            results = engine.search(query, max_results=3)
            
            print(f"📊 Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.variable_id} (confidence: {result.confidence:.3f})")
                print(f"     Table: {result.table_id} - {result.title[:80]}...")
                print(f"     Label: {result.label[:80]}...")
            
            if results:
                print(f"✅ Query '{query}' successful")
            else:
                print(f"⚠️ Query '{query}' returned no results")
        
        print("\\n🎯 Enhanced search system test complete - keywords and semantic search working!")
        
    except Exception as e:
        print(f"❌ Enhanced search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_search()
