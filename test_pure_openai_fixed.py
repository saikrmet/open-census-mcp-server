#!/usr/bin/env python3
"""Test pure OpenAI embeddings - bypass broken variables database"""

import sys
from pathlib import Path
import os

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "knowledge-base"))

def test_pure_openai_search():
    """Test search with ONLY OpenAI embeddings"""
    
    print("🔍 Testing Pure OpenAI Search")
    print("=" * 40)
    
    # Test OpenAI embedding generation
    try:
        from openai import OpenAI
        
        client = OpenAI()
        print("✅ OpenAI client created")
        
        # Test embedding generation
        test_queries = ["population", "median household income", "B01003_001E"]
        
        for query in test_queries:
            print(f"\n🧪 Testing: '{query}'")
            
            response = client.embeddings.create(
                input=[query],
                model="text-embedding-3-large"
            )
            
            embedding = response.data[0].embedding
            print(f"✅ Generated embedding: {len(embedding)} dimensions")
            
            # Convert to numpy array (what FAISS expects)
            import numpy as np
            embedding_array = np.array([embedding], dtype=np.float32)
            print(f"✅ Numpy array shape: {embedding_array.shape}")
            
    except Exception as e:
        print(f"❌ OpenAI embedding test failed: {e}")
        return False
    
    # Test table search with OpenAI (this should work)
    print(f"\n📊 Testing Table Search (OpenAI)")
    print("-" * 30)
    
    try:
        # Import from the original kb_search.py
        from kb_search import TableCatalogSearch
        
        # Force it to use OpenAI
        table_search = TableCatalogSearch("knowledge-base/table-catalog")
        
        # Check if it's using OpenAI
        print(f"Using OpenAI embeddings: {table_search.use_openai_embeddings}")
        
        # Test search
        results = table_search.search_tables("population", k=3)
        print(f"✅ Table search returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['table_id']}: confidence={result['confidence']:.3f}")
            
    except Exception as e:
        print(f"❌ Table search test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the main search engine
    print(f"\n🔍 Testing Full Search Engine")
    print("-" * 30)
    
    try:
        from kb_search import create_search_engine
        
        engine = create_search_engine()
        print("✅ Search engine created")
        
        # Test with a simple query
        results = engine.search("population", max_results=3)
        print(f"✅ Search returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.variable_id}: {result.label[:50]}... (conf: {result.confidence:.3f})")
            
    except Exception as e:
        print(f"❌ Full search test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    test_pure_openai_search()
