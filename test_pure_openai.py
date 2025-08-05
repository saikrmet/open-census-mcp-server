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
        # Import the table search class
        import sys
        sys.path.insert(0, "knowledge-base")
        
        # This is hacky but let's test if table search works with OpenAI
        from kb_search_clean import TableCatalogSearch
        
        # Force it to use OpenAI
        table_search = TableCatalogSearch("knowledge-base/table-catalog")
        
        # Manually force OpenAI embeddings
        if hasattr(table_search, '_setup_openai_embeddings'):
            table_search._setup_openai_embeddings()
            print("✅ Forced table search to use OpenAI")
        
        # Test search
        results = table_search.search_tables("population", k=3)
        print(f"✅ Table search returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['table_id']}: confidence={result['confidence']:.3f}")
            
    except Exception as e:
        print(f"❌ Table search test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    test_pure_openai_search()
