#!/usr/bin/env python3
"""Debug OpenAI embeddings - check what's actually being used"""

import os
from pathlib import Path

def check_openai_setup():
    """Check OpenAI API configuration"""
    
    print("🔍 OpenAI Configuration Check")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OPENAI_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    # Test OpenAI import
    try:
        from openai import OpenAI
        print("✅ OpenAI library imports successfully")
        
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created")
        
        # Test embedding call
        print("\n🧪 Testing embedding generation...")
        response = client.embeddings.create(
            input=["test population query"],
            model="text-embedding-3-large"
        )
        
        embedding = response.data[0].embedding
        print(f"✅ OpenAI embedding created: {len(embedding)} dimensions")
        
        if len(embedding) == 3072:
            print("✅ Correct embedding dimension (3072)")
            return True
        else:
            print(f"❌ Wrong embedding dimension: {len(embedding)} (expected 3072)")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

def check_database_dimensions():
    """Check what dimensions the databases expect"""
    
    print("\n🗄️ Database Dimension Check")  
    print("=" * 40)
    
    # Check variables FAISS
    variables_dir = Path("knowledge-base/variables-faiss")
    build_info_file = variables_dir / "build_info.json"
    
    if build_info_file.exists():
        import json
        with open(build_info_file) as f:
            build_info = json.load(f)
        
        dim = build_info.get('embedding_dimension', 'unknown')
        model = build_info.get('embedding_model', 'unknown')
        print(f"Variables DB: {dim} dimensions, model: {model}")
        
        if dim == 3072:
            print("✅ Variables built with OpenAI embeddings")
        else:
            print(f"❌ Variables built with wrong dimensions: {dim}")
    else:
        print("❌ Variables build_info.json not found")
    
    # Check table catalog
    catalog_dir = Path("knowledge-base/table-catalog")
    mapping_file = catalog_dir / "table_mapping_enhanced.json"
    
    if mapping_file.exists():
        import json
        with open(mapping_file) as f:
            mapping_data = json.load(f)
        
        dim = mapping_data.get('embedding_dimension', 'unknown')
        print(f"Tables DB: {dim} dimensions")
        
        if dim == 3072:
            print("✅ Tables built with OpenAI embeddings")
        else:
            print(f"❌ Tables built with wrong dimensions: {dim}")
    else:
        print("❌ Tables mapping file not found")

if __name__ == "__main__":
    openai_ok = check_openai_setup()
    check_database_dimensions()
    
    if openai_ok:
        print("\n🎯 Action Plan:")
        print("1. Force all search engines to use OpenAI embeddings")
        print("2. Remove all SentenceTransformers fallbacks")
        print("3. Ensure dimension consistency (3072 everywhere)")
    else:
        print("\n❌ Fix OpenAI setup first!")
