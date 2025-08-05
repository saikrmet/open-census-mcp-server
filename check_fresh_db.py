#!/usr/bin/env python3
"""Check dimensions of the files in variables-db (the fresh ones)"""

import faiss
import json
from pathlib import Path

def check_fresh_variables_db():
    """Check the fresh variables-db directory"""
    
    print("🔍 Checking Fresh Variables Database")
    print("=" * 40)
    
    variables_db_dir = Path("knowledge-base/variables-db")
    
    # Check FAISS file
    faiss_file = variables_db_dir / "variables.faiss"
    if faiss_file.exists():
        try:
            index = faiss.read_index(str(faiss_file))
            dimensions = index.d
            count = index.ntotal
            
            embed_type = "OpenAI ✅" if dimensions == 3072 else "SentenceTransformers ❌"
            
            print(f"📄 variables.faiss:")
            print(f"   Dimensions: {dimensions} ({embed_type})")
            print(f"   Vectors: {count:,}")
            
        except Exception as e:
            print(f"❌ Error reading FAISS: {e}")
    
    # Check build_info.json
    build_info_file = variables_db_dir / "build_info.json"
    if build_info_file.exists():
        try:
            with open(build_info_file) as f:
                build_info = json.load(f)
            
            print(f"\n📋 build_info.json:")
            print(f"   Embedding model: {build_info.get('embedding_model', 'unknown')}")
            print(f"   Embedding dimensions: {build_info.get('embedding_dimension', 'unknown')}")
            print(f"   Variable count: {build_info.get('variable_count', 'unknown')}")
            print(f"   Build timestamp: {build_info.get('build_timestamp', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Error reading build_info: {e}")

def show_fix_options():
    """Show how to fix the directory mismatch"""
    
    print(f"\n🛠️ Fix Options")
    print("=" * 20)
    print("Option 1: Move files to expected location")
    print("  mv knowledge-base/variables-db/* knowledge-base/variables-faiss/")
    print()
    print("Option 2: Update kb_search.py to look in variables-db")
    print("  Change variables_dir default from 'variables-faiss' to 'variables-db'")

if __name__ == "__main__":
    check_fresh_variables_db()
    show_fix_options()
