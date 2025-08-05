#!/usr/bin/env python3
"""Check actual FAISS index dimensions vs build_info.json"""

import faiss
import json
from pathlib import Path

def check_actual_faiss_dimensions():
    """Check what dimensions the FAISS indices actually have"""
    
    print("🔍 Checking Actual FAISS Dimensions")
    print("=" * 40)
    
    # Check variables FAISS
    variables_dir = Path("knowledge-base/variables-faiss")
    variables_faiss = variables_dir / "variables.faiss"
    build_info_file = variables_dir / "build_info.json"
    
    if variables_faiss.exists():
        try:
            # Load FAISS index
            index = faiss.read_index(str(variables_faiss))
            actual_dim = index.d
            print(f"Variables FAISS actual dimensions: {actual_dim}")
            
            if actual_dim == 3072:
                print("✅ Variables FAISS is OpenAI (3072)")
            elif actual_dim == 768:
                print("❌ Variables FAISS is SentenceTransformers (768)")
            else:
                print(f"⚠️ Variables FAISS is unknown model ({actual_dim})")
                
        except Exception as e:
            print(f"❌ Failed to read variables FAISS: {e}")
    else:
        print("❌ Variables FAISS not found")
    
    # Check build_info.json
    if build_info_file.exists():
        with open(build_info_file) as f:
            build_info = json.load(f)
        reported_dim = build_info.get('embedding_dimension', 'unknown')
        print(f"build_info.json reports: {reported_dim} dimensions")
        
        if actual_dim != reported_dim:
            print(f"⚠️ MISMATCH: FAISS={actual_dim}, build_info={reported_dim}")
            print("The build_info.json is wrong!")
    else:
        print("❌ build_info.json not found")
    
    # Check tables FAISS for comparison
    print(f"\n📊 Checking Tables FAISS")
    tables_dir = Path("knowledge-base/table-catalog")
    tables_faiss = tables_dir / "table_embeddings_enhanced.faiss"
    
    if tables_faiss.exists():
        try:
            index = faiss.read_index(str(tables_faiss))
            tables_dim = index.d
            print(f"Tables FAISS actual dimensions: {tables_dim}")
            
            if tables_dim == 3072:
                print("✅ Tables FAISS is OpenAI (3072)")
        except Exception as e:
            print(f"❌ Failed to read tables FAISS: {e}")
    
    print(f"\n🎯 Conclusion:")
    if 'actual_dim' in locals() and actual_dim == 3072:
        print("✅ Variables FAISS is correct (OpenAI)")
        print("❌ build_info.json is wrong - need to fix detection logic")
    elif 'actual_dim' in locals() and actual_dim == 768:
        print("❌ Variables FAISS needs to be rebuilt with OpenAI")
    else:
        print("❓ Unable to determine variables FAISS dimensions")

if __name__ == "__main__":
    check_actual_faiss_dimensions()
