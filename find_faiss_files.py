#!/usr/bin/env python3
"""Find all FAISS files and check their dimensions + timestamps"""

import os
import time
from pathlib import Path
import faiss

def find_all_faiss_files():
    """Find all .faiss files in the project and check their dimensions"""
    
    print("🔍 Searching for ALL .faiss files")
    print("=" * 40)
    
    project_root = Path("/Users/brock/Documents/GitHub/census-mcp-server")
    faiss_files = []
    
    # Search recursively for .faiss files
    for faiss_file in project_root.rglob("*.faiss"):
        faiss_files.append(faiss_file)
    
    if not faiss_files:
        print("❌ No .faiss files found in project")
        return
    
    print(f"Found {len(faiss_files)} FAISS files:\n")
    
    current_time = time.time()
    
    for faiss_file in sorted(faiss_files):
        try:
            # Get file info
            stat = faiss_file.stat()
            size_mb = stat.st_size / (1024 * 1024)
            modified_time = stat.st_mtime
            age_hours = (current_time - modified_time) / 3600
            
            # Load FAISS index to check dimensions
            index = faiss.read_index(str(faiss_file))
            dimensions = index.d
            count = index.ntotal
            
            # Determine embedding type
            if dimensions == 3072:
                embed_type = "OpenAI ✅"
            elif dimensions == 768:
                embed_type = "SentenceTransformers ❌"
            else:
                embed_type = f"Unknown ({dimensions})"
            
            # Show age
            if age_hours < 1:
                age_str = f"{age_hours * 60:.0f}m ago 🔥"
            elif age_hours < 24:
                age_str = f"{age_hours:.1f}h ago"
            else:
                age_str = f"{age_hours / 24:.1f}d ago"
            
            print(f"📁 {faiss_file.relative_to(project_root)}")
            print(f"   Dimensions: {dimensions} ({embed_type})")
            print(f"   Vectors: {count:,}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Modified: {age_str}")
            print()
            
        except Exception as e:
            print(f"❌ Error reading {faiss_file}: {e}")
            print()

def check_expected_locations():
    """Check where kb_search.py expects to find files"""
    
    print("🎯 Expected File Locations")
    print("=" * 30)
    
    expected_locations = [
        "knowledge-base/variables-faiss/variables.faiss",
        "knowledge-base/table-catalog/table_embeddings_enhanced.faiss",
        "knowledge-base/table-catalog/table_embeddings.faiss"
    ]
    
    project_root = Path("/Users/brock/Documents/GitHub/census-mcp-server")
    
    for location in expected_locations:
        full_path = project_root / location
        if full_path.exists():
            try:
                index = faiss.read_index(str(full_path))
                dimensions = index.d
                embed_type = "OpenAI ✅" if dimensions == 3072 else "SentenceTransformers ❌"
                print(f"✅ {location}: {dimensions} dims ({embed_type})")
            except Exception as e:
                print(f"❌ {location}: Error reading - {e}")
        else:
            print(f"❌ {location}: Not found")

def find_recent_build_outputs():
    """Find any files created in the last few hours"""
    
    print(f"\n⏰ Recent Build Outputs (last 2 hours)")
    print("=" * 40)
    
    project_root = Path("/Users/brock/Documents/GitHub/census-mcp-server")
    current_time = time.time()
    recent_files = []
    
    # Look for recent files
    for file_path in project_root.rglob("*"):
        if file_path.is_file():
            try:
                stat = file_path.stat()
                age_hours = (current_time - stat.st_mtime) / 3600
                
                # Show files created/modified in last 2 hours
                if age_hours < 2:
                    size_mb = stat.st_size / (1024 * 1024)
                    recent_files.append((file_path, age_hours, size_mb))
            except:
                continue
    
    if recent_files:
        recent_files.sort(key=lambda x: x[1])  # Sort by age
        
        for file_path, age_hours, size_mb in recent_files:
            age_str = f"{age_hours * 60:.0f}m ago"
            print(f"📄 {file_path.relative_to(project_root)}")
            print(f"   Size: {size_mb:.1f} MB, Modified: {age_str}")
            
            # Check if it's a FAISS file
            if file_path.suffix == '.faiss':
                try:
                    index = faiss.read_index(str(file_path))
                    dimensions = index.d
                    embed_type = "OpenAI ✅" if dimensions == 3072 else "SentenceTransformers ❌"
                    print(f"   🎯 FAISS: {dimensions} dims ({embed_type})")
                except:
                    pass
            print()
    else:
        print("No files modified in the last 2 hours")

if __name__ == "__main__":
    find_all_faiss_files()
    check_expected_locations()
    find_recent_build_outputs()
