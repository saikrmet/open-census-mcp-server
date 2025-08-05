#!/usr/bin/env python3
"""Find FAISS files specifically - ignore the cache files"""

import time
from pathlib import Path
import faiss

def find_faiss_files_only():
    """Find only .faiss files and check their timestamps"""
    
    print("🔍 Searching for .faiss files ONLY")
    print("=" * 40)
    
    project_root = Path("/Users/brock/Documents/GitHub/census-mcp-server")
    current_time = time.time()
    
    faiss_files = list(project_root.rglob("*.faiss"))
    
    if not faiss_files:
        print("❌ No .faiss files found")
        return
    
    print(f"Found {len(faiss_files)} FAISS files:\n")
    
    for faiss_file in sorted(faiss_files):
        try:
            stat = faiss_file.stat()
            age_hours = (current_time - stat.st_mtime) / 3600
            size_mb = stat.st_size / (1024 * 1024)
            
            # Load FAISS to check dimensions
            index = faiss.read_index(str(faiss_file))
            dimensions = index.d
            count = index.ntotal
            
            embed_type = "OpenAI ✅" if dimensions == 3072 else "SentenceTransformers ❌"
            
            # Age formatting
            if age_hours < 1:
                age_str = f"{age_hours * 60:.0f} minutes ago 🔥"
            elif age_hours < 24:
                age_str = f"{age_hours:.1f} hours ago"
            else:
                age_str = f"{age_hours / 24:.1f} days ago"
            
            print(f"📁 {faiss_file}")
            print(f"   Dimensions: {dimensions} ({embed_type})")
            print(f"   Vectors: {count:,}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Age: {age_str}")
            print()
            
        except Exception as e:
            print(f"❌ Error reading {faiss_file}: {e}")

def check_build_directories():
    """Check common build output directories"""
    
    print("📁 Checking Build Output Directories")
    print("=" * 40)
    
    project_root = Path("/Users/brock/Documents/GitHub/census-mcp-server")
    
    # Common build directories
    dirs_to_check = [
        "knowledge-base/variables-faiss",
        "knowledge-base/variables-db", 
        "knowledge-base/variables",
        "variables-faiss",
        "variables-db",
        "variables"
    ]
    
    for dir_name in dirs_to_check:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name} exists")
            
            # Check for FAISS files in this directory
            faiss_files = list(dir_path.glob("*.faiss"))
            json_files = list(dir_path.glob("*.json"))
            
            if faiss_files:
                for f in faiss_files:
                    stat = f.stat()
                    age_mins = (time.time() - stat.st_mtime) / 60
                    print(f"   📄 {f.name} (modified {age_mins:.0f}m ago)")
            
            if json_files:
                for f in json_files:
                    if f.name in ['build_info.json', 'variables_metadata.json']:
                        stat = f.stat()
                        age_mins = (time.time() - stat.st_mtime) / 60
                        print(f"   📋 {f.name} (modified {age_mins:.0f}m ago)")
        else:
            print(f"❌ {dir_name} does not exist")

if __name__ == "__main__":
    find_faiss_files_only()
    print()
    check_build_directories()
