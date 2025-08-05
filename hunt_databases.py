#!/usr/bin/env python3
"""Find ALL the databases and check their actual locations vs expected"""

import time
from pathlib import Path
import json

def find_all_databases():
    """Find every database in the project"""
    
    print("🔍 HUNTING DOWN ALL DATABASES")
    print("=" * 50)
    
    project_root = Path("/Users/brock/Documents/GitHub/census-mcp-server")
    
    # Search for different database types
    database_indicators = {
        "FAISS": "*.faiss",
        "ChromaDB": "chroma.sqlite3", 
        "SQLite": "*.db",
        "Metadata": "*metadata.json",
        "Build Info": "build_info.json"
    }
    
    found_databases = {}
    
    for db_type, pattern in database_indicators.items():
        files = list(project_root.rglob(pattern))
        if files:
            found_databases[db_type] = files
    
    # Show findings
    for db_type, files in found_databases.items():
        print(f"\n📁 {db_type} Files Found:")
        print("-" * 30)
        
        current_time = time.time()
        
        for file_path in sorted(files):
            try:
                stat = file_path.stat()
                age_hours = (current_time - stat.st_mtime) / 3600
                size_mb = stat.st_size / (1024 * 1024)
                
                if age_hours < 1:
                    age_str = f"{age_hours * 60:.0f}m ago 🔥"
                elif age_hours < 24:
                    age_str = f"{age_hours:.1f}h ago"
                else:
                    age_str = f"{age_hours / 24:.1f}d ago"
                
                rel_path = file_path.relative_to(project_root)
                print(f"   {rel_path}")
                print(f"     Size: {size_mb:.1f}MB, Modified: {age_str}")
                
                # Special handling for build info
                if file_path.name == "build_info.json":
                    try:
                        with open(file_path) as f:
                            build_info = json.load(f)
                        
                        model = build_info.get('embedding_model', 'unknown')
                        dims = build_info.get('embedding_dimension', 'unknown')
                        count = build_info.get('variable_count', build_info.get('chunk_count', 'unknown'))
                        
                        print(f"     📊 Model: {model}, Dims: {dims}, Count: {count}")
                    except:
                        pass
                
                print()
                
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")

def check_expected_vs_actual():
    """Check where kb_search expects files vs where they actually are"""
    
    print("🎯 EXPECTED vs ACTUAL LOCATIONS")
    print("=" * 50)
    
    project_root = Path("/Users/brock/Documents/GitHub/census-mcp-server")
    
    # What kb_search.py expects
    expected_locations = {
        "Variables DB": "knowledge-base/variables-db/",
        "Methodology DB": "knowledge-base/methodology-db/", 
        "Geography DB": "knowledge-base/geo-db/",
        "Table Catalog": "knowledge-base/table-catalog/"
    }
    
    print("Expected locations:")
    for name, location in expected_locations.items():
        full_path = project_root / location
        if full_path.exists():
            print(f"✅ {name}: {location}")
            
            # List key files in directory
            key_files = list(full_path.glob("*.faiss")) + list(full_path.glob("*.db")) + list(full_path.glob("chroma.sqlite3"))
            for f in key_files[:3]:  # Show first 3
                stat = f.stat()
                size_mb = stat.st_size / (1024 * 1024)
                age_hours = (time.time() - stat.st_mtime) / 3600
                age_str = f"{age_hours:.1f}h ago" if age_hours < 48 else f"{age_hours/24:.1f}d ago"
                print(f"     📄 {f.name} ({size_mb:.1f}MB, {age_str})")
        else:
            print(f"❌ {name}: {location} - NOT FOUND")

def find_methodology_databases():
    """Specifically hunt down methodology databases"""
    
    print("\n🔍 METHODOLOGY DATABASE HUNT")
    print("=" * 40)
    
    project_root = Path("/Users/brock/Documents/GitHub/census-mcp-server")
    
    # Look for ChromaDB collections (methodology uses ChromaDB)
    chroma_files = list(project_root.rglob("chroma.sqlite3"))
    
    print(f"Found {len(chroma_files)} ChromaDB files:")
    
    current_time = time.time()
    
    for chroma_file in sorted(chroma_files):
        try:
            stat = chroma_file.stat()
            age_hours = (current_time - stat.st_mtime) / 3600
            size_mb = stat.st_size / (1024 * 1024)
            
            age_str = f"{age_hours:.1f}h ago" if age_hours < 48 else f"{age_hours/24:.1f}d ago"
            
            rel_path = chroma_file.relative_to(project_root)
            print(f"\n📄 {rel_path}")
            print(f"   Size: {size_mb:.1f}MB, Modified: {age_str}")
            
            # Check if this directory has a collection
            parent_dir = chroma_file.parent
            collection_files = list(parent_dir.glob("*"))
            print(f"   Directory contains {len(collection_files)} files")
            
            # Look for collection names
            for f in collection_files:
                if f.is_file() and f.name not in ['chroma.sqlite3']:
                    print(f"     - {f.name}")
            
        except Exception as e:
            print(f"❌ Error reading {chroma_file}: {e}")

if __name__ == "__main__":
    find_all_databases()
    print("\n" + "="*70 + "\n")
    check_expected_vs_actual()
    find_methodology_databases()
