#!/usr/bin/env python3
"""
Debug script to find where the table catalog files actually are
and check if B17001 poverty tables exist
"""

import os
import json
from pathlib import Path

def find_files():
    """Find all table catalog files in the project"""
    print("🔍 Searching for table catalog files...")
    
    # Search from current directory up
    current_dir = Path.cwd()
    search_paths = [
        current_dir,
        current_dir.parent,
        current_dir / "table-catalog",
        current_dir.parent / "table-catalog",
        current_dir / "knowledge-base" / "table-catalog",
        current_dir.parent / "knowledge-base" / "table-catalog"
    ]
    
    found_files = []
    
    for search_path in search_paths:
        if search_path.exists():
            print(f"Checking: {search_path}")
            
            # Look for table_catalog.json
            catalog_file = search_path / "table_catalog.json"
            if catalog_file.exists():
                found_files.append(("table_catalog.json", catalog_file))
                print(f"  ✅ Found: {catalog_file}")
            
            # Look for any JSON files with "catalog" or "table" in name
            if search_path.is_dir():
                for file in search_path.glob("*catalog*.json"):
                    found_files.append(("catalog file", file))
                    print(f"  📁 Found: {file}")
                    
                for file in search_path.glob("*table*.json"):
                    found_files.append(("table file", file))
                    print(f"  📁 Found: {file}")
    
    print(f"\n📊 Total files found: {len(found_files)}")
    return found_files

def check_poverty_tables(catalog_file):
    """Check if poverty tables exist in a catalog file"""
    print(f"\n🔍 Checking poverty tables in: {catalog_file}")
    
    try:
        with open(catalog_file, 'r') as f:
            catalog_data = json.load(f)
        
        if 'tables' not in catalog_data:
            print("❌ No 'tables' key found")
            return False
        
        tables = catalog_data['tables']
        print(f"📊 Total tables: {len(tables)}")
        
        # Look for poverty tables
        poverty_tables = []
        b17_tables = []
        
        for table in tables:
            table_id = table.get('table_id', '')
            title = table.get('title', '').lower()
            universe = table.get('universe', '').lower()
            
            # Check for B17 tables
            if table_id.startswith('B17'):
                b17_tables.append(table)
            
            # Check for poverty in title or universe
            if 'poverty' in title or 'poverty' in universe:
                poverty_tables.append(table)
        
        print(f"🏷️  B17* tables found: {len(b17_tables)}")
        print(f"🏷️  Poverty-related tables: {len(poverty_tables)}")
        
        # Check specifically for B17001
        b17001 = None
        for table in tables:
            if table.get('table_id') == 'B17001':
                b17001 = table
                break
        
        if b17001:
            print(f"\n✅ B17001 FOUND!")
            print(f"   Title: {b17001.get('title', 'N/A')}")
            print(f"   Universe: {b17001.get('universe', 'N/A')}")
            print(f"   Survey programs: {b17001.get('survey_programs', [])}")
            return True
        else:
            print(f"\n❌ B17001 NOT FOUND")
            
            # Show first few B17 tables we do have
            if b17_tables:
                print(f"\nFirst 5 B17* tables we do have:")
                for table in b17_tables[:5]:
                    print(f"   {table.get('table_id')}: {table.get('title', 'N/A')[:60]}...")
            
            return False
    
    except Exception as e:
        print(f"❌ Error reading catalog: {e}")
        return False

def check_directory_structure():
    """Check the current working directory and structure"""
    print("📁 Current directory structure:")
    current_dir = Path.cwd()
    print(f"   Current: {current_dir}")
    
    # List contents of current directory
    print(f"\n📂 Contents of {current_dir}:")
    try:
        for item in sorted(current_dir.iterdir()):
            if item.is_dir():
                print(f"   📁 {item.name}/")
            else:
                print(f"   📄 {item.name}")
    except Exception as e:
        print(f"   ❌ Error listing directory: {e}")

def main():
    """Main debug function"""
    print("🐛 DEBUG: Table Catalog Location and Poverty Tables")
    print("=" * 60)
    
    # Check directory structure
    check_directory_structure()
    
    # Find catalog files
    found_files = find_files()
    
    if not found_files:
        print("\n❌ No table catalog files found!")
        print("You may need to run the table catalog builder first.")
        return
    
    # Check each catalog file for poverty tables
    poverty_found = False
    for file_type, file_path in found_files:
        if "catalog" in str(file_path).lower():
            if check_poverty_tables(file_path):
                poverty_found = True
    
    if not poverty_found:
        print("\n🚨 PROBLEM IDENTIFIED:")
        print("   B17001 poverty table not found in any catalog")
        print("   This explains why 'poverty rate' searches return 0 results")
        print("\n💡 SOLUTION:")
        print("   1. Check if your table catalog builder includes B17001")
        print("   2. Rebuild the table catalog if necessary")
        print("   3. Verify source data includes poverty tables")

if __name__ == "__main__":
    main()
