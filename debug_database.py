#!/usr/bin/env python3
"""
Database Debug - Check Geographic Database Contents
Direct database queries to see why Seattle/Portland resolution fails
"""

import sys
import sqlite3
from pathlib import Path

def debug_database():
    """Debug the geographic database directly"""
    
    current_dir = Path(__file__).parent
    
    # Find the database
    possible_paths = [
        current_dir / "knowledge-base" / "geo-db" / "geography.db",
        current_dir / "knowledge-base" / "geography.db",
    ]
    
    db_path = None
    for path in possible_paths:
        if path.exists():
            db_path = path
            break
    
    if not db_path:
        print("❌ No geographic database found!")
        return
    
    print(f"🗄️ Database: {db_path}")
    print(f"Size: {db_path.stat().st_size / (1024*1024):.1f} MB")
    print()
    
    # Connect and inspect
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check tables
    print("📋 TABLES:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count:,} rows")
    print()
    
    # Check if places table exists
    if 'places' not in tables:
        print("❌ CRITICAL: 'places' table missing!")
        conn.close()
        return
    
    # Check places table schema
    print("🏗️ PLACES TABLE SCHEMA:")
    cursor.execute("PRAGMA table_info(places)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    column_names = [col[1] for col in columns]
    print()
    
    # Look for Seattle specifically
    print("🔍 SEARCHING FOR SEATTLE:")
    
    # Try different variations
    search_queries = [
        ("Exact match", "SELECT * FROM places WHERE name = 'Seattle' AND state_abbrev = 'WA' LIMIT 1"),
        ("Case insensitive", "SELECT * FROM places WHERE LOWER(name) = 'seattle' AND state_abbrev = 'WA' LIMIT 1"),
        ("name_lower column", "SELECT * FROM places WHERE name_lower = 'seattle' AND state_abbrev = 'WA' LIMIT 1"),
        ("Partial match", "SELECT * FROM places WHERE name LIKE '%Seattle%' AND state_abbrev = 'WA' LIMIT 3"),
        ("Any WA city with 'sea'", "SELECT * FROM places WHERE name LIKE '%sea%' AND state_abbrev = 'WA' LIMIT 5")
    ]
    
    for desc, query in search_queries:
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            print(f"  {desc}: {len(results)} results")
            for result in results[:2]:  # Show first 2 results
                print(f"    {result}")
        except Exception as e:
            print(f"  {desc}: ERROR - {e}")
    
    print()
    
    # Check Washington state cities
    print("🌲 WASHINGTON STATE CITIES (sample):")
    try:
        cursor.execute("SELECT name, place_fips, population FROM places WHERE state_abbrev = 'WA' ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC LIMIT 10")
        results = cursor.fetchall()
        for result in results:
            print(f"  {result[0]} (FIPS: {result[1]}, Pop: {result[2]})")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print()
    
    # Check hot cache issue - are the columns different?
    print("🔥 HOT CACHE vs LIVE QUERY COMPARISON:")
    print("Hot cache worked for states but failed for places...")
    
    # Check if funcstat filtering was the issue
    if 'funcstat' in column_names:
        print("  ⚠️ funcstat column exists - checking values:")
        cursor.execute("SELECT DISTINCT funcstat FROM places WHERE state_abbrev = 'WA' LIMIT 10")
        funcstat_values = cursor.fetchall()
        print(f"    Funcstat values: {funcstat_values}")
    else:
        print("  ✅ No funcstat column (good - code was fixed)")
    
    conn.close()
    print()
    print("🎯 ANALYSIS:")
    print("If Seattle shows up in the database queries above, the issue is in the")
    print("CompleteGeographicHandler._place_state_lookup() method logic.")
    print("If Seattle is NOT in the database, we have a data completeness issue.")

if __name__ == "__main__":
    debug_database()
