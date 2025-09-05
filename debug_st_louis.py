#!/usr/bin/env python3
"""
Debug St. Louis entries in the geography database
"""

import sqlite3
from pathlib import Path

def debug_st_louis():
    """Find all St. Louis related entries"""
    db_path = Path("knowledge-base/geo-db/geography.db")
    
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("🔍 Searching for St. Louis entries...\n")
    
    # Search 1: All places with "Louis" in the name
    print("📍 Places containing 'Louis':")
    cursor.execute("""
        SELECT name, state_abbrev, population, place_fips, state_fips
        FROM places 
        WHERE name LIKE '%Louis%'
        ORDER BY state_abbrev, name
    """)
    
    results = cursor.fetchall()
    if results:
        for result in results:
            pop = f"{result['population']:,}" if result['population'] else "No pop data"
            print(f"   {result['name']}, {result['state_abbrev']} - {pop} - FIPS: {result['state_fips']}-{result['place_fips']}")
    else:
        print("   ❌ No places with 'Louis' found")
    
    print("\n📍 Places containing 'St' in Missouri:")
    cursor.execute("""
        SELECT name, state_abbrev, population, place_fips, state_fips
        FROM places 
        WHERE name LIKE '%St%' AND state_abbrev = 'MO'
        ORDER BY name
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    if results:
        for result in results:
            pop = f"{result['population']:,}" if result['population'] else "No pop data"
            print(f"   {result['name']}, {result['state_abbrev']} - {pop} - FIPS: {result['state_fips']}-{result['place_fips']}")
    else:
        print("   ❌ No places with 'St' in Missouri found")
    
    print("\n🏛️ Counties containing 'Louis':")
    cursor.execute("""
        SELECT name, state_abbrev, county_fips, state_fips
        FROM counties 
        WHERE name LIKE '%Louis%'
        ORDER BY state_abbrev, name
    """)
    
    results = cursor.fetchall()
    if results:
        for result in results:
            print(f"   {result['name']}, {result['state_abbrev']} - FIPS: {result['state_fips']}-{result['county_fips']}")
    else:
        print("   ❌ No counties with 'Louis' found")
    
    print("\n🔍 Checking name_variations table:")
    cursor.execute("""
        SELECT canonical_name, variation, geography_type
        FROM name_variations 
        WHERE variation LIKE '%Louis%' OR canonical_name LIKE '%Louis%'
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    if results:
        for result in results:
            print(f"   {result['canonical_name']} ↔ {result['variation']} ({result['geography_type']})")
    else:
        print("   ❌ No name variations with 'Louis' found")
    
    print("\n📊 Database table sizes:")
    tables = ['places', 'counties', 'states', 'cbsas', 'zctas', 'name_variations']
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"   {table}: {count:,} records")
    
    print("\n📝 Sample Missouri places (first 10):")
    cursor.execute("""
        SELECT name, place_fips
        FROM places 
        WHERE state_abbrev = 'MO'
        ORDER BY name
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    for result in results:
        print(f"   {result['name']} - FIPS: {result['place_fips']}")
    
    conn.close()

if __name__ == "__main__":
    debug_st_louis()
