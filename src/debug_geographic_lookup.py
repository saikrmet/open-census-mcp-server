#!/usr/bin/env python3
"""
Debug the geographic lookup process step by step
"""

import sqlite3
from pathlib import Path
from data_retrieval.geographic_handler import GeographicHandler

def debug_lookup_process():
    """Debug each step of the lookup process"""
    
    # Test case
    location = "Brainerd, MN"
    
    print(f"🔍 Debugging: {location}")
    
    # Step 1: Test the parsing
    gh = GeographicHandler()
    parsed = gh._parse_location_components(location)
    print(f"1. Parsed components: {parsed}")
    
    if not parsed:
        print("❌ Parsing failed!")
        return
    
    city_name = parsed['city']
    state_abbrev = parsed['state']
    print(f"2. City: '{city_name}', State: '{state_abbrev}'")
    
    # Step 2: Test hot cache
    location_key = location.lower()
    if location_key in gh.hot_cache:
        print(f"3. ✅ Hot cache hit: {gh.hot_cache[location_key]}")
        return
    else:
        print(f"3. ❌ Hot cache miss (key: '{location_key}')")
    
    # Step 3: Test exact lookup with debug
    print("4. Testing exact database lookup...")
    result = test_exact_lookup_debug(gh.conn, city_name, state_abbrev)
    if result:
        print(f"4. ✅ Exact lookup worked: {result}")
        return
    else:
        print("4. ❌ Exact lookup failed")
    
    # Step 4: Test fuzzy lookup
    print("5. Testing fuzzy lookup...")
    result = gh._fuzzy_database_lookup(city_name, state_abbrev)
    if result:
        print(f"5. ✅ Fuzzy lookup worked: {result}")
        return
    else:
        print("5. ❌ Fuzzy lookup failed")
    
    # Step 5: Check what happens in state fallback
    print("6. Testing state fallback...")
    result = gh._resolve_state_only(state_abbrev)
    print(f"6. State fallback result: {result}")

def test_exact_lookup_debug(conn, city_name, state_abbrev):
    """Test the exact lookup with debug output"""
    cursor = conn.cursor()
    
    print(f"   Trying exact match: '{city_name}' in '{state_abbrev}'")
    cursor.execute("""
        SELECT place_fips, state_fips, state_abbrev, name
        FROM places 
        WHERE name_lower = LOWER(?) AND state_abbrev = ?
        LIMIT 1
    """, (city_name, state_abbrev))
    
    row = cursor.fetchone()
    if row:
        print(f"   ✅ Exact match found: {row['name']}")
        return {'name': row['name'], 'place_fips': row['place_fips']}
    else:
        print(f"   ❌ No exact match")
    
    # Try with place types
    place_types = ['city', 'town', 'village', 'borough', 'township']
    for place_type in place_types:
        test_name = f"{city_name} {place_type}"
        print(f"   Trying with place type: '{test_name}'")
        
        cursor.execute("""
            SELECT place_fips, state_fips, state_abbrev, name
            FROM places 
            WHERE name_lower = LOWER(?) AND state_abbrev = ?
            LIMIT 1
        """, (test_name, state_abbrev))
        
        row = cursor.fetchone()
        if row:
            print(f"   ✅ Place type match found: {row['name']}")
            return {'name': row['name'], 'place_fips': row['place_fips']}
        else:
            print(f"   ❌ No match for '{test_name}'")
    
    return None

if __name__ == "__main__":
    debug_lookup_process()
