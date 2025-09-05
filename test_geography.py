#!/usr/bin/env python3
"""
Test Geography Database - Command Line Script

Usage: python test_geography.py [location]
Example: python test_geography.py "St. Louis"
"""

import sys
import sqlite3
from pathlib import Path

def test_database_exists():
    """Check if geography database exists and basic structure"""
    db_paths = [
        Path("knowledge-base/geo-db/geography.db"),
        Path("knowledge-base/geography.db"),
        Path("../knowledge-base/geo-db/geography.db"),
        Path("../knowledge-base/geography.db"),
        Path("geography.db")
    ]
    
    for db_path in db_paths:
        if db_path.exists():
            print(f"✅ Found database: {db_path}")
            return str(db_path)
    
    print("❌ No geography database found")
    return None

def test_database_structure(db_path):
    """Test basic database structure"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📊 Tables found: {tables}")
        
        # Count records in key tables
        for table in ['places', 'counties', 'states', 'cbsas']:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   {table}: {count:,} records")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database structure test failed: {e}")
        return False

def test_basic_queries(db_path):
    """Test basic geographic queries"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Test 1: Find major cities
        print("\n🏙️ Testing major cities...")
        test_cities = ["New York", "Los Angeles", "Chicago", "Houston"]
        
        for city in test_cities:
            cursor.execute("""
                SELECT name, state_abbrev, population
                FROM places 
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY population DESC NULLS LAST
                LIMIT 1
            """, (f"%{city}%",))
            
            result = cursor.fetchone()
            if result:
                pop = f"{result['population']:,}" if result['population'] else "Unknown"
                print(f"   ✅ {result['name']}, {result['state_abbrev']} - Pop: {pop}")
            else:
                print(f"   ❌ {city} not found")
        
        # Test 2: St. Louis specific searches
        print("\n🔍 Testing St. Louis variations...")
        st_louis_variants = [
            "St. Louis",
            "Saint Louis",
            "St Louis",
            "St. Louis city"
        ]
        
        for variant in st_louis_variants:
            cursor.execute("""
                SELECT name, state_abbrev, population, place_fips, state_fips
                FROM places 
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY population DESC NULLS LAST
                LIMIT 3
            """, (f"%{variant.replace('.', '')}%",))
            
            results = cursor.fetchall()
            if results:
                print(f"   '{variant}' found {len(results)} matches:")
                for result in results:
                    pop = f"{result['population']:,}" if result['population'] else "Unknown"
                    fips = f"{result['state_fips']}-{result['place_fips']}"
                    print(f"      {result['name']}, {result['state_abbrev']} - Pop: {pop} - FIPS: {fips}")
            else:
                print(f"   ❌ '{variant}' not found")
        
        # Test 3: Missouri places
        print("\n🏛️ Testing Missouri places...")
        cursor.execute("""
            SELECT name, population, place_fips
            FROM places 
            WHERE state_abbrev = 'MO'
            ORDER BY population DESC NULLS LAST
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        if results:
            print("   Top Missouri places:")
            for result in results:
                pop = f"{result['population']:,}" if result['population'] else "Unknown"
                print(f"      {result['name']} - Pop: {pop} - FIPS: {result['place_fips']}")
        else:
            print("   ❌ No Missouri places found")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        return False

def test_geographic_handler(db_path, location):
    """Test the actual GeographicHandler class"""
    try:
        # Add src directory to path for imports
        sys.path.append("src")
        sys.path.append("src/data_retrieval")
        
        from data_retrieval.geographic_handler import GeographicHandler
        
        print(f"\n🧭 Testing GeographicHandler with '{location}'...")
        
        handler = GeographicHandler(db_path)
        results = handler.search_locations(location, max_results=5)
        
        if results:
            print(f"   ✅ Found {len(results)} matches:")
            for i, result in enumerate(results, 1):
                fips_parts = []
                if result.get('state_fips'):
                    fips_parts.append(f"state:{result['state_fips']}")
                if result.get('place_fips'):
                    fips_parts.append(f"place:{result['place_fips']}")
                fips_str = ", ".join(fips_parts) if fips_parts else "No FIPS"
                
                print(f"      {i}. {result['name']} ({result['geography_type']})")
                print(f"         Confidence: {result['confidence']:.3f}")
                if result.get('population'):
                    print(f"         Population: {result['population']:,}")
                print(f"         FIPS: {fips_str}")
        else:
            print(f"   ❌ No matches found for '{location}'")
            
        return len(results) > 0
        
    except ImportError as e:
        print(f"❌ Could not import GeographicHandler: {e}")
        return False
    except Exception as e:
        print(f"❌ GeographicHandler test failed: {e}")
        return False

def main():
    """Main test function"""
    location = sys.argv[1] if len(sys.argv) > 1 else "St. Louis"
    
    print(f"🧪 Testing Geography Database with location: '{location}'\n")
    
    # Test 1: Database exists
    db_path = test_database_exists()
    if not db_path:
        sys.exit(1)
    
    # Test 2: Database structure
    if not test_database_structure(db_path):
        sys.exit(1)
    
    # Test 3: Basic queries
    if not test_basic_queries(db_path):
        sys.exit(1)
    
    # Test 4: GeographicHandler class
    test_geographic_handler(db_path, location)
    
    print(f"\n🎯 Test complete for '{location}'")

if __name__ == "__main__":
    main()
