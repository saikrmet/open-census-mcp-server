#!/usr/bin/env python3
"""
Quick Test - Pacific Northwest Geographic Resolution
Tests exactly what failed in your test report using known good data
"""

import sys
import asyncio
import os
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Set environment if .env file exists
env_file = current_dir / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Loaded environment from {env_file}")
    except ImportError:
        print("⚠️ python-dotenv not installed, using system environment")

async def test_basic_functionality():
    """Test the exact scenarios from your failed test report"""
    
    print("🧪 TESTING BASIC CENSUS MCP FUNCTIONALITY")
    print("="*60)
    print("Testing the exact Pacific Northwest scenarios that failed")
    print()
    
    # Test 1: Function naming issue (Priority 1)
    print("🔧 TEST 1: MCP Server Function Import")
    try:
        from census_mcp_server import call_tool
        print("✅ MCP server imported successfully")
        
        # Test the statistical consultation function (the one that was broken)
        test_args = {"query": "What is median household income?", "location": ""}
        try:
            # This call should no longer fail with NameError
            result = await call_tool("get_statistical_consultation", test_args)
            print("✅ Statistical consultation function call works")
            if result and result[0].text:
                print(f"📋 Got response: {len(result[0].text)} characters")
        except NameError as e:
            print(f"❌ STILL BROKEN - NameError: {e}")
        except Exception as e:
            print(f"⚠️ Other error (may be normal): {e}")
    except Exception as e:
        print(f"❌ Import failed: {e}")
    
    print()
    
    # Test 2: Geographic Resolution (Priority 2)  
    print("🌍 TEST 2: Geographic Resolution for Known Locations")
    
    # Test with your exact Pacific Northwest examples
    test_locations = [
        ("Washington", "Should resolve to state FIPS 53"),
        ("Oregon", "Should resolve to state FIPS 41"),
        ("Seattle, WA", "Should resolve to place FIPS 53:63000"),
        ("Portland, OR", "Should resolve to place FIPS 41:59000")
    ]
    
    try:
        from data_retrieval.python_census_api import PythonCensusAPI
        api = PythonCensusAPI()
        print("✅ Census API initialized")
        
        for location, expected in test_locations:
            try:
                if hasattr(api, 'geo_handler'):
                    geo_result = api.geo_handler.resolve_location(location)
                    if 'error' in geo_result:
                        print(f"❌ {location}: {geo_result['error']}")
                    else:
                        geo_type = geo_result.get('geography_type', 'unknown')
                        method = geo_result.get('resolution_method', 'unknown')
                        print(f"✅ {location}: {geo_type} via {method}")
                else:
                    print(f"❌ {location}: No geographic handler")
            except Exception as e:
                print(f"❌ {location}: Exception - {e}")
    except Exception as e:
        print(f"❌ Census API initialization failed: {e}")
    
    print()
    
    # Test 3: Actual Census API Calls (The real test)
    print("📊 TEST 3: Actual Census API Calls")
    print("Testing with known valid FIPS codes and variables")
    
    # Test cases from your report - states should definitely work
    test_cases = [
        ("Washington", ["B01003_001E"], "WA population"),
        ("Oregon", ["B01003_001E"], "OR population"),  
        ("Seattle, WA", ["B01003_001E"], "Seattle population"),
    ]
    
    try:
        api = PythonCensusAPI()
        
        for location, variables, description in test_cases:
            try:
                print(f"🔍 Testing: {description}")
                result = await api.get_demographic_data(location, variables, year=2023, survey="acs5")
                
                if 'error' in result:
                    step = result.get('step_failed', 'unknown')
                    print(f"❌ Failed at {step}: {result['error']}")
                    
                    # Show what did work
                    if 'location_resolved' in result:
                        print(f"   ✅ Geography resolved: {result['location_resolved']}")
                    if 'variables_validated' in result:
                        print(f"   ✅ Variables: {result['variables_validated']}")
                else:
                    # Success!
                    resolved_name = result['resolved_location']['name']
                    data = result.get('data', {})
                    if 'B01003_001E' in data and data['B01003_001E'].get('estimate'):
                        pop = data['B01003_001E']['estimate']
                        print(f"✅ {resolved_name}: Population = {pop:,.0f}")
                    else:
                        print(f"✅ {resolved_name}: API call succeeded but no population data")
                
            except Exception as e:
                print(f"❌ {description}: System error - {e}")
            
            print()
    
    except Exception as e:
        print(f"❌ Could not initialize API for testing: {e}")

async def test_database_paths():
    """Check if the geographic database exists where expected"""
    print("🗄️  DATABASE PATH CHECK")
    print("="*30)
    
    possible_db_paths = [
        current_dir / "knowledge-base" / "geo-db" / "geography.db",
        current_dir / "knowledge-base" / "geography.db",
    ]
    
    for path in possible_db_paths:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ Found: {path} ({size_mb:.1f} MB)")
            
            # Try to connect
            try:
                import sqlite3
                conn = sqlite3.connect(str(path))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                print(f"   Tables: {', '.join(tables[:5])}")
                
                if 'places' in tables:
                    cursor.execute("SELECT COUNT(*) FROM places")
                    place_count = cursor.fetchone()[0]
                    print(f"   Places: {place_count:,}")
                
                conn.close()
                print("   Database connection: OK")
            except Exception as e:
                print(f"   Database connection: FAILED - {e}")
            
            return True
        else:
            print(f"❌ Not found: {path}")
    
    return False

if __name__ == "__main__":
    print("🚀 QUICK TEST - Pacific Northwest Census Data")
    print("Testing the exact failures from your test report")
    print()
    
    # Check database first
    db_exists = asyncio.run(test_database_paths())
    print()
    
    if not db_exists:
        print("⚠️ WARNING: No geographic database found!")
        print("Geographic resolution will be severely limited")
        print()
    
    # Run functionality tests
    asyncio.run(test_basic_functionality())
    
    print()
    print("🎯 SUMMARY")
    print("If you see ✅ for all tests, the main issues are fixed!")
    print("If you see ❌, we need to debug further.")
