#!/usr/bin/env python3
"""
Complete Census MCP Diagnostic - Pacific Northwest Test Case
Tests the exact issues from your test report using known FIPS codes
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiagnosticResults:
    def __init__(self):
        self.tests = []
        self.failures = []
        self.successes = []
    
    def add_test(self, name, success, details):
        self.tests.append({"name": name, "success": success, "details": details})
        if success:
            self.successes.append(name)
        else:
            self.failures.append(name)
    
    def report(self):
        print("\n" + "="*60)
        print("DIAGNOSTIC REPORT")
        print("="*60)
        print(f"Total tests: {len(self.tests)}")
        print(f"Successes: {len(self.successes)}")
        print(f"Failures: {len(self.failures)}")
        
        if self.failures:
            print(f"\n❌ FAILURES ({len(self.failures)}):")
            for failure in self.failures:
                print(f"  - {failure}")
        
        if self.successes:
            print(f"\n✅ SUCCESSES ({len(self.successes)}):")
            for success in self.successes:
                print(f"  - {success}")

async def test_mcp_server_function_naming():
    """Test Priority 1: MCP server function naming issue"""
    results = DiagnosticResults()
    
    print("\n🔍 TESTING MCP SERVER FUNCTION NAMING")
    print("="*50)
    
    try:
        from census_mcp_server import server_instance
        results.add_test("MCP Server Import", True, "Server instance loaded")
        
        # Check if LLM advisor exists
        if hasattr(server_instance, 'llm_advisor') and server_instance.llm_advisor:
            results.add_test("LLM Advisor Present", True, "LLM advisor initialized")
            
            # Test the consult method
            if hasattr(server_instance.llm_advisor, 'consult'):
                results.add_test("Consult Method Exists", True, "consult() method found")
                
                # Test a basic consultation
                try:
                    consultation = server_instance.llm_advisor.consult("What is median household income?", "Austin, TX")
                    if consultation and hasattr(consultation, 'confidence'):
                        results.add_test("Basic Consultation", True, f"Confidence: {consultation.confidence:.1%}")
                    else:
                        results.add_test("Basic Consultation", False, "Invalid consultation response")
                except Exception as e:
                    results.add_test("Basic Consultation", False, f"Error: {str(e)}")
            else:
                results.add_test("Consult Method Exists", False, "consult() method missing")
        else:
            results.add_test("LLM Advisor Present", False, "LLM advisor not initialized")
        
        # Test the problematic function name issue
        from census_mcp_server import call_tool
        try:
            # This should reveal the naming issue
            test_args = {"query": "test query", "location": ""}
            result = await call_tool("get_statistical_consultation", test_args)
            results.add_test("Statistical Consultation Tool", True, "Function call succeeded")
        except NameError as e:
            if "_get_claude_consultation" in str(e):
                results.add_test("Function Naming Issue", False, f"FOUND THE BUG: {str(e)}")
            else:
                results.add_test("Statistical Consultation Tool", False, f"NameError: {str(e)}")
        except Exception as e:
            results.add_test("Statistical Consultation Tool", False, f"Other error: {str(e)}")
        
    except ImportError as e:
        results.add_test("MCP Server Import", False, f"Import error: {str(e)}")
    
    return results

async def test_geographic_resolution():
    """Test Priority 2: Geographic resolution for known FIPS codes"""
    results = DiagnosticResults()
    
    print("\n🌍 TESTING GEOGRAPHIC RESOLUTION")
    print("="*50)
    
    # Test with known Pacific Northwest locations and FIPS codes
    test_locations = [
        # States (should work)
        ("Washington", "53", "state"),
        ("WA", "53", "state"),
        
        # Major cities (should work if geographic resolution is functional)
        ("Seattle, WA", ("53", "63000"), "place"),
        ("Portland, OR", ("41", "59000"), "place"),
        ("Boise, ID", ("16", "08830"), "place"),
        ("Anchorage, AK", ("02", "03000"), "place"),
        
        # Counties (test county resolution)
        ("King County, WA", ("53", "033"), "county"),
        ("Multnomah County, OR", ("41", "051"), "county"),
    ]
    
    try:
        from data_retrieval.python_census_api import PythonCensusAPI
        api = PythonCensusAPI()
        results.add_test("Census API Init", True, "API client initialized")
        
        # Test geographic handler directly
        if hasattr(api, 'geo_handler'):
            results.add_test("Geographic Handler Present", True, "Geographic handler loaded")
            
            for location, expected_fips, geo_type in test_locations:
                try:
                    geo_result = api.geo_handler.resolve_location(location)
                    
                    if 'error' in geo_result:
                        results.add_test(f"Resolve '{location}'", False, f"Error: {geo_result['error']}")
                    else:
                        # Check if we got the expected geography type
                        if geo_result.get('geography_type') == geo_type:
                            results.add_test(f"Resolve '{location}'", True, f"Type: {geo_type}, Method: {geo_result.get('resolution_method')}")
                        else:
                            results.add_test(f"Resolve '{location}'", False, f"Expected {geo_type}, got {geo_result.get('geography_type')}")
                
                except Exception as e:
                    results.add_test(f"Resolve '{location}'", False, f"Exception: {str(e)}")
        else:
            results.add_test("Geographic Handler Present", False, "No geographic handler")
    
    except Exception as e:
        results.add_test("Census API Init", False, f"Error: {str(e)}")
    
    return results

async def test_census_api_calls():
    """Test actual Census API calls with known FIPS codes"""
    results = DiagnosticResults()
    
    print("\n📊 TESTING CENSUS API CALLS")
    print("="*50)
    
    # Test with state-level data (should definitely work)
    test_cases = [
        ("Washington", ["B01003_001E"], "Population for Washington state"),
        ("Oregon", ["B01003_001E"], "Population for Oregon state"),
        ("Seattle, WA", ["B01003_001E"], "Population for Seattle"),
        ("Austin, TX", ["B01003_001E"], "Population for Austin (control case)"),
    ]
    
    try:
        from data_retrieval.python_census_api import PythonCensusAPI
        api = PythonCensusAPI()
        
        for location, variables, description in test_cases:
            try:
                result = await api.get_demographic_data(location, variables, year=2023, survey="acs5")
                
                if 'error' in result:
                    step_failed = result.get('step_failed', 'unknown')
                    results.add_test(f"API Call: {description}", False, f"Failed at {step_failed}: {result['error']}")
                else:
                    data = result.get('data', {})
                    if 'B01003_001E' in data and data['B01003_001E'].get('estimate'):
                        population = data['B01003_001E']['estimate']
                        results.add_test(f"API Call: {description}", True, f"Population: {population:,.0f}")
                    else:
                        results.add_test(f"API Call: {description}", False, "No population data returned")
            
            except Exception as e:
                results.add_test(f"API Call: {description}", False, f"Exception: {str(e)}")
    
    except Exception as e:
        results.add_test("Census API Setup", False, f"Setup error: {str(e)}")
    
    return results

async def test_database_connectivity():
    """Test database connections and schema"""
    results = DiagnosticResults()
    
    print("\n🗄️  TESTING DATABASE CONNECTIVITY")
    print("="*50)
    
    # Test gazetteer database
    try:
        from data_retrieval.geographic_handler import CompleteGeographicHandler
        
        # Try to find the database
        possible_paths = [
            current_dir / "knowledge-base" / "geo-db" / "geography.db",
            current_dir / "knowledge-base" / "geography.db",
        ]
        
        db_found = False
        for path in possible_paths:
            if path.exists():
                results.add_test("Gazetteer Database Found", True, f"Path: {path}")
                db_found = True
                
                # Test database connection
                try:
                    geo_handler = CompleteGeographicHandler(path)
                    results.add_test("Database Connection", True, "Connected successfully")
                    
                    # Test coverage stats
                    if hasattr(geo_handler, 'coverage_stats'):
                        stats = geo_handler.coverage_stats
                        results.add_test("Database Coverage", True, f"Places: {stats.get('places', 0):,}, Counties: {stats.get('counties', 0):,}")
                    
                    # Test a simple query
                    try:
                        test_result = geo_handler.resolve_location("United States")
                        if test_result.get('geography_type') == 'us':
                            results.add_test("Basic Query Test", True, "US resolution works")
                        else:
                            results.add_test("Basic Query Test", False, f"Unexpected result: {test_result}")
                    except Exception as e:
                        results.add_test("Basic Query Test", False, f"Query error: {str(e)}")
                    
                    geo_handler.close()
                    break
                    
                except Exception as e:
                    results.add_test("Database Connection", False, f"Connection error: {str(e)}")
        
        if not db_found:
            results.add_test("Gazetteer Database Found", False, "Database not found in expected locations")
    
    except ImportError as e:
        results.add_test("Geographic Handler Import", False, f"Import error: {str(e)}")
    
    return results

async def main():
    """Run complete diagnostic"""
    
    print("🚀 STARTING COMPLETE CENSUS MCP DIAGNOSTIC")
    print("Testing the exact issues from your Pacific Northwest test report")
    print("Using known FIPS codes to isolate problems")
    
    # Run all diagnostic tests
    test_results = []
    
    # Priority 1: Function naming issue
    mcp_results = await test_mcp_server_function_naming()
    test_results.append(("MCP Server Function Naming", mcp_results))
    
    # Priority 2: Geographic resolution
    geo_results = await test_geographic_resolution()
    test_results.append(("Geographic Resolution", geo_results))
    
    # Test actual API calls
    api_results = await test_census_api_calls()
    test_results.append(("Census API Calls", api_results))
    
    # Test database connectivity
    db_results = await test_database_connectivity()
    test_results.append(("Database Connectivity", db_results))
    
    # Generate comprehensive report
    print("\n" + "="*80)
    print("COMPREHENSIVE DIAGNOSTIC REPORT")
    print("="*80)
    
    total_tests = 0
    total_failures = 0
    total_successes = 0
    
    for test_name, results in test_results:
        print(f"\n📋 {test_name.upper()}")
        print("-" * len(test_name))
        results.report()
        
        total_tests += len(results.tests)
        total_failures += len(results.failures)
        total_successes += len(results.successes)
    
    print(f"\n🎯 OVERALL SUMMARY")
    print(f"Total tests run: {total_tests}")
    print(f"Total successes: {total_successes}")
    print(f"Total failures: {total_failures}")
    print(f"Success rate: {total_successes/total_tests*100:.1f}%")
    
    # Priority recommendations
    print(f"\n🚨 PRIORITY FIXES NEEDED:")
    priority_issues = []
    
    for test_name, results in test_results:
        if results.failures:
            priority_issues.append(f"{test_name}: {len(results.failures)} issues")
    
    for i, issue in enumerate(priority_issues, 1):
        print(f"{i}. {issue}")

if __name__ == "__main__":
    asyncio.run(main())
