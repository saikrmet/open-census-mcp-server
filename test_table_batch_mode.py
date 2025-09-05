#!/usr/bin/env python3
"""
Census MCP Server - Table Batch Mode Test Suite

Comprehensive test script to validate all functionality before presentation:
1. Individual variable queries (existing functionality)
2. Table batch mode (new functionality) 
3. Natural language concept resolution
4. Error handling and fallbacks
5. Statistical consultation

Run this to verify everything works before your presentation!
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from census_mcp_server import server_instance
from data_retrieval.python_census_api import PythonCensusAPI
from data_retrieval.table_resolver import TableResolver
from utils.table_formatter import TableFormatter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPTestSuite:
    """Comprehensive test suite for MCP server functionality"""
    
    def __init__(self):
        self.api = None
        self.table_resolver = None
        self.formatter = None
        self.test_results = []
    
    async def run_all_tests(self):
        """Run complete test suite"""
        
        print("🚀 Census MCP Server - Table Batch Mode Test Suite")
        print("=" * 60)
        
        # Initialize components
        await self._test_initialization()
        
        if not self.api:
            print("❌ Cannot continue - API initialization failed")
            return
        
        # Core functionality tests
        await self._test_geographic_resolution()
        await self._test_individual_variables()
        await self._test_table_batch_mode()
        await self._test_natural_language_concepts()
        await self._test_error_handling()
        await self._test_statistical_consultation()
        
        # Print summary
        self._print_test_summary()
    
    async def _test_initialization(self):
        """Test component initialization"""
        print("\n📋 Testing Component Initialization")
        print("-" * 40)
        
        try:
            # Test Census API
            self.api = PythonCensusAPI()
            if self.api:
                print("✅ Census API initialized")
                self._record_result("initialization", "census_api", True, "Census API ready")
            else:
                print("❌ Census API failed to initialize")
                self._record_result("initialization", "census_api", False, "API initialization failed")
                return
            
            # Test Table Resolver
            self.table_resolver = TableResolver()
            if self.table_resolver:
                print("✅ Table Resolver initialized")
                print(f"   Available concepts: {len(self.table_resolver.get_available_concepts())}")
                print(f"   Available tables: {len(self.table_resolver.get_available_tables())}")
                self._record_result("initialization", "table_resolver", True, "Table resolver ready")
            else:
                print("⚠️ Table Resolver failed - fallback mode")
                self._record_result("initialization", "table_resolver", False, "Using fallback mode")
            
            # Test Table Formatter
            self.formatter = TableFormatter()
            print("✅ Table Formatter initialized")
            self._record_result("initialization", "table_formatter", True, "Formatter ready")
            
            # Test Server Instance
            if server_instance and server_instance.census_api:
                print("✅ MCP Server instance ready")
                self._record_result("initialization", "mcp_server", True, "Server ready")
            else:
                print("⚠️ MCP Server instance not fully initialized")
                self._record_result("initialization", "mcp_server", False, "Server issues")
        
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            self._record_result("initialization", "system", False, str(e))
    
    async def _test_geographic_resolution(self):
        """Test geographic resolution"""
        print("\n🌍 Testing Geographic Resolution")
        print("-" * 40)
        
        test_locations = [
            "United States",
            "Austin, TX", 
            "Washington",
            "New York, NY"
        ]
        
        for location in test_locations:
            try:
                # Test through the full API
                result = await self.api.get_demographic_data(
                    location=location,
                    variables=["B01003_001E"],  # Simple population variable
                    year=2023,
                    survey="acs5"
                )
                
                if 'error' not in result:
                    resolved_name = result['resolved_location']['name']
                    method = result['resolved_location']['resolution_method']
                    print(f"✅ {location} → {resolved_name} ({method})")
                    self._record_result("geographic", location, True, f"Resolved to {resolved_name}")
                else:
                    print(f"❌ {location} → {result['error']}")
                    self._record_result("geographic", location, False, result['error'])
            
            except Exception as e:
                print(f"❌ {location} → System error: {e}")
                self._record_result("geographic", location, False, str(e))
    
    async def _test_individual_variables(self):
        """Test individual variable queries (existing functionality)"""
        print("\n📊 Testing Individual Variable Queries")
        print("-" * 40)
        
        test_cases = [
            ("Austin, TX", ["B01003_001E"], "Total population"),
            ("United States", ["B19013_001E"], "Median household income"),
            ("Seattle, WA", ["B25001_001E", "B25003_001E"], "Multiple housing variables")
        ]
        
        for location, variables, description in test_cases:
            try:
                result = await self.api.get_demographic_data(
                    location=location,
                    variables=variables,
                    year=2023,
                    survey="acs5"
                )
                
                if 'error' not in result:
                    data_count = len(result.get('data', {}))
                    print(f"✅ {description}: {location} → {data_count} variables retrieved")
                    
                    # Show sample data
                    data = result.get('data', {})
                    for var_id, var_data in list(data.items())[:2]:  # Show first 2
                        formatted = var_data.get('formatted', 'No data')
                        print(f"   {var_id}: {formatted}")
                    
                    self._record_result("individual_vars", description, True, f"{data_count} vars")
                else:
                    print(f"❌ {description}: {result['error']}")
                    self._record_result("individual_vars", description, False, result['error'])
            
            except Exception as e:
                print(f"❌ {description}: System error: {e}")
                self._record_result("individual_vars", description, False, str(e))
    
    async def _test_table_batch_mode(self):
        """Test new table batch mode functionality"""
        print("\n📋 Testing Table Batch Mode")
        print("-" * 40)
        
        if not hasattr(self.api, 'get_table_data'):
            print("❌ get_table_data method not available")
            self._record_result("table_batch", "method_check", False, "Method missing")
            return
        
        test_cases = [
            ("Austin, TX", ["B19013"], "Income distribution table"),
            ("United States", ["B01003"], "Population table"), 
            ("Seattle, WA", ["B25001", "B25003"], "Multiple housing tables"),
            ("Dallas, TX", ["B15003"], "Education attainment table")
        ]
        
        for location, table_ids, description in test_cases:
            try:
                result = await self.api.get_table_data(
                    location=location,
                    table_ids=table_ids,
                    year=2023,
                    survey="acs5"
                )
                
                if 'error' not in result:
                    tables = result.get('tables', {})
                    total_vars = sum(t.get('variable_count', 0) for t in tables.values())
                    
                    print(f"✅ {description}: {location}")
                    print(f"   Tables retrieved: {len(tables)}")
                    print(f"   Total variables: {total_vars}")
                    
                    # Show sample table info
                    for table_id, table_data in tables.items():
                        title = table_data.get('title', 'Unknown')[:50] + "..."
                        var_count = table_data.get('variable_count', 0)
                        print(f"   {table_id}: {title} ({var_count} vars)")
                    
                    self._record_result("table_batch", description, True, f"{len(tables)} tables, {total_vars} vars")
                else:
                    print(f"❌ {description}: {result['error']}")
                    self._record_result("table_batch", description, False, result['error'])
            
            except Exception as e:
                print(f"❌ {description}: System error: {e}")
                self._record_result("table_batch", description, False, str(e))
    
    async def _test_natural_language_concepts(self):
        """Test natural language concept resolution"""
        print("\n💬 Testing Natural Language Concepts")
        print("-" * 40)
        
        if not self.table_resolver:
            print("⚠️ Table resolver not available - skipping concept tests")
            return
        
        test_concepts = [
            "income distribution",
            "housing units", 
            "population",
            "education",
            "employment"
        ]
        
        for concept in test_concepts:
            try:
                # Test concept resolution
                resolution = self.table_resolver.resolve_tables([concept])
                
                if 'error' not in resolution:
                    resolved = resolution['resolved_tables']
                    total_vars = sum(len(t['variables']) for t in resolved)
                    
                    print(f"✅ '{concept}' → {len(resolved)} tables, {total_vars} variables")
                    
                    for table_info in resolved[:2]:  # Show first 2 tables
                        table_id = table_info['table_id']
                        var_count = len(table_info['variables'])
                        method = table_info['resolution_method']
                        print(f"   {table_id}: {var_count} vars ({method})")
                    
                    self._record_result("concepts", concept, True, f"{len(resolved)} tables")
                    
                    # Test full table data retrieval with concept
                    try:
                        result = await self.api.get_table_data(
                            location="Austin, TX",
                            table_ids=[concept],
                            year=2023,
                            survey="acs5"
                        )
                        
                        if 'error' not in result:
                            print(f"   ✅ Full data retrieval successful")
                        else:
                            print(f"   ⚠️ Full data retrieval failed: {result['error']}")
                    
                    except Exception as e:
                        print(f"   ⚠️ Full data retrieval error: {e}")
                
                else:
                    print(f"❌ '{concept}' → {resolution['error']}")
                    if resolution.get('suggestions'):
                        print(f"   💡 Suggestions: {resolution['suggestions']}")
                    self._record_result("concepts", concept, False, resolution['error'])
            
            except Exception as e:
                print(f"❌ '{concept}' → System error: {e}")
                self._record_result("concepts", concept, False, str(e))
    
    async def _test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n⚠️ Testing Error Handling")
        print("-" * 40)
        
        error_test_cases = [
            # Geographic errors
            ("Invalid Location", ["B01003_001E"], "invalid_location"),
            ("Austin, TX", ["INVALID_VAR"], "invalid_variable"),
            ("Austin, TX", ["unknown_table"], "invalid_table"),
            ("", ["B01003_001E"], "empty_location"),
            ("Austin, TX", [], "empty_variables")
        ]
        
        for location, variables_or_tables, test_type in error_test_cases:
            try:
                if test_type in ["invalid_table", "empty_variables"]:
                    # Test table batch mode errors
                    if hasattr(self.api, 'get_table_data'):
                        result = await self.api.get_table_data(
                            location=location,
                            table_ids=variables_or_tables,
                            year=2023,
                            survey="acs5"
                        )
                    else:
                        result = {'error': 'get_table_data not available'}
                else:
                    # Test individual variable errors
                    result = await self.api.get_demographic_data(
                        location=location,
                        variables=variables_or_tables,
                        year=2023,
                        survey="acs5"
                    )
                
                if 'error' in result:
                    print(f"✅ {test_type}: Proper error handling")
                    print(f"   Error: {result['error'][:100]}...")
                    self._record_result("error_handling", test_type, True, "Error handled properly")
                else:
                    print(f"⚠️ {test_type}: Expected error but got success")
                    self._record_result("error_handling", test_type, False, "Should have errored")
            
            except Exception as e:
                print(f"✅ {test_type}: Exception caught properly")
                print(f"   Exception: {str(e)[:100]}...")
                self._record_result("error_handling", test_type, True, "Exception handled")
    
    async def _test_statistical_consultation(self):
        """Test LLM statistical consultation"""
        print("\n🧠 Testing Statistical Consultation")
        print("-" * 40)
        
        if not server_instance or not server_instance.llm_advisor:
            print("⚠️ LLM advisor not available - skipping consultation tests")
            return
        
        consultation_queries = [
            "What variables should I use for income analysis?",
            "How do I compare housing costs between cities?",
            "What are the limitations of ACS poverty data?",
            "Which table shows educational attainment?"
        ]
        
        for query in consultation_queries:
            try:
                consultation = server_instance.llm_advisor.consult(query, "Austin, TX")
                
                print(f"✅ Query: {query[:40]}...")
                print(f"   Confidence: {consultation.confidence:.1%}")
                print(f"   Recommendations: {len(consultation.recommended_variables)}")
                print(f"   Advice: {consultation.expert_advice[:80]}...")
                
                self._record_result("consultation", query[:30], True, 
                                  f"Confidence: {consultation.confidence:.1%}")
            
            except Exception as e:
                print(f"❌ Query failed: {e}")
                self._record_result("consultation", query[:30], False, str(e))
    
    def _record_result(self, category: str, test_name: str, success: bool, details: str):
        """Record test result"""
        self.test_results.append({
            'category': category,
            'test_name': test_name,
            'success': success,
            'details': details
        })
    
    def _print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        # Group results by category
        categories = {}
        for result in self.test_results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0, 'failed': 0}
            
            categories[cat]['total'] += 1
            if result['success']:
                categories[cat]['passed'] += 1
            else:
                categories[cat]['failed'] += 1
        
        # Print category summaries
        for category, stats in categories.items():
            total = stats['total']
            passed = stats['passed']
            failed = stats['failed']
            success_rate = (passed / total * 100) if total > 0 else 0
            
            status = "✅" if failed == 0 else "⚠️" if passed > failed else "❌"
            print(f"{status} {category.upper()}: {passed}/{total} passed ({success_rate:.0f}%)")
        
        # Overall summary
        total_tests = len(self.test_results)
        total_passed = sum(1 for r in self.test_results if r['success'])
        total_failed = total_tests - total_passed
        overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🎯 OVERALL: {total_passed}/{total_tests} tests passed ({overall_rate:.0f}%)")
        
        # Readiness assessment
        print("\n🚀 PRESENTATION READINESS:")
        
        # Check critical components
        critical_passed = 0
        critical_total = 0
        
        for category in ['initialization', 'geographic', 'table_batch']:
            if category in categories:
                if categories[category]['failed'] == 0:
                    critical_passed += 1
                critical_total += 1
        
        if critical_passed == critical_total:
            print("✅ READY FOR PRESENTATION! All critical components working.")
        elif critical_passed >= critical_total * 0.8:
            print("⚠️ MOSTLY READY - Some minor issues but core functionality works.")
        else:
            print("❌ NOT READY - Critical issues need to be resolved first.")
        
        # Print any failures
        failures = [r for r in self.test_results if not r['success']]
        if failures:
            print(f"\n⚠️ ISSUES TO REVIEW ({len(failures)}):")
            for failure in failures:
                print(f"   • {failure['category']}.{failure['test_name']}: {failure['details']}")


async def main():
    """Run the comprehensive test suite"""
    test_suite = MCPTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
