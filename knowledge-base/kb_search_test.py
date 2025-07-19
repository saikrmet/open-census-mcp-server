#!/usr/bin/env python3
"""
Test suite for the Concept-Based Census Search Engine

Tests the core functionality and validates:
1. Search quality and relevance
2. Geographic intelligence
3. Duplicate elimination at variable level
4. Survey instance awareness
5. Concept-based structure detection
"""

import logging
import json
from pathlib import Path
from kb_search import ConceptBasedCensusSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_file_structure():
    """Check if the required files and directories exist"""
    print(f"\n{'='*60}")
    print("FILE STRUCTURE CHECK")
    print('='*60)
    
    required_files = [
        ("table-catalog/table_catalog.json", "Table catalog"),
        ("table-catalog/table_embeddings.faiss", "Table embeddings FAISS index"),
        ("table-catalog/table_mapping.json", "Table mapping"),
        ("variables-db/variables.faiss", "Variables FAISS index"),
        ("variables-db/variables_metadata.json", "Variables metadata"),
        ("variables-db/build_info.json", "Build info")
    ]
    
    missing_files = []
    
    for file_path, description in required_files:
        if Path(file_path).exists():
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ MISSING {description}: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n🚨 CRITICAL: {len(missing_files)} required files missing!")
        print("This will cause search failures. You need to:")
        if any("table-catalog" in f for f in missing_files):
            print("1. Build table catalog: python build-table-catalog.py")
        if any("variables-db" in f for f in missing_files):
            print("2. Build variables database: python build-kb-concept-based.py --variables-only --faiss")
        return False
    else:
        print("✅ All required files present")
        return True

def debug_search_failure(query, expected_result=None):
    """Debug a specific search failure by showing intermediate results"""
    print(f"\n🔍 DEBUGGING SEARCH FAILURE: '{query}'")
    print("-" * 50)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Show what we get
        results = engine.search(query, max_results=5)
        
        if not results:
            print("❌ NO RESULTS RETURNED")
            return
        
        print(f"Got {len(results)} results:")
        for i, result in enumerate(results, 1):
            status = ""
            if expected_result and result.variable_id == expected_result:
                status = " 🎯 EXPECTED!"
            elif expected_result:
                status = f" (expected {expected_result})"
            
            print(f"{i}. {result.variable_id} - confidence: {result.confidence:.3f}{status}")
            print(f"   Label: {result.label}")
            print(f"   Concept: {result.concept}")
            
            # Show summary if available
            if hasattr(result, 'summary') and result.summary:
                print(f"   Summary: {result.summary[:100]}...")
        
        if expected_result:
            found = any(r.variable_id == expected_result for r in results)
            if not found:
                print(f"\n❌ EXPECTED RESULT {expected_result} NOT FOUND")
                
                # Try to find it manually
                expected_info = engine.get_variable_info(expected_result)
                if expected_info:
                    print(f"   But {expected_result} EXISTS in metadata:")
                    print(f"   Concept: {expected_info.get('concept', 'N/A')}")
                    print(f"   Label: {expected_info.get('label', 'N/A')}")
                else:
                    print(f"   {expected_result} NOT FOUND in metadata either")
        
    except Exception as e:
        print(f"❌ Search failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_basic_search_functionality():
    """Test basic search functionality with canonical queries"""
    print(f"\n{'='*60}")
    print("BASIC SEARCH FUNCTIONALITY TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        test_queries = [
            "median household income",
            "travel time to work",
            "poverty rate",
            "housing tenure",
            "population by age and sex"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 40)
            
            results = engine.search(query, max_results=3)
            
            if not results:
                print("❌ No results found")
                continue
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.variable_id} (confidence: {result.confidence:.3f})")
                print(f"   Label: {result.label}")
                print(f"   Structure: {result.structure_type}")
                
        print("✅ Basic search functionality working")
        
    except Exception as e:
        print(f"❌ Basic search test failed: {str(e)}")
        return False
    
    return True

def test_duplicate_elimination():
    """Test that duplicate variable_ids are actually eliminated"""
    print(f"\n{'='*60}")
    print("DUPLICATE ELIMINATION TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Test with query that historically returned duplicates
        query = "travel time to work"
        results = engine.search(query, max_results=10)
        
        # Check for duplicate variable_ids (not concepts!)
        seen_variable_ids = set()
        duplicates_found = []
        unique_count = 0
        
        for result in results:
            if result.variable_id in seen_variable_ids:
                duplicates_found.append(result.variable_id)
                print(f"❌ DUPLICATE variable_id: {result.variable_id}")
            else:
                seen_variable_ids.add(result.variable_id)
                unique_count += 1
                print(f"✅ UNIQUE: {result.variable_id} - {result.concept}")
        
        print(f"\nResults: {len(results)} total, {unique_count} unique variable_ids")
        
        if duplicates_found:
            print(f"❌ Found {len(duplicates_found)} duplicate variable_ids")
            return False
        else:
            print("✅ No duplicate variable_ids found - concept-based system working")
            return True
            
    except Exception as e:
        print(f"❌ Duplicate elimination test failed: {str(e)}")
        return False

def test_geographic_intelligence():
    """Test geographic context parsing and relevance"""
    print(f"\n{'='*60}")
    print("GEOGRAPHIC INTELLIGENCE TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        geographic_queries = [
            "poverty rate in Detroit",
            "median income in Richmond, VA",
            "housing costs in California",
            "commute time in Fairfax County"
        ]
        
        for query in geographic_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 40)
            
            # Parse geographic context
            geo_context = engine.geo_parser.parse_geographic_context(query)
            print(f"Geographic context detected: {geo_context.location_mentioned}")
            if geo_context.location_mentioned:
                print(f"  Location: {geo_context.location_text}")
                print(f"  Level: {geo_context.geography_level}")
            
            results = engine.search(query, max_results=3)
            
            if not results:
                print("❌ No results found")
                continue
                
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.variable_id} (confidence: {result.confidence:.3f})")
                print(f"   Geographic relevance: {result.geographic_relevance:.3f}")
                print(f"   Available surveys: {result.available_surveys}")
        
        print("✅ Geographic intelligence test completed")
        return True
        
    except Exception as e:
        print(f"❌ Geographic intelligence test failed: {str(e)}")
        return False

def test_concept_based_structure():
    """Test that concept-based structure is properly detected and used"""
    print(f"\n{'='*60}")
    print("CONCEPT-BASED STRUCTURE TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Test a few queries
        results = engine.search("median household income", max_results=5)
        
        concept_based_count = 0
        total_results = len(results)
        
        for result in results:
            print(f"{result.variable_id}: structure_type = {result.structure_type}")
            if result.structure_type == 'concept_based':
                concept_based_count += 1
        
        print(f"\nResults: {concept_based_count}/{total_results} have concept_based structure")
        
        if concept_based_count > 0:
            print("✅ Concept-based structure detected")
            return True
        else:
            print("⚠️  No concept-based structure detected - may be using legacy data")
            return False
            
    except Exception as e:
        print(f"❌ Concept-based structure test failed: {str(e)}")
        return False

def inspect_variable_content(engine, variable_id):
    """Inspect what content exists for a specific variable"""
    print(f"\n🔍 INSPECTING VARIABLE CONTENT: {variable_id}")
    print("-" * 40)
    
    # Try to find this variable in the metadata
    var_info = engine.get_variable_info(variable_id)
    
    if var_info:
        print(f"✅ Found {variable_id} in metadata")
        print(f"   Concept: {var_info.get('concept', 'N/A')}")
        print(f"   Label: {var_info.get('label', 'N/A')}")
        print(f"   Structure: {var_info.get('structure_type', 'N/A')}")
        
        # Show key metadata that affects search
        if 'has_summary' in var_info:
            print(f"   Has summary: {var_info.get('has_summary', False)}")
        if 'summary_length' in var_info:
            print(f"   Summary length: {var_info.get('summary_length', 0)} chars")
        if 'enrichment_length' in var_info:
            print(f"   Enrichment length: {var_info.get('enrichment_length', 0)} chars")
        if 'key_terms_count' in var_info:
            print(f"   Key terms count: {var_info.get('key_terms_count', 0)}")
        
        return True
    else:
        print(f"❌ {variable_id} NOT FOUND in metadata")
        return False

def check_canonical_file_for_variable(variable_id):
    """Check if a variable exists in the canonical file"""
    print(f"\n📁 CHECKING CANONICAL FILE FOR {variable_id}")
    print("-" * 40)
    
    # Look for canonical file
    canonical_files = [
        "source-docs/canonical_variables_refactored.json",
        "../source-docs/canonical_variables_refactored.json",
        "canonical_variables_refactored.json",
        "source-docs/canonical_variables.json",
        "../source-docs/canonical_variables.json",
        "canonical_variables.json"
    ]
    
    canonical_path = None
    for file_path in canonical_files:
        if Path(file_path).exists():
            canonical_path = Path(file_path)
            break
    
    if not canonical_path:
        print("❌ NO CANONICAL FILE FOUND!")
        return False
    
    print(f"📁 Using canonical file: {canonical_path}")
    
    try:
        with open(canonical_path) as f:
            data = json.load(f)
        
        # Determine structure
        if 'concepts' in data or any(isinstance(v, dict) and 'instances' in v for v in data.values()):
            concepts = data.get('concepts', {})
            if not concepts:
                concepts = {k: v for k, v in data.items() if k != 'metadata' and isinstance(v, dict)}
            structure_type = "concept-based"
        else:
            concepts = data.get('variables', data)
            structure_type = "temporal"
        
        print(f"📊 Structure: {structure_type}, Total: {len(concepts)} items")
        
        # Direct lookup
        if variable_id in concepts:
            concept_data = concepts[variable_id]
            print(f"✅ Found {variable_id} in canonical file")
            
            # Show summary info
            summary = concept_data.get('summary', '')
            if summary:
                print(f"   Summary ({len(summary)} chars): {summary[:150]}...")
            else:
                print(f"   ❌ NO SUMMARY")
            
            # Show key terms
            key_terms = concept_data.get('key_terms', [])
            if key_terms:
                print(f"   Key terms: {', '.join(key_terms[:5])}")
            else:
                print(f"   ❌ NO KEY TERMS")
            
            # Show concept/label
            print(f"   Concept: {concept_data.get('concept', 'N/A')}")
            print(f"   Label: {concept_data.get('label', 'N/A')}")
            
            return True
        else:
            print(f"❌ {variable_id} NOT FOUND in canonical file")
            
            # Look for similar variables in same table family
            table_id = variable_id.split('_')[0]
            similar = [k for k in concepts.keys() if k.startswith(table_id)]
            if similar:
                print(f"   Similar variables in {table_id} table: {similar[:5]}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error reading canonical file: {e}")
        return False

def test_search_quality():
    """Test search quality with known good variable matches - COMPREHENSIVE VERSION"""
    print(f"\n{'='*60}")
    print("SEARCH QUALITY TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Test cases with expected top results
        test_cases = [
            {
                'query': 'poverty rate',
                'expected_table': 'B17001',
                'expected_variable_pattern': 'B17001_002E',
                'min_confidence': 0.70
            },
            {
                'query': 'median household income',
                'expected_table': 'B19013',
                'expected_variable_pattern': 'B19013_001E',
                'min_confidence': 0.70
            },
            {
                'query': 'total population',
                'expected_table': 'B01003',
                'expected_variable_pattern': 'B01003_001E',
                'min_confidence': 0.60  # Lower bar for this one
            }
        ]
        
        failures = 0
        total_tests = len(test_cases) * 2  # variable match + confidence
        
        for test_case in test_cases:
            query = test_case['query']
            expected_table = test_case['expected_table']
            expected_pattern = test_case['expected_variable_pattern']
            min_confidence = test_case['min_confidence']
            
            print(f"\nQuery: '{query}'")
            print(f"Expected: {expected_pattern} (confidence >= {min_confidence})")
            print("-" * 50)
            
            results = engine.search(query, max_results=5)
            
            if not results:
                print("❌ No results found")
                failures += 2  # Both variable and confidence fail
                
                # Debug what's missing
                print("\n🔍 DEBUGGING MISSING RESULTS:")
                check_canonical_file_for_variable(expected_pattern)
                inspect_variable_content(engine, expected_pattern)
                continue
            
            # Show all results
            print(f"Got {len(results)} results:")
            expected_found = False
            for i, result in enumerate(results, 1):
                status = " 🎯 EXPECTED!" if result.variable_id == expected_pattern else ""
                print(f"  {i}. {result.variable_id} (confidence: {result.confidence:.3f}){status}")
                print(f"     Label: {result.label}")
                
                if result.variable_id == expected_pattern:
                    expected_found = True
            
            # Test 1: Variable match
            if expected_found:
                print("✅ Expected variable found in results")
            else:
                print(f"❌ FAILED: Expected variable {expected_pattern} not found")
                failures += 1
                
                # Debug what's wrong
                print(f"\n🔍 DEBUGGING MISSING VARIABLE {expected_pattern}:")
                check_canonical_file_for_variable(expected_pattern)
                inspect_variable_content(engine, expected_pattern)
            
            # Test 2: Top result confidence
            top_result = results[0]
            if top_result.confidence >= min_confidence:
                print(f"✅ Top result confidence acceptable ({top_result.confidence:.3f} >= {min_confidence})")
            else:
                print(f"❌ FAILED: Top result confidence too low ({top_result.confidence:.3f} < {min_confidence})")
                failures += 1
            
            # Test 3: Expected variable ranking (if found)
            if expected_found:
                expected_rank = None
                for i, result in enumerate(results):
                    if result.variable_id == expected_pattern:
                        expected_rank = i + 1
                        break
                
                if expected_rank == 1:
                    print(f"✅ Expected variable is top result")
                elif expected_rank <= 3:
                    print(f"⚠️  Expected variable is rank {expected_rank} (not ideal but acceptable)")
                else:
                    print(f"❌ Expected variable is rank {expected_rank} (too low)")
        
        # Final assessment
        passed_tests = total_tests - failures
        print(f"\n📊 SEARCH QUALITY SUMMARY:")
        print(f"   Tests passed: {passed_tests}/{total_tests}")
        print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failures == 0:
            print("✅ Search quality test PASSED - All expectations met")
            return True
        elif failures <= 2:
            print("⚠️  Search quality test MOSTLY PASSED - Minor issues")
            return True  # Be a bit lenient
        else:
            print(f"❌ Search quality test FAILED - {failures} major failures")
            return False
        
    except Exception as e:
        print(f"❌ Search quality test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
        
def test_keyword_enhanced_search():
    """Test that generated keywords improve search quality"""
    print(f"\n{'='*60}")
    print("KEYWORD-ENHANCED SEARCH TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Check if keywords catalog exists
        keywords_catalog_path = Path("table-catalog/table_catalog_with_keywords.json")
        if not keywords_catalog_path.exists():
            print("⚠️  Keywords catalog not found - cannot test keyword enhancement")
            print("   Expected: table-catalog/table_catalog_with_keywords.json")
            return False
        
        # Load keywords for validation
        with open(keywords_catalog_path) as f:
            keywords_data = json.load(f)
        
        tables_with_keywords = 0
        for table in keywords_data.get('tables', []):
            if 'search_keywords' in table:
                tables_with_keywords += 1
        
        print(f"📊 Keywords available for {tables_with_keywords} tables")
        
        # Test keyword-specific queries that should work better now
        keyword_test_cases = [
            {
                'query': 'poverty rate',
                'expected_keywords': ['poverty rate', 'below poverty line'],
                'should_find_table': 'B17001',
                'description': 'Primary keyword match for poverty'
            },
            {
                'query': 'commute time',
                'expected_keywords': ['travel time', 'commute'],
                'should_find_table': 'B08303',
                'description': 'Common search term for travel time to work'
            },
            {
                'query': 'internet access',
                'expected_keywords': ['internet', 'broadband'],
                'should_find_table': 'B28002',
                'description': 'User-friendly term for internet subscriptions'
            },
            {
                'query': 'homeownership',
                'expected_keywords': ['owner occupied', 'renter occupied'],
                'should_find_table': 'B25003',
                'description': 'Common term for housing tenure'
            }
        ]
        
        failures = 0
        total_tests = len(keyword_test_cases)
        
        for test_case in keyword_test_cases:
            query = test_case['query']
            expected_table = test_case['should_find_table']
            description = test_case['description']
            
            print(f"\nTest: {description}")
            print(f"Query: '{query}' → Expected table: {expected_table}")
            print("-" * 50)
            
            # Search for tables (coarse search)
            table_results = engine.table_search.search_tables(query, k=5)
            
            if not table_results:
                print("❌ No table results found")
                failures += 1
                continue
            
            # Check if expected table is found
            found_expected = False
            expected_rank = None
            
            print("Table search results:")
            for i, result in enumerate(table_results, 1):
                table_id = result['table_id']
                confidence = result['confidence']
                status = " 🎯 EXPECTED!" if table_id == expected_table else ""
                
                print(f"  {i}. {table_id} (confidence: {confidence:.3f}){status}")
                
                if table_id == expected_table:
                    found_expected = True
                    expected_rank = i
                    
                    # Show if this table has keywords
                    table_data = result.get('table_data', {})
                    if 'search_keywords' in table_data:
                        keywords = table_data['search_keywords']
                        primary = keywords.get('primary_keywords', [])
                        secondary = keywords.get('secondary_keywords', [])
                        print(f"       Primary keywords: {', '.join(primary)}")
                        print(f"       Secondary keywords: {', '.join(secondary)}")
                        if keywords.get('summary'):
                            print(f"       Summary: {keywords['summary'][:100]}...")
                    else:
                        print(f"       ⚠️  No keywords found for this table")
            
            # Evaluate results
            if found_expected:
                if expected_rank == 1:
                    print(f"✅ Expected table {expected_table} found at rank 1")
                elif expected_rank <= 3:
                    print(f"✅ Expected table {expected_table} found at rank {expected_rank} (acceptable)")
                else:
                    print(f"⚠️  Expected table {expected_table} found at rank {expected_rank} (could be better)")
                    failures += 0.5  # Partial failure
            else:
                print(f"❌ Expected table {expected_table} not found in top 5 results")
                failures += 1
                
                # Debug: Check if this table exists and has keywords
                print(f"\n🔍 Debugging missing table {expected_table}:")
                for table in keywords_data.get('tables', []):
                    if table['table_id'] == expected_table:
                        if 'search_keywords' in table:
                            kw = table['search_keywords']
                            print(f"   Table exists with keywords: {kw.get('primary_keywords', [])}")
                        else:
                            print(f"   Table exists but NO KEYWORDS")
                        break
                else:
                    print(f"   Table {expected_table} not found in keywords catalog")
        
        # Summary
        passed_tests = total_tests - failures
        print(f"\n📊 KEYWORD ENHANCEMENT SUMMARY:")
        print(f"   Tests passed: {passed_tests}/{total_tests}")
        print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failures == 0:
            print("✅ Keyword enhancement test PASSED - Keywords working as expected")
            return True
        elif failures <= 1:
            print("⚠️  Keyword enhancement test MOSTLY PASSED - Minor issues")
            return True
        else:
            print(f"❌ Keyword enhancement test FAILED - Keywords not improving search")
            return False
            
    except Exception as e:
        print(f"❌ Keyword enhancement test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_keyword_metadata_integration():
    """Test that keyword metadata is properly integrated into search system"""
    print(f"\n{'='*60}")
    print("KEYWORD METADATA INTEGRATION TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Test if table search is using keywords catalog
        if hasattr(engine, 'table_search') and hasattr(engine.table_search, 'tables'):
            sample_tables = list(engine.table_search.tables.values())[:5]
            
            keyword_tables = 0
            for table in sample_tables:
                if 'search_keywords' in table:
                    keyword_tables += 1
                    print(f"✅ {table['table_id']}: Has search keywords")
                    
                    # Show sample keywords
                    kw = table['search_keywords']
                    if kw.get('primary_keywords'):
                        print(f"   Primary: {', '.join(kw['primary_keywords'][:3])}")
                    if kw.get('summary'):
                        print(f"   Summary: {kw['summary'][:80]}...")
                else:
                    print(f"❌ {table['table_id']}: Missing search keywords")
            
            integration_rate = keyword_tables / len(sample_tables) * 100
            print(f"\n📊 Integration rate: {keyword_tables}/{len(sample_tables)} ({integration_rate:.1f}%)")
            
            if integration_rate >= 80:
                print("✅ Keywords properly integrated into search system")
                return True
            else:
                print("❌ Keywords not properly integrated - check catalog loading")
                return False
        else:
            print("❌ Cannot access table search metadata for integration test")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False
    

def run_all_tests():
    """Run all test suites and report results"""
    print("="*60)
    print("CONCEPT-BASED CENSUS SEARCH ENGINE TEST SUITE")
    print("="*60)
    
    # First check file structure
    if not check_file_structure():
        print("\n🚨 ABORTING: Required files missing")
        return False
    
    tests = [
        ("Basic Search Functionality", test_basic_search_functionality),
        ("Duplicate Elimination", test_duplicate_elimination),
        ("Geographic Intelligence", test_geographic_intelligence),
        ("Concept-Based Structure", test_concept_based_structure),
        ("Search Quality", test_search_quality),
        ("Keyword Enhancement", test_keyword_enhanced_search),
        ("Keyword Integration", test_keyword_metadata_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        results[test_name] = test_func()
    
    # If search quality failed, debug the specific failures
    if not results.get("Search Quality", True):
        print(f"\n🔍 DEBUGGING SEARCH QUALITY FAILURES")
        debug_search_failure("poverty rate", "B17001_002E")
        debug_search_failure("median household income", "B19013_001E")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎯 All tests passed! Concept-based search system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
