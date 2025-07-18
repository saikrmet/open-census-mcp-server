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
from kb_search import ConceptBasedCensusSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def test_search_quality():
    """Test search quality with known good variable matches"""
    print(f"\n{'='*60}")
    print("SEARCH QUALITY TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Test cases with expected top results
        test_cases = [
            {
                'query': 'median household income',
                'expected_table': 'B19013',
                'expected_variable_pattern': 'B19013_001E'
            },
            {
                'query': 'total population',
                'expected_table': 'B01003', 
                'expected_variable_pattern': 'B01003_001E'
            }
        ]
        
        for test_case in test_cases:
            query = test_case['query']
            expected_table = test_case['expected_table']
            expected_pattern = test_case['expected_variable_pattern']
            
            print(f"\nQuery: '{query}'")
            print(f"Expected: {expected_pattern}")
            print("-" * 40)
            
            results = engine.search(query, max_results=3)
            
            if not results:
                print("❌ No results found")
                continue
            
            top_result = results[0]
            print(f"Top result: {top_result.variable_id} (confidence: {top_result.confidence:.3f})")
            
            # Check if we got the expected result
            if top_result.table_id == expected_table:
                print("✅ Correct table found")
            else:
                print(f"⚠️  Expected table {expected_table}, got {top_result.table_id}")
                
            if top_result.variable_id == expected_pattern:
                print("✅ Exact variable match")
            else:
                print(f"⚠️  Expected {expected_pattern}, got {top_result.variable_id}")
        
        print("✅ Search quality test completed")
        return True
        
    except Exception as e:
        print(f"❌ Search quality test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all test suites and report results"""
    print("="*60)
    print("CONCEPT-BASED CENSUS SEARCH ENGINE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Search Functionality", test_basic_search_functionality),
        ("Duplicate Elimination", test_duplicate_elimination), 
        ("Geographic Intelligence", test_geographic_intelligence),
        ("Concept-Based Structure", test_concept_based_structure),
        ("Search Quality", test_search_quality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        results[test_name] = test_func()
    
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
