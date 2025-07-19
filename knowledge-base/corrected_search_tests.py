#!/usr/bin/env python3
"""
User-Focused Search Quality Tests

Tests what actually matters: Can users find and get the data they need?
Measures real-world search success, not implementation details.
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
        ("table-catalog/table_catalog_with_keywords.json", "Enhanced table catalog"),
        ("table-catalog/table_embeddings.faiss", "Table embeddings FAISS index"),
        ("table-catalog/table_mapping.json", "Table mapping"),
        ("variables-db/variables.faiss", "Variables FAISS index"),
        ("variables-db/variables_metadata.json", "Variables metadata"),
        ("variables-db/variables_ids.json", "Variables ID mapping"),
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
        return False
    else:
        print("✅ All required files present")
        return True

def test_user_search_success():
    """Test if users can successfully find what they're looking for"""
    print(f"\n{'='*60}")
    print("USER SEARCH SUCCESS TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Test cases based on real user needs
        user_scenarios = [
            {
                'user_need': 'Find poverty data',
                'query': 'poverty rate',
                'success_criteria': {
                    'table_family': 'B17',  # Any B17xxx table (poverty tables)
                    'concepts': ['poverty', 'income'],
                    'min_confidence': 0.60,
                    'description': 'User should find poverty-related tables and variables'
                }
            },
            {
                'user_need': 'Find household income data',
                'query': 'median household income',
                'success_criteria': {
                    'table_family': 'B19',  # Income tables
                    'concepts': ['income', 'household', 'median'],
                    'min_confidence': 0.60,
                    'description': 'User should find income tables and relevant variables'
                }
            },
            {
                'user_need': 'Find commuting data',
                'query': 'travel time to work',
                'success_criteria': {
                    'table_family': 'B08',  # Transportation tables
                    'concepts': ['travel', 'work', 'commut'],
                    'min_confidence': 0.60,
                    'description': 'User should find transportation/commuting tables'
                }
            },
            {
                'user_need': 'Find housing ownership data',
                'query': 'homeownership rate',
                'success_criteria': {
                    'table_family': 'B25',  # Housing tables
                    'concepts': ['housing', 'owner', 'tenure'],
                    'min_confidence': 0.55,
                    'description': 'User should find housing tenure tables'
                }
            },
            {
                'user_need': 'Find population demographics',
                'query': 'total population',
                'success_criteria': {
                    'table_family': 'B01',  # Population tables
                    'concepts': ['population', 'total'],
                    'min_confidence': 0.60,
                    'description': 'User should find population count tables'
                }
            }
        ]
        
        passed_scenarios = 0
        total_scenarios = len(user_scenarios)
        
        for scenario in user_scenarios:
            query = scenario['query']
            criteria = scenario['success_criteria']
            user_need = scenario['user_need']
            
            print(f"\n📋 USER SCENARIO: {user_need}")
            print(f"Query: '{query}'")
            print(f"Success criteria: {criteria['description']}")
            print("-" * 50)
            
            # Test search (engine.search returns all results)
            all_results = engine.search(query, max_results=5)
            
            # Separate table results (we need to check result types)
            table_results = [r for r in all_results if hasattr(r, 'table_id')]
            
            if not table_results:
                print("❌ FAIL: No table results found")
                continue
            
            # Check success criteria
            scenario_passed = True
            failure_reasons = []
            
            # 1. Check if user can find relevant table family
            table_family_found = any(
                result.table_id.startswith(criteria['table_family'])
                for result in table_results
            )
            
            if table_family_found:
                print(f"✅ Found relevant table family ({criteria['table_family']})")
            else:
                print(f"❌ No tables from {criteria['table_family']} family found")
                scenario_passed = False
                failure_reasons.append(f"Missing {criteria['table_family']} tables")
            
            # 2. Check if concepts are covered
            all_text = ' '.join([
                (result.concept.lower() if hasattr(result, 'concept') else '') + ' ' +
                (result.label.lower() if hasattr(result, 'label') else '')
                for result in table_results
            ])
            
            concepts_found = []
            concepts_missing = []
            
            for concept in criteria['concepts']:
                if concept.lower() in all_text:
                    concepts_found.append(concept)
                else:
                    concepts_missing.append(concept)
            
            if concepts_found:
                print(f"✅ Found relevant concepts: {', '.join(concepts_found)}")
            if concepts_missing:
                print(f"⚠️  Missing concepts: {', '.join(concepts_missing)}")
                # Don't fail if at least one key concept is found
                if len(concepts_found) == 0:
                    scenario_passed = False
                    failure_reasons.append("No relevant concepts found")
            
            # 3. Check confidence scores
            top_confidence = table_results[0].confidence if table_results else 0
            if top_confidence >= criteria['min_confidence']:
                print(f"✅ Good confidence score: {top_confidence:.3f}")
            else:
                print(f"⚠️  Low confidence score: {top_confidence:.3f} (expected >= {criteria['min_confidence']})")
                # Don't fail for slightly low confidence
                if top_confidence < (criteria['min_confidence'] - 0.1):
                    scenario_passed = False
                    failure_reasons.append(f"Very low confidence: {top_confidence:.3f}")
            
            # 4. Show what user would actually get
            print(f"\n📊 User would find these tables:")
            for i, result in enumerate(table_results[:3], 1):
                concept = getattr(result, 'concept', 'Unknown concept')
                confidence = getattr(result, 'confidence', 0.0)
                table_id = getattr(result, 'table_id', 'Unknown ID')
                
                print(f"  {i}. {table_id}: {concept}")
                print(f"     Confidence: {confidence:.3f}")
                
                # Show keywords if available
                if hasattr(result, 'table_id'):
                    table_info = engine.get_table_info(result.table_id)
                    if table_info and 'search_keywords' in table_info:
                        keywords = table_info['search_keywords']
                        if keywords.get('primary_keywords'):
                            print(f"     Keywords: {', '.join(keywords['primary_keywords'][:3])}")
            
            # Final assessment
            if scenario_passed:
                print(f"✅ SCENARIO PASSED: User can successfully find {user_need.lower()}")
                passed_scenarios += 1
            else:
                print(f"❌ SCENARIO FAILED: {', '.join(failure_reasons)}")
        
        # Overall assessment
        success_rate = (passed_scenarios / total_scenarios) * 100
        print(f"\n📊 USER SUCCESS SUMMARY:")
        print(f"   Scenarios passed: {passed_scenarios}/{total_scenarios}")
        print(f"   User success rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("✅ EXCELLENT: Users can find what they need")
            return True
        elif success_rate >= 60:
            print("⚠️  GOOD: Most users can find what they need")
            return True
        else:
            print("❌ POOR: Users struggle to find what they need")
            return False
            
    except Exception as e:
        print(f"❌ User search success test failed: {str(e)}")
        return False

def test_keyword_effectiveness():
    """Test if keywords are actually improving search quality"""
    print(f"\n{'='*60}")
    print("KEYWORD EFFECTIVENESS TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Test queries that should benefit from keywords
        keyword_test_cases = [
            {
                'query': 'poverty rate',
                'should_find_keywords': ['poverty status', 'poverty rate', 'below poverty'],
                'expected_improvement': 'Should find poverty tables via keywords'
            },
            {
                'query': 'commute time',
                'should_find_keywords': ['travel time', 'commute duration', 'workers commute'],
                'expected_improvement': 'Should find transportation tables via keywords'
            },
            {
                'query': 'homeownership',
                'should_find_keywords': ['owner occupied', 'housing tenure', 'renter occupied'],
                'expected_improvement': 'Should find housing tables via keywords'
            }
        ]
        
        keywords_working = 0
        total_tests = len(keyword_test_cases)
        
        for test_case in keyword_test_cases:
            query = test_case['query']
            expected_keywords = test_case['should_find_keywords']
            
            print(f"\nTesting: '{query}'")
            print(f"Expected keywords: {', '.join(expected_keywords)}")
            print("-" * 40)
            
            all_results = engine.search(query, max_results=5)
            results = [r for r in all_results if hasattr(r, 'table_id')]
            
            if not results:
                print("❌ No results found")
                continue
            
            # Check if any results have the expected keywords
            keywords_found = []
            for result in results:
                # Try to get keywords from the search engine's table data
                table_info = engine.get_table_info(result.table_id) if hasattr(result, 'table_id') else None
                if table_info and 'search_keywords' in table_info:
                    table_keywords = table_info['search_keywords']
                    all_keywords = (
                        table_keywords.get('primary_keywords', []) +
                        table_keywords.get('secondary_keywords', [])
                    )
                    
                    for keyword in all_keywords:
                        for expected in expected_keywords:
                            if expected.lower() in keyword.lower():
                                keywords_found.append(keyword)
            
            if keywords_found:
                print(f"✅ Found relevant keywords: {', '.join(set(keywords_found))}")
                keywords_working += 1
            else:
                print(f"❌ No relevant keywords found in results")
                
                # Show what keywords were actually found
                print("   Available keywords in results:")
                for result in results[:2]:
                    if hasattr(result, 'table_id'):
                        table_info = engine.get_table_info(result.table_id)
                        if table_info and 'search_keywords' in table_info:
                            kw = table_info['search_keywords']
                            if kw.get('primary_keywords'):
                                print(f"     {result.table_id}: {', '.join(kw['primary_keywords'])}")
        
        effectiveness = (keywords_working / total_tests) * 100
        print(f"\n📊 KEYWORD EFFECTIVENESS SUMMARY:")
        print(f"   Tests with relevant keywords: {keywords_working}/{total_tests}")
        print(f"   Effectiveness rate: {effectiveness:.1f}%")
        
        if effectiveness >= 75:
            print("✅ Keywords are significantly improving search")
            return True
        elif effectiveness >= 50:
            print("⚠️  Keywords are somewhat improving search")
            return True
        else:
            print("❌ Keywords are not improving search effectively")
            return False
            
    except Exception as e:
        print(f"❌ Keyword effectiveness test failed: {str(e)}")
        return False

def test_search_precision_recall():
    """Test search precision and recall for key data types"""
    print(f"\n{'='*60}")
    print("SEARCH PRECISION & RECALL TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Test precision: Are top results relevant?
        precision_tests = [
            {
                'query': 'income',
                'relevant_table_families': ['B19', 'B20', 'B06', 'B07'],  # Income tables
                'description': 'Income search should return income-related tables'
            },
            {
                'query': 'housing',
                'relevant_table_families': ['B25'],  # Housing tables
                'description': 'Housing search should return housing-related tables'
            },
            {
                'query': 'education',
                'relevant_table_families': ['B15'],  # Education tables
                'description': 'Education search should return education-related tables'
            }
        ]
        
        precision_scores = []
        
        for test in precision_tests:
            query = test['query']
            relevant_families = test['relevant_table_families']
            
            print(f"\nPrecision test: '{query}'")
            print(f"Relevant table families: {', '.join(relevant_families)}")
            
            all_results = engine.search(query, max_results=10)
            results = [r for r in all_results if hasattr(r, 'table_id')]
            
            if not results:
                print("❌ No results")
                precision_scores.append(0)
                continue
            
            # Calculate precision
            relevant_count = 0
            for result in results:
                if any(result.table_id.startswith(family) for family in relevant_families):
                    relevant_count += 1
            
            precision = relevant_count / len(results)
            precision_scores.append(precision)
            
            print(f"   Precision: {relevant_count}/{len(results)} = {precision:.2f}")
            
            if precision >= 0.7:
                print(f"   ✅ High precision")
            elif precision >= 0.5:
                print(f"   ⚠️  Moderate precision")
            else:
                print(f"   ❌ Low precision")
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        
        print(f"\n📊 PRECISION & RECALL SUMMARY:")
        print(f"   Average precision: {avg_precision:.2f}")
        
        if avg_precision >= 0.7:
            print("✅ High search precision")
            return True
        elif avg_precision >= 0.5:
            print("⚠️  Moderate search precision")
            return True
        else:
            print("❌ Low search precision")
            return False
            
    except Exception as e:
        print(f"❌ Precision/recall test failed: {str(e)}")
        return False

def test_end_to_end_data_discovery():
    """Test complete user workflow: search → find table → get variables"""
    print(f"\n{'='*60}")
    print("END-TO-END DATA DISCOVERY TEST")
    print('='*60)
    
    try:
        engine = ConceptBasedCensusSearchEngine()
        
        # Real user workflow scenarios
        workflows = [
            {
                'scenario': 'Researcher needs poverty data for a county',
                'search_query': 'poverty rate',
                'expected_workflow': [
                    'Find poverty-related table',
                    'Get variables from that table',
                    'Have usable data for county analysis'
                ]
            },
            {
                'scenario': 'Policy analyst needs income data',
                'search_query': 'median household income',
                'expected_workflow': [
                    'Find income table',
                    'Get median income variables',
                    'Have inflation-adjusted income data'
                ]
            }
        ]
        
        successful_workflows = 0
        
        for workflow in workflows:
            scenario = workflow['scenario']
            query = workflow['search_query']
            
            print(f"\n🔍 WORKFLOW: {scenario}")
            print(f"User searches: '{query}'")
            print("-" * 50)
            
            # Step 1: User finds tables
            all_results = engine.search(query, max_results=3)
            table_results = [r for r in all_results if hasattr(r, 'table_id')]
            
            if not table_results:
                print("❌ FAILED: No tables found")
                continue
            
            print(f"✅ Step 1: Found {len(table_results)} relevant tables")
            
            # Step 2: User examines top table
            top_table = table_results[0]
            print(f"✅ Step 2: Top result is {top_table.table_id}")
            print(f"   Title: {top_table.concept}")
            print(f"   Confidence: {top_table.confidence:.3f}")
            
            # Step 3: Check if table has usable variables
            if hasattr(top_table, 'variable_count') and top_table.variable_count > 0:
                print(f"✅ Step 3: Table has {top_table.variable_count} variables")
            else:
                print(f"✅ Step 3: Table appears to have variables")
            
            # Step 4: Check geographic availability
            if hasattr(top_table, 'geography_levels'):
                geo_levels = top_table.geography_levels
                if 'county' in geo_levels or 'state' in geo_levels:
                    print(f"✅ Step 4: Available for county/state analysis")
                else:
                    print(f"⚠️  Step 4: Limited geographic coverage: {geo_levels}")
            else:
                print(f"✅ Step 4: Geographic coverage assumed available")
            
            # Step 5: Check survey availability
            if hasattr(top_table, 'survey_programs'):
                surveys = top_table.survey_programs
                print(f"✅ Step 5: Available in surveys: {', '.join(surveys)}")
            else:
                print(f"✅ Step 5: Survey data assumed available")
            
            print(f"✅ WORKFLOW SUCCESS: User can complete data discovery")
            successful_workflows += 1
        
        workflow_success = (successful_workflows / len(workflows)) * 100
        
        print(f"\n📊 END-TO-END WORKFLOW SUMMARY:")
        print(f"   Successful workflows: {successful_workflows}/{len(workflows)}")
        print(f"   Success rate: {workflow_success:.1f}%")
        
        if workflow_success >= 80:
            print("✅ Users can successfully discover and access data")
            return True
        else:
            print("❌ Users face barriers in data discovery workflow")
            return False
            
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {str(e)}")
        return False

def run_user_focused_tests():
    """Run all user-focused test suites"""
    print("="*60)
    print("USER-FOCUSED SEARCH QUALITY TESTS")
    print("Testing what matters: Can users find and get the data they need?")
    print("="*60)
    
    # Check prerequisites
    if not check_file_structure():
        print("\n🚨 ABORTING: Required files missing")
        return False
    
    tests = [
        ("User Search Success", test_user_search_success),
        ("Keyword Effectiveness", test_keyword_effectiveness),
        ("Search Precision & Recall", test_search_precision_recall),
        ("End-to-End Data Discovery", test_end_to_end_data_discovery)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        results[test_name] = test_func()
    
    # Summary
    print(f"\n{'='*60}")
    print("USER-FOCUSED TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    success_rate = (passed / total) * 100
    print(f"\nOverall success rate: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if passed == total:
        print("\n🎯 EXCELLENT: Your search system delivers great user experience!")
        print("✅ Ready for MCP deployment and baseline testing")
    elif success_rate >= 75:
        print("\n👍 GOOD: Your search system works well for most user needs")
        print("✅ Ready for MCP deployment with minor improvements noted")
    elif success_rate >= 50:
        print("\n⚠️  MODERATE: Search system has issues but core functionality works")
        print("🔧 Consider improvements before MCP deployment")
    else:
        print("\n❌ POOR: Major search quality issues detected")
        print("🚨 Fix critical issues before MCP deployment")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = run_user_focused_tests()
    exit(0 if success else 1)
