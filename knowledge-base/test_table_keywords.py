#!/usr/bin/env python3
"""
Test table keyword extraction on a single table

Tests the keyword extraction logic on one table to see if it makes sense
before implementing in the full table catalog script.
"""

import json
from collections import Counter
from pathlib import Path

def get_table_keywords(variables, max_keywords=8):
    """Get top keywords from all variables in a table, filtering noise"""
    keyword_counts = Counter()
    
    # Generic terms that appear everywhere (useless for search)
    noise_terms = {
        'american community survey',
        'acs',
        'acs income data',
        'acs poverty data',
        'acs housing data',
        'margin of error',
        'estimate',
        'census',
        'survey data',
        'statistical data',
        'data',
        'statistics'
    }
    
    # Collect all key_terms from all variables in the table
    for var_id, var_data in variables.items():
        key_terms = var_data.get('key_terms', [])
        for term in key_terms:
            term_lower = term.lower()
            if term_lower not in noise_terms:  # Filter out noise
                keyword_counts[term_lower] += 1
    
    # Return top keywords with counts
    top_keywords = keyword_counts.most_common(max_keywords)
    return top_keywords

def test_table_keyword_extraction(table_id_to_test="B19013"):
    """Test keyword extraction on a specific table"""
    
    # Load canonical variables
    canonical_path = Path("source-docs/canonical_variables_refactored.json")
    if not canonical_path.exists():
        canonical_path = Path("source-docs/canonical_variables.json")
        print("⚠️  Using original canonical file - no refactored version found")
    
    if not canonical_path.exists():
        print("❌ No canonical variables file found")
        return
    
    print(f"📁 Loading variables from: {canonical_path.name}")
    
    with open(canonical_path) as f:
        data = json.load(f)
    
    # Extract concepts/variables
    if 'concepts' in data or any(isinstance(v, dict) and 'instances' in v for v in data.values()):
        concepts = data.get('concepts', {})
        if not concepts:
            concepts = {k: v for k, v in data.items() if k != 'metadata' and isinstance(v, dict)}
        structure_type = "concept-based"
    else:
        concepts = data.get('variables', data)
        structure_type = "temporal"
    
    print(f"📊 Structure: {structure_type}, Total: {len(concepts)} items")
    
    # Filter variables for the test table
    table_variables = {}
    for var_id, var_data in concepts.items():
        if var_id.startswith(table_id_to_test + "_"):
            table_variables[var_id] = var_data
    
    if not table_variables:
        print(f"❌ No variables found for table {table_id_to_test}")
        print("Available tables:")
        table_prefixes = set()
        for var_id in list(concepts.keys())[:20]:
            table_prefix = var_id.split('_')[0]
            table_prefixes.add(table_prefix)
        print(f"   Sample tables: {sorted(table_prefixes)}")
        return
    
    print(f"\n🔍 TESTING TABLE: {table_id_to_test}")
    print(f"Variables found: {len(table_variables)}")
    print("=" * 50)
    
    # Show some sample variables and their key terms
    print(f"\n📝 Sample variables and their key terms:")
    sample_count = 0
    for var_id, var_data in table_variables.items():
        if sample_count >= 5:
            break
        
        key_terms = var_data.get('key_terms', [])
        concept = var_data.get('concept', 'N/A')
        
        print(f"  {var_id}: {concept}")
        print(f"    Key terms: {key_terms}")
        
        sample_count += 1
    
    # Extract table-level keywords
    print(f"\n🎯 EXTRACTED TABLE KEYWORDS:")
    top_keywords = get_table_keywords(table_variables, max_keywords=8)
    
    for i, (keyword, count) in enumerate(top_keywords, 1):
        print(f"  {i}. '{keyword}' (appears {count} times)")
    
    # Show what the table embedding text would look like
    keywords_only = [kw for kw, count in top_keywords]
    
    print(f"\n📄 TABLE EMBEDDING TEXT PREVIEW:")
    print("Current approach (basic Census metadata):")
    print(f"  'Median Household Income in the Past 12 Months. Households. Median Household Income'")
    
    print(f"\nNew approach (with extracted keywords):")
    if keywords_only:
        print(f"  'Median Household Income in the Past 12 Months. Households. Median Household Income. Key terms: {', '.join(keywords_only)}'")
    else:
        print(f"  'Median Household Income in the Past 12 Months. Households. Median Household Income. [No keywords found]'")
    
    # Analysis
    print(f"\n📊 ANALYSIS:")
    print(f"  Total unique keywords: {len(top_keywords)}")
    print(f"  Variables processed: {len(table_variables)}")
    print(f"  Keywords per variable avg: {sum(count for _, count in top_keywords) / len(table_variables):.1f}")
    
    return top_keywords

def test_multiple_tables():
    """Test on multiple tables to see variety"""
    test_tables = ["B19013", "B17001", "B25003", "B01001", "B08303"]
    
    print("🧪 TESTING MULTIPLE TABLES")
    print("=" * 60)
    
    for table_id in test_tables:
        print(f"\n--- TABLE {table_id} ---")
        keywords = test_table_keyword_extraction(table_id)
        
        if keywords:
            keyword_list = [kw for kw, count in keywords]
            print(f"Keywords: {', '.join(keyword_list)}")
        else:
            print("No keywords extracted")

if __name__ == "__main__":
    # Test one table in detail
    test_table_keyword_extraction("B19013")  # Median household income
    
    # Uncomment to test multiple tables
    # test_multiple_tables()
