#!/usr/bin/env python3
"""Diagnostic script to find where search is failing"""

import sys
from pathlib import Path

# Add knowledge-base to path
sys.path.insert(0, str(Path(__file__).parent))

def diagnose_search_failure():
    """Step-by-step diagnosis of search pipeline"""
    print("🔍 DIAGNOSING SEARCH FAILURE...")
    
    try:
        from kb_search import ConceptBasedCensusSearchEngine
        
        engine = ConceptBasedCensusSearchEngine()
        query = 'median household income'
        
        print(f"\\n1️⃣ Testing table search for: '{query}'")
        table_results = engine.table_search.search_tables(query, k=5)
        print(f"   Found {len(table_results)} tables")
        
        if table_results:
            for i, result in enumerate(table_results[:3]):
                print(f"   {i+1}. {result['table_id']} (confidence: {result['confidence']:.3f})")
        
        print(f"\\n2️⃣ Testing variable search within top table...")
        if table_results:
            top_table = table_results[0]
            table_id = top_table['table_id']
            print(f"   Searching variables in table: {table_id}")
            
            var_results = engine.variable_search.search_within_table(table_id, query, k=5)
            print(f"   Found {len(var_results)} variables in {table_id}")
            
            if var_results:
                for i, result in enumerate(var_results[:3]):
                    var_meta = result['variable_metadata']
                    print(f"   {i+1}. {var_meta.get('variable_id', 'UNKNOWN')} (score: {result['final_score']:.3f})")
            else:
                print("   ❌ NO VARIABLES FOUND - This is the problem!")
                
                # Check if table has any variables at all
                print(f"\\n3️⃣ Checking if {table_id} has variables in metadata...")
                all_var_count = len(engine.variable_search.variables_metadata)
                print(f"   Total variables in database: {all_var_count}")
                
                # Sample a few variables to see structure
                print(f"\\n4️⃣ Sample variables structure:")
                for i, var_meta in enumerate(engine.variable_search.variables_metadata[:3]):
                    var_id = var_meta.get('variable_id', 'UNKNOWN')
                    var_table = var_id.split('_')[0] if '_' in var_id else 'UNKNOWN'
                    print(f"   Sample {i+1}: {var_id} from table {var_table}")
        else:
            print("   ❌ NO TABLES FOUND")
            
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_search_failure()
