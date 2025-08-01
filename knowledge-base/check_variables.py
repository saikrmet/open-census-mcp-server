#!/usr/bin/env python3
"""Check variable database quality"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent))

def check_variable_quality():
    """Check what's in the variable database"""
    print("🔍 CHECKING VARIABLE DATABASE QUALITY...")
    
    try:
        from kb_search import ConceptBasedVariableSearch
        
        var_search = ConceptBasedVariableSearch()
        variables = var_search.variables_metadata
        
        print(f"Total variables: {len(variables)}")
        
        # Categorize variables
        census_pattern = re.compile(r'^[BCSDP]\d+[A-Z]*_\d+[EM]$')
        valid_census = []
        invalid_variables = []
        
        for var_meta in variables:
            var_id = var_meta.get('variable_id', '')
            if census_pattern.match(var_id):
                valid_census.append(var_id)
            else:
                invalid_variables.append(var_id)
        
        print(f"\\n📊 Quality breakdown:")
        print(f"✅ Valid Census variables: {len(valid_census)}")
        print(f"❌ Invalid variables: {len(invalid_variables)}")
        
        if invalid_variables:
            print(f"\\n🚨 Invalid variables (first 10):")
            for i, var_id in enumerate(invalid_variables[:10]):
                print(f"   {i+1}. '{var_id}'")
        
        if valid_census:
            print(f"\\n✅ Valid Census variables (first 10):")
            for i, var_id in enumerate(valid_census[:10]):
                table_id = var_id.split('_')[0]
                print(f"   {i+1}. {var_id} (table: {table_id})")
        
        # Check if B19215 variables exist
        b19215_vars = [v for v in valid_census if v.startswith('B19215_')]
        print(f"\\n🎯 B19215 variables found: {len(b19215_vars)}")
        for var in b19215_vars[:5]:
            print(f"   - {var}")
            
    except Exception as e:
        print(f"❌ Check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_variable_quality()
