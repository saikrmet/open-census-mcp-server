#!/usr/bin/env python3
"""
Check the structure of complete_census_corpus.json to understand format
"""

import json
from pathlib import Path

def check_json_structure():
    """Examine the JSON structure and sample data"""
    
    json_path = Path("knowledge-base/complete_2023_acs_variables/complete_census_corpus.json")
    
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        return
    
    print(f"📁 Examining: {json_path}")
    print(f"📊 File size: {json_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 60)
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Check top-level structure
    print(f"🔍 Top-level type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"📋 Top-level keys: {list(data.keys())}")
        
        # Check if it has a variables section
        if 'variables' in data:
            variables = data['variables']
            print(f"📊 Variables type: {type(variables)}")
            print(f"📈 Total variables: {len(variables)}")
            
            # Sample a few variables to see structure
            if isinstance(variables, dict):
                sample_keys = list(variables.keys())[:5]
                print(f"\n🔬 Sample variable IDs: {sample_keys}")
                
                for key in sample_keys[:2]:
                    print(f"\n📍 Variable: {key}")
                    var_data = variables[key]
                    print(f"   Keys: {list(var_data.keys())}")
                    
                    # Show key fields
                    for field in ['label', 'concept', 'survey', 'predicateType']:
                        if field in var_data:
                            value = var_data[field]
                            display_value = value[:100] + "..." if len(str(value)) > 100 else value
                            print(f"   {field}: {display_value}")
            
            elif isinstance(variables, list):
                print(f"📈 Variables list length: {len(variables)}")
                
                # Sample first few
                for i, var_data in enumerate(variables[:2]):
                    print(f"\n📍 Variable {i+1}:")
                    print(f"   Keys: {list(var_data.keys())}")
                    
                    # Show key fields
                    for field in ['variable_id', 'label', 'concept', 'survey', 'predicateType']:
                        if field in var_data:
                            value = var_data[field]
                            display_value = value[:100] + "..." if len(str(value)) > 100 else value
                            print(f"   {field}: {display_value}")
        
        else:
            # Direct variable mapping
            sample_keys = list(data.keys())[:5]
            print(f"🔬 Sample keys: {sample_keys}")
            
            for key in sample_keys[:2]:
                print(f"\n📍 Key: {key}")
                survey_data = data[key]
                
                if isinstance(survey_data, dict):
                    print(f"   Type: {type(survey_data)}")
                    print(f"   Variables count: {len(survey_data)}")
                    
                    # Sample variables from this survey
                    var_sample = list(survey_data.keys())[:3]
                    print(f"   Sample variable IDs: {var_sample}")
                    
                    for var_id in var_sample[:1]:
                        print(f"\n   📍 Variable: {var_id}")
                        var_data = survey_data[var_id]
                        if isinstance(var_data, dict):
                            print(f"      Keys: {list(var_data.keys())}")
                            
                            # Show key fields
                            for field in ['label', 'concept', 'predicateType', 'group']:
                                if field in var_data:
                                    value = var_data[field]
                                    display_value = value[:80] + "..." if len(str(value)) > 80 else value
                                    print(f"      {field}: {display_value}")
                        else:
                            print(f"      Value: {var_data}")
                else:
                    print(f"   Type: {type(survey_data)}")
                    print(f"   Value preview: {str(survey_data)[:200]}...")
    
    elif isinstance(data, list):
        print(f"📈 List length: {len(data)}")
        
        # Sample first few items
        for i, item in enumerate(data[:2]):
            print(f"\n📍 Item {i+1}:")
            if isinstance(item, dict):
                print(f"   Keys: {list(item.keys())}")
                
                # Show key fields
                for field in ['variable_id', 'label', 'concept', 'survey', 'predicateType']:
                    if field in item:
                        value = item[field]
                        display_value = value[:100] + "..." if len(str(value)) > 100 else value
                        print(f"   {field}: {display_value}")
    
    # Check for survey distribution
    print(f"\n📊 Survey Analysis:")
    survey_counts = {}
    
    if isinstance(data, dict) and all(k in ['acs1', 'acs5'] for k in data.keys()):
        # Structure: {"acs5": {...}, "acs1": {...}}
        for survey_type, variables in data.items():
            if isinstance(variables, dict):
                survey_counts[survey_type] = len(variables)
                print(f"   {survey_type}: {len(variables):,} variables")
            else:
                print(f"   {survey_type}: {type(variables)} (unexpected format)")
    
    else:
        # Other structures - check normally
        if isinstance(data, dict) and 'variables' in data:
            variables = data['variables']
            if isinstance(variables, dict):
                for var_data in variables.values():
                    survey = var_data.get('survey', 'unknown')
                    survey_counts[survey] = survey_counts.get(survey, 0) + 1
            elif isinstance(variables, list):
                for var_data in variables:
                    survey = var_data.get('survey', 'unknown')
                    survey_counts[survey] = survey_counts.get(survey, 0) + 1
        
        elif isinstance(data, dict):
            for var_data in data.values():
                if isinstance(var_data, dict):
                    survey = var_data.get('survey', 'unknown')
                    survey_counts[survey] = survey_counts.get(survey, 0) + 1
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    survey = item.get('survey', 'unknown')
                    survey_counts[survey] = survey_counts.get(survey, 0) + 1
        
        if survey_counts:
            print("📈 Survey distribution:")
            for survey, count in sorted(survey_counts.items()):
                print(f"   {survey}: {count:,} variables")
        else:
            print("⚠️  Could not determine survey distribution")

if __name__ == "__main__":
    check_json_structure()
