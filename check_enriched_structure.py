#!/usr/bin/env python3
"""
Check the structure of 2023_ACS_Enriched_Universe_weighted.json
"""

import json
from pathlib import Path

def check_enriched_structure():
    """Examine the enriched universe structure"""
    
    enriched_path = Path("knowledge-base/2023_ACS_Enriched_Universe_weighted.json")
    
    if not enriched_path.exists():
        print(f"❌ File not found: {enriched_path}")
        return
    
    print(f"📁 Examining: {enriched_path}")
    print(f"📊 File size: {enriched_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 60)
    
    with open(enriched_path) as f:
        data = json.load(f)
    
    print(f"🔍 Top-level type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"📋 Top-level keys: {list(data.keys())}")
        
        # Check each top-level section
        for key, value in data.items():
            print(f"\n📍 Section: {key}")
            print(f"   Type: {type(value)}")
            
            if isinstance(value, list):
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   First item type: {type(value[0])}")
                    if isinstance(value[0], dict):
                        print(f"   First item keys: {list(value[0].keys())}")
                        
                        # Show sample enriched variable
                        sample = value[0]
                        print(f"\n   📍 Sample variable:")
                        for field in ['variable_id', 'label', 'concept', 'category_weights_linear', 'analysis']:
                            if field in sample:
                                val = sample[field]
                                if isinstance(val, str) and len(val) > 100:
                                    display_val = val[:100] + "..."
                                elif isinstance(val, dict):
                                    display_val = f"dict with {len(val)} keys: {list(val.keys())[:5]}"
                                else:
                                    display_val = val
                                print(f"      {field}: {display_val}")
            
            elif isinstance(value, dict):
                print(f"   Length: {len(value)}")
                sample_keys = list(value.keys())[:3]
                print(f"   Sample keys: {sample_keys}")
                
                if sample_keys:
                    sample_key = sample_keys[0]
                    sample_value = value[sample_key]
                    print(f"   Sample value type: {type(sample_value)}")
                    
                    if isinstance(sample_value, dict):
                        print(f"   Sample value keys: {list(sample_value.keys())}")
                        
                        # Show sample enriched variable
                        print(f"\n   📍 Sample variable: {sample_key}")
                        for field in ['variable_id', 'label', 'concept', 'category_weights_linear', 'analysis']:
                            if field in sample_value:
                                val = sample_value[field]
                                if isinstance(val, str) and len(val) > 100:
                                    display_val = val[:100] + "..."
                                elif isinstance(val, dict):
                                    display_val = f"dict with {len(val)} keys: {list(val.keys())[:5]}"
                                else:
                                    display_val = val
                                print(f"      {field}: {display_val}")
    
    elif isinstance(data, list):
        print(f"📈 List length: {len(data)}")
        if len(data) > 0:
            print(f"   First item type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"   First item keys: {list(data[0].keys())}")
                
                # Show sample enriched variable
                sample = data[0]
                print(f"\n📍 Sample variable:")
                for field in ['variable_id', 'label', 'concept', 'category_weights_linear', 'analysis']:
                    if field in sample:
                        val = sample[field]
                        if isinstance(val, str) and len(val) > 100:
                            display_val = val[:100] + "..."
                        elif isinstance(val, dict):
                            display_val = f"dict with {len(val)} keys: {list(val.keys())[:5]}"
                        else:
                            display_val = val
                        print(f"   {field}: {display_val}")

if __name__ == "__main__":
    check_enriched_structure()
