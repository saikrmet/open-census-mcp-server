#!/usr/bin/env python3
"""
Build comprehensive label-concept map from all Census sources.
Combines: complete_variables.csv + table shells + survey info
Output: variable_id → {label, concept, survey} mapping
"""

import json
import pandas as pd
from pathlib import Path

def build_label_concept_map():
    """Build comprehensive mapping of variable_id to official Census descriptions."""
    
    # Paths
    base_path = Path("knowledge-base")
    csv_path = base_path / "complete_2023_acs_variables" / "complete_variables.csv"
    shells_path = base_path / "source-docs" / "acs_table_shells" / "2023" / "ACS2023_Table_Shells.xlsx"
    output_path = base_path / "variable_label_map.json"
    
    print(f"▶ Loading variables from {csv_path}")
    
    # Load main variables CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Variables CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"▶ Loaded {len(df)} variables from CSV")
    
    # Build basic mapping
    label_map = {}
    
    for _, row in df.iterrows():
        variable_id = row['variable_id']
        
        # Skip non-variable entries (like 'in', 'ucgid')
        if not variable_id.startswith(('B', 'C', 'S', 'K', 'DP', 'CP', 'SP')):
            continue
            
        label_map[variable_id] = {
            'variable_id': variable_id,
            'label': str(row['label']).strip(),
            'concept': str(row['concept']).strip(),
            'survey': str(row['survey']).strip() if 'survey' in row else 'acs5',
            'table_family': str(row.get('table_family', variable_id[:3])),
            'source': 'complete_variables_csv'
        }
    
    print(f"▶ Created base mapping for {len(label_map)} variables")
    
    # Enhance with table shells if available
    if shells_path.exists():
        print(f"▶ Enhancing with table shells from {shells_path}")
        try:
            # Read table shells (assuming it has useful additional context)
            shells_df = pd.read_excel(shells_path)
            print(f"▶ Table shells loaded: {len(shells_df)} entries")
            
            # Add any additional context from shells
            # (Implementation depends on shells structure)
            
        except Exception as e:
            print(f"Warning: Could not load table shells: {e}")
    
    # Sample the results
    sample_vars = list(label_map.keys())[:5]
    print(f"\n📝 Sample mappings:")
    for var_id in sample_vars:
        entry = label_map[var_id]
        print(f"  {var_id}: '{entry['label'][:50]}...' | '{entry['concept'][:40]}...' | {entry['survey']}")
    
    # Check for B19013_001E specifically
    if 'B19013_001E' in label_map:
        b19013 = label_map['B19013_001E']
        print(f"\n✅ B19013_001E found:")
        print(f"  Label: {b19013['label']}")
        print(f"  Concept: {b19013['concept']}")
        print(f"  Survey: {b19013['survey']}")
    else:
        print("\n❌ B19013_001E not found in mapping")
    
    # Save mapping
    with open(output_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\n✅ Label-concept map saved to {output_path}")
    print(f"📊 Total variables mapped: {len(label_map)}")
    
    return output_path

if __name__ == "__main__":
    build_label_concept_map()
