#!/usr/bin/env python3
"""
Debug what text is actually being embedded for specific variables
"""

import json
import sys
from pathlib import Path

def load_official_labels(base_path):
    """Load official Census labels from the comprehensive mapping."""
    label_map_path = base_path / "variable_label_map.json"
    
    if not label_map_path.exists():
        print(f"❌ Label map not found at {label_map_path}")
        return {}
    
    try:
        with open(label_map_path) as f:
            label_map = json.load(f)
        print(f"✅ Loaded {len(label_map)} official Census labels")
        return label_map
    except Exception as e:
        print(f"❌ Could not load label map: {e}")
        return {}

def create_clean_embedding_text(record, official_labels=None):
    """Create clean embedding text using official Census descriptions."""
    variable_id = record.get('variable_id', '')
    
    # Use official Census labels when available
    if official_labels and variable_id in official_labels:
        official = official_labels[variable_id]
        label = official['label']
        concept = official['concept'] 
        survey = official['survey']
        source = "OFFICIAL"
    else:
        # Fallback to enriched data if no official mapping
        label = record.get('label', f'Variable {variable_id}')
        concept = record.get('concept', 'Unknown')
        survey = record.get('survey', 'acs5')
        source = "FALLBACK"
        
        # Clean up "Unknown" fallbacks
        if label == 'Unknown':
            label = f'Variable {variable_id}'
        if concept == 'Unknown':
            concept = 'Census Variable'
    
    # Create embedding text with Spock's format
    text = f"V_ID_{variable_id} {label} {concept} {survey}"
    
    return text.strip(), source

def main():
    # Set up paths
    base_path = Path("knowledge-base")
    universe_file = base_path / "2023_ACS_Enriched_Universe_weighted.json"
    
    if not universe_file.exists():
        print(f"❌ Universe file not found: {universe_file}")
        sys.exit(1)
    
    # Load official labels
    print("🔍 Loading official Census labels...")
    official_labels = load_official_labels(base_path)
    
    # Load universe data
    print(f"📁 Loading universe data from {universe_file}...")
    with open(universe_file) as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        if 'variables' in data:
            variables = list(data['variables'].values()) if isinstance(data['variables'], dict) else data['variables']
        else:
            variables = list(data.values())
    else:
        variables = data
    
    print(f"📊 Loaded {len(variables)} variables")
    
    # Test specific variables that failed
    test_variables = [
        "B25077_001E",  # Should be "Median value (dollars)"
        "B03003_003E",  # Should be "Hispanic or Latino"
        "B01001_020E",  # Should be age 65+ related
        "B19013_001E"   # Should be "Median household income"
    ]
    
    print("\n🎯 Testing specific variables that failed semantic search:")
    print("=" * 80)
    
    found_count = 0
    for variable_id in test_variables:
        # Find the variable in the data
        variable_record = None
        for record in variables:
            if record.get('variable_id') == variable_id:
                variable_record = record
                break
        
        if variable_record:
            found_count += 1
            text, source = create_clean_embedding_text(variable_record, official_labels)
            
            print(f"\n📍 {variable_id}:")
            print(f"   Source: {source}")
            print(f"   Embedding text: {text}")
            
            # Show original data for comparison
            orig_label = variable_record.get('label', 'N/A')
            orig_concept = variable_record.get('concept', 'N/A')
            print(f"   Original label: {orig_label}")
            print(f"   Original concept: {orig_concept}")
            
            # Check if it's in official labels
            if official_labels and variable_id in official_labels:
                official_data = official_labels[variable_id]
                print(f"   Official label: {official_data['label']}")
                print(f"   Official concept: {official_data['concept']}")
            else:
                print(f"   ❌ NOT FOUND in official labels")
        else:
            print(f"\n❌ {variable_id}: NOT FOUND in universe data")
    
    print(f"\n📈 Summary:")
    print(f"   Found {found_count}/{len(test_variables)} test variables")
    print(f"   Official labels available: {'YES' if official_labels else 'NO'}")
    
    if not official_labels:
        print(f"\n💡 Next step: Run this to generate official labels:")
        print(f"   python knowledge-base/scripts/build_label_concept_map.py \\")
        print(f"          --raw-vars knowledge-base/complete_2023_acs_variables/complete_variables.json \\")
        print(f"          --out knowledge-base/variable_label_map.json")
    
    # Sample a few random variables to see the pattern
    print(f"\n🔬 Sample of first 5 variables for pattern check:")
    print("-" * 60)
    for i, record in enumerate(variables[:5]):
        if isinstance(record, dict) and 'variable_id' in record:
            text, source = create_clean_embedding_text(record, official_labels)
            var_id = record['variable_id']
            print(f"{i+1}. {var_id} ({source}): {text[:100]}...")

if __name__ == "__main__":
    main()
