#!/usr/bin/env python3
"""
Merge official Census labels with enriched universe → canonical_variables.json

This creates a single authoritative source file combining:
- Official Census variable labels, concepts, survey info
- Enriched metadata with statistical intelligence and domain weights
- Complete variable records for RAG indexing

Output: knowledge-base/source-docs/canonical_variables.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def load_official_variables():
    """Load official Census variable data from complete_census_corpus.json"""
    
    official_path = Path("knowledge-base/complete_2023_acs_variables/complete_census_corpus.json")
    
    if not official_path.exists():
        raise FileNotFoundError(f"Official variables file not found: {official_path}")
    
    print(f"📁 Loading official variables from: {official_path}")
    
    with open(official_path) as f:
        data = json.load(f)
    
    # Expected format: {"acs5": [...], "acs1": [...]}
    if not isinstance(data, dict) or not all(k in ['acs1', 'acs5'] for k in data.keys()):
        raise ValueError(f"Unexpected JSON format in {official_path}")
    
    # Merge ACS flavors with universal survey structure
    all_variables = {}
    year = "2023"  # Current dataset year
    survey_program = "acs"  # American Community Survey
    
    for survey_type, var_list in data.items():
        if isinstance(var_list, list):
            # Map Census API names to standard flavor names
            flavor_mapping = {
                "acs5": "5yr",
                "acs1": "1yr"
            }
            flavor = flavor_mapping.get(survey_type, survey_type)
            
            print(f"📊 Loading {len(var_list)} variables from {survey_program}_{flavor}_{year}")
            
            for var_record in var_list:
                if isinstance(var_record, dict) and 'variable_id' in var_record:
                    base_var_id = var_record['variable_id']
                    
                    # Create universal compound key: survey_flavor_year_variable
                    compound_key = f"{survey_program}_{flavor}_{year}_{base_var_id}"
                    
                    # Ensure metadata fields are set correctly
                    var_record['survey_program'] = survey_program
                    var_record['flavor'] = flavor
                    var_record['year'] = year
                    var_record['base_variable_id'] = base_var_id
                    var_record['temporal_id'] = compound_key
                    
                    all_variables[compound_key] = var_record
    
    print(f"📈 Total unique variables loaded: {len(all_variables)}")
    return all_variables

def load_enriched_variables():
    """Load enriched universe with weights and statistical intelligence"""
    
    enriched_path = Path("knowledge-base/2023_ACS_Enriched_Universe_weighted.json")
    
    if not enriched_path.exists():
        print(f"⚠️  Enriched variables not found at: {enriched_path}")
        print("    Continuing with official variables only...")
        return {}
    
    print(f"📊 Loading enriched variables from: {enriched_path}")
    
    with open(enriched_path) as f:
        data = json.load(f)
    
    # Handle different formats
    enriched_vars = {}
    
    if isinstance(data, list):
        # Format: [{"variable_id": "...", ...}, ...]
        for record in data:
            if isinstance(record, dict) and 'variable_id' in record:
                enriched_vars[record['variable_id']] = record
    
    elif isinstance(data, dict):
        if 'variables' in data:
            # Format: {"variables": {...}}
            variables = data['variables']
            if isinstance(variables, dict):
                enriched_vars = variables
            elif isinstance(variables, list):
                enriched_vars = {v['variable_id']: v for v in variables if 'variable_id' in v}
        else:
            # Format: {"B01001_001E": {...}, ...}
            enriched_vars = data
    
    print(f"📈 Loaded {len(enriched_vars)} enriched variable records")
    return enriched_vars

def merge_variables(official_vars, enriched_vars):
    """Merge official and enriched variable data using full temporal compound keys"""
    
    canonical = {}
    enriched_count = 0
    
    for compound_key, official_data in official_vars.items():
        # Extract components from compound key: survey_flavor_year_variable
        parts = compound_key.split('_', 3)  # Split on first 3 underscores only
        survey_program = official_data.get('survey_program', parts[0])
        flavor = official_data.get('flavor', parts[1])
        year = official_data.get('year', parts[2])
        base_variable_id = official_data.get('base_variable_id', parts[3])
        
        # Start with official Census data
        canonical_record = {
            "temporal_id": compound_key,  # Full unique identifier
            "variable_id": base_variable_id,  # Original variable ID
            "survey_program": survey_program,  # acs, cps, decennial, etc.
            "flavor": flavor,  # 5yr, 1yr, monthly, etc.
            "year": year,
            "label": official_data.get('label', 'Unknown'),
            "concept": official_data.get('concept', 'Unknown'),
            "predicateType": official_data.get('predicateType', ''),
            "group": official_data.get('group', ''),
            "table_id": base_variable_id.split('_')[0] if '_' in base_variable_id else base_variable_id[:6],
            
            # Survey methodology metadata
            "survey_context": f"{survey_program.upper()} {flavor} {year} estimates",
            "flavor_characteristics": {
                "5yr": "5-year estimates: larger sample, more reliable, available for small geographies",
                "1yr": "1-year estimates: smaller sample, less reliable, large geographies only (65K+ population)",
                "3yr": "3-year estimates: discontinued 2015, medium sample size"
            }.get(flavor, f"Unknown flavor: {flavor}"),
            "methodological_notes": f"Based on {year} {survey_program.upper()} {flavor} methodology",
        }
        
        # Add enriched data if available (enriched data uses base variable ID)
        if base_variable_id in enriched_vars:
            enriched_data = enriched_vars[base_variable_id]
            
            # Preserve important enriched fields
            for field in [
                'category_weights_linear', 'category_weights_exponential',
                'analysis', 'enrichment_text', 'summary', 'enhanced_description',
                'statistical_methodology', 'fitness_for_use', 'caveats',
                'complexity', 'table_family', 'source', 'metadata',
                'quality_tier', 'agreement_score', 'processing_cost'
            ]:
                if field in enriched_data:
                    canonical_record[field] = enriched_data[field]
            
            enriched_count += 1
        
        canonical[compound_key] = canonical_record
    
    print(f"🔗 Merged {len(canonical)} total survey-flavor-year-variable combinations")
    print(f"📊 {enriched_count} variables have enriched metadata")
    print(f"📝 {len(canonical) - enriched_count} variables are official-only")
    print(f"🕐 All variables structured as: survey_flavor_year_variable")
    
    return canonical

def save_canonical_variables(canonical_vars, output_path):
    """Save canonical variables to JSON file"""
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_variables": len(canonical_vars),
        "enriched_count": sum(1 for v in canonical_vars.values() if 'category_weights_linear' in v),
        "description": "Canonical Census variable records combining official metadata with enriched statistical intelligence",
        "source_files": [
            "Official Census variable catalog",
            "Enriched universe with domain weights and statistical analysis"
        ]
    }
    
    # Create final output structure
    output_data = {
        "metadata": metadata,
        "variables": canonical_vars
    }
    
    # Save with pretty formatting for human readability
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, separators=(',', ': '))
    
    return metadata

def main():
    """Build canonical variables file"""
    
    print("🏗️  Building canonical variables file...")
    print("=" * 60)
    
    try:
        # Load source data
        official_vars = load_official_variables()
        enriched_vars = load_enriched_variables()
        
        # Merge the data
        canonical_vars = merge_variables(official_vars, enriched_vars)
        
        # Save output
        output_path = Path("knowledge-base/source-docs/canonical_variables.json")
        metadata = save_canonical_variables(canonical_vars, output_path)
        
        # Report success
        print("\n✅ Canonical variables file created successfully!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Total variables: {metadata['total_variables']:,}")
        print(f"📈 Enriched variables: {metadata['enriched_count']:,}")
        print(f"💾 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Next steps
        print(f"\n🚀 Next steps:")
        print(f"   1. Update project documentation with new canonical variables file")
        print(f"   2. Rebuild vector database for RAG using this file")
        print(f"   3. Update container to use new RAG pipeline")
        
    except Exception as e:
        print(f"❌ Error building canonical variables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
