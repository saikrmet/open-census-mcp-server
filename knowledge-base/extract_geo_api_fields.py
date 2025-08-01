#!/usr/bin/env python3
"""Move geographic API fields from variable index to proper geography reference"""

import json
import re
from pathlib import Path

def extract_geo_api_fields():
    """Extract geographic API fields to separate reference file"""
    
    # Geographic API fields that should be moved (not deleted)
    api_fields = {
        'in', 'ucgid', 'CD', 'UA', 'METDIV', 'SUMLEVEL', 'STATE', 'CSA', 
        'PRINCITY', 'BLKGRP', 'COUNTY', 'TRACT', 'PLACE', 'ZCTA5', 'SLDL', 
        'SLDU', 'VTD', 'CBSA', 'NECTA', 'CNECTA', 'NAME', 'GEO_ID'
    }
    
    source_file = Path("source-docs/canonical_variables_refactored.json")
    backup_file = Path("source-docs/canonical_variables_refactored_with_geo.json")
    geo_ref_dir = Path("geo-reference")
    geo_api_file = geo_ref_dir / "census_api_fields.json"
    
    print(f"🏗️  Extracting geographic API fields to proper reference...")
    
    # Create geo-reference directory
    geo_ref_dir.mkdir(exist_ok=True)
    
    # Load canonical variables
    with open(source_file, 'r') as f:
        data = json.load(f)
    
    # Backup original with geo fields
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Backup with geo fields: {backup_file}")
    
    # Extract concepts
    concepts = data.get('concepts', data)
    original_count = len(concepts)
    
    # Extract API fields
    extracted_fields = {}
    cleaned_concepts = {}
    
    for var_id, var_data in concepts.items():
        if var_id in api_fields:
            # Move to geo reference
            extracted_fields[var_id] = var_data
        else:
            # Keep in variables
            cleaned_concepts[var_id] = var_data
    
    print(f"\\n📊 Extraction results:")
    print(f"   Original concepts: {original_count}")
    print(f"   Extracted API fields: {len(extracted_fields)}")
    print(f"   Clean variable concepts: {len(cleaned_concepts)}")
    
    # Save cleaned canonical variables
    if 'concepts' in data:
        data['concepts'] = cleaned_concepts
    else:
        data = cleaned_concepts
    
    with open(source_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Create comprehensive geo API reference
    geo_api_reference = {
        "metadata": {
            "description": "Census API geographic field reference",
            "purpose": "Internal use for API call construction and geography parsing",
            "usage": "Not for user search or LLM concept retrieval",
            "extracted_from": "canonical_variables_refactored.json",
            "field_types": {
                "geographic_identifiers": ["in", "ucgid", "STATE", "COUNTY"],
                "summary_levels": ["SUMLEVEL", "CD", "UA", "TRACT", "BLKGRP"],
                "statistical_areas": ["CBSA", "CSA", "METDIV", "NECTA", "CNECTA"],
                "administrative": ["SLDL", "SLDU", "VTD", "PRINCITY"],
                "postal": ["ZCTA5"],
                "identifiers": ["NAME", "GEO_ID"]
            }
        },
        "api_fields": extracted_fields,
        "usage_examples": {
            "for_state_data": "for=state:*&in=us:1",
            "for_county_in_state": "for=county:*&in=state:06",
            "for_tract_in_county": "for=tract:*&in=state:06+county:001"
        }
    }
    
    # Save geo API reference
    with open(geo_api_file, 'w') as f:
        json.dump(geo_api_reference, f, indent=2)
    
    print(f"\\n✅ Geographic API reference created")
    print(f"📁 File: {geo_api_file}")
    print(f"\\n📋 Extracted fields by type:")
    for field_type, fields in geo_api_reference["metadata"]["field_types"].items():
        existing_fields = [f for f in fields if f in extracted_fields]
        if existing_fields:
            print(f"   {field_type}: {existing_fields}")
    
    print(f"\\n✅ Cleaned canonical variables saved")
    print(f"📁 File: {source_file}")
    
    print(f"\\n🔄 Next steps:")
    print(f"   1. Rebuild variable database: python build-kb-concept-based.py --variables-only --rebuild")
    print(f"   2. Remove geographic field filter hack from kb_search.py")
    print(f"   3. Test clean search system")
    print(f"   4. Update geo parsing logic to use {geo_api_file}")

if __name__ == "__main__":
    extract_geo_api_fields()
