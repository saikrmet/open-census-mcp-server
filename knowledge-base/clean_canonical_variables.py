#!/usr/bin/env python3
"""Clean canonical variables by removing geographic API fields"""

import json
import re
from pathlib import Path

def clean_canonical_variables():
    """Remove geographic API fields from canonical variables"""
    
    # Geographic API fields that should be removed
    geographic_fields = {
        'in', 'ucgid', 'CD', 'UA', 'METDIV', 'SUMLEVEL', 'STATE', 'CSA', 
        'PRINCITY', 'BLKGRP', 'COUNTY', 'TRACT', 'PLACE', 'ZCTA5', 'SLDL', 
        'SLDU', 'VTD', 'CBSA', 'NECTA', 'CNECTA', 'NAME', 'GEO_ID'
    }
    
    source_file = Path("source-docs/canonical_variables_refactored.json")
    backup_file = Path("source-docs/canonical_variables_refactored_backup.json")
    
    print(f"🔧 Cleaning canonical variables...")
    print(f"📁 Source: {source_file}")
    
    # Load data
    with open(source_file, 'r') as f:
        data = json.load(f)
    
    # Backup original
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Backup created: {backup_file}")
    
    # Get concepts
    concepts = data.get('concepts', data)
    original_count = len(concepts)
    
    # Remove geographic fields
    removed_fields = []
    for field in geographic_fields:
        if field in concepts:
            del concepts[field]
            removed_fields.append(field)
    
    print(f"\\n📊 Cleaning results:")
    print(f"   Original concepts: {original_count}")
    print(f"   Removed geographic fields: {len(removed_fields)}")
    print(f"   Clean concepts: {len(concepts)}")
    print(f"   Removed: {removed_fields}")
    
    # Update concepts in data structure
    if 'concepts' in data:
        data['concepts'] = concepts
    else:
        # If root level, replace entire structure
        data = {k: v for k, v in data.items() if k not in geographic_fields}
    
    # Save cleaned version
    with open(source_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\\n✅ Cleaned canonical variables saved")
    print(f"📁 File: {source_file}")
    print(f"\\n🔄 Next steps:")
    print(f"   1. Rebuild variable database: python build-kb-concept-based.py --variables-only")
    print(f"   2. Test search system")

if __name__ == "__main__":
    clean_canonical_variables()
