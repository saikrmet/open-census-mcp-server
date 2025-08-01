#!/usr/bin/env python3
"""
Debug script to check the actual embedding text and metadata for B19013_001E
"""

import json
import sys
from pathlib import Path

def check_b19013_embedding_text():
    """Check what text was used to embed B19013_001E"""
    
    # Check if we're in the right directory
    variables_dir = Path("knowledge-base/variables-db")
    if not variables_dir.exists():
        variables_dir = Path("variables-db")
    
    if not variables_dir.exists():
        print("❌ Cannot find variables-db directory")
        return
    
    # Load the variable IDs mapping
    ids_file = variables_dir / "variables_ids.json"
    if not ids_file.exists():
        print("❌ Cannot find variables_ids.json")
        return
    
    with open(ids_file) as f:
        ids_data = json.load(f)
    
    variable_ids = ids_data['variable_ids']
    print(f"Total variables in database: {len(variable_ids)}")
    
    # Find B19013_001E position
    target_var = 'B19013_001E'
    try:
        position = variable_ids.index(target_var)
        print(f"✅ Found {target_var} at position {position}")
    except ValueError:
        print(f"❌ {target_var} not found in variable_ids")
        # Check for similar variables
        b19013_vars = [v for v in variable_ids if v.startswith('B19013')]
        print(f"B19013 variables found: {b19013_vars}")
        return
    
    # Load the metadata
    metadata_file = variables_dir / "variables_metadata.json"
    if not metadata_file.exists():
        print("❌ Cannot find variables_metadata.json")
        return
    
    with open(metadata_file) as f:
        metadata_array = json.load(f)
    
    if position >= len(metadata_array):
        print(f"❌ Position {position} is out of range for metadata array (length: {len(metadata_array)})")
        return
    
    # Get the metadata for B19013_001E
    var_metadata = metadata_array[position]
    
    print(f"\n=== METADATA FOR {target_var} ===")
    print(f"Variable ID: {var_metadata.get('variable_id', 'MISSING')}")
    print(f"Concept: {var_metadata.get('concept', 'MISSING')}")
    print(f"Label: {var_metadata.get('label', 'MISSING')}")
    print(f"Structure type: {var_metadata.get('structure_type', 'MISSING')}")
    print(f"Has summary: {var_metadata.get('has_summary', False)}")
    print(f"Summary length: {var_metadata.get('summary_length', 0)}")
    print(f"Has enrichment: {var_metadata.get('has_full_enrichment', False)}")
    print(f"Enrichment length: {var_metadata.get('enrichment_length', 0)}")
    print(f"Key terms count: {var_metadata.get('key_terms_count', 0)}")
    
    # Check if we can reconstruct the embedding text
    print(f"\n=== RECONSTRUCTED EMBEDDING TEXT ===")
    
    # Try to recreate what the build script would have generated
    # This is based on the _create_concept_embedding_text method
    
    # Check if this is from canonical_variables_refactored.json
    print("Attempting to reverse-engineer the embedding text...")
    
    # Look for the source file to get the original data
    source_file = Path("knowledge-base/source-docs/canonical_variables_refactored.json")
    if source_file.exists():
        print("✅ Found canonical_variables_refactored.json - checking original data...")
        try:
            with open(source_file) as f:
                # This might be huge, so let's be careful
                print("Loading canonical data... (this may take a moment)")
                canonical_data = json.load(f)
            
            # Check structure
            if 'concepts' in canonical_data:
                concepts = canonical_data['concepts']
                print(f"Found concepts structure with {len(concepts)} variables")
            else:
                concepts = {k: v for k, v in canonical_data.items() 
                          if k != 'metadata' and isinstance(v, dict)}
                print(f"Using root-level structure with {len(concepts)} variables")
            
            # Get B19013_001E data
            if target_var in concepts:
                concept_data = concepts[target_var]
                print(f"✅ Found {target_var} in canonical data")
                
                # Reconstruct the embedding text
                parts = []
                
                # 1. Summary first
                summary = concept_data.get('summary', '')
                if summary:
                    parts.append(summary)
                    print(f"📝 Summary ({len(summary)} chars): {summary[:200]}...")
                
                # 2. Key terms
                key_terms = concept_data.get('key_terms', [])
                if key_terms:
                    if summary:
                        summary_lower = summary.lower()
                        unique_terms = [term for term in key_terms if term.lower() not in summary_lower]
                        if unique_terms:
                            parts.append(f"Key search terms: {', '.join(unique_terms)}")
                    else:
                        parts.append(f"Key search terms: {', '.join(key_terms)}")
                    print(f"🔑 Key terms: {key_terms}")
                
                # 3. Census identifiers
                parts.append(f"Census variable identifier: {target_var}")
                concept = concept_data.get('concept', 'Unknown')
                label = concept_data.get('label', 'Unknown')
                if concept != 'Unknown':
                    parts.append(f"Official Census concept: {concept}")
                if label != 'Unknown':
                    parts.append(f"Official Census label: {label}")
                
                # 4. Enrichment text
                enrichment = concept_data.get('enrichment_text', '')
                if enrichment:
                    parts.append(enrichment)
                    print(f"📚 Enrichment ({len(enrichment)} chars): {enrichment[:200]}...")
                
                # 5. Show the final embedding text
                embedding_text = ". ".join(parts) + "."
                print(f"\n=== FINAL EMBEDDING TEXT ===")
                print(f"Length: {len(embedding_text)} characters")
                print(f"Text: {embedding_text[:500]}...")
                
                # Check if it contains income-related terms
                income_terms = ['median household income', 'household income', 'income', 'median income']
                found_terms = [term for term in income_terms if term in embedding_text.lower()]
                print(f"\n🔍 Income-related terms found: {found_terms}")
                
                if not found_terms:
                    print("⚠️  WARNING: No income-related terms found in embedding text!")
                    print("This explains why search for 'median household income' fails.")
                
            else:
                print(f"❌ {target_var} not found in canonical concepts")
                
        except Exception as e:
            print(f"Error loading canonical data: {e}")
    else:
        print("❌ Cannot find canonical_variables_refactored.json")
    
    # Also check what other income variables exist
    print(f"\n=== OTHER INCOME VARIABLES ===")
    income_vars = [v for v in variable_ids if 'B19' in v and ('001E' in v or '002E' in v)]
    print(f"Found {len(income_vars)} potential income variables:")
    for var in income_vars[:10]:  # Show first 10
        try:
            pos = variable_ids.index(var)
            meta = metadata_array[pos]
            concept = meta.get('concept', 'Unknown')
            label = meta.get('label', 'Unknown')
            print(f"  {var}: {concept} - {label}")
        except:
            print(f"  {var}: metadata error")

if __name__ == "__main__":
    check_b19013_embedding_text()
