#!/usr/bin/env python3
"""
Quick script to check the actual structure of canonical_variables_refactored.json
"""

import json
from pathlib import Path

def check_canonical_structure():
    canonical_path = Path("knowledge-base/source-docs/canonical_variables_refactored.json")
    
    print("Checking canonical file structure...")
    print(f"File exists: {canonical_path.exists()}")
    
    if not canonical_path.exists():
        print("❌ File not found")
        return
    
    try:
        with open(canonical_path) as f:
            # Read just the beginning to see structure
            first_chunk = f.read(2000)
            print("\nFirst 2000 characters:")
            print(first_chunk)
            
            # Reset and load the actual structure
            f.seek(0)
            data = json.load(f)
            
            print(f"\n=== ACTUAL STRUCTURE ===")
            print(f"Top-level keys: {list(data.keys())}")
            
            # Check if 'concepts' key exists
            if 'concepts' in data:
                concepts = data['concepts']
                print(f"Found 'concepts' key with {len(concepts)} items")
                
                # Check for B19013_001E specifically
                if 'B19013_001E' in concepts:
                    print("✅ B19013_001E found in concepts")
                    sample_data = concepts['B19013_001E']
                    print(f"Sample B19013_001E data keys: {list(sample_data.keys())}")
                else:
                    print("❌ B19013_001E NOT found in concepts")
                    # Show first few concept keys
                    concept_keys = list(concepts.keys())
                    print(f"First 10 concept keys: {concept_keys[:10]}")
                    
                    # Check for any B19013 variables
                    b19013_vars = [k for k in concept_keys if k.startswith('B19013')]
                    print(f"B19013 variables found: {b19013_vars}")
            else:
                print("No 'concepts' key found")
                # Check if variables are at root level
                non_metadata_keys = [k for k in data.keys() if k != 'metadata' and isinstance(data[k], dict)]
                print(f"Non-metadata dict keys: {len(non_metadata_keys)}")
                if non_metadata_keys:
                    print(f"First few keys: {non_metadata_keys[:10]}")
                    
                    # Check for B19013_001E at root level
                    if 'B19013_001E' in data:
                        print("✅ B19013_001E found at root level")
                    else:
                        print("❌ B19013_001E NOT found at root level")
                        b19013_vars = [k for k in non_metadata_keys if k.startswith('B19013')]
                        print(f"B19013 variables found: {b19013_vars}")
                        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_canonical_structure()
