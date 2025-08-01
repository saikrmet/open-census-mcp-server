#!/usr/bin/env python3
"""
Post-build validation script for Census MCP variables database

Usage:
    python validate_kb_concept_based_build.py --variables-dir variables-db [--test-searches]
    python validate_kb_concept_based_build.py --variables-dir variables-db --test-searches --specific-tests
"""

import json
import sys
import argparse
from pathlib import Path
import faiss
import numpy as np

def validate_array_synchronization(variables_dir: Path) -> bool:
    """Validate that all arrays are synchronized"""
    print("🔍 Validating array synchronization...")
    
    # Load all files
    ids_file = variables_dir / "variables_ids.json"
    metadata_file = variables_dir / "variables_metadata.json"
    faiss_file = variables_dir / "variables.faiss"
    
    if not all(f.exists() for f in [ids_file, metadata_file, faiss_file]):
        missing = [f.name for f in [ids_file, metadata_file, faiss_file] if not f.exists()]
        print(f"❌ Missing required files: {missing}")
        return False
    
    # Load data
    with open(ids_file) as f:
        ids_data = json.load(f)
        variable_ids = ids_data['variable_ids']
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    index = faiss.read_index(str(faiss_file))
    
    # Check lengths
    if not (len(variable_ids) == len(metadata) == index.ntotal):
        print(f"❌ Length mismatch: ids={len(variable_ids)}, metadata={len(metadata)}, index={index.ntotal}")
        return False
    
    # Spot check metadata alignment
    sample_indices = [0, len(variable_ids)//2, -1]
    for i in sample_indices:
        expected_id = variable_ids[i]
        actual_id = metadata[i].get('variable_id')
        if expected_id != actual_id:
            print(f"❌ Metadata mismatch at position {i}: expected {expected_id}, got {actual_id}")
            return False
    
    print(f"✅ Array synchronization validated: {len(variable_ids)} variables")
    return True

def validate_search_functionality(variables_dir: Path) -> bool:
    """Test basic search functionality"""
    print("🔍 Validating search functionality...")
    
    try:
        # Add parent directory to path for imports
        sys.path.append(str(variables_dir.parent))
        from kb_search import ConceptBasedCensusSearchEngine
        
        engine = ConceptBasedCensusSearchEngine(
            catalog_dir=str(variables_dir.parent / "table-catalog"),
            variables_dir=str(variables_dir)
        )
        
        # Test searches
        test_queries = [
            "income",
            "population",
            "median household income",
            "unemployment rate"
        ]
        
        for query in test_queries:
            results = engine.search(query, max_results=5)
            if not results:
                print(f"❌ No results for '{query}'")
                return False
            print(f"✅ '{query}': {len(results)} results, top: {results[0].variable_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Search validation failed: {e}")
        return False

def validate_specific_variables(variables_dir: Path) -> bool:
    """Test specific variables that were problematic"""
    print("🔍 Validating specific known variables...")
    
    try:
        # Add parent directory to path for imports
        sys.path.append(str(variables_dir.parent))
        from kb_search import ConceptBasedCensusSearchEngine
        
        engine = ConceptBasedCensusSearchEngine(
            catalog_dir=str(variables_dir.parent / "table-catalog"),
            variables_dir=str(variables_dir)
        )
        
        # Test the specific problem case: B19013_001E
        print("  Testing B19013_001E (median household income)...")
        
        # Test 1: Direct variable lookup
        try:
            var_info = engine.get_variable_info('B19013_001E')
            if var_info:
                print(f"  ✅ Direct lookup works: {var_info.get('concept_name', 'Unknown')}")
            else:
                print(f"  ❌ Direct lookup failed for B19013_001E")
                return False
        except Exception as e:
            print(f"  ❌ Direct lookup error: {e}")
            return False
        
        # Test 2: Search for median household income
        results = engine.search('median household income', max_results=10)
        b19013_rank = next((i for i, r in enumerate(results) if r.variable_id == 'B19013_001E'), -1)
        
        if b19013_rank == -1:
            print(f"  ❌ B19013_001E not found in search for 'median household income'")
            print(f"     Top results: {[r.variable_id for r in results[:5]]}")
            return False
        elif b19013_rank <= 2:  # Should be in top 3
            print(f"  ✅ B19013_001E found at rank {b19013_rank} for 'median household income'")
        else:
            print(f"  ⚠️  B19013_001E found but at rank {b19013_rank} (expected top 3)")
        
        # Test 3: Search for the ID directly
        results = engine.search('B19013', max_results=5)
        b19013_found = any(r.variable_id == 'B19013_001E' for r in results)
        
        if b19013_found:
            print(f"  ✅ B19013_001E found when searching for 'B19013'")
        else:
            print(f"  ❌ B19013_001E not found when searching for 'B19013'")
            print(f"     Results: {[r.variable_id for r in results]}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Specific variable validation failed: {e}")
        return False

def validate_build_metadata(variables_dir: Path) -> bool:
    """Validate build metadata and files"""
    print("🔍 Validating build metadata...")
    
    build_info_file = variables_dir / "build_info.json"
    if not build_info_file.exists():
        print("❌ build_info.json missing")
        return False
    
    with open(build_info_file) as f:
        build_info = json.load(f)
    
    # Check for critical flags
    required_flags = ['has_id_mapping', 'arrays_synchronized', 'cross_validation_passed']
    for flag in required_flags:
        if not build_info.get(flag, False):
            print(f"❌ Build flag {flag} is False or missing")
            return False
    
    print("✅ Build metadata validation passed")
    print(f"   Variables: {build_info.get('variable_count', 0):,}")
    print(f"   Structure: {build_info.get('structure_type', 'unknown')}")
    print(f"   Model: {build_info.get('model_name', 'unknown')}")
    
    return True

def check_file_integrity(variables_dir: Path) -> bool:
    """Check file sizes and basic integrity"""
    print("🔍 Checking file integrity...")
    
    required_files = [
        "variables.faiss",
        "variables_metadata.json",
        "variables_ids.json",
        "build_info.json"
    ]
    
    for filename in required_files:
        file_path = variables_dir / filename
        if not file_path.exists():
            print(f"❌ Missing file: {filename}")
            return False
        
        size = file_path.stat().st_size
        if size == 0:
            print(f"❌ Empty file: {filename}")
            return False
        
        print(f"✅ {filename}: {size:,} bytes")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Validate Census MCP build')
    parser.add_argument('--variables-dir', default='variables-db', help='Variables database directory')
    parser.add_argument('--test-searches', action='store_true', help='Test search functionality')
    parser.add_argument('--specific-tests', action='store_true', help='Test specific problematic variables')
    
    args = parser.parse_args()
    variables_dir = Path(args.variables_dir)
    
    if not variables_dir.exists():
        print(f"❌ Variables directory not found: {variables_dir}")
        sys.exit(1)
    
    print(f"Validating build in: {variables_dir}")
    print("=" * 60)
    
    # Run validations
    validations = [
        ("File Integrity", lambda: check_file_integrity(variables_dir)),
        ("Build Metadata", lambda: validate_build_metadata(variables_dir)),
        ("Array Synchronization", lambda: validate_array_synchronization(variables_dir))
    ]
    
    if args.test_searches:
        validations.append(("Search Functionality", lambda: validate_search_functionality(variables_dir)))
    
    if args.specific_tests:
        validations.append(("Specific Variables", lambda: validate_specific_variables(variables_dir)))
    
    all_passed = True
    for name, validation_func in validations:
        print(f"\n=== {name} ===")
        try:
            passed = validation_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            all_passed = False
    
    print(f"\n{'=' * 60}")
    print(f"=== FINAL RESULT ===")
    if all_passed:
        print("✅ All validations passed")
        print("🚀 Build is ready for production use!")
        sys.exit(0)
    else:
        print("❌ Some validations failed")
        print("🔧 Rebuild required to fix synchronization issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
