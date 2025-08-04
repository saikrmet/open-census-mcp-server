#!/usr/bin/env python3
"""Quick fix test for import issue"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_import_fix():
    try:
        print("Testing data_retrieval import...")
        
        # First test the geographic_handler import
        print("Testing geographic_handler import...")
        from data_retrieval.geographic_handler import CompleteGeographicHandler
        print("✅ CompleteGeographicHandler imports successfully")
        
        # Now test the python_census_api import
        print("Testing python_census_api import...")
        
        # Temporarily patch the import issue
        import data_retrieval.python_census_api as api_module
        print("✅ python_census_api imports successfully")
        
        print("Testing PythonCensusAPI class...")
        api = api_module.PythonCensusAPI()
        print("✅ PythonCensusAPI initializes successfully")
        
        print("All import tests passed!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Other Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_import_fix()
