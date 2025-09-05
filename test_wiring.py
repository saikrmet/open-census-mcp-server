#!/usr/bin/env python3
"""
Test script to verify statistical advisor wiring
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Add knowledge-base directory to path
kb_path = Path(__file__).parent / "knowledge-base"
sys.path.insert(0, str(kb_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if we can import all required components"""
    try:
        from llm_statistical_advisor import create_llm_statistical_advisor
        logger.info("✅ LLM Statistical Advisor import successful")
    except Exception as e:
        logger.error(f"❌ LLM Statistical Advisor import failed: {e}")
        return False
    
    try:
        from kb_search import create_search_engine
        logger.info("✅ Search engine import successful")
    except Exception as e:
        logger.error(f"❌ Search engine import failed: {e}")
        return False
    
    try:
        from geographic_parsing import create_geographic_parser
        logger.info("✅ Geographic parsing import successful")
    except Exception as e:
        logger.error(f"❌ Geographic parsing import failed: {e}")
        return False
    
    return True

def test_search_engine_creation():
    """Test search engine creation"""
    try:
        from kb_search import create_search_engine
        
        # Try to create search engine
        kb_dir = Path(__file__).parent / "knowledge-base"
        gazetteer_path = kb_dir / "geo-db" / "geography.db"
        
        logger.info(f"Knowledge base dir: {kb_dir} (exists: {kb_dir.exists()})")
        logger.info(f"Gazetteer path: {gazetteer_path} (exists: {gazetteer_path.exists()})")
        
        search_engine = create_search_engine(
            knowledge_base_dir=str(kb_dir),
            gazetteer_db_path=str(gazetteer_path) if gazetteer_path.exists() else None
        )
        
        logger.info("✅ Search engine created successfully")
        
        # Test if it has the expected attributes
        has_geo_parser = hasattr(search_engine, 'geo_parser')
        has_variable_search = hasattr(search_engine, 'variable_search')
        has_methodology_search = hasattr(search_engine, 'methodology_search')
        
        logger.info(f"Search engine attributes:")
        logger.info(f"  geo_parser: {'✅' if has_geo_parser else '❌'}")
        logger.info(f"  variable_search: {'✅' if has_variable_search else '❌'}")
        logger.info(f"  methodology_search: {'✅' if has_methodology_search else '❌'}")
        
        return search_engine, has_geo_parser and has_variable_search
        
    except Exception as e:
        logger.error(f"❌ Search engine creation failed: {e}")
        return None, False

def test_statistical_advisor_creation():
    """Test statistical advisor creation"""
    try:
        from llm_statistical_advisor import create_llm_statistical_advisor
        
        advisor = create_llm_statistical_advisor()
        logger.info("✅ Statistical advisor created successfully")
        
        # Test if it has the expected methods
        has_set_tools = hasattr(advisor, 'set_tools')
        has_consult = hasattr(advisor, 'consult')
        
        logger.info(f"Statistical advisor methods:")
        logger.info(f"  set_tools: {'✅' if has_set_tools else '❌'}")
        logger.info(f"  consult: {'✅' if has_consult else '❌'}")
        
        return advisor, has_set_tools and has_consult
        
    except Exception as e:
        logger.error(f"❌ Statistical advisor creation failed: {e}")
        return None, False

def test_wiring():
    """Test the complete wiring"""
    logger.info("🔧 Testing statistical advisor wiring...")
    
    # Create search engine
    search_engine, search_ok = test_search_engine_creation()
    if not search_ok:
        logger.error("❌ Cannot test wiring without functional search engine")
        return False
    
    # Create statistical advisor
    advisor, advisor_ok = test_statistical_advisor_creation()
    if not advisor_ok:
        logger.error("❌ Cannot test wiring without functional statistical advisor")
        return False
    
    # Test wiring
    try:
        advisor.set_tools(
            geo_parser=search_engine.geo_parser,
            variable_search=search_engine.variable_search,
            methodology_search=search_engine.methodology_search
        )
        logger.info("✅ Tools wired successfully")
        
        # Test that tools are actually set
        has_geo = advisor.geo_parser is not None
        has_var = advisor.variable_search is not None
        has_method = advisor.methodology_search is not None
        
        logger.info(f"Wiring verification:")
        logger.info(f"  geo_parser connected: {'✅' if has_geo else '❌'}")
        logger.info(f"  variable_search connected: {'✅' if has_var else '❌'}")
        logger.info(f"  methodology_search connected: {'✅' if has_method else '❌'}")
        
        if has_geo and has_var:
            logger.info("🎯 Statistical advisor fully wired and ready!")
            return True
        else:
            logger.error("❌ Wiring incomplete - some tools missing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Wiring failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Testing statistical advisor wiring components...")
    
    # Test 1: Imports
    if not test_imports():
        logger.error("❌ Import test failed - cannot proceed")
        return False
    
    # Test 2: Complete wiring
    if test_wiring():
        logger.info("🎉 All tests passed! Statistical advisor is ready for integration.")
        return True
    else:
        logger.error("❌ Wiring test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
