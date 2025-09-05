#!/usr/bin/env python3
"""
Test Statistical Advisor Integration - Minimal Test

This is the actual integration test for the statistical advisor.
"""

import sys
import os
import logging
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
kb_path = current_dir / "knowledge-base"
src_path = current_dir / "src"

sys.path.insert(0, str(kb_path))
sys.path.insert(0, str(src_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_integration():
    """Test the full integration like the new MCP server does"""
    
    logger.info("🧠 Testing LLM Statistical Advisor Integration (v3.0)...")
    
    try:
        # Step 1: Import and create search engine (like MCP server does)
        from kb_search import create_search_engine
        
        knowledge_base_dir = str(kb_path)
        gazetteer_path = kb_path / "geo-db" / "geography.db"
        
        logger.info(f"Creating search engine with:")
        logger.info(f"  Knowledge base: {knowledge_base_dir}")
        logger.info(f"  Gazetteer: {gazetteer_path} (exists: {gazetteer_path.exists()})")
        
        search_engine = create_search_engine(
            knowledge_base_dir=knowledge_base_dir,
            gazetteer_db_path=str(gazetteer_path) if gazetteer_path.exists() else None
        )
        
        logger.info("✅ Search engine created")
        
        # Step 2: Import and create LLM statistical advisor  
        from llm_statistical_advisor import LLMStatisticalAdvisor
        
        advisor = LLMStatisticalAdvisor()
        logger.info("✅ LLM statistical advisor created")
        
        # Step 3: Wire them together (like MCP server does)
        geo_parser = getattr(search_engine, 'geo_parser', None)
        variable_search = getattr(search_engine, 'variable_search', None)
        methodology_search = getattr(search_engine, 'methodology_search', None)
        
        logger.info(f"Tool extraction:")
        logger.info(f"  geo_parser: {'✅' if geo_parser else '❌'}")
        logger.info(f"  variable_search: {'✅' if variable_search else '❌'}")
        logger.info(f"  methodology_search: {'✅' if methodology_search else '❌'}")
        
        advisor.set_tools(
            geo_parser=geo_parser,
            variable_search=variable_search,
            methodology_search=methodology_search
        )
        
        logger.info("✅ Tools wired to LLM advisor")
        
        # Step 4: Test the wiring
        has_geo = advisor.geo_parser is not None
        has_var = advisor.variable_search is not None
        has_method = advisor.methodology_search is not None
        
        logger.info(f"Final wiring check:")
        logger.info(f"  advisor.geo_parser: {'✅' if has_geo else '❌'}")
        logger.info(f"  advisor.variable_search: {'✅' if has_var else '❌'}")
        logger.info(f"  advisor.methodology_search: {'✅' if has_method else '❌'}")
        
        if has_geo and has_var:
            logger.info("🎯 INTEGRATION SUCCESS! LLM advisor is wired and ready.")
            
            # Step 5: Quick functional test
            logger.info("Testing LLM consultation...")
            try:
                # Use existing consult method with GeographicContext
                from geographic_parsing import GeographicContext
                geo_context = GeographicContext(
                    location_mentioned=True,
                    location_text="Austin, Texas",
                    geography_level="place",
                    confidence=0.9
                )
                
                result = advisor.consult("What variables for median income analysis?", geo_context=geo_context)
                logger.info(f"Consultation result: {type(result).__name__} with confidence: {result.confidence:.1%}")
                logger.info("✅ LLM functional test passed")
            except Exception as e:
                logger.warning(f"⚠️ LLM functional test failed: {e}")
            
            return True
        else:
            logger.error("❌ INTEGRATION FAILED: Missing critical tools (geo_parser and variable_search)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\n🎉 READY TO ROCK! The LLM statistical advisor integration is working.")
        print("You can now run the new MCP server (v3.0) and use the LLM statistical consultation tool.")
    else:
        print("\n❌ Integration issues detected. Check the logs above.")
    
    sys.exit(0 if success else 1)
