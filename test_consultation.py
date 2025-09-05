#!/usr/bin/env python3
"""
Test the MCP Statistical Consultation Tool

This script tests just the statistical consultation functionality
without running the full MCP server.
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Setup paths like the MCP server does
current_dir = Path(__file__).parent
kb_path = current_dir / "knowledge-base"
src_path = current_dir / "src"

sys.path.insert(0, str(kb_path))
sys.path.insert(0, str(src_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_statistical_consultation():
    """Test the LLM statistical consultation like the new MCP tool does"""
    
    logger.info("🧠 Testing LLM Statistical Consultation Tool (v3.0)...")
    
    try:
        # Replicate the new MCP server initialization
        from kb_search import create_search_engine
        from llm_statistical_advisor import LLMStatisticalAdvisor
        
        # Initialize search engine
        knowledge_base_dir = str(kb_path)
        gazetteer_path = kb_path / "geo-db" / "geography.db"
        
        logger.info("Initializing search engine...")
        search_engine = create_search_engine(
            knowledge_base_dir=knowledge_base_dir,
            gazetteer_db_path=str(gazetteer_path) if gazetteer_path.exists() else None
        )
        
        # Initialize LLM statistical advisor
        logger.info("Initializing LLM statistical advisor...")
        statistical_advisor = LLMStatisticalAdvisor()
        
        # Wire tools like the original MCP server
        logger.info("Wiring tools...")
        geo_parser = getattr(search_engine, 'geo_parser', None)
        variable_search = getattr(search_engine, 'variable_search', None)
        methodology_search = getattr(search_engine, 'methodology_search', None)
        
        statistical_advisor.set_tools(
            geo_parser=geo_parser,
            variable_search=variable_search,
            methodology_search=methodology_search
        )
        
        logger.info("✅ LLM statistical advisor initialized and wired")
        
        # Test consultation - replicate the original MCP tool logic
        query = "What variables should I use for median household income analysis?"
        location = "Austin, Texas"
        
        logger.info(f"Testing LLM consultation: '{query}' in {location}")
        
        # Parse geographic context like original MCP tool does
        geo_context = None
        if location and search_engine:
            try:
                geo_context = search_engine.geo_parser.parse_geographic_context(location)
                logger.info(f"Geographic context: {location} → {geo_context.geography_level if geo_context.location_mentioned else 'not found'}")
            except Exception as e:
                logger.warning(f"Geographic parsing failed: {e}")
        
        # Get consultation like original MCP tool does
        logger.info("Requesting LLM statistical consultation...")
        consultation = statistical_advisor.consult(
            query=query,
            geo_context=geo_context
        )
        
        # Display results like MCP tool would format them
        logger.info("📊 CONSULTATION RESULTS:")
        logger.info(f"Query: {consultation.query}")
        logger.info(f"Confidence: {consultation.confidence:.1%}")
        logger.info(f"Expert Advice: {consultation.expert_advice}")
        logger.info(f"Recommended Variables: {len(consultation.recommended_variables)}")
        
        for i, var_rec in enumerate(consultation.recommended_variables, 1):
            logger.info(f"  {i}. {var_rec.variable_id}: {var_rec.statistical_rationale}")
        
        logger.info(f"Geographic Guidance: {consultation.geographic_guidance}")
        logger.info(f"Limitations: {len(consultation.limitations)} noted")
        logger.info(f"Routing path: {consultation.routing_path}")
        
        logger.info("🎯 LLM STATISTICAL CONSULTATION TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM statistical consultation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the consultation test"""
    success = await test_statistical_consultation()
    
    if success:
        print("\n🎉 LLM Statistical Consultation is working!")
        print("The new MCP tool (v3.0) should work properly.")
    else:
        print("\n❌ LLM statistical consultation has issues.")
        print("Check the logs above for details.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
