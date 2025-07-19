#!/usr/bin/env python3
"""
Census MCP Server - Updated for Concept-Based Search Architecture

Key updates for v2.4:
- Uses ConceptBasedCensusSearchEngine instead of legacy search
- Updated file paths for concept-based structure
- Enhanced table catalog with keywords
- Proper error handling for missing files
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Updated imports for concept-based architecture
try:
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    sys.path.append(str(project_root / "knowledge-base"))
    
    from kb_search import ConceptBasedCensusSearchEngine  # Updated search engine
    from src.data_retrieval.python_census_api import PythonCensusAPI
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CensusMCPServer:
    """
    Census MCP server with concept-based search architecture.
    
    Key improvements:
    - Uses ConceptBasedCensusSearchEngine (no duplicates)
    - Enhanced table catalog with keywords
    - Survey instance awareness
    - Proper geographic context parsing
    """
    
    def __init__(self):
        """Initialize server with concept-based components."""
        logger.info("🚀 Initializing Census MCP Server v2.4 (Concept-Based)")
        
        # Initialize concept-based search engine
        try:
            logger.info("Loading concept-based search engine...")
            
            # Get absolute paths relative to script location
            script_dir = Path(__file__).parent  # src/
            project_root = script_dir.parent    # project root
            
            catalog_path = project_root / "knowledge-base" / "table-catalog"
            variables_path = project_root / "knowledge-base" / "variables-db"
            
            self.search_engine = ConceptBasedCensusSearchEngine(
                catalog_dir=str(catalog_path),
                variables_dir=str(variables_path)
            )
            logger.info("✅ Concept-based search engine loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load search engine: {e}")
            raise
        
        # Initialize Python Census API
        try:
            logger.info("Loading Python Census API...")
            self.census_api = PythonCensusAPI()
            logger.info("✅ Python Census API loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Census API: {e}")
            raise
        
        # Create MCP server instance
        self.server = Server("census-mcp-v2.4")
        
        # Register tools
        self._register_tools()
        
        logger.info("✅ Census MCP Server v2.4 initialized successfully")
        logger.info("🎯 Features: Concept-based search, keyword enhancement, survey intelligence")
    
    def _register_tools(self):
        """Register MCP tools with enhanced concept-based capabilities."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Return list of available Census tools."""
            return [
                Tool(
                    name="get_demographic_data",
                    description="Get Census demographic data for locations using intelligent variable resolution",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location (e.g., 'Detroit, MI', 'California', 'Harris County, TX')"
                            },
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Natural language variable descriptions (e.g., 'median household income', 'poverty rate')"
                            },
                            "survey": {
                                "type": "string",
                                "description": "ACS survey: 'acs5' (5-year, default) or 'acs1' (1-year)",
                                "default": "acs5"
                            },
                            "year": {
                                "type": "integer",
                                "description": "Data year (default: 2023)",
                                "default": 2023
                            }
                        },
                        "required": ["location", "variables"]
                    }
                ),
                Tool(
                    name="search_census_variables",
                    description="Search for Census variables using natural language with concept-based intelligence",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description of data needed"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10
                            },
                            "geographic_context": {
                                "type": "string",
                                "description": "Geographic context if relevant (e.g., 'county level', 'state level')",
                                "default": ""
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="find_census_tables",
                    description="Find Census tables using keywords and natural language",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description of table topic"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of tables to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="compare_locations",
                    description="Compare demographic data across multiple locations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "locations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of locations to compare"
                            },
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Variables to compare across locations"
                            },
                            "survey": {
                                "type": "string",
                                "description": "ACS survey type",
                                "default": "acs5"
                            }
                        },
                        "required": ["locations", "variables"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool execution with enhanced error handling."""
            try:
                if name == "get_demographic_data":
                    return await self._get_demographic_data(arguments)
                elif name == "search_census_variables":
                    return await self._search_census_variables(arguments)
                elif name == "find_census_tables":
                    return await self._find_census_tables(arguments)
                elif name == "compare_locations":
                    return await self._compare_locations(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Unknown tool: {name}"
                    )]
            except Exception as e:
                logger.error(f"❌ Error executing tool {name}: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"❌ Error executing {name}: {str(e)}\n\nThis may be due to missing data files or API issues. Please try again or contact support."
                )]
    
    async def _search_census_variables(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Search for Census variables using concept-based intelligence."""
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 10)
        geographic_context = arguments.get("geographic_context", "")
        
        if not query:
            return [TextContent(
                type="text",
                text="❌ Please provide a search query."
            )]
        
        try:
            logger.info(f"🔍 Searching variables: '{query}'")
            
            # Use concept-based search
            results = self.search_engine.search(query, max_results=max_results)
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"❌ No variables found for query: '{query}'\n\nTry using different keywords or broader terms."
                )]
            
            # Format results with enhanced metadata
            response_parts = [
                f"🎯 Found {len(results)} variables for: '{query}'\n"
            ]
            
            for i, result in enumerate(results, 1):
                response_parts.append(f"\n**{i}. {result.variable_id}**")
                response_parts.append(f"   📊 **Label**: {result.label}")
                response_parts.append(f"   🎯 **Concept**: {result.concept}")
                response_parts.append(f"   📈 **Confidence**: {result.confidence:.3f}")
                
                # Show survey availability
                if hasattr(result, 'available_surveys') and result.available_surveys:
                    surveys = ', '.join(result.available_surveys)
                    response_parts.append(f"   📅 **Available in**: {surveys}")
                
                # Show geographic levels if available
                if hasattr(result, 'geography_levels') and result.geography_levels:
                    geo_levels = ', '.join(result.geography_levels[:3])
                    response_parts.append(f"   🗺️  **Geography**: {geo_levels}")
                
                # Show table family
                table_id = result.variable_id.split('_')[0] if '_' in result.variable_id else 'Unknown'
                response_parts.append(f"   📋 **Table**: {table_id}")
            
            response_parts.append(f"\n💡 **Tip**: Use 'get_demographic_data' to retrieve actual values for these variables.")
            
            return [TextContent(
                type="text",
                text="\n".join(response_parts)
            )]
            
        except Exception as e:
            logger.error(f"Variable search error: {e}")
            return [TextContent(
                type="text",
                text=f"❌ Search error: {str(e)}"
            )]
    
    async def _find_census_tables(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Find Census tables using keyword-enhanced search."""
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        
        if not query:
            return [TextContent(
                type="text",
                text="❌ Please provide a search query."
            )]
        
        try:
            logger.info(f"🔍 Searching tables: '{query}'")
            
            # Search for tables (get mixed results and filter for tables)
            all_results = self.search_engine.search(query, max_results=max_results * 2)
            table_results = [r for r in all_results if hasattr(r, 'table_id')][:max_results]
            
            if not table_results:
                return [TextContent(
                    type="text",
                    text=f"❌ No tables found for query: '{query}'\n\nTry using different keywords."
                )]
            
            # Format table results with keywords
            response_parts = [
                f"📋 Found {len(table_results)} tables for: '{query}'\n"
            ]
            
            for i, result in enumerate(table_results, 1):
                response_parts.append(f"\n**{i}. Table {result.table_id}**")
                
                # Get table info for keywords
                table_info = self.search_engine.get_table_info(result.table_id)
                
                if table_info:
                    response_parts.append(f"   📊 **Title**: {table_info.get('title', 'Unknown')}")
                    response_parts.append(f"   👥 **Universe**: {table_info.get('universe', 'Unknown')}")
                    
                    # Show keywords if available
                    keywords = table_info.get('search_keywords', {})
                    if keywords.get('primary_keywords'):
                        kw_text = ', '.join(keywords['primary_keywords'][:3])
                        response_parts.append(f"   🔑 **Keywords**: {kw_text}")
                    
                    if keywords.get('summary'):
                        summary = keywords['summary'][:150] + "..." if len(keywords['summary']) > 150 else keywords['summary']
                        response_parts.append(f"   📝 **Summary**: {summary}")
                    
                    # Show survey availability
                    surveys = table_info.get('survey_programs', [])
                    if surveys:
                        response_parts.append(f"   📅 **Surveys**: {', '.join(surveys)}")
                
                response_parts.append(f"   📈 **Confidence**: {result.confidence:.3f}")
            
            response_parts.append(f"\n💡 **Tip**: Use table IDs with 'search_census_variables' to find specific variables.")
            
            return [TextContent(
                type="text",
                text="\n".join(response_parts)
            )]
            
        except Exception as e:
            logger.error(f"Table search error: {e}")
            return [TextContent(
                type="text",
                text=f"❌ Table search error: {str(e)}"
            )]
    
    async def _get_demographic_data(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get demographic data using intelligent variable resolution."""
        location = arguments.get("location", "")
        variables = arguments.get("variables", [])
        survey = arguments.get("survey", "acs5")
        year = arguments.get("year", 2023)
        
        if not location or not variables:
            return [TextContent(
                type="text",
                text="❌ Please provide both location and variables."
            )]
        
        try:
            logger.info(f"📊 Getting data for {location}: {variables}")
            
            # Use the working Python Census API directly
            result = await self.census_api.get_acs_data(
                location=location,
                variables=variables,
                year=year,
                survey=survey
            )
            
            # Format comprehensive response
            if result.get('error'):
                error_response = f"# 🏛️ Official Census Data for {location}\n\n"
                error_response += f"❌ **Error**: {result['error']}\n\n"
                error_response += "**Common Issues:**\n"
                error_response += "• Location spelling (try 'Baltimore, MD' instead of 'Baltimore')\n"
                error_response += "• Variable not available at this geographic level\n"
                error_response += "• Data suppressed for privacy (small sample sizes)\n\n"
                error_response += "**💡 Tip**: Use 'search_census_variables' to find available variables."
                return [TextContent(type="text", text=error_response)]
            
            # Format successful response
            response_parts = [
                f"# 🏛️ Official Census Data for {result.get('location', location)}\n",
                f"📅 Survey: {survey.upper()} {year}",
                f"🗺️  Geography: {result.get('geography_level', 'Unknown')} level\n"
            ]
            
            response_parts.append("## 📈 **Data Results:**")
            
            data = result.get('data', {})
            for var in variables:
                if var in data:
                    var_data = data[var]
                    # Get the actual estimate value - handle both string and numeric formats
                    estimate = var_data.get('estimate', 'N/A')
                    moe = var_data.get('moe', 'N/A')
                    raw_value = var_data.get('raw_value', None)
                    
                    response_parts.append(f"### {var.title()}")
                    
                    # Use raw_value if estimate is formatted as "0" but raw_value exists
                    if estimate in ['0', '0.0', 0] and raw_value and raw_value > 0:
                        # Format the raw value properly
                        if raw_value >= 1000:
                            display_value = f"${raw_value:,.0f}" if 'income' in var.lower() or 'earning' in var.lower() or 'salary' in var.lower() else f"{raw_value:,.0f}"
                        else:
                            display_value = str(raw_value)
                        response_parts.append(f"**Value**: {display_value}")
                    else:
                        response_parts.append(f"**Value**: {estimate}")
                    
                    if moe != 'N/A' and moe not in ['±0', '0']:
                        response_parts.append(f"**Margin of Error**: {moe}")
                    
                    # Show raw value for debugging if different from estimate
                    if raw_value and str(raw_value) != str(estimate):
                        response_parts.append(f"**Raw Value**: {raw_value}")
                    
                    response_parts.append("")
                else:
                    response_parts.append(f"### {var.title()}")
                    response_parts.append("**Status**: Data not available")
                    response_parts.append("")
            
            # Add resolution metadata if available
            if 'semantic_intelligence' in result and result['semantic_intelligence']:
                response_parts.append("## 🎯 **Variable Resolution:**")
                for intel in result['semantic_intelligence']:
                    var_name = intel.get('variable', '')
                    census_var = intel.get('census_variable', [])
                    confidence = intel.get('confidence', 0)
                    method = intel.get('resolution_method', 'unknown')
                    
                    if census_var:
                        response_parts.append(f"• **{var_name}** → {', '.join(census_var)} (confidence: {confidence:.2f}, method: {method})")
                response_parts.append("")
            
            # Add methodological context
            response_parts.extend([
                "---",
                "## 🏛️ **Official Data Source**",
                "**Source**: US Census Bureau American Community Survey",
                "**Authority**: Official demographic statistics with scientific sampling",
                "**Quality**: All estimates include margins of error at 90% confidence level",
                "**Note**: Small differences may not be statistically significant"
            ])
            
            return [TextContent(
                type="text",
                text="\n".join(response_parts)
            )]
            
        except Exception as e:
            logger.error(f"Data retrieval error: {e}")
            return [TextContent(
                type="text",
                text=f"❌ Data retrieval error: {str(e)}"
            )]
    
    async def _compare_locations(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Compare data across multiple locations."""
        locations = arguments.get("locations", [])
        variables = arguments.get("variables", [])
        survey = arguments.get("survey", "acs5")
        
        if len(locations) < 2:
            return [TextContent(
                type="text",
                text="❌ Please provide at least 2 locations to compare."
            )]
        
        if not variables:
            return [TextContent(
                type="text",
                text="❌ Please provide variables to compare."
            )]
        
        try:
            logger.info(f"📊 Comparing {len(locations)} locations for {len(variables)} variables")
            
            # Get data for each location using the working Python API
            comparison_data = {}
            
            for location in locations:
                result = await self.census_api.get_acs_data(
                    location=location,
                    variables=variables,
                    survey=survey
                )
                comparison_data[location] = result
            
            # Format comparison
            response_parts = [
                f"# 📊 **Location Comparison** ({survey.upper()})",
                f"📍 Locations: {', '.join(locations)}",
                f"📈 Variables: {', '.join(variables)}\n"
            ]
            
            # Create comparison table for each variable
            for var in variables:
                response_parts.append(f"## {var.title()}")
                response_parts.append("| Location | Value | Margin of Error | Status |")
                response_parts.append("|----------|-------|-----------------|--------|")
                
                for location in locations:
                    result = comparison_data.get(location, {})
                    data = result.get('data', {})
                    
                    if result.get('error'):
                        response_parts.append(f"| {location} | Error | - | {result['error'][:50]}... |")
                    elif var in data:
                        var_data = data[var]
                        estimate = var_data.get('estimate', 'N/A')
                        raw_value = var_data.get('raw_value', None)
                        moe = var_data.get('moe', 'N/A')
                        
                        # Use raw_value if estimate shows as "0" but raw_value exists
                        if estimate in ['0', '0.0', 0] and raw_value and raw_value > 0:
                            if raw_value >= 1000:
                                display_value = f"${raw_value:,.0f}" if 'income' in var.lower() or 'earning' in var.lower() or 'salary' in var.lower() else f"{raw_value:,.0f}"
                            else:
                                display_value = str(raw_value)
                            response_parts.append(f"| {location} | {display_value} | {moe} | ✅ |")
                        else:
                            response_parts.append(f"| {location} | {estimate} | {moe} | ✅ |")
                    else:
                        response_parts.append(f"| {location} | N/A | - | No data |")
                
                response_parts.append("")  # Add spacing
            
            response_parts.extend([
                "---",
                "**Source**: US Census Bureau American Community Survey",
                "**Note**: All estimates include margins of error. Statistical significance testing recommended for comparisons."
            ])
            
            return [TextContent(
                type="text",
                text="\n".join(response_parts)
            )]
            
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            return [TextContent(
                type="text",
                text=f"❌ Comparison error: {str(e)}"
            )]

async def main():
    """Main server entry point."""
    try:
        # Initialize server
        server = CensusMCPServer()
        
        # Run stdio server
        async with stdio_server() as (read_stream, write_stream):
            logger.info("🚀 Census MCP Server v2.4 running...")
            await server.server.run(
                read_stream,
                write_stream,
                server.server.create_initialization_options()
            )
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
