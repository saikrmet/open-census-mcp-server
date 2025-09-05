#!/usr/bin/env python3
"""
Clean Census MCP Server - Working Tools Only

Removed the broken get_census_data tool. Uses only reliable two-step workflow:
1. resolve_geography() - Find FIPS codes for locations
2. get_demographic_data() - Get Census data with FIPS codes
3. search_census_variables() - Find variable IDs (when available)
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add knowledge-base directory to Python path FIRST
current_dir = Path(__file__).parent
knowledge_base_dir = current_dir.parent / "knowledge-base"
sys.path.insert(0, str(knowledge_base_dir))

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Local imports with proper error handling
try:
    # Change working directory to knowledge-base for imports
    original_cwd = os.getcwd()
    os.chdir(str(knowledge_base_dir))
    
    from kb_search import ConceptBasedCensusSearchEngine
    
    # Change back to original directory
    os.chdir(original_cwd)
    
    logger.info("✅ Successfully imported ConceptBasedCensusSearchEngine")
    
except ImportError as e:
    logger.error(f"Failed to import ConceptBasedCensusSearchEngine: {e}")
    ConceptBasedCensusSearchEngine = None

try:
    from data_retrieval.python_census_api import PythonCensusAPI
    from data_retrieval.geographic_handler import GeographicHandler
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Create stub classes for testing
    class PythonCensusAPI:
        def get_acs_data(self, **kwargs):
            return {"error": "PythonCensusAPI not available"}
    
    class GeographicHandler:
        def __init__(self, db_path):
            pass
        def search_locations(self, location, max_results=10):
            return []

class CensusMCPServer:
    """
    Clean Census MCP server with reliable tools only.
    
    Working tools:
    - resolve_geography: Find FIPS codes for locations
    - get_demographic_data: Get Census data using FIPS codes  
    - search_census_variables: Find variable IDs (when available)
    """
    
    def __init__(self):
        """Initialize server components."""
        logger.info("Initializing Clean Census MCP Server...")
        
        try:
            geography_db_path = self._find_geography_db()
            logger.info(f"Using geography database: {geography_db_path}")
            self.geo_handler = GeographicHandler(geography_db_path)
        except Exception as e:
            logger.error(f"Could not initialize geographic handler: {e}")
            self.geo_handler = None
        
        try:
            logger.info("Initializing concept-based search engine...")
            if ConceptBasedCensusSearchEngine:
                self.search_engine = ConceptBasedCensusSearchEngine(
                    catalog_dir=str(knowledge_base_dir / "table-catalog"),
                    variables_dir=str(knowledge_base_dir / "variables-db"),
                    methodology_dir=str(knowledge_base_dir / "methodology-db")
                )
                logger.info("✅ Concept-based search engine initialized")
            else:
                self.search_engine = None
                logger.warning("Concept-based search engine not available")
        except Exception as e:
            logger.error(f"Could not initialize search engine: {e}")
            self.search_engine = None
        
        try:
            logger.info("Initializing Python Census API...")
            self.census_api = PythonCensusAPI()
        except Exception as e:
            logger.error(f"Could not initialize Census API: {e}")
            self.census_api = None
        
        # Create MCP server instance
        self.server = Server("census-mcp")
        
        # Register tools
        self._register_tools()
        
        logger.info("✅ Clean Census MCP Server initialized")
    
    def _find_geography_db(self) -> str:
        """Find geography database in correct location."""
        current_dir = Path(__file__).parent
        possible_paths = [
            current_dir.parent / "knowledge-base" / "geo-db" / "geography.db",
            current_dir.parent / "knowledge-base" / "geography.db",
            current_dir / "geography.db",
            Path("./geography.db")
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found geography database: {path}")
                return str(path.resolve())
        
        default_path = current_dir.parent / "knowledge-base" / "geo-db" / "geography.db"
        logger.warning(f"Geography database not found, using default: {default_path}")
        return str(default_path)
    
    def _register_tools(self):
        """Register working tools only - removed broken get_census_data."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Return list of working Census tools."""
            tools = [
                Tool(
                    name="resolve_geography",
                    description="Find FIPS codes and geographic identifiers for locations. Use this first to get proper codes for Census queries.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to resolve (e.g. 'Houston, TX', 'California', 'Kings County, NY')"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10
                            }
                        },
                        "required": ["location"]
                    }
                ),
                Tool(
                    name="get_demographic_data",
                    description="Get Census demographic data using FIPS codes. Supports single locations and batch queries (use '*' for all counties/places in a state). Always use resolve_geography first to get FIPS codes.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Census variable IDs (e.g. 'B19013_001E' for median income, 'B01003_001E' for population)"
                            },
                            "geography_type": {
                                "type": "string",
                                "description": "Type of geography: place, county, state, cbsa, zcta, us",
                                "enum": ["place", "county", "state", "cbsa", "zcta", "us"]
                            },
                            "state_fips": {"type": "string", "description": "2-digit state FIPS code (required for most geographies)"},
                            "place_fips": {"type": "string", "description": "5-digit place FIPS code (for cities/towns) or '*' for all places in state"},
                            "county_fips": {"type": "string", "description": "3-digit county FIPS code or '*' for all counties in state"},
                            "cbsa_code": {"type": "string", "description": "5-digit CBSA code (metro areas)"},
                            "zcta_code": {"type": "string", "description": "5-digit ZCTA code (ZIP code areas)"}
                        },
                        "required": ["variables", "geography_type"]
                    }
                )
            ]
            
            # Add variable search if available
            if self.search_engine:
                tools.append(Tool(
                    name="search_census_variables",
                    description="Search for Census variable IDs by concept or keyword. Use when you need to find the right variable codes for demographic data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description of desired data (e.g. 'median income', 'poverty rate', 'education')"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum results to return",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                ))
            
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "resolve_geography":
                    return await self._resolve_geography(arguments)
                elif name == "get_demographic_data":
                    return await self._get_demographic_data(arguments)
                elif name == "search_census_variables":
                    return await self._search_census_variables(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Unknown tool: {name}"
                    )]
            except Exception as e:
                logger.error(f"Tool error in {name}: {e}")
                return [TextContent(
                    type="text",
                    text=f"❌ Error in {name}: {str(e)}"
                )]
    
    async def _resolve_geography(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Resolve geography using gazetteer database."""
        location = arguments.get("location", "").strip()
        max_results = arguments.get("max_results", 10)
        
        if not location:
            return [TextContent(type="text", text="Please provide a location to search for.")]
        
        if not self.geo_handler:
            return [TextContent(type="text", text="Geographic handler not available.")]
        
        logger.info(f"Resolving geography: '{location}'")
        
        try:
            matches = self.geo_handler.search_locations(location, max_results)
            
            if not matches:
                return [TextContent(
                    type="text",
                    text=f"No geographic matches found for '{location}'. Try adding state context or using full place names."
                )]
            
            # Format results with comprehensive geographic metadata
            response_parts = [f"**Geographic Resolution Results for '{location}':**\n"]
            
            for i, match in enumerate(matches, 1):
                response_parts.append(f"**{i}. {match['name']} ({match['geography_type'].title()})**")
                if match.get('state_abbrev'):
                    response_parts.append(f"   **State**: {match['state_abbrev']}")
                response_parts.append(f"   **Confidence**: {match['confidence']:.3f}")
                
                # Comprehensive geographic identifiers
                identifiers = []
                if match.get('state_fips'):
                    identifiers.append(f"state_fips: '{match['state_fips']}'")
                if match.get('place_fips'):
                    identifiers.append(f"place_fips: '{match['place_fips']}'")
                if match.get('county_fips'):
                    identifiers.append(f"county_fips: '{match['county_fips']}'")
                if match.get('cbsa_code'):
                    identifiers.append(f"cbsa_code: '{match['cbsa_code']}'")
                if match.get('zcta_code'):
                    identifiers.append(f"zcta_code: '{match['zcta_code']}'")
                
                if identifiers:
                    response_parts.append(f"   **Geographic Codes**: {', '.join(identifiers)}")
                response_parts.append("")
            
            response_parts.extend([
                "**Comprehensive Data Retrieval Options:**",
                "",
                "**Single Location Analysis:**",
                "```",
                f"get_demographic_data(",
                f"  geography_type='{matches[0]['geography_type']}',",
                f"  state_fips='{matches[0].get('state_fips', 'XX')}',",
            ])
            
            if matches[0].get('place_fips'):
                response_parts.append(f"  place_fips='{matches[0]['place_fips']}',")
            elif matches[0].get('county_fips'):
                response_parts.append(f"  county_fips='{matches[0]['county_fips']}',")
            elif matches[0].get('cbsa_code'):
                response_parts.append(f"  cbsa_code='{matches[0]['cbsa_code']}',")
            
            response_parts.extend([
                f"  variables=['B19013_001E', 'B01003_001E']  # Income + Population",
                ")",
                "```",
                "",
                "**Batch Analysis (All Counties/Places in State):**",
                "```",
                f"get_demographic_data(",
                f"  geography_type='county',",
                f"  state_fips='{matches[0].get('state_fips', 'XX')}',",
                f"  county_fips='*',  # All counties - systematic analysis",
                f"  variables=['B17001_002E']  # Poverty data for ranking",
                ")",
                "```",
                "",
                "**Variable Discovery:**",
                "```",
                "search_census_variables('median household income')",
                "search_census_variables('educational attainment')",
                "search_census_variables('housing costs')",
                "```"
            ])
            
            return [TextContent(type="text", text="\n".join(response_parts))]
        
        except Exception as e:
            logger.error(f"Geography resolution error: {e}")
            return [TextContent(type="text", text=f"Error resolving geography: {str(e)}")]
    
    async def _get_demographic_data(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get demographic data using resolved FIPS codes. Supports batch queries."""
        if not self.census_api:
            return [TextContent(type="text", text="Census API not available.")]
        
        variables = arguments.get("variables", [])
        geography_type = arguments.get("geography_type", "")
        
        if not variables or not geography_type:
            return [TextContent(type="text", text="Please provide both variables and geography_type.")]
        
        # Build geography parameters
        geo_params = {"geography_type": geography_type}
        
        if geography_type == "place":
            state_fips = arguments.get("state_fips")
            place_fips = arguments.get("place_fips")
            if not state_fips or not place_fips:
                return [TextContent(type="text", text="Place geography requires both state_fips and place_fips.")]
            geo_params.update({"state_fips": state_fips, "place_fips": place_fips})
        
        elif geography_type == "county":
            state_fips = arguments.get("state_fips")
            county_fips = arguments.get("county_fips")
            if not state_fips or not county_fips:
                return [TextContent(type="text", text="County geography requires both state_fips and county_fips.")]
            geo_params.update({"state_fips": state_fips, "county_fips": county_fips})
        
        elif geography_type == "state":
            state_fips = arguments.get("state_fips")
            if not state_fips:
                return [TextContent(type="text", text="State geography requires state_fips.")]
            geo_params.update({"state_fips": state_fips})
        
        elif geography_type == "cbsa":
            cbsa_code = arguments.get("cbsa_code")
            if not cbsa_code:
                return [TextContent(type="text", text="CBSA geography requires cbsa_code.")]
            geo_params.update({"cbsa_code": cbsa_code})
        
        elif geography_type == "zcta":
            zcta_code = arguments.get("zcta_code")
            if not zcta_code:
                return [TextContent(type="text", text="ZCTA geography requires zcta_code.")]
            geo_params.update({"zcta_code": zcta_code})
        
        elif geography_type == "us":
            pass
        
        logger.info(f"Getting demographic data: {variables} for {geo_params}")
        
        try:
            results = self.census_api.get_acs_data(variables=variables, **geo_params)
            
            if "error" in results:
                return [TextContent(type="text", text=f"Census API error: {results['error']}")]
            
            # Format response - handle both single and batch queries
            response_parts = [
                f"**Official Census Data Results**",
                f"**Geography**: {geography_type.title()}",
            ]
            
            if results.get("data"):
                # Check if this is a batch query
                if results.get("batch_query"):
                    # Format batch results
                    total_geos = results.get("total_geographies", 0)
                    response_parts.extend([
                        f"**Batch Query**: {total_geos} geographies (sorted by first variable)",
                        f"**Location**: {results.get('location_name', 'Multiple geographies')}",
                        "",
                        "| Rank | Location | Value | Variable |",
                        "|------|----------|-------|----------|"
                    ])
                    
                    for rank, geography in enumerate(results["data"], 1):
                        location_name = geography["location_name"]
                        # Get first variable for display
                        for var_id, var_data in geography["data"].items():
                            estimate = var_data.get("estimate", "N/A")
                            if isinstance(estimate, (int, float)) and estimate != "N/A":
                                estimate = f"{estimate:,}"
                            label = var_data.get("label", var_id)
                            response_parts.append(f"| {rank} | {location_name} | {estimate} | {label} |")
                            break  # Only show first variable in batch summary
                    
                    # Add note about additional variables
                    if len(variables) > 1:
                        response_parts.extend([
                            "",
                            f"**Note**: Showing first variable ({variables[0]}) for ranking. Full data includes {len(variables)} variables per geography."
                        ])
                    
                else:
                    # Format single geography results
                    response_parts.append(f"**Location**: {results.get('location_name', 'Unknown')}")
                    response_parts.extend([
                        "",
                        "| Variable | Value | Margin of Error |",
                        "|----------|--------|-----------------|"
                    ])
                    
                    for var_id, var_data in results["data"].items():
                        if isinstance(var_data, dict):
                            estimate = var_data.get("estimate", "N/A")
                            moe = var_data.get("moe", "N/A")
                            label = var_data.get("label", var_id)
                            
                            # Format numbers with commas
                            if isinstance(estimate, (int, float)) and estimate != "N/A":
                                estimate = f"{estimate:,}"
                            if isinstance(moe, (int, float)) and moe != "N/A":
                                moe = f"±{moe:,}"
                            
                            response_parts.append(f"| {label} | {estimate} | {moe} |")
            
            response_parts.append(f"\n**Source**: {results.get('source', 'U.S. Census Bureau ACS 2023')}")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
        
        except Exception as e:
            logger.error(f"Census API error: {e}")
            return [TextContent(type="text", text=f"Error retrieving Census data: {str(e)}")]
    
    async def _search_census_variables(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Search variables using concept-based engine."""
        if not self.search_engine:
            return [TextContent(type="text", text="Variable search not available.")]
        
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 10)
        
        if not query:
            return [TextContent(type="text", text="Please provide a search query.")]
        
        try:
            logger.info(f"Searching variables: '{query}'")
            results = self.search_engine.search(query, max_results=max_results)
            
            if not results:
                return [TextContent(type="text", text=f"No variables found for query: '{query}'")]
            
            response_parts = [f"**Found {len(results)} variables for: '{query}'**\n"]
            
            for i, result in enumerate(results, 1):
                response_parts.append(f"**{i}. {result.variable_id}**")
                response_parts.append(f"   **Label**: {result.label}")
                response_parts.append(f"   **Concept**: {getattr(result, 'concept', 'Unknown')}")
                response_parts.append(f"   **Confidence**: {result.confidence:.3f}")
                response_parts.append("")
            
            response_parts.append("**Next step**: Use these variable IDs with `get_demographic_data`")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Variable search error: {e}")
            return [TextContent(type="text", text=f"Search error: {str(e)}")]

async def main():
    """Main server entry point."""
    logger.info("Starting Clean Census MCP Server...")
    
    try:
        census_server = CensusMCPServer()
        
        async with stdio_server() as (read_stream, write_stream):
            await census_server.server.run(
                read_stream,
                write_stream,
                census_server.server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        print(f"Server error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
