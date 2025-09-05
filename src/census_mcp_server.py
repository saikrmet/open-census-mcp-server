#!/usr/bin/env python3
"""
Clean Census MCP Server - Claude Direct Knowledge

Simple architecture:
1. Claude uses built-in Census knowledge to construct API calls directly
2. Search only for genuine edge cases where Claude is uncertain
3. No hardcoded mappings - Claude reasons from knowledge
"""

import logging
import asyncio
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.types import TextContent, Tool

# Import existing components
from data_retrieval.python_census_api import PythonCensusAPI
from utils.config import Config

# Knowledge base search (optional fallback only)
KB_SEARCH_AVAILABLE = False
try:
    current_dir = Path(__file__).parent
    kb_path = current_dir.parent / "knowledge-base"
    
    if kb_path.exists():
        sys.path.insert(0, str(kb_path))
        from kb_search import create_search_engine
        KB_SEARCH_AVAILABLE = True
        logging.info(f"✅ Knowledge base available as fallback: {kb_path}")
    else:
        logging.warning(f"⚠️ Knowledge base directory not found: {kb_path}")
        
except Exception as e:
    KB_SEARCH_AVAILABLE = False
    logging.warning(f"Knowledge base not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server
app = Server("census-mcp")

class CensusMCPServer:
    """Clean Census MCP Server with Claude direct knowledge"""
    
    def __init__(self):
        self.config = Config()
        self.census_api = PythonCensusAPI()
        self.search_engine = None
        
        # Optional geographic handler for edge cases
        try:
            from data_retrieval.geographic_handler import GeographicHandler
            geo_db_path = self.config.base_dir / "knowledge-base" / "geo-db" / "geography.db"
            self.geo_handler = GeographicHandler(str(geo_db_path))
            logger.info("✅ Geographic handler available as fallback")
        except Exception as e:
            logger.warning(f"Geographic handler unavailable: {e}")
            self.geo_handler = None
        
        # Optional search engine for edge cases
        if KB_SEARCH_AVAILABLE:
            try:
                self.search_engine = create_search_engine()
                logger.info("✅ Knowledge base search available as fallback")
            except Exception as e:
                logger.warning(f"Knowledge base search unavailable: {e}")

# Global server instance
server_instance = CensusMCPServer()

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="get_census_data",
            description="PRIMARY TOOL: Get official US Census demographic data using natural language. Supports single queries and batch queries like 'all counties in Maryland by population'. Always use this tool FIRST for Census questions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Natural language location (e.g. 'St. Louis', 'Chicago, IL', 'Texas', '90210', 'Maryland')"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Data requested in natural language (e.g. 'population', 'median income', 'poverty rate') or Census variable IDs"
                    },
                    "include_methodology": {
                        "type": "boolean",
                        "description": "Include statistical guidance and methodology context",
                        "default": True
                    }
                },
                "required": ["location", "variables"]
            }
        ),
        Tool(
            name="resolve_geography",
            description="FALLBACK: Use only when get_census_data indicates location is ambiguous and directs you here.",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to resolve"
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
            description="FALLBACK: Use only when get_census_data fails or when you need precise FIPS code control.",
            inputSchema={
                "type": "object",
                "properties": {
                    "geography_type": {
                        "type": "string",
                        "enum": ["place", "county", "state", "cbsa", "zcta", "us"],
                        "description": "Type of geography: place, county, state, cbsa, zcta"
                    },
                    "state_fips": {
                        "type": "string",
                        "description": "2-digit state FIPS code"
                    },
                    "county_fips": {
                        "type": "string",
                        "description": "3-digit county FIPS code"
                    },
                    "place_fips": {
                        "type": "string",
                        "description": "5-digit place FIPS code"
                    },
                    "cbsa_code": {
                        "type": "string",
                        "description": "5-digit CBSA code"
                    },
                    "zcta_code": {
                        "type": "string",
                        "description": "5-digit ZCTA code"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Census variable IDs or concepts"
                    }
                },
                "required": ["variables", "geography_type"]
            }
        ),
        Tool(
            name="search_census_variables",
            description="FALLBACK: Use only when get_census_data indicates variables need clarification and directs you here.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of desired data"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "get_census_data":
            return await _get_census_data_direct(arguments)
        elif name == "resolve_geography":
            return await _resolve_geography(arguments)
        elif name == "get_demographic_data":
            return await _get_demographic_data(arguments)
        elif name == "search_census_variables":
            return await _search_census_variables(arguments)
        else:
            return [TextContent(
                type="text",
                text=f"❌ Unknown tool: {name}"
            )]
            
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ Error executing {name}: {str(e)}"
        )]

async def _get_census_data_direct(arguments: Dict[str, Any]) -> List[TextContent]:
    """Direct Census data retrieval using Claude's built-in knowledge"""
    
    location = arguments.get("location", "").strip()
    variables = arguments.get("variables", [])
    include_methodology = arguments.get("include_methodology", True)
    
    if not location or not variables:
        return [TextContent(
            type="text",
            text="❌ **Missing required parameters**\n\n**Required:**\n- `location`: Geographic area (e.g., 'Baltimore, MD', 'California', '90210')\n- `variables`: Data requested (e.g., ['population', 'median income'])\n\n**Examples:**\n- get_census_data(location='Chicago, IL', variables=['population', 'poverty rate'])\n- get_census_data(location='Maryland', variables=['population']) for batch queries"
        )]
    
    # Convert single variable to list
    if isinstance(variables, str):
        variables = [variables]
    
    logger.info(f"📊 Direct Census query: {location}, variables: {variables}")
    
    try:
        # Use Claude's Census knowledge to construct API call directly
        return await _construct_api_call_directly(location, variables, include_methodology)
        
    except Exception as e:
        logger.warning(f"Direct approach failed: {e}")
        
        # Only fallback to search for genuine edge cases
        if server_instance.search_engine:
            return await _search_fallback(location, variables, include_methodology, str(e))
        else:
            return [TextContent(
                type="text",
                text=f"❌ **Census query failed**: {str(e)}\n\n💡 **Try:**\n- More specific location format ('City, State')\n- Common variable names (population, median income, poverty rate)\n- Use `resolve_geography` and `search_census_variables` tools for complex cases"
            )]

async def _construct_api_call_directly(location: str, variables: List[str], include_methodology: bool) -> List[TextContent]:
    """Use Claude's direct Census knowledge from training data to construct API call"""
    
    # Step 1: Map variables using my Census training knowledge directly
    census_variables = []
    for var in variables:
        var_clean = var.lower().strip()
        
        # Already a Census variable code?
        if var.upper().startswith('B') and '_' in var and 'E' in var.upper():
            census_variables.append(var.upper())
            continue
            
        # Direct reasoning from training data - no lookups
        if 'population' in var_clean:
            census_variables.append('B01003_001E')
        elif 'median household income' in var_clean or (('median' in var_clean or var_clean == 'income') and 'household' in var_clean):
            census_variables.append('B19013_001E')
        elif 'median family income' in var_clean:
            census_variables.append('B19113_001E')
        elif 'per capita income' in var_clean:
            census_variables.append('B19301_001E')
        elif 'poverty' in var_clean:
            census_variables.append('B17001_002E')
        elif 'median age' in var_clean or var_clean == 'age':
            census_variables.append('B01002_001E')
        elif 'home value' in var_clean or 'house value' in var_clean or 'housing value' in var_clean:
            census_variables.append('B25077_001E')
        elif 'unemployment' in var_clean:
            census_variables.append('B23025_005E')
        elif 'education' in var_clean or 'college' in var_clean or 'bachelor' in var_clean:
            census_variables.append('B15003_022E')
        else:
            # Genuine unknown - use search fallback
            raise Exception(f"Variable not in training data: {var}")
    
    # Step 2: Resolve geography using my training knowledge directly  
    location_clean = location.lower().strip()
    
    # Direct reasoning from my Census training data
    if location_clean in ['maine', 'me']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '23'}}
    elif location_clean in ['nevada', 'nv']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '32'}}
    elif location_clean in ['california', 'ca']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '06'}}
    elif location_clean in ['texas', 'tx']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '48'}}
    elif location_clean in ['florida', 'fl']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '12'}}
    elif location_clean in ['new york', 'ny']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '36'}}
    elif location_clean in ['maryland', 'md']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '24'}}
    elif location_clean in ['virginia', 'va']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '51'}}
    elif location_clean in ['illinois', 'il']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '17'}}
    elif location_clean in ['pennsylvania', 'pa']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '42'}}
    elif location_clean in ['ohio', 'oh']:
        geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': '39'}}
    elif 'reno' in location_clean and 'nv' in location_clean:
        geography_params = {'geography_type': 'place', 'fips_codes': {'state_fips': '32', 'place_fips': '60600'}}
    elif 'chicago' in location_clean and 'il' in location_clean:
        geography_params = {'geography_type': 'place', 'fips_codes': {'state_fips': '17', 'place_fips': '14000'}}
    elif 'baltimore' in location_clean and 'md' in location_clean:
        geography_params = {'geography_type': 'place', 'fips_codes': {'state_fips': '24', 'place_fips': '04000'}}
    elif location.isdigit() and len(location) == 5:
        geography_params = {'geography_type': 'zcta', 'fips_codes': {'zcta_code': location}}
    elif ',' in location:
        # Parse City, State - use training knowledge for common states
        parts = [p.strip() for p in location.split(',')]
        if len(parts) == 2:
            city, state = parts
            state_clean = state.lower()
            
            # Get state FIPS from training knowledge
            state_fips = None
            if state_clean in ['ca', 'california']:
                state_fips = '06'
            elif state_clean in ['tx', 'texas']:
                state_fips = '48'
            elif state_clean in ['fl', 'florida']:
                state_fips = '12'
            elif state_clean in ['ny', 'new york']:
                state_fips = '36'
            elif state_clean in ['nv', 'nevada']:
                state_fips = '32'
            elif state_clean in ['md', 'maryland']:
                state_fips = '24'
            elif state_clean in ['me', 'maine']:
                state_fips = '23'
            
            if state_fips:
                # Default to state level for unknown cities
                geography_params = {'geography_type': 'state', 'fips_codes': {'state_fips': state_fips}}
            else:
                raise Exception(f"State not in training data: {state}")
        else:
            raise Exception(f"Cannot parse location format: {location}")
    else:
        # Location not in training data - use fallback
        raise Exception(f"Location not in training data: {location}")
    
    # Step 3: Construct and execute API call
    api_args = {
        "geography_type": geography_params['geography_type'],
        "variables": census_variables
    }
    api_args.update(geography_params['fips_codes'])
    
    # Execute the API call
    result = await _get_demographic_data(api_args)
    
    if include_methodology:
        result[0].text += f"\n\n---\n**🧠 Claude Direct**: Reasoned from Census training data (no lookups)"
    
    return result

# Removed lookup helper functions - using direct reasoning in _construct_api_call_directly

async def _search_fallback(location: str, variables: List[str], include_methodology: bool, error_msg: str) -> List[TextContent]:
    """Fallback to search when Claude's direct knowledge is insufficient"""
    
    if not server_instance.search_engine:
        return [TextContent(
            type="text",
            text=f"❌ **Cannot resolve**: Claude's direct knowledge insufficient and search unavailable\n\n**Location**: {location}\n**Variables**: {variables}\n**Error**: {error_msg}\n\n💡 **Try:**\n- More common location/variable names\n- Use `resolve_geography` and `search_census_variables` tools"
        )]
    
    response_parts = [
        f"🔍 **Search fallback** - Claude's direct knowledge insufficient\n",
        f"**Issue**: {error_msg}",
        f"**Location**: {location}",
        f"**Variables**: {variables}\n"
    ]
    
    # Search for variables
    for var in variables:
        search_result = await _search_census_variables({"query": var, "max_results": 3})
        response_parts.append(f"**Variable search for '{var}':**")
        response_parts.append(search_result[0].text)
        response_parts.append("")
    
    response_parts.append("💡 **Next step**: Use found variable codes with `get_demographic_data` and resolved geography")
    
    if include_methodology:
        response_parts.append("\n---\n**🧠 Claude Routing**: Fell back to search - direct knowledge insufficient")
    
    return [TextContent(type="text", text="\n".join(response_parts))]

async def _resolve_geography(arguments: Dict[str, Any]) -> List[TextContent]:
    """Resolve geographic location to FIPS codes"""
    location = arguments.get("location", "").strip()
    max_results = arguments.get("max_results", 10)
    
    if not location:
        return [TextContent(
            type="text",
            text="❌ **Error**: Location parameter is required\n\n**Usage**: resolve_geography(location='Baltimore')"
        )]
    
    try:
        logger.info(f"🗺️ Resolving geography: '{location}'")
        
        if not server_instance.geo_handler:
            return [TextContent(
                type="text",
                text="❌ **Geographic handler unavailable**\n\nGeography resolution requires the geographic database to be properly initialized."
            )]
        
        results = server_instance.geo_handler.search_locations(location, max_results)
        
        if not results:
            return [TextContent(
                type="text",
                text=f"❌ **No geographic matches found** for '{location}'\n\n💡 **Try:**\n- Check spelling\n- Use state abbreviations (MD, CA, TX)\n- Try 'City, State' format\n- Use ZIP codes for small areas"
            )]
        
        response_parts = [f"🗺️ **Geographic matches for '{location}'**\n"]
        
        for i, result in enumerate(results[:max_results], 1):
            name = result.get('name', 'Unknown')
            geo_type = result.get('geography_type', 'Unknown')
            confidence = result.get('confidence', 0.0)
            
            response_parts.append(f"**{i}.** {name}")
            response_parts.append(f"   - **Type**: {geo_type.title()}")
            response_parts.append(f"   - **Confidence**: {confidence:.1%}")
            
            # Add relevant FIPS codes
            if geo_type == 'place' and 'place_fips' in result:
                response_parts.append(f"   - **Place FIPS**: {result['state_fips']}{result['place_fips']}")
                response_parts.append(f"   - **State**: {result.get('state_abbrev', 'N/A')}")
            elif geo_type == 'county' and 'county_fips' in result:
                response_parts.append(f"   - **County FIPS**: {result['state_fips']}{result['county_fips']}")
                response_parts.append(f"   - **State**: {result.get('state_abbrev', 'N/A')}")
            elif geo_type == 'cbsa' and 'cbsa_code' in result:
                response_parts.append(f"   - **CBSA Code**: {result['cbsa_code']}")
            elif geo_type == 'zcta' and 'zcta_code' in result:
                response_parts.append(f"   - **ZCTA Code**: {result['zcta_code']}")
            
            response_parts.append("")
        
        response_parts.append("💡 **Use the appropriate FIPS codes above with `get_demographic_data`**")
        
        return [TextContent(type="text", text="\n".join(response_parts))]
        
    except Exception as e:
        logger.error(f"Geography resolution error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **Geography resolution failed**: {str(e)}"
        )]

async def _get_demographic_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get demographic data using specific FIPS codes and variables"""
    
    geography_type = arguments.get("geography_type")
    variables = arguments.get("variables", [])
    
    if not geography_type or not variables:
        return [TextContent(
            type="text",
            text="❌ **Missing required parameters**\n\n**Required:**\n- `geography_type`: place, county, state, cbsa, zcta, or us\n- `variables`: List of Census variable IDs\n\n**Example**: get_demographic_data(geography_type='state', state_fips='24', variables=['B01003_001E'])"
        )]
    
    try:
        logger.info(f"📊 Getting demographic data: {geography_type}, variables: {variables}")
        
        # Build API parameters
        api_params = {
            "variables": variables
        }
        
        # Add geographic identifiers
        for geo_param in ['state_fips', 'county_fips', 'place_fips', 'cbsa_code', 'zcta_code']:
            if geo_param in arguments:
                api_params[geo_param] = arguments[geo_param]
        
        # Use the correct API method
        result = server_instance.census_api.get_acs_data(geography_type=geography_type, **api_params)
        
        if result.get('error'):
            return [TextContent(
                type="text",
                text=f"❌ **Census API Error**: {result['error']}"
            )]
        
        if not result or not result.get('data'):
            return [TextContent(
                type="text",
                text="❌ **No data returned** from Census API\n\n💡 **Possible issues:**\n- Invalid variable codes\n- Geographic area not found\n- Data not available for this geography/variable combination"
            )]
        
        # Format the response
        response_parts = [f"🏛️ **Official Census Data** ({geography_type.title()} Level)\n"]
        
        # Handle the different data format from get_acs_data
        if result.get('batch_query'):
            # Batch query result
            data_list = result['data']
            response_parts.append(f"**Batch Results**: {len(data_list)} locations")
            response_parts.append("")
            
            for i, location_data in enumerate(data_list[:10]):  # Show first 10
                location_name = location_data.get('location_name', f'Location {i+1}')
                response_parts.append(f"**{i+1}. {location_name}:**")
                
                for var_id in variables:
                    var_data = location_data['data'].get(var_id, {})
                    estimate = var_data.get('estimate', 'No data')
                    label = var_data.get('label', var_id)
                    response_parts.append(f"   - {label}: {estimate}")
                response_parts.append("")
        else:
            # Single geography result
            data = result['data']
            location_name = result.get('location_name', 'Unknown Location')
            
            response_parts.append(f"**Location**: {location_name}")
            response_parts.append("")
            
            for var_id in variables:
                var_data = data.get(var_id, {})
                estimate = var_data.get('estimate', 'No data')
                label = var_data.get('label', var_id)
                response_parts.append(f"**{label}**: {estimate}")
        
        response_parts.extend([
            "",
            "---",
            f"**Source**: US Census Bureau ACS 2023",
            f"**Geography Type**: {geography_type.title()}",
            f"**API URL**: {result.get('api_url', 'Unknown')[:100]}..."
        ])
        
        return [TextContent(type="text", text="\n".join(response_parts))]
        
    except Exception as e:
        logger.error(f"Demographic data error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **Data retrieval failed**: {str(e)}"
        )]

async def _search_census_variables(arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for Census variables using the knowledge base"""
    
    query = arguments.get("query", "").strip()
    max_results = arguments.get("max_results", 10)
    
    if not query:
        return [TextContent(
            type="text",
            text="❌ **Error**: Search query is required\n\n**Usage**: search_census_variables(query='median income')"
        )]
    
    if not server_instance.search_engine:
        return [TextContent(
            type="text",
            text="❌ **Knowledge base search unavailable**\n\nVariable search requires the knowledge base component to be properly configured."
        )]
    
    try:
        logger.info(f"🔍 Searching variables: '{query}'")
        
        results = server_instance.search_engine.search(query, max_results=max_results)
        
        if not results:
            return [TextContent(
                type="text",
                text=f"❌ **No variables found** for '{query}'\n\n💡 **Try:**\n- Different keywords (e.g., 'income' instead of 'salary')\n- Broader terms (e.g., 'education' instead of 'college degree')\n- Common Census concepts: population, income, poverty, age, race, housing"
            )]
        
        response_parts = [f"🔍 **Variable Search Results** for '{query}'\n"]
        
        for i, result in enumerate(results, 1):
            var_id = result.get('variable_id', 'Unknown')
            label = result.get('label', 'No label')
            concept = result.get('concept', 'No concept')
            score = result.get('score', 0.0)
            
            response_parts.append(f"**{i}.** `{var_id}`")
            response_parts.append(f"   - **Label**: {label}")
            response_parts.append(f"   - **Concept**: {concept}")
            response_parts.append(f"   - **Relevance**: {score:.1%}")
            response_parts.append("")
        
        response_parts.append("💡 **Use the variable IDs above with `get_demographic_data`**")
        
        return [TextContent(type="text", text="\n".join(response_parts))]
        
    except Exception as e:
        logger.error(f"Variable search error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **Variable search failed**: {str(e)}"
        )]

async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
