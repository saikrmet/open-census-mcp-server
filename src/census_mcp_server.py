#!/usr/bin/env python3
"""
Clean Census MCP Server - Claude-First with Smart Routing

Simple, reliable architecture:
1. Claude's knowledge for query parsing and confidence assessment
2. Smart routing: high confidence → direct API, medium → search-assisted, low → guidance
3. Clean tool separation with no external LLM dependencies
4. One smart tool that orchestrates everything: get_census_data
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

# Knowledge base search (optional)
KB_SEARCH_AVAILABLE = False
try:
    current_dir = Path(__file__).parent
    kb_path = current_dir.parent / "knowledge-base"
    
    if kb_path.exists():
        sys.path.insert(0, str(kb_path))
        from kb_search import create_search_engine
        KB_SEARCH_AVAILABLE = True
        logging.info(f"✅ Knowledge base available: {kb_path}")
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
    """Clean Census MCP Server with Claude-first architecture"""
    
    def __init__(self):
        self.config = Config()
        self.census_api = PythonCensusAPI(self.config.census_api_key)
        self.search_engine = None
        
        if KB_SEARCH_AVAILABLE:
            try:
                self.search_engine = create_search_engine()
                logger.info("✅ Knowledge base search engine initialized")
            except Exception as e:
                logger.warning(f"Knowledge base search unavailable: {e}")
    
    def parse_query_with_claude_knowledge(self, query: str, location: str = "") -> Dict[str, Any]:
        """
        Parse query using Claude's built-in Census knowledge
        Returns confidence score and routing decision
        """
        query_lower = query.lower()
        location_lower = location.lower()
        combined = f"{query_lower} {location_lower}".strip()
        
        parsed = {
            'query': query,
            'location': location,
            'variables': [],
            'geography_type': None,
            'confidence': 0.0,
            'routing': 'guidance'  # guidance, search_assisted, direct_api
        }
        
        # Location confidence (Claude knows these patterns)
        location_confidence = 0.0
        if any(pattern in location_lower for pattern in [
            'county', 'city', 'state', 'zip', 'zcta', 'cbsa', 'msa'
        ]):
            location_confidence = 0.8
        elif location:
            location_confidence = 0.6
        
        # Variable confidence (Claude knows common Census concepts)
        variable_confidence = 0.0
        variables = []
        
        # High confidence variables (Claude knows these codes)
        if 'population' in query_lower:
            variables.append('B01003_001E')  # Total population
            variable_confidence = 0.9
        elif 'median income' in query_lower or 'household income' in query_lower:
            variables.append('B19013_001E')  # Median household income
            variable_confidence = 0.9
        elif 'poverty' in query_lower:
            variables.append('B17001_002E')  # Poverty status
            variable_confidence = 0.8
        elif 'median age' in query_lower:
            variables.append('B01002_001E')  # Median age
            variable_confidence = 0.9
        elif 'race' in query_lower or 'ethnicity' in query_lower:
            variables.extend(['B02001_002E', 'B02001_003E', 'B03003_003E'])  # Race/ethnicity
            variable_confidence = 0.8
        elif 'education' in query_lower:
            variables.append('B15003_022E')  # Educational attainment
            variable_confidence = 0.7
        elif 'housing' in query_lower and 'median' in query_lower:
            variables.append('B25077_001E')  # Median home value
            variable_confidence = 0.8
        elif 'unemployment' in query_lower:
            variables.append('B23025_005E')  # Unemployment
            variable_confidence = 0.8
        else:
            # Medium confidence - might need search assistance
            variable_confidence = 0.3
        
        # Geography type detection (Claude knowledge)
        if 'county' in combined or 'counties' in combined:
            parsed['geography_type'] = 'county'
        elif 'state' in combined or any(state in location_lower for state in ['ca', 'ny', 'tx', 'fl']):
            parsed['geography_type'] = 'state'
        elif 'city' in combined or ',' in location:
            parsed['geography_type'] = 'place'
        elif 'zip' in combined or 'zcta' in combined:
            parsed['geography_type'] = 'zcta'
        
        # Overall confidence calculation
        parsed['confidence'] = (location_confidence + variable_confidence) / 2
        parsed['variables'] = variables
        
        # Routing decision based on confidence
        if parsed['confidence'] >= 0.8:
            parsed['routing'] = 'direct_api'
        elif parsed['confidence'] >= 0.5:
            parsed['routing'] = 'search_assisted'
        else:
            parsed['routing'] = 'guidance'
        
        return parsed

# Global server instance
server_instance = CensusMCPServer()

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="get_census_data",
            description="PRIMARY TOOL: Get official US Census demographic data using natural language. Always use this tool FIRST for Census questions. Uses Claude's knowledge when confident, falls back to RAG tools when uncertain.",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Natural language location (e.g. 'St. Louis', 'Chicago, IL', 'Texas', '90210')"
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
            return await _get_census_data_smart(arguments)
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

async def _get_census_data_smart(arguments: Dict[str, Any]) -> List[TextContent]:
    """Smart Census data retrieval using Claude's knowledge for routing"""
    
    location = arguments.get("location", "").strip()
    variables = arguments.get("variables", [])
    include_methodology = arguments.get("include_methodology", True)
    
    if not location or not variables:
        return [TextContent(
            type="text",
            text="❌ **Missing required parameters**\n\n**Required:**\n- `location`: Geographic area (e.g., 'Baltimore, MD', 'California', '90210')\n- `variables`: Data requested (e.g., ['population', 'median income'])\n\n**Example:** get_census_data(location='Chicago, IL', variables=['population', 'poverty rate'])"
        )]
    
    # Convert single variable to list
    if isinstance(variables, str):
        variables = [variables]
    
    # Parse query using Claude's knowledge
    query_text = f"Get {', '.join(variables)} for {location}"
    parsed = server_instance.parse_query_with_claude_knowledge(query_text, location)
    
    logger.info(f"📊 Smart routing: confidence={parsed['confidence']:.2f}, routing={parsed['routing']}")
    
    # Route based on confidence
    if parsed['routing'] == 'direct_api':
        # High confidence - use Claude's variable mapping directly
        return await _execute_direct_api_call(location, parsed['variables'], parsed, include_methodology)
    
    elif parsed['routing'] == 'search_assisted':
        # Medium confidence - use search to validate variables
        return await _execute_search_assisted_call(location, variables, parsed, include_methodology)
    
    else:
        # Low confidence - provide guidance
        return await _provide_usage_guidance(query_text, location, variables)

async def _execute_direct_api_call(location: str, variables: List[str], parsed: Dict, include_methodology: bool) -> List[TextContent]:
    """Execute high-confidence API call using Claude's variable knowledge"""
    
    try:
        # Use the demographic data tool with Claude's parsed information
        api_args = {
            "variables": variables,
            "geography_type": parsed.get('geography_type', 'state')
        }
        
        # Add location resolution here (simplified for demo)
        if parsed.get('geography_type') == 'state':
            # Simple state name to FIPS mapping (Claude knowledge)
            state_fips_map = {
                'california': '06', 'texas': '48', 'florida': '12', 'new york': '36',
                'pennsylvania': '42', 'illinois': '17', 'ohio': '39', 'georgia': '13',
                'north carolina': '37', 'michigan': '26', 'maryland': '24'
            }
            state_name = location.lower().replace(' state', '')
            if state_name in state_fips_map:
                api_args['state_fips'] = state_fips_map[state_name]
        
        result = await _get_demographic_data(api_args)
        
        if include_methodology:
            result[0].text += f"\n\n---\n**🧠 Claude Routing**: High confidence ({parsed['confidence']:.2f}) - used direct API call with built-in variable knowledge"
        
        return result
        
    except Exception as e:
        logger.error(f"Direct API call failed: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **Direct API call failed**: {str(e)}\n\n💡 **Suggestion**: Try using `search_census_variables` to find the right variable codes first."
        )]

async def _execute_search_assisted_call(location: str, variables: List[str], parsed: Dict, include_methodology: bool) -> List[TextContent]:
    """Execute search-assisted call for medium confidence queries"""
    
    response_parts = [f"🔍 **Search-assisted query** (confidence: {parsed['confidence']:.2f})\n"]
    
    # Search for each variable to get proper codes
    found_variables = []
    for var in variables:
        search_result = await _search_census_variables({"query": var, "max_results": 3})
        if search_result and "❌" not in search_result[0].text:
            response_parts.append(f"**Variable search for '{var}':**")
            response_parts.append(search_result[0].text)
            # Extract variable codes from search results (simplified)
            # In real implementation, parse the search results properly
            found_variables.append(var)
    
    if found_variables:
        response_parts.append(f"\n💡 **Next step**: Use `get_demographic_data` with the variable codes found above.")
    
    return [TextContent(type="text", text="\n".join(response_parts))]

async def _provide_usage_guidance(query: str, location: str, variables: List[str]) -> List[TextContent]:
    """Provide usage guidance for low-confidence queries"""
    
    guidance = [
        f"🤔 **Query needs clarification** (query: '{query}')\n",
        "**Issues identified:**"
    ]
    
    if not location:
        guidance.append("- ❌ **Location unclear**: Please specify city, county, state, or ZIP code")
    
    if not variables:
        guidance.append("- ❌ **Variables unclear**: Please specify what demographic data you need")
    
    guidance.extend([
        "\n**💡 Examples of clear queries:**",
        "- `location='Baltimore, MD', variables=['population', 'median income']`",
        "- `location='California', variables=['poverty rate', 'unemployment']`",
        "- `location='90210', variables=['median home value', 'education level']`",
        "\n**🔧 Tools to help clarify:**",
        "- Use `resolve_geography` to find the right location codes",
        "- Use `search_census_variables` to find variable definitions",
        "\n**📚 Common variables:**",
        "- Population: 'population', 'total population'",
        "- Income: 'median income', 'household income', 'per capita income'",
        "- Poverty: 'poverty rate', 'poverty status'",
        "- Age: 'median age', 'age distribution'",
        "- Education: 'education level', 'college degree', 'high school'",
        "- Housing: 'median home value', 'housing costs', 'rent'"
    ])
    
    return [TextContent(type="text", text="\n".join(guidance))]

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
        results = server_instance.census_api.resolve_geography(location, max_results)
        
        if not results:
            return [TextContent(
                type="text",
                text=f"❌ **No geographic matches found** for '{location}'\n\n💡 **Try:**\n- Check spelling\n- Use state abbreviations (MD, CA, TX)\n- Try 'City, State' format\n- Use ZIP codes for small areas"
            )]
        
        response_parts = [f"🗺️ **Geographic matches for '{location}'**\n"]
        
        for i, result in enumerate(results[:max_results], 1):
            response_parts.append(f"**{i}.** {result.get('name', 'Unknown')}")
            response_parts.append(f"   - **Type**: {result.get('geography_type', 'Unknown')}")
            response_parts.append(f"   - **FIPS**: {result.get('fips_code', 'N/A')}")
            response_parts.append(f"   - **State**: {result.get('state', 'N/A')}")
            response_parts.append("")
        
        response_parts.append("💡 **Use the FIPS codes above with `get_demographic_data`**")
        
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
            "geography_type": geography_type,
            "variables": variables
        }
        
        # Add geographic identifiers
        for geo_param in ['state_fips', 'county_fips', 'place_fips', 'cbsa_code', 'zcta_code']:
            if geo_param in arguments:
                api_params[geo_param] = arguments[geo_param]
        
        result = server_instance.census_api.get_demographic_data(**api_params)
        
        if not result or not result.get('data'):
            return [TextContent(
                type="text",
                text="❌ **No data returned** from Census API\n\n💡 **Possible issues:**\n- Invalid variable codes\n- Geographic area not found\n- Data not available for this geography/variable combination"
            )]
        
        # Format the response
        response_parts = [f"🏛️ **Official Census Data** ({geography_type.title()} Level)\n"]
        
        data = result['data'][0] if result['data'] else {}
        location_name = result.get('location_name', 'Unknown Location')
        
        response_parts.append(f"**Location**: {location_name}")
        response_parts.append("")
        
        for var_id in variables:
            value = data.get(var_id, 'No data')
            response_parts.append(f"**{var_id}**: {value}")
        
        response_parts.extend([
            "",
            "---",
            f"**Source**: US Census Bureau ACS 2023",
            f"**Geography Type**: {geography_type.title()}",
            f"**Query Time**: {result.get('query_time', 'Unknown')}"
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
        
        results = server_instance.search_engine.search(query, limit=max_results)
        
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
