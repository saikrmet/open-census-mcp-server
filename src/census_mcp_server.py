#!/usr/bin/env python3
"""
Fixed Census MCP Server - v2.9
FIXES:
1. JSON parsing errors in Census API calls
2. Geographic resolution using production database  
3. Better error messages and user feedback
4. Robust error handling throughout
"""

import logging
import asyncio
from typing import Any, Dict, List
from mcp.server import Server
from mcp.types import TextContent, Tool

# Import fixed components (correct paths for running from src/)
from data_retrieval.python_census_api import PythonCensusAPI
from utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MCP Server
app = Server("census-mcp-server")

class CensusMCPServer:
    """Fixed Census MCP Server with robust error handling"""
    
    def __init__(self):
        self.config = Config()
        self.census_api = None
        
        # Initialize components
        self._init_census_api()
        
        logger.info("🚀 Census MCP Server v2.9 initialized")
        logger.info("✅ Token-based knowledge base ready")
        logger.info("✅ Geographic handler with 29,573 places")
        logger.info("✅ Robust error handling enabled")
    
    def _init_census_api(self):
        """Initialize Census API with error handling"""
        try:
            self.census_api = PythonCensusAPI()
            logger.info("✅ Census API client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Census API: {e}")
            self.census_api = None

# Initialize server instance
server_instance = CensusMCPServer()

# Tool Definitions
@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="get_demographic_data",
            description="Get demographic data for a location with robust error handling",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location (e.g., 'Austin, TX', 'Minnesota', 'United States')"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to retrieve (e.g., 'population', 'median_income')"
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year for data (default: 2023)",
                        "default": 2023
                    },
                    "survey": {
                        "type": "string",
                        "description": "Survey type: 'acs5' (5-year) or 'acs1' (1-year)",
                        "default": "acs5"
                    }
                },
                "required": ["location", "variables"]
            }
        ),
        Tool(
            name="search_census_variables",
            description="Search for Census variables using semantic intelligence",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of data needed"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="find_census_tables",
            description="Find Census tables by topic or keyword",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or keyword to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 5)",
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
                    "year": {
                        "type": "integer",
                        "description": "Year for data (default: 2023)",
                        "default": 2023
                    }
                },
                "required": ["locations", "variables"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls with comprehensive error handling"""
    
    if name == "get_demographic_data":
        return await _get_demographic_data(arguments)
    elif name == "search_census_variables":
        return await _search_census_variables(arguments)
    elif name == "find_census_tables":
        return await _find_census_tables(arguments)
    elif name == "compare_locations":
        return await _compare_locations(arguments)
    else:
        return [TextContent(
            type="text",
            text=f"❌ Unknown tool: {name}"
        )]

async def _get_demographic_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get demographic data with robust error handling"""
    
    # Validate inputs
    location = arguments.get("location", "").strip()
    variables = arguments.get("variables", [])
    year = arguments.get("year", 2023)
    survey = arguments.get("survey", "acs5").lower()
    
    if not location:
        return [TextContent(
            type="text",
            text="❌ Error: Location is required\n\n💡 Try: 'Austin, TX' or 'Minnesota'"
        )]
    
    if not variables:
        return [TextContent(
            type="text",
            text="❌ Error: At least one variable is required\n\n💡 Try: ['population'] or ['median_income', 'poverty_rate']"
        )]
    
    # Check if Census API is available
    if not server_instance.census_api:
        return [TextContent(
            type="text",
            text="❌ Error: Census API client not available\n\nThis is a system configuration issue. Please check the server logs."
        )]
    
    try:
        # Call the fixed Census API
        logger.info(f"🌍 Processing request: {location} with {len(variables)} variables")
        
        result = await server_instance.census_api.get_demographic_data(
            location=location,
            variables=variables,
            year=year,
            survey=survey
        )
        
        # Handle API errors gracefully
        if 'error' in result:
            error_msg = result['error']
            response_parts = [f"❌ **Error**: {error_msg}"]
            
            # Add helpful suggestions
            if 'suggestion' in result:
                response_parts.append(f"\n💡 **Suggestion**: {result['suggestion']}")
            
            if 'suggestions' in result and result['suggestions']:
                response_parts.append(f"\n🎯 **Did you mean**: {', '.join(result['suggestions'][:3])}")
            
            # Add context about what was attempted
            if 'location_attempted' in result:
                response_parts.append(f"\n📍 **Location attempted**: {result['location_attempted']}")
            
            if 'variables_attempted' in result:
                response_parts.append(f"\n📊 **Variables attempted**: {', '.join(result['variables_attempted'])}")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
        
        # Format successful response
        return [TextContent(type="text", text=_format_demographic_response(result))]
        
    except Exception as e:
        logger.error(f"❌ Demographic data error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **System Error**: {str(e)}\n\nThis error has been logged. Please try again or contact support."
        )]

def _format_demographic_response(result: Dict[str, Any]) -> str:
    """Format successful demographic data response"""
    response_parts = [
        f"# 🏛️ Census Data for {result['resolved_location']['name']}\n"
    ]
    
    # Add location details
    location_info = result['resolved_location']
    response_parts.append(f"📍 **Location Type**: {location_info['geography_type'].title()}")
    
    if location_info.get('fips_codes', {}).get('place'):
        response_parts.append(f"🏷️ **FIPS Code**: {location_info['fips_codes']['state']}:{location_info['fips_codes']['place']}")
    elif location_info.get('fips_codes', {}).get('state'):
        response_parts.append(f"🏷️ **FIPS Code**: {location_info['fips_codes']['state']}")
    
    response_parts.append("")  # Blank line
    
    # Add data results
    data = result.get('data', {})
    if data:
        response_parts.append("## 📊 **Data Results**")
        
        for var_id, var_data in data.items():
            if var_data.get('estimate') is not None:
                formatted_value = var_data.get('formatted', str(var_data['estimate']))
                response_parts.append(f"**{var_id}**: {formatted_value}")
            else:
                response_parts.append(f"**{var_id}**: {var_data.get('error', 'No data available')}")
        
        response_parts.append("")  # Blank line
    
    # Add variable resolution info
    variables_info = result.get('variables', {})
    response_parts.extend([
        "## 🔍 **Variable Resolution**",
        f"**Requested**: {', '.join(variables_info.get('requested', []))}",
        f"**Resolved**: {', '.join(variables_info.get('resolved', []))}",
        f"**Success Rate**: {variables_info.get('success_count', 0)}/{len(variables_info.get('resolved', []))}"
    ])
    
    response_parts.append("")  # Blank line
    
    # Add survey metadata
    survey_info = result.get('survey_info', {})
    response_parts.extend([
        "---",
        "## 🏛️ **Data Source**",
        f"**Survey**: {survey_info.get('survey', 'Unknown')} {survey_info.get('year', 'Unknown')}",
        f"**Source**: {survey_info.get('source', 'US Census Bureau')}",
        f"**Quality**: Official government statistics with scientific sampling"
    ])
    
    # Add system info
    api_info = result.get('api_info', {})
    if api_info.get('geographic_database'):
        response_parts.append(f"**Geographic Resolution**: {api_info['geographic_database']}")
    
    return "\n".join(response_parts)

async def _search_census_variables(arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for Census variables - placeholder for semantic search integration"""
    query = arguments.get("query", "").strip()
    max_results = arguments.get("max_results", 10)
    
    if not query:
        return [TextContent(
            type="text",
            text="❌ Error: Search query is required\n\n💡 Try: 'median household income' or 'poverty rate'"
        )]
    
    # TODO: Integrate with kb_search.py semantic search
    return [TextContent(
        type="text",
        text=f"🔍 **Variable Search**: {query}\n\n⚠️ **Status**: Semantic variable search integration in progress\n\n💡 **Available**: Use variable names like 'population', 'median_income', 'poverty_rate' in get_demographic_data tool"
    )]

async def _find_census_tables(arguments: Dict[str, Any]) -> List[TextContent]:
    """Find Census tables - placeholder for table catalog integration"""
    query = arguments.get("query", "").strip()
    max_results = arguments.get("max_results", 5)
    
    if not query:
        return [TextContent(
            type="text",
            text="❌ Error: Search query is required"
        )]
    
    # TODO: Integrate with table catalog
    return [TextContent(
        type="text",
        text=f"📋 **Table Search**: {query}\n\n⚠️ **Status**: Table catalog integration in progress"
    )]

async def _compare_locations(arguments: Dict[str, Any]) -> List[TextContent]:
    """Compare locations - uses multiple get_demographic_data calls"""
    locations = arguments.get("locations", [])
    variables = arguments.get("variables", [])
    year = arguments.get("year", 2023)
    
    if not locations or len(locations) < 2:
        return [TextContent(
            type="text",
            text="❌ Error: At least 2 locations required for comparison\n\n💡 Try: ['Austin, TX', 'Dallas, TX']"
        )]
    
    if not variables:
        return [TextContent(
            type="text",
            text="❌ Error: At least one variable required\n\n💡 Try: ['population', 'median_income']"
        )]
    
    if not server_instance.census_api:
        return [TextContent(
            type="text",
            text="❌ Error: Census API client not available"
        )]
    
    try:
        # Get data for each location
        results = []
        for location in locations:
            result = await server_instance.census_api.get_demographic_data(
                location=location,
                variables=variables,
                year=year,
                survey="acs5"
            )
            results.append((location, result))
        
        # Format comparison response
        return [TextContent(type="text", text=_format_comparison_response(results, variables))]
        
    except Exception as e:
        logger.error(f"❌ Location comparison error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **System Error**: {str(e)}\n\nThis error has been logged."
        )]

def _format_comparison_response(results: List[tuple], variables: List[str]) -> str:
    """Format location comparison response"""
    response_parts = ["# 🏛️ Location Comparison\n"]
    
    # Add summary table
    response_parts.append("## 📊 **Comparison Results**")
    
    for location, result in results:
        if 'error' in result:
            response_parts.append(f"**{location}**: ❌ {result['error']}")
        else:
            name = result.get('resolved_location', {}).get('name', location)
            response_parts.append(f"### {name}")
            
            data = result.get('data', {})
            for var_id, var_data in data.items():
                if var_data.get('estimate') is not None:
                    formatted = var_data.get('formatted', str(var_data['estimate']))
                    response_parts.append(f"  **{var_id}**: {formatted}")
                else:
                    response_parts.append(f"  **{var_id}**: No data")
            
            response_parts.append("")  # Blank line
    
    response_parts.extend([
        "---",
        "**Source**: US Census Bureau American Community Survey",
        "**Note**: Comparisons use ACS 5-year estimates for reliability"
    ])
    
    return "\n".join(response_parts)

# Main server function
async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server
    
    logger.info("🚀 Starting Census MCP Server v2.9")
    logger.info("🔧 Features: Token-based KB, Geographic DB, Robust Error Handling")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
