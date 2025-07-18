#!/usr/bin/env python3
"""
Census MCP Server - Containerized Census expertise via MCP protocol

Provides natural language access to US Census data through:
- Python Census API client (replaces R tidycensus)
- Dual-path vector DB with FAISS variables + ChromaDB methodology
- Statistical validation and geographic resolution

Architecture components:
- MCP Server (this file) - Protocol interface
- Dual-Path Knowledge Base - FAISS + ChromaDB for instant startup
- Python Census API - Direct api.census.gov client
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

# Local imports
from knowledge.vector_db import DualPathKnowledgeBase
from data_retrieval.python_census_api import PythonCensusAPI
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CensusMCPServer:
    """
    Main MCP server class that orchestrates Census data requests.
    
    Handles:
    - MCP protocol communication
    - Semantic variable mapping via dual-path vector DB
    - Python Census API for data retrieval
    - Response formatting with statistical context
    """
    
    def __init__(self):
        """Initialize server with EAGER loading - no lazy initialization."""
        logger.info("🏛️ Starting Census MCP Server (Python API Mode)...")
        print("Starting Census MCP Server with Python Census API...", file=sys.stderr)
        
        # Load configuration
        self.config = Config()
        
        # EAGER INITIALIZATION - Load everything now
        self._init_knowledge_base()
        self._init_python_api()
        
        # Create MCP server instance
        self.server = Server("census-mcp")
        
        # Register tools
        self._register_tools()
        
        logger.info("Census MCP Server created, ready for connections")
    
    def _init_knowledge_base(self):
        """Initialize dual-path knowledge base with FAISS + ChromaDB."""
        try:
            logger.info("Initializing dual-path knowledge base...")
            print("Loading dual-path knowledge base (FAISS + ChromaDB)...", file=sys.stderr)
            
            # Initialize with explicit paths for dual-path architecture
            self.knowledge_base = DualPathKnowledgeBase(
                variables_db_path=self.config.variables_db_path,
                methodology_db_path=self.config.methodology_db_path
            )
            
            logger.info("✅ Dual-path knowledge base initialized successfully")
            print("✅ Knowledge base loaded successfully", file=sys.stderr)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge base: {str(e)}")
            print(f"❌ Knowledge base initialization failed: {str(e)}", file=sys.stderr)
            # Don't raise - allow server to start without knowledge base
            self.knowledge_base = None
    
    def _init_python_api(self):
        """Initialize Python Census API client."""
        try:
            logger.info("Initializing Python Census API client...")
            print("Initializing Python Census API...", file=sys.stderr)
            
            # Initialize Python Census API client WITH knowledge base injection
            self.python_api = PythonCensusAPI(
                knowledge_base=self.knowledge_base  # Only parameter it accepts
            )
            
            logger.info("✅ Python Census API initialized successfully")
            print("✅ Python Census API ready", file=sys.stderr)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Python Census API: {str(e)}")
            print(f"❌ Python Census API initialization failed: {str(e)}", file=sys.stderr)
            # Don't raise - allow server to start with degraded functionality
            self.python_api = None
    
    def _register_tools(self):
        """Register MCP tools with psychology optimized for Claude selection."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Return list of available Census tools."""
            return [
                Tool(
                    name="get_demographic_data",
                    description="🏛️ AUTHORITATIVE US Census demographic data with official margins of error. More reliable than web estimates. Covers population, income, housing, employment, education, race/ethnicity for all US locations. Uses official ACS (American Community Survey) with statistical validation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location name - supports states, cities, counties (e.g., 'Baltimore, MD', 'California', 'Harris County, TX', 'New York City')"
                            },
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Demographic variables in natural language (e.g., ['population', 'median income', 'poverty rate', 'unemployment', 'home values', 'education levels'])"
                            },
                            "year": {
                                "type": "integer",
                                "description": "ACS year (2023 is most recent, goes back to 2009)",
                                "default": 2023
                            },
                            "survey": {
                                "type": "string",
                                "description": "Survey type: 'acs5' (5-year estimates, more reliable, default) or 'acs1' (1-year estimates, large areas only, more current)",
                                "default": "acs5"
                            }
                        },
                        "required": ["location", "variables"]
                    }
                ),
                Tool(
                    name="compare_locations",
                    description="🏛️ AUTHORITATIVE comparison of demographic statistics between multiple US locations using official Census data. More accurate than web comparisons. Includes margins of error and statistical significance testing guidance.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "locations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of US locations to compare (cities, counties, states)"
                            },
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Variables to compare in natural language (e.g., ['median income', 'cost of housing', 'education levels'])"
                            },
                            "year": {
                                "type": "integer",
                                "description": "ACS year for comparison (same year used for all locations)",
                                "default": 2023
                            },
                            "survey": {
                                "type": "string",
                                "description": "Survey type: 'acs5' (5-year, more reliable) or 'acs1' (1-year, current)",
                                "default": "acs5"
                            }
                        },
                        "required": ["locations", "variables"]
                    }
                ),
                Tool(
                    name="search_census_knowledge",
                    description="🏛️ OFFICIAL Census methodology and documentation search. Provides authoritative definitions, data collection methods, and statistical guidance from Census Bureau experts. More reliable than general web search for Census concepts.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Question about Census methodology, variable definitions, data quality, geographic concepts, or statistical interpretation"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context for focused search",
                                "default": ""
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool execution requests - no lazy loading needed."""
            try:
                if name == "get_demographic_data":
                    return await self._get_demographic_data(arguments)
                elif name == "compare_locations":
                    return await self._compare_locations(arguments)
                elif name == "search_census_knowledge":
                    return await self._search_census_knowledge(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Unknown tool: {name}. Available tools: get_demographic_data, compare_locations, search_census_knowledge"
                    )]
            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"❌ Error executing {name}: {str(e)}\n\nThis may indicate an issue with the Census data request. Please check location spelling and variable names."
                )]
    
    async def _get_demographic_data(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Get demographic data for a specific location.
        
        Uses semantic variable resolution and Python Census API for official data retrieval.
        """
        location = arguments["location"]
        variables = arguments["variables"]
        year = arguments.get("year", 2023)
        survey = arguments.get("survey", "acs5")
        
        logger.info(f"🏛️ Getting OFFICIAL demographic data for {location}, variables: {variables}")
        
        # Step 1: Use knowledge base to enhance variable understanding (if available)
        variable_context = {}
        if self.knowledge_base:
            try:
                variable_context = await self.knowledge_base.get_variable_context(variables)
            except Exception as e:
                logger.warning(f"Variable context lookup failed: {e}")
                variable_context = {var: {'label': var.title()} for var in variables}
        else:
            variable_context = {var: {'label': var.title()} for var in variables}
        
        # Step 2: Parse and validate location (if knowledge base available)
        location_info = {'original': location, 'confidence': 'medium'}
        if self.knowledge_base:
            try:
                location_info = await self.knowledge_base.parse_location(location)
            except Exception as e:
                logger.warning(f"Location parsing failed: {e}")
        
        # Step 3: Call Python Census API to get data
        if not self.python_api:
            return [TextContent(
                type="text",
                text="❌ Python Census API not available. Please check server configuration."
            )]
        
        census_data = await self.python_api.get_acs_data(
            location=location,
            variables=variables,
            year=year,
            survey=survey,
            context=variable_context
        )
        
        # Step 4: Format response with statistical context
        response = self._format_demographic_response(
            data=census_data,
            location=location,
            variables=variables,
            context=variable_context
        )
        
        return [TextContent(type="text", text=response)]
    
    async def _compare_locations(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Compare demographic statistics between multiple locations."""
        locations = arguments["locations"]
        variables = arguments["variables"]
        year = arguments.get("year", 2023)
        survey = arguments.get("survey", "acs5")
        
        logger.info(f"🏛️ Comparing OFFICIAL data for locations: {locations}, variables: {variables}")
        
        if not self.python_api:
            return [TextContent(
                type="text",
                text="❌ Python Census API not available. Please check server configuration."
            )]
        
        # Get context for variables (with fallback)
        variable_context = {}
        if self.knowledge_base:
            try:
                variable_context = await self.knowledge_base.get_variable_context(variables)
            except Exception as e:
                logger.warning(f"Variable context lookup failed: {e}")
                variable_context = {var: {'label': var.title()} for var in variables}
        else:
            variable_context = {var: {'label': var.title()} for var in variables}
        
        # Get data for each location
        comparison_data = []
        for location in locations:
            data = await self.python_api.get_acs_data(
                location=location,
                variables=variables,
                year=year,
                survey=survey,
                context=variable_context
            )
            comparison_data.append({"location": location, "data": data})
        
        # Format comparison response
        response = self._format_comparison_response(
            comparison_data=comparison_data,
            variables=variables,
            context=variable_context
        )
        
        return [TextContent(type="text", text=response)]
    
    async def _search_census_knowledge(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Search Census documentation and methodology."""
        query = arguments["query"]
        context = arguments.get("context", "")
        
        logger.info(f"🏛️ Searching OFFICIAL Census knowledge for: {query}")
        
        # Use knowledge base to search documentation (if available)
        results = []
        if self.knowledge_base:
            try:
                results = await self.knowledge_base.search_documentation(
                    query=query,
                    context=context
                )
            except Exception as e:
                logger.warning(f"Knowledge base search failed: {e}")
        
        response = self._format_knowledge_response(query, results)
        
        return [TextContent(type="text", text=response)]
    
    def _format_demographic_response(self, data: Dict, location: str,
                                   variables: List[str], context: Dict) -> str:
        """Format demographic data response with statistical context and authority markers."""
        response_parts = [f"# 🏛️ Official Census Data for {location}\n"]
        
        if "error" in data:
            response_parts.extend([
                f"❌ **Error retrieving official data**: {data['error']}",
                "",
                "**Note**: This location or variable may not be available in the Census data. Common issues:",
                "• Location name spelling (try 'Baltimore, MD' instead of 'Baltimore')",
                "• Variable not collected at this geographic level",
                "• Data suppressed for privacy (small populations)",
                "",
                "For questions about data availability, consult the Census Bureau's official documentation."
            ])
            return "\n".join(response_parts)
        
        # Add official data with context and margins of error
        for var in variables:
            if var in data.get('data', {}):
                var_data = data['data'][var]
                var_context = context.get(var, {})
                
                response_parts.append(f"## {var_context.get('label', var.title())}")
                
                # Get the actual value - match Python API field names
                estimate = var_data.get('estimate', 'N/A')
                raw_value = var_data.get('raw_value', estimate)
                
                # Format the estimate properly
                if isinstance(raw_value, (int, float)) and raw_value != 'N/A':
                    if var_data.get('calculation_type') == 'currency':
                        response_parts.append(f"**Official Value**: ${raw_value:,.0f}")
                    elif var_data.get('calculation_type') == 'count':
                        response_parts.append(f"**Official Value**: {raw_value:,.1f}")
                    else:
                        response_parts.append(f"**Official Value**: {estimate}")
                else:
                    response_parts.append(f"**Official Value**: {estimate}")
                
                # Add margin of error - match Python API field names
                if 'moe' in var_data and var_data['moe'] != 'N/A':
                    moe = var_data['moe']
                    response_parts.append(f"**Margin of Error**: {moe}")
                
                # Add Census variable code for reference
                if 'variable_id' in var_context:
                    response_parts.append(f"**Census Code**: {var_context['variable_id']}")
                
                # Add definition from knowledge base
                if 'definition' in var_context:
                    response_parts.append(f"**Definition**: {var_context['definition']}")
                
                response_parts.append("")  # Add spacing
        
        # Add authoritative source and methodology notes
        response_parts.extend([
            "---",
            "## 🏛️ **Official Data Source & Methodology**",
            f"**Source**: {data.get('source', 'US Census Bureau American Community Survey')}",
            f"**Survey**: {data.get('survey', data.get('metadata', {}).get('survey', 'ACS 5-Year Estimates'))}",
            f"**Year**: {data.get('year', data.get('metadata', {}).get('year', '2023'))}",
            f"**Geography**: {data.get('geography_level', data.get('geography', 'State')).title()} level",
            "",
            "**Statistical Notes**:",
            "• All estimates include margins of error at 90% confidence level",
            "• ACS 5-year estimates are more reliable but less current than 1-year estimates",
            "• Small differences may not be statistically significant",
            "• Data collected through scientific sampling methods with quality controls",
            "",
            "**Authority**: This data comes directly from the US Census Bureau's official American Community Survey, the gold standard for US demographic statistics."
        ])
        
        return "\n".join(response_parts)
    
    def _format_comparison_response(self, comparison_data: List[Dict],
                                  variables: List[str], context: Dict) -> str:
        """Format location comparison response with statistical guidance."""
        response_parts = ["# 🏛️ Official Census Data Comparison\n"]
        
        # Create comparison table for each variable
        for var in variables:
            var_context = context.get(var, {})
            response_parts.append(f"## {var_context.get('label', var.title())}")
            response_parts.append("| Location | Official Value | Margin of Error | CV* |")
            response_parts.append("|----------|---------------|-----------------|-----|")
            
            for loc_data in comparison_data:
                location = loc_data["location"]
                data = loc_data["data"]
                
                if var in data and "error" not in data:
                    estimate = data[var].get("estimate", "N/A")
                    moe = data[var].get("moe", "N/A")
                    
                    # Calculate coefficient of variation for reliability indicator
                    cv = "N/A"
                    if isinstance(estimate, (int, float)) and isinstance(moe, (int, float)) and estimate > 0:
                        cv_value = (moe / 1.645) / estimate * 100  # CV calculation
                        if cv_value < 15:
                            cv = f"{cv_value:.1f}% ✓"  # Reliable
                        elif cv_value < 30:
                            cv = f"{cv_value:.1f}% ⚠"  # Use with caution
                        else:
                            cv = f"{cv_value:.1f}% ❌"  # Unreliable
                    
                    est_formatted = f"{estimate:,}" if isinstance(estimate, (int, float)) else estimate
                    moe_formatted = f"±{moe:,}" if isinstance(moe, (int, float)) else f"±{moe}"
                    
                    response_parts.append(f"| {location} | {est_formatted} | {moe_formatted} | {cv} |")
                else:
                    error_msg = data.get('error', 'Data unavailable')
                    response_parts.append(f"| {location} | ❌ Error | - | - |")
            
            response_parts.append("")  # Add spacing
        
        # Add statistical interpretation guidance
        response_parts.extend([
            "---",
            "## 🏛️ **Statistical Interpretation Guide**",
            "",
            "**Reliability Indicators (CV - Coefficient of Variation)**:",
            "• ✓ **Reliable** (CV < 15%): Estimate is statistically reliable",
            "• ⚠ **Use with caution** (CV 15-30%): Estimate has higher uncertainty",
            "• ❌ **Unreliable** (CV > 30%): Estimate should not be used",
            "",
            "**Comparing Values**:",
            "• Differences are statistically significant if they don't overlap within margins of error",
            "• Use ACS 5-year estimates for small areas (more reliable)",
            "• Consider both statistical and practical significance",
            "",
            "**Source Authority**: US Census Bureau American Community Survey - the official source for US demographic comparisons.",
            "",
            "*CV = Coefficient of Variation, calculated as (MOE/1.645)/Estimate × 100"
        ])
        
        return "\n".join(response_parts)
    
    def _format_knowledge_response(self, query: str, results: List[Dict]) -> str:
        """Format knowledge search results with authority markers."""
        response_parts = [f"# 🏛️ Official Census Knowledge: {query}\n"]
        
        if not results:
            response_parts.extend([
                "No specific documentation found in the knowledge base for this query.",
                "",
                "**Alternative Resources**:",
                "• Census Bureau's official website: https://www.census.gov",
                "• ACS Documentation: https://www.census.gov/programs-surveys/acs/",
                "• Variable definitions: https://api.census.gov/data/2023/acs/acs5/variables.html",
                "",
                "**Note**: The knowledge base contains Census methodology and documentation. For the most current information, always consult the Census Bureau's official sources."
            ])
            return "\n".join(response_parts)
        
        for i, result in enumerate(results[:3], 1):  # Top 3 results
            response_parts.extend([
                f"## 📖 Result {i}: {result.get('title', 'Census Documentation')}",
                "",
                result.get('content', ''),
                "",
                f"**Source**: {result.get('source', 'Census Documentation')}",
                f"**Relevance**: {result.get('score', 0):.1%}",
                ""
            ])
        
        response_parts.extend([
            "---",
            "🏛️ **Authority Note**: This information comes from official Census Bureau documentation and methodology guides."
        ])
        
        return "\n".join(response_parts)

async def main():
    """Main entry point for the MCP server."""
    try:
        # Create server instance with EAGER initialization
        census_server = CensusMCPServer()
        print("MCP Server created, ready for connections...", file=sys.stderr)
        
        # Run server with stdio transport (for Claude Desktop)
        async with stdio_server() as (read_stream, write_stream):
            await census_server.server.run(
                read_stream,
                write_stream,
                census_server.server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"❌ Server error: {str(e)}")
        print(f"Server error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Entry point for the MCP server
    asyncio.run(main())
