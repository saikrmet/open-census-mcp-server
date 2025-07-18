#!/usr/bin/env python3
"""
Census MCP Server - Containerized Census expertise via MCP protocol

Provides natural language access to US Census data through:
- Dual-path vector DB for variables and methodology
- Python Census API integration for data retrieval
- Statistical expert analysis prompts for intelligent responses
- Geographic validation and statistical context
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
from mcp import types

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
    - MCP protocol communication with prompts for statistical expertise
    - Query parsing and context enrichment via dual-path vector DB
    - Python Census API coordination for data retrieval
    - Response formatting and statistical validation
    """
    
    def __init__(self):
        """Initialize server components."""
        self.config = Config()
        
        # Initialize dual-path knowledge base
        logger.info("Initializing dual-path knowledge base...")
        self.knowledge_base = DualPathKnowledgeBase(
            variables_db_path=self.config.variables_db_path,
            methodology_db_path=self.config.methodology_db_path
        )
        
        # Initialize Python Census API client
        logger.info("Initializing Python Census API client...")
        self.census_api = PythonCensusAPI(knowledge_base=self.knowledge_base)
        
        # Create MCP server instance
        self.server = Server("census-mcp")
        
        # Register tools and prompts
        self._register_tools()
        self._register_prompts()
        
        logger.info("Census MCP Server initialized successfully")
    
    def _register_tools(self):
        """Register MCP tools that will be available to Claude Desktop."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """Return list of available Census tools."""
            return [
                types.Tool(
                    name="get_demographic_data",
                    description="Get ACS demographic data for a location with statistical context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location (e.g., 'Austin, Texas', 'Maryland', 'Baltimore County, MD')"
                            },
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Demographic variables (e.g., ['median income', 'population', 'poverty rate'])"
                            },
                            "year": {
                                "type": "integer",
                                "description": "ACS year (default: 2023)",
                                "default": 2023
                            },
                            "survey": {
                                "type": "string",
                                "description": "Survey type: 'acs1' or 'acs5' (default: 'acs5')",
                                "default": "acs5"
                            }
                        },
                        "required": ["location", "variables"]
                    }
                ),
                types.Tool(
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
                            "year": {"type": "integer", "default": 2023},
                            "survey": {"type": "string", "default": "acs5"}
                        },
                        "required": ["locations", "variables"]
                    }
                ),
                types.Tool(
                    name="search_census_knowledge",
                    description="Search Census methodology and documentation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for Census methodology/documentation"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context about what you're looking for",
                                "default": ""
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
            """Handle tool calls from Claude."""
            try:
                if name == "get_demographic_data":
                    result = await self._get_demographic_data(**arguments)
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "compare_locations":
                    result = await self._compare_locations(**arguments)
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "search_census_knowledge":
                    result = await self._search_census_knowledge(**arguments)
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Tool {name} error: {str(e)}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _register_prompts(self):
        """Register MCP prompts for statistical expert analysis."""
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[types.Prompt]:
            """List available prompts"""
            return [
                types.Prompt(
                    name="statistical_expert_analysis",
                    description="Analyze Census/ACS data with statistical expertise and plain language explanation",
                    arguments=[
                        types.PromptArgument(
                            name="data_result",
                            description="Raw Census data result from API call",
                            required=True,
                        ),
                        types.PromptArgument(
                            name="user_query",
                            description="Original user question",
                            required=True,
                        ),
                        types.PromptArgument(
                            name="methodology_notes",
                            description="Any methodological concerns or data limitations",
                            required=False,
                        ),
                    ],
                ),
                types.Prompt(
                    name="census_consultation",
                    description="Act as a Census Bureau statistical consultant for any data question",
                    arguments=[
                        types.PromptArgument(
                            name="user_question",
                            description="User's question about Census data or methodology",
                            required=True,
                        ),
                        types.PromptArgument(
                            name="available_data",
                            description="Any available data or context",
                            required=False,
                        ),
                    ],
                ),
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: dict[str, str] | None
        ) -> types.GetPromptResult:
            """Get prompt content"""
            
            if name == "statistical_expert_analysis":
                if not arguments:
                    raise ValueError("Arguments required for statistical_expert_analysis")
                
                data_result = arguments.get("data_result", "")
                user_query = arguments.get("user_query", "")
                methodology_notes = arguments.get("methodology_notes", "")
                
                prompt_content = f"""
You are a senior Census Bureau statistician and ACS (American Community Survey) data expert. Your role is to serve as a trusted statistical consultant, not just a data retriever.

ANALYSIS TASK:
User asked: "{user_query}"
Raw data result: {data_result}
{f"Methodological notes: {methodology_notes}" if methodology_notes else ""}

PROVIDE A RESPONSE THAT:

1. **Explains the data clearly**: Translate the numbers into plain language that non-statisticians understand
2. **Provides context**: What do these numbers actually mean in practical terms?
3. **Includes reliability assessment**: Explain margins of error and what they represent
4. **Flags limitations**: Point out any methodological concerns, biases, or fitness-for-use issues
5. **Suggests follow-ups**: Ask clarifying questions or suggest related data that might be useful
6. **Routes appropriately**: If Census data isn't the best source, recommend better alternatives

RESPONSE STYLE:
- Conversational and helpful, not academic or technical
- Start with the answer, then provide context
- Use phrases like "This means approximately..." and "You should be aware that..."
- Proactively explain why this data should or shouldn't be trusted
- End with a question or suggestion for further analysis

STATISTICAL WARNINGS TO CONSIDER:
- Large margins of error relative to estimates
- Small sample sizes affecting reliability  
- Mixed categories that might not match user intent
- Temporal factors affecting comparability
- Geographic aggregation issues
- When other agencies (BLS, BEA) might be better sources

Remember: Make Census data accessible, trustworthy, and actionable for non-experts.
"""
                
                return types.GetPromptResult(
                    description="Statistical expert analysis of Census/ACS data",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=prompt_content
                            ),
                        ),
                    ],
                )
            
            elif name == "census_consultation":
                if not arguments:
                    raise ValueError("Arguments required for census_consultation")
                
                user_question = arguments.get("user_question", "")
                available_data = arguments.get("available_data", "")
                
                prompt_content = f"""
You are a senior Census Bureau statistician and data expert providing consultation services. A user has come to you with a question about Census data.

USER QUESTION: "{user_question}"
{f"AVAILABLE DATA/CONTEXT: {available_data}" if available_data else ""}

As a statistical expert, provide guidance that includes:

1. **Direct answer** to their question if possible
2. **Data source recommendations** - which Census surveys/tables are most appropriate
3. **Methodological guidance** - what they should know about data quality, limitations, sampling
4. **Alternative sources** - when BLS, BEA, or other agencies might be better
5. **Follow-up questions** - what additional information would help you provide better guidance

Be conversational, helpful, and educational. Your goal is to help them get the right data in the right way for their specific needs.

Always include:
- Plain language explanations
- Warnings about data limitations when relevant
- Suggestions for next steps
- Questions to better understand their needs

Act as a trusted advisor who wants to ensure they use data appropriately and understand its limitations.
"""
                
                return types.GetPromptResult(
                    description="Census Bureau statistical consultation",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=prompt_content
                            ),
                        ),
                    ],
                )
            
            else:
                raise ValueError(f"Unknown prompt: {name}")
    
    async def _get_demographic_data(self, location: str, variables: List[str],
                                  year: int = 2023, survey: str = "acs5") -> str:
        """Get demographic data for a location with enhanced response."""
        try:
            # Get data from Census API
            result = await self.census_api.get_acs_data(location, variables, year, survey)
            
            if "error" in result:
                error_response = f"# 🏛️ Official Census Data for {location}\n\n"
                error_response += f"❌ **Error retrieving official data**: {result['error']}\n\n"
                error_response += "**Note**: This location or variable may not be available in the Census data. "
                error_response += "Common issues:\n"
                error_response += "• Location name spelling (try 'Baltimore, MD' instead of 'Baltimore')\n"
                error_response += "• Variable not collected at this geographic level\n"
                error_response += "• Data suppressed for privacy (small populations)\n\n"
                error_response += "For questions about data availability, consult the Census Bureau's official documentation."
                error_response += "\n\n**💡 Suggestion**: Use the 'statistical_expert_analysis' or 'census_consultation' prompt for expert guidance on this issue."
                return error_response
            
            # Format successful response with suggestion to use expert analysis
            formatted_response = self._format_demographic_response(result, variables)
            formatted_response += "\n\n**💡 Expert Analysis Available**: Use the 'statistical_expert_analysis' prompt with this data for detailed statistical interpretation and guidance."
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in get_demographic_data: {str(e)}")
            return f"Error retrieving demographic data: {str(e)}"
    
    async def _compare_locations(self, locations: List[str], variables: List[str],
                               year: int = 2023, survey: str = "acs5") -> str:
        """Compare demographic data across multiple locations."""
        try:
            comparison_data = []
            
            for location in locations:
                result = await self.census_api.get_acs_data(location, variables, year, survey)
                comparison_data.append({
                    "location": location,
                    "data": result.get("data", {}),
                    "error": result.get("error")
                })
            
            formatted_response = self._format_comparison_response(comparison_data, variables)
            formatted_response += "\n\n**💡 Expert Analysis Available**: Use the 'statistical_expert_analysis' prompt for detailed interpretation of these comparisons."
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in compare_locations: {str(e)}")
            return f"Error comparing locations: {str(e)}"
    
    async def _search_census_knowledge(self, query: str, context: str = "") -> str:
        """Search Census methodology and documentation."""
        try:
            # Use methodology database for conceptual queries
            results = self.knowledge_base.search_methodology(query, k=5)
            
            if not results:
                no_results_response = f"# 🏛️ Official Census Knowledge: {query}\n\n"
                no_results_response += "No specific documentation found in the knowledge base for this query.\n\n"
                no_results_response += "**Alternative Resources**:\n"
                no_results_response += "• Census Bureau's official website: https://www.census.gov\n"
                no_results_response += "• ACS Documentation: https://www.census.gov/programs-surveys/acs/\n"
                no_results_response += "• Variable definitions: https://api.census.gov/data/2023/acs/acs5/variables.html\n\n"
                no_results_response += "**Note**: The knowledge base contains Census methodology and documentation. "
                no_results_response += "For the most current information, always consult the Census Bureau's official sources."
                no_results_response += "\n\n**💡 Expert Consultation**: Use the 'census_consultation' prompt for personalized guidance on this topic."
                return no_results_response
            
            formatted_response = self._format_knowledge_response(query, results)
            formatted_response += "\n\n**💡 Expert Consultation**: Use the 'census_consultation' prompt for personalized guidance on this methodology topic."
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in search_census_knowledge: {str(e)}")
            return f"Error searching knowledge base: {str(e)}"
    
    def _format_demographic_response(self, result: Dict, variables: List[str]) -> str:
        """Format demographic data response with statistical context."""
        response_parts = [f"# 🏛️ Official Census Data for {result.get('location', 'Unknown Location')}\n"]
        
        data = result.get("data", {})
        
        for var in variables:
            if var in data and "error" not in data[var]:
                var_data = data[var]
                estimate = var_data.get("estimate", "N/A")
                moe = var_data.get("moe", "N/A")
                
                response_parts.extend([
                    f"## {var.title()}",
                    f"**Official Value**: {estimate}",
                    f"**Margin of Error**: ±{moe}",
                    f"**Definition**: Census variable: {var}",
                    ""
                ])
            else:
                response_parts.extend([
                    f"## {var.title()}",
                    "**Status**: Data not available or error in retrieval",
                    ""
                ])
        
        # Add methodological context
        response_parts.extend([
            "---",
            "## 🏛️ **Official Data Source & Methodology**",
            "**Source**: US Census Bureau American Community Survey",
            f"**Survey**: ACS 5-Year Estimates",
            f"**Year**: {result.get('year', 2023)}",
            f"**Geography**: {result.get('geography_level', 'Unknown')} level",
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
    
    def _format_comparison_response(self, comparison_data: List[Dict], variables: List[str]) -> str:
        """Format location comparison response."""
        response_parts = ["# 📊 Location Comparison\n"]
        
        # Create comparison table for each variable
        for var in variables:
            response_parts.append(f"## {var.title()}")
            response_parts.append("| Location | Value | Margin of Error |")
            response_parts.append("|----------|-------|-----------------|")
            
            for loc_data in comparison_data:
                location = loc_data["location"]
                data = loc_data["data"]
                
                if var in data and "error" not in data:
                    estimate = data[var].get("estimate", "N/A")
                    moe = data[var].get("moe", "N/A")
                    response_parts.append(f"| {location} | {estimate} | ±{moe} |")
                else:
                    response_parts.append(f"| {location} | Error | - |")
            
            response_parts.append("")  # Add spacing
        
        response_parts.extend([
            "---",
            "**Source**: US Census Bureau American Community Survey",
            "**Note**: All estimates include margins of error. Statistical significance testing recommended for comparisons."
        ])
        
        return "\n".join(response_parts)
    
    def _format_knowledge_response(self, query: str, results: List[Dict]) -> str:
        """Format knowledge search results."""
        response_parts = [f"# 🏛️ Official Census Knowledge: {query}\n"]
        
        if not results:
            response_parts.append("No relevant documentation found for this query.")
            return "\n".join(response_parts)
        
        for i, result in enumerate(results[:3], 1):  # Top 3 results
            response_parts.extend([
                f"## Result {i}: {result.get('title', 'Census Documentation')}",
                result.get('content', ''),
                f"**Source**: {result.get('source', 'Census Bureau Documentation')}",
                f"**Relevance**: {result.get('score', 0):.3f}",
                ""
            ])
        
        response_parts.extend([
            "---",
            "**Note**: This information comes from Census Bureau methodology documentation and training materials."
        ])
        
        return "\n".join(response_parts)

async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting Census MCP Server with Statistical Expert Prompts...")
    
    try:
        # Create server instance
        census_server = CensusMCPServer()
        
        # Run server with stdio transport (for Claude Desktop)
        async with stdio_server() as (read_stream, write_stream):
            await census_server.server.run(
                read_stream,
                write_stream,
                census_server.server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Entry point for the MCP server
    asyncio.run(main())
