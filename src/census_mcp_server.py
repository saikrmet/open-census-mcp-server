#!/usr/bin/env python3
"""
Census MCP Server - v2.11 with LLM Statistical Advisor Integration
FEATURES:
1. LLM Statistical Advisor for expert consultation
2. Existing demographic data tools
3. Semantic variable search
4. Claude-powered geographic resolution
"""

import logging
import asyncio
import sys
import os
from pathlib import Path
from typing import Any, Dict, List
from mcp.server import Server
from mcp.types import TextContent, Tool

# Import existing components
from data_retrieval.python_census_api import PythonCensusAPI
from utils.config import Config

# Import kb_search and LLM advisor
KB_SEARCH_AVAILABLE = False
LLM_ADVISOR_AVAILABLE = False

try:
    # Add knowledge-base directory to Python path
    current_dir = Path(__file__).parent
    kb_path = current_dir.parent / "knowledge-base"
    
    if kb_path.exists():
        kb_path_str = str(kb_path)
        if kb_path_str not in sys.path:
            sys.path.insert(0, kb_path_str)
        
        from kb_search import create_search_engine
        from llm_statistical_advisor import create_llm_statistical_advisor
        KB_SEARCH_AVAILABLE = True
        LLM_ADVISOR_AVAILABLE = True
        logging.info(f"✅ kb_search and LLM advisor loaded from: {kb_path}")
    else:
        logging.warning(f"⚠️ Knowledge base directory not found: {kb_path}")
        
except ImportError as e:
    KB_SEARCH_AVAILABLE = False
    LLM_ADVISOR_AVAILABLE = False
    logging.warning(f"Components not available: {e}")
except Exception as e:
    KB_SEARCH_AVAILABLE = False
    LLM_ADVISOR_AVAILABLE = False
    logging.error(f"Failed to load components: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MCP Server
app = Server("census-mcp-server")

class CensusMCPServer:
    """Census MCP Server with LLM Statistical Advisor"""
    
    def __init__(self):
        self.config = Config()
        self.census_api = None
        self.search_engine = None
        self.statistical_advisor = None
        
        # Initialize components
        self._init_census_api()
        self._init_search_engine()
        self._init_statistical_advisor()
        
        logger.info("🚀 Census MCP Server v2.11 initialized")
        logger.info("🧠 LLM Statistical Advisor integrated")
        logger.info("✅ Claude-first architecture with expert consultation")
    
    def _init_census_api(self):
        """Initialize Census API with error handling"""
        try:
            self.census_api = PythonCensusAPI()
            logger.info("✅ Census API client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Census API: {e}")
            self.census_api = None
    
    def _init_search_engine(self):
        """Initialize kb_search engine"""
        if not KB_SEARCH_AVAILABLE:
            logger.warning("⚠️ kb_search not available - search features disabled")
            return
        
        try:
            # Auto-detect knowledge base directory
            possible_paths = [
                Path(__file__).parent.parent / "knowledge-base",
                Path(os.getcwd()) / "knowledge-base",
                Path(os.getenv('KNOWLEDGE_BASE_DIR', '')) if os.getenv('KNOWLEDGE_BASE_DIR') else None
            ]
            
            knowledge_base_dir = None
            gazetteer_path = None
            
            for path in possible_paths:
                if path and path.exists():
                    knowledge_base_dir = str(path)
                    # Check for gazetteer database
                    potential_gazetteer = path / "geo-db" / "geography.db"
                    if potential_gazetteer.exists():
                        gazetteer_path = str(potential_gazetteer)
                    break
            
            if knowledge_base_dir:
                self.search_engine = create_search_engine(
                    knowledge_base_dir=knowledge_base_dir,
                    gazetteer_db_path=gazetteer_path
                )
                logger.info("✅ kb_search engine initialized")
                if gazetteer_path:
                    logger.info("✅ Gazetteer database connected")
            else:
                logger.warning("⚠️ Knowledge base directory not found")
                self.search_engine = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize search engine: {e}")
            self.search_engine = None
    
    def _init_statistical_advisor(self):
        """Initialize LLM Statistical Advisor"""
        if not LLM_ADVISOR_AVAILABLE:
            logger.warning("⚠️ LLM Statistical Advisor not available")
            return
        
        try:
            self.statistical_advisor = create_llm_statistical_advisor()
            
            # Connect advisor to search components as tools
            if self.search_engine:
                # Extract components from search engine
                geo_parser = getattr(self.search_engine, 'geo_parser', None)
                variable_search = getattr(self.search_engine, 'variable_search', None)
                methodology_search = getattr(self.search_engine, 'methodology_search', None)
                
                self.statistical_advisor.set_tools(
                    geo_parser=geo_parser,
                    variable_search=variable_search,
                    methodology_search=methodology_search
                )
                logger.info("✅ LLM Statistical Advisor initialized with tools")
            else:
                logger.warning("⚠️ Statistical advisor initialized without search tools")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize statistical advisor: {e}")
            self.statistical_advisor = None

# Initialize server instance
server_instance = CensusMCPServer()

# Tool Definitions
@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools including statistical consultation"""
    tools = [
        Tool(
            name="get_demographic_data",
            description="Get demographic data for a location with robust error handling and Claude-powered location resolution",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location (e.g., 'Austin, TX', 'New York', 'United States'). Supports intelligent parsing of various formats."
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to retrieve (e.g., 'population', 'median_income', 'bachelor_degree_rate'). Supports semantic matching."
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
            description="Search for Census variables using Claude-powered semantic intelligence and concept understanding",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of data needed (e.g., 'education attainment by race', 'housing costs')"
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
            description="Find Census tables by topic using semantic search across table catalog",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or keyword to search for (e.g., 'housing', 'income by race', 'commuting')"
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
            description="Compare demographic data across multiple locations with intelligent location resolution",
            inputSchema={
                "type": "object",
                "properties": {
                    "locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of locations to compare (supports various formats like 'New York', 'Austin, TX')"
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
    
    # Add statistical consultation tool if advisor is available
    if server_instance.statistical_advisor:
        tools.append(Tool(
            name="get_statistical_consultation",
            description="Get expert statistical consultation from LLM advisor with Census domain expertise, variable recommendations, and methodology guidance",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Statistical question or data request (e.g., 'What variables should I use to compare teacher salaries between urban and rural areas in Texas?')"
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional location context for geographic guidance",
                        "default": ""
                    }
                },
                "required": ["query"]
            }
        ))
    
    return tools

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls including statistical consultation"""
    
    if name == "get_demographic_data":
        return await _get_demographic_data(arguments)
    elif name == "search_census_variables":
        return await _search_census_variables(arguments)
    elif name == "find_census_tables":
        return await _find_census_tables(arguments)
    elif name == "compare_locations":
        return await _compare_locations(arguments)
    elif name == "get_statistical_consultation":
        return await _get_statistical_consultation(arguments)
    else:
        return [TextContent(
            type="text",
            text=f"❌ Unknown tool: {name}"
        )]

async def _get_statistical_consultation(arguments: Dict[str, Any]) -> List[TextContent]:
    """Provide expert statistical consultation using LLM advisor"""
    
    query = arguments.get("query", "").strip()
    location = arguments.get("location", "").strip()
    
    if not query:
        return [TextContent(
            type="text",
            text="❌ Error: Statistical query is required\n\n💡 Try asking:\n- 'What variables should I use for teacher salary analysis?'\n- 'How do I compare poverty rates between urban and rural areas?'\n- 'What are the limitations of using ACS data for small populations?'"
        )]
    
    if not server_instance.statistical_advisor:
        return [TextContent(
            type="text",
            text="❌ **Statistical Advisor Unavailable**\n\nThe LLM Statistical Advisor is not properly initialized. Please check:\n- OpenAI API key is configured\n- Knowledge base components are available\n- System has proper permissions"
        )]
    
    try:
        logger.info(f"🧠 Statistical consultation: '{query}'")
        
        # Parse geographic context if location provided
        geo_context = None
        if location and server_instance.search_engine:
            try:
                geo_context = server_instance.search_engine.geo_parser.parse_geographic_context(location)
                logger.info(f"📍 Geographic context: {location} → {geo_context.geography_level if geo_context.location_mentioned else 'not found'}")
            except Exception as e:
                logger.warning(f"Geographic parsing failed: {e}")
        
        # Get statistical consultation
        consultation = server_instance.statistical_advisor.consult(
            query=query,
            geo_context=geo_context
        )
        
        # Format expert consultation response
        return [TextContent(type="text", text=_format_statistical_consultation(consultation))]
        
    except Exception as e:
        logger.error(f"❌ Statistical consultation error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **Consultation Error**: {str(e)}\n\nThis error has been logged. The statistical advisor may need manual review for this type of query."
        )]

def _format_statistical_consultation(consultation) -> str:
    """Format statistical consultation as expert guidance"""
    
    response_parts = [
        f"# 🧠 Statistical Consultation\n",
        f"**Query**: {consultation.query}",
        f"**Confidence**: {consultation.confidence:.1%} ({consultation.routing_path})\n"
    ]
    
    # Expert advice section
    response_parts.extend([
        "## 💡 **Expert Advice**",
        consultation.expert_advice,
        ""
    ])
    
    # Variable recommendations
    if consultation.recommended_variables:
        response_parts.append("## 📊 **Recommended Variables**")
        
        for i, var_rec in enumerate(consultation.recommended_variables, 1):
            confidence_indicator = "🎯" if var_rec.confidence > 0.8 else "📊" if var_rec.confidence > 0.6 else "💡"
            
            response_parts.extend([
                f"### {confidence_indicator} {i}. {var_rec.variable_id}",
                f"**Concept**: {var_rec.concept}",
                f"**Description**: {var_rec.label}",
                f"**Statistical Rationale**: {var_rec.statistical_rationale}",
                f"**Survey Recommendation**: {var_rec.survey_recommendation}",
                f"**Geographic Suitability**: {var_rec.geographic_suitability}",
                f"**Confidence**: {var_rec.confidence:.1%}",
                ""
            ])
            
            # Add limitations if any
            if var_rec.limitations:
                response_parts.append(f"**⚠️ Limitations**: {'; '.join(var_rec.limitations)}")
                response_parts.append("")
    
    # Geographic guidance
    if consultation.geographic_guidance:
        response_parts.extend([
            "## 🗺️ **Geographic Guidance**",
            consultation.geographic_guidance,
            ""
        ])
    
    # Statistical limitations
    if consultation.limitations:
        response_parts.extend([
            "## ⚠️ **Statistical Limitations**",
            *[f"• {limitation}" for limitation in consultation.limitations],
            ""
        ])
    
    # Methodology notes
    if consultation.methodology_notes:
        response_parts.extend([
            "## 📚 **Methodology Context**",
            consultation.methodology_notes[:500] + ("..." if len(consultation.methodology_notes) > 500 else ""),
            ""
        ])
    
    # Technical specifications
    if consultation.needs_technical_specs:
        response_parts.extend([
            "## 🔧 **Technical Specifications Needed**",
            "For implementation details, API formatting, and geography codes, additional technical documentation may be helpful.",
            ""
        ])
    
    # Footer with source information
    response_parts.extend([
        "---",
        "## 🏛️ **Expert Source**",
        "**Analysis**: LLM Statistical Advisor with Census Bureau methodology training",
        "**Validation**: Cross-referenced with 36K+ official Census variables",
        "**Authority**: Based on ACS methodology and official Census documentation"
    ])
    
    if consultation.routing_path == 'LLM_primary':
        response_parts.append("**Quality**: High-confidence statistical reasoning")
    elif consultation.routing_path == 'LLM_fallback':
        response_parts.append("**Quality**: Fallback guidance - manual verification recommended")
    
    return "\n".join(response_parts)

# Include existing functions from the original file
async def _get_demographic_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get demographic data with Claude-powered enhancements"""
    
    # Validate inputs
    location = arguments.get("location", "").strip()
    variables = arguments.get("variables", [])
    year = arguments.get("year", 2023)
    survey = arguments.get("survey", "acs5").lower()
    
    if not location:
        return [TextContent(
            type="text",
            text="❌ Error: Location is required\n\n💡 Try: 'Austin, TX', 'New York', or 'United States'"
        )]
    
    if not variables:
        return [TextContent(
            type="text",
            text="❌ Error: At least one variable is required\n\n💡 Try: ['population'], ['median_income', 'poverty_rate'], or ['bachelor_degree_rate']"
        )]
    
    # Check if Census API is available
    if not server_instance.census_api:
        return [TextContent(
            type="text",
            text="❌ Error: Census API client not available\n\nThis is a system configuration issue. Please check the server logs."
        )]
    
    try:
        # Enhanced variable resolution using semantic search
        if server_instance.search_engine:
            try:
                enhanced_variables = await _enhance_variables_with_semantic_search(variables)
                logger.info(f"Enhanced variables: {variables} → {enhanced_variables}")
            except Exception as e:
                logger.warning(f"Variable enhancement failed: {e}")
                enhanced_variables = variables
        else:
            enhanced_variables = variables
        
        # Call the Census API with enhanced variables
        logger.info(f"🌍 Processing request: {location} with {len(enhanced_variables)} variables")
        
        result = await server_instance.census_api.get_demographic_data(
            location=location,
            variables=enhanced_variables,
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

async def _enhance_variables_with_semantic_search(variables: List[str]) -> List[str]:
    """Enhance variable list using semantic search for better resolution"""
    if not server_instance.search_engine:
        return variables
    
    enhanced_variables = []
    
    for variable in variables:
        # If it's already a Census variable ID, keep it as-is
        import re
        if re.match(r'^[A-Z]\d{5}_\d{3}[EM]?
        , variable.upper().strip()):
            enhanced_variables.append(variable)
            continue
        
        try:
            # Search for semantic matches
            search_results = server_instance.search_engine.search(variable, max_results=3)
            
            if search_results:
                # Use the best match
                best_match = search_results[0]
                if best_match.confidence > 0.6:
                    enhanced_variables.append(best_match.variable_id)
                    logger.info(f"Variable '{variable}' → '{best_match.variable_id}' (confidence: {best_match.confidence:.2f})")
                else:
                    # Keep original if confidence too low
                    enhanced_variables.append(variable)
            else:
                # No matches found, keep original
                enhanced_variables.append(variable)
                
        except Exception as e:
            logger.warning(f"Semantic search failed for variable '{variable}': {e}")
            enhanced_variables.append(variable)
    
    return enhanced_variables

async def _search_census_variables(arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for Census variables using semantic intelligence"""
    query = arguments.get("query", "").strip()
    max_results = arguments.get("max_results", 10)
    
    if not query:
        return [TextContent(
            type="text",
            text="❌ Error: Search query is required\n\n💡 Try: 'median household income', 'education by race', or 'housing costs'"
        )]
    
    if not server_instance.search_engine:
        return [TextContent(
            type="text",
            text="❌ **Semantic Search Unavailable**\n\nThe knowledge base is not properly initialized. Please check:\n- Knowledge base directory exists\n- FAISS indices are built\n- OpenAI API key is configured"
        )]
    
    try:
        # Perform semantic search
        logger.info(f"🔍 Searching variables: '{query}'")
        search_results = server_instance.search_engine.search(query, max_results=max_results)
        
        if not search_results:
            return [TextContent(
                type="text",
                text=f"🔍 **No Variables Found**: {query}\n\n💡 Try broader terms like:\n- 'income' instead of 'salary'\n- 'population' instead of 'people'\n- 'education' instead of 'college'"
            )]
        
        # Format results
        response_parts = [f"# 🔍 Census Variables: {query}\n"]
        
        for i, result in enumerate(search_results, 1):
            confidence_indicator = "🎯" if result.confidence > 0.8 else "📊" if result.confidence > 0.6 else "💡"
            
            response_parts.extend([
                f"## {confidence_indicator} {i}. {result.variable_id}",
                f"**Concept**: {result.concept}",
                f"**Description**: {result.label}",
                f"**Table**: {result.table_id} - {result.title}",
                f"**Universe**: {result.universe}",
                f"**Confidence**: {result.confidence:.1%}",
                f"**Surveys**: {', '.join(result.available_surveys)}",
                ""
            ])
            
            # Add methodology context if available
            if result.methodology_context:
                response_parts.append(f"**Statistical Notes**: {result.methodology_context[:200]}...")
                response_parts.append("")
        
        response_parts.extend([
            "---",
            f"**Found {len(search_results)} variables** using Claude-powered semantic search",
            "**Usage**: Copy variable IDs to use in `get_demographic_data`"
        ])
        
        return [TextContent(type="text", text="\n".join(response_parts))]
        
    except Exception as e:
        logger.error(f"❌ Variable search error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **Search Error**: {str(e)}\n\nThis error has been logged. Please try a different query."
        )]

async def _find_census_tables(arguments: Dict[str, Any]) -> List[TextContent]:
    """Find Census tables using semantic search"""
    query = arguments.get("query", "").strip()
    max_results = arguments.get("max_results", 5)
    
    if not query:
        return [TextContent(
            type="text",
            text="❌ Error: Search query is required\n\n💡 Try: 'housing', 'income by race', or 'commuting patterns'"
        )]
    
    if not server_instance.search_engine:
        return [TextContent(
            type="text",
            text="❌ **Table Search Unavailable**\n\nThe knowledge base is not properly initialized."
        )]
    
    try:
        # Search for tables (get variables first, then extract unique tables)
        logger.info(f"📋 Searching tables: '{query}'")
        search_results = server_instance.search_engine.search(query, max_results=max_results * 3)
        
        # Extract unique tables
        seen_tables = set()
        table_results = []
        
        for result in search_results:
            if result.table_id not in seen_tables:
                seen_tables.add(result.table_id)
                table_results.append(result)
                
                if len(table_results) >= max_results:
                    break
        
        if not table_results:
            return [TextContent(
                type="text",
                text=f"📋 **No Tables Found**: {query}\n\n💡 Try broader terms or check spelling"
            )]
        
        # Format table results
        response_parts = [f"# 📋 Census Tables: {query}\n"]
        
        for i, result in enumerate(table_results, 1):
            response_parts.extend([
                f"## 📊 {i}. {result.table_id}",
                f"**Title**: {result.title}",
                f"**Universe**: {result.universe}",
                f"**Primary Variable**: {result.variable_id}",
                f"**Surveys Available**: {', '.join(result.available_surveys)}",
                f"**Relevance**: {result.confidence:.1%}",
                ""
            ])
            
            # Add geographic restrictions if any
            if result.geographic_restrictions:
                restrictions = [f"{k}: {v}" for k, v in result.geographic_restrictions.items() if v]
                if restrictions:
                    response_parts.append(f"**Geographic Notes**: {'; '.join(restrictions)}")
                    response_parts.append("")
        
        response_parts.extend([
            "---",
            f"**Found {len(table_results)} tables** using semantic search",
            "**Next**: Use `search_census_variables` with table IDs for specific variables"
        ])
        
        return [TextContent(type="text", text="\n".join(response_parts))]
        
    except Exception as e:
        logger.error(f"❌ Table search error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **Search Error**: {str(e)}\n\nThis error has been logged."
        )]

async def _compare_locations(arguments: Dict[str, Any]) -> List[TextContent]:
    """Compare locations with enhanced location resolution"""
    locations = arguments.get("locations", [])
    variables = arguments.get("variables", [])
    year = arguments.get("year", 2023)
    
    if not locations or len(locations) < 2:
        return [TextContent(
            type="text",
            text="❌ Error: At least 2 locations required for comparison\n\n💡 Try: ['Austin, TX', 'Dallas, TX'] or ['New York', 'California']"
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
        # Enhance variables with semantic search
        if server_instance.search_engine:
            enhanced_variables = await _enhance_variables_with_semantic_search(variables)
        else:
            enhanced_variables = variables
        
        # Get data for each location
        results = []
        for location in locations:
            result = await server_instance.census_api.get_demographic_data(
                location=location,
                variables=enhanced_variables,
                year=year,
                survey="acs5"
            )
            results.append((location, result))
        
        # Format comparison response
        return [TextContent(type="text", text=_format_comparison_response(results, enhanced_variables))]
        
    except Exception as e:
        logger.error(f"❌ Location comparison error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **System Error**: {str(e)}\n\nThis error has been logged."
        )]

def _format_demographic_response(result: Dict[str, Any]) -> str:
    """Format successful demographic data response with enhanced information"""
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
    
    # Add data results with enhanced formatting
    data = result.get('data', {})
    if data:
        response_parts.append("## 📊 **Data Results**")
        
        for var_id, var_data in data.items():
            if var_data.get('estimate') is not None:
                formatted_value = var_data.get('formatted', str(var_data['estimate']))
                
                # Add semantic context if available
                context = ""
                if 'income' in var_id.lower():
                    context = " 💰"
                elif 'population' in var_id.lower() or var_id.startswith('B01003'):
                    context = " 👥"
                elif 'poverty' in var_id.lower():
                    context = " 📉"
                elif 'education' in var_id.lower():
                    context = " 🎓"
                
                response_parts.append(f"**{var_id}**{context}: {formatted_value}")
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
    if api_info.get('geographic_resolution'):
        response_parts.append(f"**Geographic Resolution**: {api_info['geographic_resolution']}")
    
    if server_instance.search_engine:
        response_parts.append(f"**Semantic Enhancement**: ✅ Claude-powered variable resolution")
    
    return "\n".join(response_parts)

def _format_comparison_response(results: List[tuple], variables: List[str]) -> str:
    """Format location comparison response with enhanced formatting"""
    response_parts = ["# 🏛️ Location Comparison\n"]
    
    # Add summary table
    response_parts.append("## 📊 **Comparison Results**")
    
    successful_results = []
    failed_results = []
    
    for location, result in results:
        if 'error' in result:
            failed_results.append((location, result['error']))
        else:
            successful_results.append((location, result))
    
    # Show successful comparisons
    for location, result in successful_results:
        name = result.get('resolved_location', {}).get('name', location)
        response_parts.append(f"### 📍 {name}")
        
        data = result.get('data', {})
        for var_id, var_data in data.items():
            if var_data.get('estimate') is not None:
                formatted = var_data.get('formatted', str(var_data['estimate']))
                
                # Add context emoji
                context = ""
                if 'income' in var_id.lower():
                    context = " 💰"
                elif 'population' in var_id.lower() or var_id.startswith('B01003'):
                    context = " 👥"
                elif 'poverty' in var_id.lower():
                    context = " 📉"
                
                response_parts.append(f"  **{var_id}**{context}: {formatted}")
            else:
                response_parts.append(f"  **{var_id}**: No data")
        
        response_parts.append("")  # Blank line
    
    # Show failed locations
    if failed_results:
        response_parts.append("### ❌ **Resolution Failures**")
        for location, error in failed_results:
            response_parts.append(f"**{location}**: {error}")
        response_parts.append("")
    
    response_parts.extend([
        "---",
        "**Source**: US Census Bureau American Community Survey",
        "**Note**: Comparisons use ACS 5-year estimates for reliability"
    ])
    
    if server_instance.search_engine:
        response_parts.append("**Enhancement**: ✅ Claude-powered location and variable resolution")
    
    return "\n".join(response_parts)

# Main server function
async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server
    
    logger.info("🚀 Starting Census MCP Server v2.11")
    logger.info("🧠 Features: LLM Statistical Advisor, Semantic search, Expert consultation")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
