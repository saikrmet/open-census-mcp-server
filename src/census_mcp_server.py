#!/usr/bin/env python3
"""
Census MCP Server - Simple vs Complex Query Routing Implementation

Flow:
- Simple queries (single variable + single location) → Direct LLM URL construction
- Complex queries (everything else) → Full methodology RAG treatment
"""

import asyncio
import json
import logging
import os
import sys
import re
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

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

@dataclass
class Geography:
    name: str
    geo_type: str  # state, place, county, zcta, etc.
    state_fips: Optional[str] = None
    place_fips: Optional[str] = None
    county_fips: Optional[str] = None
    zcta_code: Optional[str] = None

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
    Census MCP server with simple vs complex query routing.
    
    Simple: Single variable + single location -> Direct LLM construction
    Complex: Everything else -> Methodology RAG
    """
    
    def __init__(self):
        """Initialize server components."""
        logger.info("Initializing Census MCP Server with Simple/Complex Routing...")
        
        self.api_key = os.environ.get('CENSUS_API_KEY')
        self.base_url = "https://api.census.gov/data/2022/acs/acs5"
        
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
        
        logger.info("✅ Census MCP Server with Simple/Complex Routing initialized")
    
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
        """Register tools with simple/complex routing."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Return list of available Census tools."""
            tools = [
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
                
                # FALLBACK TOOLS
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
                            "variables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Census variable IDs or concepts"
                            },
                            "geography_type": {
                                "type": "string",
                                "description": "Type of geography: place, county, state, cbsa, zcta",
                                "enum": ["place", "county", "state", "cbsa", "zcta", "us"]
                            },
                            "state_fips": {"type": "string", "description": "2-digit state FIPS code"},
                            "place_fips": {"type": "string", "description": "5-digit place FIPS code"},
                            "county_fips": {"type": "string", "description": "3-digit county FIPS code"},
                            "cbsa_code": {"type": "string", "description": "5-digit CBSA code"},
                            "zcta_code": {"type": "string", "description": "5-digit ZCTA code"}
                        },
                        "required": ["variables", "geography_type"]
                    }
                )
            ]
            
            # Add variable search if available
            if self.search_engine:
                tools.append(Tool(
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
                ))
            
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "get_census_data":
                    return await self._get_census_data_with_routing(arguments)
                elif name == "resolve_geography":
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
    
    async def _get_census_data_with_routing(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Main routing logic: Simple vs Complex path decision.
        
        Simple: Single variable + single location -> Direct LLM construction
        Complex: Everything else -> Methodology RAG
        """
        location = arguments.get("location", "").strip()
        variables = arguments.get("variables", [])
        include_methodology = arguments.get("include_methodology", True)
        
        if not location or not variables:
            return [TextContent(
                type="text",
                text="Please provide both location and variables."
            )]
        
        # Parse locations to detect multi-location queries
        locations = self._parse_locations(location)
        
        # ROUTING DECISION
        if len(variables) == 1 and len(locations) == 1:
            logger.info(f"🚀 SIMPLE PATH: {variables[0]} for {location}")
            return await self._simple_direct_construction(location, variables[0])
        else:
            logger.info(f"🧠 COMPLEX PATH: {len(variables)} variables, {len(locations)} locations")
            return await self._complex_methodology_guided(location, variables, include_methodology)
    
    def _parse_locations(self, location: str) -> List[str]:
        """Parse location string to detect multiple locations."""
        # Simple heuristics for multi-location detection
        if " and " in location.lower():
            return [loc.strip() for loc in location.split(" and ")]
        if " vs " in location.lower():
            return [loc.strip() for loc in location.split(" vs ")]
        if ", " in location and location.count(",") > 1:
            # "Seattle, WA, Portland, OR" style
            parts = location.split(", ")
            if len(parts) >= 4:  # Assume city, state, city, state pattern
                return [f"{parts[i]}, {parts[i+1]}" for i in range(0, len(parts)-1, 2)]
        
        return [location.strip()]
    
    async def _simple_direct_construction(self, location: str, variable: str) -> List[TextContent]:
        """
        Simple path: Single variable, single location.
        LLM constructs Census API URL directly - no RAG needed.
        """
        try:
            # Step 1: LLM parses geography
            geo_info = self._llm_parse_geography(location)
            if not geo_info:
                # Fallback to complex path on parsing failure
                logger.warning(f"Simple path geo parsing failed for: {location}")
                return await self._complex_methodology_guided(location, [variable], True)
            
            # Step 2: LLM resolves variable
            resolved_var = self._llm_resolve_variable(variable)
            if not resolved_var:
                logger.warning(f"Simple path variable resolution failed for: {variable}")
                return await self._complex_methodology_guided(location, [variable], True)
            
            # Step 3: Construct API parameters directly
            params = self._build_simple_api_params(resolved_var, geo_info)
            if "error" in params:
                logger.warning(f"Simple path API params failed: {params['error']}")
                return await self._complex_methodology_guided(location, [variable], True)
            
            # Step 4: Make direct API call
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Step 5: Parse and return results
            data = response.json()
            return self._format_simple_response(data, resolved_var, location, geo_info)
            
        except Exception as e:
            logger.warning(f"Simple path failed: {e}, falling back to complex")
            # Always fall back to complex methodology path on any failure
            return await self._complex_methodology_guided(location, [variable], True)
    
    def _llm_parse_geography(self, location: str) -> Optional[Geography]:
        """
        LLM geographic parsing - I CAN do this 99% of the time.
        
        Examples:
        - "Seattle, WA" -> Geography(name="Seattle", geo_type="place", state_fips="53", place_fips="63000")
        - "Texas" -> Geography(name="Texas", geo_type="state", state_fips="48")  
        - "90210" -> Geography(name="90210", geo_type="zcta", zcta_code="90210")
        """
        location = location.strip()
        
        # ZIP code pattern
        if re.match(r'^\d{5}$', location):
            return Geography(
                name=location,
                geo_type="zcta",
                zcta_code=location
            )
        
        # State patterns
        state_mappings = {
            "texas": Geography("Texas", "state", "48"),
            "tx": Geography("Texas", "state", "48"),
            "california": Geography("California", "state", "06"),
            "ca": Geography("California", "state", "06"),
            "florida": Geography("Florida", "state", "12"),
            "fl": Geography("Florida", "state", "12"),
            "new york": Geography("New York", "state", "36"),
            "ny": Geography("New York", "state", "36"),
            "washington": Geography("Washington", "state", "53"),
            "wa": Geography("Washington", "state", "53"),
            "oregon": Geography("Oregon", "state", "41"),
            "or": Geography("Oregon", "state", "41"),
            "illinois": Geography("Illinois", "state", "17"),
            "il": Geography("Illinois", "state", "17"),
            "missouri": Geography("Missouri", "state", "29"),
            "mo": Geography("Missouri", "state", "29"),
        }
        
        if location.lower() in state_mappings:
            return state_mappings[location.lower()]
        
        # City, State patterns
        city_state_pattern = r'^(.+),\s*([A-Z]{2})$'
        match = re.match(city_state_pattern, location)
        if match:
            city, state = match.groups()
            
            # Major city mappings (I know these)
            major_cities = {
                ("seattle", "wa"): Geography("Seattle, WA", "place", "53", "63000"),
                ("portland", "or"): Geography("Portland, OR", "place", "41", "59000"),
                ("austin", "tx"): Geography("Austin, TX", "place", "48", "05000"),
                ("chicago", "il"): Geography("Chicago, IL", "place", "17", "14000"),
                ("houston", "tx"): Geography("Houston, TX", "place", "48", "35000"),
                ("phoenix", "az"): Geography("Phoenix, AZ", "place", "04", "55000"),
                ("dallas", "tx"): Geography("Dallas, TX", "place", "48", "19000"),
                ("san antonio", "tx"): Geography("San Antonio, TX", "place", "48", "65000"),
                ("richmond", "va"): Geography("Richmond, VA", "place", "51", "67000"),
                ("st. louis", "mo"): Geography("St. Louis, MO", "place", "29", "65000"),
                ("st louis", "mo"): Geography("St. Louis, MO", "place", "29", "65000"),
                ("new york", "ny"): Geography("New York, NY", "place", "36", "51000"),
                ("los angeles", "ca"): Geography("Los Angeles, CA", "place", "06", "44000"),
                ("denver", "co"): Geography("Denver, CO", "place", "08", "20000"),
            }
            
            key = (city.lower(), state.lower())
            if key in major_cities:
                return major_cities[key]
        
        # If parsing fails, return None to trigger complex path fallback
        logger.info(f"Geographic parsing uncertain for: {location}")
        return None
    
    def _llm_resolve_variable(self, variable: str) -> Optional[str]:
        """
        LLM variable resolution - I know the common Census variables.
        """
        var_lower = variable.lower().strip()
        
        # Census variable IDs - already resolved
        if variable.upper().endswith('E') and '_' in variable and len(variable) >= 10:
            return variable.upper()
        
        # Common concepts I know well
        variable_mappings = {
            "population": "B01003_001E",
            "total population": "B01003_001E",
            "median income": "B19013_001E",
            "median household income": "B19013_001E",
            "poverty": "B17001_002E",
            "poverty rate": "B17001_002E",
            "unemployment": "B23025_005E",
            "unemployment rate": "B23025_005E",
            "median age": "B01002_001E",
            "age": "B01002_001E",
            "median rent": "B25064_001E",
            "rent": "B25064_001E",
            "median home value": "B25077_001E",
            "home value": "B25077_001E",
            "households": "B25001_001E",
            "total households": "B25001_001E",
        }
        
        if var_lower in variable_mappings:
            return variable_mappings[var_lower]
        
        # If uncertain, return None to trigger complex path
        logger.info(f"Variable resolution uncertain for: {variable}")
        return None
    
    def _build_simple_api_params(self, variable: str, geo_info: Geography) -> Dict[str, Any]:
        """Build Census API parameters for simple queries."""
        params = {"get": variable}
        
        if self.api_key:
            params["key"] = self.api_key
        
        # Build geography clauses based on geo_info
        if geo_info.geo_type == "state":
            params["for"] = f"state:{geo_info.state_fips}"
        
        elif geo_info.geo_type == "place":
            params["for"] = f"place:{geo_info.place_fips}"
            params["in"] = f"state:{geo_info.state_fips}"
        
        elif geo_info.geo_type == "county":
            params["for"] = f"county:{geo_info.county_fips}"
            params["in"] = f"state:{geo_info.state_fips}"
        
        elif geo_info.geo_type == "zcta":
            params["for"] = f"zip code tabulation area:{geo_info.zcta_code}"
        
        else:
            return {"error": f"Unsupported geography type: {geo_info.geo_type}"}
        
        return params
    
    def _format_simple_response(self, data: List[List], variable: str, location: str, geo_info: Geography) -> List[TextContent]:
        """Format simple Census API response."""
        if len(data) < 2:
            return [TextContent(type="text", text="❌ Invalid Census API response format")]
        
        headers = data[0]
        values = data[1]
        
        # Find variable value
        try:
            var_index = headers.index(variable)
            estimate = values[var_index]
            
            # Format the estimate
            formatted_estimate = "N/A"
            if estimate and estimate != "-":
                try:
                    num_estimate = float(estimate)
                    formatted_estimate = f"{num_estimate:,.0f}"
                except ValueError:
                    formatted_estimate = str(estimate)
            
            response_parts = [
                f"**Official Census Data Results**",
                f"📍 **Location**: {geo_info.name}",
                f"📊 **Variable**: {variable}",
                f"📈 **Value**: {formatted_estimate}",
                f"📋 **Source**: U.S. Census Bureau ACS 5-Year 2022",
                f"🚀 **Method**: Simple direct construction (optimized path)"
            ]
            
            # Add context note based on geography type
            if geo_info.geo_type == "place":
                response_parts.append("")
                response_parts.append("**Note**: Data reflects incorporated city limits, not metropolitan area.")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except (ValueError, IndexError) as e:
            return [TextContent(type="text", text=f"❌ Could not extract variable {variable}: {e}")]
    
    async def _complex_methodology_guided(self, location: str, variables: List[str], include_methodology: bool = True) -> List[TextContent]:
        """
        Complex path: Multiple variables/locations or uncertain queries.
        Uses full methodology RAG for intelligent variable selection and geography choices.
        """
        logger.info(f"Complex path processing: {len(variables)} variables for {location}")
        
        # For now, fall back to the existing streamlined implementation
        # In a full implementation, this would:
        # 1. Consult variable ontology for concept mapping
        # 2. Check geography compatibility matrix
        # 3. Validate statistical reliability (MOE analysis)
        # 4. Select optimal dataset (ACS1 vs ACS5)
        # 5. Execute multi-step data retrieval with proper joins
        
        # Delegate to existing complex implementation
        arguments = {
            "location": location,
            "variables": variables,
            "include_methodology": include_methodology
        }
        
        return await self._get_census_data_streamlined(arguments)
    
    async def _get_census_data_streamlined(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Existing streamlined implementation for complex queries.
        Uses confidence-based RAG fallbacks.
        """
        location = arguments.get("location", "").strip()
        variables = arguments.get("variables", [])
        include_methodology = arguments.get("include_methodology", True)
        
        if not location or not variables:
            return [TextContent(
                type="text",
                text="Please provide both location and variables."
            )]
        
        logger.info(f"Complex Census query: {variables} for {location}")
        
        # Step 1: Resolve location using confidence-based approach
        geo_result = self._resolve_location_with_confidence(location)
        
        if geo_result["confidence"] < 0.6:
            # Low confidence - direct to RAG geography tool
            logger.info(f"Low geo confidence ({geo_result['confidence']:.2f}), directing to RAG")
            return [TextContent(
                type="text",
                text=f"Location '{location}' is ambiguous. Use resolve_geography('{location}') to see disambiguation options."
            )]
        
        # Step 2: Resolve variables using confidence-based approach
        var_result = self._resolve_variables_with_confidence(variables)
        
        if var_result["confidence"] < 0.6:
            # Low confidence - direct to RAG variable tool
            logger.info(f"Low variable confidence ({var_result['confidence']:.2f}), directing to RAG")
            return [TextContent(
                type="text",
                text=f"Variables {variables} need clarification. Use search_census_variables('{' '.join(variables)}') to find specific Census codes."
            )]
        
        # Step 3: Use Census API for official data
        if not self.census_api:
            return [TextContent(type="text", text="Census API not available.")]
        
        try:
            results = self.census_api.get_acs_data(
                variables=list(var_result["variables"].keys()),
                **geo_result["parameters"]
            )
            
            if "error" in results:
                return [TextContent(type="text", text=f"Census API error: {results['error']}")]
            
            # Step 4: Format with methodology context
            return await self._format_response_with_methodology(
                results, geo_result, var_result, location, variables, include_methodology
            )
            
        except Exception as e:
            logger.error(f"Census API error: {e}")
            return [TextContent(type="text", text=f"Error retrieving data: {str(e)}")]
    
    def _resolve_location_with_confidence(self, location: str) -> Dict[str, Any]:
        """Use my knowledge to resolve location, return confidence score."""
        
        location_lower = location.lower().strip()
        
        # High confidence locations I know well
        if location_lower in ["st. louis", "saint louis", "st louis"]:
            return {
                "confidence": 0.95,
                "parameters": {"geography_type": "place", "state_fips": "29", "place_fips": "65000"},
                "resolved_name": "St. Louis city, Missouri"
            }
        elif location_lower == "chicago":
            return {
                "confidence": 0.95,
                "parameters": {"geography_type": "place", "state_fips": "17", "place_fips": "14000"},
                "resolved_name": "Chicago city, Illinois"
            }
        elif location_lower == "new york":
            return {
                "confidence": 0.90,
                "parameters": {"geography_type": "place", "state_fips": "36", "place_fips": "51000"},
                "resolved_name": "New York city, New York"
            }
        elif location_lower in ["texas", "tx"]:
            return {
                "confidence": 0.95,
                "parameters": {"geography_type": "state", "state_fips": "48"},
                "resolved_name": "Texas"
            }
        elif location_lower in ["missouri", "mo"]:
            return {
                "confidence": 0.95,
                "parameters": {"geography_type": "state", "state_fips": "29"},
                "resolved_name": "Missouri"
            }
        elif location_lower in ["illinois", "il"]:
            return {
                "confidence": 0.95,
                "parameters": {"geography_type": "state", "state_fips": "17"},
                "resolved_name": "Illinois"
            }
        elif location_lower in ["california", "ca"]:
            return {
                "confidence": 0.95,
                "parameters": {"geography_type": "state", "state_fips": "06"},
                "resolved_name": "California"
            }
        
        # High confidence - "City, ST" format is unambiguous
        elif "," in location and len(location.split(",")[1].strip()) == 2:
            # Format like "City, ST" - high confidence
            return {"confidence": 0.75, "parameters": {}, "resolved_name": location}
        
        # Low confidence - need RAG assistance
        else:
            return {"confidence": 0.3, "parameters": {}, "resolved_name": location}
    
    def _resolve_variables_with_confidence(self, variables: List[str]) -> Dict[str, Any]:
        """Use my knowledge to resolve variables, return confidence score."""
        
        resolved_vars = {}
        total_confidence = 0
        
        for var in variables:
            var_lower = var.lower().strip()
            
            # Census variable IDs - highest confidence
            if var.upper().endswith('E') and '_' in var and len(var) >= 10:
                resolved_vars[var.upper()] = {"label": var.upper(), "confidence": 1.0}
                total_confidence += 1.0
            
            # Common concepts I know well
            elif var_lower in ["population", "total population"]:
                resolved_vars["B01003_001E"] = {"label": "Total Population", "confidence": 0.95}
                total_confidence += 0.95
            elif var_lower in ["median income", "median household income"]:
                resolved_vars["B19013_001E"] = {"label": "Median Household Income", "confidence": 0.95}
                total_confidence += 0.95
            elif var_lower in ["poverty", "poverty rate"]:
                resolved_vars["B17001_002E"] = {"label": "Income Below Poverty Level", "confidence": 0.90}
                total_confidence += 0.90
            elif var_lower in ["unemployment", "unemployment rate"]:
                resolved_vars["B23025_005E"] = {"label": "Unemployed Population", "confidence": 0.85}
                total_confidence += 0.85
            elif var_lower in ["median age", "age"]:
                resolved_vars["B01002_001E"] = {"label": "Median Age", "confidence": 0.90}
                total_confidence += 0.90
            
            # Lower confidence - might need RAG
            else:
                total_confidence += 0.4
        
        avg_confidence = total_confidence / len(variables) if variables else 0
        
        return {
            "confidence": avg_confidence,
            "variables": resolved_vars,
            "resolved_count": len(resolved_vars)
        }
    
    async def _format_response_with_methodology(self, results: Dict, geo_result: Dict,
                                              var_result: Dict, original_location: str,
                                              original_vars: List[str], include_methodology: bool) -> List[TextContent]:
        """Format response with official data and methodology guidance."""
        
        response_parts = [
            f"**Official Census Data Results**",
            f"📍 **Location**: {results.get('location_name', geo_result['resolved_name'])}",
            f"📋 **Source**: {results.get('source', 'U.S. Census Bureau ACS 2022')}",
            f"🧠 **Method**: Complex methodology-guided path"
        ]
        
        # Data table
        if results.get("data"):
            response_parts.extend([
                "",
                "| Variable | Value | Margin of Error |",
                "|----------|-------|-----------------|"
            ])
            
            for var_id, var_data in results["data"].items():
                if isinstance(var_data, dict):
                    estimate = var_data.get("estimate", "N/A")
                    moe = var_data.get("moe", "N/A")
                    label = var_data.get("label", var_id)
                    
                    # Format numbers
                    if isinstance(estimate, (int, float)) and estimate != "N/A":
                        estimate = f"{estimate:,}"
                    if isinstance(moe, (int, float)) and moe != "N/A":
                        moe = f"±{moe:,}"
                    
                    response_parts.append(f"| {label} | {estimate} | {moe} |")
        
        # Add methodology context if requested
        if include_methodology and self.search_engine:
            try:
                # Search for methodology context using the search engine
                methodology_query = f"{' '.join(original_vars)} {original_location} methodology"
                methodology_results = self.search_engine._search_methodology(methodology_query)
                
                if methodology_results:
                    response_parts.extend([
                        "",
                        "**Statistical Context:**",
                        methodology_results[:200] + "..."
                    ])
            except Exception as e:
                logger.warning(f"Could not retrieve methodology context: {e}")
        
        # Add data fitness warnings for specific cases
        if geo_result["parameters"].get("geography_type") == "place":
            response_parts.append("")
            response_parts.append("**Note**: This data is for the incorporated city limits only, not the metropolitan area.")
        
        return [TextContent(type="text", text="\n".join(response_parts))]
    
    # FALLBACK TOOL IMPLEMENTATIONS (unchanged from original)
    async def _resolve_geography(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """FALLBACK: Resolve geography using gazetteer database."""
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
            
            # Format results for Claude to reason with
            response_parts = [f"Geographic matches for '{location}':"]
            
            for i, match in enumerate(matches, 1):
                response_parts.append(f"\n{i}. {match['name']} ({match['geography_type'].title()})")
                if match.get('state_abbrev'):
                    response_parts.append(f"   State: {match['state_abbrev']}")
                response_parts.append(f"   Confidence: {match['confidence']:.3f}")
                
                # Add FIPS codes for API calls
                fips_info = []
                if match.get('state_fips'):
                    fips_info.append(f"state_fips: {match['state_fips']}")
                if match.get('place_fips'):
                    fips_info.append(f"place_fips: {match['place_fips']}")
                if match.get('county_fips'):
                    fips_info.append(f"county_fips: {match['county_fips']}")
                
                if fips_info:
                    response_parts.append(f"   FIPS: {', '.join(fips_info)}")
            
            response_parts.append(f"\nUse get_demographic_data with the FIPS codes from your chosen location.")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
        
        except Exception as e:
            logger.error(f"Geography resolution error: {e}")
            return [TextContent(type="text", text=f"Error resolving geography: {str(e)}")]
    
    async def _get_demographic_data(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """FALLBACK: Get demographic data using resolved FIPS codes. Now supports batch queries."""
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
                f"Geography: {geography_type.title()}",
            ]
            
            if results.get("data"):
                # Check if this is a batch query
                if results.get("batch_query"):
                    # Format batch results
                    total_geos = results.get("total_geographies", 0)
                    response_parts.extend([
                        f"**Batch Query: {total_geos} geographies (sorted by first variable)**",
                        f"Location: {results.get('location_name', 'Multiple geographies')}",
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
                    
                    # Add note about sorting and additional variables
                    if len(variables) > 1:
                        response_parts.extend([
                            "",
                            f"**Note**: Showing first variable ({variables[0]}) for ranking. Full data includes {len(variables)} variables per geography."
                        ])
                    
                else:
                    # Format single geography results (original behavior)
                    response_parts.append(f"Location: {results.get('location_name', 'Unknown')}")
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
            
            response_parts.append(f"\nSource: {results.get('source', 'U.S. Census Bureau ACS 2023')}")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
        
        except Exception as e:
            logger.error(f"Census API error: {e}")
            return [TextContent(type="text", text=f"Error retrieving Census data: {str(e)}")]
    
    async def _search_census_variables(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """FALLBACK: Search variables using concept-based engine."""
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
            
            response_parts = [f"Found {len(results)} variables for: '{query}'"]
            
            for i, result in enumerate(results, 1):
                response_parts.append(f"\n{i}. {result.variable_id}")
                response_parts.append(f"   {result.label}")
                response_parts.append(f"   Confidence: {result.confidence:.3f}")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Variable search error: {e}")
            return [TextContent(type="text", text=f"Search error: {str(e)}")]

async def main():
    """Main server entry point."""
    logger.info("Starting Census MCP Server with Simple/Complex Routing...")
    
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
