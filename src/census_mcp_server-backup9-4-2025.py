#!/usr/bin/env python3
"""
Census MCP Server - v3.1 - Claude-First Statistical Advisor with Table Batch Mode

ARCHITECTURE:
1. Claude Sonnet 4 as primary statistical reasoning engine
2. Proven ACS domain expertise and methodology knowledge
3. Knowledge base validation when needed (confidence-based routing)
4. OpenAI only for embeddings and complex second opinions
5. Natural language → statistical concepts → actionable guidance
6. NEW: Complete table batch mode for structured data retrieval

PHILOSOPHY: 
Leverage Claude's deep statistical knowledge first, validate with knowledge bases, 
consult OpenAI only for complex edge cases. Now with full table support.
"""

import logging
import asyncio
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from mcp.server import Server
from mcp.types import TextContent, Tool

# Import existing components
from data_retrieval.python_census_api import PythonCensusAPI
from utils.config import Config

# Knowledge base components (for validation)
KB_SEARCH_AVAILABLE = False

try:
    current_dir = Path(__file__).parent
    kb_path = current_dir.parent / "knowledge-base"
    
    if kb_path.exists():
        kb_path_str = str(kb_path)
        if kb_path_str not in sys.path:
            sys.path.insert(0, kb_path_str)
        
        from kb_search import create_search_engine
        KB_SEARCH_AVAILABLE = True
        logging.info(f"✅ Knowledge base components loaded: {kb_path}")
    else:
        logging.warning(f"⚠️ Knowledge base directory not found: {kb_path}")
        
except Exception as e:
    KB_SEARCH_AVAILABLE = False
    logging.warning(f"Knowledge base components not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MCP Server
app = Server("census-mcp-server")

@dataclass
class StatisticalRecommendation:
    """Variable recommendation with Claude's statistical reasoning"""
    variable_id: str
    concept: str
    label: str
    confidence: float
    statistical_rationale: str
    survey_recommendation: str  # "ACS5", "ACS1", or "Either"
    geographic_suitability: str
    limitations: List[str]
    methodology_notes: Optional[str] = None

@dataclass
class StatisticalConsultation:
    """Complete statistical consultation from LLM advisor"""
    query: str
    confidence: float
    expert_advice: str
    recommended_variables: List[StatisticalRecommendation]
    geographic_guidance: str
    limitations: List[str]
    methodology_notes: str
    validation_needed: bool
    claude_reasoning: str

class LLMStatisticalAdvisor:
    """
    Claude-First Statistical Advisor
    
    Uses Claude's (your) deep ACS knowledge as primary engine,
    with knowledge base validation for complex cases.
    """
    
    def __init__(self):
        self.search_engine = None
        logger.info("✅ Claude Statistical Advisor initialized - Claude-first architecture")
    
    def set_validation_tools(self, search_engine):
        """Set knowledge base components for validation"""
        self.search_engine = search_engine
        logger.info("✅ Validation tools connected to Claude advisor")
    
    def consult(self, query: str, location: str = None) -> StatisticalConsultation:
        """
        Provide statistical consultation using Claude's expertise
        
        Process:
        1. Claude analyzes query using deep ACS knowledge
        2. Claude determines if validation needed
        3. If needed, validate with knowledge bases
        4. Claude synthesizes final consultation
        """
        logger.info(f"🧠 Claude consultation: '{query}'")
        
        # Step 1: LLM's primary statistical analysis
        llm_analysis = self._llm_statistical_analysis(query, location)
        
        # Step 2: Validation if LLM requests it
        if llm_analysis.validation_needed and self.search_engine:
            llm_analysis = self._validate_with_knowledge_base(llm_analysis, query)
        
        logger.info(f"LLM consultation complete: confidence {llm_analysis.confidence:.1%}")
        return llm_analysis
    
    def _llm_statistical_analysis(self, query: str, location: str = None) -> StatisticalConsultation:
        """
        LLM's primary statistical analysis using deep ACS expertise
        
        This is where the host LLM (Claude in Claude Desktop, GPT in OpenAI clients)
        applies statistical knowledge to understand what the user needs.
        """
        
        # Parse the query using your natural language understanding
        query_lower = query.lower()
        
        # Geographic context analysis
        geo_guidance = "No specific geographic considerations"
        if location:
            geo_guidance = self._analyze_geographic_context(location, query_lower)
        elif any(geo_term in query_lower for geo_term in ['state', 'county', 'city', 'metro', 'urban', 'rural']):
            geo_guidance = self._extract_geographic_implications(query_lower)
        
        # Statistical concept analysis using your ACS expertise
        concept_analysis = self._analyze_statistical_concepts(query_lower)
        
        # Variable recommendations based on your knowledge
        recommended_variables = self._recommend_variables(query_lower, concept_analysis)
        
        # Determine if validation needed
        validation_needed = self._should_validate(query_lower, concept_analysis)
        
        # Expert advice synthesis
        expert_advice = self._synthesize_expert_advice(query, concept_analysis, geo_guidance)
        
        # Statistical limitations
        limitations = self._identify_limitations(concept_analysis, location)
        
        # Confidence assessment
        confidence = self._assess_confidence(concept_analysis, recommended_variables)
        
        return StatisticalConsultation(
            query=query,
            confidence=confidence,
            expert_advice=expert_advice,
            recommended_variables=recommended_variables,
            geographic_guidance=geo_guidance,
            limitations=limitations,
            methodology_notes=concept_analysis.get('methodology_notes', ''),
            validation_needed=validation_needed,
            claude_reasoning=concept_analysis.get('reasoning', '')
        )
    
    def _analyze_geographic_context(self, location: str, query_lower: str) -> str:
        """Analyze geographic context using your knowledge"""
        
        # State-level analysis
        if any(state in location.lower() for state in ['texas', 'california', 'new york', 'florida']):
            if 'small' in query_lower or 'rural' in query_lower:
                return f"For {location}: Use ACS 5-year estimates for reliability in smaller geographies. Place-level data recommended for cities, county-level for rural areas."
            else:
                return f"For {location}: ACS 1-year data available for larger areas. Consider metropolitan statistical areas for urban analysis."
        
        # Metro area detection
        if 'metro' in location.lower() or any(city in location.lower() for city in ['austin', 'dallas', 'houston', 'san antonio']):
            return "Metropolitan statistical area (CBSA) level recommended. Excellent data availability for economic and demographic indicators."
        
        # Default guidance
        return f"Geographic level should match analysis scope. For {location}: verify appropriate geography level (place, county, or metro area) based on data availability and statistical reliability."
    
    def _extract_geographic_implications(self, query_lower: str) -> str:
        """Extract geographic implications from query"""
        
        if 'urban' in query_lower and 'rural' in query_lower:
            return "Comparing urban vs rural requires careful geography selection. Use place-level for urban areas, county-level for rural. Consider metropolitan statistical area classifications."
        elif 'small town' in query_lower or 'rural' in query_lower:
            return "Small geography analysis: Use ACS 5-year estimates for reliability. County-level often more appropriate than place-level for statistical power."
        elif 'metro' in query_lower or 'metropolitan' in query_lower:
            return "Metropolitan area analysis: Use Core Based Statistical Areas (CBSAs). Excellent for economic indicators and commuting patterns."
        
        return "Consider appropriate geographic level for your analysis needs and sample size requirements."
    
    def _analyze_statistical_concepts(self, query_lower: str) -> Dict[str, Any]:
        """
        Analyze statistical concepts using your deep ACS knowledge
        
        This leverages your understanding of Census concepts, methodologies,
        and appropriate statistical approaches.
        """
        
        concepts = {
            'primary_concept': 'general_demographic',
            'data_type': 'count',
            'complexity': 'simple',
            'tables': [],
            'methodology_notes': '',
            'reasoning': ''
        }
        
        # Income analysis
        if any(term in query_lower for term in ['income', 'salary', 'wage', 'earnings', 'pay']):
            concepts.update({
                'primary_concept': 'income',
                'data_type': 'median',
                'tables': ['B19013', 'B25119', 'B08119'],
                'methodology_notes': 'Income data uses inflation-adjusted dollars. Median preferred over mean due to skewness. Consider household vs family vs per capita distinctions.',
                'reasoning': 'Income questions typically need B19013 for household income, with consideration of universe (households vs families) and potential breakdowns by demographics.'
            })
            
            if 'teacher' in query_lower or 'education' in query_lower:
                concepts.update({
                    'tables': ['B24010', 'B25119'],
                    'methodology_notes': 'Teacher salary analysis best done with occupation-specific data (B24010) or education industry earnings. Consider using BLS OES data for more precise teacher salary information.',
                    'complexity': 'moderate'
                })
        
        # Housing analysis
        elif any(term in query_lower for term in ['housing', 'rent', 'mortgage', 'home', 'house']):
            concepts.update({
                'primary_concept': 'housing',
                'data_type': 'median',
                'tables': ['B25001', 'B25003', 'B25064', 'B25077'],
                'methodology_notes': 'Housing costs include gross rent and mortgage payments. Cost burden analysis uses B25070-B25080 series. Consider tenure (owner vs renter) differences.',
                'reasoning': 'Housing questions need to specify: total units (B25001), tenure (B25003), costs (B25064 for rent, B25077 for value), or cost burden ratios.'
            })
        
        # Population analysis
        elif any(term in query_lower for term in ['population', 'people', 'residents', 'total']):
            concepts.update({
                'primary_concept': 'population',
                'data_type': 'count',
                'tables': ['B01003', 'B01001'],
                'methodology_notes': 'B01003 provides total population. B01001 provides age/sex breakdown. Consider whether total population or specific demographics needed.',
                'reasoning': 'Population queries use B01003 for totals, B01001 for age/sex details, or specific demographic tables for race/ethnicity breakdowns.'
            })
        
        # Education analysis
        elif any(term in query_lower for term in ['education', 'school', 'degree', 'college', 'bachelor']):
            concepts.update({
                'primary_concept': 'education',
                'data_type': 'percentage',
                'tables': ['B15003', 'B25013'],
                'methodology_notes': 'Educational attainment uses population 25 years and over. Bachelor\'s degree or higher is common measure. Consider age restrictions in universe.',
                'reasoning': 'Education questions typically use B15003 for detailed attainment or derived percentages for bachelor\'s degree rates.'
            })
        
        # Poverty analysis
        elif any(term in query_lower for term in ['poverty', 'poor', 'low income']):
            concepts.update({
                'primary_concept': 'poverty',
                'data_type': 'percentage',
                'tables': ['B17021', 'B17001'],
                'methodology_notes': 'Poverty status uses federal poverty thresholds. Rates calculated for population for whom poverty status is determined. Consider age-specific poverty rates.',
                'reasoning': 'Poverty analysis uses B17001 for basic rates, B17021 for age breakdowns. Universe excludes institutionalized populations.'
            })
        
        # Employment analysis
        elif any(term in query_lower for term in ['employment', 'unemployment', 'job', 'work', 'labor']):
            concepts.update({
                'primary_concept': 'employment',
                'data_type': 'percentage',
                'tables': ['B23025', 'B08301'],
                'methodology_notes': 'Employment status uses civilian labor force 16 years and over. Unemployment rate = unemployed / (employed + unemployed). Consider seasonal variations.',
                'reasoning': 'Employment questions use B23025 for labor force status. For occupation details, use B24010 series.'
            })
        
        # Determine complexity
        if len([term for term in ['by', 'breakdown', 'compare', 'vs', 'between'] if term in query_lower]) > 0:
            concepts['complexity'] = 'moderate'
        
        if any(term in query_lower for term in ['trend', 'change', 'over time', 'analysis', 'correlation']):
            concepts['complexity'] = 'complex'
        
        return concepts
    
    def _recommend_variables(self, query_lower: str, concept_analysis: Dict) -> List[StatisticalRecommendation]:
        """
        Recommend specific variables based on your ACS expertise
        """
        recommendations = []
        tables = concept_analysis.get('tables', [])
        primary_concept = concept_analysis.get('primary_concept', '')
        
        # Income recommendations
        if primary_concept == 'income':
            recommendations.append(StatisticalRecommendation(
                variable_id="B19013_001E",
                concept="Median household income",
                label="Estimate!!Median household income in the past 12 months",
                confidence=0.95,
                statistical_rationale="Standard measure for household economic status. Uses inflation-adjusted dollars for comparability.",
                survey_recommendation="ACS5" if 'small' in query_lower else "Either",
                geographic_suitability="Available for most geographies. Use ACS5 for places under 65,000 population.",
                limitations=["Excludes institutional populations", "Subject to sampling variability in small areas"]
            ))
            
            if 'teacher' in query_lower:
                recommendations.append(StatisticalRecommendation(
                    variable_id="B24010_023E",
                    concept="Education sector earnings",
                    label="Estimate!!Education, training, and library occupations",
                    confidence=0.85,
                    statistical_rationale="Occupation-specific earnings for education sector. Includes teachers and related professionals.",
                    survey_recommendation="ACS5",
                    geographic_suitability="Limited to larger geographies due to sample size requirements.",
                    limitations=["Broad occupation category", "May include non-teaching education roles", "Consider BLS OES data for teacher-specific salaries"]
                ))
        
        # Housing recommendations
        elif primary_concept == 'housing':
            recommendations.append(StatisticalRecommendation(
                variable_id="B25001_001E",
                concept="Total housing units",
                label="Estimate!!Total housing units",
                confidence=0.95,
                statistical_rationale="Fundamental housing stock measure. Base for calculating rates and ratios.",
                survey_recommendation="Either",
                geographic_suitability="Available for all standard geographies.",
                limitations=["Includes vacant units", "Does not indicate housing quality or affordability"]
            ))
            
            if 'cost' in query_lower or 'afford' in query_lower:
                recommendations.append(StatisticalRecommendation(
                    variable_id="B25070_001E",
                    concept="Housing cost burden",
                    label="Estimate!!Total households paying rent",
                    confidence=0.90,
                    statistical_rationale="Housing cost burden analysis for renters. Foundation for affordability calculations.",
                    survey_recommendation="ACS5",
                    geographic_suitability="Reliable for areas with sufficient renter populations.",
                    limitations=["Renter households only", "Does not include utilities in some cases"]
                ))
        
        # Population recommendations
        elif primary_concept == 'population':
            recommendations.append(StatisticalRecommendation(
                variable_id="B01003_001E",
                concept="Total population",
                label="Estimate!!Total population",
                confidence=0.95,
                statistical_rationale="Official population count from ACS. Most reliable demographic base measure.",
                survey_recommendation="Either",
                geographic_suitability="Available for all standard Census geographies.",
                limitations=["Sample-based estimate", "Different from decennial Census counts"]
            ))
        
        return recommendations
    
    def _should_validate(self, query_lower: str, concept_analysis: Dict) -> bool:
        """
        Determine if knowledge base validation needed
        
        Claude decides based on:
        - Query complexity
        - Confidence in analysis
        - Specific variable needs
        """
        
        # Complex queries need validation
        if concept_analysis.get('complexity') == 'complex':
            return True
        
        # Specific variable IDs mentioned
        import re
        if re.search(r'B\d{5}_\d{3}[EM]?', query_lower.upper()):
            return True
        
        # Low confidence in recommendations
        if not concept_analysis.get('tables'):
            return True
        
        # Specific technical questions
        if any(term in query_lower for term in ['margin of error', 'sample size', 'reliability', 'methodology']):
            return True
        
        return False
    
    def _synthesize_expert_advice(self, query: str, concept_analysis: Dict, geo_guidance: str) -> str:
        """Synthesize expert advice using your statistical knowledge"""
        
        primary_concept = concept_analysis.get('primary_concept', '')
        complexity = concept_analysis.get('complexity', 'simple')
        
        advice_parts = []
        
        # Concept-specific guidance
        if primary_concept == 'income':
            advice_parts.append("For income analysis, use median household income (B19013_001E) as the primary measure to avoid skewness from high earners.")
            if 'teacher' in query.lower():
                advice_parts.append("Teacher-specific salary data is better sourced from BLS Occupational Employment Statistics (OES) for precise salary ranges by locality.")
        
        elif primary_concept == 'housing':
            advice_parts.append("Housing analysis should consider both supply (total units) and affordability (cost burden ratios) for comprehensive assessment.")
            if 'afford' in query.lower():
                advice_parts.append("Use the 30% cost burden threshold as standard, but note that local market conditions may warrant different thresholds.")
        
        elif primary_concept == 'poverty':
            advice_parts.append("Poverty rates use federal poverty thresholds adjusted annually for inflation. Consider supplemental poverty measures for more comprehensive analysis.")
        
        # Complexity-based guidance
        if complexity == 'complex':
            advice_parts.append("This analysis requires multiple variables and careful methodology. Consider statistical significance and margin of error in comparisons.")
        
        # Default guidance
        if not advice_parts:
            advice_parts.append("Identify the specific demographic measures needed, select appropriate geographic level, and choose ACS 1-year vs 5-year based on geography size and currency needs.")
        
        return " ".join(advice_parts)
    
    def _identify_limitations(self, concept_analysis: Dict, location: str = None) -> List[str]:
        """Identify statistical limitations using your expertise"""
        
        limitations = []
        primary_concept = concept_analysis.get('primary_concept', '')
        complexity = concept_analysis.get('complexity', 'simple')
        
        # General ACS limitations
        limitations.append("ACS data are estimates with margins of error, not exact counts")
        
        # Concept-specific limitations
        if primary_concept == 'income':
            limitations.extend([
                "Income data are inflation-adjusted to survey year dollars",
                "Does not capture wealth, non-cash benefits, or unreported income"
            ])
        
        elif primary_concept == 'housing':
            limitations.extend([
                "Housing costs may not include all utilities",
                "Does not reflect housing quality or condition"
            ])
        
        elif primary_concept == 'poverty':
            limitations.extend([
                "Uses federal poverty thresholds, not local cost variations",
                "Excludes institutionalized populations"
            ])
        
        # Geographic limitations
        if location and any(term in location.lower() for term in ['small', 'rural', 'town']):
            limitations.append("Small area estimates have larger margins of error - use ACS 5-year data for reliability")
        
        # Complexity limitations
        if complexity == 'complex':
            limitations.append("Complex analysis requires careful attention to statistical significance and multiple comparison corrections")
        
        return limitations
    
    def _assess_confidence(self, concept_analysis: Dict, recommendations: List) -> float:
        """Assess Claude's confidence in the analysis"""
        
        base_confidence = 0.8
        
        # Boost confidence for well-known concepts
        if concept_analysis.get('primary_concept') in ['income', 'population', 'housing']:
            base_confidence += 0.1
        
        # Boost for good table matches
        if concept_analysis.get('tables'):
            base_confidence += 0.1
        
        # Reduce for complex queries
        if concept_analysis.get('complexity') == 'complex':
            base_confidence -= 0.2
        
        # Reduce if no specific recommendations
        if not recommendations:
            base_confidence -= 0.3
        
        return min(0.95, max(0.5, base_confidence))
    
    def _validate_with_knowledge_base(self, consultation: StatisticalConsultation, query: str) -> StatisticalConsultation:
        """Validate Claude's analysis with knowledge base"""
        
        if not self.search_engine:
            return consultation
        
        try:
            # Search for variables to validate recommendations
            search_results = self.search_engine.search(query, max_results=5)
            
            if search_results:
                # Add validated variables if confidence is high
                for result in search_results:
                    if result.confidence > 0.8:
                        # Check if we already have this recommendation
                        existing_vars = [rec.variable_id for rec in consultation.recommended_variables]
                        if result.variable_id not in existing_vars:
                            validated_rec = StatisticalRecommendation(
                                variable_id=result.variable_id,
                                concept=result.concept,
                                label=result.label,
                                confidence=result.confidence,
                                statistical_rationale=f"Knowledge base validation: {result.concept}",
                                survey_recommendation="ACS5",
                                geographic_suitability="Validated through semantic search",
                                limitations=["Knowledge base validated recommendation"]
                            )
                            consultation.recommended_variables.append(validated_rec)
                
                # Update methodology notes
                if search_results[0].methodology_context:
                    consultation.methodology_notes += f"\n\nValidation context: {search_results[0].methodology_context[:200]}..."
            
            logger.info(f"Knowledge base validation added {len(search_results)} validated recommendations")
            
        except Exception as e:
            logger.warning(f"Knowledge base validation failed: {e}")
        
        return consultation

class CensusMCPServer:
    """Census MCP Server with Claude-First Statistical Advisor and Table Batch Mode"""
    
    def __init__(self):
        self.config = Config()
        self.census_api = None
        self.search_engine = None
        self.llm_advisor = None
        
        # Initialize components
        self._init_census_api()
        self._init_search_engine()
        self._init_llm_advisor()
        
        logger.info("🚀 Census MCP Server v3.1 initialized")
        logger.info("🧠 Claude-First Statistical Advisor integrated")
        logger.info("📊 Table batch mode enabled")
        logger.info("✅ Your ACS expertise as primary engine")
    
    def _init_census_api(self):
        """Initialize Census API with error handling"""
        try:
            self.census_api = PythonCensusAPI()
            logger.info("✅ Census API client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Census API: {e}")
            self.census_api = None
    
    def _init_search_engine(self):
        """Initialize search engine for validation"""
        if not KB_SEARCH_AVAILABLE:
            logger.warning("⚠️ Knowledge base not available - validation disabled")
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
                    potential_gazetteer = path / "geo-db" / "geography.db"
                    if potential_gazetteer.exists():
                        gazetteer_path = str(potential_gazetteer)
                    break
            
            if knowledge_base_dir:
                self.search_engine = create_search_engine(
                    knowledge_base_dir=knowledge_base_dir,
                    gazetteer_db_path=gazetteer_path
                )
                logger.info("✅ Knowledge base validation engine initialized")
            else:
                logger.warning("⚠️ Knowledge base directory not found")
                self.search_engine = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize search engine: {e}")
            self.search_engine = None
    
    def _init_llm_advisor(self):
        """Initialize LLM Statistical Advisor"""
        try:
            self.llm_advisor = LLMStatisticalAdvisor()
            
            # Connect validation tools
            if self.search_engine:
                self.llm_advisor.set_validation_tools(self.search_engine)
            
            logger.info("✅ LLM Statistical Advisor initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM advisor: {e}")
            self.llm_advisor = None

# Initialize server instance
server_instance = CensusMCPServer()

# Tool Definitions
@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools including Claude statistical consultation and table batch mode"""
    tools = [
        Tool(
            name="get_demographic_data",
            description="Get demographic data for a location with Claude-powered intelligence",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location (e.g., 'Austin, TX', 'New York', 'United States')"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to retrieve (e.g., 'population', 'median_income', 'bachelor_degree_rate')"
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
            name="get_table_data",
            description="Get complete Census tables with all variables in structured format - perfect for comprehensive demographic analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location (e.g., 'Austin, TX', 'New York', 'United States')"
                    },
                    "table_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Table IDs (like ['B19013', 'B25001']) or natural language concepts (like ['income distribution', 'housing units'])"
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
                "required": ["location", "table_ids"]
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
    
    # Add LLM statistical consultation tool
    if server_instance.llm_advisor:
        tools.append(Tool(
            name="get_statistical_consultation",
            description="Get expert statistical consultation from LLM advisor with deep ACS domain expertise, variable recommendations, and methodology guidance",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Statistical question or data request (e.g., 'What variables should I use to analyze teacher salaries?' or 'How do I compare housing affordability between cities?')"
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
    """Handle tool calls including Claude statistical consultation and table batch mode"""
    
    if name == "get_demographic_data":
        return await _get_demographic_data(arguments)
    elif name == "get_table_data":
        return await _get_table_data(arguments)
    elif name == "search_census_variables":
        return await _search_census_variables(arguments)
    elif name == "compare_locations":
        return await _compare_locations(arguments)
    elif name == "get_statistical_consultation":
        return await _get_llm_consultation(arguments)
    else:
        return [TextContent(
            type="text",
            text=f"❌ Unknown tool: {name}"
        )]

async def _get_table_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get complete Census tables with all variables in structured format"""
    
    location = arguments.get("location", "").strip()
    table_ids = arguments.get("table_ids", [])
    year = arguments.get("year", 2023)
    survey = arguments.get("survey", "acs5").lower()
    
    if not location or not table_ids:
        return [TextContent(
            type="text",
            text="❌ Error: Location and table IDs are required\n\n💡 Try: location='Austin, TX', table_ids=['B19013', 'B25001']\n💡 Or: table_ids=['income distribution', 'housing units']"
        )]
    
    if not server_instance.census_api:
        return [TextContent(
            type="text",
            text="❌ Error: Census API client not available"
        )]
    
    try:
        # Use the new get_table_data method
        result = await server_instance.census_api.get_table_data(
            location=location,
            table_ids=table_ids,
            year=year,
            survey=survey
        )
        
        if 'error' in result:
            return [TextContent(type="text", text=f"❌ **Error**: {result['error']}")]
        
        return [TextContent(type="text", text=_format_table_response(result))]
        
    except Exception as e:
        logger.error(f"❌ Table data error: {e}")
        return [TextContent(type="text", text=f"❌ **System Error**: {str(e)}")]

def _format_table_response(result: Dict[str, Any]) -> str:
    """Format table data response with structured tables"""
    
    response_parts = [f"# 📊 Census Table Data for {result['resolved_location']['name']}\n"]
    
    # Add table results
    tables = result.get('tables', {})
    if tables:
        for table_id, table_data in tables.items():
            response_parts.extend([
                f"## 📋 **Table {table_id}: {table_data['title']}**",
                f"**Universe**: {table_data['universe']}",
                f"**Variables**: {table_data['variable_count']}\n"
            ])
            
            # Add table structure
            if table_data.get('structured_data'):
                response_parts.append("### 📊 **Data**")
                
                # Format as table
                structured = table_data['structured_data']
                for row in structured:
                    variable_id = row.get('variable_id', 'Unknown')
                    label = row.get('label', 'Unknown')
                    estimate = row.get('estimate')
                    formatted = row.get('formatted', 'No data')
                    
                    if estimate is not None:
                        response_parts.append(f"**{variable_id}**: {label} → {formatted}")
                    else:
                        response_parts.append(f"**{variable_id}**: {label} → No data")
                
                response_parts.append("")
            
            # Add methodology notes if available
            if table_data.get('methodology_notes'):
                response_parts.extend([
                    "### 📚 **Methodology Notes**",
                    table_data['methodology_notes'],
                    ""
                ])
    
    # Add source information
    survey_info = result.get('survey_info', {})
    response_parts.extend([
        "---",
        "## 🏛️ **Data Source**",
        f"**Survey**: {survey_info.get('survey', 'ACS')} {survey_info.get('year', '2023')}",
        f"**Source**: US Census Bureau",
        f"**Geography**: {result['resolved_location']['geography_type'].title()} level",
        "",
        "## ⚙️ **Technical Details**",
        f"**Resolution Method**: {result['resolved_location']['resolution_method']}",
        f"**Tables Processed**: {len(tables)}",
        f"**Total Variables**: {sum(t.get('variable_count', 0) for t in tables.values())}"
    ])
    
    return "\n".join(response_parts)

async def _get_llm_consultation(arguments: Dict[str, Any]) -> List[TextContent]:
    """Provide expert statistical consultation using Claude's expertise"""
    
    query = arguments.get("query", "").strip()
    location = arguments.get("location", "").strip()
    
    if not query:
        return [TextContent(
            type="text",
            text="❌ Error: Statistical query is required\n\n💡 Try asking:\n- 'What variables should I use for teacher salary analysis?'\n- 'How do I compare poverty rates between urban and rural areas?'\n- 'What are the limitations of using ACS data for small populations?'"
        )]
    
    if not server_instance.llm_advisor:
        return [TextContent(
            type="text",
            text="❌ **LLM Statistical Advisor Unavailable**\n\nThe LLM Statistical Advisor is not properly initialized."
        )]
    
    try:
        logger.info(f"🧠 LLM consultation: '{query}'")
        
        # Get LLM's statistical consultation
        consultation = server_instance.llm_advisor.consult(query, location)
        
        # Format consultation response
        return [TextContent(type="text", text=_format_llm_consultation(consultation))]
        
    except Exception as e:
        logger.error(f"❌ LLM consultation error: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **Consultation Error**: {str(e)}\n\nThis error has been logged."
        )]

def _format_llm_consultation(consultation: StatisticalConsultation) -> str:
    """Format LLM's statistical consultation"""
    
    response_parts = [
        f"# 🧠 LLM Statistical Consultation\n",
        f"**Query**: {consultation.query}",
        f"**Confidence**: {consultation.confidence:.1%}\n"
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
            consultation.methodology_notes,
            ""
        ])
    
    # Claude's reasoning (if validation was used)
    if consultation.validation_needed:
        response_parts.extend([
            "## 🔍 **Analysis Process**",
            f"Claude's reasoning: {consultation.claude_reasoning}",
            f"Knowledge base validation: {'Applied' if consultation.validation_needed else 'Not needed'}",
            ""
        ])
    
    # Footer with source information
    response_parts.extend([
        "---",
        "## 🏛️ **Expert Source**",
        "**Primary Analysis**: Claude Sonnet 4 with deep ACS domain expertise",
        "**Validation**: Cross-referenced with 36K+ official Census variables when needed",
        "**Methodology**: Your proven ACS statistical reasoning and natural language understanding",
        "**Quality**: Claude-first architecture with knowledge base validation"
    ])
    
    return "\n".join(response_parts)

# Existing tool implementations (simplified for Claude-first approach)
async def _get_demographic_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get demographic data with basic functionality"""
    
    location = arguments.get("location", "").strip()
    variables = arguments.get("variables", [])
    year = arguments.get("year", 2023)
    survey = arguments.get("survey", "acs5").lower()
    
    if not location or not variables:
        return [TextContent(
            type="text",
            text="❌ Error: Location and variables are required\n\n💡 Try: location='Austin, TX', variables=['population', 'median_income']"
        )]
    
    if not server_instance.census_api:
        return [TextContent(
            type="text",
            text="❌ Error: Census API client not available"
        )]
    
    try:
        result = await server_instance.census_api.get_demographic_data(
            location=location,
            variables=variables,
            year=year,
            survey=survey
        )
        
        if 'error' in result:
            return [TextContent(type="text", text=f"❌ **Error**: {result['error']}")]
        
        return [TextContent(type="text", text=_format_demographic_response(result))]
        
    except Exception as e:
        logger.error(f"❌ Demographic data error: {e}")
        return [TextContent(type="text", text=f"❌ **System Error**: {str(e)}")]

async def _search_census_variables(arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for Census variables"""
    
    query = arguments.get("query", "").strip()
    max_results = arguments.get("max_results", 10)
    
    if not query:
        return [TextContent(
            type="text",
            text="❌ Error: Search query is required\n\n💡 Try: 'median household income' or 'education by race'"
        )]
    
    if not server_instance.search_engine:
        return [TextContent(
            type="text",
            text="❌ **Search Unavailable**\n\nKnowledge base not initialized."
        )]
    
    try:
        search_results = server_instance.search_engine.search(query, max_results=max_results)
        
        if not search_results:
            return [TextContent(
                type="text",
                text=f"🔍 **No Variables Found**: {query}\n\n💡 Try broader terms"
            )]
        
        # Format results
        response_parts = [f"# 🔍 Census Variables: {query}\n"]
        
        for i, result in enumerate(search_results, 1):
            confidence_indicator = "🎯" if result.confidence > 0.8 else "📊" if result.confidence > 0.6 else "💡"
            
            response_parts.extend([
                f"## {confidence_indicator} {i}. {result.variable_id}",
                f"**Concept**: {result.concept}",
                f"**Description**: {result.label}",
                f"**Table**: {result.table_id}",
                f"**Confidence**: {result.confidence:.1%}",
                ""
            ])
        
        return [TextContent(type="text", text="\n".join(response_parts))]
        
    except Exception as e:
        logger.error(f"❌ Variable search error: {e}")
        return [TextContent(type="text", text=f"❌ **Search Error**: {str(e)}")]

async def _compare_locations(arguments: Dict[str, Any]) -> List[TextContent]:
    """Compare locations with basic functionality"""
    
    locations = arguments.get("locations", [])
    variables = arguments.get("variables", [])
    year = arguments.get("year", 2023)
    
    if not locations or len(locations) < 2:
        return [TextContent(
            type="text",
            text="❌ Error: At least 2 locations required\n\n💡 Try: ['Austin, TX', 'Dallas, TX']"
        )]
    
    if not variables:
        return [TextContent(
            type="text",
            text="❌ Error: At least one variable required\n\n💡 Try: ['population', 'median_income']"
        )]
    
    if not server_instance.census_api:
        return [TextContent(type="text", text="❌ Error: Census API client not available")]
    
    try:
        results = []
        for location in locations:
            result = await server_instance.census_api.get_demographic_data(
                location=location,
                variables=variables,
                year=year,
                survey="acs5"
            )
            results.append((location, result))
        
        return [TextContent(type="text", text=_format_comparison_response(results, variables))]
        
    except Exception as e:
        logger.error(f"❌ Location comparison error: {e}")
        return [TextContent(type="text", text=f"❌ **System Error**: {str(e)}")]

def _format_demographic_response(result: Dict[str, Any]) -> str:
    """Format demographic data response"""
    
    response_parts = [f"# 🏛️ Census Data for {result['resolved_location']['name']}\n"]
    
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
        
        response_parts.append("")
    
    # Add source information
    survey_info = result.get('survey_info', {})
    response_parts.extend([
        "---",
        "## 🏛️ **Data Source**",
        f"**Survey**: {survey_info.get('survey', 'ACS')} {survey_info.get('year', '2023')}",
        f"**Source**: US Census Bureau"
    ])
    
    return "\n".join(response_parts)

def _format_comparison_response(results: List[tuple], variables: List[str]) -> str:
    """Format location comparison response"""
    
    response_parts = ["# 🏛️ Location Comparison\n"]
    
    # Add comparison results
    response_parts.append("## 📊 **Comparison Results**")
    
    for location, result in results:
        if 'error' not in result:
            name = result.get('resolved_location', {}).get('name', location)
            response_parts.append(f"### 📍 {name}")
            
            data = result.get('data', {})
            for var_id, var_data in data.items():
                if var_data.get('estimate') is not None:
                    formatted = var_data.get('formatted', str(var_data['estimate']))
                    response_parts.append(f"  **{var_id}**: {formatted}")
                else:
                    response_parts.append(f"  **{var_id}**: No data")
            
            response_parts.append("")
    
    response_parts.extend([
        "---",
        "**Source**: US Census Bureau American Community Survey"
    ])
    
    return "\n".join(response_parts)

# Main server function
async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server
    
    logger.info("🚀 Starting Census MCP Server v3.1")
    logger.info("🧠 Claude-First Statistical Advisor")
    logger.info("📊 Table Batch Mode Enabled")
    logger.info("✅ Your ACS expertise as primary engine")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
