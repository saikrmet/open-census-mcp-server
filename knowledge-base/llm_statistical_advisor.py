#!/usr/bin/env python3
"""
LLM Statistical Advisor - Core Census Statistical Reasoning Engine

Provides expert-level statistical consultation using LLM reasoning with:
- Census methodology knowledge
- Geographic intelligence via handler calls
- Variable validation via semantic search
- Survey selection logic (ACS1 vs ACS5)
- Statistical fitness assessments

Acts as orchestrator that uses modular components as tools.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from openai import OpenAI

# Import existing modular components as tools
from geographic_parsing import GeographicContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StatisticalRecommendation:
    """Recommended variable with statistical rationale"""
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
    """Complete statistical consultation response"""
    query: str
    confidence: float
    expert_advice: str
    recommended_variables: List[StatisticalRecommendation]
    geographic_guidance: str
    limitations: List[str]
    methodology_notes: str
    needs_technical_specs: bool
    routing_path: str
    technical_specs: Optional[Dict] = None

class LLMStatisticalAdvisor:
    """
    Core statistical reasoning engine that provides Census expertise
    
    Uses LLM for statistical reasoning while calling modular components
    for validation and technical details.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY required for statistical advisor")
        
        # Initialize component references (will be set by parent system)
        self.geo_parser = None
        self.variable_search = None
        self.methodology_search = None
        
        logger.info("✅ LLM Statistical Advisor initialized")
    
    def set_tools(self, geo_parser, variable_search, methodology_search):
        """Set references to modular components for tool calls"""
        self.geo_parser = geo_parser
        self.variable_search = variable_search
        self.methodology_search = methodology_search
        logger.info("✅ Statistical advisor tools configured")
    
    def consult(self, query: str, geo_context: GeographicContext = None, 
                var_context: Dict = None) -> StatisticalConsultation:
        """
        Provide expert statistical consultation
        
        Args:
            query: User's question or request
            geo_context: Parsed geographic context
            var_context: Variable preprocessing context
            
        Returns:
            Complete statistical consultation with recommendations
        """
        logger.info(f"Statistical consultation: '{query}'")
        
        # Step 1: Initial LLM statistical analysis
        initial_analysis = self._get_initial_statistical_analysis(query, geo_context, var_context)
        
        # Step 2: Variable validation and discovery
        validated_variables = self._validate_and_discover_variables(
            initial_analysis, geo_context
        )
        
        # Step 3: Methodology context if needed
        methodology_context = self._get_methodology_context(
            query, initial_analysis.get('methodology_keywords', [])
        )
        
        # Step 4: Final expert synthesis
        final_consultation = self._synthesize_expert_consultation(
            query, initial_analysis, validated_variables, methodology_context, geo_context
        )
        
        logger.info(f"Statistical consultation complete: confidence {final_consultation.confidence:.3f}")
        return final_consultation
    
    def _get_initial_statistical_analysis(self, query: str, 
                                        geo_context: GeographicContext = None,
                                        var_context: Dict = None) -> Dict:
        """Get initial LLM analysis of statistical requirements"""
        
        geo_info = ""
        if geo_context and geo_context.location_mentioned:
            geo_info = f"\nGeographic context: {geo_context.location_text} ({geo_context.geography_level})"
        
        var_info = ""
        if var_context:
            var_info = f"\nVariable preprocessing: {var_context.get('search_strategy', 'standard')} search"
        
        prompt = f"""You are a Census Bureau statistical expert providing consultation on data requests.

Query: "{query}"{geo_info}{var_info}

Provide initial statistical analysis covering:

1. Statistical Requirements:
   - What type of demographic/economic measure is needed?
   - What population universe should be considered?
   - Are there sampling/reliability considerations?

2. Variable Strategy:
   - What Census table families are most relevant? (e.g., B01003 for population, B19013 for income)
   - Should focus on totals (_001E) or breakdowns (_002E, _003E, etc.)?
   - Any occupation, race, age, or other demographic breakdowns needed?

3. Geographic Considerations:
   - What geographic level is most appropriate for this analysis?
   - Are there any geographic limitations or reliability concerns?
   - Would metropolitan area vs city vs county make a difference?

4. Survey Selection Logic:
   - ACS 1-year vs 5-year considerations for this geography and variable type
   - Sample size and reliability trade-offs
   - Any time-series or trend analysis needs?

5. Statistical Limitations:
   - Potential data quality issues to warn about
   - Margin of error considerations
   - Universe/denominator issues
   - Any methodological caveats?

6. Methodology Keywords:
   - Key methodology concepts to research further (e.g., "poverty calculation", "urban definition")

Respond in JSON format with keys: statistical_requirements, variable_strategy, geographic_considerations, survey_selection, limitations, methodology_keywords, confidence_level (0.0-1.0)"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            analysis_text = response.choices[0].message.content
            
            # Try to parse JSON, fallback to structured text if needed
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                # Fallback: extract key insights from text
                analysis = {
                    "statistical_requirements": "Standard demographic analysis",
                    "variable_strategy": "Use appropriate Census tables",
                    "geographic_considerations": "Match geography to analysis needs",
                    "survey_selection": "ACS5 for reliability, ACS1 for currency",
                    "limitations": ["Standard margin of error considerations"],
                    "methodology_keywords": [],
                    "confidence_level": 0.7,
                    "raw_response": analysis_text
                }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            
            # Fallback to basic analysis
            return {
                "statistical_requirements": "Basic demographic data request",
                "variable_strategy": "Standard variable selection needed",
                "geographic_considerations": "Geographic level appropriate for query",
                "survey_selection": "ACS5 recommended for reliability",
                "limitations": ["Standard data quality considerations"],
                "methodology_keywords": [],
                "confidence_level": 0.5,
                "fallback": True
            }
    
    def _validate_and_discover_variables(self, analysis: Dict, 
                                       geo_context: GeographicContext) -> List[Dict]:
        """Use semantic search to validate and discover relevant variables"""
        
        if not self.variable_search:
            logger.warning("Variable search not available for validation")
            return []
        
        variable_strategy = analysis.get('variable_strategy', '')
        
        # Extract potential search terms
        search_terms = []
        
        # Look for table families mentioned (e.g., "B01003", "B19013")
        import re
        table_matches = re.findall(r'B\d{5}', variable_strategy)
        search_terms.extend(table_matches)
        
        # Look for concept keywords
        concept_keywords = [
            'population', 'income', 'poverty', 'employment', 'housing', 
            'education', 'occupation', 'race', 'age', 'gender'
        ]
        
        for keyword in concept_keywords:
            if keyword.lower() in variable_strategy.lower():
                search_terms.append(keyword)
        
        # Perform semantic search validation
        validated_variables = []
        
        for search_term in search_terms[:3]:  # Limit to top 3 terms
            try:
                # Global search for discovery
                results = self.variable_search.search_variables_global(
                    search_term, geo_context, k=5
                )
                
                for result in results:
                    var_metadata = result['variable_metadata']
                    variable_id = var_metadata.get('variable_id', '')
                    
                    validated_variables.append({
                        'variable_id': variable_id,
                        'concept': var_metadata.get('concept', ''),
                        'label': var_metadata.get('label', ''),
                        'search_term': search_term,
                        'semantic_score': result['semantic_score'],
                        'geographic_score': result['geographic_score'],
                        'final_score': result['final_score'],
                        'metadata': var_metadata
                    })
                    
            except Exception as e:
                logger.warning(f"Variable validation failed for '{search_term}': {e}")
        
        # Sort by final score and return top candidates
        validated_variables.sort(key=lambda x: x['final_score'], reverse=True)
        return validated_variables[:10]  # Top 10 candidates
    
    def _get_methodology_context(self, query: str, methodology_keywords: List[str]) -> Optional[str]:
        """Get methodology context for statistical guidance"""
        
        if not self.methodology_search or not methodology_keywords:
            return None
        
        try:
            # Search for methodology docs using keywords
            search_query = " ".join(methodology_keywords[:3])  # Top 3 keywords
            
            methodology_results = self.methodology_search.search_methodology(search_query, k=3)
            
            if methodology_results:
                # Combine top methodology insights
                context_parts = []
                for result in methodology_results[:2]:  # Top 2 results
                    content = result.get('content', '')[:300]  # Limit length
                    context_parts.append(content)
                
                return " ".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Methodology search failed: {e}")
        
        return None
    
    def _synthesize_expert_consultation(self, query: str, analysis: Dict, 
                                      validated_variables: List[Dict],
                                      methodology_context: Optional[str],
                                      geo_context: GeographicContext) -> StatisticalConsultation:
        """Synthesize final expert consultation using LLM"""
        
        # Prepare context for final synthesis
        var_context = ""
        if validated_variables:
            var_context = "\nValidated Variables Found:\n"
            for var in validated_variables[:5]:  # Top 5
                var_context += f"- {var['variable_id']}: {var['label']} (confidence: {var['final_score']:.3f})\n"
        
        method_context = ""
        if methodology_context:
            method_context = f"\nMethodology Context:\n{methodology_context}"
        
        geo_info = ""
        if geo_context and geo_context.location_mentioned:
            geo_info = f"\nGeographic Context: {geo_context.location_text} ({geo_context.geography_level})"
        
        prompt = f"""You are providing final statistical consultation as a Census Bureau expert.

Original Query: "{query}"

Initial Analysis:
{json.dumps(analysis, indent=2)}
{var_context}{method_context}{geo_info}

Provide expert consultation covering:

1. Expert Advice (2-3 sentences):
   - Clear, actionable guidance for this data request
   - Key statistical considerations

2. Recommended Variables (if any found):
   - Which specific Census variables to use and why
   - Statistical rationale for each recommendation
   - Survey recommendation (ACS1 vs ACS5) with justification

3. Geographic Guidance:
   - Optimal geographic level for this analysis
   - Any geographic limitations or alternatives

4. Limitations & Caveats:
   - Key statistical limitations to be aware of
   - Data quality considerations
   - Methodological caveats

5. Technical Specs Needed:
   - Whether API reference docs would help (true/false)

Provide practical, expert-level guidance that a trained Census analyst would give.

Respond in JSON format with keys: expert_advice, recommended_variables (array), geographic_guidance, limitations (array), technical_specs_needed (boolean), confidence (0.0-1.0)"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            synthesis_text = response.choices[0].message.content
            
            try:
                synthesis = json.loads(synthesis_text)
            except json.JSONDecodeError:
                # Fallback synthesis
                synthesis = {
                    "expert_advice": "Statistical analysis requires careful variable selection and geographic consideration.",
                    "recommended_variables": [],
                    "geographic_guidance": "Choose geographic level appropriate for analysis scope.",
                    "limitations": ["Standard margin of error and sampling considerations apply."],
                    "technical_specs_needed": False,
                    "confidence": 0.6
                }
            
            # Convert to StatisticalConsultation object
            recommendations = []
            
            for var_rec in synthesis.get('recommended_variables', []):
                if isinstance(var_rec, dict):
                    # Find matching validated variable for metadata
                    var_metadata = None
                    var_id = var_rec.get('variable_id', '')
                    
                    for val_var in validated_variables:
                        if val_var['variable_id'] == var_id:
                            var_metadata = val_var
                            break
                    
                    if var_metadata:
                        recommendations.append(StatisticalRecommendation(
                            variable_id=var_id,
                            concept=var_metadata['concept'],
                            label=var_metadata['label'],
                            confidence=var_metadata['final_score'],
                            statistical_rationale=var_rec.get('rationale', 'Recommended based on statistical analysis'),
                            survey_recommendation=var_rec.get('survey', 'ACS5'),
                            geographic_suitability=var_rec.get('geographic_notes', 'Suitable for requested geography'),
                            limitations=var_rec.get('limitations', [])
                        ))
            
            return StatisticalConsultation(
                query=query,
                confidence=synthesis.get('confidence', 0.7),
                expert_advice=synthesis.get('expert_advice', ''),
                recommended_variables=recommendations,
                geographic_guidance=synthesis.get('geographic_guidance', ''),
                limitations=synthesis.get('limitations', []),
                methodology_notes=methodology_context or '',
                needs_technical_specs=synthesis.get('technical_specs_needed', False),
                routing_path='LLM_primary'
            )
            
        except Exception as e:
            logger.error(f"Expert synthesis failed: {e}")
            
            # Fallback consultation
            return StatisticalConsultation(
                query=query,
                confidence=0.5,
                expert_advice="Statistical consultation requires expert review of available Census variables and methodology.",
                recommended_variables=[],
                geographic_guidance="Geographic level should match analysis requirements.",
                limitations=["Manual review recommended for optimal variable selection."],
                methodology_notes='',
                needs_technical_specs=False,
                routing_path='LLM_fallback'
            )
    
    def quick_variable_assessment(self, variable_id: str, 
                                geographic_level: str = None) -> Dict:
        """Quick assessment of a specific variable's fitness"""
        
        if not self.variable_search:
            return {"error": "Variable search not available"}
        
        try:
            var_info = self.variable_search.get_variable_info(variable_id)
            
            if not var_info:
                return {"error": f"Variable {variable_id} not found"}
            
            # LLM assessment of variable fitness
            prompt = f"""Assess this Census variable for statistical fitness:

Variable: {variable_id}
Label: {var_info.get('label', 'N/A')}
Concept: {var_info.get('concept', 'N/A')}
Geographic Level Requested: {geographic_level or 'Not specified'}

Provide brief assessment:
1. What this variable measures
2. Statistical reliability considerations  
3. Geographic suitability notes
4. Any important limitations

Keep response under 200 words, practical and expert-focused."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            assessment = response.choices[0].message.content
            
            return {
                "variable_id": variable_id,
                "assessment": assessment,
                "metadata": var_info
            }
            
        except Exception as e:
            logger.error(f"Variable assessment failed: {e}")
            return {"error": f"Assessment failed: {str(e)}"}

# Factory function for integration
def create_llm_statistical_advisor() -> LLMStatisticalAdvisor:
    """Create LLM Statistical Advisor instance"""
    return LLMStatisticalAdvisor()

if __name__ == "__main__":
    # Test the statistical advisor
    try:
        advisor = create_llm_statistical_advisor()
        
        # Mock geographic context
        from geographic_parsing import GeographicContext
        geo_context = GeographicContext(
            location_mentioned=True,
            location_text="Austin, Texas",
            geography_level="place"
        )
        
        # Test consultation
        consultation = advisor.consult(
            "What's the median household income in Austin, Texas?",
            geo_context=geo_context
        )
        
        print(f"Consultation confidence: {consultation.confidence:.3f}")
        print(f"Expert advice: {consultation.expert_advice}")
        print(f"Recommended variables: {len(consultation.recommended_variables)}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
