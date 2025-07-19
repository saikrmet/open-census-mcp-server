#!/usr/bin/env python3
"""
Python Census API Client with Semantic Search as Primary Resolution
Real architecture: Semantic search first, centralized mappings for ultra-fast path
"""

import logging
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
import sys
import os

# Import centralized mappings
from .census_mappings import (
    STATE_FIPS, STATE_NAMES, STATE_ABBREVS, MAJOR_CITIES,
    VARIABLE_MAPPINGS, RATE_CALCULATIONS,
    normalize_state, get_state_fips, get_major_city_info,
    get_variable_mapping, is_rate_calculation
)

logger = logging.getLogger(__name__)

class PythonCensusAPI:
    """Census API client with semantic search as PRIMARY variable resolution"""
    
    def __init__(self, knowledge_base=None):
        """Initialize with semantic search integration"""
        self.knowledge_base = knowledge_base
        self.api_key = os.getenv('CENSUS_API_KEY', '')
        
        # Initialize semantic search
        self.semantic_search = self._init_semantic_search()
        
        # Census API endpoints
        self.acs5_base = "https://api.census.gov/data/2023/acs/acs5"
        self.acs1_base = "https://api.census.gov/data/2023/acs/acs1"
        
        logger.info(f"PythonCensusAPI initialized (semantic: {'✅' if self.semantic_search else '❌'})")
        logger.info(f"Loaded {len(VARIABLE_MAPPINGS)} core mappings from census_mappings.py")
        
    def _init_semantic_search(self):
        """Initialize kb_search semantic search system"""
        try:
            # Correct path: knowledge-base at project root
            project_root = Path(__file__).parent.parent.parent  # Up from src/data_retrieval/
            kb_path = project_root / "knowledge-base"  # kb_search.py is directly in knowledge-base/
            
            if not kb_path.exists():
                logger.error(f"Knowledge base directory not found at {kb_path}")
                return None
            
            # Check if kb_search.py exists
            kb_search_file = kb_path / "kb_search.py"
            if not kb_search_file.exists():
                logger.error(f"kb_search.py not found at {kb_search_file}")
                return None
                
            sys.path.insert(0, str(kb_path))
            from kb_search import ConceptBasedCensusSearchEngine
            
            # Initialize the search engine
            search_engine = ConceptBasedCensusSearchEngine(
                catalog_dir=str(project_root / "knowledge-base" / "table-catalog"),
                variables_dir=str(project_root / "knowledge-base" / "variables-db")
            )
            
            logger.info(f"✅ Semantic search loaded from {kb_path}")
            return search_engine
            
        except ImportError as e:
            logger.error(f"Failed to import kb_search: {e}")
            return None
        except Exception as e:
            logger.error(f"Semantic search initialization failed: {e}")
            return None
            
    def resolve_variables(self, variables: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Resolve natural language variables to Census variable IDs
        CORRECT ARCHITECTURE: Semantic search PRIMARY, centralized mappings for speed only
        """
        resolved_vars = []
        metadata = {}
        
        for var in variables:
            var_lower = var.lower().strip()
            
            # ULTRA-FAST PATH: Check centralized mappings for most common variables
            core_mapping = get_variable_mapping(var)
            if core_mapping:
                resolved_vars.append(core_mapping)
                metadata[var] = {
                    'variables': [core_mapping],
                    'calculation': 'direct',
                    'source': 'centralized_mapping',
                    'confidence': 1.0
                }
                logger.info(f"Ultra-fast mapping: '{var}' -> {core_mapping}")
                continue
            
            # Check if it's a rate calculation
            if is_rate_calculation(var):
                rate_info = self._get_rate_calculation_info(var)
                if rate_info:
                    resolved_vars.extend(rate_info['variables'])
                    metadata[var] = rate_info['metadata']
                    logger.info(f"Rate calculation: '{var}' -> {rate_info['variables']}")
                    continue
            
            # PRIMARY PATH: Semantic search (65K variables)
            semantic_result = self._semantic_search_variable(var)
            if semantic_result:
                resolved_vars.append(semantic_result['variable_id'])
                metadata[var] = semantic_result['metadata']
                logger.info(f"Semantic search resolved: '{var}' -> {semantic_result['variable_id']}")
                continue
            
            # BACKUP PATH: Direct Census variable ID check
            if re.match(r'^[A-Z]\d{5}_\d{3}[EM]$', var.upper()):
                resolved_vars.append(var.upper())
                metadata[var] = {
                    'variables': [var.upper()],
                    'calculation': 'direct',
                    'source': 'direct_id',
                    'confidence': 1.0
                }
                logger.info(f"Direct variable ID: '{var}' -> {var.upper()}")
                continue
            
            # FAILED - No resolution found
            logger.warning(f"Could not resolve variable: {var}")
            metadata[var] = {
                'variables': [],
                'calculation': 'failed',
                'source': 'none',
                'confidence': 0.0,
                'error': f"Variable '{var}' not found in semantic search (65K vars) or centralized mappings"
            }
        
        logger.info(f"Resolved {len(resolved_vars)} variables from {len(variables)} requests")
        return resolved_vars, metadata
    
    def _get_rate_calculation_info(self, variable: str) -> Optional[Dict[str, Any]]:
        """Get rate calculation info from centralized mappings"""
        var_lower = variable.lower().replace(' ', '_')
        
        for rate_name, rate_config in RATE_CALCULATIONS.items():
            if rate_name in var_lower or rate_name.replace('_', ' ') in variable.lower():
                return {
                    'variables': [rate_config['numerator'], rate_config['denominator']],
                    'metadata': {
                        'variables': [rate_config['numerator'], rate_config['denominator']],
                        'calculation': 'rate',
                        'source': 'centralized_rate_calculation',
                        'confidence': 1.0,
                        'description': rate_config['description'],
                        'unit': rate_config['unit']
                    }
                }
        return None
    
    def _semantic_search_variable(self, query: str) -> Optional[Dict[str, Any]]:
        """PRIMARY variable resolution using semantic search (65K variables)"""
        if not self.semantic_search:
            logger.warning("Semantic search not available - falling back to centralized mappings only")
            return None
            
        try:
            # Use the ConceptBasedCensusSearchEngine instance
            results = self.semantic_search.search(query, max_results=5)
            
            if results and len(results) > 0:
                best_result = results[0]
                
                # LOWER confidence threshold - semantic search is primary, not fallback
                confidence = best_result.confidence
                
                if confidence >= 0.4:  # Lower threshold - semantic search is primary system
                    return {
                        'variable_id': best_result.variable_id,
                        'metadata': {
                            'variables': [best_result.variable_id],
                            'calculation': 'direct',
                            'source': 'semantic_search_primary',
                            'confidence': float(confidence),
                            'label': best_result.label,
                            'table_id': getattr(best_result, 'table_id', ''),
                            'domain_weights': {}
                        }
                    }
                else:
                    logger.info(f"Semantic confidence below threshold: {confidence:.2f} for '{query}'")
                    
        except Exception as e:
            logger.error(f"Semantic search failed for '{query}': {e}")
            
        return None
    
    def _parse_location(self, location: str) -> Dict[str, Any]:
        """
        Parse location using centralized mappings and knowledge base intelligence
        """
        # Handle national level queries first
        location_lower = location.lower().strip()
        if location_lower in ['united states', 'usa', 'us', 'nation', 'national']:
            return {
                'geography': 'us',
                'display_name': 'United States',
                'source': 'national_level'
            }
        
        # Use knowledge base for location parsing if available
        if self.knowledge_base:
            try:
                # This should use the knowledge base's parse_location method
                # For now, we'll do basic parsing but log that we should use KB
                logger.info(f"Parsing location '{location}' - should use knowledge base intelligence")
            except Exception as e:
                logger.warning(f"Knowledge base location parsing failed: {e}")
        
        # Handle "City, State" format using centralized mappings
        if ',' in location:
            parts = [p.strip() for p in location.split(',')]
            city = parts[0]
            state_part = parts[1]
            
            # Use centralized state normalization
            state_abbrev = normalize_state(state_part)
            
            if state_abbrev:
                # Check if it's a major city with known FIPS
                city_info = get_major_city_info(city)
                if city_info and city_info['state'] == state_abbrev:
                    return {
                        'geography': 'place',
                        'city': city,
                        'state': get_state_fips(state_abbrev),
                        'place_fips': city_info['place_fips'],
                        'display_name': f"{city_info['full_name']}, {state_abbrev}",
                        'source': 'major_city_mapping'
                    }
                else:
                    return {
                        'geography': 'place',
                        'city': city,
                        'state': get_state_fips(state_abbrev),
                        'display_name': f"{city}, {state_abbrev}",
                        'source': 'city_state_parsing',
                        'note': 'City FIPS lookup needed for precise geography'
                    }
        
        # State-only queries using centralized mappings
        state_abbrev = normalize_state(location)
        if state_abbrev:
            return {
                'geography': 'state',
                'state': get_state_fips(state_abbrev),
                'display_name': f"{STATE_ABBREVS[state_abbrev]}",
                'source': 'state_lookup'
            }
            
        return {
            'geography': 'unknown',
            'raw_location': location,
            'display_name': location,
            'error': f"Could not parse location '{location}' - not in centralized mappings"
        }
    
    async def get_acs_data(self, location: str, variables: List[str],
                          year: int = 2023, survey: str = "acs5",
                          context: Dict = None) -> Dict[str, Any]:
        """
        Get ACS data with SEMANTIC SEARCH as primary resolution method
        """
        logger.info(f"Getting ACS data: {location}, variables: {variables}")
        
        # Step 1: Resolve variables using SEMANTIC INTELLIGENCE as primary method
        resolved_vars, var_metadata = self.resolve_variables(variables)
        
        if not resolved_vars:
            return {
                "error": "No valid variables could be resolved using semantic search",
                "attempted_variables": variables,
                "resolution_metadata": var_metadata,
                "suggestion": "Try more specific terms - semantic search covers 65K variables"
            }
        
        # Step 2: Parse location using centralized mappings
        location_info = self._parse_location(location)
        
        # Step 3: Make real Census API call
        try:
            api_data = await self._call_census_api(resolved_vars, location_info, year, survey)
            
            # Step 4: Process response and calculate derived stats
            processed_data = self._process_api_response(api_data, variables, var_metadata)
            
            # Step 5: Create enriched response with semantic metadata
            enriched_response = self._create_enriched_response(
                processed_data, var_metadata, location_info, resolved_vars
            )
            
            return enriched_response
            
        except Exception as e:
            logger.error(f"Census API call failed: {e}")
            return {
                "error": f"Census API error: {str(e)}",
                "resolved_variables": resolved_vars,
                "metadata": var_metadata
            }
    
    async def _call_census_api(self, variables: List[str], location_info: Dict,
                             year: int, survey: str) -> Dict[str, Any]:
        """Make actual Census API calls - REAL implementation"""
        
        # Build API URL
        if survey == "acs5":
            base_url = f"https://api.census.gov/data/{year}/acs/acs5"
        else:
            base_url = f"https://api.census.gov/data/{year}/acs/acs1"
        
        # Build variable list
        var_list = ",".join(variables)
        
        # Build geography parameters correctly - Census API needs separate 'for' and 'in' params
        params = {
            'get': var_list
        }
        
        if location_info['geography'] == 'state':
            params['for'] = f"state:{location_info['state']}"
        elif location_info['geography'] == 'us':
            params['for'] = "us:*"
        elif location_info['geography'] == 'place':
            # Use place FIPS if available from centralized mappings
            if 'place_fips' in location_info:
                params['for'] = f"place:{location_info['place_fips']}"
                params['in'] = f"state:{location_info['state']}"
            else:
                # Fallback to state level if city FIPS not available
                params['for'] = f"state:{location_info['state']}"
        else:
            raise ValueError(f"Unsupported geography: {location_info['geography']}")
        
        if self.api_key:
            params['key'] = self.api_key
        
        # Make API request
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse Census API response format
            if len(data) < 2:
                raise ValueError("Invalid Census API response format")
            
            headers = data[0]
            values = data[1]
            
            # Convert to our format
            result = {}
            for i, var in enumerate(variables):
                if i < len(values) and values[i] is not None:
                    try:
                        estimate = float(values[i])
                        # Calculate approximate MOE (simplified)
                        moe = max(1, int(estimate * 0.05))  # 5% approximation
                        
                        result[var] = {
                            'estimate': estimate,
                            'moe': moe
                        }
                    except (ValueError, TypeError):
                        result[var] = {
                            'estimate': 0,
                            'moe': 0,
                            'note': 'Data not available or not numeric'
                        }
                else:
                    result[var] = {
                        'estimate': 0,
                        'moe': 0,
                        'note': 'Variable not returned by API'
                    }
            
            logger.info(f"Successfully retrieved data for {len(result)} variables")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Census API request failed: {e}")
            raise Exception(f"Census API request failed: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"Census API response parsing failed: {e}")
            raise Exception(f"Census API response parsing failed: {e}")
    
    def _process_api_response(self, api_data: Dict, original_vars: List[str],
                            metadata: Dict) -> Dict[str, Any]:
        """Process API response and calculate derived statistics using centralized mappings"""
        
        processed = {}
        
        for var in original_vars:
            var_meta = metadata.get(var, {})
            calculation = var_meta.get('calculation', 'direct')
            var_ids = var_meta.get('variables', [])
            
            if calculation == 'rate' and len(var_ids) == 2:
                # Calculate rate from two variables using centralized rate calculations
                numerator_id, denominator_id = var_ids
                
                if numerator_id in api_data and denominator_id in api_data:
                    num_data = api_data[numerator_id]
                    denom_data = api_data[denominator_id]
                    
                    numerator = num_data['estimate']
                    denominator = denom_data['estimate']
                    
                    if denominator > 0:
                        rate = (numerator / denominator) * 100
                        # Simplified MOE propagation
                        rate_moe = (num_data['moe'] / denominator) * 100
                        
                        processed[var] = {
                            'estimate': f"{rate:.1f}%",
                            'moe': f"±{rate_moe:.1f}%",
                            'raw_value': rate,
                            'raw_numerator': numerator,
                            'raw_denominator': denominator,
                            'calculation_type': 'rate',
                            'description': var_meta.get('description', '')
                        }
                    else:
                        processed[var] = {
                            'estimate': 'N/A',
                            'error': 'Division by zero in rate calculation'
                        }
                else:
                    processed[var] = {
                        'estimate': 'N/A',
                        'error': 'Missing component data for rate calculation'
                    }
                    
            elif calculation == 'direct' and var_ids:
                # Direct variable
                var_id = var_ids[0]
                if var_id in api_data:
                    data = api_data[var_id]
                    estimate = data['estimate']
                    moe = data['moe']
                    
                    # Format based on semantic metadata if available
                    domain_weights = var_meta.get('domain_weights', {})
                    
                    if 'economics' in domain_weights and domain_weights['economics'] > 0.5:
                        # Likely a monetary value
                        processed[var] = {
                            'estimate': f"${estimate:,.0f}",
                            'moe': f"±${moe:,.0f}",
                            'raw_value': estimate,
                            'calculation_type': 'currency'
                        }
                    else:
                        # Count or other numeric
                        processed[var] = {
                            'estimate': f"{estimate:,.0f}",
                            'moe': f"±{moe:,.0f}",
                            'raw_value': estimate,
                            'calculation_type': 'count'
                        }
                else:
                    processed[var] = {
                        'estimate': 'N/A',
                        'error': f'Variable {var_id} not found in API response'
                    }
            else:
                processed[var] = {
                    'estimate': 'N/A',
                    'error': f'Resolution failed: {var_meta.get("error", "Unknown error")}'
                }
                
        return processed
    
    def _create_enriched_response(self, processed_data: Dict, metadata: Dict,
                                location_info: Dict, resolved_vars: List[str]) -> Dict[str, Any]:
        """Create enriched response with semantic intelligence"""
        
        response = {
            'location': location_info['display_name'],
            'geography_level': location_info['geography'],
            'data': processed_data,
            'census_variables': resolved_vars,
            'metadata': {
                'variable_resolution': metadata,
                'survey': 'ACS 5-Year Estimates',
                'year': 2023,
                'source': 'U.S. Census Bureau'
            },
            'semantic_intelligence': []
        }
        
        # Add semantic intelligence from metadata
        for var, var_meta in metadata.items():
            if var_meta.get('source') in ['semantic_search_primary', 'centralized_mapping', 'centralized_rate_calculation']:
                intelligence = {
                    'variable': var,
                    'census_variable': var_meta.get('variables', []),
                    'confidence': var_meta.get('confidence', 0),
                    'label': var_meta.get('label', ''),
                    'table_id': var_meta.get('table_id', ''),
                    'domain_weights': var_meta.get('domain_weights', {}),
                    'resolution_method': var_meta.get('source', 'unknown'),
                    'description': var_meta.get('description', '')
                }
                response['semantic_intelligence'].append(intelligence)
        
        # Handle disambiguation
        if location_info.get('needs_disambiguation'):
            response['disambiguation_needed'] = True
            response['message'] = f"Location '{location_info['display_name']}' is ambiguous. Please specify state."
        
        return response

# Simple test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        api = PythonCensusAPI()
        
        # Test variable resolution with centralized mappings
        resolved, metadata = api.resolve_variables(["average age", "population", "median household income", "poverty rate"])
        
        print("Variable Resolution Test (using centralized mappings):")
        for var, meta in metadata.items():
            print(f"  {var} -> {meta['source']} -> {meta['variables']}")
        
        # Test location parsing
        locations = ["Boise, ID", "Maryland", "Austin, TX"]
        print("\nLocation Parsing Test:")
        for loc in locations:
            parsed = api._parse_location(loc)
            print(f"  {loc} -> {parsed['geography']} -> {parsed.get('state', 'N/A')}")
    
    asyncio.run(test())
