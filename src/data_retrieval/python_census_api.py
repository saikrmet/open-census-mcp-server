"""
Enhanced Python Census API - Wire up Semantic Intelligence
Replaces stub implementation with real Census API + semantic search
NOW USES CENTRALIZED MAPPINGS + 67K SEMANTIC FALLBACK!
"""

import os
import sys
import logging
import json
import aiohttp
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import centralized mappings
from .census_mappings import (
    STATE_FIPS, STATE_ABBREVS, STATE_NAMES, MAJOR_CITIES,
    VARIABLE_MAPPINGS, RATE_CALCULATIONS, VALID_STATE_ABBREVS,
    get_state_fips, normalize_state, get_major_city_info,
    is_rate_calculation, get_variable_mapping
)

# Add knowledge-base to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Assuming this file is in src/data_retrieval/
kb_path = project_root / "knowledge-base"
sys.path.insert(0, str(kb_path))

# Clean deterministic approach - no complex semantic search
SEMANTIC_SEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class PythonCensusAPI:
    """
    Direct Python replacement for R tidycensus with semantic intelligence.
    Now uses centralized mappings + 67K semantic search fallback!
    """
    
    def __init__(self, config=None, api_key=None, knowledge_base=None):
        self.config = config
        self.api_key = api_key or os.getenv("CENSUS_API_KEY")
        self.base_url = "https://api.census.gov/data"
        self.knowledge_base = knowledge_base  # Inject knowledge base for semantic search
        
        logger.info("PythonCensusAPI initialized with centralized mappings + semantic fallback")

    async def resolve_variables(self, variables: List[str]) -> List[Dict[str, Any]]:
        """
        Resolve variable names using centralized mappings + 67K semantic fallback.
        
        Args:
            variables: List of variable names (human-readable or codes)
            
        Returns:
            List of resolved variable info with census codes
        """
        resolved = []
        
        for var in variables:
            var_lower = var.lower().strip()
            
            # Check if it's already a Census code
            if var.upper().startswith('B') and '_' in var:
                resolved.append({
                    'original_query': var,
                    'variable_id': var.upper(),
                    'label': f"Variable {var.upper()}",
                    'table_id': var.upper().split('_')[0],
                    'is_rate_calculation': False,
                    'source': 'direct_code'
                })
                continue
            
            # Check for rate calculations first
            if is_rate_calculation(var):
                rate_match = None
                for rate_name, rate_config in RATE_CALCULATIONS.items():
                    if rate_name.replace('_', ' ') in var_lower or rate_name in var_lower:
                        rate_match = rate_config
                        rate_match['rate_name'] = rate_name
                        break
                
                if rate_match:
                    # Rate calculation - need both numerator and denominator
                    resolved.append({
                        'original_query': var,
                        'variable_id': rate_match['numerator'],
                        'label': f"Numerator for {rate_match['description']}",
                        'table_id': rate_match['numerator'].split('_')[0],
                        'is_rate_calculation': True,
                        'rate_config': rate_match,
                        'source': 'rate_calculation'
                    })
                    resolved.append({
                        'original_query': var,
                        'variable_id': rate_match['denominator'],
                        'label': f"Denominator for {rate_match['description']}",
                        'table_id': rate_match['denominator'].split('_')[0],
                        'is_rate_calculation': True,
                        'rate_config': rate_match,
                        'source': 'rate_calculation'
                    })
                    continue
            
            # Try fast hardcoded lookup first
            mapped_var = get_variable_mapping(var)
            if mapped_var:
                # For income variables, also get margin of error
                estimate_var = mapped_var
                margin_var = mapped_var.replace('_001E', '_001M')
                
                resolved.append({
                    'original_query': var,
                    'variable_id': estimate_var,
                    'label': f"Census estimate for {var}",
                    'table_id': estimate_var.split('_')[0],
                    'is_rate_calculation': False,
                    'source': 'hardcoded_mapping'
                })
                
                # Add margin of error variable
                resolved.append({
                    'original_query': var + ' (margin of error)',
                    'variable_id': margin_var,
                    'label': f"Margin of error for {var}",
                    'table_id': margin_var.split('_')[0],
                    'is_rate_calculation': False,
                    'is_margin_of_error': True,
                    'source': 'hardcoded_mapping'
                })
                continue
            
            # 🧠 SEMANTIC SEARCH FALLBACK - Use the 67K variable corpus
            logger.info(f"Hardcoded lookup failed for '{var}', trying semantic search...")
            try:
                if self.knowledge_base:
                    semantic_results = await self.knowledge_base.search_variables(var)
                    if semantic_results and semantic_results.get('variable_code'):
                        # Found it in the 67K corpus!
                        variable_code = semantic_results['variable_code']
                        logger.info(f"✅ Semantic search found: '{var}' → {variable_code}")
                        
                        resolved.append({
                            'original_query': var,
                            'variable_id': variable_code,
                            'label': semantic_results.get('label', f"Census variable for {var}"),
                            'table_id': variable_code.split('_')[0],
                            'is_rate_calculation': False,
                            'source': 'semantic_search',
                            'semantic_confidence': semantic_results.get('confidence', 0.0)
                        })
                        
                        # Also add margin of error for estimate variables
                        if variable_code.endswith('_001E'):
                            margin_var = variable_code.replace('_001E', '_001M')
                            resolved.append({
                                'original_query': var + ' (margin of error)',
                                'variable_id': margin_var,
                                'label': f"Margin of error for {var}",
                                'table_id': margin_var.split('_')[0],
                                'is_rate_calculation': False,
                                'is_margin_of_error': True,
                                'source': 'semantic_search'
                            })
                        continue
                    else:
                        logger.warning(f"Semantic search returned no results for: {var}")
                else:
                    logger.info(f"Knowledge base not available for semantic search: {var}")
            except Exception as e:
                logger.warning(f"Semantic search failed for '{var}': {e}")
            
            # Final fallback - log unknown variable
            logger.warning(f"❌ Could not resolve variable: {var}")
            resolved.append({
                'original_query': var,
                'variable_id': None,
                'label': f"Unknown variable: {var}",
                'table_id': None,
                'is_rate_calculation': False,
                'error': 'Variable not found in hardcoded mappings or 67K semantic search',
                'source': 'failed'
            })
        
        # Log resolution summary
        resolved_count = len([r for r in resolved if r.get('variable_id')])
        total_count = len([r for r in resolved if not r.get('is_margin_of_error')])
        logger.info(f"Variable resolution: {resolved_count}/{total_count} variables resolved")
        
        return resolved

    def parse_location(self, location: str) -> Dict[str, Any]:
        """
        Parse location string into geography components using centralized mappings.
        
        Args:
            location: Human-readable location
            
        Returns:
            Dictionary with geography info
        """
        location = location.strip()
        
        # Check major cities first for exact matches
        city_info = get_major_city_info(location)
        if city_info:
            return {
                'geography': 'place',
                'state': city_info['state'],
                'city': city_info['full_name'],
                'place_fips': city_info['place_fips'],
                'display_name': city_info['full_name']
            }
        
        # Handle state-level queries using centralized function
        state = normalize_state(location)
        if state:
            return {
                'geography': 'state',
                'state': state,
                'fips_code': get_state_fips(state),
                'display_name': STATE_ABBREVS.get(state, location)
            }
        
        # Check if Baltimore needs disambiguation
        if ',' in location:
            parts = [p.strip() for p in location.split(',')]
            if len(parts) == 2:
                city, state_part = parts
                
                # Parse state using centralized function
                state = normalize_state(state_part)
                
                if state:
                    # Check if this is a known major city
                    city_key = city.lower()
                    city_info = get_major_city_info(city_key)
                    if city_info and city_info['state'] == state:
                        return {
                            'geography': 'place',
                            'state': state,
                            'city': city_info['full_name'],
                            'place_fips': city_info['place_fips'],
                            'display_name': f"{city_info['full_name']}, {STATE_ABBREVS[state]}"
                        }
                    
                    # Default place handling
                    return {
                        'geography': 'place',
                        'state': state,
                        'city': city,
                        'display_name': location
                    }
        
        # Default to treating as place name
        return {
            'geography': 'place',
            'state': None,
            'city': location,
            'display_name': location
        }

    async def get_acs_data(self, location: str, variables: List[str],
                          year: int = 2023, survey: str = "acs5",
                          context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get ACS data using semantic intelligence and real Census API.
        Now uses centralized mappings + 67K semantic fallback for all lookups.
        
        Args:
            location: Human-readable location name
            variables: List of variable names (human-readable or codes)
            year: ACS year
            survey: ACS survey type (acs1 or acs5)
            context: Optional context from knowledge base
            
        Returns:
            Dictionary with Census data and metadata
        """
        try:
            # Resolve variables using centralized mappings + semantic fallback
            resolved_vars = await self.resolve_variables(variables)
            
            # Parse location using centralized mappings
            location_data = self.parse_location(location)
            
            logger.info(f"Retrieving ACS data: {location} ({location_data['geography']}) "
                       f"for {len(resolved_vars)} resolved variables")
            
            # Check for rate calculations
            rate_data = {}
            direct_variables = []
            
            for var_info in resolved_vars:
                if var_info.get('is_rate_calculation'):
                    rate_name = var_info['rate_config']['rate_name']
                    if rate_name not in rate_data:
                        rate_data[rate_name] = {
                            'config': var_info['rate_config'],
                            'numerator_var': var_info['rate_config']['numerator'],
                            'denominator_var': var_info['rate_config']['denominator'],
                            'original_query': var_info['original_query']
                        }
                else:
                    if var_info['variable_id']:
                        direct_variables.append(var_info)
            
            # Get all unique variable codes needed
            all_variables = set()
            for var_info in direct_variables:
                if var_info['variable_id']:
                    all_variables.add(var_info['variable_id'])
            
            for rate_name, rate_info in rate_data.items():
                all_variables.add(rate_info['numerator_var'])
                all_variables.add(rate_info['denominator_var'])
            
            if not all_variables:
                return {
                    'error': 'No valid variables could be resolved',
                    'location': location,
                    'variables': variables,
                    'resolved_vars': resolved_vars,
                    'success': False
                }
            
            # Make Census API call
            api_data = await self._call_census_api(
                list(all_variables), location_data, year, survey
            )
            
            # Process results
            result = self._format_results(
                api_data, resolved_vars, rate_data, location_data, year, survey, list(all_variables)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving ACS data: {str(e)}")
            return {
                'error': str(e),
                'location': location,
                'variables': variables,
                'success': False
            }

    async def _call_census_api(self, variables: List[str], location_data: Dict,
                             year: int, survey: str) -> Dict[str, Any]:
        """
        Make actual Census API call using async HTTP.
        Uses centralized FIPS mappings for geography parameters.
        
        Args:
            variables: List of Census variable codes
            location_data: Parsed location information
            year: ACS year
            survey: ACS survey type
            
        Returns:
            Raw Census API response data
        """
        # Construct API URL
        url = f"{self.base_url}/{year}/acs/{survey}"
        
        # Build parameters
        params = {
            'get': ','.join(variables + ['NAME'])
        }
        
        if self.api_key:
            params['key'] = self.api_key
        
        # Add geography parameters using centralized FIPS mappings
        if location_data['geography'] == 'state':
            state_fips = get_state_fips(location_data['state'])
            params['for'] = f"state:{state_fips}"
        elif location_data['geography'] == 'place':
            if location_data.get('state') and location_data.get('place_fips'):
                # Use known FIPS code from centralized mappings
                state_fips = get_state_fips(location_data['state'])
                params['for'] = f"place:{location_data['place_fips']}"
                params['in'] = f"state:{state_fips}"
            elif location_data.get('state'):
                # General place search within state
                state_fips = get_state_fips(location_data['state'])
                params['for'] = "place:*"
                params['in'] = f"state:{state_fips}"
            else:
                # National place search - more complex, use simplified approach
                params['for'] = "place:*"
        else:
            # Default to national
            params['for'] = "us:*"
        
        # Make async request
        logger.info(f"Census API call: {url} with params: {params}")
        
        try:
            # Check if aiohttp is available
            try:
                import aiohttp
            except ImportError:
                import requests
                response = requests.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    raise ValueError(f"Census API error {response.status_code}: {response.text}")
                data = response.json()
            else:
                # Use aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=30) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise ValueError(f"Census API error {response.status}: {error_text}")
                        
                        data = await response.json()
                
            if not data or len(data) < 2:
                raise ValueError("No data returned from Census API")
            
            # Filter for specific location if needed
            filtered_rows = self._filter_location_results(data[1:], location_data)
            
            return {
                'headers': data[0],
                'rows': filtered_rows,
                'location_data': location_data,
                'variables': variables
            }
            
        except Exception as e:
            raise ValueError(f"Census API call failed: {str(e)}")

    def _filter_location_results(self, rows: List[List], location_data: Dict) -> List[List]:
        """
        Filter Census API results to match the requested location.
        """
        if not rows or location_data['geography'] != 'place':
            return rows
        
        city_name = location_data.get('city', '').lower()
        if not city_name:
            return rows
        
        # Filter rows where NAME contains the city name
        filtered = []
        for row in rows:
            # NAME field is at index 1, not 0 (headers: ['B01003_001E', 'NAME', 'state', 'place'])
            name_field = row[1] if len(row) > 1 else ""
            
            # Look for the city name in the full place name
            if city_name in name_field.lower():
                filtered.append(row)
                # For debugging - log what we found
                logger.info(f"Found match: '{name_field}' for requested city '{city_name}'")
        
        # Return first match if found, otherwise first row as fallback
        if filtered:
            logger.info(f"Returning filtered result for '{city_name}': {filtered[0][1]}")
            return filtered[:1]
        else:
            logger.warning(f"No matches found for '{city_name}', returning first result: {rows[0][1] if rows else 'no data'}")
            return rows[:1]

    def _format_results(self, api_data: Dict, resolved_vars: List[Dict],
                       rate_data: Dict, location_data: Dict,
                       year: int, survey: str, all_variables: List[str]) -> Dict[str, Any]:
        """
        Format Census API results with proper metadata and rate calculations.
        """
        headers = api_data['headers']
        rows = api_data['rows']
        
        if not rows:
            return {
                'error': 'No matching locations found',
                'success': False,
                'location': location_data['display_name']
            }
        
        # Take first matching row (could be enhanced with better location matching)
        data_row = rows[0]
        
        # Create data dictionary
        data_dict = dict(zip(headers, data_row))
        
        # Process direct variables
        results = {}
        metadata = {
            'source': 'U.S. Census Bureau',
            'survey': f"American Community Survey {survey.upper()}",
            'year': year,
            'location': data_dict.get('NAME', location_data['display_name']),
            'geographic_level': location_data['geography'],
            'variables_used': [],
            'methodology_notes': [],
            'semantic_intelligence': SEMANTIC_SEARCH_AVAILABLE,
            'resolution_sources': {}  # Track how variables were resolved
        }
        
        for var_info in resolved_vars:
            if not var_info.get('is_rate_calculation') and var_info['variable_id']:
                var_id = var_info['variable_id']
                value = data_dict.get(var_id)
                
                # Track resolution source
                source = var_info.get('source', 'unknown')
                metadata['resolution_sources'][var_info['original_query']] = source
                
                if value is not None:
                    try:
                        numeric_value = float(value) if value != '-' else None
                    except (ValueError, TypeError):
                        numeric_value = None
                    
                    # Handle margin of error variables
                    if var_info.get('is_margin_of_error'):
                        # Find the corresponding estimate to attach MOE to
                        base_query = var_info['original_query'].replace(' (margin of error)', '')
                        if base_query in results:
                            results[base_query]['margin_of_error'] = numeric_value
                            results[base_query]['margin_of_error_variable'] = var_id
                        continue
                    
                    result_entry = {
                        'value': numeric_value,
                        'variable_id': var_id,
                        'label': var_info['label'],
                        'table_id': var_info['table_id'],
                        'resolution_source': source
                    }
                    
                    # Add semantic confidence if available
                    if 'semantic_confidence' in var_info:
                        result_entry['semantic_confidence'] = var_info['semantic_confidence']
                    
                    results[var_info['original_query']] = result_entry
                    
                    metadata['variables_used'].append({
                        'query': var_info['original_query'],
                        'variable_id': var_id,
                        'label': var_info['label'],
                        'source': source
                    })
        
        # Process rate calculations using centralized config
        for rate_name, rate_info in rate_data.items():
            numerator = data_dict.get(rate_info['numerator_var'])
            denominator = data_dict.get(rate_info['denominator_var'])
            
            if numerator is not None and denominator is not None:
                try:
                    num_val = float(numerator) if numerator != '-' else 0
                    den_val = float(denominator) if denominator != '-' else 0
                    
                    if den_val > 0:
                        rate_value = (num_val / den_val) * 100
                        
                        results[rate_info['original_query']] = {
                            'value': round(rate_value, 1),
                            'numerator': num_val,
                            'denominator': den_val,
                            'numerator_variable': rate_info['numerator_var'],
                            'denominator_variable': rate_info['denominator_var'],
                            'description': rate_info['config']['description'],
                            'unit': 'percentage',
                            'resolution_source': 'rate_calculation'
                        }
                        
                        metadata['methodology_notes'].append(
                            f"{rate_info['config']['description']}: "
                            f"Calculated as {rate_info['numerator_var']} / {rate_info['denominator_var']} × 100"
                        )
                    else:
                        results[rate_info['original_query']] = {
                            'value': None,
                            'error': 'Division by zero (denominator is 0)',
                            'description': rate_info['config']['description']
                        }
                except (ValueError, TypeError):
                    results[rate_info['original_query']] = {
                        'value': None,
                        'error': 'Invalid numeric data',
                        'description': rate_info['config']['description']
                    }
        
        return {
            'success': True,
            'data': results,
            'metadata': metadata,
            'location': location_data['display_name'],
            'semantic_intelligence': bool(self.knowledge_base),
            'api_call_details': {
                'url': f"{self.base_url}/{year}/acs/{survey}",
                'variables_requested': all_variables,
                'geography_params': {
                    'geography_type': location_data['geography'],
                    'location_parsed': location_data
                },
                'api_key_used': 'REDACTED' if self.api_key else 'None'
            }
        }
