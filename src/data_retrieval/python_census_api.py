"""
Python Census API Client - Fixed Geographic Resolution

Replaces broken hardcoded city lookup with SQLite database resolution.
Fixes the #1 user-facing failure mode: "St. Louis, MO" returning state data instead of city data.
"""

import requests
import logging
import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import the new geographic handler
from .geographic_handler import GeographicHandler

logger = logging.getLogger(__name__)

class PythonCensusAPI:
    """Python-based Census API client with bulletproof geographic resolution"""
    
    def __init__(self):
        self.api_key = os.getenv('CENSUS_API_KEY')
        self.base_url = "https://api.census.gov/data"
        self.geo_handler = GeographicHandler()
        
        logger.info("✅ PythonCensusAPI initialized with geographic handler and semantic variable resolution")
    
    async def get_acs_data(self, location: str, variables: List[str],
                          year: int = 2023, survey: str = "acs5",
                          context: Dict = None) -> Dict[str, Any]:
        """
        Get ACS data for location and variables
        
        Args:
            location: Location string like "St. Louis, MO"
            variables: List of variable names or Census variable IDs
            year: Data year (default 2023)
            survey: Survey type (acs1, acs5, default acs5)
            context: Additional context from knowledge base
            
        Returns:
            Dictionary with data, metadata, and any errors
        """
        try:
            # Step 1: Parse location with enhanced geographic handler
            location_result = self.geo_handler.resolve_location(location)
            
            # Handle resolution errors gracefully
            if 'error' in location_result:
                return {
                    'error': location_result['error'],
                    'suggestions': location_result.get('suggestions', []),
                    'location': location,
                    'variables': variables
                }
            
            logger.info(f"✅ Location resolved: {location} → {location_result['geography']} via {location_result.get('resolution_method', 'unknown')}")
            
            # Step 2: Resolve variables using ConceptBasedCensusSearchEngine
            resolved_variables = self._resolve_variables(variables)
            
            # Step 3: Make Census API call with proper geographic parameters
            census_data = await self._call_census_api(
                variables=resolved_variables,
                location_info=location_result,
                year=year,
                survey=survey
            )
            
            # Step 4: Format response with resolution metadata
            return self._format_response(
                data=census_data,
                location=location,
                location_info=location_result,
                variables=variables,
                resolved_variables=resolved_variables,
                year=year,
                survey=survey
            )
            
        except Exception as e:
            logger.error(f"Failed to get ACS data for {location}: {str(e)}")
            
            # Enhanced error handling with contextual help
            if "resolve location" in str(e).lower():
                suggestions = self.geo_handler.get_suggestions(location)
                suggestion_text = f". Try: {', '.join(suggestions[:3])}" if suggestions else ""
                error_msg = f"Geographic resolution failed: {location}{suggestion_text}"
            else:
                error_msg = f"Census API error: {str(e)}"
            
            return {
                'error': error_msg,
                'location': location,
                'variables': variables,
                'resolution_method': 'failed'
            }
    
    def _resolve_variables(self, variables: List[str]) -> List[str]:
        """Resolve variable names to Census variable IDs using ConceptBasedCensusSearchEngine"""
        resolved = []
        
        for var in variables:
            # If already a Census variable ID (like B01003_001E), use as-is
            if self._is_census_variable_id(var):
                resolved.append(var)
                continue
            
            # Use ConceptBasedCensusSearchEngine for intelligent variable resolution
            try:
                # Import here to avoid circular imports
                from knowledge_base.kb_search import ConceptBasedCensusSearchEngine
                
                # Initialize search engine (this should be cached in production)
                search_engine = ConceptBasedCensusSearchEngine()
                
                # Search for the best matching variable
                search_results = search_engine.search(var, max_results=1)
                
                if search_results and len(search_results) > 0:
                    best_match = search_results[0]
                    resolved.append(best_match.variable_id)
                    logger.debug(f"✅ Variable resolved: '{var}' → {best_match.variable_id} (confidence: {best_match.confidence:.3f})")
                else:
                    # Fallback to basic mappings only if semantic search fails
                    fallback_var = self._fallback_variable_mapping(var)
                    if fallback_var:
                        resolved.append(fallback_var)
                        logger.debug(f"⚠️ Fallback mapping: '{var}' → {fallback_var}")
                    else:
                        # Last resort - pass through as-is
                        resolved.append(var)
                        logger.warning(f"❌ Could not resolve variable: '{var}'")
                        
            except Exception as e:
                logger.warning(f"Semantic variable resolution failed for '{var}': {e}")
                # Fallback to basic mappings
                fallback_var = self._fallback_variable_mapping(var)
                if fallback_var:
                    resolved.append(fallback_var)
                else:
                    resolved.append(var)
        
        return resolved
    
    def _fallback_variable_mapping(self, var: str) -> Optional[str]:
        """Minimal fallback mappings when semantic search fails"""
        var_lower = var.lower().strip()
        
        # Only the most basic mappings as emergency fallback
        basic_mappings = {
            'total_population': 'B01003_001E',
            'median_income': 'B19013_001E',
            'poverty_rate': 'B17001_002E',
            'housing_units': 'B25001_001E',
            'unemployment_rate': 'B23025_005E'
        }
        
        # Direct mapping
        if var_lower in basic_mappings:
            return basic_mappings[var_lower]
        
        # Simple keyword matching as last resort
        if 'population' in var_lower or 'pop' in var_lower:
            return basic_mappings['total_population']
        elif 'income' in var_lower:
            return basic_mappings['median_income']
        elif 'poverty' in var_lower:
            return basic_mappings['poverty_rate']
        elif 'housing' in var_lower or 'home' in var_lower:
            return basic_mappings['housing_units']
        elif 'unemployment' in var_lower or 'jobless' in var_lower:
            return basic_mappings['unemployment_rate']
        
        return None
    
    def _is_census_variable_id(self, var: str) -> bool:
        """Check if string looks like a Census variable ID"""
        # Census variables look like: B01003_001E, C24010_001E, etc.
        import re
        pattern = r'^[A-Z]+\d+[A-Z]*_\d+[A-Z]*$'
        return bool(re.match(pattern, var))
    
    async def _call_census_api(self, variables: List[str], location_info: Dict[str, Any],
                              year: int, survey: str) -> Dict[str, Any]:
        """Make actual Census API call with proper geographic parameters"""
        
        # Build API URL
        url = f"{self.base_url}/{year}/acs/{survey}"
        
        # Build parameters
        params = {
            'get': ','.join(['NAME'] + variables)
        }
        
        # Add API key if available
        if self.api_key:
            params['key'] = self.api_key
        
        # Set geography parameters based on location info
        geography = location_info['geography']
        
        if geography == 'us':
            params['for'] = 'us:*'
        elif geography == 'state':
            params['for'] = f"state:{location_info['state_fips']}"
        elif geography == 'place':
            params['for'] = f"place:{location_info['place_fips']}"
            params['in'] = f"state:{location_info['state_fips']}"
        else:
            raise ValueError(f"Unsupported geography type: {geography}")
        
        logger.debug(f"Census API call: {url} with params {params}")
        
        # Make request
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Census API response: {len(data)} rows")
            
            return self._parse_census_response(data, variables)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Census API request failed: {e}")
            raise RuntimeError(f"Census API request failed: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Census API response: {e}")
            raise RuntimeError(f"Invalid response from Census API: {e}")
    
    def _parse_census_response(self, data: List[List], variables: List[str]) -> Dict[str, Any]:
        """Parse Census API JSON response"""
        if not data or len(data) < 2:
            raise ValueError("No data returned from Census API")
        
        headers = data[0]
        rows = data[1:]
        
        # Find variable columns
        result = {}
        
        for row in rows:
            row_data = dict(zip(headers, row))
            
            # Extract NAME (geographic name)
            result['name'] = row_data.get('NAME', 'Unknown')
            
            # Extract each variable
            for var in variables:
                if var in row_data:
                    value = row_data[var]
                    
                    # Handle null values
                    if value is None or value == 'null' or value == '':
                        result[var] = {'estimate': None, 'error': 'No data available'}
                    else:
                        try:
                            # Convert to number if possible
                            numeric_value = float(value)
                            result[var] = {
                                'estimate': numeric_value,
                                'formatted': f"{numeric_value:,.0f}" if numeric_value >= 1000 else str(numeric_value)
                            }
                        except (ValueError, TypeError):
                            result[var] = {'estimate': value, 'formatted': str(value)}
        
        return result
    
    def _format_response(self, data: Dict, location: str, location_info: Dict,
                        variables: List[str], resolved_variables: List[str],
                        year: int, survey: str) -> Dict[str, Any]:
        """Format final response with metadata"""
        
        return {
            'location': location,
            'location_info': location_info,
            'data': data,
            'variables_requested': variables,
            'variables_resolved': resolved_variables,
            'year': year,
            'survey': survey.upper(),
            'source': 'US Census Bureau American Community Survey',
            'api_version': 'python_census_api_v2.5'
        }


# For testing and validation
async def test_geographic_resolution():
    """Test the fixed geographic resolution"""
    api = PythonCensusAPI()
    
    test_locations = [
        "St. Louis, MO",      # Should return city data, not state data
        "Kansas City, MO",    # Should return city data
        "Milwaukee, WI",      # Should return city data
        "Cleveland, OH",      # Should return city data
        "New York, NY",       # Should continue working
        "Los Angeles, CA"     # Should continue working
    ]
    
    print("Testing geographic resolution:")
    print("=" * 50)
    
    for location in test_locations:
        try:
            result = await api.get_acs_data(
                location=location,
                variables=['total_population'],
                year=2023,
                survey='acs5'
            )
            
            if 'error' in result:
                print(f"❌ {location}: {result['error']}")
            else:
                location_info = result['location_info']
                pop_data = result['data'].get('B01003_001E', {})
                population = pop_data.get('estimate', 'N/A')
                
                print(f"✅ {location}: {location_info['geography']} level, pop: {population}")
                
        except Exception as e:
            print(f"❌ {location}: Exception - {e}")
    
    print("=" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_geographic_resolution())
