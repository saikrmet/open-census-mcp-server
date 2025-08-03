#!/usr/bin/env python3
"""
Python Census API Client with Environment Loading Fix
FIXES:
1. Loads .env file properly
2. Graceful fallback when geography.db missing
3. Can work without geographic database for basic functionality
4. ✅ FIXED: Removed print statements that contaminated MCP protocol stdout
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in parent directory (common pattern)
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"✅ Loaded environment from: {env_path}")  # ← FIXED: print → logging.info
    else:
        # Try current directory
        load_dotenv()
        logging.info("✅ Loaded environment from current directory")  # ← FIXED: print → logging.info
except ImportError:
    logging.warning("⚠️ python-dotenv not installed, using system environment only")  # ← FIXED: print → logging.warning

# Import the production geographic handler
try:
    from .geographic_handler import GeographicHandler
except ImportError:
    from geographic_handler import GeographicHandler

logger = logging.getLogger(__name__)

class PythonCensusAPI:
    """Production-ready Census API client with fallback capabilities"""
    
    def __init__(self):
        self.base_url = "https://api.census.gov/data"
        self.api_key = os.getenv('CENSUS_API_KEY')  # Should load from .env now
        
        # Initialize geographic handler with graceful fallback
        self.geo_handler = None
        self._init_geographic_handler()
        
        # Core variable mappings for fast path
        self.core_mappings = {
            'population': 'B01003_001E',
            'total_population': 'B01003_001E',
            'median_household_income': 'B19013_001E',
            'median_income': 'B19013_001E',
            'poverty_rate': 'B17001_002E',  # Numerator for rate calculation
            'poverty_total': 'B17001_001E',  # Denominator for rate calculation
            'median_age': 'B01002_001E',
            'unemployment_rate': 'B23025_005E',  # Numerator
            'labor_force': 'B23025_002E',  # Denominator
        }
        
        # Report status
        if self.api_key:
            logger.info("✅ Census API key loaded from environment")
        else:
            logger.warning("⚠️ Census API key not found - some features may be limited")
    
    def _init_geographic_handler(self):
        """Initialize geographic handler with fallback for missing database"""
        try:
            self.geo_handler = GeographicHandler()
            logger.info("✅ Geographic handler initialized with production database")
        except Exception as e:
            logger.warning(f"⚠️ Geographic handler unavailable: {e}")
            logger.info("💡 System will use basic state-level geography resolution")
            self.geo_handler = None
    
    async def get_demographic_data(self, location: str, variables: List[str],
                                 year: int = 2023, survey: str = "acs5") -> Dict[str, Any]:
        """
        Get demographic data with fallback geographic resolution
        """
        try:
            # Step 1: Resolve geographic location
            logger.info(f"🌍 Resolving location: {location}")
            location_info = self._resolve_location(location)
            if not location_info or 'error' in location_info:
                return {
                    'error': f"Could not resolve location: {location}",
                    'suggestion': location_info.get('suggestions', []) if location_info else [],
                    'location_attempted': location
                }
            
            # Step 2: Resolve variables to Census IDs
            logger.info(f"🔍 Resolving variables: {variables}")
            resolved_variables = self._resolve_variables(variables)
            if not resolved_variables:
                return {
                    'error': f"Could not resolve any variables: {variables}",
                    'variables_attempted': variables
                }
            
            # Step 3: Make Census API call
            logger.info(f"📊 Fetching data from Census API")
            census_data = await self._fetch_census_data(
                resolved_variables, location_info, year, survey
            )
            
            if 'error' in census_data:
                return census_data
            
            # Step 4: Process and format response
            return self._format_response(
                census_data, location, location_info,
                variables, resolved_variables, year, survey
            )
            
        except Exception as e:
            logger.error(f"❌ Demographic data error: {e}")
            return {
                'error': f"System error: {str(e)}",
                'location_attempted': location,
                'variables_attempted': variables
            }
    
    def _resolve_location(self, location: str) -> Optional[Dict[str, Any]]:
        """Resolve location with fallback to basic parsing if database unavailable"""
        
        # Try production geographic handler first
        if self.geo_handler:
            try:
                result = self.geo_handler.resolve_location(location)
                if result and 'error' not in result:
                    logger.info(f"✅ Location resolved via database: {location}")
                    return result
                else:
                    logger.warning(f"⚠️ Database resolution failed, trying fallback")
            except Exception as e:
                logger.error(f"Geographic handler error: {e}")
        
        # Fallback to basic parsing
        return self._basic_location_parsing(location)
    
    def _basic_location_parsing(self, location: str) -> Optional[Dict[str, Any]]:
        """Basic location parsing fallback when database unavailable"""
        location = location.strip().lower()
        
        # Handle US/national
        if location in ['united states', 'usa', 'us', 'america', 'national']:
            return {
                'geography': 'us',
                'state_fips': None,
                'place_fips': None,
                'state_abbrev': None,
                'name': 'United States',
                'resolution_method': 'basic_national'
            }
        
        # For now, just return an error suggesting they need the geography database
        return {
            'error': f"Geographic database required for location resolution: {location}",
            'suggestion': "Build geography database or use 'United States' for national data",
            'database_location': "Expected at: ../knowledge-base/geo-db/geography.db"
        }
    
    def _resolve_variables(self, variables: List[str]) -> List[str]:
        """Resolve variable names to Census variable IDs"""
        resolved = []
        
        for var in variables:
            # Check if already a Census ID (B19013_001E format)
            if self._is_census_variable_id(var):
                resolved.append(var)
                continue
            
            # Check core mappings first (fastest path)
            var_lower = var.lower().replace(' ', '_')
            if var_lower in self.core_mappings:
                resolved.append(self.core_mappings[var_lower])
                continue
            
            # TODO: Integrate with semantic search (kb_search.py)
            # For now, log unmapped variables
            logger.warning(f"⚠️ Could not resolve variable: {var}")
        
        return resolved
    
    def _is_census_variable_id(self, var: str) -> bool:
        """Check if string is already a Census variable ID"""
        import re
        pattern = r'^[A-Z]\d{5}_\d{3}[E|M]?$'
        return bool(re.match(pattern, var.upper()))
    
    async def _fetch_census_data(self, variables: List[str], location_info: Dict,
                                year: int, survey: str) -> Dict[str, Any]:
        """
        Fetch data from Census API with robust error handling
        """
        try:
            # Build API URL
            url = f"{self.base_url}/{year}/acs/{survey}"
            
            # Build geography parameters (now returns dict, not string)
            geography_params = self._build_geography_string(location_info)
            if not geography_params:
                return {'error': f"Could not build geography for {location_info}"}
            
            # Build request parameters
            params = {
                'get': ','.join(variables + ['NAME'])
            }
            
            # Add geography parameters (separate 'for' and 'in' keys)
            params.update(geography_params)
            
            # Add API key if available
            if self.api_key:
                params['key'] = self.api_key
            
            logger.info(f"🌐 Census API call: {url} with {len(variables)} variables")
            logger.debug(f"Parameters: {params}")
            
            # Make request with timeout and error handling
            response = requests.get(url, params=params, timeout=30)
            
            # Check HTTP status
            if response.status_code != 200:
                error_msg = f"Census API HTTP {response.status_code}"
                if response.text:
                    error_msg += f": {response.text[:200]}"
                return {'error': error_msg}
            
            # Check for empty response (FIXES JSON parsing error)
            if not response.text or not response.text.strip():
                return {
                    'error': 'Empty response from Census API',
                    'suggestion': 'The requested data may not be available for this geography/year combination'
                }
            
            # Parse JSON with error handling
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error. Response text: {response.text[:500]}")
                return {
                    'error': f'Invalid JSON response from Census API',
                    'details': str(e),
                    'suggestion': 'The Census API may be experiencing issues'
                }
            
            # Validate data structure
            if not data or not isinstance(data, list) or len(data) < 2:
                return {
                    'error': 'No data returned from Census API',
                    'suggestion': 'Try a different geography level or check if data is available for this year'
                }
            
            logger.info(f"✅ Census API success: {len(data)-1} rows returned")
            return self._parse_census_response(data, variables)
            
        except requests.exceptions.Timeout:
            return {'error': 'Census API request timed out (30 seconds)'}
        except requests.exceptions.ConnectionError:
            return {'error': 'Could not connect to Census API'}
        except requests.exceptions.RequestException as e:
            return {'error': f'Census API request failed: {str(e)}'}
        except Exception as e:
            logger.error(f"Unexpected error in Census API call: {e}")
            return {'error': f'Unexpected error: {str(e)}'}
    
    def _build_geography_string(self, location_info: Dict) -> Optional[Dict[str, str]]:
        """Build Census API geography parameters as separate dict entries"""
        geography = location_info.get('geography')
        
        if geography == 'us':
            return {'for': 'us:*'}
        elif geography == 'state':
            state_fips = location_info.get('state_fips')
            if state_fips:
                return {'for': f'state:{state_fips}'}
        elif geography == 'place':
            state_fips = location_info.get('state_fips')
            place_fips = location_info.get('place_fips')
            if state_fips and place_fips:
                return {
                    'for': f'place:{place_fips}',
                    'in': f'state:{state_fips}'
                }
        elif geography == 'county':
            state_fips = location_info.get('state_fips')
            county_fips = location_info.get('county_fips')
            if state_fips and county_fips:
                return {
                    'for': f'county:{county_fips}',
                    'in': f'state:{state_fips}'
                }
        else:
            logger.error(f"Unknown geography type: {geography}")
        
        return None
    
    def _parse_census_response(self, data: List[List], variables: List[str]) -> Dict[str, Any]:
        """Parse Census API JSON response into structured data"""
        if not data or len(data) < 2:
            return {'error': 'Invalid data structure from Census API'}
        
        headers = data[0]
        rows = data[1:]
        
        result = {'data': {}}
        
        # Process each row (usually just one for specific geographies)
        for row in rows:
            if len(row) != len(headers):
                logger.warning(f"Row length mismatch: {len(row)} vs {len(headers)}")
                continue
            
            row_data = dict(zip(headers, row))
            
            # Extract geographic name
            result['name'] = row_data.get('NAME', 'Unknown Location')
            
            # Extract each variable
            for var in variables:
                if var in row_data:
                    value = row_data[var]
                    
                    # Handle null/missing values
                    if value is None or value == 'null' or value == '' or value == '-':
                        result['data'][var] = {
                            'estimate': None,
                            'error': 'Data not available'
                        }
                    else:
                        try:
                            # Convert to number
                            numeric_value = float(value)
                            
                            # Format based on likely data type
                            if var.endswith('_001E') and 'income' in var.lower():
                                # Income variables - format as currency
                                formatted = f"${numeric_value:,.0f}"
                            elif numeric_value >= 1000:
                                # Large numbers - add commas
                                formatted = f"{numeric_value:,.0f}"
                            else:
                                # Small numbers or rates
                                formatted = f"{numeric_value:g}"
                            
                            result['data'][var] = {
                                'estimate': numeric_value,
                                'formatted': formatted
                            }
                            
                        except (ValueError, TypeError):
                            # Non-numeric value
                            result['data'][var] = {
                                'estimate': value,
                                'formatted': str(value)
                            }
        
        return result
    
    def _format_response(self, census_data: Dict, location: str, location_info: Dict,
                        variables: List[str], resolved_variables: List[str],
                        year: int, survey: str) -> Dict[str, Any]:
        """Format final response with metadata"""
        
        return {
            'location': location,
            'resolved_location': {
                'name': census_data.get('name', location),
                'geography_type': location_info.get('geography', 'unknown'),
                'resolution_method': location_info.get('resolution_method', 'unknown'),
                'fips_codes': {
                    'state': location_info.get('state_fips'),
                    'place': location_info.get('place_fips'),
                    'county': location_info.get('county_fips')
                }
            },
            'data': census_data.get('data', {}),
            'variables': {
                'requested': variables,
                'resolved': resolved_variables,
                'success_count': len([v for v in resolved_variables if v in census_data.get('data', {})])
            },
            'survey_info': {
                'year': year,
                'survey': survey.upper(),
                'source': 'US Census Bureau American Community Survey'
            },
            'api_info': {
                'version': 'python_census_api_v2.9_mcp_clean',
                'geographic_resolution': 'Production DB' if self.geo_handler else 'Basic fallback',
                'environment_loaded': bool(self.api_key)
            }
        }

# Test function for validation (only runs when called directly, not during MCP import)
async def test_with_fallback():
    """Test the fixed API with fallback capabilities"""
    api = PythonCensusAPI()
    
    test_cases = [
        ("Minnesota", ["population"]),           # Should work with basic fallback
        ("United States", ["population"]),      # Should work
        ("Texas", ["median_household_income"]), # Should work with basic fallback
    ]
    
    for location, variables in test_cases:
        logger.info(f"\n🧪 Testing: {location} with {variables}")  # ← FIXED: print → logger.info
        try:
            result = await api.get_demographic_data(location, variables)
            if 'error' in result:
                logger.error(f"❌ Error: {result['error']}")  # ← FIXED: print → logger.error
                if 'suggestion' in result:
                    logger.info(f"💡 Suggestion: {result['suggestion']}")  # ← FIXED: print → logger.info
            else:
                logger.info(f"✅ Success: {result['resolved_location']['name']}")  # ← FIXED: print → logger.info
                logger.info(f"   Method: {result['resolved_location']['resolution_method']}")  # ← FIXED: print → logger.info
                for var, data in result['data'].items():
                    logger.info(f"   {var}: {data.get('formatted', data.get('estimate'))}")  # ← FIXED: print → logger.info
        except Exception as e:
            logger.error(f"❌ Exception: {e}")  # ← FIXED: print → logger.error

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_with_fallback())
