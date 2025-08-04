#!/usr/bin/env python3
"""
Python Census API Client - v2.10 Geography-First Integration
CRITICAL: Geography resolution is the foundation - ALL other operations depend on it
WORKFLOW: Location → FIPS codes → Variables → API call
NO BANDAIDS - Clean integration with gazetteer database as primary requirement
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
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"✅ Loaded environment from: {env_path}")
    else:
        load_dotenv()
        logging.info("✅ Loaded environment from current directory")
except ImportError:
    logging.warning("⚠️ python-dotenv not installed, using system environment only")

# Import geographic handler - properly integrated with CompleteGeographicHandler
GEO_HANDLER_AVAILABLE = False
try:
    from .geographic_handler import CompleteGeographicHandler
    GEO_HANDLER_AVAILABLE = True
    logging.info("✅ CompleteGeographicHandler loaded successfully")
except ImportError as e:
    GEO_HANDLER_AVAILABLE = False
    logging.warning(f"Geographic handler not available: {e}")

if not GEO_HANDLER_AVAILABLE:
    raise ImportError("❌ CRITICAL: geographic_handler.CompleteGeographicHandler is required for geographic resolution")

logger = logging.getLogger(__name__)

class PythonCensusAPI:
    """Geography-first Census API client - geographic resolution is the foundation"""
    
    def __init__(self):
        self.base_url = "https://api.census.gov/data"
        self.api_key = os.getenv('CENSUS_API_KEY')
        
        # Initialize CRITICAL geographic handler - system cannot function without it
        self.geo_handler = self._init_geographic_handler_required()
        
        # Report initialization status
        if self.api_key:
            logger.info("✅ Census API key loaded")
        else:
            logger.warning("⚠️ Census API key missing - rate limits may apply")
        
        logger.info("🌍 Geography-first architecture initialized")
    
    def _init_geographic_handler_required(self) -> CompleteGeographicHandler:
        """Initialize geographic handler - REQUIRED for system functionality"""
        
        if not GEO_HANDLER_AVAILABLE:
            raise RuntimeError("❌ CRITICAL: Geographic handler not available")
        
        # Find gazetteer database - REQUIRED for proper geographic resolution
        gazetteer_paths = [
            Path(__file__).parent.parent.parent / "knowledge-base" / "geography.db",
            Path(__file__).parent.parent.parent / "knowledge-base" / "geo-db" / "geography.db",
            Path(__file__).parent.parent.parent / "geography.db",
            Path(os.getcwd()) / "knowledge-base" / "geography.db",
            Path(os.getenv('GAZETTEER_DB_PATH', '')) if os.getenv('GAZETTEER_DB_PATH') else None
        ]
        
        gazetteer_path = None
        for path in gazetteer_paths:
            if path and path.exists():
                gazetteer_path = path
                logger.info(f"✅ Found gazetteer database: {path}")
                break
        
        if not gazetteer_path:
            logger.error("❌ CRITICAL: Gazetteer database not found")
            logger.error("Geographic resolution will be severely limited without it")
            logger.error("Expected locations:")
            for path in gazetteer_paths[:4]:
                logger.error(f"  - {path}")
        
        try:
            # Initialize with the found database path
            geo_handler = CompleteGeographicHandler(gazetteer_path)
            
            # Validate that geographic handler is functional
            test_result = geo_handler.resolve_location("United States")
            if 'error' in test_result or test_result.get('geography_type') != 'us':
                raise RuntimeError("Geographic handler failed basic functionality test")
            
            logger.info("✅ Geographic handler validated and ready")
            logger.info(f"✅ Coverage: {getattr(geo_handler, 'coverage_stats', {})}")
            return geo_handler
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Geographic handler initialization failed: {e}")
            raise RuntimeError(f"Cannot initialize geographic handler: {e}")
    
    async def get_demographic_data(self, location: str, variables: List[str],
                                 year: int = 2023, survey: str = "acs5") -> Dict[str, Any]:
        """
        Geography-first demographic data retrieval
        
        WORKFLOW:
        1. Location string → Geographic parsing (CompleteGeographicHandler)
        2. Geographic context → FIPS codes (Gazetteer)
        3. Variables → Census IDs (Validation only - semantic resolution happens upstream)
        4. FIPS + Census IDs → Census API call
        """
        
        # STEP 1: Geographic resolution - THE MOST CRITICAL STEP
        logger.info(f"🌍 STEP 1 - Geographic resolution: '{location}'")
        
        try:
            location_result = self._resolve_geography_foundation(location)
            
            if 'error' in location_result:
                # Geographic resolution failed - NOTHING else can proceed
                logger.error(f"❌ Geographic resolution failed for '{location}'")
                return {
                    'error': f"Geographic resolution failed: {location_result['error']}",
                    'location_attempted': location,
                    'step_failed': 'geographic_resolution',
                    'resolution_details': location_result,
                    'help': "Geographic resolution is required for all Census API calls. Check gazetteer database and location format.",
                    'critical_requirement': "Valid FIPS codes are mandatory for Census Bureau API access"
                }
            
            logger.info(f"✅ Geographic resolution successful: {location_result['resolved_name']}")
            logger.info(f"   Method: {location_result['resolution_method']}")
            logger.info(f"   Geography: {location_result['geography_type']}")
            
            # Log FIPS codes appropriately based on geography type
            if location_result['geography_type'] == 'place':
                logger.info(f"   FIPS: {location_result.get('state_fips', 'N/A')}:{location_result.get('place_fips', 'N/A')}")
            elif location_result['geography_type'] == 'state':
                logger.info(f"   FIPS: {location_result.get('state_fips', 'N/A')}")
            elif location_result['geography_type'] == 'county':
                logger.info(f"   FIPS: {location_result.get('state_fips', 'N/A')}:{location_result.get('county_fips', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Geographic resolution system failure: {e}")
            return {
                'error': f"Geographic resolution system failure: {str(e)}",
                'location_attempted': location,
                'step_failed': 'geographic_system',
                'help': "This indicates a problem with the geographic handler or gazetteer database"
            }
        
        # STEP 2: Variable validation - Secondary to geography
        logger.info(f"🔍 STEP 2 - Variable validation: {variables}")
        
        try:
            validated_variables = self._validate_census_variables(variables)
            
            if not validated_variables['valid_variables']:
                logger.warning(f"⚠️ No valid Census variable IDs found in: {variables}")
                return {
                    'error': f"No valid Census variable IDs provided",
                    'variables_attempted': variables,
                    'step_failed': 'variable_validation',
                    'validation_details': validated_variables,
                    'help': "Variables must be proper Census IDs (e.g., 'B01003_001E'). Use semantic search to find correct IDs.",
                    'location_resolved': location_result['resolved_name']  # Geography succeeded
                }
            
            logger.info(f"✅ Variable validation: {len(validated_variables['valid_variables'])} valid variables")
            
        except Exception as e:
            logger.error(f"❌ Variable validation error: {e}")
            return {
                'error': f"Variable validation failed: {str(e)}",
                'variables_attempted': variables,
                'step_failed': 'variable_validation',
                'location_resolved': location_result['resolved_name']  # Geography succeeded
            }
        
        # STEP 3: Census API call - Both geography and variables validated
        logger.info(f"📊 STEP 3 - Census API call")
        
        try:
            census_data = await self._execute_census_api_call(
                validated_variables['valid_variables'],
                location_result,
                year,
                survey
            )
            
            if 'error' in census_data:
                logger.error(f"❌ Census API call failed")
                return {
                    'error': f"Census API call failed: {census_data['error']}",
                    'step_failed': 'census_api_call',
                    'location_resolved': location_result['resolved_name'],
                    'variables_validated': validated_variables['valid_variables'],
                    'api_details': census_data
                }
            
            logger.info(f"✅ Census API call successful")
            
        except Exception as e:
            logger.error(f"❌ Census API call system error: {e}")
            return {
                'error': f"Census API system error: {str(e)}",
                'step_failed': 'census_api_system',
                'location_resolved': location_result['resolved_name'],
                'variables_validated': validated_variables['valid_variables']
            }
        
        # STEP 4: Response formatting
        logger.info(f"📋 STEP 4 - Response formatting")
        
        return self._format_geography_first_response(
            location, variables, location_result, validated_variables,
            census_data, year, survey
        )
    
    def _resolve_geography_foundation(self, location: str) -> Dict[str, Any]:
        """
        Foundation geographic resolution using CompleteGeographicHandler
        
        Returns complete geographic context with FIPS codes or clear failure
        """
        
        try:
            # Use CompleteGeographicHandler.resolve_location() method
            geo_result = self.geo_handler.resolve_location(location)
            
            if 'error' in geo_result:
                return {
                    'error': geo_result['error'],
                    'resolution_method': geo_result.get('resolution_method', 'unknown'),
                    'suggestions': geo_result.get('suggestions', []),
                    'help': geo_result.get('help', {})
                }
            
            # Convert CompleteGeographicHandler result to expected format
            return self._convert_geo_result_to_census_format(geo_result)
            
        except Exception as e:
            return {
                'error': f"Geographic resolution failed: {str(e)}",
                'resolution_method': 'system_error'
            }
    
    def _convert_geo_result_to_census_format(self, geo_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert CompleteGeographicHandler result to Census API format"""
        
        geography_type = geo_result.get('geography_type')
        
        # Base result structure
        result = {
            'geography_type': geography_type,
            'resolved_name': geo_result.get('name', 'Unknown'),
            'resolution_method': geo_result.get('resolution_method', 'unknown')
        }
        
        # Add geography-specific FIPS codes
        if geography_type == 'us':
            result.update({
                'state_fips': None,
                'place_fips': None,
                'county_fips': None,
                'cbsa_code': None,
                'zcta_code': None
            })
        
        elif geography_type == 'state':
            result.update({
                'state_fips': geo_result.get('state_fips'),
                'state_abbrev': geo_result.get('state_abbrev'),
                'place_fips': None,
                'county_fips': None
            })
        
        elif geography_type == 'place':
            result.update({
                'state_fips': geo_result.get('state_fips'),
                'place_fips': geo_result.get('place_fips'),
                'state_abbrev': geo_result.get('state_abbrev'),
                'county_fips': None
            })
        
        elif geography_type == 'county':
            result.update({
                'state_fips': geo_result.get('state_fips'),
                'county_fips': geo_result.get('county_fips'),
                'state_abbrev': geo_result.get('state_abbrev'),
                'place_fips': None
            })
        
        elif geography_type == 'cbsa':
            result.update({
                'cbsa_code': geo_result.get('cbsa_code'),
                'cbsa_type': geo_result.get('cbsa_type'),
                'state_fips': None,
                'place_fips': None,
                'county_fips': None
            })
        
        elif geography_type == 'zcta':
            result.update({
                'zcta_code': geo_result.get('zcta_code'),
                'state_fips': None,  # ZCTAs can cross state boundaries
                'place_fips': None,
                'county_fips': None
            })
        
        # Add coordinates if available
        if 'lat' in geo_result and 'lon' in geo_result:
            result.update({
                'lat': geo_result['lat'],
                'lon': geo_result['lon']
            })
        
        return result
    
    def _validate_census_variables(self, variables: List[str]) -> Dict[str, Any]:
        """
        Validate Census variable IDs - NO semantic resolution here
        
        This function only validates format - semantic resolution should happen upstream
        """
        
        valid_variables = []
        invalid_variables = []
        
        for var in variables:
            if self._is_valid_census_variable_id(var):
                valid_variables.append(var.upper().strip())
            else:
                invalid_variables.append(var)
        
        return {
            'valid_variables': valid_variables,
            'invalid_variables': invalid_variables,
            'validation_method': 'format_check_only',
            'note': 'Semantic variable resolution should happen before calling this function'
        }
    
    def _is_valid_census_variable_id(self, var: str) -> bool:
        """Check if variable is a proper Census variable ID format"""
        import re
        # Match pattern like B01003_001E or B01003_001M
        pattern = r'^[A-Z]\d{5}_\d{3}[EM]?$'
        return bool(re.match(pattern, var.upper().strip()))
    
    async def _execute_census_api_call(self, variables: List[str], location_info: Dict,
                                     year: int, survey: str) -> Dict[str, Any]:
        """Execute Census API call with validated geography and variables"""
        
        try:
            # Build API URL
            url = f"{self.base_url}/{year}/acs/{survey}"
            
            # Build geography parameters from FIPS codes
            geography_params = self._build_geography_parameters(location_info)
            if not geography_params:
                return {
                    'error': 'Could not build geography parameters from FIPS codes',
                    'location_info': location_info
                }
            
            # Build request parameters
            params = {
                'get': ','.join(variables + ['NAME'])
            }
            params.update(geography_params)
            
            if self.api_key:
                params['key'] = self.api_key
            
            logger.info(f"🌐 Census API: {url}")
            logger.debug(f"Parameters: {params}")
            
            # Execute request
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                return {
                    'error': f"Census API HTTP {response.status_code}",
                    'response_text': response.text[:500],
                    'url': url,
                    'parameters': params
                }
            
            if not response.text.strip():
                return {
                    'error': 'Empty response from Census API',
                    'url': url,
                    'parameters': params
                }
            
            # Parse response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                return {
                    'error': f'Invalid JSON from Census API: {str(e)}',
                    'response_preview': response.text[:300]
                }
            
            if not data or not isinstance(data, list) or len(data) < 2:
                return {
                    'error': 'No data rows returned from Census API',
                    'response': data
                }
            
            return self._parse_census_response(data, variables)
            
        except requests.exceptions.Timeout:
            return {'error': 'Census API timeout (30 seconds)'}
        except requests.exceptions.ConnectionError:
            return {'error': 'Cannot connect to Census API'}
        except Exception as e:
            return {'error': f'Census API system error: {str(e)}'}
    
    def _build_geography_parameters(self, location_info: Dict) -> Optional[Dict[str, str]]:
        """Build Census API geography parameters from FIPS codes"""
        
        geography_type = location_info.get('geography_type')
        
        if geography_type == 'us':
            return {'for': 'us:*'}
        
        elif geography_type == 'state':
            state_fips = location_info.get('state_fips')
            if not state_fips:
                logger.error("Missing state FIPS code for state-level query")
                return None
            return {'for': f'state:{state_fips}'}
        
        elif geography_type == 'place':
            state_fips = location_info.get('state_fips')
            place_fips = location_info.get('place_fips')
            if not state_fips or not place_fips:
                logger.error(f"Missing FIPS codes for place query - state: {state_fips}, place: {place_fips}")
                return None
            return {
                'for': f'place:{place_fips}',
                'in': f'state:{state_fips}'
            }
        
        elif geography_type == 'county':
            state_fips = location_info.get('state_fips')
            county_fips = location_info.get('county_fips')
            if not state_fips or not county_fips:
                logger.error(f"Missing FIPS codes for county query - state: {state_fips}, county: {county_fips}")
                return None
            return {
                'for': f'county:{county_fips}',
                'in': f'state:{state_fips}'
            }
        
        elif geography_type == 'zcta':
            zcta_code = location_info.get('zcta_code')
            if not zcta_code:
                logger.error("Missing ZCTA code for ZIP code query")
                return None
            return {'for': f'zip code tabulation area:{zcta_code}'}
        
        else:
            logger.error(f"Unsupported geography type: {geography_type}")
            return None
    
    def _parse_census_response(self, data: List[List], variables: List[str]) -> Dict[str, Any]:
        """Parse Census API response"""
        
        headers = data[0]
        rows = data[1:]
        
        result = {'data': {}}
        
        for row in rows:
            if len(row) != len(headers):
                continue
            
            row_data = dict(zip(headers, row))
            result['name'] = row_data.get('NAME', 'Unknown')
            
            for var in variables:
                if var in row_data:
                    value = row_data[var]
                    if value in [None, 'null', '', '-']:
                        result['data'][var] = {'estimate': None, 'error': 'No data'}
                    else:
                        try:
                            numeric_value = float(value)
                            result['data'][var] = {
                                'estimate': numeric_value,
                                'formatted': f"{numeric_value:,.0f}" if numeric_value >= 1000 else f"{numeric_value:g}"
                            }
                        except (ValueError, TypeError):
                            result['data'][var] = {'estimate': value, 'formatted': str(value)}
        
        return result
    
    def _format_geography_first_response(self, original_location: str, original_variables: List[str],
                                       location_result: Dict, variable_result: Dict,
                                       census_data: Dict, year: int, survey: str) -> Dict[str, Any]:
        """Format response emphasizing geography-first workflow"""
        
        return {
            'location': original_location,
            'resolved_location': {
                'name': location_result['resolved_name'],
                'geography_type': location_result['geography_type'],
                'resolution_method': location_result['resolution_method'],
                'fips_codes': {
                    'state': location_result.get('state_fips'),
                    'place': location_result.get('place_fips'),
                    'county': location_result.get('county_fips'),
                    'cbsa': location_result.get('cbsa_code'),
                    'zcta': location_result.get('zcta_code')
                }
            },
            'data': census_data.get('data', {}),
            'variables': {
                'requested': original_variables,
                'validated': variable_result['valid_variables'],
                'invalid': variable_result['invalid_variables'],
                'success_count': len([v for v in variable_result['valid_variables']
                                    if v in census_data.get('data', {})
                                    and census_data['data'][v].get('estimate') is not None])
            },
            'survey_info': {
                'year': year,
                'survey': survey.upper(),
                'source': 'US Census Bureau American Community Survey'
            },
            'workflow_info': {
                'version': 'python_census_api_v2.10_geography_first',
                'architecture': 'geography_first',
                'steps_completed': ['geographic_resolution', 'variable_validation', 'census_api_call', 'response_formatting'],
                'geographic_foundation': True,
                'no_bandaid_fixes': True
            }
        }

# Test function for geography-first workflow
async def test_geography_first():
    """Test geography-first workflow - geography is the foundation"""
    
    logger.info("🧪 Testing geography-first workflow")
    
    try:
        api = PythonCensusAPI()
        logger.info("✅ Geography-first API initialized")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Cannot initialize geography-first API: {e}")
        return
    
    # Test cases prioritizing geographic resolution
    test_cases = [
        # Basic geographic tests
        ("United States", ["B01003_001E"]),     # National level (no FIPS needed)
        ("New York", ["B01003_001E"]),          # State resolution (requires gazetteer)
        ("Austin, TX", ["B01003_001E"]),        # Place resolution (requires gazetteer)
        
        # Geography failure scenarios
        ("Mars Colony", ["B01003_001E"]),       # Should fail at geography step
        ("United States", ["invalid_var"]),     # Should fail at variable step
    ]
    
    for location, variables in test_cases:
        logger.info(f"\n🌍 Testing geography-first: '{location}' with {variables}")
        
        try:
            result = await api.get_demographic_data(location, variables)
            
            if 'error' in result:
                step_failed = result.get('step_failed', 'unknown')
                logger.warning(f"⚠️ Failed at step: {step_failed}")
                logger.warning(f"   Error: {result['error']}")
                
                if 'location_resolved' in result:
                    logger.info(f"   ✅ Geography resolved: {result['location_resolved']}")
                if 'variables_validated' in result:
                    logger.info(f"   ✅ Variables validated: {result['variables_validated']}")
            else:
                logger.info(f"✅ Complete success: {result['resolved_location']['name']}")
                logger.info(f"   Geography: {result['resolved_location']['resolution_method']}")
                logger.info(f"   Variables: {len(result['variables']['validated'])} validated")
                
        except Exception as e:
            logger.error(f"❌ System error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_geography_first())
