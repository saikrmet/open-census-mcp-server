#!/usr/bin/env python3
"""
Fixed Python Census API - Hybrid approach with Batch Query Support

Accepts either:
1. Pre-constructed Census API URLs (for Claude's direct construction)
2. Parameters (for internal URL construction)

NEW: Supports batch queries (county:*, place:*) returning multiple rows
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class PythonCensusAPI:
    """
    Hybrid Python Census API client with batch query support.
    
    Can handle both pre-constructed URLs and parameter-based construction.
    Supports batch queries like county:* returning multiple geographies.
    """
    
    def __init__(self):
        """Initialize Census API client."""
        self.base_url = "https://api.census.gov/data/2023/acs/acs5"
        self.api_key = os.getenv("CENSUS_API_KEY")
        
        logger.info("✅ PythonCensusAPI initialized (hybrid mode)")
    
    def get_acs_data(self, url: str = None, variables: List[str] = None, geography_type: str = None, **geo_params) -> Dict[str, Any]:
        """
        Get ACS data using either pre-constructed URL or parameters.
        
        Args:
            url: Pre-constructed Census API URL (takes priority)
            variables: List of variable IDs (for parameter mode)
            geography_type: Geography type (for parameter mode)
            **geo_params: FIPS codes (for parameter mode)
        
        Returns:
            Dict with data, location_name, source, error
        """
        
        # Mode 1: Pre-constructed URL
        if url:
            return self._execute_census_url(url)
        
        # Mode 2: Parameter construction
        elif variables and geography_type:
            return self._execute_with_parameters(variables, geography_type, geo_params)
        
        else:
            return {"error": "Must provide either 'url' or 'variables + geography_type'"}
    
    def _execute_census_url(self, url: str) -> Dict[str, Any]:
        """Execute pre-constructed Census API URL."""
        
        logger.info(f"🌐 Executing pre-constructed URL: {url[:100]}...")
        
        # Validate it's a Census API URL
        if not url.startswith("https://api.census.gov/"):
            return {"error": "Invalid Census API URL - must start with https://api.census.gov/"}
        
        # Add API key if available and not already present
        if self.api_key and "key=" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}key={self.api_key}"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                error_msg = f"Census API HTTP {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON response: {e}"}
            
            if isinstance(data, list) and len(data) >= 2:
                # Normal Census API response: [headers, ...data_rows]
                return self._format_census_response(data, url)
            else:
                return {"error": f"Unexpected Census API response format: {data}"}
                
        except requests.RequestException as e:
            return {"error": f"Request failed: {e}"}
    
    def _execute_with_parameters(self, variables: List[str], geography_type: str, geo_params: Dict) -> Dict[str, Any]:
        """Execute using parameter-based URL construction."""
        
        logger.info(f"📊 Constructing URL from parameters: {variables} for {geography_type}")
        
        # Build Census API parameters
        params = {
            "get": ",".join(variables)
        }
        
        if self.api_key:
            params["key"] = self.api_key
        
        # Build geography clauses
        if geography_type == "us":
            params["for"] = "us:1"
        
        elif geography_type == "state":
            state_fips = geo_params.get("state_fips")
            if not state_fips:
                return {"error": "State geography requires state_fips"}
            params["for"] = f"state:{state_fips}"
        
        elif geography_type == "county":
            state_fips = geo_params.get("state_fips")
            county_fips = geo_params.get("county_fips")
            if not state_fips or not county_fips:
                return {"error": "County geography requires state_fips and county_fips"}
            params["for"] = f"county:{county_fips}"
            params["in"] = f"state:{state_fips}"
        
        elif geography_type == "place":
            state_fips = geo_params.get("state_fips")
            place_fips = geo_params.get("place_fips")
            if not state_fips or not place_fips:
                return {"error": "Place geography requires state_fips and place_fips"}
            params["for"] = f"place:{place_fips}"
            params["in"] = f"state:{state_fips}"
        
        elif geography_type == "zcta":
            zcta_code = geo_params.get("zcta_code")
            if not zcta_code:
                return {"error": "ZCTA geography requires zcta_code"}
            params["for"] = f"zip code tabulation area:{zcta_code}"
        
        elif geography_type == "cbsa":
            cbsa_code = geo_params.get("cbsa_code")
            if not cbsa_code:
                return {"error": "CBSA geography requires cbsa_code"}
            params["for"] = f"metropolitan statistical area/micropolitan statistical area:{cbsa_code}"
        
        else:
            return {"error": f"Unsupported geography type: {geography_type}"}
        
        # Make the request
        try:
            logger.info(f"🌐 Census API request: {self.base_url} with params {params}")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code != 200:
                return {"error": f"Census API HTTP {response.status_code}: {response.text}"}
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON response: {e}"}
            
            if isinstance(data, list) and len(data) >= 2:
                return self._format_census_response(data, response.url)
            else:
                return {"error": f"Unexpected Census API response format: {data}"}
                
        except requests.RequestException as e:
            return {"error": f"Request failed: {e}"}
    
    def _format_census_response(self, data: List, url: str) -> Dict[str, Any]:
        """
        Format Census API response into structured data.
        
        NEW: Handles both single and batch queries (multiple rows).
        """
        
        headers = data[0]
        rows = data[1:]
        
        if not rows:
            return {"error": "No data returned from Census API"}
        
        # Detect batch query (multiple rows = batch, single row = specific geography)
        is_batch_query = len(rows) > 1
        
        if is_batch_query:
            return self._format_batch_response(headers, rows, url)
        else:
            return self._format_single_response(headers, rows[0], url)
    
    def _format_single_response(self, headers: List, data_row: List, url: str) -> Dict[str, Any]:
        """Format single geography response (original behavior)."""
        
        # Map headers to values
        row_data = dict(zip(headers, data_row))
        
        # Format the response
        formatted_data = {}
        notes = []
        
        for header, value in row_data.items():
            # Skip geographic identifiers (state, county, place codes)
            if header in ['state', 'county', 'place', 'zip code tabulation area', 'metropolitan statistical area/micropolitan statistical area']:
                continue
                
            # Handle null values
            if value is None or value == "null" or value == -666666666:
                formatted_data[header] = {
                    "estimate": "Data not available",
                    "moe": "N/A",
                    "label": self._get_variable_label(header)
                }
                notes.append(f"{header}: Data suppressed or not available")
            else:
                # Convert to numeric if possible
                try:
                    numeric_value = float(value)
                    formatted_data[header] = {
                        "estimate": int(numeric_value) if numeric_value.is_integer() else numeric_value,
                        "moe": "See ACS documentation",
                        "label": self._get_variable_label(header)
                    }
                except (ValueError, TypeError):
                    formatted_data[header] = {
                        "estimate": value,
                        "moe": "N/A",
                        "label": self._get_variable_label(header)
                    }
        
        # Build location name
        location_name = self._build_location_name(row_data)
        
        return {
            "data": formatted_data,
            "location_name": location_name,
            "notes": notes,
            "source": "U.S. Census Bureau, American Community Survey 2023 5-Year Estimates",
            "api_url": url,
            "raw_census_data": row_data
        }
    
    def _format_batch_response(self, headers: List, data_rows: List[List], url: str) -> Dict[str, Any]:
        """
        Format batch query response (NEW: handles multiple geographies).
        
        Returns all geographies with their data, sorted by first variable value.
        """
        
        batch_results = []
        variable_columns = []
        
        # Identify variable columns (non-geographic identifiers)
        for header in headers:
            if header not in ['state', 'county', 'place', 'zip code tabulation area',
                             'metropolitan statistical area/micropolitan statistical area', 'NAME']:
                variable_columns.append(header)
        
        for data_row in data_rows:
            row_data = dict(zip(headers, data_row))
            
            # Format variables for this geography
            formatted_data = {}
            for header in variable_columns:
                value = row_data.get(header)
                
                if value is None or value == "null" or value == -666666666:
                    formatted_data[header] = {
                        "estimate": "Data not available",
                        "moe": "N/A",
                        "label": self._get_variable_label(header)
                    }
                else:
                    try:
                        numeric_value = float(value)
                        formatted_data[header] = {
                            "estimate": int(numeric_value) if numeric_value.is_integer() else numeric_value,
                            "moe": "See ACS documentation",
                            "label": self._get_variable_label(header)
                        }
                    except (ValueError, TypeError):
                        formatted_data[header] = {
                            "estimate": value,
                            "moe": "N/A",
                            "label": self._get_variable_label(header)
                        }
            
            # Build result for this geography
            geography_result = {
                "data": formatted_data,
                "location_name": self._build_location_name(row_data),
                "geography_codes": {k: v for k, v in row_data.items() if k in ['state', 'county', 'place']},
                "sort_value": self._get_sort_value(formatted_data, variable_columns[0])
            }
            
            batch_results.append(geography_result)
        
        # Sort by first variable value (descending)
        batch_results.sort(key=lambda x: x["sort_value"], reverse=True)
        
        # Remove sort_value from final results
        for result in batch_results:
            del result["sort_value"]
        
        return {
            "data": batch_results,
            "location_name": f"Batch query: {len(batch_results)} geographies",
            "source": "U.S. Census Bureau, American Community Survey 2023 5-Year Estimates",
            "api_url": url,
            "batch_query": True,
            "total_geographies": len(batch_results),
            "variables": variable_columns
        }
    
    def _get_sort_value(self, formatted_data: Dict, variable_id: str) -> float:
        """Get numeric sort value for ranking geographies."""
        try:
            estimate = formatted_data.get(variable_id, {}).get("estimate", 0)
            return float(estimate) if estimate != "Data not available" else 0
        except (ValueError, TypeError):
            return 0
    
    def _get_variable_label(self, variable_id: str) -> str:
        """Get human-readable label for Census variable."""
        
        # Common variable labels
        variable_labels = {
            "B01003_001E": "Total Population",
            "B19013_001E": "Median Household Income",
            "B25064_001E": "Median Gross Rent",
            "B17001_002E": "Income Below Poverty Level",
            "B23025_005E": "Unemployed Population",
            "B01002_001E": "Median Age",
            "B15003_022E": "Bachelor's Degree or Higher",
            "B25001_001E": "Total Housing Units"
        }
        
        return variable_labels.get(variable_id, variable_id)
    
    def _build_location_name(self, row_data: Dict) -> str:
        """Build human-readable location name from Census response."""
        
        # Try to use the NAME field if available
        if "NAME" in row_data:
            return row_data["NAME"]
        
        # Fallback construction
        parts = []
        
        if "place" in row_data:
            parts.append(f"Place {row_data['place']}")
        elif "county" in row_data:
            parts.append(f"County {row_data['county']}")
        
        if "state" in row_data:
            parts.append(f"State {row_data['state']}")
        
        return ", ".join(parts) if parts else "Unknown Location"

    # Utility methods for external use
    def construct_census_url(self, variables: List[str], geography_type: str, **geo_params) -> str:
        """Utility: Construct Census API URL from parameters (for Claude to use)."""
        
        params = {"get": ",".join(variables)}
        
        if self.api_key:
            params["key"] = self.api_key
        
        # Geography clauses (simplified for common cases)
        if geography_type == "place":
            state_fips = geo_params.get("state_fips")
            place_fips = geo_params.get("place_fips")
            params["for"] = f"place:{place_fips}"
            params["in"] = f"state:{state_fips}"
        elif geography_type == "state":
            state_fips = geo_params.get("state_fips")
            params["for"] = f"state:{state_fips}"
        elif geography_type == "us":
            params["for"] = "us:1"
        
        # Build URL
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.base_url}?{param_string}"
