#!/usr/bin/env python3
"""
Geographic Handler - FIXED - Simple and Effective Search

Key fix: Use LOWER(name) instead of non-existent name_lower column
Simplified search logic: find everything that matches, let Claude decide
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)

class GeographicHandler:
    """
    Database-driven geographic resolution with FIXED search patterns.
    """
    
    def __init__(self, geography_db_path: str):
        """Initialize with path to geography database."""
        self.db_path = Path(geography_db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Geography database not found: {self.db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        logger.info(f"✅ GeographicHandler initialized with {self.db_path}")
    
    def search_locations(self, location: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Simple, effective search: find everything that matches, let Claude decide.
        """
        if not location or not location.strip():
            return []
        
        location = location.strip()
        logger.info(f"🔍 Searching for: '{location}'")
        
        results = []
        
        # Extract state context (e.g. "St. Louis, MO" -> state = "MO")
        state_filter = None
        if ',' in location:
            parts = location.split(',')
            if len(parts) == 2:
                main_location = parts[0].strip()
                potential_state = parts[1].strip().upper()
                if len(potential_state) == 2:
                    state_filter = potential_state
                    location = main_location
        
        # Search places
        results.extend(self._search_places(location, state_filter, max_results // 2))
        
        # Search counties
        results.extend(self._search_counties(location, state_filter, max_results // 2))
        
        # Search CBSAs
        results.extend(self._search_cbsas(location, max_results // 4))
        
        # Search ZCTAs if it looks like a ZIP
        if self._is_zip_code(location):
            results.extend(self._search_zctas(location, max_results // 4))
        
        # Remove duplicates and sort by confidence
        unique_results = self._deduplicate_results(results)
        sorted_results = sorted(unique_results, key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"✅ Found {len(sorted_results)} matches for '{location}'")
        return sorted_results[:max_results]
    
    def _search_places(self, name: str, state_filter: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Search places with simple LIKE matching."""
        cursor = self.conn.cursor()
        
        if state_filter:
            query = """
                SELECT 'place' as geography_type, name, state_abbrev, state_fips, place_fips
                FROM places 
                WHERE LOWER(name) LIKE LOWER(?) AND state_abbrev = ?
                ORDER BY name
                LIMIT ?
            """
            params = [f"%{name}%", state_filter, limit]
        else:
            query = """
                SELECT 'place' as geography_type, name, state_abbrev, state_fips, place_fips
                FROM places 
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY name
                LIMIT ?
            """
            params = [f"%{name}%", limit]
        
        cursor.execute(query, params)
        results = []
        
        for row in cursor.fetchall():
            confidence = self._calculate_confidence(name, row['name'])
            # Boost confidence for state matches
            if state_filter and row['state_abbrev'] == state_filter:
                confidence += 0.2
                
            results.append({
                'geography_type': 'place',
                'name': row['name'],
                'state_abbrev': row['state_abbrev'],
                'state_fips': row['state_fips'],
                'place_fips': row['place_fips'],
                'confidence': min(1.0, confidence)
            })
        
        return results
    
    def _search_counties(self, name: str, state_filter: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Search counties with simple LIKE matching."""
        cursor = self.conn.cursor()
        
        if state_filter:
            query = """
                SELECT 'county' as geography_type, name, state_abbrev, state_fips, county_fips
                FROM counties 
                WHERE LOWER(name) LIKE LOWER(?) AND state_abbrev = ?
                ORDER BY name
                LIMIT ?
            """
            params = [f"%{name}%", state_filter, limit]
        else:
            query = """
                SELECT 'county' as geography_type, name, state_abbrev, state_fips, county_fips
                FROM counties 
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY name
                LIMIT ?
            """
            params = [f"%{name}%", limit]
        
        cursor.execute(query, params)
        results = []
        
        for row in cursor.fetchall():
            confidence = self._calculate_confidence(name, row['name'])
            # Boost confidence for state matches
            if state_filter and row['state_abbrev'] == state_filter:
                confidence += 0.2
                
            results.append({
                'geography_type': 'county',
                'name': row['name'],
                'state_abbrev': row['state_abbrev'],
                'state_fips': row['state_fips'],
                'county_fips': row['county_fips'],
                'confidence': min(1.0, confidence)
            })
        
        return results
    
    def _search_cbsas(self, name: str, limit: int) -> List[Dict[str, Any]]:
        """Search metro/micro areas."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT 'cbsa' as geography_type, name, cbsa_code
            FROM cbsas 
            WHERE LOWER(name) LIKE LOWER(?)
            ORDER BY name
            LIMIT ?
        """
        
        cursor.execute(query, [f"%{name}%", limit])
        results = []
        
        for row in cursor.fetchall():
            results.append({
                'geography_type': 'cbsa',
                'name': row['name'],
                'cbsa_code': row['cbsa_code'],
                'confidence': self._calculate_confidence(name, row['name'])
            })
        
        return results
    
    def _search_zctas(self, zip_code: str, limit: int) -> List[Dict[str, Any]]:
        """Search ZIP Code Tabulation Areas."""
        clean_zip = re.sub(r'[^0-9]', '', zip_code)
        if len(clean_zip) != 5:
            return []
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT zcta_code FROM zctas WHERE zcta_code = ? LIMIT ?", [clean_zip, limit])
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'geography_type': 'zcta',
                'name': f"ZIP Code {row['zcta_code']}",
                'zcta_code': row['zcta_code'],
                'confidence': 1.0
            })
        
        return results
    
    def _is_zip_code(self, location: str) -> bool:
        """Check if location looks like a ZIP code."""
        clean_location = re.sub(r'[^0-9]', '', location)
        return len(clean_location) == 5 and clean_location.isdigit()
    
    def _calculate_confidence(self, query: str, match: str) -> float:
        """Simple confidence calculation."""
        query_lower = query.lower().strip()
        match_lower = match.lower().strip()
        
        # Exact match
        if query_lower == match_lower:
            return 1.0
        
        # Query is contained in match or vice versa
        if query_lower in match_lower:
            return 0.9
        if match_lower in query_lower:
            return 0.8
        
        # Partial overlap
        query_words = set(query_lower.split())
        match_words = set(match_lower.split())
        
        if query_words & match_words:  # Any word overlap
            overlap = len(query_words & match_words)
            total = len(query_words | match_words)
            return 0.5 + (0.3 * overlap / total)
        
        return 0.3  # Minimal match
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results."""
        seen = set()
        unique_results = []
        
        for result in results:
            # Create unique key
            if result['geography_type'] == 'place':
                key = f"place_{result['state_fips']}_{result['place_fips']}"
            elif result['geography_type'] == 'county':
                key = f"county_{result['state_fips']}_{result['county_fips']}"
            elif result['geography_type'] == 'cbsa':
                key = f"cbsa_{result['cbsa_code']}"
            elif result['geography_type'] == 'zcta':
                key = f"zcta_{result['zcta_code']}"
            else:
                key = f"{result['geography_type']}_{result['name']}"
            
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cleanup database connection."""
        self.close()
