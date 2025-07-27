"""
Geographic Handler for Census MCP - SQLite-based location resolution

Replaces the broken hardcoded MAJOR_CITIES approach with comprehensive 
database lookup covering 29,573 US places.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

logger = logging.getLogger(__name__)

class GeographicHandler:
    """Handles geographic location resolution using SQLite database"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Find the database relative to project root
            current_dir = Path(__file__).parent  # src/data_retrieval/
            project_root = current_dir.parent.parent  # project root
            db_path = project_root / "knowledge-base" / "geo-db" / "geography.db"
        self.db_path = Path(db_path)
        self.conn = None
        self.hot_cache = {}
        self._init_database()
        self._load_hot_cache()
    
    def _init_database(self):
        """Initialize database connection"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Geography database not found at {self.db_path}")
        
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            logger.info(f"✅ Geographic database connected: {self.db_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to geography database: {e}")
    
    def _load_hot_cache(self):
        """Load top 500 cities into memory for fast access"""
        try:
            cursor = self.conn.cursor()
            # Get most populous places for hot cache
            cursor.execute("""
                SELECT name_lower, place_fips, state_fips, state_abbrev, name
                FROM places 
                ORDER BY CASE 
                    WHEN name_lower IN ('new york', 'los angeles', 'chicago', 'houston', 
                                       'philadelphia', 'phoenix', 'san antonio', 'san diego',
                                       'dallas', 'austin', 'jacksonville', 'fort worth',
                                       'columbus', 'charlotte', 'san francisco', 'indianapolis',
                                       'seattle', 'denver', 'washington', 'boston') THEN 0
                    ELSE 1 
                END, name
                LIMIT 500
            """)
            
            for row in cursor.fetchall():
                cache_key = f"{row['name_lower']}, {row['state_abbrev'].lower()}"
                self.hot_cache[cache_key] = {
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name'],
                    'geography': 'place'
                }
            
            logger.info(f"✅ Hot cache loaded: {len(self.hot_cache)} cities")
            
        except Exception as e:
            logger.warning(f"Failed to load hot cache: {e}")
            self.hot_cache = {}
    
    def resolve_location(self, location_string: str) -> Dict[str, Any]:
        """
        Resolve location string with GPT-level contextual reasoning
        
        Multi-strategy approach:
        1. Hot cache (instant)
        2. Exact database match  
        3. Fuzzy matching with name variations
        4. Cross-type lookup (places → counties for independent cities)
        5. Spatial reasoning (metro areas, adjacent counties)
        6. Smart suggestions on failure
        """
        location = location_string.strip()
        
        # National level
        if location.lower() in ['united states', 'usa', 'us', 'america']:
            return {
                'geography': 'us',
                'state_fips': None,
                'place_fips': None,
                'state_abbrev': None,
                'resolution_method': 'national_keyword'
            }
        
        # Strategy 1: Hot cache check (fastest path)
        location_lower = location.lower()
        if location_lower in self.hot_cache:
            logger.debug(f"✅ Hot cache hit: {location}")
            result = self.hot_cache[location_lower].copy()
            result['resolution_method'] = 'hot_cache'
            return result
        
        # Strategy 2: Parse and normalize location components
        parsed_variants = self._generate_location_variants(location)
        if not parsed_variants:
            return self._fail_with_suggestions(location, "Could not parse location format")
        
        # Strategy 3: Try each variant with escalating strategies
        for variant in parsed_variants:
            city_name = variant['city']
            state = variant['state']
            
            # 3a: Exact database match
            result = self._exact_database_lookup(city_name, state)
            if result:
                result['resolution_method'] = 'exact_match'
                result['variant_used'] = f"{city_name}, {state}"
                logger.debug(f"✅ Exact match: {location} → {result}")
                return result
            
            # 3b: Fuzzy matching using name_variations table
            result = self._fuzzy_database_lookup(city_name, state)
            if result:
                result['resolution_method'] = 'fuzzy_match'
                result['variant_used'] = f"{city_name}, {state}"
                logger.debug(f"✅ Fuzzy match: {location} → {result}")
                return result
            
            # 3c: Cross-type lookup (independent cities as county-equivalents)
            result = self._county_equivalent_lookup(city_name, state)
            if result:
                result['resolution_method'] = 'county_equivalent'
                result['variant_used'] = f"{city_name}, {state}"
                logger.debug(f"✅ County-equivalent match: {location} → {result}")
                return result
        
        # Strategy 4: Metro area / CBSA lookup
        result = self._metro_area_lookup(location)
        if result:
            result['resolution_method'] = 'metro_area'
            logger.debug(f"✅ Metro area match: {location} → {result}")
            return result
        
        # Strategy 5: State-only fallback (graceful degradation)
        for variant in parsed_variants:
            state_result = self._resolve_state_only(variant['state'])
            if state_result:
                state_result['resolution_method'] = 'state_fallback'
                state_result['warning'] = f"City '{variant['city']}' not found, using state-level data"
                logger.debug(f"⚠️ State fallback: {location} → {variant['state']}")
                return state_result
        
        # Strategy 6: Intelligent failure with suggestions
        return self._fail_with_suggestions(location, "No geographic match found")
    
    def _generate_location_variants(self, location: str) -> List[Dict[str, str]]:
        """Generate intelligent variants of the location string"""
        variants = []
        
        # Parse basic format first
        parsed = self._parse_location_components(location)
        if not parsed:
            return variants
        
        city_name = parsed['city']
        state = parsed['state']
        
        # Generate name variants using linguistic intelligence
        city_variants = self._generate_city_name_variants(city_name)
        
        for city_variant in city_variants:
            variants.append({'city': city_variant, 'state': state})
        
        return variants
    
    def _generate_city_name_variants(self, city_name: str) -> List[str]:
        """Generate intelligent city name variants"""
        variants = [city_name.lower()]
        name_lower = city_name.lower()
        
        # Saint/St. variations (both directions)
        if 'saint ' in name_lower:
            variants.append(name_lower.replace('saint ', 'st. '))
            variants.append(name_lower.replace('saint ', 'st '))
        elif 'st. ' in name_lower:
            variants.append(name_lower.replace('st. ', 'saint '))
            variants.append(name_lower.replace('st. ', 'st '))
        elif ' st ' in name_lower:
            variants.append(name_lower.replace(' st ', ' saint '))
            variants.append(name_lower.replace(' st ', ' st. '))
        
        # Remove/add common suffixes intelligently
        suffixes_to_try = ['city', 'town', 'village', 'borough']
        
        # If it has a suffix, try without it
        for suffix in suffixes_to_try:
            if name_lower.endswith(f' {suffix}'):
                variants.append(name_lower.replace(f' {suffix}', ''))
        
        # If it doesn't have a suffix, try adding common ones
        if not any(name_lower.endswith(f' {suffix}') for suffix in suffixes_to_try):
            variants.append(f"{name_lower} city")
        
        return list(set(variants))  # Remove duplicates
    
    def _exact_database_lookup(self, city_name: str, state_abbrev: str) -> Optional[Dict[str, Any]]:
        """Exact database lookup with precise matching"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT place_fips, state_fips, state_abbrev, name
                FROM places 
                WHERE name_lower = ? AND state_abbrev = ?
                LIMIT 1
            """, (city_name.lower(), state_abbrev))
            
            row = cursor.fetchone()
            if row:
                return {
                    'geography': 'place',
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name']
                }
            return None
        except Exception as e:
            logger.error(f"Exact database lookup failed: {e}")
            return None
    
    def _fuzzy_database_lookup(self, city_name: str, state_abbrev: str) -> Optional[Dict[str, Any]]:
        """Fuzzy matching using name_variations table and similarity"""
        try:
            cursor = self.conn.cursor()
            
            # First try name_variations table
            cursor.execute("""
                SELECT p.place_fips, p.state_fips, p.state_abbrev, p.name
                FROM name_variations nv
                JOIN places p ON nv.place_fips = p.place_fips AND nv.state_fips = p.state_fips
                WHERE nv.variation = ? AND p.state_abbrev = ? AND nv.geography_type = 'place'
                LIMIT 1
            """, (city_name.lower(), state_abbrev))
            
            row = cursor.fetchone()
            if row:
                return {
                    'geography': 'place',
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name']
                }
            
            # Fall back to LIKE matching for partial matches
            cursor.execute("""
                SELECT place_fips, state_fips, state_abbrev, name, name_lower
                FROM places 
                WHERE state_abbrev = ? AND (
                    name_lower LIKE ? OR 
                    name_lower LIKE ?
                )
                ORDER BY 
                    CASE WHEN name_lower = ? THEN 0 ELSE 1 END,
                    LENGTH(name_lower)
                LIMIT 3
            """, (
                state_abbrev,
                f"%{city_name.lower()}%",
                f"{city_name.lower()}%",
                city_name.lower()
            ))
            
            rows = cursor.fetchall()
            if rows:
                # Return best match (prefer exact substring matches)
                best_row = rows[0]
                return {
                    'geography': 'place',
                    'place_fips': best_row['place_fips'],
                    'state_fips': best_row['state_fips'],
                    'state_abbrev': best_row['state_abbrev'],
                    'name': best_row['name']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Fuzzy database lookup failed: {e}")
            return None
    
    def _county_equivalent_lookup(self, city_name: str, state_abbrev: str) -> Optional[Dict[str, Any]]:
        """Check for independent cities in counties table (like St. Louis)"""
        try:
            cursor = self.conn.cursor()
            
            # Look for city name in counties table (independent cities)
            cursor.execute("""
                SELECT county_fips, state_fips, state_abbrev, name
                FROM counties 
                WHERE name_lower LIKE ? AND state_abbrev = ?
                ORDER BY 
                    CASE WHEN name_lower = ? THEN 0 ELSE 1 END,
                    LENGTH(name_lower)
                LIMIT 1
            """, (f"%{city_name.lower()}%", state_abbrev, city_name.lower()))
            
            row = cursor.fetchone()
            if row:
                # Return as county geography for independent cities
                return {
                    'geography': 'county',
                    'county_fips': row['county_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"County equivalent lookup failed: {e}")
            return None
    
    def _metro_area_lookup(self, location: str) -> Optional[Dict[str, Any]]:
        """Look for metro/micro statistical areas"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT cbsa_code, name, cbsa_type
                FROM cbsas 
                WHERE name_lower LIKE ?
                ORDER BY LENGTH(name_lower)
                LIMIT 1
            """, (f"%{location.lower()}%",))
            
            row = cursor.fetchone()
            if row:
                return {
                    'geography': 'cbsa',
                    'cbsa_code': row['cbsa_code'],
                    'name': row['name'],
                    'cbsa_type': row['cbsa_type']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Metro area lookup failed: {e}")
            return None
    
    def _fail_with_suggestions(self, location: str, reason: str) -> Dict[str, Any]:
        """Intelligent failure with helpful suggestions"""
        suggestions = self.get_suggestions(location, limit=3)
        
        error_msg = f"Could not resolve location: {location}"
        if suggestions:
            error_msg += f". Did you mean: {', '.join(suggestions[:3])}?"
        
        return {
            'error': error_msg,
            'reason': reason,
            'suggestions': suggestions,
            'resolution_method': 'failed'
        }
    
    def _parse_location_components(self, location: str) -> Optional[Dict[str, str]]:
        """Parse 'City, ST' format"""
        # Handle comma-separated format: "St. Louis, MO"
        if ',' in location:
            parts = [part.strip() for part in location.split(',')]
            if len(parts) == 2:
                city_name = parts[0]
                state = parts[1].upper()
                return {'city': city_name, 'state': state}
        
        # Handle space-separated format: "St. Louis MO"
        parts = location.split()
        if len(parts) >= 2:
            # Last part might be state
            potential_state = parts[-1].upper()
            if len(potential_state) == 2 and potential_state.isalpha():
                city_name = ' '.join(parts[:-1])
                return {'city': city_name, 'state': potential_state}
        
        return None
    
    def _database_lookup(self, city_name: str, state_abbrev: str) -> Optional[Dict[str, Any]]:
        """Database lookup for city in state"""
        try:
            cursor = self.conn.cursor()
            
            # Exact match first
            cursor.execute("""
                SELECT place_fips, state_fips, state_abbrev, name
                FROM places 
                WHERE name_lower = ? AND state_abbrev = ?
                LIMIT 1
            """, (city_name.lower(), state_abbrev))
            
            row = cursor.fetchone()
            if row:
                return {
                    'geography': 'place',
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name']
                }
            
            # Fuzzy match for common variations
            cursor.execute("""
                SELECT place_fips, state_fips, state_abbrev, name, name_lower
                FROM places 
                WHERE state_abbrev = ? AND (
                    name_lower LIKE ? OR 
                    name_lower LIKE ? OR
                    name_lower LIKE ?
                )
                LIMIT 5
            """, (
                state_abbrev,
                f"%{city_name.lower()}%",
                city_name.lower().replace('saint', 'st.'),
                city_name.lower().replace('st.', 'saint')
            ))
            
            rows = cursor.fetchall()
            if rows:
                # Return best match (exact substring match preferred)
                city_lower = city_name.lower()
                for row in rows:
                    if row['name_lower'] == city_lower:
                        return {
                            'geography': 'place',
                            'place_fips': row['place_fips'],
                            'state_fips': row['state_fips'],
                            'state_abbrev': row['state_abbrev'],
                            'name': row['name']
                        }
                
                # Return first match if no exact match
                row = rows[0]
                return {
                    'geography': 'place',
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Database lookup failed: {e}")
            return None
    
    def _resolve_state_only(self, state_abbrev: str) -> Optional[Dict[str, Any]]:
        """Resolve to state-level geography"""
        state_fips_map = {
            'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
            'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16',
            'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22',
            'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
            'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
            'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40',
            'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
            'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
            'WI': '55', 'WY': '56', 'DC': '11', 'PR': '72'
        }
        
        state_fips = state_fips_map.get(state_abbrev)
        if state_fips:
            return {
                'geography': 'state',
                'state_fips': state_fips,
                'place_fips': None,
                'state_abbrev': state_abbrev
            }
        
        return None
    
    def get_suggestions(self, location: str, limit: int = 5) -> List[str]:
        """Get suggestions for failed location lookups"""
        try:
            parsed = self._parse_location_components(location)
            if not parsed:
                return []
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT name, state_abbrev 
                FROM places 
                WHERE name_lower LIKE ? 
                ORDER BY name 
                LIMIT ?
            """, (f"%{parsed['city'].lower()}%", limit))
            
            return [f"{row['name']}, {row['state_abbrev']}" for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []
