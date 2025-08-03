#!/usr/bin/env python3
"""
Production Geographic Handler
- Uses SQLite database with 29,573 places
- Hot cache for top 500 cities (<1ms lookups)
- Multi-strategy resolution with fuzzy matching
- Proper FIPS code resolution for Census API
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

logger = logging.getLogger(__name__)

class GeographicHandler:
    """Production geographic resolution with SQLite database"""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            # Geography database is at ../knowledge-base/geo-db/geography.db
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent  # Go up to project root
            db_path = project_root / "knowledge-base" / "geo-db" / "geography.db"
        
        self.db_path = Path(db_path)
        self.conn = None
        self.hot_cache = {}
        
        # Initialize database connection
        self._init_database()
        self._load_hot_cache()
    
    def _init_database(self):
        """Initialize SQLite database connection"""
        try:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Geography database not found: {self.db_path}")
            
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            
            # Verify database structure
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['places', 'counties', 'states']
            missing_tables = [t for t in required_tables if t not in tables]
            
            if missing_tables:
                raise ValueError(f"Database missing required tables: {missing_tables}")
            
            logger.info(f"✅ Geographic database connected: {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize geographic database: {e}")
            raise
    
    def _load_hot_cache(self):
        """Load top 500 cities into memory for instant lookups"""
        if not self.conn:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Load top cities by population and major cities (using actual schema)
            cursor.execute("""
                SELECT place_fips, state_fips, state_abbrev, name, 
                       LOWER(name || ', ' || state_abbrev) as cache_key
                FROM places 
                ORDER BY 
                    CASE WHEN LOWER(name) IN (
                        'new york', 'los angeles', 'chicago', 'houston', 'phoenix',
                        'philadelphia', 'san antonio', 'san diego', 'dallas', 'austin',
                        'jacksonville', 'fort worth', 'columbus', 'charlotte', 'detroit',
                        'el paso', 'seattle', 'denver', 'washington', 'boston',
                        'nashville', 'baltimore', 'oklahoma city', 'portland', 'las vegas',
                        'milwaukee', 'albuquerque', 'tucson', 'fresno', 'sacramento',
                        'kansas city', 'mesa', 'atlanta', 'omaha', 'colorado springs',
                        'raleigh', 'long beach', 'virginia beach', 'miami', 'minneapolis',
                        'oakland', 'tulsa', 'arlington', 'new orleans', 'wichita',
                        'cleveland', 'tampa', 'bakersfield', 'honolulu', 'st. louis'
                    ) THEN 0 ELSE 1 END,
                    COALESCE(population, 0) DESC,
                    land_area DESC
                LIMIT 500
            """)
            
            for row in cursor.fetchall():
                cache_key = row['cache_key']
                self.hot_cache[cache_key] = {
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name'],
                    'geography': 'place',
                    'resolution_method': 'hot_cache'
                }
            
            logger.info(f"✅ Hot cache loaded: {len(self.hot_cache)} cities")
            
        except Exception as e:
            logger.warning(f"Failed to load hot cache: {e}")
            self.hot_cache = {}
    
    def resolve_location(self, location_string: str) -> Dict[str, Any]:
        """
        Resolve location string to proper FIPS codes
        
        Multi-strategy approach:
        1. Hot cache (instant for top 500 cities)
        2. Exact database match
        3. Fuzzy matching with name variations  
        4. State-only fallback
        5. Smart suggestions on failure
        
        Returns:
            Dict with geography, state_fips, place_fips, state_abbrev, name, resolution_method
        """
        location = location_string.strip()
        
        # Handle national level
        if location.lower() in ['united states', 'usa', 'us', 'america', 'national']:
            return {
                'geography': 'us',
                'state_fips': None,
                'place_fips': None,
                'state_abbrev': None,
                'name': 'United States',
                'resolution_method': 'national_keyword'
            }
        
        # Strategy 1: Hot cache check (fastest path <1ms)
        location_key = location.lower()
        if location_key in self.hot_cache:
            logger.debug(f"✅ Hot cache hit: {location}")
            return self.hot_cache[location_key]
        
        # Strategy 2: Parse location components
        parsed = self._parse_location_components(location)
        if not parsed:
            return self._fail_with_suggestions(location, "Could not parse location format")
        
        city_name = parsed['city']
        state_abbrev = parsed['state']
        
        # Strategy 3: Exact database match
        result = self._exact_database_lookup(city_name, state_abbrev)
        if result:
            logger.debug(f"✅ Exact database match: {location}")
            result['resolution_method'] = 'exact_match'
            return result
        
        # Strategy 4: Fuzzy matching
        result = self._fuzzy_database_lookup(city_name, state_abbrev)
        if result:
            logger.debug(f"✅ Fuzzy match: {location} → {result['name']}")
            result['resolution_method'] = 'fuzzy_match'
            return result
        
        # Strategy 5: State-only fallback
        result = self._resolve_state_only(state_abbrev)
        if result:
            logger.debug(f"✅ State fallback: {location} → {state_abbrev}")
            result['resolution_method'] = 'state_fallback'
            return result
        
        # Strategy 6: Final failure with suggestions
        return self._fail_with_suggestions(location, "Location not found in database")
    
    def _parse_location_components(self, location: str) -> Optional[Dict[str, str]]:
        """Parse various location formats into city and state components"""
        location = location.strip()
        
        # Format: "City, ST" (most common)
        if ',' in location:
            parts = [part.strip() for part in location.split(',')]
            if len(parts) == 2:
                city_name = parts[0]
                state = parts[1].upper()
                
                # Validate state abbreviation
                if len(state) == 2 and state.isalpha():
                    return {'city': city_name, 'state': state}
        
        # Format: "City ST" (space-separated)
        parts = location.split()
        if len(parts) >= 2:
            potential_state = parts[-1].upper()
            if len(potential_state) == 2 and potential_state.isalpha():
                city_name = ' '.join(parts[:-1])
                return {'city': city_name, 'state': potential_state}
        
        # Single word - might be a state
        if len(location.split()) == 1:
            state_result = self._resolve_state_name(location)
            if state_result:
                return {'city': None, 'state': state_result['state_abbrev']}
        
        return None
    
    def _exact_database_lookup(self, city_name: str, state_abbrev: str) -> Optional[Dict[str, Any]]:
        """Exact database lookup for city in state with place type handling"""
        if not self.conn or not city_name:
            return None
        
        try:
            cursor = self.conn.cursor()
            
            # Try exact match first
            cursor.execute("""
                SELECT place_fips, state_fips, state_abbrev, name
                FROM places 
                WHERE name_lower = LOWER(?) AND state_abbrev = ?
                LIMIT 1
            """, (city_name, state_abbrev))
            
            row = cursor.fetchone()
            if row:
                return {
                    'geography': 'place',
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name']
                }
            
            # Try with common place type suffixes
            place_types = ['city', 'town', 'village', 'borough', 'township']
            for place_type in place_types:
                cursor.execute("""
                    SELECT place_fips, state_fips, state_abbrev, name
                    FROM places 
                    WHERE name_lower = LOWER(?) AND state_abbrev = ?
                    LIMIT 1
                """, (f"{city_name} {place_type}", state_abbrev))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'geography': 'place',
                        'place_fips': row['place_fips'],
                        'state_fips': row['state_fips'],
                        'state_abbrev': row['state_abbrev'],
                        'name': row['name']
                    }
            
        except Exception as e:
            logger.error(f"Database lookup error: {e}")
        
        return None
    
    def _fuzzy_database_lookup(self, city_name: str, state_abbrev: str) -> Optional[Dict[str, Any]]:
        """Fuzzy matching for common city name variations (using actual schema)"""
        if not self.conn or not city_name:
            return None
        
        try:
            cursor = self.conn.cursor()
            
            # Generate name variations
            variations = self._generate_name_variations(city_name)
            
            for variation in variations:
                cursor.execute("""
                    SELECT place_fips, state_fips, state_abbrev, name
                    FROM places 
                    WHERE name_lower = LOWER(?) AND state_abbrev = ?
                    LIMIT 1
                """, (variation, state_abbrev))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'geography': 'place',
                        'place_fips': row['place_fips'],
                        'state_fips': row['state_fips'],
                        'state_abbrev': row['state_abbrev'],
                        'name': row['name'],
                        'original_query': city_name,
                        'matched_variation': variation
                    }
            
        except Exception as e:
            logger.error(f"Fuzzy lookup error: {e}")
        
        return None
    
    def _generate_name_variations(self, city_name: str) -> List[str]:
        """Generate common variations of city names"""
        variations = [city_name]
        name_lower = city_name.lower()
        
        # Saint <-> St. variations
        if name_lower.startswith('saint '):
            variations.append('St. ' + city_name[6:])
        elif name_lower.startswith('st. '):
            variations.append('Saint ' + city_name[4:])
        elif name_lower.startswith('st '):
            variations.append('Saint ' + city_name[3:])
        
        # Remove periods (St. Paul → St Paul)
        if '.' in city_name:
            variations.append(city_name.replace('.', ''))
        
        # Add periods (St Paul → St. Paul)
        if ' st ' in name_lower:
            variations.append(city_name.replace(' St ', ' St. '))
        
        return variations
    
    def _resolve_state_only(self, state_abbrev: str) -> Optional[Dict[str, Any]]:
        """Resolve state-level geography (using actual schema)"""
        if not self.conn or not state_abbrev:
            return None
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT state_fips, state_abbrev, state_name
                FROM states 
                WHERE state_abbrev = ?
                LIMIT 1
            """, (state_abbrev,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'geography': 'state',
                    'state_fips': row['state_fips'],
                    'place_fips': None,
                    'state_abbrev': row['state_abbrev'],
                    'name': row['state_name']
                }
                
        except Exception as e:
            logger.error(f"State lookup error: {e}")
        
        return None
    
    def _resolve_state_name(self, state_name: str) -> Optional[Dict[str, Any]]:
        """Resolve full state name to abbreviation (using actual schema)"""
        if not self.conn:
            return None
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT state_fips, state_abbrev, state_name
                FROM states 
                WHERE LOWER(state_name) = LOWER(?)
                LIMIT 1
            """, (state_name,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'geography': 'state',
                    'state_fips': row['state_fips'],
                    'place_fips': None,
                    'state_abbrev': row['state_abbrev'],
                    'name': row['state_name']
                }
                
        except Exception as e:
            logger.error(f"State name lookup error: {e}")
        
        return None
    
    def _fail_with_suggestions(self, location: str, reason: str) -> Dict[str, Any]:
        """Generate helpful suggestions when location resolution fails"""
        suggestions = []
        
        # Try to extract partial matches for suggestions
        if self.conn:
            try:
                parsed = self._parse_location_components(location)
                if parsed and parsed['city']:
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        SELECT name, state_abbrev
                        FROM places 
                        WHERE LOWER(name) LIKE LOWER(?) AND funcstat = 'A'
                        ORDER BY CAST(land_area_sqm AS REAL) DESC
                        LIMIT 3
                    """, (f"%{parsed['city']}%",))
                    
                    for row in cursor.fetchall():
                        suggestions.append(f"{row['name']}, {row['state_abbrev']}")
                        
            except Exception as e:
                logger.error(f"Error generating suggestions: {e}")
        
        return {
            'error': f"Could not resolve location: {location}",
            'reason': reason,
            'suggestions': suggestions,
            'resolution_method': 'failed',
            'help': "Try format: 'City, ST' (e.g., 'Austin, TX') or state names (e.g., 'Minnesota')"
        }

    def get_location_suggestions(self, partial_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get location suggestions for partial matches"""
        if not self.conn:
            return []
        
        suggestions = []
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT name, state_abbrev, place_fips, state_fips
                FROM places 
                WHERE LOWER(name) LIKE LOWER(?) AND funcstat = 'A'
                ORDER BY CAST(land_area_sqm AS REAL) DESC
                LIMIT ?
            """, (f"%{partial_name}%", limit))
            
            for row in cursor.fetchall():
                suggestions.append({
                    'name': f"{row['name']}, {row['state_abbrev']}",
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'geography': 'place'
                })
                
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
        
        return suggestions

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

# Test function
def test_geographic_handler():
    """Test the geographic handler with problematic locations"""
    gh = GeographicHandler()
    
    test_locations = [
        "Brainerd, MN",      # Should work now
        "St. Louis, MO",     # Should work now
        "Kansas City, MO",   # Should work now
        "Minneapolis, MN",   # Should continue working
        "Austin, TX",        # Should continue working
        "New York, NY",      # Should continue working
        "Nonexistent, XX",   # Should fail gracefully with suggestions
    ]
    
    for location in test_locations:
        print(f"\n🧪 Testing: {location}")
        try:
            result = gh.resolve_location(location)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                if result.get('suggestions'):
                    print(f"💡 Suggestions: {', '.join(result['suggestions'][:3])}")
            else:
                print(f"✅ Success: {result['name']}")
                print(f"   Geography: {result['geography']}")
                print(f"   Method: {result['resolution_method']}")
                if result.get('place_fips'):
                    print(f"   FIPS: {result['state_fips']}:{result['place_fips']}")
                else:
                    print(f"   FIPS: {result['state_fips']}")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    gh.close()

if __name__ == "__main__":
    test_geographic_handler()
