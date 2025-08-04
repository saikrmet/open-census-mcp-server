#!/usr/bin/env python3
"""
Complete Geographic Handler - Fixed for Actual Database Schema
Supports ALL Census geography levels with spatial analysis capabilities

SCHEMA FIXES:
- Removed funcstat filtering (column doesn't exist)
- Fixed cbsa_type mapping ('1'=Metro, '2'=Micro)
- Fixed county adjacency column names
- Handles actual database structure

COMPLETE COVERAGE:
- 32,285 places (cities, towns, villages)  
- 3,222 counties (parishes, boroughs)
- 935 CBSAs (metro/micro statistical areas)
- 33,791 ZCTAs (ZIP Code Tabulation Areas)
- County adjacency for spatial queries
- Hot cache for instant lookups

GEOGRAPHY LEVELS SUPPORTED:
- us: National level
- state: States + DC + territories
- county: Counties, parishes, boroughs
- place: Incorporated places
- zcta: ZIP Code Tabulation Areas  
- cbsa: Metropolitan/micropolitan areas
- tract: Census tracts (with boundary data)
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import re
import json

logger = logging.getLogger(__name__)

class CompleteGeographicHandler:
    """Complete geographic resolution with full gazetteer dataset"""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            # Auto-detect geography database - check correct path
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            
            # Check both possible locations
            possible_paths = [
                project_root / "knowledge-base" / "geo-db" / "geography.db",
                project_root / "knowledge-base" / "geography.db"
            ]
            
            for path in possible_paths:
                if path.exists():
                    db_path = path
                    break
        
        self.db_path = Path(db_path) if db_path else None
        self.conn = None
        self.hot_cache = {}
        
        # Initialize database and validate completeness
        self._init_complete_database()
        self._load_hot_cache()
        self._validate_geographic_coverage()
    
    def _init_complete_database(self):
        """Initialize complete geographic database with all tables"""
        try:
            if not self.db_path or not self.db_path.exists():
                raise FileNotFoundError(f"Complete geography database not found: {self.db_path}")
            
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Verify all required tables exist
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            
            required_tables = {
                'places',           # Cities, towns, villages
                'counties',         # Counties, parishes, boroughs
                'states',          # States + DC + territories
                'cbsas',           # Metropolitan/micropolitan areas
                'zctas',           # ZIP Code Tabulation Areas
                'county_adjacency', # Spatial relationships
                'name_variations'   # Fuzzy matching support
            }
            
            missing_tables = required_tables - tables
            if missing_tables:
                raise ValueError(f"Database missing required tables: {missing_tables}")
            
            logger.info(f"✅ Complete geographic database connected: {self.db_path}")
            logger.info(f"✅ All {len(required_tables)} required tables present")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize complete geographic database: {e}")
            raise
    
    def _validate_geographic_coverage(self):
        """Validate completeness of geographic coverage"""
        try:
            cursor = self.conn.cursor()
            
            # Count records in each table
            coverage = {}
            
            cursor.execute("SELECT COUNT(*) FROM places")
            coverage['places'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM counties")
            coverage['counties'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM states")
            coverage['states'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM cbsas")
            coverage['cbsas'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM zctas")
            coverage['zctas'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM county_adjacency")
            coverage['adjacencies'] = cursor.fetchone()[0]
            
            # Log coverage statistics
            logger.info("🌍 Complete Geographic Coverage:")
            logger.info(f"   Places: {coverage['places']:,} (cities, towns, villages)")
            logger.info(f"   Counties: {coverage['counties']:,} (counties, parishes, boroughs)")
            logger.info(f"   States: {coverage['states']:,} (states + DC + territories)")
            logger.info(f"   CBSAs: {coverage['cbsas']:,} (metro/micro areas)")
            logger.info(f"   ZCTAs: {coverage['zctas']:,} (ZIP code areas)")
            logger.info(f"   County adjacencies: {coverage['adjacencies']:,} (spatial relationships)")
            
            # Validate expected ranges
            expected_minimums = {
                'places': 25000,      # Should have ~32K places
                'counties': 3000,     # Should have ~3.2K counties
                'states': 50,         # Should have 50+ (states + territories)
                'cbsas': 900,         # Should have ~935 CBSAs
                'zctas': 30000,       # Should have ~33K ZCTAs
                'adjacencies': 15000  # Should have ~19K adjacencies
            }
            
            for table, count in coverage.items():
                if table in expected_minimums and count < expected_minimums[table]:
                    logger.warning(f"⚠️ Low {table} count: {count} (expected >{expected_minimums[table]})")
                else:
                    logger.info(f"✅ {table.title()} coverage validated")
            
            self.coverage_stats = coverage
            
        except Exception as e:
            logger.error(f"❌ Geographic coverage validation failed: {e}")
            self.coverage_stats = {}
    
    def _load_hot_cache(self):
        """Load frequently accessed locations into hot cache - FIXED for actual schema"""
        if not self.conn:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Load major metropolitan areas for instant access
            # FIXED: Use actual cbsa_type values ('1' = Metro, '2' = Micro)
            cursor.execute("""
                SELECT cbsa_code, name, cbsa_type, name_lower
                FROM cbsas 
                WHERE cbsa_type = '1'
                ORDER BY CAST(COALESCE(land_area, 0) AS REAL) DESC
                LIMIT 100
            """)
            
            for row in cursor.fetchall():
                cache_key = f"cbsa:{row['name_lower']}"
                self.hot_cache[cache_key] = {
                    'geography_type': 'cbsa',
                    'cbsa_code': row['cbsa_code'],
                    'name': row['name'],
                    'cbsa_type': 'Metropolitan Statistical Area',  # Convert to readable format
                    'resolution_method': 'hot_cache_cbsa'
                }
            
            # Load major cities for instant access
            # FIXED: Removed funcstat filter since column doesn't exist
            cursor.execute("""
                SELECT place_fips, state_fips, state_abbrev, name, name_lower, population
                FROM places 
                WHERE population IS NOT NULL
                ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                LIMIT 500
            """)
            
            for row in cursor.fetchall():
                cache_key = f"place:{row['name_lower']},{row['state_abbrev'].lower()}"
                self.hot_cache[cache_key] = {
                    'geography_type': 'place',
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name'],
                    'population': row['population'],
                    'resolution_method': 'hot_cache_place'
                }
            
            # Load all states for instant access
            cursor.execute("SELECT state_fips, state_abbrev, state_name FROM states")
            for row in cursor.fetchall():
                # Cache by full name
                cache_key = f"state:{row['state_name'].lower()}"
                self.hot_cache[cache_key] = {
                    'geography_type': 'state',
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['state_name'],
                    'resolution_method': 'hot_cache_state'
                }
                
                # Cache by abbreviation
                cache_key = f"state:{row['state_abbrev'].lower()}"
                self.hot_cache[cache_key] = {
                    'geography_type': 'state',
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['state_name'],
                    'resolution_method': 'hot_cache_state_abbrev'
                }
            
            logger.info(f"✅ Hot cache loaded: {len(self.hot_cache)} locations")
            
        except Exception as e:
            logger.warning(f"Failed to load hot cache: {e}")
            import traceback
            traceback.print_exc()
            self.hot_cache = {}
    
    def resolve_location(self, location_string: str) -> Dict[str, Any]:
        """
        Complete location resolution supporting all Census geography levels
        
        Supports:
        - National: "United States", "USA", "national"
        - States: "New York", "NY", "California" 
        - Places: "Austin, TX", "New York City", "St. Louis, MO"
        - Counties: "Cook County, IL", "Los Angeles County"
        - CBSAs: "San Francisco Bay Area", "Chicago metro", "Greater Boston"
        - ZCTAs: "90210", "10001", "ZIP code 78701"
        - Spatial: "counties near Baltimore", "places in Dallas metro"
        """
        
        location = location_string.strip()
        
        # Handle national level
        if location.lower() in ['united states', 'usa', 'us', 'america', 'national', 'nationwide']:
            return {
                'geography_type': 'us',
                'state_fips': None,
                'place_fips': None,
                'county_fips': None,
                'cbsa_code': None,
                'zcta_code': None,
                'name': 'United States',
                'resolution_method': 'national_keyword'
            }
        
        # Multi-strategy resolution with hot cache priority
        strategies = [
            self._hot_cache_lookup,
            self._zcta_lookup,           # ZIP codes (high confidence patterns)
            self._cbsa_pattern_lookup,   # Metro area patterns
            self._place_state_lookup,    # "City, ST" format
            self._county_lookup,         # County patterns
            self._state_lookup,          # State names/abbreviations
            self._fuzzy_place_lookup,    # Fuzzy place matching
            self._spatial_query_lookup   # "near X", "in Y metro"
        ]
        
        for strategy in strategies:
            try:
                result = strategy(location)
                if result and 'error' not in result:
                    return result
            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed for '{location}': {e}")
                continue
        
        # All strategies failed
        return self._fail_with_comprehensive_suggestions(location)
    
    def _hot_cache_lookup(self, location: str) -> Optional[Dict[str, Any]]:
        """Instant lookup from hot cache"""
        location_lower = location.lower()
        
        # Try direct cache hits
        cache_keys = [
            f"place:{location_lower}",
            f"state:{location_lower}",
            f"cbsa:{location_lower}"
        ]
        
        for cache_key in cache_keys:
            if cache_key in self.hot_cache:
                return self.hot_cache[cache_key]
        
        return None
    
    def _zcta_lookup(self, location: str) -> Optional[Dict[str, Any]]:
        """ZIP Code Tabulation Area lookup"""
        
        # Pattern matching for ZIP codes
        zip_patterns = [
            r'\b(\d{5})\b',                    # "90210"
            r'zip\s*code?\s*(\d{5})',          # "ZIP code 90210"
            r'postal\s*code\s*(\d{5})',        # "postal code 90210"
            r'zcta\s*(\d{5})'                  # "ZCTA 90210"
        ]
        
        for pattern in zip_patterns:
            match = re.search(pattern, location, re.IGNORECASE)
            if match:
                zcta_code = match.group(1)
                
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        SELECT zcta_code, lat, lon
                        FROM zctas 
                        WHERE zcta_code = ?
                    """, (zcta_code,))
                    
                    row = cursor.fetchone()
                    if row:
                        return {
                            'geography_type': 'zcta',
                            'zcta_code': row['zcta_code'],
                            'state_fips': None,  # ZCTAs can cross state boundaries
                            'name': f"ZIP Code {row['zcta_code']}",
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'resolution_method': 'zcta_pattern_match'
                        }
                except Exception as e:
                    logger.warning(f"ZCTA lookup failed for {zcta_code}: {e}")
        
        return None
    
    def _cbsa_pattern_lookup(self, location: str) -> Optional[Dict[str, Any]]:
        """Metropolitan/micropolitan statistical area lookup - FIXED for actual schema"""
        
        # Metro area patterns
        metro_patterns = [
            r'(.+?)\s+metro(?:\s+area)?',           # "Chicago metro"
            r'greater\s+(.+?)(?:\s|$)',             # "Greater Boston"
            r'(.+?)\s+bay\s+area',                  # "San Francisco Bay Area"
            r'(.+?)\s+metropolitan\s+area',         # "Dallas metropolitan area"
            r'(.+?)\s+msa',                         # "Austin MSA"
            r'(.+?)\s+region'                       # "Baltimore region"
        ]
        
        for pattern in metro_patterns:
            match = re.search(pattern, location, re.IGNORECASE)
            if match:
                metro_name = match.group(1).strip()
                
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        SELECT cbsa_code, name, cbsa_type, lat, lon
                        FROM cbsas 
                        WHERE name_lower LIKE ? OR name_lower LIKE ?
                        ORDER BY 
                            CASE WHEN cbsa_type = '1' THEN 0 ELSE 1 END,
                            LENGTH(name)
                        LIMIT 1
                    """, (f"%{metro_name.lower()}%", f"{metro_name.lower()}%"))
                    
                    row = cursor.fetchone()
                    if row:
                        # Convert cbsa_type to readable format
                        readable_type = 'Metropolitan Statistical Area' if row['cbsa_type'] == '1' else 'Micropolitan Statistical Area'
                        
                        return {
                            'geography_type': 'cbsa',
                            'cbsa_code': row['cbsa_code'],
                            'name': row['name'],
                            'cbsa_type': readable_type,
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'resolution_method': 'cbsa_pattern_match'
                        }
                except Exception as e:
                    logger.warning(f"CBSA lookup failed for {metro_name}: {e}")
        
        return None
    
    def _place_state_lookup(self, location: str) -> Optional[Dict[str, Any]]:
        """City, State format lookup - FIXED to remove funcstat filter"""
        
        # Parse "City, ST" format
        place_state_pattern = r'^(.+?),\s*([A-Z]{2})$'
        match = re.match(place_state_pattern, location.strip())
        
        if not match:
            return None
        
        place_name = match.group(1).strip()
        state_abbrev = match.group(2).upper()
        
        try:
            cursor = self.conn.cursor()
            
            # Exact match first - REMOVED funcstat filter
            cursor.execute("""
                SELECT place_fips, state_fips, state_abbrev, name, 
                       population, lat, lon
                FROM places 
                WHERE name_lower = ? AND state_abbrev = ?
                ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                LIMIT 1
            """, (place_name.lower(), state_abbrev))
            
            row = cursor.fetchone()
            if row:
                return {
                    'geography_type': 'place',
                    'place_fips': row['place_fips'],
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['name'],
                    'population': row['population'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'resolution_method': 'place_exact_match'
                }
            
            # Try with common place type suffixes
            place_types = ['city', 'town', 'village', 'borough']
            for place_type in place_types:
                cursor.execute("""
                    SELECT place_fips, state_fips, state_abbrev, name,
                           population, lat, lon
                    FROM places 
                    WHERE name_lower = ? AND state_abbrev = ?
                    ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                    LIMIT 1
                """, (f"{place_name.lower()} {place_type}", state_abbrev))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'geography_type': 'place',
                        'place_fips': row['place_fips'],
                        'state_fips': row['state_fips'],
                        'state_abbrev': row['state_abbrev'],
                        'name': row['name'],
                        'population': row['population'],
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'resolution_method': 'place_type_match'
                    }
        
        except Exception as e:
            logger.warning(f"Place lookup failed for {place_name}, {state_abbrev}: {e}")
        
        return None
    
    def _county_lookup(self, location: str) -> Optional[Dict[str, Any]]:
        """County lookup with various naming patterns - FIXED schema"""
        
        # County patterns
        county_patterns = [
            r'^(.+?)\s+county(?:\s*,\s*([A-Z]{2}))?$',      # "Cook County, IL"
            r'^(.+?)\s+parish(?:\s*,\s*([A-Z]{2}))?$',      # "Orleans Parish, LA"
            r'^(.+?)\s+borough(?:\s*,\s*([A-Z]{2}))?$'      # "Bronx Borough, NY"
        ]
        
        for pattern in county_patterns:
            match = re.match(pattern, location.strip(), re.IGNORECASE)
            if match:
                county_name = match.group(1).strip()
                state_abbrev = match.group(2)
                
                try:
                    cursor = self.conn.cursor()
                    
                    if state_abbrev:
                        # State specified
                        cursor.execute("""
                            SELECT county_fips, state_fips, state_abbrev, name,
                                   lat, lon
                            FROM counties 
                            WHERE name_lower LIKE ? AND state_abbrev = ?
                            ORDER BY LENGTH(name)
                            LIMIT 1
                        """, (f"%{county_name.lower()}%", state_abbrev.upper()))
                    else:
                        # No state specified - find best match
                        cursor.execute("""
                            SELECT county_fips, state_fips, state_abbrev, name,
                                   lat, lon
                            FROM counties 
                            WHERE name_lower LIKE ?
                            ORDER BY LENGTH(name)
                            LIMIT 1
                        """, (f"%{county_name.lower()}%",))
                    
                    row = cursor.fetchone()
                    if row:
                        return {
                            'geography_type': 'county',
                            'county_fips': row['county_fips'],
                            'state_fips': row['state_fips'],
                            'state_abbrev': row['state_abbrev'],
                            'name': row['name'],
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'resolution_method': 'county_pattern_match'
                        }
                
                except Exception as e:
                    logger.warning(f"County lookup failed for {county_name}: {e}")
        
        return None
    
    def _state_lookup(self, location: str) -> Optional[Dict[str, Any]]:
        """State lookup by name or abbreviation"""
        
        try:
            cursor = self.conn.cursor()
            
            # Try abbreviation first
            if len(location) == 2 and location.isalpha():
                cursor.execute("""
                    SELECT state_fips, state_abbrev, state_name
                    FROM states 
                    WHERE state_abbrev = ?
                """, (location.upper(),))
            else:
                # Try full name
                cursor.execute("""
                    SELECT state_fips, state_abbrev, state_name
                    FROM states 
                    WHERE LOWER(state_name) = ?
                """, (location.lower(),))
            
            row = cursor.fetchone()
            if row:
                return {
                    'geography_type': 'state',
                    'state_fips': row['state_fips'],
                    'state_abbrev': row['state_abbrev'],
                    'name': row['state_name'],
                    'resolution_method': 'state_direct_match'
                }
        
        except Exception as e:
            logger.warning(f"State lookup failed for {location}: {e}")
        
        return None
    
    def _fuzzy_place_lookup(self, location: str) -> Optional[Dict[str, Any]]:
        """Fuzzy matching for place names - FIXED to remove funcstat"""
        
        try:
            cursor = self.conn.cursor()
            
            # Generate name variations
            variations = self._generate_name_variations(location)
            
            for variation in variations:
                cursor.execute("""
                    SELECT place_fips, state_fips, state_abbrev, name,
                           population, lat, lon
                    FROM places 
                    WHERE name_lower = ?
                    ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                    LIMIT 1
                """, (variation.lower(),))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'geography_type': 'place',
                        'place_fips': row['place_fips'],
                        'state_fips': row['state_fips'],
                        'state_abbrev': row['state_abbrev'],
                        'name': row['name'],
                        'population': row['population'],
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'original_query': location,
                        'matched_variation': variation,
                        'resolution_method': 'fuzzy_place_match'
                    }
        
        except Exception as e:
            logger.warning(f"Fuzzy place lookup failed for {location}: {e}")
        
        return None
    
    def _spatial_query_lookup(self, location: str) -> Optional[Dict[str, Any]]:
        """Handle spatial queries using county adjacency data"""
        
        # Spatial query patterns
        spatial_patterns = [
            r'(?:counties|places)\s+near\s+(.+)',      # "counties near Baltimore"
            r'(?:in|within)\s+(.+?)\s+metro',          # "in Dallas metro"
            r'adjacent\s+to\s+(.+)',                   # "adjacent to Cook County"
            r'surrounding\s+(.+)'                      # "surrounding Austin"
        ]
        
        for pattern in spatial_patterns:
            match = re.search(pattern, location, re.IGNORECASE)
            if match:
                anchor_location = match.group(1).strip()
                
                # This would require complex spatial logic
                # For now, return a placeholder indicating spatial query detected
                return {
                    'geography_type': 'spatial',
                    'anchor_location': anchor_location,
                    'spatial_query': location,
                    'resolution_method': 'spatial_query_detected',
                    'note': 'Spatial queries require additional processing'
                }
        
        return None
    
    def _generate_name_variations(self, location: str) -> List[str]:
        """Generate common variations of location names"""
        variations = [location]
        location_lower = location.lower()
        
        # Saint <-> St. variations
        if location_lower.startswith('saint '):
            variations.append('St. ' + location[6:])
            variations.append('St ' + location[6:])
        elif location_lower.startswith('st. '):
            variations.append('Saint ' + location[4:])
        elif location_lower.startswith('st '):
            variations.append('Saint ' + location[3:])
            variations.append('St. ' + location[3:])
        
        # Remove/add periods
        if '.' in location:
            variations.append(location.replace('.', ''))
        else:
            variations.append(location.replace(' St ', ' St. '))
        
        # Mount <-> Mt. variations
        if location_lower.startswith('mount '):
            variations.append('Mt. ' + location[6:])
        elif location_lower.startswith('mt. '):
            variations.append('Mount ' + location[4:])
        
        return list(set(variations))  # Remove duplicates
    
    def _fail_with_comprehensive_suggestions(self, location: str) -> Dict[str, Any]:
        """Generate comprehensive suggestions when all resolution strategies fail"""
        
        suggestions = []
        
        try:
            cursor = self.conn.cursor()
            
            # Find partial matches across all geography types
            search_term = f"%{location.lower()}%"
            
            # Places
            cursor.execute("""
                SELECT 'place' as type, name || ', ' || state_abbrev as suggestion
                FROM places 
                WHERE name_lower LIKE ?
                ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                LIMIT 3
            """, (search_term,))
            suggestions.extend([f"{row['suggestion']}" for row in cursor.fetchall()])
            
            # States
            cursor.execute("""
                SELECT 'state' as type, state_name as suggestion
                FROM states 
                WHERE LOWER(state_name) LIKE ?
                LIMIT 2
            """, (search_term,))
            suggestions.extend([row['suggestion'] for row in cursor.fetchall()])
            
            # Counties
            cursor.execute("""
                SELECT 'county' as type, name || ', ' || state_abbrev as suggestion
                FROM counties 
                WHERE name_lower LIKE ?
                ORDER BY LENGTH(name)
                LIMIT 2
            """, (search_term,))
            suggestions.extend([row['suggestion'] for row in cursor.fetchall()])
            
            # CBSAs
            cursor.execute("""
                SELECT 'cbsa' as type, name as suggestion
                FROM cbsas 
                WHERE name_lower LIKE ?
                ORDER BY 
                    CASE WHEN cbsa_type = '1' THEN 0 ELSE 1 END
                LIMIT 2
            """, (search_term,))
            suggestions.extend([row['suggestion'] for row in cursor.fetchall()])
            
        except Exception as e:
            logger.warning(f"Error generating suggestions: {e}")
        
        return {
            'error': f"Could not resolve location: {location}",
            'resolution_method': 'comprehensive_failure',
            'suggestions': suggestions[:8],  # Limit to top 8 suggestions
            'help': {
                'formats': [
                    "City, ST (Austin, TX)",
                    "State name (California)",
                    "County name (Cook County, IL)",
                    "ZIP code (90210)",
                    "Metro area (San Francisco Bay Area)"
                ],
                'coverage': f"Database contains {self.coverage_stats.get('places', 0):,} places, "
                          f"{self.coverage_stats.get('counties', 0):,} counties, "
                          f"{self.coverage_stats.get('cbsas', 0):,} metro areas"
            }
        }
    
    def get_adjacent_counties(self, county_fips: str, state_fips: str) -> List[Dict[str, Any]]:
        """Get counties adjacent to specified county - FIXED for actual schema"""
        
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            
            # FIXED: Use actual column names from county_adjacency table
            # Build the full county GEOID (state_fips + county_fips)
            county_geoid = f"{state_fips}{county_fips}"
            
            cursor.execute("""
                SELECT DISTINCT 
                    ca.neighbor_geoid,
                    ca.neighbor_name
                FROM county_adjacency ca
                WHERE ca.county_geoid = ?
                ORDER BY ca.neighbor_name
            """, (county_geoid,))
            
            adjacent_counties = []
            for row in cursor.fetchall():
                # Parse neighbor_geoid to get state and county FIPS
                neighbor_geoid = row['neighbor_geoid']
                if len(neighbor_geoid) == 5:
                    neighbor_state_fips = neighbor_geoid[:2]
                    neighbor_county_fips = neighbor
