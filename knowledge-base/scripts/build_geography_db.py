#!/usr/bin/env python3
"""
Build optimized Census Gazetteer SQLite database from raw files.
Replaces hardcoded MAJOR_CITIES with comprehensive geographic intelligence.

Input: 44MB of raw Census gazetteer files
Output: Fast, indexed SQLite database for geographic resolution
"""

import sqlite3
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GazetteerProcessor:
    """Process Census gazetteer files into optimized SQLite database."""
    
    def __init__(self, source_dir: str, output_db: str):
        self.source_dir = Path(source_dir)
        self.output_db = Path(output_db)
        self.conn = None
        
        # Auto-detect files from your actual directory structure
        self.gazetteer_files = self._detect_gazetteer_files()
        
        logger.info(f"Gazetteer processor initialized")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_db}")
    
    def _detect_gazetteer_files(self) -> Dict[str, str]:
        """Auto-detect gazetteer files from your directory."""
        files = {}
        
        # Debug the path resolution
        absolute_path = self.source_dir.resolve()
        logger.info(f"Resolved absolute path: {absolute_path}")
        logger.info(f"Directory exists: {self.source_dir.exists()}")
        
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            logger.error(f"Absolute path: {absolute_path}")
            # List what IS in the parent directory
            parent = self.source_dir.parent
            if parent.exists():
                logger.error(f"Contents of parent directory {parent}:")
                for item in parent.iterdir():
                    logger.error(f"  {item.name}")
            return files
        
        all_files = list(self.source_dir.glob("*.txt"))
        logger.info(f"Found {len(all_files)} .txt files in {self.source_dir}")
        
        for file_path in all_files:
            filename = file_path.name
            filename_lower = filename.lower()
            logger.info(f"Checking file: {filename}")
            
            # Map based on your actual file naming patterns
            if '_place_' in filename_lower and 'national' in filename_lower:
                files['places'] = filename
            elif '_counties_' in filename_lower and 'national' in filename_lower:
                files['counties'] = filename
            elif '_cbsa_' in filename_lower and 'national' in filename_lower:
                files['cbsas'] = filename
            elif '_zcta_' in filename_lower and 'national' in filename_lower:
                files['zctas'] = filename
            elif 'adjacency' in filename_lower:
                files['adjacency'] = filename
        
        logger.info(f"Detected {len(files)} gazetteer files: {list(files.keys())}")
        return files
    
    def build_database(self):
        """Build complete geographic database from gazetteer files."""
        try:
            self._create_database()
            self._process_places()
            self._process_counties()
            self._process_cbsas()
            self._process_zctas()
            self._process_adjacency()  # Add spatial relationships
            self._create_indexes()
            self._create_hot_cache_view()
            self._validate_database()
            
            logger.info("✅ Geographic database built successfully")
            
        except Exception as e:
            logger.error(f"❌ Database build failed: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()
    
    def _create_database(self):
        """Create SQLite database with optimized schema."""
        if self.output_db.exists():
            self.output_db.unlink()
            logger.info("Removed existing database")
        
        self.conn = sqlite3.connect(self.output_db)
        
        # Places table (cities, towns, CDPs)
        self.conn.execute("""
            CREATE TABLE places (
                place_fips TEXT NOT NULL,
                state_fips TEXT NOT NULL,
                name TEXT NOT NULL,
                name_lower TEXT NOT NULL,
                state_abbrev TEXT NOT NULL,
                lat REAL,
                lon REAL,
                land_area REAL,
                water_area REAL,
                population INTEGER,
                PRIMARY KEY (place_fips, state_fips)
            )
        """)
        
        # Counties table
        self.conn.execute("""
            CREATE TABLE counties (
                county_fips TEXT NOT NULL,
                state_fips TEXT NOT NULL, 
                name TEXT NOT NULL,
                name_lower TEXT NOT NULL,
                state_abbrev TEXT NOT NULL,
                lat REAL,
                lon REAL,
                land_area REAL,
                water_area REAL,
                PRIMARY KEY (county_fips, state_fips)
            )
        """)
        
        # CBSAs (Metro/Micro areas)
        self.conn.execute("""
            CREATE TABLE cbsas (
                cbsa_code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_lower TEXT NOT NULL,
                cbsa_type TEXT,
                lat REAL,
                lon REAL,
                land_area REAL,
                water_area REAL
            )
        """)
        
        # ZIP Code Tabulation Areas
        self.conn.execute("""
            CREATE TABLE zctas (
                zcta_code TEXT PRIMARY KEY,
                lat REAL,
                lon REAL,
                land_area REAL,
                water_area REAL
            )
        """)
        
        # Common name variations for fuzzy matching
        self.conn.execute("""
            CREATE TABLE name_variations (
                canonical_name TEXT NOT NULL,
                variation TEXT NOT NULL,
                geography_type TEXT NOT NULL,
                place_fips TEXT,
                state_fips TEXT,
                PRIMARY KEY (variation, geography_type)
            )
        """)
        
        # County adjacency for spatial queries
        self.conn.execute("""
            CREATE TABLE county_adjacency (
                county_geoid TEXT NOT NULL,
                county_name TEXT NOT NULL,
                neighbor_geoid TEXT NOT NULL,
                neighbor_name TEXT NOT NULL,
                PRIMARY KEY (county_geoid, neighbor_geoid),
                FOREIGN KEY (county_geoid) REFERENCES counties(county_fips),
                FOREIGN KEY (neighbor_geoid) REFERENCES counties(county_fips)
            )
        """)
    
    def _process_places(self):
        """Process places gazetteer file with correct column parsing."""
        places_file = self.source_dir / self.gazetteer_files['places']
        
        if not places_file.exists():
            logger.warning(f"Places file not found: {places_file}")
            return
        
        logger.info(f"Processing places from {places_file}")
        
        # Build state abbreviation to FIPS mapping
        state_abbrev_to_fips = {
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
        
        count = 0
        with open(places_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # Skip header row
            
            for row in reader:
                if len(row) < 12:  # Places file has 12 columns
                    continue
                
                try:
                    # Correct Places file layout:
                    # Column 0: USPS (State abbreviation)
                    # Column 1: GEOID (State FIPS + Place FIPS)
                    # Column 2: ANSICODE
                    # Column 3: NAME
                    # Column 4: LSAD (Legal/Statistical Area Description)
                    # Column 5: FUNCSTAT (Functional Status)
                    # Column 6: ALAND
                    # Column 7: AWATER
                    # Column 8: ALAND_SQMI
                    # Column 9: AWATER_SQMI
                    # Column 10: INTPTLAT
                    # Column 11: INTPTLONG
                    
                    state_abbrev = row[0].strip()
                    geoid = row[1].strip()
                    name = row[3].strip()
                    funcstat = row[5].strip()
                    lat = float(row[10]) if row[10].strip() else None
                    lon = float(row[11]) if row[11].strip() else None
                    land_area = float(row[6]) if row[6].strip() else None
                    water_area = float(row[7]) if row[7].strip() else None
                    
                    # Skip non-active places (F = fictional, N = not defined)
                    if funcstat not in ['A', 'S']:  # A = Active, S = Statistical
                        continue
                    
                    # Get state FIPS from abbreviation
                    state_fips = state_abbrev_to_fips.get(state_abbrev)
                    if not state_fips:
                        continue
                    
                    # Extract place FIPS from GEOID
                    # GEOID format: SSFFFFFF (2-digit state + 5-digit place)
                    if len(geoid) >= 7:
                        place_fips = geoid[2:]  # Remove state part, keep place part
                    else:
                        continue
                    
                    name_lower = name.lower()
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO places 
                        (place_fips, state_fips, name, name_lower, state_abbrev, 
                         lat, lon, land_area, water_area)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (place_fips, state_fips, name, name_lower, state_abbrev,
                          lat, lon, land_area, water_area))
                    
                    # Add common name variations
                    self._add_name_variations(name, 'place', place_fips, state_fips)
                    
                    count += 1
                    if count % 1000 == 0:
                        logger.info(f"Processed {count} places...")
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipped malformed place row: {e}")
                    continue
        
        self.conn.commit()
        logger.info(f"✅ Processed {count} places")
        
    def _extract_state_mappings(self) -> Dict[str, str]:
        """Extract state FIPS to abbreviation mapping from gazetteer files."""
        # This method is no longer needed since we're using state abbreviations directly from the files
        # But keeping it for potential future use
        return {}
        with open(places_file, 'r', encoding='utf-8') as f:
            # Skip header if exists
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                if len(row) < 10:  # Skip malformed rows
                    continue
                
                try:
                    # Parse gazetteer format (adjust indexes based on actual file structure)
                    state_fips = row[0]  # USPS
                    place_fips = row[1]  # GEOID
                    name = row[3]        # NAME
                    lat = float(row[8]) if row[8] else None      # INTPTLAT
                    lon = float(row[9]) if row[9] else None      # INTPTLON
                    land_area = float(row[5]) if row[5] else None  # ALAND
                    water_area = float(row[6]) if row[6] else None # AWATER
                    
                    state_abbrev = state_fips_to_abbrev.get(state_fips, 'XX')
                    name_lower = name.lower()
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO places 
                        (place_fips, state_fips, name, name_lower, state_abbrev, 
                         lat, lon, land_area, water_area)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (place_fips, state_fips, name, name_lower, state_abbrev,
                          lat, lon, land_area, water_area))
                    
                    # Add common name variations
                    self._add_name_variations(name, 'place', place_fips, state_fips)
                    
                    count += 1
                    if count % 1000 == 0:
                        logger.info(f"Processed {count} places...")
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipped malformed place row: {e}")
                    continue
        
        self.conn.commit()
        logger.info(f"✅ Processed {count} places")
    
    def _process_counties(self):
        """Process counties gazetteer file using official column layout."""
        counties_file = self.source_dir / self.gazetteer_files.get('counties', '')
        
        if not counties_file.exists():
            logger.warning(f"Counties file not found: {counties_file}")
            return
        
        logger.info(f"Processing counties from {counties_file}")
        
        count = 0
        with open(counties_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                if len(row) < 10:  # Counties file has 10 columns per PDF
                    continue
                
                try:
                    # Official Counties file layout from PDF:
                    # Column 1: USPS (State abbreviation)
                    # Column 2: GEOID (State FIPS + County FIPS)
                    # Column 3: ANSICODE
                    # Column 4: NAME
                    # Column 5: ALAND
                    # Column 6: AWATER
                    # Column 7: ALAND_SQMI
                    # Column 8: AWATER_SQMI
                    # Column 9: INTPTLAT
                    # Column 10: INTPTLONG
                    
                    state_abbrev = row[0].strip()
                    geoid = row[1].strip()
                    name = row[3].strip()
                    lat = float(row[8]) if row[8].strip() else None
                    lon = float(row[9]) if row[9].strip() else None
                    land_area = float(row[4]) if row[4].strip() else None
                    water_area = float(row[5]) if row[5].strip() else None
                    
                    # Extract state and county FIPS from GEOID
                    if len(geoid) >= 5:  # State (2) + County (3) = 5 digits
                        state_fips = geoid[:2]
                        county_fips = geoid[2:]
                    else:
                        continue
                    
                    name_lower = name.lower()
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO counties
                        (county_fips, state_fips, name, name_lower, state_abbrev,
                         lat, lon, land_area, water_area)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (county_fips, state_fips, name, name_lower, state_abbrev,
                          lat, lon, land_area, water_area))
                    
                    # Add name variations
                    self._add_name_variations(name, 'county', county_fips, state_fips)
                    
                    count += 1
                    if count % 100 == 0:
                        logger.info(f"Processed {count} counties...")
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipped malformed county row: {e}")
                    continue
        
        self.conn.commit()
        logger.info(f"✅ Processed {count} counties")
    
    def _process_cbsas(self):
        """Process CBSAs gazetteer file using official column layout."""
        cbsas_file = self.source_dir / self.gazetteer_files.get('cbsas', '')
        
        if not cbsas_file.exists():
            logger.warning(f"CBSAs file not found: {cbsas_file}")
            return
        
        logger.info(f"Processing CBSAs from {cbsas_file}")
        
        count = 0
        with open(cbsas_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                if len(row) < 10:  # CBSAs file has 10 columns per PDF
                    continue
                
                try:
                    # Official CBSAs file layout from PDF:
                    # Column 1: CSAFP (Combined statistical area code, may be blank)
                    # Column 2: GEOID (Metro/Micro statistical area code)
                    # Column 3: NAME
                    # Column 4: CBSA_TYPE (Metropolitan/Micropolitan)
                    # Column 5: ALAND
                    # Column 6: AWATER
                    # Column 7: ALAND_SQMI
                    # Column 8: AWATER_SQMI
                    # Column 9: INTPTLAT
                    # Column 10: INTPTLONG
                    
                    cbsa_code = row[1].strip()  # GEOID
                    name = row[2].strip()
                    cbsa_type = row[3].strip()
                    lat = float(row[8]) if row[8].strip() else None
                    lon = float(row[9]) if row[9].strip() else None
                    land_area = float(row[4]) if row[4].strip() else None
                    water_area = float(row[5]) if row[5].strip() else None
                    
                    name_lower = name.lower()
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO cbsas
                        (cbsa_code, name, name_lower, cbsa_type,
                         lat, lon, land_area, water_area)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (cbsa_code, name, name_lower, cbsa_type,
                          lat, lon, land_area, water_area))
                    
                    # Add name variations for metro areas
                    self._add_name_variations(name, 'cbsa', cbsa_code, None)
                    
                    count += 1
                    if count % 100 == 0:
                        logger.info(f"Processed {count} CBSAs...")
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipped malformed CBSA row: {e}")
                    continue
        
        self.conn.commit()
        logger.info(f"✅ Processed {count} CBSAs")
    
    def _process_zctas(self):
        """Process ZIP Code Tabulation Areas using official column layout."""
        zctas_file = self.source_dir / self.gazetteer_files.get('zctas', '')
        
        if not zctas_file.exists():
            logger.info("ZCTAs file not found - skipping ZIP code support")
            return
        
        logger.info(f"Processing ZCTAs from {zctas_file}")
        
        count = 0
        with open(zctas_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                if len(row) < 7:  # ZCTAs file has 7 columns per PDF
                    continue
                
                try:
                    # Official ZCTAs file layout from PDF:
                    # Column 1: GEOID (Five digit ZIP Code)
                    # Column 2: ALAND
                    # Column 3: AWATER
                    # Column 4: ALAND_SQMI
                    # Column 5: AWATER_SQMI
                    # Column 6: INTPTLAT
                    # Column 7: INTPTLONG
                    
                    zcta_code = row[0].strip()
                    lat = float(row[5]) if row[5].strip() else None
                    lon = float(row[6]) if row[6].strip() else None
                    land_area = float(row[1]) if row[1].strip() else None
                    water_area = float(row[2]) if row[2].strip() else None
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO zctas
                        (zcta_code, lat, lon, land_area, water_area)
                        VALUES (?, ?, ?, ?, ?)
                    """, (zcta_code, lat, lon, land_area, water_area))
                    
                    count += 1
                    if count % 1000 == 0:
                        logger.info(f"Processed {count} ZCTAs...")
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipped malformed ZCTA row: {e}")
                    continue
        
        self.conn.commit()
        logger.info(f"✅ Processed {count} ZIP code areas")
    
    def _process_adjacency(self):
        """Process county adjacency file for spatial queries."""
        adjacency_file = self.source_dir / self.gazetteer_files.get('adjacency', '')
        
        if not adjacency_file.exists():
            logger.warning(f"Adjacency file not found: {adjacency_file}")
            return
        
        logger.info(f"Processing county adjacency from {adjacency_file}")
        
        count = 0
        with open(adjacency_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            
            # Skip header if it exists
            first_row = next(reader, None)
            if first_row and 'County Name' in first_row[0]:
                pass  # Header row, already consumed
            else:
                # No header, process this row
                if first_row and len(first_row) >= 4:
                    county_name = first_row[0].strip()
                    county_geoid = first_row[1].strip()
                    neighbor_name = first_row[2].strip()
                    neighbor_geoid = first_row[3].strip()
                    
                    # Skip self-references
                    if county_geoid != neighbor_geoid:
                        self.conn.execute("""
                            INSERT OR REPLACE INTO county_adjacency
                            (county_geoid, county_name, neighbor_geoid, neighbor_name)
                            VALUES (?, ?, ?, ?)
                        """, (county_geoid, county_name, neighbor_geoid, neighbor_name))
                        count += 1
            
            # Process remaining rows
            for row in reader:
                if len(row) < 4:
                    continue
                
                try:
                    county_name = row[0].strip()
                    county_geoid = row[1].strip()
                    neighbor_name = row[2].strip()
                    neighbor_geoid = row[3].strip()
                    
                    # Skip self-references (county adjacent to itself)
                    if county_geoid == neighbor_geoid:
                        continue
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO county_adjacency
                        (county_geoid, county_name, neighbor_geoid, neighbor_name)
                        VALUES (?, ?, ?, ?)
                    """, (county_geoid, county_name, neighbor_geoid, neighbor_name))
                    
                    count += 1
                    if count % 1000 == 0:
                        logger.info(f"Processed {count} adjacency relationships...")
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipped malformed adjacency row: {e}")
                    continue
        
        self.conn.commit()
        logger.info(f"✅ Processed {count} county adjacency relationships")
    
    def _add_name_variations(self, name: str, geo_type: str,
                           place_fips: str = None, state_fips: str = None):
        """Add common name variations for fuzzy matching."""
        name_lower = name.lower()
        variations = set()
        
        # Saint/St. variations
        if 'saint ' in name_lower:
            variations.add(name_lower.replace('saint ', 'st. '))
            variations.add(name_lower.replace('saint ', 'st '))
        elif 'st. ' in name_lower:
            variations.add(name_lower.replace('st. ', 'saint '))
            variations.add(name_lower.replace('st. ', 'st '))
        elif 'st ' in name_lower and not name_lower.endswith('st'):
            variations.add(name_lower.replace('st ', 'saint '))
            variations.add(name_lower.replace('st ', 'st. '))
        
        # Remove common suffixes for matching
        suffixes = ['city', 'town', 'village', 'borough']
        for suffix in suffixes:
            if name_lower.endswith(f' {suffix}'):
                variations.add(name_lower.replace(f' {suffix}', ''))
        
        # Add the original name
        variations.add(name_lower)
        
        # Insert variations
        for variation in variations:
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO name_variations
                    (canonical_name, variation, geography_type, place_fips, state_fips)
                    VALUES (?, ?, ?, ?, ?)
                """, (name_lower, variation, geo_type, place_fips, state_fips))
            except sqlite3.Error:
                pass  # Skip duplicates
    
    def _create_indexes(self):
        """Create performance indexes for fast lookups."""
        logger.info("Creating database indexes...")
        
        # Places indexes
        self.conn.execute("CREATE INDEX idx_places_name_state ON places(name_lower, state_abbrev)")
        self.conn.execute("CREATE INDEX idx_places_state ON places(state_abbrev)")
        self.conn.execute("CREATE INDEX idx_places_fips ON places(place_fips)")
        
        # Counties indexes
        self.conn.execute("CREATE INDEX idx_counties_name_state ON counties(name_lower, state_abbrev)")
        self.conn.execute("CREATE INDEX idx_counties_state ON counties(state_abbrev)")
        
        # CBSAs indexes
        self.conn.execute("CREATE INDEX idx_cbsas_name ON cbsas(name_lower)")
        
        # County adjacency indexes for spatial queries
        self.conn.execute("CREATE INDEX idx_adjacency_county ON county_adjacency(county_geoid)")
        self.conn.execute("CREATE INDEX idx_adjacency_neighbor ON county_adjacency(neighbor_geoid)")
        
        # Name variations index
        self.conn.execute("CREATE INDEX idx_variations_lookup ON name_variations(variation, geography_type)")
        
        self.conn.commit()
        logger.info("✅ Database indexes created")
    
    def _create_hot_cache_view(self):
        """Create view for hot cache based on actual data patterns."""
        # Use data-driven approach: top places by population/land area
        self.conn.execute("""
            CREATE VIEW hot_cache_places AS
            SELECT 
                name_lower || ', ' || state_abbrev as lookup_key,
                place_fips,
                state_fips, 
                name,
                state_abbrev,
                lat,
                lon,
                population
            FROM places 
            WHERE population IS NOT NULL
            ORDER BY population DESC
            LIMIT 500
        """)
        
        logger.info("✅ Hot cache view created (top 500 by population)")
    
    def _validate_database(self):
        """Validate database contents and performance."""
        cursor = self.conn.cursor()
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM places")
        places_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM counties")
        counties_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM cbsas")
        cbsas_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM name_variations")
        variations_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM county_adjacency")
        adjacency_count = cursor.fetchone()[0]
        
        logger.info(f"Database validation:")
        logger.info(f"  Places: {places_count:,}")
        logger.info(f"  Counties: {counties_count:,}")
        logger.info(f"  CBSAs: {cbsas_count:,}")
        logger.info(f"  County adjacencies: {adjacency_count:,}")
        logger.info(f"  Name variations: {variations_count:,}")
        
        # Test with actual data - use top 5 places by population
        cursor.execute("""
            SELECT name_lower, state_abbrev, place_fips, name
            FROM places 
            WHERE population IS NOT NULL
            ORDER BY population DESC 
            LIMIT 5
        """)
        
        logger.info("Testing lookups with actual top cities:")
        for row in cursor.fetchall():
            name_lower, state_abbrev, place_fips, name = row
            logger.info(f"  ✅ {name_lower}, {state_abbrev} -> FIPS {place_fips} ({name})")
        
        # Test lookup performance
        import time
        cursor.execute("SELECT name_lower, state_abbrev FROM places LIMIT 10")
        test_cities = cursor.fetchall()
        
        start_time = time.time()
        for city, state in test_cities:
            cursor.execute("""
                SELECT place_fips FROM places 
                WHERE name_lower = ? AND state_abbrev = ?
            """, (city, state))
            cursor.fetchone()
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Lookup performance: {elapsed:.1f}ms for 10 queries ({elapsed/10:.1f}ms avg)")
        
        # Database size
        db_size = self.output_db.stat().st_size / (1024 * 1024)
        logger.info(f"Database size: {db_size:.1f} MB")

def main():
    """Build the geographic database."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Census Gazetteer Database")
    parser.add_argument("--source", required=True,
                       help="Directory containing gazetteer files")
    parser.add_argument("--output", required=True,
                       help="Output SQLite database file")
    
    args = parser.parse_args()
    
    processor = GazetteerProcessor(args.source, args.output)
    processor.build_database()
    
    print(f"✅ Geographic database built: {args.output}")
    print("Next step: Update python_census_api.py to use this database")

if __name__ == "__main__":
    main()
