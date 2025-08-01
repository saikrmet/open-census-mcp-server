#!/usr/bin/env python3
"""
Enhanced Census Gazetteer Database Builder with Reference Tables Integration.
Builds proper hierarchical schema: Nation -> Regions -> Divisions -> States -> Counties -> Places

Input: 
- Raw gazetteer files (places, counties, CBSAs, etc.)
- Clean reference tables (states.csv, regions-divisions.csv, etc.)
Output: Complete hierarchical geographic database
"""

import sqlite3
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalGazetteerProcessor:
    """Enhanced processor with proper hierarchical schema and reference tables."""
    
    def __init__(self, source_dir: str, reference_dir: str, output_db: str):
        self.source_dir = Path(source_dir)
        self.reference_dir = Path(reference_dir)
        self.output_db = Path(output_db)
        self.conn = None
        
        # Auto-detect gazetteer files
        self.gazetteer_files = self._detect_gazetteer_files()
        
        logger.info(f"Enhanced Gazetteer processor initialized")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Reference: {self.reference_dir}")
        logger.info(f"Output: {self.output_db}")
    
    def _detect_gazetteer_files(self) -> Dict[str, str]:
        """Auto-detect gazetteer files from directory."""
        files = {}
        
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return files
        
        all_files = list(self.source_dir.glob("*.txt"))
        logger.info(f"Found {len(all_files)} .txt files in {self.source_dir}")
        
        for file_path in all_files:
            filename = file_path.name
            filename_lower = filename.lower()
            
            # Map based on file naming patterns
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
        
        logger.info(f"Detected gazetteer files: {list(files.keys())}")
        return files
    
    def build_database(self):
        """Build complete hierarchical geographic database."""
        try:
            self._create_hierarchical_schema()
            self._load_reference_tables()
            self._process_counties()
            self._process_places()
            self._process_cbsas()
            self._process_zctas()
            self._process_adjacency()
            self._create_indexes()
            self._validate_hierarchy()
            
            logger.info("✅ Hierarchical geographic database built successfully")
            
        except Exception as e:
            logger.error(f"❌ Database build failed: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()
    
    def _create_hierarchical_schema(self):
        """Create proper hierarchical schema with foreign keys."""
        if self.output_db.exists():
            self.output_db.unlink()
            logger.info("Removed existing database")
        
        self.conn = sqlite3.connect(self.output_db)
        
        # Regions table (top level)
        self.conn.execute("""
            CREATE TABLE regions (
                region_code TEXT PRIMARY KEY,
                region_name TEXT NOT NULL
            )
        """)
        
        # Divisions table (under regions)
        self.conn.execute("""
            CREATE TABLE divisions (
                division_code TEXT PRIMARY KEY,
                division_name TEXT NOT NULL,
                region_code TEXT NOT NULL,
                FOREIGN KEY (region_code) REFERENCES regions(region_code)
            )
        """)
        
        # States table (under divisions) - THE MISSING PIECE
        self.conn.execute("""
            CREATE TABLE states (
                state_fips TEXT PRIMARY KEY,
                state_abbrev TEXT UNIQUE NOT NULL,
                state_name TEXT NOT NULL,
                region_code TEXT NOT NULL,
                division_code TEXT NOT NULL,
                FOREIGN KEY (region_code) REFERENCES regions(region_code),
                FOREIGN KEY (division_code) REFERENCES divisions(division_code)
            )
        """)
        
        # Counties table (under states) - ENHANCED with FK
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
                PRIMARY KEY (county_fips, state_fips),
                FOREIGN KEY (state_fips) REFERENCES states(state_fips)
            )
        """)
        
        # Places table (under states/counties) - ENHANCED with FK
        self.conn.execute("""
            CREATE TABLE places (
                place_fips TEXT NOT NULL,
                state_fips TEXT NOT NULL,
                county_fips TEXT,  -- NEW: Link to county (nullable for cross-county places)
                name TEXT NOT NULL,
                name_lower TEXT NOT NULL,
                state_abbrev TEXT NOT NULL,
                lat REAL,
                lon REAL,
                land_area REAL,
                water_area REAL,
                population INTEGER,
                PRIMARY KEY (place_fips, state_fips),
                FOREIGN KEY (state_fips) REFERENCES states(state_fips),
                FOREIGN KEY (county_fips, state_fips) REFERENCES counties(county_fips, state_fips)
            )
        """)
        
        # CBSAs (cross-boundary metro areas)
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
        
        # ZIP Code Tabulation Areas (cross-boundary)
        self.conn.execute("""
            CREATE TABLE zctas (
                zcta_code TEXT PRIMARY KEY,
                lat REAL,
                lon REAL,
                land_area REAL,
                water_area REAL
            )
        """)
        
        # Name variations for fuzzy matching
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
        
        logger.info("✅ Hierarchical schema created with proper foreign keys")
    
    def _load_reference_tables(self):
        """Load clean reference data from CSV files."""
        logger.info("Loading reference tables...")
        
        # Load regions
        regions_file = self.reference_dir / "regions-divisions.csv"
        if regions_file.exists():
            with open(regions_file, 'r') as f:
                reader = csv.DictReader(f)
                regions = {}
                for row in reader:
                    region_code = row['region_code']
                    region_name = row['region_name']
                    division_code = row['division_code']
                    division_name = row['division_name']
                    
                    # Insert region (deduplicated)
                    if region_code not in regions:
                        self.conn.execute("""
                            INSERT OR REPLACE INTO regions (region_code, region_name)
                            VALUES (?, ?)
                        """, (region_code, region_name))
                        regions[region_code] = True
                    
                    # Insert division
                    self.conn.execute("""
                        INSERT OR REPLACE INTO divisions 
                        (division_code, division_name, region_code)
                        VALUES (?, ?, ?)
                    """, (division_code, division_name, region_code))
            
            logger.info("✅ Loaded regions and divisions")
        
        # Load states with hierarchy
        states_file = self.reference_dir / "states.csv"
        if states_file.exists():
            with open(states_file, 'r') as f:
                reader = csv.DictReader(f)
                states_count = 0
                for row in reader:
                    self.conn.execute("""
                        INSERT OR REPLACE INTO states 
                        (state_fips, state_abbrev, state_name, region_code, division_code)
                        VALUES (?, ?, ?, ?, ?)
                    """, (row['state_fips'], row['state_abbrev'], row['state_name'],
                          row['region_code'], row['division_code']))
                    states_count += 1
            
            logger.info(f"✅ Loaded {states_count} states with hierarchy")
        
        self.conn.commit()
    
    def _process_counties(self):
        """Process counties with proper foreign key relationships."""
        counties_file = self.source_dir / self.gazetteer_files.get('counties', '')
        
        if not counties_file.exists():
            logger.warning(f"Counties file not found: {counties_file}")
            return
        
        logger.info(f"Processing counties with hierarchy from {counties_file}")
        
        count = 0
        with open(counties_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                if len(row) < 10:
                    continue
                
                try:
                    state_abbrev = row[0].strip()
                    geoid = row[1].strip()
                    name = row[3].strip()
                    lat = float(row[8]) if row[8].strip() else None
                    lon = float(row[9]) if row[9].strip() else None
                    land_area = float(row[4]) if row[4].strip() else None
                    water_area = float(row[5]) if row[5].strip() else None
                    
                    # Extract state and county FIPS from GEOID
                    if len(geoid) >= 5:
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
        logger.info(f"✅ Processed {count} counties with state foreign keys")
    
    def _process_places(self):
        """Process places with enhanced hierarchy and county linking."""
        places_file = self.source_dir / self.gazetteer_files['places']
        
        if not places_file.exists():
            logger.warning(f"Places file not found: {places_file}")
            return
        
        logger.info(f"Processing places with hierarchy from {places_file}")
        
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
                if len(row) < 12:
                    continue
                
                try:
                    state_abbrev = row[0].strip()
                    geoid = row[1].strip()
                    name = row[3].strip()
                    funcstat = row[5].strip()
                    lat = float(row[10]) if row[10].strip() else None
                    lon = float(row[11]) if row[11].strip() else None
                    land_area = float(row[6]) if row[6].strip() else None
                    water_area = float(row[7]) if row[7].strip() else None
                    
                    # Skip non-active places
                    if funcstat not in ['A', 'S']:
                        continue
                    
                    # Get state FIPS from abbreviation
                    state_fips = state_abbrev_to_fips.get(state_abbrev)
                    if not state_fips:
                        continue
                    
                    # Extract place FIPS from GEOID
                    if len(geoid) >= 7:
                        place_fips = geoid[2:]  # Remove state part
                    else:
                        continue
                    
                    name_lower = name.lower()
                    
                    # TODO: Add county linking logic here for places
                    # For now, leaving county_fips as NULL
                    county_fips = None
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO places 
                        (place_fips, state_fips, county_fips, name, name_lower, state_abbrev, 
                         lat, lon, land_area, water_area)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (place_fips, state_fips, county_fips, name, name_lower, state_abbrev,
                          lat, lon, land_area, water_area))
                    
                    # Add name variations
                    self._add_name_variations(name, 'place', place_fips, state_fips)
                    
                    count += 1
                    if count % 1000 == 0:
                        logger.info(f"Processed {count} places...")
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipped malformed place row: {e}")
                    continue
        
        self.conn.commit()
        logger.info(f"✅ Processed {count} places with state foreign keys")
    
    def _process_cbsas(self):
        """Process CBSAs (unchanged from original)."""
        cbsas_file = self.source_dir / self.gazetteer_files.get('cbsas', '')
        
        if not cbsas_file.exists():
            logger.warning(f"CBSAs file not found: {cbsas_file}")
            return
        
        logger.info(f"Processing CBSAs from {cbsas_file}")
        
        count = 0
        with open(cbsas_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                if len(row) < 10:
                    continue
                
                try:
                    cbsa_code = row[1].strip()
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
        """Process ZCTAs (unchanged from original)."""
        zctas_file = self.source_dir / self.gazetteer_files.get('zctas', '')
        
        if not zctas_file.exists():
            logger.info("ZCTAs file not found - skipping ZIP code support")
            return
        
        logger.info(f"Processing ZCTAs from {zctas_file}")
        
        count = 0
        with open(zctas_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                if len(row) < 7:
                    continue
                
                try:
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
        """Process county adjacency (unchanged from original)."""
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
        """Add common name variations for fuzzy matching (unchanged from original)."""
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
        """Create performance indexes including hierarchy indexes."""
        logger.info("Creating hierarchical database indexes...")
        
        # Hierarchy indexes
        self.conn.execute("CREATE INDEX idx_states_region ON states(region_code)")
        self.conn.execute("CREATE INDEX idx_states_division ON states(division_code)")
        self.conn.execute("CREATE INDEX idx_states_abbrev ON states(state_abbrev)")
        
        # Places indexes
        self.conn.execute("CREATE INDEX idx_places_name_state ON places(name_lower, state_abbrev)")
        self.conn.execute("CREATE INDEX idx_places_state ON places(state_abbrev)")
        self.conn.execute("CREATE INDEX idx_places_fips ON places(place_fips)")
        
        # Counties indexes
        self.conn.execute("CREATE INDEX idx_counties_name_state ON counties(name_lower, state_abbrev)")
        self.conn.execute("CREATE INDEX idx_counties_state ON counties(state_abbrev)")
        
        # CBSAs indexes
        self.conn.execute("CREATE INDEX idx_cbsas_name ON cbsas(name_lower)")
        
        # County adjacency indexes
        self.conn.execute("CREATE INDEX idx_adjacency_county ON county_adjacency(county_geoid)")
        self.conn.execute("CREATE INDEX idx_adjacency_neighbor ON county_adjacency(neighbor_geoid)")
        
        # Name variations index
        self.conn.execute("CREATE INDEX idx_variations_lookup ON name_variations(variation, geography_type)")
        
        self.conn.commit()
        logger.info("✅ Hierarchical database indexes created")
    
    def _validate_hierarchy(self):
        """Validate hierarchical database structure and foreign key relationships."""
        cursor = self.conn.cursor()
        
        # Count records by level
        cursor.execute("SELECT COUNT(*) FROM regions")
        regions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM divisions")
        divisions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM states")
        states_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM counties")
        counties_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM places")
        places_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM cbsas")
        cbsas_count = cursor.fetchone()[0]
        
        logger.info(f"Hierarchical database validation:")
        logger.info(f"  Regions: {regions_count}")
        logger.info(f"  Divisions: {divisions_count}")
        logger.info(f"  States: {states_count}")
        logger.info(f"  Counties: {counties_count:,}")
        logger.info(f"  Places: {places_count:,}")
        logger.info(f"  CBSAs: {cbsas_count:,}")
        
        # Test hierarchical queries
        logger.info("Testing hierarchical queries:")
        
        # Test: "Maryland" state lookup
        cursor.execute("""
            SELECT state_name, region_code, division_code 
            FROM states 
            WHERE state_abbrev = 'MD' OR state_name = 'Maryland'
        """)
        md_result = cursor.fetchone()
        if md_result:
            logger.info(f"  ✅ Maryland -> {md_result[0]}, Region {md_result[1]}, Division {md_result[2]}")
        
        # Test: "Counties in Maryland"
        cursor.execute("""
            SELECT COUNT(*) 
            FROM counties c
            JOIN states s ON c.state_fips = s.state_fips
            WHERE s.state_abbrev = 'MD'
        """)
        md_counties = cursor.fetchone()[0]
        logger.info(f"  ✅ Counties in Maryland: {md_counties}")
        
        # Test: Sample city hierarchical lookup
        cursor.execute("""
            SELECT p.name, p.state_abbrev, s.state_name, s.region_code
            FROM places p
            JOIN states s ON p.state_fips = s.state_fips
            WHERE p.name_lower = 'baltimore' AND p.state_abbrev = 'MD'
        """)
        baltimore_result = cursor.fetchone()
        if baltimore_result:
            logger.info(f"  ✅ Baltimore hierarchical lookup: {baltimore_result}")
        
        # Database size
        db_size = self.output_db.stat().st_size / (1024 * 1024)
        logger.info(f"Database size: {db_size:.1f} MB")
        
        logger.info("🎯 Hierarchical database validation complete!")


def main():
    """Build the enhanced hierarchical geographic database."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Enhanced Hierarchical Census Database")
    parser.add_argument("--source", required=True,
                       help="Directory containing gazetteer files")
    parser.add_argument("--reference", required=True,
                       help="Directory containing reference CSV files")
    parser.add_argument("--output", required=True,
                       help="Output SQLite database file")
    
    args = parser.parse_args()
    
    processor = HierarchicalGazetteerProcessor(args.source, args.reference, args.output)
    processor.build_database()
    
    print(f"✅ Enhanced hierarchical geographic database built: {args.output}")
    print("\nWhat this enables:")
    print("  • 'Maryland' -> Resolves via states table")
    print("  • 'Counties in Maryland' -> Works via foreign key joins")
    print("  • 'Baltimore, MD' -> Full hierarchical context")
    print("  • Regional analysis -> Nations → Regions → Divisions → States")
    print("\nNext step: Update python_census_api.py to use this hierarchical database")


if __name__ == "__main__":
    main()
