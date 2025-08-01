#!/usr/bin/env python3
"""
Build official geographic reference tables from Census sources.
Creates clean CSV files for hierarchical database building.

Input: source-docs/geographic-reference/*.txt
Output: knowledge-base/geo-reference-data/*.csv
"""

import csv
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_states_table(source_dir: Path, output_path: Path):
    """Build states.csv from national_state2020.txt + region mappings."""
    
    # Official Census Region/Division mappings from hierarchy docs
    region_mappings = {
        '01': ('3', '6'),  # Alabama -> South, East South Central
        '02': ('4', '9'),  # Alaska -> West, Pacific
        '04': ('4', '8'),  # Arizona -> West, Mountain
        '05': ('3', '7'),  # Arkansas -> South, West South Central
        '06': ('4', '9'),  # California -> West, Pacific
        '08': ('4', '8'),  # Colorado -> West, Mountain
        '09': ('1', '1'),  # Connecticut -> Northeast, New England
        '10': ('3', '5'),  # Delaware -> South, South Atlantic
        '11': ('3', '5'),  # DC -> South, South Atlantic
        '12': ('3', '5'),  # Florida -> South, South Atlantic
        '13': ('3', '5'),  # Georgia -> South, South Atlantic
        '15': ('4', '9'),  # Hawaii -> West, Pacific
        '16': ('4', '8'),  # Idaho -> West, Mountain
        '17': ('2', '3'),  # Illinois -> Midwest, East North Central
        '18': ('2', '3'),  # Indiana -> Midwest, East North Central
        '19': ('2', '4'),  # Iowa -> Midwest, West North Central
        '20': ('2', '4'),  # Kansas -> Midwest, West North Central
        '21': ('3', '6'),  # Kentucky -> South, East South Central
        '22': ('3', '7'),  # Louisiana -> South, West South Central
        '23': ('1', '1'),  # Maine -> Northeast, New England
        '24': ('3', '5'),  # Maryland -> South, South Atlantic
        '25': ('1', '1'),  # Massachusetts -> Northeast, New England
        '26': ('2', '3'),  # Michigan -> Midwest, East North Central
        '27': ('2', '4'),  # Minnesota -> Midwest, West North Central
        '28': ('3', '6'),  # Mississippi -> South, East South Central
        '29': ('2', '4'),  # Missouri -> Midwest, West North Central
        '30': ('4', '8'),  # Montana -> West, Mountain
        '31': ('2', '4'),  # Nebraska -> Midwest, West North Central
        '32': ('4', '8'),  # Nevada -> West, Mountain
        '33': ('1', '1'),  # New Hampshire -> Northeast, New England
        '34': ('1', '2'),  # New Jersey -> Northeast, Middle Atlantic
        '35': ('4', '8'),  # New Mexico -> West, Mountain
        '36': ('1', '2'),  # New York -> Northeast, Middle Atlantic
        '37': ('3', '5'),  # North Carolina -> South, South Atlantic
        '38': ('2', '4'),  # North Dakota -> Midwest, West North Central
        '39': ('2', '3'),  # Ohio -> Midwest, East North Central
        '40': ('3', '7'),  # Oklahoma -> South, West South Central
        '41': ('4', '9'),  # Oregon -> West, Pacific
        '42': ('1', '2'),  # Pennsylvania -> Northeast, Middle Atlantic
        '44': ('1', '1'),  # Rhode Island -> Northeast, New England
        '45': ('3', '5'),  # South Carolina -> South, South Atlantic
        '46': ('2', '4'),  # South Dakota -> Midwest, West North Central
        '47': ('3', '6'),  # Tennessee -> South, East South Central
        '48': ('3', '7'),  # Texas -> South, West South Central
        '49': ('4', '8'),  # Utah -> West, Mountain
        '50': ('1', '1'),  # Vermont -> Northeast, New England
        '51': ('3', '5'),  # Virginia -> South, South Atlantic
        '53': ('4', '9'),  # Washington -> West, Pacific
        '54': ('3', '5'),  # West Virginia -> South, South Atlantic
        '55': ('2', '3'),  # Wisconsin -> Midwest, East North Central
        '56': ('4', '8'),  # Wyoming -> West, Mountain
        # Territories (assign to closest region)
        '60': ('4', '9'),  # American Samoa -> West, Pacific
        '66': ('4', '9'),  # Guam -> West, Pacific  
        '69': ('4', '9'),  # Northern Mariana Islands -> West, Pacific
        '72': ('3', '5'),  # Puerto Rico -> South, South Atlantic
        '74': ('4', '9'),  # Minor Outlying Islands -> West, Pacific
        '78': ('3', '5'),  # Virgin Islands -> South, South Atlantic
    }
    
    # Read from national_state2020.txt
    state_file = source_dir / "national_state2020.txt"
    if not state_file.exists():
        raise FileNotFoundError(f"State file not found: {state_file}")
    
    logger.info(f"Reading states from {state_file}")
    
    with open(state_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='|')
        states_data = []
        
        for row in reader:
            state_fips = row['STATEFP']
            state_abbrev = row['STATE']
            state_name = row['STATE_NAME']
            
            # Get region/division from mapping
            region_code, division_code = region_mappings.get(state_fips, ('9', '9'))
            
            states_data.append([
                state_fips, state_abbrev, state_name, region_code, division_code
            ])
    
    # Write states.csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['state_fips', 'state_abbrev', 'state_name', 'region_code', 'division_code'])
        writer.writerows(states_data)
    
    logger.info(f"✅ Created {output_path} with {len(states_data)} states")


def build_regions_divisions_table(output_path: Path):
    """Build regions-divisions.csv from official Census hierarchy."""
    
    regions_data = [
        # region_code, region_name, division_code, division_name
        ('1', 'Northeast', '1', 'New England'),
        ('1', 'Northeast', '2', 'Middle Atlantic'),
        ('2', 'Midwest', '3', 'East North Central'),
        ('2', 'Midwest', '4', 'West North Central'),
        ('3', 'South', '5', 'South Atlantic'),
        ('3', 'South', '6', 'East South Central'), 
        ('3', 'South', '7', 'West South Central'),
        ('4', 'West', '8', 'Mountain'),
        ('4', 'West', '9', 'Pacific'),
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['region_code', 'region_name', 'division_code', 'division_name'])
        writer.writerows(regions_data)
    
    logger.info(f"✅ Created {output_path} with {len(regions_data)} region/division mappings")


def build_summary_levels_table(output_path: Path):
    """Build summary-levels.csv with official Census geography codes."""
    
    # Official Census Summary Level codes from hierarchy docs
    summary_levels = [
        # summary_level, geography_type, description
        ('010', 'nation', 'United States'),
        ('020', 'region', 'Census Region'),
        ('030', 'division', 'Census Division'),
        ('040', 'state', 'State'),
        ('050', 'county', 'County'),
        ('060', 'county_subdivision', 'County Subdivision'),
        ('140', 'census_tract', 'Census Tract'),
        ('150', 'block_group', 'Block Group'), 
        ('160', 'place', 'Incorporated Place'),
        ('170', 'consolidated_city', 'Consolidated City'),
        ('230', 'aiannh', 'American Indian Area/Alaska Native Area/Hawaiian Home Land'),
        ('250', 'aiannh_onsv', 'American Indian Area/Alaska Native Area (Reservation or Statistical Entity Only)'),
        ('310', 'cbsa', 'Core Based Statistical Area'),
        ('330', 'csa', 'Combined Statistical Area'),
        ('350', 'necta', 'New England City and Town Area'),
        ('400', 'ua', 'Urban Area'),
        ('500', 'congressional_district', 'Congressional District'),
        ('610', 'state_senate', 'State Legislative District (Upper Chamber)'),
        ('620', 'state_house', 'State Legislative District (Lower Chamber)'),
        ('860', 'zcta', 'ZIP Code Tabulation Area'),
        ('950', 'school_district_elementary', 'School District (Elementary)'),
        ('960', 'school_district_secondary', 'School District (Secondary)'),
        ('970', 'school_district_unified', 'School District (Unified)'),
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['summary_level', 'geography_type', 'description'])
        writer.writerows(summary_levels)
    
    logger.info(f"✅ Created {output_path} with {len(summary_levels)} summary levels")


def build_aiannh_reference_table(source_dir: Path, output_path: Path):
    """Build aiannh-areas.csv from national tribal area files."""
    
    # Read national_aiannh2020.txt
    aiannh_file = source_dir / "national_aiannh2020.txt"
    if not aiannh_file.exists():
        raise FileNotFoundError(f"AIANNH file not found: {aiannh_file}")
    
    logger.info(f"Reading AIANNH areas from {aiannh_file}")
    
    with open(aiannh_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='|')
        aiannh_data = []
        
        for row in reader:
            aiannh_ce = row['AIANNHCE']
            aiannh_ns = row['AIANNHNS']  
            aiannh_name = row['AIANNHNAME']
            states = row['STATES']
            
            # Determine area type from code pattern
            if aiannh_ce.startswith('0'):
                area_type = 'reservation'
            elif aiannh_ce.startswith('5'):
                area_type = 'hawaiian_home_land'
            elif aiannh_ce.startswith('6') or aiannh_ce.startswith('7'):
                area_type = 'alaska_native_village'
            elif aiannh_ce.startswith('8'):
                area_type = 'tribal_designated_statistical_area'
            elif aiannh_ce.startswith('9'):
                area_type = 'state_designated_tribal_statistical_area'
            else:
                area_type = 'other'
            
            aiannh_data.append([
                aiannh_ce, aiannh_ns, aiannh_name, area_type, states
            ])
    
    # Write aiannh-areas.csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['aiannh_ce', 'aiannh_ns', 'aiannh_name', 'area_type', 'states'])
        writer.writerows(aiannh_data)
    
    logger.info(f"✅ Created {output_path} with {len(aiannh_data)} tribal areas")


def main():
    """Build all geographic reference tables."""
    
    # Paths relative to knowledge-base/scripts/ location
    script_dir = Path(__file__).parent
    kb_dir = script_dir.parent
    
    source_dir = kb_dir / "source-docs/geographic-reference"
    output_dir = kb_dir / "geo-reference-data"
    
    # Validate source directory exists
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Build reference tables
    build_states_table(source_dir, output_dir / "states.csv")
    build_regions_divisions_table(output_dir / "regions-divisions.csv")
    build_summary_levels_table(output_dir / "summary-levels.csv")
    build_aiannh_reference_table(source_dir, output_dir / "aiannh-areas.csv")
    
    logger.info("🎉 All geographic reference tables built successfully!")
    logger.info(f"Reference data: {output_dir}")
    logger.info("Ready for hierarchical database build.")


if __name__ == "__main__":
    main()
