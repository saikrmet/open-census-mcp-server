"""
Census Data Mappings - Single Source of Truth

Consolidates all hardcoded values from python_census_api.py into one file.
These can eventually be replaced with dynamic API calls, but static mappings
provide better performance and reliability for common queries.

TODO: Add API endpoints to fetch these dynamically and cache them.
"""

# =============================================================================
# GEOGRAPHY MAPPINGS
# =============================================================================

# State FIPS codes - official Census Bureau codes
STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13',
    'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19',
    'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
    'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
    'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
    'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
    'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
    'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50',
    'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56',
    'DC': '11'
}

# State abbreviations to full names
STATE_ABBREVS = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}

# Reverse mapping: full names to abbreviations
STATE_NAMES = {v.lower(): k for k, v in STATE_ABBREVS.items()}

# Major cities with known FIPS codes (fixes Baltimore bug and provides fast lookup)
# TODO: These can be fetched dynamically from Census API and cached
MAJOR_CITIES = {
    'new york': {'state': 'NY', 'place_fips': '51000', 'full_name': 'New York city'},
    'los angeles': {'state': 'CA', 'place_fips': '44000', 'full_name': 'Los Angeles city'},
    'chicago': {'state': 'IL', 'place_fips': '14000', 'full_name': 'Chicago city'},
    'houston': {'state': 'TX', 'place_fips': '35000', 'full_name': 'Houston city'},
    'philadelphia': {'state': 'PA', 'place_fips': '60000', 'full_name': 'Philadelphia city'},
    'phoenix': {'state': 'AZ', 'place_fips': '55000', 'full_name': 'Phoenix city'},
    'san antonio': {'state': 'TX', 'place_fips': '65000', 'full_name': 'San Antonio city'},
    'san diego': {'state': 'CA', 'place_fips': '66000', 'full_name': 'San Diego city'},
    'dallas': {'state': 'TX', 'place_fips': '19000', 'full_name': 'Dallas city'},
    'san jose': {'state': 'CA', 'place_fips': '68000', 'full_name': 'San Jose city'},
    'baltimore': {'state': 'MD', 'place_fips': '04000', 'full_name': 'Baltimore city'},  # Note: Need to verify with actual API test
    'detroit': {'state': 'MI', 'place_fips': '22000', 'full_name': 'Detroit city'},
    'austin': {'state': 'TX', 'place_fips': '05000', 'full_name': 'Austin city'},
    'denver': {'state': 'CO', 'place_fips': '20000', 'full_name': 'Denver city'},
    'seattle': {'state': 'WA', 'place_fips': '63000', 'full_name': 'Seattle city'},
}

# =============================================================================
# VARIABLE MAPPINGS
# =============================================================================

# Core demographic variables - human-readable to Census codes
VARIABLE_MAPPINGS = {
    # Population
    'population': 'B01003_001E',
    'total population': 'B01003_001E',
    'pop': 'B01003_001E',
    
    # Income
    'median income': 'B19013_001E',
    'median household income': 'B19013_001E',
    'household income': 'B19013_001E',
    'income': 'B19013_001E',
    'median family income': 'B19113_001E',
    'per capita income': 'B19301_001E',
    
    # Housing
    'median home value': 'B25077_001E',
    'home value': 'B25077_001E',
    'median house value': 'B25077_001E',
    'housing units': 'B25001_001E',
    'total housing units': 'B25001_001E',
    'renter occupied': 'B25003_003E',
    'owner occupied': 'B25003_002E',
    'median rent': 'B25064_001E',
    'gross rent': 'B25064_001E',
    
    # Transportation variables
    'commute time': 'B08303_001E',
    'travel time to work': 'B08303_001E',
    'mean travel time to work': 'B08303_001E',
    'average commute time': 'B08303_001E',
    'means of transportation to work': 'B08301_001E',
    'public transportation': 'B08301_010E',
    'drove alone': 'B08301_003E',
    'carpooled': 'B08301_004E',
    
    # Demographics
    'median age': 'B01002_001E',
    'age': 'B01002_001E',
    
    # Education
    'high school graduation rate': 'B15003_017E',  # High school graduate (includes equivalency)
    'college degree': 'B15003_022E',  # Bachelor's degree
    'bachelor degree': 'B15003_022E',
    
    # Race/Ethnicity
    'white alone': 'B02001_002E',
    'black alone': 'B02001_003E',
    'asian alone': 'B02001_005E',
    'hispanic': 'B03003_003E',
    'latino': 'B03003_003E',
    
    # Employment
    'unemployment': 'B23025_005E',  # Unemployed
    'labor force': 'B23025_002E',   # Labor force
    'employed': 'B23025_004E',      # Employed
}

# =============================================================================
# RATE CALCULATIONS
# =============================================================================

# Calculated rates requiring numerator/denominator
RATE_CALCULATIONS = {
    'poverty_rate': {
        'numerator': 'B17001_002E',  # Below poverty level
        'denominator': 'B17001_001E',  # Total population for whom poverty determined
        'description': 'Percentage of population below poverty level',
        'unit': 'percentage'
    },
    'unemployment_rate': {
        'numerator': 'B23025_005E',  # Unemployed
        'denominator': 'B23025_002E',  # Labor force
        'description': 'Percentage of labor force unemployed',
        'unit': 'percentage'
    },
    'renter_occupied_rate': {
        'numerator': 'B25003_003E',  # Renter occupied
        'denominator': 'B25003_001E',  # Total occupied housing units
        'description': 'Percentage of occupied housing units that are renter-occupied',
        'unit': 'percentage'
    },
    'homeownership_rate': {
        'numerator': 'B25003_002E',  # Owner occupied
        'denominator': 'B25003_001E',  # Total occupied housing units
        'description': 'Percentage of occupied housing units that are owner-occupied',
        'unit': 'percentage'
    },
    'college_graduation_rate': {
        'numerator': 'B15003_022E',  # Bachelor's degree
        'denominator': 'B15003_001E',  # Total population 25 years and over
        'description': 'Percentage of population 25+ with bachelor\'s degree or higher',
        'unit': 'percentage'
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_state_fips(state_abbrev: str) -> str:
    """Get FIPS code for state abbreviation."""
    return STATE_FIPS.get(state_abbrev.upper(), '01')  # Default to Alabama if not found

def normalize_state(state_input: str) -> str:
    """Convert state name/abbreviation to standard abbreviation."""
    state_input = state_input.strip()
    
    # Already an abbreviation
    if state_input.upper() in STATE_ABBREVS:
        return state_input.upper()
    
    # Full state name
    if state_input.lower() in STATE_NAMES:
        return STATE_NAMES[state_input.lower()]
    
    return None

def get_major_city_info(city_name: str) -> dict:
    """Get major city information including FIPS codes."""
    return MAJOR_CITIES.get(city_name.lower())

def is_rate_calculation(variable: str) -> bool:
    """Check if variable requires rate calculation."""
    var_lower = variable.lower().replace(' ', '_')
    return any(rate_name in var_lower for rate_name in RATE_CALCULATIONS.keys())

def get_variable_mapping(variable: str) -> str:
    """Get Census variable code for human-readable variable name."""
    return VARIABLE_MAPPINGS.get(variable.lower())

# =============================================================================
# API INTEGRATION ENDPOINTS (FOR FUTURE DYNAMIC LOADING)
# =============================================================================

# These endpoints can be used to fetch mappings dynamically
CENSUS_API_ENDPOINTS = {
    'states': 'https://api.census.gov/data/2023/acs/acs5?get=NAME&for=state:*',
    'variables': 'https://api.census.gov/data/2023/acs/acs5/variables.json',
    'geographies': 'https://api.census.gov/data/2023/acs/acs5/geography.json'
}

# =============================================================================
# VALIDATION SETS
# =============================================================================

# Valid state abbreviations for input validation
VALID_STATE_ABBREVS = set(STATE_ABBREVS.keys())

# Valid geography types for Census API
VALID_GEOGRAPHY_TYPES = {
    'us', 'state', 'county', 'place', 'tract', 'block group'
}

# Common variable aliases that should map to the same Census variable
VARIABLE_ALIASES = {
    'income': ['median income', 'household income', 'median household income'],
    'population': ['pop', 'total population'],
    'poverty': ['poverty rate', 'poverty_rate', 'below poverty'],
    'unemployment': ['unemployment rate', 'unemployment_rate', 'jobless rate'],
    'housing': ['housing units', 'total housing units'],
    'rent': ['median rent', 'gross rent'],
    'home value': ['median home value', 'house value', 'median house value']
}
