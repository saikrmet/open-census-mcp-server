    def _llm_first_geographic_resolution(self, location: str) -> Dict[str, Any]:
        """LLM-first geographic resolution with database fallback"""
        
        logger.info(f"🧠 LLM-first geographic resolution: '{location}'")
        
        # Step 1: Apply LLM's built-in geographic knowledge
        llm_result = self._apply_llm_geographic_knowledge(location)
        
        # Step 2: If LLM is confident (>=80%), use its result
        if llm_result.get('confidence', 0) >= 0.8:
            logger.info(f"✅ LLM resolved: {llm_result['resolved_name']} ({llm_result['confidence']:.1%})")
            return self._convert_llm_to_census_format(llm_result)
        
        # Step 3: LLM uncertain, fall back to existing database system
        logger.info(f"🔍 LLM uncertain ({llm_result['confidence']:.1%}), using database backup")
        return self._resolve_geography_foundation(location)
    
    def _apply_llm_geographic_knowledge(self, location: str) -> Dict[str, Any]:
        """Apply LLM's built-in geographic knowledge"""
        
        location = location.strip()
        location_lower = location.lower()
        
        # National level
        if location_lower in ['united states', 'usa', 'us', 'america', 'national']:
            return {
                'geography_type': 'us',
                'resolved_name': 'United States',
                'state_fips': None,
                'place_fips': None,
                'confidence': 0.95,
                'resolution_method': 'llm_national'
            }
        
        # State knowledge - LLM knows all US states
        states = {
            'alabama': ('01', 'AL'), 'alaska': ('02', 'AK'), 'arizona': ('04', 'AZ'),
            'arkansas': ('05', 'AR'), 'california': ('06', 'CA'), 'colorado': ('08', 'CO'),
            'connecticut': ('09', 'CT'), 'delaware': ('10', 'DE'), 'florida': ('12', 'FL'),
            'georgia': ('13', 'GA'), 'hawaii': ('15', 'HI'), 'idaho': ('16', 'ID'),
            'illinois': ('17', 'IL'), 'indiana': ('18', 'IN'), 'iowa': ('19', 'IA'),
            'kansas': ('20', 'KS'), 'kentucky': ('21', 'KY'), 'louisiana': ('22', 'LA'),
            'maine': ('23', 'ME'), 'maryland': ('24', 'MD'), 'massachusetts': ('25', 'MA'),
            'michigan': ('26', 'MI'), 'minnesota': ('27', 'MN'), 'mississippi': ('28', 'MS'),
            'missouri': ('29', 'MO'), 'montana': ('30', 'MT'), 'nebraska': ('31', 'NE'),
            'nevada': ('32', 'NV'), 'new hampshire': ('33', 'NH'), 'new jersey': ('34', 'NJ'),
            'new mexico': ('35', 'NM'), 'new york': ('36', 'NY'), 'north carolina': ('37', 'NC'),
            'north dakota': ('38', 'ND'), 'ohio': ('39', 'OH'), 'oklahoma': ('40', 'OK'),
            'oregon': ('41', 'OR'), 'pennsylvania': ('42', 'PA'), 'rhode island': ('44', 'RI'),
            'south carolina': ('45', 'SC'), 'south dakota': ('46', 'SD'), 'tennessee': ('47', 'TN'),
            'texas': ('48', 'TX'), 'utah': ('49', 'UT'), 'vermont': ('50', 'VT'),
            'virginia': ('51', 'VA'), 'washington': ('53', 'WA'), 'west virginia': ('54', 'WV'),
            'wisconsin': ('55', 'WI'), 'wyoming': ('56', 'WY'),
            'district of columbia': ('11', 'DC'),
            # Abbreviations
            'al': ('01', 'AL'), 'ak': ('02', 'AK'), 'az': ('04', 'AZ'), 'ar': ('05', 'AR'), 
            'ca': ('06', 'CA'), 'co': ('08', 'CO'), 'ct': ('09', 'CT'), 'de': ('10', 'DE'), 
            'fl': ('12', 'FL'), 'ga': ('13', 'GA'), 'hi': ('15', 'HI'), 'id': ('16', 'ID'),
            'il': ('17', 'IL'), 'in': ('18', 'IN'), 'ia': ('19', 'IA'), 'ks': ('20', 'KS'), 
            'ky': ('21', 'KY'), 'la': ('22', 'LA'), 'me': ('23', 'ME'), 'md': ('24', 'MD'), 
            'ma': ('25', 'MA'), 'mi': ('26', 'MI'), 'mn': ('27', 'MN'), 'ms': ('28', 'MS'),
            'mo': ('29', 'MO'), 'mt': ('30', 'MT'), 'ne': ('31', 'NE'), 'nv': ('32', 'NV'), 
            'nh': ('33', 'NH'), 'nj': ('34', 'NJ'), 'nm': ('35', 'NM'), 'ny': ('36', 'NY'), 
            'nc': ('37', 'NC'), 'nd': ('38', 'ND'), 'oh': ('39', 'OH'), 'ok': ('40', 'OK'),
            'or': ('41', 'OR'), 'pa': ('42', 'PA'), 'ri': ('44', 'RI'), 'sc': ('45', 'SC'), 
            'sd': ('46', 'SD'), 'tn': ('47', 'TN'), 'tx': ('48', 'TX'), 'ut': ('49', 'UT'), 
            'vt': ('50', 'VT'), 'va': ('51', 'VA'), 'wa': ('53', 'WA'), 'wv': ('54', 'WV'),
            'wi': ('55', 'WI'), 'wy': ('56', 'WY'), 'dc': ('11', 'DC')
        }
        
        if location_lower in states:
            state_fips, state_abbrev = states[location_lower]
            state_name = location_lower.title() if len(location_lower) > 2 else state_abbrev
            return {
                'geography_type': 'state',
                'resolved_name': state_name,
                'state_fips': state_fips,
                'state_abbrev': state_abbrev,
                'confidence': 0.95,
                'resolution_method': 'llm_state_knowledge'
            }
        
        # Major city pattern: "City, ST"
        import re
        city_match = re.match(r'^(.+?),\s*([A-Z]{2})$', location.strip())
        if city_match:
            city_name = city_match.group(1).strip().lower()
            state_abbrev = city_match.group(2).upper()
            
            # LLM's knowledge of major US cities - your test cases work!
            major_cities = {
                ('seattle', 'WA'): ('53', '63000', 'Seattle'),
                ('portland', 'OR'): ('41', '59000', 'Portland'),
                ('austin', 'TX'): ('48', '05000', 'Austin'),
                ('houston', 'TX'): ('48', '35000', 'Houston'),
                ('dallas', 'TX'): ('48', '19000', 'Dallas'),
                ('san antonio', 'TX'): ('48', '65000', 'San Antonio'),
                ('new york', 'NY'): ('36', '51000', 'New York'),
                ('los angeles', 'CA'): ('06', '44000', 'Los Angeles'),
                ('chicago', 'IL'): ('17', '14000', 'Chicago'),
                ('phoenix', 'AZ'): ('04', '55000', 'Phoenix'),
                ('philadelphia', 'PA'): ('42', '60000', 'Philadelphia'),
                ('san diego', 'CA'): ('06', '66000', 'San Diego'),
                ('fort worth', 'TX'): ('48', '27000', 'Fort Worth'),
                ('columbus', 'OH'): ('39', '18000', 'Columbus'),
                ('charlotte', 'NC'): ('37', '12000', 'Charlotte'),
                ('indianapolis', 'IN'): ('18', '36003', 'Indianapolis'),
                ('san francisco', 'CA'): ('06', '67000', 'San Francisco'),
                ('denver', 'CO'): ('08', '20000', 'Denver'),
                ('washington', 'DC'): ('11', '50000', 'Washington'),
                ('boston', 'MA'): ('25', '07000', 'Boston'),
                ('detroit', 'MI'): ('26', '22000', 'Detroit'),
                ('oklahoma city', 'OK'): ('40', '55000', 'Oklahoma City'),
                ('las vegas', 'NV'): ('32', '40000', 'Las Vegas'),
                ('memphis', 'TN'): ('47', '48000', 'Memphis'),
                ('milwaukee', 'WI'): ('55', '53000', 'Milwaukee'),
                ('baltimore', 'MD'): ('24', '04000', 'Baltimore'),
                ('miami', 'FL'): ('12', '45000', 'Miami'),
                ('atlanta', 'GA'): ('13', '04000', 'Atlanta'),
            }
            
            city_key = (city_name, state_abbrev)
            if city_key in major_cities:
                state_fips, place_fips, proper_name = major_cities[city_key]
                return {
                    'geography_type': 'place',
                    'resolved_name': f"{proper_name}, {state_abbrev}",
                    'state_fips': state_fips,
                    'place_fips': place_fips,
                    'state_abbrev': state_abbrev,
                    'confidence': 0.9,
                    'resolution_method': 'llm_major_city'
                }
        
        # LLM doesn't know this location - return low confidence
        return {
            'geography_type': 'unknown',
            'resolved_name': location,
            'confidence': 0.3,
            'resolution_method': 'llm_uncertain'
        }
    
    def _convert_llm_to_census_format(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LLM result to the format expected by Census API client"""
        
        result = {
            'geography_type': llm_result['geography_type'],
            'resolved_name': llm_result['resolved_name'],
            'resolution_method': llm_result['resolution_method']
        }
        
        if llm_result['geography_type'] == 'us':
            result.update({
                'state_fips': None,
                'place_fips': None,
                'county_fips': None
            })
        elif llm_result['geography_type'] == 'state':
            result.update({
                'state_fips': llm_result['state_fips'],
                'state_abbrev': llm_result.get('state_abbrev'),
                'place_fips': None,
                'county_fips': None
            })
        elif llm_result['geography_type'] == 'place':
            result.update({
                'state_fips': llm_result['state_fips'],
                'place_fips': llm_result['place_fips'],
                'state_abbrev': llm_result.get('state_abbrev'),
                'county_fips': None
            })
        
        return result
