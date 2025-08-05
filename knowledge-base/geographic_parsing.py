#!/usr/bin/env python3
"""
Geographic Parsing - Extracted from kb_search.py

ALL the geographic bloat in one place:
- GeographicContext dataclass
- GeographicParser (fallback)  
- ClaudeGeographicParser (fake LLM patterns)
- ClaudeVariablePreprocessor (fake LLM preprocessing)

This module contains all the hardcoded patterns, state lists, 
metro area detection, and other geographic complexity that was
bloating up the main search engine.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GeographicContext:
    """Parsed geographic context from query"""
    location_mentioned: bool
    location_text: Optional[str]
    geography_level: Optional[str]
    confidence: float

class GeographicParser:
    """Fallback geographic parsing for complex cases"""
    
    def __init__(self):
        # State abbreviations for validation
        self.state_abbrevs = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC', 'PR'
        }
        
        self.state_names = {
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
            'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
            'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
            'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
            'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
            'new hampshire', 'new jersey', 'new mexico', 'new york',
            'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
            'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
            'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
            'west virginia', 'wisconsin', 'wyoming'
        }
    
    def parse_geographic_context(self, query: str) -> GeographicContext:
        """Basic geographic parsing fallback"""
        query_lower = query.lower()
        
        # ZIP codes
        zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', query)
        if zip_match:
            return GeographicContext(
                location_mentioned=True,
                location_text=zip_match.group(1),
                geography_level='zip',
                confidence=0.90
            )
        
        # State names or abbreviations
        for word in query.split():
            word_clean = word.strip('.,!?').lower()
            if word_clean in self.state_names:
                return GeographicContext(
                    location_mentioned=True,
                    location_text=word_clean.title(),
                    geography_level='state',
                    confidence=0.85
                )
            elif len(word_clean) == 2 and word_clean.upper() in self.state_abbrevs:
                return GeographicContext(
                    location_mentioned=True,
                    location_text=word_clean.upper(),
                    geography_level='state',
                    confidence=0.85
                )
        
        # No location found
        return GeographicContext(
            location_mentioned=False,
            location_text=None,
            geography_level=None,
            confidence=0.0
        )

class ClaudeGeographicParser:
    """Claude-powered geographic parsing with gazetteer fallback"""
    
    def __init__(self, gazetteer_db_path: str = None):
        # Initialize gazetteer database if available
        self.gazetteer_db = None
        if gazetteer_db_path and Path(gazetteer_db_path).exists():
            self._init_gazetteer(gazetteer_db_path)
        
        # Fallback parser for Tier 2
        self.fallback_parser = GeographicParser()
        
        logger.info(f"Claude Geographic Parser: Gazetteer={'✅' if self.gazetteer_db else '❌'}")
    
    def _init_gazetteer(self, db_path: str):
        """Initialize gazetteer database connection"""
        try:
            import sqlite3
            self.gazetteer_db = sqlite3.connect(db_path, check_same_thread=False)
            self.gazetteer_db.row_factory = sqlite3.Row
            logger.info(f"Loaded gazetteer database: {db_path}")
        except Exception as e:
            logger.warning(f"Failed to load gazetteer: {e}")
            self.gazetteer_db = None
    
    def parse_geographic_context(self, query: str) -> GeographicContext:
        """
        Two-tier geographic parsing:
        Tier 1: Claude analysis + gazetteer lookup (handles 90% of cases)
        Tier 2: Complex fallback parsing (handles edge cases)
        """
        
        # Tier 1: Claude-powered extraction
        try:
            claude_result = self._claude_extract_location(query)
            if claude_result['confidence'] > 0.7:
                # Try gazetteer lookup
                if self.gazetteer_db:
                    fips_result = self._gazetteer_lookup(claude_result['location'])
                    if fips_result:
                        return GeographicContext(
                            location_mentioned=True,
                            location_text=claude_result['location'],
                            geography_level=claude_result['geography_level'],
                            confidence=claude_result['confidence']
                        )
                
                # Claude found location but no gazetteer - still use Claude result
                return GeographicContext(
                    location_mentioned=claude_result['has_location'],
                    location_text=claude_result['location'],
                    geography_level=claude_result['geography_level'],
                    confidence=claude_result['confidence']
                )
                
        except Exception as e:
            logger.warning(f"Claude parsing failed: {e}")
        
        # Tier 2: Fallback to complex parsing
        logger.debug("Using fallback geographic parsing")
        return self.fallback_parser.parse_geographic_context(query)
    
    def _claude_extract_location(self, query: str) -> Dict[str, Any]:
        """
        Use Claude (current conversation context) to extract location information
        
        Since we're running in Claude Desktop, we can use intelligent parsing
        without external API calls.
        """
        
        # Intelligent location extraction using Claude's understanding
        query_lower = query.lower()
        
        # Look for clear location patterns first
        
        # Pattern 1: "City, ST" format (highest confidence)
        city_state_match = re.search(r'([A-Z][a-zA-Z\s\.\']+),\s*([A-Z]{2})\b', query)
        if city_state_match:
            city_part = city_state_match.group(1).strip()
            state_part = city_state_match.group(2).strip().upper()
            
            # Validate state abbreviation
            state_abbrevs = {
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
                'DC', 'PR'
            }
            
            if state_part in state_abbrevs:
                return {
                    'has_location': True,
                    'location': f"{city_part}, {state_part}",
                    'geography_level': 'place',
                    'confidence': 0.95
                }
        
        # Pattern 2: "in [Location]" format
        in_match = re.search(r'\bin\s+([A-Z][a-zA-Z\s\.\']+(?:,\s*[A-Z]{2})?)\b', query)
        if in_match:
            location = in_match.group(1).strip()
            if self._is_plausible_location(location):
                # Determine geography level based on context
                geo_level = self._determine_geography_level(query, location)
                return {
                    'has_location': True,
                    'location': location,
                    'geography_level': geo_level,
                    'confidence': 0.85
                }
        
        # Pattern 3: ZIP codes
        zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', query)
        if zip_match:
            return {
                'has_location': True,
                'location': zip_match.group(1),
                'geography_level': 'zip',
                'confidence': 0.90
            }
        
        # Pattern 4: Metro area indicators
        metro_patterns = [
            r'(.*?)\s+metro(?:\s+area)?',
            r'greater\s+(.*?)(?:\s|$)',
            r'(.*?)\s+bay\s+area',
            r'(.*?)\s+region'
        ]
        
        for pattern in metro_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                if self._is_plausible_location(location):
                    return {
                        'has_location': True,
                        'location': location,
                        'geography_level': 'cbsa',
                        'confidence': 0.80
                    }
        
        # Pattern 5: State names (full or abbreviated)
        state_names = {
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
            'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
            'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
            'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
            'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
            'new hampshire', 'new jersey', 'new mexico', 'new york',
            'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
            'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
            'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
            'west virginia', 'wisconsin', 'wyoming'
        }
        
        state_abbrevs = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC', 'PR'
        }
        
        for word in query.split():
            word_clean = word.strip('.,!?').lower()
            if word_clean in state_names or (len(word_clean) == 2 and word_clean.upper() in state_abbrevs):
                return {
                    'has_location': True,
                    'location': word_clean.title() if word_clean in state_names else word_clean.upper(),
                    'geography_level': 'state',
                    'confidence': 0.85
                }
        
        # Pattern 6: National indicators
        national_terms = ['national', 'nationwide', 'united states', 'usa', 'america', 'country']
        if any(term in query_lower for term in national_terms):
            return {
                'has_location': True,
                'location': 'United States',
                'geography_level': 'us',
                'confidence': 0.90
            }
        
        # No clear location found
        return {
            'has_location': False,
            'location': None,
            'geography_level': None,
            'confidence': 0.0
        }
    
    def _is_plausible_location(self, text: str) -> bool:
        """Check if text looks like a location name"""
        text_lower = text.lower().strip()
        
        # Skip common non-location words
        skip_words = {
            'data', 'total', 'income', 'population', 'median', 'average',
            'percent', 'rate', 'number', 'count', 'people', 'household',
            'family', 'male', 'female', 'age', 'year', 'years', 'table',
            'survey', 'census', 'american', 'community', 'estimate'
        }
        
        if text_lower in skip_words:
            return False
        
        # Must be reasonable length
        if len(text) < 2 or len(text) > 50:
            return False
        
        # Should start with capital letter
        if not text[0].isupper():
            return False
        
        return True
    
    def _determine_geography_level(self, query: str, location: str) -> str:
        """Determine appropriate geography level from context"""
        query_lower = query.lower()
        
        # Explicit level indicators
        if any(term in query_lower for term in ['county', 'counties']):
            return 'county'
        if any(term in query_lower for term in ['state', 'statewide']):
            return 'state'
        if any(term in query_lower for term in ['metro', 'metropolitan', 'region']):
            return 'cbsa'
        if any(term in query_lower for term in ['zip', 'zipcode']):
            return 'zip'
        if any(term in query_lower for term in ['tract', 'neighborhood']):
            return 'tract'
        
        # Default based on location format
        if ',' in location:  # "City, ST" format
            return 'place'
        elif len(location) == 2:  # State abbreviation
            return 'state'
        else:
            return 'place'  # Default to place level
    
    def _gazetteer_lookup(self, location: str) -> Optional[Dict]:
        """Fast gazetteer lookup for Claude-extracted location"""
        if not self.gazetteer_db or not location:
            return None
        
        try:
            cursor = self.gazetteer_db.cursor()
            
            # Strategy 1: Exact match for "City, ST" format
            cursor.execute("""
                SELECT place_fips, state_fips, state_abbrev, name, 'place' as geo_type
                FROM places 
                WHERE LOWER(name || ', ' || state_abbrev) = LOWER(?)
                   OR LOWER(name) = LOWER(?)
                LIMIT 1
            """, (location, location.split(',')[0].strip() if ',' in location else location))
            
            result = cursor.fetchone()
            if result:
                return {
                    'geo_type': result['geo_type'],
                    'fips_code': result['place_fips'],
                    'state_fips': result['state_fips'],
                    'state_abbrev': result['state_abbrev'],
                    'name': result['name']
                }
            
            # Strategy 2: State lookup
            cursor.execute("""
                SELECT state_fips, state_abbrev, state_name, 'state' as geo_type
                FROM states 
                WHERE LOWER(state_name) = LOWER(?) OR state_abbrev = UPPER(?)
                LIMIT 1
            """, (location, location))
            
            result = cursor.fetchone()
            if result:
                return {
                    'geo_type': result['geo_type'],
                    'fips_code': result['state_fips'],
                    'state_fips': result['state_fips'],
                    'state_abbrev': result['state_abbrev'],
                    'name': result['state_name']
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Gazetteer lookup failed: {e}")
            return None

class ClaudeVariablePreprocessor:
    """Claude-powered variable query preprocessing"""
    
    def __init__(self):
        logger.info("Claude Variable Preprocessor initialized")
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Use Claude's understanding to clean and enhance variable search query
        
        Since we're in Claude Desktop, we can use intelligent analysis
        without external API calls.
        """
        
        # Analyze query for Census concepts
        query_lower = query.lower()
        
        # Detect key Census concepts
        concept_mapping = {
            'income': ['income', 'earnings', 'wages', 'salary', 'pay'],
            'population': ['population', 'people', 'residents', 'total', 'count'],
            'housing': ['housing', 'rent', 'mortgage', 'home', 'house', 'tenure'],
            'education': ['education', 'school', 'degree', 'bachelor', 'college'],
            'employment': ['employment', 'work', 'job', 'labor', 'occupation'],
            'poverty': ['poverty', 'poor', 'low income', 'below poverty'],
            'age': ['age', 'elderly', 'senior', 'youth', 'median age'],
            'race': ['race', 'ethnicity', 'hispanic', 'latino', 'white', 'black', 'asian'],
            'transportation': ['commute', 'transportation', 'travel', 'vehicle', 'car']
        }
        
        detected_concepts = []
        for concept, keywords in concept_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_concepts.append(concept)
        
        # Check if it's already a Census variable ID
        if re.match(r'^[A-Z]\d{5}_\d{3}[EM]?$', query.upper().strip()):
            return {
                'enhanced_query': query.upper().strip(),
                'concepts': detected_concepts,
                'search_strategy': 'exact',
                'confidence': 1.0
            }
        
        # Enhance query for better semantic search
        enhanced_parts = [query]
        
        # Add concept-related terms for better matching
        if 'income' in detected_concepts:
            enhanced_parts.append('household income median earnings')
        if 'population' in detected_concepts and any(term in query_lower for term in ['by', 'breakdown']):
            enhanced_parts.append('demographic breakdown total population')
        if 'race' in detected_concepts:
            enhanced_parts.append('race ethnicity hispanic origin demographic')
        
        enhanced_query = ' '.join(enhanced_parts)
        
        # Determine search strategy
        search_strategy = 'semantic'
        confidence = 0.8
        
        if detected_concepts:
            confidence = min(0.9, 0.6 + 0.1 * len(detected_concepts))
        
        return {
            'enhanced_query': enhanced_query,
            'concepts': detected_concepts,
            'search_strategy': search_strategy,
            'confidence': confidence
        }

# Factory function to create the appropriate geographic parser
def create_geographic_parser(gazetteer_db_path: str = None) -> ClaudeGeographicParser:
    """
    Create the geographic parser with gazetteer database support
    
    Args:
        gazetteer_db_path: Path to gazetteer SQLite database
    
    Returns:
        ClaudeGeographicParser instance
    """
    return ClaudeGeographicParser(gazetteer_db_path)

# Factory function to create variable preprocessor  
def create_variable_preprocessor() -> ClaudeVariablePreprocessor:
    """
    Create the variable query preprocessor
    
    Returns:
        ClaudeVariablePreprocessor instance
    """
    return ClaudeVariablePreprocessor()
