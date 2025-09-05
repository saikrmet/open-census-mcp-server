#!/usr/bin/env python3
"""
Pure LLM Geographic Resolver with Database RAG Validation

ARCHITECTURE:
1. LLM Primary (95%) - GPT-4 knows all major US locations and FIPS codes
2. Database RAG (5%) - Validates FIPS codes and handles edge cases
3. Confidence scoring - High confidence LLM results bypass validation

BENEFITS:
- Fast: One LLM call vs complex database queries
- Accurate: GPT-4 knows major FIPS codes perfectly
- Complete: Database RAG for obscure places
- Simple: No brittle regex parsing
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import sqlite3

from openai import OpenAI

# Environment setup
try:
    from dotenv import load_dotenv
    for env_path in [Path('.env'), Path(__file__).parent / '.env',
                     Path(__file__).parent.parent / '.env',
                     Path(__file__).parent.parent.parent / '.env']:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PureLLMGeographicResolver:
    """
    LLM-first geographic resolution with database RAG validation
    
    Uses GPT-4's extensive geographic knowledge as primary resolver,
    with database validation for precision and edge case handling.
    """
    
    def __init__(self, gazetteer_db_path: Optional[str] = None):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY required for LLM geographic resolution")
        
        # Initialize database for RAG validation (optional)
        self.db_path = self._find_database(gazetteer_db_path)
        self.db_conn = self._init_database() if self.db_path else None
        
        logger.info("🧠 Pure LLM Geographic Resolver initialized")
        if self.db_conn:
            logger.info("✅ Database RAG validation available")
        else:
            logger.info("⚠️ Database RAG not available - LLM only mode")
    
    def _find_database(self, gazetteer_db_path: Optional[str]) -> Optional[Path]:
        """Find geographic database for RAG validation"""
        
        if gazetteer_db_path and Path(gazetteer_db_path).exists():
            return Path(gazetteer_db_path)
        
        # Auto-detect database
        possible_paths = [
            Path(__file__).parent / "geo-db" / "geography.db",
            Path(__file__).parent.parent / "knowledge-base" / "geo-db" / "geography.db",
            Path(os.getcwd()) / "knowledge-base" / "geo-db" / "geography.db"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Auto-detected database: {path}")
                return path
        
        logger.info("No database found - pure LLM mode only")
        return None
    
    def _init_database(self) -> Optional[sqlite3.Connection]:
        """Initialize database connection for RAG validation"""
        
        try:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            
            # Quick validation
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM places")
            place_count = cursor.fetchone()[0]
            logger.info(f"Database RAG ready: {place_count:,} places available")
            
            return conn
            
        except Exception as e:
            logger.warning(f"Database RAG initialization failed: {e}")
            return None
    
    def resolve_location(self, query: str) -> Dict[str, Any]:
        """
        Pure LLM geographic resolution with optional database validation
        
        Flow:
        1. LLM resolves location with FIPS codes
        2. Confidence assessment 
        3. Database validation if needed (low confidence or user requests)
        4. Final result with accuracy metadata
        """
        logger.info(f"🧠 LLM resolving: '{query}'")
        
        # Step 1: LLM Primary Resolution
        llm_result = self._llm_resolve_location(query)
        
        if 'error' in llm_result:
            logger.warning(f"LLM resolution failed: {llm_result['error']}")
            return llm_result
        
        confidence = llm_result.get('confidence', 0.0)
        logger.info(f"🎯 LLM confidence: {confidence:.2f}")
        
        # Step 2: Validation Decision
        if confidence >= 0.85:
            logger.info("✅ High confidence - skipping validation")
            return self._add_metadata(llm_result, query, 'llm_high_confidence')
        
        elif confidence >= 0.60 and self.db_conn:
            logger.info("🔍 Medium confidence - validating with database RAG")
            return self._validate_with_database_rag(llm_result, query)
        
        elif self.db_conn:
            logger.info("⚠️ Low confidence - database RAG search")
            return self._database_rag_search(llm_result, query)
        
        else:
            logger.info("📝 No validation available - returning LLM result")
            return self._add_metadata(llm_result, query, 'llm_only_no_validation')
    
    def _llm_resolve_location(self, query: str) -> Dict[str, Any]:
        """Primary LLM geographic resolution"""
        
        try:
            prompt = f"""You are a US Census geographic expert. Resolve this location query to official Census format with FIPS codes.

Query: "{query}"

Provide the most accurate US Census geographic information in JSON format:

{{
    "geography_type": "place|county|state|us",
    "name": "Official Census name",
    "state_name": "Full state name",
    "state_abbrev": "ST",
    "state_fips": "06",
    "place_fips": "67000",
    "county_fips": "075",
    "confidence": 0.95,
    "reasoning": "Brief explanation of resolution"
}}

Rules:
- Use official Census naming (e.g., "San Francisco city", "Cook County")
- Provide correct FIPS codes for state and place/county
- Confidence 0.9+ for major cities/states, 0.7+ for smaller places, 0.5+ for uncertain
- If multiple matches exist, pick the most populous/well-known
- For ambiguous queries, state your assumption in reasoning

Examples:
- "San Francisco, CA" → San Francisco city, California (state_fips: "06", place_fips: "67000")
- "NYC" → New York city, New York (state_fips: "36", place_fips: "51000")  
- "Cook County, IL" → Cook County, Illinois (state_fips: "17", county_fips: "031")
- "Texas" → Texas (state_fips: "48")"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                logger.info(f"✅ LLM resolved: {result.get('name', 'Unknown')} (confidence: {result.get('confidence', 0):.2f})")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"LLM returned invalid JSON: {result_text[:200]}")
                return {
                    'error': 'LLM response parsing failed',
                    'raw_response': result_text[:200]
                }
            
        except Exception as e:
            logger.error(f"LLM resolution failed: {e}")
            return {
                'error': f'LLM geographic resolution failed: {str(e)}',
                'query': query
            }
    
    def _validate_with_database_rag(self, llm_result: Dict, query: str) -> Dict[str, Any]:
        """Validate LLM result against database RAG"""
        
        if not self.db_conn:
            return self._add_metadata(llm_result, query, 'validation_unavailable')
        
        try:
            geography_type = llm_result.get('geography_type')
            
            if geography_type == 'place':
                return self._validate_place(llm_result, query)
            elif geography_type == 'county':
                return self._validate_county(llm_result, query)
            elif geography_type == 'state':
                return self._validate_state(llm_result, query)
            else:
                return self._add_metadata(llm_result, query, 'validation_not_needed')
                
        except Exception as e:
            logger.warning(f"Database validation failed: {e}")
            return self._add_metadata(llm_result, query, 'validation_failed')
    
    def _validate_place(self, llm_result: Dict, query: str) -> Dict[str, Any]:
        """Validate place resolution against database"""
        
        state_fips = llm_result.get('state_fips')
        place_fips = llm_result.get('place_fips')
        
        if not state_fips or not place_fips:
            return self._add_metadata(llm_result, query, 'validation_insufficient_data')
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT name, state_abbrev, population, lat, lon
                FROM places 
                WHERE state_fips = ? AND place_fips = ?
                LIMIT 1
            """, (state_fips, place_fips))
            
            row = cursor.fetchone()
            if row:
                # Validation successful - merge database data
                validated_result = llm_result.copy()
                validated_result.update({
                    'db_name': row['name'],
                    'population': row['population'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'validation_status': 'confirmed'
                })
                
                logger.info(f"✅ Database validation confirmed: {row['name']}")
                return self._add_metadata(validated_result, query, 'llm_database_validated')
            
            else:
                logger.warning(f"❌ Database validation failed - FIPS not found: {state_fips}:{place_fips}")
                
                # Search for similar names as correction
                cursor.execute("""
                    SELECT name, state_fips, place_fips, state_abbrev
                    FROM places
                    WHERE state_fips = ? AND name_lower LIKE ?
                    ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                    LIMIT 3
                """, (state_fips, f"%{llm_result.get('name', '').lower().replace(' city', '')}%"))
                
                suggestions = [f"{row['name']}, {row['state_abbrev']} (FIPS: {row['state_fips']}:{row['place_fips']})"
                              for row in cursor.fetchall()]
                
                if suggestions:
                    llm_result['validation_status'] = 'fips_mismatch'
                    llm_result['database_suggestions'] = suggestions
                    logger.info(f"💡 Found similar places: {suggestions[:2]}")
                
                return self._add_metadata(llm_result, query, 'llm_validation_failed')
                
        except Exception as e:
            logger.error(f"Place validation error: {e}")
            return self._add_metadata(llm_result, query, 'validation_error')
    
    def _validate_state(self, llm_result: Dict, query: str) -> Dict[str, Any]:
        """Validate state resolution against database"""
        
        state_fips = llm_result.get('state_fips')
        
        if not state_fips:
            return self._add_metadata(llm_result, query, 'validation_insufficient_data')
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT state_name, state_abbrev
                FROM states 
                WHERE state_fips = ?
            """, (state_fips,))
            
            row = cursor.fetchone()
            if row:
                validated_result = llm_result.copy()
                validated_result.update({
                    'db_name': row['state_name'],
                    'validation_status': 'confirmed'
                })
                
                logger.info(f"✅ State validation confirmed: {row['state_name']}")
                return self._add_metadata(validated_result, query, 'llm_database_validated')
            else:
                logger.warning(f"❌ State validation failed - FIPS not found: {state_fips}")
                return self._add_metadata(llm_result, query, 'llm_validation_failed')
                
        except Exception as e:
            logger.error(f"State validation error: {e}")
            return self._add_metadata(llm_result, query, 'validation_error')
    
    def _validate_county(self, llm_result: Dict, query: str) -> Dict[str, Any]:
        """Validate county resolution against database"""
        
        state_fips = llm_result.get('state_fips')
        county_fips = llm_result.get('county_fips')
        
        if not state_fips or not county_fips:
            return self._add_metadata(llm_result, query, 'validation_insufficient_data')
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT name, state_abbrev, lat, lon
                FROM counties 
                WHERE state_fips = ? AND county_fips = ?
            """, (state_fips, county_fips))
            
            row = cursor.fetchone()
            if row:
                validated_result = llm_result.copy()
                validated_result.update({
                    'db_name': row['name'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'validation_status': 'confirmed'
                })
                
                logger.info(f"✅ County validation confirmed: {row['name']}")
                return self._add_metadata(validated_result, query, 'llm_database_validated')
            else:
                logger.warning(f"❌ County validation failed - FIPS not found: {state_fips}:{county_fips}")
                return self._add_metadata(llm_result, query, 'llm_validation_failed')
                
        except Exception as e:
            logger.error(f"County validation error: {e}")
            return self._add_metadata(llm_result, query, 'validation_error')
    
    def _database_rag_search(self, llm_result: Dict, query: str) -> Dict[str, Any]:
        """Database RAG search for low confidence cases"""
        
        logger.info(f"🔍 Database RAG search for: '{query}'")
        
        if not self.db_conn:
            return self._add_metadata(llm_result, query, 'rag_unavailable')
        
        try:
            cursor = self.db_conn.cursor()
            search_term = f"%{query.lower()}%"
            
            # Search places first (most common)
            cursor.execute("""
                SELECT 'place' as type, name, state_fips, place_fips, state_abbrev, population
                FROM places 
                WHERE name_lower LIKE ?
                ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                LIMIT 5
            """, (search_term,))
            
            rag_results = cursor.fetchall()
            
            if rag_results:
                best_match = rag_results[0]
                
                rag_result = {
                    'geography_type': 'place',
                    'name': best_match['name'],
                    'state_fips': best_match['state_fips'],
                    'place_fips': best_match['place_fips'],
                    'state_abbrev': best_match['state_abbrev'],
                    'population': best_match['population'],
                    'confidence': 0.7,  # RAG confidence
                    'reasoning': f'Database RAG match from {len(rag_results)} options',
                    'rag_alternatives': [f"{r['name']}, {r['state_abbrev']}" for r in rag_results[1:4]]
                }
                
                logger.info(f"✅ Database RAG found: {best_match['name']}, {best_match['state_abbrev']}")
                return self._add_metadata(rag_result, query, 'database_rag_search')
            
            else:
                logger.warning(f"❌ Database RAG found no matches")
                return self._add_metadata(llm_result, query, 'rag_no_matches')
                
        except Exception as e:
            logger.error(f"Database RAG search failed: {e}")
            return self._add_metadata(llm_result, query, 'rag_search_error')
    
    def _add_metadata(self, result: Dict, query: str, resolution_method: str) -> Dict[str, Any]:
        """Add resolution metadata"""
        
        result = result.copy()
        result.update({
            'original_query': query,
            'resolution_method': resolution_method,
            'resolver': 'pure_llm_geographic_resolver'
        })
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics"""
        
        return {
            'resolver_type': 'pure_llm_geographic_resolver',
            'llm_available': bool(os.getenv('OPENAI_API_KEY')),
            'database_rag_available': self.db_conn is not None,
            'architecture': 'llm_primary_database_rag_validation'
        }

# Factory function
def create_pure_llm_resolver(gazetteer_db_path: Optional[str] = None) -> PureLLMGeographicResolver:
    """Create Pure LLM Geographic Resolver with optional database RAG"""
    return PureLLMGeographicResolver(gazetteer_db_path)

if __name__ == "__main__":
    # Test the pure LLM resolver
    logger.info("🧪 Testing Pure LLM Geographic Resolver...")
    
    try:
        resolver = create_pure_llm_resolver()
        
        test_cases = [
            "San Francisco, CA",
            "Austin, TX",
            "NYC",
            "Boston, MA",
            "Cook County, IL",
            "California",
            "United States",
            "Obscure Town, MT"  # Edge case
        ]
        
        for location in test_cases:
            logger.info(f"\n🌍 Testing: '{location}'")
            
            try:
                result = resolver.resolve_location(location)
                
                if 'error' in result:
                    logger.warning(f"❌ Failed: {result['error']}")
                else:
                    name = result.get('name', 'Unknown')
                    confidence = result.get('confidence', 0)
                    method = result.get('resolution_method', 'unknown')
                    
                    logger.info(f"✅ Success: {name} (confidence: {confidence:.2f})")
                    logger.info(f"   Method: {method}")
                    
                    if result.get('geography_type') == 'place':
                        fips = f"{result.get('state_fips', 'XX')}:{result.get('place_fips', 'XXXXX')}"
                        logger.info(f"   FIPS: {fips}")
                    elif result.get('geography_type') == 'state':
                        logger.info(f"   FIPS: {result.get('state_fips', 'XX')}")
                    
                    if 'validation_status' in result:
                        logger.info(f"   Validation: {result['validation_status']}")
                
            except Exception as e:
                logger.error(f"❌ Error testing '{location}': {e}")
        
        # Test stats
        stats = resolver.get_stats()
        logger.info(f"\n📊 Resolver stats: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
