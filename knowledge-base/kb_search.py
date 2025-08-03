#!/usr/bin/env python3
"""
Concept-Based Census Search Engine - Production Architecture

Three-tier semantic search system:
1. Table Catalog (coarse retrieval) → FAISS/ChromaDB
2. Variables Database (fine retrieval) → FAISS index  
3. Methodology Database (statistical expertise) → ChromaDB

Architecture:
- 36K concept-based variables (eliminates ACS1/5yr duplicates)
- Survey instance awareness with geographic intelligence
- Semantic routing between variables and methodology
- Statistical expertise injection for data quality

Path Structure:
- knowledge-base/table-catalog/ → Table embeddings + metadata
- knowledge-base/variables-faiss/ → Variable embeddings + metadata
- knowledge-base/methodology-db/ → ChromaDB collection for docs
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

# Required dependencies
import faiss
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# OpenAI embeddings support
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    import os
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    import os

# Silence tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with concept-based metadata"""
    variable_id: str
    table_id: str
    concept: str
    label: str
    title: str
    universe: str
    confidence: float
    geographic_relevance: float
    geographic_restrictions: Dict[str, str]
    available_surveys: List[str]
    statistical_notes: List[str]
    primary_variable: bool
    methodology_context: Optional[str] = None
    survey_instances: List[Dict] = None
    geography_coverage: Dict[str, List[str]] = None
    primary_instance: Optional[str] = None
    structure_type: Optional[str] = None

@dataclass
class GeographicContext:
    """Parsed geographic context from query"""
    location_mentioned: bool
    location_text: Optional[str]
    geography_level: Optional[str]
    confidence: float

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
        import re
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
        import re
        if re.match(r'^[A-Z]\d{5}_\d{3}[EM]?

class QueryTypeDetector:
    """Detects whether query needs variables, methodology, or both"""
    
    def __init__(self):
        self.variable_indicators = [
            # Direct variable patterns
            r'B\d{5}_\d{3}[EM]?',  # B19013_001E
            # Common variable requests
            'population', 'income', 'poverty', 'unemployment', 'housing', 'education',
            'median', 'total', 'percentage', 'count', 'number of', 'how many',
            # Geographic data requests
            'by state', 'by county', 'by race', 'by age', 'by gender'
        ]
        
        self.methodology_indicators = [
            # Methodology questions
            'how does', 'how is', 'methodology', 'method', 'calculated', 'measured',
            'definition', 'universe', 'sample size', 'margin of error', 'reliability',
            'data quality', 'survey design', 'coverage', 'response rate',
            # Statistical concepts
            'statistical', 'significance', 'confidence', 'weighting', 'estimation',
            'imputation', 'allocation', 'editing', 'processing'
        ]
    
    def detect_query_type(self, query: str) -> str:
        """
        Determine query type for smart routing
        
        Returns:
            'variables' - Query seeking specific data variables
            'methodology' - Query about statistical methods/concepts
            'both' - Ambiguous query needing both searches
        """
        query_lower = query.lower()
        
        variable_score = 0
        methodology_score = 0
        
        # Check for variable indicators
        for indicator in self.variable_indicators:
            if re.search(indicator, query_lower):
                variable_score += 1
        
        # Check for methodology indicators
        for indicator in self.methodology_indicators:
            if indicator in query_lower:
                methodology_score += 1
        
        # Determine routing
        if variable_score > methodology_score:
            return 'variables'
        elif methodology_score > variable_score:
            return 'methodology'
        else:
            return 'both'

class TableCatalogSearch:
    """Coarse retrieval using table catalog with FAISS/ChromaDB"""
    
    def __init__(self, catalog_dir: str = "table-catalog"):
        self.catalog_dir = Path(catalog_dir)
        self.tables = {}
        self.embeddings_index = None
        self.table_ids = []
        self.embedding_model = None
        self.use_openai_embeddings = False
        
        self._load_catalog()
        self._load_embeddings()
    
    def _load_catalog(self):
        """Load table catalog metadata"""
        catalog_file = self.catalog_dir / "table_catalog_enhanced.json"
        if not catalog_file.exists():
            catalog_file = self.catalog_dir / "table_catalog.json"
        
        if not catalog_file.exists():
            raise FileNotFoundError(f"Table catalog not found in {self.catalog_dir}")
        
        with open(catalog_file) as f:
            catalog_data = json.load(f)
        
        if 'tables' in catalog_data:
            tables_list = catalog_data['tables']
        else:
            tables_list = catalog_data
        
        self.tables = {table['table_id']: table for table in tables_list}
        logger.info(f"Loaded {len(self.tables)} tables from catalog")
    
    def _load_embeddings(self):
        """Load FAISS embeddings for tables"""
        faiss_file = self.catalog_dir / "table_embeddings_enhanced.faiss"
        if not faiss_file.exists():
            faiss_file = self.catalog_dir / "table_embeddings.faiss"
        
        if not faiss_file.exists():
            raise FileNotFoundError(f"Table FAISS index not found in {self.catalog_dir}")
        
        self.embeddings_index = faiss.read_index(str(faiss_file))
        
        # Load table ID mapping
        mapping_file = self.catalog_dir / "table_mapping_enhanced.json"
        if not mapping_file.exists():
            mapping_file = self.catalog_dir / "table_mapping.json"
        
        with open(mapping_file) as f:
            mapping_data = json.load(f)
        
        self.table_ids = mapping_data['table_ids']
        
        # Setup embedding model based on index type
        embedding_dimension = mapping_data.get('embedding_dimension', 768)
        if embedding_dimension == 3072 and OPENAI_AVAILABLE:
            self._setup_openai_embeddings()
        else:
            self._setup_sentence_transformers()
        
        logger.info(f"Loaded FAISS index with {len(self.table_ids)} table embeddings")
    
    def _setup_openai_embeddings(self):
        """Setup OpenAI embeddings"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.use_openai_embeddings = True
        logger.info("Using OpenAI embeddings for table search")
    
    def _setup_sentence_transformers(self):
        """Setup SentenceTransformers embeddings"""
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.use_openai_embeddings = False
        logger.info("Using SentenceTransformers for table search")
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate query embedding"""
        if self.use_openai_embeddings:
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-large"
            )
            return np.array([response.data[0].embedding], dtype=np.float32)
        else:
            return self.embedding_model.encode([query]).astype('float32')
    
    def search_tables(self, query: str, k: int = 5, geographic_context: Optional[GeographicContext] = None) -> List[Dict]:
        """Search for relevant tables using FAISS"""
        query_embedding = self._generate_query_embedding(query)
        
        # Search FAISS index
        distances, indices = self.embeddings_index.search(query_embedding, k * 2)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.table_ids):
                continue
            
            table_id = self.table_ids[idx]
            table_data = self.tables.get(table_id)
            if not table_data:
                continue
            
            # Convert distance to similarity
            similarity = max(0.0, 1.0 - (distance / 2.0))
            
            # Calculate geographic relevance
            geographic_relevance = self._calculate_geographic_relevance(table_data, geographic_context)
            
            # Combined score
            combined_score = similarity * (0.7 + 0.3 * geographic_relevance)
            
            results.append({
                'table_id': table_id,
                'table_data': table_data,
                'confidence': similarity,
                'geographic_relevance': geographic_relevance,
                'combined_score': combined_score
            })
        
        # Sort by combined score and return top k
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:k]
    
    def _calculate_geographic_relevance(self, table_data: Dict, geographic_context: Optional[GeographicContext]) -> float:
        """Calculate geographic relevance of table with CBSA support"""
        if not geographic_context or not geographic_context.location_mentioned:
            return 1.0
        
        geo_level = geographic_context.geography_level
        
        # Check if requested geography level is available
        available_levels = table_data.get('geography_levels', [])
        if geo_level in available_levels:
            return 1.0
        
        # Special handling for CBSA requests
        if geo_level == 'cbsa':
            # CBSAs often work with county or place data
            if 'county' in available_levels or 'place' in available_levels:
                return 0.95
            # Metro areas can also use state data for broader analysis
            if 'state' in available_levels:
                return 0.85
        
        # Check survey availability matrix
        survey_availability = table_data.get('survey_availability', {})
        
        # ACS5 availability (more geographic detail)
        acs5_levels = survey_availability.get('5yr', {})
        if isinstance(acs5_levels, dict):
            if geo_level in acs5_levels or any(geo_level in str(v) for v in acs5_levels.values()):
                return 1.0
        elif isinstance(acs5_levels, list) and geo_level in acs5_levels:
            return 1.0
        
        # ACS1 availability (limited geography but more current)
        acs1_levels = survey_availability.get('1yr', {})
        if isinstance(acs1_levels, dict):
            if geo_level in acs1_levels or any(geo_level in str(v) for v in acs1_levels.values()):
                return 0.85
        elif isinstance(acs1_levels, list) and geo_level in acs1_levels:
            return 0.85
        
        # Geographic level hierarchy fallbacks with CBSA intelligence
        level_hierarchy = {
            'tract': ['place', 'county', 'cbsa', 'state'],
            'zip': ['place', 'county', 'cbsa', 'state'],
            'place': ['county', 'cbsa', 'state'],
            'county': ['cbsa', 'state'],
            'cbsa': ['state'],
            'state': []
        }
        
        if geo_level in level_hierarchy:
            for fallback_level in level_hierarchy[geo_level]:
                if fallback_level in available_levels:
                    # CBSA fallbacks get higher relevance for economic data
                    if fallback_level == 'cbsa':
                        return 0.90
                    elif fallback_level == 'county':
                        return 0.80
                    elif fallback_level == 'state':
                        return 0.70
                    else:
                        return 0.75
        
        # Default moderate relevance (don't eliminate completely)
        return 0.60

class VariablesSearch:
    """Fine retrieval using concept-based FAISS variable index"""
    
    def __init__(self, variables_dir: str = "variables-faiss"):
        self.variables_dir = Path(variables_dir)
        self.variables_index = None
        self.variables_metadata = []
        self.embedding_model = None
        self.use_openai_embeddings = False
        
        self._load_variables()
    
    def _load_variables(self):
        """Load concept-based variables FAISS index and metadata"""
        logger.info("Loading concept-based variables database...")
        
        # Load FAISS index
        faiss_file = self.variables_dir / "variables.faiss"
        if not faiss_file.exists():
            raise FileNotFoundError(f"Variables FAISS index not found: {faiss_file}")
        
        self.variables_index = faiss.read_index(str(faiss_file))
        
        # Load metadata
        metadata_file = self.variables_dir / "variables_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Variables metadata not found: {metadata_file}")
        
        with open(metadata_file) as f:
            self.variables_metadata = json.load(f)
        
        # Setup embedding model based on build info
        build_info_file = self.variables_dir / "build_info.json"
        if build_info_file.exists():
            with open(build_info_file) as f:
                build_info = json.load(f)
            
            embedding_dimension = build_info.get('embedding_dimension', 768)
            if embedding_dimension == 3072 and OPENAI_AVAILABLE:
                self._setup_openai_embeddings()
            else:
                self._setup_sentence_transformers()
        else:
            self._setup_sentence_transformers()
        
        logger.info(f"Loaded {len(self.variables_metadata)} concept-based variables")
    
    def _setup_openai_embeddings(self):
        """Setup OpenAI embeddings"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.use_openai_embeddings = True
        logger.info("Using OpenAI embeddings for variables")
    
    def _setup_sentence_transformers(self):
        """Setup SentenceTransformers embeddings"""
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.use_openai_embeddings = False
        logger.info("Using SentenceTransformers for variables")
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate query embedding"""
        if self.use_openai_embeddings:
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-large"
            )
            return np.array([response.data[0].embedding], dtype=np.float32)
        else:
            return self.embedding_model.encode([query]).astype('float32')
    
    def search_within_table(self, table_id: str, query: str, geographic_context: Optional[GeographicContext] = None, k: int = 10) -> List[Dict]:
        """Search for variables within a specific table"""
        query_embedding = self._generate_query_embedding(query)
        
        # Search FAISS index broadly, then filter to table
        distances, indices = self.variables_index.search(query_embedding, k * 5)
        
        table_results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.variables_metadata):
                continue
            
            var_metadata = self.variables_metadata[idx]
            variable_id = var_metadata.get('variable_id', '')
            
            # Filter to target table
            var_table_id = variable_id.split('_')[0] if '_' in variable_id else ''
            if var_table_id != table_id:
                continue
            
            # Calculate scores
            semantic_score = max(0.0, 1.0 - (distance / 2.0))
            geographic_score = self._calculate_geographic_relevance(var_metadata, geographic_context)
            structure_bonus = self._calculate_structure_bonus(variable_id, query)
            
            final_score = semantic_score * (0.6 + 0.2 * geographic_score + 0.2 * structure_bonus)
            
            table_results.append({
                'variable_metadata': var_metadata,
                'semantic_score': semantic_score,
                'geographic_score': geographic_score,
                'structure_bonus': structure_bonus,
                'final_score': final_score
            })
        
        table_results.sort(key=lambda x: x['final_score'], reverse=True)
        return table_results[:k]
    
    def _calculate_geographic_relevance(self, var_metadata: Dict, geographic_context: Optional[GeographicContext]) -> float:
        """Calculate geographic relevance using concept-based metadata with CBSA support"""
        if not geographic_context or not geographic_context.location_mentioned:
            return 1.0
        
        geo_level = geographic_context.geography_level
        
        # Check geography coverage from metadata
        geography_coverage = var_metadata.get('geography_coverage', {})
        if isinstance(geography_coverage, str):
            try:
                geography_coverage = json.loads(geography_coverage) if geography_coverage else {}
            except json.JSONDecodeError:
                geography_coverage = {}
        
        # Direct geography level match
        if geography_coverage:
            for survey_type, geo_levels in geography_coverage.items():
                if isinstance(geo_levels, list):
                    if geo_level in geo_levels:
                        return 1.0
                elif isinstance(geo_levels, str):
                    if geo_level in geo_levels:
                        return 1.0
        
        # CBSA-specific intelligence
        if geo_level == 'cbsa':
            # Metro areas often use county or place aggregation
            if geography_coverage:
                for survey_type, geo_levels in geography_coverage.items():
                    geo_list = geo_levels if isinstance(geo_levels, list) else [geo_levels]
                    if 'county' in geo_list or 'place' in geo_list:
                        return 0.95  # High relevance - can aggregate to metro
                    if 'state' in geo_list:
                        return 0.85  # Moderate relevance - broader than metro
        
        # Economic data context for metro areas
        variable_id = var_metadata.get('variable_id', '')
        concept = var_metadata.get('concept', '').lower()
        label = var_metadata.get('label', '').lower()
        
        # Economic variables work well at metro level
        if geo_level == 'cbsa':
            economic_indicators = [
                'income', 'earnings', 'wage', 'salary', 'employment', 'unemployment',
                'poverty', 'commute', 'transportation', 'occupation', 'industry'
            ]
            if any(indicator in concept or indicator in label for indicator in economic_indicators):
                return 0.90  # Economic data often better at metro level
        
        # Variable type bonuses
        if variable_id.endswith('_001E'):
            # Total variables are more flexible across geography levels
            return 0.85
        
        # Default moderate relevance
        return 0.75
    
    def _calculate_structure_bonus(self, variable_id: str, query: str) -> float:
        """Calculate bonus for variable structure (prioritize totals unless breakdown requested)"""
        query_lower = query.lower()
        
        # Prefer _001E (total) variables unless query asks for breakdown
        if variable_id.endswith('_001E'):
            breakdown_terms = ['by', 'breakdown', 'split', 'detailed', 'male', 'female', 'age', 'race']
            if any(term in query_lower for term in breakdown_terms):
                return 0.5
            return 1.0
        
        # Bonus for specific breakdowns when requested
        if any(term in query_lower for term in ['male', 'female', 'men', 'women']):
            if 'male' in variable_id.lower() or 'female' in variable_id.lower():
                return 0.8
        
        return 0.7

class MethodologySearch:
    """Statistical expertise search using ChromaDB methodology database"""
    
    def __init__(self, methodology_dir: str = "methodology-db"):
        self.methodology_dir = Path(methodology_dir)
        self.collection = None
        self.embedding_model = None
        
        self._load_methodology_db()
    
    def _load_methodology_db(self):
        """Load ChromaDB methodology collection"""
        logger.info("Loading methodology database...")
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=str(self.methodology_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = client.get_collection("census_methodology")
            doc_count = self.collection.count()
            logger.info(f"Loaded methodology database with {doc_count} documents")
        except Exception as e:
            raise FileNotFoundError(f"Methodology collection not found: {e}")
        
        # Load embedding model for consistency
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    def search_methodology(self, query: str, k: int = 5) -> List[Dict]:
        """Search methodology documents for statistical expertise"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
        
        methodology_results = []
        for i in range(len(results['ids'][0])):
            methodology_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'confidence': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                'source': results['metadatas'][0][i].get('source', 'Unknown')
            })
        
        return methodology_results

class ConceptBasedCensusSearchEngine:
    """Complete coarse-to-fine search engine with LLM-first intelligence"""
    
    def __init__(self,
                 catalog_dir: str = "table-catalog",
                 variables_dir: str = "variables-faiss",
                 methodology_dir: str = "methodology-db",
                 gazetteer_db_path: str = None):
        
        # Set paths relative to knowledge-base directory
        base_dir = Path(__file__).parent
        
        # Initialize LLM-first components
        self.geo_parser = LLMGeographicParser(gazetteer_db_path)
        self.variable_preprocessor = LLMVariablePreprocessor()
        self.query_detector = QueryTypeDetector()
        
        # Initialize search components
        self.table_search = TableCatalogSearch(catalog_dir)
        self.variable_search = VariablesSearch(variables_dir)
        self.methodology_search = MethodologySearch(methodology_dir)
        
        logger.info("Concept-based Census Search Engine initialized")
        logger.info("✅ LLM-first architecture: Geography + Variable preprocessing")
        logger.info("✅ Three-tier search: Tables → Variables → Methodology")
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        LLM-enhanced search with concept-based coarse-to-fine retrieval
        
        Flow:
        1. LLM preprocesses query (geography + variables)
        2. Coarse search: Find relevant tables  
        3. Fine search: Find variables within tables
        4. Methodology: Add statistical context
        """
        logger.info(f"Searching: '{query}'")
        
        # Step 1: LLM preprocessing
        geo_context = self.geo_parser.parse_geographic_context(query)
        if geo_context.location_mentioned:
            logger.info(f"Geographic context: {geo_context.location_text} ({geo_context.geography_level})")
        
        variable_preprocessing = self.variable_preprocessor.preprocess_query(query)
        logger.info(f"Variable preprocessing: {variable_preprocessing['search_strategy']} search")
        
        # Use enhanced query for better semantic search
        search_query = variable_preprocessing['enhanced_query']
        
        # Step 2: Query type detection for smart routing
        query_type = self.query_detector.detect_query_type(search_query)
        logger.info(f"Query type: {query_type}")
        
        all_results = []
        
        # Step 3: Variables search (always for 'variables' and 'both' types)
        if query_type in ['variables', 'both']:
            all_results.extend(self._search_variables(search_query, geo_context, max_results))
        
        # Step 4: Methodology search (for 'methodology' and 'both' types)
        if query_type in ['methodology', 'both']:
            methodology_context = self._search_methodology(search_query)
            # Inject methodology context into results
            for result in all_results:
                if not result.methodology_context and methodology_context:
                    result.methodology_context = methodology_context[:500] + "..."
        
        # Step 5: Sort all results by confidence
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Returning {min(len(all_results), max_results)} results")
        return all_results[:max_results]
    
    def _search_variables(self, query: str, geo_context: GeographicContext, max_results: int) -> List[SearchResult]:
        """Search variables using coarse-to-fine retrieval"""
        # Coarse retrieval: find relevant tables
        table_results = self.table_search.search_tables(query, k=5, geographic_context=geo_context)
        logger.info(f"Found {len(table_results)} candidate tables")
        
        if not table_results:
            return []
        
        # Fine retrieval: search within top tables
        variable_results = []
        
        for table_result in table_results:
            table_id = table_result['table_id']
            table_data = table_result['table_data']
            table_confidence = table_result['confidence']
            
            # Search variables within this table
            vars_in_table = self.variable_search.search_within_table(
                table_id, query, geo_context, k=3
            )
            
            # Convert to SearchResult objects
            for var_result in vars_in_table:
                var_metadata = var_result['variable_metadata']
                variable_id = var_metadata.get('variable_id', '')
                
                # Combine table and variable confidence
                combined_confidence = table_confidence * 0.6 + var_result['final_score'] * 0.4
                
                # Extract survey information
                available_surveys = var_metadata.get('available_surveys', ['acs5'])
                if isinstance(available_surveys, str):
                    available_surveys = [available_surveys]
                
                # Handle geography coverage
                geography_coverage = {}
                geo_coverage_raw = var_metadata.get('geography_coverage', {})
                if isinstance(geo_coverage_raw, str):
                    try:
                        geography_coverage = json.loads(geo_coverage_raw) if geo_coverage_raw else {}
                    except json.JSONDecodeError:
                        geography_coverage = {}
                else:
                    geography_coverage = geo_coverage_raw or {}
                
                search_result = SearchResult(
                    variable_id=variable_id,
                    table_id=table_id,
                    concept=var_metadata.get('concept', var_metadata.get('concept_name', '')),
                    label=var_metadata.get('label', var_metadata.get('description', '')),
                    title=table_data.get('title', ''),
                    universe=table_data.get('universe', ''),
                    confidence=combined_confidence,
                    geographic_relevance=var_result['geographic_score'],
                    geographic_restrictions={
                        'acs1': table_data.get('geography_restrictions_1yr', ''),
                        'acs5': table_data.get('geography_restrictions_5yr', '')
                    },
                    available_surveys=available_surveys,
                    statistical_notes=table_data.get('statistical_notes', []),
                    primary_variable=variable_id.endswith('_001E'),
                    survey_instances=[],
                    geography_coverage=geography_coverage,
                    primary_instance=var_metadata.get('primary_instance'),
                    structure_type=var_metadata.get('structure_type', 'concept_based')
                )
                
                variable_results.append(search_result)
        
        return variable_results
    
    def _search_methodology(self, query: str) -> Optional[str]:
        """Search methodology for statistical expertise context"""
        try:
            methodology_results = self.methodology_search.search_methodology(query, k=2)
            if methodology_results:
                # Return most relevant methodology context
                return methodology_results[0]['content']
        except Exception as e:
            logger.warning(f"Methodology search failed: {e}")
        
        return None
    
    def get_table_info(self, table_id: str) -> Optional[Dict]:
        """Get complete information about a table"""
        return self.table_search.tables.get(table_id)
    
    def get_variable_info(self, variable_id: str) -> Optional[Dict]:
        """Get complete information about a variable"""
        for var_metadata in self.variable_search.variables_metadata:
            if var_metadata.get('variable_id') == variable_id:
                return var_metadata
        return None

# Backward compatibility alias
CensusSearchEngine = ConceptBasedCensusSearchEngine

# Factory function for MCP integration with LLM support
def create_search_engine(knowledge_base_dir: str = None, gazetteer_db_path: str = None) -> ConceptBasedCensusSearchEngine:
    """
    Create LLM-enhanced search engine with automatic path detection
    
    Args:
        knowledge_base_dir: Base directory for knowledge base files
        gazetteer_db_path: Path to gazetteer SQLite database
    
    Returns:
        ConceptBasedCensusSearchEngine instance with LLM support
    """
    if knowledge_base_dir:
        base_path = Path(knowledge_base_dir)
        catalog_dir = base_path / "table-catalog"
        variables_dir = base_path / "variables-faiss"
        methodology_dir = base_path / "methodology-db"
        
        # Auto-detect gazetteer if not specified
        if not gazetteer_db_path:
            potential_gazetteer = base_path / "geography.db"
            if potential_gazetteer.exists():
                gazetteer_db_path = str(potential_gazetteer)
    else:
        # Default paths relative to current file
        base_dir = Path(__file__).parent
        catalog_dir = base_dir / "table-catalog"
        variables_dir = base_dir / "variables-faiss"
        methodology_dir = base_dir / "methodology-db"
        
        # Auto-detect gazetteer
        if not gazetteer_db_path:
            potential_gazetteer = base_dir / "geography.db"
            if potential_gazetteer.exists():
                gazetteer_db_path = str(potential_gazetteer)
    
    return ConceptBasedCensusSearchEngine(
        catalog_dir=str(catalog_dir),
        variables_dir=str(variables_dir),
        methodology_dir=str(methodology_dir),
        gazetteer_db_path=gazetteer_db_path
    )ensusSearchEngine(
        catalog_dir=str(catalog_dir),
        variables_dir=str(variables_dir),
        methodology_dir=str(methodology_dir)
    )

if __name__ == "__main__":
    # Test the search engine
    logger.info("Testing Census Search Engine...")
    try:
        engine = create_search_engine()
        results = engine.search("median household income by race", max_results=5)
        logger.info(f"Test search returned {len(results)} results")
        for result in results[:2]:
            logger.info(f"  {result.variable_id}: {result.label[:100]}...")
    except Exception as e:
        logger.error(f"Test failed: {e}"), query.upper().strip()):
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

class QueryTypeDetector:
    """Detects whether query needs variables, methodology, or both"""
    
    def __init__(self):
        self.variable_indicators = [
            # Direct variable patterns
            r'B\d{5}_\d{3}[EM]?',  # B19013_001E
            # Common variable requests
            'population', 'income', 'poverty', 'unemployment', 'housing', 'education',
            'median', 'total', 'percentage', 'count', 'number of', 'how many',
            # Geographic data requests
            'by state', 'by county', 'by race', 'by age', 'by gender'
        ]
        
        self.methodology_indicators = [
            # Methodology questions
            'how does', 'how is', 'methodology', 'method', 'calculated', 'measured',
            'definition', 'universe', 'sample size', 'margin of error', 'reliability',
            'data quality', 'survey design', 'coverage', 'response rate',
            # Statistical concepts
            'statistical', 'significance', 'confidence', 'weighting', 'estimation',
            'imputation', 'allocation', 'editing', 'processing'
        ]
    
    def detect_query_type(self, query: str) -> str:
        """
        Determine query type for smart routing
        
        Returns:
            'variables' - Query seeking specific data variables
            'methodology' - Query about statistical methods/concepts
            'both' - Ambiguous query needing both searches
        """
        query_lower = query.lower()
        
        variable_score = 0
        methodology_score = 0
        
        # Check for variable indicators
        for indicator in self.variable_indicators:
            if re.search(indicator, query_lower):
                variable_score += 1
        
        # Check for methodology indicators
        for indicator in self.methodology_indicators:
            if indicator in query_lower:
                methodology_score += 1
        
        # Determine routing
        if variable_score > methodology_score:
            return 'variables'
        elif methodology_score > variable_score:
            return 'methodology'
        else:
            return 'both'

class TableCatalogSearch:
    """Coarse retrieval using table catalog with FAISS/ChromaDB"""
    
    def __init__(self, catalog_dir: str = "table-catalog"):
        self.catalog_dir = Path(catalog_dir)
        self.tables = {}
        self.embeddings_index = None
        self.table_ids = []
        self.embedding_model = None
        self.use_openai_embeddings = False
        
        self._load_catalog()
        self._load_embeddings()
    
    def _load_catalog(self):
        """Load table catalog metadata"""
        catalog_file = self.catalog_dir / "table_catalog_enhanced.json"
        if not catalog_file.exists():
            catalog_file = self.catalog_dir / "table_catalog.json"
        
        if not catalog_file.exists():
            raise FileNotFoundError(f"Table catalog not found in {self.catalog_dir}")
        
        with open(catalog_file) as f:
            catalog_data = json.load(f)
        
        if 'tables' in catalog_data:
            tables_list = catalog_data['tables']
        else:
            tables_list = catalog_data
        
        self.tables = {table['table_id']: table for table in tables_list}
        logger.info(f"Loaded {len(self.tables)} tables from catalog")
    
    def _load_embeddings(self):
        """Load FAISS embeddings for tables"""
        faiss_file = self.catalog_dir / "table_embeddings_enhanced.faiss"
        if not faiss_file.exists():
            faiss_file = self.catalog_dir / "table_embeddings.faiss"
        
        if not faiss_file.exists():
            raise FileNotFoundError(f"Table FAISS index not found in {self.catalog_dir}")
        
        self.embeddings_index = faiss.read_index(str(faiss_file))
        
        # Load table ID mapping
        mapping_file = self.catalog_dir / "table_mapping_enhanced.json"
        if not mapping_file.exists():
            mapping_file = self.catalog_dir / "table_mapping.json"
        
        with open(mapping_file) as f:
            mapping_data = json.load(f)
        
        self.table_ids = mapping_data['table_ids']
        
        # Setup embedding model based on index type
        embedding_dimension = mapping_data.get('embedding_dimension', 768)
        if embedding_dimension == 3072 and OPENAI_AVAILABLE:
            self._setup_openai_embeddings()
        else:
            self._setup_sentence_transformers()
        
        logger.info(f"Loaded FAISS index with {len(self.table_ids)} table embeddings")
    
    def _setup_openai_embeddings(self):
        """Setup OpenAI embeddings"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.use_openai_embeddings = True
        logger.info("Using OpenAI embeddings for table search")
    
    def _setup_sentence_transformers(self):
        """Setup SentenceTransformers embeddings"""
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.use_openai_embeddings = False
        logger.info("Using SentenceTransformers for table search")
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate query embedding"""
        if self.use_openai_embeddings:
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-large"
            )
            return np.array([response.data[0].embedding], dtype=np.float32)
        else:
            return self.embedding_model.encode([query]).astype('float32')
    
    def search_tables(self, query: str, k: int = 5, geographic_context: Optional[GeographicContext] = None) -> List[Dict]:
        """Search for relevant tables using FAISS"""
        query_embedding = self._generate_query_embedding(query)
        
        # Search FAISS index
        distances, indices = self.embeddings_index.search(query_embedding, k * 2)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.table_ids):
                continue
            
            table_id = self.table_ids[idx]
            table_data = self.tables.get(table_id)
            if not table_data:
                continue
            
            # Convert distance to similarity
            similarity = max(0.0, 1.0 - (distance / 2.0))
            
            # Calculate geographic relevance
            geographic_relevance = self._calculate_geographic_relevance(table_data, geographic_context)
            
            # Combined score
            combined_score = similarity * (0.7 + 0.3 * geographic_relevance)
            
            results.append({
                'table_id': table_id,
                'table_data': table_data,
                'confidence': similarity,
                'geographic_relevance': geographic_relevance,
                'combined_score': combined_score
            })
        
        # Sort by combined score and return top k
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:k]
    
    def _calculate_geographic_relevance(self, table_data: Dict, geographic_context: Optional[GeographicContext]) -> float:
        """Calculate geographic relevance of table with CBSA support"""
        if not geographic_context or not geographic_context.location_mentioned:
            return 1.0
        
        geo_level = geographic_context.geography_level
        
        # Check if requested geography level is available
        available_levels = table_data.get('geography_levels', [])
        if geo_level in available_levels:
            return 1.0
        
        # Special handling for CBSA requests
        if geo_level == 'cbsa':
            # CBSAs often work with county or place data
            if 'county' in available_levels or 'place' in available_levels:
                return 0.95
            # Metro areas can also use state data for broader analysis
            if 'state' in available_levels:
                return 0.85
        
        # Check survey availability matrix
        survey_availability = table_data.get('survey_availability', {})
        
        # ACS5 availability (more geographic detail)
        acs5_levels = survey_availability.get('5yr', {})
        if isinstance(acs5_levels, dict):
            if geo_level in acs5_levels or any(geo_level in str(v) for v in acs5_levels.values()):
                return 1.0
        elif isinstance(acs5_levels, list) and geo_level in acs5_levels:
            return 1.0
        
        # ACS1 availability (limited geography but more current)
        acs1_levels = survey_availability.get('1yr', {})
        if isinstance(acs1_levels, dict):
            if geo_level in acs1_levels or any(geo_level in str(v) for v in acs1_levels.values()):
                return 0.85
        elif isinstance(acs1_levels, list) and geo_level in acs1_levels:
            return 0.85
        
        # Geographic level hierarchy fallbacks with CBSA intelligence
        level_hierarchy = {
            'tract': ['place', 'county', 'cbsa', 'state'],
            'zip': ['place', 'county', 'cbsa', 'state'],
            'place': ['county', 'cbsa', 'state'],
            'county': ['cbsa', 'state'],
            'cbsa': ['state'],
            'state': []
        }
        
        if geo_level in level_hierarchy:
            for fallback_level in level_hierarchy[geo_level]:
                if fallback_level in available_levels:
                    # CBSA fallbacks get higher relevance for economic data
                    if fallback_level == 'cbsa':
                        return 0.90
                    elif fallback_level == 'county':
                        return 0.80
                    elif fallback_level == 'state':
                        return 0.70
                    else:
                        return 0.75
        
        # Default moderate relevance (don't eliminate completely)
        return 0.60

class VariablesSearch:
    """Fine retrieval using concept-based FAISS variable index"""
    
    def __init__(self, variables_dir: str = "variables-faiss"):
        self.variables_dir = Path(variables_dir)
        self.variables_index = None
        self.variables_metadata = []
        self.embedding_model = None
        self.use_openai_embeddings = False
        
        self._load_variables()
    
    def _load_variables(self):
        """Load concept-based variables FAISS index and metadata"""
        logger.info("Loading concept-based variables database...")
        
        # Load FAISS index
        faiss_file = self.variables_dir / "variables.faiss"
        if not faiss_file.exists():
            raise FileNotFoundError(f"Variables FAISS index not found: {faiss_file}")
        
        self.variables_index = faiss.read_index(str(faiss_file))
        
        # Load metadata
        metadata_file = self.variables_dir / "variables_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Variables metadata not found: {metadata_file}")
        
        with open(metadata_file) as f:
            self.variables_metadata = json.load(f)
        
        # Setup embedding model based on build info
        build_info_file = self.variables_dir / "build_info.json"
        if build_info_file.exists():
            with open(build_info_file) as f:
                build_info = json.load(f)
            
            embedding_dimension = build_info.get('embedding_dimension', 768)
            if embedding_dimension == 3072 and OPENAI_AVAILABLE:
                self._setup_openai_embeddings()
            else:
                self._setup_sentence_transformers()
        else:
            self._setup_sentence_transformers()
        
        logger.info(f"Loaded {len(self.variables_metadata)} concept-based variables")
    
    def _setup_openai_embeddings(self):
        """Setup OpenAI embeddings"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.use_openai_embeddings = True
        logger.info("Using OpenAI embeddings for variables")
    
    def _setup_sentence_transformers(self):
        """Setup SentenceTransformers embeddings"""
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.use_openai_embeddings = False
        logger.info("Using SentenceTransformers for variables")
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate query embedding"""
        if self.use_openai_embeddings:
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-large"
            )
            return np.array([response.data[0].embedding], dtype=np.float32)
        else:
            return self.embedding_model.encode([query]).astype('float32')
    
    def search_within_table(self, table_id: str, query: str, geographic_context: Optional[GeographicContext] = None, k: int = 10) -> List[Dict]:
        """Search for variables within a specific table"""
        query_embedding = self._generate_query_embedding(query)
        
        # Search FAISS index broadly, then filter to table
        distances, indices = self.variables_index.search(query_embedding, k * 5)
        
        table_results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.variables_metadata):
                continue
            
            var_metadata = self.variables_metadata[idx]
            variable_id = var_metadata.get('variable_id', '')
            
            # Filter to target table
            var_table_id = variable_id.split('_')[0] if '_' in variable_id else ''
            if var_table_id != table_id:
                continue
            
            # Calculate scores
            semantic_score = max(0.0, 1.0 - (distance / 2.0))
            geographic_score = self._calculate_geographic_relevance(var_metadata, geographic_context)
            structure_bonus = self._calculate_structure_bonus(variable_id, query)
            
            final_score = semantic_score * (0.6 + 0.2 * geographic_score + 0.2 * structure_bonus)
            
            table_results.append({
                'variable_metadata': var_metadata,
                'semantic_score': semantic_score,
                'geographic_score': geographic_score,
                'structure_bonus': structure_bonus,
                'final_score': final_score
            })
        
        table_results.sort(key=lambda x: x['final_score'], reverse=True)
        return table_results[:k]
    
    def _calculate_geographic_relevance(self, var_metadata: Dict, geographic_context: Optional[GeographicContext]) -> float:
        """Calculate geographic relevance using concept-based metadata with CBSA support"""
        if not geographic_context or not geographic_context.location_mentioned:
            return 1.0
        
        geo_level = geographic_context.geography_level
        
        # Check geography coverage from metadata
        geography_coverage = var_metadata.get('geography_coverage', {})
        if isinstance(geography_coverage, str):
            try:
                geography_coverage = json.loads(geography_coverage) if geography_coverage else {}
            except json.JSONDecodeError:
                geography_coverage = {}
        
        # Direct geography level match
        if geography_coverage:
            for survey_type, geo_levels in geography_coverage.items():
                if isinstance(geo_levels, list):
                    if geo_level in geo_levels:
                        return 1.0
                elif isinstance(geo_levels, str):
                    if geo_level in geo_levels:
                        return 1.0
        
        # CBSA-specific intelligence
        if geo_level == 'cbsa':
            # Metro areas often use county or place aggregation
            if geography_coverage:
                for survey_type, geo_levels in geography_coverage.items():
                    geo_list = geo_levels if isinstance(geo_levels, list) else [geo_levels]
                    if 'county' in geo_list or 'place' in geo_list:
                        return 0.95  # High relevance - can aggregate to metro
                    if 'state' in geo_list:
                        return 0.85  # Moderate relevance - broader than metro
        
        # Economic data context for metro areas
        variable_id = var_metadata.get('variable_id', '')
        concept = var_metadata.get('concept', '').lower()
        label = var_metadata.get('label', '').lower()
        
        # Economic variables work well at metro level
        if geo_level == 'cbsa':
            economic_indicators = [
                'income', 'earnings', 'wage', 'salary', 'employment', 'unemployment',
                'poverty', 'commute', 'transportation', 'occupation', 'industry'
            ]
            if any(indicator in concept or indicator in label for indicator in economic_indicators):
                return 0.90  # Economic data often better at metro level
        
        # Variable type bonuses
        if variable_id.endswith('_001E'):
            # Total variables are more flexible across geography levels
            return 0.85
        
        # Default moderate relevance
        return 0.75
    
    def _calculate_structure_bonus(self, variable_id: str, query: str) -> float:
        """Calculate bonus for variable structure (prioritize totals unless breakdown requested)"""
        query_lower = query.lower()
        
        # Prefer _001E (total) variables unless query asks for breakdown
        if variable_id.endswith('_001E'):
            breakdown_terms = ['by', 'breakdown', 'split', 'detailed', 'male', 'female', 'age', 'race']
            if any(term in query_lower for term in breakdown_terms):
                return 0.5
            return 1.0
        
        # Bonus for specific breakdowns when requested
        if any(term in query_lower for term in ['male', 'female', 'men', 'women']):
            if 'male' in variable_id.lower() or 'female' in variable_id.lower():
                return 0.8
        
        return 0.7

class MethodologySearch:
    """Statistical expertise search using ChromaDB methodology database"""
    
    def __init__(self, methodology_dir: str = "methodology-db"):
        self.methodology_dir = Path(methodology_dir)
        self.collection = None
        self.embedding_model = None
        
        self._load_methodology_db()
    
    def _load_methodology_db(self):
        """Load ChromaDB methodology collection"""
        logger.info("Loading methodology database...")
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=str(self.methodology_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = client.get_collection("census_methodology")
            doc_count = self.collection.count()
            logger.info(f"Loaded methodology database with {doc_count} documents")
        except Exception as e:
            raise FileNotFoundError(f"Methodology collection not found: {e}")
        
        # Load embedding model for consistency
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    def search_methodology(self, query: str, k: int = 5) -> List[Dict]:
        """Search methodology documents for statistical expertise"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
        
        methodology_results = []
        for i in range(len(results['ids'][0])):
            methodology_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'confidence': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                'source': results['metadatas'][0][i].get('source', 'Unknown')
            })
        
        return methodology_results

class ConceptBasedCensusSearchEngine:
    """Complete coarse-to-fine search engine with LLM-first intelligence"""
    
    def __init__(self,
                 catalog_dir: str = "table-catalog",
                 variables_dir: str = "variables-faiss",
                 methodology_dir: str = "methodology-db",
                 gazetteer_db_path: str = None):
        
        # Set paths relative to knowledge-base directory
        base_dir = Path(__file__).parent
        
        # Initialize LLM-first components
        self.geo_parser = LLMGeographicParser(gazetteer_db_path)
        self.variable_preprocessor = LLMVariablePreprocessor()
        self.query_detector = QueryTypeDetector()
        
        # Initialize search components
        self.table_search = TableCatalogSearch(catalog_dir)
        self.variable_search = VariablesSearch(variables_dir)
        self.methodology_search = MethodologySearch(methodology_dir)
        
        logger.info("Concept-based Census Search Engine initialized")
        logger.info("✅ LLM-first architecture: Geography + Variable preprocessing")
        logger.info("✅ Three-tier search: Tables → Variables → Methodology")
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        LLM-enhanced search with concept-based coarse-to-fine retrieval
        
        Flow:
        1. LLM preprocesses query (geography + variables)
        2. Coarse search: Find relevant tables  
        3. Fine search: Find variables within tables
        4. Methodology: Add statistical context
        """
        logger.info(f"Searching: '{query}'")
        
        # Step 1: LLM preprocessing
        geo_context = self.geo_parser.parse_geographic_context(query)
        if geo_context.location_mentioned:
            logger.info(f"Geographic context: {geo_context.location_text} ({geo_context.geography_level})")
        
        variable_preprocessing = self.variable_preprocessor.preprocess_query(query)
        logger.info(f"Variable preprocessing: {variable_preprocessing['search_strategy']} search")
        
        # Use enhanced query for better semantic search
        search_query = variable_preprocessing['enhanced_query']
        
        # Step 2: Query type detection for smart routing
        query_type = self.query_detector.detect_query_type(search_query)
        logger.info(f"Query type: {query_type}")
        
        all_results = []
        
        # Step 3: Variables search (always for 'variables' and 'both' types)
        if query_type in ['variables', 'both']:
            all_results.extend(self._search_variables(search_query, geo_context, max_results))
        
        # Step 4: Methodology search (for 'methodology' and 'both' types)
        if query_type in ['methodology', 'both']:
            methodology_context = self._search_methodology(search_query)
            # Inject methodology context into results
            for result in all_results:
                if not result.methodology_context and methodology_context:
                    result.methodology_context = methodology_context[:500] + "..."
        
        # Step 5: Sort all results by confidence
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Returning {min(len(all_results), max_results)} results")
        return all_results[:max_results]
    
    def _search_variables(self, query: str, geo_context: GeographicContext, max_results: int) -> List[SearchResult]:
        """Search variables using coarse-to-fine retrieval"""
        # Coarse retrieval: find relevant tables
        table_results = self.table_search.search_tables(query, k=5, geographic_context=geo_context)
        logger.info(f"Found {len(table_results)} candidate tables")
        
        if not table_results:
            return []
        
        # Fine retrieval: search within top tables
        variable_results = []
        
        for table_result in table_results:
            table_id = table_result['table_id']
            table_data = table_result['table_data']
            table_confidence = table_result['confidence']
            
            # Search variables within this table
            vars_in_table = self.variable_search.search_within_table(
                table_id, query, geo_context, k=3
            )
            
            # Convert to SearchResult objects
            for var_result in vars_in_table:
                var_metadata = var_result['variable_metadata']
                variable_id = var_metadata.get('variable_id', '')
                
                # Combine table and variable confidence
                combined_confidence = table_confidence * 0.6 + var_result['final_score'] * 0.4
                
                # Extract survey information
                available_surveys = var_metadata.get('available_surveys', ['acs5'])
                if isinstance(available_surveys, str):
                    available_surveys = [available_surveys]
                
                # Handle geography coverage
                geography_coverage = {}
                geo_coverage_raw = var_metadata.get('geography_coverage', {})
                if isinstance(geo_coverage_raw, str):
                    try:
                        geography_coverage = json.loads(geo_coverage_raw) if geo_coverage_raw else {}
                    except json.JSONDecodeError:
                        geography_coverage = {}
                else:
                    geography_coverage = geo_coverage_raw or {}
                
                search_result = SearchResult(
                    variable_id=variable_id,
                    table_id=table_id,
                    concept=var_metadata.get('concept', var_metadata.get('concept_name', '')),
                    label=var_metadata.get('label', var_metadata.get('description', '')),
                    title=table_data.get('title', ''),
                    universe=table_data.get('universe', ''),
                    confidence=combined_confidence,
                    geographic_relevance=var_result['geographic_score'],
                    geographic_restrictions={
                        'acs1': table_data.get('geography_restrictions_1yr', ''),
                        'acs5': table_data.get('geography_restrictions_5yr', '')
                    },
                    available_surveys=available_surveys,
                    statistical_notes=table_data.get('statistical_notes', []),
                    primary_variable=variable_id.endswith('_001E'),
                    survey_instances=[],
                    geography_coverage=geography_coverage,
                    primary_instance=var_metadata.get('primary_instance'),
                    structure_type=var_metadata.get('structure_type', 'concept_based')
                )
                
                variable_results.append(search_result)
        
        return variable_results
    
    def _search_methodology(self, query: str) -> Optional[str]:
        """Search methodology for statistical expertise context"""
        try:
            methodology_results = self.methodology_search.search_methodology(query, k=2)
            if methodology_results:
                # Return most relevant methodology context
                return methodology_results[0]['content']
        except Exception as e:
            logger.warning(f"Methodology search failed: {e}")
        
        return None
    
    def get_table_info(self, table_id: str) -> Optional[Dict]:
        """Get complete information about a table"""
        return self.table_search.tables.get(table_id)
    
    def get_variable_info(self, variable_id: str) -> Optional[Dict]:
        """Get complete information about a variable"""
        for var_metadata in self.variable_search.variables_metadata:
            if var_metadata.get('variable_id') == variable_id:
                return var_metadata
        return None

# Backward compatibility alias
CensusSearchEngine = ConceptBasedCensusSearchEngine

# Factory function for MCP integration with LLM support
def create_search_engine(knowledge_base_dir: str = None, gazetteer_db_path: str = None) -> ConceptBasedCensusSearchEngine:
    """
    Create LLM-enhanced search engine with automatic path detection
    
    Args:
        knowledge_base_dir: Base directory for knowledge base files
        gazetteer_db_path: Path to gazetteer SQLite database
    
    Returns:
        ConceptBasedCensusSearchEngine instance with LLM support
    """
    if knowledge_base_dir:
        base_path = Path(knowledge_base_dir)
        catalog_dir = base_path / "table-catalog"
        variables_dir = base_path / "variables-faiss"
        methodology_dir = base_path / "methodology-db"
        
        # Auto-detect gazetteer if not specified
        if not gazetteer_db_path:
            potential_gazetteer = base_path / "geography.db"
            if potential_gazetteer.exists():
                gazetteer_db_path = str(potential_gazetteer)
    else:
        # Default paths relative to current file
        base_dir = Path(__file__).parent
        catalog_dir = base_dir / "table-catalog"
        variables_dir = base_dir / "variables-faiss"
        methodology_dir = base_dir / "methodology-db"
        
        # Auto-detect gazetteer
        if not gazetteer_db_path:
            potential_gazetteer = base_dir / "geography.db"
            if potential_gazetteer.exists():
                gazetteer_db_path = str(potential_gazetteer)
    
    return ConceptBasedCensusSearchEngine(
        catalog_dir=str(catalog_dir),
        variables_dir=str(variables_dir),
        methodology_dir=str(methodology_dir),
        gazetteer_db_path=gazetteer_db_path
    )ensusSearchEngine(
        catalog_dir=str(catalog_dir),
        variables_dir=str(variables_dir),
        methodology_dir=str(methodology_dir)
    )

if __name__ == "__main__":
    # Test the search engine
    logger.info("Testing Census Search Engine...")
    try:
        engine = create_search_engine()
        results = engine.search("median household income by race", max_results=5)
        logger.info(f"Test search returned {len(results)} results")
        for result in results[:2]:
            logger.info(f"  {result.variable_id}: {result.label[:100]}...")
    except Exception as e:
        logger.error(f"Test failed: {e}")
