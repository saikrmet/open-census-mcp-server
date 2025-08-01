#!/usr/bin/env python3
"""
Coarse-to-Fine Census Variable Search with Concept-Based Intelligence

Updated to use concept-based canonical variables structure:
- Eliminates duplicate variables (65K → 36K concepts)
- Survey instance awareness (ACS1/5yr metadata)
- Clean table catalog with survey intelligence
- FAISS-based variable search for performance

Architecture:
1. Table catalog for coarse retrieval (concept → table)
2. FAISS variable index for fine retrieval (table → variable)
3. Geographic intelligence for spatial validation and relevance

Usage:
    from kb_search import ConceptBasedCensusSearchEngine
    
    engine = ConceptBasedCensusSearchEngine()
    results = engine.search("commute time in Richmond, VA")
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# OpenAI for embeddings (optional)
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
    # Look for .env in parent directory
    env_path = Path(__file__).parent.parent / '.env'
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
    title: str  # Official table title
    universe: str  # Official universe
    confidence: float
    geographic_relevance: float
    geographic_restrictions: Dict[str, str]  # {acs1: restrictions, acs5: restrictions}
    available_surveys: List[str]  # [acs1, acs5]
    statistical_notes: List[str]
    primary_variable: bool
    methodology_context: Optional[str] = None
    # New concept-based fields
    survey_instances: List[Dict] = None
    geography_coverage: Dict[str, List[str]] = None
    primary_instance: Optional[str] = None
    structure_type: Optional[str] = None

@dataclass
class GeographicContext:
    """Parsed geographic context from query"""
    location_mentioned: bool
    location_text: Optional[str]
    geography_level: Optional[str]  # state, county, place, tract
    confidence: float

class GeographicParser:
    """Extracts and validates geographic context from queries"""
    
    def __init__(self):
        # Geographic patterns for detection
        self.location_patterns = [
            r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "in Virginia"
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})',  # "Richmond, VA"
            r'\b([A-Z][a-z]+\s+County)\b',  # "Fairfax County"
            r'zip\s*code\s*(\d{5})',  # "zip code 23220"
            r'\b(\d{5})\b',  # "23220"
        ]
        
        self.geography_indicators = {
            'state': ['state', 'statewide'],
            'county': ['county', 'counties'],
            'place': ['city', 'cities', 'town', 'place'],
            'tract': ['tract', 'neighborhood'],
            'zip': ['zip', 'zipcode', 'postal'],
            'metro': ['metro', 'metropolitan', 'msa']
        }
    
    def parse_geographic_context(self, query: str) -> GeographicContext:
        """Extract geographic context from query"""
        query_lower = query.lower()
        
        # Check for location mentions
        location_text = None
        for pattern in self.location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location_text = match.group(0)
                break
        
        # Determine geography level
        geography_level = None
        for level, indicators in self.geography_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                geography_level = level
                break
        
        # Default to place level if location mentioned but no level specified
        if location_text and not geography_level:
            geography_level = 'place'
        
        return GeographicContext(
            location_mentioned=bool(location_text),
            location_text=location_text,
            geography_level=geography_level,
            confidence=0.8 if location_text else 0.0
        )

class ConceptBasedTableCatalogSearch:
    """Coarse retrieval using concept-based table catalog"""
    
    def __init__(self, catalog_dir: str = "table-catalog"):
        self.catalog_dir = Path(catalog_dir)
        self.tables = None
        self.embeddings_index = None
        self.table_ids = None
        self.embedding_model = None
        
        self._load_catalog()
    
    def _load_catalog(self):
        """Load concept-based table catalog and embeddings"""
        logger.info("Loading concept-based table catalog...")
        
        # Load table catalog
        catalog_file = self.catalog_dir / "table_catalog_enhanced.json"
        if not catalog_file.exists():
            raise FileNotFoundError(f"Enhanced table catalog not found: {catalog_file}")
        
        with open(catalog_file) as f:
            catalog_data = json.load(f)
        
        # Validate catalog structure
        if 'tables' not in catalog_data:
            raise ValueError("Invalid catalog structure: missing 'tables' key")
        
        self.tables = {table['table_id']: table for table in catalog_data['tables']}
        
        # Check for concept-based improvements
        metadata = catalog_data.get('metadata', {})
        model_version = metadata.get('model_version', 'unknown')
        
        logger.info(f"Loaded {len(self.tables)} tables (catalog version: {model_version})")
        
        # Load FAISS embeddings
        faiss_file = self.catalog_dir / "table_embeddings_enhanced.faiss"
        if not faiss_file.exists():
            raise FileNotFoundError(f"Enhanced FAISS index not found: {faiss_file}")
        
        self.embeddings_index = faiss.read_index(str(faiss_file))
        
        # Load table ID mapping and detect embedding type
        mapping_file = self.catalog_dir / "table_mapping_enhanced.json"
        if not mapping_file.exists():
            raise FileNotFoundError(f"Enhanced table mapping not found: {mapping_file}")
        
        with open(mapping_file) as f:
            mapping_data = json.load(f)
        self.table_ids = mapping_data['table_ids']
        
        # Detect embedding type and initialize appropriate model
        embedding_type = mapping_data.get('embedding_type', 'sentence_transformers')
        embedding_dimension = mapping_data.get('embedding_dimension', 768)
        
        if embedding_type == 'openai' or embedding_dimension == 3072:
            # Use OpenAI embeddings
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package required for OpenAI table embeddings. Install with: pip install openai")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required for OpenAI embeddings")
            self.openai_client = OpenAI(api_key=api_key)
            self.use_openai_embeddings = True
            logger.info(f"Using OpenAI embeddings for table search (dimension: {embedding_dimension})")
        else:
            # Use SentenceTransformers
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.use_openai_embeddings = False
            logger.info(f"Using SentenceTransformers for table search (dimension: {embedding_dimension})")
        
        logger.info(f"Loaded FAISS index with {len(self.table_ids)} embeddings")
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate query embedding using the same model as the index"""
        if self.use_openai_embeddings:
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-large"
            )
            embedding = np.array([response.data[0].embedding], dtype=np.float32)
            return embedding
        else:
            return self.embedding_model.encode([query]).astype('float32')
    
    def search_tables(self, query: str, k: int = 5,
                     geographic_context: Optional[GeographicContext] = None) -> List[Dict]:
        """Search for relevant tables using FAISS index"""
        
        # Generate query embedding using the same model as the index
        query_embedding = self._generate_query_embedding(query)
        
        # Search FAISS index for similar tables
        distances, indices = self.embeddings_index.search(query_embedding, k * 2)  # Get extra for filtering
        
        # Convert results to table data
        table_scores = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.table_ids):
                continue
                
            table_id = self.table_ids[idx]
            table_data = self.tables.get(table_id)
            
            if not table_data:
                logger.warning(f"Table {table_id} not found in catalog")
                continue
            
            # Convert L2 distance to similarity score (0-1)
            # L2 distance of 0 = perfect match, larger distances = less similar
            similarity = max(0.0, 1.0 - (distance / 2.0))  # Normalize distance to 0-1 range
            
            table_scores.append({
                'table_id': table_id,
                'table_data': table_data,
                'confidence': similarity,
                'distance': distance
            })
        
        # Apply geographic filtering and return top k
        results = []
        for table_result in table_scores[:k * 2]:  # Get extra for filtering
            try:
                # Apply geographic filtering if context provided
                geographic_relevance = 1.0
                if geographic_context and geographic_context.location_mentioned:
                    geographic_relevance = self._calculate_geographic_relevance(
                        table_result['table_data'], geographic_context
                    )
                
                # Combined score
                combined_score = table_result['confidence'] * (0.7 + 0.3 * geographic_relevance)
                
                table_result['geographic_relevance'] = geographic_relevance
                table_result['combined_score'] = combined_score
                results.append(table_result)
                
            except Exception as e:
                logger.warning(f"Error calculating geographic relevance: {e}")
                continue
        
        # Sort by combined score and return top k
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:k]
    
    def _calculate_geographic_relevance(self, table_data: Dict,
                                      geographic_context: GeographicContext) -> float:
        """Calculate geographic relevance of table for query context"""
        if not geographic_context.geography_level:
            return 1.0
        
        # Check if requested geography level is available
        available_levels = table_data.get('geography_levels', [])
        
        if geographic_context.geography_level in available_levels:
            return 1.0
        
        # Check survey availability matrix for more detailed geography info
        survey_availability = table_data.get('survey_availability', {})
        
        # If ACS5 has the geography level, it's available
        if geographic_context.geography_level in survey_availability.get('5yr', {}):
            return 1.0
        
        # If only ACS1 has it, partial relevance
        if geographic_context.geography_level in survey_availability.get('1yr', {}):
            return 0.8
        
        # Penalty for unavailable geography, but don't eliminate
        return 0.6

class ConceptBasedVariableSearch:
    """Fine retrieval using concept-based FAISS variable index"""
    
    def __init__(self, variables_dir: str = "variables-db"):
        self.variables_dir = Path(variables_dir)
        self.variables_index = None
        self.variables_metadata = None
        self.embedding_model = None
        
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
        
        # Load build info to understand structure and embedding model
        build_info_file = self.variables_dir / "build_info.json"
        self.use_openai_embeddings = False
        
        if build_info_file.exists():
            with open(build_info_file) as f:
                build_info = json.load(f)
            structure_type = build_info.get('structure_type', 'unknown')
            embedding_dimension = build_info.get('embedding_dimension', 768)
            
            # Detect if OpenAI embeddings were used (3072 dimensions)
            if embedding_dimension == 3072:
                self.use_openai_embeddings = True
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI package required to query OpenAI-embedded index. Install with: pip install openai")
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable required for OpenAI embeddings")
                self.openai_client = OpenAI(api_key=api_key)
                logger.info(f"Using OpenAI embeddings (dimension: {embedding_dimension})")
            else:
                self.use_openai_embeddings = False
                self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                logger.info(f"Using SentenceTransformers (dimension: {embedding_dimension})")
            
            logger.info(f"Variables database structure: {structure_type}")
        else:
            # Fallback to sentence transformers
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("No build info found, defaulting to SentenceTransformers")
        
        logger.info(f"Loaded {len(self.variables_metadata)} concept-based variables")
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate query embedding using the same model as the index"""
        if self.use_openai_embeddings:
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-large"
            )
            embedding = np.array([response.data[0].embedding], dtype=np.float32)
            return embedding
        else:
            return self.embedding_model.encode([query]).astype('float32')
    
    def search_within_table(self, table_id: str, query: str,
                          geographic_context: Optional[GeographicContext] = None,
                          k: int = 10) -> List[Dict]:
        """Search for variables within a specific table using concept-based index"""
        
        # Create query embedding using the same model as the index
        query_embedding = self._generate_query_embedding(query)
        
        # Search FAISS index for broader results
        distances, indices = self.variables_index.search(query_embedding, k * 5)  # Get extra for table filtering
        
        # Filter results to only include variables from the specified table
        table_results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.variables_metadata):
                continue
            
            var_metadata = self.variables_metadata[idx]
            variable_id = var_metadata.get('variable_id', '')
            
            # Extract table ID from variable ID
            var_table_id = variable_id.split('_')[0] if '_' in variable_id else ''
            
            # Skip if not from the target table
            if var_table_id != table_id:
                continue
            
            # Calculate semantic score (convert L2 distance to similarity)
            semantic_score = max(0.0, 1.0 - (distance / 2.0))
            
            # Calculate geographic relevance
            geographic_score = 1.0
            if geographic_context and geographic_context.location_mentioned:
                geographic_score = self._calculate_variable_geographic_relevance(
                    var_metadata, geographic_context
                )
            
            # Prioritize _001E (total) variables unless query is specific
            structure_bonus = self._calculate_structure_bonus(
                variable_id, query
            )
            
            # Combined score
            final_score = semantic_score * (0.6 + 0.2 * geographic_score + 0.2 * structure_bonus)
            
            table_results.append({
                'variable_metadata': var_metadata,
                'semantic_score': semantic_score,
                'geographic_score': geographic_score,
                'structure_bonus': structure_bonus,
                'final_score': final_score,
                'distance': distance
            })
        
        # Sort by final score and return top k
        table_results.sort(key=lambda x: x['final_score'], reverse=True)
        return table_results[:k]
    
    def _calculate_variable_geographic_relevance(self, var_metadata: Dict,
                                               geographic_context: GeographicContext) -> float:
        """Calculate geographic relevance using concept-based metadata"""
        if not geographic_context.location_mentioned:
            return 1.0
        
        # Check geography coverage from concept-based metadata
        geography_coverage_raw = var_metadata.get('geography_coverage', {})
        
        # Handle geography coverage (might be string or dict)
        if isinstance(geography_coverage_raw, str):
            try:
                geography_coverage = json.loads(geography_coverage_raw) if geography_coverage_raw else {}
            except json.JSONDecodeError:
                geography_coverage = {}
        else:
            geography_coverage = geography_coverage_raw or {}
        
        # If we have geography coverage info, use it
        if geography_coverage:
            # Check if requested geography level is available in any survey
            geo_level = geographic_context.geography_level
            if geo_level:
                for survey_type, geo_levels in geography_coverage.items():
                    if isinstance(geo_levels, list) and geo_level in geo_levels:
                        return 1.0
                    elif isinstance(geo_levels, str) and geo_level in geo_levels:
                        return 1.0
        
        # Default to moderate relevance
        return 0.8
    
    def _calculate_structure_bonus(self, variable_id: str, query: str) -> float:
        """Calculate bonus for variable structure (prioritize totals)"""
        query_lower = query.lower()
        
        # Prefer _001E (total) variables unless query asks for breakdown
        if variable_id.endswith('_001E'):
            # Check if query asks for specific breakdown
            breakdown_terms = ['by', 'breakdown', 'split', 'detailed', 'male', 'female', 'age', 'race']
            if any(term in query_lower for term in breakdown_terms):
                return 0.5  # Reduced bonus for totals when breakdown requested
            return 1.0  # Full bonus for totals
        
        # Check if this variable matches requested breakdown
        if any(term in query_lower for term in ['male', 'female', 'men', 'women']):
            if 'male' in variable_id.lower() or 'female' in variable_id.lower():
                return 0.8
        
        return 0.7  # Default for non-total variables

class ConceptBasedCensusSearchEngine:
    """Complete coarse-to-fine search engine with concept-based intelligence"""
    
    def __init__(self, catalog_dir: str = "table-catalog",
                 variables_dir: str = "variables-db"):
        
        self.geo_parser = GeographicParser()
        self.table_search = ConceptBasedTableCatalogSearch(catalog_dir)
        self.variable_search = ConceptBasedVariableSearch(variables_dir)
        
        logger.info("Concept-based Census Search Engine initialized")
        logger.info("🎯 Key improvements: No duplicate variables, survey instance awareness")
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Complete search with concept-based coarse-to-fine retrieval"""
        logger.info(f"Searching: '{query}'")
        
        # Parse geographic context
        geo_context = self.geo_parser.parse_geographic_context(query)
        if geo_context.location_mentioned:
            logger.info(f"Geographic context: {geo_context.location_text} ({geo_context.geography_level})")
        
        # Coarse retrieval: find relevant tables
        table_results = self.table_search.search_tables(query, k=10, geographic_context=geo_context)  # Increased from 5 to 10
        logger.info(f"Found {len(table_results)} candidate tables")
        
        # Fine retrieval: search within top tables
        all_results = []
        
        for table_result in table_results:
            table_id = table_result['table_id']
            table_data = table_result['table_data']
            table_confidence = table_result['confidence']
            
            # Search variables within this table
            variable_results = self.variable_search.search_within_table(
                table_id, query, geo_context, k=5
            )
            
            # Convert to SearchResult objects
            for var_result in variable_results:
                var_metadata = var_result['variable_metadata']
                variable_id = var_metadata.get('variable_id', '')
                
                # Combine table and variable confidence
                combined_confidence = table_confidence * 0.6 + var_result['final_score'] * 0.4
                
                # Determine if this is the primary variable
                is_primary = (variable_id == table_data.get('primary_variable') or
                            variable_id.endswith('_001E'))
                
                # Extract survey instance information
                survey_instances = []
                geography_coverage = {}
                primary_instance = None
                
                # Handle concept-based metadata
                if var_metadata.get('structure_type') == 'concept_based':
                    # This is from the concept-based structure
                    available_surveys = var_metadata.get('available_surveys', [])
                    
                    # Handle geography_coverage (might be string or dict)
                    geo_coverage_raw = var_metadata.get('geography_coverage', {})
                    if isinstance(geo_coverage_raw, str):
                        try:
                            geography_coverage = json.loads(geo_coverage_raw) if geo_coverage_raw else {}
                        except json.JSONDecodeError:
                            geography_coverage = {}
                    else:
                        geography_coverage = geo_coverage_raw or {}
                    
                    primary_instance = var_metadata.get('primary_instance')
                else:
                    # This is from the original structure
                    available_surveys = ['acs5']  # Default assumption
                
                # Create geographic restrictions for backward compatibility
                geographic_restrictions = {
                    'acs1': table_data.get('geography_restrictions_1yr', ''),
                    'acs5': table_data.get('geography_restrictions_5yr', '')
                }
                
                search_result = SearchResult(
                    variable_id=variable_id,
                    table_id=table_id,
                    concept=var_metadata.get('concept_name', var_metadata.get('concept', '')),  # Use concept_name first
                    label=var_metadata.get('description', var_metadata.get('label', '')),  # Use description first
                    title=table_data.get('title', ''),
                    universe=table_data.get('universe', ''),
                    confidence=combined_confidence,
                    geographic_relevance=var_result['geographic_score'],
                    geographic_restrictions=geographic_restrictions,
                    available_surveys=available_surveys,
                    statistical_notes=table_data.get('statistical_notes', []),
                    primary_variable=is_primary,
                    # New concept-based fields
                    survey_instances=survey_instances,
                    geography_coverage=geography_coverage,
                    primary_instance=primary_instance,
                    structure_type=var_metadata.get('structure_type', 'unknown')
                )
                
                all_results.append(search_result)
        
        # Sort all results by confidence and return top results
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Returning {min(len(all_results), max_results)} results")
        return all_results[:max_results]
    
    def get_table_info(self, table_id: str) -> Optional[Dict]:
        """Get complete information about a table"""
        return self.table_search.tables.get(table_id)
    
    def get_variable_info(self, variable_id: str) -> Optional[Dict]:
        """Get complete information about a variable from concept-based metadata"""
        # Search through metadata for the variable
        for var_metadata in self.variable_search.variables_metadata:
            if var_metadata.get('variable_id') == variable_id:
                return var_metadata
        return None

# Backward compatibility alias
CensusSearchEngine = ConceptBasedCensusSearchEngine
