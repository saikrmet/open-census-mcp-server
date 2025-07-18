#!/usr/bin/env python3
"""
Coarse-to-Fine Census Variable Search with Geographic Intelligence

Implements the complete search system using:
1. Table catalog for coarse retrieval (concept → table)
2. Canonical variables for fine retrieval (table → variable)
3. Geographic intelligence for spatial validation and relevance

Usage:
    from coarse_fine_search import CensusSearchEngine
    
    engine = CensusSearchEngine()
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with complete metadata"""
    variable_id: str
    table_id: str
    concept: str
    label: str
    title: str  # Official table title
    universe: str  # Official universe
    confidence: float
    geographic_relevance: float
    geographic_restrictions: Dict[str, str]
    available_surveys: List[str]
    statistical_notes: List[str]
    primary_variable: bool
    methodology_context: Optional[str] = None

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

class TableCatalogSearch:
    """Coarse retrieval using table catalog"""
    
    def __init__(self, catalog_dir: str = "table-catalog"):
        self.catalog_dir = Path(catalog_dir)
        self.tables = None
        self.embeddings_index = None
        self.table_ids = None
        self.embedding_model = None
        
        self._load_catalog()
    
    def _load_catalog(self):
        """Load table catalog and embeddings"""
        logger.info("Loading table catalog...")
        
        # Load table catalog
        catalog_file = self.catalog_dir / "table_catalog.json"
        with open(catalog_file) as f:
            catalog_data = json.load(f)
        
        self.tables = {table['table_id']: table for table in catalog_data['tables']}
        logger.info(f"Loaded {len(self.tables)} tables")
        
        # Load FAISS embeddings
        faiss_file = self.catalog_dir / "table_embeddings.faiss"
        self.embeddings_index = faiss.read_index(str(faiss_file))
        
        # Load table ID mapping
        mapping_file = self.catalog_dir / "table_mapping.json"
        with open(mapping_file) as f:
            mapping_data = json.load(f)
        self.table_ids = mapping_data['table_ids']
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        logger.info(f"Loaded FAISS index with {len(self.table_ids)} embeddings")
    
    def search_tables(self, query: str, k: int = 5, 
                     geographic_context: Optional[GeographicContext] = None) -> List[Dict]:
        """Search for relevant tables using coarse retrieval"""
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.embeddings_index.search(query_embedding, k * 2)  # Get extra for filtering
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            table_id = self.table_ids[idx]
            table_data = self.tables[table_id]
            
            # Calculate confidence (convert L2 distance to similarity)
            confidence = max(0.0, 1.0 - (distance / 2.0))
            
            # Apply geographic filtering if context provided
            geographic_relevance = 1.0
            if geographic_context and geographic_context.location_mentioned:
                geographic_relevance = self._calculate_geographic_relevance(
                    table_data, geographic_context
                )
            
            # Combined score
            combined_score = confidence * (0.7 + 0.3 * geographic_relevance)
            
            results.append({
                'table_id': table_id,
                'table_data': table_data,
                'confidence': confidence,
                'geographic_relevance': geographic_relevance,
                'combined_score': combined_score
            })
        
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
        
        # Penalty for unavailable geography, but don't eliminate
        return 0.6

class VariableSearch:
    """Fine retrieval within table families"""
    
    def __init__(self, canonical_path: str = "source-docs/canonical_variables.json",
                 geo_scalars_path: str = "concepts/geo_similarity_scalars.json"):
        self.canonical_path = Path(canonical_path)
        self.geo_scalars_path = Path(geo_scalars_path)
        self.variables = None
        self.geo_scalars = None
        
        self._load_data()
    
    def _load_data(self):
        """Load canonical variables and geographic scalars"""
        logger.info("Loading canonical variables...")
        
        # Load canonical variables
        with open(self.canonical_path) as f:
            data = json.load(f)
        self.variables = data.get('variables', data)
        
        # Load geographic scalars
        try:
            with open(self.geo_scalars_path) as f:
                self.geo_scalars = json.load(f)
            logger.info(f"Loaded geographic scalars for {len(self.geo_scalars)} variables")
        except FileNotFoundError:
            logger.warning(f"Geographic scalars not found at {self.geo_scalars_path}")
            self.geo_scalars = {}
        
        logger.info(f"Loaded {len(self.variables)} canonical variables")
    
    def search_within_table(self, table_id: str, query: str, 
                          geographic_context: Optional[GeographicContext] = None,
                          k: int = 10) -> List[Dict]:
        """Search for variables within a specific table"""
        
        # Get all variables for this table
        table_variables = {}
        for temporal_id, var_data in self.variables.items():
            var_table_id = var_data.get('table_id', var_data.get('variable_id', '').split('_')[0])
            if var_table_id == table_id:
                table_variables[temporal_id] = var_data
        
        if not table_variables:
            return []
        
        # Score variables within table
        scored_variables = []
        query_lower = query.lower()
        
        for temporal_id, var_data in table_variables.items():
            variable_id = var_data.get('variable_id', temporal_id)
            
            # Calculate semantic relevance
            semantic_score = self._calculate_semantic_relevance(var_data, query_lower)
            
            # Calculate geographic relevance
            geographic_score = 1.0
            if geographic_context and geographic_context.location_mentioned:
                geographic_score = self._calculate_variable_geographic_relevance(
                    variable_id, geographic_context
                )
            
            # Prioritize _001E (total) variables unless query is specific
            structure_bonus = self._calculate_structure_bonus(variable_id, query_lower)
            
            # Combined score
            final_score = semantic_score * (0.6 + 0.2 * geographic_score + 0.2 * structure_bonus)
            
            scored_variables.append({
                'temporal_id': temporal_id,
                'variable_data': var_data,
                'semantic_score': semantic_score,
                'geographic_score': geographic_score,
                'structure_bonus': structure_bonus,
                'final_score': final_score
            })
        
        # Sort by final score
        scored_variables.sort(key=lambda x: x['final_score'], reverse=True)
        
        return scored_variables[:k]
    
    def _calculate_semantic_relevance(self, var_data: Dict, query_lower: str) -> float:
        """Calculate semantic relevance of variable to query"""
        score = 0.0
        
        # Check label
        label = var_data.get('label', '').lower()
        if any(word in label for word in query_lower.split()):
            score += 0.3
        
        # Check concept
        concept = var_data.get('concept', '').lower()
        if any(word in concept for word in query_lower.split()):
            score += 0.4
        
        # Check enrichment text (sample for performance)
        enrichment = var_data.get('enrichment_text', '')[:500].lower()
        query_words = query_lower.split()
        if len(query_words) > 0:
            matches = sum(1 for word in query_words if word in enrichment)
            score += 0.3 * (matches / len(query_words))
        
        return min(score, 1.0)
    
    def _calculate_variable_geographic_relevance(self, variable_id: str, 
                                               geographic_context: GeographicContext) -> float:
        """Calculate geographic relevance using geo scalars"""
        if not self.geo_scalars:
            return 1.0
        
        # Get geographic scalar for this variable
        geo_score = self.geo_scalars.get(variable_id, 0.0)
        
        # Higher geo scores are more relevant for geographic queries
        if geographic_context.location_mentioned:
            return min(geo_score * 2.0, 1.0)  # Boost geographic variables
        
        return 1.0
    
    def _calculate_structure_bonus(self, variable_id: str, query_lower: str) -> float:
        """Calculate bonus for variable structure (prioritize totals)"""
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

class CensusSearchEngine:
    """Complete coarse-to-fine search engine with geographic intelligence"""
    
    def __init__(self, catalog_dir: str = "table-catalog",
                 canonical_path: str = "source-docs/canonical_variables.json",
                 geo_scalars_path: str = "concepts/geo_similarity_scalars.json"):
        
        self.geo_parser = GeographicParser()
        self.table_search = TableCatalogSearch(catalog_dir)
        self.variable_search = VariableSearch(canonical_path, geo_scalars_path)
        
        logger.info("Census Search Engine initialized")
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Complete search with coarse-to-fine retrieval"""
        logger.info(f"Searching: '{query}'")
        
        # Parse geographic context
        geo_context = self.geo_parser.parse_geographic_context(query)
        if geo_context.location_mentioned:
            logger.info(f"Geographic context: {geo_context.location_text} ({geo_context.geography_level})")
        
        # Coarse retrieval: find relevant tables
        table_results = self.table_search.search_tables(query, k=5, geographic_context=geo_context)
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
                var_data = var_result['variable_data']
                variable_id = var_data.get('variable_id', var_result['temporal_id'])
                
                # Combine table and variable confidence
                combined_confidence = table_confidence * 0.6 + var_result['final_score'] * 0.4
                
                # Determine if this is the primary variable
                is_primary = (variable_id == table_data.get('primary_variable') or 
                            variable_id.endswith('_001E'))
                
                search_result = SearchResult(
                    variable_id=variable_id,
                    table_id=table_id,
                    concept=var_data.get('concept', ''),
                    label=var_data.get('label', ''),
                    title=table_data.get('title', ''),
                    universe=table_data.get('universe', ''),
                    confidence=combined_confidence,
                    geographic_relevance=var_result['geographic_score'],
                    geographic_restrictions={
                        'acs1': table_data.get('geography_restrictions_1yr', ''),
                        'acs5': table_data.get('geography_restrictions_5yr', '')
                    },
                    available_surveys=table_data.get('survey_programs', []),
                    statistical_notes=table_data.get('statistical_notes', []),
                    primary_variable=is_primary
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
        """Get complete information about a variable"""
        for temporal_id, var_data in self.variable_search.variables.items():
            if var_data.get('variable_id') == variable_id:
                return var_data
        return None

def test_search_engine():
    """Test the search engine with sample queries"""
    logger.info("Testing Census Search Engine...")
    
    engine = CensusSearchEngine()
    
    test_queries = [
        "commute time in Richmond, VA",
        "median household income",
        "poverty rate in Detroit",
        "housing tenure by race",
        "travel time to work",
        "population by age and sex"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        results = engine.search(query, max_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.variable_id} (confidence: {result.confidence:.3f})")
            print(f"   Table: {result.table_id} - {result.title}")
            print(f"   Universe: {result.universe}")
            print(f"   Label: {result.label}")
            print(f"   Geographic relevance: {result.geographic_relevance:.3f}")
            print(f"   Primary variable: {result.primary_variable}")
            if result.statistical_notes:
                print(f"   Notes: {', '.join(result.statistical_notes[:2])}")

if __name__ == "__main__":
    test_search_engine()
