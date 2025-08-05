#!/usr/bin/env python3
"""
Variables Search Component - Clean extraction from kb_search.py

Fine retrieval using concept-based FAISS variable index with OpenAI embeddings.
Handles semantic search within specific tables and geographic relevance scoring.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Required dependencies
import faiss
from openai import OpenAI
import os

# Import geographic context
from geographic_parsing import GeographicContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIEmbeddings:
    """OpenAI embeddings for variable search"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "text-embedding-3-large"
        self.dimensions = 3072
        
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY required")
        
        logger.debug("OpenAI embeddings initialized: text-embedding-3-large (3072d)")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        embeddings = np.array([data.embedding for data in response.data], dtype=np.float32)
        return embeddings

class VariablesSearch:
    """
    Fine retrieval using concept-based FAISS variable index
    
    Provides semantic search within specific tables with geographic awareness
    and variable structure intelligence.
    """
    
    def __init__(self, variables_dir: str = "variables-db"):
        self.variables_dir = Path(variables_dir)
        self.variables_index = None
        self.variables_metadata = []
        self.embedding_model = OpenAIEmbeddings()
        
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
        
        # Verify build info shows OpenAI embeddings
        build_info_file = self.variables_dir / "build_info.json"
        if build_info_file.exists():
            with open(build_info_file) as f:
                build_info = json.load(f)
            
            embedding_dimension = build_info.get('embedding_dimension', 0)
            if embedding_dimension != 3072:
                logger.warning(f"Variables database dimension mismatch: expected 3072, got {embedding_dimension}")
            
            model_name = build_info.get('model', 'unknown')
            if 'text-embedding-3-large' not in model_name:
                logger.warning(f"Variables database built with {model_name}, expected text-embedding-3-large")
        
        logger.info(f"✅ Variables loaded: {len(self.variables_metadata)} variables, 3072d OpenAI embeddings")
    
    def search_within_table(self, table_id: str, query: str, 
                          geographic_context: Optional[GeographicContext] = None, 
                          k: int = 10) -> List[Dict]:
        """
        Search for variables within a specific table
        
        Args:
            table_id: Target table ID (e.g., "B01003")
            query: Search query text
            geographic_context: Geographic context for relevance scoring
            k: Number of results to return
            
        Returns:
            List of variable results with scores and metadata
        """
        query_embedding = self.embedding_model.encode([query])
        
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
    
    def search_variables_global(self, query: str, 
                              geographic_context: Optional[GeographicContext] = None,
                              k: int = 20) -> List[Dict]:
        """
        Search across all variables globally (not table-specific)
        
        Args:
            query: Search query text
            geographic_context: Geographic context for relevance scoring
            k: Number of results to return
            
        Returns:
            List of variable results with scores and metadata
        """
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS index globally
        distances, indices = self.variables_index.search(query_embedding, k * 2)
        
        global_results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.variables_metadata):
                continue
            
            var_metadata = self.variables_metadata[idx]
            variable_id = var_metadata.get('variable_id', '')
            
            # Calculate scores
            semantic_score = max(0.0, 1.0 - (distance / 2.0))
            geographic_score = self._calculate_geographic_relevance(var_metadata, geographic_context)
            structure_bonus = self._calculate_structure_bonus(variable_id, query)
            
            final_score = semantic_score * (0.6 + 0.2 * geographic_score + 0.2 * structure_bonus)
            
            global_results.append({
                'variable_metadata': var_metadata,
                'semantic_score': semantic_score,
                'geographic_score': geographic_score,
                'structure_bonus': structure_bonus,
                'final_score': final_score
            })
        
        global_results.sort(key=lambda x: x['final_score'], reverse=True)
        return global_results[:k]
    
    def _calculate_geographic_relevance(self, var_metadata: Dict, 
                                      geographic_context: Optional[GeographicContext]) -> float:
        """Calculate geographic relevance using concept-based metadata"""
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
            if geography_coverage:
                for survey_type, geo_levels in geography_coverage.items():
                    geo_list = geo_levels if isinstance(geo_levels, list) else [geo_levels]
                    if 'county' in geo_list or 'place' in geo_list:
                        return 0.95
                    if 'state' in geo_list:
                        return 0.85
        
        # Economic data context for metro areas
        variable_id = var_metadata.get('variable_id', '')
        concept = var_metadata.get('concept', '').lower()
        label = var_metadata.get('label', '').lower()
        
        if geo_level == 'cbsa':
            economic_indicators = [
                'income', 'earnings', 'wage', 'salary', 'employment', 'unemployment',
                'poverty', 'commute', 'transportation', 'occupation', 'industry'
            ]
            if any(indicator in concept or indicator in label for indicator in economic_indicators):
                return 0.90
        
        # Primary variable bonus
        if variable_id.endswith('_001E'):
            return 0.85
        
        return 0.75
    
    def _calculate_structure_bonus(self, variable_id: str, query: str) -> float:
        """Calculate bonus for variable structure"""
        query_lower = query.lower()
        
        # Primary variables (_001E) are typically totals
        if variable_id.endswith('_001E'):
            breakdown_terms = ['by', 'breakdown', 'split', 'detailed', 'male', 'female', 'age', 'race']
            if any(term in query_lower for term in breakdown_terms):
                return 0.5  # User wants breakdown, not total
            return 1.0  # User wants total, perfect match
        
        # Gender-specific variables
        if any(term in query_lower for term in ['male', 'female', 'men', 'women']):
            if 'male' in variable_id.lower() or 'female' in variable_id.lower():
                return 0.8
        
        # Age-specific variables
        if any(term in query_lower for term in ['age', 'years old', 'elderly', 'senior']):
            if any(age_term in variable_id.lower() for age_term in ['age', 'years']):
                return 0.8
        
        return 0.7
    
    def get_variable_info(self, variable_id: str) -> Optional[Dict]:
        """Get complete information about a specific variable"""
        for var_metadata in self.variables_metadata:
            if var_metadata.get('variable_id') == variable_id:
                return var_metadata
        return None
    
    def get_table_variables(self, table_id: str) -> List[Dict]:
        """Get all variables for a specific table"""
        table_variables = []
        for var_metadata in self.variables_metadata:
            variable_id = var_metadata.get('variable_id', '')
            var_table_id = variable_id.split('_')[0] if '_' in variable_id else ''
            if var_table_id == table_id:
                table_variables.append(var_metadata)
        
        # Sort by variable ID for consistent ordering
        table_variables.sort(key=lambda x: x.get('variable_id', ''))
        return table_variables
    
    def get_stats(self) -> Dict:
        """Get statistics about the variables database"""
        if not self.variables_metadata:
            return {}
        
        # Count by table
        table_counts = {}
        for var_metadata in self.variables_metadata:
            variable_id = var_metadata.get('variable_id', '')
            table_id = variable_id.split('_')[0] if '_' in variable_id else 'unknown'
            table_counts[table_id] = table_counts.get(table_id, 0) + 1
        
        return {
            'total_variables': len(self.variables_metadata),
            'total_tables': len(table_counts),
            'largest_table': max(table_counts.items(), key=lambda x: x[1]) if table_counts else None,
            'embedding_dimensions': 3072,
            'index_type': 'FAISS with OpenAI embeddings'
        }

# Factory function for easy integration
def create_variables_search(variables_dir: str = None) -> VariablesSearch:
    """Create VariablesSearch instance with path resolution"""
    if variables_dir:
        return VariablesSearch(variables_dir)
    
    # Default path relative to current file
    base_dir = Path(__file__).parent
    default_dir = base_dir / "variables-db"
    
    return VariablesSearch(str(default_dir))

if __name__ == "__main__":
    # Test the variables search
    try:
        variables = create_variables_search()
        stats = variables.get_stats()
        logger.info(f"Variables database stats: {stats}")
        
        # Test search within table
        results = variables.search_within_table("B01003", "total population", k=3)
        logger.info(f"Found {len(results)} variables for 'total population' in B01003")
        
        for result in results[:2]:
            var_id = result['variable_metadata'].get('variable_id', '')
            score = result['final_score']
            logger.info(f"  {var_id}: {score:.3f}")
            
    except Exception as e:
        logger.error(f"Variables search test failed: {e}")
