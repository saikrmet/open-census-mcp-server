#!/usr/bin/env python3
"""
Census Search Engine - Clean Modular Architecture

Orchestrates semantic search using extracted components:
- TableCatalogSearch: Coarse table retrieval  
- VariablesSearch: Fine variable search within tables
- MethodologySearch: Statistical expertise context
- Geographic parsing: Location intelligence

Clean separation of concerns with lightweight orchestration.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import os

# Load environment variables more robustly
try:
    from dotenv import load_dotenv
    # Try multiple .env locations
    for env_path in [
        Path('.env'),
        Path(__file__).parent / '.env',
        Path(__file__).parent.parent / '.env'
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

# Import modular components
from variable_search import VariablesSearch, create_variables_search
from geographic_parsing import (
    create_geographic_parser,
    create_variable_preprocessor,
    GeographicContext
)

# Local search components (to be extracted)
import faiss
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with concept-based metadata"""
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

class QueryTypeDetector:
    """Detects whether query needs variables, methodology, or both"""
    
    def __init__(self):
        self.variable_indicators = [
            r'B\d{5}_\d{3}[EM]?',  # B19013_001E
            'population', 'income', 'poverty', 'unemployment', 'housing', 'education',
            'median', 'total', 'percentage', 'count', 'number of', 'how many',
            'by state', 'by county', 'by race', 'by age', 'by gender'
        ]
        
        self.methodology_indicators = [
            'how does', 'how is', 'methodology', 'method', 'calculated', 'measured',
            'definition', 'universe', 'sample size', 'margin of error', 'reliability',
            'data quality', 'survey design', 'coverage', 'response rate',
            'statistical', 'significance', 'confidence', 'weighting', 'estimation'
        ]
    
    def detect_query_type(self, query: str) -> str:
        """Determine query type for smart routing"""
        query_lower = query.lower()
        
        variable_score = sum(1 for indicator in self.variable_indicators
                           if re.search(indicator, query_lower))
        methodology_score = sum(1 for indicator in self.methodology_indicators
                              if indicator in query_lower)
        
        if variable_score > methodology_score:
            return 'variables'
        elif methodology_score > variable_score:
            return 'methodology'
        else:
            return 'both'

class OpenAIEmbeddings:
    """OpenAI embeddings for search components"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "text-embedding-3-large"
        self.dimensions = 3072
        
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY required")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(input=texts, model=self.model)
        return np.array([data.embedding for data in response.data], dtype=np.float32)

class TableCatalogSearch:
    """Coarse retrieval using table catalog with FAISS"""
    
    def __init__(self, catalog_dir: str = "table-catalog"):
        self.catalog_dir = Path(catalog_dir)
        self.tables = {}
        self.embeddings_index = None
        self.table_ids = []
        self.embedding_model = OpenAIEmbeddings()
        
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
            self.tables = {table['table_id']: table for table in catalog_data['tables']}
        else:
            self.tables = {table['table_id']: table for table in catalog_data}
        
        logger.info(f"Loaded {len(self.tables)} tables from catalog")
    
    def _load_embeddings(self):
        """Load FAISS embeddings for tables"""
        faiss_file = self.catalog_dir / "table_embeddings_enhanced.faiss"
        if not faiss_file.exists():
            faiss_file = self.catalog_dir / "table_embeddings.faiss"
        
        if not faiss_file.exists():
            raise FileNotFoundError(f"Table FAISS index not found in {self.catalog_dir}")
        
        self.embeddings_index = faiss.read_index(str(faiss_file))
        
        mapping_file = self.catalog_dir / "table_mapping_enhanced.json"
        if not mapping_file.exists():
            mapping_file = self.catalog_dir / "table_mapping.json"
        
        with open(mapping_file) as f:
            mapping_data = json.load(f)
        
        self.table_ids = mapping_data['table_ids']
        logger.info(f"✅ Table FAISS index loaded: {len(self.table_ids)} tables")
    
    def search_tables(self, query: str, k: int = 5,
                     geographic_context: Optional[GeographicContext] = None) -> List[Dict]:
        """Search for relevant tables using FAISS"""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.embeddings_index.search(query_embedding, k * 2)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.table_ids):
                continue
            
            table_id = self.table_ids[idx]
            table_data = self.tables.get(table_id)
            if not table_data:
                continue
            
            similarity = max(0.0, 1.0 - (distance / 2.0))
            geographic_relevance = self._calculate_geographic_relevance(table_data, geographic_context)
            combined_score = similarity * (0.7 + 0.3 * geographic_relevance)
            
            results.append({
                'table_id': table_id,
                'table_data': table_data,
                'confidence': similarity,
                'geographic_relevance': geographic_relevance,
                'combined_score': combined_score
            })
        
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:k]
    
    def _calculate_geographic_relevance(self, table_data: Dict,
                                      geographic_context: Optional[GeographicContext]) -> float:
        """Calculate geographic relevance of table"""
        if not geographic_context or not geographic_context.location_mentioned:
            return 1.0
        
        geo_level = geographic_context.geography_level
        available_levels = table_data.get('geography_levels', [])
        
        if geo_level in available_levels:
            return 1.0
        
        # Geographic hierarchy fallbacks
        if geo_level == 'cbsa':
            if 'county' in available_levels or 'place' in available_levels:
                return 0.95
            if 'state' in available_levels:
                return 0.85
        
        return 0.75

class MethodologySearch:
    """Statistical expertise search using ChromaDB"""
    
    def __init__(self, methodology_dir: str = "methodology-db"):
        self.methodology_dir = Path(methodology_dir)
        self.collection = None
        self.embedding_model = OpenAIEmbeddings()
        
        self._load_methodology_db()
    
    def _load_methodology_db(self):
        """Load ChromaDB methodology collection"""
        client = chromadb.PersistentClient(
            path=str(self.methodology_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = client.get_collection("census_methodology")
            doc_count = self.collection.count()
            logger.info(f"✅ Methodology database loaded: {doc_count} documents")
        except Exception as e:
            logger.warning(f"Methodology collection not found: {e}")
            self.collection = None
    
    def search_methodology(self, query: str, k: int = 5) -> List[Dict]:
        """Search methodology documents for statistical expertise"""
        if not self.collection:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
        
        methodology_results = []
        for i in range(len(results['ids'][0])):
            methodology_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'confidence': 1.0 - results['distances'][0][i],
                'source': results['metadatas'][0][i].get('source', 'Unknown')
            })
        
        return methodology_results

class CensusSearchEngine:
    """
    Clean modular Census search engine
    
    Uses extracted components for coarse-to-fine semantic search:
    1. TableCatalogSearch → Find relevant tables
    2. VariablesSearch → Find variables within tables  
    3. MethodologySearch → Add statistical context
    """
    
    def __init__(self,
                 catalog_dir: str = "table-catalog",
                 variables_dir: str = "variables-db",
                 methodology_dir: str = "methodology-db",
                 gazetteer_db_path: str = None):
        
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        # Initialize geographic intelligence
        self.geo_parser = create_geographic_parser(gazetteer_db_path)
        self.variable_preprocessor = create_variable_preprocessor()
        self.query_detector = QueryTypeDetector()
        
        # Initialize modular search components
        self.table_search = TableCatalogSearch(catalog_dir)
        self.variable_search = create_variables_search(variables_dir)
        self.methodology_search = MethodologySearch(methodology_dir)
        
        logger.info("✅ Clean modular Census search engine initialized")
        logger.info("✅ Components: TableSearch + VariablesSearch + MethodologySearch + Geographic")
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Semantic search with modular architecture
        """
        logger.info(f"Searching: '{query}'")
        
        # Step 1: Check for exact variable ID pattern
        variable_id_match = re.match(r'^([A-Z]\d{5})_\d{3}[EM]?$', query.strip().upper())
        if variable_id_match:
            table_id = variable_id_match.group(1)
            logger.info(f"Exact variable ID detected: {query} → table {table_id}")
            return self._search_exact_variable(query.strip().upper(), table_id, max_results)
        
        # Step 2: Parse context
        geo_context = self.geo_parser.parse_geographic_context(query)
        if geo_context.location_mentioned:
            logger.info(f"Geographic context: {geo_context.location_text} ({geo_context.geography_level})")
        
        variable_preprocessing = self.variable_preprocessor.preprocess_query(query)
        search_query = variable_preprocessing['enhanced_query']
        
        # Step 3: Query type detection
        query_type = self.query_detector.detect_query_type(search_query)
        logger.info(f"Query type: {query_type}")
        
        all_results = []
        
        # Step 4: Variables search using modular components
        if query_type in ['variables', 'both']:
            all_results.extend(self._search_variables_modular(search_query, geo_context, max_results))
        
        # Step 5: Add methodology context
        if query_type in ['methodology', 'both']:
            methodology_context = self._search_methodology(search_query)
            for result in all_results:
                if not result.methodology_context and methodology_context:
                    result.methodology_context = methodology_context[:500] + "..."
        
        # Step 6: Sort and return
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        logger.info(f"Returning {min(len(all_results), max_results)} results")
        return all_results[:max_results]
    
    def _search_variables_modular(self, query: str, geo_context: GeographicContext,
                                max_results: int) -> List[SearchResult]:
        """Search variables using modular coarse-to-fine retrieval"""
        
        # Coarse retrieval: find relevant tables
        table_results = self.table_search.search_tables(query, k=5, geographic_context=geo_context)
        logger.info(f"Found {len(table_results)} candidate tables")
        
        if not table_results:
            return []
        
        # Fine retrieval: search within top tables using extracted component
        variable_results = []
        
        for table_result in table_results:
            table_id = table_result['table_id']
            table_data = table_result['table_data']
            table_confidence = table_result['confidence']
            
            # Use extracted VariablesSearch component
            vars_in_table = self.variable_search.search_within_table(
                table_id, query, geo_context, k=3
            )
            
            # Convert to SearchResult objects
            for var_result in vars_in_table:
                var_metadata = var_result['variable_metadata']
                variable_id = var_metadata.get('variable_id', '')
                
                combined_confidence = table_confidence * 0.6 + var_result['final_score'] * 0.4
                
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
                    concept=var_metadata.get('concept', ''),
                    label=var_metadata.get('label', ''),
                    title=table_data.get('title', ''),
                    universe=table_data.get('universe', ''),
                    confidence=combined_confidence,
                    geographic_relevance=var_result['geographic_score'],
                    geographic_restrictions={
                        'acs1': table_data.get('geography_restrictions_1yr', ''),
                        'acs5': table_data.get('geography_restrictions_5yr', '')
                    },
                    available_surveys=var_metadata.get('available_surveys', ['acs5']),
                    statistical_notes=table_data.get('statistical_notes', []),
                    primary_variable=variable_id.endswith('_001E'),
                    survey_instances=[],
                    geography_coverage=geography_coverage,
                    primary_instance=var_metadata.get('primary_instance'),
                    structure_type=var_metadata.get('structure_type', 'concept_based')
                )
                
                variable_results.append(search_result)
        
        return variable_results
    
    def _search_exact_variable(self, variable_id: str, table_id: str,
                             max_results: int) -> List[SearchResult]:
        """Search for exact variable ID using modular components"""
        
        # Check if table exists
        table_data = self.table_search.tables.get(table_id)
        if not table_data:
            logger.warning(f"Table {table_id} not found for variable {variable_id}")
            return []
        
        logger.info(f"Direct table lookup: {table_id} - {table_data.get('title', 'Unknown')}")
        
        # Use extracted VariablesSearch component for exact lookup
        vars_in_table = self.variable_search.search_within_table(
            table_id, variable_id, None, k=max_results
        )
        
        if not vars_in_table:
            logger.warning(f"Variable {variable_id} not found in table {table_id}")
            return []
        
        # Convert to SearchResult objects
        variable_results = []
        
        for var_result in vars_in_table:
            var_metadata = var_result['variable_metadata']
            found_variable_id = var_metadata.get('variable_id', '')
            
            # Prioritize exact matches
            is_exact_match = found_variable_id == variable_id
            confidence = 1.0 if is_exact_match else var_result['final_score']
            
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
                variable_id=found_variable_id,
                table_id=table_id,
                concept=var_metadata.get('concept', ''),
                label=var_metadata.get('label', ''),
                title=table_data.get('title', ''),
                universe=table_data.get('universe', ''),
                confidence=confidence,
                geographic_relevance=1.0,  # Direct lookup has perfect relevance
                geographic_restrictions={
                    'acs1': table_data.get('geography_restrictions_1yr', ''),
                    'acs5': table_data.get('geography_restrictions_5yr', '')
                },
                available_surveys=var_metadata.get('available_surveys', ['acs5']),
                statistical_notes=table_data.get('statistical_notes', []),
                primary_variable=found_variable_id.endswith('_001E'),
                survey_instances=[],
                geography_coverage=geography_coverage,
                primary_instance=var_metadata.get('primary_instance'),
                structure_type=var_metadata.get('structure_type', 'concept_based')
            )
            
            variable_results.append(search_result)
        
        # Sort by exact match first, then by confidence
        variable_results.sort(key=lambda x: (not (x.variable_id == variable_id), -x.confidence))
        
        logger.info(f"Direct lookup found {len(variable_results)} results, exact match: {any(r.variable_id == variable_id for r in variable_results)}")
        return variable_results
    
    def _search_methodology(self, query: str) -> Optional[str]:
        """Search methodology for statistical expertise context"""
        try:
            methodology_results = self.methodology_search.search_methodology(query, k=2)
            if methodology_results:
                return methodology_results[0]['content']
        except Exception as e:
            logger.warning(f"Methodology search failed: {e}")
        
        return None
    
    def get_table_info(self, table_id: str) -> Optional[Dict]:
        """Get complete information about a table"""
        return self.table_search.tables.get(table_id)
    
    def get_variable_info(self, variable_id: str) -> Optional[Dict]:
        """Get complete information about a variable using modular component"""
        return self.variable_search.get_variable_info(variable_id)
    
    def get_stats(self) -> Dict:
        """Get statistics about the search engine components"""
        variable_stats = self.variable_search.get_stats()
        table_count = len(self.table_search.tables)
        
        return {
            'tables': table_count,
            'variables': variable_stats.get('total_variables', 0),
            'methodology_docs': self.methodology_search.collection.count() if self.methodology_search.collection else 0,
            'architecture': 'modular',
            'components': ['TableCatalogSearch', 'VariablesSearch', 'MethodologySearch', 'GeographicParsing']
        }

# Backward compatibility aliases
ConceptBasedCensusSearchEngine = CensusSearchEngine

# Factory function for MCP integration
def create_search_engine(knowledge_base_dir: str = None,
                        gazetteer_db_path: str = None) -> CensusSearchEngine:
    """Create clean modular search engine"""
    
    if knowledge_base_dir:
        base_path = Path(knowledge_base_dir)
        catalog_dir = base_path / "table-catalog"
        variables_dir = base_path / "variables-db"
        methodology_dir = base_path / "methodology-db"
        
        # Auto-detect gazetteer if not specified
        if not gazetteer_db_path:
            potential_gazetteer = base_path / "geo-db" / "geography.db"
            if potential_gazetteer.exists():
                gazetteer_db_path = str(potential_gazetteer)
    else:
        # Default paths relative to current file
        base_dir = Path(__file__).parent
        catalog_dir = base_dir / "table-catalog"
        variables_dir = base_dir / "variables-db"
        methodology_dir = base_dir / "methodology-db"
        
        # Auto-detect gazetteer
        if not gazetteer_db_path:
            potential_gazetteer = base_dir / "geo-db" / "geography.db"
            if potential_gazetteer.exists():
                gazetteer_db_path = str(potential_gazetteer)
    
    return CensusSearchEngine(
        catalog_dir=str(catalog_dir),
        variables_dir=str(variables_dir),
        methodology_dir=str(methodology_dir),
        gazetteer_db_path=gazetteer_db_path
    )

if __name__ == "__main__":
    # Test the modular search engine
    logger.info("Testing Clean Modular Census Search Engine...")
    try:
        engine = create_search_engine()
        stats = engine.get_stats()
        logger.info(f"Engine stats: {stats}")
        
        results = engine.search("B01003_001E", max_results=3)
        logger.info(f"Test search returned {len(results)} results")
        for result in results[:2]:
            logger.info(f"  {result.variable_id}: {result.label[:100]}...")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
