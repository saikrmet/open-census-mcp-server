#!/usr/bin/env python3
"""
Clean Census Search Engine - Geography-Free Variable Discovery

Pure focus on:
- Variable search (concept → Census variable IDs)
- Methodology RAG (statistical context)
- Table search (Census table discovery)

NO geographic parsing - that's handled by MCP resolve_geography tool.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Vector search imports
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None
    SentenceTransformer = None

# Methodology search imports
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VariableResult:
    """Variable search result with confidence score."""
    variable_id: str
    label: str
    concept: str
    table_id: str
    confidence: float
    universe: Optional[str] = None
    methodology_notes: Optional[str] = None

@dataclass
class TableResult:
    """Table search result."""
    table_id: str
    title: str
    universe: str
    subject_area: str
    confidence: float

class VariablesSearch:
    """FAISS-based variable search engine."""
    
    def __init__(self, variables_dir: str):
        self.variables_dir = Path(variables_dir)
        self.model = None
        self.index = None
        self.metadata = None
        self.variable_ids = None
        
        if FAISS_AVAILABLE:
            self._load_index()
        else:
            logger.warning("FAISS not available - variable search disabled")
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        try:
            # Load FAISS index
            index_path = self.variables_dir / "variables.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} variables")
            
            # Load metadata
            metadata_path = self.variables_dir / "variables_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ Loaded metadata for {len(self.metadata)} variables")
            
            # Load variable IDs
            ids_path = self.variables_dir / "variables_ids.json"
            if ids_path.exists():
                with open(ids_path) as f:
                    data = json.load(f)
                    self.variable_ids = data.get('variable_ids', [])
                logger.info(f"✅ Loaded {len(self.variable_ids)} variable IDs")
            
            # Load sentence transformer model
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("✅ Loaded sentence transformer model")
            
        except Exception as e:
            logger.error(f"Failed to load variables search: {e}")
            self.index = None
    
    def search(self, query: str, max_results: int = 10) -> List[VariableResult]:
        """Search for variables by concept."""
        if not self.index or not self.model:
            logger.warning("Variables search not available")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), max_results)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.variable_ids):
                    variable_id = self.variable_ids[idx]
                    metadata = self.metadata.get(variable_id, {})
                    
                    result = VariableResult(
                        variable_id=variable_id,
                        label=metadata.get('label', variable_id),
                        concept=metadata.get('concept', ''),
                        table_id=variable_id.split('_')[0] if '_' in variable_id else '',
                        confidence=float(1.0 - score),  # Convert distance to similarity
                        universe=metadata.get('universe'),
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Variable search error: {e}")
            return []

class TableCatalogSearch:
    """FAISS-based table search using existing embeddings."""
    
    def __init__(self, catalog_dir: str):
        self.catalog_dir = Path(catalog_dir)
        self.model = None
        self.index = None
        self.table_ids = None
        self.tables = {}
        
        if FAISS_AVAILABLE:
            self._load_index()
        else:
            logger.warning("FAISS not available - table search disabled")
    
    def _load_index(self):
        """Load FAISS table index and metadata."""
        try:
            # Load FAISS index
            index_path = self.catalog_dir / "table_embeddings.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"✅ Loaded table FAISS index with {self.index.ntotal} tables")
            
            # Load table IDs mapping
            mapping_path = self.catalog_dir / "table_mapping.json"
            if mapping_path.exists():
                with open(mapping_path) as f:
                    data = json.load(f)
                    self.table_ids = data.get('table_ids', [])
                logger.info(f"✅ Loaded {len(self.table_ids)} table IDs")
            
            # Load table catalog metadata
            catalog_path = self.catalog_dir / "table_catalog.json"
            if catalog_path.exists():
                with open(catalog_path) as f:
                    data = json.load(f)
                    self.tables = data.get('tables', {})
                logger.info(f"✅ Loaded metadata for {len(self.tables)} tables")
            
            # Load sentence transformer model
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("✅ Loaded sentence transformer model for tables")
            
        except Exception as e:
            logger.error(f"Failed to load table search: {e}")
            self.index = None
    
    def search(self, query: str, max_results: int = 10) -> List[TableResult]:
        """Search tables using FAISS semantic similarity."""
        if not self.index or not self.model:
            logger.warning("Table search not available")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), max_results)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.table_ids):
                    table_id = self.table_ids[idx]
                    table_info = self.tables.get(table_id, {})
                    
                    result = TableResult(
                        table_id=table_id,
                        title=table_info.get('title', table_id),
                        universe=table_info.get('universe', ''),
                        subject_area=table_info.get('subject_area', ''),
                        confidence=float(1.0 - score)  # Convert distance to similarity
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Table search error: {e}")
            return []

class MethodologySearch:
    """ChromaDB-based methodology search."""
    
    def __init__(self, methodology_dir: str):
        self.methodology_dir = Path(methodology_dir)
        self.collection = None
        
        if CHROMADB_AVAILABLE:
            self._load_collection()
        else:
            logger.warning("ChromaDB not available - methodology search disabled")
    
    def _load_collection(self):
        """Load ChromaDB collection."""
        try:
            client = chromadb.PersistentClient(path=str(self.methodology_dir))
            self.collection = client.get_collection("census_methodology")
            count = self.collection.count()
            logger.info(f"✅ Loaded methodology collection with {count} documents")
        except Exception as e:
            logger.warning(f"Methodology search not available: {e}")
            self.collection = None
    
    def search(self, query: str, max_results: int = 5) -> str:
        """Search methodology documents."""
        if not self.collection:
            return ""
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results
            )
            
            if results['documents']:
                # Concatenate top results
                context = "\n\n".join(results['documents'][0])
                return context[:1000]  # Limit length
            
        except Exception as e:
            logger.error(f"Methodology search error: {e}")
        
        return ""

class ConceptBasedCensusSearchEngine:
    """
    Geography-free Census search engine.
    
    Focuses purely on:
    - Variable discovery (concept → Census variable IDs)
    - Statistical methodology context
    - Table catalog search
    
    NO geographic parsing - handled by MCP tools.
    """
    
    def __init__(self, catalog_dir: str = None, variables_dir: str = None, methodology_dir: str = None):
        """Initialize search components."""
        
        # Set default paths
        if not catalog_dir or not variables_dir or not methodology_dir:
            base_dir = Path(__file__).parent
            catalog_dir = catalog_dir or str(base_dir / "table-catalog")
            variables_dir = variables_dir or str(base_dir / "variables-db")
            methodology_dir = methodology_dir or str(base_dir / "methodology-db")
        
        # Initialize search components
        self.table_search = TableCatalogSearch(catalog_dir)
        self.variables_search = VariablesSearch(variables_dir)
        self.methodology_search = MethodologySearch(methodology_dir)
        
        logger.info("✅ ConceptBasedCensusSearchEngine initialized (geography-free)")
    
    def search(self, query: str, max_results: int = 10) -> List[VariableResult]:
        """
        Main search interface - find Census variables by concept.
        
        Args:
            query: Natural language concept (e.g. "median household income")
            max_results: Maximum number of results to return
            
        Returns:
            List of VariableResult with Census variable IDs and metadata
        """
        
        # Check for direct variable ID patterns first
        if self._is_variable_id(query):
            return self._direct_variable_lookup(query)
        
        # Use synonym mappings for common concepts
        synonymized_query = self._apply_synonyms(query)
        
        # Search variables using FAISS
        results = self.variables_search.search(synonymized_query, max_results)
        
        # Add methodology context to top results
        if results and self.methodology_search.collection:
            methodology_context = self.methodology_search.search(query)
            if methodology_context:
                # Add to top result
                results[0].methodology_notes = methodology_context[:200]
        
        return results
    
    def _is_variable_id(self, query: str) -> bool:
        """Check if query is already a Census variable ID."""
        import re
        # Pattern: B19013_001E, S1501_C01_001E, etc.
        pattern = r'^[A-Z]+[0-9]+[A-Z]*_[0-9]+[A-Z]*$'
        return bool(re.match(pattern, query.upper()))
    
    def _direct_variable_lookup(self, variable_id: str) -> List[VariableResult]:
        """Direct lookup for variable IDs."""
        variable_id = variable_id.upper()
        
        if self.variables_search.metadata and variable_id in self.variables_search.metadata:
            metadata = self.variables_search.metadata[variable_id]
            result = VariableResult(
                variable_id=variable_id,
                label=metadata.get('label', variable_id),
                concept=metadata.get('concept', ''),
                table_id=variable_id.split('_')[0],
                confidence=1.0,
                universe=metadata.get('universe')
            )
            return [result]
        
        return []
    
    def _apply_synonyms(self, query: str) -> str:
        """Apply synonym mappings for common Census concepts."""
        query_lower = query.lower().strip()
        
        # Common synonym mappings
        synonyms = {
            "population": "total population",
            "median income": "median household income",
            "poverty": "poverty rate income below poverty level",
            "unemployment": "unemployment rate labor force",
            "median age": "median age by sex",
            "housing": "housing units occupied",
            "rent": "median gross rent",
            "home value": "median value owner occupied housing",
            "education": "educational attainment",
            "commute": "means of transportation to work",
        }
        
        for key, expanded in synonyms.items():
            if key in query_lower:
                return query_lower.replace(key, expanded)
        
        return query
    
    def search_tables(self, query: str, max_results: int = 10) -> List[TableResult]:
        """Search Census tables by concept."""
        return self.table_search.search(query, max_results)
    
    def _search_methodology(self, query: str) -> str:
        """Search methodology documents (internal use)."""
        return self.methodology_search.search(query)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            'variables_count': len(self.variables_search.variable_ids) if self.variables_search.variable_ids else 0,
            'tables_count': len(self.table_search.tables),
            'methodology_count': self.methodology_search.collection.count() if self.methodology_search.collection else 0,
            'architecture': 'geography_free',
            'components': ['TableCatalogSearch', 'VariablesSearch', 'MethodologySearch']
        }

# Backward compatibility aliases
CensusSearchEngine = ConceptBasedCensusSearchEngine

# Factory function for MCP integration
def create_search_engine(knowledge_base_dir: str = None) -> ConceptBasedCensusSearchEngine:
    """Create geography-free search engine for variable discovery."""
    
    if knowledge_base_dir:
        base_path = Path(knowledge_base_dir)
        catalog_dir = base_path / "table-catalog"
        variables_dir = base_path / "variables-db"
        methodology_dir = base_path / "methodology-db"
    else:
        # Default paths relative to current file
        base_dir = Path(__file__).parent
        catalog_dir = base_dir / "table-catalog"
        variables_dir = base_dir / "variables-db"
        methodology_dir = base_dir / "methodology-db"
    
    return ConceptBasedCensusSearchEngine(
        catalog_dir=str(catalog_dir),
        variables_dir=str(variables_dir),
        methodology_dir=str(methodology_dir)
    )

if __name__ == "__main__":
    # Test the geography-free search engine
    logger.info("Testing Geography-Free Census Search Engine...")
    try:
        engine = create_search_engine()
        stats = engine.get_stats()
        logger.info(f"Engine stats: {stats}")
        
        results = engine.search("median household income", max_results=3)
        logger.info(f"Test search returned {len(results)} results")
        for result in results[:2]:
            logger.info(f"  {result.variable_id}: {result.label[:100]}...")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
