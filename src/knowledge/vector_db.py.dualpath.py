#!/usr/bin/env python3
"""
Dual-Path Vector Database for Census MCP Server

Handles separated RAG functionality using:
- Sentence transformers for high-quality local embeddings (NO API key required)
- TWO ChromaDB collections optimized for different retrieval patterns:
  1. Variables DB: 65K canonical variables → entity lookup, GraphRAG potential
  2. Methodology DB: Documentation, guides, PDFs → conceptual search
- Location parsing and geographic knowledge
- Smart query routing based on content type
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib

# Vector DB and embedding imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

from utils.config import get_config

logger = logging.getLogger(__name__)

class DualPathKnowledgeBase:
    """
    Dual-path vector database for optimized Census expertise retrieval.
    
    Provides:
    - Variables database: Optimized for entity lookup (variable IDs, concepts)
    - Methodology database: Optimized for conceptual search (documentation, guides)
    - Smart query routing based on query type detection
    - Location parsing and geographic knowledge
    - Unified interface for seamless integration
    """
    
    def __init__(self,
                 variables_db_path: Optional[Path] = None,
                 methodology_db_path: Optional[Path] = None,
                 corpus_path: Optional[Path] = None):
        """
        Initialize dual-path knowledge base.
        
        Args:
            variables_db_path: Path to variables vector database
            methodology_db_path: Path to methodology vector database  
            corpus_path: Path to legacy corpus (for compatibility)
        """
        self.config = get_config()
        
        # Database paths from config
        self.variables_db_path = variables_db_path or self.config.variables_db_path
        self.methodology_db_path = methodology_db_path or self.config.methodology_db_path
        self.corpus_path = corpus_path  # Keep for legacy compatibility but not used
        
        # Initialize components
        self._init_embedding_model()
        self._init_dual_vector_dbs()
        self._init_variable_contexts()
        self._init_location_knowledge()
        
        logger.info("Dual-Path Knowledge Base initialized successfully")
    
    def _init_embedding_model(self):
        """Initialize sentence transformer for local embeddings."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available. RAG functionality disabled.")
            self.embedding_model = None
            self.model_name = None
            return
        
        try:
            # Use the same model that built the databases
            model_name = 'sentence-transformers/all-mpnet-base-v2'  # 768-dim
            
            logger.info(f"Loading sentence transformer model: {model_name}")
            
            # Load sentence transformer model (uses cached version)
            self.embedding_model = SentenceTransformer(model_name, cache_folder='./model_cache')
            self.model_name = model_name
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Sentence transformer model loaded successfully")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Dimensions: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {str(e)}")
            self.embedding_model = None
            self.model_name = None
    
    def _init_dual_vector_dbs(self):
        """Initialize both ChromaDB collections for dual-path retrieval."""
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available. Vector search disabled.")
            self.variables_client = None
            self.methodology_client = None
            self.variables_collection = None
            self.methodology_collection = None
            return
        
        try:
            # Initialize Variables Database
            if self.variables_db_path.exists():
                self.variables_client = chromadb.PersistentClient(
                    path=str(self.variables_db_path),
                    settings=Settings(anonymized_telemetry=False, allow_reset=False)
                )
                self.variables_collection = self.variables_client.get_collection("census_variables")
                var_count = self.variables_collection.count()
                logger.info(f"✅ Variables DB loaded: {var_count:,} variables")
            else:
                logger.warning(f"Variables database not found at {self.variables_db_path}")
                self.variables_client = None
                self.variables_collection = None
            
            # Initialize Methodology Database
            if self.methodology_db_path.exists():
                self.methodology_client = chromadb.PersistentClient(
                    path=str(self.methodology_db_path),
                    settings=Settings(anonymized_telemetry=False, allow_reset=False)
                )
                self.methodology_collection = self.methodology_client.get_collection("census_methodology")
                method_count = self.methodology_collection.count()
                logger.info(f"✅ Methodology DB loaded: {method_count:,} documents")
            else:
                logger.warning(f"Methodology database not found at {self.methodology_db_path}")
                self.methodology_client = None
                self.methodology_collection = None
            
        except Exception as e:
            logger.error(f"Failed to initialize vector databases: {str(e)}")
            self.variables_client = None
            self.methodology_client = None
            self.variables_collection = None
            self.methodology_collection = None
    
    def _init_variable_contexts(self):
        """Initialize static variable context knowledge."""
        # Common Census variable patterns and contexts
        self.variable_contexts = {
            'b01001': {
                'label': 'Total Population',
                'definition': 'Universe: Total population',
                'table': 'Sex by Age'
            },
            'b19013': {
                'label': 'Median Household Income',
                'definition': 'Median household income in the past 12 months (in inflation-adjusted dollars)',
                'table': 'Median Household Income'
            },
            'b25001': {
                'label': 'Total Housing Units',
                'definition': 'Universe: Housing units',
                'table': 'Housing Units'
            },
            'b08303': {
                'label': 'Travel Time to Work',
                'definition': 'Universe: Workers 16 years and over who did not work from home',
                'table': 'Travel Time to Work'
            }
        }
    
    def _init_location_knowledge(self):
        """Initialize geographic location patterns."""
        self.location_patterns = {
            'major_cities': {
                'new york': {'full_name': 'New York, NY', 'state': 'NY', 'type': 'city'},
                'los angeles': {'full_name': 'Los Angeles, CA', 'state': 'CA', 'type': 'city'},
                'chicago': {'full_name': 'Chicago, IL', 'state': 'IL', 'type': 'city'},
                'houston': {'full_name': 'Houston, TX', 'state': 'TX', 'type': 'city'},
                'phoenix': {'full_name': 'Phoenix, AZ', 'state': 'AZ', 'type': 'city'},
                'philadelphia': {'full_name': 'Philadelphia, PA', 'state': 'PA', 'type': 'city'},
                'san antonio': {'full_name': 'San Antonio, TX', 'state': 'TX', 'type': 'city'},
                'san diego': {'full_name': 'San Diego, CA', 'state': 'CA', 'type': 'city'},
                'dallas': {'full_name': 'Dallas, TX', 'state': 'TX', 'type': 'city'},
                'san jose': {'full_name': 'San Jose, CA', 'state': 'CA', 'type': 'city'}
            },
            'ambiguous_names': {
                'springfield': ['IL', 'MA', 'MO', 'OH'],
                'columbus': ['OH', 'GA'],
                'richmond': ['VA', 'CA'],
                'portland': ['OR', 'ME'],
                'washington': ['DC', 'state']
            }
        }
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect whether query is asking for variables or methodology.
        
        Args:
            query: Search query
            
        Returns:
            'variables', 'methodology', or 'both'
        """
        query_lower = query.lower().strip()
        
        # Variable indicators
        variable_patterns = [
            r'B\d{5}',  # Table codes like B01001
            r'variable\s+\w+',  # "variable B01001"
            r'median\s+\w+\s+income',  # "median household income"
            r'total\s+population',
            r'housing\s+units',
            r'demographics?',
            r'race\s+and\s+ethnicity',
            r'age\s+and\s+sex',
            r'education\s+level',
            r'employment\s+status',
            r'poverty\s+status'
        ]
        
        # Methodology indicators
        methodology_patterns = [
            r'how\s+(does|do)\s+',  # "how does Census collect"
            r'methodology',
            r'sample\s+size',
            r'margin\s+of\s+error',
            r'survey\s+design',
            r'data\s+collection',
            r'statistical\s+reliability',
            r'data\s+quality',
            r'sampling\s+methods?',
            r'questionnaire',
            r'interview\s+process',
            r'response\s+rate',
            r'weighting\s+procedures?'
        ]
        
        # Check for variable patterns
        has_variable_indicators = any(re.search(pattern, query_lower) for pattern in variable_patterns)
        
        # Check for methodology patterns
        has_methodology_indicators = any(re.search(pattern, query_lower) for pattern in methodology_patterns)
        
        if has_methodology_indicators and not has_variable_indicators:
            return 'methodology'
        elif has_variable_indicators and not has_methodology_indicators:
            return 'variables'
        else:
            # When ambiguous, search both but prioritize methodology for conceptual queries
            conceptual_keywords = ['explain', 'understand', 'learn', 'guide', 'tutorial', 'documentation']
            if any(keyword in query_lower for keyword in conceptual_keywords):
                return 'methodology'
            return 'both'
    
    async def search_variables(self, query: str, context: str = "",
                             top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search variables database for entity lookup.
        
        Args:
            query: Search query
            context: Additional context
            top_k: Number of results to return
            
        Returns:
            List of relevant variables with metadata
        """
        if not self.embedding_model or not self.variables_collection:
            logger.warning("Cannot search variables: embedding model or variables DB not available")
            return []
        
        try:
            # Prepare search query
            search_text = f"{query} {context}".strip()
            top_k = top_k or 10
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([search_text])
            
            # Search variables database
            results = self.variables_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results for variables
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'source': 'variables_db',
                    'type': 'variable',
                    'variable_id': results['metadatas'][0][i].get('temporal_id', ''),
                    'label': results['metadatas'][0][i].get('label', ''),
                    'concept': results['metadatas'][0][i].get('concept', '')
                }
                formatted_results.append(result)
            
            # Filter by reasonable distance threshold for variables (more lenient)
            filtered_results = [r for r in formatted_results if r['distance'] < 1.5]
            
            logger.info(f"Variables search: {len(filtered_results)} relevant results from {len(results['documents'][0])} total")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Variables search failed: {str(e)}")
            return []
    
    async def search_methodology(self, query: str, context: str = "",
                               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search methodology database for conceptual information.
        
        Args:
            query: Search query
            context: Additional context
            top_k: Number of results to return
            
        Returns:
            List of relevant methodology documents with metadata
        """
        if not self.embedding_model or not self.methodology_collection:
            logger.warning("Cannot search methodology: embedding model or methodology DB not available")
            return []
        
        try:
            # Prepare search query
            search_text = f"{query} {context}".strip()
            top_k = top_k or 8
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([search_text])
            
            # Search methodology database
            results = self.methodology_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results for methodology
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'source': 'methodology_db',
                    'type': 'methodology',
                    'category': results['metadatas'][0][i].get('category', ''),
                    'source_file': results['metadatas'][0][i].get('source_file', ''),
                    'title': results['metadatas'][0][i].get('source_file', '').split('/')[-1]
                }
                formatted_results.append(result)
            
            # Filter by distance threshold for methodology (stricter for quality)
            filtered_results = [r for r in formatted_results if r['distance'] < 1.2]
            
            logger.info(f"Methodology search: {len(filtered_results)} relevant results from {len(results['documents'][0])} total")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Methodology search failed: {str(e)}")
            return []
    
    async def search_documentation(self, query: str, context: str = "",
                                 top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Smart search that routes queries to appropriate database(s).
        
        Args:
            query: Search query
            context: Additional context for the search
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata from appropriate database(s)
        """
        query_type = self._detect_query_type(query)
        all_results = []
        
        logger.info(f"Smart routing detected query type: {query_type} for query: '{query}'")
        
        if query_type == 'variables':
            # Search only variables database
            results = await self.search_variables(query, context, top_k)
            all_results.extend(results)
            
        elif query_type == 'methodology':
            # Search only methodology database
            results = await self.search_methodology(query, context, top_k)
            all_results.extend(results)
            
        else:  # query_type == 'both'
            # Search both, but prioritize based on results quality
            var_results = await self.search_variables(query, context, top_k // 2)
            method_results = await self.search_methodology(query, context, top_k // 2)
            
            # Combine and sort by score/relevance
            all_results.extend(var_results)
            all_results.extend(method_results)
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit to top_k total results
            if top_k:
                all_results = all_results[:top_k]
        
        logger.info(f"Documentation search complete: {len(all_results)} total results")
        return all_results
    
    async def get_variable_context(self, variables: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get context information for Census variables using variables database.
        
        Args:
            variables: List of variable names/IDs
            
        Returns:
            Dictionary mapping variables to context information
        """
        context = {}
        
        for var in variables:
            var_lower = var.lower().strip()
            
            # Start with static context
            var_context = self.variable_contexts.get(var_lower, {
                'label': var.title(),
                'definition': f"Census variable: {var}",
                'notes': 'Variable definition not available in local knowledge base'
            })
            
            # Enhance with variables database search
            try:
                search_results = await self.search_variables(f"variable {var}", top_k=3)
                if search_results:
                    best_match = search_results[0]
                    var_context.update({
                        'label': best_match.get('label', var_context.get('label')),
                        'concept': best_match.get('concept', ''),
                        'variable_id': best_match.get('variable_id', ''),
                        'source': 'variables_database',
                        'confidence': 'high' if best_match['score'] > 0.8 else 'medium'
                    })
            except Exception as e:
                logger.warning(f"Variable database search failed for {var}: {str(e)}")
            
            context[var] = var_context
        
        return context
    
    async def parse_location(self, location: str) -> Dict[str, Any]:
        """
        Parse and enhance location information with geographic context.
        
        Args:
            location: Location string
            
        Returns:
            Enhanced location information with geographic context
        """
        location_lower = location.lower().strip()
        
        # Check major cities
        if location_lower in self.location_patterns['major_cities']:
            city_info = self.location_patterns['major_cities'][location_lower]
            return {
                'original': location,
                'normalized': city_info['full_name'],
                'state': city_info['state'],
                'type': city_info['type'],
                'confidence': 'high'
            }
        
        # Check for ambiguous names
        for ambiguous, states in self.location_patterns['ambiguous_names'].items():
            if ambiguous in location_lower:
                return {
                    'original': location,
                    'ambiguous': True,
                    'possible_states': states,
                    'recommendation': f"Specify state for {ambiguous} (e.g., '{ambiguous}, {states[0]}')",
                    'confidence': 'low'
                }
        
        # Default parsing
        return {
            'original': location,
            'normalized': location,
            'confidence': 'medium'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive knowledge base statistics."""
        stats = {
            'embedding_model': self.model_name if self.embedding_model else 'Not available',
            'embedding_dimensions': self.embedding_dimension if self.embedding_model else 0,
            'variables_db_available': self.variables_collection is not None,
            'methodology_db_available': self.methodology_collection is not None,
            'variables_db_path': str(self.variables_db_path),
            'methodology_db_path': str(self.methodology_db_path),
            'variable_contexts': len(self.variable_contexts),
            'location_patterns': len(self.location_patterns['major_cities'])
        }
        
        if self.variables_collection:
            try:
                stats['variables_count'] = self.variables_collection.count()
            except:
                stats['variables_count'] = 'Unknown'
        
        if self.methodology_collection:
            try:
                stats['methodology_count'] = self.methodology_collection.count()
            except:
                stats['methodology_count'] = 'Unknown'
        
        return stats

# Legacy compatibility alias
KnowledgeBase = DualPathKnowledgeBase
