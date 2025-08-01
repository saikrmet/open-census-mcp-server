#!/usr/bin/env python3
"""
OpenAI-Only Knowledge Base Builder - Clean Implementation
Builds TWO separate vector databases optimized for different retrieval patterns:

1. VARIABLES DATABASE: 36K concept-based variables → FAISS index (fast loading) OR ChromaDB  
2. METHODOLOGY DATABASE: Documentation, guides, PDFs → ChromaDB (conceptual search)

Key Features:
- OpenAI embeddings only (text-embedding-3-large)
- Bulletproof retry logic with exponential backoff
- Token counting and truncation
- Disk caching to avoid re-embedding
- Array synchronization validation

Usage:
    python build-kb-openai-clean.py --variables-only --faiss
    python build-kb-openai-clean.py --methodology-only  
    python build-kb-openai-clean.py --both --faiss
"""

import os
import sys
import numpy as np
import logging
import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
import re

# Bulletproofing imports
from tenacity import retry, stop_after_attempt, wait_exponential
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass

# Document processing
import PyPDF2
from bs4 import BeautifulSoup
import pandas as pd

# Vector DB and embeddings
import chromadb
from chromadb.config import Settings

# OpenAI for embeddings
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    raise ImportError("OpenAI package required. Install with: pip install openai")

# FAISS for variables database
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress OpenAI HTTP request logging spam
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Progress logger
progress_logger = logging.getLogger('progress')
progress_logger.setLevel(logging.INFO)
progress_handler = logging.StreamHandler()
progress_handler.setFormatter(logging.Formatter('%(message)s'))
progress_logger.addHandler(progress_handler)
progress_logger.propagate = False

def get_text_hash(text: str) -> str:
    """Get stable hash for text caching"""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken if available"""
    if not TIKTOKEN_AVAILABLE:
        return len(text) // 4  # Rough approximation
    
    try:
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        return len(encoding.encode(text))
    except:
        return len(text) // 4

def truncate_text_to_tokens(text: str, max_tokens: int = 7500) -> str:
    """Truncate text to stay under token limit"""
    if not TIKTOKEN_AVAILABLE:
        max_chars = max_tokens * 4
        return text[:max_chars] if len(text) > max_chars else text
    
    try:
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])
    except:
        max_chars = max_tokens * 4
        return text[:max_chars] if len(text) > max_chars else text

@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)
def create_embeddings_with_retry(client, texts: List[str]) -> List[List[float]]:
    """Create embeddings with bulletproof retry logic"""
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-large"
    )
    return [item.embedding for item in response.data]

def show_progress(current, total, prefix="Progress", bar_length=50):
    """Show a clean progress bar"""
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f'\r{prefix}: [{bar}] {percent:.1%} ({current}/{total})', end='', flush=True)
    if current == total:
        print()

def extract_content(file_path: Path) -> str:
    """Extract text content from various file types"""
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return extract_pdf_content(file_path)
    elif suffix in ['.md', '.txt']:
        return extract_text_content(file_path)
    elif suffix == '.html':
        return extract_html_content(file_path)
    elif suffix in ['.csv', '.xlsx']:
        return extract_tabular_content(file_path)
    else:
        return extract_text_content(file_path)

def extract_pdf_content(file_path: Path) -> str:
    """Extract text from PDF files"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {e}")
        return ""

def extract_text_content(file_path: Path) -> str:
    """Extract text from text files"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return ""

def extract_html_content(file_path: Path) -> str:
    """Extract text from HTML files"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return soup.get_text()
    except Exception as e:
        logger.error(f"Error extracting HTML {file_path}: {e}")
        return ""

def extract_tabular_content(file_path: Path) -> str:
    """Extract summary from CSV/Excel files"""
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, nrows=100)
        else:
            df = pd.read_excel(file_path, nrows=100)
        
        summary = f"Table: {file_path.name}\n"
        summary += f"Columns: {', '.join(df.columns)}\n"
        summary += f"Sample data:\n{df.head().to_string()}\n"
        return summary
    except Exception as e:
        logger.error(f"Error extracting tabular data {file_path}: {e}")
        return ""

def create_chunks(content: str, file_path: Path) -> List[Dict]:
    """Robust chunking with fallback for documents without paragraph breaks"""
    chunks = []
    
    # Clean text
    content = re.sub(r'\s+', ' ', content).strip()
    
    if len(content) < 100:
        return chunks
    
    # Settings
    chunk_size = 1000
    overlap_size = 200
    max_safe_size = 5000  # Absolute max before we force split
    
    # First try: paragraph-based splitting
    paragraphs = content.split('\n\n')
    
    # If we only got 1 paragraph and it's huge, use fallback splitting
    if len(paragraphs) == 1 and len(paragraphs[0]) > max_safe_size:
        # Fallback: split by sentences first
        sentences = re.split(r'[.!?]+\s+', content)
        if len(sentences) > 1:
            paragraphs = sentences
        else:
            # Last resort: split by fixed character chunks
            paragraphs = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size-overlap_size)]
    
    current_chunk = ""
    chunk_num = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If this single paragraph is still too large, force split it
        if len(paragraph) > max_safe_size:
            # Split oversized paragraph into smaller pieces
            for i in range(0, len(paragraph), max_safe_size-overlap_size):
                piece = paragraph[i:i+max_safe_size]
                if len(piece.strip()) > 100:
                    chunks.append(create_chunk_metadata(
                        piece.strip(), file_path, chunk_num
                    ))
                    chunk_num += 1
            continue
        
        # Normal processing
        test_chunk = current_chunk + " " + paragraph if current_chunk else paragraph
        
        if len(test_chunk) > chunk_size and current_chunk:
            # Save current chunk
            if len(current_chunk.strip()) > 100:
                chunks.append(create_chunk_metadata(
                    current_chunk.strip(), file_path, chunk_num
                ))
                chunk_num += 1
            
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else current_chunk
            current_chunk = overlap_text + " " + paragraph if overlap_text else paragraph
        else:
            current_chunk = test_chunk
    
    # Add final chunk
    if current_chunk and len(current_chunk.strip()) > 100:
        # Final safety check
        if len(current_chunk) > max_safe_size:
            # Force split the final chunk
            for i in range(0, len(current_chunk), max_safe_size-overlap_size):
                piece = current_chunk[i:i+max_safe_size]
                if len(piece.strip()) > 100:
                    chunks.append(create_chunk_metadata(
                        piece.strip(), file_path, chunk_num
                    ))
                    chunk_num += 1
        else:
            chunks.append(create_chunk_metadata(
                current_chunk.strip(), file_path, chunk_num
            ))
    
    return chunks

def create_chunk_metadata(text: str, file_path: Path, chunk_num: int) -> Dict:
    """Create metadata for a text chunk"""
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:6]
    chunk_id = f"{file_path.stem}_{file_hash}_{chunk_num}_{content_hash}"
    
    return {
        'id': chunk_id,
        'text': text,
        'metadata': {
            'source': str(file_path),
            'filename': file_path.name,
            'chunk_number': chunk_num,
            'file_type': file_path.suffix[1:] if file_path.suffix else 'unknown'
        }
    }

def process_single_file(file_path: Path, openai_client) -> List[Dict]:
    """Process a single file and return chunks with embeddings"""
    try:
        content = extract_content(file_path)
        if not content.strip():
            return []
        
        chunks = create_chunks(content, file_path)
        
        # Generate embeddings for all chunks with caching
        cache_dir = Path("./embedding_cache")
        cache_dir.mkdir(exist_ok=True)
        
        for chunk in chunks:
            text = chunk['text']
            
            # Token pre-flight check
            token_count = count_tokens(text)
            if token_count > 7500:
                logger.warning(f"Truncating text: {token_count} tokens -> 7500")
                text = truncate_text_to_tokens(text, 7500)
            
            # Check cache first
            text_hash = get_text_hash(text)
            cache_file = cache_dir / f"{text_hash}.npy"
            
            if cache_file.exists():
                embedding = np.load(cache_file).tolist()
                chunk['embedding'] = embedding
            else:
                # Need to embed this text
                try:
                    response_embeddings = create_embeddings_with_retry(openai_client, [text])
                    embedding = response_embeddings[0]
                    chunk['embedding'] = embedding
                    
                    # Cache the result
                    np.save(cache_file, np.array(embedding))
                except Exception as e:
                    logger.error(f"Failed to embed text after retries: {e}")
                    # Use zero embedding as fallback
                    chunk['embedding'] = [0.0] * 3072
            
            # Rate limiting
            time.sleep(0.1)
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

class OpenAIKnowledgeBuilder:
    """
    Builds separated knowledge bases using OpenAI embeddings only:
    - Variables DB: Entity lookup for 36K concept-based variables (no duplicates)
    - Methodology DB: Conceptual search for documentation
    
    Key improvements:
    - OpenAI embeddings only (no sentence-transformers)
    - Bulletproof retry logic and caching
    - Array synchronization validation
    """
    
    def __init__(self, source_dir: Path, build_mode: str,
                 variables_dir: Path = None, methodology_dir: Path = None,
                 test_mode: bool = False, workers: int = 1, use_faiss: bool = False):
        
        self.source_dir = Path(source_dir)
        self.build_mode = build_mode
        self.variables_dir = Path(variables_dir) if variables_dir else None
        self.methodology_dir = Path(methodology_dir) if methodology_dir else None
        self.test_mode = test_mode
        self.workers = workers
        self.use_faiss = use_faiss and (build_mode in ['variables', 'both'])
        
        # Stats tracking
        self.variables_stats = {'concepts_processed': 0, 'survey_instances_processed': 0, 'errors': 0}
        self.methodology_stats = {'files_processed': 0, 'chunks_created': 0, 'errors': 0}
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        # Initialize databases
        self._init_databases()
        
        # Validate FAISS
        if self.use_faiss and not FAISS_AVAILABLE:
            logger.error("FAISS requested but not available. Install with: pip install faiss-cpu")
            raise ImportError("FAISS not available")
        
        logger.info(f"🚀 OpenAI Knowledge Builder initialized:")
        logger.info(f"   Build mode: {build_mode}")
        logger.info(f"   Variables dir: {variables_dir}")
        logger.info(f"   Variables backend: {'FAISS' if self.use_faiss else 'ChromaDB'}")
        logger.info(f"   Methodology dir: {methodology_dir}")
        logger.info(f"   Test mode: {test_mode}")
        logger.info(f"   🎯 OpenAI embeddings only (text-embedding-3-large)")
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.openai_client = OpenAI(api_key=api_key)
        logger.info(f"✅ OpenAI client initialized")
    
    def _generate_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        embeddings = []
        batch_size = 100  # OpenAI batch limit
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Only log every 5th batch to reduce noise
            if (i//batch_size + 1) % 5 == 0 or i == 0:
                logger.info(f"🧠 Generating OpenAI embeddings for batch {i//batch_size + 1}/{total_batches} ({len(batch_texts)} texts)...")
            
            response = self.openai_client.embeddings.create(
                input=batch_texts,
                model="text-embedding-3-large"
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            # Rate limiting
            time.sleep(0.1)
        
        return np.array(embeddings, dtype=np.float32)
    
    def _init_databases(self):
        """Initialize ChromaDB collections based on build mode"""
        self.variables_client = None
        self.methodology_client = None
        self.variables_collection = None
        self.methodology_collection = None
        
        if self.build_mode in ['variables', 'both']:
            self.variables_dir.mkdir(parents=True, exist_ok=True)
            self.variables_client = chromadb.PersistentClient(
                path=str(self.variables_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            try:
                self.variables_collection = self.variables_client.get_collection("census_variables")
            except:
                self.variables_collection = self.variables_client.create_collection(
                    "census_variables",
                    metadata={"description": "Census concept-based variables for entity lookup"}
                )
            logger.info(f"✅ Variables ChromaDB initialized: {self.variables_dir}")
        
        if self.build_mode in ['methodology', 'both']:
            self.methodology_dir.mkdir(parents=True, exist_ok=True)
            self.methodology_client = chromadb.PersistentClient(
                path=str(self.methodology_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            try:
                self.methodology_collection = self.methodology_client.get_collection("census_methodology")
            except:
                self.methodology_collection = self.methodology_client.create_collection(
                    "census_methodology",
                    metadata={"description": "Census methodology and documentation"}
                )
            logger.info(f"✅ Methodology ChromaDB initialized: {self.methodology_dir}")
    
    def build_knowledge_bases(self, rebuild: bool = False):
        """Main orchestration method"""
        start_time = time.time()
        
        print(f"🚀 Starting OpenAI-only knowledge base build...")
        
        if self.build_mode in ['variables', 'both']:
            self._build_variables_database(rebuild)
        
        if self.build_mode in ['methodology', 'both']:
            self._build_methodology_database(rebuild)
        
        build_time = time.time() - start_time
        self._display_final_stats(build_time)
    
    def _build_variables_database(self, rebuild: bool = False):
        """Build the concept-based variables database"""
        canonical_path = self.source_dir / "canonical_variables_refactored.json"
        if not canonical_path.exists():
            canonical_path = self.source_dir / "canonical_variables.json"
            if not canonical_path.exists():
                raise FileNotFoundError(f"No canonical variables file found in {self.source_dir}")
        
        logger.info(f"📁 Using canonical variables: {canonical_path}")
        
        if self.use_faiss:
            self._build_variables_faiss(canonical_path)
        else:
            self._build_variables_chromadb(canonical_path)
    
    def _load_canonical_variables(self, canonical_path: Path) -> tuple[dict, bool]:
        """Load canonical variables and detect structure type"""
        with open(canonical_path) as f:
            data = json.load(f)
        
        is_refactored = self._is_refactored_structure(data)
        
        if is_refactored:
            logger.info("🎯 Detected CONCEPT-BASED structure (refactored)")
            
            if 'concepts' in data and data['concepts']:
                concepts = data['concepts']
                logger.info(f"Using 'concepts' key: {len(concepts)} variables found")
            else:
                concepts = {k: v for k, v in data.items()
                           if k != 'metadata' and isinstance(v, dict)}
                logger.info(f"Using root-level parsing: {len(concepts)} variables found")
            
            if not concepts:
                raise ValueError("No concepts found in canonical data - check file structure")
        else:
            logger.info("📦 Detected TEMPORAL structure (original)")
            concepts = data.get('variables', data)
            
            if not concepts:
                raise ValueError("No variables found in canonical data")
        
        logger.info(f"✅ Loaded {len(concepts)} variables for processing")
        return concepts, is_refactored
    
    def _is_refactored_structure(self, data: dict) -> bool:
        """Detect if this is the new concept-based structure"""
        if 'metadata' in data and isinstance(data.get('metadata'), dict):
            metadata = data['metadata']
            if metadata.get('structure_type') == 'concept_based':
                return True
            if 'consolidation_stats' in metadata:
                return True
        
        sample_keys = list(data.keys())[:5]
        for key in sample_keys:
            if key == 'metadata':
                continue
            if isinstance(data.get(key), dict):
                item = data[key]
                if 'concept_name' in item or 'survey_instances' in item:
                    return True
        
        return False
    
    def _create_concept_embedding_text(self, variable_id: str, concept_data: dict, is_refactored: bool) -> tuple[str, dict]:
        """Create optimized embedding text and metadata for concept"""
        if is_refactored:
            concept_name = concept_data.get('concept', variable_id)
            label = concept_data.get('label', '')
            summary = concept_data.get('summary', '')
            key_terms = concept_data.get('key_terms', [])
            
            parts = []
            
            if concept_name and concept_name != variable_id:
                parts.append(concept_name)
            
            if label and label.lower() != concept_name.lower():
                parts.append(label)
            
            if summary:
                summary_first = summary.split('.')[0] + '.' if '.' in summary else summary
                parts.append(summary_first)
            
            if key_terms and isinstance(key_terms, list):
                terms_text = ', '.join(key_terms[:5])
                parts.append(f"Key terms: {terms_text}")
            
            instances = concept_data.get('instances', [])
            if instances:
                survey_types = set()
                for instance in instances:
                    if isinstance(instance, dict) and 'survey_type' in instance:
                        survey_types.add(instance['survey_type'])
                if survey_types:
                    parts.append(f"Available in: {', '.join(sorted(survey_types))}")
                
                self.variables_stats['survey_instances_processed'] += len(instances)
            
            metadata = {
                'variable_id': variable_id,
                'concept_name': concept_name,
                'description': label,
                'summary': summary,
                'key_terms': ', '.join(key_terms[:10]) if isinstance(key_terms, list) else str(key_terms),
                'survey_instances_count': len(instances) if isinstance(instances, list) else 0,
                'structure_type': 'concept_based'
            }
        else:
            label = concept_data.get('label', variable_id)
            description = concept_data.get('description', '')
            
            parts = [label]
            if description and description.lower() != label.lower():
                parts.append(description)
            
            metadata = {
                'variable_id': variable_id,
                'label': label,
                'description': description,
                'structure_type': 'temporal_based'
            }
            
            for field in ['universe', 'group', 'survey']:
                if field in concept_data:
                    metadata[field] = concept_data[field]
        
        text_parts = [part for part in parts if part and part.strip()]
        text = " | ".join(text_parts) + "." if text_parts else f"{variable_id}."
        return text, metadata
    
    def _build_variables_faiss(self, canonical_path: Path):
        """Build concept-based variables database using FAISS index"""
        logger.info("🎯 Processing canonical variables for FAISS database...")
        
        concepts, is_refactored = self._load_canonical_variables(canonical_path)
        logger.info(f"📊 Found {len(concepts)} {'concepts' if is_refactored else 'temporal variables'}")
        
        concept_items = list(concepts.items())
        if self.test_mode:
            concept_items = concept_items[:1000]
            logger.info(f"🧪 Test mode: Limited to {len(concept_items)} variables")
        
        all_texts = []
        all_metadata = []
        all_embeddings = []
        all_variable_ids = []
        
        batch_size = 1000
        total_batches = (len(concept_items) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(concept_items), batch_size)):
            batch = concept_items[i:i + batch_size]
            logger.info(f"🔄 Processing FAISS batch {batch_num + 1}/{total_batches}")
            
            batch_texts = []
            batch_metadata = []
            batch_variable_ids = []
            
            for variable_id, concept_data in batch:
                text, metadata = self._create_concept_embedding_text(variable_id, concept_data, is_refactored)
                
                batch_texts.append(text)
                batch_metadata.append(metadata)
                batch_variable_ids.append(variable_id)
            
            logger.info(f"🧠 Generating embeddings for {len(batch_texts)} variables...")
            embeddings = self._generate_openai_embeddings(batch_texts)
            
            # Array synchronization validation
            if not (len(batch_texts) == len(batch_metadata) == len(batch_variable_ids) == len(embeddings)):
                raise RuntimeError(
                    f"Array synchronization error in batch {batch_num + 1}: "
                    f"texts={len(batch_texts)}, metadata={len(batch_metadata)}, "
                    f"ids={len(batch_variable_ids)}, embeddings={len(embeddings)}"
                )
            
            all_texts.extend(batch_texts)
            all_metadata.extend(batch_metadata)
            all_embeddings.extend(embeddings)
            all_variable_ids.extend(batch_variable_ids)
            
            self.variables_stats['concepts_processed'] += len(batch_texts)
        
        # Final synchronization validation
        if not (len(all_texts) == len(all_metadata) == len(all_embeddings) == len(all_variable_ids)):
            raise RuntimeError(
                f"Final array synchronization error: "
                f"texts={len(all_texts)}, metadata={len(all_metadata)}, "
                f"embeddings={len(all_embeddings)}, ids={len(all_variable_ids)}"
            )
        
        logger.info(f"✅ Array synchronization validated: {len(all_variable_ids)} variables")
        
        # Build FAISS index
        logger.info(f"🔧 Building FAISS index for {len(all_embeddings)} variables...")
        
        embeddings_array = np.array(all_embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Save FAISS index
        faiss_path = self.variables_dir / "variables.faiss"
        faiss.write_index(index, str(faiss_path))
        logger.info(f"💾 FAISS index saved: {faiss_path}")
        
        # Save metadata
        metadata_path = self.variables_dir / "variables_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        logger.info(f"💾 Metadata saved: {metadata_path}")
        
        # Save variable ID mapping
        ids_mapping = {
            'variable_ids': all_variable_ids,
            'total_variables': len(all_variable_ids),
            'embedding_dimension': dimension,
            'created_timestamp': time.time(),
            'source_file': canonical_path.name,
            'structure_type': 'concept_based' if is_refactored else 'temporal_based'
        }
        
        ids_path = self.variables_dir / "variables_ids.json"
        with open(ids_path, 'w') as f:
            json.dump(ids_mapping, f, indent=2)
        logger.info(f"💾 Variable IDs mapping saved: {ids_path}")
        
        # Save build info
        build_info = {
            'embedding_model': 'text-embedding-3-large',
            'embedding_dimension': dimension,
            'variable_count': len(all_embeddings),
            'structure_type': 'concept_based' if is_refactored else 'temporal_based',
            'source_file': canonical_path.name,
            'build_timestamp': time.time(),
            'index_type': 'faiss_flat_l2',
            'survey_instances_processed': self.variables_stats['survey_instances_processed'],
            'has_id_mapping': True,
            'arrays_synchronized': True
        }
        
        build_info_path = self.variables_dir / "build_info.json"
        with open(build_info_path, 'w') as f:
            json.dump(build_info, f, indent=2)
        logger.info(f"💾 Build info saved: {build_info_path}")
        
        structure_note = "concept-based variables" if is_refactored else "temporal variables"
        logger.info(f"✅ FAISS variables database complete: {len(all_embeddings)} {structure_note}")
    
    def _build_variables_chromadb(self, canonical_path: Path):
        """Build concept-based variables database using ChromaDB"""
        logger.info("🎯 Processing canonical variables for ChromaDB...")
        
        # Clear existing collection if rebuilding
        try:
            self.variables_client.delete_collection("census_variables")
            self.variables_collection = self.variables_client.create_collection(
                "census_variables",
                metadata={"description": "Census concept-based variables for entity lookup"}
            )
            logger.info("🗑️ Cleared existing variables collection")
        except:
            pass
        
        concepts, is_refactored = self._load_canonical_variables(canonical_path)
        logger.info(f"📊 Found {len(concepts)} {'concepts' if is_refactored else 'temporal variables'}")
        
        concept_items = list(concepts.items())
        if self.test_mode:
            concept_items = concept_items[:1000]
            logger.info(f"🧪 Test mode: Limited to {len(concept_items)} variables")
        
        batch_size = 1000
        total_batches = (len(concept_items) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(concept_items), batch_size)):
            batch = concept_items[i:i + batch_size]
            logger.info(f"🔄 Processing ChromaDB batch {batch_num + 1}/{total_batches}")
            
            ids = []
            texts = []
            metadatas = []
            
            for variable_id, concept_data in batch:
                text, metadata = self._create_concept_embedding_text(variable_id, concept_data, is_refactored)
                
                ids.append(variable_id)
                texts.append(text)
                metadatas.append(metadata)
            
            # Add to ChromaDB
            self.variables_collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.variables_stats['concepts_processed'] += len(batch)
        
        structure_note = "concept-based variables" if is_refactored else "temporal variables"
        logger.info(f"✅ ChromaDB variables database complete: {self.variables_stats['concepts_processed']} {structure_note}")
    
    def _build_methodology_database(self, rebuild: bool = False):
        """Build the methodology database from documentation files"""
        logger.info("📚 Building methodology database...")
        
        # Clear existing collection if rebuilding
        if rebuild:
            try:
                self.methodology_client.delete_collection("census_methodology")
                self.methodology_collection = self.methodology_client.create_collection(
                    "census_methodology",
                    metadata={"description": "Census methodology and documentation"}
                )
                logger.info("🗑️ Cleared existing methodology collection")
            except:
                pass
        
        # Find documentation files
        exclude_patterns = [
            'canonical_variables.json',
            'canonical_variables_refactored.json',
            'failed_variables_retry.json',
            'acs1_raw.json',
            'acs5_raw.json',
            'raw_data',
            'data_dumps',
            '__pycache__',
            '.git',
            'node_modules',
            'build',
            'dist',
            '.DS_Store'
        ]
        
        doc_files = []
        for f in self.source_dir.rglob('*'):
            if (f.is_file() and
                f.suffix.lower() in ['.pdf', '.md', '.txt', '.html', '.htm', '.csv', '.xlsx', '.Rmd'] and
                not any(exclude in str(f).lower() for exclude in exclude_patterns) and
                not any(part.startswith('.') for part in f.parts)):
                
                # Skip very large files that will hit OpenAI token limits
                if f.stat().st_size > 5 * 1024 * 1024:  # Skip files > 5MB
                    continue
                    
                # Skip very large files in test mode
                if self.test_mode and f.stat().st_size > 1 * 1024 * 1024:
                    continue
                    
                doc_files.append(f)
        
        if not doc_files:
            logger.warning(f"No documentation files found in {self.source_dir}")
            return
        
        logger.info(f"📄 Found {len(doc_files)} documentation files")
        
        # Process files sequentially (simplified for reliability)
        print(f"Processing {len(doc_files)} methodology files...")
        
        for i, file_path in enumerate(doc_files):
            try:
                # Show progress every 10 files
                if (i + 1) % 10 == 0 or i == 0 or i == len(doc_files) - 1:
                    show_progress(i + 1, len(doc_files), "File Processing")
                
                chunks = process_single_file(file_path, self.openai_client)
                
                if chunks:
                    # Add to ChromaDB in batches
                    batch_size = 500
                    for j in range(0, len(chunks), batch_size):
                        batch = chunks[j:j + batch_size]
                        
                        texts = [c['text'] for c in batch]
                        ids = [c['id'] for c in batch]
                        metadatas = [c['metadata'] for c in batch]
                        embeddings = [c['embedding'] for c in batch]
                        
                        self.methodology_collection.add(
                            documents=texts,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            ids=ids
                        )
                    
                    self.methodology_stats['files_processed'] += 1
                    self.methodology_stats['chunks_created'] += len(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.methodology_stats['errors'] += 1
        
        logger.info(f"✅ Methodology database complete: {self.methodology_stats['chunks_created']} chunks from {self.methodology_stats['files_processed']} files")
    
    def _display_final_stats(self, build_time):
        """Display simple completion message"""
        print(f"🎉 Build complete in {build_time:.1f}s!")
        
        if self.build_mode in ['variables', 'both']:
            print(f"   Variables: {self.variables_stats['concepts_processed']:,} concepts")
        
        if self.build_mode in ['methodology', 'both']:
            print(f"   Methodology: {self.methodology_stats['chunks_created']:,} chunks from {self.methodology_stats['files_processed']:,} files")
            if self.methodology_stats['errors'] > 0:
                print(f"   Errors: {self.methodology_stats['errors']}")

def main():
    parser = argparse.ArgumentParser(description='OpenAI-Only Knowledge Base Builder')
    parser.add_argument('--variables-only', action='store_true', help='Build only variables database')
    parser.add_argument('--methodology-only', action='store_true', help='Build only methodology database')
    parser.add_argument('--both', action='store_true', help='Build both databases')
    parser.add_argument('--faiss', action='store_true', help='Use FAISS for variables database (faster loading)')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild existing databases')
    parser.add_argument('--test-mode', action='store_true', help='Test with subset of data')
    parser.add_argument('--source-dir', default='source-docs', help='Source directory')
    parser.add_argument('--variables-dir', default='variables-db', help='Variables database directory')
    parser.add_argument('--methodology-dir', default='methodology-db', help='Methodology database directory')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (unused in this version)')
    
    args = parser.parse_args()
    
    # Determine build mode
    if args.variables_only:
        build_mode = 'variables'
    elif args.methodology_only:
        build_mode = 'methodology'
    elif args.both:
        build_mode = 'both'
    else:
        # Default to both if no mode specified
        build_mode = 'both'
        logger.info("No build mode specified, defaulting to --both")
    
    # Validate FAISS usage
    if args.faiss and build_mode == 'methodology':
        logger.warning("FAISS flag ignored - only applies to variables database")
        args.faiss = False
    
    builder = OpenAIKnowledgeBuilder(
        source_dir=Path(args.source_dir),
        build_mode=build_mode,
        variables_dir=Path(args.variables_dir),
        methodology_dir=Path(args.methodology_dir),
        test_mode=args.test_mode,
        workers=args.workers,
        use_faiss=args.faiss
    )
    
    builder.build_knowledge_bases(rebuild=args.rebuild)
    logger.info("🚀 OpenAI-only knowledge base build completed!")

if __name__ == "__main__":
    main()
