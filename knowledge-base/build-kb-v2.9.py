#!/usr/bin/env python3
"""
OpenAI Knowledge Base Builder v2.9 - Token-Based Chunking
MINIMAL UPGRADE FROM v2.8.2:
- Token-based chunking instead of character-based (800 token chunks)
- All v2.8.2 performance optimizations preserved
- Zero complexity added, maximum compatibility

Usage:
    python build-kb-v2.9.py --methodology-only --workers 8
    python build-kb-v2.9.py --both --faiss --workers 6
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress OpenAI HTTP request logging spam
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global cache directory
CACHE_DIR = Path("./embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

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

def split_by_tokens(text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    """Split text into token-sized chunks with overlap"""
    if not TIKTOKEN_AVAILABLE:
        # Fallback to character-based splitting
        char_size = target_tokens * 4
        overlap_size = overlap_tokens * 4
        return [text[i:i+char_size] for i in range(0, len(text), char_size-overlap_size)]
    
    try:
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        tokens = encoding.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), target_tokens - overlap_tokens):
            chunk_tokens = tokens[i:i + target_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    except:
        # Fallback if tiktoken fails
        char_size = target_tokens * 4
        overlap_size = overlap_tokens * 4
        return [text[i:i+char_size] for i in range(0, len(text), char_size-overlap_size)]

def get_token_overlap(text: str, overlap_tokens: int) -> str:
    """Get overlap text by tokens from end of text"""
    if not TIKTOKEN_AVAILABLE:
        overlap_chars = overlap_tokens * 4
        return text[-overlap_chars:] if len(text) > overlap_chars else text
    
    try:
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        tokens = encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        overlap_tokens_slice = tokens[-overlap_tokens:]
        return encoding.decode(overlap_tokens_slice)
    except:
        overlap_chars = overlap_tokens * 4
        return text[-overlap_chars:] if len(text) > overlap_chars else text

def create_chunks(content: str, file_path: Path) -> List[Dict]:
    """Token-based chunking - MAIN CHANGE from v2.8.2"""
    chunks = []
    
    # Clean text
    content = re.sub(r'\s+', ' ', content).strip()
    
    if len(content) < 100:
        return chunks
    
    # Token-based settings instead of character-based
    target_tokens = 800  # Good balance for embeddings
    overlap_tokens = 150
    max_safe_tokens = 1200  # Force split if larger
    
    # Same paragraph-based splitting logic as v2.8.2
    paragraphs = content.split('\n\n')
    
    # If we only got 1 paragraph and it's huge, use fallback splitting
    if len(paragraphs) == 1 and count_tokens(paragraphs[0]) > max_safe_tokens:
        # Fallback: split by sentences first
        sentences = re.split(r'[.!?]+\s+', content)
        if len(sentences) > 1:
            paragraphs = sentences
        else:
            # Last resort: split by token chunks
            paragraphs = split_by_tokens(content, target_tokens, overlap_tokens)
    
    current_chunk = ""
    chunk_num = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If this single paragraph is still too large, force split it
        para_tokens = count_tokens(paragraph)
        if para_tokens > max_safe_tokens:
            # Split oversized paragraph into token-sized pieces
            pieces = split_by_tokens(paragraph, target_tokens, overlap_tokens)
            
            for piece in pieces:
                if count_tokens(piece) > 50:  # Minimum viable chunk
                    chunks.append(create_chunk_metadata(
                        piece.strip(), file_path, chunk_num
                    ))
                    chunk_num += 1
            continue
        
        # Normal processing - check token count of combined chunk
        test_chunk = current_chunk + " " + paragraph if current_chunk else paragraph
        test_tokens = count_tokens(test_chunk)
        
        if test_tokens > target_tokens and current_chunk:
            # Save current chunk
            if count_tokens(current_chunk) > 50:
                chunks.append(create_chunk_metadata(
                    current_chunk.strip(), file_path, chunk_num
                ))
                chunk_num += 1
            
            # Start new chunk with token-based overlap
            overlap_text = get_token_overlap(current_chunk, overlap_tokens)
            current_chunk = overlap_text + " " + paragraph if overlap_text else paragraph
        else:
            current_chunk = test_chunk
    
    # Add final chunk
    if current_chunk and count_tokens(current_chunk) > 50:
        # Final safety check
        if count_tokens(current_chunk) > max_safe_tokens:
            # Force split the final chunk
            pieces = split_by_tokens(current_chunk, target_tokens, overlap_tokens)
            for piece in pieces:
                if count_tokens(piece) > 50:
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
    """Create metadata for a text chunk with token count"""
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
            'file_type': file_path.suffix[1:] if file_path.suffix else 'unknown',
            'token_count': count_tokens(text),
            'chunking_version': 'v2.9_token_based'
        }
    }

def get_cached_embedding(text: str) -> Optional[List[float]]:
    """Get cached embedding if it exists"""
    text_hash = get_text_hash(text)
    cache_file = CACHE_DIR / f"{text_hash}.npy"
    
    if cache_file.exists():
        try:
            return np.load(cache_file).tolist()
        except:
            # Corrupted cache file, remove it
            cache_file.unlink(missing_ok=True)
            return None
    return None

def cache_embedding(text: str, embedding: List[float]):
    """Cache an embedding to disk"""
    text_hash = get_text_hash(text)
    cache_file = CACHE_DIR / f"{text_hash}.npy"
    
    try:
        np.save(cache_file, np.array(embedding))
    except Exception as e:
        logger.warning(f"Failed to cache embedding: {e}")

def batch_embed_texts_massive(openai_client, texts: List[str]) -> List[List[float]]:
    """
    MASSIVE batch embedding - process hundreds/thousands of texts efficiently
    """
    if not texts:
        return []
    
    embeddings = []
    texts_to_embed = []
    embedding_indices = []
    
    # Check cache for each text
    for i, text in enumerate(texts):
        # Token pre-flight check
        token_count = count_tokens(text)
        if token_count > 7500:
            text = truncate_text_to_tokens(text, 7500)
        
        cached = get_cached_embedding(text)
        if cached is not None:
            embeddings.append(cached)
        else:
            embeddings.append(None)  # Placeholder
            texts_to_embed.append(text)
            embedding_indices.append(i)
    
    # Batch embed any uncached texts
    if texts_to_embed:
        cache_hits = len(texts) - len(texts_to_embed)
        logger.info(f"🚀 MASSIVE BATCH: Embedding {len(texts_to_embed)} new texts (found {cache_hits} in cache)")
        
        batch_size = 100  # OpenAI batch limit
        new_embeddings = []
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                batch_embeddings = create_embeddings_with_retry(openai_client, batch_texts)
                new_embeddings.extend(batch_embeddings)
                
                # Cache each embedding
                for text, embedding in zip(batch_texts, batch_embeddings):
                    cache_embedding(text, embedding)
                
                # Progress every 5 batches or on last batch
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"   Batch {batch_num}/{total_batches} complete ({len(batch_texts)} texts)")
                
                # Minimal rate limiting
                time.sleep(0.02)
                
            except Exception as e:
                logger.error(f"Failed to embed batch {batch_num} after retries: {e}")
                # Use zero embeddings as fallback
                fallback_embeddings = [[0.0] * 3072] * len(batch_texts)
                new_embeddings.extend(fallback_embeddings)
        
        # Fill in the new embeddings
        for idx, embedding in zip(embedding_indices, new_embeddings):
            embeddings[idx] = embedding
    
    return embeddings

def process_methodology_worker_crossfile(args):
    """
    CROSS-FILE BATCHING: Worker processes ALL files, collects ALL chunks, then batch embeds everything
    """
    worker_id, files_batch, api_key, min_file_size = args
    
    try:
        # Initialize OpenAI client for this worker
        openai_client = OpenAI(api_key=api_key)
        
        all_chunks = []
        files_processed = 0
        files_skipped = 0
        skipped_files = []
        errors = 0
        
        logger.info(f"Worker {worker_id}: Processing {len(files_batch)} files...")
        
        # PHASE 1: Process all files and collect chunks (no embedding yet)
        for i, file_path in enumerate(files_batch):
            try:
                # Check file size and skip tiny files
                file_size = file_path.stat().st_size
                if file_size < min_file_size:
                    skipped_files.append({
                        'file': str(file_path),
                        'size_bytes': file_size,
                        'reason': f'File too small ({file_size} bytes < {min_file_size} bytes)'
                    })
                    files_skipped += 1
                    continue
                
                # Progress logging every 50 files
                if (i + 1) % 50 == 0 or i == 0 or i == len(files_batch) - 1:
                    logger.info(f"Worker {worker_id}: Processed {i + 1}/{len(files_batch)} files")
                
                content = extract_content(file_path)
                if not content.strip():
                    continue
                
                chunks = create_chunks(content, file_path)
                if chunks:
                    all_chunks.extend(chunks)
                    files_processed += 1
                
            except Exception as e:
                logger.error(f"Worker {worker_id}: Error processing {file_path}: {e}")
                errors += 1
        
        logger.info(f"Worker {worker_id}: Collected {len(all_chunks)} chunks from {files_processed} files")
        logger.info(f"Worker {worker_id}: Skipped {files_skipped} small files")
        
        # PHASE 2: MASSIVE batch embedding of all collected chunks
        if all_chunks:
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            embeddings = batch_embed_texts_massive(openai_client, chunk_texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk['embedding'] = embedding
        
        return {
            'worker_id': worker_id,
            'chunks': all_chunks,
            'files_processed': files_processed,
            'files_skipped': files_skipped,
            'skipped_files': skipped_files,
            'chunks_created': len(all_chunks),
            'errors': errors
        }
        
    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {e}")
        return {
            'worker_id': worker_id,
            'chunks': [],
            'files_processed': 0,
            'files_skipped': 0,
            'skipped_files': [],
            'chunks_created': 0,
            'errors': len(files_batch)
        }

class OpenAIKnowledgeBuilder:
    """
    High-performance knowledge base builder with cross-file batching and token-based chunking
    """
    
    def __init__(self, source_dir: Path, build_mode: str,
                 variables_dir: Path = None, methodology_dir: Path = None,
                 test_mode: bool = False, workers: int = 4, use_faiss: bool = False,
                 min_file_size: int = 1024):  # Skip files smaller than 1KB
        
        self.source_dir = Path(source_dir)
        self.build_mode = build_mode
        self.variables_dir = Path(variables_dir) if variables_dir else None
        self.methodology_dir = Path(methodology_dir) if methodology_dir else None
        self.test_mode = test_mode
        self.workers = max(1, min(workers, multiprocessing.cpu_count()))
        self.use_faiss = use_faiss and (build_mode in ['variables', 'both'])
        self.min_file_size = min_file_size
        
        # Stats tracking
        self.variables_stats = {'concepts_processed': 0, 'survey_instances_processed': 0, 'errors': 0}
        self.methodology_stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'chunks_created': 0,
            'errors': 0,
            'skipped_files': []
        }
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        # Initialize databases
        self._init_databases()
        
        # Validate FAISS
        if self.use_faiss and not FAISS_AVAILABLE:
            logger.error("FAISS requested but not available. Install with: pip install faiss-cpu")
            raise ImportError("FAISS not available")
        
        logger.info(f"🚀 OpenAI Knowledge Builder v2.9 initialized:")
        logger.info(f"   Build mode: {build_mode}")
        logger.info(f"   Variables dir: {variables_dir}")
        logger.info(f"   Variables backend: {'FAISS' if self.use_faiss else 'ChromaDB'}")
        logger.info(f"   Methodology dir: {methodology_dir}")
        logger.info(f"   Workers: {self.workers}")
        logger.info(f"   Min file size: {self.min_file_size} bytes")
        logger.info(f"   Test mode: {test_mode}")
        logger.info(f"   🎯 Token-based chunking (800 tokens/chunk)")
        logger.info(f"   🎯 OpenAI embeddings only (text-embedding-3-large)")
        logger.info(f"   ⚡ CROSS-FILE BATCHING enabled for maximum performance")
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.openai_client = OpenAI(api_key=self.api_key)
        logger.info(f"✅ OpenAI client initialized")
    
    def _generate_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API with massive batching"""
        logger.info(f"🧠 Generating embeddings for {len(texts)} texts...")
        
        embeddings = batch_embed_texts_massive(self.openai_client, texts)
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
        
        print(f"🚀 Starting TOKEN-BASED CHUNKING knowledge base build...")
        print(f"   Workers: {self.workers}")
        print(f"   Cache directory: {CACHE_DIR}")
        print(f"   Min file size: {self.min_file_size} bytes")
        print(f"   Target chunk size: 800 tokens")
        
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
                # NO ARBITRARY TRUNCATION - use full summary
                parts.append(summary)
            
            if key_terms and isinstance(key_terms, list):
                # NO ARBITRARY TRUNCATION - use all key terms
                terms_text = ', '.join(key_terms)
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
                'key_terms': ', '.join(key_terms) if isinstance(key_terms, list) else str(key_terms),
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
        
        batch_size = 1000
        total_batches = (len(concept_items) + batch_size - 1) // batch_size
        
        all_texts = []
        all_metadata = []
        all_embeddings = []
        all_variable_ids = []
        
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
            'arrays_synchronized': True,
            'performance_version': 'v2.9-token-based'
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
        except Exception:
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
        """Build the methodology database with CROSS-FILE BATCHING"""
        logger.info("📚 Building methodology database with CROSS-FILE BATCHING...")
        
        # Clear existing collection if rebuilding
        if rebuild:
            try:
                self.methodology_client.delete_collection("census_methodology")
                self.methodology_collection = self.methodology_client.create_collection(
                    "census_methodology",
                    metadata={"description": "Census methodology and documentation"}
                )
                logger.info("🗑️ Cleared existing methodology collection")
            except Exception:
                pass
        
        # Find documentation files - NO ARBITRARY SIZE LIMITS
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
                
                doc_files.append(f)
        
        if not doc_files:
            logger.warning(f"No documentation files found in {self.source_dir}")
            return
        
        total_size = sum(f.stat().st_size for f in doc_files)
        logger.info(f"📄 Found {len(doc_files)} documentation files ({total_size / 1024 / 1024:.1f} MB total)")
        logger.info(f"🎯 Processing ALL files - token truncation will handle oversized content")
        
        # Split files among workers
        files_per_worker = max(1, len(doc_files) // self.workers)
        worker_args = []
        
        for i in range(self.workers):
            start_idx = i * files_per_worker
            if i == self.workers - 1:  # Last worker gets remaining files
                end_idx = len(doc_files)
            else:
                end_idx = (i + 1) * files_per_worker
            
            files_batch = doc_files[start_idx:end_idx]
            if files_batch:
                worker_args.append((i, files_batch, self.api_key, self.min_file_size))
        
        logger.info(f"🚀 Starting {len(worker_args)} workers for CROSS-FILE BATCHING...")
        
        # Process files in parallel with cross-file batching
        all_chunks = []
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            future_to_worker = {executor.submit(process_methodology_worker_crossfile, args): args[0]
                              for args in worker_args}
            
            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    result = future.result()
                    all_chunks.extend(result['chunks'])
                    
                    self.methodology_stats['files_processed'] += result['files_processed']
                    self.methodology_stats['files_skipped'] += result['files_skipped']
                    self.methodology_stats['chunks_created'] += result['chunks_created']
                    self.methodology_stats['errors'] += result['errors']
                    self.methodology_stats['skipped_files'].extend(result['skipped_files'])
                    
                    logger.info(f"✅ Worker {worker_id} completed: {result['files_processed']} files processed, "
                              f"{result['chunks_created']} chunks, {result['files_skipped']} files skipped, "
                              f"{result['errors']} errors")
                    
                except Exception as e:
                    logger.error(f"❌ Worker {worker_id} failed: {e}")
                    self.methodology_stats['errors'] += 1
        
        logger.info(f"🔗 Merging {len(all_chunks)} chunks into methodology database...")
        
        # Add all chunks to ChromaDB in batches
        if all_chunks:
            batch_size = 1000
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                
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
                
                if batch_num % 10 == 0 or batch_num == total_batches:
                    logger.info(f"   Merged batch {batch_num}/{total_batches}")
        
        # Save skipped files log (only files below min_file_size)
        if self.methodology_stats['skipped_files']:
            skipped_log_path = self.methodology_dir / "skipped_files.json"
            with open(skipped_log_path, 'w') as f:
                json.dump({
                    'total_skipped': len(self.methodology_stats['skipped_files']),
                    'skipped_files': self.methodology_stats['skipped_files'],
                    'summary': {
                        'min_file_size_bytes': self.min_file_size,
                        'total_files_found': len(doc_files) + len(self.methodology_stats['skipped_files']),
                        'files_processed': self.methodology_stats['files_processed'],
                        'files_skipped': len(self.methodology_stats['skipped_files']),
                        'reason': f'Files below {self.min_file_size} bytes (web scraping artifacts)'
                    }
                }, f, indent=2)
            logger.info(f"💾 Skipped files log saved: {skipped_log_path}")
        
        logger.info(f"✅ Methodology database complete: {self.methodology_stats['chunks_created']} chunks from {self.methodology_stats['files_processed']} files")
        logger.info(f"🚫 Total files skipped: {self.methodology_stats['files_skipped']} (below {self.min_file_size} bytes)")
        if self.methodology_stats['errors'] > 0:
            logger.warning(f"⚠️ {self.methodology_stats['errors']} files had errors")
    
    def _display_final_stats(self, build_time):
        """Display comprehensive completion statistics"""
        print(f"\n🎉 TOKEN-BASED CHUNKING build complete in {build_time:.1f}s!")
        print(f"⚡ Performance: {self.workers} workers, massive batching enabled")
        print(f"🎯 Chunking: 800 tokens per chunk (vs 1000 chars in v2.8.2)")
        
        if self.build_mode in ['variables', 'both']:
            print(f"   📊 Variables: {self.variables_stats['concepts_processed']:,} concepts")
            print(f"      Backend: {'FAISS' if self.use_faiss else 'ChromaDB'}")
        
        if self.build_mode in ['methodology', 'both']:
            chunks_per_sec = self.methodology_stats['chunks_created'] / max(build_time, 1)
            print(f"   📚 Methodology: {self.methodology_stats['chunks_created']:,} chunks from {self.methodology_stats['files_processed']:,} files")
            print(f"      Performance: {chunks_per_sec:.1f} chunks/second")
            print(f"      Files skipped: {self.methodology_stats['files_skipped']:,} (too small)")
            if self.methodology_stats['errors'] > 0:
                print(f"      ⚠️ Errors: {self.methodology_stats['errors']}")
        
        # Cache statistics
        cache_files = len(list(CACHE_DIR.glob("*.npy")))
        print(f"   💾 Cache: {cache_files:,} embeddings cached")
        print(f"   🗂️ Skipped files log: methodology-db/skipped_files.json")

def main():
    parser = argparse.ArgumentParser(description='High-Performance OpenAI Knowledge Base Builder v2.9 - Token-Based Chunking')
    parser.add_argument('--variables-only', action='store_true', help='Build only variables database')
    parser.add_argument('--methodology-only', action='store_true', help='Build only methodology database')
    parser.add_argument('--both', action='store_true', help='Build both databases')
    parser.add_argument('--faiss', action='store_true', help='Use FAISS for variables database (faster loading)')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild existing databases')
    parser.add_argument('--test-mode', action='store_true', help='Test with subset of data')
    parser.add_argument('--source-dir', default='source-docs', help='Source directory')
    parser.add_argument('--variables-dir', default='variables-db', help='Variables database directory')
    parser.add_argument('--methodology-dir', default='methodology-db', help='Methodology database directory')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for methodology processing')
    parser.add_argument('--min-file-size', type=int, default=1024, help='Skip files smaller than this size in bytes')
    
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
    
    # Performance recommendations
    if args.workers > 8:
        logger.warning(f"High worker count ({args.workers}) may hit OpenAI rate limits. Consider 4-8 workers.")
    
    builder = OpenAIKnowledgeBuilder(
        source_dir=Path(args.source_dir),
        build_mode=build_mode,
        variables_dir=Path(args.variables_dir),
        methodology_dir=Path(args.methodology_dir),
        test_mode=args.test_mode,
        workers=args.workers,
        use_faiss=args.faiss,
        min_file_size=args.min_file_size
    )
    
    try:
        builder.build_knowledge_bases(rebuild=args.rebuild)
        logger.info("🚀 Token-based chunking knowledge base build completed successfully!")
    except KeyboardInterrupt:
        logger.info("⏹️ Build interrupted by user")
    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        raise

if __name__ == "__main__":
    main()
