#!/usr/bin/env python3
"""
Dual-Path Knowledge Base Builder - Concept-Based Variables Architecture
Builds TWO separate vector databases optimized for different retrieval patterns:

1. VARIABLES DATABASE: 36K concept-based variables → FAISS index (fast loading) OR ChromaDB
2. METHODOLOGY DATABASE: Documentation, guides, PDFs → ChromaDB (conceptual search)

Key Update: Handles canonical_variables_refactored.json with concept-based structure
- Eliminates duplicate variables (65K → 36K concepts)
- Survey instance awareness (ACS1/5yr as instances, not separate variables)
- Rich metadata preservation for intelligent search

Usage:
    python build-kb-concept-based.py --variables-only --output-dir variables-db --faiss
    python build-kb-concept-based.py --methodology-only --output-dir methodology-db
    python build-kb-concept-based.py --both --variables-dir variables-db --methodology-dir methodology-db --faiss
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import re

# Document processing
import PyPDF2
from bs4 import BeautifulSoup
import markdown
import pandas as pd

# Vector DB and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# FAISS for variables database
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - KB-BUILD - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def chunk_content(text, file_path, category, source_path, worker_id=None):
    """Smart chunking: structured data extraction for special files, recursive chunking for regular text"""
    chunks = []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) < 100:
        return chunks
    
    # For structured data files (like Census variables), use structured extraction
    is_structured = any(indicator in file_path.name.lower()
                       for indicator in ['variables', 'api', 'definitions', 'zcta', 'rel'])
    
    if is_structured and len(text) > 5000:  # Large structured files
        return chunk_structured_document(text, file_path, category, source_path, worker_id)
    
    # Standard recursive chunking for regular documents
    chunk_size = 1000
    overlap = 200  # 20% overlap
    
    # Split by paragraphs first (natural boundaries)
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    chunk_num = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # Test if adding this paragraph would exceed chunk size
        test_chunk = current_chunk + " " + paragraph if current_chunk else paragraph
        
        if len(test_chunk) > chunk_size and current_chunk:
            # Save current chunk
            if len(current_chunk.strip()) > 100:
                chunks.append(create_chunk_metadata(
                    current_chunk.strip(), file_path, category, chunk_num, source_path, worker_id
                ))
                chunk_num += 1
            
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + paragraph if overlap_text else paragraph
        else:
            current_chunk = test_chunk
    
    # Add final chunk
    if current_chunk and len(current_chunk.strip()) > 100:
        chunks.append(create_chunk_metadata(
            current_chunk.strip(), file_path, category, chunk_num, source_path, worker_id
        ))
    
    return chunks

def chunk_structured_document(text, file_path, category, source_path, worker_id=None):
    """Handle structured documents by splitting on natural entity boundaries"""
    chunks = []
    chunk_num = 0
    
    # Try splitting on common structured boundaries
    split_patterns = [
        r'\n(?=[A-Z]\d{5}_\d{3})',  # Census variable codes
        r'\n(?=Table [A-Z]\d+)',    # Table definitions
        r'\n(?=\w+:)',              # Key-value pairs
        r'\n\n',                    # Paragraph breaks
        r'\n'                       # Line breaks (last resort)
    ]
    
    sections = [text]  # Start with full text
    
    # Try each split pattern until chunks are reasonable size
    for pattern in split_patterns:
        new_sections = []
        for section in sections:
            if len(section) <= 2000:  # Reasonable size for structured content
                new_sections.append(section)
            else:
                # Split this section
                parts = re.split(pattern, section)
                new_sections.extend(parts)
        sections = new_sections
        
        # Check if we're at reasonable size now
        if all(len(s) <= 2000 for s in sections):
            break
    
    # Create chunks from sections
    for section in sections:
        section = section.strip()
        if len(section) >= 100:  # Minimum chunk size
            chunks.append(create_chunk_metadata(
                section, file_path, category, chunk_num, source_path, worker_id
            ))
            chunk_num += 1
    
    return chunks

def create_chunk_metadata(text, file_path, category, chunk_num, source_path, worker_id=None):
    """Create metadata for a text chunk"""
    
    # Generate globally unique ID using full file path + content
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:6]
    worker_prefix = f"w{worker_id}_" if worker_id is not None else ""
    chunk_id = f"{worker_prefix}{file_path.stem}_{file_hash}_{chunk_num}_{content_hash}"
    
    return {
        'id': chunk_id,
        'text': text,
        'metadata': {
            'source_file': str(file_path.relative_to(source_path)),
            'category': category,
            'chunk_number': chunk_num,
            'file_name': file_path.name,
            'file_type': file_path.suffix,
            'text_length': len(text)
        }
    }

def worker_process_files(files_chunk, worker_id, source_dir, temp_dir, model_name):
    """Process files in parallel worker with clean chunking"""
    
    # Set environment for offline model loading
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Load model from local cache
    model = SentenceTransformer(model_name, cache_folder='./model_cache', device='cpu')
    source_path = Path(source_dir)
    
    all_chunks = []
    files_processed = 0
    errors = 0
    
    # Progress tracking
    total_files = len(files_chunk)
    progress_interval = max(50, total_files // 10)
    
    print(f"🔄 Worker {worker_id}: Starting {total_files} files...")
    
    for idx, (file_path, category) in enumerate(files_chunk):
        try:
            # Progress updates
            if idx > 0 and idx % progress_interval == 0:
                percent = (idx / total_files) * 100
                print(f"Worker {worker_id}: {percent:.0f}% ({idx}/{total_files})")
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        if page.extract_text():
                            text += page.extract_text()
            elif file_path.suffix.lower() in ['.html', '.htm']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
            elif file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    md_content = f.read()
                    # Convert markdown to HTML then extract text
                    html = markdown.markdown(md_content)
                    soup = BeautifulSoup(html, 'html.parser')
                    text = soup.get_text()
            elif file_path.suffix.lower() == '.xlsx':
                excel_file = pd.ExcelFile(file_path)
                text_parts = []
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        if not df.empty:
                            text_parts.append(f"Sheet: {sheet_name}")
                            text_parts.append(f"Columns: {', '.join(df.columns.astype(str))}")
                            for _, row in df.head(10).iterrows():
                                row_text = ' | '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                                if row_text:
                                    text_parts.append(row_text)
                    except:
                        continue
                text = '\n'.join(text_parts)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                text = json.dumps(data)[:8000]
            elif file_path.suffix.lower() == '.csv':
                try:
                    df = pd.read_csv(file_path, nrows=100)
                    text = f"CSV: {file_path.name}\nColumns: {', '.join(df.columns)}\n"
                    text += df.to_string()
                except:
                    text = ""
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            if len(text.strip()) < 100:
                continue
            
            # Use clean recursive chunking with worker ID
            chunks = chunk_content(text, file_path, category, source_path, worker_id)
            
            # Generate embeddings
            if chunks:
                texts = [c['text'] for c in chunks]
                embeddings = model.encode(texts, show_progress_bar=False)
                
                for i, chunk in enumerate(chunks):
                    chunk['embedding'] = embeddings[i].tolist()
                
                all_chunks.extend(chunks)
                files_processed += 1
                
        except Exception as e:
            print(f"❌ Worker {worker_id}: ERROR in {file_path.name}: {str(e)}")
            errors += 1
            continue
    
    # Save to temp file
    temp_file = Path(temp_dir) / f"worker_{worker_id}.json"
    with open(temp_file, 'w') as f:
        json.dump({
            'chunks': all_chunks,
            'files_processed': files_processed,
            'errors': errors
        }, f)
    
    print(f"✅ Worker {worker_id}: COMPLETE - {files_processed} files, {len(all_chunks)} chunks, {errors} errors")
    return {'files_processed': files_processed, 'chunks_created': len(all_chunks), 'errors': errors}

class ConceptBasedKnowledgeBuilder:
    """
    Builds separated knowledge bases optimized for different retrieval patterns:
    - Variables DB: Entity lookup for 36K concept-based variables (no duplicates)
    - Methodology DB: Conceptual search for documentation
    
    Key improvements:
    - Handles canonical_variables_refactored.json structure
    - Survey instance awareness (ACS1/5yr metadata)
    - Rich metadata preservation for intelligent search
    """
    
    def __init__(self, source_dir: Path, build_mode: str,
                 variables_dir: Path = None, methodology_dir: Path = None,
                 test_mode: bool = False, model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 workers: int = 5, use_faiss: bool = False):
        
        self.source_dir = Path(source_dir)
        self.build_mode = build_mode  # 'variables', 'methodology', or 'both'
        self.variables_dir = Path(variables_dir) if variables_dir else None
        self.methodology_dir = Path(methodology_dir) if methodology_dir else None
        self.test_mode = test_mode
        self.model_name = model_name
        self.workers = workers
        
        # Stats tracking
        self.variables_stats = {
            'concepts_processed': 0,
            'survey_instances_processed': 0,
            'files_processed': 0,
            'chunks_created': 0,
            'errors': 0
        }
        self.methodology_stats = {'files_processed': 0, 'chunks_created': 0, 'errors': 0}
        
        # Initialize model
        self._init_model()
        
        # Initialize databases based on build mode
        self._init_databases()
        
        self.use_faiss = use_faiss and (build_mode in ['variables', 'both'])
        
        # Validate FAISS availability
        if self.use_faiss and not FAISS_AVAILABLE:
            logger.error("FAISS requested but not available. Install with: pip install faiss-cpu")
            raise ImportError("FAISS not available")
        
        logger.info(f"🚀 Concept-Based Knowledge Builder initialized:")
        logger.info(f"   Build mode: {build_mode}")
        logger.info(f"   Variables dir: {variables_dir}")
        logger.info(f"   Variables backend: {'FAISS' if self.use_faiss else 'ChromaDB'}")
        logger.info(f"   Methodology dir: {methodology_dir}")
        logger.info(f"   Test mode: {test_mode}")
        logger.info(f"   🎯 CONCEPT-BASED: Handles refactored canonical variables")
    
    def _init_model(self):
        """Initialize embedding model with local caching"""
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        logger.info(f"🔄 Loading model: {self.model_name} (768 dimensions)")
        self.embedding_model = SentenceTransformer(self.model_name, cache_folder='./model_cache')
        logger.info(f"✅ Model cached locally in ./model_cache")
    
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
            logger.info(f"✅ Variables database initialized: {self.variables_dir}")
        
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
                    metadata={"description": "Census methodology and documentation for conceptual search"}
                )
            logger.info(f"✅ Methodology database initialized: {self.methodology_dir}")
    
    def build_knowledge_bases(self, rebuild: bool = False):
        """Build the knowledge bases according to the specified mode"""
        logger.info(f"🚀 Building concept-based knowledge bases - Mode: {self.build_mode}")
        start_time = time.time()
        
        if rebuild:
            self._rebuild_collections()
        
        if self.build_mode in ['variables', 'both']:
            logger.info("🎯 Building CONCEPT-BASED VARIABLES database...")
            self._build_variables_database()
        
        if self.build_mode in ['methodology', 'both']:
            logger.info("📚 Building METHODOLOGY database...")
            self._build_methodology_database()
        
        build_time = time.time() - start_time
        self._display_final_stats(build_time)
    
    def _rebuild_collections(self):
        """Rebuild collections if they exist"""
        if self.build_mode in ['variables', 'both']:
            if self.use_faiss:
                # Remove existing FAISS files
                faiss_files = ['variables.faiss', 'variables_metadata.json']
                for fname in faiss_files:
                    fpath = self.variables_dir / fname
                    if fpath.exists():
                        fpath.unlink()
                        logger.info(f"🔄 Removed existing FAISS file: {fname}")
            else:
                # Rebuild ChromaDB collection
                if self.variables_client:
                    try:
                        self.variables_client.delete_collection("census_variables")
                        self.variables_collection = self.variables_client.create_collection(
                            "census_variables",
                            metadata={"description": "Census concept-based variables for entity lookup"}
                        )
                        logger.info("🔄 Variables collection rebuilt")
                    except:
                        pass
        
        if self.methodology_client and self.build_mode in ['methodology', 'both']:
            try:
                self.methodology_client.delete_collection("census_methodology")
                self.methodology_collection = self.methodology_client.create_collection(
                    "census_methodology",
                    metadata={"description": "Census methodology and documentation for conceptual search"}
                )
                logger.info("🔄 Methodology collection rebuilt")
            except:
                pass
    
    def _build_variables_database(self):
        """Build variables database using concept-based structure"""
        # Look for refactored canonical variables first, fallback to original
        canonical_path = self.source_dir / "canonical_variables_refactored.json"
        if not canonical_path.exists():
            canonical_path = self.source_dir / "canonical_variables.json"
            logger.warning("⚠️  Using original canonical_variables.json - refactored version not found")
            logger.warning("⚠️  Consider running refactor script first for optimal performance")
        
        if not canonical_path.exists():
            logger.error("❌ No canonical variables file found - cannot build variables database")
            return
        
        logger.info(f"📁 Using canonical variables: {canonical_path.name}")
        
        if self.use_faiss:
            self._build_variables_faiss(canonical_path)
        else:
            self._build_variables_chromadb(canonical_path)
    
    def _is_refactored_structure(self, data: dict) -> bool:
        """Detect if this is the refactored concept-based structure"""
        # Check for refactored structure indicators
        if 'metadata' in data and 'concepts' in data:
            return True
        
        # Check if any top-level entries have 'instances' array
        for key, value in data.items():
            if isinstance(value, dict) and 'instances' in value:
                return True
        
        return False
    
    def _load_canonical_variables(self, canonical_path: Path) -> tuple[dict, bool]:
        """Load canonical variables and detect structure type"""
        with open(canonical_path) as f:
            data = json.load(f)
        
        is_refactored = self._is_refactored_structure(data)
        
        if is_refactored:
            logger.info("🎯 Detected CONCEPT-BASED structure (refactored)")
            concepts = data.get('concepts', {})
            if not concepts:
                # Handle case where concepts are at root level
                concepts = {k: v for k, v in data.items() if k != 'metadata' and isinstance(v, dict)}
        else:
            logger.info("📦 Detected TEMPORAL structure (original)")
            concepts = data.get('variables', data)
        
        return concepts, is_refactored
    
    def _create_concept_embedding_text(self, variable_id: str, concept_data: dict, is_refactored: bool) -> tuple[str, dict]:
        """Create optimized embedding text and metadata for a concept - SUMMARY-FIRST VERSION
        
        Prioritizes search-optimized summaries while preserving all Census precision and semantic intelligence.
        Order: Summary → Key Terms → Administrative Precision → Full Enrichment → Domain Context
        """
        
        if is_refactored:
            # OPTIMIZED ORDERING FOR SEARCH RELEVANCE
            parts = []
            
            # 1. SUMMARY FIRST (search-optimized signal gets highest embedding weight)
            summary = concept_data.get('summary', '')
            if summary:
                parts.append(summary)
                logger.debug(f"Added summary: {len(summary)} chars for {variable_id}")
            
            # 2. KEY TERMS (only if NOT already in summary to avoid repetition)
            key_terms = concept_data.get('key_terms', [])
            if key_terms and summary:
                # Filter out terms already in summary to avoid dilution
                summary_lower = summary.lower()
                unique_terms = [term for term in key_terms if term.lower() not in summary_lower]
                if unique_terms:
                    parts.append(f"Key search terms: {', '.join(unique_terms)}")
                    logger.debug(f"Added {len(unique_terms)} unique key terms for {variable_id}")
            elif key_terms and not summary:
                # If no summary, include all key terms
                parts.append(f"Key search terms: {', '.join(key_terms)}")
                logger.debug(f"Added {len(key_terms)} key terms for {variable_id}")
            
            # 3. CENSUS ADMINISTRATIVE PRECISION (essential technical vocabulary)
            parts.append(f"Census variable identifier: {variable_id}")
            
            concept = concept_data.get('concept', 'Unknown')
            label = concept_data.get('label', 'Unknown')
            
            if concept != 'Unknown':
                parts.append(f"Official Census concept: {concept}")
            if label != 'Unknown':
                parts.append(f"Official Census label: {label}")
            
            # 4. FULL ENRICHMENT TEXT (preserve your $160 semantic intelligence)
            enrichment = concept_data.get('enrichment_text', '')
            if enrichment:
                parts.append(enrichment)  # Keep ALL the expensive analysis
                logger.debug(f"Added full enrichment text: {len(enrichment)} chars for {variable_id}")
            
            # 5. CATEGORY WEIGHTS (domain expertise context)
            weights = concept_data.get('category_weights_linear', {})
            if weights:
                # Keep your existing nuanced threshold
                weight_strs = [f"{k}: {v:.2f}" for k, v in weights.items() if v > 0.05]
                if weight_strs:
                    parts.append(f"Domain expertise: {', '.join(weight_strs)}")
                    logger.debug(f"Added {len(weight_strs)} category weights for {variable_id}")
            
            # 6. SURVEY METHODOLOGY INTELLIGENCE (controlled to avoid verbosity)
            instances = concept_data.get('instances', [])
            if instances:
                # Add survey type summary first (most useful)
                survey_types = list(set(inst.get('survey_type', '') for inst in instances))
                if survey_types:
                    parts.append(f"Available surveys: {', '.join(filter(None, survey_types))}")
                
                # Add unique datasets and sample characteristics (avoid repetition)
                datasets = list(set(inst.get('dataset', '') for inst in instances if inst.get('dataset')))
                if datasets:
                    parts.append(f"Survey datasets: {', '.join(datasets)}")
                
                # Add unique sample characteristics (condensed)
                sample_chars = list(set(inst.get('sample_characteristics', '') for inst in instances if inst.get('sample_characteristics')))
                if sample_chars:
                    # Take only the first unique characteristic to avoid verbosity
                    parts.append(f"Survey methodology: {sample_chars[0]}")
                    
                logger.debug(f"Added survey context for {len(instances)} instances")
            
            # Create comprehensive metadata with rich context (preserved from original)
            metadata = {
                'variable_id': variable_id,
                'concept': concept,
                'label': label,
                'source_file': 'canonical_variables_refactored.json',
                'category': 'canonical_variables',
                'structure_type': 'concept_based',
                'available_surveys': concept_data.get('available_surveys', []),
                'geography_coverage': str(concept_data.get('geography_coverage', {})),
                'primary_instance': concept_data.get('primary_instance', ''),
                'instance_count': len(instances),
                'enrichment_length': len(enrichment),
                'category_count': len(weights),
                'has_full_enrichment': len(enrichment) > 1000,
                'has_summary': bool(summary),
                'summary_length': len(summary) if summary else 0,
                'key_terms_count': len(key_terms) if key_terms else 0
            }
            
            # Track survey instance processing (preserved from original)
            self.variables_stats['survey_instances_processed'] += len(instances)
            
        else:
            # Original temporal structure - apply same principles
            temporal_id = variable_id
            parts = []
            
            # 1. Summary first (if available)
            summary = concept_data.get('summary', '')
            if summary:
                parts.append(summary)
            
            # 2. Key terms
            key_terms = concept_data.get('key_terms', [])
            if key_terms:
                if summary:
                    summary_lower = summary.lower()
                    unique_terms = [term for term in key_terms if term.lower() not in summary_lower]
                    if unique_terms:
                        parts.append(f"Key search terms: {', '.join(unique_terms)}")
                else:
                    parts.append(f"Key search terms: {', '.join(key_terms)}")
            
            # 3. Administrative identifiers
            parts.append(f"Variable {temporal_id}")
            
            concept = concept_data.get('concept', 'Unknown')
            label = concept_data.get('label', 'Unknown')
            
            if concept != 'Unknown':
                parts.append(f"Concept: {concept}")
            if label != 'Unknown':
                parts.append(f"Label: {label}")
            
            # 4. Full enrichment (no truncation)
            enrichment = concept_data.get('enrichment_text', '')
            if enrichment:
                parts.append(enrichment)
            
            # 5. Category weights
            weights = concept_data.get('category_weights_linear', {})
            if weights:
                weight_strs = [f"{k}: {v:.2f}" for k, v in weights.items() if v > 0.05]
                if weight_strs:
                    parts.append(f"Categories: {', '.join(weight_strs)}")
            
            # 6. Survey context
            if concept_data.get('survey_context'):
                parts.append(f"Survey: {concept_data['survey_context']}")
            
            metadata = {
                'temporal_id': temporal_id,
                'variable_id': concept_data.get('variable_id', ''),
                'concept': concept,
                'label': label,
                'source_file': 'canonical_variables.json',
                'category': 'canonical_variables',
                'structure_type': 'temporal_based',
                'has_summary': bool(summary),
                'has_enrichment': bool(enrichment)
            }
        
        # Join all parts into rich embedding text
        text = ". ".join(parts) + "."
        
        # Enhanced logging
        if is_refactored:
            logger.debug(f"✅ Summary-first embedding created for {variable_id}: "
                       f"Summary: {len(summary) if summary else 0} chars, "
                       f"Enrichment: {len(enrichment) if enrichment else 0} chars, "
                       f"Categories: {len(weights)}, "
                       f"Survey instances: {len(instances)}")
        
        return text, metadata
    

    def _build_variables_faiss(self, canonical_path: Path):
        """Build concept-based variables database using FAISS index"""
        logger.info("🎯 Processing canonical variables for FAISS database...")
        
        concepts, is_refactored = self._load_canonical_variables(canonical_path)
        logger.info(f"📊 Found {len(concepts)} {'concepts' if is_refactored else 'temporal variables'}")
        
        # Process variables in batches for memory efficiency
        concept_items = list(concepts.items())
        if self.test_mode:
            concept_items = concept_items[:1000]
            logger.info(f"🧪 Test mode: Limited to {len(concept_items)} variables")
        
        all_texts = []
        all_metadata = []
        all_embeddings = []
        all_variable_ids = []  # 🔧 FIX: Track variable IDs for mapping
        
        batch_size = 1000
        total_batches = (len(concept_items) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(concept_items), batch_size)):
            batch = concept_items[i:i + batch_size]
            logger.info(f"🔄 Processing FAISS batch {batch_num + 1}/{total_batches}")
            
            batch_texts = []
            batch_metadata = []
            batch_variable_ids = []  # 🔧 FIX: Track IDs for this batch
            
            for variable_id, concept_data in batch:
                # Create optimized embedding text and metadata
                text, metadata = self._create_concept_embedding_text(variable_id, concept_data, is_refactored)
                
                batch_texts.append(text)
                batch_metadata.append(metadata)
                batch_variable_ids.append(variable_id)  # 🔧 FIX: Store variable ID
            
            # Generate embeddings for batch
            logger.info(f"🧠 Generating embeddings for {len(batch_texts)} variables...")
            embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
            
            all_texts.extend(batch_texts)
            all_metadata.extend(batch_metadata)
            all_embeddings.extend(embeddings)
            all_variable_ids.extend(batch_variable_ids)  # 🔧 FIX: Add to main list
            
            self.variables_stats['concepts_processed'] += len(batch_texts)
        
        # Build FAISS index
        logger.info(f"🔧 Building FAISS index for {len(all_embeddings)} variables...")
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Create FAISS index (L2 distance, good for semantic similarity)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings_array)
        
        # Save FAISS index
        faiss_path = self.variables_dir / "variables.faiss"
        faiss.write_index(index, str(faiss_path))
        logger.info(f"💾 FAISS index saved: {faiss_path}")
        
        # Save metadata separately
        metadata_path = self.variables_dir / "variables_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        logger.info(f"💾 Metadata saved: {metadata_path}")
        
        # 🔧 FIX: Save variable ID mapping (THE MISSING PIECE!)
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
            'model_name': self.model_name,
            'embedding_dimension': dimension,
            'variable_count': len(all_embeddings),
            'structure_type': 'concept_based' if is_refactored else 'temporal_based',
            'source_file': canonical_path.name,
            'build_timestamp': time.time(),
            'index_type': 'faiss_flat_l2',
            'survey_instances_processed': self.variables_stats['survey_instances_processed'],
            'has_id_mapping': True  # 🔧 FIX: Mark that ID mapping exists
        }
        
        build_info_path = self.variables_dir / "build_info.json"
        with open(build_info_path, 'w') as f:
            json.dump(build_info, f, indent=2)
        logger.info(f"💾 Build info saved: {build_info_path}")
        
        structure_note = "concept-based variables" if is_refactored else "temporal variables"
        logger.info(f"✅ FAISS variables database complete: {len(all_embeddings)} {structure_note}")
        logger.info(f"✅ ID mapping created: {len(all_variable_ids)} variable IDs")
    
    def _build_variables_chromadb(self, canonical_path: Path):
        """Build concept-based variables database using ChromaDB"""
        logger.info("🎯 Processing canonical variables for ChromaDB database...")
        
        concepts, is_refactored = self._load_canonical_variables(canonical_path)
        logger.info(f"📊 Found {len(concepts)} {'concepts' if is_refactored else 'temporal variables'}")
        
        # Process in batches
        concept_items = list(concepts.items())
        batch_size = 200 if self.test_mode else 1000
        total_batches = (len(concept_items) + batch_size - 1) // batch_size
        
        if self.test_mode:
            concept_items = concept_items[:1000]
            logger.info(f"🧪 Test mode: Limited to {len(concept_items)} variables")
        
        for batch_num, i in enumerate(range(0, len(concept_items), batch_size)):
            batch = concept_items[i:i + batch_size]
            
            logger.info(f"🔄 Processing variables batch {batch_num + 1}/{total_batches}")
                
            chunks = []
            for variable_id, concept_data in batch:
                # Create optimized embedding text and metadata
                text, metadata = self._create_concept_embedding_text(variable_id, concept_data, is_refactored)
                
                chunk_id = f"var_{variable_id}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                
                chunks.append({
                    'id': chunk_id,
                    'text': text,
                    'metadata': metadata
                })
            
            # Generate embeddings and store
            if chunks:
                logger.info(f"🧠 Generating embeddings for {len(chunks)} variables...")
                texts = [c['text'] for c in chunks]
                embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
                
                # Store in batches of 500
                for j in range(0, len(chunks), 500):
                    batch_chunks = chunks[j:j + 500]
                    batch_texts = [c['text'] for c in batch_chunks]
                    batch_ids = [c['id'] for c in batch_chunks]
                    batch_meta = [c['metadata'] for c in batch_chunks]
                    batch_embeddings = embeddings[j:j + 500].tolist()
                    
                    self.variables_collection.add(
                        documents=batch_texts,
                        embeddings=batch_embeddings,
                        metadatas=batch_meta,
                        ids=batch_ids
                    )
                    
                    logger.info(f"💾 Variables: Stored batch {j//500 + 1}: {len(batch_chunks)} variables")
                
                self.variables_stats['concepts_processed'] += len(chunks)
        
        structure_note = "concept-based variables" if is_refactored else "temporal variables"
        logger.info(f"✅ Variables database complete: {self.variables_stats['concepts_processed']} {structure_note}")
    
    def _build_methodology_database(self):
        """Build methodology-only database from documentation files"""
        # Create temp directory for parallel processing
        temp_dir = self.methodology_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect methodology files (exclude canonical_variables files)
        logger.info("📁 Collecting methodology files...")
        all_files = []
        patterns = ['*.pdf', '*.html', '*.htm', '*.md', '*.txt', '*.Rmd', '*.xlsx']
        
        # Exclude canonical variables files and focus on documentation
        exclude_patterns = [
            'canonical_variables.json',
            'canonical_variables_refactored.json',
            'acs1_raw.json',
            'acs5_raw.json',
            'raw_data',
            'data_dumps'
        ]
        
        for pattern in patterns:
            pattern_files = list(self.source_dir.rglob(pattern))
            logger.info(f"   Found {len(pattern_files)} {pattern} files")
            
            for file_path in pattern_files:
                # Skip hidden files and excluded patterns
                if (any(part.startswith('.') for part in file_path.parts) or
                    any(exclude in str(file_path).lower() for exclude in exclude_patterns)):
                    continue
                
                if self.test_mode and file_path.stat().st_size > 10 * 1024 * 1024:
                    continue
                
                # Determine category from path
                category = file_path.parts[1] if len(file_path.parts) > 1 else 'general'
                all_files.append((file_path, category))
        
        if self.test_mode:
            all_files = all_files[:100]
            logger.info(f"🧪 Test mode: Limited to {len(all_files)} files")
        
        logger.info(f"🚀 Processing {len(all_files)} methodology files with {self.workers} workers")
        
        # Split files among workers
        chunk_size = max(1, len(all_files) // self.workers)
        file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
        
        # Process in parallel
        logger.info("🔄 Starting parallel workers for methodology...")
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(worker_process_files, chunk, i, str(self.source_dir),
                              str(temp_dir), self.model_name)
                for i, chunk in enumerate(file_chunks) if chunk
            ]
            
            completed = 0
            for future in futures:
                result = future.result()
                completed += 1
                logger.info(f"✅ Methodology Worker {completed} completed: {result}")
                self.methodology_stats['files_processed'] += result['files_processed']
                self.methodology_stats['chunks_created'] += result['chunks_created']
                self.methodology_stats['errors'] += result['errors']
        
        # Merge temp files into methodology database
        self._merge_methodology_temp_files(temp_dir)
        
        # Cleanup
        self._cleanup_temp_files(temp_dir)
    
    def _merge_methodology_temp_files(self, temp_dir):
        """Merge temporary files into methodology ChromaDB"""
        logger.info("🔄 Merging methodology temp files into ChromaDB...")
        
        temp_files = list(temp_dir.glob("worker_*.json"))
        logger.info(f"📁 Found {len(temp_files)} temp files to merge")
        
        total_merged = 0
        for i, temp_file in enumerate(temp_files):
            logger.info(f"📥 Merging {temp_file.name} ({i+1}/{len(temp_files)})...")
            
            with open(temp_file) as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            if not chunks:
                logger.info(f"   ⚠️  No chunks in {temp_file.name}")
                continue
            
            # Store in batches
            batches = (len(chunks) + 499) // 500
            for j in range(0, len(chunks), 500):
                batch = chunks[j:j + 500]
                batch_num = j // 500 + 1
                
                logger.info(f"   💾 Methodology: Storing batch {batch_num}/{batches}: {len(batch)} chunks")
                
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
            
            total_merged += len(chunks)
            logger.info(f"   ✅ Merged {len(chunks)} chunks from {temp_file.name}")
        
        logger.info(f"🎉 Methodology merge complete: {total_merged} total chunks merged")
    
    def _cleanup_temp_files(self, temp_dir):
        """Clean up temporary files"""
        for temp_file in temp_dir.glob("worker_*.json"):
            temp_file.unlink()
        
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    
    def _display_final_stats(self, build_time):
        """Display comprehensive build statistics"""
        logger.info("🎉 CONCEPT-BASED DUAL-PATH BUILD COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Build time: {build_time:.2f}s")
        logger.info(f"Build mode: {self.build_mode}")
        
        if self.build_mode in ['variables', 'both']:
            backend_type = "FAISS" if self.use_faiss else "ChromaDB"
            logger.info(f"\n🎯 CONCEPT-BASED VARIABLES DATABASE ({self.variables_dir}) - {backend_type}:")
            logger.info(f"   Concepts processed: {self.variables_stats['concepts_processed']:,}")
            logger.info(f"   Survey instances processed: {self.variables_stats['survey_instances_processed']:,}")
            
            if self.use_faiss:
                faiss_path = self.variables_dir / "variables.faiss"
                metadata_path = self.variables_dir / "variables_metadata.json"
                build_info_path = self.variables_dir / "build_info.json"
                
                logger.info(f"   FAISS index: {faiss_path}")
                logger.info(f"   Metadata file: {metadata_path}")
                logger.info(f"   Build info: {build_info_path}")
                
                if faiss_path.exists():
                    size_mb = faiss_path.stat().st_size / 1024 / 1024
                    logger.info(f"   Index size: {size_mb:.1f} MB")
                
                # Show structure type from build info
                if build_info_path.exists():
                    with open(build_info_path) as f:
                        build_info = json.load(f)
                    structure_type = build_info.get('structure_type', 'unknown')
                    logger.info(f"   Structure type: {structure_type}")
            else:
                if self.variables_collection:
                    total_docs = self.variables_collection.count()
                    logger.info(f"   Total documents: {total_docs:,}")
        
        if self.build_mode in ['methodology', 'both']:
            logger.info(f"\n📚 METHODOLOGY DATABASE ({self.methodology_dir}) - ChromaDB:")
            logger.info(f"   Files processed: {self.methodology_stats['files_processed']:,}")
            logger.info(f"   Chunks created: {self.methodology_stats['chunks_created']:,}")
            logger.info(f"   Errors: {self.methodology_stats['errors']}")
            if self.methodology_collection:
                total_docs = self.methodology_collection.count()
                logger.info(f"   Total documents: {total_docs:,}")
        
        logger.info(f"\n🎯 KEY IMPROVEMENTS:")
        logger.info(f"   ✅ Concept-based structure (eliminates duplicates)")
        logger.info(f"   ✅ Survey instance awareness (ACS1/5yr metadata)")
        logger.info(f"   ✅ Rich metadata preservation for intelligent search")
        logger.info(f"   ✅ Automatic structure detection and handling")
        
        logger.info(f"\n💡 NEXT STEPS:")
        if self.build_mode in ['variables', 'both']:
            if self.use_faiss:
                logger.info(f"   Variables: FAISS index for lightning-fast concept lookup")
            else:
                logger.info(f"   Variables: ChromaDB for concept lookup and GraphRAG potential")
        if self.build_mode in ['methodology', 'both']:
            logger.info(f"   Methodology: ChromaDB optimized for conceptual search")
        logger.info(f"   Ready for MCP server integration with concept-based intelligence!")

def main():
    parser = argparse.ArgumentParser(description='Concept-Based Dual-Path Knowledge Base Builder')
    parser.add_argument('--variables-only', action='store_true', help='Build only variables database')
    parser.add_argument('--methodology-only', action='store_true', help='Build only methodology database')
    parser.add_argument('--both', action='store_true', help='Build both databases')
    parser.add_argument('--faiss', action='store_true', help='Use FAISS for variables database (faster loading)')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild existing databases')
    parser.add_argument('--test-mode', action='store_true', help='Test with subset of data')
    parser.add_argument('--source-dir', default='source-docs', help='Source directory')
    parser.add_argument('--variables-dir', default='variables-db', help='Variables database directory')
    parser.add_argument('--methodology-dir', default='methodology-db', help='Methodology database directory')
    parser.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2', help='Embedding model')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    
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
    
    builder = ConceptBasedKnowledgeBuilder(
        source_dir=Path(args.source_dir),
        build_mode=build_mode,
        variables_dir=Path(args.variables_dir),
        methodology_dir=Path(args.methodology_dir),
        test_mode=args.test_mode,
        model_name=args.model,
        workers=args.workers,
        use_faiss=args.faiss
    )
    
    builder.build_knowledge_bases(rebuild=args.rebuild)
    logger.info("🚀 Concept-based dual-path knowledge base build completed!")

if __name__ == "__main__":
    main()
