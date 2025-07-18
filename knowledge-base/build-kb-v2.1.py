#!/usr/bin/env python3
"""
Knowledge Base Vectorization Script - Parallel Processing Version
Processes source documents into ChromaDB vector database using sentence transformers with parallel workers

Enhanced with:
- Excel, JSON, CSV support
- Canonical variables as individual documents  
- Parallel processing with temp files
- Clean recursive chunking (paragraphs → size limits)
- Semantic precision via all-mpnet-base-v2 (768-dim, local cache)

Usage:
    python build-kb.py --model sentence-transformers/all-mpnet-base-v2 --workers 5 --rebuild
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
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
    """Create metadata for a text chunk - fixed missing function"""
    
    # Generate globally unique ID using full file path + content
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:6]  # Include path in ID
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
                    from bs4 import BeautifulSoup
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
                    import markdown
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

class KnowledgeBaseBuilder:
    def __init__(self, source_dir: Path, output_dir: Path, test_mode: bool = False,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2", workers: int = 5):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.test_mode = test_mode
        self.model_name = model_name
        self.workers = workers
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {'files_processed': 0, 'chunks_created': 0, 'canonical_variables': 0, 'errors': 0}
        
        # Initialize model and ChromaDB - force local cache
        os.environ['HF_HUB_OFFLINE'] = '1'  # Force offline mode after download
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid worker conflicts
        
        # Download and cache model locally
        logger.info(f"🔄 Loading model: {model_name} (768 dimensions)")
        self.embedding_model = SentenceTransformer(model_name, cache_folder='./model_cache')
        logger.info(f"✅ Model cached locally in ./model_cache")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.output_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        try:
            self.collection = self.chroma_client.get_collection("census_knowledge")
        except:
            self.collection = self.chroma_client.create_collection("census_knowledge")
    
    def build_knowledge_base(self, rebuild: bool = False):
        logger.info("🚀 Building knowledge base with clean recursive chunking")
        start_time = time.time()
        
        if rebuild:
            try:
                self.chroma_client.delete_collection("census_knowledge")
                self.collection = self.chroma_client.create_collection("census_knowledge")
                logger.info("Rebuilt collection")
            except:
                pass
        
        # Process canonical variables first
        self._process_canonical_variables()
        
        # Process other files in parallel
        self._process_files_parallel()
        
        # Merge temp files
        self._merge_temp_files()
        
        # Cleanup
        self._cleanup_temp_files()
        
        build_time = time.time() - start_time
        logger.info(f"✅ Build complete in {build_time:.2f}s - {self.stats}")
    
    def _process_canonical_variables(self):
        canonical_path = self.source_dir / "canonical_variables.json"
        if not canonical_path.exists():
            logger.info("⚠️  No canonical_variables.json found - skipping")
            return
        
        logger.info("🎯 Processing canonical variables...")
        
        with open(canonical_path) as f:
            data = json.load(f)
        
        variables = data.get('variables', data)
        logger.info(f"📊 Found {len(variables)} canonical variables")
        
        # Process in batches
        var_items = list(variables.items())
        batch_size = 200 if self.test_mode else 1000
        total_batches = (len(var_items) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(var_items), batch_size)):
            batch = var_items[i:i + batch_size]
            if self.test_mode and i > 500:
                break
            
            logger.info(f"🔄 Processing canonical variables batch {batch_num + 1}/{total_batches}")
                
            chunks = []
            for temporal_id, var_data in batch:
                # Create variable text
                parts = [f"Variable {temporal_id}"]
                
                label = var_data.get('label', 'Unknown')
                concept = var_data.get('concept', 'Unknown')
                
                if label != 'Unknown':
                    parts.append(f"Label: {label}")
                if concept != 'Unknown':
                    parts.append(f"Concept: {concept}")
                
                # Add context
                if var_data.get('survey_context'):
                    parts.append(f"Survey: {var_data['survey_context']}")
                
                # Add weights
                weights = var_data.get('category_weights_linear', {})
                if weights:
                    weight_strs = [f"{k}: {v:.2f}" for k, v in weights.items() if v > 0.1]
                    if weight_strs:
                        parts.append(f"Weights: {', '.join(weight_strs)}")
                
                text = ". ".join(parts) + "."
                chunk_id = f"var_{temporal_id}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                
                chunks.append({
                    'id': chunk_id,
                    'text': text,
                    'metadata': {
                        'source_file': 'canonical_variables.json',
                        'category': 'canonical_variables',
                        'temporal_id': temporal_id,
                        'variable_id': var_data.get('variable_id', ''),
                        'file_type': 'canonical_variable'
                    }
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
                    
                    self.collection.add(
                        documents=batch_texts,
                        embeddings=batch_embeddings,
                        metadatas=batch_meta,
                        ids=batch_ids
                    )
                    
                    logger.info(f"💾 Stored batch {j//500 + 1}: {len(batch_chunks)} variables")
                
                self.stats['canonical_variables'] += len(chunks)
        
        logger.info(f"✅ Processed {self.stats['canonical_variables']} canonical variables")
    
    def _process_files_parallel(self):
        # Collect all files
        logger.info("📁 Collecting files to process...")
        all_files = []
        patterns = ['*.pdf', '*.html', '*.htm', '*.md', '*.txt', '*.Rmd', '*.xlsx', '*.json', '*.csv']
        
        for pattern in patterns:
            pattern_files = list(self.source_dir.rglob(pattern))
            logger.info(f"   Found {len(pattern_files)} {pattern} files")
            
            for file_path in pattern_files:
                if (not any(part.startswith('.') for part in file_path.parts) and
                    file_path.name != 'canonical_variables.json'):
                    
                    if self.test_mode and file_path.stat().st_size > 10 * 1024 * 1024:
                        continue
                        
                    # Determine category from path
                    category = file_path.parts[1] if len(file_path.parts) > 1 else 'general'
                    all_files.append((file_path, category))
        
        if self.test_mode:
            all_files = all_files[:200]
            logger.info(f"🧪 Test mode: Limited to {len(all_files)} files")
        
        logger.info(f"🚀 Processing {len(all_files)} files with {self.workers} workers")
        
        # Split files among workers
        chunk_size = max(1, len(all_files) // self.workers)
        file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
        
        logger.info(f"📊 File distribution:")
        for i, chunk in enumerate(file_chunks):
            logger.info(f"   Worker {i}: {len(chunk)} files")
        
        # Process in parallel
        logger.info("🔄 Starting parallel workers...")
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(worker_process_files, chunk, i, str(self.source_dir),
                              str(self.temp_dir), self.model_name)
                for i, chunk in enumerate(file_chunks) if chunk
            ]
            
            completed = 0
            for future in futures:
                result = future.result()
                completed += 1
                logger.info(f"✅ Worker {completed} completed: {result}")
                self.stats['files_processed'] += result['files_processed']
                self.stats['chunks_created'] += result['chunks_created']
                self.stats['errors'] += result['errors']
    
    def _merge_temp_files(self):
        logger.info("🔄 Merging temporary files into ChromaDB...")
        
        temp_files = list(self.temp_dir.glob("worker_*.json"))
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
                
                logger.info(f"   💾 Storing batch {batch_num}/{batches}: {len(batch)} chunks")
                
                texts = [c['text'] for c in batch]
                ids = [c['id'] for c in batch]
                metadatas = [c['metadata'] for c in batch]
                embeddings = [c['embedding'] for c in batch]
                
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            
            total_merged += len(chunks)
            logger.info(f"   ✅ Merged {len(chunks)} chunks from {temp_file.name}")
        
        logger.info(f"🎉 Merge complete: {total_merged} total chunks merged")
    
    def _cleanup_temp_files(self):
        for temp_file in self.temp_dir.glob("worker_*.json"):
            temp_file.unlink()
        
        if not any(self.temp_dir.iterdir()):
            self.temp_dir.rmdir()

def main():
    parser = argparse.ArgumentParser(description='Build Knowledge Base with Clean Recursive Chunking')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild')
    parser.add_argument('--test-mode', action='store_true', help='Test with subset')
    parser.add_argument('--source-dir', default='source-docs', help='Source directory')
    parser.add_argument('--output-dir', default='vector-db', help='Output directory')
    parser.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2', help='Model name')
    parser.add_argument('--workers', type=int, default=5, help='Number of workers')
    
    args = parser.parse_args()
    
    builder = KnowledgeBaseBuilder(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        test_mode=args.test_mode,
        model_name=args.model,
        workers=args.workers
    )
    
    builder.build_knowledge_base(rebuild=args.rebuild)
    logger.info("✅ Knowledge base build completed!")

if __name__ == "__main__":
    main()
