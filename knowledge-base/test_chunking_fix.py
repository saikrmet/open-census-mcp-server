#!/usr/bin/env python3
"""
Direct inline test of the fixed chunking logic
"""
import re
from pathlib import Path
from typing import List, Dict
import hashlib

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

def create_chunks(content: str, file_path: Path) -> List[Dict]:
    """Proven chunking logic with aggressive multi-level fallback"""
    chunks = []
    
    # Clean text
    content = re.sub(r'\s+', ' ', content).strip()
    
    if len(content) < 100:
        return chunks
    
    # Use proven settings from build-kb.py
    chunk_size = 800      # Target size
    max_chunk_size = 1200 # Hard limit to stay under token limits  
    overlap = 150         # Overlap for context preservation
    
    # Multi-level splitting: paragraphs → sentences → words
    paragraphs = content.split('\n\n')
    
    current_chunk = ""
    chunk_num = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If paragraph itself is too large, split by sentences
        if len(paragraph) > max_chunk_size:
            sentences = re.split(r'[.!?]+', paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # If sentence is still too large, split by words (aggressive fallback)
                if len(sentence) > max_chunk_size:
                    words = sentence.split()
                    current_sentence = ""
                    
                    for word in words:
                        if len(current_sentence + " " + word) > max_chunk_size and current_sentence:
                            # Store the current sentence chunk
                            if len(current_sentence.strip()) > 100:
                                chunks.append(create_chunk_metadata(
                                    current_sentence.strip(), file_path, chunk_num
                                ))
                                chunk_num += 1
                            
                            # Start new chunk with overlap
                            overlap_words = current_sentence.split()[-20:]  # Last 20 words
                            current_sentence = " ".join(overlap_words) + " " + word
                        else:
                            current_sentence += " " + word if current_sentence else word
                    
                    # Add final sentence chunk
                    if len(current_sentence.strip()) > 100:
                        if len(current_chunk + " " + current_sentence) <= max_chunk_size:
                            current_chunk += " " + current_sentence if current_chunk else current_sentence
                        else:
                            # Store current chunk first
                            if len(current_chunk.strip()) > 100:
                                chunks.append(create_chunk_metadata(
                                    current_chunk.strip(), file_path, chunk_num
                                ))
                                chunk_num += 1
                            current_chunk = current_sentence
                else:
                    # Normal sentence, try to add to current chunk
                    if len(current_chunk + " " + sentence) > chunk_size and current_chunk:
                        # Store current chunk
                        if len(current_chunk.strip()) > 100:
                            chunks.append(create_chunk_metadata(
                                current_chunk.strip(), file_path, chunk_num
                            ))
                            chunk_num += 1
                        
                        # Start new chunk with overlap
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
        else:
            # Normal paragraph processing
            if len(current_chunk + " " + paragraph) > chunk_size and current_chunk:
                # Store current chunk
                if len(current_chunk.strip()) > 100:
                    chunks.append(create_chunk_metadata(
                        current_chunk.strip(), file_path, chunk_num
                    ))
                    chunk_num += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + paragraph
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph
    
    # Add final chunk
    if len(current_chunk.strip()) > 100:
        # Ensure final chunk isn't too large
        if len(current_chunk) > max_chunk_size:
            # Split final chunk if needed
            words = current_chunk.split()
            while len(" ".join(words)) > max_chunk_size and len(words) > 10:
                chunk_words = words[:len(words)//2]
                chunks.append(create_chunk_metadata(
                    " ".join(chunk_words), file_path, chunk_num
                ))
                chunk_num += 1
                words = words[len(words)//2:]
            
            if words:
                chunks.append(create_chunk_metadata(
                    " ".join(words), file_path, chunk_num
                ))
        else:
            chunks.append(create_chunk_metadata(
                current_chunk.strip(), file_path, chunk_num
            ))
    
    return chunks

def test_chunking():
    """Test the chunking with problematic text sizes"""
    
    # Test 1: Massive single paragraph (common PDF issue)
    massive_text = "This is a sentence that appears in a PDF with no paragraph breaks. " * 300  # ~21,000 chars
    chunks = create_chunks(massive_text, Path("massive.pdf"))
    
    print(f"=== CHUNKING TEST RESULTS ===")
    print(f"Input: {len(massive_text):,} characters")
    print(f"Output: {len(chunks)} chunks")
    
    max_size = 0
    over_limit = []
    for i, chunk in enumerate(chunks):
        size = len(chunk['text'])
        max_size = max(max_size, size)
        if size > 1200:
            over_limit.append((i, size))
        print(f"  Chunk {i}: {size:,} chars")
    
    print(f"\nMax chunk size: {max_size:,} chars")
    print(f"Chunks over 1200 limit: {len(over_limit)}")
    if over_limit:
        for chunk_id, size in over_limit:
            print(f"  Chunk {chunk_id}: {size:,} chars (OVER LIMIT!)")
    
    # Test 2: No-break text (worst case)
    no_break_text = "verylongwordwithnobreaksatall" * 500  # ~15,000 chars, no spaces
    chunks2 = create_chunks(no_break_text, Path("nobreaks.txt"))
    
    print(f"\n=== NO-BREAKS TEST ===")
    print(f"Input: {len(no_break_text):,} characters (no spaces)")
    print(f"Output: {len(chunks2)} chunks")
    
    max_size2 = 0
    for i, chunk in enumerate(chunks2):
        size = len(chunk['text'])
        max_size2 = max(max_size2, size)
        print(f"  Chunk {i}: {size:,} chars")
    
    print(f"Max chunk size: {max_size2:,} chars")
    
    # Overall validation
    success = max(max_size, max_size2) <= 1200
    print(f"\n🎯 CHUNKING FIX: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"All chunks under 1200 chars: {success}")
    
    return success

if __name__ == "__main__":
    test_chunking()
