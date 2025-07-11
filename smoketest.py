#!/usr/bin/env python3
"""
Quick smoke test for clean embeddings
"""
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load components
print("🔍 Loading clean index...")
index = faiss.read_index('knowledge-base/stats-index/variables_bge.faiss')
with open('knowledge-base/stats-index/variables_meta.json') as f:
    meta = json.load(f)

print(f"✅ Index: {index.ntotal} vectors, {index.d} dimensions")
print(f"✅ Metadata: {len(meta)} entries")

# Check metadata structure
sample = meta[0]
print(f"✅ Sample keys: {list(sample.keys())}")

# Check for weights
if 'weights' in sample:
    print(f"✅ Weights found: {list(sample['weights'].keys())}")
else:
    print("❌ No weights in metadata")

# Test semantic search
print("\n🔍 Testing semantic search...")
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

test_queries = [
    "median household income",
    "percent renter occupied", 
    "foreign born population"
]

for query in test_queries:
    # Embed query
    query_vec = model.encode([query])
    
    # Search
    scores, indices = index.search(query_vec, 3)
    
    print(f"\n📊 Query: '{query}'")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        var = meta[idx]
        print(f"  {i+1}. {var['variable_id']} - {var['label'][:50]}... (score: {score:.3f})")

print("\n🎯 Quick validation:")
print("- Does 'median household income' return B19013_001E?")
print("- Are scores reasonable (>0.3 for good matches)?")
print("- No more spam-contaminated results?")
