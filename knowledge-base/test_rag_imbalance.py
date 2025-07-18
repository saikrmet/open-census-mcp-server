#!/usr/bin/env python3
"""
Direct ChromaDB test to prove RAG imbalance hypothesis
"""

import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

def test_rag_imbalance():
    """Test methodology vs variable dominance in existing ChromaDB"""
    
    # Load the CORRECT embedding model that built the ChromaDB
    print("Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')  # 768-dim
    print(f"Model loaded: {embedding_model.get_sentence_embedding_dimension()}-dimensional embeddings")
    
    # Connect directly to ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(Path("vector-db")))
        collection = client.get_collection("census_knowledge")
        print(f"✅ Connected to collection with {collection.count():,} documents")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("Make sure you're running from knowledge-base/ directory")
        return
    
    # Test queries that should find methodology docs
    methodology_queries = [
        "ACS sampling methodology",
        "margin of error calculation",
        "survey design principles",
        "how does Census collect data",
        "statistical reliability",
        "data quality issues"
    ]
    
    print("\n🔍 Testing Methodology Queries")
    print("=" * 60)
    
    total_variable_hits = 0
    total_methodology_hits = 0
    
    for query in methodology_queries:
        print(f"\nQuery: '{query}'")
        
        try:
            # Generate query embedding with correct model
            query_embedding = embedding_model.encode([query])
            
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=5,
                include=['metadatas', 'distances']
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                print("  No results")
                continue
            
            variable_count = 0
            methodology_count = 0
            
            print("  Top 5 results:")
            for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                category = metadata.get('category', 'unknown')
                source = metadata.get('source_file', metadata.get('source', 'unknown'))
                
                # Classify result type
                if (category == 'canonical_variables' or
                    'canonical_variables' in source or
                    'Variable:' in metadata.get('title', '')):
                    variable_count += 1
                    result_type = "🔢 VARIABLE"
                else:
                    methodology_count += 1
                    result_type = "📄 METHODOLOGY"
                
                print(f"    {i+1}. {result_type} | {source[:50]}... | dist: {distance:.3f}")
            
            total_variable_hits += variable_count
            total_methodology_hits += methodology_count
            
            # Query assessment
            if variable_count > methodology_count:
                print("  ⚠️  VARIABLES DOMINATING")
            elif methodology_count > variable_count:
                print("  ✅ Methodology ranking well")
            else:
                print("  ➡️  Mixed results")
                
        except Exception as e:
            print(f"  ❌ Query failed: {e}")
    
    # Overall imbalance assessment
    print(f"\n📊 OVERALL RESULTS")
    print("=" * 40)
    print(f"Total variable hits in top-5: {total_variable_hits}")
    print(f"Total methodology hits in top-5: {total_methodology_hits}")
    
    if total_variable_hits > total_methodology_hits * 2:
        print("🚨 SEVERE IMBALANCE - Variables dominating methodology queries")
        print("   Recommendation: SEPARATE THE DATABASES")
    elif total_variable_hits > total_methodology_hits:
        print("⚠️  MODERATE IMBALANCE - Variables crowding methodology")
        print("   Recommendation: Consider separation or better ranking")
    else:
        print("✅ Reasonable balance maintained")
    
    # Collection composition analysis
    print(f"\n📈 Collection Composition")
    print("=" * 30)
    
    try:
        # Sample analysis (avoid loading entire collection)
        sample_size = min(5000, collection.count())
        sample = collection.get(limit=sample_size, include=['metadatas'])
        
        category_counts = {}
        for metadata in sample['metadatas']:
            category = metadata.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        total_sampled = len(sample['metadatas'])
        
        for category, count in sorted(category_counts.items()):
            percentage = (count / total_sampled) * 100
            print(f"  {category}: {count:,} docs ({percentage:.1f}%)")
        
        # Calculate variable dominance ratio
        var_count = category_counts.get('canonical_variables', 0)
        other_count = total_sampled - var_count
        
        if other_count > 0:
            ratio = var_count / other_count
            print(f"\n📊 Variables:Other ratio: {ratio:.1f}:1")
            
            if ratio > 10:
                print("🚨 SEVERE IMBALANCE (>10:1) - Separation critical")
            elif ratio > 5:
                print("⚠️  HIGH IMBALANCE (>5:1) - Separation recommended")
            else:
                print("✅ Manageable ratio - unified approach viable")
        
    except Exception as e:
        print(f"⚠️  Could not analyze composition: {e}")
    
    print(f"\n💡 VERDICT:")
    print("If variables dominated methodology queries above,")
    print("your hypothesis is CONFIRMED - separate the databases.")

if __name__ == "__main__":
    test_rag_imbalance()
