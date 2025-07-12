#!/usr/bin/env python3
"""
Knowledge Base Smoke Tests
Tests retrieval quality and semantic precision of the built vector database

Usage:
    python smoke_test.py
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KBSmokeTest:
    def __init__(self, vector_db_path="knowledge-base/vector-db"):
        """Initialize test with vector DB and embedding model"""
        self.vector_db_path = Path(vector_db_path)
        
        # Load same model used for building
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='./model_cache')
        
        # Connect to ChromaDB
        logger.info("Connecting to vector database...")
        self.client = chromadb.PersistentClient(
            path=str(self.vector_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection("census_knowledge")
        
        total_docs = self.collection.count()
        logger.info(f"Connected to KB with {total_docs:,} documents")
    
    def search(self, query, top_k=5):
        """Search the knowledge base"""
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Format results
        formatted = []
        for i in range(len(results['documents'][0])):
            similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
            formatted.append({
                'text': results['documents'][0][i][:200] + "..." if len(results['documents'][0][i]) > 200 else results['documents'][0][i],
                'similarity': similarity,
                'metadata': results['metadatas'][0][i],
                'source': results['metadatas'][0][i].get('source_file', 'Unknown')
            })
        
        return formatted
    
    def run_smoke_tests(self):
        """Run comprehensive smoke tests"""
        
        test_queries = [
            # Variable precision tests
            {
                'category': 'Variable Precision',
                'query': 'median household income',
                'expect': ['B19013', 'household', 'median', 'income'],
                'avoid': ['mean', 'per capita', 'family income']
            },
            {
                'category': 'Variable Precision',
                'query': 'poverty rate',
                'expect': ['poverty', 'rate', 'B17001'],
                'avoid': ['poverty threshold', 'income']
            },
            {
                'category': 'Variable Precision',
                'query': 'unemployment rate',
                'expect': ['unemployment', 'B23025', 'labor force'],
                'avoid': ['employment', 'labor participation']
            },
            
            # Geographic precision tests
            {
                'category': 'Geographic Concepts',
                'query': 'census tract boundaries',
                'expect': ['tract', 'geographic', 'boundary'],
                'avoid': ['block group', 'county']
            },
            {
                'category': 'Geographic Concepts',
                'query': 'ZCTA zip code areas',
                'expect': ['ZCTA', 'zip', 'tabulation'],
                'avoid': ['census tract', 'county']
            },
            
            # Methodology tests
            {
                'category': 'Statistical Methods',
                'query': 'margin of error calculation',
                'expect': ['margin', 'error', 'MOE', 'confidence'],
                'avoid': ['estimate', 'sample']
            },
            {
                'category': 'Statistical Methods',
                'query': 'ACS survey methodology',
                'expect': ['ACS', 'survey', 'methodology', 'sample'],
                'avoid': ['decennial', 'census']
            },
            
            # Technical documentation tests
            {
                'category': 'R/tidycensus',
                'query': 'get_acs function parameters',
                'expect': ['get_acs', 'function', 'parameters'],
                'avoid': ['get_decennial', 'load_variables']
            },
            {
                'category': 'R/tidycensus',
                'query': 'tidycensus installation',
                'expect': ['install', 'tidycensus', 'package'],
                'avoid': ['censusapi', 'acs']
            },
            
            # Edge cases and disambiguation
            {
                'category': 'Disambiguation',
                'query': 'Washington state vs DC',
                'expect': ['Washington', 'state', 'DC'],
                'avoid': []  # Just check if it returns relevant results
            },
            {
                'category': 'Complex Queries',
                'query': 'educational attainment by race and ethnicity',
                'expect': ['education', 'attainment', 'race', 'ethnicity'],
                'avoid': ['income', 'employment']
            }
        ]
        
        print("\n" + "="*80)
        print("CENSUS KNOWLEDGE BASE SMOKE TESTS")
        print("="*80)
        
        passed = 0
        failed = 0
        
        for test in test_queries:
            print(f"\n📊 {test['category']}: {test['query']}")
            print("-" * 60)
            
            try:
                results = self.search(test['query'], top_k=3)
                
                if not results:
                    print("❌ FAIL: No results returned")
                    failed += 1
                    continue
                
                # Check top result
                top_result = results[0]
                top_text = top_result['text'].lower()
                
                # Score the result
                expect_score = sum(1 for term in test['expect'] if term.lower() in top_text)
                avoid_score = sum(1 for term in test['avoid'] if term.lower() in top_text)
                
                print(f"🎯 Top result ({top_result['similarity']:.3f}): {top_result['source']}")
                print(f"📝 Text: {top_result['text']}")
                
                # Scoring
                if expect_score >= len(test['expect']) // 2 and avoid_score == 0:
                    print(f"✅ PASS: Found {expect_score}/{len(test['expect'])} expected terms, avoided unwanted terms")
                    passed += 1
                elif expect_score > 0:
                    print(f"⚠️  PARTIAL: Found {expect_score}/{len(test['expect'])} expected, {avoid_score} unwanted")
                    passed += 0.5
                    failed += 0.5
                else:
                    print(f"❌ FAIL: Found {expect_score}/{len(test['expect'])} expected terms")
                    failed += 1
                    
                # Show other results
                for i, result in enumerate(results[1:], 2):
                    print(f"   {i}. ({result['similarity']:.3f}) {result['source']}")
                    
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                failed += 1
        
        # Summary
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*80)
        print("SMOKE TEST SUMMARY")
        print("="*80)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 KNOWLEDGE BASE QUALITY: EXCELLENT")
        elif success_rate >= 60:
            print("⚠️  KNOWLEDGE BASE QUALITY: GOOD - Some tuning needed")
        else:
            print("🚨 KNOWLEDGE BASE QUALITY: POOR - Needs investigation")
        
        return success_rate >= 70  # Return True if acceptable quality

def main():
    """Run smoke tests"""
    tester = KBSmokeTest()
    success = tester.run_smoke_tests()
    
    if success:
        print("\n🚀 Knowledge base ready for production!")
    else:
        print("\n🔧 Knowledge base needs tuning before production use.")

if __name__ == "__main__":
    main()
