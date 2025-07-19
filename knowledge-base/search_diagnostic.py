#!/usr/bin/env python3
"""
Search System Diagnostic Script

Inspects exactly what's happening in your search system:
1. What text is being embedded for variables
2. What text is being embedded for queries  
3. What the top FAISS matches actually are
4. Whether keywords are being used properly
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchDiagnostic:
    """Diagnostic tool for search system debugging"""
    
    def __init__(self):
        self.canonical_vars = {}
        self.variable_metadata = {}
        self.table_metadata = {}
        self.embedding_model = None
        self.variable_index = None
        self.table_index = None
        self.variable_ids = []
        self.table_ids = []
        
        self._load_data()
    
    def _load_data(self):
        """Load all search system data"""
        print("🔍 LOADING SEARCH SYSTEM DATA")
        print("=" * 50)
        
        # 1. Load canonical variables
        canonical_paths = [
            "source-docs/canonical_variables_refactored.json",
            "../source-docs/canonical_variables_refactored.json",
            "canonical_variables_refactored.json"
        ]
        
        for path in canonical_paths:
            if Path(path).exists():
                print(f"📁 Loading canonical variables: {path}")
                with open(path) as f:
                    data = json.load(f)
                
                if 'concepts' in data:
                    self.canonical_vars = data['concepts']
                else:
                    self.canonical_vars = {k: v for k, v in data.items() if k != 'metadata'}
                
                print(f"   Loaded {len(self.canonical_vars)} canonical variables")
                break
        else:
            print("❌ No canonical variables file found!")
        
        # 2. Load variable metadata
        metadata_path = Path("variables-db/variables_metadata.json")
        if metadata_path.exists():
            print(f"📁 Loading variable metadata: {metadata_path}")
            with open(metadata_path) as f:
                self.variable_metadata = json.load(f)
            print(f"   Loaded metadata for {len(self.variable_metadata)} variables")
        else:
            print("⚠️  Variable metadata not found")
        
        # 3. Load table catalog with keywords
        table_paths = [
            "table-catalog/table_catalog_with_keywords.json",
            "table-catalog/table_catalog.json"
        ]
        
        for path in table_paths:
            if Path(path).exists():
                print(f"📁 Loading table catalog: {path}")
                with open(path) as f:
                    table_data = json.load(f)
                
                self.table_metadata = {table['table_id']: table for table in table_data.get('tables', [])}
                
                # Check for keywords
                keywords_count = sum(1 for table in self.table_metadata.values() if 'search_keywords' in table)
                print(f"   Loaded {len(self.table_metadata)} tables")
                print(f"   Tables with keywords: {keywords_count}")
                break
        else:
            print("❌ No table catalog found!")
        
        # 4. Load FAISS indices
        variable_faiss = Path("variables-db/variables.faiss")
        if variable_faiss.exists():
            print(f"📁 Loading variable FAISS index: {variable_faiss}")
            self.variable_index = faiss.read_index(str(variable_faiss))
            print(f"   Variable index size: {self.variable_index.ntotal}")
        else:
            print("⚠️  Variable FAISS index not found")
        
        table_faiss = Path("table-catalog/table_embeddings.faiss")
        if table_faiss.exists():
            print(f"📁 Loading table FAISS index: {table_faiss}")
            self.table_index = faiss.read_index(str(table_faiss))
            print(f"   Table index size: {self.table_index.ntotal}")
        else:
            print("⚠️  Table FAISS index not found")
        
        # 5. Load ID mappings
        variable_mapping = Path("variables-db/variables_ids.json")
        if variable_mapping.exists():
            with open(variable_mapping) as f:
                mapping_data = json.load(f)
            self.variable_ids = mapping_data.get('variable_ids', [])
            print(f"   Variable ID mapping: {len(self.variable_ids)} IDs")
        
        table_mapping = Path("table-catalog/table_mapping.json")
        if table_mapping.exists():
            with open(table_mapping) as f:
                mapping_data = json.load(f)
            self.table_ids = mapping_data.get('table_ids', [])
            print(f"   Table ID mapping: {len(self.table_ids)} IDs")
        
        # 6. Load embedding model
        print("🤖 Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("   Model loaded successfully")
    
    def inspect_variable_embedding_text(self, variable_id: str):
        """Show exactly what text is being embedded for a variable"""
        print(f"\n🔍 INSPECTING VARIABLE EMBEDDING TEXT: {variable_id}")
        print("-" * 60)
        
        # Check canonical variables
        if variable_id in self.canonical_vars:
            var_data = self.canonical_vars[variable_id]
            print("✅ Found in canonical variables")
            
            # Show key fields that might be embedded
            fields_to_check = ['label', 'concept', 'summary', 'key_terms', 'enrichment_text']
            
            for field in fields_to_check:
                value = var_data.get(field, '')
                if value:
                    if isinstance(value, list):
                        value = ', '.join(value)
                    print(f"   {field}: {value[:150]}{'...' if len(str(value)) > 150 else ''}")
                else:
                    print(f"   {field}: [EMPTY]")
            
            # Try to reconstruct likely embedding text
            print(f"\n📝 LIKELY EMBEDDING TEXT:")
            text_parts = []
            
            # This should match your actual embedding construction logic
            if var_data.get('label'):
                text_parts.append(f"Variable {variable_id}: {var_data['label']}")
            if var_data.get('concept'):
                text_parts.append(f"Concept: {var_data['concept']}")
            if var_data.get('summary'):
                text_parts.append(f"Summary: {var_data['summary']}")
            if var_data.get('key_terms'):
                key_terms = var_data['key_terms']
                if isinstance(key_terms, list):
                    text_parts.append(f"Key terms: {', '.join(key_terms)}")
            
            embedding_text = " | ".join(text_parts)
            print(f"   {embedding_text}")
            print(f"   Length: {len(embedding_text)} characters")
            
        else:
            print("❌ NOT FOUND in canonical variables")
        
        # Check variable metadata
        if variable_id in self.variable_metadata:
            meta = self.variable_metadata[variable_id]
            print(f"\n✅ Found in variable metadata")
            print(f"   Structure type: {meta.get('structure_type', 'unknown')}")
            if 'summary_length' in meta:
                print(f"   Summary length: {meta['summary_length']} chars")
            if 'key_terms_count' in meta:
                print(f"   Key terms count: {meta['key_terms_count']}")
        else:
            print("❌ NOT FOUND in variable metadata")
    
    def inspect_table_embedding_text(self, table_id: str):
        """Show exactly what text is being embedded for a table"""
        print(f"\n🔍 INSPECTING TABLE EMBEDDING TEXT: {table_id}")
        print("-" * 60)
        
        if table_id in self.table_metadata:
            table_data = self.table_metadata[table_id]
            print("✅ Found in table catalog")
            
            # Show basic metadata
            print(f"   Title: {table_data.get('title', 'N/A')}")
            print(f"   Universe: {table_data.get('universe', 'N/A')}")
            print(f"   Concept: {table_data.get('concept', 'N/A')}")
            
            # Show keywords if available
            if 'search_keywords' in table_data:
                keywords = table_data['search_keywords']
                print(f"\n🔑 SEARCH KEYWORDS:")
                
                primary = keywords.get('primary_keywords', [])
                secondary = keywords.get('secondary_keywords', [])
                summary = keywords.get('summary', '')
                
                print(f"   Primary: {', '.join(primary)}")
                print(f"   Secondary: {', '.join(secondary)}")
                print(f"   Summary: {summary}")
                
                # Reconstruct likely embedding text (WITH keywords)
                print(f"\n📝 LIKELY EMBEDDING TEXT (with keywords):")
                text_parts = []
                
                if primary:
                    text_parts.append(f"Primary search terms: {', '.join(primary)}")
                if summary:
                    text_parts.append(f"Summary: {summary}")
                text_parts.extend([
                    f"Title: {table_data['title']}",
                    f"Universe: {table_data['universe']}",
                    f"Concept: {table_data['concept']}"
                ])
                if secondary:
                    text_parts.append(f"Related terms: {', '.join(secondary)}")
                
                embedding_text = '. '.join(text_parts)
                print(f"   {embedding_text}")
                print(f"   Length: {len(embedding_text)} characters")
            else:
                print(f"\n⚠️  NO SEARCH KEYWORDS - using basic metadata only")
                # Basic embedding text without keywords
                text_parts = [
                    table_data['title'],
                    table_data['universe'], 
                    table_data['concept']
                ]
                embedding_text = '. '.join(text_parts)
                print(f"📝 BASIC EMBEDDING TEXT:")
                print(f"   {embedding_text}")
        else:
            print("❌ NOT FOUND in table catalog")
    
    def debug_query_search(self, query: str, search_type: str = "both"):
        """Debug a specific query and show top matches"""
        print(f"\n🔍 DEBUGGING QUERY: '{query}'")
        print("=" * 60)
        
        # Create query embedding
        print(f"📝 Query embedding input: '{query}'")
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        print(f"   Embedding shape: {query_embedding.shape}")
        
        if search_type in ["both", "tables"] and self.table_index:
            print(f"\n📊 TABLE SEARCH RESULTS:")
            print("-" * 30)
            
            distances, indices = self.table_index.search(query_embedding, 10)
            
            for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                
                table_id = self.table_ids[idx] if idx < len(self.table_ids) else f"UNKNOWN_IDX_{idx}"
                confidence = max(0.0, 1.0 - (distance / 2.0))
                
                print(f"{rank+1}. {table_id} (distance: {distance:.3f}, confidence: {confidence:.3f})")
                
                if table_id in self.table_metadata:
                    table = self.table_metadata[table_id]
                    print(f"   Title: {table.get('title', 'N/A')[:80]}...")
                    
                    if 'search_keywords' in table:
                        kw = table['search_keywords']
                        if kw.get('primary_keywords'):
                            print(f"   Primary keywords: {', '.join(kw['primary_keywords'])}")
                        if kw.get('summary'):
                            print(f"   Summary: {kw['summary'][:100]}...")
                    else:
                        print(f"   ⚠️  No keywords available")
                else:
                    print(f"   ❌ Table metadata not found")
                print()
        
        if search_type in ["both", "variables"] and self.variable_index:
            print(f"\n📊 VARIABLE SEARCH RESULTS:")
            print("-" * 30)
            
            distances, indices = self.variable_index.search(query_embedding, 10)
            
            for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                
                variable_id = self.variable_ids[idx] if idx < len(self.variable_ids) else f"UNKNOWN_IDX_{idx}"
                confidence = max(0.0, 1.0 - (distance / 2.0))
                
                print(f"{rank+1}. {variable_id} (distance: {distance:.3f}, confidence: {confidence:.3f})")
                
                if variable_id in self.canonical_vars:
                    var = self.canonical_vars[variable_id]
                    print(f"   Label: {var.get('label', 'N/A')[:80]}...")
                    print(f"   Concept: {var.get('concept', 'N/A')}")
                    
                    if var.get('summary'):
                        print(f"   Summary: {var['summary'][:100]}...")
                    if var.get('key_terms'):
                        terms = var['key_terms']
                        if isinstance(terms, list):
                            print(f"   Key terms: {', '.join(terms[:5])}")
                else:
                    print(f"   ❌ Variable not found in canonical data")
                print()
    
    def run_full_diagnostic(self):
        """Run comprehensive diagnostic"""
        print("🔍 COMPREHENSIVE SEARCH SYSTEM DIAGNOSTIC")
        print("=" * 70)
        
        # 1. Inspect key variables
        key_variables = ["B17001_002E", "B19013_001E", "B08303_001E"]
        for var_id in key_variables:
            self.inspect_variable_embedding_text(var_id)
        
        # 2. Inspect key tables  
        key_tables = ["B17001", "B19013", "B08303"]
        for table_id in key_tables:
            self.inspect_table_embedding_text(table_id)
        
        # 3. Debug problem queries
        problem_queries = [
            "poverty rate",
            "median household income", 
            "travel time to work",
            "commute time"
        ]
        
        for query in problem_queries:
            self.debug_query_search(query, search_type="both")
        
        print("\n🎯 DIAGNOSTIC COMPLETE")
        print("Look for:")
        print("1. ❌ Missing variables/tables in canonical data")
        print("2. 🔑 Tables without search keywords") 
        print("3. 📊 Low confidence scores for expected matches")
        print("4. 🔍 Mismatched embedding text vs expectations")


def main():
    """Run diagnostic based on command line args"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Search System Diagnostic Tool')
    parser.add_argument('--variable', help='Inspect specific variable embedding text')
    parser.add_argument('--table', help='Inspect specific table embedding text')
    parser.add_argument('--query', help='Debug specific search query')
    parser.add_argument('--search-type', choices=['tables', 'variables', 'both'], 
                       default='both', help='Type of search to debug')
    parser.add_argument('--full', action='store_true', help='Run full comprehensive diagnostic')
    
    args = parser.parse_args()
    
    diagnostic = SearchDiagnostic()
    
    if args.full:
        diagnostic.run_full_diagnostic()
    elif args.variable:
        diagnostic.inspect_variable_embedding_text(args.variable)
    elif args.table:
        diagnostic.inspect_table_embedding_text(args.table)
    elif args.query:
        diagnostic.debug_query_search(args.query, args.search_type)
    else:
        print("No specific diagnostic requested. Use --full for comprehensive diagnostic.")
        print("Or use --variable, --table, or --query for specific inspections.")


if __name__ == "__main__":
    main()
