#!/usr/bin/env python3
"""
Enhanced Table Catalog Extractor with Keyword Support

Builds Census table catalog from concept-based canonical variables and 
official table metadata. Now supports enhanced embeddings with keywords.

Usage:
    python enhanced_build_table_catalog.py --use-keywords
    
Output:
    - table_catalog_enhanced.json: Enhanced table catalog with keywords
    - table_embeddings_enhanced.faiss: Enhanced embeddings including keywords
"""

import json
import re
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import existing classes from original build script
from build_table_catalog_refactored import (
    SurveyInstance, ConceptVariable, TableCatalog,
    OfficialLabelParser, ConceptBasedTableCatalogExtractor
)

class EnhancedTableCatalogExtractor(ConceptBasedTableCatalogExtractor):
    """Enhanced extractor that includes keywords in embeddings"""
    
    def __init__(self, 
                 canonical_path: str = "source-docs/canonical_variables_refactored.json",
                 table_list_path: str = "source-docs/acs_table_shells/2023_DataProductList.xlsx",
                 keywords_catalog_path: str = "table-catalog/table_catalog_with_keywords.json"):
        super().__init__(canonical_path, table_list_path)
        self.keywords_catalog_path = Path(keywords_catalog_path)
        self.keywords_data = self._load_keywords_catalog()
    
    def _load_keywords_catalog(self) -> Dict:
        """Load catalog with generated keywords"""
        if not self.keywords_catalog_path.exists():
            logger.warning(f"Keywords catalog not found: {self.keywords_catalog_path}")
            logger.warning("Will create embeddings without keywords")
            return {}
        
        with open(self.keywords_catalog_path, 'r') as f:
            keywords_catalog = json.load(f)
        
        # Extract keywords by table_id
        keywords_by_table = {}
        for table in keywords_catalog.get('tables', []):
            table_id = table['table_id']
            search_keywords = table.get('search_keywords', {})
            if search_keywords:
                keywords_by_table[table_id] = search_keywords
        
        logger.info(f"Loaded keywords for {len(keywords_by_table)} tables")
        return keywords_by_table
    
    def create_enhanced_embeddings(self, catalogs: List[TableCatalog]) -> Tuple[np.ndarray, List[str]]:
        """Create enhanced embeddings including keywords when available"""
        logger.info("Creating enhanced table embeddings with keywords...")
        
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        embedding_texts = []
        table_ids = []
        keywords_used_count = 0
        
        for catalog in catalogs:
            # Start with core metadata (prioritized order for search)
            text_parts = []
            
            # Get keywords if available
            keywords = self.keywords_data.get(catalog.table_id, {})
            
            # 1. Primary keywords first (highest search priority)
            if keywords.get('primary_keywords'):
                primary_kw = ', '.join(keywords['primary_keywords'])
                text_parts.append(f"Primary search terms: {primary_kw}")
                keywords_used_count += 1
            
            # 2. Summary (user-friendly explanation)
            if keywords.get('summary'):
                text_parts.append(f"Summary: {keywords['summary']}")
            
            # 3. Official metadata
            text_parts.extend([
                f"Title: {catalog.title}",
                f"Universe: {catalog.universe}",
                f"Concept: {catalog.concept}"
            ])
            
            # 4. Secondary keywords (additional search terms)
            if keywords.get('secondary_keywords'):
                secondary_kw = ', '.join(keywords['secondary_keywords'])
                text_parts.append(f"Related terms: {secondary_kw}")
            
            # 5. Survey context (for disambiguation)
            if len(catalog.survey_programs) == 1:
                if 'acs1' in catalog.survey_programs:
                    text_parts.append("1-year survey data")
                else:
                    text_parts.append("5-year survey data")
            else:
                text_parts.append("Available in both 1-year and 5-year surveys")
            
            # 6. Key dimensional info (limited to avoid noise)
            if catalog.dimensional_categories:
                dims = sorted(list(catalog.dimensional_categories))[:3]
                text_parts.append(f"Dimensions: {', '.join(dims)}")
            
            # Clean and combine
            embedding_text = '. '.join(filter(None, text_parts))
            # Remove Census jargon that doesn't help search
            embedding_text = re.sub(r'\b(Estimate!!|Margin of Error!!)\b', '', embedding_text)
            embedding_text = re.sub(r'\s+', ' ', embedding_text).strip()
            
            embedding_texts.append(embedding_text)
            table_ids.append(catalog.table_id)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(embedding_texts, show_progress_bar=True)
        
        logger.info(f"Generated enhanced embeddings for {len(embeddings)} tables")
        logger.info(f"Keywords used in {keywords_used_count} embeddings")
        
        return embeddings, table_ids
    
    def save_enhanced_catalog(self, catalogs: List[TableCatalog], embeddings: np.ndarray,
                            table_ids: List[str], output_dir: str = "table-catalog"):
        """Save enhanced catalog and embeddings"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format (reuse parent method logic)
        catalog_data = []
        for catalog in catalogs:
            # Manual serialization to handle complex nested objects
            catalog_dict = {
                'table_id': catalog.table_id,
                'title': catalog.title,
                'universe': catalog.universe,
                'concept': catalog.concept,
                'data_product_type': catalog.data_product_type,
                'survey_programs': catalog.survey_programs,
                'geography_restrictions_1yr': catalog.geography_restrictions_1yr,
                'geography_restrictions_5yr': catalog.geography_restrictions_5yr,
                'geography_levels': catalog.geography_levels,
                'variable_count': catalog.variable_count,
                'primary_variable': catalog.primary_variable,
                'dimensional_categories': list(catalog.dimensional_categories),
                'methodology_topics': catalog.methodology_topics,
                'statistical_notes': catalog.statistical_notes,
                'survey_availability': dict(catalog.survey_availability),
                'variables': []
            }
            
            # Add keywords if available
            if catalog.table_id in self.keywords_data:
                catalog_dict['search_keywords'] = self.keywords_data[catalog.table_id]
            
            # Serialize variables manually
            for var in catalog.variables:
                var_dict = {
                    'variable_id': var.variable_id,
                    'concept': var.concept,
                    'label': var.label,
                    'table_id': var.table_id,
                    'available_surveys': var.available_surveys,
                    'geography_coverage': dict(var.geography_coverage),
                    'primary_instance': var.primary_instance,
                    'dimensional_tags': dict(var.dimensional_tags),
                    'category_weights': dict(var.category_weights),
                    'is_estimate': var.is_estimate,
                    'is_margin_error': var.is_margin_error,
                    'survey_instances': []
                }
                
                # Serialize survey instances manually
                for instance in var.survey_instances:
                    instance_dict = {
                        'dataset': instance.dataset,
                        'survey_type': instance.survey_type,
                        'year': instance.year,
                        'geography_vintage': instance.geography_vintage,
                        'geography_restrictions': instance.geography_restrictions,
                        'sample_characteristics': instance.sample_characteristics,
                        'status': instance.status
                    }
                    var_dict['survey_instances'].append(instance_dict)
                
                catalog_dict['variables'].append(var_dict)
            
            catalog_data.append(catalog_dict)
        
        # Enhanced metadata
        keywords_used = len([t for t in table_ids if t in self.keywords_data])
        
        # Save enhanced table catalog
        catalog_file = output_path / "table_catalog_enhanced.json"
        with open(catalog_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model_version': '3.1_enhanced_with_keywords',
                    'total_tables': len(catalogs),
                    'total_concepts': sum(c.variable_count for c in catalogs),
                    'keywords_integrated': keywords_used,
                    'keywords_coverage': f"{keywords_used}/{len(catalogs)} ({keywords_used/len(catalogs)*100:.1f}%)",
                    'extraction_stats': dict(self.stats),
                    'embedding_model_used': 'sentence-transformers/all-mpnet-base-v2',
                    'enhancement_features': [
                        'Primary keywords prioritized in embeddings',
                        'User-friendly summaries included',
                        'Secondary keywords for broader recall',
                        'Survey context for disambiguation',
                        'Optimized text order for search relevance'
                    ],
                    'data_sources': {
                        'table_metadata': str(self.table_list_path),
                        'variable_details': str(self.canonical_path),
                        'search_keywords': str(self.keywords_catalog_path)
                    }
                },
                'tables': catalog_data
            }, f, indent=2)
        
        logger.info(f"Saved enhanced table catalog to {catalog_file}")
        
        # Save enhanced FAISS embeddings
        embeddings_array = embeddings.astype('float32')
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        faiss_file = output_path / "table_embeddings_enhanced.faiss"
        faiss.write_index(index, str(faiss_file))
        
        # Save enhanced table ID mapping
        mapping_file = output_path / "table_mapping_enhanced.json"
        with open(mapping_file, 'w') as f:
            json.dump({
                'table_ids': table_ids,
                'embedding_dimension': dimension,
                'total_embeddings': len(table_ids),
                'keywords_integrated': keywords_used,
                'embedding_enhancement': 'keywords_v1.0'
            }, f, indent=2)
        
        logger.info(f"Saved enhanced FAISS embeddings to {faiss_file}")
        
        # Save dimensional vocabulary
        vocab = self.extract_dimensional_vocabulary(catalogs)
        vocab_file = output_path / "dimensional_vocabulary_enhanced.json"
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent=2)
        
        logger.info(f"Saved enhanced dimensional vocabulary to {vocab_file}")


def main():
    """Main enhanced extraction process"""
    parser = argparse.ArgumentParser(description='Enhanced Table Catalog Builder with Keywords')
    parser.add_argument('--use-keywords', action='store_true', 
                       help='Use keywords catalog for enhanced embeddings')
    parser.add_argument('--keywords-catalog', 
                       default='table-catalog/table_catalog_with_keywords.json',
                       help='Path to catalog with generated keywords')
    parser.add_argument('--output-suffix', default='enhanced',
                       help='Suffix for output files')
    
    args = parser.parse_args()
    
    if args.use_keywords:
        logger.info("🚀 Starting Enhanced Table Catalog Extraction with Keywords...")
        logger.info("📦 Using refactored canonical variables + generated keywords")
        logger.info("🎯 Creating search-optimized embeddings")
        
        extractor = EnhancedTableCatalogExtractor(
            keywords_catalog_path=args.keywords_catalog
        )
    else:
        logger.info("🚀 Starting Standard Table Catalog Extraction...")
        extractor = ConceptBasedTableCatalogExtractor()
    
    # Load data (same as original)
    table_metadata = extractor.load_table_metadata()
    concepts = extractor.load_refactored_canonical_variables()
    
    # Group concepts by table
    table_variables = extractor.group_variables_by_table(concepts)
    
    # JOIN with enhanced survey awareness
    catalogs = extractor.join_table_data(table_metadata, table_variables)
    
    # Create embeddings (enhanced or standard)
    if args.use_keywords:
        embeddings, table_ids = extractor.create_enhanced_embeddings(catalogs)
        extractor.save_enhanced_catalog(catalogs, embeddings, table_ids)
        
        logger.info("✅ Enhanced table catalog extraction complete!")
        logger.info("🎯 Enhanced features delivered:")
        logger.info("   - Keywords prioritized in embeddings")
        logger.info("   - User-friendly summaries included")
        logger.info("   - Optimized search relevance")
        logger.info("📁 Enhanced output files:")
        logger.info("   - table-catalog/table_catalog_enhanced.json")
        logger.info("   - table-catalog/table_embeddings_enhanced.faiss")
        logger.info("   - table-catalog/table_mapping_enhanced.json")
        
    else:
        embeddings, table_ids = extractor.create_table_embeddings(catalogs)
        extractor.save_catalog(catalogs, embeddings, table_ids)
        
        logger.info("✅ Standard table catalog extraction complete!")
    
    # Print comprehensive statistics
    extractor.print_statistics(catalogs)

if __name__ == "__main__":
    main()
