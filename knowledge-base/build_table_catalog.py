#!/usr/bin/env python3
"""
Table Catalog Extractor - Complete Census Metadata Catalog Builder

Joins official Census table metadata (DataProductList.xlsx) with enriched 
canonical variables to create a unified catalog for coarse-to-fine retrieval.

Like a SQL query walking into a bar: "May I JOIN you?"

Usage:
    python build_table_catalog.py
    
Output:
    - table_catalog.json: Complete table catalog with metadata
    - table_embeddings.faiss: Embeddings for coarse retrieval
    - dimensional_vocabulary.json: Extracted dimensional tags
"""

import json
import re
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Variable:
    """Individual variable within a table"""
    variable_id: str
    temporal_id: str
    label: str
    concept: str
    survey_context: str
    acs5_available: bool
    acs1_available: bool
    dimensional_tags: Dict[str, str]
    is_estimate: bool
    is_margin_error: bool
    category_weights: Dict[str, float]

@dataclass
class TableCatalog:
    """Complete table entry in the catalog"""
    table_id: str
    title: str  # Official from DataProductList
    universe: str  # Official from DataProductList
    concept: str  # Derived from variables
    data_product_type: str  # From DataProductList
    survey_programs: List[str]  # ['acs1', 'acs5'] from DataProductList
    geography_restrictions_1yr: str  # From DataProductList
    geography_restrictions_5yr: str  # From DataProductList
    geography_levels: List[str]  # Parsed from restrictions
    variable_count: int
    variables: List[Variable]
    primary_variable: Optional[str]  # Usually _001E
    dimensional_categories: Set[str]
    methodology_topics: List[str]
    statistical_notes: List[str]

class DimensionalTagger:
    """Extracts dimensional tags from variable labels and concepts"""
    
    def __init__(self):
        # Controlled vocabulary for dimensional tags
        self.sex_patterns = {
            'male': ['male', 'men', 'man'],
            'female': ['female', 'women', 'woman'],
            'total': ['total', 'both sexes']
        }
        
        self.age_patterns = {
            'under_5': ['under 5', 'less than 5'],
            '5_to_9': ['5 to 9'],
            '10_to_14': ['10 to 14'],
            '15_to_17': ['15 to 17'],
            '18_to_24': ['18 to 24'],
            '25_to_34': ['25 to 34'],
            '35_to_44': ['35 to 44'],
            '45_to_54': ['45 to 54'],
            '55_to_64': ['55 to 64'],
            '65_plus': ['65 and over', '65 years and over', '65+'],
            'working_age': ['16 years and over', '18 years and over'],
            'total': ['total', 'all ages']
        }
        
        self.income_patterns = {
            'under_10k': ['less than $10,000', 'under $10,000'],
            '10k_to_15k': ['$10,000 to $14,999'],
            '15k_to_25k': ['$15,000 to $24,999'],
            '25k_to_35k': ['$25,000 to $34,999'],
            '35k_to_50k': ['$35,000 to $49,999'],
            '50k_to_75k': ['$50,000 to $74,999'],
            '75k_to_100k': ['$75,000 to $99,999'],
            '100k_plus': ['$100,000 or more', '$100,000 and over'],
            'median': ['median'],
            'aggregate': ['aggregate'],
            'total': ['total']
        }
        
        self.race_patterns = {
            'white': ['white alone', 'white'],
            'black': ['black or african american', 'black'],
            'asian': ['asian alone', 'asian'],
            'hispanic': ['hispanic or latino', 'hispanic'],
            'native': ['american indian and alaska native'],
            'pacific': ['native hawaiian and other pacific islander'],
            'other': ['some other race', 'other'],
            'two_plus': ['two or more races'],
            'total': ['total']
        }
        
        self.tenure_patterns = {
            'owner': ['owner occupied', 'owned'],
            'renter': ['renter occupied', 'rented'],
            'total': ['total']
        }
        
        self.time_patterns = {
            'less_5min': ['less than 5 minutes'],
            '5_to_9min': ['5 to 9 minutes'],
            '10_to_14min': ['10 to 14 minutes'],
            '15_to_19min': ['15 to 19 minutes'],
            '20_to_24min': ['20 to 24 minutes'],
            '25_to_29min': ['25 to 29 minutes'],
            '30_to_34min': ['30 to 34 minutes'],
            '35_plus_min': ['35 minutes or more', '35 or more minutes'],
            'total': ['total']
        }
        
        self.education_patterns = {
            'less_than_hs': ['less than high school', 'no schooling'],
            'high_school': ['high school graduate', 'ged'],
            'some_college': ['some college', 'associates'],
            'bachelors': ['bachelor', "bachelor's"],
            'graduate': ['graduate', 'professional', 'masters', 'doctorate'],
            'total': ['total']
        }
    
    def extract_tags(self, label: str, concept: str) -> Dict[str, str]:
        """Extract dimensional tags from variable label and concept"""
        tags = {}
        label_lower = label.lower()
        concept_lower = concept.lower()
        text = f"{label_lower} {concept_lower}"
        
        # Check each dimensional category
        for category, patterns_dict in [
            ('sex', self.sex_patterns),
            ('age', self.age_patterns),
            ('income', self.income_patterns),
            ('race', self.race_patterns),
            ('tenure', self.tenure_patterns),
            ('travel_time', self.time_patterns),
            ('education', self.education_patterns)
        ]:
            for tag_value, patterns in patterns_dict.items():
                for pattern in patterns:
                    if pattern in text:
                        tags[category] = tag_value
                        break
                if category in tags:
                    break
        
        # Special handling for estimate vs margin of error
        if '!!margin of error' in label_lower or label.endswith('_M'):
            tags['stat_type'] = 'margin_error'
        elif '!!estimate' in label_lower or label.endswith('_E'):
            tags['stat_type'] = 'estimate'
        
        return tags

class TableCatalogExtractor:
    """Main extractor that joins table metadata with variable details"""
    
    def __init__(self, 
                 canonical_path: str = "source-docs/canonical_variables.json",
                 table_list_path: str = "source-docs/acs_table_shells/2023_DataProductList.xlsx"):
        self.canonical_path = Path(canonical_path)
        self.table_list_path = Path(table_list_path)
        self.tagger = DimensionalTagger()
        self.embedding_model = None
        
        # Statistics
        self.stats = {
            'variables_processed': 0,
            'tables_from_excel': 0,
            'tables_with_variables': 0,
            'tables_without_variables': 0,
            'dimensional_tags_extracted': 0,
            'missing_concepts': 0
        }
    
    def load_table_metadata(self) -> pd.DataFrame:
        """Load official table metadata from DataProductList.xlsx"""
        logger.info(f"Loading table metadata from {self.table_list_path}")
        
        # Read Excel file
        df = pd.read_excel(self.table_list_path)
        
        # Clean column names (remove any whitespace/special chars)
        df.columns = df.columns.str.strip()
        
        # Expected columns based on your sample
        expected_cols = [
            'Table ID', 'Table Title', 'Table Universe', 'Data Product Type', 
            'Year', '1-Year Geography Restrictions\n(with Summary Levels in Parentheses)',
            '5-Year Geography Restrictions\n(with Summary Levels in Parentheses)'
        ]
        
        # Handle potential column name variations
        col_mapping = {}
        for expected in expected_cols:
            for actual in df.columns:
                if 'Table ID' in expected and 'Table ID' in actual:
                    col_mapping[actual] = 'table_id'
                elif 'Table Title' in expected and 'Table Title' in actual:
                    col_mapping[actual] = 'title'
                elif 'Table Universe' in expected and 'Table Universe' in actual:
                    col_mapping[actual] = 'universe'
                elif 'Data Product Type' in expected and 'Product Type' in actual:
                    col_mapping[actual] = 'data_product_type'
                elif 'Year' in expected and 'Year' in actual:
                    col_mapping[actual] = 'year'
                elif '1-Year Geography' in expected and '1-Year' in actual:
                    col_mapping[actual] = 'geo_1yr'
                elif '5-Year Geography' in expected and '5-Year' in actual:
                    col_mapping[actual] = 'geo_5yr'
        
        # Rename columns
        df = df.rename(columns=col_mapping)
        
        # Filter for ACS tables (B, C, S, DP prefixes)
        acs_tables = df[df['table_id'].str.match(r'^[BCSDP]\d+[A-Z]*$', na=False)]
        
        logger.info(f"Loaded {len(acs_tables)} ACS tables from Excel")
        self.stats['tables_from_excel'] = len(acs_tables)
        
        return acs_tables
    
    def load_canonical_variables(self) -> Dict:
        """Load canonical variables from JSON"""
        logger.info(f"Loading canonical variables from {self.canonical_path}")
        
        with open(self.canonical_path, 'r') as f:
            data = json.load(f)
        
        variables = data.get('variables', data)
        logger.info(f"Loaded {len(variables)} canonical variables")
        return variables
    
    def extract_table_id(self, variable_id: str) -> str:
        """Extract table ID from variable ID (B08303_001E -> B08303)"""
        return variable_id.split('_')[0]
    
    def determine_survey_availability(self, temporal_id: str) -> Tuple[bool, bool]:
        """Determine ACS1 and ACS5 availability from temporal ID"""
        acs5_available = 'acs_5yr' in temporal_id or '5yr' in temporal_id
        acs1_available = 'acs_1yr' in temporal_id or '1yr' in temporal_id
        return acs5_available, acs1_available
    
    def parse_geography_levels(self, geo_1yr: str, geo_5yr: str) -> List[str]:
        """Parse geography levels from restriction strings"""
        geo_levels = set()
        
        # Common geography mappings
        geo_mappings = {
            'nation': ['010', 'national'],
            'state': ['040', 'state'],
            'county': ['050', 'county'],
            'place': ['160', 'place'],
            'tract': ['140', 'tract'],
            'block_group': ['150', 'block group'],
            'zcta': ['860', 'zcta', 'zip'],
            'metro': ['310', 'metro', 'msa']
        }
        
        # Check both 1-year and 5-year restrictions
        restriction_text = f"{geo_1yr or ''} {geo_5yr or ''}".lower()
        
        for geo_level, identifiers in geo_mappings.items():
            if any(identifier in restriction_text for identifier in identifiers):
                geo_levels.add(geo_level)
        
        # If no specific levels found, assume standard availability
        if not geo_levels:
            geo_levels = {'state', 'county', 'place'}
        
        return sorted(list(geo_levels))
    
    def group_variables_by_table(self, variables: Dict) -> Dict[str, List[Variable]]:
        """Group variables by table ID"""
        logger.info("Grouping variables by table...")
        
        tables = defaultdict(list)
        
        for temporal_id, var_data in variables.items():
            variable_id = var_data.get('variable_id', temporal_id)
            table_id = self.extract_table_id(variable_id)
            
            # Skip if missing essential data
            if not var_data.get('concept') or not var_data.get('label'):
                self.stats['missing_concepts'] += 1
                continue
            
            # Create Variable object
            acs5_avail, acs1_avail = self.determine_survey_availability(temporal_id)
            dimensional_tags = self.tagger.extract_tags(
                var_data.get('label', ''),
                var_data.get('concept', '')
            )
            
            if dimensional_tags:
                self.stats['dimensional_tags_extracted'] += 1
            
            variable = Variable(
                variable_id=variable_id,
                temporal_id=temporal_id,
                label=var_data.get('label', ''),
                concept=var_data.get('concept', ''),
                survey_context=var_data.get('survey_context', ''),
                acs5_available=acs5_avail,
                acs1_available=acs1_avail,
                dimensional_tags=dimensional_tags,
                is_estimate=variable_id.endswith('_E'),
                is_margin_error=variable_id.endswith('_M'),
                category_weights=var_data.get('category_weights_linear', {})
            )
            
            tables[table_id].append(variable)
            self.stats['variables_processed'] += 1
        
        logger.info(f"Grouped {self.stats['variables_processed']} variables into {len(tables)} tables")
        return dict(tables)
    
    def join_table_data(self, table_metadata: pd.DataFrame, 
                       table_variables: Dict[str, List[Variable]]) -> List[TableCatalog]:
        """JOIN the two tables! Official metadata + variable details"""
        logger.info("🍺 Joining table metadata with variable data...")
        
        catalogs = []
        
        for _, table_row in table_metadata.iterrows():
            table_id = table_row['table_id']
            variables = table_variables.get(table_id, [])
            
            if not variables:
                self.stats['tables_without_variables'] += 1
                continue
            
            self.stats['tables_with_variables'] += 1
            
            # Get primary concept from variables
            concepts = [v.concept for v in variables if v.concept]
            primary_concept = Counter(concepts).most_common(1)[0][0] if concepts else table_row['title']
            
            # Determine survey programs from Excel year column
            survey_programs = []
            year_data = str(table_row.get('year', '')).strip()
            if '1' in year_data:
                survey_programs.append('acs1')
            if '5' in year_data:
                survey_programs.append('acs5')
            
            # Parse geography levels
            geography_levels = self.parse_geography_levels(
                table_row.get('geo_1yr', ''),
                table_row.get('geo_5yr', '')
            )
            
            # Find primary variable (usually _001E)
            primary_variable = None
            for var in variables:
                if var.variable_id.endswith('_001E') and var.is_estimate:
                    primary_variable = var.variable_id
                    break
            
            # Collect dimensional categories
            dimensional_categories = set()
            for var in variables:
                dimensional_categories.update(var.dimensional_tags.keys())
            
            # Generate methodology topics
            methodology_topics = [primary_concept.lower()]
            if dimensional_categories:
                methodology_topics.extend([f"{cat}_analysis" for cat in dimensional_categories])
            
            # Generate statistical notes
            statistical_notes = []
            if 'sex' in dimensional_categories:
                statistical_notes.append("Available by sex (male/female)")
            if 'age' in dimensional_categories:
                statistical_notes.append("Available by age groups")
            if 'race' in dimensional_categories:
                statistical_notes.append("Available by race/ethnicity")
            if any(v.is_margin_error for v in variables):
                statistical_notes.append("Includes margins of error for statistical reliability")
            
            # Add survey-specific notes
            if len(survey_programs) == 1:
                if 'acs1' in survey_programs:
                    statistical_notes.append("ACS 1-year only: more current but limited geography")
                else:
                    statistical_notes.append("ACS 5-year only: broader geography but less current")
            
            catalog = TableCatalog(
                table_id=table_id,
                title=table_row['title'],
                universe=table_row['universe'],
                concept=primary_concept,
                data_product_type=table_row.get('data_product_type', 'Unknown'),
                survey_programs=survey_programs,
                geography_restrictions_1yr=table_row.get('geo_1yr', ''),
                geography_restrictions_5yr=table_row.get('geo_5yr', ''),
                geography_levels=geography_levels,
                variable_count=len(variables),
                variables=variables,
                primary_variable=primary_variable,
                dimensional_categories=dimensional_categories,
                methodology_topics=methodology_topics,
                statistical_notes=statistical_notes
            )
            
            catalogs.append(catalog)
        
        logger.info(f"✅ Successfully joined data: {len(catalogs)} complete table catalogs")
        logger.info(f"   Tables with variables: {self.stats['tables_with_variables']}")
        logger.info(f"   Tables without variables: {self.stats['tables_without_variables']}")
        
        return catalogs
    
    def create_table_embeddings(self, catalogs: List[TableCatalog]) -> Tuple[np.ndarray, List[str]]:
        """Create embeddings for table concepts for coarse retrieval"""
        logger.info("Creating table embeddings...")
        
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        embedding_texts = []
        table_ids = []
        
        for catalog in catalogs:
            # Use official title + universe + concept for clean embeddings
            text_parts = [
                catalog.title,  # Official Census table title
                catalog.universe,  # Official universe definition
                catalog.concept  # Primary concept from variables
            ]
            
            # Add key dimensional info if meaningful
            if catalog.dimensional_categories:
                dims = list(catalog.dimensional_categories)[:3]  # Limit to avoid noise
                text_parts.append(f"by {', '.join(dims)}")
            
            # Create clean embedding text
            embedding_text = '. '.join(filter(None, text_parts))
            
            # Remove Census formatting noise
            embedding_text = re.sub(r'\b(Estimate!!|Margin of Error!!)\b', '', embedding_text)
            embedding_text = re.sub(r'\s+', ' ', embedding_text).strip()
            
            embedding_texts.append(embedding_text)
            table_ids.append(catalog.table_id)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(embedding_texts, show_progress_bar=True)
        
        logger.info(f"Generated embeddings for {len(embeddings)} tables")
        return embeddings, table_ids
    
    def save_catalog(self, catalogs: List[TableCatalog], embeddings: np.ndarray, 
                    table_ids: List[str], output_dir: str = "table-catalog"):
        """Save complete catalog and embeddings"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert catalogs to serializable format
        catalog_data = []
        for catalog in catalogs:
            catalog_dict = asdict(catalog)
            # Convert sets to lists for JSON serialization
            catalog_dict['dimensional_categories'] = list(catalog_dict['dimensional_categories'])
            catalog_data.append(catalog_dict)
        
        # Save table catalog
        catalog_file = output_path / "table_catalog.json"
        with open(catalog_file, 'w') as f:
            json.dump({
                'metadata': {
                    'total_tables': len(catalogs),
                    'total_variables': sum(c.variable_count for c in catalogs),
                    'extraction_stats': self.stats,
                    'model_used': 'sentence-transformers/all-mpnet-base-v2',
                    'data_sources': {
                        'table_metadata': str(self.table_list_path),
                        'variable_details': str(self.canonical_path)
                    }
                },
                'tables': catalog_data
            }, f, indent=2)
        
        logger.info(f"💾 Saved table catalog to {catalog_file}")
        
        # Save embeddings as FAISS index
        embeddings_array = embeddings.astype('float32')
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        faiss_file = output_path / "table_embeddings.faiss"
        faiss.write_index(index, str(faiss_file))
        
        # Save table ID mapping
        mapping_file = output_path / "table_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump({
                'table_ids': table_ids,
                'embedding_dimension': dimension,
                'total_embeddings': len(table_ids)
            }, f, indent=2)
        
        logger.info(f"🔍 Saved FAISS embeddings to {faiss_file}")
        
        # Save dimensional vocabulary
        vocab = self.extract_dimensional_vocabulary(catalogs)
        vocab_file = output_path / "dimensional_vocabulary.json"
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent=2)
        
        logger.info(f"📝 Saved dimensional vocabulary to {vocab_file}")
    
    def extract_dimensional_vocabulary(self, catalogs: List[TableCatalog]) -> Dict:
        """Extract dimensional vocabulary from all tables"""
        vocabulary = defaultdict(Counter)
        
        for catalog in catalogs:
            for var in catalog.variables:
                for dim_type, dim_value in var.dimensional_tags.items():
                    vocabulary[dim_type][dim_value] += 1
        
        # Convert to regular dict for serialization
        return {dim_type: dict(values) for dim_type, values in vocabulary.items()}
    
    def print_statistics(self, catalogs: List[TableCatalog]):
        """Print extraction and join statistics"""
        logger.info("=== TABLE CATALOG JOIN COMPLETE ===")
        logger.info(f"📊 Tables from Excel: {self.stats['tables_from_excel']:,}")
        logger.info(f"📊 Variables processed: {self.stats['variables_processed']:,}")
        logger.info(f"✅ Tables with variables: {self.stats['tables_with_variables']:,}")
        logger.info(f"❌ Tables without variables: {self.stats['tables_without_variables']:,}")
        logger.info(f"🏷️  Variables with dimensional tags: {self.stats['dimensional_tags_extracted']:,}")
        logger.info(f"⚠️  Missing concepts: {self.stats['missing_concepts']:,}")
        
        # Table size distribution
        table_sizes = [c.variable_count for c in catalogs]
        logger.info(f"📈 Avg variables per table: {np.mean(table_sizes):.1f}")
        logger.info(f"📈 Max variables in table: {max(table_sizes)}")
        
        # Survey program coverage
        acs5_tables = sum(1 for c in catalogs if 'acs5' in c.survey_programs)
        acs1_tables = sum(1 for c in catalogs if 'acs1' in c.survey_programs)
        both_tables = sum(1 for c in catalogs if len(c.survey_programs) == 2)
        logger.info(f"📅 Tables with ACS5: {acs5_tables:,}")
        logger.info(f"📅 Tables with ACS1: {acs1_tables:,}")
        logger.info(f"📅 Tables with both: {both_tables:,}")
        
        # Dimensional categories
        all_dimensions = set()
        for c in catalogs:
            all_dimensions.update(c.dimensional_categories)
        logger.info(f"🏷️  Unique dimensional categories: {len(all_dimensions)}")
        logger.info(f"🏷️  Categories: {sorted(all_dimensions)}")
        
        # Sample some tables
        logger.info("\n📋 Sample table catalogs:")
        for i, catalog in enumerate(catalogs[:3]):
            logger.info(f"   {i+1}. {catalog.table_id}: {catalog.title}")
            logger.info(f"      Universe: {catalog.universe}")
            logger.info(f"      Variables: {catalog.variable_count}, Primary: {catalog.primary_variable}")
            logger.info(f"      Dimensions: {sorted(catalog.dimensional_categories)}")

def main():
    """Main extraction and join process"""
    logger.info("🚀 Starting Table Catalog Extraction with Official Metadata...")
    
    extractor = TableCatalogExtractor()
    
    # Load both data sources
    table_metadata = extractor.load_table_metadata()
    variables = extractor.load_canonical_variables()
    
    # Group variables by table
    table_variables = extractor.group_variables_by_table(variables)
    
    # JOIN the tables! 🍺
    catalogs = extractor.join_table_data(table_metadata, table_variables)
    
    # Create embeddings
    embeddings, table_ids = extractor.create_table_embeddings(catalogs)
    
    # Save everything
    extractor.save_catalog(catalogs, embeddings, table_ids)
    
    # Print stats
    extractor.print_statistics(catalogs)
    
    logger.info("✅ Table catalog extraction complete!")
    logger.info("📁 Output files:")
    logger.info("   - table-catalog/table_catalog.json")
    logger.info("   - table-catalog/table_embeddings.faiss")
    logger.info("   - table-catalog/dimensional_vocabulary.json")
    logger.info("🎯 Ready for coarse-to-fine retrieval implementation!")

if __name__ == "__main__":
    main()
