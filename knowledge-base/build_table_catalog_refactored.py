#!/usr/bin/env python3
"""
Table Catalog Extractor - Concept-Based Architecture

Builds Census table catalog from concept-based canonical variables and 
official table metadata. Eliminates duplicate variables and enables 
survey-aware intelligence.

Usage:
    python build_table_catalog_refactored.py
    
Output:
    - table_catalog.json: Clean table catalog with survey instances
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
class SurveyInstance:
    """Survey instance information from refactored structure"""
    dataset: str
    survey_type: str  # 1yr, 5yr
    year: str
    geography_vintage: str
    geography_restrictions: List[str]
    sample_characteristics: str
    status: str

@dataclass
class ConceptVariable:
    """Variable concept with survey instances"""
    variable_id: str
    concept: str
    label: str
    table_id: str
    survey_instances: List[SurveyInstance]
    available_surveys: List[str]
    geography_coverage: Dict[str, List[str]]
    primary_instance: str
    dimensional_tags: Dict[str, str]
    category_weights: Dict[str, float]
    is_estimate: bool
    is_margin_error: bool

@dataclass
class TableCatalog:
    """Complete table entry with survey-aware metadata"""
    table_id: str
    title: str
    universe: str
    concept: str
    data_product_type: str
    survey_programs: List[str]
    geography_restrictions_1yr: str
    geography_restrictions_5yr: str
    geography_levels: List[str]
    variable_count: int
    variables: List[ConceptVariable]
    primary_variable: Optional[str]
    dimensional_categories: Set[str]
    methodology_topics: List[str]
    statistical_notes: List[str]
    survey_availability: Dict[str, Dict[str, List[str]]]  # {survey_type: {geo_level: [restrictions]}}

class OfficialLabelParser:
    """Parse Census variable labels using official structure"""
    
    def __init__(self):
        # Only parse what Census actually provides in labels
        pass
    
    def parse_label_hierarchy(self, label: str) -> Dict[str, str]:
        """Parse Census variable label hierarchy using official delimiters"""
        tags = {}
        
        # Parse the official "!!" delimited structure
        if '!!' in label:
            parts = label.split('!!')
            
            # Remove common prefixes
            if parts and parts[0].lower() in ['estimate', 'margin of error']:
                tags['stat_type'] = 'estimate' if parts[0].lower() == 'estimate' else 'margin_error'
                parts = parts[1:]
            
            # Extract hierarchy levels (without imposing our own categories)
            for i, part in enumerate(parts):
                if part.strip():
                    tags[f'level_{i}'] = part.strip()
        
        # Only add what we can definitively determine from the variable ID
        variable_id = label.split('_')[-1] if '_' in label else ''
        if variable_id.endswith('_E'):
            tags['stat_type'] = 'estimate'
        elif variable_id.endswith('_M'):
            tags['stat_type'] = 'margin_error'
        
        return tags
    
    def extract_tags(self, label: str, concept: str) -> Dict[str, str]:
        """Extract only what can be definitively determined from official labels"""
        return self.parse_label_hierarchy(label)

class ConceptBasedTableCatalogExtractor:
    """Enhanced extractor for concept-based canonical variables"""
    
    def __init__(self,
                 canonical_path: str = "source-docs/canonical_variables_refactored.json",
                 table_list_path: str = "source-docs/acs_table_shells/2023_DataProductList.xlsx"):
        self.canonical_path = Path(canonical_path)
        self.table_list_path = Path(table_list_path)
        self.tagger = OfficialLabelParser()
        self.embedding_model = None
        
        # Enhanced statistics tracking
        self.stats = {
            'concepts_processed': 0,
            'survey_instances_processed': 0,
            'tables_from_excel': 0,
            'tables_with_variables': 0,
            'tables_without_variables': 0,
            'dimensional_tags_extracted': 0,
            'survey_type_distribution': defaultdict(int),
            'geography_vintage_distribution': defaultdict(int),
            'multi_survey_concepts': 0
        }
    
    def load_refactored_canonical_variables(self) -> Dict:
        """Load concept-based canonical variables"""
        logger.info(f"Loading refactored canonical variables from {self.canonical_path}")
        
        with open(self.canonical_path, 'r') as f:
            data = json.load(f)
        
        # Extract concepts from the refactored structure
        concepts = data.get('concepts', {})
        if not concepts:
            # Fallback to root level if structure is different
            concepts = {k: v for k, v in data.items() if k != 'metadata'}
        
        logger.info(f"Loaded {len(concepts)} concept-based variables")
        return concepts
    
    def load_table_metadata(self) -> pd.DataFrame:
        """Load official table metadata from DataProductList.xlsx"""
        logger.info(f"Loading table metadata from {self.table_list_path}")
        
        df = pd.read_excel(self.table_list_path)
        df.columns = df.columns.str.strip()
        
        # Handle column mapping
        col_mapping = {}
        for col in df.columns:
            if 'Table ID' in col:
                col_mapping[col] = 'table_id'
            elif 'Table Title' in col:
                col_mapping[col] = 'title'
            elif 'Table Universe' in col:
                col_mapping[col] = 'universe'
            elif 'Data Product Type' in col:
                col_mapping[col] = 'data_product_type'
            elif 'Year' in col:
                col_mapping[col] = 'year'
            elif '1-Year' in col:
                col_mapping[col] = 'geo_1yr'
            elif '5-Year' in col:
                col_mapping[col] = 'geo_5yr'
        
        df = df.rename(columns=col_mapping)
        
        # Filter for ACS tables
        acs_tables = df[df['table_id'].str.match(r'^[BCSDP]\d+[A-Z]*$', na=False)]
        
        self.stats['tables_from_excel'] = len(acs_tables)
        logger.info(f"Loaded {len(acs_tables)} ACS tables from Excel")
        
        return acs_tables
    
    def parse_survey_instances(self, concept_data: Dict) -> List[SurveyInstance]:
        """Parse survey instances from concept-based structure"""
        instances = []
        
        for instance_data in concept_data.get('instances', []):
            # Parse geography restrictions
            geo_restrictions = instance_data.get('geography_restrictions', [])
            if isinstance(geo_restrictions, str):
                geo_restrictions = [geo_restrictions]
            
            instance = SurveyInstance(
                dataset=instance_data.get('dataset', ''),
                survey_type=instance_data.get('survey_type', ''),
                year=instance_data.get('year', ''),
                geography_vintage=instance_data.get('geography_vintage', ''),
                geography_restrictions=geo_restrictions,
                sample_characteristics=instance_data.get('sample_characteristics', ''),
                status=instance_data.get('status', 'active')
            )
            
            instances.append(instance)
            
            # Track statistics
            self.stats['survey_instances_processed'] += 1
            self.stats['survey_type_distribution'][instance.survey_type] += 1
            self.stats['geography_vintage_distribution'][instance.geography_vintage] += 1
        
        return instances
    
    def create_concept_variable(self, variable_id: str, concept_data: Dict) -> ConceptVariable:
        """Create ConceptVariable from concept-based structure"""
        
        # Parse survey instances
        survey_instances = self.parse_survey_instances(concept_data)
        
        # Extract metadata
        available_surveys = concept_data.get('available_surveys', [])
        geography_coverage = concept_data.get('geography_coverage', {})
        primary_instance = concept_data.get('primary_instance', '')
        
        # Extract dimensional tags
        dimensional_tags = self.tagger.extract_tags(
            concept_data.get('label', ''),
            concept_data.get('concept', '')
        )
        
        if dimensional_tags:
            self.stats['dimensional_tags_extracted'] += 1
        
        # Track multi-survey concepts
        if len(available_surveys) > 1:
            self.stats['multi_survey_concepts'] += 1
        
        return ConceptVariable(
            variable_id=variable_id,
            concept=concept_data.get('concept', ''),
            label=concept_data.get('label', ''),
            table_id=concept_data.get('table_id', variable_id.split('_')[0]),
            survey_instances=survey_instances,
            available_surveys=available_surveys,
            geography_coverage=geography_coverage,
            primary_instance=primary_instance,
            dimensional_tags=dimensional_tags,
            category_weights=concept_data.get('category_weights_linear', {}),
            is_estimate=variable_id.endswith('_E'),
            is_margin_error=variable_id.endswith('_M')
        )
    
    def group_variables_by_table(self, concepts: Dict) -> Dict[str, List[ConceptVariable]]:
        """Group concept-based variables by table"""
        logger.info("Grouping concept-based variables by table...")
        
        tables = defaultdict(list)
        
        for variable_id, concept_data in concepts.items():
            # Skip if missing essential data
            if not concept_data.get('concept') or not concept_data.get('label'):
                continue
            
            concept_var = self.create_concept_variable(variable_id, concept_data)
            tables[concept_var.table_id].append(concept_var)
            
            self.stats['concepts_processed'] += 1
        
        logger.info(f"Grouped {self.stats['concepts_processed']} concepts into {len(tables)} tables")
        return dict(tables)
    
    def parse_geography_levels(self, geo_1yr: str, geo_5yr: str) -> List[str]:
        """Parse geography levels from restriction strings using official Census Summary Level Codes
        
        Source: https://www.census.gov/programs-surveys/geography/technical-documentation/naming-convention/cartographic-boundary-file/carto-boundary-summary-level.html
        """
        geo_levels = set()
        
        # Complete Official Census Summary Level Code mappings
        # Source: Census Bureau Geographic Summary Level Codes (see URL above)
        official_geo_mappings = {
            'region': ['020'],  # Region
            'division': ['030'],  # Division
            'state': ['040'],   # State
            'county': ['050'],  # State-County
            'county_subdivision': ['060'],  # State-County-County Subdivision
            'subminor_civil_division': ['067'],  # State-County-County Subdivision-Subminor Civil Division
            'tract': ['140'],   # State-County-Census Tract
            'block_group': ['150'],  # State-County-Census Tract-Block Group
            'place': ['160'],   # State-Place
            'consolidated_city': ['170'],  # State-Consolidated City
            'alaska_native_regional_corp': ['230'],  # State-Alaska Native Regional Corporation
            'american_indian_area': ['250'],  # American Indian Area/Alaska Native Area/Hawaiian Home Land
            'tribal_subdivision': ['251'],  # American Indian Area-Tribal Subdivision/Remainder
            'reservation_statistical': ['252'],  # American Indian Area/Alaska Native Area (Reservation or Statistical Entity Only)
            'trust_land_hawaiian_home': ['254'],  # American Indian Area (Off-Reservation Trust Land Only)/Hawaiian Home Land
            'tribal_census_tract': ['256'],  # American Indian Area-Tribal Census Tract
            'tribal_block_group': ['258'],  # American Indian Area-Tribal Census Tract-Tribal Block Group
            'metro_micro_statistical_area': ['310'],  # Metropolitan Statistical Area/Micropolitan Statistical Area
            'metro_division': ['314'],  # Metropolitan Statistical Area-Metropolitan Division
            'combined_statistical_area': ['330'],  # Combined Statistical Area
            'combined_statistical_metro': ['332'],  # Combined Statistical Area-Metropolitan Statistical Area/Micropolitan Statistical Area
            'combined_necta': ['335'],  # Combined New England City and Town Area
            'combined_necta_necta': ['337'],  # Combined New England City and Town Area-New England City and Town Area
            'necta': ['350'],  # New England City and Town Area
            'necta_principal_city': ['352'],  # New England City and Town Area-State-Principal City
            'necta_division': ['355'],  # New England City and Town Area (NECTA)-NECTA Division
            'state_necta_principal_city': ['361'],  # State-New England City and Town Area-Principal City
            'congressional_district': ['500'],  # State-Congressional District (111th)
            'state_legislative_upper': ['610'],  # State-State Legislative District (Upper Chamber)
            'state_legislative_lower': ['620'],  # State-State Legislative District (Lower Chamber)
            'voting_district': ['700'],  # State-County-Voting District/Remainder
            'zcta': ['860'],    # 5-Digit ZIP Code Tabulation Area
            'school_district_elementary': ['950'],  # State-School District (Elementary)/Remainder
            'school_district_secondary': ['960'],  # State-School District (Secondary)/Remainder
            'school_district_unified': ['970'],  # State-School District (Unified)/Remainder
        }
        
        restriction_text = f"{geo_1yr or ''} {geo_5yr or ''}".lower()
        
        # Parse using official summary level codes and common text patterns
        for geo_level, codes in official_geo_mappings.items():
            # Check for summary level codes
            if any(code in restriction_text for code in codes):
                geo_levels.add(geo_level)
            
            # Also check for common text patterns (backup for human-readable text)
            text_patterns = {
                'state': ['state'],
                'county': ['county'],
                'place': ['place', 'city'],
                'tract': ['tract'],
                'block_group': ['block group'],
                'zcta': ['zip', 'zcta'],
                'metro': ['metro', 'msa', 'micropolitan'],
                'congressional_district': ['congressional'],
                'county_subdivision': ['subdivision'],
            }
            
            if geo_level in text_patterns:
                if any(pattern in restriction_text for pattern in text_patterns[geo_level]):
                    geo_levels.add(geo_level)
        
        # Conservative default: if nothing specified, assume basic levels
        # (Based on typical ACS table availability, not assumption)
        if not geo_levels:
            geo_levels = {'state', 'county', 'place'}
        
        return sorted(list(geo_levels))
    
    def create_survey_availability_matrix(self, variables: List[ConceptVariable]) -> Dict[str, Dict[str, List[str]]]:
        """Create survey availability matrix for table"""
        availability = defaultdict(lambda: defaultdict(list))
        
        for var in variables:
            for instance in var.survey_instances:
                survey_type = instance.survey_type
                for geo_restriction in instance.geography_restrictions:
                    availability[survey_type][geo_restriction].append(var.variable_id)
        
        return dict(availability)
    
    def join_table_data(self, table_metadata: pd.DataFrame,
                       table_variables: Dict[str, List[ConceptVariable]]) -> List[TableCatalog]:
        """JOIN table metadata with concept-based variable data"""
        logger.info("Joining table metadata with concept-based variables...")
        
        catalogs = []
        
        for _, table_row in table_metadata.iterrows():
            table_id = table_row['table_id']
            variables = table_variables.get(table_id, [])
            
            if not variables:
                self.stats['tables_without_variables'] += 1
                continue
            
            self.stats['tables_with_variables'] += 1
            
            # Get primary concept
            concepts = [v.concept for v in variables if v.concept]
            primary_concept = Counter(concepts).most_common(1)[0][0] if concepts else table_row['title']
            
            # Determine survey programs from variables
            all_surveys = set()
            for var in variables:
                all_surveys.update(var.available_surveys)
            
            survey_programs = []
            if '1yr' in all_surveys:
                survey_programs.append('acs1')
            if '5yr' in all_surveys:
                survey_programs.append('acs5')
            
            # Parse geography levels
            geography_levels = self.parse_geography_levels(
                table_row.get('geo_1yr', ''),
                table_row.get('geo_5yr', '')
            )
            
            # Find primary variable
            primary_variable = None
            for var in variables:
                if var.variable_id.endswith('_001E') and var.is_estimate:
                    primary_variable = var.variable_id
                    break
            
            # Collect dimensional categories
            dimensional_categories = set()
            for var in variables:
                dimensional_categories.update(var.dimensional_tags.keys())
            
            # Create survey availability matrix
            survey_availability = self.create_survey_availability_matrix(variables)
            
            # Generate enhanced statistical notes
            statistical_notes = []
            if dimensional_categories:
                dim_list = sorted(dimensional_categories)
                statistical_notes.append(f"Available by: {', '.join(dim_list)}")
            
            # Survey-specific notes
            if len(survey_programs) == 1:
                if 'acs1' in survey_programs:
                    statistical_notes.append("ACS 1-year only: Current data, limited geography")
                else:
                    statistical_notes.append("ACS 5-year only: All geographies, averaged data")
            else:
                statistical_notes.append("Available in both ACS 1-year and 5-year surveys")
            
            # Geography availability notes
            if 'tract' in geography_levels:
                statistical_notes.append("Available at census tract level")
            if 'block_group' in geography_levels:
                statistical_notes.append("Available at block group level")
            
            # Margin of error availability
            if any(v.is_margin_error for v in variables):
                statistical_notes.append("Includes margins of error for reliability assessment")
            
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
                methodology_topics=[primary_concept.lower()],
                statistical_notes=statistical_notes,
                survey_availability=survey_availability
            )
            
            catalogs.append(catalog)
        
        logger.info(f"Successfully joined {len(catalogs)} table catalogs")
        logger.info(f"Tables with variables: {self.stats['tables_with_variables']}")
        logger.info(f"Tables without variables: {self.stats['tables_without_variables']}")
        
        return catalogs
    
    def create_table_embeddings(self, catalogs: List[TableCatalog]) -> Tuple[np.ndarray, List[str]]:
        """Create clean embeddings for concept-based tables"""
        logger.info("Creating table embeddings for concept-based structure...")
        
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        embedding_texts = []
        table_ids = []
        
        for catalog in catalogs:
            # Create clean embedding text using official metadata
            text_parts = [
                catalog.title,
                catalog.universe,
                catalog.concept
            ]
            
            # Add key dimensional info (limited to avoid noise)
            if catalog.dimensional_categories:
                dims = sorted(list(catalog.dimensional_categories))[:3]
                text_parts.append(f"by {', '.join(dims)}")
            
            # Add survey availability context
            if len(catalog.survey_programs) == 1:
                if 'acs1' in catalog.survey_programs:
                    text_parts.append("1-year survey")
                else:
                    text_parts.append("5-year survey")
            
            # Clean embedding text
            embedding_text = '. '.join(filter(None, text_parts))
            embedding_text = re.sub(r'\b(Estimate!!|Margin of Error!!)\b', '', embedding_text)
            embedding_text = re.sub(r'\s+', ' ', embedding_text).strip()
            
            embedding_texts.append(embedding_text)
            table_ids.append(catalog.table_id)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(embedding_texts, show_progress_bar=True)
        
        logger.info(f"Generated clean embeddings for {len(embeddings)} tables")
        return embeddings, table_ids
    
    def save_catalog(self, catalogs: List[TableCatalog], embeddings: np.ndarray,
                    table_ids: List[str], output_dir: str = "table-catalog"):
        """Save concept-based catalog and embeddings"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format manually (avoid asdict issues)
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
        
        # Save table catalog
        catalog_file = output_path / "table_catalog.json"
        with open(catalog_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model_version': '3.0_concept_based',
                    'total_tables': len(catalogs),
                    'total_concepts': sum(c.variable_count for c in catalogs),
                    'extraction_stats': dict(self.stats),
                    'model_used': 'sentence-transformers/all-mpnet-base-v2',
                    'data_sources': {
                        'table_metadata': str(self.table_list_path),
                        'variable_details': str(self.canonical_path)
                    },
                    'key_improvements': [
                        'Eliminated duplicate variables',
                        'Survey-aware metadata',
                        'Geography intelligence',
                        'Concept-based structure'
                    ]
                },
                'tables': catalog_data
            }, f, indent=2)
        
        logger.info(f"Saved concept-based table catalog to {catalog_file}")
        
        # Save FAISS embeddings
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
        
        logger.info(f"Saved FAISS embeddings to {faiss_file}")
        
        # Save dimensional vocabulary
        vocab = self.extract_dimensional_vocabulary(catalogs)
        vocab_file = output_path / "dimensional_vocabulary.json"
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent=2)
        
        logger.info(f"Saved dimensional vocabulary to {vocab_file}")
    
    def extract_dimensional_vocabulary(self, catalogs: List[TableCatalog]) -> Dict:
        """Extract dimensional vocabulary from concept-based tables"""
        vocabulary = defaultdict(Counter)
        
        for catalog in catalogs:
            for var in catalog.variables:
                for dim_type, dim_value in var.dimensional_tags.items():
                    vocabulary[dim_type][dim_value] += 1
        
        return {dim_type: dict(values) for dim_type, values in vocabulary.items()}
    
    def print_statistics(self, catalogs: List[TableCatalog]):
        """Print enhanced statistics for concept-based extraction"""
        logger.info("=== CONCEPT-BASED TABLE CATALOG COMPLETE ===")
        logger.info(f"📊 Concepts processed: {self.stats['concepts_processed']:,}")
        logger.info(f"📊 Survey instances: {self.stats['survey_instances_processed']:,}")
        logger.info(f"📊 Tables from Excel: {self.stats['tables_from_excel']:,}")
        logger.info(f"✅ Tables with variables: {self.stats['tables_with_variables']:,}")
        logger.info(f"❌ Tables without variables: {self.stats['tables_without_variables']:,}")
        logger.info(f"🏷️  Dimensional tags extracted: {self.stats['dimensional_tags_extracted']:,}")
        logger.info(f"🔀 Multi-survey concepts: {self.stats['multi_survey_concepts']:,}")
        
        # Survey type distribution
        logger.info(f"\n📅 Survey type distribution:")
        for survey_type, count in self.stats['survey_type_distribution'].items():
            logger.info(f"   {survey_type}: {count:,} instances")
        
        # Geography vintage distribution
        logger.info(f"\n🗺️  Geography vintage distribution:")
        for vintage, count in self.stats['geography_vintage_distribution'].items():
            logger.info(f"   {vintage}: {count:,} instances")
        
        # Table statistics
        variable_counts = [c.variable_count for c in catalogs]
        logger.info(f"\n📈 Average concepts per table: {np.mean(variable_counts):.1f}")
        logger.info(f"📈 Max concepts in table: {max(variable_counts)}")
        
        # Survey availability
        both_surveys = sum(1 for c in catalogs if len(c.survey_programs) == 2)
        acs1_only = sum(1 for c in catalogs if c.survey_programs == ['acs1'])
        acs5_only = sum(1 for c in catalogs if c.survey_programs == ['acs5'])
        
        logger.info(f"\n📊 Survey availability:")
        logger.info(f"   Both ACS1 & ACS5: {both_surveys:,}")
        logger.info(f"   ACS1 only: {acs1_only:,}")
        logger.info(f"   ACS5 only: {acs5_only:,}")
        
        # Sample tables
        logger.info(f"\n📋 Sample concept-based tables:")
        for i, catalog in enumerate(catalogs[:3]):
            logger.info(f"   {i+1}. {catalog.table_id}: {catalog.title}")
            logger.info(f"      Universe: {catalog.universe}")
            logger.info(f"      Concepts: {catalog.variable_count}")
            logger.info(f"      Surveys: {catalog.survey_programs}")
            logger.info(f"      Dimensions: {sorted(catalog.dimensional_categories)}")

def main():
    """Main concept-based extraction process"""
    logger.info("🚀 Starting Concept-Based Table Catalog Extraction...")
    logger.info("📦 Using refactored canonical variables structure")
    logger.info("🎯 Eliminating duplicate variables and enabling survey intelligence")
    
    extractor = ConceptBasedTableCatalogExtractor()
    
    # Load concept-based data
    table_metadata = extractor.load_table_metadata()
    concepts = extractor.load_refactored_canonical_variables()
    
    # Group concepts by table
    table_variables = extractor.group_variables_by_table(concepts)
    
    # JOIN with enhanced survey awareness
    catalogs = extractor.join_table_data(table_metadata, table_variables)
    
    # Create clean embeddings
    embeddings, table_ids = extractor.create_table_embeddings(catalogs)
    
    # Save everything
    extractor.save_catalog(catalogs, embeddings, table_ids)
    
    # Print comprehensive statistics
    extractor.print_statistics(catalogs)
    
    logger.info("✅ Concept-based table catalog extraction complete!")
    logger.info("🎯 Key improvements delivered:")
    logger.info("   - Eliminated duplicate variables")
    logger.info("   - Survey-aware metadata")
    logger.info("   - Geography intelligence")
    logger.info("   - Clean, concept-based structure")
    logger.info("📁 Output files:")
    logger.info("   - table-catalog/table_catalog.json")
    logger.info("   - table-catalog/table_embeddings.faiss")
    logger.info("   - table-catalog/dimensional_vocabulary.json")
    logger.info("🎯 Ready for concept-based search system!")

if __name__ == "__main__":
    main()
