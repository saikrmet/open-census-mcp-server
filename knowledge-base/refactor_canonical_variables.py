#!/usr/bin/env python3
"""
Canonical Variables Refactor - Concept-Based Data Model

Transforms the current duplicated canonical variables structure into a proper
concept-based model following Census statistical ontology best practices.

Structure:
- Concept level: Single definition per variable concept
- Instance level: Multiple survey/year combinations per concept

Usage:
    python refactor_canonical_variables.py
    
Output:
    - canonical_variables_refactored.json: New concept-based structure
    - refactor_analysis.json: Analysis of consolidation process
"""

import json
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SurveyInstance:
    """Single survey instance of a variable concept"""
    dataset: str  # acs2023_1yr, acs2023_5yr
    temporal_id: str  # Original temporal ID
    survey_program: str  # acs
    survey_type: str  # 1yr, 5yr
    year: str  # 2023
    geography_vintage: str  # 2010, 2020 (Census geography boundaries used)
    status: str  # active, introduced, discontinued, definition_changed
    survey_context: str
    flavor_characteristics: str
    methodological_notes: str
    geography_restrictions: List[str]
    sample_characteristics: str
    quality_tier: str
    agreement_score: float
    processing_cost: float
    # Future extension fields
    definition_version: Optional[str] = None  # For tracking definition changes
    notes: Optional[str] = None  # Human-readable change notes

@dataclass
class VariableConcept:
    """Consolidated variable concept with multiple survey instances"""
    variable_id: str  # B08303_001E
    concept: str  # Travel Time to Work
    label: str  # Estimate!!Total:
    predicate_type: str  # int
    group: str  # B08303
    table_id: str  # B08303
    table_family: str  # B08
    
    # Versioning and lineage (future-ready)
    version: int  # 1, 2, 3... for definition changes
    valid_from: Optional[str]  # First year this definition is valid
    valid_to: Optional[str]  # Last year (None = current)
    
    # Consolidated metadata (single copy per version)
    enrichment_text: str
    category_weights_linear: Dict[str, float]
    complexity: str
    
    # Survey instances across years
    instances: List[SurveyInstance]
    
    # Derived metadata
    available_surveys: List[str]  # [acs1, acs5]
    available_years: List[str]  # [2023, 2024, ...]
    geography_coverage: Dict[str, List[str]]  # {acs1: [state, county], acs5: [state, county, place, tract]}
    primary_instance: str  # Preferred instance (usually latest acs5)
    
    # Future extension fields for lineage tracking
    replaces: Optional[List[str]] = None  # Variables this replaced
    replaced_by: Optional[List[str]] = None  # Variables that replaced this
    related_variables: Optional[List[str]] = None  # Related/similar variables
    comparability_notes: Optional[str] = None  # Human notes about comparability

class CanonicalVariablesRefactor:
    """Refactors canonical variables from duplicated to concept-based structure"""
    
    def __init__(self, input_path: str = "source-docs/canonical_variables.json"):
        self.input_path = Path(input_path)
        self.stats = {
            'original_variables': 0,
            'unique_concepts': 0,
            'consolidation_ratio': 0.0,
            'survey_distributions': defaultdict(int),
            'quality_issues': [],
            'auto_generated_concepts': 0
        }
    
    def load_original_variables(self) -> Dict:
        """Load the original canonical variables"""
        logger.info(f"Loading original canonical variables from {self.input_path}")
        
        with open(self.input_path) as f:
            data = json.load(f)
        
        variables = data.get('variables', data)
        self.stats['original_variables'] = len(variables)
        
        logger.info(f"Loaded {len(variables)} original variable entries")
        return variables
    
    def extract_variable_id(self, temporal_id: str) -> str:
        """Extract variable ID from temporal ID"""
        # acs_5yr_2023_B08303_001E -> B08303_001E
        parts = temporal_id.split('_')
        if len(parts) >= 4:
            return '_'.join(parts[3:])  # B08303_001E
        return temporal_id
    
    def parse_temporal_id(self, temporal_id: str) -> Tuple[str, str, str]:
        """Parse temporal ID into components"""
        # acs_5yr_2023_B08303_001E -> (acs, 5yr, 2023)
        parts = temporal_id.split('_')
        if len(parts) >= 3:
            survey_program = parts[0]  # acs
            survey_type = parts[1]  # 5yr
            year = parts[2]  # 2023
            return survey_program, survey_type, year
        return 'unknown', 'unknown', 'unknown'
    
    def auto_generate_geographic_concept(self, variable_id: str, label: str) -> Optional[str]:
        """Auto-generate concept for geographic identifier variables"""
        if not label:
            return None
        
        label_lower = label.lower()
        
        # Geographic identifier patterns
        geo_mappings = {
            'congressional district': 'Geographic Identifier - Congressional District',
            'urban area': 'Geographic Identifier - Urban Area',
            'metropolitan division': 'Geographic Identifier - Metropolitan Division',
            'summary level': 'Geographic Identifier - Summary Level Code',
            'state': 'Geographic Identifier - State',
            'combined statistical area': 'Geographic Identifier - Combined Statistical Area',
            'principal city': 'Geographic Identifier - Principal City',
            'block group': 'Geographic Identifier - Block Group',
            'school district': 'Geographic Identifier - School District',
            'tribal census tract': 'Geographic Identifier - Tribal Census Tract',
            'county': 'Geographic Identifier - County',
            'place': 'Geographic Identifier - Place',
            'tract': 'Geographic Identifier - Census Tract',
            'zip code': 'Geographic Identifier - ZIP Code Tabulation Area',
            'metro': 'Geographic Identifier - Metropolitan Statistical Area'
        }
        
        # Handle special cases by variable ID
        if variable_id == 'NATION':
            return 'Geographic Identifier - National Level'
        elif variable_id == 'CONCIT':
            return 'Geographic Identifier - Consolidated City'
        
        # Check for geographic patterns
        for pattern, concept in geo_mappings.items():
            if pattern in label_lower:
                return concept
        
        # Default geographic concept if it looks like a geo identifier
        geo_indicators = ['geographic', 'geography', 'geo', 'boundary', 'area', 'region', 'division']
        if any(indicator in label_lower for indicator in geo_indicators):
            return f"Geographic Identifier - {label}"
        
        return None
    def determine_geography_vintage(self, year: str) -> str:
        """Determine geography vintage based on year"""
        # Critical boundary change: 2020 Census boundaries vs 2010
        year_int = int(year)
        if year_int >= 2020:
            return "2020"  # 2020+ uses 2020 Census boundaries
        else:
            return "2010"  # 2009-2019 uses 2010 Census boundaries
    def determine_geography_restrictions(self, survey_type: str, enrichment_text: str) -> List[str]:
        """Determine geography restrictions from survey type and enrichment"""
        if survey_type == '1yr':
            # ACS1 limited to areas ≥65k population
            return ['state', 'county', 'place_65k+', 'metro']
        elif survey_type == '5yr':
            # ACS5 available for all geography levels
            return ['nation', 'state', 'county', 'place', 'tract', 'block_group', 'zcta']
        else:
            return ['state', 'county']  # Default
    
    def consolidate_by_concept(self, original_variables: Dict) -> Dict[str, VariableConcept]:
        """Consolidate variables by concept, combining survey instances"""
        logger.info("Consolidating variables by concept...")
        
        # Group by variable_id (e.g., B08303_001E)
        concept_groups = defaultdict(list)
        
        for temporal_id, var_data in original_variables.items():
            variable_id = self.extract_variable_id(temporal_id)
            
            # Handle missing concepts - try to auto-generate for geographic identifiers
            concept = var_data.get('concept', '')
            label = var_data.get('label', '')
            
            if not concept and label:
                # Try to auto-generate concept for geographic identifiers
                auto_concept = self.auto_generate_geographic_concept(variable_id, label)
                if auto_concept:
                    concept = auto_concept
                    self.stats['auto_generated_concepts'] += 1
                    logger.debug(f"Auto-generated concept for {variable_id}: {auto_concept}")
            
            # Skip only if we still have no concept and no label
            if not concept or not label:
                self.stats['quality_issues'].append(f"Missing concept/label: {temporal_id}")
                continue
            
            # Update the var_data with auto-generated concept if needed
            if not var_data.get('concept') and concept:
                var_data = var_data.copy()  # Don't modify original
                var_data['concept'] = concept
            
            concept_groups[variable_id].append((temporal_id, var_data))
        
        logger.info(f"Found {len(concept_groups)} unique variable concepts")
        self.stats['unique_concepts'] = len(concept_groups)
        self.stats['consolidation_ratio'] = self.stats['unique_concepts'] / self.stats['original_variables']
        
        # Build consolidated concepts
        consolidated = {}
        
        for variable_id, instances_data in concept_groups.items():
            concept = self._build_concept_from_instances(variable_id, instances_data)
            if concept:
                consolidated[variable_id] = concept
        
        logger.info(f"Successfully consolidated {len(consolidated)} variable concepts")
        return consolidated
    
    def _build_concept_from_instances(self, variable_id: str, instances_data: List[Tuple[str, Dict]]) -> Optional[VariableConcept]:
        """Build a VariableConcept from multiple survey instances"""
        
        if not instances_data:
            return None
        
        # Use first instance as template for shared metadata
        template_temporal_id, template_data = instances_data[0]
        
        # Build survey instances
        survey_instances = []
        available_surveys = set()
        available_years = set()
        geography_coverage = {}
        
        for temporal_id, var_data in instances_data:
            survey_program, survey_type, year = self.parse_temporal_id(temporal_id)
            
            # Create dataset identifier
            dataset = f"{survey_program}{year}_{survey_type}"
            
            # Determine geography vintage and restrictions
            geography_vintage = self.determine_geography_vintage(year)
            geography_restrictions = self.determine_geography_restrictions(
                survey_type,
                var_data.get('enrichment_text', '')
            )
            
            # Sample characteristics
            if survey_type == '1yr':
                sample_chars = "Smaller sample, more current (12 months), limited geography"
            elif survey_type == '5yr':
                sample_chars = "Larger sample, less current (60-month average), all geographies"
            else:
                sample_chars = "Unknown sample characteristics"
            
            instance = SurveyInstance(
                dataset=dataset,
                temporal_id=temporal_id,
                survey_program=survey_program,
                survey_type=survey_type,
                year=year,
                geography_vintage=geography_vintage,  # Critical for boundary comparability
                status='active',  # Default to active, future updates will set proper status
                survey_context=var_data.get('survey_context', ''),
                flavor_characteristics=var_data.get('flavor_characteristics', ''),
                methodological_notes=var_data.get('methodological_notes', ''),
                geography_restrictions=geography_restrictions,
                sample_characteristics=sample_chars,
                quality_tier=var_data.get('quality_tier', ''),
                agreement_score=var_data.get('agreement_score', 0.0),
                processing_cost=var_data.get('processing_cost', 0.0),
                definition_version=None,  # Future: track definition changes
                notes=None  # Future: human-readable change notes
            )
            
            survey_instances.append(instance)
            available_surveys.add(survey_type)
            available_years.add(year)
            geography_coverage[survey_type] = geography_restrictions
            
            # Track survey distribution
            self.stats['survey_distributions'][survey_type] += 1
        
        # Choose primary instance (prefer latest 5yr for broader geography coverage)
        primary_instance = None
        latest_year = max(available_years) if available_years else "2023"
        
        if '5yr' in available_surveys:
            primary_instance = f"{template_data.get('survey_program', 'acs')}{latest_year}_5yr"
        else:
            primary_instance = survey_instances[0].dataset
        
        # Create consolidated concept
        concept = VariableConcept(
            variable_id=variable_id,
            concept=template_data.get('concept', ''),
            label=template_data.get('label', ''),
            predicate_type=template_data.get('predicateType', ''),
            group=template_data.get('group', ''),
            table_id=template_data.get('table_id', ''),
            table_family=template_data.get('table_family', ''),
            
            # Versioning (future-ready)
            version=1,  # Start with version 1
            valid_from=min(available_years) if available_years else "2023",
            valid_to=None,  # None = current/active
            
            # Single copy of rich metadata
            enrichment_text=template_data.get('enrichment_text', ''),
            category_weights_linear=template_data.get('category_weights_linear', {}),
            complexity=template_data.get('complexity', ''),
            
            # Survey instances
            instances=survey_instances,
            
            # Derived metadata
            available_surveys=sorted(list(available_surveys)),
            available_years=sorted(list(available_years)),
            geography_coverage=geography_coverage,
            primary_instance=primary_instance,
            
            # Lineage fields (future extension)
            replaces=None,
            replaced_by=None,
            related_variables=None,
            comparability_notes=None
        )
        
        return concept
    
    def save_refactored_variables(self, consolidated: Dict[str, VariableConcept],
                                output_path: str = "canonical_variables_refactored.json"):
        """Save refactored canonical variables"""
        output_file = Path(output_path)
        
        # Convert to serializable format
        serializable_data = {}
        for variable_id, concept in consolidated.items():
            concept_dict = asdict(concept)
            serializable_data[variable_id] = concept_dict
        
        # Create output structure
        output_data = {
            'metadata': {
                'model_version': '2.0_concept_based',
                'generated_from': str(self.input_path),
                'consolidation_stats': self.stats,
                'description': 'Future-ready concept-based canonical variables with temporal extension support',
                'structure': {
                    'concept_level': 'Single definition per variable concept with versioning support',
                    'instance_level': 'Multiple survey/year combinations per concept with status tracking',
                    'temporal_ready': 'Supports definition changes, discontinuities, and lineage tracking',
                    'key_benefits': [
                        'Eliminates search result duplicates',
                        'Single copy of enrichment metadata per version',
                        'Proper statistical entity modeling',
                        'Geographic restrictions per survey type',
                        'Ready for 2024+ data integration',
                        'Supports variable lineage and definition changes'
                    ]
                }
            },
            'concepts': serializable_data
        }
        
        # Save refactored file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"💾 Saved refactored canonical variables to {output_file}")
        
        # Save analysis
        analysis_file = output_file.parent / "refactor_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump({
                'consolidation_analysis': self.stats,
                'examples': self._generate_examples(consolidated)
            }, f, indent=2)
        
        logger.info(f"📊 Saved refactor analysis to {analysis_file}")
    
    def _generate_examples(self, consolidated: Dict[str, VariableConcept]) -> Dict:
        """Generate examples for analysis"""
        examples = {}
        
        # Find example with multiple survey types
        for variable_id, concept in list(consolidated.items())[:5]:
            examples[variable_id] = {
                'concept': concept.concept,
                'available_surveys': concept.available_surveys,
                'geography_coverage': concept.geography_coverage,
                'instance_count': len(concept.instances),
                'primary_instance': concept.primary_instance
            }
        
        return examples
    
    def print_consolidation_stats(self, consolidated: Dict[str, VariableConcept]):
        """Print detailed consolidation statistics"""
        logger.info("=== CANONICAL VARIABLES REFACTOR COMPLETE ===")
        logger.info(f"📊 Original variables: {self.stats['original_variables']:,}")
        logger.info(f"📊 Unique concepts: {self.stats['unique_concepts']:,}")
        logger.info(f"📊 Consolidation ratio: {self.stats['consolidation_ratio']:.1%}")
        logger.info(f"📊 Space savings: {(1 - self.stats['consolidation_ratio']):.1%}")
        
        # Survey distribution
        logger.info(f"\n📅 Survey type distribution:")
        for survey_type, count in self.stats['survey_distributions'].items():
            logger.info(f"   {survey_type}: {count:,} instances")
        
        # Quality issues
        if self.stats['quality_issues']:
            logger.info(f"\n⚠️  Quality issues found: {len(self.stats['quality_issues'])}")
            for issue in self.stats['quality_issues'][:5]:
                logger.info(f"   {issue}")
            if len(self.stats['quality_issues']) > 5:
                logger.info(f"   ... and {len(self.stats['quality_issues']) - 5} more")
        
        # Auto-generated concepts
        if self.stats['auto_generated_concepts'] > 0:
            logger.info(f"\n🔧 Auto-generated concepts: {self.stats['auto_generated_concepts']}")
            logger.info("   Geographic identifiers now properly included")
        
        # Examples of multi-survey concepts
        multi_survey = [c for c in consolidated.values() if len(c.available_surveys) > 1]
        logger.info(f"\n🔀 Concepts with multiple surveys: {len(multi_survey):,}")
        
        # Sample concept
        if consolidated:
            sample_id = list(consolidated.keys())[0]
            sample = consolidated[sample_id]
            logger.info(f"\n📋 Sample concept: {sample_id}")
            logger.info(f"   Concept: {sample.concept}")
            logger.info(f"   Available surveys: {sample.available_surveys}")
            logger.info(f"   Geography coverage: {sample.geography_coverage}")
            logger.info(f"   Instances: {len(sample.instances)}")

def main():
    """Main refactoring process"""
    logger.info("🚀 Starting Future-Ready Canonical Variables Refactor...")
    logger.info("📅 Current scope: 2023 data consolidation")
    logger.info("🔮 Future ready: 2024+ integration, definition tracking, lineage support")
    
    refactor = CanonicalVariablesRefactor()
    
    # Load original variables
    original = refactor.load_original_variables()
    
    # Consolidate by concept
    consolidated = refactor.consolidate_by_concept(original)
    
    # Save refactored structure
    refactor.save_refactored_variables(consolidated)
    
    # Print statistics
    refactor.print_consolidation_stats(consolidated)
    
    logger.info("✅ Future-ready canonical variables refactor complete!")
    logger.info("📁 Output files:")
    logger.info("   - canonical_variables_refactored.json")
    logger.info("   - refactor_analysis.json")
    logger.info("🎯 Ready to rebuild search systems with concept-based structure!")
    logger.info("📅 Ready to integrate 2024 data when available!")
    logger.info("🔄 Structure supports definition changes and variable lineage tracking!")

if __name__ == "__main__":
    main()
