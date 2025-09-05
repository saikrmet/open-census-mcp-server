#!/usr/bin/env python3
"""
Table Resolver - Convert table IDs and concepts to Census variables

Handles:
1. Direct table IDs (B19013 → list of variables)
2. Natural language concepts (income distribution → B19013 → variables) 
3. Table metadata and variable metadata lookup
4. Fallback to hardcoded common tables if catalog unavailable

Architecture: Table catalog first, hardcoded fallbacks second
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TableResolver:
    """Resolve table IDs and concepts to Census variables"""
    
    def __init__(self, knowledge_base_dir: Optional[str] = None):
        self.knowledge_base_dir = knowledge_base_dir or self._find_knowledge_base()
        self.table_catalog = None
        self.variable_metadata = None
        
        # Try to load table catalog
        self._load_table_catalog()
        
        # Initialize concept mappings (fallback)
        self._init_concept_mappings()
        
        # Initialize common table definitions (fallback)
        self._init_common_tables()
        
        logger.info(f"✅ TableResolver initialized")
        logger.info(f"   Catalog loaded: {'Yes' if self.table_catalog else 'No (using fallbacks)'}")
        logger.info(f"   Knowledge base: {self.knowledge_base_dir}")
    
    def _find_knowledge_base(self) -> Optional[str]:
        """Auto-detect knowledge base directory"""
        
        possible_paths = [
            Path(__file__).parent.parent.parent / "knowledge-base",
            Path(__file__).parent.parent / "knowledge-base",
            Path.cwd() / "knowledge-base"
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                return str(path)
        
        return None
    
    def _load_table_catalog(self):
        """Load table catalog from knowledge base"""
        
        if not self.knowledge_base_dir:
            logger.warning("⚠️ No knowledge base directory found - using fallbacks only")
            return
        
        catalog_paths = [
            Path(self.knowledge_base_dir) / "table-catalog" / "table_catalog_enhanced.json",
            Path(self.knowledge_base_dir) / "table-catalog" / "table_catalog.json",
            Path(self.knowledge_base_dir) / "table-catalog" / "table_catalog_with_keywords.json"
        ]
        
        for catalog_path in catalog_paths:
            if catalog_path.exists():
                try:
                    with open(catalog_path, 'r') as f:
                        data = json.load(f)
                    
                    # Handle different catalog formats
                    if isinstance(data, dict):
                        if 'tables' in data:  # Structured format
                            self.table_catalog = {t['table_id']: t for t in data['tables']}
                        else:  # Direct table mapping format
                            self.table_catalog = data
                    
                    logger.info(f"✅ Loaded table catalog: {catalog_path}")
                    logger.info(f"   Tables available: {len(self.table_catalog)}")
                    return
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load catalog {catalog_path}: {e}")
                    continue
        
        logger.warning("⚠️ No table catalog found - using fallbacks only")
    
    def _init_concept_mappings(self):
        """Initialize natural language concept mappings"""
        
        self.concept_mappings = {
            # Income concepts
            'income': ['B19013'],
            'income distribution': ['B19013'],
            'household income': ['B19013'],
            'median income': ['B19013'],
            'family income': ['B19101'],
            'per capita income': ['B19301'],
            
            # Housing concepts
            'housing': ['B25001', 'B25003'],
            'housing units': ['B25001'],
            'housing tenure': ['B25003'],
            'home value': ['B25077'],
            'rent': ['B25064'],
            'housing costs': ['B25070', 'B25080'],
            
            # Population concepts
            'population': ['B01003'],
            'total population': ['B01003'],
            'age': ['B01001'],
            'age and sex': ['B01001'],
            
            # Education concepts
            'education': ['B15003'],
            'educational attainment': ['B15003'],
            'school enrollment': ['B14001'],
            
            # Employment concepts
            'employment': ['B23025'],
            'labor force': ['B23025'],
            'unemployment': ['B23025'],
            'occupation': ['B24010'],
            
            # Poverty concepts
            'poverty': ['B17001'],
            'poverty status': ['B17001'],
            
            # Transportation concepts
            'commuting': ['B08301'],
            'transportation': ['B08301'],
            'travel time': ['B08303']
        }
    
    def _init_common_tables(self):
        """Initialize common table definitions as fallbacks"""
        
        self.common_tables = {
            'B01001': {
                'table_id': 'B01001',
                'title': 'SEX BY AGE',
                'universe': 'Total population',
                'variables': self._generate_b01001_variables(),
                'methodology_notes': 'Age and sex characteristics of the population. Universe includes all persons.'
            },
            'B01003': {
                'table_id': 'B01003',
                'title': 'TOTAL POPULATION',
                'universe': 'Total population',
                'variables': ['B01003_001E', 'B01003_001M'],
                'methodology_notes': 'Total population count from ACS sample data.'
            },
            'B19013': {
                'table_id': 'B19013',
                'title': 'MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS)',
                'universe': 'Households',
                'variables': ['B19013_001E', 'B19013_001M'],
                'methodology_notes': 'Median household income in inflation-adjusted dollars. Excludes group quarters population.'
            },
            'B25001': {
                'table_id': 'B25001',
                'title': 'HOUSING UNITS',
                'universe': 'Housing units',
                'variables': ['B25001_001E', 'B25001_001M'],
                'methodology_notes': 'Total housing units including occupied and vacant units.'
            },
            'B25003': {
                'table_id': 'B25003',
                'title': 'TENURE',
                'universe': 'Occupied housing units',
                'variables': ['B25003_001E', 'B25003_001M', 'B25003_002E', 'B25003_002M',
                            'B25003_003E', 'B25003_003M'],
                'methodology_notes': 'Housing tenure (owner vs renter occupied). Universe is occupied housing units only.'
            },
            'B25064': {
                'table_id': 'B25064',
                'title': 'MEDIAN GROSS RENT (DOLLARS)',
                'universe': 'Renter-occupied housing units paying cash rent',
                'variables': ['B25064_001E', 'B25064_001M'],
                'methodology_notes': 'Median gross rent including utilities. Renter-occupied units only.'
            },
            'B25077': {
                'table_id': 'B25077',
                'title': 'MEDIAN VALUE (DOLLARS)',
                'universe': 'Owner-occupied housing units',
                'variables': ['B25077_001E', 'B25077_001M'],
                'methodology_notes': 'Median home value. Owner-occupied units only.'
            },
            'B15003': {
                'table_id': 'B15003',
                'title': 'EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER',
                'universe': 'Population 25 years and over',
                'variables': self._generate_b15003_variables(),
                'methodology_notes': 'Educational attainment levels. Universe is population 25 years and over.'
            },
            'B23025': {
                'table_id': 'B23025',
                'title': 'EMPLOYMENT STATUS FOR THE POPULATION 16 YEARS AND OVER',
                'universe': 'Population 16 years and over',
                'variables': self._generate_b23025_variables(),
                'methodology_notes': 'Employment status including labor force participation. Population 16 years and over.'
            },
            'B17001': {
                'table_id': 'B17001',
                'title': 'POVERTY STATUS IN THE PAST 12 MONTHS BY SEX BY AGE',
                'universe': 'Population for whom poverty status is determined',
                'variables': self._generate_b17001_variables(),
                'methodology_notes': 'Poverty status by demographic characteristics. Excludes institutionalized population.'
            }
        }
    
    def resolve_tables(self, table_ids: List[str]) -> Dict[str, Any]:
        """
        Resolve table IDs or concepts to variables
        
        Returns:
        {
            'resolved_tables': [
                {
                    'table_id': 'B19013',
                    'input': 'income distribution', 
                    'variables': ['B19013_001E', 'B19013_001M'],
                    'resolution_method': 'concept_mapping'
                }
            ],
            'unresolved': ['unknown_concept']
        }
        """
        
        resolved_tables = []
        unresolved = []
        
        for table_input in table_ids:
            table_input_clean = table_input.strip()
            
            # Try direct table ID first
            if self._is_table_id(table_input_clean):
                resolved = self._resolve_direct_table_id(table_input_clean)
                if resolved:
                    resolved['input'] = table_input_clean
                    resolved_tables.append(resolved)
                else:
                    unresolved.append(table_input_clean)
            
            # Try concept mapping
            else:
                resolved = self._resolve_concept(table_input_clean)
                if resolved:
                    for table_data in resolved:
                        table_data['input'] = table_input_clean
                        resolved_tables.append(table_data)
                else:
                    unresolved.append(table_input_clean)
        
        if unresolved:
            return {
                'error': f'Could not resolve table identifiers: {unresolved}',
                'resolved_tables': resolved_tables,
                'unresolved': unresolved,
                'suggestions': self._suggest_alternatives(unresolved)
            }
        
        return {
            'resolved_tables': resolved_tables,
            'unresolved': []
        }
    
    def _is_table_id(self, table_input: str) -> bool:
        """Check if input looks like a direct table ID"""
        # Pattern like B19013, S2401, DP05, etc.
        return bool(re.match(r'^[A-Z]{1,2}\d{5}[A-Z]*$', table_input.upper()))
    
    def _resolve_direct_table_id(self, table_id: str) -> Optional[Dict[str, Any]]:
        """Resolve direct table ID to variables"""
        
        table_id_upper = table_id.upper()
        
        # Try table catalog first
        if self.table_catalog and table_id_upper in self.table_catalog:
            catalog_entry = self.table_catalog[table_id_upper]
            
            # Handle different catalog formats
            variables = []
            if isinstance(catalog_entry, dict):
                if 'variables' in catalog_entry:
                    # Extract variable IDs from variable objects
                    for var in catalog_entry['variables']:
                        if isinstance(var, dict):
                            var_id = var.get('variable_id') or var.get('temporal_id', '').split('.')[-1]
                            if var_id:
                                variables.append(var_id)
                        else:
                            variables.append(str(var))
                else:
                    # Look for variable-like keys
                    variables = [k for k in catalog_entry.keys() if re.match(r'^[A-Z]\d{5}_\d{3}[EM]$', k)]
            
            if variables:
                return {
                    'table_id': table_id_upper,
                    'variables': variables,
                    'resolution_method': 'table_catalog'
                }
        
        # Fallback to common tables
        if table_id_upper in self.common_tables:
            table_info = self.common_tables[table_id_upper]
            return {
                'table_id': table_id_upper,
                'variables': table_info['variables'],
                'resolution_method': 'common_table_fallback'
            }
        
        # Generate standard variables for known table patterns
        generated_vars = self._generate_standard_table_variables(table_id_upper)
        if generated_vars:
            return {
                'table_id': table_id_upper,
                'variables': generated_vars,
                'resolution_method': 'pattern_generation'
            }
        
        return None
    
    def _resolve_concept(self, concept: str) -> Optional[List[Dict[str, Any]]]:
        """Resolve natural language concept to table(s)"""
        
        concept_lower = concept.lower().strip()
        
        # Direct concept mapping
        if concept_lower in self.concept_mappings:
            table_ids = self.concept_mappings[concept_lower]
            resolved = []
            
            for table_id in table_ids:
                table_data = self._resolve_direct_table_id(table_id)
                if table_data:
                    table_data['resolution_method'] = 'concept_mapping'
                    resolved.append(table_data)
            
            return resolved if resolved else None
        
        # Partial matching
        for concept_key, table_ids in self.concept_mappings.items():
            if concept_lower in concept_key or concept_key in concept_lower:
                resolved = []
                for table_id in table_ids:
                    table_data = self._resolve_direct_table_id(table_id)
                    if table_data:
                        table_data['resolution_method'] = 'partial_concept_match'
                        resolved.append(table_data)
                
                return resolved if resolved else None
        
        return None
    
    def _generate_standard_table_variables(self, table_id: str) -> List[str]:
        """Generate standard variable list for common table patterns"""
        
        # Most tables have at least _001E (total) and _001M (margin of error)
        base_vars = [f"{table_id}_001E", f"{table_id}_001M"]
        
        # Income tables (B19xxx) typically have just totals
        if table_id.startswith('B19'):
            return base_vars
        
        # Population tables (B01xxx) can have many categories
        if table_id == 'B01001':  # Age by sex
            return self._generate_b01001_variables()
        elif table_id.startswith('B01'):
            return base_vars
        
        # Housing tables (B25xxx)
        if table_id == 'B25003':  # Tenure
            return ['B25003_001E', 'B25003_001M', 'B25003_002E', 'B25003_002M', 'B25003_003E', 'B25003_003M']
        elif table_id.startswith('B25'):
            return base_vars
        
        # Education tables (B15xxx)
        if table_id == 'B15003':
            return self._generate_b15003_variables()
        elif table_id.startswith('B15'):
            return base_vars
        
        # Employment tables (B23xxx)
        if table_id == 'B23025':
            return self._generate_b23025_variables()
        elif table_id.startswith('B23'):
            return base_vars
        
        # Default: assume basic total + margin of error
        return base_vars
    
    def _generate_b01001_variables(self) -> List[str]:
        """Generate B01001 (Age by Sex) variables"""
        variables = []
        
        # Total, Male, Female
        for base in ['B01001_001', 'B01001_002', 'B01001_026']:
            variables.extend([f"{base}E", f"{base}M"])
        
        # Age groups for males (003-025) and females (027-049)
        for i in range(3, 26):  # Male age groups
            variables.extend([f"B01001_{i:03d}E", f"B01001_{i:03d}M"])
        
        for i in range(27, 50):  # Female age groups
            variables.extend([f"B01001_{i:03d}E", f"B01001_{i:03d}M"])
        
        return variables
    
    def _generate_b15003_variables(self) -> List[str]:
        """Generate B15003 (Educational Attainment) variables"""
        variables = []
        
        # Educational attainment has ~25 categories
        for i in range(1, 26):
            variables.extend([f"B15003_{i:03d}E", f"B15003_{i:03d}M"])
        
        return variables
    
    def _generate_b23025_variables(self) -> List[str]:
        """Generate B23025 (Employment Status) variables"""
        return [
            'B23025_001E', 'B23025_001M',  # Total
            'B23025_002E', 'B23025_002M',  # In labor force
            'B23025_003E', 'B23025_003M',  # Civilian labor force
            'B23025_004E', 'B23025_004M',  # Employed
            'B23025_005E', 'B23025_005M',  # Unemployed
            'B23025_006E', 'B23025_006M',  # Armed forces
            'B23025_007E', 'B23025_007M'   # Not in labor force
        ]
    
    def _generate_b17001_variables(self) -> List[str]:
        """Generate B17001 (Poverty Status) variables"""
        variables = []
        
        # Poverty status by age and sex has many categories
        for i in range(1, 60):  # B17001 has ~59 variables
            variables.extend([f"B17001_{i:03d}E", f"B17001_{i:03d}M"])
        
        return variables
    
    def get_table_metadata(self, table_id: str) -> Dict[str, Any]:
        """Get metadata for a table"""
        
        table_id_upper = table_id.upper()
        
        # Try table catalog first
        if self.table_catalog and table_id_upper in self.table_catalog:
            catalog_entry = self.table_catalog[table_id_upper]
            if isinstance(catalog_entry, dict):
                return {
                    'title': catalog_entry.get('title', f'Table {table_id_upper}'),
                    'universe': catalog_entry.get('universe', 'Unknown'),
                    'methodology_notes': catalog_entry.get('methodology_notes', ''),
                    'source': 'table_catalog'
                }
        
        # Fallback to common tables
        if table_id_upper in self.common_tables:
            table_info = self.common_tables[table_id_upper]
            return {
                'title': table_info['title'],
                'universe': table_info['universe'],
                'methodology_notes': table_info['methodology_notes'],
                'source': 'common_table_fallback'
            }
        
        # Default metadata
        return {
            'title': f'Table {table_id_upper}',
            'universe': 'Unknown',
            'methodology_notes': f'Standard Census table {table_id_upper}',
            'source': 'default'
        }
    
    def get_variable_metadata(self, variable_id: str) -> Dict[str, Any]:
        """Get metadata for a specific variable"""
        
        variable_id_upper = variable_id.upper()
        
        # Extract table ID from variable ID
        table_id = variable_id_upper.split('_')[0]
        
        # Try to find in table catalog
        if self.table_catalog and table_id in self.table_catalog:
            catalog_entry = self.table_catalog[table_id]
            if isinstance(catalog_entry, dict) and 'variables' in catalog_entry:
                for var in catalog_entry['variables']:
                    if isinstance(var, dict):
                        var_id = var.get('variable_id') or var.get('temporal_id', '').split('.')[-1]
                        if var_id == variable_id_upper:
                            return {
                                'label': var.get('label', 'Unknown'),
                                'concept': var.get('concept', ''),
                                'is_estimate': variable_id_upper.endswith('E'),
                                'is_margin_error': variable_id_upper.endswith('M'),
                                'source': 'table_catalog'
                            }
        
        # Generate basic metadata
        is_estimate = variable_id_upper.endswith('E')
        is_margin = variable_id_upper.endswith('M')
        
        # Extract variable number
        var_parts = variable_id_upper.split('_')
        if len(var_parts) == 2:
            var_num = var_parts[1][:3]  # Get just the number part
            
            if var_num == '001':
                label_base = 'Total'
            else:
                label_base = f'Category {var_num}'
            
            if is_estimate:
                label = f"Estimate!!{label_base}"
            elif is_margin:
                label = f"Margin of Error!!{label_base}"
            else:
                label = label_base
        else:
            label = 'Unknown variable'
        
        return {
            'label': label,
            'concept': f'Variable from table {table_id}',
            'is_estimate': is_estimate,
            'is_margin_error': is_margin,
            'source': 'generated'
        }
    
    def _suggest_alternatives(self, unresolved: List[str]) -> List[str]:
        """Suggest alternative concepts for unresolved inputs"""
        
        suggestions = []
        
        for item in unresolved:
            item_lower = item.lower()
            
            # Suggest related concepts
            if any(term in item_lower for term in ['money', 'pay', 'wage', 'earn']):
                suggestions.extend(['income', 'household income', 'median income'])
            
            elif any(term in item_lower for term in ['home', 'house', 'residence']):
                suggestions.extend(['housing', 'housing units', 'home value'])
            
            elif any(term in item_lower for term in ['people', 'person', 'resident']):
                suggestions.extend(['population', 'total population'])
            
            elif any(term in item_lower for term in ['school', 'college', 'degree']):
                suggestions.extend(['education', 'educational attainment'])
            
            elif any(term in item_lower for term in ['work', 'job', 'employ']):
                suggestions.extend(['employment', 'labor force', 'unemployment'])
            
            elif any(term in item_lower for term in ['poor', 'poverty']):
                suggestions.extend(['poverty', 'poverty status'])
        
        # Remove duplicates and limit suggestions
        return list(set(suggestions))[:5]
    
    def get_available_concepts(self) -> List[str]:
        """Get list of available concept mappings"""
        return sorted(self.concept_mappings.keys())
    
    def get_available_tables(self) -> List[str]:
        """Get list of available table IDs"""
        available = set()
        
        # From catalog
        if self.table_catalog:
            available.update(self.table_catalog.keys())
        
        # From common tables
        available.update(self.common_tables.keys())
        
        return sorted(available)


if __name__ == "__main__":
    # Test table resolver
    resolver = TableResolver()
    
    test_cases = [
        ['B19013'],  # Direct table ID
        ['income distribution'],  # Concept
        ['housing', 'population'],  # Multiple concepts
        ['B25001', 'unknown_concept'],  # Mixed valid/invalid
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case}")
        result = resolver.resolve_tables(test_case)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            if result.get('suggestions'):
                print(f"💡 Suggestions: {result['suggestions']}")
        else:
            print(f"✅ Resolved {len(result['resolved_tables'])} tables")
            for table in result['resolved_tables']:
                print(f"   {table['table_id']}: {len(table['variables'])} variables ({table['resolution_method']})")
