#!/usr/bin/env python3
"""
Table Formatter - Format Census table data for presentation

Handles:
1. Structured table formatting with proper headers
2. Variable grouping and hierarchy 
3. Statistical formatting (numbers, percentages, etc.)
4. Margin of error integration
5. Table metadata presentation

Architecture: Clean, readable table output optimized for Claude Desktop display
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)

class TableFormatter:
    """Format Census table data for clean presentation"""
    
    def __init__(self):
        logger.info("✅ TableFormatter initialized")
    
    def format_table_collection(self, tables: Dict[str, Any], location_name: str) -> str:
        """
        Format multiple tables into a cohesive presentation
        
        Args:
            tables: Dict of table_id -> table_data
            location_name: Geographic location name
            
        Returns:
            Formatted markdown string
        """
        
        response_parts = [
            f"# 📊 Census Table Data for {location_name}\n"
        ]
        
        # Summary information
        total_tables = len(tables)
        total_variables = sum(t.get('variable_count', 0) for t in tables.values())
        
        response_parts.extend([
            "## 📋 **Summary**",
            f"**Tables Retrieved**: {total_tables}",
            f"**Total Variables**: {total_variables}",
            f"**Location**: {location_name}\n"
        ])
        
        # Format each table
        for i, (table_id, table_data) in enumerate(tables.items(), 1):
            formatted_table = self.format_single_table(table_id, table_data, i)
            response_parts.append(formatted_table)
        
        return "\n".join(response_parts)
    
    def format_single_table(self, table_id: str, table_data: Dict[str, Any], table_number: int = 1) -> str:
        """
        Format a single table with all its variables
        
        Args:
            table_id: Census table ID (e.g., 'B19013')
            table_data: Table data dictionary
            table_number: Sequential number for display
            
        Returns:
            Formatted markdown string for the table
        """
        
        parts = [
            f"## 📋 **Table {table_number}: {table_id}**",
            f"**Title**: {table_data.get('title', f'Table {table_id}')}",
            f"**Universe**: {table_data.get('universe', 'Unknown')}",
            f"**Variables**: {table_data.get('variable_count', 0)}\n"
        ]
        
        # Format structured data
        structured_data = table_data.get('structured_data', [])
        if structured_data:
            formatted_data = self._format_structured_data(structured_data, table_id)
            parts.extend([
                "### 📊 **Data**",
                formatted_data,
                ""
            ])
        
        # Add methodology notes
        methodology = table_data.get('methodology_notes', '').strip()
        if methodology:
            parts.extend([
                "### 📚 **Methodology Notes**",
                methodology,
                ""
            ])
        
        # Add primary variable callout
        primary_var = table_data.get('primary_variable')
        if primary_var:
            primary_data = next((row for row in structured_data if row['variable_id'] == primary_var), None)
            if primary_data and primary_data.get('estimate') is not None:
                parts.extend([
                    f"### 🎯 **Key Statistic**",
                    f"**{primary_var}**: {primary_data.get('formatted', 'No data')} ({primary_data.get('label', 'Unknown')})",
                    ""
                ])
        
        return "\n".join(parts)
    
    def _format_structured_data(self, data: List[Dict[str, Any]], table_id: str) -> str:
        """Format the structured data rows"""
        
        if not data:
            return "No data available"
        
        # Group estimates and margins of error
        estimates = [row for row in data if row['variable_id'].endswith('E')]
        margins = [row for row in data if row['variable_id'].endswith('M')]
        
        # Create margin lookup
        margin_lookup = {row['variable_id'].replace('M', 'E'): row for row in margins}
        
        # Determine formatting approach based on table type
        if len(estimates) <= 5:
            return self._format_simple_table(estimates, margin_lookup)
        else:
            return self._format_hierarchical_table(estimates, margin_lookup, table_id)
    
    def _format_simple_table(self, estimates: List[Dict], margin_lookup: Dict) -> str:
        """Format simple tables with few variables"""
        
        parts = []
        
        for row in estimates:
            variable_id = row['variable_id']
            label = self._clean_label(row.get('label', 'Unknown'))
            estimate = row.get('estimate')
            formatted = row.get('formatted', 'No data')
            
            # Add margin of error if available
            margin_row = margin_lookup.get(variable_id)
            margin_info = ""
            if margin_row and margin_row.get('estimate') is not None:
                margin_val = margin_row['estimate']
                margin_info = f" (±{margin_val:,.0f})" if margin_val > 0 else ""
            
            # Format the row
            if estimate is not None:
                if row.get('is_total', False):
                    parts.append(f"🎯 **{variable_id}**: {formatted}{margin_info} - {label}")
                else:
                    parts.append(f"• **{variable_id}**: {formatted}{margin_info} - {label}")
            else:
                parts.append(f"• **{variable_id}**: No data - {label}")
        
        return "\n".join(parts)
    
    def _format_hierarchical_table(self, estimates: List[Dict], margin_lookup: Dict, table_id: str) -> str:
        """Format complex tables with hierarchical structure"""
        
        # Group variables by category
        grouped = self._group_variables_by_hierarchy(estimates, table_id)
        
        parts = []
        
        for group_name, variables in grouped.items():
            if group_name != "Total":
                parts.append(f"\n**{group_name}**")
            
            for row in variables:
                variable_id = row['variable_id']
                label = self._clean_label(row.get('label', 'Unknown'))
                estimate = row.get('estimate')
                formatted = row.get('formatted', 'No data')
                
                # Add margin of error if available
                margin_row = margin_lookup.get(variable_id)
                margin_info = ""
                if margin_row and margin_row.get('estimate') is not None:
                    margin_val = margin_row['estimate']
                    margin_info = f" (±{margin_val:,.0f})" if margin_val > 0 else ""
                
                # Format with appropriate indent
                if row.get('is_total', False) or group_name == "Total":
                    icon = "🎯"
                    indent = ""
                else:
                    icon = "•"
                    indent = "  " if group_name != "Total" else ""
                
                if estimate is not None:
                    parts.append(f"{indent}{icon} **{variable_id}**: {formatted}{margin_info} - {label}")
                else:
                    parts.append(f"{indent}{icon} **{variable_id}**: No data - {label}")
        
        return "\n".join(parts)
    
    def _group_variables_by_hierarchy(self, estimates: List[Dict], table_id: str) -> Dict[str, List[Dict]]:
        """Group variables by logical hierarchy based on table type"""
        
        grouped = {"Total": []}
        
        for row in estimates:
            variable_id = row['variable_id']
            label = row.get('label', '')
            
            # Total variables always go first
            if row.get('is_total', False) or variable_id.endswith('_001E'):
                grouped["Total"].append(row)
                continue
            
            # Table-specific grouping
            if table_id == 'B01001':  # Age by sex
                group = self._categorize_age_sex_variable(label)
            elif table_id == 'B15003':  # Education
                group = self._categorize_education_variable(label)
            elif table_id == 'B23025':  # Employment
                group = self._categorize_employment_variable(label)
            elif table_id == 'B17001':  # Poverty
                group = self._categorize_poverty_variable(label)
            elif table_id.startswith('B25'):  # Housing
                group = self._categorize_housing_variable(label)
            else:
                group = "Other Variables"
            
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(row)
        
        # Remove empty groups
        return {k: v for k, v in grouped.items() if v}
    
    def _categorize_age_sex_variable(self, label: str) -> str:
        """Categorize B01001 age/sex variables"""
        label_lower = label.lower()
        
        if 'male' in label_lower and 'female' not in label_lower:
            return "Male by Age"
        elif 'female' in label_lower:
            return "Female by Age"
        else:
            return "Age Groups"
    
    def _categorize_education_variable(self, label: str) -> str:
        """Categorize B15003 education variables"""
        label_lower = label.lower()
        
        if any(term in label_lower for term in ['high school', 'ged']):
            return "High School"
        elif any(term in label_lower for term in ['college', 'bachelor', 'associate']):
            return "College"
        elif any(term in label_lower for term in ['graduate', 'master', 'doctoral', 'professional']):
            return "Graduate/Professional"
        elif any(term in label_lower for term in ['less than', 'no school', '9th grade', '10th grade', '11th grade']):
            return "Less than High School"
        else:
            return "Educational Attainment"
    
    def _categorize_employment_variable(self, label: str) -> str:
        """Categorize B23025 employment variables"""
        label_lower = label.lower()
        
        if 'labor force' in label_lower:
            return "Labor Force Status"
        elif any(term in label_lower for term in ['employed', 'unemployed']):
            return "Employment Status"
        elif 'armed forces' in label_lower:
            return "Military"
        else:
            return "Employment"
    
    def _categorize_poverty_variable(self, label: str) -> str:
        """Categorize B17001 poverty variables"""
        label_lower = label.lower()
        
        if 'male' in label_lower:
            return "Male by Poverty Status"
        elif 'female' in label_lower:
            return "Female by Poverty Status"
        elif any(term in label_lower for term in ['below', 'above']):
            return "Poverty Status"
        else:
            return "Poverty by Demographics"
    
    def _categorize_housing_variable(self, label: str) -> str:
        """Categorize B25xxx housing variables"""
        label_lower = label.lower()
        
        if any(term in label_lower for term in ['owner', 'owned']):
            return "Owner-Occupied"
        elif any(term in label_lower for term in ['renter', 'rent']):
            return "Renter-Occupied"
        elif 'vacant' in label_lower:
            return "Vacant Units"
        else:
            return "Housing Characteristics"
    
    def _clean_label(self, label: str) -> str:
        """Clean Census variable labels for better readability"""
        
        if not label or label == 'Unknown':
            return 'Unknown'
        
        # Remove Census formatting artifacts
        cleaned = label.replace('Estimate!!', '').replace('Margin of Error!!', '')
        
        # Clean up extra whitespace and separators
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'^[:\-\s]+|[:\-\s]+$', '', cleaned)
        
        # Capitalize properly
        if cleaned.islower():
            cleaned = cleaned.title()
        
        return cleaned.strip()
    
    def format_comparison_tables(self, comparison_data: List[Tuple[str, Dict]], table_ids: List[str]) -> str:
        """
        Format table data for multiple locations in comparison format
        
        Args:
            comparison_data: List of (location_name, tables_dict) tuples
            table_ids: List of requested table IDs
            
        Returns:
            Formatted comparison markdown
        """
        
        response_parts = [
            "# 📊 Census Table Comparison\n",
            f"**Tables**: {', '.join(table_ids)}",
            f"**Locations**: {len(comparison_data)}\n"
        ]
        
        # For each table, show comparison across locations
        for table_id in table_ids:
            response_parts.append(f"## 📋 **Table {table_id} Comparison**")
            
            # Get table metadata from first location that has this table
            table_meta = None
            for _, tables_dict in comparison_data:
                if table_id in tables_dict:
                    table_meta = tables_dict[table_id]
                    break
            
            if table_meta:
                response_parts.extend([
                    f"**Title**: {table_meta.get('title', f'Table {table_id}')}",
                    f"**Universe**: {table_meta.get('universe', 'Unknown')}\n"
                ])
            
            # Compare primary variables across locations
            for location_name, tables_dict in comparison_data:
                if table_id in tables_dict:
                    table_data = tables_dict[table_id]
                    primary_var = table_data.get('primary_variable')
                    
                    if primary_var:
                        structured_data = table_data.get('structured_data', [])
                        primary_data = next((row for row in structured_data if row['variable_id'] == primary_var), None)
                        
                        if primary_data and primary_data.get('estimate') is not None:
                            formatted_val = primary_data.get('formatted', 'No data')
                            response_parts.append(f"📍 **{location_name}**: {formatted_val}")
                        else:
                            response_parts.append(f"📍 **{location_name}**: No data")
                else:
                    response_parts.append(f"📍 **{location_name}**: Table not available")
            
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def format_table_summary(self, tables: Dict[str, Any]) -> str:
        """
        Create a brief summary of tables retrieved
        
        Args:
            tables: Dict of table_id -> table_data
            
        Returns:
            Brief summary string
        """
        
        if not tables:
            return "No tables retrieved"
        
        total_vars = sum(t.get('variable_count', 0) for t in tables.values())
        
        summary_parts = [
            f"📊 **{len(tables)} tables retrieved** with **{total_vars} total variables**:"
        ]
        
        for table_id, table_data in tables.items():
            title = table_data.get('title', f'Table {table_id}')
            var_count = table_data.get('variable_count', 0)
            
            # Get primary statistic if available
            primary_var = table_data.get('primary_variable')
            primary_stat = ""
            if primary_var:
                structured_data = table_data.get('structured_data', [])
                primary_data = next((row for row in structured_data if row['variable_id'] == primary_var), None)
                if primary_data and primary_data.get('estimate') is not None:
                    primary_stat = f" → {primary_data.get('formatted', 'No data')}"
            
            summary_parts.append(f"• **{table_id}**: {title} ({var_count} vars){primary_stat}")
        
        return "\n".join(summary_parts)


class TableValidationError(Exception):
    """Exception raised when table data validation fails"""
    pass


def validate_table_data(table_data: Dict[str, Any]) -> bool:
    """
    Validate that table data has required structure
    
    Args:
        table_data: Table data dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        TableValidationError: If validation fails
    """
    
    required_keys = ['structured_data', 'title', 'universe', 'variable_count']
    
    for key in required_keys:
        if key not in table_data:
            raise TableValidationError(f"Missing required key: {key}")
    
    structured_data = table_data['structured_data']
    if not isinstance(structured_data, list):
        raise TableValidationError("structured_data must be a list")
    
    # Validate each data row
    required_row_keys = ['variable_id', 'label', 'estimate', 'formatted']
    
    for i, row in enumerate(structured_data):
        if not isinstance(row, dict):
            raise TableValidationError(f"Row {i} is not a dictionary")
        
        for key in required_row_keys:
            if key not in row:
                raise TableValidationError(f"Row {i} missing required key: {key}")
        
        # Validate variable ID format
        var_id = row['variable_id']
        if not re.match(r'^[A-Z]\d{5}_\d{3}[EM]?$', var_id):
            raise TableValidationError(f"Invalid variable ID format: {var_id}")
    
    return True


def format_number(value: float, value_type: str = 'count') -> str:
    """
    Format numbers appropriately based on type
    
    Args:
        value: Numeric value to format
        value_type: Type of value ('count', 'currency', 'percentage', 'rate')
        
    Returns:
        Formatted string
    """
    
    if value is None:
        return "No data"
    
    try:
        num_val = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    if value_type == 'currency':
        if num_val >= 1000:
            return f"${num_val:,.0f}"
        else:
            return f"${num_val:.2f}"
    
    elif value_type == 'percentage':
        return f"{num_val:.1f}%"
    
    elif value_type == 'rate':
        return f"{num_val:.2f}"
    
    else:  # count
        if num_val >= 1000:
            return f"{num_val:,.0f}"
        else:
            return f"{num_val:g}"


def infer_value_type(variable_id: str, label: str) -> str:
    """
    Infer the type of value based on variable ID and label
    
    Args:
        variable_id: Census variable ID
        label: Variable label
        
    Returns:
        Value type ('count', 'currency', 'percentage', 'rate')
    """
    
    label_lower = label.lower()
    
    # Currency indicators
    if any(term in label_lower for term in ['dollar', 'income', 'value', 'rent', 'cost', '$']):
        return 'currency'
    
    # Percentage indicators
    if any(term in label_lower for term in ['percent', 'rate', '%']):
        return 'percentage'
    
    # Rate indicators (like unemployment rate)
    if 'rate' in label_lower and 'dollar' not in label_lower:
        return 'rate'
    
    # Default to count
    return 'count'


if __name__ == "__main__":
    # Test table formatter
    formatter = TableFormatter()
    
    # Sample table data
    sample_table_data = {
        'B19013': {
            'title': 'MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS)',
            'universe': 'Households',
            'variable_count': 2,
            'structured_data': [
                {
                    'variable_id': 'B19013_001E',
                    'label': 'Estimate!!Median household income in the past 12 months (in 2023 inflation-adjusted dollars)',
                    'estimate': 75000.0,
                    'formatted': '$75,000',
                    'is_total': True
                },
                {
                    'variable_id': 'B19013_001M',
                    'label': 'Margin of Error!!Median household income in the past 12 months (in 2023 inflation-adjusted dollars)',
                    'estimate': 2500.0,
                    'formatted': '$2,500',
                    'is_total': False
                }
            ],
            'primary_variable': 'B19013_001E',
            'methodology_notes': 'Median household income in inflation-adjusted dollars. Excludes group quarters population.'
        }
    }
    
    # Test formatting
    print("🧪 Testing table formatter:")
    result = formatter.format_table_collection(sample_table_data, "Austin, TX")
    print(result)
    
    # Test validation
    try:
        validate_table_data(sample_table_data['B19013'])
        print("\n✅ Table data validation passed")
    except TableValidationError as e:
        print(f"\n❌ Table data validation failed: {e}")
    
    # Test number formatting
    print(f"\n🧪 Number formatting tests:")
    print(f"Currency: {format_number(75000, 'currency')}")
    print(f"Count: {format_number(250000, 'count')}")
    print(f"Percentage: {format_number(15.7, 'percentage')}")
