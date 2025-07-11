#!/usr/bin/env python3
"""
Keyword-based Census search system - replaces broken BGE semantic search
"""

import json
import re
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

class KeywordCensusSearch:
    """Fast, reliable keyword-based search for Census variables"""
    
    def __init__(self, cache_dir: str = "knowledge-base/catalog_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Keyword index: token -> set of variable IDs
        self.keyword_index = collections.defaultdict(set)
        
        # Variable metadata: variable_id -> {label, concept, etc}
        self.variable_metadata = {}
        
        # Domain weights from your enrichment
        self.domain_weights = {}
        
        # Tokenizer for consistent keyword extraction
        self.tokenizer = re.compile(r"[A-Za-z0-9]+")
        
        # Direct mappings for common queries
        self.direct_mappings = {
            "median household income": "B19013_001E",
            "house cost": "B25077_001E",
            "home value": "B25077_001E",
            "home prices": "B25077_001E",
            "average house cost": "B25077_001E",
            "latino population": "B03003_003E",
            "hispanic population": "B03003_003E",
            "how latino": "B03003_003E",
            "elderly population": "B01001_020E",
            "people 65 and over": "B01001_020E",
            "how many elderly": "B01001_020E",
            "seniors": "B01001_020E",
            "poverty rate": "S1701_C03_001E",
            "unemployment rate": "S2301_C04_001E",
            "median rent": "B25064_001E",
            "gross rent": "B25064_001E",
        }
        
        # Keyword synonyms
        self.synonyms = {
            "latino": "hispanic",
            "house": "home",
            "cost": "value",
            "average": "median",
            "elderly": "65",
            "seniors": "65",
            "old": "65",
            "poor": "poverty",
            "jobless": "unemployment",
            "rent": "gross",
        }
        
    def build_index(self, dataset: str = "2023/acs/acs5"):
        """Build keyword index from Census API catalog"""
        
        print(f"🔍 Building keyword index for {dataset}...")
        
        # Cache the catalog
        cache_file = self.cache_dir / f"{dataset.replace('/', '_')}_variables.json"
        
        if not cache_file.exists():
            print(f"📡 Downloading catalog from Census API...")
            url = f"https://api.census.gov/data/{dataset}/variables.json"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            cache_file.write_text(response.text)
        
        # Load catalog
        catalog = json.loads(cache_file.read_text())["variables"]
        print(f"📊 Loaded {len(catalog)} variables")
        
        # Build keyword index
        for variable_id, metadata in catalog.items():
            # Skip group metadata and other non-variable entries
            if 'label' not in metadata or not metadata.get('label'):
                continue
            
            # Skip if it looks like metadata (not a variable)
            if variable_id in ['for', 'in', 'ucgid', 'NAME']:
                continue
                
            # Extract text for indexing
            label = metadata.get('label', '')
            concept = metadata.get('concept', '')
            text = f"{label} {concept}".lower()
            
            # Store metadata
            self.variable_metadata[variable_id] = {
                'label': label,
                'concept': concept,
                'predicateType': metadata.get('predicateType', ''),
                'group': metadata.get('group', ''),
            }
            
            # Index all tokens
            tokens = self.tokenizer.findall(text)
            for token in tokens:
                self.keyword_index[token].add(variable_id)
        
        # Add synonym mappings
        self._add_synonyms()
        
        # Load domain weights if available
        self._load_domain_weights()
        
        print(f"✅ Built index with {len(self.keyword_index)} unique tokens")
        
    def _add_synonyms(self):
        """Add synonym mappings to keyword index"""
        for synonym, canonical in self.synonyms.items():
            if canonical in self.keyword_index:
                self.keyword_index[synonym].update(self.keyword_index[canonical])
                
    def _load_domain_weights(self):
        """Load domain weights from enriched data"""
        weights_file = Path("knowledge-base/2023_ACS_Enriched_Universe_weighted.json")
        
        if not weights_file.exists():
            print("⚠️  No domain weights file found - continuing without weights")
            return
            
        try:
            with open(weights_file) as f:
                data = json.load(f)
                
            if isinstance(data, list):
                variables = data
            elif isinstance(data, dict):
                variables = list(data.values()) if 'variables' not in data else list(data['variables'].values())
            else:
                variables = []
                
            for record in variables:
                if isinstance(record, dict) and 'variable_id' in record:
                    variable_id = record['variable_id']
                    weights = record.get('category_weights_linear', {})
                    if weights:
                        self.domain_weights[variable_id] = weights
                        
            print(f"📈 Loaded domain weights for {len(self.domain_weights)} variables")
            
        except Exception as e:
            print(f"⚠️  Could not load domain weights: {e}")
    
    def search(self, query: str, k: int = 10, domain_filter: Optional[str] = None) -> List[Dict]:
        """Search for variables using keyword matching"""
        
        # Check direct mappings first
        query_clean = query.lower().strip()
        if query_clean in self.direct_mappings:
            variable_id = self.direct_mappings[query_clean]
            result = self._get_variable_result(variable_id)
            if result:
                result['confidence'] = 1.0
                result['match_type'] = 'direct'
                return [result]
        
        # Keyword search
        return self._keyword_search(query, k, domain_filter)
    
    def _keyword_search(self, query: str, k: int, domain_filter: Optional[str] = None) -> List[Dict]:
        """Perform keyword-based search"""
        
        # Extract tokens from query
        tokens = self.tokenizer.findall(query.lower())
        if not tokens:
            return []
        
        # Count matches per variable
        variable_scores = collections.Counter()
        
        for token in tokens:
            for variable_id in self.keyword_index.get(token, []):
                variable_scores[variable_id] += 1
        
        # Filter by domain if specified
        if domain_filter and self.domain_weights:
            filtered_scores = {}
            for variable_id, score in variable_scores.items():
                weights = self.domain_weights.get(variable_id, {})
                domain_weight = weights.get(domain_filter, 0)
                if domain_weight > 0.3:  # Threshold for domain relevance
                    # Boost score by domain weight
                    boosted_score = score * (1 + domain_weight)
                    filtered_scores[variable_id] = boosted_score
            variable_scores = filtered_scores
        
        # Get top results
        top_variables = [vid for vid, _ in variable_scores.most_common(k)]
        
        # Build results
        results = []
        total_tokens = len(tokens)
        
        for variable_id in top_variables:
            result = self._get_variable_result(variable_id)
            if result:
                # Calculate confidence based on token matches
                matches = variable_scores[variable_id]
                confidence = min(matches / total_tokens, 1.0)
                
                result['confidence'] = confidence
                result['match_type'] = 'keyword'
                result['token_matches'] = matches
                results.append(result)
        
        return results
    
    def _get_variable_result(self, variable_id: str) -> Optional[Dict]:
        """Get formatted result for a variable"""
        
        if variable_id not in self.variable_metadata:
            return None
            
        metadata = self.variable_metadata[variable_id]
        weights = self.domain_weights.get(variable_id, {})
        
        return {
            'variable_id': variable_id,
            'label': metadata['label'],
            'concept': metadata['concept'],
            'table_id': variable_id.split('_')[0],
            'weights': weights,
            'score': 1.0  # Placeholder for compatibility
        }
    
    def search_by_id(self, variable_id: str) -> Optional[Dict]:
        """Direct lookup by variable ID"""
        return self._get_variable_result(variable_id)
    
    def get_suggestions(self, query: str, threshold: float = 0.6) -> List[str]:
        """Get suggestions when confidence is low"""
        
        results = self.search(query, k=5)
        
        if not results or (results and results[0]['confidence'] < threshold):
            suggestions = []
            for result in results[:3]:
                suggestion = f"{result['variable_id']} – {result['label'][:60]}..."
                suggestions.append(suggestion)
            return suggestions
        
        return []

# Usage example
def main():
    """Example usage and testing"""
    
    # Initialize search system
    search_system = KeywordCensusSearch()
    search_system.build_index()
    
    # Test the failed queries
    test_queries = [
        "average house cost",
        "latino population",
        "household income",
        "how many elderly"
    ]
    
    print("\n🧪 Testing keyword search:")
    print("=" * 60)
    
    for query in test_queries:
        results = search_system.search(query, k=3)
        
        print(f"\nQuery: '{query}'")
        if results:
            top_result = results[0]
            print(f"  ✅ {top_result['variable_id']} (confidence: {top_result['confidence']:.2f})")
            print(f"     {top_result['label'][:70]}...")
            
            if top_result['confidence'] < 0.6:
                suggestions = search_system.get_suggestions(query)
                if suggestions:
                    print(f"  💡 Suggestions:")
                    for suggestion in suggestions:
                        print(f"     {suggestion}")
        else:
            print(f"  ❌ No results found")

if __name__ == "__main__":
    main()
