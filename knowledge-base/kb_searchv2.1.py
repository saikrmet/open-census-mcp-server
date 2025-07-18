"""
ACS Variable Search with Clean Embeddings and Post-Ranking
Updated to use the same FAISS database as the MCP server dual-path system
"""
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# --- Global configuration - UPDATED to match MCP server FAISS database ---
_BASE = Path(__file__).parent          # knowledge-base/
_VECT = _BASE / "variables-faiss/variables.faiss"
_META = _BASE / "variables-faiss/variables_metadata.json"
_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- Global state (lazy loaded) ---------------------------------------------
_index = None
_meta = None
_model = None

def _load():
    """Load FAISS index, metadata, and model once."""
    global _index, _meta, _model
    if _index is None:
        try:
            logger.info(f"Loading FAISS index from {_VECT}")
            _index = faiss.read_index(str(_VECT))
            
            logger.info(f"Loading metadata from {_META}")
            with open(_META) as f:
                _meta = json.load(f)
            
            logger.info(f"Loading model {_MODEL}")
            _model = SentenceTransformer(_MODEL, cache_folder='./model_cache')
            
            logger.info(f"✅ kb_search loaded: {len(_meta)} variables, model ready")
            
        except FileNotFoundError as e:
            logger.error(f"❌ kb_search files not found: {e}")
            logger.error(f"Expected files:")
            logger.error(f"  FAISS index: {_VECT}")
            logger.error(f"  Metadata: {_META}")
            raise
        except Exception as e:
            logger.error(f"❌ kb_search initialization failed: {e}")
            raise

def _extract_table_id(variable_id):
    """Extract table ID from variable ID (B19013_001E -> B19013)."""
    if not variable_id or '_' not in variable_id:
        return "UNKNOWN"
    return variable_id.split("_")[0]

def _post_rank(hits, domain=None):
    """Apply weight boosting and table-family decay."""
    if not hits:
        return hits
    
    # 0. Pre-filter by domain if specified
    if domain:
        hits = [h for h in hits if h.get("weights", {}).get(domain, 0) > 0.3]
        if not hits:
            return hits
    
    # 1. Apply exponential weight boosting
    for h in hits:
        if domain:
            w = h.get("weights", {}).get(domain, 0)
            boost = 1.0 + 1.5 * (w ** 2) if w > 0 else 1.0
        else:
            boost = 1.0
        h["re_rank"] = h["score"] * boost
    
    # 2. Apply table family decay
    TABLE_DECAY = 0.90
    family_seen = {}
    
    # Sort by boosted score first
    hits_sorted = sorted(hits, key=lambda x: x.get("re_rank", x.get("score", 0)), reverse=True)
    
    for h in hits_sorted:
        table_id = h.get("table_id", "UNKNOWN")
        decay_power = family_seen.get(table_id, 0)
        current_score = h.get("re_rank", h.get("score", 0))
        h["re_rank"] = current_score * (TABLE_DECAY ** decay_power)
        family_seen[table_id] = decay_power + 1
    
    # 3. Handle E/M pairs - ensure estimate comes before margin
    final_hits = sorted(hits, key=lambda x: x.get("re_rank", x.get("score", 0)), reverse=True)
    for i in range(len(final_hits) - 1):
        var_id = final_hits[i].get("variable_id", "")
        if var_id.endswith("_M"):
            base_id = var_id[:-2] + "_E"
            # Look for paired estimate in next few positions
            for j in range(i + 1, min(i + 5, len(final_hits))):
                if final_hits[j].get("variable_id", "") == base_id:
                    # Swap if margin ranked higher than estimate
                    final_hits[i], final_hits[j] = final_hits[j], final_hits[i]
                    break
    
    return final_hits

def search(query: str, k: int = 50, domain_filter: str = None):
    """
    Search ACS variables using clean embeddings with post-ranking.
    
    Args:
        query: Natural language search query
        k: Number of results to return (after post-ranking)
        domain_filter: Optional domain to filter by (e.g., 'economics', 'demographics')
    
    Returns:
        List of dicts with variable_id, label, weights, score, re_rank
    """
    try:
        _load()
        
        if not _model or not _index or not _meta:
            logger.error("kb_search not properly initialized")
            return []
        
        # Encode query with normalization to match FAISS index
        query_vec = _model.encode([query], normalize_embeddings=True).astype("float32")
        
        # Search with larger buffer for post-ranking
        search_k = min(k * 3, 150)  # Search more to allow for filtering/reranking
        D, I = _index.search(query_vec, search_k)
        
        # Build hits
        hits = []
        for idx, distance in zip(I[0], D[0]):
            if idx == -1 or idx >= len(_meta):  # FAISS returns -1 for invalid indices
                continue
                
            m = _meta[idx]
            
            # Ensure required fields exist - handle metadata structure
            variable_id = m.get("variable_id", m.get("temporal_id", f"UNKNOWN_{idx}"))
            
            # Convert distance to similarity score (FAISS returns L2 distances)
            score = max(0.0, 1.0 - (float(distance) / 2.0))
            
            hits.append({
                "variable_id": variable_id,
                "table_id": _extract_table_id(variable_id),
                "label": m.get("label", "No label available"),
                "weights": m.get("weights", {}),
                "score": score
            })
        
        # Apply post-ranking
        ranked_hits = _post_rank(hits, domain_filter)
        
        # Return top k after ranking
        result = ranked_hits[:k]
        logger.info(f"kb_search returned {len(result)} results for '{query}'")
        return result
        
    except Exception as e:
        logger.error(f"kb_search failed for query '{query}': {e}")
        return []

def search_by_id(variable_id: str):
    """
    Direct lookup by variable ID.
    
    Args:
        variable_id: Exact variable ID (e.g., 'B19013_001E')
    
    Returns:
        Dict with variable info or None if not found
    """
    try:
        _load()
        
        if not _meta:
            logger.error("kb_search metadata not loaded")
            return None
        
        # Linear search through metadata (could be optimized with index)
        for m in _meta:
            # Check both variable_id and temporal_id fields
            if m.get("variable_id") == variable_id or m.get("temporal_id") == variable_id:
                return {
                    "variable_id": m.get("variable_id", m.get("temporal_id", variable_id)),
                    "table_id": _extract_table_id(variable_id),
                    "label": m.get("label", "No label available"),
                    "weights": m.get("weights", {}),
                    "score": 1.0,  # Perfect match
                    "re_rank": 1.0
                }
        
        logger.warning(f"Variable {variable_id} not found in metadata")
        return None
        
    except Exception as e:
        logger.error(f"search_by_id failed for {variable_id}: {e}")
        return None

def get_available_domains():
    """Get list of available domain filters."""
    try:
        _load()
        
        if not _meta or len(_meta) == 0:
            return []
        
        # Get unique domain keys from first variable that has weights
        for m in _meta:
            weights = m.get("weights", {})
            if weights:
                return list(weights.keys())
        
        return []
        
    except Exception as e:
        logger.error(f"get_available_domains failed: {e}")
        return []

# --- Synonym mapping for canonical queries -----------------------------------
SYNONYMS = {
    "median household income": "B19013_001E",
    "poverty rate": "B17001_002E",
    "unemployment rate": "B23025_005E",
    "foreign born population": "B05002_013E",
    "percent renter occupied": "B25003_003E",
    "median home value": "B25077_001E",
    "median age": "B01002_001E",
    "total population": "B01003_001E",
    "population": "B01003_001E",  # Added for common usage
    "teacher salary": "B24022_011E",
    "education occupations income": "B24022_011E",
    "commute time": "B08303_001E",
    "travel time to work": "B08303_001E",
    # Add more as discovered
}

def search_with_synonyms(query: str, k: int = 50, domain_filter: str = None):
    """
    Search with synonym checking first, then semantic search.
    
    Args:
        query: Natural language search query
        k: Number of results to return
        domain_filter: Optional domain filter
    
    Returns:
        List of search results
    """
    try:
        # Check for exact synonym match first (fastest path)
        query_lower = query.lower().strip()
        if query_lower in SYNONYMS:
            logger.info(f"Found synonym match for '{query}' -> {SYNONYMS[query_lower]}")
            direct_result = search_by_id(SYNONYMS[query_lower])
            if direct_result:
                return [direct_result]
        
        # Fall back to semantic search
        logger.info(f"Using semantic search for '{query}'")
        return search(query, k, domain_filter)
        
    except Exception as e:
        logger.error(f"search_with_synonyms failed for '{query}': {e}")
        return []

# Test function
def test_search():
    """Test the search functionality"""
    test_queries = [
        "teacher salary",
        "population",
        "median income",
        "poverty rate",
        "median age",
        "B19013_001E"
    ]
    
    print("Testing kb_search...")
    
    for query in test_queries:
        print(f"\nTesting: '{query}'")
        try:
            results = search_with_synonyms(query, k=3)
            if results:
                for i, r in enumerate(results):
                    print(f"  {i+1}. {r.get('variable_id', 'N/A')} - {r.get('label', 'N/A')[:60]}...")
                    print(f"      Score: {r.get('score', 0):.3f}, Re-rank: {r.get('re_rank', 0):.3f}")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    # Test when run directly
    test_search()
