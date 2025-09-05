"""
Configuration management for Census MCP Server

Handles paths, environment variables, and settings for containerized deployment.
Supports both local development and Docker container environments.
Pure Python implementation with direct Census API access.
Updated for dual-path vector database architecture with OpenAI embeddings.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration manager for Census MCP Server.
    
    Loads settings from environment variables, config files, and defaults.
    Container-aware for seamless Docker deployment.
    Pure Python with direct Census API access.
    Supports dual-path vector database architecture with OpenAI embeddings.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to JSON config file
        """
        self.config_file = config_file
        self._load_config()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from environment, files, and defaults."""
        
        # Load .env file first
        try:
            from dotenv import load_dotenv
            env_file = self.base_dir / ".env" if hasattr(self, 'base_dir') else Path(__file__).parent.parent.parent / ".env"
            load_dotenv(env_file)
            logger.info(f"✅ Loaded .env file from {env_file}")
        except ImportError:
            logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")
        
        # Base paths - container-aware
        self.base_dir = Path(os.getenv('CENSUS_MCP_BASE', Path(__file__).parent.parent.parent))
        self.data_dir = self.base_dir / "data"
        self.config_dir = self.base_dir / "config"
        
        # Knowledge Base paths
        self.knowledge_corpus_path = self.data_dir / "knowledge_corpus"
        
        # Dual-Path Vector Database Configuration
        self.knowledge_base_dir = self.base_dir / "knowledge-base"
        self.variables_db_path = self.knowledge_base_dir / "variables-faiss"
        self.methodology_db_path = self.knowledge_base_dir / "methodology-db"
        
        # Legacy vector DB path (for backward compatibility)
        self.vector_db_path = self.data_dir / "vector_db"
        self.vector_db_type = os.getenv('VECTOR_DB_TYPE', 'dual-path')
        
        # Census API Configuration
        self.census_api_key = os.getenv('CENSUS_API_KEY')
        self.census_base_url = os.getenv('CENSUS_BASE_URL', 'https://api.census.gov/data')
        
        # LLM Configuration
        self.default_llm = os.getenv('DEFAULT_LLM', 'claude')
        self.claude_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Embedding Configuration - Updated for OpenAI
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
        self.embedding_dimension = self._get_embedding_dimension()
        
        # Server Configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '300'))
        
        # Vector Search Configuration
        self.vector_search_top_k = int(os.getenv('VECTOR_SEARCH_TOP_K', '5'))
        self.vector_search_threshold = float(os.getenv('VECTOR_SEARCH_THRESHOLD', '0.3'))
        
        # Container-specific settings
        self.is_container = os.getenv('CENSUS_MCP_CONTAINER', 'false').lower() == 'true'
        self.container_data_mount = os.getenv('CONTAINER_DATA_MOUNT', '/app/data')
        
        # Load from config file if provided
        if self.config_file and Path(self.config_file).exists():
            self._load_config_file()
        
        # Load default config if it exists
        default_config = self.config_dir / "mcp_config.json"
        if default_config.exists():
            self._load_config_file(str(default_config))
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension based on model."""
        model_dimensions = {
            # OpenAI models
            'text-embedding-3-large': 3072,
            'text-embedding-3-small': 1536,
            'text-embedding-ada-002': 1536,
            # Legacy sentence transformer models (deprecated)
            'all-mpnet-base-v2': 768,
            'all-MiniLM-L6-v2': 384,
            'all-MiniLM-L12-v2': 384,
            'all-distilroberta-v1': 768,
            'paraphrase-mpnet-base-v2': 768,
        }
        
        return int(os.getenv('EMBEDDING_DIMENSION',
                           model_dimensions.get(self.embedding_model, 3072)))
    
    def _load_config_file(self, config_path: Optional[str] = None):
        """Load configuration from JSON file."""
        try:
            config_path = config_path or self.config_file
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            for key, value in file_config.items():
                if not hasattr(self, key) or getattr(self, key) is None:
                    setattr(self, key, value)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Could not load config file {config_path}: {e}")
    
    def _validate_config(self):
        """Validate configuration and create necessary directories."""
        
        # Create data directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_corpus_path.mkdir(parents=True, exist_ok=True)
        
        # Dual-path database directories
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        # Note: Don't auto-create variables-faiss and methodology-db as they should be built explicitly
        
        # Legacy vector DB directory (for backward compatibility)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Validate embedding model
        if not self._validate_embedding_model():
            logger.warning(f"Embedding model {self.embedding_model} may not be available.")
        
        # Log configuration status
        self._log_config_status()
    
    def _validate_embedding_model(self) -> bool:
        """Validate that the embedding model is available."""
        try:
            if self.embedding_model.startswith('text-embedding-'):
                # OpenAI model - check for API key
                return bool(self.openai_api_key)
            elif self.embedding_model in ['all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'all-MiniLM-L12-v2']:
                # Legacy sentence transformers model
                try:
                    import sentence_transformers
                    return True
                except ImportError:
                    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
                    return False
            return True
        except Exception:
            return False
    
    def _log_config_status(self):
        """Log current configuration status."""
        logger.info("Census MCP Server Configuration:")
        logger.info(f"  Base directory: {self.base_dir}")
        logger.info(f"  Data directory: {self.data_dir}")
        
        # Dual-path vector database status
        logger.info(f"  Knowledge base directory: {self.knowledge_base_dir}")
        logger.info(f"  Variables DB path: {self.variables_db_path}")
        logger.info(f"  Variables DB exists: {self.variables_db_path.exists()}")
        logger.info(f"  Methodology DB path: {self.methodology_db_path}")
        logger.info(f"  Methodology DB exists: {self.methodology_db_path.exists()}")
        
        # Legacy path
        logger.info(f"  Legacy vector DB path: {self.vector_db_path}")
        
        logger.info(f"  Container mode: {self.is_container}")
        logger.info(f"  Vector DB type: {self.vector_db_type}")
        
        # Updated embedding model logging
        if self.embedding_model.startswith('text-embedding-'):
            api_status = "with API key" if self.openai_api_key else "missing API key"
            logger.info(f"  Embedding model: {self.embedding_model} (OpenAI, {api_status})")
        else:
            logger.info(f"  Embedding model: {self.embedding_model} (local)")
            
        logger.info(f"  Embedding dimensions: {self.embedding_dimension}")
        logger.info(f"  Census API base: {self.census_base_url}")
        logger.info(f"  Log level: {self.log_level}")
        
        # Check for API keys with updated messages
        api_keys_status = {
            "Census API": "✓" if self.census_api_key else "⚠ (optional)",
            "Claude API": "✓" if self.claude_api_key else "✗",
        }
        
        # OpenAI key status depends on embedding model
        if self.embedding_model.startswith('text-embedding-'):
            api_keys_status["OpenAI API"] = "✓" if self.openai_api_key else "✗ (required for embeddings)"
        else:
            api_keys_status["OpenAI API"] = "✓" if self.openai_api_key else "⚠ (not needed for local embeddings)"
        
        for api, status in api_keys_status.items():
            logger.info(f"  {api} key: {status}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                value = getattr(self, attr_name)
                if 'api_key' not in attr_name.lower():
                    config_dict[attr_name] = str(value) if isinstance(value, Path) else value
        return config_dict

# Global configuration instance
_config_instance = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def reload_config(config_file: Optional[str] = None) -> Config:
    """Reload the global configuration."""
    global _config_instance
    _config_instance = Config(config_file)
    return _config_instance
