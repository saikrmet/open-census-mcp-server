# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Census MCP Server that provides demographic data access through the Model Context Protocol. It combines Python-based MCP server functionality with R's tidycensus package for Census API interactions and includes an embedded knowledge base with vector search capabilities.

## Architecture

**Core Components:**
- `src/census_mcp_server.py` - Main MCP server with Claude-first statistical advisor
- `src/data_retrieval/python_census_api.py` - Python wrapper for R tidycensus calls
- `knowledge-base/` - Vector database and semantic search components
- `knowledge-base/llm_statistical_advisor.py` - LLM-powered statistical consultation

**Key Architecture Pattern:**
The system uses a "Claude-first" statistical advisor approach where Claude Sonnet 4 provides primary statistical reasoning, with knowledge base validation when needed for complex cases.

## Development Commands

### Building the Project

**Docker Container (Production):**
```bash
# Build knowledge base first (required before container build)
cd knowledge-base/
python build-kb.py --both --variables-dir ../data/vector_db --methodology-dir ../data/vector_db

# Build container with embedded knowledge base
./build.sh
```

**Knowledge Base Development:**
```bash
# Build variables database only (fast entity lookup)
cd knowledge-base/
python build-kb.py --variables-only --output-dir variables-db --faiss

# Build methodology database only (conceptual search)
python build-kb.py --methodology-only --output-dir methodology-db

# Build both databases
python build-kb.py --both --variables-dir variables-db --methodology-dir methodology-db
```

### Running the Server

**Development:**
```bash
# Local development (requires R + tidycensus + Python environment)
python src/census_mcp_server.py

# With environment setup
export CENSUS_API_KEY=your_key_here
export PYTHONPATH=/path/to/project/src
python src/census_mcp_server.py
```

**Production:**
```bash
# Docker container
docker run -e CENSUS_API_KEY=your_key census-mcp:latest

# Docker Compose
docker-compose up
```

### Testing

**Quick Tests:**
```bash
# Test basic functionality
python quick_test.py

# Test LLM integration
python test_llm_first.py

# Test consultation features
python test_consultation.py
```

**Knowledge Base Testing:**
```bash
cd knowledge-base/
# Test search functionality
python kb_search_test.py

# Test build process
./run_test.sh
```

## Development Environment Setup

**Dependencies:**
- Python 3.11+ with requirements from `requirements.txt`
- R with tidycensus, dplyr, jsonlite packages
- ChromaDB and sentence-transformers for vector search
- Optional: FAISS for faster variable lookup

**Environment Variables:**
- `CENSUS_API_KEY` - Census Bureau API key (optional but recommended)
- `VECTOR_DB_TYPE=chromadb` - Vector database type
- `R_EXECUTABLE` - Path to Rscript executable
- `PYTHONPATH` - Should include `/src` directory

## Key Files and Their Purposes

**Main Application:**
- `src/census_mcp_server.py` - Primary MCP server implementation
- `src/data_retrieval/python_census_api.py` - Census API integration layer
- `src/utils/config.py` - Configuration management

**Knowledge Base System:**
- `knowledge-base/build-kb.py` - Dual-path knowledge base builder (variables + methodology)
- `knowledge-base/kb_search.py` - Semantic search engine
- `knowledge-base/llm_statistical_advisor.py` - Statistical consultation system

**Configuration:**
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment specification
- `docker-compose.yml` - Container orchestration
- `Dockerfile` - Production container definition

## Statistical Consultation System

The server includes a sophisticated LLM-powered statistical advisor that:

1. **Primary Analysis:** Uses Claude's deep ACS knowledge for statistical reasoning
2. **Variable Recommendation:** Suggests appropriate Census variables based on query analysis
3. **Geographic Guidance:** Provides geography-specific advice (ACS1 vs ACS5, sample sizes)
4. **Methodology Notes:** Explains limitations and proper statistical interpretation
5. **Validation:** Cross-references with 36K+ official Census variables when needed

Access through the `get_statistical_consultation` MCP tool.

## Vector Database Architecture

**Dual-Path Design:**
- **Variables Database:** 65K+ canonical Census variables optimized for entity lookup (FAISS or ChromaDB)
- **Methodology Database:** Documentation and guides optimized for conceptual search (ChromaDB)

This separation allows for fast variable lookup while maintaining rich contextual search capabilities.

## Container Deployment

The production container is self-contained with:
- Pre-built vector database (85MB)
- Cached sentence transformer models
- R environment with tidycensus
- All Python dependencies

Container builds require the vector database to be built first using `build-kb.py`.

## Census API Integration

The system wraps R's tidycensus package through Python subprocess calls, providing:
- Geographic resolution and validation
- Variable mapping and metadata
- Proper error handling and rate limiting
- Support for ACS 1-year and 5-year estimates

Geographic queries are resolved through tidycensus's built-in geography handling.