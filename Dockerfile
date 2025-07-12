# ---------------------------------------------
# Census‑MCP runtime image – version 2.0
# ---------------------------------------------
# * Python 3.11‑slim as base
# * R + tidycensus for data retrieval helpers
# * Pre-cached sentence transformer model (all-mpnet-base-v2)
# * Runtime‑only KB assets (ChromaDB + FAISS, concepts, geo scalars)
# * No raw PDFs / build scripts → lean container
#
# Build:
#   docker build -t ghcr.io/brockwebb/open-census-mcp:2.0 .
# ---------------------------------------------
FROM python:3.11-slim AS base
LABEL maintainer="brockwebb" \
      org.opencontainers.image.source="https://github.com/brockwebb/open-census-mcp-server"

# --------------------------------------------------
# 1️⃣  System‑level dependencies (R + dev headers)
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        r-base r-base-dev \
        r-cran-dplyr r-cran-jsonlite \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
        libgdal-dev libudunits2-dev libproj-dev libgeos-dev \
        libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
        libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
        build-essential \
        cmake \
        libabsl-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# 2️⃣  Python dependencies
# --------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# 2.5️⃣  Pre-download sentence transformer model
# --------------------------------------------------
RUN python -c "from sentence_transformers import SentenceTransformer; \
               import os; \
               os.makedirs('/app/model_cache', exist_ok=True); \
               print('🔄 Downloading all-mpnet-base-v2 model (~400MB)...'); \
               model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', \
                                         cache_folder='/app/model_cache'); \
               print(f'✅ Model cached: {model.get_sentence_embedding_dimension()} dimensions'); \
               print(f'   Max sequence length: {model.max_seq_length}'); \
               print('   Container is now self-contained!');"

# Set environment variable so the app knows where to find the cached model
ENV SENTENCE_TRANSFORMERS_CACHE=/app/model_cache

# --------------------------------------------------
# 3️⃣  R packages needed by tidycensus
# --------------------------------------------------
# Install tidycensus (sf dependencies should now work)
RUN R -e "options(repos = c(CRAN = 'https://cloud.r-project.org/')); \
          install.packages('tidycensus', dependencies=TRUE); \
          if (!require('tidycensus', quietly=TRUE)) { \
            cat('TIDYCENSUS INSTALLATION FAILED\n'); \
            q(status=1) \
          } else { \
            cat('tidycensus installed successfully\n') \
          }"

# Verify all packages work
RUN R -e "library(jsonlite); library(dplyr); library(tidycensus); cat('All R packages loaded successfully\n')"

# --------------------------------------------------
# 4️⃣  Runtime‑only KB assets + application code
# --------------------------------------------------
# ChromaDB knowledge base (main vector database)
COPY knowledge-base/vector-db/            /app/data/vector_db/

# Remove obsolete stats-index (kb_search.py was removed)
# COPY knowledge-base/stats-index/          /app/stats-index/

# Concepts + geo scalars
COPY knowledge-base/concepts/concept_backbone.ttl \
     knowledge-base/concepts/variable_links.ttl \
     knowledge-base/concepts/geo_similarity_scalars.json \
                                          /app/data/concepts/

# Static geo limits for GeoAdvisor
COPY data/table_geos.json                 /app/data/

# Core data files
COPY knowledge-base/2023_ACS_Enriched_Universe.json /app/
COPY knowledge-base/scripts/COOS_Complete_Ontology.json /app/

# Application code
COPY src/                                 /app/src/

# --------------------------------------------------
# 5️⃣  Environment configuration
# --------------------------------------------------
ENV VECTOR_DB_TYPE=chromadb \
    PYTHONPATH=/app:/app/src \
    CENSUS_MCP_CONTAINER=true \
    R_EXECUTABLE=/usr/bin/Rscript \
    EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# --------------------------------------------------
# 6️⃣  Drop root privileges
# --------------------------------------------------
RUN useradd -m -u 1000 census && chown -R census:census /app
USER census

# --------------------------------------------------
# 7️⃣  Container health‑check & launch
# --------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from census_mcp_server import health_check; exit(0 if health_check() else 1)"

CMD ["python", "src/census_mcp_server.py"]
