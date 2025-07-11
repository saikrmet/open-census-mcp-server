# ---------------------------------------------
# Census‑MCP runtime image – version 2.0
# ---------------------------------------------
# * Python 3.11‑slim as base
# * R + tidycensus for data retrieval helpers
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
# 2️⃣  Python dependencies + model pre-caching
# --------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir faiss-cpu sentence-transformers

# Pre-download and cache the BGE model with explicit cache directory
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
RUN mkdir -p /app/.cache/sentence_transformers && \
    python -c "import sys; print('Pre-caching BGE model...', file=sys.stderr); from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-large-en-v1.5'); print('BGE model cached successfully!', file=sys.stderr)"

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

# Cache sentence transformer models to avoid re-downloading
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# --------------------------------------------------
# 4️⃣  Runtime‑only KB assets + application code
# --------------------------------------------------
# ChromaDB knowledge base (main vector database)
COPY knowledge-base/vector-db/            /app/data/vector_db/


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

# Application code and configuration
COPY src/                                 /app/src/
COPY config/                              /app/config/

# --------------------------------------------------
# 5️⃣  Environment configuration
# --------------------------------------------------
ENV VECTOR_DB_TYPE=chromadb \
    EMBEDDING_MODEL=BAAI/bge-large-en-v1.5 \
    PYTHONPATH=/app:/app/src \
    CENSUS_MCP_CONTAINER=true \
    R_EXECUTABLE=/usr/bin/Rscript

# --------------------------------------------------
# 6️⃣  Drop root privileges
# --------------------------------------------------
RUN useradd -m -u 1000 census && chown -R census:census /app && chown -R census:census /app/.cache
USER census

# --------------------------------------------------
# 7️⃣  Container health‑check & launch
# --------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \

CMD ["python", "src/census_mcp_server.py"]
