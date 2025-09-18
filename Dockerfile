FROM ghcr.io/brockwebb/open-census-mcp:2.0

# (Optional) update requirements
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Override CMD to wrap the old one
CMD ["mcp-proxy", "--host", "0.0.0.0", "--port", "8000", "--", "python", "src/census_mcp_server.py"]