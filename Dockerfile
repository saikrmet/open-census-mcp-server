FROM ghcr.io/brockwebb/open-census-mcp:2.0

# (Optional) update requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Override CMD to wrap the old one
CMD ["mcp-proxy", "--host", "0.0.0.0", "--port", "8000", "--", "python", "/app/src/census_mcp_server.py"]