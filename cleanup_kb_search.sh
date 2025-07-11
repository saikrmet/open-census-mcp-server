#!/bin/bash
"""
Clean up kb_search references and move old files to archive
"""

set -e  # Exit on any error

echo "🧹 Cleaning up kb_search system for RAG-only architecture..."
echo "============================================================"

# Create archive directory
ARCHIVE_DIR="../archive-opencensusmcp"
mkdir -p "$ARCHIVE_DIR"
echo "📁 Created archive directory: $ARCHIVE_DIR"

# Move old kb_search related files to archive
echo ""
echo "📦 Moving files to archive..."

# Move kb_search files
if [ -f "knowledge-base/kb_search.py" ]; then
    mv "knowledge-base/kb_search.py" "$ARCHIVE_DIR/"
    echo "   ✅ Moved kb_search.py to archive"
fi

if [ -f "sanity_check.py" ]; then
    mv "sanity_check.py" "$ARCHIVE_DIR/"
    echo "   ✅ Moved sanity_check.py to archive"
fi

if [ -f "kbsearchtest.py" ]; then
    mv "kbsearchtest.py" "$ARCHIVE_DIR/"
    echo "   ✅ Moved kbsearchtest.py to archive"
fi

# Move any stats-index FAISS files (old semantic search)
if [ -d "knowledge-base/stats-index" ]; then
    mv "knowledge-base/stats-index" "$ARCHIVE_DIR/"
    echo "   ✅ Moved stats-index/ to archive"
fi

# Move build_stats_faiss.py (no longer needed)
if [ -f "knowledge-base/scripts/build_stats_faiss.py" ]; then
    mv "knowledge-base/scripts/build_stats_faiss.py" "$ARCHIVE_DIR/"
    echo "   ✅ Moved build_stats_faiss.py to archive"
fi

# Clean up Python cache
if [ -d "knowledge-base/__pycache__" ]; then
    rm -rf "knowledge-base/__pycache__"
    echo "   ✅ Cleaned Python cache"
fi

echo ""
echo "📝 Updating configuration files..."

# Update Dockerfile
if [ -f "Dockerfile" ]; then
    echo "   🐳 Updating Dockerfile..."
    
    # Create backup
    cp "Dockerfile" "$ARCHIVE_DIR/Dockerfile.bak"
    
    # Remove kb_search related lines
    sed -i.tmp '/# Install additional packages for kb_search/d' Dockerfile
    sed -i.tmp '/# FAISS stats index (kb_search.py)/d' Dockerfile
    sed -i.tmp '/COPY knowledge-base\/kb_search.py/d' Dockerfile
    sed -i.tmp '/COPY knowledge-base\/stats-index/d' Dockerfile
    sed -i.tmp '/from kb_search import search/d' Dockerfile
    
    # Clean up temp file
    rm -f Dockerfile.tmp
    
    echo "      ✅ Removed kb_search references from Dockerfile"
fi

# Update docker-compose.yml
if [ -f "docker-compose.yml" ]; then
    echo "   🐙 Updating docker-compose.yml..."
    
    # Create backup
    cp "docker-compose.yml" "$ARCHIVE_DIR/docker-compose.yml.bak"
    
    # Remove kb_search health check
    sed -i.tmp '/from kb_search import search/d' docker-compose.yml
    sed -i.tmp '/Test actual kb_search functionality/d' docker-compose.yml
    
    # Clean up temp file
    rm -f docker-compose.yml.tmp
    
    echo "      ✅ Removed kb_search health check from docker-compose.yml"
fi

# Update any requirements.txt if it has FAISS
if [ -f "requirements.txt" ]; then
    if grep -q "faiss" requirements.txt; then
        echo "   📋 Found FAISS in requirements.txt..."
        cp "requirements.txt" "$ARCHIVE_DIR/requirements.txt.bak"
        
        # Comment out FAISS requirement
        sed -i.tmp 's/^faiss/#faiss/' requirements.txt
        rm -f requirements.txt.tmp
        
        echo "      ✅ Commented out FAISS requirement"
    fi
fi

echo ""
echo "🔍 Checking for remaining kb_search references..."

# Check for any remaining references
REMAINING=$(grep -r "kb_search" . --exclude-dir=.git --exclude-dir="$ARCHIVE_DIR" 2>/dev/null || true)

if [ -n "$REMAINING" ]; then
    echo "⚠️  Found remaining references:"
    echo "$REMAINING"
    echo ""
    echo "Please review and update these manually."
else
    echo "✅ No remaining kb_search references found!"
fi

echo ""
echo "📊 Archive contents:"
ls -la "$ARCHIVE_DIR/"

echo ""
echo "🎉 Cleanup complete!"
echo ""
echo "📋 Summary:"
echo "   • Moved old kb_search files to $ARCHIVE_DIR"
echo "   • Updated Dockerfile and docker-compose.yml"
echo "   • Cleaned Python cache"
echo "   • Ready for RAG-only architecture"
echo ""
echo "🚀 Next steps:"
echo "   1. Rebuild vector database with canonical_variables.json"
echo "   2. Test RAG-only pipeline"
echo "   3. Update health checks to use vector search"
echo "   4. Ship container v2"
