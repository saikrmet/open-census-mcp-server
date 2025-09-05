#!/usr/bin/env python3
"""
Statistical Advisor Status Check & Integration Guide

This script checks if everything is ready for the statistical advisor integration
and provides clear next steps.
"""

import sys
import os
import json
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    # Try multiple .env locations
    for env_path in [
        Path('.env'),
        Path(__file__).parent / '.env',
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"📄 Loaded .env from: {env_path}")
            break
except ImportError:
    print("⚠️  python-dotenv not installed - relying on system environment variables")

# Setup paths
current_dir = Path(__file__).parent
kb_path = current_dir / "knowledge-base"
src_path = current_dir / "src"

def check_files():
    """Check if all required files exist"""
    print("📁 File System Check:")
    
    required_files = [
        ("LLM Statistical Advisor", kb_path / "llm_statistical_advisor.py"),
        ("Search Engine", kb_path / "kb_search.py"),
        ("Geographic Parsing", kb_path / "geographic_parsing.py"),
        ("Variable Search", kb_path / "variable_search.py"),
        ("MCP Server", src_path / "census_mcp_server.py"),
    ]
    
    required_dirs = [
        ("Variables DB", kb_path / "variables-db"),
        ("Table Catalog", kb_path / "table-catalog"),
        ("Methodology DB", kb_path / "methodology-db"),
    ]
    
    optional_files = [
        ("Gazetteer DB", kb_path / "geo-db" / "geography.db"),
    ]
    
    all_good = True
    
    for name, path in required_files:
        exists = path.exists()
        print(f"  {'✅' if exists else '❌'} {name}: {path}")
        if not exists:
            all_good = False
    
    for name, path in required_dirs:
        exists = path.exists() and path.is_dir()
        print(f"  {'✅' if exists else '❌'} {name}: {path}")
        if not exists:
            all_good = False
    
    for name, path in optional_files:
        exists = path.exists()
        print(f"  {'🔧' if exists else '⚠️ '} {name}: {path} ({'present' if exists else 'optional'})")
    
    return all_good

def check_environment():
    """Check environment variables"""
    print("\\n🔑 Environment Check:")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    print(f"  {'✅' if openai_key else '❌'} OPENAI_API_KEY: {'Present' if openai_key else 'Missing'}")
    print(f"  {'🔧' if anthropic_key else '⚠️ '} ANTHROPIC_API_KEY: {'Present' if anthropic_key else 'Optional'}")
    
    if not openai_key:
        print("    ⚠️  OpenAI API key is required for embeddings and LLM reasoning")
        return False
    
    return True

def check_knowledge_base():
    """Check knowledge base structure"""
    print("\\n📚 Knowledge Base Check:")
    
    # Check variables database
    variables_dir = kb_path / "variables-db"
    if variables_dir.exists():
        faiss_file = variables_dir / "variables.faiss"
        metadata_file = variables_dir / "variables_metadata.json"
        
        print(f"  {'✅' if faiss_file.exists() else '❌'} Variables FAISS index: {faiss_file.exists()}")
        print(f"  {'✅' if metadata_file.exists() else '❌'} Variables metadata: {metadata_file.exists()}")
        
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                print(f"    📊 Variables count: {len(metadata):,}")
            except:
                print("    ⚠️  Could not read metadata file")
    else:
        print("  ❌ Variables database directory missing")
        return False
    
    # Check table catalog
    catalog_dir = kb_path / "table-catalog"
    if catalog_dir.exists():
        catalog_files = list(catalog_dir.glob("table_catalog*.json"))
        print(f"  {'✅' if catalog_files else '❌'} Table catalog files: {len(catalog_files)} found")
    else:
        print("  ❌ Table catalog directory missing")
        return False
    
    # Check methodology database
    methodology_dir = kb_path / "methodology-db"
    if methodology_dir.exists():
        print(f"  {'✅' if methodology_dir.exists() else '❌'} Methodology database: present")
    else:
        print("  ⚠️  Methodology database missing (optional)")
    
    return True

def test_imports():
    """Test if components can be imported"""
    print("\\n🔧 Import Test:")
    
    sys.path.insert(0, str(kb_path))
    sys.path.insert(0, str(src_path))
    
    components = [
        ("LLM Statistical Advisor", "llm_statistical_advisor", "LLMStatisticalAdvisor"),
        ("Search Engine", "kb_search", "create_search_engine"), 
        ("Geographic Parsing", "geographic_parsing", "create_geographic_parser"),
        ("Variable Search", "variable_search", "create_variables_search"),
    ]
    
    all_imports_good = True
    
    for name, module, function in components:
        try:
            exec(f"from {module} import {function}")
            print(f"  ✅ {name}: import successful")
        except Exception as e:
            print(f"  ❌ {name}: import failed - {e}")
            all_imports_good = False
    
    return all_imports_good

def provide_next_steps(files_ok, env_ok, kb_ok, imports_ok):
    """Provide next steps based on status"""
    print("\\n🎯 Status Summary:")
    
    overall_status = files_ok and env_ok and kb_ok and imports_ok
    
    print(f"  Files: {'✅' if files_ok else '❌'}")
    print(f"  Environment: {'✅' if env_ok else '❌'}")
    print(f"  Knowledge Base: {'✅' if kb_ok else '❌'}")
    print(f"  Imports: {'✅' if imports_ok else '❌'}")
    
    print("\\n🚀 Next Steps:")
    
    if overall_status:
        print("  🎉 ALL SYSTEMS GO! You can now:")
        print("    1. Run the integration test:")
        print("       python3 integration_test.py")
        print()
        print("    2. Test LLM statistical consultation:")
        print("       python3 test_consultation.py")
        print()
        print("    3. Run the new MCP server (v3.0):")
        print("       python3 src/census_mcp_server.py")
        print()
        print("    4. Test with new LLM-first architecture in Claude Desktop")
        
    else:
        print("  ❌ Issues detected. Fix these first:")
        
        if not files_ok:
            print("    • Missing required files - check the file paths above")
        
        if not env_ok:
            print("    • Set OPENAI_API_KEY environment variable")
            print("      export OPENAI_API_KEY='your-key-here'")
        
        if not kb_ok:
            print("    • Knowledge base needs to be built:")
            print("      cd knowledge-base && python3 build-kb-concept-based.py")
        
        if not imports_ok:
            print("    • Import errors - check Python dependencies:")
            print("      pip install -r requirements.txt")

def main():
    """Run the complete status check"""
    print("🧠 LLM Statistical Advisor Integration Status Check")
    print("=" * 60)
    
    files_ok = check_files()
    env_ok = check_environment()
    kb_ok = check_knowledge_base()
    imports_ok = test_imports()
    
    provide_next_steps(files_ok, env_ok, kb_ok, imports_ok)

if __name__ == "__main__":
    main()
