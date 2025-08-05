#!/usr/bin/env python3
"""Debug the variables build process to see why it's creating 768-dim embeddings"""

import sys
import os
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv()

def test_build_embedding_generation():
    """Test what embeddings the build script actually generates"""
    
    print("🔍 Debugging Variables Build Process")
    print("=" * 50)
    
    # Test OpenAI embeddings directly
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No OpenAI API key found")
            return
        
        client = OpenAI(api_key=api_key)
        print(f"✅ OpenAI client created with key: {api_key[:10]}...{api_key[-4:]}")
        
        # Test the exact embedding call used in build script
        test_text = "B01003_001E Total population"
        
        print(f"\n🧪 Testing OpenAI embedding generation:")
        print(f"Text: '{test_text}'")
        print(f"Model: text-embedding-3-large")
        
        response = client.embeddings.create(
            input=[test_text],
            model="text-embedding-3-large"
        )
        
        embedding = response.data[0].embedding
        print(f"✅ OpenAI embedding generated: {len(embedding)} dimensions")
        
        if len(embedding) == 3072:
            print("✅ OpenAI is working correctly (3072 dimensions)")
        else:
            print(f"❌ OpenAI returned wrong dimensions: {len(embedding)}")
            return
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return
    
    # Now check what the build script's method produces
    print(f"\n🔧 Testing Build Script Methods")
    print("-" * 30)
    
    try:
        # Import the build script
        sys.path.insert(0, 'knowledge-base')
        
        # Import specific methods from build script
        from build_kb_v2_9 import OpenAIKnowledgeBuilder
        
        # Create a test builder
        builder = OpenAIKnowledgeBuilder(
            source_dir=Path("knowledge-base/source-docs"),
            build_mode="variables",
            variables_dir=Path("knowledge-base/variables-faiss"),
            use_faiss=True
        )
        
        print(f"✅ Build script imported successfully")
        print(f"Client initialized: {hasattr(builder, 'openai_client')}")
        
        # Test the builder's embedding method
        test_texts = ["B01003_001E Total population", "B19013_001E Median household income"]
        
        print(f"\n🧠 Testing builder._generate_openai_embeddings()...")
        embeddings = builder._generate_openai_embeddings(test_texts)
        
        print(f"✅ Builder generated embeddings: {embeddings.shape}")
        
        if embeddings.shape[1] == 3072:
            print("✅ Builder is using OpenAI correctly (3072 dimensions)")
        else:
            print(f"❌ Builder is using wrong embeddings: {embeddings.shape[1]} dimensions")
        
    except Exception as e:
        print(f"❌ Build script test failed: {e}")
        import traceback
        traceback.print_exc()

def check_build_logs():
    """Check recent build logs for clues"""
    
    print(f"\n📋 Checking Recent Build Activity")
    print("-" * 30)
    
    # Check if build-kb-v2.9.py was recently run
    build_script = Path("knowledge-base/build-kb-v2.9.py")
    if build_script.exists():
        stat = build_script.stat()
        print(f"Build script last modified: {stat.st_mtime}")
    
    # Check variables directory files
    variables_dir = Path("knowledge-base/variables-faiss")
    if variables_dir.exists():
        print(f"\nVariables directory contents:")
        for file in variables_dir.iterdir():
            if file.is_file():
                stat = file.stat()
                print(f"  {file.name}: {stat.st_size} bytes, modified {stat.st_mtime}")

if __name__ == "__main__":
    test_build_embedding_generation()
    check_build_logs()
