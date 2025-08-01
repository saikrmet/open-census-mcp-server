#!/usr/bin/env python3
"""
Quick test to run the fixed methodology build and see if token errors are resolved
"""

import subprocess
import sys
from pathlib import Path

def test_methodology_build():
    """Test the methodology build with the chunking fix"""
    
    # Change to knowledge-base directory
    kb_dir = Path("/Users/brock/Documents/GitHub/census-mcp-server/knowledge-base")
    
    print("🧪 Testing methodology build with chunking fix...")
    print(f"Working directory: {kb_dir}")
    
    # Run the methodology build
    cmd = [
        sys.executable, "build-kb-concept-based.py",
        "--methodology-only", 
        "--use-openai",
        "--test-mode",
        "--rebuild"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=kb_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("\n=== STDOUT ===")
            print(result.stdout)
        
        if result.stderr:
            print("\n=== STDERR ===")
            print(result.stderr)
        
        # Check for token limit errors
        if "maximum context length" in result.stderr or "requested" in result.stderr and "tokens" in result.stderr:
            print("\n❌ TOKEN LIMIT ERRORS STILL PRESENT")
            return False
        elif result.returncode == 0:
            print("\n✅ BUILD COMPLETED SUCCESSFULLY")
            return True
        else:
            print(f"\n⚠️ BUILD FAILED WITH EXIT CODE {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ BUILD TIMED OUT")
        return False
    except Exception as e:
        print(f"\n💥 ERROR RUNNING BUILD: {e}")
        return False

if __name__ == "__main__":
    success = test_methodology_build()
    sys.exit(0 if success else 1)
