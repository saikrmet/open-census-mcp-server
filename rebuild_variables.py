#!/usr/bin/env python3
"""Quick test to run the variables rebuild correctly"""

import subprocess
import sys
from pathlib import Path
import os

def rebuild_variables():
    """Rebuild variables database with proper working directory"""
    
    print("🔧 Rebuilding Variables Database")
    print("=" * 40)
    
    # Change to knowledge-base directory
    kb_dir = Path("knowledge-base")
    if not kb_dir.exists():
        print("❌ knowledge-base directory not found")
        return False
    
    build_script = kb_dir / "build-kb-v2.9.py"
    if not build_script.exists():
        print("❌ build-kb-v2.9.py not found")
        return False
    
    print(f"✅ Found build script: {build_script}")
    print(f"✅ Working directory: {kb_dir.absolute()}")
    
    # Set environment
    env = os.environ.copy()
    
    # Build command
    cmd = [
        sys.executable, 
        "build-kb-v2.9.py", 
        "--variables-only", 
        "--faiss", 
        "--rebuild"
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print(f"📁 In directory: {kb_dir.absolute()}")
    print("=" * 40)
    
    try:
        # Run the build script in the correct directory
        result = subprocess.run(
            cmd,
            cwd=kb_dir,
            env=env,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Variables rebuild completed successfully!")
            return True
        else:
            print(f"❌ Variables rebuild failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run build script: {e}")
        return False

if __name__ == "__main__":
    success = rebuild_variables()
    
    if success:
        print("\n🎯 Next step: Run check_faiss_dimensions.py to verify 3072 dimensions")
    else:
        print("\n❌ Fix the build script issues first")
