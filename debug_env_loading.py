#!/usr/bin/env python3
"""Fix environment loading and check what's in .env"""

import os
from pathlib import Path

def debug_env_loading():
    """Debug environment variable loading"""
    
    print("🔍 Environment Loading Debug")
    print("=" * 40)
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file found: {env_file.absolute()}")
        
        # Read .env file content
        with open(env_file) as f:
            content = f.read()
        
        print(f"📄 .env file content ({len(content)} chars):")
        print("-" * 20)
        
        # Show sanitized content
        for line in content.split('\n'):
            if line.strip() and not line.startswith('#'):
                if 'OPENAI_API_KEY' in line:
                    key_part = line.split('=')[1] if '=' in line else 'missing'
                    print(f"OPENAI_API_KEY={key_part[:10]}...{key_part[-4:] if len(key_part) > 14 else key_part}")
                else:
                    print(line)
        print("-" * 20)
        
    else:
        print(f"❌ .env file not found at: {env_file.absolute()}")
        
        # Check other possible locations
        possible_locations = [
            Path("../.env"),
            Path("src/.env"),
            Path("knowledge-base/.env")
        ]
        
        for loc in possible_locations:
            if loc.exists():
                print(f"📍 Found .env at: {loc.absolute()}")
                break
    
    # Try loading with python-dotenv
    try:
        from dotenv import load_dotenv
        print("\n🔧 Testing dotenv loading...")
        
        # Try different load strategies
        strategies = [
            ("Current directory", lambda: load_dotenv()),
            ("Explicit path", lambda: load_dotenv(".env")),
            ("Find .env", lambda: load_dotenv(find_dotenv=True))
        ]
        
        for name, loader in strategies:
            print(f"  Testing {name}...")
            success = loader()
            api_key = os.getenv('OPENAI_API_KEY')
            
            if api_key:
                print(f"  ✅ {name}: Found key {api_key[:10]}...{api_key[-4:]}")
                return True
            else:
                print(f"  ❌ {name}: No key found")
        
    except ImportError:
        print("❌ python-dotenv not installed")
        
        # Try manual loading
        print("🔧 Trying manual .env loading...")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip() and '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                        if key == 'OPENAI_API_KEY':
                            print(f"✅ Manually loaded: {key}={value[:10]}...{value[-4:]}")
                            return True
    
    return False

if __name__ == "__main__":
    success = debug_env_loading()
    
    # Test final result
    print(f"\n🎯 Final Check:")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OPENAI_API_KEY available: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("❌ OPENAI_API_KEY still not available")
        
        print("\n💡 Solutions:")
        print("1. Check .env file format (no quotes, no spaces around =)")
        print("2. Try: export OPENAI_API_KEY=your_key_here")
        print("3. Check file permissions on .env")
