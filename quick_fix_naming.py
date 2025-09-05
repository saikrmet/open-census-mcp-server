#!/usr/bin/env python3
"""
Quick Fix for MCP Server Function Naming Issue
Fixes the _get_claude_consultation vs _get_llm_consultation naming mismatch
"""

import sys
from pathlib import Path

def fix_function_naming():
    """Fix the function naming issue in census_mcp_server.py"""
    
    src_dir = Path(__file__).parent / "src"
    server_file = src_dir / "census_mcp_server.py"
    
    if not server_file.exists():
        print(f"❌ Server file not found: {server_file}")
        return False
    
    # Read the file
    with open(server_file, 'r') as f:
        content = f.read()
    
    # Find the problematic line
    if "_get_claude_consultation(arguments)" in content:
        print("🔍 Found the bug: _get_claude_consultation should be _get_llm_consultation")
        
        # Fix the function call
        fixed_content = content.replace(
            "_get_claude_consultation(arguments)",
            "_get_llm_consultation(arguments)"
        )
        
        # Write the fixed version
        backup_file = server_file.with_suffix('.py.backup')
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"✅ Created backup: {backup_file}")
        
        with open(server_file, 'w') as f:
            f.write(fixed_content)
        print(f"✅ Fixed function naming in: {server_file}")
        
        return True
    else:
        print("🤔 Function naming issue not found - may already be fixed")
        return False

if __name__ == "__main__":
    print("🔧 APPLYING QUICK FIX FOR MCP SERVER FUNCTION NAMING")
    success = fix_function_naming()
    
    if success:
        print("\n✅ Fix applied successfully")
        print("You should now be able to use the get_statistical_consultation tool")
    else:
        print("\n⚠️ Fix not applied - please check manually")
