#!/usr/bin/env python3
"""Quick MCP server test"""

import sys
import asyncio
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

async def test_mcp_server():
    try:
        print("Testing MCP server import...")
        from census_mcp_server import CensusMCPServer
        print("✅ MCP server imports successfully")
        
        print("Testing server initialization...")
        server = CensusMCPServer()
        print("✅ MCP server initializes successfully")
        
        print("Testing components...")
        print(f"  Census API: {'✅' if server.census_api else '❌'}")
        print(f"  Search Engine: {'✅' if server.search_engine else '❌'}")
        
        print("All MCP tests passed!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
