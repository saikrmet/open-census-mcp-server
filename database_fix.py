#!/usr/bin/env python3
"""
Database Query Fix - Handle "city" suffix in place names
The database stores "Seattle city" but users search for "Seattle"
"""

def show_database_fix():
    """Show the specific database query fix needed"""
    
    print("🔧 DATABASE QUERY FIX NEEDED")
    print("="*40)
    
    print("\n🔍 PROBLEM IDENTIFIED:")
    print("Database stores: 'Seattle city'")
    print("User searches: 'Seattle, WA'") 
    print("Query fails: name_lower = 'seattle' (doesn't match 'seattle city')")
    
    print("\n📁 FILE TO FIX:")
    print("src/data_retrieval/geographic_handler.py")
    print("Method: CompleteGeographicHandler._place_state_lookup()")
    
    print("\n🔍 FIND (around line 300):")
    print("""
            cursor.execute('''
                SELECT place_fips, state_fips, state_abbrev, name,
                       population, lat, lon
                FROM places 
                WHERE name_lower = ? AND state_abbrev = ?
                ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                LIMIT 1
            ''', (place_name.lower(), state_abbrev))
""")
    
    print("\n✏️ REPLACE WITH:")
    print("""
            # Try exact match first
            cursor.execute('''
                SELECT place_fips, state_fips, state_abbrev, name,
                       population, lat, lon
                FROM places 
                WHERE name_lower = ? AND state_abbrev = ?
                ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                LIMIT 1
            ''', (place_name.lower(), state_abbrev))
            
            row = cursor.fetchone()
            if not row:
                # Try with city suffix
                cursor.execute('''
                    SELECT place_fips, state_fips, state_abbrev, name,
                           population, lat, lon
                    FROM places 
                    WHERE name_lower = ? AND state_abbrev = ?
                    ORDER BY CAST(COALESCE(population, 0) AS INTEGER) DESC
                    LIMIT 1
                ''', (f"{place_name.lower()} city", state_abbrev))
                
                row = cursor.fetchone()
""")
    
    print("\n🧪 TEST AFTER FIX:")
    print("Seattle, WA should resolve to:")
    print("  FIPS: 53063000") 
    print("  Name: Seattle city")
    print("  Method: database_with_city_suffix")
    
    print("\n⚡ HOWEVER - LLM-FIRST APPROACH IS BETTER:")
    print("Instead of fixing database queries, implement LLM-first resolution")
    print("LLM knows Seattle = 53:63000 without database lookup")

def show_implementation_priority():
    """Show which approach to implement first"""
    
    print("\n🎯 IMPLEMENTATION PRIORITY")
    print("="*30)
    
    print("\n🥇 OPTION 1 (RECOMMENDED): LLM-First Approach")
    print("   - Fixes Seattle, Portland, and all major cities instantly") 
    print("   - No database dependency for common locations")
    print("   - Aligns with your architectural vision")
    print("   - Enables advanced analytics with batch operations")
    print("   - Time: 15 minutes to implement")
    
    print("\n🥈 OPTION 2 (BANDAID): Database Query Fix")
    print("   - Only fixes the 'city' suffix issue")
    print("   - Still database-dependent")
    print("   - Doesn't address architectural problems")
    print("   - Won't support advanced use cases well")
    print("   - Time: 10 minutes to implement")
    
    print("\n💡 RECOMMENDATION:")
    print("Implement LLM-first approach instead of database fix")
    print("It solves the root cause, not just the symptom")

def main():
    show_database_fix()
    show_implementation_priority()
    
    print("\n🚀 NEXT STEPS:")
    print("1. Test fixed LLM approach: python test_llm_first.py")
    print("2. Implement LLM-first in census_mcp_server")
    print("3. Test end-to-end: python quick_test_pnw.py")
    print("4. Move to batch operations for advanced analytics")

if __name__ == "__main__":
    main()
