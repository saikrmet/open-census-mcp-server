#!/usr/bin/env python3
"""Check gazetteer database schema - correct path"""

import sqlite3
from pathlib import Path

def check_gazetteer_schema():
    # Correct path based on search results
    db_path = Path(__file__).parent / "knowledge-base" / "geo-db" / "geography.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"✅ Found gazetteer: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\n📋 Tables ({len(tables)}): {tables}")
        
        # Check schema for each table
        for table in tables:
            print(f"\n🔍 {table} schema:")
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            column_names = []
            for col in columns:
                column_names.append(col[1])
                print(f"  {col[1]} ({col[2]})")
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  📊 {count:,} rows")
            
            # Show sample data if not too many columns
            if len(column_names) <= 8:
                cursor.execute(f"SELECT * FROM {table} LIMIT 2")
                sample = cursor.fetchall()
                if sample:
                    for i, row in enumerate(sample):
                        print(f"  Sample {i+1}: {dict(zip(column_names, row))}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_gazetteer_schema()
