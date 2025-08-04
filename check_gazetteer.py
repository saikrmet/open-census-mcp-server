#!/usr/bin/env python3
"""Check gazetteer database schema"""

import sqlite3
from pathlib import Path

def check_gazetteer_schema():
    # Find the gazetteer database
    possible_paths = [
        Path(__file__).parent / "knowledge-base" / "geography.db",
        Path(__file__).parent / "geography.db"
    ]
    
    db_path = None
    for path in possible_paths:
        if path.exists():
            db_path = path
            break
    
    if not db_path:
        print("❌ No gazetteer database found")
        return
    
    print(f"✅ Found gazetteer: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\n📋 Tables: {tables}")
        
        # Check schema for each table
        for table in tables:
            print(f"\n🔍 {table} schema:")
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Show sample data
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            sample = cursor.fetchall()
            if sample:
                print(f"  Sample rows: {len(sample)}")
                for row in sample[:1]:  # Just first row
                    print(f"    {row}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    check_gazetteer_schema()
