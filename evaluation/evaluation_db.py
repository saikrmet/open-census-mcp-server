#!/usr/bin/env python3
"""
Census MCP Evaluation Database System - FIXED VERSION
Stores and scores manual test results for systematic evaluation

Usage:
    python evaluation_db.py --add-baseline
    python evaluation_db.py --score-run baseline
    python evaluation_db.py --compare-runs baseline improved
"""

import sqlite3
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

class CensusMCPEvaluator:
    def __init__(self, db_path: str = "census_mcp_evaluation.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with evaluation schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    timestamp TEXT NOT NULL,
                    total_queries INTEGER,
                    overall_score REAL,
                    passed_queries INTEGER
                );
                
                CREATE TABLE IF NOT EXISTS query_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    query_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_category TEXT NOT NULL,
                    
                    -- MCP Behavior
                    mcp_tool_called TEXT,
                    mcp_parameters TEXT,
                    mcp_success BOOLEAN,
                    
                    -- Response Quality (INCREASED TO 3000 CHARS)
                    final_answer TEXT,
                    census_variables TEXT,
                    margin_of_error TEXT,
                    methodology_notes TEXT,
                    
                    -- Scoring (0.0 to 1.0)
                    correctness REAL,
                    plan_quality REAL,
                    tool_coordination REAL,
                    limitation_handling REAL,
                    disambiguation REAL,
                    methodology_guidance REAL,
                    
                    -- Overall
                    passed BOOLEAN,
                    failure_reason TEXT,
                    notes TEXT,
                    
                    FOREIGN KEY (run_id) REFERENCES test_runs (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_run_id ON query_tests (run_id);
                CREATE INDEX IF NOT EXISTS idx_query_id ON query_tests (query_id);
            ''')
    
    def _create_test_run(self, run_name: str, description: str) -> int:
        """Create a new test run and return run_id"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO test_runs (run_name, description, timestamp, total_queries)
                VALUES (?, ?, ?, 0)
            ''', (run_name, description, datetime.now().isoformat()))
            return cursor.lastrowid
    
    def _add_query_test(self, run_id: int, test_data: Dict[str, Any]):
        """Add a single query test result"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO query_tests (
                    run_id, query_id, query_text, query_category,
                    mcp_tool_called, mcp_parameters, mcp_success,
                    final_answer, census_variables, margin_of_error, methodology_notes,
                    correctness, plan_quality, tool_coordination, limitation_handling,
                    disambiguation, methodology_guidance, passed, failure_reason, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, test_data["query_id"], test_data["query_text"], test_data["query_category"],
                test_data["mcp_tool_called"], test_data["mcp_parameters"], test_data["mcp_success"],
                test_data["final_answer"], test_data["census_variables"], test_data["margin_of_error"],
                test_data["methodology_notes"], test_data["correctness"], test_data["plan_quality"],
                test_data["tool_coordination"], test_data["limitation_handling"], test_data["disambiguation"],
                test_data["methodology_guidance"], test_data["passed"], test_data.get("failure_reason", ""),
                test_data.get("notes", "")
            ))
    
    def _update_run_summary(self, run_id: int):
        """Update test run with summary statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            stats = cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed,
                    AVG(correctness) as avg_correctness
                FROM query_tests WHERE run_id = ?
            ''', (run_id,)).fetchone()
            
            overall_score = stats[1] / stats[0] if stats[0] > 0 else 0
            
            conn.execute('''
                UPDATE test_runs 
                SET total_queries = ?, passed_queries = ?, overall_score = ?
                WHERE id = ?
            ''', (stats[0], stats[1], overall_score, run_id))
    
    def score_run(self, run_name: str):
        """Print detailed scoring for a test run - FIXED VERSION"""
        with sqlite3.connect(self.db_path) as conn:
            run_info = conn.execute('''
                SELECT * FROM test_runs WHERE run_name = ?
            ''', (run_name,)).fetchone()
            
            if not run_info:
                print(f"❌ Run '{run_name}' not found")
                return
            
            results = conn.execute('''
                SELECT * FROM query_tests WHERE run_id = ? ORDER BY query_id
            ''', (run_info[0],)).fetchall()
            
            print(f"\n{'='*70}")
            print(f"📊 CENSUS MCP EVALUATION - {run_name.upper()}")
            print(f"{'='*70}")
            
            if len(run_info) > 5 and run_info[5] is not None:
                print(f"Overall Score: {run_info[5]:.1%} ({run_info[6] if len(run_info) > 6 else 0}/{run_info[4] if len(run_info) > 4 else 0} passed)")
            else:
                print("Overall Score: Calculating...")
            print(f"Test Date: {run_info[3] if len(run_info) > 3 else 'Unknown'}")
            
            # Category breakdown - SAFE VERSION
            categories = {}
            for result in results:
                category = result[4] if len(result) > 4 else "unknown"
                if category not in categories:
                    categories[category] = {"total": 0, "passed": 0, "avg_scores": []}
                categories[category]["total"] += 1
                
                # Safe access to 'passed' field
                passed = result[22] if len(result) > 22 and result[22] is not None else False
                if passed:
                    categories[category]["passed"] += 1
                    
                # Calculate behavioral average safely
                try:
                    behavioral_scores = []
                    for i in range(15, 21):  # correctness through methodology
                        if len(result) > i and result[i] is not None:
                            behavioral_scores.append(float(result[i]))
                    
                    if behavioral_scores:
                        avg_score = sum(behavioral_scores) / len(behavioral_scores)
                        categories[category]["avg_scores"].append(avg_score)
                except (ValueError, TypeError):
                    pass
            
            print(f"\n📈 Results by Category:")
            for category, stats in categories.items():
                pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
                avg_score = sum(stats["avg_scores"]) / len(stats["avg_scores"]) if stats["avg_scores"] else 0
                print(f"  {category}: {pct:.1f}% pass rate, {avg_score:.2f} avg behavioral score")
            
            # Behavioral scores - SAFE VERSION
            if results:
                def safe_avg(index):
                    values = []
                    for r in results:
                        if len(r) > index and r[index] is not None:
                            try:
                                values.append(float(r[index]))
                            except (ValueError, TypeError):
                                pass
                    return sum(values) / len(values) if values else 0.0
                
                print(f"\n🎯 Behavioral Scores (Average):")
                print(f"  Correctness: {safe_avg(15):.2f}")
                print(f"  Plan Quality: {safe_avg(16):.2f}")
                print(f"  Tool Coordination: {safe_avg(17):.2f}")
                print(f"  Limitation Handling: {safe_avg(18):.2f}")
                print(f"  Disambiguation: {safe_avg(19):.2f}")
                print(f"  Methodology Guidance: {safe_avg(20):.2f}")
            
            # Failed queries - SAFE VERSION
            failures = []
            successes = []
            for r in results:
                passed = r[22] if len(r) > 22 and r[22] is not None else False
                correctness = r[15] if len(r) > 15 and r[15] is not None else 0
                
                if not passed:
                    failures.append(r)
                elif passed and float(correctness) >= 0.8:
                    successes.append(r)
            
            if failures:
                print(f"\n❌ Failed Queries ({len(failures)}):")
                for fail in failures:
                    query_id = fail[2] if len(fail) > 2 else "Unknown"
                    query_text = fail[3] if len(fail) > 3 else "Unknown query"
                    failure_reason = fail[23] if len(fail) > 23 else "No reason provided"
                    print(f"  {query_id}: {query_text}")
                    print(f"    Reason: {failure_reason}")
            
            if successes:
                print(f"\n✅ Success Stories ({len(successes)}):")
                for success in successes:
                    query_id = success[2] if len(success) > 2 else "Unknown"
                    query_text = success[3] if len(success) > 3 else "Unknown query"
                    notes = success[24] if len(success) > 24 else "Success"
                    print(f"  {query_id}: {query_text}")
                    print(f"    Notes: {notes}")
    
    def compare_runs(self, baseline_run: str, comparison_run: str):
        """Enhanced comparison showing confidence and behavioral improvements"""
        with sqlite3.connect(self.db_path) as conn:
            baseline = conn.execute('''
                SELECT * FROM test_runs WHERE run_name = ?
            ''', (baseline_run,)).fetchone()
            
            comparison = conn.execute('''
                SELECT * FROM test_runs WHERE run_name = ?
            ''', (comparison_run,)).fetchone()
            
            if not baseline or not comparison:
                print("❌ One or both runs not found")
                return
            
            # Get detailed behavioral scores
            baseline_scores = conn.execute('''
                SELECT query_id, correctness, plan_quality, tool_coordination, 
                       limitation_handling, disambiguation, methodology_guidance, passed
                FROM query_tests WHERE run_id = ? ORDER BY query_id
            ''', (baseline[0],)).fetchall()
            
            comparison_scores = conn.execute('''
                SELECT query_id, correctness, plan_quality, tool_coordination,
                       limitation_handling, disambiguation, methodology_guidance, passed
                FROM query_tests WHERE run_id = ? ORDER BY query_id
            ''', (comparison[0],)).fetchall()
            
            print(f"\n{'='*70}")
            print(f"📊 ENHANCED COMPARISON: {baseline_run} vs {comparison_run}")
            print(f"{'='*70}")
            
            # Binary pass rate (old metric)
            baseline_pass_rate = baseline[5] or 0
            comparison_pass_rate = comparison[5] or 0
            pass_improvement = comparison_pass_rate - baseline_pass_rate
            
            print(f"📈 Pass Rate:")
            print(f"  {baseline_run}: {baseline_pass_rate:.1%} ({baseline[6]}/{baseline[4]})")
            print(f"  {comparison_run}: {comparison_pass_rate:.1%} ({comparison[6]}/{comparison[4]})")
            print(f"  Change: {pass_improvement:+.1%}")
            
            # Calculate behavioral score improvements
            def safe_avg(scores, index):
                values = []
                for s in scores:
                    if len(s) > index and s[index] is not None:
                        try:
                            values.append(float(s[index]))
                        except (ValueError, TypeError):
                            pass
                return sum(values) / len(values) if values else 0.0
            
            baseline_behavioral = {
                'correctness': safe_avg(baseline_scores, 1),
                'plan_quality': safe_avg(baseline_scores, 2),
                'tool_coordination': safe_avg(baseline_scores, 3),
                'limitation_handling': safe_avg(baseline_scores, 4),
                'disambiguation': safe_avg(baseline_scores, 5),
                'methodology_guidance': safe_avg(baseline_scores, 6)
            }
            
            comparison_behavioral = {
                'correctness': safe_avg(comparison_scores, 1),
                'plan_quality': safe_avg(comparison_scores, 2),
                'tool_coordination': safe_avg(comparison_scores, 3),
                'limitation_handling': safe_avg(comparison_scores, 4),
                'disambiguation': safe_avg(comparison_scores, 5),
                'methodology_guidance': safe_avg(comparison_scores, 6)
            }
            
            print(f"\n🎯 Behavioral Score Improvements:")
            improvements = []
            for metric in baseline_behavioral.keys():
                baseline_val = baseline_behavioral[metric]
                comparison_val = comparison_behavioral[metric]
                improvement = comparison_val - baseline_val
                improvements.append(improvement)
                
                # Color code improvements
                if improvement > 0.1:
                    icon = "🔥"
                elif improvement > 0.05:
                    icon = "📈"
                elif improvement > 0:
                    icon = "↗️"
                elif improvement == 0:
                    icon = "➡️"
                else:
                    icon = "📉"
                    
                print(f"  {icon} {metric.replace('_', ' ').title()}: {baseline_val:.2f} → {comparison_val:.2f} ({improvement:+.2f})")
            
            # Overall behavioral improvement
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0
            print(f"\n📊 Overall Behavioral Improvement: {avg_improvement:+.2f}")
            
            # Show most improved queries
            query_improvements = []
            for baseline_q, comparison_q in zip(baseline_scores, comparison_scores):
                if len(baseline_q) > 6 and len(comparison_q) > 6:
                    try:
                        baseline_vals = [float(baseline_q[j]) for j in range(1, 7) if baseline_q[j] is not None]
                        comparison_vals = [float(comparison_q[j]) for j in range(1, 7) if comparison_q[j] is not None]
                        
                        if baseline_vals and comparison_vals:
                            baseline_avg = sum(baseline_vals) / len(baseline_vals)
                            comparison_avg = sum(comparison_vals) / len(comparison_vals)
                            improvement = comparison_avg - baseline_avg
                            query_improvements.append((baseline_q[0], improvement, baseline_avg, comparison_avg))
                    except (ValueError, TypeError):
                        pass
            
            # Sort by improvement
            query_improvements.sort(key=lambda x: x[1], reverse=True)
            
            if query_improvements:
                print(f"\n🏆 Most Improved Queries:")
                for query_id, improvement, baseline_avg, comparison_avg in query_improvements[:3]:
                    if improvement > 0:
                        print(f"  {query_id}: {baseline_avg:.2f} → {comparison_avg:.2f} (+{improvement:.2f})")
                
                print(f"\n📉 Most Degraded Queries:")
                for query_id, improvement, baseline_avg, comparison_avg in query_improvements[-3:]:
                    if improvement < 0:
                        print(f"  {query_id}: {baseline_avg:.2f} → {comparison_avg:.2f} ({improvement:.2f})")
            
            # Summary verdict with better nuance
            print(f"\n🎯 VERDICT:")
            if avg_improvement > 0.1:
                print("🔥 SIGNIFICANT IMPROVEMENT - System intelligence substantially upgraded")
            elif avg_improvement > 0.05:
                print("📈 MEANINGFUL IMPROVEMENT - Clear progress in system quality")
            elif avg_improvement > 0.01:
                print("↗️ MODEST IMPROVEMENT - Incremental progress detected")
            elif avg_improvement > -0.01:
                print("➡️ STABLE - No significant change in system performance")
            elif avg_improvement > -0.05:
                print("📉 SLIGHT REGRESSION - Minor decline in performance")
            else:
                print("⚠️ SIGNIFICANT REGRESSION - System performance declined")

def main():
    parser = argparse.ArgumentParser(description="Census MCP Evaluation Database")
    parser.add_argument("--score-run", help="Score and display run results")
    parser.add_argument("--compare-runs", nargs=2, help="Compare two runs")
    
    args = parser.parse_args()
    
    evaluator = CensusMCPEvaluator()
    
    if args.score_run:
        evaluator.score_run(args.score_run)
    elif args.compare_runs:
        evaluator.compare_runs(args.compare_runs[0], args.compare_runs[1])
    else:
        print("Use --help to see available commands")

if __name__ == "__main__":
    main()
