#!/usr/bin/env python3
"""
Census MCP Evaluation Database System
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
    def __init__(self, db_path: str = "evaluation/census_mcp_evaluation.db"):
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
                    
                    -- Response Quality
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
    
    def add_baseline_data(self):
        """Add baseline test results from Claude's comprehensive testing log"""
        
        baseline_tests = [
            # Original 6 tests
            {
                "query_id": "Q01",
                "query_text": "What's the median household income in Baltimore, Maryland?",
                "query_category": "basic_demographic",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Baltimore, MD", "variables": ["median household income"], "year": 2023}),
                "mcp_success": True,
                "final_answer": "Baltimore's median household income is $59,623 (±$1,145)",
                "census_variables": "B19013_001",
                "margin_of_error": "±1,145 (1.9%)",
                "methodology_notes": "ACS 5-year estimates, 90% confidence level",
                "correctness": 1.0,
                "plan_quality": 0.9,
                "tool_coordination": 0.8,
                "limitation_handling": 0.7,
                "disambiguation": 0.8,
                "methodology_guidance": 0.9,
                "passed": True,
                "notes": "Perfect execution - correct tool, real data, proper MOE"
            },
            {
                "query_id": "Q02",
                "query_text": "What's the total population and median age in Austin, Texas?",
                "query_category": "multi_variable",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Austin, TX", "variables": ["total population", "median age"], "year": 2023}),
                "mcp_success": True,
                "final_answer": "Austin population: 967,862 (±373), median age: 34.5 (±0.2)",
                "census_variables": "B01003_001, B01002_001",
                "margin_of_error": "±373 (0.0%), ±0.2 (0.6%)",
                "methodology_notes": "ACS 5-year estimates, very tight margins indicate reliable data",
                "correctness": 1.0,
                "plan_quality": 0.9,
                "tool_coordination": 0.9,
                "limitation_handling": 0.8,
                "disambiguation": 0.8,
                "methodology_guidance": 0.8,
                "passed": True,
                "notes": "Excellent multi-variable handling, consistent with previous query"
            },
            {
                "query_id": "Q03",
                "query_text": "What's the population of 123 Main Street?",
                "query_category": "limitation_handling",
                "mcp_tool_called": None,
                "mcp_parameters": None,
                "mcp_success": False,
                "final_answer": "Census doesn't report individual address data",
                "census_variables": None,
                "margin_of_error": None,
                "methodology_notes": "Privacy protection, smallest unit is census block",
                "correctness": 1.0,
                "plan_quality": 0.9,
                "tool_coordination": 0.0,
                "limitation_handling": 1.0,
                "disambiguation": 0.9,
                "methodology_guidance": 0.8,
                "passed": True,
                "notes": "Perfect limitation handling - correctly rejected impossible query"
            },
            {
                "query_id": "Q04",
                "query_text": "What's the median income in Washington?",
                "query_category": "disambiguation",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Washington", "variables": ["median household income"], "year": 2023}),
                "mcp_success": True,
                "final_answer": "Washington State median income: $106,287 (±$1,803)",
                "census_variables": "B19013_001",
                "margin_of_error": "±1,803 (1.7%)",
                "methodology_notes": "Defaulted to Washington State, should have requested clarification",
                "correctness": 0.7,
                "plan_quality": 0.6,
                "tool_coordination": 0.7,
                "limitation_handling": 0.5,
                "disambiguation": 0.3,
                "methodology_guidance": 0.7,
                "passed": False,
                "failure_reason": "Should have requested clarification for Washington State vs DC",
                "notes": "System made assumption instead of asking for clarification"
            },
            {
                "query_id": "Q05",
                "query_text": "What's the average teacher salary in Texas?",
                "query_category": "complex_occupation",
                "mcp_tool_called": "search_census_knowledge, get_demographic_data",
                "mcp_parameters": json.dumps({"multiple_attempts": "tried knowledge search, then wrong variables"}),
                "mcp_success": False,
                "final_answer": "System failed to find teacher-specific salary data",
                "census_variables": "B19013_001 (incorrect, should be B24121_157E)",
                "margin_of_error": None,
                "methodology_notes": "Variable mapping failure, ACS has occupation data but MCP can't access it",
                "correctness": 0.1,
                "plan_quality": 0.4,
                "tool_coordination": 0.3,
                "limitation_handling": 0.2,
                "disambiguation": 0.5,
                "methodology_guidance": 0.2,
                "passed": False,
                "failure_reason": "Failed to map natural language to correct census variables (B24121 series)",
                "notes": "Major limitation - natural language processing breaks on specialized queries"
            },
            {
                "query_id": "Q06",
                "query_text": "What's the total population of Austin, Texas? (repeat)",
                "query_category": "consistency_test",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Austin, TX", "variables": ["total population"], "year": 2023}),
                "mcp_success": True,
                "final_answer": "Austin population: 967,862 (±373) - identical to previous",
                "census_variables": "B01003_001",
                "margin_of_error": "±373 (0.0%)",
                "methodology_notes": "Consistent response, no memory of previous query",
                "correctness": 1.0,
                "plan_quality": 0.8,
                "tool_coordination": 0.8,
                "limitation_handling": 0.7,
                "disambiguation": 0.8,
                "methodology_guidance": 0.7,
                "passed": True,
                "notes": "Good consistency, but Claude complained about repetition"
            },
            # NEW TESTS Q07-Q15
            {
                "query_id": "Q07",
                "query_text": "What's the poverty rate in Detroit, Michigan?",
                "query_category": "derived_statistic",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Detroit, Michigan", "variables": ["poverty rate"], "year": 2023}),
                "mcp_success": False,
                "final_answer": "197,473 people in poverty (count, not rate)",
                "census_variables": "B17001_002",
                "margin_of_error": "±5,240 (2.7%)",
                "methodology_notes": "ACS 5-year estimates, returned count instead of percentage rate",
                "correctness": 0.3,
                "plan_quality": 0.4,
                "tool_coordination": 0.7,
                "limitation_handling": 0.2,
                "disambiguation": 0.8,
                "methodology_guidance": 0.3,
                "passed": False,
                "failure_reason": "Variable mapping failure - returned count instead of rate percentage",
                "notes": "Classic derived statistic failure - needs calculation not raw count"
            },
            {
                "query_id": "Q08",
                "query_text": "What's the unemployment rate and median age in Cleveland, Ohio?",
                "query_category": "multi_variable_mixed",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Cleveland, Ohio", "variables": ["unemployment rate", "median age"], "year": 2023}),
                "mcp_success": False,
                "final_answer": "177,863 (invalid unemployment), 36.3 years (correct median age)",
                "census_variables": "B01001_002 (wrong), B01002_001 (correct)",
                "margin_of_error": "±2,056 (1.2%), ±0.5 (1.4%)",
                "methodology_notes": "Partial success - age correct, unemployment mapped to population count",
                "correctness": 0.5,
                "plan_quality": 0.4,
                "tool_coordination": 0.7,
                "limitation_handling": 0.3,
                "disambiguation": 0.8,
                "methodology_guidance": 0.4,
                "passed": False,
                "failure_reason": "Unemployment rate mapped to wrong variable (population count)",
                "notes": "50% success rate - demonstrates inconsistent variable mapping"
            },
            {
                "query_id": "Q09",
                "query_text": "What's the population of Springfield?",
                "query_category": "geographic_ambiguity",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Springfield", "variables": ["total population"], "year": 2023}),
                "mcp_success": False,
                "final_answer": "8,157 (population of unknown Springfield)",
                "census_variables": "B01003_001",
                "margin_of_error": "±28 (0.3%)",
                "methodology_notes": "Data accurate but for unknown Springfield - critical disambiguation failure",
                "correctness": 0.1,
                "plan_quality": 0.3,
                "tool_coordination": 0.7,
                "limitation_handling": 0.1,
                "disambiguation": 0.0,
                "methodology_guidance": 0.2,
                "passed": False,
                "failure_reason": "Arbitrary Springfield selection without clarification (30+ cities named Springfield)",
                "notes": "Data technically accurate but meaningless without knowing which Springfield"
            },
            {
                "query_id": "Q10",
                "query_text": "What's the median salary for software developers in Seattle?",
                "query_category": "occupation_specific",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Seattle", "variables": ["median salary for software developers"], "year": 2023}),
                "mcp_success": False,
                "final_answer": "$121,984 (incorrect - household income, not developer salary)",
                "census_variables": "B19013_001 (wrong - household income, not occupation earnings)",
                "margin_of_error": "±1,723 (1.4%)",
                "methodology_notes": "Severe mapping failure - occupation query mapped to household income",
                "correctness": 0.2,
                "plan_quality": 0.3,
                "tool_coordination": 0.7,
                "limitation_handling": 0.1,
                "disambiguation": 0.8,
                "methodology_guidance": 0.2,
                "passed": False,
                "failure_reason": "Occupation salary mapped to household income variable",
                "notes": "Should use B24xxx series or route to BLS for occupation-specific data"
            },
            {
                "query_id": "Q11",
                "query_text": "How many people don't have health insurance in Houston, Texas?",
                "query_category": "health_insurance",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Houston, Texas", "variables": ["people without health insurance"], "year": 2023}),
                "mcp_success": False,
                "final_answer": "2,300,419 (total population, not uninsured count)",
                "census_variables": "B01003_001 (wrong - total population, not health insurance)",
                "margin_of_error": "±172 (0.0%)",
                "methodology_notes": "Complete variable mapping failure - health insurance → total population",
                "correctness": 0.0,
                "plan_quality": 0.2,
                "tool_coordination": 0.7,
                "limitation_handling": 0.0,
                "disambiguation": 0.8,
                "methodology_guidance": 0.1,
                "passed": False,
                "failure_reason": "Health insurance query mapped to total population variable",
                "notes": "Should use B27xxx series for health insurance coverage data"
            },
            {
                "query_id": "Q12",
                "query_text": "What's the homeownership rate in Atlanta, Georgia?",
                "query_category": "housing_rate",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Atlanta, Georgia", "variables": ["homeownership rate"], "year": 2023}),
                "mcp_success": False,
                "final_answer": "420,600 (housing count, not rate percentage)",
                "census_variables": "B25077_001 (wrong - housing count/value, not ownership rate)",
                "margin_of_error": "±11,515 (2.7%)",
                "methodology_notes": "Rate query returned count instead of percentage",
                "correctness": 0.2,
                "plan_quality": 0.3,
                "tool_coordination": 0.7,
                "limitation_handling": 0.2,
                "disambiguation": 0.8,
                "methodology_guidance": 0.2,
                "passed": False,
                "failure_reason": "Homeownership rate mapped to housing count instead of percentage",
                "notes": "Should use B25003 series to calculate ownership percentage"
            },
            {
                "query_id": "Q13",
                "query_text": "What was the population growth in Austin from 2020 to 2023?",
                "query_category": "temporal_comparison",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Austin, Texas", "variables": ["population growth"], "year": 2023}),
                "mcp_success": False,
                "final_answer": "967,862 (2023 population, not growth calculation)",
                "census_variables": "B01003_001 (correct variable, wrong approach)",
                "margin_of_error": "±373 (0.0%)",
                "methodology_notes": "Cannot handle temporal comparisons - single year data only",
                "correctness": 0.1,
                "plan_quality": 0.2,
                "tool_coordination": 0.6,
                "limitation_handling": 0.1,
                "disambiguation": 0.7,
                "methodology_guidance": 0.2,
                "passed": False,
                "failure_reason": "No temporal comparison capability - returned single-year data",
                "notes": "Needs multi-year query capability or route to Census population estimates"
            },
            {
                "query_id": "Q14",
                "query_text": "What's the poverty rate in census tract 1001 in Baltimore?",
                "query_category": "small_geography",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "census tract 1001 in Baltimore", "variables": ["poverty rate"], "year": 2023}),
                "mcp_success": False,
                "final_answer": "Error - geographic parsing failure",
                "census_variables": None,
                "margin_of_error": None,
                "methodology_notes": "Cannot parse census tract geography without proper FIPS codes",
                "correctness": 0.0,
                "plan_quality": 0.1,
                "tool_coordination": 0.0,
                "limitation_handling": 0.3,
                "disambiguation": 0.2,
                "methodology_guidance": 0.2,
                "passed": False,
                "failure_reason": "Geographic parsing failure for census tract specification",
                "notes": "Needs proper geographic hierarchy handling for sub-city units"
            },
            {
                "query_id": "Q15",
                "query_text": "What's the crime rate in Denver?",
                "query_category": "data_boundary",
                "mcp_tool_called": "get_demographic_data",
                "mcp_parameters": json.dumps({"location": "Denver", "variables": ["crime rate"], "year": 2023}),
                "mcp_success": False,
                "final_answer": "API error - crime data not in Census",
                "census_variables": None,
                "margin_of_error": None,
                "methodology_notes": "Census doesn't collect crime statistics - wrong data source",
                "correctness": 0.0,
                "plan_quality": 0.1,
                "tool_coordination": 0.0,
                "limitation_handling": 0.2,
                "disambiguation": 0.8,
                "methodology_guidance": 0.3,
                "passed": False,
                "failure_reason": "Data source error - Census doesn't have crime statistics",
                "notes": "Should route to FBI UCR or local police data, not Census"
            }
        ]
        
        # Create baseline run
        run_id = self._create_test_run("baseline", "Initial manual testing baseline from Claude Desktop")
        
        # Add all test results
        for test in baseline_tests:
            self._add_query_test(run_id, test)
        
        # Update run summary
        self._update_run_summary(run_id)
        
        print(f"✅ Added {len(baseline_tests)} baseline tests to run ID {run_id}")
        
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
            # Calculate summary stats
            cursor = conn.cursor()
            stats = cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed,
                    AVG(correctness) as avg_correctness,
                    AVG(plan_quality) as avg_plan,
                    AVG(tool_coordination) as avg_coordination
                FROM query_tests WHERE run_id = ?
            ''', (run_id,)).fetchone()
            
            overall_score = stats[1] / stats[0] if stats[0] > 0 else 0
            
            # Update run record
            conn.execute('''
                UPDATE test_runs 
                SET total_queries = ?, passed_queries = ?, overall_score = ?
                WHERE id = ?
            ''', (stats[0], stats[1], overall_score, run_id))
    
    def score_run(self, run_name: str):
        """Print detailed scoring for a test run"""
        with sqlite3.connect(self.db_path) as conn:
            # Get run info
            run_info = conn.execute('''
                SELECT * FROM test_runs WHERE run_name = ?
            ''', (run_name,)).fetchone()
            
            if not run_info:
                print(f"❌ Run '{run_name}' not found")
                return
            
            # Get detailed results
            results = conn.execute('''
                SELECT * FROM query_tests WHERE run_id = ? ORDER BY query_id
            ''', (run_info[0],)).fetchall()
            
            print(f"\n{'='*70}")
            print(f"📊 CENSUS MCP EVALUATION - {run_name.upper()}")
            print(f"{'='*70}")
            print(f"Overall Score: {run_info[5]:.1%} ({run_info[6]}/{run_info[4]} passed)")
            print(f"Test Date: {run_info[3]}")
            
            # Category breakdown
            categories = {}
            for result in results:
                category = result[4]  # query_category
                if category not in categories:
                    categories[category] = {"total": 0, "passed": 0, "avg_scores": []}
                categories[category]["total"] += 1
                if len(result) > 22 and result[22]:  # passed (check bounds)
                    categories[category]["passed"] += 1
                # Add average behavioral scores (check bounds and type)
                if len(result) > 20:
                    try:
                        behavioral_scores = [float(result[i]) for i in range(15, 21)]
                        behavioral_avg = sum(behavioral_scores) / len(behavioral_scores)
                        categories[category]["avg_scores"].append(behavioral_avg)
                    except (ValueError, TypeError):
                        # Skip if any scores are not numeric
                        pass
            
            print(f"\n📈 Results by Category:")
            for category, stats in categories.items():
                pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
                avg_score = sum(stats["avg_scores"]) / len(stats["avg_scores"]) if stats["avg_scores"] else 0
                print(f"  {category}: {pct:.1f}% pass rate, {avg_score:.2f} avg behavioral score")
            
            # Detailed behavioral scores
            if results:
                try:
                    avg_correctness = sum(float(r[15]) for r in results) / len(results)
                    avg_plan = sum(float(r[16]) for r in results) / len(results)
                    avg_coordination = sum(float(r[17]) for r in results) / len(results)
                    avg_limitation = sum(float(r[18]) for r in results) / len(results)
                    avg_disambiguation = sum(float(r[19]) for r in results) / len(results)
                    avg_methodology = sum(float(r[20]) for r in results) / len(results)
                    
                    print(f"\n🎯 Behavioral Scores (Average):")
                    print(f"  Correctness: {avg_correctness:.2f}")
                    print(f"  Plan Quality: {avg_plan:.2f}")
                    print(f"  Tool Coordination: {avg_coordination:.2f}")
                    print(f"  Limitation Handling: {avg_limitation:.2f}")
                    print(f"  Disambiguation: {avg_disambiguation:.2f}")
                    print(f"  Methodology Guidance: {avg_methodology:.2f}")
                except (ValueError, TypeError):
                    print(f"\n🎯 Behavioral Scores: Error calculating averages")
            
            # Failed queries
            failures = [r for r in results if not r[22]]
            if failures:
                print(f"\n❌ Failed Queries ({len(failures)}):")
                for fail in failures:
                    print(f"  {fail[2]}: {fail[3]} - {fail[23]}")
            
            # Success stories
            successes = [r for r in results if r[22] and r[15] >= 0.8]  # High correctness
            if successes:
                print(f"\n✅ Success Stories ({len(successes)}):")
                for success in successes:
                    print(f"  {success[2]}: {success[3]} - {success[24]}")
    
    def compare_runs(self, baseline_run: str, comparison_run: str):
        """Compare two test runs"""
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
            
            print(f"\n{'='*70}")
            print(f"📊 COMPARISON: {baseline_run} vs {comparison_run}")
            print(f"{'='*70}")
            
            baseline_score = baseline[5] or 0
            comparison_score = comparison[5] or 0
            improvement = comparison_score - baseline_score
            
            print(f"{baseline_run}: {baseline_score:.1%} ({baseline[6]}/{baseline[4]})")
            print(f"{comparison_run}: {comparison_score:.1%} ({comparison[6]}/{comparison[4]})")
            print(f"Improvement: {improvement:+.1%}")
            
            if improvement > 0:
                print("🎉 IMPROVEMENT DETECTED!")
            elif improvement < 0:
                print("⚠️ REGRESSION DETECTED!")
            else:
                print("➡️ NO CHANGE")
    
    def export_failing_queries(self, run_name: str) -> List[Dict]:
        """Export failing queries for improvement focus"""
        with sqlite3.connect(self.db_path) as conn:
            run_info = conn.execute('''
                SELECT id FROM test_runs WHERE run_name = ?
            ''', (run_name,)).fetchone()
            
            if not run_info:
                return []
            
            failures = conn.execute('''
                SELECT query_text, failure_reason, notes, correctness, plan_quality
                FROM query_tests 
                WHERE run_id = ? AND passed = 0
                ORDER BY correctness ASC
            ''', (run_info[0],)).fetchall()
            
            return [
                {
                    "query": f[0],
                    "failure_reason": f[1],
                    "notes": f[2],
                    "correctness": f[3],
                    "plan_quality": f[4]
                }
                for f in failures
            ]

def main():
    parser = argparse.ArgumentParser(description="Census MCP Evaluation Database")
    parser.add_argument("--add-baseline", action="store_true", help="Add baseline test data")
    parser.add_argument("--score-run", help="Score and display run results")
    parser.add_argument("--compare-runs", nargs=2, help="Compare two runs")
    parser.add_argument("--export-failures", help="Export failing queries from run")
    
    args = parser.parse_args()
    
    evaluator = CensusMCPEvaluator()
    
    if args.add_baseline:
        evaluator.add_baseline_data()
    
    elif args.score_run:
        evaluator.score_run(args.score_run)
    
    elif args.compare_runs:
        evaluator.compare_runs(args.compare_runs[0], args.compare_runs[1])
    
    elif args.export_failures:
        failures = evaluator.export_failing_queries(args.export_failures)
        print(f"\n🎯 Failed Queries from {args.export_failures}:")
        for i, failure in enumerate(failures, 1):
            print(f"{i}. {failure['query']}")
            print(f"   Reason: {failure['failure_reason']}")
            print(f"   Score: {failure['correctness']:.2f} correctness, {failure['plan_quality']:.2f} plan")
            print()
    
    else:
        print("Use --help to see available commands")

if __name__ == "__main__":
    main()
