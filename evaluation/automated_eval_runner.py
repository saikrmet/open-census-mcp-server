#!/usr/bin/env python3
"""
Automated Census MCP Evaluation Runner
Uses Anthropic API to evaluate MCP performance against benchmark queries

Usage:
    python automated_eval_runner.py --run baseline
    python automated_eval_runner.py --run experiment --compare 1
"""

import json
import sqlite3
import time
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPEvaluator:
    def __init__(self, db_path: str = "evaluation.db"):
        self.db_path = Path(db_path)
        
        # Load API key from .env
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=self.api_key)
        
        # Initialize database
        self._init_database()
        
        # Load benchmark queries
        self.benchmark_queries = self._load_benchmark_queries()
        
    def _init_database(self):
        """Initialize SQLite database with evaluation schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    system_version TEXT NOT NULL,
                    description TEXT,
                    total_queries INTEGER,
                    overall_score REAL,
                    passed_queries INTEGER,
                    execution_time_seconds REAL
                );
                
                CREATE TABLE IF NOT EXISTS query_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    query_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    expected_result TEXT NOT NULL,
                    
                    full_response TEXT,
                    mcp_calls_json TEXT,
                    reasoning_chain TEXT,
                    
                    plan_quality REAL,
                    tool_coordination REAL,
                    adaptation REAL,
                    reasoning_clarity REAL,
                    
                    correctness REAL,
                    passed BOOLEAN,
                    failure_reason TEXT,
                    
                    execution_time_ms INTEGER,
                    
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_run_id ON query_results (run_id);
                CREATE INDEX IF NOT EXISTS idx_query_id ON query_results (query_id);
            ''')
    
    def _load_benchmark_queries(self) -> List[Dict]:
        """Load benchmark queries"""
        return [
            {"id": "Q01", "type": "variable", "category": "core_demographics", "query": "total population", "truth": ["B01003_001E"]},
            {"id": "Q02", "type": "variable", "category": "core_demographics", "query": "median household income", "truth": ["B19013_001E"]},
            {"id": "Q03", "type": "variable", "category": "core_demographics", "query": "poverty count", "truth": ["B17001_002E"]},
            {"id": "Q04", "type": "variable", "category": "core_demographics", "query": "race breakdown", "truth": ["B02001_002E", "B02001_003E", "B02001_004E"]},
            {"id": "Q05", "type": "calculation", "category": "core_demographics", "query": "bachelor's degree or higher", "truth": ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]},
            {"id": "Q06", "type": "variable", "category": "core_demographics", "query": "without health insurance", "truth": ["B27001_017E"]},
            {"id": "Q07", "type": "variable", "category": "core_demographics", "query": "median age", "truth": ["B01002_001E"]},
            {"id": "Q08", "type": "variable", "category": "housing_transport", "query": "owner vs renter occupied", "truth": ["B25003_002E", "B25003_003E"]},
            {"id": "Q09", "type": "variable", "category": "housing_transport", "query": "average commute time", "truth": ["B08303_001E"]},
            {"id": "Q10", "type": "geographic", "category": "geographic_complexity", "query": "most educated state", "truth": "state_comparison_B15003"},
            {"id": "Q11", "type": "disambiguation", "category": "geographic_complexity", "query": "Washington income", "truth": "clarification_needed"},
            {"id": "Q12", "type": "caveat", "category": "edge_cases", "query": "teacher salary", "truth": "educator_aggregation_caveat"},
            {"id": "Q13", "type": "limitation", "category": "confusing", "query": "population of 123 Main Street", "truth": "address_level_not_available"},
            {"id": "Q14", "type": "limitation", "category": "confusing", "query": "undocumented immigrants in Dallas", "truth": "immigration_status_not_collected"},
            {"id": "Q15", "type": "concept", "category": "confusing", "query": "people who are both Black and Hispanic", "truth": "race_ethnicity_separate_variables"},
            {"id": "Q16", "type": "clarification", "category": "confusing", "query": "average mean income", "truth": "median_vs_mean_clarification"},
            {"id": "Q17", "type": "temporal", "category": "confusing", "query": "unemployment rate for March 2025", "truth": "monthly_data_not_available"},
            {"id": "Q18", "type": "limitation", "category": "confusing", "query": "people with Medicaid coverage", "truth": "public_insurance_aggregated"},
            {"id": "Q19", "type": "temporal", "category": "confusing", "query": "population in 2010", "truth": "decennial_vs_acs_confusion"},
            {"id": "Q20", "type": "limitation", "category": "confusing", "query": "commute by Uber or rideshare", "truth": "rideshare_not_specified"}
        ]
    
    def run_evaluation(self, run_name: str, description: str = "") -> int:
        """Run complete evaluation and return run_id"""
        logger.info(f"Starting evaluation run: {run_name}")
        logger.info(f"Testing {len(self.benchmark_queries)} queries")
        
        # Create run record
        run_id = self._create_run_record(run_name, description)
        start_time = time.time()
        
        results = []
        for i, query in enumerate(self.benchmark_queries, 1):
            logger.info(f"Query {i}/{len(self.benchmark_queries)}: {query['query']}")
            
            query_start = time.time()
            result = self._evaluate_single_query(query, run_id)
            result["execution_time_ms"] = (time.time() - query_start) * 1000
            
            self._save_query_result(result, run_id)
            results.append(result)
            
            # Log result
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            logger.info(f"  {status} (correctness: {result['correctness']:.2f}, plan: {result['plan_quality']:.2f})")
        
        # Update run record with summary
        total_time = time.time() - start_time
        self._update_run_summary(run_id, results, total_time)
        
        return run_id
    
    def _create_run_record(self, system_version: str, description: str) -> int:
        """Create evaluation run record and return run_id"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO evaluation_runs (timestamp, system_version, description, total_queries)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), system_version, description, len(self.benchmark_queries)))
            return cursor.lastrowid
    
    def _evaluate_single_query(self, query: Dict, run_id: int) -> Dict:
        """Evaluate a single query using Anthropic API with MCP"""
        try:
            # Create the API call with MCP-specific prompt
            prompt = self._create_mcp_prompt(query["query"])
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract response content
            response_text = response.content[0].text if response.content else ""
            
            # Parse MCP function calls (if any)
            mcp_calls = self._extract_mcp_calls(response_text)
            
            # Score the response using both traditional and behavioral metrics
            scores = self._score_response_comprehensive(response_text, query, mcp_calls)
            
            return {
                "query_id": query["id"],
                "query_text": query["query"],
                "query_type": query["type"],
                "category": query["category"],
                "expected_result": json.dumps(query["truth"]),
                "full_response": response_text,
                "mcp_calls_json": json.dumps(mcp_calls),
                "reasoning_chain": self._extract_reasoning_chain(response_text),
                **scores
            }
            
        except Exception as e:
            logger.error(f"Error evaluating query {query['id']}: {str(e)}")
            return self._create_error_result(query, str(e))
    
    def _create_mcp_prompt(self, user_query: str) -> str:
        """Create prompt that encourages MCP tool usage"""
        return f"""You have access to Census MCP tools that can help answer demographic questions. Please use the appropriate Census tools to answer this question:

"{user_query}"

If the question involves demographic data, locations, or Census concepts, please use the Census MCP tools (get_demographic_data, compare_locations, or search_census_knowledge) to provide an accurate, well-sourced answer.

For questions that can't be answered with Census data, please explain why and suggest appropriate alternatives."""
    
    def _extract_mcp_calls(self, response_text: str) -> List[Dict]:
        """Extract MCP function calls from response text"""
        # Look for common MCP function patterns
        mcp_functions = ["get_demographic_data", "compare_locations", "search_census_knowledge"]
        calls = []
        
        for func in mcp_functions:
            if func in response_text:
                calls.append({"function": func, "detected": True})
        
        return calls
    
    def _extract_reasoning_chain(self, response_text: str) -> str:
        """Extract reasoning chain from response"""
        # Simple extraction - look for step-by-step reasoning
        lines = response_text.split('\n')
        reasoning_lines = []
        
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in ['first', 'then', 'next', 'step', 'because', 'therefore']):
                reasoning_lines.append(line)
        
        return ' | '.join(reasoning_lines[:5])  # First 5 reasoning steps
    
    def _score_response_comprehensive(self, response_text: str, query: Dict, mcp_calls: List[Dict]) -> Dict:
        """Score response across multiple dimensions"""
        # Traditional correctness scoring
        correctness = self._score_correctness(response_text, query)
        
        # Behavioral scoring (from agent evaluation paper)
        plan_quality = self._score_plan_quality(response_text, query, mcp_calls)
        tool_coordination = self._score_tool_coordination(mcp_calls, query)
        adaptation = self._score_adaptation(response_text, query)
        reasoning_clarity = self._score_reasoning_clarity(response_text)
        
        # Overall pass/fail
        passed = correctness >= 0.7 and plan_quality >= 0.6
        failure_reason = "" if passed else self._get_failure_reason(correctness, plan_quality, query)
        
        return {
            "correctness": correctness,
            "plan_quality": plan_quality,
            "tool_coordination": tool_coordination,
            "adaptation": adaptation,
            "reasoning_clarity": reasoning_clarity,
            "passed": passed,
            "failure_reason": failure_reason
        }
    
    def _score_correctness(self, response_text: str, query: Dict) -> float:
        """Score traditional correctness"""
        query_type = query["type"]
        expected = query["truth"]
        response_lower = response_text.lower()
        
        if query_type == "variable":
            # Check if expected variable IDs are mentioned
            if isinstance(expected, list):
                found = sum(1 for var in expected if var.lower() in response_lower)
                return min(found / len(expected), 1.0)
            else:
                return 1.0 if str(expected).lower() in response_lower else 0.0
        
        elif query_type == "limitation":
            # Check for limitation acknowledgment
            limitation_words = ["not available", "limitation", "cannot", "not collected"]
            return 1.0 if any(word in response_lower for word in limitation_words) else 0.0
        
        elif query_type == "disambiguation":
            # Check for clarification request
            clarification_words = ["clarify", "which", "specify", "need more", "ambiguous"]
            return 1.0 if any(word in response_lower for word in clarification_words) else 0.0
        
        elif query_type == "caveat":
            # Check for appropriate warning/caveat
            caveat_words = ["caveat", "limitation", "note that", "however", "but"]
            return 1.0 if any(word in response_lower for word in caveat_words) else 0.0
        
        else:
            # Default scoring for other types
            return 0.8 if len(response_text) > 100 else 0.3
    
    def _score_plan_quality(self, response_text: str, query: Dict, mcp_calls: List[Dict]) -> float:
        """Score quality of strategy/plan chosen"""
        query_type = query["type"]
        
        # Check if appropriate MCP tools were used
        if query_type in ["variable", "geographic", "calculation"]:
            # Should use demographic tools
            demographic_tools = ["get_demographic_data", "compare_locations"]
            if any(call["function"] in demographic_tools for call in mcp_calls):
                return 0.9
            elif "search_census_knowledge" in [call["function"] for call in mcp_calls]:
                return 0.7  # Knowledge search is reasonable but not optimal
            else:
                return 0.3  # No appropriate tools used
        
        elif query_type in ["limitation", "caveat", "concept"]:
            # Should use knowledge search or explain without data
            if "search_census_knowledge" in [call["function"] for call in mcp_calls]:
                return 0.9
            elif len(response_text) > 200:  # Substantial explanation without tools
                return 0.7
            else:
                return 0.4
        
        else:
            # For other types, check if response is substantive
            return 0.8 if len(response_text) > 150 else 0.5
    
    def _score_tool_coordination(self, mcp_calls: List[Dict], query: Dict) -> float:
        """Score how well tools were coordinated"""
        if not mcp_calls:
            return 0.0
        
        # Simple coordination scoring
        if len(mcp_calls) == 1:
            return 0.8  # Single appropriate tool
        elif len(mcp_calls) > 1:
            return 0.9  # Multiple tools suggest good coordination
        else:
            return 0.0
    
    def _score_adaptation(self, response_text: str, query: Dict) -> float:
        """Score adaptation to query challenges"""
        response_lower = response_text.lower()
        
        # Check for adaptive language
        adaptive_phrases = [
            "instead", "alternative", "however", "but", "although",
            "different approach", "another way", "let me try"
        ]
        
        adaptation_score = sum(1 for phrase in adaptive_phrases if phrase in response_lower)
        return min(adaptation_score / 3.0, 1.0)  # Max score with 3+ adaptive phrases
    
    def _score_reasoning_clarity(self, response_text: str) -> float:
        """Score clarity of reasoning chain"""
        # Simple heuristics for reasoning clarity
        reasoning_indicators = [
            "because", "therefore", "first", "then", "next", "finally",
            "this means", "as a result", "consequently"
        ]
        
        response_lower = response_text.lower()
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        
        # Score based on structure and reasoning indicators
        structure_score = 0.3 if len(response_text) > 200 else 0.1
        reasoning_score = min(reasoning_count / 5.0, 0.7)  # Max 0.7 for reasoning indicators
        
        return structure_score + reasoning_score
    
    def _get_failure_reason(self, correctness: float, plan_quality: float, query: Dict) -> str:
        """Generate failure reason description"""
        reasons = []
        
        if correctness < 0.7:
            reasons.append(f"Low correctness ({correctness:.2f})")
        
        if plan_quality < 0.6:
            reasons.append(f"Poor plan quality ({plan_quality:.2f})")
        
        return "; ".join(reasons) if reasons else "Unknown failure"
    
    def _create_error_result(self, query: Dict, error_msg: str) -> Dict:
        """Create error result record"""
        return {
            "query_id": query["id"],
            "query_text": query["query"],
            "query_type": query["type"],
            "category": query["category"],
            "expected_result": json.dumps(query["truth"]),
            "full_response": f"ERROR: {error_msg}",
            "mcp_calls_json": "[]",
            "reasoning_chain": "",
            "correctness": 0.0,
            "plan_quality": 0.0,
            "tool_coordination": 0.0,
            "adaptation": 0.0,
            "reasoning_clarity": 0.0,
            "passed": False,
            "failure_reason": f"System error: {error_msg}"
        }
    
    def _save_query_result(self, result: Dict, run_id: int):
        """Save query result to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO query_results (
                    run_id, query_id, query_text, query_type, category, expected_result,
                    full_response, mcp_calls_json, reasoning_chain,
                    plan_quality, tool_coordination, adaptation, reasoning_clarity,
                    correctness, passed, failure_reason, execution_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, result["query_id"], result["query_text"], result["query_type"],
                result["category"], result["expected_result"], result["full_response"],
                result["mcp_calls_json"], result["reasoning_chain"],
                result["plan_quality"], result["tool_coordination"], result["adaptation"],
                result["reasoning_clarity"], result["correctness"], result["passed"],
                result["failure_reason"], result["execution_time_ms"]
            ))
    
    def _update_run_summary(self, run_id: int, results: List[Dict], total_time: float):
        """Update run record with summary statistics"""
        passed_count = sum(1 for r in results if r["passed"])
        overall_score = passed_count / len(results) if results else 0
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE evaluation_runs 
                SET overall_score = ?, passed_queries = ?, execution_time_seconds = ?
                WHERE id = ?
            ''', (overall_score, passed_count, total_time, run_id))
    
    def print_run_summary(self, run_id: int):
        """Print summary of evaluation run"""
        with sqlite3.connect(self.db_path) as conn:
            # Get run info
            run_info = conn.execute(
                "SELECT * FROM evaluation_runs WHERE id = ?", (run_id,)
            ).fetchone()
            
            if not run_info:
                print(f"Run {run_id} not found")
                return
            
            # Get detailed results
            results = conn.execute(
                "SELECT * FROM query_results WHERE run_id = ?", (run_id,)
            ).fetchall()
            
            print(f"\n{'='*60}")
            print(f"EVALUATION RESULTS - {run_info[2]} ({run_info[1]})")
            print(f"{'='*60}")
            print(f"Overall Score: {run_info[5]:.1%} ({run_info[6]}/{run_info[4]})")
            print(f"Execution Time: {run_info[7]:.1f} seconds")
            
            # Category breakdown
            category_stats = {}
            for result in results:
                category = result[5]  # category column
                if category not in category_stats:
                    category_stats[category] = {"total": 0, "passed": 0}
                category_stats[category]["total"] += 1
                if result[15]:  # passed column
                    category_stats[category]["passed"] += 1
            
            print(f"\nBy Category:")
            for category, stats in category_stats.items():
                pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"  {category}: {pct:.1f}% ({stats['passed']}/{stats['total']})")
            
            # Average behavioral scores
            if results:
                avg_plan = sum(r[9] for r in results) / len(results)  # plan_quality
                avg_tool = sum(r[10] for r in results) / len(results)  # tool_coordination
                avg_adapt = sum(r[11] for r in results) / len(results)  # adaptation
                avg_reasoning = sum(r[12] for r in results) / len(results)  # reasoning_clarity
                
                print(f"\nBehavioral Scores (Avg):")
                print(f"  Plan Quality: {avg_plan:.2f}")
                print(f"  Tool Coordination: {avg_tool:.2f}")
                print(f"  Adaptation: {avg_adapt:.2f}")
                print(f"  Reasoning Clarity: {avg_reasoning:.2f}")
    
    def compare_runs(self, run1_id: int, run2_id: int):
        """Compare two evaluation runs"""
        with sqlite3.connect(self.db_path) as conn:
            run1 = conn.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run1_id,)).fetchone()
            run2 = conn.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run2_id,)).fetchone()
            
            if not run1 or not run2:
                print("One or both runs not found")
                return
            
            print(f"\n{'='*60}")
            print(f"COMPARISON: {run1[2]} vs {run2[2]}")
            print(f"{'='*60}")
            print(f"{run1[2]}: {run1[5]:.1%} ({run1[6]}/{run1[4]})")
            print(f"{run2[2]}: {run2[5]:.1%} ({run2[6]}/{run2[4]})")
            
            improvement = run2[5] - run1[5]
            print(f"Improvement: {improvement:+.1%}")
            print(f"Result: {'✅ IMPROVED' if improvement > 0 else '❌ REGRESSED' if improvement < 0 else '➡️ NO CHANGE'}")

def main():
    parser = argparse.ArgumentParser(description="Run Census MCP automated evaluation")
    parser.add_argument("--run", required=True, help="Run name/version (e.g., 'baseline', 'v2.1')")
    parser.add_argument("--description", default="", help="Run description")
    parser.add_argument("--compare", help="Compare with previous run ID")
    
    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run evaluation
    evaluator = MCPEvaluator()
    run_id = evaluator.run_evaluation(args.run, args.description)
    
    # Print results
    evaluator.print_run_summary(run_id)
    
    # Compare if requested
    if args.compare:
        try:
            baseline_id = int(args.compare)
            evaluator.compare_runs(baseline_id, run_id)
        except ValueError:
            print(f"Invalid comparison run ID: {args.compare}")
    
    print(f"\nResults saved to database: {evaluator.db_path}")
    print(f"Run ID: {run_id}")

if __name__ == "__main__":
    main()
