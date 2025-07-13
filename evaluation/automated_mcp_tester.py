#!/usr/bin/env python3
"""
Automated Census MCP Tester
Connects directly to MCP server and runs benchmark queries programmatically

Usage (run from evaluation/ folder):
    python automated_mcp_tester.py --run-name v2.1-python --description "Python-only implementation"
    python automated_mcp_tester.py --run-name v2.1-python --single-query "What's the poverty rate in Detroit?"
"""

import asyncio
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import sys
import re
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import the evaluation database
from evaluation_db import CensusMCPEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPDirectClient:
    """Direct client to test MCP server without Claude Desktop"""
    
    def __init__(self, server_path: str, python_path: str = None):
        # Adjust paths relative to evaluation folder
        if not Path(server_path).is_absolute():
            # Assume relative to project root from evaluation folder
            self.server_path = Path(__file__).parent.parent / server_path
        else:
            self.server_path = Path(server_path)
            
        self.python_path = python_path or "/opt/anaconda3/envs/census-mcp/bin/python"
        self.server_process = None
        
    async def start_server(self):
        """Start the MCP server process"""
        try:
            # Set up environment for MCP server
            project_root = Path(__file__).parent.parent
            env = {
                **os.environ,  # Inherit current environment
                'PYTHONPATH': str(project_root / 'src'),
                'PYTHONUNBUFFERED': '1',
                'LOG_LEVEL': 'INFO',
                'CENSUS_API_KEY': os.getenv('CENSUS_API_KEY', '')
            }
            
            self.server_process = await asyncio.create_subprocess_exec(
                self.python_path,
                str(self.server_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            logger.info(f"Started MCP server: PID {self.server_process.pid}")
            
            # Wait a moment for server to initialize
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    async def send_mcp_message(self, message: Dict) -> Dict:
        """Send MCP message and get response"""
        if not self.server_process:
            raise RuntimeError("Server not started")
        
        try:
            # Send message
            message_json = json.dumps(message) + '\n'
            self.server_process.stdin.write(message_json.encode())
            await self.server_process.stdin.drain()
            
            # Read response
            response_line = await self.server_process.stdout.readline()
            response = json.loads(response_line.decode().strip())
            
            return response
            
        except Exception as e:
            logger.error(f"MCP communication error: {e}")
            return {"error": str(e)}
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Call a specific MCP tool"""
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self.send_mcp_message(message)
        return response
    
    async def stop_server(self):
        """Stop the MCP server"""
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()
            logger.info("MCP server stopped")

class BenchmarkQueries:
    """Benchmark query definitions"""
    
    QUERIES = [
        {
            "query_id": "Q01",
            "query_text": "What's the median household income in Baltimore, Maryland?",
            "query_category": "basic_demographic",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Baltimore, Maryland",
                "variables": ["median household income"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q02", 
            "query_text": "What's the total population and median age in Austin, Texas?",
            "query_category": "multi_variable",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Austin, Texas",
                "variables": ["total population", "median age"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q03",
            "query_text": "What's the population of 123 Main Street?",
            "query_category": "limitation_handling",
            "tool_name": "get_demographic_data", 
            "arguments": {
                "location": "123 Main Street",
                "variables": ["population"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q04",
            "query_text": "What's the median income in Washington?",
            "query_category": "disambiguation",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Washington",
                "variables": ["median income"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q05",
            "query_text": "What's the average teacher salary in Texas?",
            "query_category": "complex_occupation",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Texas",
                "variables": ["average teacher salary"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q06",
            "query_text": "What's the total population of Austin, Texas? (repeat)",
            "query_category": "consistency_test",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Austin, Texas",
                "variables": ["total population"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q07",
            "query_text": "What's the poverty rate in Detroit, Michigan?",
            "query_category": "derived_statistic",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Detroit, Michigan",
                "variables": ["poverty rate"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q08",
            "query_text": "What's the unemployment rate and median age in Cleveland, Ohio?",
            "query_category": "multi_variable_mixed",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Cleveland, Ohio",
                "variables": ["unemployment rate", "median age"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q09",
            "query_text": "What's the population of Springfield?",
            "query_category": "geographic_ambiguity",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Springfield",
                "variables": ["population"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q10",
            "query_text": "What's the median salary for software developers in Seattle?",
            "query_category": "occupation_specific",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Seattle",
                "variables": ["median salary for software developers"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q11",
            "query_text": "How many people don't have health insurance in Houston, Texas?",
            "query_category": "health_insurance",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Houston, Texas",
                "variables": ["people without health insurance"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q12",
            "query_text": "What's the homeownership rate in Atlanta, Georgia?",
            "query_category": "housing_rate",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Atlanta, Georgia",
                "variables": ["homeownership rate"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q13",
            "query_text": "What was the population growth in Austin from 2020 to 2023?",
            "query_category": "temporal_comparison",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Austin, Texas",
                "variables": ["population growth"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q14",
            "query_text": "What's the poverty rate in census tract 1001 in Baltimore?",
            "query_category": "small_geography",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "census tract 1001 in Baltimore",
                "variables": ["poverty rate"],
                "year": 2023,
                "survey": "acs5"
            }
        },
        {
            "query_id": "Q15",
            "query_text": "What's the crime rate in Denver?",
            "query_category": "data_boundary",
            "tool_name": "get_demographic_data",
            "arguments": {
                "location": "Denver",
                "variables": ["crime rate"],
                "year": 2023,
                "survey": "acs5"
            }
        }
    ]

class AutomatedMCPTester:
    """Main tester class"""
    
    def __init__(self, server_path: str, python_path: str = None):
        self.client = MCPDirectClient(server_path, python_path)
        self.evaluator = CensusMCPEvaluator()
        
    async def run_single_query(self, query_def: Dict) -> Dict:
        """Run a single query and return results"""
        logger.info(f"Running query {query_def['query_id']}: {query_def['query_text']}")
        
        try:
            # Call the MCP tool
            response = await self.client.call_tool(
                query_def['tool_name'],
                query_def['arguments']
            )
            
            # Parse response
            if 'error' in response:
                return {
                    'success': False,
                    'error': response['error'],
                    'raw_response': response
                }
            
            # Extract result content
            result_content = ""
            if 'result' in response and isinstance(response['result'], list):
                for item in response['result']:
                    if item.get('type') == 'text':
                        result_content += item.get('text', '')
            
            return {
                'success': True,
                'content': result_content,
                'raw_response': response
            }
            
        except Exception as e:
            logger.error(f"Query {query_def['query_id']} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'raw_response': None
            }
    
    def analyze_response(self, query_def: Dict, result: Dict) -> Dict:
        """Analyze response and score it"""
        
        # Basic success check
        if not result['success']:
            return {
                'passed': False,
                'correctness': 0.0,
                'plan_quality': 0.0,
                'tool_coordination': 0.0,
                'limitation_handling': 0.0,
                'disambiguation': 0.0,
                'methodology_guidance': 0.0,
                'failure_reason': result.get('error', 'Unknown error'),
                'notes': 'Query execution failed'
            }
        
        content = result.get('content', '')
        
        # Extract key information
        has_percentage = '%' in content
        has_margin_error = '±' in content
        has_census_code = re.search(r'B\d{5}_\d{3}', content)
        has_methodology = any(term in content.lower() for term in ['acs', 'survey', 'estimate', 'confidence'])
        
        # Score based on query type
        scores = self._score_by_category(query_def, content, result)
        
        return scores
    
    def _score_by_category(self, query_def: Dict, content: str, result: Dict) -> Dict:
        """Score response based on query category"""
        
        category = query_def['query_category']
        base_scores = {
            'passed': False,
            'correctness': 0.0,
            'plan_quality': 0.5,
            'tool_coordination': 0.7,  # Tool was called
            'limitation_handling': 0.5,
            'disambiguation': 0.5,
            'methodology_guidance': 0.3,
            'failure_reason': '',
            'notes': ''
        }
        
        # Category-specific scoring
        if category == 'derived_statistic' and 'poverty rate' in query_def['query_text']:
            # Check if we got a percentage (our fix!)
            if '%' in content:
                base_scores['correctness'] = 1.0
                base_scores['passed'] = True
                base_scores['notes'] = 'SUCCESS: Returns percentage rate instead of count'
            else:
                base_scores['correctness'] = 0.3
                base_scores['failure_reason'] = 'Returned count instead of percentage rate'
                
        elif category == 'basic_demographic':
            # Should work well
            if '±' in content and '$' in content:  # Has MOE and currency
                base_scores['correctness'] = 1.0
                base_scores['passed'] = True
            else:
                base_scores['correctness'] = 0.5
                
        elif category == 'limitation_handling':
            # Should reject impossible queries
            if 'error' in content.lower() or 'cannot' in content.lower():
                base_scores['correctness'] = 1.0
                base_scores['limitation_handling'] = 1.0
                base_scores['passed'] = True
            else:
                base_scores['limitation_handling'] = 0.0
                
        # Add methodology guidance scoring
        if any(term in content.lower() for term in ['acs', 'margin of error', 'confidence']):
            base_scores['methodology_guidance'] = 0.8
            
        return base_scores
    
    async def run_benchmark_suite(self, run_name: str, description: str = ""):
        """Run all benchmark queries"""
        
        logger.info(f"Starting benchmark suite: {run_name}")
        
        # Start MCP server
        if not await self.client.start_server():
            logger.error("Failed to start MCP server")
            return
        
        try:
            # Create test run
            run_id = self.evaluator._create_test_run(run_name, description)
            
            results = []
            
            # Run each query
            for query_def in BenchmarkQueries.QUERIES:
                logger.info(f"Testing {query_def['query_id']}: {query_def['query_text']}")
                
                # Execute query
                result = await self.run_single_query(query_def)
                
                # Analyze and score
                scores = self.analyze_response(query_def, result)
                
                # Build test data
                # Build test data - ensure no dict values, force correct types
                test_data = {
                    'query_id': str(query_def['query_id']),
                    'query_text': str(query_def['query_text']),
                    'query_category': str(query_def['query_category']),
                    'mcp_tool_called': str(query_def['tool_name']),
                    'mcp_parameters': json.dumps(query_def['arguments']),
                    'mcp_success': bool(result['success']),
                    'final_answer': str(result.get('content', ''))[:500],
                    'census_variables': str(self._extract_census_vars(result.get('content', '')) or ''),
                    'margin_of_error': str(self._extract_moe(result.get('content', '')) or ''),
                    'methodology_notes': str(self._extract_methodology(result.get('content', '')) or ''),
                    'correctness': float(scores['correctness']),
                    'plan_quality': float(scores['plan_quality']),
                    'tool_coordination': float(scores['tool_coordination']),
                    'limitation_handling': float(scores['limitation_handling']),
                    'disambiguation': float(scores['disambiguation']),
                    'methodology_guidance': float(scores['methodology_guidance']),
                    'passed': bool(scores['passed']),
                    'failure_reason': str(scores['failure_reason']),
                    'notes': str(scores['notes'])
                }
                
                
                # Add to database
                self.evaluator._add_query_test(run_id, test_data)
                results.append(test_data)
                
                # Brief delay between queries
                await asyncio.sleep(1)
            
            # Update run summary
            self.evaluator._update_run_summary(run_id)
            
            # Print summary
            passed = sum(1 for r in results if r['passed'])
            total = len(results)
            logger.info(f"Benchmark complete: {passed}/{total} passed ({passed/total:.1%})")
            
        finally:
            await self.client.stop_server()
    
    def _extract_census_vars(self, content: str) -> Optional[str]:
        """Extract Census variable codes from response"""
        matches = re.findall(r'B\d{5}_\d{3}[A-Z]*', content)
        return ', '.join(matches) if matches else None
    
    def _extract_moe(self, content: str) -> Optional[str]:
        """Extract margin of error from response"""
        moe_match = re.search(r'±[^)]+\)', content)
        return moe_match.group() if moe_match else None
    
    def _extract_methodology(self, content: str) -> Optional[str]:
        """Extract methodology notes"""
        if 'acs' in content.lower():
            return "ACS estimates mentioned"
        return None

async def main():
    parser = argparse.ArgumentParser(description="Automated Census MCP Tester")
    parser.add_argument("--server-path", default="src/census_mcp_server.py", help="Path to MCP server (relative to project root)")
    parser.add_argument("--python-path", default="/opt/anaconda3/envs/census-mcp/bin/python", help="Path to Python executable")
    parser.add_argument("--run-name", required=True, help="Test run name")
    parser.add_argument("--description", default="", help="Test run description")
    parser.add_argument("--single-query", help="Run single query instead of full suite")
    
    args = parser.parse_args()
    
    # Ensure we're in the evaluation folder
    if not Path("evaluation_db.py").exists():
        print("❌ Error: Run this script from the evaluation/ folder")
        print("   cd /Users/brock/Documents/GitHub/census-mcp-server/evaluation")
        print("   python automated_mcp_tester.py --run-name v2.1-python")
        return
    
    tester = AutomatedMCPTester(args.server_path, args.python_path)
    
    if args.single_query:
        # Find matching query
        matching_queries = [q for q in BenchmarkQueries.QUERIES if args.single_query.lower() in q['query_text'].lower()]
        if matching_queries:
            query_def = matching_queries[0]
            await tester.client.start_server()
            try:
                result = await tester.run_single_query(query_def)
                print(f"Query: {query_def['query_text']}")
                print(f"Success: {result['success']}")
                print(f"Content: {result.get('content', 'No content')}")
            finally:
                await tester.client.stop_server()
        else:
            print(f"No query found matching: {args.single_query}")
    else:
        # Run full benchmark suite
        await tester.run_benchmark_suite(args.run_name, args.description)

if __name__ == "__main__":
    asyncio.run(main())
