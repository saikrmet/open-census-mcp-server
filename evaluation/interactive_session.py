#!/usr/bin/env python3
"""
Interactive Census MCP Evaluation Session - ROBUST VERSION
No more data loss, handles 3000 char responses properly

Usage:
    python interactive_session.py
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from evaluation_db import CensusMCPEvaluator

class InteractiveSession:
    """Robust interactive evaluation session - no more data loss"""
    
    def __init__(self):
        self.evaluator = CensusMCPEvaluator("census_mcp_evaluation.db")
        self.state_file = Path("session_state.json")
        
        # Load benchmark queries
        with open("benchmark_queries.json", 'r') as f:
            data = json.load(f)
            self.queries = data['queries']
        
        # Standard prompt template
        self.prompt_template = """I'm systematically evaluating this Census MCP system. Please answer the following question and then provide structured evaluation data:

QUESTION: {question}

After providing your answer, please include this structured evaluation section:

=== EVALUATION DATA ===
MCP_TOOL_CALLED: [exact tool name(s) and parameters used]
CENSUS_VARIABLES: [specific variable codes like B19013_001E]
API_CALLS: [exact API call(s) used]
DATA_RETURNED: [key numbers/values from response]
MARGIN_OF_ERROR: [MOE values if provided]
METHODOLOGY_NOTES: [statistical caveats, survey info, limitations]
GEOGRAPHIC_LEVEL: [state/county/place/tract etc.]
SUCCESS: [true/false - did the MCP work as expected]
ISSUES_ENCOUNTERED: [any problems, failures, or limitations]
DISAMBIGUATION_NEEDED: [true/false - was location/question ambiguous]
ROUTING_SUGGESTION: [if other data sources recommended]
CONFIDENCE_LEVEL: [your assessment 0.0-1.0 of response quality]
=== END EVALUATION DATA ==="""
    
    def save_state(self, state: Dict):
        """Save session state with backup"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            # Create backup
            backup_file = self.state_file.with_suffix('.backup.json')
            with open(backup_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save state: {e}")
    
    def load_state(self) -> Dict:
        """Load session state with backup fallback"""
        for state_file in [self.state_file, self.state_file.with_suffix('.backup.json')]:
            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {state_file}: {e}")
        return {}
    
    def start_new_run(self):
        """Start a new evaluation run"""
        print("🏛️ Census MCP Evaluation Session")
        print("=" * 50)
        
        run_name = input("Enter run name (e.g., v2.1-semantic): ").strip()
        if not run_name:
            print("❌ Run name required")
            return False
        
        description = input("Enter description (optional): ").strip()
        
        try:
            # Create database run
            run_id = self.evaluator._create_test_run(run_name, description)
            
            # Initialize state
            state = {
                'run_name': run_name,
                'run_id': run_id,
                'current_query_index': 0,
                'completed_queries': [],
                'started_at': datetime.now().isoformat()
            }
            
            self.save_state(state)
            print(f"✅ Started evaluation run: {run_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating run: {e}")
            return False
    
    def parse_response(self, query: Dict, response_text: str) -> Dict:
        """Parse Claude's response - ROBUST VERSION"""
        
        # Extract evaluation data section
        eval_match = re.search(
            r'=== EVALUATION DATA ===(.*?)=== END EVALUATION DATA ===',
            response_text,
            re.DOTALL
        )
        
        if not eval_match:
            print("⚠️ Warning: No evaluation data section found")
            eval_section = ""
        else:
            eval_section = eval_match.group(1)
        
        # Parse fields with robust defaults
        def extract_field(field_name: str, default: str = "not provided") -> str:
            pattern = rf'{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)'
            match = re.search(pattern, eval_section, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip().strip('[]')
                return value if value else default
            return default
        
        # Extract main answer - INCREASED TO 3000 CHARS
        if eval_match:
            main_answer = response_text[:eval_match.start()].strip()
        else:
            main_answer = response_text.strip()
        
        # Truncate to 3000 chars with smart sentence boundary
        if len(main_answer) > 3000:
            truncated = main_answer[:3000]
            last_period = truncated.rfind('.')
            if last_period > 2000:  # If we can get substantial content
                main_answer = truncated[:last_period + 1] + " [TRUNCATED]"
            else:
                main_answer = truncated + " [TRUNCATED]"
        
        # Parse evaluation fields
        mcp_tool = extract_field("MCP_TOOL_CALLED", "unknown")
        census_vars = extract_field("CENSUS_VARIABLES", "unknown")
        data_returned = extract_field("DATA_RETURNED", "unknown")
        moe = extract_field("MARGIN_OF_ERROR", "not provided")
        methodology = extract_field("METHODOLOGY_NOTES", "not provided")
        success_text = extract_field("SUCCESS", "false").lower()
        success = success_text in ["true", "yes", "1", "success"]
        issues = extract_field("ISSUES_ENCOUNTERED", "none reported")
        
        # Parse confidence level robustly
        confidence_text = extract_field("CONFIDENCE_LEVEL", "0.5")
        try:
            confidence_match = re.search(r'(\d+\.?\d*)', confidence_text)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 10.0
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.5
        except (ValueError, AttributeError):
            confidence = 0.5
        
        # Score the response
        scores = self.score_response(query, main_answer, {
            'success': success,
            'confidence': confidence,
            'census_vars': census_vars,
            'data_returned': data_returned,
            'issues': issues
        })
        
        # Return complete test data
        return {
            'query_id': query['query_id'],
            'query_text': query['query_text'],
            'query_category': query['query_category'],
            'mcp_tool_called': mcp_tool,
            'mcp_parameters': json.dumps({"tool": mcp_tool, "success": success}),
            'mcp_success': success,
            'final_answer': main_answer,  # Now 3000 chars max
            'census_variables': census_vars,
            'margin_of_error': moe,
            'methodology_notes': methodology,
            'correctness': float(scores['correctness']),
            'plan_quality': float(scores['plan_quality']),
            'tool_coordination': float(scores['tool_coordination']),
            'limitation_handling': float(scores['limitation_handling']),
            'disambiguation': float(scores['disambiguation']),
            'methodology_guidance': float(scores['methodology_guidance']),
            'passed': bool(scores['passed']),
            'failure_reason': scores['failure_reason'] or "",
            'notes': scores['notes'] or ""
        }
    
    def score_response(self, query: Dict, answer: str, eval_data: Dict) -> Dict:
        """Score the response based on category and content"""
        
        category = query['query_category']
        
        scores = {
            'correctness': 0.5,
            'plan_quality': 0.5,
            'tool_coordination': 0.7 if eval_data['success'] else 0.2,
            'limitation_handling': 0.5,
            'disambiguation': 0.5,
            'methodology_guidance': 0.3,
            'passed': False,
            'failure_reason': '',
            'notes': ''
        }
        
        # Category-specific scoring
        if category == 'derived_statistic' and 'poverty rate' in query['query_text']:
            if '%' in answer and any(num in answer for num in ['31.', '32.', '33.', '34.']):
                scores['correctness'] = 1.0
                scores['passed'] = True
                scores['notes'] = '🎉 SUCCESS: Returns percentage rate (v2.1 fix working!)'
            elif '%' in answer:
                scores['correctness'] = 0.8
                scores['passed'] = True
                scores['notes'] = 'Returns percentage but unexpected value'
            else:
                scores['correctness'] = 0.2
                scores['failure_reason'] = 'Still returning count instead of percentage'
        
        elif category == 'basic_demographic':
            if eval_data['success'] and '$' in answer and '±' in answer:
                scores['correctness'] = 1.0
                scores['passed'] = True
                scores['notes'] = 'Proper income data with MOE'
        
        elif category == 'limitation_handling' or category == 'data_boundary':
            if not eval_data['success'] or 'cannot' in answer.lower() or 'error' in answer.lower():
                scores['correctness'] = 1.0
                scores['limitation_handling'] = 1.0
                scores['passed'] = True
                scores['notes'] = 'Correctly rejected inappropriate query'
        
        elif category == 'geographic_ambiguity':
            if 'clarification' in answer.lower() or 'which' in answer.lower():
                scores['disambiguation'] = 1.0
                scores['passed'] = True
            else:
                scores['disambiguation'] = 0.2
                scores['failure_reason'] = 'Should have requested clarification'
        
        # Methodology bonus
        if any(term in answer.lower() for term in ['acs', 'margin of error', 'confidence', 'survey']):
            scores['methodology_guidance'] = 0.8
        
        return scores
    
    def continue_session(self):
        """Continue existing session with bulletproof error handling"""
        state = self.load_state()
        
        if not state:
            print("No session in progress. Starting new run...")
            if not self.start_new_run():
                return
            state = self.load_state()
        
        current_index = state['current_query_index']
        
        while current_index < len(self.queries):
            query = self.queries[current_index]
            
            # Show progress
            print(f"\n📊 Progress: {current_index + 1}/{len(self.queries)}")
            print(f"🏷️ Category: {query['query_category']}")
            print(f"🔍 Query ID: {query['query_id']}")
            print("=" * 70)
            
            # Generate prompt
            prompt = self.prompt_template.format(question=query['query_text'])
            print("📋 Copy this prompt to Claude Desktop:")
            print("-" * 50)
            print(prompt)
            print("-" * 50)
            
            # Get response
            print("\n📥 Paste Claude's FULL response below:")
            print("📝 Include everything from answer through '=== END EVALUATION DATA ==='")
            print("🔄 Type 'DONE' on a new line when finished pasting:")
            
            response_lines = []
            
            while True:
                try:
                    line = input()
                    
                    if line.strip().upper() == 'DONE':
                        print("✅ Processing response...")
                        break
                    elif line.strip() == "=== END EVALUATION DATA ===":
                        response_lines.append(line)
                        print("✅ Detected end marker. Processing...")
                        break
                    else:
                        response_lines.append(line)
                        
                except KeyboardInterrupt:
                    print("\n⚠️ Interrupted. Options:")
                    print("1. Skip this query")
                    print("2. Save and exit")
                    print("3. Continue entering response")
                    
                    choice = input("Choice (1/2/3): ").strip()
                    if choice == "1":
                        current_index += 1
                        state['current_query_index'] = current_index
                        self.save_state(state)
                        break
                    elif choice == "2":
                        self.save_state(state)
                        print("💾 Session saved. Run again to continue.")
                        return
                    else:
                        print("Continuing...")
                        continue
            
            response_text = '\n'.join(response_lines)
            
            if not response_text.strip():
                print("❌ Empty response. Skipping...")
                current_index += 1
                state['current_query_index'] = current_index
                self.save_state(state)
                continue
            
            # Process response
            try:
                parsed_data = self.parse_response(query, response_text)
                
                # Save immediately to prevent data loss
                self.evaluator._add_query_test(state['run_id'], parsed_data)
                print(f"💾 Saved to database immediately")
                
                # Show result
                print(f"\n✅ Processed {query['query_id']}")
                print(f"   Success: {parsed_data['passed']}")
                print(f"   Score: {parsed_data['correctness']:.1f}")
                print(f"   Answer length: {len(parsed_data['final_answer'])} chars")
                if parsed_data['failure_reason']:
                    print(f"   Issue: {parsed_data['failure_reason']}")
                
                # Update state
                state['completed_queries'].append(query['query_id'])
                state['current_query_index'] = current_index + 1
                self.save_state(state)
                
                current_index += 1
                
            except Exception as e:
                print(f"❌ Error processing response: {e}")
                import traceback
                traceback.print_exc()
                
                print("\nOptions:")
                print("1. Retry this query")
                print("2. Skip this query")
                print("3. Save and exit")
                
                choice = input("Choice (1/2/3): ").strip()
                if choice == "2":
                    current_index += 1
                    state['current_query_index'] = current_index
                    self.save_state(state)
                elif choice == "3":
                    self.save_state(state)
                    print("💾 Session saved.")
                    return
        
        # Evaluation complete
        try:
            self.finish_evaluation(state)
        except Exception as e:
            print(f"⚠️ Error in final scoring: {e}")
            print("Your data is saved. View results with:")
            print(f"python evaluation_db.py --score-run '{state['run_name']}'")
    
    def finish_evaluation(self, state: Dict):
        """Finish evaluation and show results"""
        print("\n🎉 EVALUATION COMPLETE!")
        print("=" * 50)
        
        # Update database
        self.evaluator._update_run_summary(state['run_id'])
        
        # Show results
        self.evaluator.score_run(state['run_name'])
        
        # Clean up
        self.state_file.unlink(missing_ok=True)
        Path(self.state_file.name + '.backup.json').unlink(missing_ok=True)
        
        print(f"\n📊 Compare results:")
        print(f"python evaluation_db.py --compare-runs baseline {state['run_name']}")
    
    def show_menu(self):
        """Show main menu"""
        print("\n🏛️ Census MCP Evaluation Session")
        print("=" * 40)
        print("1. Start new evaluation run")
        print("2. Continue existing session")
        print("3. Show session status")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            if self.start_new_run():
                self.continue_session()
        elif choice == "2":
            self.continue_session()
        elif choice == "3":
            state = self.load_state()
            if state:
                completed = len(state['completed_queries'])
                total = len(self.queries)
                print(f"Run: {state['run_name']}")
                print(f"Progress: {completed}/{total} queries")
            else:
                print("No session in progress")
        elif choice == "4":
            print("👋 Goodbye!")
            return
        else:
            print("Invalid choice")
        
        # Loop back to menu
        self.show_menu()

def main():
    session = InteractiveSession()
    session.show_menu()

if __name__ == "__main__":
    main()
