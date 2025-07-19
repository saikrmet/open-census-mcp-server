#!/usr/bin/env python3
"""
Production Table Keyword Generator using GPT-4.1

Generates search-optimized keywords for ACS tables with production-grade infrastructure:
- Exponential backoff and retry logic
- Safe resume with checkpoint system
- Parallel processing with rate limiting
- Comprehensive error handling and logging
- Safe file handling (never overwrites original catalog)
"""

import json
import openai
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class KeywordResult:
    """Container for keyword generation results"""
    table_id: str
    primary_keywords: List[str]
    secondary_keywords: List[str]
    summary: str
    processing_cost: float
    processing_time: float
    timestamp: float
    error: Optional[str] = None


class ProductionTableKeywordGenerator:
    """Production-grade table keyword generator with robust error handling"""
    
    def __init__(self, openai_api_key: str, rate_limit_rpm: int = 1000):
        """Initialize with API key and rate limiting"""
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Rate limiting
        self.rate_limit_rpm = rate_limit_rpm
        self.min_delay = 60.0 / rate_limit_rpm  # seconds between calls
        self.last_call_time = 0
        self.call_lock = Lock()
        
        # Statistics
        self.total_cost = 0.0
        self.successful_count = 0
        self.failed_count = 0
        self.retry_count = 0
        
        logger.info(f"Initialized with rate limit: {rate_limit_rpm} RPM")
    
    def deduplicate_keywords(self, keywords: List[str]) -> List[str]:
        """Remove near-duplicate keywords using string similarity"""
        if not keywords:
            return keywords
        
        def similarity(a: str, b: str) -> float:
            """Calculate simple word overlap similarity"""
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            intersection = words_a.intersection(words_b)
            union = words_a.union(words_b)
            return len(intersection) / len(union) if union else 0
        
        deduplicated = []
        for keyword in keywords:
            # Check if this keyword is too similar to any existing keyword
            is_duplicate = False
            for existing in deduplicated:
                if similarity(keyword, existing) > 0.7:  # 70% word overlap threshold
                    # Keep the shorter, cleaner version
                    if len(keyword) < len(existing):
                        deduplicated.remove(existing)
                        deduplicated.append(keyword)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(keyword)
        
        return deduplicated
    
    def _estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate cost for GPT-4.1-mini"""
        # GPT-4.1-mini pricing (current rates)
        input_cost_per_1m = 0.150   # $0.150 per 1M input tokens
        output_cost_per_1m = 0.600  # $0.600 per 1M output tokens
        
        # Rough token estimation (4 characters per token)
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        
        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
        
        return input_cost + output_cost
    
    def _rate_limited_call(self, prompt: str, max_retries: int = 3) -> str:
        """Make rate-limited API call with exponential backoff"""
        
        # Rate limiting
        with self.call_lock:
            time_since_last = time.time() - self.last_call_time
            if time_since_last < self.min_delay:
                sleep_time = self.min_delay - time_since_last
                logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            self.last_call_time = time.time()
        
        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-1106-preview",  # GPT-4.1
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Track cost
                cost = self._estimate_cost(prompt, response_text)
                self.total_cost += cost
                
                return response_text
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "rate_limit" in error_str or "429" in error_str:
                    # Rate limit hit - exponential backoff
                    wait_time = (2 ** attempt) * 30  # 30s, 60s, 120s
                    logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    self.retry_count += 1
                    continue
                    
                elif "503" in error_str or "502" in error_str or "server" in error_str:
                    # Server error - retry with backoff
                    wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s
                    logger.warning(f"Server error (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    self.retry_count += 1
                    continue
                    
                else:
                    # Other error - don't retry
                    logger.error(f"API error: {e}")
                    raise e
        
        # All retries exhausted
        raise Exception(f"Failed after {max_retries} retries")
    
    def generate_table_keywords(self, table_id: str, title: str, universe: str, concept: str) -> KeywordResult:
        """Generate keywords for a single table with error handling"""
        start_time = time.time()
        
        # Expert-refined prompt following ACS specialist recommendations
        prompt = f"""Given the ACS table details below, generate 3-5 **primary keywords**, 2-4 **secondary keywords**, and a **one-sentence summary**.

- Primary keywords should represent the main concepts and most likely search terms.
- Secondary keywords should cover alternative phrasings and related but less central ideas.
- Summary should be a single, plain-language sentence explaining what the table contains.
- Avoid excessive duplication and keep each keyword distinct unless necessary for accuracy.
- Use language an informed layperson would use.
- Return as JSON only, no additional text.

Example output:
```json
{{
  "primary_keywords": ["poverty rate", "below poverty line", "poverty by age"],
  "secondary_keywords": ["economic hardship", "low income population"],
  "summary": "Population data showing poverty status broken down by age and sex for people whose poverty status can be determined."
}}
```

**Table Title:** "{title}"
**Concept:** "{concept}"
**Universe:** "{universe}"

Return only the JSON object, no additional text."""
        
        try:
            # Make API call with retries
            response_text = self._rate_limited_call(prompt)
            
            # Clean up potential markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON response
            keywords_data = json.loads(response_text)
            
            # Validate structure
            if not isinstance(keywords_data, dict):
                raise ValueError("Response is not a dictionary")
            
            required_fields = ["primary_keywords", "secondary_keywords", "summary"]
            if not all(field in keywords_data for field in required_fields):
                raise ValueError(f"Missing required fields: {required_fields}")
            
            # Clean up keywords
            primary = [kw.strip() for kw in keywords_data["primary_keywords"] if kw.strip()]
            secondary = [kw.strip() for kw in keywords_data["secondary_keywords"] if kw.strip()]
            
            # Enhanced deduplication
            primary = self.deduplicate_keywords(primary)
            secondary = self.deduplicate_keywords(secondary)
            
            # Get summary
            summary = keywords_data.get("summary", "").strip()
            
            self.successful_count += 1
            processing_time = time.time() - start_time
            
            return KeywordResult(
                table_id=table_id,
                primary_keywords=primary[:5],  # Limit primary to 5
                secondary_keywords=secondary[:3],  # Limit secondary to 3
                summary=summary,
                processing_cost=self.total_cost,  # Running total
                processing_time=processing_time,
                timestamp=time.time()
            )
            
        except json.JSONDecodeError as e:
            self.failed_count += 1
            error_msg = f"JSON decode error: {e}"
            logger.error(f"{table_id}: {error_msg}")
            return KeywordResult(
                table_id=table_id,
                primary_keywords=[],
                secondary_keywords=[],
                summary="",
                processing_cost=0.0,
                processing_time=time.time() - start_time,
                timestamp=time.time(),
                error=error_msg
            )
            
        except Exception as e:
            self.failed_count += 1
            error_msg = f"Generation error: {str(e)}"
            logger.error(f"{table_id}: {error_msg}")
            return KeywordResult(
                table_id=table_id,
                primary_keywords=[],
                secondary_keywords=[],
                summary="",
                processing_cost=0.0,
                processing_time=time.time() - start_time,
                timestamp=time.time(),
                error=error_msg
            )
    
    def load_existing_results(self, output_file: str) -> Dict[str, Dict]:
        """Load existing results for safe resume"""
        if not os.path.exists(output_file):
            return {}
        
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            
            logger.info(f"Loaded {len(existing_data)} existing results for resume")
            return existing_data
                
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
            return {}
    
    def save_checkpoint(self, results: List[KeywordResult], output_file: str):
        """Save checkpoint during processing (results only, not enhanced catalog)"""
        try:
            # Convert results to simple dict format for checkpointing
            results_dict = {}
            for result in results:
                results_dict[result.table_id] = {
                    "primary_keywords": result.primary_keywords,
                    "secondary_keywords": result.secondary_keywords,
                    "summary": result.summary,
                    "processing_time": result.processing_time,
                    "timestamp": result.timestamp,
                    "error": result.error
                }
            
            # Atomic write
            temp_file = output_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            os.rename(temp_file, output_file)
            
            logger.info(f"Checkpoint saved: {len([r for r in results if not r.error])} successful results")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def save_enhanced_catalog(self, results: List[KeywordResult], original_catalog_file: str, output_catalog_file: str):
        """Save enhanced catalog with keywords without overwriting original"""
        try:
            # Load original catalog
            with open(original_catalog_file, 'r') as f:
                catalog_data = json.load(f)
            
            # Create lookup of results by table_id
            results_dict = {result.table_id: result for result in results if not result.error}
            
            # Enhance tables with keywords
            enhanced_tables = []
            integrated_count = 0
            
            for table in catalog_data['tables']:
                table_id = table['table_id']
                
                # Create copy to avoid modifying original
                enhanced_table = table.copy()
                
                if table_id in results_dict:
                    result = results_dict[table_id]
                    enhanced_table['search_keywords'] = {
                        'primary_keywords': result.primary_keywords,
                        'secondary_keywords': result.secondary_keywords,
                        'summary': result.summary,
                        'metadata': {
                            'generated_by': 'gpt-4.1-production',
                            'processing_time': result.processing_time,
                            'timestamp': result.timestamp,
                            'processing_cost': result.processing_cost
                        }
                    }
                    integrated_count += 1
                
                enhanced_tables.append(enhanced_table)
            
            # Update metadata
            enhanced_catalog = catalog_data.copy()
            enhanced_catalog['tables'] = enhanced_tables
            enhanced_catalog['metadata'] = catalog_data['metadata'].copy()
            enhanced_catalog['metadata']['search_keywords_generated'] = True
            enhanced_catalog['metadata']['keyword_generation_timestamp'] = time.time()
            enhanced_catalog['metadata']['keyword_integration_count'] = integrated_count
            enhanced_catalog['metadata']['enhanced_catalog_version'] = 'keywords_v1.0'
            
            # Atomic write to new file
            temp_file = output_catalog_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(enhanced_catalog, f, indent=2)
            os.rename(temp_file, output_catalog_file)
            
            logger.info(f"✅ Enhanced catalog saved to: {output_catalog_file}")
            logger.info(f"✅ Integrated keywords for {integrated_count}/{len(enhanced_tables)} tables")
            logger.info(f"🔒 Original catalog preserved: {original_catalog_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enhanced catalog: {e}")
            return False
    
    def process_tables_parallel(self, tables: List[Dict], output_file: str,
                              max_workers: int = 5, checkpoint_interval: int = 50) -> List[KeywordResult]:
        """Process tables in parallel with checkpoints"""
        
        results = []
        processed_count = 0
        
        # Process in batches for checkpointing
        batch_size = checkpoint_interval
        for batch_start in range(0, len(tables), batch_size):
            batch_end = min(batch_start + batch_size, len(tables))
            batch_tables = tables[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start+1}-{batch_end} of {len(tables)}")
            
            # Process batch in parallel
            batch_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tables in batch
                future_to_table = {
                    executor.submit(
                        self.generate_table_keywords,
                        table['table_id'],
                        table['title'],
                        table['universe'],
                        table['concept']
                    ): table for table in batch_tables
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_table):
                    table = future_to_table[future]
                    try:
                        result = future.result(timeout=120)  # 2 minute timeout
                        batch_results.append(result)
                        processed_count += 1
                        
                        if result.error:
                            logger.error(f"❌ {table['table_id']}: {result.error}")
                        else:
                            logger.info(f"✅ {table['table_id']}: {len(result.primary_keywords)} primary, {len(result.secondary_keywords)} secondary")
                        
                    except Exception as e:
                        logger.error(f"❌ {table['table_id']}: Timeout or executor error: {e}")
                        # Create error result
                        error_result = KeywordResult(
                            table_id=table['table_id'],
                            primary_keywords=[],
                            secondary_keywords=[],
                            summary="",
                            processing_cost=0.0,
                            processing_time=0.0,
                            timestamp=time.time(),
                            error=f"Executor error: {str(e)}"
                        )
                        batch_results.append(error_result)
                        processed_count += 1
            
            results.extend(batch_results)
            
            # Save checkpoint
            self.save_checkpoint(results, output_file)
            
            # Progress report
            success_rate = self.successful_count / (self.successful_count + self.failed_count) * 100 if (self.successful_count + self.failed_count) > 0 else 0
            avg_cost = self.total_cost / self.successful_count if self.successful_count > 0 else 0
            
            logger.info(f"Progress: {processed_count}/{len(tables)} | "
                       f"Success: {success_rate:.1f}% | "
                       f"Avg cost: ${avg_cost:.4f} | "
                       f"Total: ${self.total_cost:.2f} | "
                       f"Retries: {self.retry_count}")
        
        return results


def test_keyword_generation():
    """Test keyword generation on 10 diverse sample tables"""
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable required")
        return
    
    generator = ProductionTableKeywordGenerator(api_key, rate_limit_rpm=1000)
    
    # 10 diverse test tables
    test_tables = [
        {
            'table_id': 'B16001',
            'title': 'Language Spoken at Home by Ability to Speak English for the Population 5 Years and Over',
            'universe': 'Population 5 years and over',
            'concept': 'Language Spoken at Home'
        },
        {
            'table_id': 'B27001',
            'title': 'Health Insurance Coverage Status by Sex by Age',
            'universe': 'Civilian noninstitutionalized population',
            'concept': 'Health Insurance Coverage'
        },
        {
            'table_id': 'B28002',
            'title': 'Presence and Types of Internet Subscriptions in Household',
            'universe': 'Households',
            'concept': 'Internet Subscription'
        },
        {
            'table_id': 'B21001',
            'title': 'Sex by Age by Veteran Status for the Civilian Population 18 Years and Over',
            'universe': 'Civilian population 18 years and over',
            'concept': 'Veteran Status'
        },
        {
            'table_id': 'B18101',
            'title': 'Sex by Age by Disability Status',
            'universe': 'Civilian noninstitutionalized population',
            'concept': 'Disability Status'
        },
        {
            'table_id': 'B10051',
            'title': 'Grandparents Living with Own Grandchildren Under 18 Years by Responsibility for Own Grandchildren',
            'universe': 'Population 30 years and over',
            'concept': 'Grandparent Caregivers'
        },
        {
            'table_id': 'B05002',
            'title': 'Place of Birth by Nativity and Citizenship Status',
            'universe': 'Total population',
            'concept': 'Nativity and Citizenship'
        },
        {
            'table_id': 'B26001',
            'title': 'Group Quarters Population by Sex by Age',
            'universe': 'Population in group quarters',
            'concept': 'Group Quarters'
        },
        {
            'table_id': 'B25034',
            'title': 'Year Structure Built',
            'universe': 'Housing units',
            'concept': 'Year Structure Built'
        },
        {
            'table_id': 'B13016',
            'title': 'Women 15 to 50 Years Who Had a Birth in the Past 12 Months by Marital Status and Age',
            'universe': 'Women 15 to 50 years',
            'concept': 'Fertility'
        }
    ]
    
    print("🧪 TESTING PRODUCTION TABLE KEYWORD GENERATION")
    print("=" * 70)
    print("Features: GPT-4.1, deduplication, summaries, error handling\n")
    
    results = []
    for table in test_tables:
        print(f"📊 TABLE {table['table_id']}: {table['title'][:60]}...")
        
        result = generator.generate_table_keywords(
            table['table_id'],
            table['title'],
            table['universe'],
            table['concept']
        )
        
        if result.error:
            print(f"❌ Failed: {result.error}")
        else:
            print(f"🎯 Primary Keywords: {', '.join(result.primary_keywords)}")
            print(f"🔧 Secondary Keywords: {', '.join(result.secondary_keywords)}")
            print(f"📝 Summary: {result.summary}")
            print(f"💰 Cost: ${result.processing_cost:.4f}")
        
        results.append(result)
        print()
    
    # Final statistics
    print("📊 TEST COMPLETE")
    print(f"✅ Successful: {generator.successful_count}")
    print(f"❌ Failed: {generator.failed_count}")
    print(f"💰 Total cost: ${generator.total_cost:.4f}")
    print(f"🔄 Retries: {generator.retry_count}")


def main():
    """Main production execution"""
    parser = argparse.ArgumentParser(description='Production Table Keyword Generator')
    parser.add_argument('--catalog-file', default='table-catalog/table_catalog.json', help='Input table catalog')
    parser.add_argument('--output-file', default='table_keywords_results.json', help='Output results file')
    parser.add_argument('--enhanced-catalog-file', default='table-catalog/table_catalog_with_keywords.json', help='Output enhanced catalog file')
    parser.add_argument('--test-only', action='store_true', help='Run test on 10 sample tables')
    parser.add_argument('--max-tables', type=int, help='Limit number of tables for testing')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from existing results')
    parser.add_argument('--force-restart', action='store_true', help='Delete existing results and start fresh')
    parser.add_argument('--checkpoint-interval', type=int, default=50, help='Save checkpoint every N tables')
    parser.add_argument('--max-workers', type=int, default=5, help='Parallel processing workers')
    parser.add_argument('--rate-limit', type=int, default=1000, help='API requests per minute')
    
    args = parser.parse_args()
    
    # Test mode
    if args.test_only:
        test_keyword_generation()
        return 0
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable required")
        return 1
    
    # Initialize generator
    generator = ProductionTableKeywordGenerator(api_key, rate_limit_rpm=args.rate_limit)
    
    # Load table catalog
    try:
        with open(args.catalog_file, 'r') as f:
            catalog_data = json.load(f)
        tables = catalog_data.get('tables', [])
        logger.info(f"Loaded {len(tables)} tables from catalog")
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")
        return 1
    
    # Handle force restart
    if args.force_restart and os.path.exists(args.output_file):
        os.remove(args.output_file)
        logger.warning(f"🔥 Force restart: Deleted {args.output_file}")
    
    # Load existing results for resume
    existing_results = generator.load_existing_results(args.output_file) if args.resume else {}
    
    # Filter out already processed tables
    if existing_results:
        original_count = len(tables)
        tables = [t for t in tables if t['table_id'] not in existing_results]
        logger.info(f"Resume: Skipping {original_count - len(tables)} already processed tables")
    
    # Limit tables if requested
    if args.max_tables and args.max_tables < len(tables):
        tables = tables[:args.max_tables]
        logger.info(f"Limited to {args.max_tables} tables for testing")
    
    if not tables:
        logger.info("🎉 All tables already processed!")
        return 0
    
    # Cost estimation
    estimated_cost = len(tables) * 0.002  # GPT-4.1-mini estimate
    logger.info(f"Processing {len(tables)} tables")
    logger.info(f"Estimated cost: ${estimated_cost:.2f}")
    
    # Confirmation
    if not args.resume:
        response = input(f"Generate keywords for {len(tables)} tables? (estimated cost ${estimated_cost:.2f}) [y/N]: ")
        if response.lower() != 'y':
            logger.info("Cancelled")
            return 0
    
    # Process tables
    start_time = time.time()
    
    try:
        results = generator.process_tables_parallel(
            tables,
            args.output_file,
            max_workers=args.max_workers,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Final save and enhanced catalog creation
        generator.save_checkpoint(results, args.output_file)
        
        # Create enhanced catalog without overwriting original
        logger.info("🔧 Creating enhanced catalog with keywords...")
        generator.save_enhanced_catalog(results, args.catalog_file, args.enhanced_catalog_file)
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    # Final report
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info("✅ PRODUCTION RUN COMPLETE")
    logger.info(f"✅ Successful: {generator.successful_count}")
    logger.info(f"❌ Failed: {generator.failed_count}")
    logger.info(f"💰 Total cost: ${generator.total_cost:.2f}")
    logger.info(f"🔄 Retries: {generator.retry_count}")
    logger.info(f"⏱️ Processing time: {processing_time:.1f}s")
    logger.info(f"📁 Results saved to: {args.output_file}")
    logger.info(f"📁 Enhanced catalog saved to: {args.enhanced_catalog_file}")
    logger.info(f"🔒 Original catalog preserved: {args.catalog_file}")
    logger.info("💡 Next: Update table embeddings to include keywords")
    
    return 0


if __name__ == "__main__":
    exit(main())
