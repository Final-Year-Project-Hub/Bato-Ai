"""
LLM Benchmarking Tool
Measures response times, token usage, and model performance.
"""

import asyncio
import time
import sys
import os
import argparse
from typing import List, Dict, Any
from statistics import mean, median

sys.path.append(os.getcwd())

from app.core.multi_llm import MultiModelLLMManager
from app.core.config import settings


class LLMBenchmark:
    """Benchmark LLM performance."""
    
    def __init__(self, api_token: str):
        self.manager = MultiModelLLMManager(api_token=api_token)
        self.results = {
            "query_analyzer": [],
            "generator": [],
            "validator": []
        }
    
    async def benchmark_model(
        self,
        model_name: str,
        llm,
        prompt: str,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Benchmark a specific model."""
        
        print(f"\n🧪 Testing {model_name}...")
        print(f"   Prompt length: {len(prompt)} chars")
        
        times = []
        errors = 0
        
        for i in range(iterations):
            try:
                start = time.time()
                response = await llm.ainvoke(prompt)
                end = time.time()
                
                elapsed = end - start
                times.append(elapsed)
                
                response_text = response.content if hasattr(response, "content") else str(response)
                
                print(f"   Iteration {i+1}/{iterations}: {elapsed:.2f}s ({len(response_text)} chars)")
                
            except Exception as e:
                errors += 1
                print(f"   Iteration {i+1}/{iterations}: ERROR - {str(e)[:50]}")
        
        if times:
            return {
                "model": model_name,
                "iterations": iterations,
                "successful": len(times),
                "errors": errors,
                "mean_time": mean(times),
                "median_time": median(times),
                "min_time": min(times),
                "max_time": max(times),
                "times": times
            }
        else:
            return {
                "model": model_name,
                "iterations": iterations,
                "successful": 0,
                "errors": errors,
                "error": "All iterations failed"
            }
    
    async def run_comprehensive_benchmark(self, iterations: int = 5):
        """Run comprehensive LLM benchmarks."""
        
        print("=" * 70)
        print("LLM PERFORMANCE BENCHMARK")
        print("=" * 70)
        print(f"\nIterations per model: {iterations}")
        print(f"Timeout: {settings.LLM_TIMEOUT}s")
        print()
        
        # Test prompts
        query_prompt = """Extract structured information from: "I want to learn React, I'm a beginner"
        
Return JSON with: goal, intent, proficiency, tech_stack, confidence"""
        
        generation_prompt = """Create a learning roadmap for: Learn Python basics
        
Return JSON with phases, topics, subtopics. Keep it concise (3-4 phases)."""
        
        validation_prompt = """Validate this roadmap structure:
{
  "goal": "Learn React",
  "phases": [{"title": "Basics", "topics": []}]
}

Return: {"valid": true/false, "issues": []}"""
        
        # 1. Query Analyzer Model
        print("📊 Test 1: Query Analyzer Model")
        print("-" * 70)
        result = await self.benchmark_model(
            settings.QUERY_ANALYSIS_MODEL,
            self.manager.get_query_analyzer_llm(),
            query_prompt,
            iterations
        )
        self.results["query_analyzer"] = result
        
        # 2. Generator Model
        print("\n📊 Test 2: Generator Model")
        print("-" * 70)
        result = await self.benchmark_model(
            settings.GENERATION_MODEL,
            self.manager.get_generator_llm(),
            generation_prompt,
            iterations
        )
        self.results["generator"] = result
        
        # 3. Validator Model
        print("\n📊 Test 3: Validator Model")
        print("-" * 70)
        result = await self.benchmark_model(
            settings.VALIDATION_MODEL,
            self.manager.get_validator_llm(),
            validation_prompt,
            iterations
        )
        self.results["validator"] = result
        
        # Summary
        print("\n\n📊 SUMMARY")
        print("=" * 70)
        
        for model_type, result in self.results.items():
            print(f"\n{model_type.upper().replace('_', ' ')}:")
            print(f"   Model: {result.get('model', 'N/A')}")
            
            if result.get("successful", 0) > 0:
                print(f"   Success rate: {result['successful']}/{result['iterations']}")
                print(f"   Mean time: {result['mean_time']:.2f}s")
                print(f"   Median time: {result['median_time']:.2f}s")
                print(f"   Range: {result['min_time']:.2f}s - {result['max_time']:.2f}s")
                
                # Performance rating
                avg = result['mean_time']
                if avg < 5:
                    print(f"   Rating: ⭐⭐⭐ FAST")
                elif avg < 15:
                    print(f"   Rating: ⭐⭐ GOOD")
                elif avg < 30:
                    print(f"   Rating: ⭐ ACCEPTABLE")
                else:
                    print(f"   Rating: ⚠️  SLOW")
            else:
                print(f"   ❌ All attempts failed")
                if "error" in result:
                    print(f"   Error: {result['error']}")
        
        # Overall assessment
        print("\n\n🎯 OVERALL ASSESSMENT")
        print("=" * 70)
        
        total_avg = mean([
            r["mean_time"] for r in self.results.values()
            if r.get("successful", 0) > 0
        ])
        
        print(f"\nAverage response time across all models: {total_avg:.2f}s")
        
        # Estimate total roadmap generation time
        # Typical flow: query analysis (1x) + generation (1x) + optional validation
        estimated_total = (
            self.results["query_analyzer"].get("mean_time", 0) +
            self.results["generator"].get("mean_time", 0)
        )
        
        print(f"Estimated roadmap generation time: {estimated_total:.2f}s")
        print(f"   (Query analysis + Generation, excluding retrieval)")
        
        if estimated_total < 45:
            print(f"\n✅ EXCELLENT: Well under 60s target!")
        elif estimated_total < 60:
            print(f"\n✅ GOOD: Within 60s target")
        else:
            print(f"\n⚠️  WARNING: Exceeds 60s target")
            print(f"\n💡 Recommendations:")
            print(f"   - Consider using faster models")
            print(f"   - Reduce LLM_MAX_TOKENS")
            print(f"   - Optimize prompts for brevity")
        
        print("\n" + "=" * 70)


def main():
    """Run LLM benchmark."""
    
    parser = argparse.ArgumentParser(description="Benchmark LLM performance")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per model (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Get API token
    api_token = settings.get_hf_token()
    
    if not api_token:
        print("❌ Error: HUGGINGFACE_API_TOKEN not set in .env")
        sys.exit(1)
    
    # Run benchmark
    benchmark = LLMBenchmark(api_token)
    asyncio.run(benchmark.run_comprehensive_benchmark(args.iterations))


if __name__ == "__main__":
    main()
