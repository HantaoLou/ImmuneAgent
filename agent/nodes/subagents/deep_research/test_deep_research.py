#!/usr/bin/env python3
"""
Simple test script to verify DeepSeek-powered Deep Research functionality.
Based on complete understanding of the framework architecture.
"""

import asyncio
import sys
from pathlib import Path

# Add agent directory to Python path for full path imports
agent_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(agent_dir))

# Import the public API
from nodes.subagents.deep_research import (
    run_deep_research,
    get_default_config,
    deep_researcher_builder,
)


async def test_deep_research(question: str):
    """
    Test the Deep Research functionality with DeepSeek models.
    
    Args:
        question: The research question to investigate
        
    Returns:
        dict: The final state containing the research report
    """
    
    print(f"🔬 Starting Deep Research for: '{question}'")
    print("=" * 80)
    
    # Get config to display (config is automatically loaded by run_deep_research)
    config = get_default_config()
    
    # Print loaded configuration
    print("📋 Loaded Configuration:")
    print(f"   Research Model: {config['configurable']['research_model']}")
    print(f"   Summarization Model: {config['configurable']['summarization_model']}")
    print(f"   Final Report Model: {config['configurable']['final_report_model']}")
    if config['configurable'].get('vector_scoring_model'):
        print(f"   Vector Scoring Model: {config['configurable']['vector_scoring_model']}")
    if config['configurable'].get('vector_summarization_model'):
        print(f"   Vector Summarization Model: {config['configurable']['vector_summarization_model']}")
    print()
    
    # Check available tools before execution
    print("🔧 Checking available tools...")
    from nodes.subagents.deep_research.utils import get_all_tools
    tools = await get_all_tools(config)
    print(f"✅ Available tools ({len(tools)}):")
    for tool in tools:
        tool_name = tool.name if hasattr(tool, 'name') else str(type(tool).__name__)
        print(f"   - {tool_name}")
    
    # Execute the Deep Research workflow using the simplified API
    try:
        print("\n🚀 Executing Deep Research workflow...")
        print("📋 Configuration:")
        print(f"   - Models: DeepSeek (chat + reasoner)")
        print(f"   - Search API: {config['configurable']['search_api']}") 
        print(f"   - Max iterations: {config['configurable']['max_researcher_iterations']}")
        print(f"   - Concurrent units: {config['configurable']['max_concurrent_research_units']}")
        print("\n⏳ Processing (this may take a few minutes)...\n")
        
        # Use the simplified public API - returns full state for detailed output
        result = await run_deep_research(question, return_full_state=True)
        
        # Extract and display results
        final_report = result.get("final_report", "No report generated")
        research_brief = result.get("research_brief", "No research brief")
        messages = result.get("messages", [])
        
        print("✅ Deep Research completed successfully!")
        print("=" * 80)
        print("📋 RESEARCH BRIEF:")
        print(research_brief)
        print("\n" + "=" * 80)
        print("📄 FINAL RESEARCH REPORT:")
        print(final_report)
        print("\n" + "=" * 80)
        print(f"📊 WORKFLOW MESSAGES: {len(messages)} total messages")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during Deep Research: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

async def run_multiple_tests():
    """Run multiple test cases to verify different scenarios."""
    
    test_cases = [
        "研究CRISPR基因编辑技术在免疫治疗中的最新应用和发展前景",
        "分析人工智能在新药研发中的应用现状和技术挑战", 
        "What are the latest developments in CAR-T cell therapy for solid tumors?"
    ]
    
    results = []
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n{'='*100}")
        print(f"🧪 TEST CASE {i}/{len(test_cases)}")
        print(f"{'='*100}")
        
        result = await test_deep_research(question)
        results.append({
            "question": question,
            "success": result is not None,
            "result": result
        })
        
        if i < len(test_cases):
            print("\n⏸️  Waiting 5 seconds before next test...")
            await asyncio.sleep(5)
    
    # Summary
    print(f"\n{'='*100}")
    print("📊 TEST SUMMARY")
    print(f"{'='*100}")
    
    successful_tests = sum(1 for r in results if r["success"])
    print(f"✅ Successful tests: {successful_tests}/{len(test_cases)}")
    
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"   Test {i}: {status} - {result['question'][:50]}...")
    
    return results

def main():
    """Main function to run the Deep Research test."""
    
    print("🧬 DeepSeek Deep Research Test")
    print("=" * 100)
    
    # Check if running single test or multiple tests
    if len(sys.argv) > 1:
        # Single test with custom question
        question = " ".join(sys.argv[1:])
        print(f"🎯 Running single test with custom question")
        result = asyncio.run(test_deep_research(question))
        
        if result:
            print(f"\n🎉 Test completed successfully!")
        else:
            print(f"\n💥 Test failed!")
            
    else:
        # Run multiple test cases
        print(f"🎯 Running multiple test cases")
        results = asyncio.run(run_multiple_tests())
        
        successful_tests = sum(1 for r in results if r["success"])
        if successful_tests == len(results):
            print(f"\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main()
