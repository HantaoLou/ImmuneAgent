"""Quick test for the Tavily + Qdrant literature search pipeline."""

import asyncio
import os
import sys
from pathlib import Path

# Load env vars from deep_research .env
env_path = Path(__file__).parent.parent / "deep_research" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value and not os.environ.get(key):
                os.environ[key] = value

# Add agent dir to path
agent_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(agent_dir))

from nodes.subagents.subagent_paperQA.paper_retrieval import (
    extract_search_queries,
    search_tavily,
    search_qdrant,
    discover_papers,
    safe_paper_pipeline,
)


async def test_query_extraction():
    print("=" * 60)
    print("TEST 1: Query Extraction")
    print("=" * 60)
    question = "What is the role of CD4+ T cells in adaptive immunity and how do they interact with MHC class II molecules?"
    queries = extract_search_queries(question)
    for i, q in enumerate(queries):
        print(f"  Query {i+1}: {q}")
    print()


async def test_tavily_search():
    print("=" * 60)
    print("TEST 2: Tavily Search")
    print("=" * 60)
    api_key = os.environ.get("TAVILY_API_KEY", "")
    print(f"  TAVILY_API_KEY: {'set (' + api_key[:8] + '...)' if api_key else 'NOT SET'}")

    results = await search_tavily("CD4+ T cells adaptive immunity MHC class II", max_results=3)
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"    [{r['source']}] {r['title'][:80]} (score: {r['score']:.2f})")
        print(f"      URL: {r.get('url', 'N/A')[:80]}")
        print(f"      Snippet: {r['snippet'][:120]}...")
    print()


async def test_qdrant_search():
    print("=" * 60)
    print("TEST 3: Qdrant Search")
    print("=" * 60)
    host = os.environ.get("QDRANT_HOST", "")
    collection = os.environ.get("QDRANT_COLLECTION", "")
    print(f"  QDRANT_HOST: {host or 'NOT SET'}")
    print(f"  QDRANT_COLLECTION: {collection or 'NOT SET'}")
    print(f"  EMBEDDING_PROVIDER: {os.environ.get('EMBEDDING_PROVIDER', 'NOT SET')}")
    print(f"  EMBEDDING_MODEL: {os.environ.get('EMBEDDING_MODEL', 'NOT SET')}")

    results = await search_qdrant("CD4+ T cells adaptive immunity", max_results=3)
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"    [{r['source']}] {r['title'][:80]} (similarity: {r['score']:.3f})")
        print(f"      Snippet: {r['snippet'][:120]}...")
    print()


async def test_discover():
    print("=" * 60)
    print("TEST 4: discover_papers (Tavily + Qdrant combined)")
    print("=" * 60)
    results = await discover_papers(
        "What is the role of CD4+ T cells in adaptive immunity?",
        max_per_source=3,
    )
    print(f"  Total results: {len(results)}")
    sources = {}
    for r in results:
        sources.setdefault(r["source"], 0)
        sources[r["source"]] += 1
    print(f"  By source: {sources}")
    for r in results[:5]:
        print(f"    [{r['source']}] {r['title'][:80]}")
    print()


async def test_full_pipeline():
    print("=" * 60)
    print("TEST 5: safe_paper_pipeline (full end-to-end)")
    print("=" * 60)
    result = await safe_paper_pipeline(
        question="What is the role of CD4+ T cells in adaptive immunity and how do they interact with MHC class II molecules?",
        max_papers=5,
        timeout=60.0,
    )

    if result is None:
        print("  Pipeline returned None (no results or failure)")
    else:
        print(f"  papers_discovered: {result['papers_discovered']}")
        print(f"  papers_indexed: {result['papers_indexed']}")
        print(f"  confidence: {result['confidence']:.3f}")
        print(f"  sources: {result.get('sources', [])}")
        print(f"  evidence_items: {len(result['evidence_items'])}")
        print(f"  evidence_text_block length: {len(result['evidence_text_block'])} chars")
        print()
        print("  --- Evidence Text Block (first 800 chars) ---")
        print(result["evidence_text_block"][:800])
    print()


async def main():
    print(f"\nPython: {sys.version}")
    print(f"CWD: {os.getcwd()}\n")

    await test_query_extraction()
    await test_tavily_search()
    await test_qdrant_search()
    await test_discover()
    await test_full_pipeline()

    print("All tests complete.")


if __name__ == "__main__":
    asyncio.run(main())
