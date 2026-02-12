#!/usr/bin/env python3
"""
Comprehensive Test for 8 Collections Retrieval

This test validates the entire retrieval pipeline using retrieve() as entry point.
It verifies each of the 8 configured collections can retrieve data.

Test Strategy:
1. First test each collection individually with _retrieve_from_single_collection()
2. Then test the full retrieve() function with all collections
3. Diagnose any collection returning 0 results
"""

import os
import sys
import asyncio
import time

# Add agent path
sys.path.insert(0, '/data/server/ImmuneAgent_2.0/agent')

# Load environment variables
from dotenv import load_dotenv
load_dotenv('/data/server/ImmuneAgent_2.0/agent/nodes/subagents/deep_research/.env')


def print_separator(title: str, char: str = "="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_result(success: bool, message: str):
    status = "✅" if success else "❌"
    print(f"{status} {message}")


# ============================================================================
# Test 1: Verify all 8 collections are configured
# ============================================================================
def test_collection_configuration():
    print_separator("Test 1: Collection Configuration Verification")
    
    collections_str = os.getenv("QDRANT_COLLECTIONS", "")
    collections = [c.strip() for c in collections_str.split(",") if c.strip()]
    
    print(f"QDRANT_COLLECTIONS = {collections_str}")
    print(f"Parsed collections ({len(collections)}):")
    
    from nodes.subagents.deep_research.vector_search_tool import (
        COLLECTION_CONFIG, get_collection_config
    )
    
    results = {}
    for col in collections:
        cfg = get_collection_config(col)
        in_config = col in COLLECTION_CONFIG
        print(f"  - {col}: dim={cfg['dimension']}, source_key={cfg['source_key']}, configured={in_config}")
        results[col] = cfg
    
    if len(collections) == 8:
        print_result(True, f"All 8 collections configured")
    else:
        print_result(False, f"Expected 8 collections, got {len(collections)}")
    
    return collections, results


# ============================================================================
# Test 2: Test each collection individually with similarity_search
# ============================================================================
def test_individual_similarity_search(collections: list):
    print_separator("Test 2: Individual Collection Similarity Search")
    
    from nodes.subagents.deep_research.vector_search_tool import _get_vector_store
    
    config = {'configurable': {}}
    test_query = "immune system T cell response"
    
    results = {}
    
    for col in collections:
        print(f"\n--- {col} ---")
        try:
            start = time.time()
            vs = _get_vector_store(col, config)
            
            # Direct similarity search (bypasses retriever logic)
            docs = vs.similarity_search(test_query, k=3)
            elapsed = time.time() - start
            
            results[col] = {
                "count": len(docs),
                "time": elapsed,
                "error": None
            }
            
            if docs:
                print_result(True, f"Retrieved {len(docs)} docs in {elapsed:.2f}s")
                # Show first doc source
                source = docs[0].metadata.get('source') or docs[0].metadata.get('source_file', 'N/A')
                print(f"    Sample source: {str(source)[:60]}...")
            else:
                print_result(False, f"Retrieved 0 docs (collection may be empty or query mismatch)")
                
        except Exception as e:
            results[col] = {"count": 0, "time": 0, "error": str(e)}
            print_result(False, f"Error: {e}")
    
    return results


# ============================================================================
# Test 3: Test each collection with _retrieve_from_single_collection
# ============================================================================
def test_individual_retriever(collections: list):
    print_separator("Test 3: Individual Collection Retriever (_retrieve_from_single_collection)")
    
    from nodes.subagents.deep_research.vector_search_tool import _retrieve_from_single_collection
    
    config = {'configurable': {}}
    test_queries = ["immune response mechanism", "T cell activation"]
    
    results = {}
    
    for col in collections:
        print(f"\n--- {col} ---")
        try:
            start = time.time()
            docs, _ = _retrieve_from_single_collection(col, test_queries, config, k_per_query=3)
            elapsed = time.time() - start
            
            results[col] = {
                "count": len(docs),
                "time": elapsed,
                "error": None
            }
            
            if docs:
                print_result(True, f"Retrieved {len(docs)} docs in {elapsed:.2f}s")
                # Check _collection metadata
                has_collection_tag = all(d.metadata.get('_collection') == col for d in docs)
                print(f"    _collection tag correct: {has_collection_tag}")
            else:
                print_result(False, f"Retrieved 0 docs")
                
        except Exception as e:
            import traceback
            results[col] = {"count": 0, "time": 0, "error": str(e)}
            print_result(False, f"Error: {e}")
            traceback.print_exc()
    
    return results


# ============================================================================
# Test 4: Test full retrieve_doc() with all collections
# ============================================================================
def test_retrieve_doc_full():
    print_separator("Test 4: Full retrieve_doc() with All Collections")
    
    from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
    
    config = {'configurable': {}}
    test_queries = ["CAR-T cell therapy", "virus infection mechanism"]
    
    print(f"Queries: {test_queries}")
    print()
    
    try:
        start = time.time()
        results = retrieve_doc(query=test_queries, config=config, k_per_query=3)
        elapsed = time.time() - start
        
        print(f"\n--- Results ---")
        print(f"Total documents: {len(results)}")
        print(f"Time: {elapsed:.2f}s")
        
        if results:
            print_result(True, f"retrieve_doc() returned {len(results)} documents")
            
            # Analyze by source
            sources = {}
            for doc in results:
                src = doc.source[:50] if doc.source else "unknown"
                sources[src] = sources.get(src, 0) + 1
            
            print(f"\nUnique sources: {len(sources)}")
            for src, count in list(sources.items())[:5]:
                print(f"  - {src}... ({count})")
        else:
            print_result(False, "retrieve_doc() returned 0 documents")
        
        return results
        
    except Exception as e:
        import traceback
        print_result(False, f"Error: {e}")
        traceback.print_exc()
        return []


# ============================================================================
# Test 5: Test async retrieve() - the actual entry point
# ============================================================================
async def test_retrieve_async():
    print_separator("Test 5: Async retrieve() - Production Entry Point")
    
    from nodes.subagents.deep_research.vector_search_tool import retrieve
    
    config = {'configurable': {}}
    test_queries = ["immune checkpoint inhibitor", "viral protein structure"]
    
    print(f"Queries: {test_queries}")
    print()
    
    try:
        start = time.time()
        result_str = await retrieve(query=test_queries, config=config, k_per_query=3)
        elapsed = time.time() - start
        
        print(f"\n--- Results ---")
        print(f"Result length: {len(result_str)} characters")
        print(f"Time: {elapsed:.2f}s")
        
        if result_str:
            print_result(True, f"retrieve() returned {len(result_str)} chars")
            
            # Count documents by looking for RetrievedDocument patterns
            doc_count = result_str.count("source=")
            print(f"Estimated document count: {doc_count}")
            
            # Show preview
            preview = result_str[:500].replace('\n', ' ')
            print(f"\nPreview: {preview}...")
        else:
            print_result(False, "retrieve() returned empty string")
        
        return result_str
        
    except Exception as e:
        import traceback
        print_result(False, f"Error: {e}")
        traceback.print_exc()
        return ""


# ============================================================================
# Summary and Diagnosis
# ============================================================================
def summarize_results(similarity_results: dict, retriever_results: dict):
    print_separator("Summary & Diagnosis", "=")
    
    print("Collection Results:")
    print(f"{'Collection':<25} {'Similarity':<12} {'Retriever':<12} {'Status':<10}")
    print("-" * 60)
    
    all_ok = True
    for col in similarity_results:
        sim_count = similarity_results[col]["count"]
        ret_count = retriever_results.get(col, {}).get("count", 0)
        
        if sim_count > 0 and ret_count > 0:
            status = "✅ OK"
        elif sim_count > 0 and ret_count == 0:
            status = "⚠️ Retriever Issue"
            all_ok = False
        elif sim_count == 0 and ret_count == 0:
            status = "❌ No Data"
            all_ok = False
        else:
            status = "❓ Unknown"
            all_ok = False
        
        print(f"{col:<25} {sim_count:<12} {ret_count:<12} {status}")
    
    print()
    if all_ok:
        print_result(True, "All 8 collections working correctly!")
    else:
        print_result(False, "Some collections have issues - see diagnosis above")
        
        # Provide diagnosis for failing collections
        print("\n--- Diagnosis ---")
        for col in similarity_results:
            sim = similarity_results[col]
            ret = retriever_results.get(col, {})
            
            if sim["error"]:
                print(f"{col}: Similarity search error - {sim['error']}")
            elif ret.get("error"):
                print(f"{col}: Retriever error - {ret['error']}")
            elif sim["count"] == 0:
                print(f"{col}: Query does not match content (try domain-specific query)")


# ============================================================================
# Main
# ============================================================================
def main():
    print_separator("8-Collection Comprehensive Retrieval Test", "#")
    
    # Test 1: Configuration
    collections, configs = test_collection_configuration()
    
    if not collections:
        print("No collections configured. Exiting.")
        return
    
    # Test 2: Similarity search
    similarity_results = test_individual_similarity_search(collections)
    
    # Test 3: Retriever
    retriever_results = test_individual_retriever(collections)
    
    # Test 4: Full retrieve_doc
    test_retrieve_doc_full()
    
    # Test 5: Async retrieve
    asyncio.run(test_retrieve_async())
    
    # Summary
    summarize_results(similarity_results, retriever_results)


if __name__ == "__main__":
    main()
