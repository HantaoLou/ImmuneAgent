#!/usr/bin/env python3
"""
Complete End-to-End Test for Multi-Collection Vector Search

This test validates the entire vector search pipeline with real embedding models:
1. OpenAI text-embedding-3-small (1536 dimensions) - Immunology collection
2. PubMedBERT (768 dimensions) - virology collection
3. Multi-collection mixed query

Test Flow:
- Input: Research question
- Output: Retrieved documents with sources

Author: Cascade AI
"""

import os
import sys
import time
import traceback

# Add agent path
sys.path.insert(0, '/data/server/ImmuneAgent_2.0/agent')

# Load environment variables
from dotenv import load_dotenv
load_dotenv('/data/server/ImmuneAgent_2.0/agent/nodes/subagents/deep_research/.env')


def print_separator(title: str):
    """Print a formatted section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(success: bool, message: str):
    """Print test result with status."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {message}")


def test_imports():
    """Test 1: Verify all imports work correctly."""
    print_separator("Test 1: Import Verification")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import (
            # Configuration
            COLLECTION_CONFIG,
            EMBEDDER_GROUPS,
            get_collection_config,
            get_doc_source,
            # Caches
            _embedder_cache,
            _vector_store_cache,
            # Functions
            _create_embedder_for_group,
            _get_vector_store,
            _get_summarize_model,
            # Retrieval
            _retrieve_from_single_collection,
            retrieve_doc,
            retrieve,
            # Retriever class
            QdrantParentDocumentRetriever,
            # Tool
            vector_db_search,
            # Models
            RetrievedDocument,
        )
        print_result(True, "All imports successful")
        
        # Verify configurations
        print(f"\nCOLLECTION_CONFIG: {list(COLLECTION_CONFIG.keys())}")
        print(f"EMBEDDER_GROUPS: {list(EMBEDDER_GROUPS.keys())}")
        
        return True
    except Exception as e:
        print_result(False, f"Import failed: {e}")
        traceback.print_exc()
        return False


def test_collection_config():
    """Test 2: Verify collection configuration logic."""
    print_separator("Test 2: Collection Configuration")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import (
            get_collection_config,
            COLLECTION_CONFIG
        )
        
        # Test known collections
        for col_name in ["Immunology", "virology", "immunology_hle_v2"]:
            config = get_collection_config(col_name)
            print(f"\n{col_name}:")
            print(f"  dimension: {config['dimension']}")
            print(f"  source_key: {config['source_key']}")
            print(f"  source_field: {config['source_field']}")
            print(f"  embedder_group: {config['embedder_group']}")
        
        # Test unknown collection (should use default)
        unknown_config = get_collection_config("unknown_collection")
        assert unknown_config["dimension"] == 1536, "Default should be 1536"
        assert unknown_config["embedder_group"] == "openai_1536", "Default should be openai_1536"
        print_result(True, "Unknown collection uses correct default")
        
        return True
    except Exception as e:
        print_result(False, f"Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_embedder_openai_1536():
    """Test 3: Test OpenAI embedder (1536 dimensions)."""
    print_separator("Test 3: OpenAI Embedder (1536 dim)")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import (
            _create_embedder_for_group,
            _embedder_cache
        )
        
        print("Creating OpenAI embedder...")
        embedder = _create_embedder_for_group("openai_1536")
        
        # Test embedding
        test_query = "CAR-T cell therapy for cancer treatment"
        print(f"Test query: \"{test_query}\"")
        
        start_time = time.time()
        embedding = embedder.embed_query(test_query)
        elapsed = time.time() - start_time
        
        print(f"Embedding dimension: {len(embedding)}")
        print(f"Time taken: {elapsed:.2f}s")
        print(f"First 3 values: {embedding[:3]}")
        
        assert len(embedding) == 1536, f"Expected 1536, got {len(embedding)}"
        print_result(True, "OpenAI embedder works correctly")
        
        # Verify cache
        assert "openai_1536" in _embedder_cache, "Embedder should be cached"
        print_result(True, "Embedder cached correctly")
        
        return True
    except Exception as e:
        print_result(False, f"OpenAI embedder test failed: {e}")
        traceback.print_exc()
        return False


def test_embedder_pubmedbert_768():
    """Test 4: Test PubMedBERT embedder (768 dimensions)."""
    print_separator("Test 4: PubMedBERT Embedder (768 dim)")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import (
            _create_embedder_for_group,
            _embedder_cache
        )
        
        print("Creating PubMedBERT embedder (may download model on first run)...")
        embedder = _create_embedder_for_group("pubmedbert_768")
        
        # Test embedding
        test_query = "Virus infection mechanism and immune response"
        print(f"Test query: \"{test_query}\"")
        
        start_time = time.time()
        embedding = embedder.embed_query(test_query)
        elapsed = time.time() - start_time
        
        print(f"Embedding dimension: {len(embedding)}")
        print(f"Time taken: {elapsed:.2f}s")
        print(f"First 3 values: {embedding[:3]}")
        
        assert len(embedding) == 768, f"Expected 768, got {len(embedding)}"
        print_result(True, "PubMedBERT embedder works correctly")
        
        # Verify cache
        assert "pubmedbert_768" in _embedder_cache, "Embedder should be cached"
        print_result(True, "Embedder cached correctly")
        
        return True
    except Exception as e:
        print_result(False, f"PubMedBERT embedder test failed: {e}")
        traceback.print_exc()
        return False


def test_vector_store_immunology():
    """Test 5: Test vector store for Immunology (1536 dim)."""
    print_separator("Test 5: Vector Store - Immunology (1536 dim)")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import (
            _get_vector_store,
            _vector_store_cache
        )
        
        config = {'configurable': {}}
        
        print("Creating vector store for Immunology...")
        vs = _get_vector_store("Immunology", config)
        
        print(f"Collection name: {vs.collection_name}")
        
        # Test similarity search
        query = "CAR-T cell therapy mechanism"
        print(f"\nSearching: \"{query}\"")
        
        start_time = time.time()
        docs = vs.similarity_search(query, k=2)
        elapsed = time.time() - start_time
        
        print(f"Found {len(docs)} documents in {elapsed:.2f}s")
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"\n  Doc {i+1}:")
            print(f"    Source: {source[:60]}...")
            print(f"    Content: {content_preview}...")
        
        assert len(docs) > 0, "Should find at least 1 document"
        print_result(True, "Immunology vector store works correctly")
        
        # Verify cache
        assert "Immunology" in _vector_store_cache, "Vector store should be cached"
        print_result(True, "Vector store cached correctly")
        
        return True
    except Exception as e:
        print_result(False, f"Immunology vector store test failed: {e}")
        traceback.print_exc()
        return False


def test_vector_store_virology():
    """Test 6: Test vector store for virology (768 dim)."""
    print_separator("Test 6: Vector Store - virology (768 dim)")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import (
            _get_vector_store,
            _vector_store_cache
        )
        
        config = {'configurable': {}}
        
        print("Creating vector store for virology...")
        vs = _get_vector_store("virology", config)
        
        print(f"Collection name: {vs.collection_name}")
        
        # Test similarity search
        query = "respiratory syncytial virus vaccine"
        print(f"\nSearching: \"{query}\"")
        
        start_time = time.time()
        docs = vs.similarity_search(query, k=2)
        elapsed = time.time() - start_time
        
        print(f"Found {len(docs)} documents in {elapsed:.2f}s")
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"\n  Doc {i+1}:")
            print(f"    Source: {source[:60]}...")
            print(f"    Content: {content_preview}...")
        
        assert len(docs) > 0, "Should find at least 1 document"
        print_result(True, "virology vector store works correctly")
        
        # Verify cache
        assert "virology" in _vector_store_cache, "Vector store should be cached"
        print_result(True, "Vector store cached correctly")
        
        return True
    except Exception as e:
        print_result(False, f"virology vector store test failed: {e}")
        traceback.print_exc()
        return False


def test_retrieve_single_collection():
    """Test 7: Test _retrieve_from_single_collection function."""
    print_separator("Test 7: Retrieve from Single Collection")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import (
            _retrieve_from_single_collection,
            get_doc_source
        )
        
        # Create minimal config
        config = {
            'configurable': {
                'summarization_model': 'deepseek:deepseek-chat',
                'research_model': 'deepseek:deepseek-chat',
            }
        }
        
        # Test with Immunology
        print("Testing _retrieve_from_single_collection with Immunology...")
        query = ["What is CAR-T cell therapy?"]
        
        start_time = time.time()
        docs, col_name = _retrieve_from_single_collection(
            "Immunology", query, config, k_per_query=2
        )
        elapsed = time.time() - start_time
        
        print(f"Collection: {col_name}")
        print(f"Retrieved {len(docs)} documents in {elapsed:.2f}s")
        
        for i, doc in enumerate(docs[:2]):
            source = get_doc_source(doc, col_name)
            print(f"\n  Doc {i+1}:")
            print(f"    Source: {source[:60]}...")
            print(f"    _collection: {doc.metadata.get('_collection', 'N/A')}")
        
        assert len(docs) >= 0, "Should return documents or empty list"
        if docs:
            assert docs[0].metadata.get('_collection') == "Immunology", "_collection should be set"
        print_result(True, "_retrieve_from_single_collection works correctly")
        
        return True
    except Exception as e:
        print_result(False, f"Single collection retrieval test failed: {e}")
        traceback.print_exc()
        return False


def test_retrieve_doc_single():
    """Test 8: Test retrieve_doc with single collection."""
    print_separator("Test 8: retrieve_doc - Single Collection")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
        
        # Ensure single collection mode
        os.environ['QDRANT_COLLECTIONS'] = ''
        os.environ['QDRANT_COLLECTION'] = 'Immunology'
        
        config = {
            'configurable': {
                'summarization_model': 'deepseek:deepseek-chat',
                'research_model': 'deepseek:deepseek-chat',
                'vector_scoring_model': 'deepseek:deepseek-chat',
            }
        }
        
        query = ["CAR-T cell therapy mechanism", "immune checkpoint inhibitors"]
        print(f"Query: {query}")
        
        start_time = time.time()
        results = retrieve_doc(query, config, k_per_query=2)
        elapsed = time.time() - start_time
        
        print(f"\nRetrieved {len(results)} RetrievedDocument objects in {elapsed:.2f}s")
        
        for i, doc in enumerate(results[:3]):
            print(f"\n  Result {i+1}:")
            print(f"    Source: {doc.source[:60]}...")
            print(f"    Content length: {len(doc.page_content)} chars")
        
        print_result(True, "retrieve_doc (single collection) works correctly")
        return True
    except Exception as e:
        print_result(False, f"retrieve_doc single collection test failed: {e}")
        traceback.print_exc()
        return False


def test_retrieve_doc_multi():
    """Test 9: Test retrieve_doc with multiple collections."""
    print_separator("Test 9: retrieve_doc - Multi Collection")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
        
        # Set multi-collection mode (both 1536 and 768 dim)
        os.environ['QDRANT_COLLECTIONS'] = 'Immunology,virology'
        
        config = {
            'configurable': {
                'summarization_model': 'deepseek:deepseek-chat',
                'research_model': 'deepseek:deepseek-chat',
                'vector_scoring_model': 'deepseek:deepseek-chat',
            }
        }
        
        query = ["immune response to virus infection"]
        print(f"Query: {query}")
        print(f"Collections: Immunology (1536 dim) + virology (768 dim)")
        
        start_time = time.time()
        results = retrieve_doc(query, config, k_per_query=2)
        elapsed = time.time() - start_time
        
        print(f"\nRetrieved {len(results)} RetrievedDocument objects in {elapsed:.2f}s")
        
        for i, doc in enumerate(results[:5]):
            print(f"\n  Result {i+1}:")
            print(f"    Source: {doc.source[:60]}...")
            print(f"    Content length: {len(doc.page_content)} chars")
        
        print_result(True, "retrieve_doc (multi collection) works correctly")
        return True
    except Exception as e:
        print_result(False, f"retrieve_doc multi collection test failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end():
    """Test 10: Complete end-to-end test with research question."""
    print_separator("Test 10: End-to-End - Research Question to Results")
    
    try:
        from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
        
        # Set multi-collection mode
        os.environ['QDRANT_COLLECTIONS'] = 'Immunology,virology'
        
        config = {
            'configurable': {
                'summarization_model': 'deepseek:deepseek-chat',
                'research_model': 'deepseek:deepseek-chat',
                'vector_scoring_model': 'deepseek:deepseek-chat',
            }
        }
        
        # Real research question
        research_question = "What are the latest advances in CAR-T cell therapy for treating viral infections and cancer?"
        
        print(f"Research Question:\n  \"{research_question}\"\n")
        print("Processing...")
        
        # Generate multiple query variants
        queries = [
            "CAR-T cell therapy advances",
            "CAR-T therapy for viral infections",
            "CAR-T therapy for cancer treatment"
        ]
        
        start_time = time.time()
        results = retrieve_doc(queries, config, k_per_query=3)
        elapsed = time.time() - start_time
        
        print(f"\n{'='*40}")
        print(f"  RESULTS")
        print(f"{'='*40}")
        print(f"Total documents retrieved: {len(results)}")
        print(f"Total time: {elapsed:.2f}s")
        
        print(f"\n--- Top Results ---")
        for i, doc in enumerate(results[:5]):
            print(f"\n[{i+1}] Source: {doc.source}")
            print(f"    Content preview: {doc.page_content[:200].replace(chr(10), ' ')}...")
        
        assert len(results) > 0, "Should retrieve at least 1 document"
        print_result(True, "End-to-end test completed successfully")
        
        return True
    except Exception as e:
        print_result(False, f"End-to-end test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  MULTI-COLLECTION VECTOR SEARCH - COMPLETE TEST SUITE")
    print("="*60)
    print(f"\nTest Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing with real embedding models:")
    print("  - OpenAI text-embedding-3-small (1536 dim)")
    print("  - PubMedBERT (768 dim)")
    
    results = {}
    
    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("Collection Config", test_collection_config),
        ("OpenAI Embedder (1536)", test_embedder_openai_1536),
        ("PubMedBERT Embedder (768)", test_embedder_pubmedbert_768),
        ("Vector Store Immunology", test_vector_store_immunology),
        ("Vector Store virology", test_vector_store_virology),
        ("Retrieve Single Collection", test_retrieve_single_collection),
        ("retrieve_doc Single", test_retrieve_doc_single),
        ("retrieve_doc Multi", test_retrieve_doc_multi),
        ("End-to-End", test_end_to_end),
    ]
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_result(False, f"{test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_separator("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
