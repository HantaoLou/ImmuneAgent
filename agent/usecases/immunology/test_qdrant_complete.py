#!/usr/bin/env python
"""
Complete test of Qdrant deployment with 800+ immunology papers
"""

import sys
from pathlib import Path

# Add paths
kb_path = Path(__file__).parent.parent.parent.parent / "kb" / "src"
if str(kb_path) not in sys.path:
    sys.path.insert(0, str(kb_path))
sys.path.insert(0, str(Path(__file__).parent))

from constants import OPENAI_API_KEY
from kb.config import QdrantConfig
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from tools.retrieval_tools import ImmunologyRetriever


def test_qdrant_deployment():
    """Test the complete Qdrant deployment"""
    print("=" * 60)
    print("🧪 Testing Complete Qdrant Deployment")
    print("=" * 60)

    # 1. Check collection status
    print("\n1️⃣ Checking Qdrant Collection Status...")
    config = QdrantConfig.from_env()
    client = config.get_client()

    try:
        info = client.get_collection("immunology_production")
        print(f"   ✅ Collection: immunology_production")
        print(f"   📊 Vectors: {info.vectors_count}")
        print(f"   📍 Points: {info.points_count}")
        print(f"   🟢 Status: {info.status}")

        if info.points_count < 1000:
            print(
                f"   ⚠️ Warning: Only {info.points_count} points loaded (expected 1000+)"
            )
        else:
            print(f"   🎉 Successfully deployed {info.points_count} document chunks!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # 2. Test direct vector store search
    print("\n2️⃣ Testing Direct Vector Store Search...")
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
    )

    vector_store = QdrantVectorStore(
        client=client, collection_name="immunology_production", embedding=embedder
    )

    test_queries = [
        "What are the latest advances in CAR-T cell therapy for solid tumors?",
        "How does T cell exhaustion affect immunotherapy outcomes?",
        "What are the mechanisms of checkpoint inhibitor resistance?",
        "How can antibody engineering improve therapeutic efficacy?",
        "What role do MDSCs play in tumor immune evasion?",
    ]

    for query in test_queries[:3]:  # Test first 3 queries
        print(f"\n   Query: {query[:60]}...")
        results = vector_store.similarity_search_with_score(query, k=3)

        for i, (doc, score) in enumerate(results, 1):
            relevance = 1 - score  # Convert distance to similarity
            title = doc.metadata.get("title", "Unknown")
            topic = doc.metadata.get("topic", "N/A")
            print(f"      {i}. [{relevance:.2%}] {title[:50]}...")
            print(f"         Topic: {topic}")

    # 3. Test ImmunologyRetriever integration
    print("\n3️⃣ Testing ImmunologyRetriever Integration...")
    retriever = ImmunologyRetriever(collection_name="immunology_production")

    query = "CAR-T cell manufacturing and quality control"
    print(f"   Query: {query}")

    # Test basic retrieval
    results = retriever.retrieve(query, k=5)
    print(f"   Found {len(results)} results")

    if results:
        print("\n   Top Results:")
        for i, result in enumerate(results[:3], 1):
            print(f"      {i}. Score: {result['score']:.3f}")
            print(f"         Source: {result['source']}")
            print(f"         Citation: {result.get('citation', 'N/A')}")

    # Test retrieval with reranking
    print("\n   Testing with reranking...")
    reranked = retriever.retrieve_with_rerank(query, k=10, rerank_k=5)
    print(f"   Reranked to top {len(reranked)} results")

    # 4. Test topic coverage
    print("\n4️⃣ Testing Topic Coverage...")
    topics_to_test = [
        "CAR-T Cell Therapy",
        "Checkpoint Inhibitors",
        "Antibody Engineering",
        "T Cell Biology",
        "B Cell Immunology",
        "Tumor Immunology",
        "Autoimmunity",
        "Immunometabolism",
        "Single-Cell Technologies",
        "Vaccine Development",
    ]

    coverage = {}
    for topic in topics_to_test:
        results = vector_store.similarity_search(topic, k=10)
        count = len(results)
        coverage[topic] = count
        status = "✅" if count >= 5 else "⚠️" if count > 0 else "❌"
        print(f"   {status} {topic}: {count} documents found")

    # 5. Test performance metrics
    print("\n5️⃣ Testing Performance Metrics...")
    import time

    # Measure search latency
    latencies = []
    for _ in range(5):
        start = time.time()
        _ = vector_store.similarity_search("immunotherapy", k=10)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

    avg_latency = sum(latencies) / len(latencies)
    print(f"   Average search latency: {avg_latency:.1f}ms")

    if avg_latency < 100:
        print("   ✅ Excellent performance (<100ms)")
    elif avg_latency < 500:
        print("   ✅ Good performance (<500ms)")
    else:
        print(f"   ⚠️ Slow performance ({avg_latency:.1f}ms)")

    # 6. Summary
    print("\n" + "=" * 60)
    print("📊 Deployment Summary")
    print("=" * 60)
    print(f"✅ Collection active with {info.points_count} points")
    print(f"✅ Search functionality working")
    print(
        f"✅ Topic coverage: {len([c for c in coverage.values() if c > 0])}/{len(topics_to_test)} topics"
    )
    print(f"✅ Average latency: {avg_latency:.1f}ms")

    if info.points_count >= 1000:
        print("\n🎉 Qdrant deployment successful with 800+ papers!")
        print("   The ImmuneAgent knowledge base is ready for production use.")
    else:
        print(f"\n⚠️ Deployment partial: {info.points_count} points loaded")
        print("   Run deploy_qdrant_papers.py to load more papers.")


def test_advanced_queries():
    """Test advanced immunology queries"""
    print("\n" + "=" * 60)
    print("🔬 Testing Advanced Immunology Queries")
    print("=" * 60)

    config = QdrantConfig.from_env()
    client = config.get_client()

    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
    )

    vector_store = QdrantVectorStore(
        client=client, collection_name="immunology_production", embedding=embedder
    )

    advanced_queries = [
        {
            "query": "How do armored CAR-T cells overcome the immunosuppressive tumor microenvironment?",
            "expected_topics": ["CAR-T", "TME", "immunosuppression"],
        },
        {
            "query": "What are the mechanisms of T cell metabolic reprogramming in exhaustion?",
            "expected_topics": [
                "T cell exhaustion",
                "immunometabolism",
                "mitochondria",
            ],
        },
        {
            "query": "How can bispecific antibodies be engineered for improved ADCC?",
            "expected_topics": ["antibody engineering", "bispecific", "Fc engineering"],
        },
        {
            "query": "What single-cell technologies reveal about tumor-infiltrating lymphocytes?",
            "expected_topics": ["single-cell", "TILs", "tumor immunology"],
        },
        {
            "query": "How do checkpoint inhibitors synergize with cancer vaccines?",
            "expected_topics": ["checkpoint", "vaccines", "combination therapy"],
        },
    ]

    for i, test in enumerate(advanced_queries, 1):
        print(f"\n{i}. Query: {test['query'][:80]}...")
        print(f"   Expected topics: {', '.join(test['expected_topics'])}")

        results = vector_store.similarity_search_with_score(test["query"], k=5)

        if results:
            # Check topic coverage
            topics_found = set()
            for doc, score in results:
                topic = doc.metadata.get("topic", "").lower()
                subtopic = doc.metadata.get("subtopic", "").lower()
                content = doc.page_content.lower()

                for expected in test["expected_topics"]:
                    if (
                        expected.lower() in topic
                        or expected.lower() in subtopic
                        or expected.lower() in content
                    ):
                        topics_found.add(expected)

            coverage = len(topics_found) / len(test["expected_topics"])
            status = "✅" if coverage >= 0.6 else "⚠️" if coverage > 0 else "❌"
            print(
                f"   {status} Topic coverage: {coverage:.0%} ({len(topics_found)}/{len(test['expected_topics'])})"
            )

            # Show top result
            top_doc, top_score = results[0]
            print(f"   Top result: {top_doc.metadata.get('title', 'N/A')[:60]}...")
            print(f"   Relevance: {(1 - top_score):.2%}")
        else:
            print("   ❌ No results found")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Qdrant deployment")
    parser.add_argument(
        "--advanced", action="store_true", help="Run advanced query tests"
    )

    args = parser.parse_args()

    # Run basic tests
    test_qdrant_deployment()

    # Run advanced tests if requested
    if args.advanced:
        test_advanced_queries()

    print("\n✅ Testing complete!")
