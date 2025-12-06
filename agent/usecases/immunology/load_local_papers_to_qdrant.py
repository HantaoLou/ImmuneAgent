#!/usr/bin/env python
"""
Load locally stored papers from kb/data into Qdrant
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Add paths
kb_path = Path(__file__).parent.parent.parent.parent / "kb" / "src"
if str(kb_path) not in sys.path:
    sys.path.insert(0, str(kb_path))
sys.path.insert(0, str(Path(__file__).parent))

from constants import OPENAI_API_KEY
from kb.config import QdrantConfig
from kb.config.config import ModelConfig, get_text_splitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Paths
KB_DATA_PATH = Path(__file__).parent.parent.parent.parent / "kb" / "data"
PAPERS_PATH = KB_DATA_PATH / "papers"
METADATA_PATH = KB_DATA_PATH / "metadata"


def load_local_papers() -> List[Document]:
    """Load papers from local kb/data folder"""
    documents = []

    if not PAPERS_PATH.exists():
        print(f"❌ No local papers found at: {PAPERS_PATH}")
        print("   Run download_papers_locally.py first to download papers")
        return documents

    print(f"📂 Loading papers from: {PAPERS_PATH}")

    # Load all text files
    for category_dir in PAPERS_PATH.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            papers = list(category_dir.glob("*.txt"))

            print(f"  📁 {category_name}: {len(papers)} papers")

            for paper_file in papers:
                # Load paper content
                with open(paper_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Try to load metadata
                paper_id = paper_file.stem
                metadata_file = METADATA_PATH / f"{paper_id}.json"

                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {
                        "source": paper_id,
                        "category": category_name,
                        "file": str(paper_file.name),
                    }

                # Create document
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

    print(f"\n✅ Loaded {len(documents)} papers from local storage")
    return documents


def load_to_qdrant(
    documents: List[Document],
    collection_name: str = "immunology_local",
    recreate: bool = False,
):
    """
    Load documents into Qdrant

    Args:
        documents: List of documents to load
        collection_name: Name of Qdrant collection
        recreate: Whether to recreate the collection
    """
    if not documents:
        print("❌ No documents to load")
        return

    print(f"\n🚀 Loading {len(documents)} documents to Qdrant")
    print(f"   Collection: {collection_name}")

    # Initialize Qdrant
    qdrant_config = QdrantConfig.from_env()
    client = qdrant_config.get_client()

    # Handle collection
    if recreate:
        try:
            client.delete_collection(collection_name)
            print(f"   🗑️ Deleted existing collection")
        except:
            pass

    # Create or verify collection
    try:
        info = client.get_collection(collection_name)
        print(f"   ✅ Using existing collection with {info.vectors_count} vectors")
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI embedding size
                distance=Distance.COSINE,
            ),
        )
        print(f"   📝 Created new collection")

    # Initialize embedder and vector store
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
    )

    vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embedder
    )

    # Chunk documents
    print("\n✂️ Chunking documents...")
    text_splitter = get_text_splitter(ModelConfig.from_env())
    chunked_docs = []

    for doc in tqdm(documents, desc="Chunking"):
        if len(doc.page_content) > 1000:
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
                chunked_docs.append(chunk)
        else:
            chunked_docs.append(doc)

    print(f"   Created {len(chunked_docs)} chunks")

    # Add to Qdrant
    print("\n💾 Adding to Qdrant...")
    batch_size = 50

    for i in tqdm(range(0, len(chunked_docs), batch_size), desc="Uploading"):
        batch = chunked_docs[i : i + batch_size]
        try:
            vector_store.add_documents(batch)
        except Exception as e:
            print(f"   ⚠️ Error in batch {i // batch_size}: {e}")

    # Verify
    info = client.get_collection(collection_name)
    print(f"\n✅ Upload complete!")
    print(f"   Collection: {collection_name}")
    print(f"   Total vectors: {info.vectors_count}")
    print(f"   Total points: {info.points_count}")

    # Test search
    print("\n🔍 Testing search...")
    test_query = "CAR-T cell therapy"
    results = vector_store.similarity_search(test_query, k=3)
    print(f"   Query: {test_query}")
    print(f"   Found {len(results)} results")

    if results:
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            category = doc.metadata.get("category", "Unknown")
            print(f"   {i}. Source: {source}, Category: {category}")


def merge_collections(
    source_collection: str = "immunology_local",
    target_collection: str = "immunology_production",
):
    """
    Merge local collection into production collection

    Args:
        source_collection: Source collection name
        target_collection: Target collection name
    """
    print(f"\n🔄 Merging {source_collection} -> {target_collection}")

    qdrant_config = QdrantConfig.from_env()
    client = qdrant_config.get_client()

    try:
        source_info = client.get_collection(source_collection)
        target_info = client.get_collection(target_collection)

        print(f"   Source: {source_info.points_count} points")
        print(f"   Target: {target_info.points_count} points")

        # In a real implementation, we would:
        # 1. Read all points from source
        # 2. Add them to target
        # 3. Handle duplicates

        print("   ⚠️ Merge operation would require batch point transfer")
        print("   Use Qdrant's scroll API for production implementation")

    except Exception as e:
        print(f"   ❌ Error: {e}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Load local papers to Qdrant")
    parser.add_argument(
        "--collection", default="immunology_local", help="Collection name"
    )
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    parser.add_argument(
        "--merge", action="store_true", help="Merge with production collection"
    )

    args = parser.parse_args()

    # Load local papers
    documents = load_local_papers()

    if documents:
        # Load to Qdrant
        load_to_qdrant(
            documents, collection_name=args.collection, recreate=args.recreate
        )

        # Merge if requested
        if args.merge:
            merge_collections(
                source_collection=args.collection,
                target_collection="immunology_production",
            )
    else:
        print("\n📝 To download papers locally, run:")
        print("   python download_papers_locally.py")


if __name__ == "__main__":
    main()
