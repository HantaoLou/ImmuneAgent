# some of the codes are copied from https://github.com/wylswz/langchain-examples

import argparse

from diskcache import Cache
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from kb.cli.filter import filter_by_entropy, read_and_chunk_pdfs
from kb.cli.url import load_url_docs
from kb.config import QdrantConfig, get_embedder
from kb.config.config import get_text_splitter

cache = Cache("tmp")

# be careful, this is related to embedder
VECTOR_SIZE = 768


def md5(filename):
    import codecs
    import hashlib

    return hashlib.md5(codecs.encode(filename)).hexdigest()


def _dedup_by_page_content(documents: list[Document]):
    hashes = set()
    ret = []
    # 需要先按照文章排序
    # 如果有两篇完全一样的文章，避免各过滤掉一半
    documents = sorted(documents, key=lambda x: x.metadata.get("source", ""))
    for doc in documents:
        h = md5(doc.page_content)
        if h not in hashes:
            hashes.add(h)
            ret.append(doc)
    return ret


def add_documents(collection_name, documents: list[Document], reload=False):
    documents = _dedup_by_page_content(documents=documents)
    config = QdrantConfig.from_env()
    client = config.get_client()
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    vector = Qdrant(
        client=client, embedding=get_embedder(), collection_name=collection_name
    )
    grouped = {}
    for doc in documents:
        key = doc.metadata["source"]
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(doc)
    for src, docs in grouped.items():
        if reload:
            points = vector.client.query_points(
                collection_name=collection_name,
                limit=65536,  # 应该够了吧。。。。。
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source", match=MatchValue(value=src)
                        )
                    ]
                ),
            ).points
            print(f"deleting {len(points)} old vectors with same source {src}")
            if points:
                deleted = vector.client.delete(
                    collection_name=collection_name,
                    points_selector=[p.id for p in points],
                )
                print(deleted)
        vector.add_documents(docs)


def init_vector_store(path, collection_name, reload=False):
    chunks = read_and_chunk_pdfs(path)
    add_documents(collection_name, _filter_chunks(chunks, 1.0 / 10), reload)


def _filter_chunks(chunks: list[Document], ratio: float):
    chunks_and_entropy = filter_by_entropy(chunks, 0)
    sorted_chunks = sorted(chunks_and_entropy, key=lambda x: x[1], reverse=True)
    # Drop 1/10 document with lowest entropy
    drop_cnt = int(len(sorted_chunks) * ratio)
    documents = [chunk for chunk, _ in sorted_chunks[:-drop_cnt]]
    return documents


def load_url(args):
    url = args.url
    collection_name = args.collection_name
    reload = bool(args.reload)
    dryrun = bool(args.dryrun)
    docs = load_url_docs(url)
    chunked = get_text_splitter().split_documents(docs)
    filtered = _filter_chunks(chunked, 1.0 / 10)
    if not dryrun:
        add_documents(collection_name, filtered, reload)
    else:
        for d in filtered:
            print(d.page_content)
            print("=" * 20)


def load_doc(args):
    """Load a document into the vector store."""
    path = args.path
    reload = bool(args.reload)

    # Initialize vector store with the document
    init_vector_store(path, args.collection_name, reload)
    print(f"Successfully loaded document: {path}")


def query(args):
    collection_name = args.collection_name
    query_str = args.query
    top_k = args.top_k
    from kb.vectorstore import get_vector_store

    store = get_vector_store(collection_name)
    res = store.similarity_search(query_str, k=top_k)
    for d in res:
        print(d.metadata)
        print(d.page_content)
        print("=" * 20)


def drop_collection(args):
    collection_name = args.collection_name
    QdrantConfig.from_env().get_client().delete_collection(collection_name)
    print(f"Collection '{collection_name}' has been dropped from the vector store.")


def list_collections(args):
    collections = QdrantConfig.from_env().get_client().get_collections().collections
    for collection in collections:
        print(collection.name)


def register_commands(subparsers: argparse.Action):
    """Register all available commands with their arguments."""
    # Add load-doc command

    load_doc_parser = subparsers.add_parser(
        "load-doc", help="Load a document into the vector store"
    )
    load_doc_parser.add_argument("--path", type=str, help="Path to the document file")
    load_doc_parser.add_argument(
        "--collection_name", type=str, help="Collection name", default="default"
    )
    load_doc_parser.add_argument(
        "--reload", action="store_true", help="Remove existing vector from same source"
    )

    load_url_parser = subparsers.add_parser(
        "load-url", help="Load a document from url into the vector store"
    )
    load_url_parser.add_argument("--url", type=str, help="URL to the document file")
    load_url_parser.add_argument(
        "--collection_name", type=str, help="Collection name", default="default"
    )
    load_url_parser.add_argument(
        "--dryrun", action="store_true", help="Don't add to vector store"
    )
    load_url_parser.add_argument(
        "--reload", action="store_true", help="Remove existing vector from same source"
    )

    query = subparsers.add_parser("query", help="Query the vector store")
    query.add_argument(
        "--collection_name", type=str, help="Collection name", default="default"
    )
    query.add_argument("--query", type=str, help="Query string")
    query.add_argument(
        "--top_k", type=int, default=5, help="Number of top results to return"
    )

    drop_collection = subparsers.add_parser(
        "drop-collection", help="Drop a collection from the vector store"
    )
    drop_collection.add_argument(
        "--collection_name", type=str, help="Collection name to drop", default="default"
    )

    _ = subparsers.add_parser(
        "list-collections", help="List all collections in the vector store"
    )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Document loading and processing tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register all commands
    register_commands(subparsers)

    args = parser.parse_args()

    # Map commands to their handler functions
    command_handlers = {
        "load-doc": load_doc,
        "load-url": load_url,
        "query": query,
        "drop-collection": drop_collection,
        "list-collections": list_collections,
    }

    # Execute the command if it exists
    if args.command and args.command in command_handlers:
        handler = command_handlers[args.command]
        handler(args)
    else:
        parser.print_help()
