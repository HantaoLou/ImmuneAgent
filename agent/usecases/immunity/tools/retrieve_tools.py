import hashlib
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from diskcache import Cache
from kb.config.config import QdrantConfig
from kb.vectorstore.store import (
    KEY_SRC,
    QdrantParentDocumentRetriever,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from common.factory import get_summarize_model
from common.util.retrieval_utils import (
    clean_document_content,
    is_academic_noise,
    model_filter_and_rank,
    remove_think_tags,
)
from usecases.immunity.common.doi_utils import DOIMetadataRetriever
from usecases.immunity.common.pmc_api_utils import PMCAPIUtils
from usecases.immunity.common.utils import smart_truncate_abstract
from usecases.immunity.state.state import RetrievalState
from usecases.immunity.tools.web_retrieval_tools import WebRetrievalTool

# extract_citation_from_chunks function has been replaced by DOIMetadataRetriever
# Now directly use DOI tool to get more accurate and complete Citation information


c = Cache("/tmp/antibody_gen/web_search/")


def _check_api_key_and_clear_cache():
    """Check if API key has changed, clear search_one cache if it has"""
    cache_dir = Path("/tmp/antibody_gen/web_search/")
    api_key_file = cache_dir / ".current_api_key_hash"

    # Get current API key hash
    current_key = os.environ.get("TAVILY_API_KEY", "")
    current_hash = hashlib.md5(current_key.encode()).hexdigest()[:8]

    # Get stored hash
    stored_hash = ""
    if api_key_file.exists():
        stored_hash = api_key_file.read_text().strip()

    # Clear cache if key has changed
    if current_hash != stored_hash:
        # Clear search_one related cache
        keys_to_delete = []
        for key in c.iterkeys():
            if isinstance(key, tuple) and len(key) > 0:
                if key[0] == "tools.retrieve_tools.search_one":
                    keys_to_delete.append(key)

        for key in keys_to_delete:
            del c[key]

        # Update stored hash
        api_key_file.parent.mkdir(parents=True, exist_ok=True)
        api_key_file.write_text(current_hash)

        if keys_to_delete:
            print(f"API key changed, cleared {len(keys_to_delete)} cache entries")


def _filter_chunks(chunks: list[Document]) -> list[Document]:
    """Filter documents"""
    return filter(lambda x: not is_academic_noise(x.page_content), chunks)


def _create_embedder_from_immunity_config(config):
    """Create embedder from immunity_config"""
    embedding_model = config["configurable"]["model_config"]["embedding_model"]
    provider = embedding_model["provider"]
    model = embedding_model["model"]
    params = embedding_model.get("params", {})

    if provider == "OpenAI":  # 注意大小写
        api_key = params.get("api_key")
        base_url = params.get("base_url")

        return OpenAIEmbeddings(
            model=model, openai_api_key=api_key, openai_api_base=base_url
        )
    else:
        return OllamaEmbeddings(model=model, **params)


class RetrievedDocument(BaseModel):
    source: str = Field(
        description="source of the document, can be a url or a file path"
    )
    page_content: str = Field(description="content of the document")

    def __str__(self) -> str:
        return f"""
<document>
    <source>{self.source}</source>
    <content>{self.page_content}</content>
</document>
"""

    def __repr__(self) -> str:
        return self.__str__()


@tool(parse_docstring=True)
def retrieve_doc(
    query: List[str],
    config: RunnableConfig,
    k_per_query: int = 5,
    state: Optional[RetrievalState] = None,
) -> list[RetrievedDocument]:
    """
    Retrieve related documents from the knowledge base. Source of the document is in "source" field of metadata

    Args:
        query: List of query strings. It is recommended that each string can be 16 to 128 tokens in lenth, so that the query can capture the user's intention.
        config: RunnableConfig.
        k_per_query: number of retrived documents for each query.

    Returns:
        Retrieved documents with source and page content, where source is the path or url of original paper. Source of the document is included in <source> tag
    """
    try:
        # Create Qdrant configuration
        qdrant_config = QdrantConfig(
            host="117.148.176.36",
            port=6333,
            api_key=None,
            prefer_grpc=False,
            https=False,
            timeout=300,
        )

        # Create independent vector_store without depending on kb module
        embedder = _create_embedder_from_immunity_config(config)
        client = qdrant_config.get_client()

        vector_store = QdrantVectorStore(
            client=client, embedding=embedder, collection_name="Immunology"
        )

        retriever = QdrantParentDocumentRetriever(
            summarize_model=get_summarize_model(config),
            vector_store=vector_store,
            role="computational antibody design expert",
            retriever_kwargs={
                "search_type": "mmr",
                "search_kwargs": {"k": k_per_query, "lambda_mult": 0.65},
            },
            chunk_filter=_filter_chunks,
        )
        all_docs = []
        seen_docs = set()

        # Execute retrieval sequentially to avoid vector database concurrency conflicts
        results = []
        for q in query:
            try:
                docs = retriever.invoke(
                    q
                )  # Fix: pass single query string q instead of entire list query

                # Some retrievers may include <think/> tags
                # Initialize DOI metadata retriever
                doi_retriever = DOIMetadataRetriever()
                for doc in docs:
                    doc.page_content = remove_think_tags(doc.page_content)
                    # Get and process metadata information from original_chunks
                    original_chunks = doc.metadata.get("original_chunks")
                    if original_chunks:
                        print(f"Found {len(original_chunks)} original chunks")

                        # Check DOI field directly at metadata level, discard if not found
                        has_doi = False
                        doi_value = None
                        for chunk in original_chunks:
                            if hasattr(chunk, "metadata") and "doi" in chunk.metadata:
                                doi_value = chunk.metadata["doi"]
                                if doi_value and str(doi_value).strip():
                                    has_doi = True
                                    break

                        if not has_doi:
                            print(
                                "  ✗ DOI field not found in metadata or DOI is empty, discarding document"
                            )
                            print("-" * 50)
                            continue

                        # Use DOIMetadataRetriever to get Citation information
                        try:
                            # Get complete Citation information through DOI
                            citation = doi_retriever.get_metadata_by_doi(doi_value)

                            if citation:
                                # Only add to state.citations when state is not None
                                if state is not None:
                                    state.citations.append(citation)
                            else:
                                print(
                                    f"  ✗ Unable to get Citation information through DOI {doi_value}, discarding document"
                                )
                                print("-" * 50)

                        except Exception as e:
                            print(
                                f"  ✗ DOI metadata retrieval failed: {str(e)}, discarding document"
                            )
                            print("-" * 50)

                    else:
                        print("original_chunks data not found, discarding document")
                results.append(docs)
                print(f"Query '{q[:50]}...' retrieved {len(docs)} documents")
            except Exception as e:
                print(f"Retriever query failed: {e}")
                results.append([])

        # Merge results and deduplicate
        total_retrieved = 0
        for docs in results:
            total_retrieved += len(docs)

            for doc in docs:
                doc_hash = hash(doc.page_content)
                if doc_hash not in seen_docs:
                    seen_docs.add(doc_hash)
                    all_docs.append(doc)

        # Content cleaning and filtering
        scored_docs = []
        for doc in all_docs:
            content = clean_document_content(doc.page_content.strip())
            if not is_academic_noise(content):
                doc.page_content = content
                scored_docs.append(doc)

        print(
            f"Total retrieved {total_retrieved} documents, {len(all_docs)} after deduplication, {len(scored_docs)} after cleaning"
        )

        # Model filtering and ranking
        if len(scored_docs) > 0:
            top_docs = model_filter_and_rank(scored_docs, query, config)
            print(
                f"Obtained {len(top_docs)} high-quality documents after model filtering"
            )
        else:
            top_docs = []
        return [
            RetrievedDocument(
                source=doc.metadata[KEY_SRC], page_content=doc.page_content
            )
            for doc in top_docs
        ]
    except Exception as e:
        import traceback

        print(f"Retrieval query failed: {e}")
        print(f"Detailed error information: {traceback.format_exc()}")
        return []


@tool(parse_docstring=True)
def retrieve(
    query: List[str],
    config: RunnableConfig,
    k_per_query: int = 10,
    state: Optional[RetrievalState] = None,
) -> str:
    """
    Retrieve related documents from the knowledge base.

    Args:
        query: List of query strings.
        config: RunnableConfig.
        k_per_query: number of retrived documents for each query

    Returns:
        Retrieved documents with page_content and source. The source of a document can be a url or a file path. Source of the document is included in <source> tag
    """
    try:
        top_docs = retrieve_doc.invoke(
            {"query": query, "k_per_query": k_per_query, "state": state}, config
        )
        # Build context
        context = "\n\n".join([str(doc) for doc in top_docs])
        return context
    except Exception as e:
        print(f"Retrieval query failed: {e}")
        return ""


def search_one(q: str, k_per_query: int = 20) -> str:
    _check_api_key_and_clear_cache()  # 检查API密钥变化并清除缓存

    academic_search = TavilySearch(
        max_results=k_per_query,
        include_answer=True,
        include_raw_content=True,
        search_depth="advanced",
        include_domains=[
            "pubmed.ncbi.nlm.nih.gov",
            "scholar.google.com",
            "arxiv.org",
            "biorxiv.org",
            "nature.com",
            "science.org",
            "cell.com",
            "pnas.org",
            "ncbi.nlm.nih.gov",
            "doi.org",
            "researchgate.net",
            "semanticscholar.org",
        ],
    )
    results = academic_search.invoke({"query": q})
    return results


@tool(parse_docstring=True)
def web_search_node(
    query: List[str],
    k_per_query: int = 10,
    k_total: int = 10,
    state: Optional[RetrievalState] = None,
    config: RunnableConfig = None,
) -> str:
    """
    Search for online resources related to the query.

    Args:
        query: List of query strings.
        k_per_query: number of retrived documents for each query.
        k_total: number of documents in final results.
        state: Optional[RetrievalState] = None,
        config: RunnableConfig = None,

    Returns:
        Search results.
    """
    print(f"Online search query: {query}")

    # Get TAVILY_API_KEY from config, use default if config is None or not configured
    tavily_api_key = None
    if config and "configurable" in config:
        tavily_api_key = config["configurable"].get("tavily_api_key")

    # Set TAVILY_API_KEY - prioritize config setting, then environment variable, finally default value
    if not os.environ.get("TAVILY_API_KEY"):
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        else:
            # Keep original default value as fallback
            from config.api_keys import APIKeys
            os.environ["TAVILY_API_KEY"] = APIKeys.TAVILY_API_KEY

    try:
        all_results = []  # Store all results
        seen_urls = set()  # URL set for deduplication

        for q in query:
            results = search_one(q, k_per_query)
            if "results" in results:
                # First filter noise content using is_academic_noise method, then filter content shorter than 500 characters
                filtered_results = []
                for result in results["results"]:
                    content = result.get("content", "")
                    if (
                        not is_academic_noise(content) and len(content) >= 500
                    ):  # Filter academic noise and insufficient length content
                        filtered_results.append(result)

                # Sort filtered results by score, take top 5 highest scoring records
                sorted_results = sorted(
                    filtered_results, key=lambda x: x.get("score", 0), reverse=True
                )[:5]

                # Deduplicate based on URL
                for result in sorted_results:
                    url = result.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(result)

        # Sort all results by score, take top 10
        final_results = sorted(
            all_results, key=lambda x: x.get("score", 0), reverse=True
        )[:k_total]

        # First filter query results to find DOI, put them in variable collection
        extracted_dois = []  # DOI variable collection
        pmc_utils = PMCAPIUtils()  # Create PMC API tool instance
        # Initialize DOI metadata retriever
        doi_retriever = DOIMetadataRetriever()
        for result in final_results:
            if result.get("url", ""):
                url = result.get("url", "")
                # Prioritize PMC API to get DOI (for PMC URLs)
                if pmc_utils.is_pmc_url(url):
                    doi = pmc_utils.get_doi_from_url(url)
                    if doi:  # If DOI found, add to collection
                        extracted_dois.append(doi)
                        # Get complete Citation information through DOI
                        citation = doi_retriever.get_metadata_by_doi(doi)

                        if citation:
                            # Print citation object JSON format data
                            import json

                            try:
                                # Convert Citation object to dictionary format
                                citation_dict = {
                                    "title": citation.title,
                                    "authors": citation.authors,
                                    "journal": citation.journal,
                                    "year": citation.year,
                                    "volume": citation.volume,
                                    "issue": getattr(citation, "issue", ""),
                                    "pages": citation.pages,
                                    "doi": citation.doi,
                                    "url": getattr(citation, "url", ""),
                                    "pmid": citation.pmid,
                                    "abstract": citation.abstract,
                                    "citation_key": getattr(
                                        citation, "citation_key", ""
                                    ),
                                }
                                print("Citation object JSON format data:")
                                print(
                                    json.dumps(
                                        citation_dict, ensure_ascii=False, indent=2
                                    )
                                )
                                print("-" * 50)
                            except Exception as e:
                                print(f"Error converting Citation to JSON: {e}")
                            # Only add to state.citations when state is not None
                            if state:
                                state.citations.append(citation)

        # Extract content and concatenate
        content_list = [
            f"""
<document>
    <source>{result.get("url", "")}</source>
    <content>{result.get("content", "")}</content>
</document>
"""
            for result in final_results
            if result.get("content", "")
        ]
        combined_content = "\n\n".join(content_list)
        return combined_content
    except Exception as e:
        print(f"Search query failed: {e}")
        return ""


@tool(parse_docstring=True)
async def web_retrieval_search(query: List[str]) -> str:
    """
    Web retrieval tool for fetching real scientific papers from web sources.

    Args:
        query: Search query for scientific papers
    """
    import json

    try:
        # Initialize web retrieval tool
        web_tool = WebRetrievalTool()
        # Merge results from all queries, sort by relevance_score and select top 5
        all_papers = []
        for q in query:
            papers = await web_tool.search_all_sources(q, max_per_source=5)
            # Sort by relevance_score in descending order, select top 5 most relevant papers
            if papers and hasattr(papers[0], "relevance_score"):
                sorted_papers = sorted(
                    papers, key=lambda p: getattr(p, "relevance_score", 0), reverse=True
                )
                top_papers = sorted_papers[:5]  # Select top 5 most relevant
                all_papers.extend(top_papers)
            else:
                # If no relevance_score field, use original logic
                all_papers.extend(papers)

        # Deduplicate based on abstract hash, similar to lines 180-188 logic
        seen_abstracts = set()
        unique_papers = []

        for paper in all_papers:
            # Get paper's abstract attribute, use empty string if not exists
            abstract = getattr(paper, "abstract", "")
            if abstract:
                # Calculate abstract hash for deduplication
                abstract_hash = hash(abstract)
                if abstract_hash not in seen_abstracts:
                    seen_abstracts.add(abstract_hash)
                    unique_papers.append(paper)
            else:
                # Keep paper even if abstract is empty
                unique_papers.append(paper)

        # Use deduplicated papers
        all_papers = unique_papers

        # Extract content and concatenate, format consistent with web_search_node
        content_list = [
            f"""
<document>
    <source>{paper.source}</source>
    <content>{smart_truncate_abstract(paper.abstract)}</content>
</document>
"""
            for paper in all_papers
            if paper.abstract  # Ensure abstract is not empty
        ]
        combined_content = "\n\n".join(content_list)
        return combined_content

    except Exception as e:
        # Return empty document format in case of exception
        return f"<document><source>error</source><content>Web retrieval failed: {str(e)}</content></document>"
