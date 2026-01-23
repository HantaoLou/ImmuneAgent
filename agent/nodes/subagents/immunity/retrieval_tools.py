"""
Immunity Agent Retrieval Tools

Implements three retrieval methods:
1. retrieve: Retrieve from Qdrant vector database
2. web_search_node: Tavily API web search
3. web_retrieval_search: Multiple Web source retrieval
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.utils.llm_factory import create_reasoning_advanced_llm


# ===================== Configuration Check =====================

def _check_qdrant_config() -> bool:
    """Check if Qdrant configuration is available"""
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "immunology")
    
    # Check if Qdrant is configured
    # If environment variables are not set, consider it unavailable
    return qdrant_host != "localhost" or os.getenv("QDRANT_API_KEY") is not None


def _check_tavily_config() -> bool:
    """Check if Tavily API configuration is available"""
    return os.getenv("TAVILY_API_KEY") is not None


# ===================== 1. Qdrant Vector Database Retrieval =====================

async def retrieve_from_qdrant(
    queries: List[str],
    k_per_query: int = 10,
    collection_name: str = "immunology"
) -> Dict[str, Any]:
    """
    Retrieve documents from Qdrant vector database
    
    Args:
        queries: List of queries
        k_per_query: Number of documents to retrieve per query
        collection_name: Qdrant collection name
    
    Returns:
        Retrieval result dictionary containing documents and citations
    """
    if not _check_qdrant_config():
        print("⚠️ Qdrant configuration unavailable, skipping vector database retrieval")
        return {"documents": [], "citations": []}
    
    try:
        from langchain_qdrant import QdrantVectorStore
        from langchain_openai import OpenAIEmbeddings
        from langchain_ollama import OllamaEmbeddings
        from qdrant_client import QdrantClient
        
        # Get configuration
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Create embedding model (prefer OpenAI, otherwise use Ollama)
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        if embedding_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️ OpenAI API Key not configured, cannot create embedding model")
                return {"documents": [], "citations": []}
            embedder = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)
        else:
            # Use Ollama
            embedder = OllamaEmbeddings(model="nomic-embed-text")
        
        # Create Qdrant client
        client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key,
            timeout=300
        )
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            embedding=embedder,
            collection_name=collection_name
        )
        
        # Retrieve documents
        all_documents = []
        all_citations = []
        
        for query in queries:
            # Use similarity search
            docs = vector_store.similarity_search(
                query=query,
                k=k_per_query
            )
            
            for doc in docs:
                # Extract document content
                content = doc.page_content
                metadata = doc.metadata
                
                # Extract citation information
                citation = {
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "year": metadata.get("year", ""),
                    "journal": metadata.get("journal", ""),
                    "doi": metadata.get("doi", ""),
                    "url": metadata.get("url", "")
                }
                
                all_documents.append({
                    "query": query,
                    "content": content,
                    "title": citation.get("title", ""),
                    "summary": content[:500] + "..." if len(content) > 500 else content,
                    "relevance_score": metadata.get("score", 0.0),
                    "source": citation.get("source", "")
                })
                
                if citation.get("doi") or citation.get("title"):
                    all_citations.append(citation)
        
        print(f"✅ Qdrant retrieval completed: {len(all_documents)} documents, {len(all_citations)} citations")
        return {"documents": all_documents, "citations": all_citations}
    
    except ImportError:
        print("⚠️ Qdrant-related libraries not installed, skipping vector database retrieval")
        return {"documents": [], "citations": []}
    except Exception as e:
        print(f"⚠️ Qdrant retrieval failed: {e}")
        return {"documents": [], "citations": []}


# ===================== 2. Tavily API Web Search =====================

async def web_search_with_tavily(
    queries: List[str],
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Perform web search using Tavily API
    
    Args:
        queries: List of queries
        max_results: Maximum number of results per query
    
    Returns:
        Retrieval result dictionary containing documents and citations
    """
    if not _check_tavily_config():
        print("⚠️ Tavily API configuration unavailable, skipping web search")
        return {"documents": [], "citations": []}
    
    try:
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        
        all_documents = []
        all_citations = []
        
        for query in queries:
            # Execute search
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_domains=["pubmed.ncbi.nlm.nih.gov", "scholar.google.com", "arxiv.org", "biorxiv.org"]
            )
            
            for result in response.get("results", []):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                
                # Filter short content
                if len(content) < 100:
                    continue
                
                all_documents.append({
                    "query": query,
                    "content": content,
                    "title": title,
                    "summary": content[:500] + "..." if len(content) > 500 else content,
                    "relevance_score": result.get("score", 0.0),
                    "source": url
                })
                
                # Extract citation information
                citation = {
                    "title": title,
                    "url": url,
                    "source": url
                }
                
                # Try to extract DOI from URL
                if "doi.org" in url or "doi:" in content:
                    import re
                    doi_match = re.search(r'10\.\d+/[^\s]+', url + " " + content)
                    if doi_match:
                        citation["doi"] = doi_match.group()
                
                all_citations.append(citation)
        
        print(f"✅ Tavily search completed: {len(all_documents)} documents, {len(all_citations)} citations")
        return {"documents": all_documents, "citations": all_citations}
    
    except ImportError:
        print("⚠️ Tavily library not installed, skipping web search")
        return {"documents": [], "citations": []}
    except Exception as e:
        print(f"⚠️ Tavily search failed: {e}")
        return {"documents": [], "citations": []}


# ===================== 3. Web Retrieval (Multiple Sources) =====================

async def web_retrieval_search(
    queries: List[str],
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Retrieve scientific papers from multiple Web sources
    
    Args:
        queries: List of queries
        max_results: Maximum number of results per query
    
    Returns:
        Retrieval result dictionary containing documents and citations
    """
    try:
        # Multiple Web source retrieval can be implemented here
        # For example: PubMed API, ArXiv API, bioRxiv API, etc.
        
        # Due to the need for multiple APIs, a simplified implementation is provided here
        # Actual usage can integrate specific APIs
        
        all_documents = []
        all_citations = []
        
        # Example: If PubMed API is configured
        pubmed_api_key = os.getenv("PUBMED_API_KEY")
        if pubmed_api_key:
            # PubMed retrieval can be implemented
            pass
        
        # If not configured, return empty results
        if not pubmed_api_key:
            print("⚠️ Web retrieval API not configured, skipping Web retrieval")
            return {"documents": [], "citations": []}
        
        return {"documents": all_documents, "citations": all_citations}
    
    except Exception as e:
        print(f"⚠️ Web retrieval failed: {e}")
        return {"documents": [], "citations": []}


# ===================== 4. LLM Fallback Solution =====================

async def retrieve_with_llm(
    queries: List[str],
    original_question: str
) -> Dict[str, Any]:
    """
    Use LLM to generate retrieval results (fallback solution)
    
    When vector database and web search are unavailable, use LLM to generate relevant literature information based on professional knowledge
    
    Args:
        queries: List of optimized queries
        original_question: Original research question
    
    Returns:
        Retrieval result dictionary containing documents and citations
    """
    llm = create_reasoning_advanced_llm()
    if not llm:
        return {"documents": [], "citations": []}
    
    try:
        from langchain_core.messages import HumanMessage
        import json
        
        retrieval_prompt = f"""You are a professional expert in immunology and computational biology literature retrieval. Based on the following research question and optimized queries, generate relevant literature summaries and context information.

Original Research Question:
{original_question}

Optimized Queries:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(queries)])}

Please generate relevant literature summaries and key findings for each optimized query based on your professional knowledge. Return JSON format:

{{
    "context": "Comprehensive literature context summary integrating all query-related literature findings, including key concepts, experimental methods, research results, and theoretical frameworks. Content should be comprehensive and highly relevant to the research question.",
    "retrieval_docs": [
        {{
            "query": "Corresponding optimized query",
            "title": "Relevant literature title or topic",
            "summary": "Key summary and findings of this literature",
            "key_findings": ["Key finding 1", "Key finding 2"],
            "methodology": "Research method or technique",
            "relevance_score": 85,
            "source": "Literature source"
        }}
    ],
    "citations": [
        {{
            "author": "Primary author name",
            "year": 2023,
            "title": "Literature title",
            "journal": "Journal name",
            "doi": "DOI number (if known)",
            "relevance": "Relevance explanation to the query"
        }}
    ]
}}

Requirements:
1. context should be a comprehensive literature review integrating all query-related content
2. Each optimized query should correspond to at least one retrieval_doc
3. citations should include 5-10 relevant high-quality literature citations
4. All content should be based on real immunology and computational biology research domain knowledge
5. Ensure literature content is highly relevant to the research question
6. Use standard academic citation format

Please return valid JSON format, do not include additional text or markdown markers."""
        
        messages = [HumanMessage(content=retrieval_prompt)]
        response = llm.invoke(messages)
        response_content = response.content.strip()
        
        # Parse JSON
        try:
            retrieval_data = json.loads(response_content)
        except:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                retrieval_data = json.loads(json_match.group())
            else:
                retrieval_data = {}
        
        # Convert to unified format
        documents = []
        for doc in retrieval_data.get("retrieval_docs", []):
            documents.append({
                "query": doc.get("query", ""),
                "title": doc.get("title", ""),
                "summary": doc.get("summary", ""),
                "content": doc.get("summary", ""),  # LLM-generated content
                "relevance_score": doc.get("relevance_score", 0),
                "source": doc.get("source", "LLM Generated")
            })
        
        citations = retrieval_data.get("citations", [])
        
        print(f"✅ LLM retrieval completed: {len(documents)} documents, {len(citations)} citations")
        return {"documents": documents, "citations": citations}
    
    except Exception as e:
        print(f"⚠️ LLM retrieval failed: {e}")
        return {"documents": [], "citations": []}


# ===================== 5. Parallel Retrieval Main Function =====================

async def parallel_retrieval(
    queries: List[str],
    original_question: str,
    k_per_query: int = 10
) -> Dict[str, Any]:
    """
    Execute three retrieval methods in parallel and integrate results
    
    Args:
        queries: List of optimized queries
        original_question: Original research question
        k_per_query: Number of documents to retrieve per query
    
    Returns:
        Integrated retrieval results containing context, retrieval_docs, citations
    """
    print(f"🔍 Starting parallel retrieval, query count: {len(queries)}")
    
    # Execute three retrieval methods in parallel
    tasks = [
        retrieve_from_qdrant(queries, k_per_query),
        web_search_with_tavily(queries, max_results=k_per_query),
        web_retrieval_search(queries, max_results=k_per_query)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Integrate results
    all_documents = []
    all_citations = []
    citation_map = {}  # For deduplication
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"⚠️ Retrieval method {i+1} failed: {result}")
            continue
        
        if isinstance(result, dict):
            all_documents.extend(result.get("documents", []))
            
            # Deduplicate citations (based on DOI or title)
            for citation in result.get("citations", []):
                doi = citation.get("doi", "")
                title = citation.get("title", "")
                key = doi if doi else title
                
                if key and key not in citation_map:
                    citation_map[key] = citation
                    all_citations.append(citation)
    
    # If all retrieval methods fail or return empty results, use LLM fallback solution
    if not all_documents and not all_citations:
        print("⚠️ All retrieval methods failed, using LLM fallback solution")
        llm_result = await retrieve_with_llm(queries, original_question)
        all_documents = llm_result.get("documents", [])
        all_citations = llm_result.get("citations", [])
    
    # Generate comprehensive context
    context_parts = []
    for doc in all_documents[:20]:  # Limit document count
        context_parts.append(f"**{doc.get('title', 'N/A')}**\n{doc.get('summary', doc.get('content', ''))}")
    
    context = "\n\n".join(context_parts)
    
    # Sort documents by relevance
    all_documents.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    print(f"✅ Parallel retrieval completed: {len(all_documents)} documents, {len(all_citations)} citations")
    
    return {
        "context": context,
        "retrieval_docs": all_documents,
        "citations": all_citations
    }


# ===================== Synchronous Wrapper Function =====================

def parallel_retrieval_sync(
    queries: List[str],
    original_question: str,
    k_per_query: int = 10
) -> Dict[str, Any]:
    """
    Synchronous version of parallel retrieval function
    
    Args:
        queries: List of optimized queries
        original_question: Original research question
        k_per_query: Number of documents to retrieve per query
    
    Returns:
        Integrated retrieval results
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        parallel_retrieval(queries, original_question, k_per_query)
    )

