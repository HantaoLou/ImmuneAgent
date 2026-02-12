"""X-Masters Solver Tools

Provides simplified, synchronous wrappers for web search and knowledge base retrieval.
These tools are injected into the CodeActAgent's execution namespace.

Design decisions:
    - web_search: Uses TavilyClient (sync) directly, no async wrapping needed
    - knowledge_search: Uses retrieve_doc() which is already synchronous
    - Both load configs from .env via load_dotenv()
    - Returns formatted strings suitable for LLM consumption
"""

import os

# Disable HuggingFace network requests to avoid Thread-auto_conversion errors
# when huggingface.co is unreachable
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
_env_loaded = False


def _ensure_env_loaded():
    """Load .env file if not already loaded.
    
    Searches for .env in multiple locations (priority order):
        1. agent/nodes/subagents/deep_research/.env (has multi-collection config)
        2. open_deep_research/.env (fallback)
        3. Project root .env
    """
    global _env_loaded
    if _env_loaded:
        return
    
    from dotenv import load_dotenv
    
    # Find project root (ImmuneAgent_2.0)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent.parent  # x_masters -> subagents -> nodes -> agent -> root
    
    # Priority 1: deep_research/.env (has QDRANT_COLLECTIONS with multiple collections)
    deep_research_env = current_dir.parent / "deep_research" / ".env"
    if deep_research_env.exists():
        load_dotenv(deep_research_env, override=True)
        logger.info(f"Loaded environment from {deep_research_env}")
    
    # Priority 2: open_deep_research/.env (fallback for other configs)
    odr_env = project_root / "open_deep_research" / ".env"
    if odr_env.exists():
        load_dotenv(odr_env, override=False)  # Don't override deep_research config
        logger.info(f"Loaded environment from {odr_env}")
    
    # Priority 3: root .env if exists
    root_env = project_root / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)
        logger.info(f"Loaded environment from {root_env}")
    
    _env_loaded = True


# ---------------------------------------------------------------------------
# Web Search (Tavily)
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information using Tavily.
    
    This is a synchronous wrapper around Tavily's search API.
    Use this tool when you need to find current information, facts,
    or research findings from the internet.
    
    Args:
        query: Search query string (natural language)
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted string containing search results with titles, URLs, and content
        
    Example:
        >>> results = web_search("CAR-T cell therapy mechanism")
        >>> print(results)
    """
    _ensure_env_loaded()
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "[Error] TAVILY_API_KEY not configured. Cannot perform web search."
    
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=False,  # Keep response concise
        )
        
        # Format results for LLM consumption
        results = response.get("results", [])
        if not results:
            return f"[Web Search] No results found for: {query}"
        
        formatted = [f"[Web Search Results for: {query}]\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")[:500]  # Truncate long content
            formatted.append(f"\n--- Result {i} ---")
            formatted.append(f"Title: {title}")
            formatted.append(f"URL: {url}")
            formatted.append(f"Content: {content}")
        
        return "\n".join(formatted)
        
    except ImportError:
        return "[Error] tavily package not installed. Run: pip install tavily-python"
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"[Error] Web search failed: {str(e)}"


# ---------------------------------------------------------------------------
# Knowledge Base Search (Qdrant Vector DB)
# ---------------------------------------------------------------------------
def knowledge_search(query: str, k: int = 5, collections: Optional[List[str]] = None) -> str:
    """Search the internal knowledge base using vector similarity.
    
    This tool queries the Qdrant vector database for relevant documents
    based on semantic similarity. Use this for accessing:
    - Historical research data and experimental results
    - Internal publications and technical reports
    - Domain-specific knowledge (immunology, genomics, etc.)
    
    Args:
        query: Search query string (natural language)
        k: Number of results to return per query (default: 5)
        collections: Specific collections to search (default: uses QDRANT_COLLECTIONS env var)
        
    Returns:
        Formatted string containing retrieved documents with sources
        
    Example:
        >>> results = knowledge_search("CRISPR gene editing protocols")
        >>> print(results)
    """
    _ensure_env_loaded()
    
    # Check Qdrant configuration
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = os.getenv("QDRANT_PORT")
    
    if not qdrant_host or not qdrant_port:
        return "[Error] Qdrant not configured (missing QDRANT_HOST or QDRANT_PORT). Cannot perform knowledge search."
    
    # Set QDRANT_COLLECTIONS if not already set
    # This is needed by retrieve_doc() — it reads QDRANT_COLLECTIONS (plural)
    if collections:
        os.environ["QDRANT_COLLECTIONS"] = ",".join(collections)
    elif not os.getenv("QDRANT_COLLECTIONS"):
        # Fall back to QDRANT_COLLECTION (singular) if set
        single_collection = os.getenv("QDRANT_COLLECTION")
        if single_collection:
            os.environ["QDRANT_COLLECTIONS"] = single_collection
        else:
            return "[Error] No Qdrant collections configured. Set QDRANT_COLLECTIONS environment variable."
    
    try:
        # Import retrieve_doc (synchronous function)
        import sys
        agent_dir = Path(__file__).parent.parent.parent.parent
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        
        from nodes.subagents.deep_research.vector_search_tool import retrieve_doc
        
        # retrieve_doc expects a list of queries and returns RetrievedDocument objects
        docs = retrieve_doc(query=[query], config=None, k_per_query=k)
        
        if not docs:
            return f"[Knowledge Search] No results found for: {query}"
        
        # Format results for LLM consumption
        formatted = [f"[Knowledge Search Results for: {query}]\n"]
        for i, doc in enumerate(docs, 1):
            source = getattr(doc, 'source', 'Unknown source')
            content = getattr(doc, 'page_content', str(doc))[:800]  # Truncate
            formatted.append(f"\n--- Document {i} ---")
            formatted.append(f"Source: {source}")
            formatted.append(f"Content: {content}")
        
        return "\n".join(formatted)
        
    except ImportError as e:
        logger.error(f"Knowledge search import failed: {e}")
        return f"[Error] Knowledge search dependencies not available: {str(e)}"
    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        return f"[Error] Knowledge search failed: {str(e)}"


# ---------------------------------------------------------------------------
# Tool injection helper
# ---------------------------------------------------------------------------
def get_solver_tools() -> dict:
    """Get all tools to inject into the Solver's execution namespace.
    
    Returns:
        Dict mapping function names to function objects.
        These will be added to executor._persistent_namespace.
    """
    base_tools = {
        "web_search": web_search,
        "knowledge_search": knowledge_search,
    }

    # Add biomedical database query tools (simplified wrappers)
    try:
        from nodes.subagents.x_masters.biomedical_tools import get_biomedical_tools
        bio_tools = get_biomedical_tools()
        base_tools.update(bio_tools)
        logger.info(f"Loaded {len(bio_tools)} biomedical tools")
    except Exception as e:
        logger.warning(f"Failed to load biomedical tools: {e}")

    return base_tools


def get_lightweight_tools() -> dict:
    """Get tools WITHOUT knowledge_search for downstream agents (Critic/Rewriter/Selector).

    These agents receive pre-retrieved context via their prompt, so they don't
    need the expensive knowledge_search tool.  They keep web_search for real-time
    fact-checking and biomedical DB tools for structured queries.

    Returns:
        Dict mapping function names to function objects.
    """
    base_tools = {
        "web_search": web_search,
    }

    try:
        from nodes.subagents.x_masters.biomedical_tools import get_biomedical_tools
        bio_tools = get_biomedical_tools()
        base_tools.update(bio_tools)
    except Exception as e:
        logger.warning(f"Failed to load biomedical tools: {e}")

    return base_tools


def inject_tools_to_namespace(namespace: dict) -> None:
    """Inject all Solver tools into the given namespace.
    
    Args:
        namespace: The execution namespace dict (e.g., executor._persistent_namespace)
    """
    tools = get_solver_tools()
    namespace.update(tools)
    logger.info(f"Injected {len(tools)} tools into namespace: {list(tools.keys())}")


def inject_lightweight_tools_to_namespace(namespace: dict) -> None:
    """Inject lightweight tools (no knowledge_search) into the given namespace.

    Args:
        namespace: The execution namespace dict.
    """
    tools = get_lightweight_tools()
    namespace.update(tools)
    logger.info(f"Injected {len(tools)} lightweight tools into namespace: {list(tools.keys())}")


def make_tracked_knowledge_search(collector: list):
    """Create a wrapped knowledge_search that records results into a collector.

    The returned function has the same signature as knowledge_search and returns
    the same result, but also appends each result to the collector list.  This
    allows the caller to harvest all knowledge_search results after the agent
    finishes, without modifying the agent's execution logic.

    Args:
        collector: A mutable list that will accumulate search results.
                   Each entry is the raw result string from knowledge_search.

    Returns:
        A wrapped knowledge_search function.
    """
    def tracked_knowledge_search(query, k=5, collections=None):
        result = knowledge_search(query, k, collections)
        # Only collect meaningful results (skip errors and empty results)
        if not result.startswith("[Error") and "No results found" not in result:
            collector.append(result)
        return result
    return tracked_knowledge_search


def merge_search_results(all_results: list[str]) -> str:
    """Merge and deduplicate search results from multiple Solver instances.

    Args:
        all_results: List of raw knowledge_search result strings from all Solvers.

    Returns:
        Merged context string, or empty string if no results.
    """
    if not all_results:
        return ""

    # Deduplicate by content (different Solvers may search the same thing)
    seen = set()
    unique = []
    for result in all_results:
        # Use first 200 chars as dedup key (same document = same prefix)
        key = result[:200]
        if key not in seen:
            seen.add(key)
            unique.append(result)

    merged = "\n\n".join(unique)
    logger.info(f"Merged {len(all_results)} search results → {len(unique)} unique, {len(merged)} chars")
    return merged
