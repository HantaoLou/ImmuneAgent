"""Search tools for ImmuneAgent.

Provides simplified, synchronous wrappers for web search and knowledge base retrieval.
These tools are injected into agent execution namespaces.

Design decisions:
    - web_search: Uses TavilyClient (sync) directly, no async wrapping needed
    - knowledge_search: Uses retrieve_doc() which is already synchronous
    - Both load configs from .env via load_dotenv()
    - Returns formatted strings suitable for LLM consumption

LangChain 1.0+ Compatibility:
    - All tools use @tool decorator from langchain_core.tools
    - Tools can be directly bound to LLM via .bind_tools()
    - Backward compatible with namespace injection pattern
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

from langchain_core.tools import tool

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

    # Find project root: tools/ -> agent/ -> project root
    agent_dir = Path(__file__).parent.parent
    project_root = agent_dir.parent

    # Priority 1: deep_research/.env (has QDRANT_COLLECTIONS with multiple collections)
    deep_research_env = agent_dir / "nodes" / "subagents" / "deep_research" / ".env"
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
@tool
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
    """
    _ensure_env_loaded()

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "[Error] TAVILY_API_KEY not configured. Cannot perform web search."

    try:
        import time as _time
        import random as _random
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)

        # Retry with exponential backoff to handle rate limiting
        # when multiple agents call Tavily concurrently
        max_retries = 3
        results = []
        for attempt in range(max_retries):
            try:
                response = client.search(
                    query=query,
                    max_results=max_results,
                    include_raw_content=False,
                )
                results = response.get("results", [])
                if results:
                    break
                # Empty results — might be rate-limited, retry
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + _random.uniform(0, 1)
                    logger.warning(f"web_search got 0 results (attempt {attempt+1}), retrying in {wait:.1f}s")
                    _time.sleep(wait)
            except Exception as retry_err:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + _random.uniform(0, 1)
                    logger.warning(f"web_search error (attempt {attempt+1}): {retry_err}, retrying in {wait:.1f}s")
                    _time.sleep(wait)
                else:
                    raise

        if not results:
            return f"[Web Search] No results found for: {query}"

        # Format results for LLM consumption
        formatted = [f"[Web Search Results for: {query}]\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")[:2000]  # Truncate long content
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
# Read Webpage Content
# ---------------------------------------------------------------------------
@tool
def read_webpage(url: str, max_chars: int = 10000) -> str:
    """Fetch and extract the main text content from a webpage URL.

    Use this tool when web_search returns a relevant URL (e.g. a paper, article,
    or documentation page) and you need to read the full content rather than
    just the search snippet. This is essential for understanding detailed
    methodology, results, and conclusions from scientific papers.

    This tool can read most URLs including PMC, PubMed, bioRxiv, Wiley,
    ScienceDirect, and other academic publishers that normally block direct
    access.

    Args:
        url: The URL to fetch (must start with http:// or https://)
        max_chars: Maximum characters to return (default: 10000)

    Returns:
        Extracted text content from the webpage, or an error message.
    """
    if not url.startswith(("http://", "https://")):
        return f"[Error] Invalid URL: {url}. Must start with http:// or https://"

    # --- Strategy 1: Tavily Extract (bypasses 403/reCAPTCHA) ---
    _ensure_env_loaded()
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.extract(urls=[url], extract_depth="advanced")
            results = response.get("results", [])
            if results:
                text = results[0].get("raw_content") or results[0].get("text", "")
                text = text.strip()
                if len(text) > 200:
                    if len(text) > max_chars:
                        text = text[:max_chars] + "\n\n[... content truncated ...]"
                    return f"[Webpage Content from: {url}]\n\n{text}"
        except Exception as e:
            logger.warning(f"Tavily extract failed for {url}: {e}")

    # --- Strategy 2: Direct HTTP fetch with trafilatura ---
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()

        html = raw.decode("utf-8", errors="replace")

        # Try trafilatura first (best for article extraction)
        try:
            import trafilatura
            text = trafilatura.extract(html, include_comments=False, include_tables=True)
            if text and len(text.strip()) > 200:
                text = text.strip()
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n[... content truncated ...]"
                return f"[Webpage Content from: {url}]\n\n{text}"
        except ImportError:
            pass

        # Fallback: simple HTML tag stripping
        import re as _re
        text = _re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=_re.DOTALL | _re.IGNORECASE)
        text = _re.sub(r'<[^>]+>', ' ', text)
        text = _re.sub(r'\s+', ' ', text).strip()
        import html as _html_mod
        text = _html_mod.unescape(text)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... content truncated ...]"

        if len(text.strip()) < 100:
            return f"[Error] Could not extract meaningful content from: {url}"

        return f"[Webpage Content from: {url}]\n\n{text}"

    except Exception as e:
        logger.error(f"read_webpage failed for {url}: {e}")
        return f"[Error] Failed to read {url}: {str(e)}"


# ---------------------------------------------------------------------------
# Knowledge Base Search (Qdrant Vector DB)
# ---------------------------------------------------------------------------
@tool
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
        agent_dir = Path(__file__).parent.parent  # tools/ -> agent/
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
# Tool registry
# ---------------------------------------------------------------------------
def get_search_tools() -> list:
    """Return a list of all search tool functions as LangChain tools.
    
    Returns:
        List of LangChain tool objects that can be directly bound to LLM.
        
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> tools = get_search_tools()
        >>> llm_with_tools = llm.bind_tools(tools)
    """
    return [web_search, read_webpage, knowledge_search]


def get_search_tools_dict() -> dict:
    """Return a dict of search tool functions for backward compatibility.
    
    This function is kept for backward compatibility with the existing
    namespace injection pattern.
    """
    return {
        "web_search": web_search,
        "read_webpage": read_webpage,
        "knowledge_search": knowledge_search,
    }
