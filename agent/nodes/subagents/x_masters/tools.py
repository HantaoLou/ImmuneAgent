"""X-Masters Solver Tools

Provides simplified, synchronous wrappers for web search and knowledge base retrieval.
These tools are injected into the CodeActAgent's execution namespace.

Design decisions:
    - web_search: Uses unified_search with multi-provider fallback (Tavily, SerpAPI, Bing, DuckDuckGo)
    - knowledge_search: Uses retrieve_doc() which is already synchronous
    - Both load configs from .env via load_dotenv()
    - Returns formatted strings suitable for LLM consumption

Integration Note:
    - Core search/memory tools are now imported from agent.tools (common_tools)
    - This module provides injection helpers and X-Masters-specific wrappers
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
# Web Search (Unified with multi-provider fallback)
# ---------------------------------------------------------------------------

# Global flag to indicate if web search is completely unavailable
_all_search_providers_failed = False


def _is_search_available() -> bool:
    """Check if any search provider is still available."""
    if _all_search_providers_failed:
        return False
    try:
        from agent.utils.unified_search import get_search_status
        status = get_search_status()
        available = any(
            s['enabled'] and s['has_api_key'] and not s['quota_exceeded']
            for s in status.values()
        )
        return available
    except:
        return True  # Assume available if can't check


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information using unified search with automatic fallback.
    
    This function tries multiple search providers in order:
    1. Tavily (primary) - Best for AI/LLM applications
    2. SerpAPI (fallback) - Google search results
    3. Bing Search API (fallback) - Microsoft search
    4. DuckDuckGo (free fallback) - Privacy-focused search
    
    When a provider's quota is exceeded, it automatically falls back to the next.
    
    Args:
        query: Search query string (natural language)
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted string containing search results with titles, URLs, and content
        
    Example:
        >>> results = web_search("CAR-T cell therapy mechanism")
        >>> print(results)
    """
    global _all_search_providers_failed
    
    # Check if search is already known to be unavailable
    if _all_search_providers_failed:
        return "[Search Unavailable] All search providers have failed. Please proceed with your existing knowledge and the provided context. Do not attempt more searches."
    
    _ensure_env_loaded()
    
    # ========== FIX: Smart query compression for providers with character limits ==========
    # Tavily has a 400 character limit. Compress long queries while preserving key information.
    MAX_QUERY_LENGTH = 350  # Leave some margin
    original_query = query
    if len(query) > MAX_QUERY_LENGTH:
        query = _compress_search_query(query, MAX_QUERY_LENGTH)
        logger.info(f"Query compressed from {len(original_query)} to {len(query)} chars: {query[:100]}...")
    
    try:
        from agent.utils.unified_search import web_search_unified, get_search_status
        
        # Log current search status
        status = get_search_status()
        available_providers = [p for p, s in status.items() if s['enabled'] and s['has_api_key'] and not s['quota_exceeded']]
        logger.info(f"Available search providers: {available_providers}")
        
        # If no providers available, mark as failed
        if not available_providers:
            _all_search_providers_failed = True
            return "[Search Unavailable] All search providers have failed or quota exceeded. Please proceed with your existing knowledge and the provided context. Do not attempt more searches."
        
        result = web_search_unified(query, max_results)
        
        # Check if result indicates all providers failed
        if "All search providers failed" in result:
            _all_search_providers_failed = True
            return f"{result}\n\n[Search Unavailable] Please proceed with your existing knowledge. Do not attempt more searches."
        
        return result
        
    except ImportError as e:
        logger.warning(f"Unified search not available, falling back to Tavily only: {e}")
        return _web_search_tavily_fallback(query, max_results)
    except Exception as e:
        logger.error(f"Unified search failed: {e}")
        return f"[Error] Web search failed: {str(e)}\n\n[Search Unavailable] Please proceed with your existing knowledge."


def _compress_search_query(query: str, max_length: int = 350) -> str:
    """Compress a long search query while preserving key information.
    
    Strategy:
    1. Extract key terms (scientific names, technical terms, important concepts)
    2. Remove filler words and redundant phrases
    3. Prioritize: core terms > conditions > context
    4. Keep the most informative parts
    
    Args:
        query: Original query string
        max_length: Maximum allowed length
        
    Returns:
        Compressed query string
    """
    import re
    
    # If already short enough, return as is
    if len(query) <= max_length:
        return query
    
    # Step 1: Extract key terms (capitalized words, scientific terms, numbers with units)
    # These are usually the most important for scientific/biomedical queries
    key_patterns = [
        r'\b[A-Z][a-z]+(?:\'s)?\b',  # Capitalized words (Watterson's, pi, etc.)
        r'\b[A-Z]{2,}\b',  # Acronyms (DNA, RNA, SNP)
        r'\b[a-z]+_[a-z]+\b',  # snake_case terms
        r'\b\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?\b',  # Numbers with optional units
        r'\b(?:estimator|diversity|nucleotide|variant|sample|bias|calculate|formula)\b',  # Key scientific terms
    ]
    
    key_terms = set()
    for pattern in key_patterns:
        matches = re.findall(pattern, query)
        key_terms.update(matches)
    
    # Step 2: Split into sentences and prioritize
    sentences = re.split(r'[.!?]', query)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Prioritize sentences that contain key terms
    scored_sentences = []
    for sent in sentences:
        score = sum(1 for term in key_terms if term.lower() in sent.lower())
        scored_sentences.append((score, sent))
    
    # Sort by score (descending) then by position (earlier is better for ties)
    scored_sentences.sort(key=lambda x: (-x[0], sentences.index(x[1]) if x[1] in sentences else 999))
    
    # Step 3: Build compressed query from best sentences
    compressed_parts = []
    current_length = 0
    
    for score, sent in scored_sentences:
        # Clean up the sentence
        sent = sent.strip()
        if not sent:
            continue
            
        # Estimate length with space
        new_length = current_length + len(sent) + 2  # +2 for ". " 
        
        if new_length <= max_length:
            compressed_parts.append(sent)
            current_length = new_length
        else:
            # Try to fit a truncated version
            remaining = max_length - current_length - 2
            if remaining > 30:  # Only if we can fit something meaningful
                truncated = sent[:remaining]
                # Find last complete word
                last_space = truncated.rfind(' ')
                if last_space > remaining // 2:
                    truncated = truncated[:last_space]
                compressed_parts.append(truncated + "...")
            break
    
    # If we got nothing, fall back to simple truncation with word boundary
    if not compressed_parts:
        truncated = query[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length // 2:
            return truncated[:last_space]
        return truncated
    
    # Join with proper punctuation
    result = '. '.join(compressed_parts)
    if not result.endswith('.') and not result.endswith('...'):
        result += '.'
    
    return result


def _web_search_tavily_fallback(query: str, max_results: int = 5) -> str:
    """Fallback to Tavily-only search if unified search fails."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "[Error] TAVILY_API_KEY not configured. Cannot perform web search."
    
    try:
        import time as _time
        import random as _random
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=api_key)
        
        # Single attempt (no retries - let caller handle fallback)
        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=False,
        )
        results = response.get("results", [])
        
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
def read_webpage(url: str, max_chars: int = 10000) -> str:
    """Fetch and extract the main text content from a webpage URL.

    Use this tool when web_search returns a relevant URL (e.g. a paper, article,
    or documentation page) and you need to read the full content rather than
    just the search snippet.  This is essential for understanding detailed
    methodology, results, and conclusions from scientific papers.

    This tool can read most URLs including PMC, PubMed, bioRxiv, Wiley,
    ScienceDirect, and other academic publishers that normally block direct
    access.

    Args:
        url: The URL to fetch (must start with http:// or https://)
        max_chars: Maximum characters to return (default: 10000)

    Returns:
        Extracted text content from the webpage, or an error message.

    Example:
        >>> text = read_webpage("https://pmc.ncbi.nlm.nih.gov/articles/PMC12345678/")
        >>> print(text[:500])
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
    
    This function now imports from common_tools (agent/tools/) as the single
    source of truth for all domain-specific tools.
    
    Tool categories loaded:
        - Search tools: web_search, read_webpage, knowledge_search
        - Memory tools: agent_memory
        - Biomedical tools: 24 database query functions
        - Literature tools: 6 academic search functions
        - Clinical tools: 5 clinical data functions
        - Reference tools: 10 biological database functions
        - Docx tools: parse_docx
    
    Returns:
        Dict mapping function names to function objects.
        These will be added to executor._persistent_namespace.
    """
    # Import from common_tools (single source of truth)
    try:
        from agent.tools import get_all_common_tools
        all_tools = get_all_common_tools()
        logger.info(f"Loaded {len(all_tools)} tools from common_tools")
        return all_tools
    except Exception as e:
        logger.warning(f"Failed to load from common_tools, falling back to legacy: {e}")
        # Fallback to legacy loading
        base_tools = {
            "web_search": web_search,
            "knowledge_search": knowledge_search,
            "read_webpage": read_webpage,
        }
        try:
            from nodes.subagents.x_masters.biomedical_tools import get_biomedical_tools
            bio_tools = get_biomedical_tools()
            base_tools.update(bio_tools)
            logger.info(f"Loaded {len(bio_tools)} biomedical tools (legacy)")
        except Exception as e2:
            logger.warning(f"Failed to load biomedical tools: {e2}")
        return base_tools


def get_lightweight_tools() -> dict:
    """Get tools WITHOUT knowledge_search for downstream agents (Critic/Rewriter/Selector).

    These agents receive pre-retrieved context via their prompt, so they don't
    need the expensive knowledge_search tool.  They keep:
        - web_search for real-time fact-checking
        - read_webpage for URL content extraction
        - All biomedical, literature, clinical, and reference tools
    
    Returns:
        Dict mapping function names to function objects.
    """
    # Import from common_tools
    try:
        from agent.tools import get_all_common_tools
        all_tools = get_all_common_tools()
        # Remove knowledge_search for lightweight mode
        lightweight_tools = {k: v for k, v in all_tools.items() if k != "knowledge_search"}
        logger.info(f"Loaded {len(lightweight_tools)} lightweight tools from common_tools")
        return lightweight_tools
    except Exception as e:
        logger.warning(f"Failed to load from common_tools, falling back to legacy: {e}")
        # Fallback
        base_tools = {
            "web_search": web_search,
            "read_webpage": read_webpage,
        }
        try:
            from nodes.subagents.x_masters.biomedical_tools import get_biomedical_tools
            bio_tools = get_biomedical_tools()
            base_tools.update(bio_tools)
        except Exception as e2:
            logger.warning(f"Failed to load biomedical tools: {e2}")
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


# ---------------------------------------------------------------------------
# Subgraph-specific tool configurations
# ---------------------------------------------------------------------------

def get_tools_for_subgraph(subgraph_name: str) -> dict:
    """Get a customized tool set for a specific subgraph.
    
    Different subgraphs have different tool requirements:
    
    - general_qa: All tools (full research capability)
    - immunity: All tools + emphasis on literature/reference tools
    - deep_research: All tools (primary research)
    - code_act: Core tools only (web_search, memory, docx)
    - executor: Minimal tools (for code execution)
    
    Args:
        subgraph_name: Name of the subgraph (e.g., "general_qa", "immunity", "code_act")
        
    Returns:
        Dict mapping function names to function objects for that subgraph.
    """
    try:
        from agent.tools import get_all_common_tools
        all_tools = get_all_common_tools()
    except Exception as e:
        logger.warning(f"Failed to load common_tools: {e}")
        return get_solver_tools()
    
    # Define which tools each subgraph should have
    subgraph_configs = {
        "general_qa": None,  # None = all tools
        "immunity": None,    # All tools
        "deep_research": None,  # All tools
        "supervisor": None,  # All tools
        
        # Code execution subgraphs - lighter toolset
        "code_act": {
            "web_search", "read_webpage", "knowledge_search",
            "agent_memory", "parse_docx"
        },
        
        "executor": {
            "web_search", "read_webpage", "agent_memory", "parse_docx"
        },
        
        "task_decomposition": {
            "web_search", "read_webpage", "knowledge_search"
        },
        
        # Literature-focused subgraphs
        "literature_review": {
            # Search tools
            "web_search", "read_webpage", "knowledge_search",
            # Literature tools
            "search_pubmed", "get_pubmed_abstract", "search_semantic_scholar",
            "get_paper_citations", "search_preprints", "search_europe_pmc",
        },
        
        # Clinical data focused
        "clinical_analysis": {
            # Search tools
            "web_search", "read_webpage", "knowledge_search",
            # Clinical tools
            "search_clinical_trials", "query_open_targets", "search_clinvar",
            "query_pharmgkb", "search_openfda",
            # Reference tools for clinical context
            "query_uniprot", "query_kegg", "query_reactome",
        },
        
        # Bioinformatics focused
        "bioinformatics": {
            # Search tools
            "web_search", "read_webpage", "knowledge_search",
            # Biomedical tools (all)
            "query_kg", "query_expression", "query_disease_gene", "query_gene",
            "query_protein_atlas", "query_omim", "query_ppi", "query_drug_interaction",
            "query_binding", "query_variant", "query_gwas", "query_genebass",
            "query_tcr", "query_mirna_target", "query_mirna_validated", "query_sgrna",
            "query_go", "query_hpo", "query_geneset", "query_drug_for_disease",
            "query_disease_for_drug", "query_depmap", "query_cell_markers",
            "query_virus_host", "query_drug_repurposing",
            # Reference tools
            "query_uniprot", "query_interpro", "query_imgt", "query_reactome",
            "query_string_db", "query_kegg", "query_pdb_search", "query_vdjdb",
            "query_ipd_hla", "query_immport",
        },
    }
    
    config = subgraph_configs.get(subgraph_name)
    
    if config is None:
        # Full tool set
        logger.info(f"Subgraph '{subgraph_name}' gets all {len(all_tools)} tools")
        return all_tools
    
    # Filter to specific tools
    filtered = {k: v for k, v in all_tools.items() if k in config}
    logger.info(f"Subgraph '{subgraph_name}' gets {len(filtered)}/{len(all_tools)} tools")
    return filtered


def get_tool_descriptions_for_prompt(subgraph_name: str = None) -> str:
    """Generate a formatted string of tool descriptions for LLM prompts.
    
    This is useful for including tool information in system prompts.
    
    Args:
        subgraph_name: Optional subgraph name to filter tools
        
    Returns:
        Formatted string describing available tools
    """
    if subgraph_name:
        tools = get_tools_for_subgraph(subgraph_name)
    else:
        try:
            from agent.tools import get_all_common_tools
            tools = get_all_common_tools()
        except:
            tools = get_solver_tools()
    
    descriptions = []
    descriptions.append("## Available Tools\n")
    
    # Group tools by category
    categories = {
        "Search Tools": ["web_search", "read_webpage", "knowledge_search"],
        "Memory Tools": ["agent_memory"],
        "Document Parsing": ["parse_docx"],
        "Literature Search": ["search_pubmed", "get_pubmed_abstract", "search_semantic_scholar",
                              "get_paper_citations", "search_preprints", "search_europe_pmc"],
        "Clinical Data": ["search_clinical_trials", "query_open_targets", "search_clinvar",
                          "query_pharmgkb", "search_openfda"],
        "Biomedical Databases": ["query_kg", "query_expression", "query_disease_gene", "query_gene",
                                 "query_protein_atlas", "query_omim", "query_ppi", "query_drug_interaction",
                                 "query_binding", "query_variant", "query_gwas", "query_genebass",
                                 "query_tcr", "query_mirna_target", "query_mirna_validated", "query_sgrna",
                                 "query_go", "query_hpo", "query_geneset", "query_drug_for_disease",
                                 "query_disease_for_drug", "query_depmap", "query_cell_markers",
                                 "query_virus_host", "query_drug_repurposing"],
        "Reference Databases": ["query_uniprot", "query_interpro", "query_imgt", "query_reactome",
                                "query_string_db", "query_kegg", "query_pdb_search", "query_vdjdb",
                                "query_ipd_hla", "query_immport"],
    }
    
    for category, tool_names in categories.items():
        available = [name for name in tool_names if name in tools]
        if available:
            descriptions.append(f"\n### {category}\n")
            for name in available:
                func = tools[name]
                doc = func.__doc__ or "No description available"
                # Extract first line of docstring
                first_line = doc.strip().split('\n')[0]
                descriptions.append(f"- **{name}**: {first_line}")
    
    return "\n".join(descriptions)
