"""
Unified Search Tool - Multi-provider web search with automatic fallback

Supports multiple search providers:
1. Tavily (primary) - Best for AI/LLM applications
2. SerpAPI (fallback) - Google search results
3. Bing Search API (fallback) - Microsoft search
4. DuckDuckGo (free, no API key required) - Privacy-focused search

Features:
- Automatic quota detection and fallback
- Global state tracking for disabled providers
- Unified search interface
- Retry with exponential backoff
- Both sync and async interfaces
- Automatic query truncation for Tavily (400 char limit)

Usage:
    from agent.utils.unified_search import unified_search, unified_search_async
    results = unified_search("CAR-T cell therapy mechanism")
    results = await unified_search_async("CAR-T cell therapy mechanism")
"""

import os
import logging
import time
import random
import asyncio
from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Tavily has a 400 character query limit
MAX_TAVILY_QUERY_LENGTH = 395  # Leave small margin


def _compress_search_query(query: str, max_length: int = MAX_TAVILY_QUERY_LENGTH) -> str:
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
    
    original_len = len(query)
    
    # Step 1: Extract key terms (capitalized words, scientific terms, numbers with units)
    # These are usually the most important for scientific/biomedical queries
    key_patterns = [
        r'\b[A-Z][a-z]+(?:\'s)?\b',  # Capitalized words (Watterson's, pi, etc.)
        r'\b[A-Z]{2,}\b',  # Acronyms (DNA, RNA, SNP, CAR-T)
        r'\b[a-z]+_[a-z]+\b',  # snake_case terms
        r'\b\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?\b',  # Numbers with optional units
        r'\b(?:estimator|diversity|nucleotide|variant|sample|bias|calculate|formula|antibody|sequence|protein|gene|cell|therapy|drug|target)\b',  # Key scientific terms
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
            result = truncated[:last_space]
        else:
            result = truncated
        logger.info(f"Compressed Tavily query from {original_len} to {len(result)} chars (fallback truncation)")
        return result
    
    # Join with proper punctuation
    result = '. '.join(compressed_parts)
    if not result.endswith('.') and not result.endswith('...'):
        result += '.'
    
    logger.info(f"Compressed Tavily query from {original_len} to {len(result)} chars (smart compression)")
    return result


# Alias for backwards compatibility
_truncate_query_for_tavily = _compress_search_query


class SearchProvider(str, Enum):
    TAVILY = "tavily"
    SERPAPI = "serpapi"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"


@dataclass
class SearchProviderState:
    """Tracks the state of each search provider"""
    enabled: bool = True
    failure_count: int = 0
    last_failure_time: float = 0
    quota_exceeded: bool = False
    last_error: str = ""


@dataclass
class UnifiedSearchConfig:
    """Configuration for unified search"""
    # Provider states
    providers: Dict[str, SearchProviderState] = field(default_factory=dict)
    
    # Thresholds
    max_failures_before_disable: int = 3
    cooldown_seconds: float = 300  # 5 minutes before re-trying a disabled provider
    
    # Retry settings
    max_retries: int = 2
    base_backoff: float = 1.0
    
    # Provider priority
    provider_priority: List[str] = field(default_factory=lambda: [
        SearchProvider.TAVILY.value,
        SearchProvider.SERPAPI.value,
        SearchProvider.BING.value,
        SearchProvider.DUCKDUCKGO.value,
    ])


# Global state
_config = UnifiedSearchConfig()


def _get_provider_state(provider: str) -> SearchProviderState:
    """Get or create state for a provider"""
    if provider not in _config.providers:
        _config.providers[provider] = SearchProviderState()
    return _config.providers[provider]


def _should_use_provider(provider: str) -> bool:
    """Check if a provider should be used based on its state"""
    state = _get_provider_state(provider)
    
    if not state.enabled:
        return False
    
    if state.quota_exceeded:
        # Check if cooldown period has passed
        if time.time() - state.last_failure_time > _config.cooldown_seconds:
            logger.info(f"Provider {provider} cooldown expired, re-enabling")
            state.quota_exceeded = False
            state.failure_count = 0
            return True
        return False
    
    return True


def _mark_provider_failure(provider: str, error: str, is_quota_error: bool = False):
    """Mark a provider as failed"""
    state = _get_provider_state(provider)
    state.failure_count += 1
    state.last_failure_time = time.time()
    state.last_error = error
    
    if is_quota_error:
        state.quota_exceeded = True
        logger.warning(f"Provider {provider} quota exceeded: {error}")
    elif state.failure_count >= _config.max_failures_before_disable:
        state.enabled = False
        logger.warning(f"Provider {provider} disabled after {state.failure_count} failures: {error}")


def _mark_provider_success(provider: str):
    """Mark a provider as successful"""
    state = _get_provider_state(provider)
    state.failure_count = 0
    state.quota_exceeded = False
    state.last_error = ""


def _is_quota_error(error: str) -> bool:
    """Check if an error indicates quota/API limit exceeded"""
    quota_indicators = [
        "usage limit",
        "rate limit",
        "quota exceeded",
        "plan's limit",
        "exceeds your plan",
        "too many requests",
        "api limit",
    ]
    error_lower = error.lower()
    return any(indicator in error_lower for indicator in quota_indicators)


# ---------------------------------------------------------------------------
# Search Provider Implementations (Sync)
# ---------------------------------------------------------------------------

def _search_tavily(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search using Tavily API"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not configured")
    
    # CRITICAL FIX: Truncate query to fit Tavily's 400 character limit
    query = _truncate_query_for_tavily(query)
    
    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)
    
    response = client.search(
        query=query,
        max_results=max_results,
        include_raw_content=False,
    )
    
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:500],
            "source": "tavily"
        })
    
    return results


def _search_serpapi(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search using SerpAPI (Google results)"""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY not configured")
    
    import requests
    
    params = {
        "q": query,
        "api_key": api_key,
        "num": max_results,
    }
    
    response = requests.get("https://serpapi.com/search", params=params, timeout=30)
    data = response.json()
    
    results = []
    for r in data.get("organic_results", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", "")[:500],
            "source": "serpapi"
        })
    
    return results


def _search_bing(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search using Bing Search API"""
    api_key = os.getenv("BING_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("BING_SEARCH_API_KEY not configured")
    
    import requests
    
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": max_results,
        "mkt": "en-US"
    }
    
    response = requests.get(
        "https://api.bing.microsoft.com/v7.0/search",
        headers=headers,
        params=params,
        timeout=30
    )
    data = response.json()
    
    results = []
    for r in data.get("webPages", {}).get("value", [])[:max_results]:
        results.append({
            "title": r.get("name", ""),
            "url": r.get("url", ""),
            "snippet": r.get("snippet", "")[:500],
            "source": "bing"
        })
    
    return results


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search using DuckDuckGo (free, no API key required)"""
    try:
        # Try ddgs first (newer package)
        try:
            from ddgs import DDGS
        except ImportError:
            # Fall back to duckduckgo_search
            from duckduckgo_search import DDGS
        
        results = []
        # Use context manager for proper cleanup
        with DDGS() as ddgs:
            try:
                # Use text search with timeout
                search_results = list(ddgs.text(query, max_results=max_results))
                for r in search_results:
                    if r and isinstance(r, dict):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("href", "") or r.get("url", ""),
                            "snippet": (r.get("body", "") or r.get("snippet", ""))[:500],
                            "source": "duckduckgo"
                        })
            except Exception as search_err:
                logger.warning(f"DDGS text search error: {search_err}")
        
        if results:
            return results
            
        # If no results, try alternative method
        logger.warning(f"DuckDuckGo returned no results for: {query[:50]}")
        return []
        
    except ImportError:
        raise ValueError("ddgs or duckduckgo-search not installed. Run: pip install ddgs")
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        raise ValueError(f"DuckDuckGo search failed: {e}")


# ---------------------------------------------------------------------------
# Async Search Provider Implementations
# ---------------------------------------------------------------------------

async def _search_tavily_async(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search using Tavily API (async)"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not configured")
    
    # CRITICAL FIX: Truncate query to fit Tavily's 400 character limit
    query = _truncate_query_for_tavily(query)
    
    from tavily import AsyncTavilyClient
    client = AsyncTavilyClient(api_key=api_key)
    
    response = await client.search(
        query=query,
        max_results=max_results,
        include_raw_content=True,
    )
    
    results = []
    for r in response.get("results", []):
        raw_content = r.get("raw_content") or ""
        snippet = r.get("content", "")
        content = raw_content if len(raw_content) > len(snippet) else snippet
        
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": content,
            "snippet": snippet[:500],
            "score": r.get("score", 0.0),
            "source": "tavily",
        })
    
    return results


async def _search_serpapi_async(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search using SerpAPI (async)"""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY not configured")
    
    import aiohttp
    
    params = {
        "q": query,
        "api_key": api_key,
        "num": max_results,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get("https://serpapi.com/search", params=params, timeout=30) as response:
            data = await response.json()
    
    results = []
    for r in data.get("organic_results", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", "")[:500],
            "score": 0.5,
            "source": "serpapi"
        })
    
    return results


async def _search_duckduckgo_async(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search using DuckDuckGo (async)"""
    # DuckDuckGo search is synchronous, run in thread
    def _sync_search():
        try:
            # Try ddgs first (newer package)
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                try:
                    search_results = list(ddgs.text(query, max_results=max_results))
                    for r in search_results:
                        if r and isinstance(r, dict):
                            results.append({
                                "title": r.get("title", ""),
                                "url": r.get("href", "") or r.get("url", ""),
                                "content": (r.get("body", "") or r.get("snippet", "")),
                                "snippet": (r.get("body", "") or r.get("snippet", ""))[:500],
                                "score": 0.5,
                                "source": "duckduckgo"
                            })
                except Exception as search_err:
                    logger.warning(f"DDGS async search error: {search_err}")
            return results
        except ImportError:
            raise ValueError("ddgs or duckduckgo-search not installed")
        except Exception as e:
            logger.error(f"DuckDuckGo async search error: {e}")
            raise ValueError(f"DuckDuckGo search failed: {e}")
    
    return await asyncio.to_thread(_sync_search)


# ---------------------------------------------------------------------------
# Unified Search Functions
# ---------------------------------------------------------------------------

def unified_search(
    query: str,
    max_results: int = 5,
    preferred_provider: Optional[str] = None,
    fallback_on_failure: bool = True,
) -> Dict[str, Any]:
    """
    Unified web search with automatic provider fallback.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        preferred_provider: Preferred search provider (optional)
        fallback_on_failure: Whether to fallback to other providers on failure
        
    Returns:
        Dictionary with:
        - success: bool
        - results: List of search results
        - provider: Provider that was used
        - error: Error message if failed
    """
    # Determine provider order
    if preferred_provider and _should_use_provider(preferred_provider):
        providers_to_try = [preferred_provider]
        if fallback_on_failure:
            providers_to_try.extend([
                p for p in _config.provider_priority 
                if p != preferred_provider and _should_use_provider(p)
            ])
    else:
        providers_to_try = [
            p for p in _config.provider_priority 
            if _should_use_provider(p)
        ]
    
    if not providers_to_try:
        return {
            "success": False,
            "results": [],
            "provider": None,
            "error": "All search providers are disabled or quota exceeded"
        }
    
    # Provider implementations mapping
    provider_funcs = {
        SearchProvider.TAVILY.value: _search_tavily,
        SearchProvider.SERPAPI.value: _search_serpapi,
        SearchProvider.BING.value: _search_bing,
        SearchProvider.DUCKDUCKGO.value: _search_duckduckgo,
    }
    
    last_error = ""
    
    for provider in providers_to_try:
        if provider not in provider_funcs:
            continue
        
        # Check if provider has required API key
        if provider == SearchProvider.TAVILY.value and not os.getenv("TAVILY_API_KEY"):
            continue
        if provider == SearchProvider.SERPAPI.value and not os.getenv("SERPAPI_API_KEY"):
            continue
        if provider == SearchProvider.BING.value and not os.getenv("BING_SEARCH_API_KEY"):
            continue
        
        try:
            logger.info(f"Trying search with provider: {provider}")
            results = provider_funcs[provider](query, max_results)
            
            if results:
                _mark_provider_success(provider)
                return {
                    "success": True,
                    "results": results,
                    "provider": provider,
                    "error": None
                }
            else:
                logger.warning(f"Provider {provider} returned no results")
                
        except Exception as e:
            error_str = str(e)
            last_error = error_str
            is_quota = _is_quota_error(error_str)
            _mark_provider_failure(provider, error_str, is_quota_error=is_quota)
            
            logger.warning(f"Provider {provider} failed: {error_str}")
            
            if not fallback_on_failure:
                break
    
    return {
        "success": False,
        "results": [],
        "provider": None,
        "error": f"All search providers failed. Last error: {last_error}"
    }


async def unified_search_async(
    query: str,
    max_results: int = 5,
    preferred_provider: Optional[str] = None,
    fallback_on_failure: bool = True,
) -> Dict[str, Any]:
    """
    Async unified web search with automatic provider fallback.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        preferred_provider: Preferred search provider (optional)
        fallback_on_failure: Whether to fallback to other providers on failure
        
    Returns:
        Dictionary with:
        - success: bool
        - results: List of search results
        - provider: Provider that was used
        - error: Error message if failed
    """
    # Determine provider order
    if preferred_provider and _should_use_provider(preferred_provider):
        providers_to_try = [preferred_provider]
        if fallback_on_failure:
            providers_to_try.extend([
                p for p in _config.provider_priority 
                if p != preferred_provider and _should_use_provider(p)
            ])
    else:
        providers_to_try = [
            p for p in _config.provider_priority 
            if _should_use_provider(p)
        ]
    
    if not providers_to_try:
        return {
            "success": False,
            "results": [],
            "provider": None,
            "error": "All search providers are disabled or quota exceeded"
        }
    
    # Provider implementations mapping (async)
    provider_funcs_async = {
        SearchProvider.TAVILY.value: _search_tavily_async,
        SearchProvider.SERPAPI.value: _search_serpapi_async,
        SearchProvider.DUCKDUCKGO.value: _search_duckduckgo_async,
    }
    
    last_error = ""
    
    for provider in providers_to_try:
        if provider not in provider_funcs_async:
            continue
        
        # Check if provider has required API key
        if provider == SearchProvider.TAVILY.value and not os.getenv("TAVILY_API_KEY"):
            continue
        if provider == SearchProvider.SERPAPI.value and not os.getenv("SERPAPI_API_KEY"):
            continue
        
        try:
            logger.info(f"Trying async search with provider: {provider}")
            results = await provider_funcs_async[provider](query, max_results)
            
            if results:
                _mark_provider_success(provider)
                return {
                    "success": True,
                    "results": results,
                    "provider": provider,
                    "error": None
                }
            else:
                logger.warning(f"Provider {provider} returned no results")
                
        except Exception as e:
            error_str = str(e)
            last_error = error_str
            is_quota = _is_quota_error(error_str)
            _mark_provider_failure(provider, error_str, is_quota_error=is_quota)
            
            logger.warning(f"Provider {provider} failed: {error_str}")
            
            if not fallback_on_failure:
                break
    
    return {
        "success": False,
        "results": [],
        "provider": None,
        "error": f"All search providers failed. Last error: {last_error}"
    }


def web_search_unified(query: str, max_results: int = 5) -> str:
    """
    Unified web search that returns formatted string (compatible with x_masters tools).
    
    This is a drop-in replacement for the original web_search function.
    """
    result = unified_search(query, max_results)
    
    if not result["success"]:
        return f"[Error] Web search failed: {result['error']}"
    
    # Format results for LLM consumption
    formatted = [f"[Web Search Results for: {query}] (via {result['provider']})\n"]
    
    for i, r in enumerate(result["results"], 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("snippet", "")[:1000]
        
        formatted.append(f"\n--- Result {i} ---")
        formatted.append(f"Title: {title}")
        formatted.append(f"URL: {url}")
        formatted.append(f"snippet: {snippet}")
    
    return "\n".join(formatted)


async def search_web_async(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Async web search that returns raw results (for paper_qa compatibility).
    
    Returns list of result dicts with:
    - title: str
    - url: str
    - content: str (full content)
    - snippet: str (short snippet)
    - score: float
    - source: str
    """
    result = await unified_search_async(query, max_results)
    
    if not result["success"]:
        logger.warning(f"Async web search failed: {result['error']}")
        return []
    
    return result["results"]


def get_search_status() -> Dict[str, Any]:
    """Get the current status of all search providers"""
    status = {}
    for provider in _config.provider_priority:
        state = _get_provider_state(provider)
        
        # Check if API key is configured
        has_key = False
        if provider == SearchProvider.TAVILY.value:
            has_key = bool(os.getenv("TAVILY_API_KEY"))
        elif provider == SearchProvider.SERPAPI.value:
            has_key = bool(os.getenv("SERPAPI_API_KEY"))
        elif provider == SearchProvider.BING.value:
            has_key = bool(os.getenv("BING_SEARCH_API_KEY"))
        elif provider == SearchProvider.DUCKDUCKGO.value:
            has_key = True  # No key required
        
        status[provider] = {
            "enabled": state.enabled,
            "has_api_key": has_key,
            "quota_exceeded": state.quota_exceeded,
            "failure_count": state.failure_count,
            "last_error": state.last_error,
        }
    
    return status


def reset_search_providers():
    """Reset all search providers to enabled state"""
    for provider in _config.provider_priority:
        state = _get_provider_state(provider)
        state.enabled = True
        state.failure_count = 0
        state.quota_exceeded = False
        state.last_error = ""
    
    logger.info("All search providers have been reset")
