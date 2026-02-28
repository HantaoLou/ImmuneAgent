"""Literature discovery via Tavily + Qdrant, evidence processing via paper-qa.

Primary source: Tavily web search (academic domain filtering)
Secondary source: Qdrant vector database (local immunology knowledge base)
Optional enhancement: paper-qa Docs for LLM-scored evidence gathering

Called from knowledge_activation_node to provide literature evidence
for the Kc->Kr knowledge generation prompt.
"""

import asyncio
import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Academic domains for Tavily search filtering
TAVILY_ACADEMIC_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov",
    "scholar.google.com",
    "arxiv.org",
    "biorxiv.org",
    "nature.com",
    "sciencedirect.com",
    "cell.com",
    "springer.com",
    "wiley.com",
    "pnas.org",
    "frontiersin.org",
    "mdpi.com",
]


# ==================== Query Extraction ====================


def extract_search_queries(question: str) -> list[str]:
    """Extract 1-3 search queries from a biomedical question.

    Uses keyword extraction heuristics (no LLM call, <1ms).
    """
    q = re.sub(
        r"^(what|how|which|why|when|where|describe|explain|compare)\s+"
        r"(is|are|does|do|was|were|the|a|an)\s+",
        "",
        question.lower(),
    )
    q = q.rstrip("?. ")
    queries = [q]

    bio_terms = re.findall(
        r"(?:(?:CD\d+\+?|IL-?\d+|TNF-?\w*|IFN-?\w*|TLR\d*|MHC|HLA|"
        r"BCR|TCR|Ig[GMAED]|CRISPR|PCR|ELISA|RNA|DNA|mRNA|"
        r"[A-Z][a-z]+(?:ase|ine|oid|gen|tin|cin))\s*)+",
        question,
    )
    if bio_terms:
        queries.append(" ".join(bio_terms[:5]))

    if len(question.split()) > 15:
        first_clause = re.split(r"[,;]", q)[0].strip()
        if first_clause != q and len(first_clause.split()) >= 4:
            queries.append(first_clause)

    return queries[:3]


# ==================== Unified Search (Multi-provider with fallback) ====================


async def search_unified(
    query: str, max_results: int = 8
) -> list[dict[str, Any]]:
    """Search via unified multi-provider API with automatic fallback.

    Tries providers in order: Tavily -> SerpAPI -> DuckDuckGo
    Falls back automatically when quota exceeded or errors occur.

    Returns results from scientific publishers, PubMed, ArXiv, etc.
    """
    try:
        from agent.utils.unified_search import unified_search_async, get_search_status, SearchProvider
        
        # Log current search status
        status = get_search_status()
        available = [p for p, s in status.items() if s['enabled'] and s['has_api_key'] and not s['quota_exceeded']]
        logger.info(f"Available search providers for paper_qa: {available}")
        
        # Use unified search with Tavily as preferred (for academic domains)
        result = await unified_search_async(query, max_results, preferred_provider=SearchProvider.TAVILY.value)
        
        if result["success"]:
            # Mark results as coming from tavily for compatibility
            for r in result["results"]:
                r["source"] = "tavily"  # For compatibility with existing code
            logger.info(f"Unified search returned {len(result['results'])} results via {result['provider']}")
            return result["results"]
        else:
            logger.warning(f"Unified search failed: {result['error']}")
            return []
            
    except ImportError as e:
        logger.warning(f"Unified search not available: {e}, falling back to Tavily-only")
        return await search_tavily(query, max_results)
    except Exception as e:
        logger.warning(f"Unified search error: {e}")
        return []


async def search_tavily(
    query: str, max_results: int = 8
) -> list[dict[str, Any]]:
    """Search via Tavily API with academic domain filtering.

    Returns results from scientific publishers, PubMed, ArXiv, etc.
    Uses search_depth='advanced' for deeper content retrieval.
    """
    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        logger.warning("tavily-python not installed, skipping Tavily search")
        return []

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set, skipping Tavily search")
        return []

    # CRITICAL FIX: Truncate query to 400 characters (Tavily limit)
    if len(query) > 400:
        # Try to truncate at a sentence boundary
        truncated = query[:400]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')
        last_sentence_end = max(last_period, last_question, last_exclaim)
        
        if last_sentence_end > 200:  # Keep at least 200 chars
            truncated = truncated[:last_sentence_end + 1]
        else:
            # Just truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > 200:
                truncated = truncated[:last_space]
        
        logger.debug(f"Truncated Tavily query from {len(query)} to {len(truncated)} chars")
        query = truncated

    try:
        client = AsyncTavilyClient(api_key=api_key)
        response = await client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=True,
            include_domains=TAVILY_ACADEMIC_DOMAINS,
        )

        results = []
        for item in response.get("results", []):
            raw_content = item.get("raw_content") or ""
            snippet = item.get("content", "")
            content = raw_content if len(raw_content) > len(snippet) else snippet

            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": content,
                "snippet": snippet[:500],
                "score": item.get("score", 0.0),
                "source": "tavily",
            })

        logger.info(f"Tavily returned {len(results)} results for: {query[:60]}")
        return results
    except Exception as e:
        error_str = str(e)
        # Check if this is a quota error
        if "usage limit" in error_str.lower() or "plan" in error_str.lower():
            logger.error(f"Tavily quota exceeded: {e}")
            # Mark Tavily as quota exceeded for future calls
            try:
                from agent.utils.unified_search import _mark_provider_failure, SearchProvider
                _mark_provider_failure(SearchProvider.TAVILY.value, error_str, is_quota_error=True)
            except:
                pass
        else:
            logger.warning(f"Tavily search failed: {e}")
        return []


# ==================== Qdrant Search (SECONDARY) ====================


def _get_embeddings():
    """Get embedding model matching the Qdrant collection.

    Reads config from env vars (same as deep_research subagent):
    - EMBEDDING_PROVIDER: 'openai' or 'ollama' (default: 'openai')
    - EMBEDDING_MODEL: model name (default: 'text-embedding-3-small')
    - EMBEDDING_BASE_URL: custom API base URL (optional)
    - EMBEDDING_API_KEY: dedicated embedding API key (falls back to OPENAI_API_KEY)
    """
    provider = os.environ.get("EMBEDDING_PROVIDER", "openai")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = os.environ.get("EMBEDDING_BASE_URL", "https://xiaoai.plus/v1")
    api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")

    try:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            kwargs = {"model": model}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            return OpenAIEmbeddings(**kwargs)
        else:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=model)
    except Exception as e:
        logger.warning(f"Failed to initialize embeddings ({provider}/{model}): {e}")
        return None


async def search_qdrant(
    query: str, max_results: int = 5
) -> list[dict[str, Any]]:
    """Search Qdrant vector database for relevant domain knowledge.

    Connects to the existing Immunology collection using the same
    config as the deep_research subagent (env vars: QDRANT_HOST, etc.).
    """
    host = os.environ.get("QDRANT_HOST", "")
    if not host:
        logger.debug("QDRANT_HOST not set, skipping Qdrant search")
        return []

    port = int(os.environ.get("QDRANT_PORT", "6333"))
    collection = os.environ.get("QDRANT_COLLECTION", "Immunology")

    try:
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore

        embeddings = _get_embeddings()
        if not embeddings:
            return []

        client = QdrantClient(
            host=host, port=port, timeout=30, check_compatibility=False
        )
        store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
        )

        docs_with_scores = await asyncio.to_thread(
            store.similarity_search_with_score, query, k=max_results
        )

        results = []
        for doc, score in docs_with_scores:
            meta = doc.metadata or {}
            results.append({
                "title": meta.get("source", meta.get("title", "Knowledge Base")),
                "url": "",
                "content": doc.page_content,
                "snippet": doc.page_content[:500],
                "score": float(score),
                "source": "qdrant",
                "metadata": meta,
            })

        logger.info(f"Qdrant returned {len(results)} results for: {query[:60]}")
        return results
    except Exception as e:
        logger.warning(f"Qdrant search failed: {e}")
        return []


# ==================== Discovery (Tavily + Qdrant) ====================


def _deduplicate_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate by URL (web results) and content hash (all results)."""
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    unique = []

    for r in results:
        url = r.get("url", "")
        if url and url in seen_urls:
            continue
        content_hash = hashlib.md5(
            r.get("content", "")[:300].encode()
        ).hexdigest()
        if content_hash in seen_hashes:
            continue
        if url:
            seen_urls.add(url)
        seen_hashes.add(content_hash)
        unique.append(r)

    return unique


async def discover_papers(
    question: str, max_per_source: int = 8
) -> list[dict[str, Any]]:
    """Discover relevant content from multiple sources in parallel.

    Uses unified search (Tavily -> SerpAPI -> DuckDuckGo fallback) as primary.
    Qdrant is the secondary source (local knowledge base).
    Results are deduplicated and returned in a unified format.
    """
    queries = extract_search_queries(question)
    primary_query = queries[0]

    # Use unified search with fallback instead of Tavily-only
    web_results, qdrant_results = await asyncio.gather(
        search_unified(primary_query, max_results=max_per_source),
        search_qdrant(primary_query, max_results=max_per_source),
        return_exceptions=True,
    )

    all_results: list[dict[str, Any]] = []
    if isinstance(web_results, list):
        all_results.extend(web_results)
    else:
        logger.warning(f"Unified web search error: {web_results}")
    if isinstance(qdrant_results, list):
        all_results.extend(qdrant_results)
    else:
        logger.warning(f"Qdrant search error: {qdrant_results}")

    return _deduplicate_results(all_results)


# ==================== Paper Indexing (paper-qa, optional) ====================


async def _add_result_to_docs(
    docs, result: dict[str, Any], settings, timeout: float = 60.0
) -> bool:
    """Add a single search result to the paper-qa Docs collection."""
    title = result.get("title", "Untitled")
    content = result.get("content", "")
    url = result.get("url", "")
    source = result.get("source", "unknown")

    if not content or len(content) < 50:
        return False

    citation = f"{title}. Source: {source}"
    if url:
        citation += f" ({url})"

    # Try URL-based indexing for web sources with URLs
    if url and source == "tavily":
        try:
            docname = await asyncio.wait_for(
                docs.aadd_url(
                    url=url, citation=citation, title=title, settings=settings,
                ),
                timeout=timeout,
            )
            if docname:
                logger.debug(f"Successfully indexed via URL: '{title[:50]}'")
                return True
        except asyncio.TimeoutError:
            logger.debug(f"URL indexing timeout for '{title[:50]}' (>{timeout}s)")
        except Exception as e:
            logger.debug(f"URL indexing failed for '{title[:50]}': {type(e).__name__}: {e}")

    # Fall back to content-based indexing
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(f"Title: {title}\n\n{content}\n")
        temp_path = f.name

    try:
        docname = await asyncio.wait_for(
            docs.aadd(
                path=temp_path, citation=citation, title=title, settings=settings,
            ),
            timeout=30.0,
        )
        if docname:
            logger.debug(f"Successfully indexed via content: '{title[:50]}'")
            return True
    except asyncio.TimeoutError:
        logger.debug(f"Content indexing timeout for '{title[:50]}' (>30s)")
    except Exception as e:
        logger.debug(f"Content indexing failed for '{title[:50]}': {type(e).__name__}: {e}")
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return False


async def index_results(
    docs,
    results: list[dict[str, Any]],
    settings,
    max_items: int = 8,
) -> list[str]:
    """Index search results into a paper-qa Docs collection."""
    indexable = [
        r for r in results
        if r.get("content") and len(r.get("content", "")) >= 50
    ][:max_items]

    if not indexable:
        return []

    sem = asyncio.Semaphore(3)

    async def bounded_add(result):
        async with sem:
            return await _add_result_to_docs(docs, result, settings)

    outcomes = await asyncio.gather(
        *(bounded_add(r) for r in indexable),
        return_exceptions=True,
    )
    return [
        indexable[i].get("title", f"result_{i}")
        for i, ok in enumerate(outcomes)
        if ok is True
    ]


# ==================== Evidence Gathering (paper-qa) ====================


async def gather_evidence(
    docs, query: str, settings
) -> dict[str, Any]:
    """Gather evidence from indexed docs using paper-qa's pipeline."""
    if not docs.docs:
        return _empty_result()

    try:
        session = await docs.aget_evidence(query, settings=settings)
    except Exception as e:
        logger.error(f"Evidence gathering failed: {e}")
        return _empty_result()

    if not session.contexts:
        return _empty_result()

    try:
        session = await docs.aquery(query=session, settings=settings)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")

    evidence_items = [
        {
            "context_id": ctx.id,
            "summary": ctx.context,
            "score": ctx.score,
            "chunk_name": ctx.text.name,
            "citation": ctx.text.doc.citation,
            "docname": ctx.text.doc.docname,
        }
        for ctx in sorted(session.contexts, key=lambda c: -c.score)
        if ctx.score >= 3
    ]

    return {
        "answer": session.answer or "",
        "evidence_items": evidence_items,
        "references": session.references or "",
        "cost": session.cost,
        "context_count": len(session.contexts),
    }


def _empty_result() -> dict[str, Any]:
    return {
        "answer": "",
        "evidence_items": [],
        "references": "",
        "cost": 0.0,
        "context_count": 0,
    }


# ==================== Evidence Formatting ====================


def format_evidence_for_knowledge_activation(
    result: dict[str, Any], max_items: int = 8
) -> str:
    """Format paper-qa evidence for injection into knowledge activation prompt."""
    items = result.get("evidence_items", [])
    if not items:
        return ""

    parts = ["### Evidence from Scientific Literature\n"]
    for ev in items[:max_items]:
        parts.append(
            f"**[{ev['docname']}]** (relevance: {ev['score']}/10)\n"
            f"> {ev['summary']}\n"
            f"Source: {ev['citation']}\n\n"
        )

    answer = result.get("answer", "")
    if answer:
        parts.append(
            f"### Preliminary Answer (from paper analysis)\n{answer}\n\n"
        )

    refs = result.get("references", "")
    if refs:
        parts.append(f"### Bibliography\n{refs}\n")

    return "\n".join(parts)


def format_raw_results_for_knowledge_activation(
    results: list[dict[str, Any]], max_items: int = 10
) -> str:
    """Format raw Tavily + Qdrant results when paper-qa is not available.

    This is the default formatting path. It directly formats search
    results without requiring paper-qa for LLM-based evidence scoring.
    """
    if not results:
        return ""

    tavily_items = [r for r in results if r["source"] == "tavily"]
    qdrant_items = [r for r in results if r["source"] == "qdrant"]

    parts = ["### Evidence from Literature Search\n"]

    if tavily_items:
        parts.append("#### Web Academic Sources (Tavily)\n")
        for r in tavily_items[:max_items]:
            snippet = r["snippet"][:400] if r.get("snippet") else r.get("content", "")[:400]
            parts.append(
                f"**{r['title']}** (score: {r['score']:.2f})\n"
                f"> {snippet}\n"
            )
            if r.get("url"):
                parts.append(f"URL: {r['url']}\n")
            parts.append("\n")

    if qdrant_items:
        parts.append("#### Internal Knowledge Base (Qdrant)\n")
        for r in qdrant_items[:max_items]:
            snippet = r["snippet"][:400] if r.get("snippet") else r.get("content", "")[:400]
            parts.append(
                f"**{r['title']}** (similarity: {r['score']:.3f})\n"
                f"> {snippet}\n\n"
            )

    return "\n".join(parts)


def format_evidence_for_deep_research(
    result: dict[str, Any], max_items: int = 5
) -> str:
    """Format evidence for enriching the deep_research question."""
    items = result.get("evidence_items", [])
    if not items:
        return ""

    parts = ["Prior evidence from scientific papers:\n"]
    for ev in items[:max_items]:
        parts.append(
            f"- [{ev['docname']}] (score {ev['score']}/10): {ev['summary']}\n"
        )

    return "".join(parts)


# ==================== Confidence Computation ====================


def compute_evidence_confidence(result: dict[str, Any]) -> float:
    """Compute confidence from paper-qa scored evidence.

    confidence = top_5_mean * 0.5 + coverage * 0.3 + certainty * 0.2
    """
    items = result.get("evidence_items", [])
    if not items:
        return 0.0

    scores = sorted([ev["score"] for ev in items], reverse=True)
    top_5_mean = sum(scores[:5]) / min(len(scores), 5) / 10.0

    unique_docs = set(ev["docname"] for ev in items)
    coverage = min(len(unique_docs) / 3.0, 1.0)

    answer = result.get("answer", "")
    certainty = 0.0 if "cannot answer" in answer.lower() else 1.0

    return top_5_mean * 0.5 + coverage * 0.3 + certainty * 0.2


def compute_raw_confidence(results: list[dict[str, Any]]) -> float:
    """Compute confidence from raw Tavily + Qdrant results.

    Uses result count and average score as proxy.
    """
    if not results:
        return 0.0

    tavily_count = sum(1 for r in results if r["source"] == "tavily")
    qdrant_count = sum(1 for r in results if r["source"] == "qdrant")

    source_coverage = min((tavily_count > 0) + (qdrant_count > 0), 2) / 2.0
    result_coverage = min(len(results) / 5.0, 1.0)

    avg_score = sum(r.get("score", 0.0) for r in results) / len(results)

    return avg_score * 0.4 + result_coverage * 0.3 + source_coverage * 0.3


# ==================== Top-Level Safe Pipeline ====================


async def safe_paper_pipeline(
    question: str,
    analysis_objects: list[str] | None = None,
    max_papers: int = 8,
    timeout: float = 120.0,
) -> Optional[dict[str, Any]]:
    """Safe top-level pipeline that knowledge_activation_node calls.

    Discovery: Tavily (primary) + Qdrant (secondary)
    Processing: paper-qa if installed, else raw formatting (default)

    Returns dict with keys: evidence_items, answer, references, confidence,
    evidence_text_block, papers_discovered, papers_indexed, cost, sources.
    Returns None on any failure (graceful degradation).
    """
    try:
        # Step 1: Discover via Tavily + Qdrant
        results = await asyncio.wait_for(
            discover_papers(question, max_per_source=max_papers),
            timeout=30.0,
        )
        if not results:
            logger.info("No results from Tavily + Qdrant")
            return None

        sources_used = list(set(r["source"] for r in results))
        logger.info(
            f"Discovered {len(results)} results from {', '.join(sources_used)}"
        )

        # Step 2: Try paper-qa enhanced evidence processing
        try:
            from paperqa import Docs

            from agent.nodes.subagents.paper_qa.paperqa_config import create_paperqa_settings

            settings = create_paperqa_settings()
            docs = Docs()

            indexed = await asyncio.wait_for(
                index_results(docs, results, settings, max_items=max_papers),
                timeout=timeout * 0.6,
            )

            if indexed:
                logger.info(f"Successfully indexed {len(indexed)} papers into paper-qa")
                pqa_result = await asyncio.wait_for(
                    gather_evidence(docs, question, settings),
                    timeout=timeout * 0.4,
                )

                confidence = compute_evidence_confidence(pqa_result)
                evidence_text = format_evidence_for_knowledge_activation(pqa_result)

                if evidence_text:
                    return {
                        "evidence_items": pqa_result["evidence_items"],
                        "answer": pqa_result["answer"],
                        "references": pqa_result["references"],
                        "confidence": confidence,
                        "evidence_text_block": evidence_text,
                        "papers_discovered": len(results),
                        "papers_indexed": len(indexed),
                        "cost": pqa_result["cost"],
                        "sources": sources_used,
                    }
                else:
                    logger.warning("paper-qa indexed papers but evidence gathering returned empty text")
            else:
                logger.warning(f"paper-qa indexing failed: 0 papers indexed from {len(results)} discovered papers")
        except ImportError:
            logger.info("paper-qa not installed, using direct formatting (papers_indexed will be 0)")
        except Exception as e:
            logger.warning(f"paper-qa processing failed, falling back: {e}")

        # Step 3: Default path — format raw results directly
        evidence_text = format_raw_results_for_knowledge_activation(results)
        confidence = compute_raw_confidence(results)

        return {
            "evidence_items": [
                {
                    "title": r["title"],
                    "snippet": r["snippet"],
                    "score": r["score"],
                    "source": r["source"],
                    "url": r.get("url", ""),
                }
                for r in results
            ],
            "answer": "",
            "references": "",
            "confidence": confidence,
            "evidence_text_block": evidence_text,
            "papers_discovered": len(results),
            "papers_indexed": 0,
            "cost": 0.0,
            "sources": sources_used,
        }

    except asyncio.TimeoutError:
        logger.warning("Paper pipeline timed out")
        return None
    except Exception as e:
        logger.warning(f"Paper pipeline failed: {e}")
        return None
