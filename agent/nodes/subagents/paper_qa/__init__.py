"""Literature search subagent: Unified multi-provider + Qdrant discovery, optional paper-qa evidence.

Primary source: Unified web search (Tavily -> SerpAPI -> DuckDuckGo fallback)
Secondary source: Qdrant vector database (local immunology knowledge base)
Optional enhancement: paper-qa Docs for LLM-scored evidence gathering

Usage:
    from agent.nodes.subagents.paper_qa import safe_paper_pipeline

    result = await safe_paper_pipeline(question="What is the role of CD4+ T cells?")
    if result:
        evidence_text = result["evidence_text_block"]
        confidence = result["confidence"]
        sources = result["sources"]  # e.g. ["tavily", "qdrant"]
"""

from .paper_retrieval import (
    safe_paper_pipeline,
    discover_papers,
    search_unified,
    search_tavily,
    search_qdrant,
    format_evidence_for_knowledge_activation,
    format_raw_results_for_knowledge_activation,
    format_evidence_for_deep_research,
)
from .paperqa_cache import DocsCache

__all__ = [
    "safe_paper_pipeline",
    "discover_papers",
    "search_unified",
    "search_tavily",
    "search_qdrant",
    "format_evidence_for_knowledge_activation",
    "format_raw_results_for_knowledge_activation",
    "format_evidence_for_deep_research",
    "DocsCache",
]
