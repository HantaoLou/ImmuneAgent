"""Common tools for ImmuneAgent.

Provides domain-specific API tools (literature, clinical, reference databases)
plus utility tools (memory, DOCX parsing) that are injected into the agent
harness via Pattern A (namespace injection) and Pattern B (StructuredTools).

LangChain 1.0+ Compatibility:
    All tools use @tool decorator from langchain_core.tools and can be 
    directly bound to LLM via .bind_tools() or used with LangGraph agents.

Usage:
    # Get tools as LangChain tools for binding to LLM
    from agent.tools import get_all_langchain_tools
    tools = get_all_langchain_tools()  # List of tool objects
    
    # Bind to LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4")
    llm_with_tools = llm.bind_tools(tools)
    
    # Use with LangGraph ReAct agent
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(model="gpt-4", tools=tools)
    
    # Backward compatible: get tools as dict (namespace injection pattern)
    from agent.tools import get_all_common_tools
    tools_dict = get_all_common_tools()  # {name: function} dict
    
    # Get tools for a specific category
    from agent.tools import get_tools_by_category
    tools = get_tools_by_category()["literature"]

Tool Categories:
    - Search (3): web_search, read_webpage, knowledge_search
    - Memory (1): agent_memory
    - Biomedical (24): Gene/disease/drug/protein queries
    - Literature (6): PubMed, Semantic Scholar, preprints
    - Clinical (5): Clinical trials, Open Targets, ClinVar
    - Reference (12): UniProt, InterPro, IMGT, Reactome, etc.
    - Docx (1): parse_docx
    
Total: 52 tools available
"""

import logging
from typing import List, Dict, Callable, Any

logger = logging.getLogger(__name__)

# Tool count constants for documentation
TOOL_COUNTS = {
    "search": 3,
    "memory": 1,
    "biomedical": 24,
    "literature": 6,
    "clinical": 5,
    "reference": 12,
    "docx": 1,
    "total": 52,
}


# =============================================================================
# Primary API: Get LangChain Tools (LangChain 1.0+ compatible)
# =============================================================================

def get_all_langchain_tools() -> List:
    """Returns all common tools as LangChain tool objects.
    
    These tools can be directly bound to LLM via .bind_tools() or used
    with LangGraph agents.
    
    Returns:
        List of LangChain tool objects (Decorated with @tool).
        
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> tools = get_all_langchain_tools()
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> llm_with_tools = llm.bind_tools(tools)
    """
    tools = []

    for loader_name, loader in [
        ("search", _load_search_tools),
        ("memory", _load_memory_tools),
        ("biomedical", _load_biomedical_tools),
        ("literature", _load_literature_tools),
        ("clinical", _load_clinical_tools),
        ("reference", _load_reference_tools),
        ("docx", _load_docx_tools),
    ]:
        try:
            sub_tools = loader()
            tools.extend(sub_tools)
            logger.info(f"Loaded {len(sub_tools)} {loader_name} tools")
        except Exception as e:
            logger.warning(f"Failed to load {loader_name} tools: {e}")

    return tools


def get_langchain_tools_by_category() -> Dict[str, List]:
    """Get LangChain tools organized by category.
    
    Returns:
        Dict with category names as keys, lists of tool objects as values.
        
    Example:
        >>> tools = get_langchain_tools_by_category()
        >>> tools["literature"]  # List of literature search tools
    """
    return {
        "search": _load_search_tools(),
        "memory": _load_memory_tools(),
        "biomedical": _load_biomedical_tools(),
        "literature": _load_literature_tools(),
        "clinical": _load_clinical_tools(),
        "reference": _load_reference_tools(),
        "docx": _load_docx_tools(),
    }


# =============================================================================
# Backward Compatible API: Dict-based namespace injection
# =============================================================================

def get_all_common_tools() -> Dict[str, Callable]:
    """Returns all common tools as {name: function} dict.

    This is kept for backward compatibility with the existing namespace
    injection pattern. For new code, prefer get_all_langchain_tools().
    
    Returns:
        Dict mapping 52 tool names to function objects.
    """
    tools = {}

    for loader_name, loader in [
        ("search", _load_search_dict),
        ("memory", _load_memory_dict),
        ("biomedical", _load_biomedical_dict),
        ("literature", _load_literature_dict),
        ("clinical", _load_clinical_dict),
        ("reference", _load_reference_dict),
        ("docx", _load_docx_dict),
    ]:
        try:
            sub_tools = loader()
            tools.update(sub_tools)
            logger.info(f"Loaded {len(sub_tools)} {loader_name} tools (dict mode)")
        except Exception as e:
            logger.warning(f"Failed to load {loader_name} tools: {e}")

    return tools


def get_tools_by_category() -> Dict[str, Dict[str, Callable]]:
    """Get tools organized by category (dict-based, backward compatible).
    
    Returns:
        Dict with category names as keys, tool dicts as values.
        
    Example:
        >>> tools = get_tools_by_category()
        >>> tools["literature"]["search_pubmed"]  # PubMed search function
    """
    return {
        "search": _load_search_dict(),
        "memory": _load_memory_dict(),
        "biomedical": _load_biomedical_dict(),
        "literature": _load_literature_dict(),
        "clinical": _load_clinical_dict(),
        "reference": _load_reference_dict(),
        "docx": _load_docx_dict(),
    }


def get_tool_names() -> list:
    """Get a list of all available tool names.
    
    Returns:
        List of tool name strings.
    """
    return list(get_all_common_tools().keys())


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all tools for LLM prompts.
    
    Returns:
        Markdown-formatted string describing all tools.
    """
    tools_by_cat = get_tools_by_category()
    
    descriptions = ["# Available Tools\n"]
    descriptions.append(f"Total: {sum(len(t) for t in tools_by_cat.values())} tools\n")
    
    category_info = {
        "search": ("Search Tools", "Web and knowledge base search"),
        "memory": ("Memory Tools", "Agent state persistence"),
        "biomedical": ("Biomedical Databases", "Gene/disease/drug/protein queries"),
        "literature": ("Literature Search", "Academic paper search and retrieval"),
        "clinical": ("Clinical Data", "Clinical trials and pharmacogenomics"),
        "reference": ("Reference Databases", "Protein structures, pathways, ontologies"),
        "docx": ("Document Parsing", "DOCX file processing"),
    }
    
    for cat, (title, desc) in category_info.items():
        tools = tools_by_cat.get(cat, {})
        if tools:
            descriptions.append(f"\n## {title}\n")
            descriptions.append(f"{desc} ({len(tools)} tools)\n\n")
            for name, func in tools.items():
                doc = func.__doc__ or "No description"
                first_line = doc.strip().split('\n')[0]
                descriptions.append(f"- **{name}**: {first_line}\n")
    
    return "".join(descriptions)


# =============================================================================
# Internal loaders for LangChain tools
# =============================================================================

def _load_search_tools() -> List:
    from .search import get_search_tools
    return get_search_tools()


def _load_memory_tools() -> List:
    from .memory import get_memory_tools
    return get_memory_tools()


def _load_biomedical_tools() -> List:
    from .biomedical import get_biomedical_tools
    return get_biomedical_tools()


def _load_literature_tools() -> List:
    from .literature import get_literature_tools
    return get_literature_tools()


def _load_clinical_tools() -> List:
    from .clinical import get_clinical_tools
    return get_clinical_tools()


def _load_reference_tools() -> List:
    from .reference import get_reference_tools
    return get_reference_tools()


def _load_docx_tools() -> List:
    from .docx_parser import get_docx_tools
    return get_docx_tools()


# =============================================================================
# Internal loaders for dict-based tools (backward compatible)
# =============================================================================

def _load_search_dict() -> Dict[str, Callable]:
    from .search import get_search_tools_dict
    return get_search_tools_dict()


def _load_memory_dict() -> Dict[str, Callable]:
    from .memory import get_memory_tools_dict
    return get_memory_tools_dict()


def _load_biomedical_dict() -> Dict[str, Callable]:
    from .biomedical import get_biomedical_tools_dict
    return get_biomedical_tools_dict()


def _load_literature_dict() -> Dict[str, Callable]:
    from .literature import get_literature_tools_dict
    return get_literature_tools_dict()


def _load_clinical_dict() -> Dict[str, Callable]:
    from .clinical import get_clinical_tools_dict
    return get_clinical_tools_dict()


def _load_reference_dict() -> Dict[str, Callable]:
    from .reference import get_reference_tools_dict
    return get_reference_tools_dict()


def _load_docx_dict() -> Dict[str, Callable]:
    from .docx_parser import get_docx_tools_dict
    return get_docx_tools_dict()


# =============================================================================
# Convenience exports
# =============================================================================

# Export main functions
__all__ = [
    # LangChain 1.0+ API
    "get_all_langchain_tools",
    "get_langchain_tools_by_category",
    # Backward compatible API
    "get_all_common_tools",
    "get_tools_by_category",
    "get_tool_names",
    "get_tool_descriptions",
    # Constants
    "TOOL_COUNTS",
]
