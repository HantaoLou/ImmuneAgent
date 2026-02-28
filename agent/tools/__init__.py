"""Common tools for ImmuneAgent.

Provides domain-specific API tools (literature, clinical, reference databases)
plus utility tools (memory, DOCX parsing) that are injected into the agent
harness via Pattern A (namespace injection) and Pattern B (StructuredTools).

Usage:
    from agent.tools import get_all_common_tools
    tools = get_all_common_tools()  # {name: function} dict
    
    # Get tools for a specific subgraph
    from agent.tools import get_tools_for_subgraph
    tools = get_tools_for_subgraph("immunity")
    
    # Get tool descriptions for prompts
    from agent.tools import get_tool_descriptions
    descriptions = get_tool_descriptions()

Tool Categories:
    - Search (3): web_search, read_webpage, knowledge_search
    - Memory (1): agent_memory
    - Biomedical (24): Gene/disease/drug/protein queries
    - Literature (6): PubMed, Semantic Scholar, preprints
    - Clinical (5): Clinical trials, Open Targets, ClinVar
    - Reference (10): UniProt, InterPro, IMGT, Reactome, etc.
    - Docx (1): parse_docx
    
Total: 50 tools available
"""

import logging

logger = logging.getLogger(__name__)

# Tool count constants for documentation
TOOL_COUNTS = {
    "search": 3,
    "memory": 1,
    "biomedical": 24,
    "literature": 6,
    "clinical": 5,
    "reference": 10,
    "docx": 1,
    "total": 50,
}


def get_all_common_tools() -> dict:
    """Returns all common tools as {name: function} dict.

    Each function is synchronous, accepts primitives, returns string.
    Safe to call even if some sub-modules fail to import.
    
    Returns:
        Dict mapping 50 tool names to function objects.
    """
    tools = {}

    for loader_name, loader in [
        ("biomedical", _load_biomedical),
        ("search", _load_search),
        ("literature", _load_literature),
        ("clinical", _load_clinical),
        ("reference", _load_reference),
        ("memory", _load_memory),
        ("docx", _load_docx),
    ]:
        try:
            sub_tools = loader()
            tools.update(sub_tools)
            logger.info(f"Loaded {len(sub_tools)} {loader_name} tools")
        except Exception as e:
            logger.warning(f"Failed to load {loader_name} tools: {e}")

    return tools


def get_tools_by_category() -> dict:
    """Get tools organized by category.
    
    Returns:
        Dict with category names as keys, tool dicts as values.
        
    Example:
        >>> tools = get_tools_by_category()
        >>> tools["literature"]["search_pubmed"]  # PubMed search function
    """
    return {
        "search": _load_search(),
        "memory": _load_memory(),
        "biomedical": _load_biomedical(),
        "literature": _load_literature(),
        "clinical": _load_clinical(),
        "reference": _load_reference(),
        "docx": _load_docx(),
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


def _load_biomedical() -> dict:
    from .biomedical import get_biomedical_tools
    return get_biomedical_tools()


def _load_search() -> dict:
    from .search import get_search_tools
    return get_search_tools()


def _load_literature() -> dict:
    from .literature import get_literature_tools
    return get_literature_tools()


def _load_clinical() -> dict:
    from .clinical import get_clinical_tools
    return get_clinical_tools()


def _load_reference() -> dict:
    from .reference import get_reference_tools
    return get_reference_tools()


def _load_memory() -> dict:
    from .memory import get_memory_tools
    return get_memory_tools()


def _load_docx() -> dict:
    from .docx_parser import get_docx_tools
    return get_docx_tools()
