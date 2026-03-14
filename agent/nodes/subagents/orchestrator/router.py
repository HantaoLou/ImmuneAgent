from typing import Optional

SERVICE_TO_DOMAIN: dict[str, str] = {
    "igblast": "immune",
    "metabcr": "immune",
    "nettcr": "immune",
    "mixtcrpred": "immune",
    "flu": "immune",
    "bioinfo_bcell": "bioinformatics",
    "bioinfo_tcell": "bioinformatics",
    "bioinformatics": "bioinformatics",
    "bioinfo_immune": "bioinformatics",
    "data": "bioinformatics",
    "combine_filter": "bioinformatics",
    "ribonn": "rna",
    "gemorna": "rna",
    "codontransformer": "rna",
    "rinalmo": "rna",
    "spired_fitness": "structural",
    "foldx_saturation_scan": "structural",
}

SUBAGENT_CAPABILITIES: dict[str, dict] = {
    "immune": {
        "keywords": [
            "igblast",
            "metabcr",
            "nettcr",
            "mixtcrpred",
            "antibody",
            "antigen",
            "vdj",
            "cdr3",
            "epitope",
            "binding",
            "peptide",
            "mhc",
            "neutralization",
            "neutralizing",
            "flu",
            "influenza",
            "seroconversion",
        ],
        "mcp_tools": [
            "igblast",
            "metabcr",
            "nettcr",
            "mixtcrpred",
            "flu",
        ],
        "description": (
            "Adaptive immunity - TCR/BCR repertoire analysis, "
            "antibody-antigen binding prediction, V(D)J recombination, "
            "T cell epitope screening, influenza-specific assays"
        ),
    },
    "bioinformatics": {
        "keywords": [
            "bioinfo_bcell",
            "bioinfo_tcell",
            "bioinfo_immune",
            "bioinformatics",
            "data",
            "combine_filter",
            "filter",
            "seurat",
            "scanpy",
            "rds",
            "umap",
            "pseudotime",
            "clustering",
            "differential expression",
            "scrna",
            "single cell",
            "trajectory",
            "dimensionality reduction",
            "gene expression",
            "csv",
            "tsv",
            "merge",
            "aggregate",
            "clonotype",
            "celltype",
            "marker",
            "dotplot",
            "diversity",
            "cytotrace",
            "cellchat",
        ],
        "mcp_tools": [
            "bioinfo_bcell",
            "bioinfo_tcell",
            "bioinformatics",
            "bioinfo_immune",
            "data",
        ],
        "description": (
            "Bioinformatics data analysis & visualization - scRNA-seq, "
            "UMAP, trajectory, clonotype, cell typing, DEG, CellChat"
        ),
    },
    "rna": {
        "keywords": [
            "ribonn",
            "gemorna",
            "codontransformer",
            "rinalmo",
            "rna",
            "mrna",
            "codon",
            "ribosome",
            "translation",
            "nucleotide",
            "rna structure",
            "rna design",
        ],
        "mcp_tools": [
            "ribonn",
            "gemorna",
            "codontransformer",
            "rinalmo",
        ],
        "description": (
            "RNA analysis & design - mRNA optimization, codon usage, "
            "ribosome profiling, RNA structure prediction"
        ),
    },
    "structural": {
        "keywords": [
            "spired",
            "foldx",
            "protein structure",
            "stability",
            "mutation",
            "saturation scan",
            "fitness",
            "pdb",
            "folding",
            "thermostability",
            "ddg",
        ],
        "mcp_tools": [
            "spired_fitness",
            "foldx_saturation_scan",
        ],
        "description": (
            "Protein structure & stability - fitness landscape prediction, "
            "saturation mutagenesis scanning, stability assessment"
        ),
    },
    "general": {
        "keywords": [],
        "mcp_tools": [],
        "description": "analysis workflow",
    },
}


def route_task_by_tools(task_tools: list[str]) -> str:
    """Route a task to the best-matching domain using tool/service names."""
    if not task_tools:
        return ""
    counts: dict[str, int] = {}
    for tool in task_tools:
        domain = SERVICE_TO_DOMAIN.get(tool.lower())
        if domain:
            counts[domain] = counts.get(domain, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def route_task(task_content: str, available_agents: Optional[list[str]] = None) -> str:
    """Route a task to the best-matching subagent via keyword scoring."""
    if available_agents is None:
        available_agents = list(SUBAGENT_CAPABILITIES.keys())

    if len(available_agents) == 1:
        return available_agents[0]

    content_lower = task_content.lower()
    scores: dict[str, int] = {}
    for name in available_agents:
        caps = SUBAGENT_CAPABILITIES.get(name, {})
        keywords = caps.get("keywords", [])
        scores[name] = sum(1 for kw in keywords if kw in content_lower)

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return available_agents[0]


def get_domain_mcp_tools(agent_name: str) -> list[str]:
    """Return MCP tool names for a domain."""
    caps = SUBAGENT_CAPABILITIES.get(agent_name, {})
    return caps.get("mcp_tools", [])


def get_domain_description(agent_name: str) -> str:
    """Return human-readable description for a domain."""
    caps = SUBAGENT_CAPABILITIES.get(agent_name, {})
    return caps.get("description", agent_name)
