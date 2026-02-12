"""
Biomedical Query Tools
============================================
Standard Python tools for querying biomedical DuckDB database.
Converted from MCP tools for use in regular Python applications.

Total: 35 tools covering 88 data tables

Modules:
- core_query_tools: 11 core biomedical query tools
- disease_gene_tools: 4 disease-gene association tools
- ppi_tools: 3 protein interaction tools
- drug_tools: 2 drug interaction tools
- geneset_tools: 3 gene set tools
- misc_tools: 6 miscellaneous tools
- gwas_tools: 1 GWAS catalog query tool
- genebass_tools: 1 gene burden analysis tool
- txgnn_tools: 2 drug repurposing prediction tools
- go_tools: 3 Gene Ontology query tools
- hpo_tools: 3 Human Phenotype Ontology query tools
"""

# Core Query Tools
from .core_query_tools import (
    query_knowledge_graph, KnowledgeGraphQuery, KGNodeType,
    query_tcr_mcpas, TCRQuery,
    query_mirdb, MiRDBQuery,
    query_mirtarbase, MiRTarBaseQuery,
    query_bindingdb, BindingDBQuery,
    query_gtex_expression, GTExQuery,
    query_sgrna_human, query_sgrna_mouse, SgRNAQuery,
    query_genetic_interaction, GeneticInteractionQuery, OrganismID,
    query_variant, VariantQuery,
    get_core_database_stats,
)

# Disease/Gene Tools
from .disease_gene_tools import (
    query_disgenet, DisGeNETQuery,
    query_omim, OMIMQuery,
    query_proteinatlas, ProteinAtlasQuery,
    query_gene_info, GeneInfoQuery,
)

# PPI Tools
from .ppi_tools import (
    query_ppi, PPIQuery, PPIExperimentType, PPIOrganismID,
    query_synthetic_interaction, SyntheticInteractionQuery, SyntheticInteractionType,
    get_ppi_stats,
)

# Drug Tools
from .drug_tools import (
    query_drug_interaction, DrugInteractionQuery, DrugCategory, InteractionLevel,
    get_drug_interaction_stats,
)

# Gene Set Tools
from .geneset_tools import (
    query_msigdb, MSigDBQuery, MSigDBCollection,
    query_mousemine, MouseMineQuery, MouseMineCollection,
    get_geneset_stats,
)

# Misc Tools
from .misc_tools import (
    query_evebio, EVEBioQuery, EVEBioDataType,
    query_broad_repurposing, BroadRepurposingQuery,
    query_virus_host_ppi, VirusHostPPIQuery,
    query_depmap, DepMapQuery, DepMapDataType,
    query_celltype_marker, CellTypeMarkerQuery,
    query_czi_census, CZICensusQuery,
)

# GWAS Tools
from .gwas_tools import (
    query_gwas_catalog, GWASCatalogQuery,
)

# Genebass Tools
from .genebass_tools import (
    query_genebass, GenebassQuery, VariantType,
)

# TxGNN Tools
from .txgnn_tools import (
    query_drug_for_disease, DrugForDiseaseQuery,
    query_disease_for_drug, DiseaseForDrugQuery,
)

# GO Tools
from .go_tools import (
    query_go_term, GOTermQuery, GONamespace,
    query_go_hierarchy, GOHierarchyQuery, HierarchyDirection,
    query_go_relations, GORelationsQuery, GORelationType,
)

# HPO Tools
from .hpo_tools import (
    query_hpo_term, HPOTermQuery,
    query_hpo_hierarchy, HPOHierarchyQuery, HPOHierarchyDirection,
    query_hpo_xref, HPOXrefQuery, HPOXrefSource,
)

__all__ = [
    # Core
    "query_knowledge_graph", "KnowledgeGraphQuery", "KGNodeType",
    "query_tcr_mcpas", "TCRQuery",
    "query_mirdb", "MiRDBQuery",
    "query_mirtarbase", "MiRTarBaseQuery",
    "query_bindingdb", "BindingDBQuery",
    "query_gtex_expression", "GTExQuery",
    "query_sgrna_human", "query_sgrna_mouse", "SgRNAQuery",
    "query_genetic_interaction", "GeneticInteractionQuery", "OrganismID",
    "query_variant", "VariantQuery",
    "get_core_database_stats",
    # Disease/Gene
    "query_disgenet", "DisGeNETQuery",
    "query_omim", "OMIMQuery",
    "query_proteinatlas", "ProteinAtlasQuery",
    "query_gene_info", "GeneInfoQuery",
    # PPI
    "query_ppi", "PPIQuery", "PPIExperimentType", "PPIOrganismID",
    "query_synthetic_interaction", "SyntheticInteractionQuery", "SyntheticInteractionType",
    "get_ppi_stats",
    # Drug
    "query_drug_interaction", "DrugInteractionQuery", "DrugCategory", "InteractionLevel",
    "get_drug_interaction_stats",
    # Gene Set
    "query_msigdb", "MSigDBQuery", "MSigDBCollection",
    "query_mousemine", "MouseMineQuery", "MouseMineCollection",
    "get_geneset_stats",
    # Misc
    "query_evebio", "EVEBioQuery", "EVEBioDataType",
    "query_broad_repurposing", "BroadRepurposingQuery",
    "query_virus_host_ppi", "VirusHostPPIQuery",
    "query_depmap", "DepMapQuery", "DepMapDataType",
    "query_celltype_marker", "CellTypeMarkerQuery",
    "query_czi_census", "CZICensusQuery",
    # GWAS
    "query_gwas_catalog", "GWASCatalogQuery",
    # Genebass
    "query_genebass", "GenebassQuery", "VariantType",
    # TxGNN
    "query_drug_for_disease", "DrugForDiseaseQuery",
    "query_disease_for_drug", "DiseaseForDrugQuery",
    # GO
    "query_go_term", "GOTermQuery", "GONamespace",
    "query_go_hierarchy", "GOHierarchyQuery", "HierarchyDirection",
    "query_go_relations", "GORelationsQuery", "GORelationType",
    # HPO
    "query_hpo_term", "HPOTermQuery",
    "query_hpo_hierarchy", "HPOHierarchyQuery", "HPOHierarchyDirection",
    "query_hpo_xref", "HPOXrefQuery", "HPOXrefSource",
]
