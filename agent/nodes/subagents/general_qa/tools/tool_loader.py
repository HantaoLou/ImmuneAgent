"""
Tool Loader for General QA
============================================
Loads and converts all biomedical tools to LangChain Tool format
for use with LangChain 1.0 and LangGraph 1.0
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

# Import all tools
from agent.nodes.subagents.general_qa.tools import (
    # Core Query Tools
    query_knowledge_graph, KnowledgeGraphQuery,
    query_tcr_mcpas, TCRQuery,
    query_mirdb, MiRDBQuery,
    query_mirtarbase, MiRTarBaseQuery,
    query_bindingdb, BindingDBQuery,
    query_gtex_expression, GTExQuery,
    query_sgrna_human, query_sgrna_mouse, SgRNAQuery,
    query_genetic_interaction, GeneticInteractionQuery,
    query_variant, VariantQuery,
    get_core_database_stats,
    # Disease/Gene Tools
    query_disgenet, DisGeNETQuery,
    query_omim, OMIMQuery,
    query_proteinatlas, ProteinAtlasQuery,
    query_gene_info, GeneInfoQuery,
    # PPI Tools
    query_ppi, PPIQuery,
    query_synthetic_interaction, SyntheticInteractionQuery,
    get_ppi_stats,
    # Drug Tools
    query_drug_interaction, DrugInteractionQuery,
    get_drug_interaction_stats,
    # Gene Set Tools
    query_msigdb, MSigDBQuery,
    query_mousemine, MouseMineQuery,
    get_geneset_stats,
    # Misc Tools
    query_evebio, EVEBioQuery,
    query_broad_repurposing, BroadRepurposingQuery,
    query_virus_host_ppi, VirusHostPPIQuery,
    query_depmap, DepMapQuery,
    query_celltype_marker, CellTypeMarkerQuery,
    query_czi_census, CZICensusQuery,
    # GWAS Tools
    query_gwas_catalog, GWASCatalogQuery,
    # Genebass Tools
    query_genebass, GenebassQuery,
    # TxGNN Tools
    query_drug_for_disease, DrugForDiseaseQuery,
    query_disease_for_drug, DiseaseForDrugQuery,
    # GO Tools
    query_go_term, GOTermQuery,
    query_go_hierarchy, GOHierarchyQuery,
    query_go_relations, GORelationsQuery,
    # HPO Tools
    query_hpo_term, HPOTermQuery,
    query_hpo_hierarchy, HPOHierarchyQuery,
    query_hpo_xref, HPOXrefQuery,
)

# Import Analysis Tools (using @tool decorator - LangChain 1.0 compatible)
from agent.nodes.subagents.general_qa.tools.analysis_tools import (
    verify_multi_statement,
    calculate_modification_mass,
    analyze_sgrna,
    analyze_experimental_data,
    get_analysis_tools,
    MultiStatementInput,
    ModificationInput,
    SgRNAAnalysisInput,
    ExperimentalDataInput
)


# Wrapper functions to convert expanded parameters to Pydantic models
def _query_knowledge_graph_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_knowledge_graph that accepts expanded parameters"""
    query = KnowledgeGraphQuery(**kwargs)
    return query_knowledge_graph(query)

def _query_disgenet_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_disgenet that accepts expanded parameters"""
    query = DisGeNETQuery(**kwargs)
    return query_disgenet(query)

def _query_omim_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_omim that accepts expanded parameters"""
    query = OMIMQuery(**kwargs)
    return query_omim(query)

def _query_tcr_mcpas_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_tcr_mcpas that accepts expanded parameters"""
    query = TCRQuery(**kwargs)
    return query_tcr_mcpas(query)

def _query_mirdb_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_mirdb that accepts expanded parameters"""
    query = MiRDBQuery(**kwargs)
    return query_mirdb(query)

def _query_mirtarbase_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_mirtarbase that accepts expanded parameters"""
    query = MiRTarBaseQuery(**kwargs)
    return query_mirtarbase(query)

def _query_bindingdb_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_bindingdb that accepts expanded parameters"""
    query = BindingDBQuery(**kwargs)
    return query_bindingdb(query)

def _query_gtex_expression_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_gtex_expression that accepts expanded parameters"""
    query = GTExQuery(**kwargs)
    return query_gtex_expression(query)

def _query_sgrna_human_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_sgrna_human that accepts expanded parameters"""
    query = SgRNAQuery(**kwargs)
    return query_sgrna_human(query)

def _query_sgrna_mouse_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_sgrna_mouse that accepts expanded parameters"""
    query = SgRNAQuery(**kwargs)
    return query_sgrna_mouse(query)

def _query_genetic_interaction_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_genetic_interaction that accepts expanded parameters"""
    query = GeneticInteractionQuery(**kwargs)
    return query_genetic_interaction(query)

def _query_variant_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_variant that accepts expanded parameters"""
    query = VariantQuery(**kwargs)
    return query_variant(query)

def _query_proteinatlas_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_proteinatlas that accepts expanded parameters"""
    query = ProteinAtlasQuery(**kwargs)
    return query_proteinatlas(query)

def _query_gene_info_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_gene_info that accepts expanded parameters"""
    query = GeneInfoQuery(**kwargs)
    return query_gene_info(query)

def _query_ppi_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_ppi that accepts expanded parameters"""
    query = PPIQuery(**kwargs)
    return query_ppi(query)

def _query_synthetic_interaction_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_synthetic_interaction that accepts expanded parameters"""
    query = SyntheticInteractionQuery(**kwargs)
    return query_synthetic_interaction(query)

def _query_drug_interaction_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_drug_interaction that accepts expanded parameters"""
    query = DrugInteractionQuery(**kwargs)
    return query_drug_interaction(query)

def _query_msigdb_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_msigdb that accepts expanded parameters"""
    query = MSigDBQuery(**kwargs)
    return query_msigdb(query)

def _query_mousemine_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_mousemine that accepts expanded parameters"""
    query = MouseMineQuery(**kwargs)
    return query_mousemine(query)

def _query_evebio_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_evebio that accepts expanded parameters"""
    query = EVEBioQuery(**kwargs)
    return query_evebio(query)

def _query_broad_repurposing_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_broad_repurposing that accepts expanded parameters"""
    query = BroadRepurposingQuery(**kwargs)
    return query_broad_repurposing(query)

def _query_virus_host_ppi_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_virus_host_ppi that accepts expanded parameters"""
    query = VirusHostPPIQuery(**kwargs)
    return query_virus_host_ppi(query)

def _query_depmap_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_depmap that accepts expanded parameters"""
    query = DepMapQuery(**kwargs)
    return query_depmap(query)

def _query_celltype_marker_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_celltype_marker that accepts expanded parameters"""
    query = CellTypeMarkerQuery(**kwargs)
    return query_celltype_marker(query)

def _query_czi_census_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_czi_census that accepts expanded parameters"""
    query = CZICensusQuery(**kwargs)
    return query_czi_census(query)

def _query_gwas_catalog_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_gwas_catalog that accepts expanded parameters"""
    query = GWASCatalogQuery(**kwargs)
    return query_gwas_catalog(query)

def _query_genebass_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_genebass that accepts expanded parameters"""
    query = GenebassQuery(**kwargs)
    return query_genebass(query)

def _query_drug_for_disease_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_drug_for_disease that accepts expanded parameters"""
    query = DrugForDiseaseQuery(**kwargs)
    return query_drug_for_disease(query)

def _query_disease_for_drug_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_disease_for_drug that accepts expanded parameters"""
    query = DiseaseForDrugQuery(**kwargs)
    return query_disease_for_drug(query)

def _query_go_term_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_go_term that accepts expanded parameters"""
    query = GOTermQuery(**kwargs)
    return query_go_term(query)

def _query_go_hierarchy_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_go_hierarchy that accepts expanded parameters"""
    query = GOHierarchyQuery(**kwargs)
    return query_go_hierarchy(query)

def _query_go_relations_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_go_relations that accepts expanded parameters"""
    query = GORelationsQuery(**kwargs)
    return query_go_relations(query)

def _query_hpo_term_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_hpo_term that accepts expanded parameters"""
    query = HPOTermQuery(**kwargs)
    return query_hpo_term(query)

def _query_hpo_hierarchy_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_hpo_hierarchy that accepts expanded parameters"""
    query = HPOHierarchyQuery(**kwargs)
    return query_hpo_hierarchy(query)

def _query_hpo_xref_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """Wrapper for query_hpo_xref that accepts expanded parameters"""
    query = HPOXrefQuery(**kwargs)
    return query_hpo_xref(query)


def load_all_tools() -> List[StructuredTool]:
    """
    Load all biomedical tools as LangChain StructuredTool objects
    
    Returns:
        List of StructuredTool objects ready for LLM binding
    """
    tools = []
    
    # Core Query Tools
    tools.append(StructuredTool.from_function(
        func=_query_knowledge_graph_wrapper,
        name="query_knowledge_graph",
        description="Query biomedical knowledge graph for gene-disease-drug-pathway associations. Use for finding relationships between biological entities.",
        args_schema=KnowledgeGraphQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_tcr_mcpas_wrapper,
        name="query_tcr_mcpas",
        description="Query T cell receptor sequences and antigen specificity data. Use for immune system research and TCR-based therapies.",
        args_schema=TCRQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_mirdb_wrapper,
        name="query_mirdb",
        description="Query computational miRNA target predictions. Use for finding miRNA-mRNA regulatory relationships.",
        args_schema=MiRDBQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_mirtarbase_wrapper,
        name="query_mirtarbase",
        description="Query experimentally validated miRNA-target interactions. Use for verified miRNA regulation data.",
        args_schema=MiRTarBaseQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_bindingdb_wrapper,
        name="query_bindingdb",
        description="Query drug-target binding affinity data. Use for drug discovery and binding potency analysis.",
        args_schema=BindingDBQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_gtex_expression_wrapper,
        name="query_gtex_expression",
        description="Query human tissue gene expression profiles. Use for tissue-specific expression analysis.",
        args_schema=GTExQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_sgrna_human_wrapper,
        name="query_sgrna_human",
        description="Query human CRISPR sgRNA designs for gene knockout. Use for designing CRISPR experiments.",
        args_schema=SgRNAQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_sgrna_mouse_wrapper,
        name="query_sgrna_mouse",
        description="Query mouse CRISPR sgRNA designs for gene knockout. Use for mouse model experiments.",
        args_schema=SgRNAQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_genetic_interaction_wrapper,
        name="query_genetic_interaction",
        description="Query genetic interaction data including synthetic lethality. Use for finding gene-gene relationships.",
        args_schema=GeneticInteractionQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_variant_wrapper,
        name="query_variant",
        description="Query genetic variant data including SNP positions. Use for variant annotation and GWAS analysis.",
        args_schema=VariantQuery
    ))
    
    # Disease/Gene Tools
    tools.append(StructuredTool.from_function(
        func=_query_disgenet_wrapper,
        name="query_disgenet",
        description="Query disease-gene associations. Use for finding genes linked to diseases or diseases caused by gene mutations.",
        args_schema=DisGeNETQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_omim_wrapper,
        name="query_omim",
        description="Query Mendelian inheritance disease data. Use for single-gene disorders and inheritance patterns.",
        args_schema=OMIMQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_proteinatlas_wrapper,
        name="query_proteinatlas",
        description="Query comprehensive protein/gene annotations. Use for protein class, function, and location information.",
        args_schema=ProteinAtlasQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_gene_info_wrapper,
        name="query_gene_info",
        description="Query gene basic information including genomic coordinates. Use for gene position and transcript data.",
        args_schema=GeneInfoQuery
    ))
    
    # PPI Tools
    tools.append(StructuredTool.from_function(
        func=_query_ppi_wrapper,
        name="query_ppi",
        description="Query physical protein-protein interactions. Use for finding interaction partners and protein complexes.",
        args_schema=PPIQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_synthetic_interaction_wrapper,
        name="query_synthetic_interaction",
        description="Query synthetic lethality and genetic interactions. Use for cancer drug target discovery.",
        args_schema=SyntheticInteractionQuery
    ))
    
    # Drug Tools
    tools.append(StructuredTool.from_function(
        func=_query_drug_interaction_wrapper,
        name="query_drug_interaction",
        description="Query drug-drug interaction data. Use for checking medication safety and drug combinations.",
        args_schema=DrugInteractionQuery
    ))
    
    # Gene Set Tools
    tools.append(StructuredTool.from_function(
        func=_query_msigdb_wrapper,
        name="query_msigdb",
        description="Query MSigDB human gene sets for pathway analysis. Use for GSEA and functional annotation.",
        args_schema=MSigDBQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_mousemine_wrapper,
        name="query_mousemine",
        description="Query MouseMine mouse gene sets. Use for mouse model pathway analysis.",
        args_schema=MouseMineQuery
    ))
    
    # Misc Tools
    tools.append(StructuredTool.from_function(
        func=_query_evebio_wrapper,
        name="query_evebio",
        description="Query EVE Bio drug screening data. Use for compound activity profiles.",
        args_schema=EVEBioQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_broad_repurposing_wrapper,
        name="query_broad_repurposing",
        description="Query Broad Institute Drug Repurposing Hub. Use for drug repurposing research.",
        args_schema=BroadRepurposingQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_virus_host_ppi_wrapper,
        name="query_virus_host_ppi",
        description="Query virus-host protein interactions. Use for viral infection mechanism research.",
        args_schema=VirusHostPPIQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_depmap_wrapper,
        name="query_depmap",
        description="Query DepMap cancer dependency data. Use for cancer essential genes and drug sensitivity.",
        args_schema=DepMapQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_celltype_marker_wrapper,
        name="query_celltype_marker",
        description="Query cell type marker genes. Use for single-cell annotation and flow cytometry.",
        args_schema=CellTypeMarkerQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_czi_census_wrapper,
        name="query_czi_census",
        description="Query CZI single-cell dataset catalog. Use for discovering available scRNA-seq datasets.",
        args_schema=CZICensusQuery
    ))
    
    # GWAS Tools
    tools.append(StructuredTool.from_function(
        func=_query_gwas_catalog_wrapper,
        name="query_gwas_catalog",
        description="Query GWAS Catalog for SNP-disease associations. Use for genetic association studies.",
        args_schema=GWASCatalogQuery
    ))
    
    # Genebass Tools
    tools.append(StructuredTool.from_function(
        func=_query_genebass_wrapper,
        name="query_genebass",
        description="Query Genebass for gene-phenotype associations from rare variant burden tests. Use for loss-of-function analysis.",
        args_schema=GenebassQuery
    ))
    
    # TxGNN Tools
    tools.append(StructuredTool.from_function(
        func=_query_drug_for_disease_wrapper,
        name="query_drug_for_disease",
        description="Find predicted drugs for a disease using AI-based drug repurposing. Use for exploring treatment candidates.",
        args_schema=DrugForDiseaseQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_disease_for_drug_wrapper,
        name="query_disease_for_drug",
        description="Find predicted diseases treatable by a drug using AI-based predictions. Use for drug repurposing research.",
        args_schema=DiseaseForDrugQuery
    ))
    
    # GO Tools
    tools.append(StructuredTool.from_function(
        func=_query_go_term_wrapper,
        name="query_go_term",
        description="Search Gene Ontology terms by ID, name, or keyword. Use for biological process/function/component queries.",
        args_schema=GOTermQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_go_hierarchy_wrapper,
        name="query_go_hierarchy",
        description="Query GO term hierarchy (ancestors or descendants). Use for understanding GO term relationships.",
        args_schema=GOHierarchyQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_go_relations_wrapper,
        name="query_go_relations",
        description="Query GO term relationships (part_of, regulates, etc.). Use for exploring regulatory networks.",
        args_schema=GORelationsQuery
    ))
    
    # HPO Tools
    tools.append(StructuredTool.from_function(
        func=_query_hpo_term_wrapper,
        name="query_hpo_term",
        description="Search Human Phenotype Ontology terms. Use for phenotype queries and clinical observations.",
        args_schema=HPOTermQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_hpo_hierarchy_wrapper,
        name="query_hpo_hierarchy",
        description="Query HPO term hierarchy. Use for understanding phenotype relationships.",
        args_schema=HPOHierarchyQuery
    ))
    
    tools.append(StructuredTool.from_function(
        func=_query_hpo_xref_wrapper,
        name="query_hpo_xref",
        description="Query HPO cross-references to external coding systems. Use for mapping between medical terminologies.",
        args_schema=HPOXrefQuery
    ))
    
    # Analysis Tools (NEW - using @tool decorator, already LangChain 1.0 compatible)
    # These tools are already decorated with @tool, so we can add them directly
    analysis_tools = get_analysis_tools()
    for tool_func in analysis_tools:
        # @tool decorated functions are already LangChain Tool objects
        tools.append(tool_func)
    
    return tools


def get_tools_by_category() -> Dict[str, List[StructuredTool]]:
    """
    Get tools organized by category
    
    Returns:
        Dictionary mapping category names to tool lists
    """
    all_tools = load_all_tools()
    
    categories = {
        "core_query": [],  # Core biomedical queries
        "disease_gene": [],  # Disease-gene associations
        "drug": [],  # Drug-related queries
        "pathway": [],  # Pathway and gene set tools
        "ontology": [],  # GO and HPO ontologies
        "genetic": [],  # Genetic variants and GWAS
        "interaction": [],  # Protein and genetic interactions
        "expression": [],  # Expression and cell type tools
        "repurposing": [],  # Drug repurposing tools
        "analysis": [],  # Analysis tools (NEW - for specialized reasoning)
    }
    
    tool_name_to_category = {
        # Core query tools
        "query_knowledge_graph": "core_query",
        "query_tcr_mcpas": "core_query",
        "query_mirdb": "core_query",
        "query_mirtarbase": "core_query",
        "query_bindingdb": "core_query",
        "query_gtex_expression": "expression",
        "query_sgrna_human": "core_query",
        "query_sgrna_mouse": "core_query",
        "query_genetic_interaction": "interaction",
        "query_variant": "genetic",
        # Disease/Gene tools
        "query_disgenet": "disease_gene",
        "query_omim": "disease_gene",
        "query_proteinatlas": "disease_gene",
        "query_gene_info": "disease_gene",
        # PPI tools
        "query_ppi": "interaction",
        "query_synthetic_interaction": "interaction",
        # Drug tools
        "query_drug_interaction": "drug",
        # Gene set tools
        "query_msigdb": "pathway",
        "query_mousemine": "pathway",
        # Misc tools
        "query_evebio": "drug",
        "query_broad_repurposing": "repurposing",
        "query_virus_host_ppi": "interaction",
        "query_depmap": "disease_gene",
        "query_celltype_marker": "expression",
        "query_czi_census": "expression",
        # GWAS tools
        "query_gwas_catalog": "genetic",
        # Genebass tools
        "query_genebass": "genetic",
        # TxGNN tools
        "query_drug_for_disease": "repurposing",
        "query_disease_for_drug": "repurposing",
        # GO tools
        "query_go_term": "ontology",
        "query_go_hierarchy": "ontology",
        "query_go_relations": "ontology",
        # HPO tools
        "query_hpo_term": "ontology",
        "query_hpo_hierarchy": "ontology",
        "query_hpo_xref": "ontology",
        # Analysis tools (NEW)
        "verify_multi_statement": "analysis",
        "calculate_modification_mass": "analysis",
        "analyze_sgrna": "analysis",
        "analyze_experimental_data": "analysis",
    }
    
    for tool in all_tools:
        category = tool_name_to_category.get(tool.name, "core_query")
        categories[category].append(tool)
    
    return categories


# 节点工具使用规则（明确哪些节点可以使用工具）
NODE_TOOL_USAGE = {
    "n0_input_preprocessing": False,  # 不调用工具
    "n1_question_decomposition": False,  # 不调用工具
    "n2_calculation_algorithm_recognition": False,  # 不调用工具
    "n3_knowledge_retrieval": True,  # 优先调用领域工具
    "n4_calculation_decomposition": True,  # 可以调用工具
    "n5_algorithm_validation": True,  # 可以调用工具
    "n6_initial_inference": False,  # 工具一般不需要用于推理本身，但用于检索
    "n7_complete_inference": False,  # 工具一般不需要用于推理本身
    "n8_answer_generation": False,  # 工具一般不需要用于答案生成本身
    "n9_result_validation": True,  # 可以调用工具进行验证
    "n10_exception_handling": True,  # 所有工具用于寻找替代方案
    "n11_manual_intervention": False,  # 不调用工具
}


def get_tools_for_node(
    node_name: str,
    domain: Optional[str] = None,
    question_type: Optional[str] = None
) -> List[StructuredTool]:
    """
    Get appropriate tools for a specific node based on its function and domain
    
    Args:
        node_name: Name of the node (e.g., "n3_knowledge_retrieval")
        domain: Domain from raw_subject or core_domains
        question_type: Question type label
    
    Returns:
        List of tools appropriate for the node and domain
    """
    # 如果工具使用被明确禁用，返回空列表
    if not NODE_TOOL_USAGE.get(node_name, True):  # 默认True如果未指定
        return []
    
    categories = get_tools_by_category()
    all_tools = load_all_tools()
    tool_map = {tool.name: tool for tool in all_tools}
    
    # 获取基础工具（基于节点功能）
    base_tool_map = {
        # N0: Input preprocessing - no tools needed
        "n0_input_preprocessing": [],
        
        # N1: Question decomposition - basic entity lookup tools + analysis tools
        "n1_question_decomposition": (
            categories["disease_gene"] +
            categories["ontology"][:2] +  # GO and HPO term search
            categories["analysis"]  # Analysis tools for statement detection
        ),
        
        # N2: Calculation/algorithm recognition - no tools needed
        "n2_calculation_algorithm_recognition": [],
        
        # N3: Knowledge retrieval - ALL tools for comprehensive knowledge retrieval
        "n3_knowledge_retrieval": all_tools,
        
        # N4: Calculation decomposition - expression and variant tools for calculations
        "n4_calculation_decomposition": (
            categories["expression"] +
            categories["genetic"] +
            categories["core_query"][:3] +  # Basic query tools
            categories["analysis"]  # Include modification mass calculator
        ),
        
        # N5: Algorithm validation - pathway and interaction tools
        "n5_algorithm_validation": (
            categories["pathway"] +
            categories["interaction"] +
            categories["disease_gene"]
        ),
        
        # N6: Initial inference - knowledge retrieval tools + analysis tools
        "n6_initial_inference": (
            categories["core_query"] +
            categories["disease_gene"] +
            categories["interaction"] +
            categories["genetic"] +
            categories["analysis"]  # Analysis tools for data interpretation
        ),
        
        # N7: Complete inference - all tools for comprehensive reasoning
        "n7_complete_inference": all_tools,
        
        # N8: Answer generation - focused tools for answer refinement + analysis
        "n8_answer_generation": (
            categories["disease_gene"] +
            categories["ontology"] +
            categories["drug"] +
            categories["analysis"]  # Analysis tools for verification
        ),
        
        # N9: Result validation - validation tools + analysis tools
        "n9_result_validation": (
            categories["disease_gene"][:2] +  # DisGeNET, OMIM for validation
            categories["ontology"][:2] +  # GO, HPO for validation
            categories["analysis"]  # Analysis tools for statement verification
        ),
        
        # N10: Exception handling - all tools for finding alternatives
        "n10_exception_handling": all_tools,
        
        # N11: Manual intervention - no tools needed
        "n11_manual_intervention": [],
    }
    
    base_tools = base_tool_map.get(node_name, [])
    
    # 如果指定了领域，获取领域特定工具
    if domain or question_type:
        try:
            from agent.nodes.subagents.general_qa.prompts.domain_mapper import get_prompt_module
            
            domain_module = get_prompt_module(domain=domain, question_type=question_type)
            if hasattr(domain_module, 'get_domain_tools'):
                domain_tool_names = domain_module.get_domain_tools()
                domain_tools = [tool_map[name] for name in domain_tool_names if name in tool_map]
                
                # 合并工具，领域工具优先（去重）
                merged_tools = []
                seen_names = set()
                
                # 先添加领域工具
                for tool in domain_tools:
                    if tool.name not in seen_names:
                        merged_tools.append(tool)
                        seen_names.add(tool.name)
                
                # 再添加基础工具（如果不在领域工具中）
                for tool in base_tools:
                    if tool.name not in seen_names:
                        merged_tools.append(tool)
                        seen_names.add(tool.name)
                
                return merged_tools
        except Exception as e:
            # 如果获取领域工具失败，降级到基础工具
            print(f"Warning: Failed to get domain tools for {domain}/{question_type}: {e}")
            pass
    
    return base_tools

