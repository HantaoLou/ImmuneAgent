"""
Tool Trigger Module
============================================
根据关键词和领域自动触发相关工具调用
"""

from typing import List, Dict, Set, Optional
from langchain_core.tools import StructuredTool
import re

# 关键词到工具的映射
KEYWORD_TO_TOOLS = {
    # 药物相关
    "drug": ["query_drug_interaction", "query_drug_for_disease", "query_disease_for_drug", "query_broad_repurposing"],
    "medication": ["query_drug_interaction", "query_drug_for_disease"],
    "hypertension": ["query_drug_interaction", "query_drug_for_disease"],
    "htn": ["query_drug_interaction", "query_drug_for_disease"],
    "antihypertensive": ["query_drug_interaction"],
    "prescription": ["query_drug_interaction"],
    
    # 疾病相关
    "disease": ["query_disgenet", "query_omim", "query_knowledge_graph"],
    "syndrome": ["query_omim", "query_disgenet", "query_hpo_term"],
    "congenital": ["query_omim", "query_disgenet", "query_hpo_term"],
    "anomaly": ["query_omim", "query_disgenet", "query_hpo_term"],
    "defect": ["query_omim", "query_disgenet", "query_hpo_term"],
    
    # 基因相关
    "gene": ["query_gene_info", "query_disgenet", "query_proteinatlas", "query_knowledge_graph"],
    "protein": ["query_proteinatlas", "query_ppi", "query_knowledge_graph"],
    "mutation": ["query_disgenet", "query_omim", "query_variant"],
    "variant": ["query_variant", "query_gwas_catalog", "query_genebass"],
    
    # 表达相关
    "expression": ["query_gtex_expression", "query_depmap", "query_celltype_marker"],
    "rna-seq": ["query_gtex_expression", "query_depmap"],
    "transcriptome": ["query_gtex_expression"],
    
    # 通路相关
    "pathway": ["query_msigdb", "query_go_term", "query_go_hierarchy"],
    "metabolic": ["query_msigdb", "query_go_term"],
    "signaling": ["query_msigdb", "query_go_term"],
    
    # 相互作用相关
    "interaction": ["query_ppi", "query_synthetic_interaction", "query_genetic_interaction"],
    "binding": ["query_bindingdb", "query_ppi"],
    "affinity": ["query_bindingdb"],
    
    # 表型相关
    "phenotype": ["query_hpo_term", "query_hpo_hierarchy", "query_genebass"],
    "symptom": ["query_hpo_term", "query_disgenet"],
    
    # 计算相关
    "calculation": ["query_variant", "query_gwas_catalog"],
    "formula": ["query_bindingdb"],
    "affinity": ["query_bindingdb"],
    
    # 实验相关
    "plasmid": ["query_knowledge_graph"],  # 需要更专门的工具
    "vector": ["query_knowledge_graph"],
    "co-expression": ["query_knowledge_graph"],
    "chaperone": ["query_proteinatlas", "query_ppi"],
    
    # 结构相关
    "structure": ["query_proteinatlas", "query_go_term"],
    "folding": ["query_proteinatlas", "query_ppi"],
    "secondary structure": ["query_proteinatlas"],
    
    # 遗传相关
    "genetic": ["query_gwas_catalog", "query_genebass", "query_variant"],
    "fst": ["query_gwas_catalog", "query_genebass"],
    "differentiation": ["query_gwas_catalog"],
    
    # 过滤相关
    "filter": ["query_gtex_expression", "query_depmap"],
    "contamination": ["query_gtex_expression"],
    "log fold change": ["query_gtex_expression", "query_depmap"],
    "lfc": ["query_gtex_expression", "query_depmap"],
}

# 领域到工具的映射（基于领域模块配置）
DOMAIN_TO_TOOLS = {
    # 核心领域（基于prompt模块）
    "Genetics": [
        "query_gwas_catalog", "query_genebass", "query_variant",
        "query_omim", "query_disgenet", "query_gene_info"
    ],
    "Genomics": [
        "query_variant", "query_gwas_catalog", "query_genebass",
        "query_gene_info", "query_knowledge_graph"
    ],
    "Immunology": [
        "query_tcr_mcpas", "query_celltype_marker", "query_ppi",
        "query_knowledge_graph", "query_proteinatlas"
    ],
    "Clinical Medicine": [
        "query_drug_interaction", "query_drug_for_disease",
        "query_disease_for_drug", "query_omim", "query_disgenet",
        "query_hpo_term"
    ],
    "Bioinformatics": [
        "query_variant", "query_gwas_catalog", "query_knowledge_graph",
        "query_gene_info", "query_go_term"
    ],
    "Biochemistry": [
        "query_proteinatlas", "query_go_term", "query_bindingdb",
        "query_ppi", "query_knowledge_graph"
    ],
    "Molecular Biology": [
        "query_proteinatlas", "query_ppi", "query_gene_info",
        "query_go_term", "query_knowledge_graph"
    ],
    # 其他领域
    "Pediatrics": ["query_omim", "query_disgenet", "query_hpo_term"],
    "Respiratory Medicine": ["query_disgenet", "query_gtex_expression"],
    "Congenital Anomalies": ["query_omim", "query_disgenet", "query_hpo_term"],
    "Population Genetics": ["query_gwas_catalog", "query_genebass", "query_variant"],
    "Metabolic Pathways": ["query_msigdb", "query_go_term"],
    "Spectroscopy": ["query_proteinatlas"],
    "Protein Structure": ["query_proteinatlas", "query_ppi"],
}


def extract_keywords(text: str) -> Set[str]:
    """
    从文本中提取关键词
    
    Args:
        text: 输入文本
    
    Returns:
        关键词集合
    """
    if not text:
        return set()
    
    text_lower = text.lower()
    keywords = set()
    
    # 提取所有可能的关键词
    for keyword in KEYWORD_TO_TOOLS.keys():
        if keyword in text_lower:
            keywords.add(keyword)
    
    # 提取专业术语（大写字母开头的词）
    professional_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for term in professional_terms:
        term_lower = term.lower()
        if term_lower in KEYWORD_TO_TOOLS:
            keywords.add(term_lower)
    
    return keywords


def get_tools_by_keywords(
    text: str,
    domains: Optional[List[str]] = None,
    key_entities: Optional[List[str]] = None,
    all_tools: Optional[List[StructuredTool]] = None
) -> List[StructuredTool]:
    """
    根据关键词、领域和实体自动选择并返回相关工具
    
    Args:
        text: 问题文本
        domains: 核心领域列表
        key_entities: 关键实体列表
        all_tools: 所有可用工具列表
    
    Returns:
        应该调用的工具列表
    """
    if not all_tools:
        return []
    
    # 创建工具名称到工具的映射
    tool_map = {tool.name: tool for tool in all_tools}
    
    # 收集需要调用的工具名称
    required_tool_names = set()
    
    # 1. 基于文本关键词
    keywords = extract_keywords(text)
    for keyword in keywords:
        if keyword in KEYWORD_TO_TOOLS:
            required_tool_names.update(KEYWORD_TO_TOOLS[keyword])
    
    # 2. 基于领域
    if domains:
        for domain in domains:
            if domain in DOMAIN_TO_TOOLS:
                required_tool_names.update(DOMAIN_TO_TOOLS[domain])
    
    # 3. 基于关键实体
    if key_entities:
        entity_text = " ".join(key_entities).lower()
        entity_keywords = extract_keywords(entity_text)
        for keyword in entity_keywords:
            if keyword in KEYWORD_TO_TOOLS:
                required_tool_names.update(KEYWORD_TO_TOOLS[keyword])
    
    # 转换为工具对象
    selected_tools = []
    for tool_name in required_tool_names:
        if tool_name in tool_map:
            selected_tools.append(tool_map[tool_name])
    
    return selected_tools


def should_force_tool_usage(text: str, domains: Optional[List[str]] = None) -> bool:
    """
    判断是否应该强制使用工具
    
    Args:
        text: 问题文本
        domains: 核心领域列表
    
    Returns:
        是否应该强制使用工具
    """
    text_lower = text.lower()
    
    # 强制使用工具的关键词
    force_keywords = [
        "drug", "medication", "prescription", "recommend",
        "disease", "syndrome", "congenital", "defect",
        "gene", "protein", "mutation", "variant",
        "expression", "rna-seq", "transcriptome",
        "pathway", "metabolic", "signaling",
        "interaction", "binding", "affinity",
        "phenotype", "symptom",
        "plasmid", "vector", "co-expression",
        "structure", "folding",
        "genetic", "fst", "differentiation",
        "filter", "contamination", "log fold change",
    ]
    
    # 检查文本中是否包含强制关键词
    for keyword in force_keywords:
        if keyword in text_lower:
            return True
    
    # 检查领域
    if domains:
        force_domains = ["Clinical Medicine", "Genetics", "Pediatrics", "Respiratory Medicine"]
        for domain in domains:
            if domain in force_domains:
                return True
    
    return False

