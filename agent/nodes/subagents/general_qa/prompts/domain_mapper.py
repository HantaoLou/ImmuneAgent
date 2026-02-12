"""
Domain Mapper for Prompt Routing
Enhanced with domain identification, cross-domain detection, and caching
"""

from typing import Optional, Dict, Any, List, Tuple
import importlib
import re
import time
import threading

# 领域映射规则
DOMAIN_MAPPING = {
    # raw_subject 到 prompt 模块的映射
    "Genetics": "genetics",
    "Genomics": "genomics",
    "Immunology": "immunology",
    "Biochemistry": "biochemistry",
    "Molecular Biology": "molecular_biology",
    "Bioinformatics": "bioinformatics",
    "Computational Biology": "bioinformatics",  # 合并到 bioinformatics
    "Clinical Medicine": "clinical_medicine",
    "Medicine": "clinical_medicine",  # 合并
    "Microbiology": "microbiology",
    "Biophysics": "biophysics",
    "Neuroscience": "neuroscience",
    "Pathology": "pathology",
    "Pharmacy": "pharmacy",
    "Physiology": "physiology",
    "Anatomy": "anatomy",
    "Ecology": "ecology",
    "Public Health": "public_health",
    "Bioengineering": "bioengineering",
    "Biology": "general",  # 通用领域
    
    # question_type 到 prompt 模块的映射
    "genetics_genomics": "genetics",
    "bioinformatics": "bioinformatics",
    "clinical_medicine": "clinical_medicine",
    "protein_structure": "biochemistry",
    "signaling_pathway": "molecular_biology",
    "vdj_bcr_tcr": "immunology",
    "immune_cells": "immunology",
    "antibody": "immunology",
    "mhc_binding": "immunology",
    "microbiology": "microbiology",
    "mechanistic_reasoning": "general",
    "general_biomedical": "general",
}

# 领域核心关键词映射
DOMAIN_KEYWORDS = {
    "Genetics": [
        "allele", "genotype", "phenotype", "locus", "haplotype",
        "SNP", "variant", "mutation", "inheritance", "Hardy-Weinberg",
        "HWE", "Fst", "theta", "pi", "recombination", "linkage",
        "Mendelian", "autosomal", "X-linked", "dominant", "recessive",
        "genetic", "heredity", "chromosome", "gene", "allelic"
    ],
    "Immunology": [
        "T cell", "B cell", "NK cell", "macrophage", "dendritic cell",
        "TCR", "BCR", "MHC", "antigen", "antibody", "cytokine",
        "allelic exclusion", "allelic inclusion", "positive selection",
        "negative selection", "V(D)J", "recombination", "CD4", "CD8",
        "immune", "immunology", "lymphocyte", "leukocyte", "thymus"
    ],
    "Clinical Medicine": [
        "hypertension", "diabetes", "medication", "drug", "treatment",
        "diagnosis", "patient", "symptom", "disease", "therapy",
        "JNC8", "ADA", "guideline", "contraindication", "dosage",
        "antihypertensive", "blood pressure", "glucose", "insulin",
        "clinical", "medical", "therapeutic", "prescription"
    ],
    "Bioinformatics": [
        "theta", "pi", "Fst", "Watterson", "nucleotide diversity",
        "variant calling", "phasing", "VCF", "FASTA", "BAM",
        "chi-square", "statistical test", "population genetics",
        "sequencing", "alignment", "quality score", "imputation",
        "bioinformatics", "computational", "algorithm", "GWAS"
    ],
    "Biochemistry": [
        "concentration", "molecular weight", "enzyme", "substrate",
        "reaction", "equilibrium", "binding", "affinity", "kinetics",
        "protein", "ligand", "receptor", "pathway", "metabolism",
        "biochemical", "biochemistry", "catalysis", "inhibition"
    ],
    "Molecular Biology": [
        "transcription", "translation", "DNA", "RNA", "protein",
        "gene expression", "promoter", "enhancer", "splicing",
        "mutation", "replication", "repair", "recombination",
        "molecular", "genetic code", "codon", "ribosome"
    ],
    "Microbiology": [
        "bacteria", "virus", "microbe", "pathogen", "infection",
        "microbiology", "microbial", "bacterial", "viral"
    ],
    "Biophysics": [
        "biophysics", "membrane", "lipid", "structure", "folding",
        "spectroscopy", "NMR", "X-ray", "crystallography"
    ],
}

# 领域识别置信度阈值
DOMAIN_CONFIDENCE_THRESHOLD = 0.15  # 低于此阈值使用general
CROSS_DOMAIN_THRESHOLD = 0.20  # 跨领域识别阈值（至少2个领域超过此阈值）

# 缓存的模块实例（懒加载）
_module_cache: Dict[str, Any] = {}
_module_loading_lock = threading.Lock()  # 线程安全

# 缓存领域识别结果（5分钟有效期）
_domain_cache: Dict[Tuple[int, Optional[str]], Tuple[Optional[str], float, float]] = {}


def extract_keywords(text: str) -> List[str]:
    """从文本中提取关键词（支持大小写不敏感）"""
    if not text:
        return []
    
    text_lower = text.lower()
    keywords = []
    
    # 提取所有可能的领域关键词
    for domain, domain_keywords in DOMAIN_KEYWORDS.items():
        for keyword in domain_keywords:
            keyword_lower = keyword.lower()
            # 支持完整词匹配（避免部分匹配）
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            if re.search(pattern, text_lower):
                keywords.append(keyword)
    
    return keywords


def identify_domain(
    user_input: str,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
    """
    识别领域并返回置信度
    
    Args:
        user_input: 用户输入文本
        question_type: 问题类型（优先使用）
        core_domains: 已识别的核心领域列表
    
    Returns:
        (primary_domain, confidence, all_domain_scores)
        - primary_domain: 主要领域（None表示使用general）
        - confidence: 置信度（0-1）
        - all_domain_scores: 所有领域的得分列表 [(domain, score), ...]
    """
    # 优先使用question_type（如果已识别）
    if question_type:
        domain_module_name = DOMAIN_MAPPING.get(question_type)
        if domain_module_name:
            return question_type, 1.0, [(question_type, 1.0)]
    
    # 其次使用core_domains（如果已识别）
    if core_domains and len(core_domains) > 0:
        primary_domain = core_domains[0]
        confidence = 0.8 if len(core_domains) == 1 else 0.6  # 多领域时降低置信度
        all_scores = [(d, 0.8 if i == 0 else 0.6) for i, d in enumerate(core_domains)]
        return primary_domain, confidence, all_scores
    
    # 基于关键词匹配识别领域
    keywords = extract_keywords(user_input)
    if not keywords:
        return None, 0.0, []
    
    # 计算各领域得分
    domain_scores = {}
    text_words = set(user_input.lower().split())
    total_words = len(text_words)
    
    for domain, domain_keywords in DOMAIN_KEYWORDS.items():
        domain_keywords_lower = [kw.lower() for kw in domain_keywords]
        matched_keywords = [kw for kw in domain_keywords_lower if kw in text_words]
        # 得分 = 匹配关键词数 / 文本总词数（归一化）
        score = len(matched_keywords) / max(total_words, 1)
        domain_scores[domain] = score
    
    # 排序并选择主要领域
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    all_domain_scores = sorted_domains
    
    if not sorted_domains:
        return None, 0.0, []
    
    primary_domain, max_score = sorted_domains[0]
    
    # 检查置信度阈值
    if max_score < DOMAIN_CONFIDENCE_THRESHOLD:
        return None, 0.0, all_domain_scores
    
    # 检查是否为跨领域问题（至少2个领域超过阈值）
    if len(sorted_domains) >= 2 and sorted_domains[1][1] >= CROSS_DOMAIN_THRESHOLD:
        # 返回跨领域标记
        return "cross_domain", max_score, all_domain_scores
    
    return primary_domain, max_score, all_domain_scores


def detect_cross_domain(
    user_input: str,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    检测是否为跨领域问题
    
    Returns:
        (is_cross_domain, domains)
    """
    primary_domain, confidence, all_scores = identify_domain(
        user_input, question_type, core_domains
    )
    
    if primary_domain == "cross_domain":
        # 提取所有超过阈值的领域
        domains = [domain for domain, score in all_scores if score >= CROSS_DOMAIN_THRESHOLD]
        return True, domains[:3]  # 最多保留3个领域
    
    # 检查是否有多领域输入
    if core_domains and len(core_domains) > 1:
        return True, core_domains[:3]
    
    return False, []


def get_cached_domain(user_input: str, question_type: Optional[str] = None) -> Optional[Tuple[Optional[str], float]]:
    """获取缓存的领域识别结果"""
    cache_key = (hash(user_input[:100]), question_type)  # 只缓存前100字符
    if cache_key in _domain_cache:
        domain, confidence, timestamp = _domain_cache[cache_key]
        if time.time() - timestamp < 300:  # 5分钟有效期
            return domain, confidence
        else:
            del _domain_cache[cache_key]
    return None


def cache_domain_result(user_input: str, question_type: Optional[str], domain: Optional[str], confidence: float):
    """缓存领域识别结果"""
    cache_key = (hash(user_input[:100]), question_type)
    _domain_cache[cache_key] = (domain, confidence, time.time())
    # 清理过期缓存（简单实现，实际可用定时任务）
    if len(_domain_cache) > 1000:
        current_time = time.time()
        expired_keys = [k for k, v in _domain_cache.items() if current_time - v[2] >= 300]
        for k in expired_keys:
            del _domain_cache[k]


def get_prompt_module(
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    user_input: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> Any:
    """
    Get the appropriate prompt module with domain identification and lazy loading
    
    Args:
        domain: Domain from raw_subject (e.g., "Genetics")
        question_type: Question type (e.g., "genetics_genomics")
        user_input: User input text (for domain identification)
        core_domains: Already identified core domains list
    
    Returns:
        Prompt module instance
    """
    # 如果未提供domain，尝试从user_input识别
    if not domain and user_input:
        cached_result = get_cached_domain(user_input, question_type)
        if cached_result:
            domain, confidence = cached_result
        else:
            domain, confidence, _ = identify_domain(user_input, question_type, core_domains)
            if domain:
                cache_domain_result(user_input, question_type, domain, confidence)
    
    # 检查是否为跨领域
    if domain == "cross_domain" or (core_domains and len(core_domains) > 1):
        is_cross, domains = detect_cross_domain(user_input or "", question_type, core_domains)
        if is_cross:
            # 返回跨领域模块
            try:
                module = importlib.import_module("agent.nodes.subagents.general_qa.prompts.prompt_cross_domain")
                if hasattr(module, '_domains'):
                    module._domains = domains  # 设置涉及的领域
                return module
            except ImportError:
                pass
    
    # 优先使用 question_type，其次使用 domain
    key = question_type or domain
    
    # 懒加载：检查缓存
    if key in _module_cache:
        return _module_cache[key]
    
    # 查找映射
    module_name = DOMAIN_MAPPING.get(key, "general")
    
    # 线程安全的懒加载
    with _module_loading_lock:
        # 双重检查（避免并发加载）
        if key in _module_cache:
            return _module_cache[key]
        
        # 动态导入模块
        try:
            module = importlib.import_module(f"agent.nodes.subagents.general_qa.prompts.prompt_{module_name}")
            _module_cache[key] = module
            return module
        except ImportError:
            # 如果模块不存在，使用通用模块
            module = importlib.import_module("agent.nodes.subagents.general_qa.prompts.prompt_general")
            _module_cache[key] = module
            return module


def detect_domain_from_state(state: Any) -> Optional[str]:
    """
    Detect domain from state object
    
    Args:
        state: GeneralQAState object
    
    Returns:
        Detected domain string or None
    """
    # 优先使用 question_type_label
    if hasattr(state, 'question_type_label') and state.question_type_label:
        return state.question_type_label
    
    # 其次使用 core_domains
    if hasattr(state, 'core_domains') and state.core_domains:
        if isinstance(state.core_domains, list) and len(state.core_domains) > 0:
            return state.core_domains[0]
        elif isinstance(state.core_domains, str):
            return state.core_domains
    
    # 最后使用 question_category_standard 推断
    if hasattr(state, 'question_category_standard') and state.question_category_standard:
        category = state.question_category_standard
        if category.startswith("ProfessionalKnowledge-"):
            subcategory = category.split("-", 1)[1]
            # 尝试映射子类别到领域
            subcategory_mapping = {
                "Genetics": "Genetics",
                "Immunology": "Immunology",
                "Biochemistry": "Biochemistry",
                "LipidBiophysics": "Biophysics",
                "ProteinStructure": "Biochemistry",
            }
            return subcategory_mapping.get(subcategory)
    
    return None

