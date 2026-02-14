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
    "Genomics": "genetics",
    "Population Genetics": "genetics",
    "HWE": "genetics",
    "Fst Analysis": "genetics",
    "Linkage Analysis": "genetics",
    "GWAS": "genetics",
    "Variant": "genetics",
    
    "Immunology": "immunology",
    "T Cell": "immunology",
    "B Cell": "immunology",
    "TCR": "immunology",
    "BCR": "immunology",
    "MHC": "immunology",
    "Antigen Presentation": "immunology",
    "V(D)J": "immunology",
    "Hematopoiesis": "immunology",
    "Stem Cell": "immunology",
    
    "Biochemistry": "biochemistry",
    "Enzyme": "biochemistry",
    "Metabolism": "biochemistry",
    "Sugar Metabolism": "biochemistry",
    "Carbohydrate": "biochemistry",
    "Raffinose": "biochemistry",
    "Concentration": "biochemistry",
    "Binding": "biochemistry",
    "Kinetics": "biochemistry",
    
    "Molecular Biology": "molecular_biology",
    "Gene Expression": "molecular_biology",
    "Transcription": "molecular_biology",
    "Translation": "molecular_biology",
    "Pathway": "molecular_biology",
    
    "Bioinformatics": "bioinformatics",
    "Computational Biology": "bioinformatics",
    "Sequence Analysis": "bioinformatics",
    "Alignment": "bioinformatics",
    
    "Clinical Medicine": "clinical_medicine",
    "Medicine": "clinical_medicine",
    "Drug": "clinical_medicine",
    "Medication": "clinical_medicine",
    "Hypertension": "clinical_medicine",
    "Diabetes": "clinical_medicine",
    "Treatment": "clinical_medicine",
    
    "Microbiology": "microbiology",
    "Virus": "microbiology",
    "Bacteria": "microbiology",
    "Pathogen": "microbiology",
    "Infection": "microbiology",
    
    "Biophysics": "biophysics",
    "Membrane": "biophysics",
    "Lipid": "biophysics",
    "Cell Signaling": "biophysics",
    "Receptor": "biophysics",
    
    "Entomology": "general",  # 昆虫学没有专门模块，使用general
    "Insect Physiology": "general",
    "Aphid": "general",
    "Host Adaptation": "general",
    
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

# 细粒度领域关键词映射（增强版）
FINE_GRAINED_DOMAIN_KEYWORDS = {
    # Population Genetics 细分
    "Population Genetics": [
        "Fst", "theta", "pi", "nucleotide diversity", "Watterson",
        "Hardy-Weinberg", "HWE", "allele frequency", "genotype frequency",
        "population structure", "genetic drift", "gene flow", "migration",
        "hybrid zone", "cline", "isolation", "subpopulation"
    ],
    "Linkage Genetics": [
        "linkage disequilibrium", "LD", "haplotype", "recombination",
        "genetic map", "centimorgan", "cM", "linkage map"
    ],
    "Molecular Genetics": [
        "SNP", "variant", "mutation", "insertion", "deletion",
        "sequencing", "genotype", "allele"
    ],
    
    # Immunology 细分
    "T Cell Biology": [
        "T cell", "CD4", "CD8", "TCR", "T cell receptor",
        "thymus", "positive selection", "negative selection",
        "regulatory T cell", "Treg", "effector T cell"
    ],
    "B Cell Biology": [
        "B cell", "BCR", "B cell receptor", "antibody",
        "plasma cell", "memory B cell", "germinal center",
        "somatic hypermutation", "class switching"
    ],
    "V(D)J Recombination": [
        "V(D)J", "VDJ", "rearrangement", "recombination",
        "allelic exclusion", "allelic inclusion", "RAG",
        "junctional diversity", "N nucleotide", "P nucleotide"
    ],
    "Hematopoiesis": [
        "HSC", "hematopoietic stem cell", "MPP", "multipotent progenitor",
        "lineage", "differentiation", "bone marrow", "hematopoiesis"
    ],
    
    # Biochemistry 细分
    "Enzyme Kinetics": [
        "enzyme", "substrate", "Km", "Vmax", "kcat",
        "Michaelis-Menten", "inhibition", "catalysis"
    ],
    "Sugar Metabolism": [
        "sucrose", "raffinose", "glucose", "fructose",
        "carbohydrate", "glycolysis", "RFO", "oligosaccharide",
        "alpha-galactosidase", "galactosidase"
    ],
    "Protein Biochemistry": [
        "protein", "folding", "stability", "conformation",
        "denaturation", "aggregation", "oligomerization"
    ],
    
    # Clinical Medicine 细分
    "Hypertension Management": [
        "hypertension", "blood pressure", "antihypertensive",
        "JNC8", "ACE inhibitor", "ARB", "diuretic"
    ],
    "Drug Interaction": [
        "drug interaction", "contraindication", "adverse effect",
        "metabolism", "CYP450", "drug-drug"
    ],
    
    # Entomology (昆虫学)
    "Insect Physiology": [
        "aphid", "biotype", "host plant", "phloem sap",
        "digestive adaptation", "specialization", "host transfer"
    ],
}

# 领域核心关键词映射（保持向后兼容）
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
        "immune", "immunology", "lymphocyte", "leukocyte", "thymus",
        "HSC", "hematopoietic", "stem cell", "MPP"
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
        "biochemical", "biochemistry", "catalysis", "inhibition",
        "sucrose", "raffinose", "glucose", "carbohydrate", "sugar",
        "alpha-galactosidase", "galactosidase"
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
    "Entomology": [
        "aphid", "biotype", "insect", "host plant", "phloem",
        "cotton", "watermelon", "melon", "adaptation", "specialization"
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


def identify_fine_grained_domains(text: str) -> List[Tuple[str, str, float]]:
    """
    识别细粒度领域并映射到prompt模块
    
    Args:
        text: 用户输入文本
    
    Returns:
        List of (fine_grained_domain, prompt_module, confidence)
    """
    if not text:
        return []
    
    text_lower = text.lower()
    results = []
    
    # 检查细粒度领域关键词
    for fine_domain, keywords in FINE_GRAINED_DOMAIN_KEYWORDS.items():
        matches = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            if re.search(pattern, text_lower):
                matches += 1
        
        if matches > 0:
            confidence = matches / len(keywords)
            # 映射到prompt模块
            prompt_module = DOMAIN_MAPPING.get(fine_domain, "general")
            results.append((fine_domain, prompt_module, confidence))
    
    # 如果没有找到细粒度匹配，使用通用领域匹配
    if not results:
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                if re.search(pattern, text_lower):
                    matches += 1
            
            if matches > 0:
                confidence = matches / len(keywords)
                prompt_module = DOMAIN_MAPPING.get(domain, "general")
                results.append((domain, prompt_module, confidence))
    
    # 按置信度排序
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def map_core_domains_to_modules(core_domains: List[str], user_input: str = "") -> Tuple[str, List[str]]:
    """
    将N1识别的core_domains映射到可用的prompt模块
    
    Args:
        core_domains: N1节点识别的领域列表（如["Biochemistry", "Insect Physiology", "Sugar Metabolism"]）
        user_input: 原始用户输入（用于细粒度匹配）
    
    Returns:
        (primary_module, all_modules)
        - primary_module: 主要prompt模块
        - all_modules: 所有匹配的模块列表
    """
    if not core_domains:
        # 使用细粒度识别
        fine_results = identify_fine_grained_domains(user_input)
        if fine_results:
            return fine_results[0][1], [r[1] for r in fine_results[:3]]
        return "general", ["general"]
    
    modules = []
    for domain in core_domains:
        # 直接映射
        if domain in DOMAIN_MAPPING:
            modules.append(DOMAIN_MAPPING[domain])
        else:
            # 尝试部分匹配
            domain_lower = domain.lower()
            for key in DOMAIN_MAPPING:
                if key.lower() in domain_lower or domain_lower in key.lower():
                    modules.append(DOMAIN_MAPPING[key])
                    break
            else:
                # 使用细粒度识别
                fine_results = identify_fine_grained_domains(f"{domain} {user_input}")
                if fine_results:
                    modules.append(fine_results[0][1])
                else:
                    modules.append("general")
    
    # 去重
    unique_modules = list(dict.fromkeys(modules))
    
    # 如果只有一个模块，作为主模块
    if len(unique_modules) == 1:
        return unique_modules[0], unique_modules
    
    # 多模块情况：使用cross_domain或第一个
    if len(unique_modules) > 1:
        # 优先使用非general模块
        non_general = [m for m in unique_modules if m != "general"]
        if non_general:
            return non_general[0], unique_modules
        return unique_modules[0], unique_modules
    
    return "general", ["general"]


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
    Detect domain from state object - ENHANCED with fine-grained mapping
    
    Args:
        state: GeneralQAState object
    
    Returns:
        Detected domain string (mapped to available prompt module) or None
    """
    user_input = ""
    if hasattr(state, 'user_input'):
        user_input = state.user_input or ""
    if hasattr(state, 'cleaned_text') and state.cleaned_text:
        user_input = state.cleaned_text
    
    # 优先使用 question_type_label（但需要映射到模块）
    if hasattr(state, 'question_type_label') and state.question_type_label:
        q_type = state.question_type_label
        if q_type in DOMAIN_MAPPING:
            return DOMAIN_MAPPING[q_type]
        # 尝试细粒度匹配
        fine_results = identify_fine_grained_domains(user_input)
        if fine_results:
            return fine_results[0][1]
        return q_type
    
    # 其次使用 core_domains（应用细粒度映射）
    if hasattr(state, 'core_domains') and state.core_domains:
        if isinstance(state.core_domains, list) and len(state.core_domains) > 0:
            # 使用新的映射函数
            primary_module, all_modules = map_core_domains_to_modules(
                state.core_domains, user_input
            )
            return primary_module
        elif isinstance(state.core_domains, str):
            primary_module, _ = map_core_domains_to_modules(
                [state.core_domains], user_input
            )
            return primary_module
    
    # 最后使用 question_category_standard 推断
    if hasattr(state, 'question_category_standard') and state.question_category_standard:
        category = state.question_category_standard
        if category.startswith("ProfessionalKnowledge-"):
            subcategory = category.split("-", 1)[1]
            # 尝试映射子类别到领域
            subcategory_mapping = {
                "Genetics": "genetics",
                "Immunology": "immunology",
                "Biochemistry": "biochemistry",
                "LipidBiophysics": "biophysics",
                "ProteinStructure": "biochemistry",
            }
            mapped = subcategory_mapping.get(subcategory)
            if mapped:
                return mapped
    
    # 使用细粒度识别作为最后手段
    fine_results = identify_fine_grained_domains(user_input)
    if fine_results:
        return fine_results[0][1]
    
    return None


def detect_domain_and_modules_from_state(state: Any) -> Tuple[Optional[str], List[str]]:
    """
    检测领域并返回主要模块和所有相关模块
    
    Args:
        state: GeneralQAState object
    
    Returns:
        (primary_module, all_modules)
        - primary_module: 主要prompt模块
        - all_modules: 所有相关模块列表
    """
    user_input = ""
    if hasattr(state, 'user_input'):
        user_input = state.user_input or ""
    if hasattr(state, 'cleaned_text') and state.cleaned_text:
        user_input = state.cleaned_text
    
    core_domains = []
    if hasattr(state, 'core_domains') and state.core_domains:
        if isinstance(state.core_domains, list):
            core_domains = state.core_domains
        elif isinstance(state.core_domains, str):
            core_domains = [state.core_domains]
    
    # 使用映射函数
    primary_module, all_modules = map_core_domains_to_modules(core_domains, user_input)
    
    return primary_module, all_modules

