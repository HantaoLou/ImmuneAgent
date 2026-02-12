# General QA Prompt 优化 - 实施细节

## 一、代码结构示例

### 1.1 prompts/domain_mapper.py

```python
"""
Domain Mapper for Prompt Routing
Maps question domains to corresponding prompt modules
"""

from typing import Optional, Dict, Any
import importlib

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

# 缓存的模块实例
_module_cache: Dict[str, Any] = {}


def get_prompt_module(domain: Optional[str] = None, question_type: Optional[str] = None) -> Any:
    """
    Get the appropriate prompt module based on domain or question type
    
    Args:
        domain: Domain from raw_subject (e.g., "Genetics")
        question_type: Question type (e.g., "genetics_genomics")
    
    Returns:
        Prompt module instance
    """
    # 优先使用 question_type，其次使用 domain
    key = question_type or domain
    
    if key in _module_cache:
        return _module_cache[key]
    
    # 查找映射
    module_name = DOMAIN_MAPPING.get(key, "general")
    
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
                "Genetics": "genetics",
                "Immunology": "immunology",
                "Biochemistry": "biochemistry",
                # ... 更多映射
            }
            return subcategory_mapping.get(subcategory)
    
    return None
```

### 1.2 prompts/base.py

```python
"""
Base Prompt Templates
Contains common prompt templates used across all domains
"""

from typing import Dict, List, Any


def get_base_input_preprocessing_prompt(user_input: str) -> str:
    """
    Base template for N0: Input Preprocessing
    Domain-specific modules will enhance this template
    """
    return f"""You are a biomedical question analysis expert. Your task is to preprocess the input question, classify its type, and extract structured three-dimensional information.

Input Question:
{user_input}

Please perform the following tasks:
1. Clean the input text: Remove redundant descriptions, normalize formatting, extract core question content
2. Classify question type: Determine if this is a Multiple Choice question, Text Matching question, Mechanism Explanation question, Numerical Calculation question, Logical Calculation question, or Professional Algorithm question
   - **CRITICAL**: If question asks for "minimum number" based on grouping/logical rules (e.g., "minimum [reagents] to distinguish [entities]"), classify as "Logical Calculation", NOT "Numerical Calculation"

[... 保留原有的通用规则 ...]

Output your response in JSON format:
{{
    "cleaned_text": "cleaned question text",
    "question_type_label": "Multiple Choice|Text Matching|Mechanism Explanation|Numerical Calculation|Logical Calculation|Professional Algorithm",
    [... 保留原有的输出格式 ...]
}}
"""


def get_base_question_decomposition_prompt(
    cleaned_text: str,
    question_type_label: str,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    question_category_standard: str = None,
    category_specific_constraints: List[str] = None
) -> str:
    """Base template for N1: Question Decomposition"""
    # ... 保留原有的通用模板 ...
    pass


# ... 其他11个节点的base模板 ...
```

### 1.3 prompts/prompt-genetics.py (示例)

```python
"""
Genetics Domain-Specific Prompts
Optimized for genetics and genomics questions
"""

from typing import Dict, List, Any
from .base import (
    get_base_input_preprocessing_prompt,
    get_base_question_decomposition_prompt,
    # ... 导入其他base函数 ...
)

# 领域特定配置
DOMAIN_CONFIG = {
    "name": "Genetics",
    "priority_tools": [
        "query_gwas_catalog",
        "query_genebass",
        "query_variant",
        "query_omim",
        "query_disgenet",
        "query_gene_info"
    ],
    "common_entities": [
        "allele", "genotype", "phenotype", "locus", "haplotype",
        "SNP", "variant", "mutation", "inheritance pattern"
    ],
    "calculation_focus": [
        "Hardy-Weinberg equilibrium",
        "genetic linkage",
        "recombination frequency",
        "Fst (fixation index)",
        "nucleotide diversity (pi)",
        "Watterson's estimator (theta)"
    ],
    "validation_criteria": [
        "Must verify against population genetics principles",
        "Check HWE assumptions",
        "Verify inheritance pattern consistency"
    ]
}


def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0 prompt with genetics-specific enhancements"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    
    genetics_enhancements = f"""

**Genetics-Specific Extraction Rules:**
1. **Inheritance Pattern Recognition**: Identify and extract inheritance patterns explicitly:
   - Autosomal dominant/recessive
   - X-linked dominant/recessive
   - Mitochondrial inheritance
   - Add to structured_condition.key_features: "inheritance pattern: [pattern]"

2. **Genotype/Phenotype Extraction**: Extract genotype and phenotype relationships:
   - Identify genotype notation (e.g., "0/0", "0/1", "1/1", "AA", "Aa", "aa")
   - Extract phenotype descriptions
   - Add to structured_subject.attribute: "genotype: [genotype], phenotype: [phenotype]"

3. **Population Genetics Parameters**: Identify population genetics concepts:
   - HWE (Hardy-Weinberg equilibrium) assumptions
   - Fst, pi, theta, genetic differentiation
   - Allele frequencies, genotype frequencies
   - Add to core_keywords: ["Fst", "HWE", "pi", "theta", ...] if present

4. **Genetic Variant Notation**: Preserve exact variant notation:
   - SNP IDs (e.g., "rs123456")
   - Genomic coordinates (e.g., "chr1:123456")
   - Variant notation (e.g., "c.123A>G")
   - DO NOT modify or normalize these notations

**Genetics-Specific Category Constraints:**
- For "ProfessionalKnowledge-Genetics": ["Must verify against Mendelian genetics principles", "Check population genetics assumptions", "Verify inheritance pattern logic"]
- For "Calculation-PopulationGenetics": ["Must apply HWE equations", "Verify sample size assumptions", "Check allele frequency calculations"]
"""
    
    return base_prompt + genetics_enhancements


def get_question_decomposition_prompt(
    cleaned_text: str,
    question_type_label: str,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    question_category_standard: str = None,
    category_specific_constraints: List[str] = None
) -> str:
    """N1 prompt with genetics-specific decomposition"""
    base_prompt = get_base_question_decomposition_prompt(
        cleaned_text, question_type_label, structured_subject,
        structured_condition, structured_goal, question_category_standard,
        category_specific_constraints
    )
    
    genetics_enhancements = """

**Genetics-Specific Decomposition Patterns:**

1. **Inheritance Pattern Questions**:
   - Sub-objective 1: Identify inheritance pattern from pedigree or description
   - Sub-objective 2: Apply Mendelian genetics principles
   - Sub-objective 3: Calculate genotype/phenotype probabilities

2. **Population Genetics Questions**:
   - Sub-objective 1: Extract population parameters (allele frequencies, sample size)
   - Sub-objective 2: Apply HWE or population genetics formulas
   - Sub-objective 3: Verify assumptions and interpret results

3. **Genetic Linkage Questions**:
   - Sub-objective 1: Identify linked loci and recombination events
   - Sub-objective 2: Calculate recombination frequency
   - Sub-objective 3: Determine genetic map distances

**Genetics-Specific Domain Identification:**
- Core domains should include: "Population Genetics", "Mendelian Genetics", "Genetic Linkage", "Molecular Genetics" as appropriate
- Use precise domain names (e.g., "Population Genetics, Fst Analysis" not just "Genetics")
"""
    
    return base_prompt + genetics_enhancements


def get_knowledge_retrieval_prompt(
    cleaned_text: str,
    research_objective: str,
    core_domains: List[str],
    key_entities: List[str],
    synonyms: List[str] = None
) -> str:
    """N3 prompt with genetics-specific knowledge retrieval"""
    # ... 基于base模板，添加genetics特定的工具使用指导 ...
    pass


# ... 实现其他9个节点的prompt函数 ...


def get_domain_tools() -> List[str]:
    """Return priority tools for genetics domain"""
    return DOMAIN_CONFIG["priority_tools"]
```

### 1.4 prompt.py (统一入口，修改后)

```python
"""
Prompt definitions for General QA subgraph nodes
All prompts are in English as required.

This module serves as a unified entry point that routes to domain-specific prompts.
"""

from typing import Dict, List, Any, Optional
from .prompts.domain_mapper import get_prompt_module, detect_domain_from_state


def _get_prompt_func(func_name: str, domain: Optional[str] = None, question_type: Optional[str] = None, *args, **kwargs):
    """Internal helper to route to domain-specific prompt function"""
    module = get_prompt_module(domain=domain, question_type=question_type)
    func = getattr(module, func_name)
    return func(*args, **kwargs)


# ========== N0: Input Preprocessing & Question Classification ==========

def get_input_preprocessing_prompt(user_input: str, domain: Optional[str] = None, question_type: Optional[str] = None) -> str:
    """Prompt for N0: Input preprocessing and question classification"""
    return _get_prompt_func("get_input_preprocessing_prompt", domain, question_type, user_input)


# ========== N1: Question Decomposition & Domain Localization ==========

def get_question_decomposition_prompt(
    cleaned_text: str,
    question_type_label: str,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    question_category_standard: str = None,
    category_specific_constraints: List[str] = None,
    domain: Optional[str] = None,
    question_type: Optional[str] = None
) -> str:
    """Prompt for N1: Question decomposition and domain localization"""
    return _get_prompt_func(
        "get_question_decomposition_prompt",
        domain, question_type,
        cleaned_text, question_type_label, structured_subject,
        structured_condition, structured_goal, question_category_standard,
        category_specific_constraints
    )


# ... 其他10个节点的函数，都添加 domain 和 question_type 参数 ...
```

### 1.5 graph.py (修改节点调用)

```python
# 在节点函数中，检测领域并传递给prompt函数

def n0_input_preprocessing_node(state: GeneralQAState) -> GeneralQAState:
    """N0: Input Preprocessing & Question Classification"""
    # ... 现有代码 ...
    
    # 检测领域
    domain = detect_domain_from_state(state) if hasattr(state, 'question_type_label') else None
    
    # 调用prompt函数，传递领域信息
    prompt = get_input_preprocessing_prompt(
        state.user_input,
        domain=domain,
        question_type=getattr(state, 'question_type_label', None)
    )
    
    # ... 其余代码保持不变 ...
```

### 1.6 tools/tool_loader.py (增强工具分配)

```python
def get_tools_for_node(node_name: str, domain: Optional[str] = None, question_type: Optional[str] = None) -> List[StructuredTool]:
    """
    Get appropriate tools for a specific node based on its function and domain
    
    Args:
        node_name: Name of the node (e.g., "n3_knowledge_retrieval")
        domain: Domain from raw_subject or core_domains
        question_type: Question type label
    
    Returns:
        List of tools appropriate for the node and domain
    """
    # 获取基础工具
    base_tools = get_base_tools_for_node(node_name)
    
    # 如果指定了领域，获取领域特定的工具
    if domain or question_type:
        domain_module = domain_mapper.get_prompt_module(domain=domain, question_type=question_type)
        if hasattr(domain_module, 'get_domain_tools'):
            domain_tool_names = domain_module.get_domain_tools()
            all_tools = load_all_tools()
            tool_map = {tool.name: tool for tool in all_tools}
            domain_tools = [tool_map[name] for name in domain_tool_names if name in tool_map]
            
            # 合并工具，领域工具优先
            merged_tools = domain_tools + [t for t in base_tools if t not in domain_tools]
            return merged_tools
    
    return base_tools
```

## 二、实施优先级

### Phase 1: 基础架构（必须）
1. ✅ 创建 `prompts/` 文件夹结构
2. ✅ 实现 `prompts/base.py` 包含所有base模板
3. ✅ 实现 `prompts/domain_mapper.py` 路由逻辑
4. ✅ 实现 `prompts/prompt-general.py` 作为默认
5. ✅ 修改 `prompt.py` 作为统一入口

### Phase 2: 核心领域（高优先级）
基于题目数据统计，优先实现：
1. **Genetics** - 最高频
2. **Immunology** - 高频
3. **Clinical Medicine** - 高频
4. **Bioinformatics** - 中高频

### Phase 3: 工具分配优化
1. 增强 `tool_trigger.py` 的领域映射
2. 修改 `tool_loader.py` 支持领域参数
3. 在 `graph.py` 中传递领域信息

### Phase 4: 其他领域（逐步扩展）
按需实现剩余领域

## 三、测试策略

1. **单元测试**：测试领域路由逻辑
2. **集成测试**：使用 `csv_questions_data.json` 中的题目测试
3. **对比测试**：对比优化前后的准确率
4. **回归测试**：确保现有功能不受影响

## 四、跨领域问题处理

### 4.1 domain_mapper.py 增强

```python
def detect_cross_domain(state: Any) -> bool:
    """检测是否为跨领域问题"""
    # 检查core_domains是否包含多个领域
    if hasattr(state, 'core_domains') and state.core_domains:
        if isinstance(state.core_domains, list) and len(state.core_domains) > 1:
            return True
    
    # 检查question_type是否映射到多个领域
    if hasattr(state, 'question_type_label') and state.question_type_label:
        # 某些question_type可能涉及多个领域
        multi_domain_types = ["genetics_genomics", "bioinformatics"]  # 可能涉及多个领域
        if state.question_type_label in multi_domain_types:
            return True
    
    return False


def get_cross_domain_modules(state: Any) -> List[Any]:
    """获取跨领域问题涉及的所有领域模块"""
    domains = []
    
    if hasattr(state, 'core_domains') and state.core_domains:
        if isinstance(state.core_domains, list):
            domains = state.core_domains
        elif isinstance(state.core_domains, str):
            domains = [state.core_domains]
    
    modules = []
    for domain in domains:
        module = get_prompt_module(domain=domain)
        if module and module not in modules:
            modules.append(module)
    
    return modules
```

### 4.2 prompt-cross_domain.py

```python
"""
Cross-Domain Prompt Module
Handles questions that span multiple domains
"""

from typing import List, Dict, Any
from .base import get_base_input_preprocessing_prompt, ...
from .domain_mapper import get_cross_domain_modules


def get_input_preprocessing_prompt(user_input: str, domains: List[str] = None) -> str:
    """跨领域问题的N0 prompt，融合多个领域的规则"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    
    if not domains:
        return base_prompt
    
    # 收集所有涉及领域的增强规则
    domain_enhancements = []
    for domain in domains:
        try:
            from .domain_mapper import get_prompt_module
            domain_module = get_prompt_module(domain=domain)
            if hasattr(domain_module, 'get_domain_extraction_rules'):
                rules = domain_module.get_domain_extraction_rules()
                domain_enhancements.append(f"**{domain} Rules:**\n{rules}")
        except:
            continue
    
    # 合并所有领域的规则
    if domain_enhancements:
        merged_enhancements = "\n\n".join(domain_enhancements)
        return base_prompt + f"\n\n**Multi-Domain Extraction Rules (merged from {len(domains)} domains):**\n{merged_enhancements}"
    
    return base_prompt


def get_domain_tools(domains: List[str]) -> List[str]:
    """合并多个领域的工具列表"""
    all_tools = set()
    
    for domain in domains:
        try:
            from .domain_mapper import get_prompt_module
            domain_module = get_prompt_module(domain=domain)
            if hasattr(domain_module, 'get_domain_tools'):
                tools = domain_module.get_domain_tools()
                all_tools.update(tools)
        except:
            continue
    
    # 按领域优先级排序（Genetics > Immunology > Clinical Medicine > ...）
    priority_order = {
        "Genetics": 1,
        "Immunology": 2,
        "Clinical Medicine": 3,
        "Bioinformatics": 4,
        # ... 其他领域
    }
    
    # 按优先级排序工具
    sorted_tools = sorted(all_tools, key=lambda t: priority_order.get(t, 99))
    return sorted_tools
```

## 五、计算类问题优化

### 5.1 base.py 中的计算类模板

```python
def get_calculation_guide() -> str:
    """通用计算步骤指导，所有领域复用"""
    return """
**Calculation Step Guide (Universal - All Domains):**

1. **Parameter Extraction**: Extract all key parameters and confirm their validity
   - Identify all numerical values, units, and ranges
   - Verify parameter units are consistent (convert if necessary)
   - Check parameter ranges are valid (e.g., frequency ∈ [0,1], concentration > 0, probability ≤ 1)
   - Identify missing parameters and mark them explicitly in missing_core_entities

2. **Formula Selection**: Select appropriate formula based on domain and question type
   - For population genetics: Hardy-Weinberg equation (p²+2pq+q²=1), theta, pi, Fst formulas
   - For concentration: C1V1 = C2V2, dilution formulas
   - For statistical tests: Chi-square (χ² = Σ((O-E)²/E)), t-test, F-test formulas
   - Verify formula applicability conditions (e.g., HWE assumptions, sample size requirements)

3. **Step-by-Step Calculation**: Perform calculation showing each intermediate step
   - Show substitution of values into formula explicitly
   - Calculate intermediate results step by step
   - Track units throughout calculation (ensure unit consistency)
   - Show all mathematical operations clearly

4. **Result Verification**: Verify result conforms to domain constraints
   - Check result is within expected range:
     * Probabilities: ∈ [0,1]
     * Frequencies: ∈ [0,1]
     * Counts: ≥ 0 (integers)
     * Concentrations: > 0
   - Verify units are correct and match question requirements
   - Check result makes biological/clinical sense
   - Compare with critical/reference values if applicable (e.g., chi-square critical values)
   - Verify result satisfies all constraints from the question
"""


def get_calculation_validation_rules() -> Dict[str, List[str]]:
    """各领域的计算验证规则"""
    return {
        "Genetics": [
            "Allele frequencies must sum to 1: p + q = 1",
            "Genotype frequencies must sum to 1: p² + 2pq + q² = 1",
            "Probabilities must be ∈ [0,1]",
            "Fst must be ∈ [0,1]"
        ],
        "Biochemistry": [
            "Concentrations must be > 0",
            "Molecular weights must be > 0",
            "Reaction rates must be ≥ 0"
        ],
        "Bioinformatics": [
            "Theta and pi must be ≥ 0",
            "Fst must be ∈ [0,1]",
            "Chi-square must be ≥ 0",
            "P-values must be ∈ [0,1]"
        ],
        "Clinical Medicine": [
            "Dosages must be > 0",
            "Dosing frequencies must be positive integers",
            "Drug concentrations must be within therapeutic range"
        ]
    }
```

### 5.2 领域模块中的计算规则集成

各领域在N4（计算分解）和N8（答案生成）节点中引入计算类模板：

```python
# 在prompt-genetics.py中
def get_calculation_decomposition_prompt(...) -> str:
    base_prompt = get_base_calculation_decomposition_prompt(...)
    calculation_guide = get_calculation_guide()
    validation_rules = get_calculation_validation_rules()["Genetics"]
    
    genetics_calculation_rules = f"""
{calculation_guide}

**Genetics-Specific Calculation Rules:**

1. **Hardy-Weinberg Calculations**:
   - Verify HWE assumptions: random mating, no selection, no mutation, no migration, large population
   - Apply HWE equation: p² + 2pq + q² = 1 where p + q = 1
   - Check: p² + 2pq + q² must equal 1 (within rounding error)

2. **Population Genetics Parameters**:
   - Theta (Watterson's estimator): θ = S / Σ(1/i) where S is number of segregating sites, i from 1 to n-1
   - Pi (nucleotide diversity): π = Σ(2pq) summed over all polymorphic sites
   - Fst: Fst = (HT - HS) / HT where HT is total heterozygosity, HS is subpopulation heterozygosity
   - Verify: theta ≥ 0, pi ≥ 0, Fst ∈ [0,1]

3. **Genetic Probability Calculations**:
   - Inheritance probabilities must sum to 1 for all possible outcomes
   - Conditional probabilities: P(A|B) = P(A and B) / P(B)
   - Verify: all probabilities ∈ [0,1], sum of probabilities = 1

**Validation Rules:**
{chr(10).join(f"- {rule}" for rule in validation_rules)}
"""
    
    return base_prompt + genetics_calculation_rules
```

## 六、节点-工具调用规则

### 6.1 tool_loader.py 增强

```python
# 节点是否支持工具调用（明确规则）
NODE_TOOL_USAGE = {
    "n0_input_preprocessing": False,  # 不调用工具，纯文本预处理
    "n1_question_decomposition": False,  # 不调用工具，问题分解
    "n2_calculation_algorithm_recognition": False,  # 不调用工具，算法识别
    "n3_knowledge_retrieval": True,  # 必须调用工具，知识检索
    "n4_calculation_decomposition": True,  # 可选调用工具，获取计算参数
    "n5_algorithm_validation": True,  # 可选调用工具，验证算法适用性
    "n6_initial_inference": False,  # 不调用工具，基于已有知识推理
    "n7_complete_inference": False,  # 不调用工具，完整逻辑推理
    "n8_answer_generation": False,  # 不调用工具，答案生成
    "n9_result_validation": True,  # 可选调用工具，验证结果
    "n10_exception_handling": True,  # 可选调用工具，异常处理
    "n11_manual_intervention": False,  # 不调用工具，人工介入
}

# 节点工具调用优先级（True=必须调用，False=可选调用）
NODE_TOOL_PRIORITY = {
    "n3_knowledge_retrieval": True,  # 必须调用工具
    "n4_calculation_decomposition": False,  # 可选调用
    "n5_algorithm_validation": False,  # 可选调用
    "n9_result_validation": False,  # 可选调用
    "n10_exception_handling": False,  # 可选调用
}


def get_tools_for_node(
    node_name: str,
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    cross_domain: bool = False,
    domains: Optional[List[str]] = None
) -> List[StructuredTool]:
    """
    根据节点和领域返回工具，遵循节点工具使用规则
    
    Args:
        node_name: 节点名称
        domain: 单一领域
        question_type: 问题类型
        cross_domain: 是否为跨领域问题
        domains: 跨领域问题的领域列表
    
    Returns:
        工具列表，如果节点不支持工具调用则返回空列表
    """
    # 检查节点是否支持工具调用
    if not NODE_TOOL_USAGE.get(node_name, False):
        return []  # 不支持工具调用的节点返回空列表
    
    # 获取基础工具（基于节点）
    base_tools = get_base_tools_for_node(node_name)
    
    # 处理跨领域问题
    if cross_domain and domains:
        from .domain_mapper import get_prompt_module
        all_domain_tools = []
        for d in domains:
            try:
                domain_module = get_prompt_module(domain=d)
                if hasattr(domain_module, 'get_domain_tools'):
                    tools = domain_module.get_domain_tools()
                    all_domain_tools.extend(tools)
            except:
                continue
        
        # 去重并合并
        all_tools = load_all_tools()
        tool_map = {tool.name: tool for tool in all_tools}
        domain_tools = [tool_map[name] for name in set(all_domain_tools) if name in tool_map]
        
        # 合并工具，领域工具优先
        merged_tools = domain_tools + [t for t in base_tools if t not in domain_tools]
        return merged_tools
    
    # 处理单一领域问题
    if domain or question_type:
        from .domain_mapper import get_prompt_module
        domain_module = get_prompt_module(domain=domain, question_type=question_type)
        if hasattr(domain_module, 'get_domain_tools'):
            domain_tool_names = domain_module.get_domain_tools()
            all_tools = load_all_tools()
            tool_map = {tool.name: tool for tool in all_tools}
            domain_tools = [tool_map[name] for name in domain_tool_names if name in tool_map]
            
            # 合并工具，领域工具优先
            merged_tools = domain_tools + [t for t in base_tools if t not in domain_tools]
            return merged_tools
    
    return base_tools
```

## 七、新增领域支持体系

### 7.1 领域Prompt模板生成脚本

创建 `scripts/generate_domain_prompt.py`：

```python
#!/usr/bin/env python3
"""
自动生成领域Prompt模板
用法: 
    python generate_domain_prompt.py \
        --domain Immunology \
        --entities "T cell,B cell,antibody,TCR,BCR" \
        --tools "query_tcr_mcpas,query_celltype_marker,query_ppi" \
        --calculation-focus ""
"""

import argparse
import os
from pathlib import Path

TEMPLATE = '''"""
{domain} Domain-Specific Prompts
Optimized for {domain_lower} questions
"""

from typing import Dict, List, Any
from .base import (
    get_base_input_preprocessing_prompt,
    get_base_question_decomposition_prompt,
    get_base_calculation_algorithm_recognition_prompt,
    get_base_knowledge_retrieval_prompt,
    get_base_calculation_decomposition_prompt,
    get_base_algorithm_validation_prompt,
    get_base_initial_inference_prompt,
    get_base_complete_inference_prompt,
    get_base_answer_generation_prompt,
    get_base_result_validation_prompt,
    get_base_exception_handling_prompt,
    get_base_manual_intervention_prompt,
    get_calculation_guide,
)

# 领域特定配置
DOMAIN_CONFIG = {{
    "name": "{domain}",
    "priority_tools": {tools_list},
    "common_entities": {entities_list},
    "calculation_focus": {calculation_focus_list},
    "validation_criteria": [
        "Must verify against {domain_lower} principles",
        # TODO: Add domain-specific validation criteria
    ]
}}


def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0 prompt with {domain_lower}-specific enhancements"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    
    {domain_lower}_enhancements = f"""

**{domain}-Specific Extraction Rules:**
# TODO: Add domain-specific extraction rules
1. **Entity Extraction**: Extract {domain_lower}-specific entities
2. **Keyword Extraction**: Identify {domain_lower} core concepts
3. **Category Constraints**: Apply {domain_lower}-specific category constraints

**{domain}-Specific Category Constraints:**
- For "ProfessionalKnowledge-{domain}": ["Must verify against {domain_lower} principles"]
"""
    
    return base_prompt + {domain_lower}_enhancements


# TODO: Implement remaining 11 node prompt functions
# - get_question_decomposition_prompt
# - get_calculation_algorithm_recognition_prompt
# - get_knowledge_retrieval_prompt
# - get_calculation_decomposition_prompt
# - get_algorithm_validation_prompt
# - get_initial_inference_prompt
# - get_complete_inference_prompt
# - get_answer_generation_prompt
# - get_result_validation_prompt
# - get_exception_handling_prompt
# - get_manual_intervention_prompt


def get_domain_tools() -> List[str]:
    """Return priority tools for {domain_lower} domain"""
    return DOMAIN_CONFIG["priority_tools"]


def get_domain_extraction_rules() -> str:
    """Return domain-specific extraction rules for cross-domain merging"""
    return """
# TODO: Add domain-specific extraction rules that can be merged with other domains
"""
'''


def generate_domain_prompt(
    domain: str,
    entities: str,
    tools: str,
    calculation_focus: str = ""
):
    """生成领域Prompt文件"""
    domain_lower = domain.lower().replace(" ", "_")
    entities_list = [f'"{e.strip()}"' for e in entities.split(",")]
    tools_list = [f'"{t.strip()}"' for t in tools.split(",")]
    calculation_focus_list = [f'"{c.strip()}"' for c in calculation_focus.split(",")] if calculation_focus else []
    
    content = TEMPLATE.format(
        domain=domain,
        domain_lower=domain_lower,
        entities_list=entities_list,
        tools_list=tools_list,
        calculation_focus_list=calculation_focus_list
    )
    
    # 写入文件
    prompts_dir = Path(__file__).parent.parent / "agent" / "nodes" / "subagents" / "general_qa" / "prompts"
    output_file = prompts_dir / f"prompt-{domain_lower}.py"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✅ Generated: {output_file}")
    print(f"⚠️  TODO: Implement all 12 node prompt functions")
    print(f"⚠️  TODO: Add domain-specific rules and validation criteria")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate domain prompt template")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., Immunology)")
    parser.add_argument("--entities", required=True, help="Comma-separated entity list")
    parser.add_argument("--tools", required=True, help="Comma-separated tool list")
    parser.add_argument("--calculation-focus", default="", help="Comma-separated calculation focus list")
    
    args = parser.parse_args()
    generate_domain_prompt(args.domain, args.entities, args.tools, args.calculation_focus)
```

### 7.2 领域模块验收清单

创建 `DOMAIN_MODULE_CHECKLIST.md`：

```markdown
# 领域模块验收清单

新增领域模块必须满足以下要求才能合并：

## 1. 完整性要求 ✅

- [ ] 实现所有12个节点的Prompt函数
- [ ] 包含 `DOMAIN_CONFIG` 配置（工具列表、实体列表、验证标准）
- [ ] 实现 `get_domain_tools()` 函数
- [ ] 实现 `get_domain_extraction_rules()` 函数（用于跨领域合并）

## 2. 质量要求 ✅

- [ ] 包含领域专属规则（至少3个节点的详细增强）
- [ ] 包含计算类问题的领域特定规则（如适用）
- [ ] 包含验证标准（validation_criteria）
- [ ] 所有Prompt使用英文

## 3. 测试要求 ✅

- [ ] 配套≥5个测试用例（覆盖不同问题类型）
- [ ] 测试用例覆盖该领域的主要场景
- [ ] 通过单元测试（测试Prompt函数）
- [ ] 通过集成测试（端到端测试）

## 4. 文档要求 ✅

- [ ] 在 `CORE_DOMAINS_DETAILS.md` 中添加领域详细说明（如为核心领域）
- [ ] 更新 `domain_mapper.py` 中的领域映射
- [ ] 更新 `tool_trigger.py` 中的领域工具映射

## 验收流程

1. 代码审查：检查是否满足完整性要求
2. 功能测试：运行测试用例，验证功能正确性
3. 质量检查：检查Prompt质量和领域适配性
4. 性能测试：验证不影响系统性能
5. 合并批准：满足所有要求后合并
```

## 八、注意事项

1. **向后兼容**：保持API兼容，现有调用无需修改
2. **默认行为**：未匹配领域时使用general，确保系统稳定
3. **性能考虑**：使用模块缓存避免重复导入
4. **错误处理**：领域模块不存在时优雅降级到general
5. **跨领域处理**：自动检测并融合多领域规则
6. **计算类优化**：统一模板+领域特定规则
7. **工具调用规则**：明确节点工具使用规则，避免无效调用

