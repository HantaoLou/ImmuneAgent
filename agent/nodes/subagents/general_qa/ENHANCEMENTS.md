# General QA Prompt 优化方案 - 关键补充细节

本文档补充优化方案中的5个关键细节，确保方案完整可落地。

## 一、领域识别环节：精准识别+容错机制

### 1.1 领域识别逻辑

在 `prompts/domain_mapper.py` 中补充领域识别逻辑：

```python
"""
Domain Mapper for Prompt Routing
Enhanced with domain identification and confidence scoring
"""

from typing import Optional, Dict, Any, List, Tuple
import importlib
import re
from collections import Counter

# 领域核心关键词映射（从配置加载或硬编码）
DOMAIN_KEYWORDS = {
    "Genetics": [
        "allele", "genotype", "phenotype", "locus", "haplotype",
        "SNP", "variant", "mutation", "inheritance", "Hardy-Weinberg",
        "HWE", "Fst", "theta", "pi", "recombination", "linkage",
        "Mendelian", "autosomal", "X-linked", "dominant", "recessive"
    ],
    "Immunology": [
        "T cell", "B cell", "NK cell", "macrophage", "dendritic cell",
        "TCR", "BCR", "MHC", "antigen", "antibody", "cytokine",
        "allelic exclusion", "allelic inclusion", "positive selection",
        "negative selection", "V(D)J", "recombination", "CD4", "CD8"
    ],
    "Clinical Medicine": [
        "hypertension", "diabetes", "medication", "drug", "treatment",
        "diagnosis", "patient", "symptom", "disease", "therapy",
        "JNC8", "ADA", "guideline", "contraindication", "dosage",
        "antihypertensive", "blood pressure", "glucose", "insulin"
    ],
    "Bioinformatics": [
        "theta", "pi", "Fst", "Watterson", "nucleotide diversity",
        "variant calling", "phasing", "VCF", "FASTA", "BAM",
        "chi-square", "statistical test", "population genetics",
        "sequencing", "alignment", "quality score", "imputation"
    ],
    "Biochemistry": [
        "concentration", "molecular weight", "enzyme", "substrate",
        "reaction", "equilibrium", "binding", "affinity", "kinetics",
        "protein", "ligand", "receptor", "pathway", "metabolism"
    ],
    "Molecular Biology": [
        "transcription", "translation", "DNA", "RNA", "protein",
        "gene expression", "promoter", "enhancer", "splicing",
        "mutation", "replication", "repair", "recombination"
    ],
    # ... 其他领域
}

# 领域识别置信度阈值
DOMAIN_CONFIDENCE_THRESHOLD = 0.15  # 低于此阈值使用general
CROSS_DOMAIN_THRESHOLD = 0.20  # 跨领域识别阈值（至少2个领域超过此阈值）


def extract_keywords(text: str) -> List[str]:
    """从文本中提取关键词（支持大小写不敏感）"""
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


# 缓存领域识别结果（5分钟有效期）
_domain_cache: Dict[str, Tuple[Optional[str], float, float]] = {}  # key: (input_hash, question_type), value: (domain, confidence, timestamp)
import time

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
        _domain_cache = {k: v for k, v in _domain_cache.items() if current_time - v[2] < 300}
```

### 1.2 在domain_mapper.py中集成识别逻辑

```python
# 在原有的domain_mapper.py中补充

def get_prompt_module(
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    user_input: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> Any:
    """
    Get the appropriate prompt module with domain identification
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
        is_cross, domains = detect_cross_domain(user_input, question_type, core_domains)
        if is_cross:
            # 返回跨领域模块
            try:
                module = importlib.import_module("agent.nodes.subagents.general_qa.prompts.prompt_cross_domain")
                module._domains = domains  # 设置涉及的领域
                return module
            except ImportError:
                pass
    
    # 原有逻辑...
    key = question_type or domain
    
    if key in _module_cache:
        return _module_cache[key]
    
    module_name = DOMAIN_MAPPING.get(key, "general")
    
    try:
        module = importlib.import_module(f"agent.nodes.subagents.general_qa.prompts.prompt_{module_name}")
        _module_cache[key] = module
        return module
    except ImportError:
        module = importlib.import_module("agent.nodes.subagents.general_qa.prompts.prompt_general")
        _module_cache[key] = module
        return module
```

## 二、性能优化：懒加载+缓存机制

### 2.1 模块懒加载

```python
# prompts/domain_mapper.py 中实现懒加载

_module_cache: Dict[str, Any] = {}
_module_loading_lock = threading.Lock()  # 线程安全

def get_prompt_module_lazy(domain: str) -> Any:
    """懒加载领域模块（仅在首次调用时加载）"""
    if domain in _module_cache:
        return _module_cache[domain]
    
    with _module_loading_lock:
        # 双重检查（避免并发加载）
        if domain in _module_cache:
            return _module_cache[domain]
        
        # 加载模块
        module_name = DOMAIN_MAPPING.get(domain, "general")
        try:
            module = importlib.import_module(f"agent.nodes.subagents.general_qa.prompts.prompt_{module_name}")
            _module_cache[domain] = module
            return module
        except ImportError:
            module = importlib.import_module("agent.nodes.subagents.general_qa.prompts.prompt_general")
            _module_cache[domain] = module
            return module
```

### 2.2 跨领域规则预编译

```python
# prompts/prompt-cross_domain.py 中实现规则预编译

# 高频跨领域组合（预编译）
PRECOMPILED_CROSS_DOMAINS = {
    ("Genetics", "Bioinformatics"): {
        "extraction_rules": """
        **Genetics+Bioinformatics Combined Rules:**
        - Extract genetic variants and computational parameters
        - Identify population genetics formulas and statistical tests
        """,
        "tools": ["query_gwas_catalog", "query_genebass", "query_variant", "query_knowledge_graph"],
        "compiled_at": None  # 首次使用时编译
    },
    ("Immunology", "Clinical Medicine"): {
        "extraction_rules": """
        **Immunology+Clinical Medicine Combined Rules:**
        - Extract immune cell types and clinical treatment information
        - Identify immunotherapy and clinical decision criteria
        """,
        "tools": ["query_tcr_mcpas", "query_drug_for_disease", "query_celltype_marker"],
        "compiled_at": None
    }
}

def get_precompiled_rules(domains: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """获取预编译的跨领域规则"""
    domain_key = tuple(sorted(domains))  # 排序后作为key
    if domain_key in PRECOMPILED_CROSS_DOMAINS:
        rule = PRECOMPILED_CROSS_DOMAINS[domain_key]
        if rule["compiled_at"] is None:
            # 首次使用时编译（合并各领域的规则）
            rule["compiled_at"] = time.time()
            # 可以在这里进行规则优化、去重等操作
        return rule
    return None
```

### 2.3 工具映射缓存

```python
# tools/tool_loader.py 中实现工具映射缓存

_tool_mapping_cache: Dict[Tuple[str, Optional[str]], List[str]] = {}  # key: (node_name, domain), value: tool_names
_tool_cache_timestamp: Dict[Tuple[str, Optional[str]], float] = {}
CACHE_TTL = 300  # 5分钟

def get_cached_tools(node_name: str, domain: Optional[str] = None) -> Optional[List[str]]:
    """获取缓存的工具列表"""
    cache_key = (node_name, domain)
    if cache_key in _tool_mapping_cache:
        if time.time() - _tool_cache_timestamp.get(cache_key, 0) < CACHE_TTL:
            return _tool_mapping_cache[cache_key]
        else:
            del _tool_mapping_cache[cache_key]
            del _tool_cache_timestamp[cache_key]
    return None


def cache_tools(node_name: str, domain: Optional[str], tools: List[str]):
    """缓存工具列表"""
    cache_key = (node_name, domain)
    _tool_mapping_cache[cache_key] = tools
    _tool_cache_timestamp[cache_key] = time.time()
```

## 三、工具调用：优先级+容错策略

### 3.1 工具优先级配置

```python
# prompts/prompt-genetics.py 等各领域模块中

DOMAIN_CONFIG = {
    "name": "Genetics",
    "priority_tools": [
        "query_gwas_catalog",      # 优先级1（最高）
        "query_genebass",          # 优先级2
        "query_variant",           # 优先级3
        "query_omim",              # 优先级4
        "query_disgenet",          # 优先级5
        "query_gene_info"          # 优先级6
    ],
    "tool_priority": {  # 明确优先级映射
        "query_gwas_catalog": 1,
        "query_genebass": 2,
        "query_variant": 3,
        "query_omim": 4,
        "query_disgenet": 5,
        "query_gene_info": 6
    },
    "fallback_tools": {  # 工具失败时的降级工具
        "query_gwas_catalog": "query_variant",  # GWAS失败时使用variant
        "query_genebass": "query_gwas_catalog",  # Genebass失败时使用GWAS
    },
    # ... 其他配置
}
```

### 3.2 工具调用容错逻辑

```python
# tools/tool_loader.py 中实现工具容错

def get_tools_for_node_with_fallback(
    node_name: str,
    domain: Optional[str] = None,
    question_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    获取工具列表（包含优先级和容错信息）
    
    Returns:
        List of tool dicts with metadata: [{"tool": StructuredTool, "priority": int, "fallback": str, "retry": int}, ...]
    """
    # 获取基础工具列表
    tools = get_tools_for_node(node_name, domain, question_type)
    
    # 获取领域工具优先级
    tool_metadata = []
    if domain:
        domain_module = get_prompt_module(domain=domain)
        if hasattr(domain_module, 'DOMAIN_CONFIG'):
            config = domain_module.DOMAIN_CONFIG
            tool_priority = config.get("tool_priority", {})
            fallback_tools = config.get("fallback_tools", {})
            
            for tool in tools:
                priority = tool_priority.get(tool.name, 999)
                fallback = fallback_tools.get(tool.name)
                tool_metadata.append({
                    "tool": tool,
                    "priority": priority,
                    "fallback": fallback,
                    "retry_times": 1,  # 失败重试1次
                    "max_retries": 2   # 最多重试2次
                })
            
            # 按优先级排序
            tool_metadata.sort(key=lambda x: x["priority"])
            return tool_metadata
    
    # 无领域信息时，默认配置
    for tool in tools:
        tool_metadata.append({
            "tool": tool,
            "priority": 999,
            "fallback": None,
            "retry_times": 1,
            "max_retries": 2
        })
    
    return tool_metadata


# graph.py 中调用工具时使用容错逻辑
def _call_tool_with_fallback(tool_metadata: Dict[str, Any], tool_args: Dict[str, Any]) -> Any:
    """调用工具，支持重试和降级"""
    tool = tool_metadata["tool"]
    retry_times = tool_metadata.get("retry_times", 1)
    fallback = tool_metadata.get("fallback")
    
    for attempt in range(retry_times + 1):
        try:
            result = tool.invoke(tool_args)
            return result, None  # (result, error)
        except Exception as e:
            if attempt < retry_times:
                continue  # 重试
            else:
                # 尝试降级工具
                if fallback:
                    fallback_tool = get_tool_by_name(fallback)
                    if fallback_tool:
                        try:
                            result = fallback_tool.invoke(tool_args)
                            return result, f"Used fallback tool {fallback} after {tool.name} failed"
                        except:
                            pass
                return None, str(e)
    
    return None, "Tool call failed after retries"
```

## 四、代码复用：基于类的抽象封装

### 4.1 基类设计

```python
# prompts/base.py 中实现基类

from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseDomainPrompt(ABC):
    """所有领域Prompt的基类，实现通用逻辑"""
    
    # 子类必须定义的配置
    DOMAIN_CONFIG: Dict[str, Any] = {}
    
    def __init__(self):
        """初始化领域配置"""
        if not self.DOMAIN_CONFIG:
            raise ValueError(f"{self.__class__.__name__} must define DOMAIN_CONFIG")
    
    def get_base_prompt(self, node_name: str, *args, **kwargs) -> str:
        """加载节点基础Prompt模板"""
        base_functions = {
            "n0_input_preprocessing": get_base_input_preprocessing_prompt,
            "n1_question_decomposition": get_base_question_decomposition_prompt,
            "n2_calculation_algorithm_recognition": get_base_calculation_algorithm_recognition_prompt,
            "n3_knowledge_retrieval": get_base_knowledge_retrieval_prompt,
            "n4_calculation_decomposition": get_base_calculation_decomposition_prompt,
            "n5_algorithm_validation": get_base_algorithm_validation_prompt,
            "n6_initial_inference": get_base_initial_inference_prompt,
            "n7_complete_inference": get_base_complete_inference_prompt,
            "n8_answer_generation": get_base_answer_generation_prompt,
            "n9_result_validation": get_base_result_validation_prompt,
            "n10_exception_handling": get_base_exception_handling_prompt,
            "n11_manual_intervention": get_base_manual_intervention_prompt,
        }
        
        func = base_functions.get(node_name)
        if not func:
            raise ValueError(f"Unknown node: {node_name}")
        
        return func(*args, **kwargs)
    
    @abstractmethod
    def get_domain_enhancements(self, node_name: str, *args, **kwargs) -> str:
        """获取领域特定增强（子类必须实现）"""
        pass
    
    def get_domain_extraction_rules(self) -> str:
        """获取领域提取规则（用于跨领域合并）"""
        return self.DOMAIN_CONFIG.get("extraction_rules", "")
    
    def get_domain_tools(self) -> List[str]:
        """获取领域工具列表"""
        return self.DOMAIN_CONFIG.get("priority_tools", [])
    
    # 通用节点Prompt函数（所有领域复用）
    def get_input_preprocessing_prompt(self, user_input: str) -> str:
        """N0: Input Preprocessing"""
        base_prompt = self.get_base_prompt("n0_input_preprocessing", user_input)
        enhancements = self.get_domain_enhancements("n0_input_preprocessing", user_input=user_input)
        return base_prompt + enhancements
    
    def get_question_decomposition_prompt(
        self,
        cleaned_text: str,
        question_type_label: str,
        structured_subject: Dict[str, Any] = None,
        structured_condition: Dict[str, Any] = None,
        structured_goal: Dict[str, Any] = None,
        question_category_standard: str = None,
        category_specific_constraints: List[str] = None
    ) -> str:
        """N1: Question Decomposition"""
        base_prompt = self.get_base_prompt(
            "n1_question_decomposition",
            cleaned_text, question_type_label, structured_subject,
            structured_condition, structured_goal, question_category_standard,
            category_specific_constraints
        )
        enhancements = self.get_domain_enhancements(
            "n1_question_decomposition",
            cleaned_text=cleaned_text,
            question_type_label=question_type_label
        )
        return base_prompt + enhancements
    
    # ... 其他10个节点的通用函数（类似实现）
```

### 4.2 子类实现示例

```python
# prompts/prompt-genetics.py 使用基类

from .base import BaseDomainPrompt, get_calculation_guide

class GeneticsPrompt(BaseDomainPrompt):
    """Genetics领域Prompt实现"""
    
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
        "tool_priority": {
            "query_gwas_catalog": 1,
            "query_genebass": 2,
            "query_variant": 3,
            "query_omim": 4,
            "query_disgenet": 5,
            "query_gene_info": 6
        },
        "extraction_rules": """
        **Genetics-Specific Extraction Rules:**
        1. Inheritance Pattern Recognition
        2. Genotype/Phenotype Extraction
        3. Population Genetics Parameters
        """,
        # ... 其他配置
    }
    
    def get_domain_enhancements(self, node_name: str, **kwargs) -> str:
        """实现领域特定增强"""
        if node_name == "n0_input_preprocessing":
            return self._get_n0_enhancements(kwargs.get("user_input", ""))
        elif node_name == "n1_question_decomposition":
            return self._get_n1_enhancements()
        elif node_name == "n4_calculation_decomposition":
            return self._get_n4_enhancements()
        # ... 其他节点
        return ""
    
    def _get_n0_enhancements(self, user_input: str) -> str:
        """N0节点增强"""
        return """
**Genetics-Specific Extraction Rules:**
1. **Inheritance Pattern Recognition**: Identify and extract inheritance patterns explicitly
2. **Genotype/Phenotype Extraction**: Extract genotype and phenotype relationships
3. **Population Genetics Parameters**: Identify population genetics concepts
"""
    
    def _get_n1_enhancements(self) -> str:
        """N1节点增强"""
        return """
**Genetics-Specific Decomposition Patterns:**
1. **Inheritance Pattern Questions**: Identify pattern → Apply principles → Calculate probabilities
2. **Population Genetics Questions**: Extract parameters → Apply formulas → Verify assumptions
"""
    
    def _get_n4_enhancements(self) -> str:
        """N4节点增强（引入计算类模板）"""
        calculation_guide = get_calculation_guide()
        return f"""
{calculation_guide}

**Genetics-Specific Calculation Rules:**
- Hardy-Weinberg calculations
- Population genetics parameters
- Genetic probability calculations
"""


# 创建模块级实例（保持向后兼容）
_genetics_prompt = GeneticsPrompt()

# 导出函数（保持原有API）
def get_input_preprocessing_prompt(user_input: str) -> str:
    return _genetics_prompt.get_input_preprocessing_prompt(user_input)

def get_question_decomposition_prompt(*args, **kwargs) -> str:
    return _genetics_prompt.get_question_decomposition_prompt(*args, **kwargs)

# ... 其他函数
```

## 五、测试落地：自动化测试与指标计算

### 5.1 测试脚本模板

```python
# tests/test_domain_prompt.py

"""
领域Prompt自动化测试脚本
自动运行各领域测试用例并输出指标
"""

import json
import time
from typing import Dict, List, Tuple
from pathlib import Path

# 测试数据集路径
TEST_DATA_PATH = Path("agent/tests/csv_questions_data.json")

# 正确答案判定标准
class AnswerValidator:
    """答案验证器"""
    
    @staticmethod
    def is_correct(predicted: str, expected: str, question_type: str) -> bool:
        """
        判定答案是否正确
        
        Args:
            predicted: 预测答案
            expected: 期望答案
            question_type: 问题类型
        
        Returns:
            bool: 是否正确
        """
        predicted = predicted.strip().upper()
        expected = expected.strip().upper()
        
        # 多选题：检查是否包含所有正确答案
        if question_type == "Multi-Select":
            predicted_set = set(predicted.split(","))
            expected_set = set(expected.split(","))
            return predicted_set == expected_set
        
        # 单选题：精确匹配或语义匹配
        if question_type == "Single Choice":
            # 精确匹配
            if predicted == expected:
                return True
            # 语义匹配（如"A" vs "A. option text"）
            if predicted in expected or expected in predicted:
                return True
            return False
        
        # 数值题：允许一定误差
        if question_type == "Numeric":
            try:
                pred_num = float(predicted)
                exp_num = float(expected)
                # 允许5%误差
                return abs(pred_num - exp_num) / max(abs(exp_num), 1e-10) < 0.05
            except:
                return predicted == expected
        
        # 文本匹配：包含核心关键词
        if question_type == "Text Matching":
            # 提取核心关键词
            pred_keywords = set(predicted.lower().split())
            exp_keywords = set(expected.lower().split())
            # 至少80%关键词匹配
            if len(exp_keywords) == 0:
                return False
            match_ratio = len(pred_keywords & exp_keywords) / len(exp_keywords)
            return match_ratio >= 0.8
        
        # 默认：精确匹配
        return predicted == expected


class DomainPromptTester:
    """领域Prompt测试器"""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.test_cases = []
        self.results = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "tool_calls": {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "unnecessary": 0
            },
            "latency": []
        }
    
    def load_test_cases(self, test_data_path: Path) -> List[Dict]:
        """加载测试用例"""
        with open(test_data_path, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
        
        # 筛选该领域的题目
        domain_questions = [
            q for q in all_questions
            if q.get("raw_subject") == self.domain or 
               q.get("question_type") in self._get_domain_question_types()
        ]
        
        return domain_questions
    
    def _get_domain_question_types(self) -> List[str]:
        """获取领域对应的问题类型"""
        mapping = {
            "Genetics": ["genetics_genomics"],
            "Immunology": ["vdj_bcr_tcr", "immune_cells", "antibody", "mhc_binding"],
            "Clinical Medicine": ["clinical_medicine"],
            "Bioinformatics": ["bioinformatics"],
        }
        return mapping.get(self.domain, [])
    
    def run_test(self, test_case: Dict) -> Dict:
        """运行单个测试用例"""
        start_time = time.time()
        
        # 调用General QA系统
        from agent.nodes.subagents.general_qa.graph import create_general_qa_graph
        graph = create_general_qa_graph()
        
        # 执行问题回答
        result = graph.invoke({"user_input": test_case["question"]})
        
        latency = time.time() - start_time
        self.results["latency"].append(latency)
        
        # 验证答案
        predicted_answer = result.get("final_answer", "")
        expected_answer = test_case.get("answer", "")
        question_type = test_case.get("answer_type", "Single Choice")
        
        is_correct = AnswerValidator.is_correct(
            predicted_answer, expected_answer, question_type
        )
        
        # 统计工具调用
        tool_calls = result.get("tool_calls_history", [])
        tool_stats = self._analyze_tool_calls(tool_calls, test_case)
        
        return {
            "test_case_id": test_case.get("id"),
            "predicted": predicted_answer,
            "expected": expected_answer,
            "correct": is_correct,
            "latency": latency,
            "tool_stats": tool_stats
        }
    
    def _analyze_tool_calls(self, tool_calls: List[Dict], test_case: Dict) -> Dict:
        """分析工具调用"""
        stats = {
            "total": len(tool_calls),
            "correct": 0,
            "incorrect": 0,
            "unnecessary": 0
        }
        
        # 判断工具调用是否正确（简化实现）
        domain_tools = self._get_expected_tools()
        for call in tool_calls:
            tool_name = call.get("tool_name", "")
            if tool_name in domain_tools:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1
        
        # 判断是否有不必要的工具调用（如N0/N1节点调用工具）
        for call in tool_calls:
            node_name = call.get("node", "")
            if node_name in ["n0_input_preprocessing", "n1_question_decomposition"]:
                stats["unnecessary"] += 1
        
        return stats
    
    def _get_expected_tools(self) -> List[str]:
        """获取期望的工具列表"""
        from agent.nodes.subagents.general_qa.prompts.domain_mapper import get_prompt_module
        module = get_prompt_module(domain=self.domain)
        if hasattr(module, 'get_domain_tools'):
            return module.get_domain_tools()
        return []
    
    def run_all_tests(self, test_cases: List[Dict], max_tests: int = None) -> Dict:
        """运行所有测试用例"""
        if max_tests:
            test_cases = test_cases[:max_tests]
        
        self.results["total"] = len(test_cases)
        
        for test_case in test_cases:
            result = self.run_test(test_case)
            if result["correct"]:
                self.results["correct"] += 1
            else:
                self.results["incorrect"] += 1
            
            # 累计工具调用统计
            tool_stats = result["tool_stats"]
            self.results["tool_calls"]["total"] += tool_stats["total"]
            self.results["tool_calls"]["correct"] += tool_stats["correct"]
            self.results["tool_calls"]["incorrect"] += tool_stats["incorrect"]
            self.results["tool_calls"]["unnecessary"] += tool_stats["unnecessary"]
        
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict:
        """计算指标"""
        total = self.results["total"]
        if total == 0:
            return {}
        
        accuracy = self.results["correct"] / total
        
        tool_total = self.results["tool_calls"]["total"]
        tool_correct = self.results["tool_calls"]["correct"]
        tool_accuracy = tool_correct / tool_total if tool_total > 0 else 0.0
        
        tool_unnecessary = self.results["tool_calls"]["unnecessary"]
        unnecessary_rate = tool_unnecessary / tool_total if tool_total > 0 else 0.0
        
        avg_latency = sum(self.results["latency"]) / len(self.results["latency"]) if self.results["latency"] else 0.0
        
        return {
            "domain": self.domain,
            "accuracy": accuracy,
            "tool_accuracy": tool_accuracy,
            "unnecessary_tool_rate": unnecessary_rate,
            "avg_latency": avg_latency,
            "total_tests": total,
            "correct_tests": self.results["correct"],
            "tool_stats": self.results["tool_calls"]
        }


def run_all_domain_tests(domains: List[str] = None, max_tests_per_domain: int = 10) -> Dict:
    """运行所有领域的测试"""
    if domains is None:
        domains = ["Genetics", "Immunology", "Clinical Medicine", "Bioinformatics"]
    
    all_results = {}
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Testing domain: {domain}")
        print(f"{'='*60}")
        
        tester = DomainPromptTester(domain)
        test_cases = tester.load_test_cases(TEST_DATA_PATH)
        
        if not test_cases:
            print(f"⚠️  No test cases found for {domain}")
            continue
        
        print(f"Found {len(test_cases)} test cases, running {min(max_tests_per_domain, len(test_cases))}...")
        
        results = tester.run_all_tests(test_cases, max_tests_per_domain)
        all_results[domain] = results
        
        # 打印结果
        print(f"\nResults for {domain}:")
        print(f"  Accuracy: {results['accuracy']:.2%}")
        print(f"  Tool Accuracy: {results['tool_accuracy']:.2%}")
        print(f"  Unnecessary Tool Rate: {results['unnecessary_tool_rate']:.2%}")
        print(f"  Avg Latency: {results['avg_latency']:.3f}s")
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    overall_accuracy = sum(r["accuracy"] for r in all_results.values()) / len(all_results) if all_results else 0.0
    overall_tool_accuracy = sum(r["tool_accuracy"] for r in all_results.values()) / len(all_results) if all_results else 0.0
    
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Overall Tool Accuracy: {overall_tool_accuracy:.2%}")
    
    # 检查是否达到目标
    print(f"\nTarget Check:")
    print(f"  Accuracy ≥ 90%: {'✅' if overall_accuracy >= 0.90 else '❌'} ({overall_accuracy:.2%})")
    print(f"  Tool Accuracy ≥ 95%: {'✅' if overall_tool_accuracy >= 0.95 else '❌'} ({overall_tool_accuracy:.2%})")
    
    return all_results


if __name__ == "__main__":
    # 运行测试
    results = run_all_domain_tests(max_tests_per_domain=10)
    
    # 保存结果
    output_path = Path("agent/tests/outputs/domain_prompt_test_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Test results saved to: {output_path}")
```

### 5.2 指标计算说明

```markdown
# 测试指标计算说明

## 1. 准确率 (Accuracy)
- **公式**: accuracy = correct_tests / total_tests
- **目标**: ≥ 90%
- **说明**: 核心领域问题答案正确率

## 2. 工具调用准确率 (Tool Accuracy)
- **公式**: tool_accuracy = correct_tool_calls / total_tool_calls
- **目标**: ≥ 95%
- **说明**: 正确识别需要调用的工具的比例

## 3. 无效工具调用率 (Unnecessary Tool Rate)
- **公式**: unnecessary_rate = unnecessary_tool_calls / total_tool_calls
- **目标**: ≤ 5%
- **说明**: 不应调用工具时调用了工具的比例（如N0/N1节点）

## 4. 平均延迟 (Avg Latency)
- **公式**: avg_latency = sum(latencies) / count
- **目标**: ≤ 500ms（缓存生效后）
- **说明**: 单问题处理平均延迟

## 5. 跨领域问题解决率 (Cross-Domain Success Rate)
- **公式**: cross_domain_success = correct_cross_domain / total_cross_domain
- **目标**: ≥ 80%
- **说明**: 跨领域问题的解决成功率
```

## 六、总结

本文档补充了优化方案中的5个关键细节：

1. ✅ **领域识别**：关键词匹配+置信度阈值+缓存机制
2. ✅ **性能优化**：懒加载+规则预编译+工具映射缓存
3. ✅ **工具容错**：优先级排序+重试机制+降级策略
4. ✅ **代码复用**：基类抽象+子类实现，最大化复用
5. ✅ **测试落地**：自动化测试脚本+指标计算+结果验证

这些补充确保了优化方案的完整性和可落地性。

