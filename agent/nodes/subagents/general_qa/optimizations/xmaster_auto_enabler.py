"""
XMaster Auto-Enabler
根据问题复杂度自动决定是否启用XMaster多候选推理
"""

import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class XMasterConfig:
    """XMaster配置"""
    enabled: bool
    num_candidates: int
    num_critics: int
    num_rewriters: int
    timeout_multiplier: float  # 超时倍数
    complexity_level: str = "unknown"  # 复杂度级别标签
    reason: str = ""  # 决策原因


# 复杂度到XMaster配置的映射
COMPLEXITY_CONFIG = {
    ComplexityLevel.SIMPLE: XMasterConfig(
        enabled=False,
        num_candidates=1,
        num_critics=0,
        num_rewriters=0,
        timeout_multiplier=1.0,
        complexity_level="simple",
        reason="Simple question, XMaster not needed"
    ),
    ComplexityLevel.MODERATE: XMasterConfig(
        enabled=True,
        num_candidates=2,
        num_critics=2,
        num_rewriters=2,
        timeout_multiplier=1.3,
        complexity_level="moderate",
        reason="Moderate complexity, XMaster with 2 candidates"
    ),
    ComplexityLevel.COMPLEX: XMasterConfig(
        enabled=True,
        num_candidates=3,
        num_critics=3,
        num_rewriters=3,
        timeout_multiplier=1.6,
        complexity_level="complex",
        reason="Complex question, XMaster with 3 candidates"
    ),
    ComplexityLevel.VERY_COMPLEX: XMasterConfig(
        enabled=True,
        num_candidates=5,
        num_critics=5,
        num_rewriters=4,
        timeout_multiplier=2.0,
        complexity_level="very_complex",
        reason="Very complex question, XMaster with 5 candidates"
    )
}


# 复杂问题指示器
COMPLEXITY_INDICATORS = {
    "multi_step": [
        r"step[s]?\s+\d+",
        r"first[,\.].*then",
        r"after.*before",
        r"calculate.*explain",
        r"determine.*verify"
    ],
    "calculation": [
        r"calculate|compute|determine",
        r"what (is|are) the (ratio|percentage|probability)",
        r"how many|how much",
        r"frequency|probability|rate"
    ],
    "multi_domain": [
        r"gene|allele|genotype|phenotype",  # genetics
        r"protein|enzyme|pathway|metabolism",  # biochemistry
        r"cell|membrane|receptor|signal",  # cell biology
        r"diagnosis|treatment|patient|clinical"  # clinical
    ],
    "long_question": 500,  # 字符数阈值
    "many_options": 5  # 选项数阈值
}


def estimate_complexity(
    question: str,
    question_type: Optional[str] = None,
    options_count: int = 0
) -> ComplexityLevel:
    """
    估算问题复杂度
    
    Args:
        question: 问题文本
        question_type: 问题类型
        options_count: 选项数量
        
    Returns:
        ComplexityLevel
    """
    score = 0
    
    # 1. 长度因子
    if len(question) > COMPLEXITY_INDICATORS["long_question"]:
        score += 1
    if len(question) > 1000:
        score += 1
    if len(question) > 2000:
        score += 1
    
    # 2. 多步骤因子
    for pattern in COMPLEXITY_INDICATORS["multi_step"]:
        if re.search(pattern, question, re.IGNORECASE):
            score += 1
            break
    
    # 3. 计算因子
    for pattern in COMPLEXITY_INDICATORS["calculation"]:
        if re.search(pattern, question, re.IGNORECASE):
            score += 1
            break
    
    # 4. 多领域因子
    domain_count = 0
    for pattern in COMPLEXITY_INDICATORS["multi_domain"]:
        if re.search(pattern, question, re.IGNORECASE):
            domain_count += 1
    if domain_count >= 2:
        score += 1
    if domain_count >= 4:
        score += 1
    
    # 5. 选项数因子
    if options_count > COMPLEXITY_INDICATORS["many_options"]:
        score += 1
    if options_count > 8:
        score += 1
    
    # 6. 问题类型因子
    complex_types = ["genetics_genomics", "bioinformatics", "mhc_binding", "vdj_bcr_tcr"]
    if question_type and question_type in complex_types:
        score += 1
    
    # 映射到复杂度等级
    if score >= 6:
        return ComplexityLevel.VERY_COMPLEX
    elif score >= 4:
        return ComplexityLevel.COMPLEX
    elif score >= 2:
        return ComplexityLevel.MODERATE
    else:
        return ComplexityLevel.SIMPLE


def get_xmaster_config(
    question: str,
    question_type: Optional[str] = None,
    options_count: int = 0,
    force_enable: bool = False
) -> XMasterConfig:
    """
    获取XMaster配置
    
    Args:
        question: 问题文本
        question_type: 问题类型
        options_count: 选项数量
        force_enable: 强制启用
        
    Returns:
        XMasterConfig
    """
    if force_enable:
        return COMPLEXITY_CONFIG[ComplexityLevel.COMPLEX]
    
    complexity = estimate_complexity(question, question_type, options_count)
    return COMPLEXITY_CONFIG[complexity]


def should_enable_xmaster(
    question: str,
    question_type: Optional[str] = None,
    options_count: int = 0
) -> Tuple[bool, int]:
    """
    判断是否应该启用XMaster
    
    Returns:
        (enabled, num_candidates)
    """
    config = get_xmaster_config(question, question_type, options_count)
    return config.enabled, config.num_candidates


def count_options(question: str) -> int:
    """
    计算问题中的选项数量
    
    Args:
        question: 问题文本
        
    Returns:
        选项数量
    """
    # 方法1: 检测 "Answer Choices:" 后的选项
    if "Answer Choices:" in question:
        # 计算A., B., C. 等选项
        options_count = len(re.findall(r'\b[A-H]\.', question))
        if options_count > 0:
            return options_count
    
    # 方法2: 检测独立的选项行
    options_count = len(re.findall(r'^[A-H][\.\)]\s', question, re.MULTILINE))
    if options_count > 0:
        return options_count
    
    return 0


def integrate_with_general_qa(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    将XMaster自动启用逻辑集成到general_qa状态
    
    Args:
        state: 当前状态字典
        
    Returns:
        更新后的状态字典
    """
    question = state.get("user_input", "") or state.get("question_text", "")
    question_type = state.get("question_type")
    
    # 检测选项数量
    options_count = count_options(question)
    
    # 获取XMaster配置
    config = get_xmaster_config(question, question_type, options_count)
    
    # 更新状态
    state["xmasters_enabled"] = config.enabled
    state["num_candidates"] = config.num_candidates
    state["num_critics"] = config.num_critics
    state["num_rewriters"] = config.num_rewriters
    
    # 调整超时
    if "timeout" in state and state["timeout"]:
        original_timeout = state["timeout"]
        state["timeout"] = original_timeout * config.timeout_multiplier
        logger.info(
            f"XMaster config: enabled={config.enabled}, "
            f"candidates={config.num_candidates}, "
            f"timeout adjusted: {original_timeout}s -> {state['timeout']}s"
        )
    
    return state


class XMasterAutoEnabler:
    """XMaster自动启用管理器"""
    
    def __init__(self):
        self._config_history: list = []
    
    def configure_for_question(
        self,
        question: str,
        question_type: Optional[str] = None
    ) -> XMasterConfig:
        """为问题配置XMaster"""
        options_count = count_options(question)
        config = get_xmaster_config(question, question_type, options_count)
        
        self._config_history.append({
            "question_length": len(question),
            "question_type": question_type,
            "options_count": options_count,
            "config": config
        })
        
        return config
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._config_history:
            return {"total": 0}
        
        enabled_count = sum(1 for c in self._config_history if c["config"].enabled)
        avg_candidates = sum(
            c["config"].num_candidates for c in self._config_history
        ) / len(self._config_history)
        
        return {
            "total_questions": len(self._config_history),
            "xmaster_enabled_count": enabled_count,
            "enable_rate": enabled_count / len(self._config_history),
            "average_candidates": avg_candidates
        }

