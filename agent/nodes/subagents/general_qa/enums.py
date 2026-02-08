"""
General QA 枚举类型定义

集中管理所有枚举类型，避免循环导入。
"""

from enum import Enum


class QuestionType(str, Enum):
    """问题类型枚举（优化后的分类）"""
    JUDGMENT = "判断型"  # 判断型：需判断"是否正确""是否偏倚""属于哪类"
    CALCULATION = "计算型"  # 计算型：需计算具体数值
    ANALYSIS = "分析型"  # 分析型：需分析因果关系/影响
    ENUMERATION = "枚举型"  # 枚举型：需列举多个答案
    UNKNOWN = "unknown"  # 未知类型
    
    # 保留旧类型作为兼容（向后兼容）
    CONCEPTUAL = "conceptual"  # 概念性问题（已废弃，使用ANALYSIS）
    EXPERIMENTAL = "experimental"  # 实验性问题（已废弃，使用ANALYSIS）
    COMPARISON = "comparison"  # 比较性问题（已废弃，使用JUDGMENT）
    CAUSAL = "causal"  # 因果关系问题（已废弃，使用ANALYSIS）
    DEFINITION = "definition"  # 定义性问题（已废弃，使用JUDGMENT）


class Domain(str, Enum):
    """领域枚举"""
    BIOLOGY = "biology"  # 生物学
    CHEMISTRY = "chemistry"  # 化学
    PHYSICS = "physics"  # 物理学
    MEDICINE = "medicine"  # 医学
    IMMUNOLOGY = "immunology"  # 免疫学
    MOLECULAR_BIOLOGY = "molecular_biology"  # 分子生物学
    GENERAL = "general"  # 通用领域
    UNKNOWN = "unknown"  # 未知领域


class ReasoningStrategy(str, Enum):
    """推理策略枚举"""
    DEDUCTIVE = "deductive"  # 演绎推理
    INDUCTIVE = "inductive"  # 归纳推理
    ABDUCTIVE = "abductive"  # 溯因推理
    ANALOGICAL = "analogical"  # 类比推理
    CAUSAL = "causal"  # 因果推理
    STATISTICAL = "statistical"  # 统计推理
    DIRECT = "direct"  # 直接回答

