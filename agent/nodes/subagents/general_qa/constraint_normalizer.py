"""
通用约束关键词归一化模块

提供领域无关的约束关键词归一化功能，将自然语言约束转换为标准化的约束关键词。
所有知识点复用同一套约束关键词词典，实现通用化匹配。
"""

from typing import Dict, List, Set, Optional
import re


# ===================== 通用约束关键词词典 =====================
# 全局维护，所有知识点复用，新增约束时仅需更新此词典

CONSTRAINT_KEYWORD_DICT: Dict[str, str] = {
    # 数据完整性约束
    "无全局缺失": "no_global_missing",
    "无完全缺失": "no_global_missing",
    "无全局缺失的SNP": "no_global_missing",
    "无完全缺失的位点": "no_global_missing",
    "有全局缺失": "has_global_missing",
    "存在全局缺失": "has_global_missing",
    "存在完全缺失": "has_global_missing",
    
    # 样本量约束
    "大样本": "sample_size_large",
    "样本量任意大": "sample_size_large",
    "样本量足够大": "sample_size_large",
    "大样本量": "sample_size_large",
    "小样本": "sample_size_small",
    "样本量小": "sample_size_small",
    "样本量未知": "sample_size_unknown",
    
    # 过滤操作约束
    "随机过滤": "filtering_random",
    "随机SNV过滤": "filtering_random",
    "随机筛选": "filtering_random",
    "系统过滤": "filtering_systematic",
    "系统性过滤": "filtering_systematic",
    "少数等位基因过滤": "filtering_minority",
    "minority filtering": "filtering_minority",
    "任意过滤": "filtering_any",
    "有过滤操作": "filtering_any",
    
    # 插补操作约束
    "参考序列插补": "imputation_reference",
    "reference imputation": "imputation_reference",
    "样本特异性插补": "imputation_sample_specific",
    "sample-specific imputation": "imputation_sample_specific",
    "有插补操作": "imputation_any",
    
    # 组合约束（用+连接表示同时满足）
    "filtering_random + filtering_minority": "filtering_random + filtering_minority",
    "imputation_reference + imputation_sample_specific": "imputation_reference + imputation_sample_specific",
    
    # 其他通用约束（可根据需要扩展）
    "温度37度": "temperature_37c",
    "37°C": "temperature_37c",
    "标准条件": "standard_conditions",
}


def normalize_constraint(constraint_text: str) -> str:
    """
    将单个自然语言约束文本归一化为标准约束关键词
    
    Args:
        constraint_text: 自然语言约束文本（如"无全局缺失的SNP"）
    
    Returns:
        标准化的约束关键词（如"no_global_missing"）
    """
    if not constraint_text or not constraint_text.strip():
        return ""
    
    # 去除首尾空格并转为小写（部分匹配）
    normalized = constraint_text.strip()
    
    # 直接查找词典
    if normalized in CONSTRAINT_KEYWORD_DICT:
        return CONSTRAINT_KEYWORD_DICT[normalized]
    
    # 模糊匹配：检查是否包含词典中的键
    normalized_lower = normalized.lower()
    for key, value in CONSTRAINT_KEYWORD_DICT.items():
        if key.lower() in normalized_lower or normalized_lower in key.lower():
            return value
    
    # 如果未找到匹配，返回原文本（保留原始约束，后续可扩展词典）
    return normalized


def normalize_constraint_list(constraints: List[str]) -> List[str]:
    """
    将约束列表归一化为标准约束关键词列表
    
    Args:
        constraints: 自然语言约束列表
    
    Returns:
        标准化后的约束关键词列表（去重）
    """
    if not constraints:
        return []
    
    normalized_set: Set[str] = set()
    for constraint in constraints:
        normalized = normalize_constraint(constraint)
        if normalized:
            normalized_set.add(normalized)
    
    return list(normalized_set)


def normalize_constraint_hierarchy(constraint_hierarchy: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    归一化约束分层结构（C1/C2）
    
    Args:
        constraint_hierarchy: 约束分层字典，包含C1_core_constraint和C2_secondary_constraint
    
    Returns:
        归一化后的约束分层字典
    """
    normalized = {
        "C1_core_constraint": [],
        "C2_secondary_constraint": []
    }
    
    if "C1_core_constraint" in constraint_hierarchy:
        normalized["C1_core_constraint"] = normalize_constraint_list(
            constraint_hierarchy["C1_core_constraint"]
        )
    
    if "C2_secondary_constraint" in constraint_hierarchy:
        normalized["C2_secondary_constraint"] = normalize_constraint_list(
            constraint_hierarchy["C2_secondary_constraint"]
        )
    
    return normalized


def add_constraint_keyword(natural_language: str, standard_keyword: str) -> None:
    """
    动态添加新的约束关键词映射（运行时扩展词典）
    
    Args:
        natural_language: 自然语言约束文本
        standard_keyword: 标准化的约束关键词
    """
    CONSTRAINT_KEYWORD_DICT[natural_language] = standard_keyword


def get_all_standard_keywords() -> List[str]:
    """
    获取所有标准化的约束关键词列表
    
    Returns:
        所有标准约束关键词的列表（去重）
    """
    return list(set(CONSTRAINT_KEYWORD_DICT.values()))

