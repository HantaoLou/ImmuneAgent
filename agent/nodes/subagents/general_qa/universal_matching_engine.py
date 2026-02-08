"""
通用约束-知识匹配引擎

领域/知识点无关的通用匹配引擎，实现：
1. 约束关键词归一化
2. 核心约束C1自动锁定（从C2中筛选与Kc匹配的约束）
3. 条件-结论匹配（Kc是C2_normalized的子集 → 触发对应Kr）
4. 匹配结果生成

核心原则：纯逻辑操作，无业务知识，适配所有知识点
"""

from typing import Dict, List, Any, Set, Optional
from .constraint_normalizer import normalize_constraint_list, normalize_constraint_hierarchy


def universal_matching_engine(
    constraint_hierarchy: Dict[str, List[str]],
    conditional_knowledge: Dict[str, Any],
    analysis_object_name: str = "unknown"
) -> Dict[str, Any]:
    """
    通用约束-知识匹配引擎（领域/知识点无关）
    
    核心逻辑：
    1. 归一化C2约束为标准化关键词
    2. 从C2_normalized中筛选与任意Kc匹配的约束，作为C1
    3. 匹配成功则触发对应Kr，无匹配则触发default_Kr
    
    Args:
        constraint_hierarchy: 从parse_question获取的C1/C2约束
            - C1_core_constraint: 核心约束列表（初始为空，由引擎填充）
            - C2_secondary_constraint: 次要约束列表（原始约束，从题目提取）
        conditional_knowledge: 从activate_knowledge获取的单个知识点的Conditional Knowledge
            格式：
            {
                "Kc1": ["no_global_missing", "sample_size_large", ...],
                "Kr1": "theta_bias: UNBIASED; reason: ...",
                "Kc2": ["has_global_missing", ...],
                "Kr2": "theta_bias: BIASED_DOWN; reason: ...",
                "default_Kr": "theta_bias: UNKNOWN; need more constraints"
            }
        analysis_object_name: 分析对象名称（用于日志）
    
    Returns:
        通用匹配结果字典：
        {
            "constraint_hierarchy_updated": {
                "C1_core_constraint": [...],  # 填充后的核心约束
                "C2_secondary_constraint": [...]  # 保留原始次要约束
            },
            "preliminary_conclusion_universal": "...",  # 通用格式初步结论（Kr）
            "matching_info": {
                "matched_Kc": [...],  # 匹配成功的Kc列表
                "matching_rule": "Kc is subset of C2_normalized",
                "match_type": "full_match" | "partial_match" | "no_match",
                "match_score": 0.0-1.0
            }
        }
    """
    # 步骤1：归一化C2约束为标准化关键词
    C2_raw = constraint_hierarchy.get("C2_secondary_constraint", [])
    C2_normalized = normalize_constraint_list(C2_raw)
    C2_normalized_set = set(C2_normalized)
    
    # 步骤2：提取所有Kc条件集
    all_Kc = []
    Kc_keys = []
    for key, value in conditional_knowledge.items():
        if key.startswith("Kc") and isinstance(value, list):
            all_Kc.append(value)
            Kc_keys.append(key)
    
    if not all_Kc:
        # 如果没有Kc，返回默认结论
        return {
            "constraint_hierarchy_updated": {
                "C1_core_constraint": [],
                "C2_secondary_constraint": C2_raw
            },
            "preliminary_conclusion_universal": conditional_knowledge.get(
                "default_Kr",
                "UNKNOWN: No conditional knowledge (Kc) available"
            ),
            "matching_info": {
                "matched_Kc": [],
                "matching_rule": "No Kc available",
                "match_type": "no_match",
                "match_score": 0.0
            }
        }
    
    # 步骤3：通用核心约束C1自动锁定
    # 规则：从C2_normalized中，筛选出与任意Kc匹配的约束，作为C1
    matched_Kc_list = []
    matched_Kc_keys = []
    C1_locked_set: Set[str] = set()
    
    for i, Kc in enumerate(all_Kc):
        Kc_set = set(Kc)
        # 通用子集匹配规则：Kc是C2_normalized的子集 → 匹配成功
        if Kc_set.issubset(C2_normalized_set):
            matched_Kc_list.append(Kc)
            matched_Kc_keys.append(Kc_keys[i])
            # 将匹配的Kc中的所有约束加入C1
            C1_locked_set.update(Kc_set)
    
    C1_locked = list(C1_locked_set)
    
    # 步骤4：通用条件-结论匹配
    if matched_Kc_list:
        # 匹配成功：使用第一个匹配的Kc对应的Kr
        matched_Kc_key = matched_Kc_keys[0]
        Kr_key = matched_Kc_key.replace("Kc", "Kr")
        final_conclusion = conditional_knowledge.get(
            Kr_key,
            conditional_knowledge.get("default_Kr", "UNKNOWN: Kr not found")
        )
        
        # 计算匹配度
        matched_Kc = matched_Kc_list[0]
        match_score = len(matched_Kc) / len(C2_normalized) if C2_normalized else 1.0
        match_type = "full_match" if len(matched_Kc) == len(C2_normalized) else "partial_match"
    else:
        # 无匹配：使用default_Kr
        final_conclusion = conditional_knowledge.get(
            "default_Kr",
            "UNKNOWN: No matching Kc found, need more constraints"
        )
        match_score = 0.0
        match_type = "no_match"
    
    # 步骤5：返回通用结果
    return {
        "constraint_hierarchy_updated": {
            "C1_core_constraint": C1_locked,
            "C2_secondary_constraint": C2_raw  # 保留原始次要约束
        },
        "preliminary_conclusion_universal": final_conclusion,
        "matching_info": {
            "matched_Kc": matched_Kc_list,
            "matched_Kc_keys": matched_Kc_keys,
            "matching_rule": "Kc is subset of C2_normalized",
            "match_type": match_type,
            "match_score": match_score,
            "C1_locked": C1_locked,
            "C2_normalized": C2_normalized
        }
    }


def match_multiple_knowledge_points(
    constraint_hierarchy: Dict[str, List[str]],
    domain_knowledge: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    对多个知识点执行匹配（批量匹配）
    
    Args:
        constraint_hierarchy: 约束分层字典
        domain_knowledge: 领域知识字典，key为知识点名称，value为知识点的完整知识结构
            每个知识点的结构应包含"Conditional Knowledge"字段
    
    Returns:
        每个知识点的匹配结果字典：
        {
            "knowledge_point_1": universal_matching_engine(...),
            "knowledge_point_2": universal_matching_engine(...),
            ...
        }
    """
    results = {}
    
    for knowledge_name, knowledge_data in domain_knowledge.items():
        # 提取Conditional Knowledge
        conditional_knowledge = knowledge_data.get("Conditional Knowledge", {})
        
        if not conditional_knowledge:
            # 如果没有Conditional Knowledge，跳过
            results[knowledge_name] = {
                "constraint_hierarchy_updated": {
                    "C1_core_constraint": [],
                    "C2_secondary_constraint": constraint_hierarchy.get("C2_secondary_constraint", [])
                },
                "preliminary_conclusion_universal": "UNKNOWN: No Conditional Knowledge available",
                "matching_info": {
                    "matched_Kc": [],
                    "matching_rule": "No Conditional Knowledge available",
                    "match_type": "no_match",
                    "match_score": 0.0
                }
            }
            continue
        
        # 执行匹配
        match_result = universal_matching_engine(
            constraint_hierarchy=constraint_hierarchy,
            conditional_knowledge=conditional_knowledge,
            analysis_object_name=knowledge_name
        )
        results[knowledge_name] = match_result
    
    return results

