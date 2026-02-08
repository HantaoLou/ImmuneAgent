"""
推理规则配置系统

参数化的推导逻辑，支持不同问题类型。
"""

from typing import Dict, List, Any, Optional, Callable
from .enums import QuestionType, ReasoningStrategy


class ReasoningRule:
    """推理规则基类"""
    
    def __init__(self, question_type: QuestionType, strategy: ReasoningStrategy):
        self.question_type = question_type
        self.strategy = strategy
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        """
        应用推理规则，生成推理步骤
        
        Args:
            question: 用户问题
            knowledge_context: 知识上下文
            key_info: 关键信息
            
        Returns:
            推理步骤列表
        """
        return []
    
    def get_strategy(self) -> ReasoningStrategy:
        """获取推理策略"""
        return self.strategy


class ConceptualReasoningRule(ReasoningRule):
    """概念性问题的推理规则（演绎推理）"""
    
    def __init__(self):
        super().__init__(QuestionType.CONCEPTUAL, ReasoningStrategy.DEDUCTIVE)
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        steps = []
        
        # 步骤1：识别核心概念
        concepts = knowledge_context.get("relevant_concepts", [])
        if concepts:
            steps.append(f"识别核心概念：{', '.join(concepts[:3])}")
        
        # 步骤2：应用相关理论
        theories = knowledge_context.get("related_theories", [])
        if theories:
            steps.append(f"应用理论：{theories[0] if theories else '相关科学理论'}")
        
        # 步骤3：从一般到特殊的演绎
        key_facts = knowledge_context.get("key_facts", [])
        if key_facts:
            steps.append(f"基于事实：{key_facts[0] if key_facts else '相关科学事实'}")
        
        # 步骤4：得出结论
        steps.append("通过演绎推理得出结论")
        
        return steps


class ExperimentalReasoningRule(ReasoningRule):
    """实验性问题的推理规则（因果推理）"""
    
    def __init__(self):
        super().__init__(QuestionType.EXPERIMENTAL, ReasoningStrategy.CAUSAL)
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        steps = []
        
        # 步骤1：识别实验条件
        steps.append("分析实验条件和变量")
        
        # 步骤2：识别因果关系
        methods = knowledge_context.get("experimental_methods", [])
        if methods:
            steps.append(f"考虑实验方法：{methods[0] if methods else '相关实验方法'}")
        
        # 步骤3：分析可能的影响因素
        steps.append("分析可能的影响因素和干扰变量")
        
        # 步骤4：建立因果链
        steps.append("建立因果关系链")
        
        # 步骤5：得出结论
        steps.append("基于因果关系得出结论")
        
        return steps


class CalculationReasoningRule(ReasoningRule):
    """计算性问题的推理规则（直接推理）"""
    
    def __init__(self):
        super().__init__(QuestionType.CALCULATION, ReasoningStrategy.DIRECT)
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        steps = []
        
        # 步骤1：提取数值和参数
        steps.append("提取问题中的数值和参数")
        
        # 步骤2：识别计算公式
        steps.append("识别适用的计算公式或方法")
        
        # 步骤3：执行计算
        steps.append("执行计算步骤")
        
        # 步骤4：验证结果合理性
        steps.append("验证计算结果的合理性")
        
        return steps


class ComparisonReasoningRule(ReasoningRule):
    """比较性问题的推理规则（类比推理）"""
    
    def __init__(self):
        super().__init__(QuestionType.COMPARISON, ReasoningStrategy.ANALOGICAL)
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        steps = []
        
        # 步骤1：识别比较对象
        steps.append("识别需要比较的对象或概念")
        
        # 步骤2：确定比较维度
        steps.append("确定比较的维度和标准")
        
        # 步骤3：进行类比分析
        concepts = knowledge_context.get("relevant_concepts", [])
        if concepts:
            steps.append(f"基于相关概念进行类比：{', '.join(concepts[:2])}")
        
        # 步骤4：找出相似性和差异性
        steps.append("分析相似性和差异性")
        
        # 步骤5：得出结论
        steps.append("基于比较分析得出结论")
        
        return steps


class CausalReasoningRule(ReasoningRule):
    """因果关系问题的推理规则（溯因推理）"""
    
    def __init__(self):
        super().__init__(QuestionType.CAUSAL, ReasoningStrategy.ABDUCTIVE)
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        steps = []
        
        # 步骤1：识别观察到的现象
        steps.append("识别观察到的现象或结果")
        
        # 步骤2：提出可能的解释
        theories = knowledge_context.get("related_theories", [])
        if theories:
            steps.append(f"基于理论提出可能解释：{theories[0] if theories else '相关理论'}")
        
        # 步骤3：评估解释的合理性
        key_facts = knowledge_context.get("key_facts", [])
        if key_facts:
            steps.append(f"评估解释与事实的一致性：{key_facts[0] if key_facts else '相关事实'}")
        
        # 步骤4：选择最佳解释
        steps.append("选择最合理的解释")
        
        return steps


class DefinitionReasoningRule(ReasoningRule):
    """定义性问题的推理规则（直接推理）"""
    
    def __init__(self):
        super().__init__(QuestionType.DEFINITION, ReasoningStrategy.DIRECT)
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        steps = []
        
        # 步骤1：查找标准定义
        concepts = knowledge_context.get("relevant_concepts", [])
        if concepts:
            steps.append(f"查找相关概念的标准定义：{concepts[0] if concepts else '相关概念'}")
        
        # 步骤2：解释关键术语
        steps.append("解释定义中的关键术语")
        
        # 步骤3：提供背景信息
        key_facts = knowledge_context.get("key_facts", [])
        if key_facts:
            steps.append(f"提供相关背景：{key_facts[0] if key_facts else '相关背景'}")
        
        # 步骤4：给出完整定义
        steps.append("给出完整准确的定义")
        
        return steps


# 新的推理规则类（对应优化后的问题类型）
class JudgmentReasoningRule(ReasoningRule):
    """判断型问题的推理规则（演绎推理）"""
    
    def __init__(self):
        super().__init__(QuestionType.JUDGMENT, ReasoningStrategy.DEDUCTIVE)
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        steps = []
        
        # 步骤1：识别分析对象
        analysis_object = key_info.get("分析对象", "核心研究对象")
        steps.append(f"识别分析对象：{analysis_object}")
        
        # 步骤2：分析实验条件和约束条件
        exp_condition = key_info.get("实验条件", "无")
        constraint = key_info.get("约束条件", "无")
        if exp_condition != "无":
            steps.append(f"分析实验条件：{exp_condition}")
        if constraint != "无":
            steps.append(f"考虑约束条件：{constraint}")
        
        # 步骤3：应用相关理论进行判断
        theories = knowledge_context.get("related_theories", [])
        if theories:
            steps.append(f"应用理论进行判断：{theories[0] if theories else '相关科学理论'}")
        
        # 步骤4：得出结论
        target_output = key_info.get("目标输出", "判断结果")
        steps.append(f"得出结论：{target_output}")
        
        return steps


class AnalysisReasoningRule(ReasoningRule):
    """分析型问题的推理规则（因果推理）"""
    
    def __init__(self):
        super().__init__(QuestionType.ANALYSIS, ReasoningStrategy.CAUSAL)
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        steps = []
        
        # 步骤1：识别分析对象和目标输出
        analysis_object = key_info.get("分析对象", "核心研究对象")
        target_output = key_info.get("目标输出", "分析结果")
        steps.append(f"识别分析对象：{analysis_object}")
        steps.append(f"明确分析目标：{target_output}")
        
        # 步骤2：分析实验条件和因果关系
        exp_condition = key_info.get("实验条件", "无")
        if exp_condition != "无":
            steps.append(f"分析实验条件的影响：{exp_condition}")
        
        # 步骤3：建立因果链
        methods = knowledge_context.get("experimental_methods", [])
        if methods:
            steps.append(f"考虑实验方法：{methods[0] if methods else '相关实验方法'}")
        steps.append("建立因果关系链")
        
        # 步骤4：分析影响因素
        steps.append("分析可能的影响因素和机制")
        
        # 步骤5：得出结论
        steps.append("基于因果分析得出结论")
        
        return steps


class EnumerationReasoningRule(ReasoningRule):
    """枚举型问题的推理规则（归纳推理）"""
    
    def __init__(self):
        super().__init__(QuestionType.ENUMERATION, ReasoningStrategy.INDUCTIVE)
    
    def apply(self, question: str, knowledge_context: Dict[str, Any], 
              key_info: Dict[str, Any]) -> List[str]:
        steps = []
        
        # 步骤1：识别分析对象
        analysis_object = key_info.get("分析对象", "核心研究对象")
        steps.append(f"识别分析对象：{analysis_object}")
        
        # 步骤2：确定枚举标准
        target_output = key_info.get("目标输出", "枚举结果")
        steps.append(f"确定枚举标准：{target_output}")
        
        # 步骤3：从知识库中归纳相关项
        concepts = knowledge_context.get("relevant_concepts", [])
        if concepts:
            steps.append(f"基于相关概念归纳：{', '.join(concepts[:3])}")
        
        # 步骤4：列举所有相关项
        steps.append("列举所有符合条件的项")
        
        # 步骤5：验证完整性
        steps.append("验证枚举的完整性和准确性")
        
        return steps


# 推理规则注册表（支持新旧两种问题类型）
REASONING_RULES: Dict[Any, ReasoningRule] = {
    # 新的问题类型（优化后）
    QuestionType.JUDGMENT: JudgmentReasoningRule(),
    QuestionType.CALCULATION: CalculationReasoningRule(),
    QuestionType.ANALYSIS: AnalysisReasoningRule(),
    QuestionType.ENUMERATION: EnumerationReasoningRule(),
    # 旧的问题类型（向后兼容）
    QuestionType.CONCEPTUAL: ConceptualReasoningRule(),
    QuestionType.EXPERIMENTAL: ExperimentalReasoningRule(),
    QuestionType.COMPARISON: ComparisonReasoningRule(),
    QuestionType.CAUSAL: CausalReasoningRule(),
    QuestionType.DEFINITION: DefinitionReasoningRule(),
    # 字符串映射（支持直接从字符串获取）
    "判断型": JudgmentReasoningRule(),
    "计算型": CalculationReasoningRule(),
    "分析型": AnalysisReasoningRule(),
    "枚举型": EnumerationReasoningRule(),
}


def get_reasoning_rule(question_type: Any) -> Optional[ReasoningRule]:
    """
    根据问题类型获取推理规则
    
    支持枚举类型和字符串类型的问题类型
    
    Args:
        question_type: 问题类型（可以是QuestionType枚举或字符串）
        
    Returns:
        推理规则，如果不存在则返回None
    """
    # 如果是字符串，直接查找
    if isinstance(question_type, str):
        return REASONING_RULES.get(question_type)
    
    # 如果是枚举类型，查找枚举值
    if isinstance(question_type, QuestionType):
        return REASONING_RULES.get(question_type)
    
    # 如果都不匹配，返回None
    return None

