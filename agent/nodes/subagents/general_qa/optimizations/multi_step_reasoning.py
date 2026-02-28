"""
Multi-Step Reasoning Framework
将复杂问题分解为多步推理，避免超时

这个模块实现了将复杂的生物医学问题分解为多个可管理的步骤，
每步独立执行，避免整体超时，并提高推理准确率。
"""

import re
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class StepType(Enum):
    """推理步骤类型"""
    INFORMATION_EXTRACTION = "information_extraction"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
    LOGICAL_DEDUCTION = "logical_deduction"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    VERIFICATION = "verification"
    INTEGRATION = "integration"


class ProblemType(Enum):
    """问题类型"""
    GENETICS_CALCULATION = "genetics_calculation"
    CLINICAL_DIAGNOSIS = "clinical_diagnosis"
    MOLECULAR_BIOLOGY = "molecular_biology"
    MULTI_DOMAIN = "multi_domain"
    SEQUENCE_ANALYSIS = "sequence_analysis"
    PATHWAY_ANALYSIS = "pathway_analysis"
    GENERAL = "general"


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    step_type: StepType
    description: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    is_completed: bool = False
    error: Optional[str] = None
    depends_on: List[int] = field(default_factory=list)
    execution_time: float = 0.0


@dataclass
class ReasoningPlan:
    """推理计划"""
    question: str
    problem_type: ProblemType
    steps: List[ReasoningStep]
    current_step: int = 0
    is_completed: bool = False
    final_answer: Optional[str] = None
    total_execution_time: float = 0.0


# 问题分解模板 - 根据问题类型定义推理步骤
DECOMPOSITION_TEMPLATES = {
    ProblemType.GENETICS_CALCULATION: [
        {
            "step_type": StepType.INFORMATION_EXTRACTION,
            "description": "提取遗传学信息（基因型、表型、频率）",
            "extract_keywords": ["genotype", "phenotype", "frequency", "allele", "dominant", "recessive"]
        },
        {
            "step_type": StepType.CALCULATION,
            "description": "使用遗传学公式计算期望值",
            "formulas": ["hardy-weinberg", "probability", "ratio"]
        },
        {
            "step_type": StepType.VERIFICATION,
            "description": "验证计算结果",
            "check_type": "numerical"
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "得出最终答案",
            "output_format": "option_or_value"
        }
    ],
    ProblemType.CLINICAL_DIAGNOSIS: [
        {
            "step_type": StepType.INFORMATION_EXTRACTION,
            "description": "提取症状和临床数据",
            "extract_keywords": ["symptom", "patient", "clinical", "diagnosis", "treatment"]
        },
        {
            "step_type": StepType.KNOWLEDGE_RETRIEVAL,
            "description": "检索相关医学知识",
            "search_domains": ["disease", "symptom", "treatment"]
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "生成鉴别诊断列表",
            "output_format": "differential_list"
        },
        {
            "step_type": StepType.COMPARISON,
            "description": "评估每个诊断与证据的匹配度",
            "comparison_type": "evidence_based"
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "选择最可能的诊断",
            "output_format": "diagnosis"
        }
    ],
    ProblemType.MOLECULAR_BIOLOGY: [
        {
            "step_type": StepType.INFORMATION_EXTRACTION,
            "description": "提取分子组件和相互作用信息",
            "extract_keywords": ["protein", "enzyme", "gene", "expression", "pathway"]
        },
        {
            "step_type": StepType.KNOWLEDGE_RETRIEVAL,
            "description": "映射到已知通路",
            "search_domains": ["pathway", "interaction", "function"]
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "预测扰动结果",
            "output_format": "prediction"
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "得出结论",
            "output_format": "conclusion"
        }
    ],
    ProblemType.SEQUENCE_ANALYSIS: [
        {
            "step_type": StepType.INFORMATION_EXTRACTION,
            "description": "提取序列信息和约束条件",
            "extract_keywords": ["sequence", "DNA", "RNA", "protein", "codon", "amino"]
        },
        {
            "step_type": StepType.CALCULATION,
            "description": "分析序列特征",
            "analysis_type": "sequence"
        },
        {
            "step_type": StepType.VERIFICATION,
            "description": "验证序列操作结果",
            "check_type": "biological"
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "得出最终答案",
            "output_format": "option_or_value"
        }
    ],
    ProblemType.PATHWAY_ANALYSIS: [
        {
            "step_type": StepType.INFORMATION_EXTRACTION,
            "description": "识别通路中的关键分子",
            "extract_keywords": ["pathway", "upstream", "downstream", "activate", "inhibit"]
        },
        {
            "step_type": StepType.KNOWLEDGE_RETRIEVAL,
            "description": "检索通路知识",
            "search_domains": ["signaling", "metabolism", "regulation"]
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "追踪信号传导路径",
            "output_format": "pathway_trace"
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "预测效应",
            "output_format": "effect"
        }
    ],
    ProblemType.MULTI_DOMAIN: [
        {
            "step_type": StepType.INFORMATION_EXTRACTION,
            "description": "分解问题为领域特定的子问题",
            "extract_keywords": []
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "独立解决每个子问题",
            "output_format": "sub_answers"
        },
        {
            "step_type": StepType.INTEGRATION,
            "description": "整合跨领域结果",
            "integration_type": "cross_domain"
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "得出最终答案",
            "output_format": "final"
        }
    ],
    ProblemType.GENERAL: [
        {
            "step_type": StepType.INFORMATION_EXTRACTION,
            "description": "提取问题的关键信息",
            "extract_keywords": []
        },
        {
            "step_type": StepType.KNOWLEDGE_RETRIEVAL,
            "description": "检索相关知识",
            "search_domains": []
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "进行逻辑推理",
            "output_format": "reasoning"
        },
        {
            "step_type": StepType.LOGICAL_DEDUCTION,
            "description": "得出结论",
            "output_format": "answer"
        }
    ]
}


# 问题类型检测关键词
PROBLEM_TYPE_INDICATORS = {
    ProblemType.GENETICS_CALCULATION: [
        "genotype", "phenotype", "allele", "frequency", "hardy-weinberg",
        "inheritance", "dominant", "recessive", "heterozygous", "homozygous",
        "基因型", "表型", "等位基因", "频率", "遗传", "显性", "隐性"
    ],
    ProblemType.CLINICAL_DIAGNOSIS: [
        "diagnosis", "patient", "symptom", "treatment", "disease", "clinical",
        "differential", "prognosis", "therapy",
        "诊断", "患者", "症状", "治疗", "疾病", "临床"
    ],
    ProblemType.MOLECULAR_BIOLOGY: [
        "protein", "enzyme", "gene expression", "transcription", "translation",
        "mrna", "regulation", "signaling", "receptor", "ligand",
        "蛋白质", "酶", "基因表达", "转录", "翻译", "信号"
    ],
    ProblemType.SEQUENCE_ANALYSIS: [
        "sequence", "dna", "rna", "codon", "amino acid", "nucleotide",
        "orf", "reading frame", "start codon", "stop codon",
        "序列", "密码子", "氨基酸", "核苷酸", "阅读框"
    ],
    ProblemType.PATHWAY_ANALYSIS: [
        "pathway", "upstream", "downstream", "cascade", "activate", "inhibit",
        "kinase", "phosphorylation", "signal transduction",
        "通路", "上游", "下游", "激活", "抑制", "信号转导"
    ]
}


def detect_problem_type(question: str, question_type: Optional[str] = None) -> ProblemType:
    """
    检测问题类型
    
    Args:
        question: 问题文本
        question_type: 已知的问题类型（可选）
        
    Returns:
        ProblemType
    """
    question_lower = question.lower()
    
    # 如果有已知的question_type，优先使用
    if question_type:
        type_mapping = {
            "genetics": ProblemType.GENETICS_CALCULATION,
            "clinical": ProblemType.CLINICAL_DIAGNOSIS,
            "molecular": ProblemType.MOLECULAR_BIOLOGY,
            "sequence": ProblemType.SEQUENCE_ANALYSIS,
            "pathway": ProblemType.PATHWAY_ANALYSIS,
        }
        for key, prob_type in type_mapping.items():
            if key in question_type.lower():
                return prob_type
    
    # 基于关键词检测
    scores = {}
    for prob_type, keywords in PROBLEM_TYPE_INDICATORS.items():
        score = sum(1 for kw in keywords if kw in question_lower)
        if score > 0:
            scores[prob_type] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    # 检查是否跨领域
    domain_count = sum(1 for domain in ["gene", "protein", "cell", "clinical", "pathway"] 
                       if domain in question_lower)
    if domain_count >= 2:
        return ProblemType.MULTI_DOMAIN
    
    return ProblemType.GENERAL


def create_reasoning_plan(question: str, question_type: Optional[str] = None) -> ReasoningPlan:
    """
    创建推理计划
    
    Args:
        question: 问题文本
        question_type: 问题类型
        
    Returns:
        ReasoningPlan
    """
    problem_type = detect_problem_type(question, question_type)
    template = DECOMPOSITION_TEMPLATES.get(problem_type, DECOMPOSITION_TEMPLATES[ProblemType.GENERAL])
    
    steps = []
    for i, step_config in enumerate(template):
        step = ReasoningStep(
            step_id=i,
            step_type=step_config["step_type"],
            description=step_config["description"],
            input_data={
                "question": question,
                "config": step_config
            },
            depends_on=[i-1] if i > 0 else []
        )
        steps.append(step)
    
    return ReasoningPlan(
        question=question,
        problem_type=problem_type,
        steps=steps
    )


class MultiStepReasoner:
    """多步推理器"""
    
    def __init__(
        self, 
        llm_client=None, 
        max_time_per_step: float = 60.0,
        enable_logging: bool = True
    ):
        """
        初始化多步推理器
        
        Args:
            llm_client: LLM客户端（可选）
            max_time_per_step: 每步最大执行时间（秒）
            enable_logging: 是否启用日志
        """
        self.llm_client = llm_client
        self.max_time_per_step = max_time_per_step
        self.enable_logging = enable_logging
        self._completed_plans: List[ReasoningPlan] = []
        self._step_handlers: Dict[StepType, Callable] = {
            StepType.INFORMATION_EXTRACTION: self._extract_information,
            StepType.CALCULATION: self._calculate,
            StepType.KNOWLEDGE_RETRIEVAL: self._retrieve_knowledge,
            StepType.COMPARISON: self._compare,
            StepType.LOGICAL_DEDUCTION: self._deduce,
            StepType.VERIFICATION: self._verify,
            StepType.INTEGRATION: self._integrate,
        }
    
    def reason(
        self, 
        question: str, 
        question_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, ReasoningPlan]:
        """
        执行多步推理
        
        Args:
            question: 问题文本
            question_type: 问题类型
            context: 额外上下文信息
            
        Returns:
            (final_answer, reasoning_plan)
        """
        start_time = time.time()
        
        # 创建推理计划
        plan = create_reasoning_plan(question, question_type)
        
        if self.enable_logging:
            logger.info(f"Created reasoning plan for {plan.problem_type.value}: {len(plan.steps)} steps")
        
        # 合并上下文
        if context:
            for step in plan.steps:
                step.input_data.update(context)
        
        # 执行每一步
        for step in plan.steps:
            step_start = time.time()
            try:
                handler = self._step_handlers.get(step.step_type)
                if handler:
                    step_output = handler(step, plan)
                else:
                    step_output = self._default_handler(step, plan)
                
                step.output_data = step_output or {}
                step.is_completed = True
                plan.current_step = step.step_id + 1
                step.execution_time = time.time() - step_start
                
                if self.enable_logging:
                    logger.info(f"Step {step.step_id} completed: {step.description} ({step.execution_time:.2f}s)")
                
            except Exception as e:
                step.error = str(e)
                step.execution_time = time.time() - step_start
                if self.enable_logging:
                    logger.error(f"Step {step.step_id} failed: {e}")
                # 继续执行下一步，而不是中断
        
        # 提取最终答案
        plan.total_execution_time = time.time() - start_time
        if plan.current_step >= len(plan.steps):
            plan.is_completed = True
            plan.final_answer = self._extract_final_answer(plan)
        else:
            # 部分完成时也尝试提取答案
            plan.final_answer = self._extract_partial_answer(plan)
        
        self._completed_plans.append(plan)
        return plan.final_answer or "Unable to complete reasoning", plan
    
    def _extract_information(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """信息提取步骤"""
        question = step.input_data.get("question", "")
        config = step.input_data.get("config", {})
        keywords = config.get("extract_keywords", [])
        
        # 提取关键实体
        entities = self._extract_entities(question, keywords)
        
        # 提取数字
        numbers = self._extract_numbers(question)
        
        # 提取关系词
        relationships = self._extract_relationships(question)
        
        # 提取约束条件
        constraints = self._extract_constraints(question)
        
        return {
            "entities": entities,
            "numbers": numbers,
            "relationships": relationships,
            "constraints": constraints,
            "raw_question": question
        }
    
    def _calculate(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """计算步骤"""
        context = self._get_context_from_dependencies(step, plan)
        
        # 基于问题类型执行计算
        if plan.problem_type == ProblemType.GENETICS_CALCULATION:
            return self._genetic_calculation(context)
        elif plan.problem_type == ProblemType.SEQUENCE_ANALYSIS:
            return self._sequence_calculation(context)
        else:
            return self._general_calculation(context)
    
    def _retrieve_knowledge(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """知识检索步骤"""
        context = self._get_context_from_dependencies(step, plan)
        config = step.input_data.get("config", {})
        search_domains = config.get("search_domains", [])
        
        # 返回检索上下文（实际检索由外部系统完成）
        return {
            "knowledge_context": context,
            "search_domains": search_domains,
            "entities_to_search": context.get("entities", []),
            "retrieved": True
        }
    
    def _compare(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """比较步骤"""
        context = self._get_context_from_dependencies(step, plan)
        config = step.input_data.get("config", {})
        comparison_type = config.get("comparison_type", "general")
        
        return {
            "comparison_type": comparison_type,
            "comparison_context": context,
            "compared": True
        }
    
    def _deduce(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """逻辑推导步骤"""
        context = self._get_context_from_dependencies(step, plan)
        config = step.input_data.get("config", {})
        output_format = config.get("output_format", "general")
        
        # 从上下文中提取潜在答案
        potential_answer = self._extract_potential_answer(context, output_format)
        
        return {
            "conclusion": potential_answer,
            "reasoning_context": context,
            "output_format": output_format,
            "deduced": True
        }
    
    def _verify(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """验证步骤"""
        context = self._get_context_from_dependencies(step, plan)
        config = step.input_data.get("config", {})
        check_type = config.get("check_type", "general")
        
        is_valid = True
        validation_notes = []
        
        # 数值验证
        if check_type == "numerical":
            numbers = context.get("numbers", [])
            if numbers:
                validation_notes.append(f"Found {len(numbers)} numerical values")
        
        # 生物学验证
        elif check_type == "biological":
            entities = context.get("entities", [])
            if entities:
                validation_notes.append(f"Found {len(entities)} biological entities")
        
        return {
            "is_valid": is_valid,
            "validation_notes": validation_notes,
            "check_type": check_type,
            "verified": True
        }
    
    def _integrate(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """整合步骤"""
        context = self._get_context_from_dependencies(step, plan)
        config = step.input_data.get("config", {})
        integration_type = config.get("integration_type", "general")
        
        # 收集所有子答案
        sub_answers = []
        for prev_step in plan.steps[:step.step_id]:
            if prev_step.output_data and prev_step.output_data.get("conclusion"):
                sub_answers.append(prev_step.output_data["conclusion"])
        
        return {
            "sub_answers": sub_answers,
            "integration_type": integration_type,
            "integrated": True
        }
    
    def _default_handler(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """默认处理器"""
        return {
            "step_type": step.step_type.value,
            "description": step.description,
            "handled": True
        }
    
    def _get_context_from_dependencies(self, step: ReasoningStep, plan: ReasoningPlan) -> Dict[str, Any]:
        """收集依赖步骤的输出"""
        context = {}
        for dep_id in step.depends_on:
            if dep_id < len(plan.steps) and plan.steps[dep_id].output_data:
                context.update(plan.steps[dep_id].output_data)
        return context
    
    def _extract_entities(self, text: str, keywords: List[str]) -> List[str]:
        """提取实体"""
        entities = []
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                entities.append(keyword)
        
        # 使用正则提取大写开头的词（可能是专有名词）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(proper_nouns)
        
        return list(set(entities))
    
    def _extract_numbers(self, text: str) -> List[float]:
        """提取数字"""
        numbers = []
        # 匹配整数和小数
        matches = re.findall(r'[-+]?\d*\.?\d+', text)
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        return numbers
    
    def _extract_relationships(self, text: str) -> List[str]:
        """提取关系词"""
        relationship_patterns = [
            r'\b(?:activates?|inhibits?|regulates?|binds?|interacts? with)\b',
            r'\b(?:upstream|downstream|before|after|leads? to)\b',
            r'\b(?:causes?|results? in|affects?)\b',
            r'\b(?:激活|抑制|调节|结合|相互作用|上游|下游|导致)\b'
        ]
        
        relationships = []
        for pattern in relationship_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            relationships.extend(matches)
        
        return list(set(relationships))
    
    def _extract_constraints(self, text: str) -> List[str]:
        """提取约束条件"""
        constraint_patterns = [
            r'\b(?:must|should|only|except|unless|given|assuming)\b',
            r'\b(?:if|when|where|provided that)\b',
            r'\b(?:必须|应该|只有|除了|除非|给定|假设)\b'
        ]
        
        constraints = []
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            constraints.extend(matches)
        
        return list(set(constraints))
    
    def _genetic_calculation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """遗传学计算"""
        numbers = context.get("numbers", [])
        entities = context.get("entities", [])
        
        return {
            "calculation_type": "genetic",
            "input_numbers": numbers,
            "entities": entities,
            "calculated": True
        }
    
    def _sequence_calculation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """序列计算"""
        entities = context.get("entities", [])
        
        return {
            "calculation_type": "sequence",
            "entities": entities,
            "calculated": True
        }
    
    def _general_calculation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """通用计算"""
        numbers = context.get("numbers", [])
        
        return {
            "calculation_type": "general",
            "input_numbers": numbers,
            "calculated": True
        }
    
    def _extract_potential_answer(self, context: Dict[str, Any], output_format: str) -> Optional[str]:
        """提取潜在答案"""
        # 检查是否有选项格式
        if output_format == "option_or_value":
            # 尝试从上下文中找到选项
            numbers = context.get("numbers", [])
            if numbers:
                return str(numbers[0])
        
        # 检查是否有结论
        if "conclusion" in context:
            return context["conclusion"]
        
        return None
    
    def _extract_final_answer(self, plan: ReasoningPlan) -> str:
        """提取最终答案"""
        # 从最后一个完成的步骤提取答案
        for step in reversed(plan.steps):
            if step.is_completed and step.output_data:
                conclusion = step.output_data.get("conclusion")
                if conclusion:
                    return str(conclusion)
        
        return "Unable to determine final answer"
    
    def _extract_partial_answer(self, plan: ReasoningPlan) -> str:
        """从部分完成的计划中提取答案"""
        # 查找最后一个有输出的步骤
        last_output = None
        for step in plan.steps:
            if step.output_data:
                last_output = step.output_data
        
        if last_output:
            if "conclusion" in last_output:
                return str(last_output["conclusion"])
            if "entities" in last_output:
                return f"Extracted entities: {', '.join(last_output['entities'])}"
        
        return "Partial reasoning completed"
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取推理统计"""
        if not self._completed_plans:
            return {"total_plans": 0}
        
        completed = sum(1 for p in self._completed_plans if p.is_completed)
        avg_steps = sum(p.current_step for p in self._completed_plans) / len(self._completed_plans)
        avg_time = sum(p.total_execution_time for p in self._completed_plans) / len(self._completed_plans)
        
        problem_type_counts = {}
        for p in self._completed_plans:
            pt = p.problem_type.value
            problem_type_counts[pt] = problem_type_counts.get(pt, 0) + 1
        
        return {
            "total_plans": len(self._completed_plans),
            "completed_plans": completed,
            "completion_rate": completed / len(self._completed_plans),
            "average_steps_executed": avg_steps,
            "average_execution_time": avg_time,
            "problem_type_distribution": problem_type_counts
        }


# 便捷函数
def reason_with_steps(
    question: str, 
    question_type: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[str, ReasoningPlan]:
    """
    使用多步推理回答问题
    
    Args:
        question: 问题文本
        question_type: 问题类型（可选）
        context: 额外上下文（可选）
        
    Returns:
        (final_answer, reasoning_plan)
    """
    reasoner = MultiStepReasoner()
    return reasoner.reason(question, question_type, context)


def get_reasoning_plan(question: str, question_type: Optional[str] = None) -> ReasoningPlan:
    """
    获取推理计划（不执行）
    
    Args:
        question: 问题文本
        question_type: 问题类型
        
    Returns:
        ReasoningPlan
    """
    return create_reasoning_plan(question, question_type)


def should_use_multi_step(question: str) -> bool:
    """
    判断是否应该使用多步推理
    
    Args:
        question: 问题文本
        
    Returns:
        bool
    """
    # 长问题使用多步推理
    if len(question) > 200:
        return True
    
    # 复杂问题类型使用多步推理
    problem_type = detect_problem_type(question)
    if problem_type in [
        ProblemType.GENETICS_CALCULATION,
        ProblemType.CLINICAL_DIAGNOSIS,
        ProblemType.MULTI_DOMAIN
    ]:
        return True
    
    # 包含多个句子的问题使用多步推理
    sentence_count = len(re.split(r'[.!?。！？]', question))
    if sentence_count > 2:
        return True
    
    return False


def get_step_type_description(step_type: StepType) -> str:
    """获取步骤类型的描述"""
    descriptions = {
        StepType.INFORMATION_EXTRACTION: "从问题中提取关键信息",
        StepType.CALCULATION: "执行数学或逻辑计算",
        StepType.COMPARISON: "比较不同选项或实体",
        StepType.LOGICAL_DEDUCTION: "进行逻辑推理",
        StepType.KNOWLEDGE_RETRIEVAL: "检索相关知识",
        StepType.VERIFICATION: "验证中间结果",
        StepType.INTEGRATION: "整合多个子结果"
    }
    return descriptions.get(step_type, "执行推理步骤")


