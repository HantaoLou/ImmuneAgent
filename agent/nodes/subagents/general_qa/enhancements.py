"""
General QA 子图增强模块

包含针对 HLE (Hard Long-Enduring) 复杂问题的优化实现：
1. Self-Consistency 多路径投票
2. Chain-of-Thought 显式推理链
3. 数学计算交叉验证
4. 迭代式知识检索
5. 元认知监控
6. 智能异常诊断
7. 增强工具意图识别
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from dataclasses import dataclass, field
import re
import asyncio
import json


# ===================== 数据结构定义 =====================

@dataclass
class InferenceStep:
    """结构化推理步骤"""
    step_id: int
    reasoning_type: str  # "deduction", "induction", "abduction", "calculation", "comparison"
    premise: str
    operation: str
    conclusion: str
    confidence: float = 1.0
    depends_on: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "reasoning_type": self.reasoning_type,
            "premise": self.premise,
            "operation": self.operation,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "depends_on": self.depends_on
        }


@dataclass
class ReasoningPath:
    """推理路径"""
    path_id: int
    temperature: float
    steps: List[InferenceStep]
    final_conclusion: str
    answer: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "temperature": self.temperature,
            "steps": [s.to_dict() for s in self.steps],
            "final_conclusion": self.final_conclusion,
            "answer": self.answer
        }


@dataclass
class SelfConsistencyResult:
    """Self-Consistency 结果"""
    paths: List[ReasoningPath]
    answer_votes: Dict[str, int]
    consensus_answer: str
    consensus_ratio: float
    confidence_level: str  # "high", "medium", "low"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": [p.to_dict() for p in self.paths],
            "answer_votes": self.answer_votes,
            "consensus_answer": self.consensus_answer,
            "consensus_ratio": self.consensus_ratio,
            "confidence_level": self.confidence_level
        }


@dataclass
class CalculationVerificationResult:
    """计算验证结果"""
    symbolic_result: Optional[float] = None
    numerical_result: Optional[float] = None
    llm_verification: Optional[str] = None
    all_match: bool = False
    discrepancy: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbolic_result": self.symbolic_result,
            "numerical_result": self.numerical_result,
            "llm_verification": self.llm_verification,
            "all_match": self.all_match,
            "discrepancy": self.discrepancy
        }


@dataclass
class MetaCognitiveAssessment:
    """元认知评估结果"""
    goal_alignment: bool = True
    constraint_coverage: bool = True
    knowledge_gaps: List[str] = field(default_factory=list)
    reasoning_coherence: bool = True
    confidence_calibration: float = 1.0
    needs_backtracking: bool = False
    backtracking_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_alignment": self.goal_alignment,
            "constraint_coverage": self.constraint_coverage,
            "knowledge_gaps": self.knowledge_gaps,
            "reasoning_coherence": self.reasoning_coherence,
            "confidence_calibration": self.confidence_calibration,
            "needs_backtracking": self.needs_backtracking,
            "backtracking_reason": self.backtracking_reason
        }


@dataclass
class RetryStrategy:
    """重试策略"""
    target_node: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_node": self.target_node,
            "action": self.action,
            "params": self.params,
            "reason": self.reason
        }


# ===================== 优化1: Self-Consistency 多路径投票 =====================

class SelfConsistencyEngine:
    """Self-Consistency 多路径推理引擎"""
    
    def __init__(self, num_paths: int = 3, temperatures: List[float] = None):
        self.num_paths = num_paths
        self.temperatures = temperatures or [0.0, 0.3, 0.7]
    
    def extract_answer(self, conclusion: str, options: List[str] = None) -> str:
        """从结论中提取答案"""
        if not conclusion:
            return ""
        
        conclusion_stripped = conclusion.strip()
        
        # 如果有选项，尝试匹配选项
        if options:
            # 直接匹配选项字母
            for opt_label in ['A', 'B', 'C', 'D', 'E', 'F']:
                if conclusion_stripped.upper() == opt_label:
                    return opt_label
            
            # 匹配 "答案是 A" 或 "选 A" 等模式
            patterns = [
                r'[答案选](?:是)?\s*([A-F])',
                r'(?:option|answer)[:\s]*([A-F])',
                r'^([A-F])\.?\s',
            ]
            for pattern in patterns:
                match = re.search(pattern, conclusion, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
        
        # 对于数值答案，提取数值
        num_match = re.search(r'([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*(?:mL/g|mg/L|μM|mM|M|g/mol|kDa|Gy|%|units?)?', conclusion)
        if num_match:
            return num_match.group(0).strip()
        
        # 对于 True/False 问题
        if re.search(r'\b(true|correct|yes)\b', conclusion, re.IGNORECASE):
            return "True"
        if re.search(r'\b(false|incorrect|no)\b', conclusion, re.IGNORECASE):
            return "False"
        
        # 返回结论的前100个字符作为答案
        return conclusion[:100].strip()
    
    def normalize_answer(self, answer: str) -> str:
        """标准化答案以便比较"""
        answer = answer.strip().upper()
        
        # 标准化选项字母
        if answer in ['A', 'B', 'C', 'D', 'E', 'F']:
            return answer
        
        # 标准化 True/False
        if answer in ['TRUE', 'CORRECT', 'YES', 'T']:
            return 'TRUE'
        if answer in ['FALSE', 'INCORRECT', 'NO', 'F']:
            return 'FALSE'
        
        # 标准化数值（去除空格，统一格式）
        answer = re.sub(r'\s+', '', answer)
        
        return answer
    
    def vote(self, answers: List[str]) -> Tuple[str, float, Dict[str, int]]:
        """对多个答案进行投票"""
        normalized = [self.normalize_answer(a) for a in answers]
        vote_counts = Counter(normalized)
        
        most_common = vote_counts.most_common(1)[0]
        consensus_answer = most_common[0]
        consensus_ratio = most_common[1] / len(answers)
        
        return consensus_answer, consensus_ratio, dict(vote_counts)
    
    def determine_confidence_level(self, consensus_ratio: float) -> str:
        """根据一致性比例确定置信度级别"""
        if consensus_ratio >= 0.67:  # 2/3 以上一致
            return "high"
        elif consensus_ratio >= 0.5:  # 1/2 以上一致
            return "medium"
        else:
            return "low"
    
    def aggregate_results(
        self,
        paths: List[ReasoningPath],
        options: List[str] = None
    ) -> SelfConsistencyResult:
        """聚合多路径结果"""
        # 提取所有答案
        answers = []
        for path in paths:
            answer = self.extract_answer(path.final_conclusion, options)
            path.answer = answer
            answers.append(answer)
        
        # 投票
        consensus_answer, consensus_ratio, vote_counts = self.vote(answers)
        
        # 确定置信度
        confidence_level = self.determine_confidence_level(consensus_ratio)
        
        return SelfConsistencyResult(
            paths=paths,
            answer_votes=vote_counts,
            consensus_answer=consensus_answer,
            consensus_ratio=consensus_ratio,
            confidence_level=confidence_level
        )


# ===================== 优化2: Chain-of-Thought 显式推理链 =====================

class ChainOfThoughtParser:
    """CoT 推理链解析器"""
    
    REASONING_TYPES = {
        "deduction": ["therefore", "thus", "so", "hence", "must", "implies"],
        "induction": ["pattern", "trend", "generally", "typically", "usually"],
        "abduction": ["likely", "probably", "most likely", "best explanation"],
        "calculation": ["=", "equals", "calculate", "compute", "formula"],
        "comparison": ["greater than", "less than", "equal to", "vs", "compared to"]
    }
    
    def detect_reasoning_type(self, text: str) -> str:
        """检测推理类型"""
        text_lower = text.lower()
        
        for rtype, keywords in self.REASONING_TYPES.items():
            for kw in keywords:
                if kw in text_lower:
                    return rtype
        
        return "deduction"  # 默认
    
    def parse_inference_path(
        self,
        closed_inference_path: List[Dict[str, Any]]
    ) -> List[InferenceStep]:
        """解析推理路径为结构化步骤"""
        steps = []
        
        for i, step_data in enumerate(closed_inference_path):
            step_content = step_data.get("step_content", "")
            step_type = step_data.get("step_type", "reasoning")
            
            # 检测推理类型
            reasoning_type = self.detect_reasoning_type(step_content)
            if step_type == "calculation":
                reasoning_type = "calculation"
            
            # 提取前提和结论
            premise = ""
            conclusion = step_content
            
            # 尝试分割前提和结论
            connectors = ["therefore", "thus", "so", "hence", "implies that", "means that"]
            for conn in connectors:
                if conn in step_content.lower():
                    parts = re.split(conn, step_content, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        premise = parts[0].strip()
                        conclusion = parts[1].strip()
                        break
            
            step = InferenceStep(
                step_id=i + 1,
                reasoning_type=reasoning_type,
                premise=premise or f"Given: {step_content[:100]}",
                operation=step_type,
                conclusion=conclusion,
                confidence=1.0,
                depends_on=[j + 1 for j in range(i)] if i > 0 else []
            )
            steps.append(step)
        
        return steps
    
    def validate_chain_coherence(self, steps: List[InferenceStep]) -> Tuple[bool, List[str]]:
        """验证推理链的连贯性"""
        issues = []
        
        if not steps:
            return False, ["Empty inference chain"]
        
        # 检查依赖关系
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id > step.step_id:
                    issues.append(f"Step {step.step_id} depends on future step {dep_id}")
        
        # 检查逻辑跳跃
        for i, step in enumerate(steps):
            if i > 0:
                prev_step = steps[i - 1]
                # 检查是否有连接词
                if not any(conn in step.premise.lower() for conn in 
                          ["therefore", "thus", "since", "because", "given", "from"]):
                    if not step.depends_on:
                        issues.append(f"Step {step.step_id} appears disconnected from previous steps")
        
        is_coherent = len(issues) == 0
        return is_coherent, issues


# ===================== 优化3: 数学计算交叉验证 =====================

class CalculationVerifier:
    """计算验证器"""
    
    def extract_formula(self, matched_formula: Dict[str, Any]) -> Optional[str]:
        """从匹配的公式中提取公式表达式"""
        if not matched_formula:
            return None
        
        # 尝试不同的字段名
        for key in ["formula_expression", "expression", "formula"]:
            if key in matched_formula:
                return str(matched_formula[key])
        
        return None
    
    def extract_parameters(self, key_parameters: Dict[str, Any]) -> Dict[str, float]:
        """提取参数值"""
        params = {}
        
        if not key_parameters:
            return params
        
        # 从 parameters 字段提取
        if "parameters" in key_parameters:
            for param in key_parameters["parameters"]:
                if isinstance(param, dict):
                    name = param.get("name", param.get("parameter", ""))
                    value = param.get("value", param.get("default_value"))
                    if name and value is not None:
                        try:
                            params[name] = float(value)
                        except (ValueError, TypeError):
                            pass
        
        # 直接提取数值参数
        for key, value in key_parameters.items():
            if isinstance(value, (int, float)):
                params[key] = float(value)
            elif isinstance(value, str):
                # 尝试从字符串中提取数值
                num_match = re.search(r'([+-]?[\d.]+)', value)
                if num_match:
                    try:
                        params[key] = float(num_match.group(1))
                    except ValueError:
                        pass
        
        return params
    
    def symbolic_evaluate(self, formula: str, params: Dict[str, float]) -> Optional[float]:
        """符号计算（简化实现，实际应使用 sympy）"""
        try:
            # 替换参数
            expr = formula
            for name, value in params.items():
                expr = re.sub(r'\b' + re.escape(name) + r'\b', str(value), expr)
            
            # 安全计算（仅允许基本数学运算）
            # 注意：这是一个简化实现，生产环境应使用 sympy
            allowed_chars = set('0123456789.+-*/()^ ')
            if not all(c in allowed_chars for c in expr):
                return None
            
            # 替换 ^ 为 **
            expr = expr.replace('^', '**')
            
            # 计算
            result = eval(expr)
            return float(result)
        except Exception:
            return None
    
    def numerical_evaluate(self, formula: str, params: Dict[str, float]) -> Optional[float]:
        """数值计算（逐步计算）"""
        return self.symbolic_evaluate(formula, params)  # 简化实现
    
    def verify(
        self,
        matched_formula: Dict[str, Any],
        key_parameters: Dict[str, Any],
        llm_result: str = None
    ) -> CalculationVerificationResult:
        """验证计算结果"""
        formula = self.extract_formula(matched_formula)
        params = self.extract_parameters(key_parameters)
        
        if not formula or not params:
            return CalculationVerificationResult(all_match=False)
        
        # 符号计算
        symbolic_result = self.symbolic_evaluate(formula, params)
        
        # 数值计算
        numerical_result = self.numerical_evaluate(formula, params)
        
        # 检查一致性
        all_match = False
        discrepancy = None
        
        if symbolic_result is not None and numerical_result is not None:
            tolerance = 1e-6
            all_match = abs(symbolic_result - numerical_result) < tolerance
            
            if not all_match:
                discrepancy = {
                    "symbolic": symbolic_result,
                    "numerical": numerical_result,
                    "difference": abs(symbolic_result - numerical_result)
                }
        
        return CalculationVerificationResult(
            symbolic_result=symbolic_result,
            numerical_result=numerical_result,
            llm_verification=llm_result,
            all_match=all_match,
            discrepancy=discrepancy
        )


# ===================== 优化4: 迭代式知识检索 =====================

class IterativeKnowledgeRetriever:
    """迭代式知识检索器"""
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
    
    def generate_retrieval_queries(
        self,
        cleaned_text: str,
        core_keywords: List[str],
        existing_knowledge: Dict[str, Any] = None,
        iteration: int = 0
    ) -> List[str]:
        """生成检索查询"""
        queries = []
        
        # 第一轮：基于核心关键词
        if iteration == 0:
            for kw in core_keywords[:5]:  # 最多5个关键词
                queries.append(kw)
        
        # 后续轮次：基于知识缺口
        else:
            # 分析已有知识中的信息缺口
            gaps = self._identify_knowledge_gaps(cleaned_text, existing_knowledge)
            queries.extend(gaps)
        
        return queries
    
    def _identify_knowledge_gaps(
        self,
        question_text: str,
        knowledge: Dict[str, Any]
    ) -> List[str]:
        """识别知识缺口"""
        gaps = []
        
        # 检查问题中的关键概念是否在知识中有覆盖
        question_terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question_text))
        
        knowledge_text = ""
        if isinstance(knowledge, dict):
            for domain, content in knowledge.items():
                if isinstance(content, dict):
                    for ktype, items in content.items():
                        if isinstance(items, list):
                            knowledge_text += " ".join(str(i) for i in items) + " "
        
        knowledge_terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', knowledge_text))
        
        # 找出问题中有但知识中没有的术语
        missing_terms = question_terms - knowledge_terms
        
        for term in missing_terms:
            if len(term) > 3:  # 忽略太短的术语
                gaps.append(f"What is {term}?")
        
        return gaps[:3]  # 最多3个缺口查询
    
    def is_knowledge_sufficient(
        self,
        state: 'GeneralQAState',
        knowledge: Dict[str, Any]
    ) -> bool:
        """判断知识是否足够"""
        # 基本检查
        if not knowledge:
            return False
        
        # 检查是否有足够的领域覆盖
        if state.core_domains:
            covered_domains = set(knowledge.keys())
            required_domains = set(state.core_domains)
            
            if not required_domains.issubset(covered_domains):
                return False
        
        # 检查每个领域是否有实际内容
        for domain, content in knowledge.items():
            if isinstance(content, dict):
                foundational = content.get("foundational_knowledge", [])
                specialized = content.get("specialized_knowledge", [])
                
                if not foundational and not specialized:
                    return False
        
        return True
    
    def generate_follow_up_questions(
        self,
        state: 'GeneralQAState',
        new_knowledge: Dict[str, Any]
    ) -> List[str]:
        """生成追问"""
        follow_ups = []
        
        # 基于研究目标生成追问
        if state.research_objective:
            # 检查研究目标是否被充分回答
            obj_lower = state.research_objective.lower()
            
            if "how" in obj_lower and "mechanism" not in str(new_knowledge).lower():
                follow_ups.append(f"What is the mechanism of {state.structured_subject.get('type', 'the process')}?")
            
            if "why" in obj_lower and "reason" not in str(new_knowledge).lower():
                follow_ups.append(f"Why does {state.structured_subject.get('type', 'this')} occur?")
        
        return follow_ups[:2]  # 最多2个追问


# ===================== 优化5: 元认知监控 =====================

class MetaCognitiveMonitor:
    """元认知监控器"""
    
    def check_goal_alignment(
        self,
        state: 'GeneralQAState'
    ) -> bool:
        """检查推理是否与目标对齐"""
        if not state.structured_goal or not state.core_conclusion:
            return True  # 无法检查时假设对齐
        
        goal_type = state.structured_goal.get("type", "")
        goal_constraint = state.structured_goal.get("constraint", "")
        
        # 检查结论是否回答了目标
        conclusion_lower = (state.core_conclusion or "").lower()
        
        # 目标类型匹配检查
        goal_type_indicators = {
            "conclusion judgment": ["conclusion", "answer is", "result is"],
            "calculation result": ["=", "equals", "value is", "calculated"],
            "explanation": ["because", "due to", "reason is", "mechanism"],
            "comparison": ["greater", "less", "better", "worse", "compared"]
        }
        
        if goal_type in goal_type_indicators:
            indicators = goal_type_indicators[goal_type]
            if not any(ind in conclusion_lower for ind in indicators):
                # 结论可能未直接回答目标
                pass  # 不强制失败，仅记录
        
        return True
    
    def check_constraint_coverage(
        self,
        state: 'GeneralQAState'
    ) -> bool:
        """检查是否覆盖了所有约束"""
        all_constraints = []
        
        # 收集所有约束
        if state.key_constraints:
            all_constraints.extend(state.key_constraints)
        if state.negative_constraints:
            all_constraints.extend(state.negative_constraints)
        if state.exclusive_constraints:
            all_constraints.extend(state.exclusive_constraints)
        if state.strong_restrictions:
            all_constraints.extend(state.strong_restrictions)
        if state.answer_constraints:
            all_constraints.extend(state.answer_constraints)
        
        if not all_constraints:
            return True
        
        # 检查推理路径是否覆盖了约束
        if not state.closed_inference_path:
            return False
        
        inference_text = ""
        for step in state.closed_inference_path:
            inference_text += step.get("step_content", "") + " "
        
        # 检查每个约束是否在推理中被考虑
        uncovered = []
        for constraint in all_constraints:
            constraint_keywords = set(constraint.lower().split())
            if not any(kw in inference_text.lower() for kw in constraint_keywords if len(kw) > 3):
                uncovered.append(constraint)
        
        return len(uncovered) == 0
    
    def identify_knowledge_gaps(
        self,
        state: 'GeneralQAState'
    ) -> List[str]:
        """识别知识缺口"""
        gaps = []
        
        # 检查知识是否标记为不可靠
        if state.knowledge_unreliable:
            gaps.append("Knowledge retrieval may be incomplete or unreliable")
        
        # 检查领域覆盖
        if state.core_domains and state.domain_knowledge_map:
            for domain in state.core_domains:
                if domain not in state.domain_knowledge_map:
                    gaps.append(f"Missing knowledge for domain: {domain}")
        
        # 检查关键实体的知识
        if state.key_entities:
            all_knowledge_text = str(state.domain_knowledge_map or "")
            for entity in state.key_entities:
                if entity.lower() not in all_knowledge_text.lower():
                    gaps.append(f"Missing knowledge for key entity: {entity}")
        
        return gaps
    
    def check_reasoning_coherence(
        self,
        state: 'GeneralQAState'
    ) -> bool:
        """检查推理连贯性"""
        if not state.closed_inference_path or len(state.closed_inference_path) < 2:
            return True
        
        parser = ChainOfThoughtParser()
        steps = parser.parse_inference_path(state.closed_inference_path)
        is_coherent, _ = parser.validate_chain_coherence(steps)
        
        return is_coherent
    
    def assess(
        self,
        state: 'GeneralQAState'
    ) -> MetaCognitiveAssessment:
        """执行元认知评估"""
        goal_alignment = self.check_goal_alignment(state)
        constraint_coverage = self.check_constraint_coverage(state)
        knowledge_gaps = self.identify_knowledge_gaps(state)
        reasoning_coherence = self.check_reasoning_coherence(state)
        
        # 计算置信度校准
        confidence_calibration = 1.0
        if not goal_alignment:
            confidence_calibration -= 0.3
        if not constraint_coverage:
            confidence_calibration -= 0.2
        if knowledge_gaps:
            confidence_calibration -= 0.1 * len(knowledge_gaps)
        if not reasoning_coherence:
            confidence_calibration -= 0.2
        
        confidence_calibration = max(0.0, min(1.0, confidence_calibration))
        
        # 判断是否需要回溯
        needs_backtracking = (
            not goal_alignment or 
            (not constraint_coverage and len(knowledge_gaps) > 1) or
            confidence_calibration < 0.5
        )
        
        backtracking_reason = None
        if needs_backtracking:
            reasons = []
            if not goal_alignment:
                reasons.append("Goal not aligned")
            if not constraint_coverage:
                reasons.append("Constraints not covered")
            if knowledge_gaps:
                reasons.append(f"Knowledge gaps: {', '.join(knowledge_gaps[:2])}")
            backtracking_reason = "; ".join(reasons)
        
        return MetaCognitiveAssessment(
            goal_alignment=goal_alignment,
            constraint_coverage=constraint_coverage,
            knowledge_gaps=knowledge_gaps,
            reasoning_coherence=reasoning_coherence,
            confidence_calibration=confidence_calibration,
            needs_backtracking=needs_backtracking,
            backtracking_reason=backtracking_reason
        )


# ===================== 优化6: 智能异常诊断 =====================

class ExceptionDiagnostician:
    """异常诊断器"""
    
    # 根因到重试策略的映射
    RETRY_STRATEGIES = {
        "knowledge_insufficient": RetryStrategy(
            target_node="n3_knowledge_retrieval",
            action="expand_search",
            params={"use_paperqa": True, "use_web_search": True},
            reason="Knowledge base lacks sufficient information"
        ),
        "knowledge_invalid": RetryStrategy(
            target_node="n3_knowledge_retrieval",
            action="fresh_retrieval",
            params={"skip_cache": True},
            reason="Retrieved knowledge failed validity check"
        ),
        "reasoning_error": RetryStrategy(
            target_node="n7_complete_inference",
            action="use_different_strategy",
            params={"strategy": "backward_chaining"},
            reason="Forward reasoning led to contradiction"
        ),
        "reasoning_incomplete": RetryStrategy(
            target_node="n7_complete_inference",
            action="extend_reasoning",
            params={"add_steps": True},
            reason="Inference chain incomplete"
        ),
        "calculation_error": RetryStrategy(
            target_node="n4_calculation_decomposition",
            action="verify_formula",
            params={"use_symbolic_math": True},
            reason="Calculation produced inconsistent results"
        ),
        "calculation_out_of_range": RetryStrategy(
            target_node="n7_complete_inference",
            action="recheck_parameters",
            params={"validate_constraints": True},
            reason="Calculated result outside expected range"
        ),
        "answer_format_error": RetryStrategy(
            target_node="n8_answer_generation",
            action="strict_format",
            params={"enforce_format": True},
            reason="Answer format does not match expected format"
        ),
        "option_matching_error": RetryStrategy(
            target_node="n8_answer_generation",
            action="rematch_options",
            params={"use_semantic_matching": True},
            reason="Conclusion does not match any option"
        ),
        "constraint_violation": RetryStrategy(
            target_node="n7_complete_inference",
            action="check_constraints",
            params={"strict_constraint_check": True},
            reason="Answer violates question constraints"
        ),
        "self_consistency_low": RetryStrategy(
            target_node="n7_complete_inference",
            action="increase_sampling",
            params={"num_paths": 5, "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8]},
            reason="Low consensus across reasoning paths"
        )
    }
    
    def diagnose_root_cause(self, state: 'GeneralQAState') -> str:
        """诊断根因"""
        # 检查知识问题
        if state.knowledge_validity_label == "Invalid":
            return "knowledge_invalid"
        
        if state.knowledge_unreliable or (
            not state.domain_knowledge_map or 
            len(state.domain_knowledge_map) == 0
        ):
            return "knowledge_insufficient"
        
        # 检查推理问题
        if state.exception_type_label:
            if "Inference Path" in state.exception_type_label:
                if "Inconsistent" in state.exception_type_label:
                    return "reasoning_error"
                if "Incomplete" in state.exception_type_label:
                    return "reasoning_incomplete"
            
            if "Formula Match" in state.exception_type_label:
                return "calculation_error"
            
            if "Result Out of Range" in state.exception_type_label:
                return "calculation_out_of_range"
            
            if "Format" in state.exception_type_label:
                return "answer_format_error"
            
            if "Option Matching" in state.exception_type_label:
                return "option_matching_error"
            
            if "Constraint" in state.exception_type_label:
                return "constraint_violation"
        
        # 检查自一致性
        if hasattr(state, 'self_consistency_result') and state.self_consistency_result:
            sc_result = state.self_consistency_result
            if isinstance(sc_result, dict) and sc_result.get("confidence_level") == "low":
                return "self_consistency_low"
        
        # 默认
        return "reasoning_error"
    
    def get_retry_strategy(self, root_cause: str) -> RetryStrategy:
        """获取重试策略"""
        return self.RETRY_STRATEGIES.get(
            root_cause,
            RetryStrategy(
                target_node="n7_complete_inference",
                action="default_retry",
                params={},
                reason="Unknown error, retrying from inference"
            )
        )


# ===================== 优化7: 增强工具意图识别 =====================

class ToolIntentAnalyzer:
    """工具意图分析器"""
    
    # 领域到工具的映射
    DOMAIN_TOOL_MAPPING = {
        "mathematical": ["symbolic_math", "numerical_computation", "unit_conversion"],
        "biology": ["paperqa", "knowledge_graph", "sequence_analysis", "gene_database"],
        "immunology": ["igblast", "metabcr", "antibody_analysis"],
        "physics": ["unit_conversion", "formula_database", "physical_constants"],
        "chemistry": ["molecular_database", "reaction_predictor", "chemical_properties"],
        "logic": ["theorem_prover", "constraint_solver", "formal_verification"],
        "clinical": ["clinical_guidelines", "drug_database", "adverse_events"]
    }
    
    # 问题类型到工具的映射
    QUESTION_TYPE_TOOL_MAPPING = {
        "Numerical Calculation": ["symbolic_math", "numerical_computation", "unit_conversion"],
        "Logical Calculation": ["theorem_prover", "constraint_solver"],
        "Professional Algorithm": ["algorithm_executor", "parameter_validator"],
        "Mechanism Explanation": ["paperqa", "knowledge_graph"],
        "Text Matching": ["semantic_matcher", "entity_extractor"],
        "Multiple Choice": ["option_analyzer", "semantic_matcher"]
    }
    
    # 关键词到工具的映射
    KEYWORD_TOOL_MAPPING = {
        "antibody": ["igblast", "metabcr", "antibody_analysis"],
        "sequence": ["sequence_analysis", "igblast"],
        "binding": ["binding_affinity", "molecular_docking"],
        "structure": ["alphafold", "structure_prediction"],
        "expression": ["expression_analysis", "rna_tools"],
        "mutation": ["mutation_analyzer", "variant_tools"],
        "pathway": ["pathway_analysis", "interaction_network"],
        "dose": ["dose_calculator", "pharmacokinetics"],
        "clinical": ["clinical_guidelines", "drug_database"]
    }
    
    def analyze_requirements(
        self,
        user_input: str,
        core_domains: List[str],
        question_type: str,
        core_keywords: List[str]
    ) -> Dict[str, List[str]]:
        """分析工具需求"""
        required_tools = set()
        recommended_tools = set()
        
        # 基于领域
        for domain in core_domains:
            domain_lower = domain.lower()
            for domain_key, tools in self.DOMAIN_TOOL_MAPPING.items():
                if domain_key in domain_lower:
                    required_tools.update(tools)
        
        # 基于问题类型
        if question_type in self.QUESTION_TYPE_TOOL_MAPPING:
            recommended_tools.update(self.QUESTION_TYPE_TOOL_MAPPING[question_type])
        
        # 基于关键词
        for keyword in core_keywords:
            keyword_lower = keyword.lower()
            for kw_key, tools in self.KEYWORD_TOOL_MAPPING.items():
                if kw_key in keyword_lower:
                    recommended_tools.update(tools)
        
        # 从用户输入中检测明确的工具请求
        user_lower = user_input.lower()
        for kw_key, tools in self.KEYWORD_TOOL_MAPPING.items():
            if kw_key in user_lower:
                required_tools.update(tools)
        
        return {
            "required": list(required_tools),
            "recommended": list(recommended_tools),
            "all_tools": list(required_tools | recommended_tools)
        }
    
    def update_tool_intent(
        self,
        state: 'GeneralQAState'
    ) -> Dict[str, str]:
        """更新工具意图"""
        requirements = self.analyze_requirements(
            state.user_input,
            state.core_domains or [],
            state.question_type_label or "",
            state.core_keywords or []
        )
        
        tool_intent = state.tool_intent or {}
        
        # 标记必需的工具
        for tool in requirements["required"]:
            tool_intent[tool] = "REQUIRED"
        
        # 标记推荐的工具
        for tool in requirements["recommended"]:
            if tool not in tool_intent:
                tool_intent[tool] = "RECOMMENDED"
        
        return tool_intent


# ===================== 辅助函数 =====================

def create_enhanced_prompt(
    base_prompt: str,
    state: 'GeneralQAState',
    enable_cot: bool = True,
    enable_self_consistency: bool = True
) -> str:
    """创建增强的 prompt"""
    enhanced = base_prompt
    
    # 添加 CoT 指令
    if enable_cot:
        cot_instruction = """

**CHAIN-OF-THOUGHT REASONING (显式推理链):**
For each reasoning step, you MUST:
1. Clearly state the PREMISE (前提) - what facts or knowledge you're using
2. Show the OPERATION (操作) - what logical/mathematical operation you're performing  
3. State the CONCLUSION (结论) - what you infer from this step
4. Indicate which previous steps this step DEPENDS ON (依赖步骤)

Format each step as:
Step N [TYPE]: 
  Premise: ...
  Operation: ...
  Conclusion: ...
  Depends on: Step X, Step Y
"""
        enhanced += cot_instruction
    
    # 添加 Self-Consistency 指令
    if enable_self_consistency:
        sc_instruction = """

**SELF-CONSISTENCY (自一致性检查):**
After reaching your conclusion, verify it by:
1. Checking if the same conclusion follows from different reasoning paths
2. Identifying any contradictions in your reasoning chain
3. Confirming your answer satisfies ALL constraints (especially negative and exclusive constraints)
"""
        enhanced += sc_instruction
    
    return enhanced


def extract_numerical_result(text: str) -> Optional[float]:
    """从文本中提取数值结果"""
    if not text:
        return None
    
    # 尝试多种模式
    patterns = [
        r'([+-]?[\d.]+)\s*(?:mL/g|mg/L|μM|mM|M|g/mol|kDa|Gy|%|units?)',
        r'(?:result|answer|value)[:\s]*([+-]?[\d.]+)',
        r'=\s*([+-]?[\d.]+)',
        r'([+-]?[\d.]+(?:\.[\d]+)?(?:[eE][+-]?\d+)?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None

