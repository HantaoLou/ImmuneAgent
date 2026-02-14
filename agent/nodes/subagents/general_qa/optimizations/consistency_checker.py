"""
P1: Self-Consistency Checker

Validates answers using self-consistency across multiple reasoning paths:
- Generates multiple reasoning paths
- Compares conclusions
- Calculates agreement rate
- Flags low-confidence answers
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class ConsistencyLevel(Enum):
    """Consistency levels"""
    HIGH = "high"          # >80% agreement
    MEDIUM = "medium"      # 60-80% agreement
    LOW = "low"            # <60% agreement
    UNKNOWN = "unknown"


@dataclass
class ReasoningPath:
    """Single reasoning path"""
    path_id: int
    steps: List[str]
    conclusion: str
    confidence: float = 0.0
    reasoning_type: str = "standard"


@dataclass
class ConsistencyResult:
    """Result of consistency check"""
    paths: List[ReasoningPath]
    agreement_rate: float
    consistency_level: ConsistencyLevel
    dominant_conclusion: str
    minority_conclusions: List[str]
    warnings: List[str]


class SelfConsistencyChecker:
    """
    Check self-consistency across multiple reasoning paths
    """
    
    def __init__(self, num_paths: int = 3):
        self.num_paths = num_paths
    
    def check_consistency(self, 
                          state: Any,
                          llm: Any,
                          prompt_generator: callable = None) -> ConsistencyResult:
        """
        Generate multiple reasoning paths and check consistency
        
        Args:
            state: Current state object
            llm: LLM instance
            prompt_generator: Optional function to generate prompts
        
        Returns:
            ConsistencyResult with agreement analysis
        """
        paths = []
        
        for i in range(self.num_paths):
            # 生成不同的推理路径
            path = self._generate_reasoning_path(state, llm, i, prompt_generator)
            paths.append(path)
        
        # 计算一致性
        agreement_rate, dominant, minorities = self._calculate_agreement(paths)
        
        # 确定一致性级别
        if agreement_rate >= 0.8:
            level = ConsistencyLevel.HIGH
        elif agreement_rate >= 0.6:
            level = ConsistencyLevel.MEDIUM
        else:
            level = ConsistencyLevel.LOW
        
        # 生成警告
        warnings = []
        if level == ConsistencyLevel.LOW:
            warnings.append(f"Low self-consistency: only {agreement_rate*100:.1f}% agreement")
            warnings.append("Consider manual review or additional verification")
        if len(minorities) > 1:
            warnings.append(f"Multiple conflicting conclusions: {minorities}")
        
        return ConsistencyResult(
            paths=paths,
            agreement_rate=agreement_rate,
            consistency_level=level,
            dominant_conclusion=dominant,
            minority_conclusions=minorities,
            warnings=warnings
        )
    
    def _generate_reasoning_path(self, 
                                  state: Any,
                                  llm: Any,
                                  path_id: int,
                                  prompt_generator: callable = None) -> ReasoningPath:
        """Generate a single reasoning path"""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            # 使用不同的采样策略
            temperature = 0.3 + (path_id * 0.1)  # 略微增加温度
            
            # 生成prompt
            if prompt_generator:
                prompt = prompt_generator(state, path_id)
            else:
                prompt = self._default_prompt(state, path_id)
            
            # 调用LLM
            messages = [
                SystemMessage(content="You are an expert reasoning system. Provide step-by-step reasoning."),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # 解析响应
            steps, conclusion = self._parse_reasoning_response(response_text)
            
            return ReasoningPath(
                path_id=path_id,
                steps=steps,
                conclusion=conclusion,
                reasoning_type=f"path_{path_id}"
            )
        
        except Exception as e:
            return ReasoningPath(
                path_id=path_id,
                steps=[f"Error generating path: {e}"],
                conclusion="",
                reasoning_type="error"
            )
    
    def _default_prompt(self, state: Any, path_id: int) -> str:
        """Generate default prompt for reasoning"""
        question = getattr(state, 'cleaned_text', '') or getattr(state, 'user_input', '')
        options = getattr(state, 'question_options', [])
        
        prompt = f"""
Question: {question}

"""
        
        if options:
            prompt += "Options:\n"
            for opt_id, opt_text in enumerate(options, 1):
                prompt += f"{opt_id}. {opt_text}\n"
            prompt += "\n"
        
        prompt += f"""
Please reason step by step and provide your answer.
Reasoning path ID: {path_id + 1} (using slightly different reasoning approach)
"""
        
        return prompt
    
    def _parse_reasoning_response(self, response: str) -> Tuple[List[str], str]:
        """Parse reasoning response into steps and conclusion"""
        steps = []
        conclusion = ""
        
        # 提取步骤
        step_pattern = r'(?:Step|step)\s*\d+[:.)]?\s*([^\n]+)'
        steps = re.findall(step_pattern, response)
        
        # 如果没有明确的步骤，按行分割
        if not steps:
            lines = response.strip().split('\n')
            steps = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        # 提取结论
        conclusion_patterns = [
            r'(?:therefore|thus|conclusion|answer|final answer)[,:]?\s*([^\n.]+)',
            r'(?:the answer is|I choose|I select)\s*:?\s*([^\n.]+)',
            r'(?:option|answer)\s*[:=]?\s*([A-Ea-e]|[0-9]+)'
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                conclusion = match.group(1).strip()
                break
        
        # 如果没找到，取最后一行
        if not conclusion and steps:
            conclusion = steps[-1]
        
        return steps, conclusion
    
    def _calculate_agreement(self, 
                             paths: List[ReasoningPath]) -> Tuple[float, str, List[str]]:
        """Calculate agreement rate across paths"""
        conclusions = [p.conclusion for p in paths if p.conclusion]
        
        if not conclusions:
            return 0.0, "", []
        
        # 标准化结论
        normalized = [self._normalize_conclusion(c) for c in conclusions]
        
        # 统计频率
        conclusion_counts = {}
        for c in normalized:
            conclusion_counts[c] = conclusion_counts.get(c, 0) + 1
        
        # 找到主导结论
        dominant = max(conclusion_counts.keys(), key=lambda x: conclusion_counts[x])
        agreement_rate = conclusion_counts[dominant] / len(normalized)
        
        # 找到少数结论
        minorities = [c for c in conclusion_counts.keys() if c != dominant]
        
        return agreement_rate, dominant, minorities
    
    def _normalize_conclusion(self, conclusion: str) -> str:
        """Normalize conclusion for comparison"""
        # 提取选项字母
        match = re.search(r'\b([A-Ea-e])\b', conclusion)
        if match:
            return match.group(1).upper()
        
        # 提取数字
        match = re.search(r'\b(\d+)\b', conclusion)
        if match:
            return match.group(1)
        
        # 标准化文本
        normalized = conclusion.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized[:50]  # 截断


def validate_answer_plausibility(state: Any) -> List[str]:
    """
    Validate answer plausibility
    
    Args:
        state: Current state object
    
    Returns:
        List of validation issues
    """
    issues = []
    
    # 1. 检查答案格式
    answer_format = getattr(state, 'answer_format_label', None)
    final_answer = getattr(state, 'final_answer', None)
    options = getattr(state, 'question_options', [])
    
    if answer_format == "Single Choice" and options:
        if final_answer not in [str(i) for i in range(len(options))]:
            # 检查是否是字母格式
            if final_answer and final_answer.upper() not in [chr(65+i) for i in range(len(options))]:
                issues.append(f"Answer '{final_answer}' not in valid options")
    
    # 2. 检查数值答案范围
    if answer_format == "Numerical":
        if final_answer:
            try:
                value = float(re.search(r'[\d.]+', str(final_answer)).group())
                param_constraints = getattr(state, 'parameter_constraints', {})
                
                for param_name, constraints in param_constraints.items():
                    if isinstance(constraints, dict) and 'range' in constraints:
                        min_val = constraints['range'].get('min')
                        max_val = constraints['range'].get('max')
                        
                        if min_val is not None and value < min_val:
                            issues.append(f"Value {value} below minimum {min_val}")
                        if max_val is not None and value > max_val:
                            issues.append(f"Value {value} above maximum {max_val}")
            except (AttributeError, ValueError):
                pass
    
    # 3. 检查推理路径支持
    core_conclusion = getattr(state, 'core_conclusion', None)
    closed_path = getattr(state, 'closed_inference_path', [])
    
    if core_conclusion and closed_path:
        # 检查结论是否被推理路径支持
        path_supports = False
        conclusion_keywords = set(re.findall(r'\b\w+\b', core_conclusion.lower()))
        
        for step in closed_path:
            step_text = ""
            if isinstance(step, dict):
                step_text = step.get('conclusion', '') or step.get('inference', '')
            else:
                step_text = str(step)
            
            step_keywords = set(re.findall(r'\b\w+\b', step_text.lower()))
            overlap = conclusion_keywords & step_keywords
            
            if len(overlap) >= 3:  # 至少3个关键词重叠
                path_supports = True
                break
        
        if not path_supports:
            issues.append("Conclusion not well-supported by reasoning path")
    
    # 4. 检查知识置信度
    domain_knowledge = getattr(state, 'domain_knowledge_map', {})
    if not domain_knowledge:
        issues.append("No domain knowledge retrieved")
    
    return issues


def calculate_agreement(conclusions: List[str]) -> float:
    """
    Calculate agreement rate among conclusions
    
    Args:
        conclusions: List of conclusion strings
    
    Returns:
        Agreement rate (0.0 to 1.0)
    """
    if not conclusions:
        return 0.0
    
    # 标准化
    normalized = []
    for c in conclusions:
        c_lower = c.lower().strip()
        # 提取选项字母
        match = re.search(r'\b([a-e])\b', c_lower)
        if match:
            normalized.append(match.group(1).upper())
        else:
            normalized.append(c_lower[:30])
    
    # 统计
    counts = {}
    for c in normalized:
        counts[c] = counts.get(c, 0) + 1
    
    if not counts:
        return 0.0
    
    max_count = max(counts.values())
    return max_count / len(normalized)


def check_mcq_consistency(options: Dict[str, str],
                           conclusions: List[str]) -> Tuple[str, float]:
    """
    Check consistency for MCQ answers
    
    Args:
        options: Dict of option_id to option_text
        conclusions: List of conclusions from different paths
    
    Returns:
        Tuple of (most_common_answer, agreement_rate)
    """
    if not conclusions:
        return None, 0.0
    
    # 提取选项
    extracted = []
    for conclusion in conclusions:
        # 尝试匹配选项ID
        for opt_id in options.keys():
            if opt_id.lower() in conclusion.lower():
                extracted.append(opt_id.upper())
                break
        else:
            # 尝试匹配选项内容
            for opt_id, opt_text in options.items():
                if any(word in conclusion.lower() for word in opt_text.lower().split()[:3]):
                    extracted.append(opt_id.upper())
                    break
    
    if not extracted:
        return None, 0.0
    
    # 统计
    counts = {}
    for ans in extracted:
        counts[ans] = counts.get(ans, 0) + 1
    
    most_common = max(counts.keys(), key=lambda x: counts[x])
    agreement = counts[most_common] / len(extracted)
    
    return most_common, agreement

