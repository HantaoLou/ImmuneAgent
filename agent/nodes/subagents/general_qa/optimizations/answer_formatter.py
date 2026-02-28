"""
Answer Formatter
将答案自动转换为期望的格式

解决问题:
- 系统输出 "True" 但期望 "1 hour"
- 系统输出长解释但期望单字母选项
- 系统输出JSON格式但期望简单文本
"""

import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AnswerType(Enum):
    SINGLE_LETTER = "single_letter"      # A, B, C, D
    MULTIPLE_LETTERS = "multiple_letters"  # A, B, C
    TRUE_FALSE = "true_false"            # True/False
    NUMERIC = "numeric"                  # 75.6, 46.24
    SHORT_TEXT = "short_text"            # 简短文本
    LIST = "list"                        # 列表
    TUPLE_PAIR = "tuple_pair"            # (1,2,3), (4,5,6)
    EXPLANATION = "explanation"          # 解释性文本
    CODE = "code"                        # 代码
    FORMAT_SPECIFIC = "format_specific"  # 特定格式


@dataclass
class FormatRule:
    """格式化规则"""
    answer_type: AnswerType
    extraction_regex: str
    format_template: str
    max_length: int
    validation_regex: Optional[str] = None


# 预定义格式化规则
FORMAT_RULES = {
    # 单选题: 从文本中提取选项字母
    "single_letter": FormatRule(
        answer_type=AnswerType.SINGLE_LETTER,
        extraction_regex=r"\b([A-H])\b",
        format_template="{letter}",
        max_length=1,
        validation_regex=r"^[A-H]$"
    ),
    
    # True/False题
    "true_false": FormatRule(
        answer_type=AnswerType.TRUE_FALSE,
        extraction_regex=r"(True|False|Yes|No)",
        format_template="{answer}",
        max_length=10,
        validation_regex=r"^(True|False)$"
    ),
    
    # 数值题
    "numeric": FormatRule(
        answer_type=AnswerType.NUMERIC,
        extraction_regex=r"([-+]?\d+\.?\d*)",
        format_template="{number}",
        max_length=20,
        validation_regex=r"^[-+]?\d+\.?\d*%?$"
    ),
    
    # 列表题 (如 (1,2,3), (4,5,6))
    "tuple_pair": FormatRule(
        answer_type=AnswerType.TUPLE_PAIR,
        extraction_regex=r"\(([^)]+)\)",
        format_template="({content})",
        max_length=100
    ),
    
    # 多选题 (A, B, C)
    "multiple_letters": FormatRule(
        answer_type=AnswerType.MULTIPLE_LETTERS,
        extraction_regex=r"([A-H](?:\s*,\s*[A-H])*)",
        format_template="{letters}",
        max_length=50,
        validation_regex=r"^[A-H](\s*,\s*[A-H])*$"
    )
}


class AnswerFormatter:
    """答案格式化器"""
    
    def __init__(self):
        self.format_rules = FORMAT_RULES
        self._format_history: List[Dict[str, Any]] = []
    
    def detect_answer_type(self, answer: str) -> AnswerType:
        """
        检测答案类型
        """
        if not answer:
            return AnswerType.SHORT_TEXT
            
        answer_stripped = answer.strip()
        answer_upper = answer_stripped.upper()
        
        # True/False检测
        if answer_upper in ["TRUE", "FALSE"]:
            return AnswerType.TRUE_FALSE
        if answer_upper in ["YES", "NO"]:
            return AnswerType.TRUE_FALSE
        
        # 单字母检测 (纯A-H)
        if re.match(r"^[A-H]$", answer_upper):
            return AnswerType.SINGLE_LETTER
        
        # 多字母检测 (A, B, C)
        if re.match(r"^[A-H](\s*,\s*[A-H])*$", answer_upper):
            return AnswerType.MULTIPLE_LETTERS
        
        # 数值检测
        if re.match(r"^[-+]?\d+\.?\d*%?$", answer_stripped):
            return AnswerType.NUMERIC
        
        # 元组对检测
        if re.match(r"^\([^)]+\)\s*,\s*\([^)]+\)$", answer_stripped):
            return AnswerType.TUPLE_PAIR
        
        # 单个元组检测
        if re.match(r"^\([^)]+\)$", answer_stripped):
            return AnswerType.LIST
        
        # 代码检测
        if any(kw in answer for kw in ["def ", "import ", "function", "class ", "```"]):
            return AnswerType.CODE
        
        # 长文本检测
        if len(answer) > 200:
            return AnswerType.EXPLANATION
        
        return AnswerType.SHORT_TEXT
    
    def format_answer(
        self,
        raw_answer: str,
        expected_answer: Optional[str] = None,
        question: str = ""
    ) -> str:
        """
        格式化答案
        
        Args:
            raw_answer: 原始答案
            expected_answer: 期望答案 (用于格式参考)
            question: 问题文本
            
        Returns:
            格式化后的答案
        """
        if not raw_answer:
            return ""
        
        raw_answer = str(raw_answer).strip()
        
        # 检测原始答案类型
        raw_type = self.detect_answer_type(raw_answer)
        
        # 如果提供了期望答案，尝试匹配格式
        if expected_answer:
            expected_type = self.detect_answer_type(expected_answer)
            
            # 如果原始是解释，尝试提取
            if raw_type == AnswerType.EXPLANATION:
                extracted = self._extract_from_explanation(
                    raw_answer, expected_type, expected_answer
                )
                if extracted:
                    self._record_format(raw_answer, extracted, raw_type, expected_type)
                    return extracted
            
            # 如果类型不同但可以转换
            if raw_type != expected_type:
                converted = self._try_convert(raw_answer, raw_type, expected_type)
                if converted:
                    self._record_format(raw_answer, converted, raw_type, expected_type)
                    return converted
        
        # 尝试应用格式规则
        formatted = self._apply_format_rules(raw_answer, question)
        if formatted != raw_answer:
            self._record_format(raw_answer, formatted, raw_type, raw_type)
            return formatted
        
        return raw_answer
    
    def _extract_from_explanation(
        self,
        explanation: str,
        target_type: AnswerType,
        expected_answer: str
    ) -> Optional[str]:
        """
        从解释性文本中提取答案
        """
        explanation_lower = explanation.lower()
        
        if target_type == AnswerType.SINGLE_LETTER:
            # 策略1: 查找 "答案是 X" 模式
            patterns = [
                r"答案[是为]\s*[:：]?\s*([A-H])",
                r"answer[:\s]+([A-H])",
                r"选择[:\s]*([A-H])",
                r"option[:\s]*([A-H])",
                r"correct[:\s]+([A-H])"
            ]
            for pattern in patterns:
                match = re.search(pattern, explanation, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
            
            # 策略2: 查找最后出现的选项字母（排除选项描述中的）
            # 排除 "A." "B)" 等选项标记，只找结论中的字母
            conclusion_markers = ["therefore", "thus", "所以", "因此", "结论", "conclusion"]
            for marker in conclusion_markers:
                if marker in explanation_lower:
                    marker_pos = explanation_lower.find(marker)
                    after_marker = explanation[marker_pos:]
                    letters = re.findall(r"\b([A-H])\b", after_marker)
                    if letters:
                        return letters[-1].upper()
            
            # 策略3: 找所有独立字母，取最后一个
            letters = re.findall(r"\b([A-H])\b", explanation)
            if letters:
                return letters[-1].upper()
        
        elif target_type == AnswerType.TRUE_FALSE:
            # 提取True/False
            if re.search(r"\b(true|yes|正确|对)\b", explanation_lower):
                # 但要检查是否有"not true"之类的否定
                if not re.search(r"not\s+(true|correct)|不正确|错误|false", explanation_lower):
                    return "True"
            if re.search(r"\b(false|no|错误|不对)\b", explanation_lower):
                return "False"
        
        elif target_type == AnswerType.NUMERIC:
            # 提取数值
            # 尝试匹配期望答案的格式
            if "%" in expected_answer:
                match = re.search(r"(\d+\.?\d*)\s*%", explanation)
                if match:
                    return f"{float(match.group(1))}%"
            else:
                # 尝试找到独立的数值
                numbers = re.findall(r"\b([-+]?\d+\.?\d*)\b", explanation)
                if numbers:
                    # 尝试找到与期望答案最接近的数值
                    try:
                        expected_num = float(re.search(r"[-+]?\d+\.?\d*", expected_answer).group())
                        closest = min(numbers, key=lambda x: abs(float(x) - expected_num))
                        return closest
                    except (ValueError, AttributeError):
                        return numbers[-1]
        
        elif target_type == AnswerType.TUPLE_PAIR:
            # 提取元组对格式 (1,2,3), (4,5,6)
            tuples = re.findall(r"\(([^)]+)\)", explanation)
            if len(tuples) >= 2:
                return f"({tuples[0]}), ({tuples[1]})"
            elif len(tuples) == 1:
                return f"({tuples[0]})"
        
        elif target_type == AnswerType.MULTIPLE_LETTERS:
            # 提取多个字母
            all_letters = re.findall(r"\b([A-H])\b", explanation.upper())
            if all_letters:
                # 去重并保持顺序
                seen = set()
                unique_letters = []
                for l in all_letters:
                    if l not in seen:
                        seen.add(l)
                        unique_letters.append(l)
                return ", ".join(unique_letters)
        
        return None
    
    def _try_convert(
        self,
        answer: str,
        from_type: AnswerType,
        to_type: AnswerType
    ) -> Optional[str]:
        """尝试类型转换"""
        # 单字母 -> 多字母 (已经是单字母，不需要转换)
        if from_type == AnswerType.SINGLE_LETTER and to_type == AnswerType.MULTIPLE_LETTERS:
            return answer.upper()
        
        # 数值 -> 带百分号的数值
        if from_type == AnswerType.NUMERIC and to_type == AnswerType.NUMERIC:
            if "%" not in answer:
                return f"{answer}%"
        
        return None
    
    def _apply_format_rules(self, answer: str, question: str) -> str:
        """应用格式化规则"""
        # 检测问题类型
        question_lower = question.lower()
        
        # 多选题
        if "answer choices:" in question_lower:
            letters = re.findall(r"\b([A-H])\b", answer.upper())
            if letters:
                return letters[-1]
        
        return answer
    
    def _record_format(
        self,
        original: str,
        formatted: str,
        original_type: AnswerType,
        target_type: AnswerType
    ):
        """记录格式化历史"""
        self._format_history.append({
            "original": original[:100] if len(original) > 100 else original,
            "formatted": formatted,
            "original_type": original_type.value,
            "target_type": target_type.value
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取格式化统计"""
        if not self._format_history:
            return {"total": 0}
        
        type_conversions = {}
        for record in self._format_history:
            key = f"{record['original_type']} -> {record['target_type']}"
            type_conversions[key] = type_conversions.get(key, 0) + 1
        
        return {
            "total": len(self._format_history),
            "type_conversions": type_conversions
        }


# 便捷函数
_global_formatter: Optional[AnswerFormatter] = None


def get_formatter() -> AnswerFormatter:
    """获取全局格式化器"""
    global _global_formatter
    if _global_formatter is None:
        _global_formatter = AnswerFormatter()
    return _global_formatter


def format_answer(
    raw_answer: str,
    expected_answer: Optional[str] = None,
    question: str = ""
) -> str:
    """格式化答案的便捷函数"""
    formatter = get_formatter()
    return formatter.format_answer(raw_answer, expected_answer, question)


def detect_type(answer: str) -> str:
    """检测答案类型的便捷函数"""
    formatter = get_formatter()
    return formatter.detect_answer_type(answer).value


