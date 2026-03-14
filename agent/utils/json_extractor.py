"""
通用 JSON 提取工具函数

用于从 LLM 响应中提取 JSON 内容，支持多种格式和边界情况。
"""

import json
import re
from typing import Optional, Dict, Any, Union


def extract_json_from_llm_response(
    text: str, 
    default: Optional[Dict[str, Any]] = None,
    log_errors: bool = True
) -> Optional[Dict[str, Any]]:
    """
    从 LLM 响应中提取 JSON 对象，支持多种格式和边界情况。
    
    Args:
        text: LLM 响应文本
        default: 解析失败时返回的默认值（None 表示返回空字典）
        log_errors: 是否记录错误日志
        
    Returns:
        解析后的 JSON 字典，如果解析失败则返回 default 或空字典
    """
    if not text or not isinstance(text, str):
        if log_errors:
            print(f"[WARN] JSON extractor: Empty or invalid input text")
        return default if default is not None else {}
    
    text = text.strip()
    
    # 如果输入为空或只包含空白字符，直接返回默认值
    if not text:
        if log_errors:
            print(f"[WARN] JSON extractor: Input text is empty after stripping")
        return default if default is not None else {}
    
    # 移除 thinking 标签（GLM 原生思考模式）
    thinking_patterns = [
        r"<think[^>]*>.*?</think\s*>",
        r"<thinking[^>]*>.*?</thinking\s*>",
        r"<reasoning[^>]*>.*?</reasoning\s*>",
    ]
    for pattern in thinking_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    
    # 策略 1: 如果文本直接以 { 开头，尝试直接解析
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if log_errors:
                print(f"[WARN] JSON extractor: Direct parse failed: {e}")
            # 继续尝试其他策略
    
    # 策略 2: 尝试从代码块中提取 JSON
    json_block_patterns = [
        r"```json\s*\n?(.*?)\n?```",
        r"```\s*\n?(.*?)\n?```",
    ]
    for pattern in json_block_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            if json_str:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    if log_errors:
                        print(f"[WARN] JSON extractor: Code block parse failed: {e}")
                    continue
    
    # 策略 3: 使用括号匹配提取完整的 JSON 对象
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        json_end = -1
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_end = i
                    break
        
        if json_end > brace_start:
            json_str = text[brace_start : json_end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                if log_errors:
                    print(f"[WARN] JSON extractor: Brace matching parse failed: {e}")
    
    # 策略 4: 使用正则表达式提取（非贪婪匹配，匹配第一个完整的 JSON 对象）
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            if log_errors:
                print(f"[WARN] JSON extractor: Regex match parse failed: {e}")
    
    # 所有策略都失败
    if log_errors:
        print(f"[WARN] JSON extractor: All extraction strategies failed")
        print(f"[WARN] JSON extractor: Response preview (first 200 chars): {text[:200]}")
    
    return default if default is not None else {}


def safe_json_loads(
    text: str,
    default: Optional[Dict[str, Any]] = None,
    log_errors: bool = True
) -> Optional[Dict[str, Any]]:
    """
    安全地解析 JSON 字符串，如果失败则返回默认值。
    
    Args:
        text: JSON 字符串
        default: 解析失败时返回的默认值
        log_errors: 是否记录错误日志
        
    Returns:
        解析后的 JSON 字典，如果解析失败则返回 default
    """
    if not text or not isinstance(text, str):
        if log_errors:
            print(f"[WARN] safe_json_loads: Empty or invalid input")
        return default if default is not None else {}
    
    text = text.strip()
    if not text:
        if log_errors:
            print(f"[WARN] safe_json_loads: Input is empty after stripping")
        return default if default is not None else {}
    
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        elif isinstance(result, list):
            # 如果是数组，包装成字典
            return {"items": result}
        else:
            if log_errors:
                print(f"[WARN] safe_json_loads: Result is not a dict or list: {type(result)}")
            return default if default is not None else {}
    except json.JSONDecodeError as e:
        if log_errors:
            print(f"[WARN] safe_json_loads: JSON decode error: {e}")
            print(f"[WARN] safe_json_loads: Input preview (first 200 chars): {text[:200]}")
        return default if default is not None else {}

