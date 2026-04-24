"""
Generic JSON extraction utility functions

Used to extract JSON content from LLM responses, supporting multiple formats and edge cases.
"""

import json
import re
from typing import Optional, Dict, Any, Union


def extract_json_from_llm_response(
    text: str, default: Optional[Dict[str, Any]] = None, log_errors: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from LLM response, supporting multiple formats and edge cases.

    Args:
        text: LLM response text
        default: Default value to return on parse failure (None means return empty dict)
        log_errors: Whether to log errors

    Returns:
        Parsed JSON dict, or default/empty dict on failure
    """
    if not text or not isinstance(text, str):
        if log_errors:
            print(f"[WARN] JSON extractor: Empty or invalid input text")
        return default if default is not None else {}

    text = text.strip()

    if not text:
        if log_errors:
            print(f"[WARN] JSON extractor: Input text is empty after stripping")
        return default if default is not None else {}

    # Remove thinking tags (GLM native thinking mode)
    thinking_patterns = [
        r"<think[^>]*>.*?</think\s*>",
        r"<thinking[^>]*>.*?</thinking\s*>",
        r"<reasoning[^>]*>.*?</reasoning\s*>",
    ]
    for pattern in thinking_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()

    # Strategy 1: If text starts with {, try direct parsing
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if log_errors:
                print(f"[WARN] JSON extractor: Direct parse failed: {e}")

    # Strategy 2: Try extracting JSON from code blocks
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

    # Strategy 3: Use brace matching to extract complete JSON object
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

    # Strategy 4: Use regex extraction (non-greedy, matches first complete JSON object)
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            if log_errors:
                print(f"[WARN] JSON extractor: Regex match parse failed: {e}")

    # All strategies failed
    if log_errors:
        print(f"[WARN] JSON extractor: All extraction strategies failed")
        print(
            f"[WARN] JSON extractor: Response preview (first 200 chars): {text[:200]}"
        )

    return default if default is not None else {}


def safe_json_loads(
    text: str, default: Optional[Dict[str, Any]] = None, log_errors: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string, returning default value on failure.

    Args:
        text: JSON string
        default: Default value to return on parse failure
        log_errors: Whether to log errors

    Returns:
        Parsed JSON dict, or default on failure
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
            # If array, wrap in dict
            return {"items": result}
        else:
            if log_errors:
                print(
                    f"[WARN] safe_json_loads: Result is not a dict or list: {type(result)}"
                )
            return default if default is not None else {}
    except json.JSONDecodeError as e:
        if log_errors:
            print(f"[WARN] safe_json_loads: JSON decode error: {e}")
            print(
                f"[WARN] safe_json_loads: Input preview (first 200 chars): {text[:200]}"
            )
        return default if default is not None else {}
