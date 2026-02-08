"""
LLM Output JSON Format Auto-Fix Module

Three-level format repair strategy to handle non-standard JSON outputs from LLM,
reducing process interruption due to format errors.
"""

import json
import re
from typing import Dict, Any, Optional, Tuple, List


def fix_json_format(response_text: str, required_keys: Optional[List[str]] = None) -> Tuple[Dict[str, Any], str]:
    """
    Three-level JSON format repair strategy
    
    Args:
        response_text: LLM returned text (may contain non-standard JSON)
        required_keys: List of required keys (for validation)
    
    Returns:
        Tuple of (parsed_dict: Dict[str, Any], repair_level: str)
        repair_level can be: "none", "basic", "intermediate", "advanced"
    """
    # Level 1: Basic repair - try standard JSON parsing with lenient settings
    try:
        result = json.loads(response_text.strip())
        if isinstance(result, dict):
            return result, "none"
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON code blocks first
    json_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    return result, "basic"
            except json.JSONDecodeError:
                continue
    
    # Level 2: Intermediate repair - fix common JSON issues
    fixed_text = _intermediate_fix(response_text)
    try:
        result = json.loads(fixed_text)
        if isinstance(result, dict):
            return result, "intermediate"
    except json.JSONDecodeError:
        pass
    
    # Level 3: Advanced repair - extract and reconstruct JSON
    fixed_text = _advanced_fix(response_text, required_keys)
    try:
        result = json.loads(fixed_text)
        if isinstance(result, dict):
            return result, "advanced"
    except json.JSONDecodeError:
        pass
    
    # If all repairs fail, raise error
    raise ValueError(f"Failed to parse JSON after all repair attempts. Raw response: {response_text[:200]}")


def _intermediate_fix(text: str) -> str:
    """Intermediate repair: fix common JSON syntax issues"""
    fixed = text.strip()
    
    # Fix 1: Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    
    # Fix 2: Replace single quotes with double quotes (for keys and string values)
    # But be careful with apostrophes in text
    fixed = re.sub(r"'(\w+)':", r'"\1":', fixed)  # Keys
    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)  # String values
    
    # Fix 3: Fix unquoted keys
    fixed = re.sub(r'(\w+):', r'"\1":', fixed)
    
    # Fix 4: Fix newlines in string values (replace with \n)
    fixed = re.sub(r'"([^"]*)\n([^"]*)"', r'"\1\\n\2"', fixed)
    
    # Fix 5: Remove comments (JSON doesn't support comments)
    fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    
    return fixed


def _advanced_fix(text: str, required_keys: Optional[List[str]] = None) -> str:
    """Advanced repair: extract JSON structure and reconstruct"""
    # Try to extract the first complete JSON object
    brace_count = 0
    start_idx = -1
    json_str = ""
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx:i+1]
                break
    
    if not json_str:
        # Fallback: try to find any JSON-like structure
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
    
    if not json_str:
        # Last resort: construct minimal JSON with required keys
        if required_keys:
            json_str = "{" + ", ".join(f'"{key}": ""' for key in required_keys) + "}"
        else:
            json_str = "{}"
    
    # Apply intermediate fixes
    return _intermediate_fix(json_str)


def generate_format_fix_prompt(original_response: str, required_keys: List[str], error_message: str) -> str:
    """
    Generate format correction prompt for LLM retry
    
    Args:
        original_response: Original LLM response with format errors
        required_keys: List of required keys
        error_message: Error message from parsing attempt
    
    Returns:
        Format correction prompt string
    """
    return f"""Your previous JSON output had format errors: {error_message}

Required keys: {required_keys}

Please correct the JSON format and ensure:
1. All keys are in double quotes
2. No trailing commas
3. All string values are properly quoted
4. All required keys are present

Original output (for reference):
{original_response[:500]}

Now output the corrected JSON only, no extra text."""

