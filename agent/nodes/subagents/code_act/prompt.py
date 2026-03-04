"""
CodeAct Agent Prompt Module

Centralized management of all code generation related prompt templates.
"""

from typing import List, Dict, Any, Optional
import json


# ===================== MCP Tool Call Code Generation =====================

MCP_TOOL_CODE_SYSTEM_PROMPT = """You are a professional Python code generation expert, specializing in generating code to call MCP (Model Context Protocol) tools.

# Your Responsibilities

Based on the provided MCP tool information and parameters, generate Python code that correctly calls the tool.

# Code Generation Requirements

1. **Code must be directly executable**, use the provided `call_tool()` interface to call MCP tools
2. **Use the provided parameters**, ensure parameter types and formats are correct
3. **Handle possible errors**, add appropriate error handling
4. **Code should be concise and clear**, include necessary comments
5. **Must set result variable at the end**, containing execution results
6. **LLM decides WHAT, code handles HOW data transforms** (keep tool calls simple, move data handling into code)

# MCP Tool Call Method

**Important: Must use `core.tool_interface.call_tool` function to call MCP tools.**

Standard call pattern:
```python
from core.tool_interface import call_tool

# Call MCP tool
tool_result = call_tool(
    tool_name="tool_name",
    parameters={
        "param1": "value1",
        "param2": "value2"
    }
)

# Process result
if tool_result["status"] == "success":
    result = {
        "status": "success",
        "output": tool_result["output"]
    }
else:
    result = {
        "status": "failed",
        "error": tool_result["error"]
    }
```

# Output Format

You must only return executable Python code, do not include any explanations, markdown code block markers, or other text.
Code should:
- Import `from core.tool_interface import call_tool`
- Use `call_tool` to call MCP tools
- Process execution results
- Set result variable (dictionary format, containing status, output, etc.)

# Notes

- **Must use `call_tool` function, do not try to directly import `mcp_client` or other non-existent modules**
- Parameter values should directly use provided parameters, do not modify
- If parameters contain file paths, ensure proper path handling
- `call_tool` returns a dictionary containing `status`, `output`, `error`, `error_type`, `execution_time_ms`, `tool_name`, and `service_id`
"""


def get_mcp_tool_code_user_prompt(
    tool_name: str,
    tool_description: str,
    parameters: Dict[str, Any],
    task_description: str
) -> str:
    """
    Generate user prompt for MCP tool call code
    
    Args:
        tool_name: Tool name
        tool_description: Tool description
        parameters: Tool parameters
        task_description: Task description
    
    Returns:
        User prompt
    """
    params_str = json.dumps(parameters, ensure_ascii=False, indent=2)
    
    return f"""Please generate Python code to call the following MCP tool:

# Tool Information
Tool name: {tool_name}
Tool description: {tool_description}

# Task Description
{task_description}

# Parameters
{params_str}

# Requirements
1. Use `from core.tool_interface import call_tool` to import MCP tool call function
2. Use `call_tool(tool_name="{tool_name}", parameters=<parameter_dict>)` to call the tool
3. Process execution result (check `tool_result["status"]`)
4. Set result variable in the following format:
   result = {{
       "status": "success" or "failed",
       "output": <execution_result>,
       "error": <error_message, if any>
   }}

# Example Code Structure
```python
from core.tool_interface import call_tool

tool_result = call_tool(
    tool_name="{tool_name}",
    parameters={params_str}
)

if tool_result["status"] == "success":
    result = {{"status": "success", "output": tool_result["output"]}}
else:
    result = {{"status": "failed", "error": tool_result["error"]}}
```

Return only Python code, do not include any explanations or markdown code block markers."""


# ===================== General Code Generation =====================

CODEACT_SYSTEM_PROMPT = """You are a professional Python code generation expert, specializing in generating executable Python code based on task descriptions.

# Your Responsibilities

Based on the user's task description, generate complete, executable Python code to complete the task.

# Code Generation Principles

1. **Code must be directly executable**, include all necessary imports and dependencies
2. **Code should be robust**, include appropriate error handling
3. **Code should be clear**, include necessary comments
4. **Prioritize Python standard library**, if third-party libraries are necessary, use common ones (such as pandas, numpy, etc.)
5. **Must set result variable at the end**, containing execution results
6. **LLM decides WHAT, code handles HOW data transforms** (keep logic explicit in code)

# ⭐ CRITICAL: Data Exploration Step (REQUIRED for data processing tasks)

Before processing any data files, you MUST first explore and understand the data structure:

```python
# Step 1: Explore data structure FIRST
import pandas as pd
df = pd.read_csv(file_path)
print(f"[Data Exploration] Columns: {list(df.columns)}")
print(f"[Data Exploration] Shape: {df.shape}")
print(f"[Data Exploration] First 3 rows:\\n{df.head(3)}")
print(f"[Data Exploration] Data types:\\n{df.dtypes}")
```

This helps you:
- Identify correct column names (don't guess!)
- Understand data types
- Spot potential issues

# ⭐ CRITICAL: Output Validation (REQUIRED for computation tasks)

After computing results, you MUST validate the output is reasonable:

```python
# Step 2: Validate output before returning
if output_value == 0 or output_value is None:
    print(f"[Warning] Output seems unusual: {output_value}")
    print(f"[Debug] Please verify the computation logic and data columns used")
    # Consider: Did I use the correct columns? Is the data format correct?
```

For scoring/metrics tasks (F1, accuracy, etc.):
- Scores should typically be > 0 (unless data is truly random)
- Scores should be in valid range (e.g., F1 in [0, 1])
- If all scores are 0, you likely used wrong columns!

# Output Format

You must only return executable Python code, do not include any explanations, markdown code block markers, or other text.
Code should:
- Import necessary libraries
- **Explore data structure FIRST** (print columns, sample data)
- Implement functionality required by the task
- **Validate output is reasonable** before returning
- Handle possible errors
- Set result variable (dictionary format, containing status, output, etc.)

# Notes

- If task involves file operations, ensure proper file path and encoding handling
- If task involves data processing, ensure data format is correct
- Code should be able to run independently, not dependent on external state
- **NEVER guess column names** - always print and explore first!
"""


def get_codeact_user_prompt(
    task_description: str,
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    output_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    column_hints: Optional[Dict[str, List[str]]] = None
) -> str:
    """
    Generate user prompt for general code
    
    Args:
        task_description: Task description
        inputs: Input parameter list
        outputs: Output parameter list
        output_constraints: Output constraints, format: {"field_name": {"min": 0, "max": 1, "description": "..."}}
        column_hints: Column hints for semantic matching, format: {"possible_true_labels": [...], "possible_predictions": [...]}
    
    Returns:
        User prompt
    """
    prompt = f"""Please generate Python code to complete the following task:

# Task Description
{task_description}
"""
    
    if inputs:
        prompt += f"\n# Input Parameters\n" + "\n".join([f"- {inp}" for inp in inputs])
    
    if outputs:
        prompt += f"\n# Output Requirements\n" + "\n".join([f"- {out}" for out in outputs])
    
    # Add column hints section with semantic matching guidance
    if column_hints:
        hints_text = "\n# ⭐ Column Hints (Use SEMANTIC Matching)\n\n"
        hints_text += "**IMPORTANT: Do NOT use simple keyword matching!**\n\n"
        hints_text += "Use the following semantic matching function to identify columns:\n\n"
        hints_text += """```python
def _semantic_column_match(columns: list, candidates: list, column_data_samples: dict = None) -> tuple:
    \"\"\"Semantically match columns using strict matching rules.
    
    Args:
        columns: All available column names
        candidates: Candidate column names from hints
        column_data_samples: Optional dict mapping column names to sample data for validation
    
    Returns:
        (matched_column, confidence, match_method)
    \"\"\"
    # Normalize column names for comparison
    col_map = {col.lower().strip(): col for col in columns}
    candidates_lower = [c.lower().strip() for c in candidates]
    
    # Priority 1: Exact match (highest confidence)
    for candidate in candidates_lower:
        if candidate in col_map:
            return col_map[candidate], 1.0, "exact_match"
    
    # Priority 2: Word boundary match (column contains candidate as whole word)
    import re
    for candidate in candidates_lower:
        pattern = r'\\b' + re.escape(candidate) + r'\\b'
        for col_lower, col_original in col_map.items():
            if re.search(pattern, col_lower):
                return col_original, 0.8, "word_boundary"
    
    # Priority 3: Prefix/suffix match
    for candidate in candidates_lower:
        for col_lower, col_original in col_map.items():
            if col_lower.startswith(candidate + '_') or col_lower.endswith('_' + candidate):
                return col_original, 0.6, "prefix_suffix"
    
    # No match found
    return None, 0.0, "no_match"

# Example usage with column_hints:
# column_hints = parameters.get("column_hints", {})
# true_label_candidates = column_hints.get("possible_true_labels", [])
# pred_candidates = column_hints.get("possible_predictions", [])

# true_col, true_conf, true_method = _semantic_column_match(df.columns, true_label_candidates)
# pred_col, pred_conf, pred_method = _semantic_column_match(df.columns, pred_candidates)

# if true_conf < 0.5 or pred_conf < 0.5:
#     print(f"[Warning] Low confidence column matching!")
#     print(f"  True label: {true_col} (confidence: {true_conf}, method: {true_method})")
#     print(f"  Prediction: {pred_col} (confidence: {pred_conf}, method: {pred_method})")
#     print(f"  Available columns: {list(df.columns)}")
```

"""
        hints_text += "# Available Column Hints:\n"
        for hint_type, candidates in column_hints.items():
            hints_text += f"- **{hint_type}**: {', '.join(candidates)}\n"
        
        hints_text += """
# Matching Rules (FOLLOW STRICTLY):
1. **Exact match first**: Column name exactly matches a candidate (case-insensitive)
2. **Word boundary second**: Candidate appears as a complete word in column name
3. **Prefix/suffix third**: Column starts/ends with candidate + underscore
4. **NEVER use partial substring matching** (e.g., "label" matching "main_name" is WRONG)
5. **Validate with data**: Check if matched column contains appropriate data types

# ⭐ CRITICAL: Fallback Strategy When No Match Found

If `_semantic_column_match` returns None (no match), DO NOT raise an error immediately!
Instead, use **intelligent column selection** based on data analysis:

```python
# Fallback: Intelligent column selection when semantic matching fails
def _intelligent_column_selection(df, purpose: str, exclude_cols: list = None) -> tuple:
    \"\"\"Select appropriate column based on data characteristics.
    
    Args:
        df: DataFrame
        purpose: 'label' or 'prediction' or 'score'
        exclude_cols: Columns to exclude
    
    Returns:
        (selected_column, reason)
    \"\"\"
    exclude_cols = exclude_cols or []
    candidates = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        col_lower = col.lower()
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        total_count = len(df)
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        
        # Score each column based on purpose
        score = 0
        reasons = []
        
        if purpose == 'label':
            # Labels should have moderate cardinality (not too many unique values)
            if 2 <= unique_count <= 20:
                score += 30
                reasons.append(f"good cardinality ({unique_count} unique)")
            elif unique_count <= 100:
                score += 15
                reasons.append(f"acceptable cardinality ({unique_count} unique)")
            
            # Numeric columns are often labels for classification
            if dtype in ['int64', 'int32', 'float64']:
                score += 20
                reasons.append("numeric type")
            
            # Column name hints
            if any(hint in col_lower for hint in ['class', 'label', 'target', 'binding', 'type']):
                score += 25
                reasons.append("name hint")
        
        elif purpose == 'prediction':
            # Predictions can be scores/probabilities
            if dtype in ['float64', 'float32']:
                score += 25
                reasons.append("float type (likely probability/score)")
            
            # Column name hints
            if any(hint in col_lower for hint in ['score', 'pred', 'prob', 'output', 'result']):
                score += 25
                reasons.append("name hint")
            
            # Binary predictions
            if unique_count == 2:
                score += 20
                reasons.append("binary values")
        
        if score > 0:
            candidates.append((col, score, '; '.join(reasons)))
    
    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    if candidates:
        return candidates[0][0], candidates[0][2]
    return None, "No suitable column found"

# Usage example:
# if true_col is None:
#     true_col, reason = _intelligent_column_selection(df, 'label')
#     print(f"[Fallback] Selected '{true_col}' as label column: {reason}")
# 
# if pred_col is None:
#     pred_col, reason = _intelligent_column_selection(df, 'prediction', exclude_cols=[true_col])
#     print(f"[Fallback] Selected '{pred_col}' as prediction column: {reason}")
```

# Validation After Matching:
```python
# After identifying columns, ALWAYS validate:
print(f"[Column Matching Result]")
print(f"  True label column: {true_col}")
print(f"  Prediction column: {pred_col}")

if true_col is None or pred_col is None:
    # CRITICAL: If still no match after fallback, provide helpful error
    print(f"[Error] Could not identify required columns!")
    print(f"  Available columns: {list(df.columns)}")
    print(f"  Column dtypes: {df.dtypes.to_dict()}")
    print(f"  Sample data:\\n{df.head(3)}")
    result = {{
        "status": "failed",
        "error": f"Column identification failed. True: {{true_col}}, Pred: {{pred_col}}. Available: {{list(df.columns)}}",
        "output": None
    }}
else:
    print(f"  True label unique values: {{df[true_col].nunique()}} unique, sample: {{df[true_col].head(3).tolist()}}")
    print(f"  Prediction unique values: {{df[pred_col].nunique()}} unique, sample: {{df[pred_col].head(3).tolist()}}")
    
    # Data type validation
    if df[true_col].dtype == 'object':
        print(f"  [Warning] True label is string type, may need encoding")
    if df[pred_col].dtype == 'object':
        print(f"  [Warning] Prediction is string type, may need encoding")
```
"""
        prompt += hints_text
    
    # Add output constraints section
    if output_constraints:
        constraints_text = "\n# ⭐ Output Constraints (MUST VALIDATE)\n\n"
        constraints_text += "Your output MUST satisfy these constraints. Add validation code to check:\n\n"
        
        for field_name, constraints in output_constraints.items():
            min_val = constraints.get("min")
            max_val = constraints.get("max")
            description = constraints.get("description", "")
            
            constraint_line = f"- **{field_name}**: "
            conditions = []
            if min_val is not None:
                conditions.append(f"must be >= {min_val}")
            if max_val is not None:
                conditions.append(f"must be <= {max_val}")
            
            constraint_line += ", ".join(conditions)
            if description:
                constraint_line += f" ({description})"
            
            constraints_text += constraint_line + "\n"
        
        constraints_text += """
# Example Validation Code:
```python
# Validate output constraints
validation_warnings = []
if output.get("f1_score", 0) <= 0:
    validation_warnings.append("F1 score is 0 or negative - check if correct columns are used")
if not (0 <= output.get("precision", 0) <= 1):
    validation_warnings.append("Precision out of valid range [0, 1]")
    
if validation_warnings:
    print("[Validation Warning] " + " | ".join(validation_warnings))
```
"""
        prompt += constraints_text
    
    prompt += """
# Requirements
1. Generate complete, executable Python code
2. Include all necessary imports
3. **Explore data structure FIRST** (print columns, sample data for file-based tasks)
4. Handle possible errors
5. **Validate output against constraints** before setting result
6. Set result variable in the following format:
   result = {
       "status": "success" or "failed",
       "output": <execution_result>,
       "error": <error_message, if any>
   }

Return only Python code, do not include any explanations."""
    
    return prompt


# ===================== Code Fixing =====================

FIX_CODE_SYSTEM_PROMPT = """You are a professional Python code fixing expert, specializing in fixing errors in code.

# Your Responsibilities

Based on the provided error information and original code, generate corrected code after fixing.

# Code Fixing Principles

1. **Maintain the original logic of the code**, only fix the error parts
2. **Ensure fixed code can execute**, will not produce new errors
3. **Add necessary error handling**, avoid similar errors from occurring again
4. **Code should be clear**, include necessary comments
5. **Must set result variable at the end**, containing execution results

# Output Format

You must only return fixed Python code, do not include any explanations, markdown code block markers, or other text.
Code should:
- Fix all errors
- Maintain original functionality
- Set result variable (dictionary format, containing status, output, etc.)

# Notes

- Carefully analyze error causes, ensure fixes are correct
- Do not change the core logic of the code
- If errors are due to missing dependencies or configuration, handle them in the code
"""


def get_fix_code_user_prompt(
    previous_code: str,
    previous_error: str,
    error_category: Optional[str] = None
) -> str:
    """
    Generate user prompt for code fixing
    
    Args:
        previous_code: Previous code
        previous_error: Previous error message
        error_category: Error category
    
    Returns:
        User prompt
    """
    prompt = f"""Please fix the errors in the following code:

# Original Code
```python
{previous_code}
```

# Error Message
{previous_error}
"""
    
    if error_category:
        prompt += f"\n# Error Category\n{error_category}\n"
    
    prompt += """
# Requirements
1. Fix all errors in the code
2. Maintain the original functionality and logic of the code
3. Ensure fixed code can execute correctly
4. Set result variable in the following format:
   result = {
       "status": "success" or "failed",
       "output": <execution_result>,
       "error": <error_message, if any>
   }

Return only the fixed Python code, do not include any explanations."""
    
    return prompt


# ===================== Parameter Fixing =====================

FIX_PARAMETER_SYSTEM_PROMPT = """You are a professional Python code fixing expert, specializing in fixing parameter errors in code.

# Your Responsibilities

Based on the provided error information and original code, fix parameter-related issues (such as parameter type errors, parameter value errors, missing parameters, etc.).

# Parameter Fixing Principles

1. **Maintain the original logic of the code**, only fix parameter issues
2. **Ensure fixed code can execute**, parameter types and values are all correct
3. **Add necessary parameter validation**, avoid similar errors from occurring again
4. **Code should be clear**, include necessary comments
5. **Must set result variable at the end**, containing execution results

# Output Format

You must only return fixed Python code, do not include any explanations, markdown code block markers, or other text.
Code should:
- Fix all parameter errors
- Maintain original functionality
- Set result variable (dictionary format, containing status, output, etc.)

# Notes

- Carefully analyze the causes of parameter errors, ensure fixes are correct
- If parameter type is wrong, perform appropriate type conversion
- If parameter value is wrong, use reasonable default values or corrected values
- If parameters are missing, add necessary parameters
"""


def get_fix_parameter_user_prompt(
    previous_code: str,
    previous_error: str,
    error_category: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate user prompt for parameter fixing
    
    Args:
        previous_code: Previous code
        previous_error: Previous error message
        error_category: Error category
        parameters: Available parameters (if any)
    
    Returns:
        User prompt
    """
    prompt = f"""Please fix the parameter errors in the following code:

# Original Code
```python
{previous_code}
```

# Error Message
{previous_error}
"""
    
    if error_category:
        prompt += f"\n# Error Category\n{error_category}\n"
    
    if parameters:
        params_str = json.dumps(parameters, ensure_ascii=False, indent=2)
        prompt += f"\n# Available Parameters\n{params_str}\n"
    
    prompt += """
# Requirements
1. Fix all parameter errors in the code
2. Maintain the original functionality and logic of the code
3. Ensure fixed code can execute correctly, parameter types and values are all correct
4. Set result variable in the following format:
   result = {
       "status": "success" or "failed",
       "output": <execution_result>,
       "error": <error_message, if any>
   }

Return only the fixed Python code, do not include any explanations."""
    
    return prompt

