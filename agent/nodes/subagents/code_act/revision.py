"""
CodeAct Revision Mechanism

Reference SE-Agent's Revision mechanism, implement failure-driven strategy generation:
1. Deep self-reflection: Analyze failure causes, identify root problems
2. Orthogonal strategy generation: Generate new strategies different from failure path
3. Architecture-level improvement: Not only fix errors, but also improve overall approach
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import json

from nodes.subagents.code_act.trajectory import CodeTrajectory, TrajectoryStatus
from utils.llm_factory import create_reasoning_advanced_llm, create_code_llm


class RevisionStrategy(str, Enum):
    """Revision strategy type"""
    CODE_FIX = "code_fix"  # Code fix: Fix syntax errors, logic errors
    PARAMETER_ADJUST = "parameter_adjust"  # Parameter adjustment: Modify parameter values or types
    ARCHITECTURE_CHANGE = "architecture_change"  # Architecture improvement: Change overall implementation approach
    DEPENDENCY_RESOLVE = "dependency_resolve"  # Dependency resolution: Add missing dependencies or imports
    ENVIRONMENT_FIX = "environment_fix"  # Environment fix: Fix environment configuration issues
    DATA_TRANSFORM = "data_transform"  # Data transform: Use data transformation tools to fix format mismatches


class RevisionPlan(BaseModel):
    """Revision plan"""
    strategy: RevisionStrategy = Field(description="Strategy to adopt")
    root_cause: str = Field(description="Root cause analysis")
    action_items: List[str] = Field(default_factory=list, description="Specific action items")
    expected_outcome: str = Field(description="Expected outcome")
    confidence: float = Field(description="Confidence (0-1)", ge=0.0, le=1.0)
    orthogonal: bool = Field(default=False, description="Whether orthogonal to failure path (using different approach)")
    suggested_tool: Optional[Dict[str, Any]] = Field(default=None, description="Suggested tool for data transformation")


# Known data transformation tools for common format mismatches
DATA_TRANSFORM_TOOLS = {
    # TCR/NetTCR related transformations
    "tcr_to_nettcr": {
        "tool_name": "convert_tcr_to_nettcr_format",
        "service": "reference",
        "trigger_patterns": [
            "Missing required columns: {'A1'", "Missing required columns: {'A2'",
            "Missing required columns: {'A3'", "Missing required columns: {'B1'",
            "Missing required columns: {'B2'", "Missing required columns: {'B3'",
            "Missing required columns: {'peptide'",
            "NetTCR", "nettcr format", "A1, A2, A3, B1, B2, B3"
        ],
        "description": "Convert TCR data with V gene columns to NetTCR format (A1-A3, B1-B3, peptide)",
        "required_columns": ["TRA_v_gene", "TRB_v_gene", "CDR3a", "CDR3b"],
    },
    "format_tcr_for_nettcr": {
        "tool_name": "format_tcr_data_for_nettcr",
        "service": "nettcr",
        "trigger_patterns": [
            "CDR3a", "CDR3b", "cdr3a", "cdr3b"
        ],
        "description": "Convert TCR data with CDR3a/CDR3b columns to NetTCR format",
    },
    # FASTA conversion for igblast
    "csv_to_fasta": {
        "tool_name": "convert_csv_to_fasta",
        "service": "reference",
        "trigger_patterns": [
            "FASTA", "fasta", ".fa", ".fasta",
            "sequences must be in FASTA format",
        ],
        "description": "Convert CSV/TSV sequences to FASTA format",
    },
}


def find_data_transform_tool(error_message: str, tool_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Find an appropriate data transformation tool based on error message.
    
    Args:
        error_message: The error message from failed execution
        tool_context: The tool that failed (e.g., "validate_tcr_input", "predict_tcr_binding_ensemble")
    
    Returns:
        Suggested transformation tool info, or None if no match found
    """
    error_lower = error_message.lower()
    
    # Check for NetTCR format mismatches
    if any(p.lower() in error_lower for p in ["nettcr", "a1", "a2", "a3", "b1", "b2", "b3"]):
        if "missing required columns" in error_lower:
            # This is likely a format mismatch for NetTCR
            return DATA_TRANSFORM_TOOLS["tcr_to_nettcr"]
    
    # Check for FASTA format requirements
    if any(p in error_message for p in ["FASTA", ".fa", ".fasta"]):
        return DATA_TRANSFORM_TOOLS["csv_to_fasta"]
    
    # Context-specific recommendations
    if tool_context:
        if "validate_tcr_input" in tool_context or "predict_tcr_binding" in tool_context:
            # NetTCR tools require specific column format
            return DATA_TRANSFORM_TOOLS["tcr_to_nettcr"]
    
    return None


class RevisionAnalyzer:
    """
    Revision analyzer
    
    Responsible for analyzing failed trajectories and generating Revision plans.
    """
    
    def __init__(self):
        self.llm = create_reasoning_advanced_llm()
    
    def analyze_failure(
        self,
        failed_trajectory: CodeTrajectory,
        previous_trajectories: Optional[List[CodeTrajectory]] = None
    ) -> RevisionPlan:
        """
        Analyze failed trajectory and generate Revision plan
        
        Args:
            failed_trajectory: Failed trajectory
            previous_trajectories: Previous attempt trajectories (to avoid repeating errors)
        
        Returns:
            Revision plan
        """
        # First, check for data format issues using simple analysis (high priority)
        error_message = failed_trajectory.error_message or ""
        if "Missing required columns" in error_message or "missing columns" in error_message.lower():
            # Data format mismatch detected - use simple analysis which has tool suggestions
            print("  🔍 Detected data format mismatch, using specialized analysis")
            return self._analyze_failure_simple(failed_trajectory)
        
        if not self.llm:
            # LLM unavailable, use simple analysis
            return self._analyze_failure_simple(failed_trajectory)
        
        try:
            return self._analyze_failure_with_llm(failed_trajectory, previous_trajectories or [])
        except Exception as e:
            print(f"  ⚠ LLM analysis failed: {e}, using simple analysis")
            return self._analyze_failure_simple(failed_trajectory)
    
    def _analyze_failure_with_llm(
        self,
        failed_trajectory: CodeTrajectory,
        previous_trajectories: List[CodeTrajectory]
    ) -> RevisionPlan:
        """Use LLM for deep analysis"""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        system_prompt = """You are a code execution failure analysis expert. Your task is to deeply analyze the root cause of code execution failures and generate a Revision plan to fix the problem.

Analysis Requirements:
1. **Deep Self-Reflection**: Don't just look at surface errors, identify root problems
   - Is it code logic error? Parameter issue? Environment configuration? Missing dependencies?
   - **IMPORTANT**: Is it a DATA FORMAT MISMATCH? (e.g., "Missing required columns", "expected format X but got Y")
   - Is the error reproducible? Are there patterns?
   
2. **Orthogonal Strategy Generation**: Generate new strategies different from failure path
   - If method A was tried before, consider method B
   - Avoid repeating the same error path
   
3. **Architecture-Level Improvement**: Not only fix errors, but also improve overall approach
   - Consider if there's a better implementation way
   - Can the problem be avoided by changing architecture

4. **Data Transformation Detection** (CRITICAL):
   - If error mentions "Missing required columns" or "format mismatch", use `data_transform` strategy
   - For NetTCR tools requiring A1, A2, A3, B1, B2, B3, peptide columns:
     * If input has CDR3a, CDR3b, TRA_v_gene, TRB_v_gene columns → use `convert_tcr_to_nettcr_format` tool
   - data_transform strategy should be preferred over code_fix for format mismatches

Output Format: Revision plan in JSON format, containing:
- strategy: Strategy type (code_fix/parameter_adjust/architecture_change/dependency_resolve/environment_fix/**data_transform**)
- root_cause: Root cause analysis (detailed explanation)
- action_items: Specific action items list (each item a string)
- expected_outcome: Expected outcome description
- confidence: Confidence (0-1)
- orthogonal: Whether orthogonal to failure path (true/false)"""
        
        # Prepare failure information
        error_info = {
            "error_type": failed_trajectory.error_type,
            "error_message": failed_trajectory.error_message,
            "error_category": failed_trajectory.error_category,
            "code_preview": failed_trajectory.generated_code[:500] if failed_trajectory.generated_code else None,
            "execution_mode": failed_trajectory.execution_mode,
            "parameters": failed_trajectory.parameters
        }
        
        # Prepare previous attempt information
        previous_attempts = []
        for prev_traj in previous_trajectories[-3:]:  # Only consider last 3 attempts
            prev_info = {
                "error_type": prev_traj.error_type,
                "error_message": prev_traj.error_message[:200] if prev_traj.error_message else None,
                "strategy_used": prev_traj.metadata.get("revision_strategy") if prev_traj.metadata else None
            }
            previous_attempts.append(prev_info)
        
        user_prompt = f"""Please analyze the following code execution failure and generate a Revision plan:

Failure Information:
{json.dumps(error_info, ensure_ascii=False, indent=2)}

Previous Attempts (avoid repetition):
{json.dumps(previous_attempts, ensure_ascii=False, indent=2) if previous_attempts else "None"}

Please generate Revision plan (JSON format)."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse JSON response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        plan_data = json.loads(response_text)
        
        # Create Revision plan
        return RevisionPlan(
            strategy=RevisionStrategy(plan_data.get("strategy", "code_fix")),
            root_cause=plan_data.get("root_cause", "Unknown cause"),
            action_items=plan_data.get("action_items", []),
            expected_outcome=plan_data.get("expected_outcome", "Fix error"),
            confidence=float(plan_data.get("confidence", 0.5)),
            orthogonal=bool(plan_data.get("orthogonal", False))
        )
    
    def _analyze_failure_simple(self, failed_trajectory: CodeTrajectory) -> RevisionPlan:
        """Simple analysis (used when LLM unavailable)"""
        error_type = failed_trajectory.error_type or "Unknown"
        error_message = failed_trajectory.error_message or ""
        
        # Get tool context from trajectory
        tool_context = None
        if failed_trajectory.tools:
            tool_context = failed_trajectory.tools[0].get("tool_name", "") if failed_trajectory.tools else ""
        
        # Check for data format mismatch errors (e.g., missing required columns)
        if "Missing required columns" in error_message or "missing columns" in error_message.lower():
            # Try to find a suitable transformation tool
            suggested_tool = find_data_transform_tool(error_message, tool_context)
            
            strategy = RevisionStrategy.DATA_TRANSFORM
            root_cause = f"Data format mismatch: {error_message}"
            action_items = [
                "Analyze the expected input format for the tool",
                "Find appropriate data transformation tools",
                "Convert input data to required format before calling the tool",
                "Common transformations: CSV column renaming, format conversion (CSV->FASTA), etc."
            ]
            
            if suggested_tool:
                action_items.insert(0, f"RECOMMENDED: Use {suggested_tool['tool_name']} tool to convert data format")
                action_items.insert(1, f"  - {suggested_tool['description']}")
            
            return RevisionPlan(
                strategy=strategy,
                root_cause=root_cause,
                action_items=action_items,
                expected_outcome="Transform input data to match expected format",
                confidence=0.85,
                orthogonal=True,
                suggested_tool=suggested_tool
            )
        
        # Check for parameter errors related to data format
        if "parameter_error" in error_message.lower() or "validation" in error_message.lower():
            # Also try to find transformation tools for validation errors
            suggested_tool = find_data_transform_tool(error_message, tool_context)
            
            strategy = RevisionStrategy.PARAMETER_ADJUST
            root_cause = f"Parameter validation error: {error_message}"
            action_items = [
                "Check parameter types and formats",
                "Validate input data structure",
                "Consider using data transformation tools if format mismatch detected"
            ]
            
            if suggested_tool:
                action_items.append(f"RECOMMENDED: Use {suggested_tool['tool_name']} to fix format issues")
            
            return RevisionPlan(
                strategy=strategy,
                root_cause=root_cause,
                action_items=action_items,
                expected_outcome="Adjust parameters to match expected format",
                confidence=0.75,
                orthogonal=False,
                suggested_tool=suggested_tool
            )
        
        # Infer strategy based on error type
        if "SyntaxError" in error_type or "IndentationError" in error_type:
            strategy = RevisionStrategy.CODE_FIX
            root_cause = "Code syntax error"
            action_items = ["Check code syntax", "Fix indentation issues", "Verify code structure"]
        elif "NameError" in error_type or "ImportError" in error_type:
            strategy = RevisionStrategy.DEPENDENCY_RESOLVE
            root_cause = "Dependency or import issue"
            action_items = ["Check import statements", "Add missing dependencies", "Verify module paths"]
        elif "TypeError" in error_type or "ValueError" in error_type:
            strategy = RevisionStrategy.PARAMETER_ADJUST
            root_cause = "Parameter type or value error"
            action_items = ["Check parameter types", "Verify parameter values", "Adjust parameter format"]
        elif "AttributeError" in error_type:
            strategy = RevisionStrategy.CODE_FIX
            root_cause = "Object attribute access error"
            action_items = ["Check object type", "Verify attribute existence", "Fix attribute access"]
        else:
            strategy = RevisionStrategy.CODE_FIX
            root_cause = f"Execution error: {error_type}"
            action_items = ["Check code logic", "Add error handling", "Verify execution environment"]
        
        return RevisionPlan(
            strategy=strategy,
            root_cause=root_cause,
            action_items=action_items,
            expected_outcome="Fix error and execute successfully",
            confidence=0.6,
            orthogonal=False
        )


class RevisionExecutor:
    """
    Revision executor
    
    Generate fixed code based on Revision plan.
    """
    
    def __init__(self):
        from utils.llm_factory import create_code_llm
        self.llm = create_code_llm()
    
    def generate_revision_code(
        self,
        revision_plan: RevisionPlan,
        original_code: str,
        original_error: str,
        task_description: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Generate fixed code based on Revision plan
        
        Args:
            revision_plan: Revision plan
            original_code: Original code
            original_error: Original error message
            task_description: Task description
            parameters: Task parameters
        
        Returns:
            Fixed code
        """
        if not self.llm:
            # LLM unavailable, use simple fix
            return self._generate_revision_code_simple(revision_plan, original_code, original_error)
        
        try:
            return self._generate_revision_code_with_llm(
                revision_plan, original_code, original_error, task_description, parameters
            )
        except Exception as e:
            print(f"  ⚠ LLM failed to generate fix code: {e}, using simple fix")
            return self._generate_revision_code_simple(revision_plan, original_code, original_error)
    
    def _generate_revision_code_with_llm(
        self,
        revision_plan: RevisionPlan,
        original_code: str,
        original_error: str,
        task_description: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Use LLM to generate fix code"""
        from langchain_core.messages import SystemMessage, HumanMessage
        from nodes.subagents.code_act.prompt import FIX_CODE_SYSTEM_PROMPT
        
        # Select different prompts based on strategy
        if revision_plan.strategy == RevisionStrategy.DATA_TRANSFORM:
            # Special handling for data transformation strategy
            # IMPORTANT: We need to completely replace the original code, not fix it
            system_prompt = """You are a data transformation expert. Your task is to generate COMPLETELY NEW code that first transforms data, then calls the MCP tool.

# CRITICAL REQUIREMENT
You MUST NOT just repeat the original code. Your new code MUST:
1. First import and call the LOCAL data transformation function (convert_tcr_to_nettcr_format)
2. Parse the transformation result to get the output file path
3. Only then call the ORIGINAL MCP tool with the TRANSFORMED data path

# CRITICAL: Data Transformation Functions are LOCAL, NOT MCP Tools!

The data transformation functions (like convert_tcr_to_nettcr_format) are LOCAL Python functions in the agent codebase, NOT MCP tools!
- Import: `from tools.reference import convert_tcr_to_nettcr_format`
- Call directly as a Python function: `result = convert_tcr_to_nettcr_format(input_data=..., peptide=...)`
- The function returns a STRING with the result, NOT a dict with status
- DO NOT use call_tool() for these local functions!

# Available LOCAL Data Transformation Functions
1. `convert_tcr_to_nettcr_format` (from tools.reference)
   - Input: input_data (CSV path), peptide (target peptide sequence), output_dir (optional)
   - Output: STRING with conversion summary (parse the output path from the string)
   
# Example Code Structure (YOU MUST FOLLOW THIS PATTERN)
```python
from core.tool_interface import call_tool
from tools.reference import convert_tcr_to_nettcr_format  # LOCAL function, NOT MCP tool!
import os
import re

# Determine output directory from input path
input_path = "<original_input_path>"
output_dir = None
if "/data/sessions/" in input_path:
    parts = input_path.split("/data/sessions/")[1].split("/")
    session_id = parts[0] if parts else None
    if session_id:
        output_dir = f"/data/sessions/{session_id}/output"
        os.makedirs(output_dir, exist_ok=True)

# STEP 1: Transform the data FIRST using LOCAL function
# NOTE: This is a LOCAL Python function, call it directly (NOT via call_tool)
transform_kwargs = {
    "input_data": input_path,
    "peptide": "<target_peptide>"  # e.g., "ELAGIGILTV"
}
if output_dir:
    transform_kwargs["output_dir"] = output_dir

result_text = convert_tcr_to_nettcr_format(**transform_kwargs)
print(f"[Transform] {result_text[:500]}")

# STEP 2: Check transformation result (it's a string, not a dict)
if "Error:" in result_text or "error" in result_text.lower():
    result = {"status": "failed", "error": "Data transformation failed", "details": result_text}
else:
    # STEP 3: Parse output path from result text
    path_match = re.search(r'([^\\s]*_nettcr_format\\.csv)', result_text)
    if path_match:
        transformed_path = path_match.group(1)
    elif output_dir:
        transformed_path = os.path.join(output_dir, os.path.basename(input_path).rsplit('.', 1)[0] + "_nettcr_format.csv")
    else:
        transformed_path = input_path.rsplit('.', 1)[0] + "_nettcr_format.csv"
    
    print(f"[Transform] Transformed file: {transformed_path}")
    
    # STEP 4: Call the ORIGINAL MCP tool with TRANSFORMED data
    # NOTE: This IS an MCP tool, so use call_tool()
    tool_result = call_tool(
        "<original_tool_name>",
        {
            "input_data": transformed_path,  # Use TRANSFORMED path, not original!
            # ... other parameters
        }
    )
    result = tool_result
```

# Output Format
Generate ONLY Python code. No explanations. The code MUST include:
1. Import from tools.reference for LOCAL transformation functions
2. Direct function call (NOT call_tool) for transformation
3. Parse the string result to get output path
4. Use call_tool() for the ORIGINAL MCP tool with transformed data"""
            
            # Get original tool info from the code
            tool_info = self._extract_tool_info_from_code(original_code)
            
            user_prompt = f"""Generate COMPLETELY NEW code that first transforms data, then calls the original tool.

# IMPORTANT: You must NOT repeat the original code as-is. The original code FAILED because the data format is wrong.

Original Tool: {tool_info.get('tool_name', 'unknown')}
Original Service: {tool_info.get('service_id', 'unknown')}
Original Error: {original_error}

Parameters:
{json.dumps(parameters, ensure_ascii=False, indent=2)}

# SUGGESTED TRANSFORMATION
{f"Use: {revision_plan.suggested_tool.get('tool_name', 'convert_tcr_to_nettcr_format')}" if revision_plan.suggested_tool else "Use: convert_tcr_to_nettcr_format"}

Generate the complete fixed code NOW. Remember:
1. Import: `from tools.reference import convert_tcr_to_nettcr_format` (LOCAL function)
2. Call convert_tcr_to_nettcr_format DIRECTLY (NOT via call_tool) - it's a local Python function
3. The function returns a STRING with result - parse the output path from it
4. Then use call_tool() for the ORIGINAL MCP tool with the TRANSFORMED data path"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            code = response.content.strip()
            
            # Remove markdown code block markers
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()
            
            # Validate the code contains transformation step with correct import
            if "convert_tcr_to_nettcr_format" not in code:
                print("  ⚠ Generated code does not contain data transformation, regenerating...")
                # Force regenerate with simpler approach
                code = self._generate_data_transform_code_fallback(parameters, tool_info)
            elif "from tools.reference import" not in code and "tools.reference" not in code:
                # LLM might have used call_tool incorrectly for local function
                print("  ⚠ Generated code missing correct import for local function, regenerating...")
                code = self._generate_data_transform_code_fallback(parameters, tool_info)
            
            return code
            
        elif revision_plan.strategy == RevisionStrategy.ARCHITECTURE_CHANGE:
            system_prompt = """You are a code architecture improvement expert. Your task is to re-implement code using a completely different architectural approach based on failure analysis.

Requirements:
1. **Orthogonal Strategy**: Don't repeat the failed method, adopt a completely new implementation approach
2. **Architecture Improvement**: Consider more elegant, more robust implementation approaches
3. **Error Prevention**: Add error handling and validation in code
4. **Maintainability**: Code should be clear, readable, and maintainable

Generate complete, executable code."""
            user_prompt = f"""Based on the following Revision plan, generate fixed code:

Revision Plan:
- Strategy: {revision_plan.strategy.value}
- Root cause: {revision_plan.root_cause}
- Action items: {', '.join(revision_plan.action_items)}
- Expected outcome: {revision_plan.expected_outcome}
- Orthogonal strategy: {'Yes' if revision_plan.orthogonal else 'No'}

Original Code:
```python
{original_code}
```

Original Error:
{original_error}

Task Description: {task_description}
Parameters: {json.dumps(parameters, ensure_ascii=False)}

Please generate complete fixed code (ensure executable)."""
        else:
            system_prompt = FIX_CODE_SYSTEM_PROMPT
            user_prompt = f"""Based on the following Revision plan, generate fixed code:

Revision Plan:
- Strategy: {revision_plan.strategy.value}
- Root cause: {revision_plan.root_cause}
- Action items: {', '.join(revision_plan.action_items)}
- Expected outcome: {revision_plan.expected_outcome}
- Orthogonal strategy: {'Yes' if revision_plan.orthogonal else 'No'}

Original Code:
```python
{original_code}
```

Original Error:
{original_error}

Task Description: {task_description}
Parameters: {json.dumps(parameters, ensure_ascii=False)}

Please generate complete fixed code (ensure executable)."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        code = response.content.strip()
        
        # Remove markdown code block markers
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        return code
    
    def _extract_tool_info_from_code(self, code: str) -> Dict[str, Any]:
        """Extract tool information from generated code."""
        import re
        
        tool_info = {
            "tool_name": "unknown",
            "service_id": "unknown"
        }
        
        # Try to find tool_name in the code
        tool_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', code)
        if tool_match:
            tool_info["tool_name"] = tool_match.group(1)
        
        # Try to find service_id in the code
        service_match = re.search(r'"service_id"\s*:\s*"([^"]+)"', code)
        if service_match:
            tool_info["service_id"] = service_match.group(1)
        
        # Try to find call_tool pattern
        call_tool_match = re.search(r'call_tool\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"', code)
        if call_tool_match:
            tool_info["service_id"] = call_tool_match.group(1)
            tool_info["tool_name"] = call_tool_match.group(2)
        
        return tool_info
    
    def _generate_data_transform_code_fallback(self, parameters: Dict[str, Any], tool_info: Dict[str, Any]) -> str:
        """Generate fallback code for data transformation when LLM fails."""
        input_data = parameters.get("input_data", "INPUT_PATH")
        peptide = parameters.get("peptide", "ELAGIGILTV")  # Default peptide for MART-1
        original_tool = tool_info.get("tool_name", "predict_tcr_binding_ensemble")
        original_service = tool_info.get("service_id", "nettcr")
        
        # Get other parameters excluding input_data
        other_params = {k: v for k, v in parameters.items() if k not in ["input_data", "peptide"]}
        other_params_str = ", ".join([f'"{k}": {repr(v)}' for k, v in other_params.items()])
        if other_params_str:
            other_params_str = ", " + other_params_str
        
        return f'''from core.tool_interface import call_tool
from tools.reference import convert_tcr_to_nettcr_format
import os
import re

# Determine output directory from input path (save to same session output directory)
# If input is in /data/sessions/{session_id}/, output should go to /data/sessions/{session_id}/output/
input_path = "{input_data}"
output_dir = None
if "/data/sessions/" in input_path:
    # Extract session directory and construct output path
    parts = input_path.split("/data/sessions/")[1].split("/")
    session_id = parts[0] if parts else None
    if session_id:
        output_dir = f"/data/sessions/{{session_id}}/output"
        os.makedirs(output_dir, exist_ok=True)
        print(f"[DataTransform] Output directory: {{output_dir}}")

# STEP 1: Transform the data FIRST using local convert_tcr_to_nettcr_format function
# Note: This is a LOCAL Python function, NOT an MCP tool!
transform_kwargs = {{
    "input_data": input_path,
    "peptide": "{peptide}"
}}
if output_dir:
    transform_kwargs["output_dir"] = output_dir

# Call the local function directly (NOT via call_tool)
result_text = convert_tcr_to_nettcr_format(**transform_kwargs)
print(f"[DataTransform] Conversion result: {{result_text[:500]}}...")

# STEP 2: Check transformation result
if "Error:" in result_text or "error" in result_text.lower():
    result = {{
        "status": "failed", 
        "error": "Data transformation failed", 
        "details": result_text
    }}
else:
    # STEP 3: Parse output path from result text
    # Look for output file path in the result
    path_match = re.search(r'Output:\\s*([^\\s]+\\.csv)', result_text)
    if not path_match:
        path_match = re.search(r'Saved to:\\s*([^\\s]+\\.csv)', result_text)
    if not path_match:
        # Try to find any .csv path with _nettcr_format
        path_match = re.search(r'([^\\s]*_nettcr_format\\.csv)', result_text)
    
    if path_match:
        transformed_path = path_match.group(1)
    else:
        # Fallback: construct path from input
        transformed_path = input_path.rsplit('.', 1)[0] + "_nettcr_format.csv"
        if output_dir:
            transformed_path = os.path.join(output_dir, os.path.basename(input_path).rsplit('.', 1)[0] + "_nettcr_format.csv")
    
    print(f"[DataTransform] Data transformed successfully: {{transformed_path}}")
    
    # STEP 4: Call the ORIGINAL MCP tool with TRANSFORMED data
    tool_result = call_tool(
        "{original_tool}",
        {{
            "input_data": transformed_path{other_params_str}
        }}
    )
    result = tool_result
'''
    
    def _generate_revision_code_simple(
        self,
        revision_plan: RevisionPlan,
        original_code: str,
        original_error: str
    ) -> str:
        """Simple fix (used when LLM unavailable)"""
        # Basic error handling wrapper
        fixed_code = f"""# Revision fix: {revision_plan.root_cause}
# Strategy: {revision_plan.strategy.value}
# Action items: {', '.join(revision_plan.action_items)}

try:
{chr(10).join('    ' + line for line in original_code.split(chr(10)))}
    result = {{"status": "success", "output": "Execution successful"}}
except Exception as e:
    # Error handling
    result = {{
        "status": "failed",
        "error": str(e),
        "error_type": type(e).__name__
    }}
"""
        return fixed_code


def create_revision_plan(
    failed_trajectory: CodeTrajectory,
    previous_trajectories: Optional[List[CodeTrajectory]] = None
) -> RevisionPlan:
    """
    Convenience function to create Revision plan
    
    Args:
        failed_trajectory: Failed trajectory
        previous_trajectories: Previous attempt trajectories
    
    Returns:
        Revision plan
    """
    analyzer = RevisionAnalyzer()
    return analyzer.analyze_failure(failed_trajectory, previous_trajectories)


def execute_revision_plan(
    revision_plan: RevisionPlan,
    original_code: str,
    original_error: str,
    task_description: str,
    parameters: Dict[str, Any]
) -> str:
    """
    Convenience function to execute Revision plan
    
    Args:
        revision_plan: Revision plan
        original_code: Original code
        original_error: Original error
        task_description: Task description
        parameters: Task parameters
    
    Returns:
        Fixed code
    """
    executor = RevisionExecutor()
    return executor.generate_revision_code(
        revision_plan, original_code, original_error, task_description, parameters
    )

