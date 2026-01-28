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


class RevisionPlan(BaseModel):
    """Revision plan"""
    strategy: RevisionStrategy = Field(description="Strategy to adopt")
    root_cause: str = Field(description="Root cause analysis")
    action_items: List[str] = Field(default_factory=list, description="Specific action items")
    expected_outcome: str = Field(description="Expected outcome")
    confidence: float = Field(description="Confidence (0-1)", ge=0.0, le=1.0)
    orthogonal: bool = Field(default=False, description="Whether orthogonal to failure path (using different approach)")


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
   - Is the error reproducible? Are there patterns?
   
2. **Orthogonal Strategy Generation**: Generate new strategies different from failure path
   - If method A was tried before, consider method B
   - Avoid repeating the same error path
   
3. **Architecture-Level Improvement**: Not only fix errors, but also improve overall approach
   - Consider if there's a better implementation way
   - Can the problem be avoided by changing architecture

Output Format: Revision plan in JSON format, containing:
- strategy: Strategy type (code_fix/parameter_adjust/architecture_change/dependency_resolve/environment_fix)
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
        if revision_plan.strategy == RevisionStrategy.ARCHITECTURE_CHANGE:
            system_prompt = """You are a code architecture improvement expert. Your task is to re-implement code using a completely different architectural approach based on failure analysis.

Requirements:
1. **Orthogonal Strategy**: Don't repeat the failed method, adopt a completely new implementation approach
2. **Architecture Improvement**: Consider more elegant, more robust implementation approaches
3. **Error Prevention**: Add error handling and validation in code
4. **Maintainability**: Code should be clear, readable, and maintainable

Generate complete, executable code."""
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

