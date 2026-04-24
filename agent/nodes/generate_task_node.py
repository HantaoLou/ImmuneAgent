"""
Generate Task Node - Generates task.md and identifies missing parameters

This node replaces the task_decomposition subgraph with a HITL-enabled approach:
1. Generates task.md content for user review
2. Identifies missing parameters
3. Supports regeneration based on user feedback
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import json
import re
import os
from datetime import datetime

agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, SubTask, UserTaskType

try:
    from langchain_core.messages import HumanMessage, SystemMessage

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    HumanMessage = None
    SystemMessage = None


GENERATE_TASK_SYSTEM_PROMPT = """You are an expert task planner for bioinformatics and computational biology workflows.

Your role is to:
1. Analyze the user's request and any execution plan
2. Generate a structured task breakdown in Markdown format
3. Focus on describing WHAT needs to be done, not HOW or with WHICH tools
4. Identify missing parameters that the user needs to provide

IMPORTANT: Your task is to describe the task sequence clearly. Do NOT specify:
- Which specific tools or services to use
- Implementation details or technical approaches
- Specific software packages or libraries

The execution layer will decide the appropriate tools and implementation methods.

## Output Format

You must respond with a JSON object containing:
1. "task_md": A Markdown document describing the task breakdown
2. "missing_parameters": A list of parameters that are missing and need user input

## Task Markdown Structure

The task.md should include:
- **Overview**: Brief summary of what will be done
- **Input Requirements**: Required input data and their formats
- **Processing Steps**: Clear step-by-step workflow describing what each step accomplishes
- **Expected Outputs**: What results will be generated

## Missing Parameters Format

Each missing parameter should have:
- "name": Parameter name
- "description": What this parameter is for
- "type": Data type (string, number, file, etc.)
- "required": Whether it's required

## Example Response

```json
{
  "task_md": "# Task Breakdown\\n\\n## Overview\\nAnalyze TCR sequences...\\n\\n## Steps\\n1. Load data\\n2. Process...",
  "missing_parameters": [
    {
      "name": "input_file",
      "description": "The TCR sequence file to analyze",
      "type": "file",
      "required": true
    }
  ]
}
```

Generate a comprehensive and clear task plan that focuses on WHAT needs to be done, not how to implement it."""

GENERATE_TASK_USER_PROMPT = """## User Request
{user_input}

## Execution Plan (if provided)
{execution_plan}

## Available Files
{available_files}

## Previous Task Plan (if regenerating)
{previous_task_md}

## User Feedback (if regenerating)
{user_feedback}

---

Please generate a task breakdown (task.md) and identify any missing parameters.

**Note**: If there is a Previous Task Plan, you should revise it based on the User Feedback while keeping the good parts. Focus on addressing the user's specific concerns."""


def _extract_response_text(response) -> str:
    """Extract text from LLM response."""
    response_text = ""

    if hasattr(response, "content") and response.content:
        content = response.content
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
            response_text = "".join(texts)

    if not response_text and hasattr(response, "additional_kwargs"):
        reasoning = response.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            response_text = reasoning

    if not response_text and hasattr(response, "response_metadata"):
        metadata = response.response_metadata
        if isinstance(metadata, dict):
            for key in ["content", "text", "reasoning_content", "output"]:
                if key in metadata and metadata[key]:
                    response_text = str(metadata[key])
                    break

    if not response_text:
        if hasattr(response, "text"):
            response_text = response.text
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", str(response))

    return response_text.strip() if response_text else ""


def _parse_generate_task_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response to extract task_md and missing_parameters."""
    result = {
        "task_md": "",
        "missing_parameters": [],
    }

    thinking_patterns = [
        r"<think[^>]*>.*?</think\s*>",
        r"<thinking[^>]*>.*?</thinking\s*>",
        r"<reasoning[^>]*>.*?</reasoning\s*>",
    ]
    for pattern in thinking_patterns:
        response_text = re.sub(
            pattern, "", response_text, flags=re.DOTALL | re.IGNORECASE
        )

    response_text = response_text.strip()

    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            result["task_md"] = parsed.get("task_md", "")
            result["missing_parameters"] = parsed.get("missing_parameters", [])
            return result
    except json.JSONDecodeError:
        pass

    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    result["task_md"] = parsed.get("task_md", "")
                    result["missing_parameters"] = parsed.get("missing_parameters", [])
                    return result
            except json.JSONDecodeError:
                continue

    brace_count, start_idx = 0, -1
    for i, char in enumerate(response_text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    parsed = json.loads(response_text[start_idx : i + 1])
                    if isinstance(parsed, dict):
                        result["task_md"] = parsed.get("task_md", "")
                        result["missing_parameters"] = parsed.get(
                            "missing_parameters", []
                        )
                        return result
                except json.JSONDecodeError:
                    pass
                start_idx = -1

    if not result["task_md"]:
        if response_text.startswith("#"):
            lines = response_text.split("\n")
            if any(line.startswith("##") for line in lines):
                result["task_md"] = response_text

    return result


def _generate_fallback_task_md(state: GlobalState) -> str:
    """Generate a fallback task.md when LLM is unavailable."""
    user_input = state.user_input
    execution_plan = state.execution_plan or ""

    task_md = f"""# Task Breakdown

## Overview
{execution_plan[:500] if execution_plan else user_input[:500]}

## Input Requirements
- User input will be processed based on the request

## Processing Steps
1. Analyze user request
2. Execute required operations
3. Generate output results

## Expected Outputs
- Results will be generated based on task execution

## Notes
- This is a fallback task plan generated without LLM assistance
- Please review and provide additional parameters if needed
"""
    return task_md


def _get_available_files(state: GlobalState) -> str:
    """Get list of available files from state."""
    files_info = []

    copied_files = state.merged_result.get("copied_files", {})
    if copied_files:
        files_info.append("### Uploaded Files")
        for src, dst in copied_files.items():
            files_info.append(f"- `{dst}` (from {src})")

    input_dir = state.merged_result.get("input_dir", "")
    if input_dir:
        files_info.append(f"\n### Input Directory\n`{input_dir}`")

    if state.file_paths:
        files_info.append("\n### File Paths")
        for key, path in state.file_paths.items():
            files_info.append(f"- {key}: `{path}`")

    return "\n".join(files_info) if files_info else "No files provided"


async def generate_task_node(state: GlobalState) -> GlobalState:
    """
    Generate Task Node - Creates task.md and identifies missing parameters.

    This node:
    1. Generates a task breakdown (task.md) for user review
    2. Identifies missing parameters that need user input
    3. Supports regeneration based on user feedback

    Args:
        state: Current GlobalState

    Returns:
        Updated GlobalState with task_md_content and missing_parameters
    """
    print(f"\n{'=' * 60}")
    print("[GenerateTask] Starting task generation...")
    print(f"{'=' * 60}")

    user_input = state.user_input
    execution_plan = state.execution_plan
    user_feedback = state.user_feedback
    iteration = state.hitl_iteration

    print(f"[GenerateTask] Iteration: {iteration}")
    if user_feedback:
        print(f"[GenerateTask] User feedback: {user_feedback[:100]}...")

    available_files = _get_available_files(state)

    llm = state.get_llm(purpose="reasoning_advanced", node_name="generate_task")

    # Debug: Check progress_callback during LLM creation
    print(f"[GenerateTask] Created LLM with type: {type(llm)}")
    if hasattr(llm, "progress_callback"):
        print(
            f"[GenerateTask] LLM.progress_callback: {llm.progress_callback is not None}"
        )
    if hasattr(llm, "enable_native_thinking"):
        print(
            f"[GenerateTask] LLM.enable_native_thinking: {llm.enable_native_thinking}"
        )

    if llm is not None:
        print("[GenerateTask] Calling LLM for task generation...")

        system_prompt = GENERATE_TASK_SYSTEM_PROMPT
        user_prompt = GENERATE_TASK_USER_PROMPT.format(
            user_input=user_input,
            execution_plan=execution_plan or "Not provided",
            available_files=available_files,
            previous_task_md=state.task_md_content or "None - initial generation",
            user_feedback=user_feedback or "None - initial generation",
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = llm.invoke(messages)
            response_text = _extract_response_text(response)

            print(f"[GenerateTask] LLM response: {len(response_text)} chars")

            result = _parse_generate_task_response(response_text)

            state.task_md_content = result["task_md"]
            state.missing_parameters = result["missing_parameters"]

            if not state.task_md_content:
                state.task_md_content = _generate_fallback_task_md(state)

            print(
                f"[GenerateTask] Generated task.md: {len(state.task_md_content)} chars"
            )
            print(f"[GenerateTask] Missing parameters: {len(state.missing_parameters)}")

        except Exception as e:
            print(f"[GenerateTask] LLM call failed: {e}")
            state.task_md_content = _generate_fallback_task_md(state)
            state.missing_parameters = []
    else:
        print("[GenerateTask] LLM unavailable, using fallback")
        state.task_md_content = _generate_fallback_task_md(state)
        state.missing_parameters = []

    state.hitl_request = {
        "type": "task_review",
        "task_md": state.task_md_content,
        "missing_parameters": state.missing_parameters,
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
    }

    if not state.merged_result:
        state.merged_result = {}
    state.merged_result["task_md_content"] = state.task_md_content
    state.merged_result["missing_parameters"] = state.missing_parameters

    print(f"{'=' * 60}")
    print("[GenerateTask] Task generation complete")
    print(f"{'=' * 60}")

    return state


def generate_task_router(state: GlobalState) -> str:
    """
    Router for generate_task node.

    Returns:
        "hitl" if task needs user review
        "orchestrator" if user has confirmed (hitl_confirmed=True)
    """
    if state.hitl_confirmed:
        print("[GenerateTask] User confirmed, proceeding to orchestrator")
        return "orchestrator"

    print("[GenerateTask] Task needs user review, going to HITL")
    return "hitl"
