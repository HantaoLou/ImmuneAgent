"""React-style executor that loops over CodeAct for a single task."""

from typing import Any, Callable, Dict, List, Optional
import time

from core.react_state import ReactStep, ReactStepType


def _summarize_react_steps(steps: List[ReactStep], max_steps: int = 5) -> str:
    tail = steps[-max_steps:] if steps else []
    return " | ".join(f"{step.step_type.value}:{step.content}" for step in tail)


def execute_with_react(
    task: Any,
    state: Any,
    result: Any,
    base_execution_mode: Any,
    parameters: Dict[str, Any],
    run_codeact: Callable[[Any, Dict[str, Any], Optional[str], Optional[str], Optional[str]], Any],
    handle_streaming: Callable[[Any, Any], Optional[Dict[str, Any]]],
    classify_error: Callable[[str, str], Any],
    analyze_failure: Callable[[str, str, Any], str],
    generate_suggestions: Callable[[Any, str, int, int], List[str]],
    status_enum: Any,
    error_category_enum: Any,
    fix_code_mode: Any,
    fix_parameter_mode: Any,
    max_steps: int = 3
) -> Any:
    """
    Execute a single task with a React-style loop.

    The loop alternates between ACT/RESULT steps and selects fix_code or
    fix_parameter based on error classification.
    """
    start_time = time.time()
    react_steps: List[ReactStep] = []

    react_steps.append(ReactStep(
        step_type=ReactStepType.OBS,
        content=f"task_id={task.task_id}; mode={base_execution_mode.value}"
    ))
    react_steps.append(ReactStep(
        step_type=ReactStepType.THINK,
        content=f"max_steps={max_steps}; params={list(parameters.keys()) if parameters else []}"
    ))

    previous_code: Optional[str] = None
    previous_error: Optional[str] = None
    error_category_value: Optional[str] = None
    current_mode = base_execution_mode

    termination_reason: Optional[str] = None
    for step in range(max_steps):
        react_steps.append(ReactStep(
            step_type=ReactStepType.ACT,
            content=f"step={step + 1}; execution_mode={current_mode.value}"
        ))

        exec_result, codeact_state = run_codeact(
            current_mode,
            parameters,
            previous_code,
            previous_error,
            error_category_value
        )

        result.code = exec_result.get("code")
        status = exec_result.get("status")

        if status == "success":
            output = exec_result.get("output")
            streaming_result = handle_streaming(output, task)
            if streaming_result:
                streaming_status = streaming_result.get("status")
                if streaming_status is None:
                    result.status = status_enum.FAILED
                    result.error = streaming_result.get("error", "Streaming task returned no status")
                elif isinstance(streaming_status, status_enum):
                    result.status = streaming_status
                elif isinstance(streaming_status, str) and streaming_status.lower() == "completed":
                    result.status = status_enum.COMPLETED
                elif isinstance(streaming_status, str) and streaming_status.lower() == "failed":
                    result.status = status_enum.FAILED
                else:
                    result.status = status_enum.FAILED
                    result.error = f"Streaming task returned unknown status: {streaming_status}"
                result.output = streaming_result.get("output", output)
                if streaming_result.get("error"):
                    result.error = streaming_result.get("error")
                    result.status = status_enum.FAILED
            else:
                result.status = status_enum.COMPLETED
                result.output = output

            react_steps.append(ReactStep(
                step_type=ReactStepType.RESULT,
                content=f"status=success; output_type={type(result.output).__name__}"
            ))
            termination_reason = "success"
            break

        result.status = status_enum.FAILED
        result.error = exec_result.get("error") or "Execution failed"
        error_type = exec_result.get("error_type") or "UnknownError"
        error_for_classification = result.error if result.error else "Execution failed"
        result.error_category = classify_error(error_for_classification, error_type)
        result.failure_analysis = analyze_failure(error_for_classification, error_type, result.error_category)
        result.suggestions = generate_suggestions(
            result.error_category,
            error_for_classification,
            result.retry_count,
            state.max_retries
        )

        react_steps.append(ReactStep(
            step_type=ReactStepType.RESULT,
            content=f"status=failed; error_type={error_type}"
        ))

        previous_code = exec_result.get("code")
        previous_error = result.error
        error_category_value = (
            result.error_category.value
            if hasattr(result.error_category, "value")
            else str(result.error_category)
        )

        if result.error_category == error_category_enum.PARAMETER_ERROR:
            current_mode = fix_parameter_mode
        else:
            current_mode = fix_code_mode
    else:
        termination_reason = "max_steps_reached"

    result.execution_time = time.time() - start_time
    if not termination_reason:
        termination_reason = "failed"

    result.result_summary = {
        "react_steps": [step.model_dump() for step in react_steps],
        "react_summary": _summarize_react_steps(react_steps, max_steps=5),
        "react_max_steps": max_steps,
        "react_termination_reason": termination_reason
    }
    return result

