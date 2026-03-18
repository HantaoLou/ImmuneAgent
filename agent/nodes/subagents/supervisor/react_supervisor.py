from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import shutil
import sys
from pathlib import Path
import json

# Add agent directory to path (support import from subgraph directory)
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, UserTaskType
from nodes.subagents.supervisor.prompt import (
    TASK_CLASSIFICATION_SYSTEM_PROMPT,
    get_task_classification_user_prompt,
)

# LLM-related imports (using common LLM factory)
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_llm_with_thinking, create_reasoning_llm

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_llm = None
    create_llm_with_thinking = None
    HumanMessage = None
    SystemMessage = None
    print(
        "Warning: langchain-related libraries not installed, will use keyword matching as fallback"
    )


class ReactSupervisorState(BaseModel):
    """React Supervisor subgraph state"""

    user_input: str = Field(description="User's original input")
    user_task_type: Optional[UserTaskType] = Field(
        default=None, description="User task type"
    )
    decision: Optional[str] = Field(default=None, description="Decision label")
    reasoning: Optional[str] = Field(default=None, description="Decision reasoning")
    uploaded_files: List[str] = Field(
        default_factory=list, description="List of uploaded file paths (original paths)"
    )
    sandbox_file_paths: Dict[str, str] = Field(
        default_factory=dict,
        description="Sandbox file path mapping (original path -> sandbox path)",
    )
    sandbox_dir: str = Field(description="Sandbox directory path")
    execution_plan: Optional[str] = Field(
        default=None, description="Execution plan (if user provided a plan)"
    )
    progress_callback: Optional[Any] = Field(
        default=None, description="Progress callback for thinking capture"
    )
    session_id: Optional[str] = Field(
        default=None, description="Session ID for tracking"
    )


def _get_llm(
    progress_callback: Optional[Any] = None,
    session_id: Optional[str] = None,
    node_name: str = "supervisor",
):
    """
    获取 LLM 实例

    Args:
        progress_callback: 进度回调函数
        session_id: 会话ID
        node_name: 节点名称
    """
    if not LLM_AVAILABLE:
        return None

    if create_llm_with_thinking is not None:
        return create_llm_with_thinking(
            purpose="reasoning",
            temperature=0.1,
            progress_callback=progress_callback,
            session_id=session_id,
            node_name=node_name,
        )

    if create_reasoning_llm is not None:
        return create_reasoning_llm(temperature=0.1)

    return None


def _normalize_decision(decision: str) -> UserTaskType:
    decision_lower = (decision or "").lower()
    if "plan" in decision_lower or "execute" in decision_lower:
        return UserTaskType.EXECUTE_PLAN
    if (
        "immun" in decision_lower
        or "antigen" in decision_lower
        or "antibody" in decision_lower
    ):
        return UserTaskType.IMMUNOLOGY_TASK
    if "general" in decision_lower or "qa" in decision_lower or "q&a" in decision_lower:
        return UserTaskType.GENERAL_QA
    return UserTaskType.GENERAL_QA


def _classify_with_llm(user_input: str, llm) -> Optional[Dict[str, str]]:
    system_prompt = (
        TASK_CLASSIFICATION_SYSTEM_PROMPT
        + """

Return a JSON object with keys:
- decision: one of "General Q&A" / "Execute Given Plan" / "Immunology-Related Task"
- reasoning: a brief explanation (1-2 sentences)
"""
    )
    user_prompt = get_task_classification_user_prompt(user_input)
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        result_text = response.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.strip("`").strip()
        return json.loads(result_text)
    except Exception as e:
        error_str = str(e).lower()
        if (
            "authentication" in error_str
            or "api key" in error_str
            or "401" in error_str
        ):
            print(
                f"[WARN] LLM API Key authentication failed, fallback to keyword matching: {type(e).__name__}"
            )
        elif "rate limit" in error_str or "429" in error_str:
            print(
                f"[WARN] LLM API rate limit exceeded, fallback to keyword matching: {type(e).__name__}"
            )
        else:
            print(
                f"[WARN] LLM task type classification failed, fallback to keyword matching: {type(e).__name__}: {str(e)[:100]}"
            )
        return None


def _classify_user_task_type(
    user_input: str,
    progress_callback: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> Dict[str, str]:
    llm = _get_llm(
        progress_callback=progress_callback,
        session_id=session_id,
        node_name="supervisor",
    )
    if llm is not None:
        llm_result = _classify_with_llm(user_input, llm)
        if llm_result:
            return {
                "decision": llm_result.get("decision", ""),
                "reasoning": llm_result.get("reasoning", "Classified by LLM."),
            }

    user_input_lower = user_input.lower()
    if any(
        keyword in user_input_lower
        for keyword in [
            "execute",
            "plan",
            "step",
            "follow",
            "according to",
            "instruction",
            "执行",
            "计划",
            "步骤",
            "按照",
            "依据",
            "流程",
        ]
    ):
        return {
            "decision": "Execute Given Plan",
            "reasoning": "Matched execution-plan keywords.",
        }
    if any(
        keyword in user_input_lower
        for keyword in [
            "immun",
            "antigen",
            "antibody",
            "vaccine",
            "immune system",
            "immune cell",
            "t cell",
            "b cell",
            "immune response",
            "免疫",
            "抗原",
            "抗体",
            "疫苗",
            "免疫系统",
            "免疫细胞",
            "t细胞",
            "b细胞",
            "免疫反应",
        ]
    ):
        return {
            "decision": "Immunology-Related Task",
            "reasoning": "Matched immunology keywords.",
        }
    return {"decision": "General Q&A", "reasoning": "Defaulted to general QA."}


def user_description_classify_node(state: ReactSupervisorState) -> ReactSupervisorState:
    user_input = state.user_input
    classification = _classify_user_task_type(
        user_input,
        progress_callback=get_progress_callback_by_session(state.session_id),
        session_id=state.session_id,
    )
    state.decision = classification.get("decision")
    state.reasoning = classification.get("reasoning")
    state.user_task_type = _normalize_decision(state.decision or "")

    # Skip file processing if already done by preprocess_user_input_node
    if state.uploaded_files and not state.sandbox_file_paths:
        sandbox_path = Path(state.sandbox_dir)
        sandbox_path.mkdir(parents=True, exist_ok=True)
        for uploaded_file_path in state.uploaded_files:
            # Skip remote paths (they are handled by OpenSandbox)
            if uploaded_file_path.startswith(("/data/", "/home/", "/opt/", "/mnt/")):
                continue
            sandbox_file_path = _download_file_to_sandbox(
                uploaded_file_path, sandbox_path
            )
            if sandbox_file_path:
                state.sandbox_file_paths[uploaded_file_path] = sandbox_file_path
    return state


def _download_file_to_sandbox(
    source_file_path: str, sandbox_dir: Path
) -> Optional[str]:
    try:
        source_path = Path(source_file_path)
        if source_file_path.startswith(("http://", "https://")):
            print(
                f"Warning: URL file download functionality not yet implemented: {source_file_path}"
            )
            return None
        if source_path.exists() and source_path.is_file():
            target_file_path = sandbox_dir / source_path.name
            counter = 1
            while target_file_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                target_file_path = sandbox_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            shutil.copy2(source_path, target_file_path)
            print(f"File copied to sandbox: {source_path} -> {target_file_path}")
            return str(target_file_path)
        print(f"Warning: Source file does not exist: {source_file_path}")
        return None
    except Exception as e:
        print(f"Error: Failed to copy file to sandbox {source_file_path}: {str(e)}")
        return None


def supervisor_input_mapper(global_state: GlobalState) -> ReactSupervisorState:
    return ReactSupervisorState(
        user_input=global_state.user_input,
        user_task_type=global_state.user_task_type,
        uploaded_files=list(global_state.file_paths.values())
        if global_state.file_paths
        else [],
        sandbox_dir=global_state.sandbox_dir,
        execution_plan=global_state.execution_plan,
    )


def supervisor_output_mapper(
    subgraph_output: ReactSupervisorState | dict, global_state: GlobalState
) -> GlobalState:
    if isinstance(subgraph_output, dict):
        subgraph_output = ReactSupervisorState(**subgraph_output)

    # 确保 user_task_type 是枚举类型而不是字符串
    task_type = subgraph_output.user_task_type
    if task_type is not None:
        if isinstance(task_type, str):
            try:
                from state import UserTaskType

                task_type = UserTaskType(task_type)
            except (ValueError, KeyError):
                # 如果转换失败，保持原值
                pass
        global_state.user_task_type = task_type

    global_state.supervisor_decision = subgraph_output.decision
    global_state.supervisor_reasoning = subgraph_output.reasoning
    if subgraph_output.sandbox_file_paths:
        global_state.file_paths.update(subgraph_output.sandbox_file_paths)
    return global_state


def build_react_supervisor_subgraph():
    graph = StateGraph(ReactSupervisorState)
    graph.add_node("classify", user_description_classify_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", END)
    return graph.compile()
