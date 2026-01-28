from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import shutil
import sys
from pathlib import Path

from .prompt import TASK_CLASSIFICATION_SYSTEM_PROMPT, get_task_classification_user_prompt

# Import main graph state (for state mapping)
# Add agent directory to path (support import from subgraph directory)
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, UserTaskType

# LLM-related imports (using common LLM factory)
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_reasoning_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_llm = None
    HumanMessage = None
    SystemMessage = None
    print("Warning: langchain-related libraries not installed, will use keyword matching as fallback")


# ---------------------- Supervisor State Model ----------------------
class SupervisorState(BaseModel):
    """Supervisor Agent subgraph state"""
    user_input: str = Field(description="User's original input")
    user_task_type: Optional[UserTaskType] = Field(default=None, description="User task type")
    uploaded_files: List[str] = Field(default_factory=list, description="List of uploaded file paths (original paths)")
    sandbox_file_paths: Dict[str, str] = Field(default_factory=dict, description="Sandbox file path mapping (original path -> sandbox path)")
    sandbox_dir: str = Field(description="Sandbox directory path")
    execution_plan: Optional[str] = Field(default=None, description="Execution plan (if user provided a plan)")


# ---------------------- LLM Instantiation (using common LLM factory) ----------------------
def _get_llm():
    """
    Get reasoning model instance (for task classification)
    
    Use the common LLM factory to create a reasoning model, prioritizing models with good reasoning performance.
    
    Returns:
        LLM instance, returns None if all are unavailable
    """
    if not LLM_AVAILABLE or create_reasoning_llm is None:
        return None
    
    # Use reasoning model (for task classification)
    return create_reasoning_llm(temperature=0.1)


# ---------------------- Node 1: User Description Classification Node ----------------------
def user_description_classify_node(state: SupervisorState) -> SupervisorState:
    """
    User description classification node:
    1. Based on user input, determine which type the task belongs to: 【General Q&A】, 【Execute Given Plan】, or 【Immunology-Related Task】
    2. If user uploaded files, download them to the agreed sandbox directory
    """
    user_input = state.user_input
    
    # 1. Determine task type
    task_type = _classify_user_task_type(user_input)
    state.user_task_type = task_type
    
    # 2. Check and process uploaded files
    if state.uploaded_files:
        # Ensure sandbox directory exists
        sandbox_path = Path(state.sandbox_dir)
        sandbox_path.mkdir(parents=True, exist_ok=True)
        
        # Download/copy files to sandbox directory
        for uploaded_file_path in state.uploaded_files:
            sandbox_file_path = _download_file_to_sandbox(
                uploaded_file_path, 
                sandbox_path
            )
            if sandbox_file_path:
                state.sandbox_file_paths[uploaded_file_path] = sandbox_file_path
    
    return state


def _classify_user_task_type_with_llm(user_input: str, llm) -> Optional[UserTaskType]:
    """
    Use LLM to classify task type based on user input
    
    Args:
        user_input: User input
        llm: LLM instance
    
    Returns:
        Task type, returns None if classification fails
    """
    # Use centralized prompt templates
    system_prompt = TASK_CLASSIFICATION_SYSTEM_PROMPT
    user_prompt = get_task_classification_user_prompt(user_input)

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        result_text = response.content.strip()
        
        # Extract task type (allow partial matching for robustness)
        result_lower = result_text.lower()
        if "plan" in result_lower or "execute" in result_lower:
            return UserTaskType.EXECUTE_PLAN
        elif "immun" in result_lower or "antigen" in result_lower or "antibody" in result_lower:
            return UserTaskType.IMMUNOLOGY_TASK
        elif "general" in result_lower or "qa" in result_lower or "q&a" in result_lower:
            return UserTaskType.GENERAL_QA
        else:
            # If cannot parse, try fuzzy matching
            if any(word in result_lower for word in ["plan", "step", "execute", "instruction"]):
                return UserTaskType.EXECUTE_PLAN
            elif any(word in result_lower for word in ["immun", "antigen", "antibody", "vaccine"]):
                return UserTaskType.IMMUNOLOGY_TASK
            else:
                return UserTaskType.GENERAL_QA
                
    except Exception as e:
        # Check if it's an authentication error (API Key error)
        error_str = str(e).lower()
        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
            print(f"⚠ LLM API Key authentication failed, will use keyword matching as fallback: {type(e).__name__}")
            print(f"  Tip: Please check if the API Key in environment variables is correctly configured")
        elif "rate limit" in error_str or "429" in error_str:
            print(f"⚠ LLM API rate limit exceeded, will use keyword matching as fallback: {type(e).__name__}")
        else:
            print(f"⚠ LLM task type classification failed, will use keyword matching as fallback: {type(e).__name__}: {str(e)[:100]}")
        
        return None

def _classify_user_task_type(user_input: str) -> UserTaskType:
    """
    Classify task type based on user input (prioritize LLM, fallback to keyword matching on failure)
    
    Args:
        user_input: User input
    
    Returns:
        Task type
    """
    # Try using LLM classification (using common LLM factory)
    llm = _get_llm()
    if llm is not None:
        result = _classify_user_task_type_with_llm(user_input, llm)
        if result is not None:
            print(f"LLM classified task type: {result.value}")
            return result
    
    # When LLM is unavailable or fails, use keyword matching as fallback
    print("Using keyword matching as fallback")
    user_input_lower = user_input.lower()
    
    # Check execution plan related keywords
    if any(keyword in user_input_lower for keyword in [
        "execute", "plan", "step", "follow", "according to", "instruction",
        "执行", "计划", "步骤", "按照", "依据", "流程"
    ]):
        return UserTaskType.EXECUTE_PLAN
    
    # Check immunology related keywords
    if any(keyword in user_input_lower for keyword in [
        "immun", "antigen", "antibody", "vaccine", "immune system", "immune cell", "t cell", "b cell", "immune response",
        "免疫", "抗原", "抗体", "疫苗", "免疫系统", "免疫细胞", "t细胞", "b细胞", "免疫反应"
    ]):
        return UserTaskType.IMMUNOLOGY_TASK
    
    # Default to general Q&A
    return UserTaskType.GENERAL_QA


def _download_file_to_sandbox(source_file_path: str, sandbox_dir: Path) -> Optional[str]:
    """
    Download/copy file to sandbox directory
    
    Args:
        source_file_path: Source file path (may be URL or local path)
        sandbox_dir: Sandbox directory path
    
    Returns:
        File path in sandbox, returns None if failed
    """
    try:
        source_path = Path(source_file_path)
        
        # If it's a URL, need to download (simplified handling here, may need requests library in practice)
        if source_file_path.startswith(("http://", "https://")):
            # TODO: Implement HTTP file download logic
            # Return None for now, can be extended later
            print(f"Warning: URL file download functionality not yet implemented: {source_file_path}")
            return None
        
        # If it's a local file, copy to sandbox directory
        if source_path.exists() and source_path.is_file():
            # Generate target file path (preserve filename)
            target_file_path = sandbox_dir / source_path.name
            
            # If target file exists, add numeric suffix to avoid overwriting
            counter = 1
            while target_file_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                target_file_path = sandbox_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy file
            shutil.copy2(source_path, target_file_path)
            print(f"File copied to sandbox: {source_path} -> {target_file_path}")
            return str(target_file_path)
        else:
            print(f"Warning: Source file does not exist: {source_file_path}")
            return None
            
    except Exception as e:
        print(f"Error: Failed to copy file to sandbox {source_file_path}: {str(e)}")
        return None


# ---------------------- State Mapping Functions =====================
def supervisor_input_mapper(global_state: GlobalState) -> SupervisorState:
    """
    Main graph → subgraph state mapping
    
    Map the main graph's GlobalState to SupervisorState, extracting information needed by the subgraph.
    
    Args:
        global_state: Main graph's global state
    
    Returns:
        SupervisorState: Subgraph state
    """
    uploaded_files = list(global_state.file_paths.keys()) if global_state.file_paths else []
    return SupervisorState(
        user_input=global_state.user_input,
        user_task_type=None,  # Will be determined in subgraph
        uploaded_files=uploaded_files,
        sandbox_file_paths=dict(global_state.file_paths) if global_state.file_paths else {},
        sandbox_dir=global_state.sandbox_dir,
        execution_plan=global_state.execution_plan
    )


def supervisor_output_mapper(subgraph_output: SupervisorState | dict, global_state: GlobalState) -> GlobalState:
    """
    Subgraph → main graph state mapping
    
    Synchronize the subgraph's SupervisorState results back to the main graph's GlobalState.
    
    Args:
        subgraph_output: Subgraph output state (may be SupervisorState object or dict)
        global_state: Main graph's global state (will be updated)
    
    Returns:
        GlobalState: Updated main graph state
    """
    
    # Handle dict format state (LangGraph may return dict)
    if isinstance(subgraph_output, dict):
        subgraph_output = SupervisorState(**subgraph_output)
    
    # Store task type classification result to user_task_type
    if subgraph_output.user_task_type:
        global_state.user_task_type = subgraph_output.user_task_type
    
    # Synchronize execution plan (if execution plan was determined)
    if subgraph_output.execution_plan:
        global_state.execution_plan = subgraph_output.execution_plan
    
    # Synchronize sandbox file paths (if needed, can store to merged_result)
    if subgraph_output.sandbox_file_paths:
        global_state.file_paths = subgraph_output.sandbox_file_paths
    
    # Return updated global state
    return global_state


# ---------------------- Build Supervisor Agent Subgraph ----------------------
def build_supervisor_subgraph():
    """
    Build Supervisor Agent subgraph
    
    Use common LLM factory to create LLM instance, prioritizing models with good reasoning performance.
    
    Returns:
        Compiled subgraph
    """
    graph = StateGraph(SupervisorState)
    
    graph.add_node("classify_user_description", user_description_classify_node)
    
    graph.add_edge(START, "classify_user_description")

    graph.add_edge("classify_user_description", END)
    
    return graph.compile()
