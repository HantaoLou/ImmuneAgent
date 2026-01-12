from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import shutil
import sys
from pathlib import Path

from .prompt import TASK_CLASSIFICATION_SYSTEM_PROMPT, get_task_classification_user_prompt

# 导入主图状态（用于状态映射）
# 添加agent目录到路径（支持从子图目录导入）
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, UserTaskType

# LLM相关导入（使用公共LLM工厂）
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_reasoning_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_llm = None
    HumanMessage = None
    SystemMessage = None
    print("警告：langchain相关库未安装，将使用关键字判断作为降级方案")


# ---------------------- Supervisor 状态模型 ----------------------
class SupervisorState(BaseModel):
    """Supervisor Agent 子图状态"""
    user_input: str = Field(description="用户原始输入")
    user_task_type: Optional[UserTaskType] = Field(default=None, description="用户任务类型")
    uploaded_files: List[str] = Field(default_factory=list, description="上传的文件路径列表（原始路径）")
    sandbox_file_paths: Dict[str, str] = Field(default_factory=dict, description="沙盒文件路径映射（原始路径 -> 沙盒路径）")
    sandbox_dir: str = Field(description="沙盒目录路径")
    execution_plan: Optional[str] = Field(default=None, description="执行计划（如果用户提供了计划）")


# ---------------------- LLM实例化（使用公共LLM工厂） ----------------------
def _get_llm():
    """
    获取推理模型实例（用于任务分类）
    
    使用公共LLM工厂创建推理模型，优先使用推理性能好的模型。
    
    Returns:
        LLM实例，如果都不可用则返回None
    """
    if not LLM_AVAILABLE or create_reasoning_llm is None:
        return None
    
    # 使用推理模型（用于任务分类）
    return create_reasoning_llm(temperature=0.1)


# ---------------------- 节点1：用户描述判别节点 ----------------------
def user_description_classify_node(state: SupervisorState) -> SupervisorState:
    """
    用户描述判别节点：
    1. 根据用户输入，判断节点属于【普通问答】、【执行给定计划】、【免疫学相关任务】的哪一种
    2. 如果用户上传了文件，需要将文件下载到约定的沙盒目录中
    """
    user_input = state.user_input
    
    # 1. 判断任务类型
    task_type = _classify_user_task_type(user_input)
    state.user_task_type = task_type
    
    # 2. 检查并处理上传的文件
    # if state.uploaded_files:
    #     # 确保沙盒目录存在
    #     sandbox_path = Path(state.sandbox_dir)
    #     sandbox_path.mkdir(parents=True, exist_ok=True)
        
    #     # 下载/复制文件到沙盒目录
    #     for uploaded_file_path in state.uploaded_files:
    #         sandbox_file_path = _download_file_to_sandbox(
    #             uploaded_file_path, 
    #             sandbox_path
    #         )
    #         if sandbox_file_path:
    #             state.sandbox_file_paths[uploaded_file_path] = sandbox_file_path
    
    return state


def _classify_user_task_type_with_llm(user_input: str, llm) -> Optional[UserTaskType]:
    """
    使用LLM根据用户输入判断任务类型
    
    Args:
        user_input: 用户输入
        llm: LLM实例
    
    Returns:
        任务类型，如果判断失败返回None
    """
    # 使用集中的提示词模板
    system_prompt = TASK_CLASSIFICATION_SYSTEM_PROMPT
    user_prompt = get_task_classification_user_prompt(user_input)

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        result_text = response.content.strip()
        
        # 提取任务类型（允许部分匹配以提高鲁棒性）
        result_lower = result_text.lower()
        if "计划" in result_text:
            return UserTaskType.EXECUTE_PLAN
        elif "免疫" in result_text:
            return UserTaskType.IMMUNOLOGY_TASK
        elif "普通" in result_text or "问答" in result_text:
            return UserTaskType.GENERAL_QA
        else:
            # 如果无法解析，尝试模糊匹配
            if any(word in result_lower for word in ["plan", "计划", "步骤", "step", "执行"]):
                return UserTaskType.EXECUTE_PLAN
            elif any(word in result_lower for word in ["immun", "免疫", "antigen", "antibody", "抗原", "抗体"]):
                return UserTaskType.IMMUNOLOGY_TASK
            else:
                return UserTaskType.GENERAL_QA
                
    except Exception as e:
        print(f"错误：LLM判断任务类型失败: {e}")
        return None

def _classify_user_task_type(user_input: str) -> UserTaskType:
    """
    根据用户输入判断任务类型（优先使用LLM，失败时降级到关键字判断）
    
    Args:
        user_input: 用户输入
    
    Returns:
        任务类型
    """
    # 尝试使用LLM判断（使用公共LLM工厂）
    llm = _get_llm()
    if llm is not None:
        result = _classify_user_task_type_with_llm(user_input, llm)
        if result is not None:
            print(f"LLM判断任务类型: {result.value}")
            return result
    
    # LLM不可用或失败时，使用关键字匹配作为降级方案
    print("使用关键字匹配作为降级方案")
    user_input_lower = user_input.lower()
    
    # 检查执行计划相关关键词
    if any(keyword in user_input for keyword in ["执行", "计划", "步骤", "按照", "按照以下", "按照这个"]):
        return UserTaskType.EXECUTE_PLAN
    
    # 检查免疫学相关关键词
    if any(keyword in user_input_lower for keyword in ["免疫", "抗原", "抗体", "疫苗", "免疫系统", "免疫细胞", "t细胞", "b细胞", "免疫反应"]):
        return UserTaskType.IMMUNOLOGY_TASK
    
    # 默认为普通问答
    return UserTaskType.GENERAL_QA


def _download_file_to_sandbox(source_file_path: str, sandbox_dir: Path) -> Optional[str]:
    """
    将文件下载/复制到沙盒目录
    
    Args:
        source_file_path: 源文件路径（可能是URL或本地路径）
        sandbox_dir: 沙盒目录路径
    
    Returns:
        沙盒中的文件路径，如果失败返回None
    """
    try:
        source_path = Path(source_file_path)
        
        # 如果是URL，需要下载（这里简化处理，实际可能需要使用requests等库）
        if source_file_path.startswith(("http://", "https://")):
            # TODO: 实现HTTP文件下载逻辑
            # 这里先返回None，后续可以扩展
            print(f"警告：URL文件下载功能暂未实现: {source_file_path}")
            return None
        
        # 如果是本地文件，复制到沙盒目录
        if source_path.exists() and source_path.is_file():
            # 生成目标文件路径（保持文件名）
            target_file_path = sandbox_dir / source_path.name
            
            # 如果目标文件已存在，添加数字后缀避免覆盖
            counter = 1
            while target_file_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                target_file_path = sandbox_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # 复制文件
            shutil.copy2(source_path, target_file_path)
            print(f"文件已复制到沙盒: {source_path} -> {target_file_path}")
            return str(target_file_path)
        else:
            print(f"警告：源文件不存在: {source_file_path}")
            return None
            
    except Exception as e:
        print(f"错误：复制文件到沙盒失败 {source_file_path}: {str(e)}")
        return None


# ---------------------- 状态映射函数 =====================
def supervisor_input_mapper(global_state: GlobalState) -> SupervisorState:
    """
    主图→子图的状态映射
    
    将主图的 GlobalState 映射为 SupervisorState，提取子图需要的信息。
    
    Args:
        global_state: 主图的全局状态
    
    Returns:
        SupervisorState: 子图状态
    """
    return SupervisorState(
        user_input=global_state.user_input,
        user_task_type=None,  # 将在子图中判断
        uploaded_files=[],  # TODO: 从 global_state 中获取上传的文件信息
        sandbox_file_paths={},
        sandbox_dir=global_state.sandbox_dir,
        execution_plan=None  # TODO: 从 global_state 中提取执行计划（如果有）
    )


def supervisor_output_mapper(subgraph_output: SupervisorState | dict, global_state: GlobalState) -> GlobalState:
    """
    子图→主图的状态映射
    
    将子图的 SupervisorState 结果同步回主图的 GlobalState。
    
    Args:
        subgraph_output: 子图输出的状态（可能是 SupervisorState 对象或字典）
        global_state: 主图的全局状态（将被更新）
    
    Returns:
        GlobalState: 更新后的主图状态
    """
    
    # 处理字典格式的状态（LangGraph 可能返回字典）
    if isinstance(subgraph_output, dict):
        subgraph_output = SupervisorState(**subgraph_output)
    
    # 将任务类型判断结果存储到 user_task_type 中
    if subgraph_output.user_task_type:
        global_state.user_task_type = subgraph_output.user_task_type
    
    # 同步执行计划（如果判断出了执行计划）
    if subgraph_output.execution_plan:
        global_state.execution_plan = subgraph_output.execution_plan
    
    # 同步沙盒文件路径（如果需要，可以存储到 merged_result）
    if subgraph_output.sandbox_file_paths:
        global_state.file_paths = subgraph_output.sandbox_file_paths
    
    # 返回更新后的全局状态
    return global_state


# ---------------------- 构建 Supervisor Agent 子图 ----------------------
def build_supervisor_subgraph():
    """
    构建监督者Agent子图
    
    使用公共LLM工厂创建LLM实例，优先使用通义千问，其次使用其他模型。
    
    Returns:
        编译后的子图
    """
    graph = StateGraph(SupervisorState)
    
    # 添加节点
    graph.add_node("classify_user_description", user_description_classify_node)
    
    # 定义流转规则
    graph.add_edge(START, "classify_user_description")
    # TODO: 根据判别结果路由到不同的后续节点
    graph.add_edge("classify_user_description", END)
    
    return graph.compile()
