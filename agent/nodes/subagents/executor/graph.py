"""
Executor Agent 子图（优化版）

负责执行任务列表中的任务，主要职责：
1. 任务初始化：无依赖任务标记为就绪，有依赖任务标记为等待依赖
2. 参数推断：使用 LLM 推断任务参数值，无法推断的触发 HITL
3. 并行执行：并行执行就绪任务（有并行上限）
4. 结果推理：使用 LLM 评估结果是否满足要求，不满足触发 HITL
5. 依赖管理：任务完成后激活依赖任务
6. 任务调度：并行数量空缺时从就绪任务中取出执行
7. 结果汇总：所有任务完成后汇总结果

充分利用 LangGraph 1.0+ 特性：
- 使用 interrupt 机制实现真正的 HITL
- 使用 checkpoint 实现状态持久化
- 优化异步执行和状态管理
"""

from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
from langgraph.graph import StateGraph, START, END
try:
    from langgraph.types import interrupt, Command
    INTERRUPT_AVAILABLE = True
except ImportError:
    # 如果 interrupt 不可用，定义一个占位函数
    INTERRUPT_AVAILABLE = False
    def interrupt(value: Any = None):
        """占位 interrupt 函数（如果 LangGraph 版本不支持）"""
        raise NotImplementedError("interrupt 功能需要 LangGraph 支持，请确保已安装正确版本")
    Command = None


# ===================== Interrupt 辅助函数 =====================

def safe_interrupt(interrupt_value: Any = None) -> Optional[Any]:
    """
    安全调用 interrupt 函数
    
    LangGraph 的 interrupt 机制：
    - 首次调用时：抛出 GraphInterrupt 异常，LangGraph 会捕获并保存状态
    - 恢复时：interrupt() 会返回 Command(resume=...) 中的 resume 值
    
    Args:
        interrupt_value: 中断时传递的值（用于标识中断原因）
    
    Returns:
        如果是恢复执行，返回 resume 值；如果是首次调用，返回 None（但会抛出异常）
    """
    if not INTERRUPT_AVAILABLE:
        return None
    
    try:
        # 尝试调用 interrupt
        # 如果是首次调用，这会抛出 GraphInterrupt 异常
        # 如果是恢复执行，这会返回 resume 值
        resume_value = interrupt(interrupt_value)
        return resume_value
    except Exception as e:
        # 首次调用时，interrupt 会抛出异常（这是正常行为）
        # LangGraph 会捕获这个异常并保存状态
        # 我们不需要在这里处理，让异常向上传播
        raise
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # 如果 MemorySaver 不存在，使用简单的内存存储
    MemorySaver = None
import sys
import json
from pathlib import Path
from enum import Enum
import time

# 导入主图状态和任务模型
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from agent.state import SubTask, TaskStatus, UserTaskType, GlobalState, ParallelTaskGroup

# 导入 CodeAct 子图
from agent.nodes.subagents.code_act.graph import (
    build_codeact_subgraph,
    codeact_input_mapper,
    codeact_output_mapper,
    CodeActExecutionMode,
    CodeActState
)

# 导入 LLM 工厂
from agent.utils.llm_factory import create_reasoning_advanced_llm, create_reasoning_llm

# ===================== Executor 子图状态模型 =====================

class ExecutorTaskStatus(str, Enum):
    """Executor内部任务状态"""
    READY = "就绪"  # 无依赖或依赖已完成，可以执行
    RUNNING = "执行中"  # 正在执行
    COMPLETED = "已完成"  # 执行成功
    FAILED = "失败"  # 执行失败
    WAITING_DEPENDENCY = "等待依赖"  # 等待依赖任务完成
    WAITING_HITL_PARAMS = "等待HITL参数"  # 等待用户提供参数
    WAITING_HITL_CONFIRM = "等待HITL确认"  # 等待用户确认是否继续


class ErrorCategory(str, Enum):
    """错误分类"""
    RETRYABLE = "可重试"  # 网络错误、超时等，可以重试
    CODE_ERROR = "代码错误"  # 代码逻辑错误，需要修改代码
    PARAMETER_ERROR = "参数错误"  # 参数不正确，需要修改参数
    SYSTEM_ERROR = "系统错误"  # 系统级错误，可能需要人工干预


class TaskExecutionResult(BaseModel):
    """任务执行结果"""
    task_id: str
    status: ExecutorTaskStatus
    execution_mode: str  # "mcp_tool" 或 "codeact"
    parameters: Dict[str, Any] = Field(default_factory=dict, description="解析后的参数")
    missing_parameters: List[str] = Field(default_factory=list, description="缺失的参数列表")
    code: Optional[str] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    retry_count: int = 0
    execution_time: float = 0.0
    confidence_score: Optional[float] = Field(default=None, description="结果置信度（0-1）")
    failure_analysis: Optional[str] = Field(default=None, description="失败原因分析")
    suggestions: Optional[List[str]] = Field(default_factory=list, description="改进建议")
    result_satisfied: Optional[bool] = Field(default=None, description="结果是否满足要求")
    user_continue: Optional[bool] = Field(default=None, description="用户是否选择继续执行")


class ExecutorState(BaseModel):
    """Executor子图状态"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
        from_attributes=True
    )
    
    # 输入：来自task_decomposition的任务列表
    subtasks: List[SubTask] = Field(default_factory=list, description="待执行的子任务列表")
    parallel_task_groups: Dict[str, Any] = Field(default_factory=dict, description="并行任务组")
    
    # 任务状态管理
    task_status_map: Dict[str, ExecutorTaskStatus] = Field(default_factory=dict, description="任务ID→状态映射")
    task_results: Dict[str, TaskExecutionResult] = Field(default_factory=dict, description="任务执行结果")
    running_tasks: List[str] = Field(default_factory=list, description="当前正在运行的任务ID列表")
    
    # 执行配置
    max_parallel_tasks: int = Field(default=3, description="最大并行任务数")
    max_retries: int = Field(default=2, description="最大重试次数")
    sandbox_dir: str = Field(default="DEFAULT_SANDBOX_DIR", description="沙盒目录")
    
    # 执行统计
    total_tasks: int = 0
    completed_count: int = 0
    failed_count: int = 0
    
    # 循环检测（防止无限循环）
    activate_iteration_count: int = Field(default=0, description="连续激活迭代次数（用于检测死循环）")
    max_activate_iterations: int = Field(default=10, description="最大连续激活迭代次数")
    
    # HITL相关
    hitl_requests: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="HITL请求（task_id -> 请求信息）")
    hitl_responses: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="HITL响应（task_id -> 响应信息）")
    
    # 父状态引用（用于访问全局状态和更新HITL状态）
    # 注意：exclude=True 避免 LangGraph 序列化时验证失败，但保留在模型中供节点使用
    # 使用 Union 类型允许在验证时接受 GlobalState 实例或 None
    parent_state: Optional[GlobalState] = Field(default=None, exclude=True, description="主图状态引用")
    
    @field_validator('parent_state', mode='before')
    @classmethod
    def validate_parent_state(cls, v: Any) -> Optional[GlobalState]:
        """验证 parent_state，允许 GlobalState 实例或 None"""
        # 如果是 GlobalState 实例或 None，直接返回
        if v is None or isinstance(v, GlobalState):
            return v
        # 如果是字典（反序列化时），尝试转换为 GlobalState
        if isinstance(v, dict):
            try:
                return GlobalState.model_validate(v)
            except:
                return None
        # 其他情况返回 None
        return None


# ===================== 工具函数 =====================

def _load_tools_params_table() -> Dict[str, Dict[str, Any]]:
    """加载工具参数表"""
    tools_params_path = agent_dir / "config" / "tools_params_table.json"
    
    if not tools_params_path.exists():
        return {}
    
    try:
        with open(tools_params_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tools_params_map = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for tool_name, params_info in item.items():
                        tools_params_map[tool_name] = params_info
        elif isinstance(data, dict):
            tools_params_map = data
        
        return tools_params_map
    except Exception as e:
        print(f"⚠ 加载工具参数表失败: {e}")
        return {}


def classify_error(error: str, error_type: str) -> ErrorCategory:
    """分类错误类型"""
    error_lower = error.lower()
    error_type_lower = error_type.lower()
    
    # 可重试的错误
    retryable_keywords = [
        "timeout", "timed out", "connection", "network", "temporary",
        "rate limit", "429", "503", "502", "retry", "busy"
    ]
    if any(keyword in error_lower or keyword in error_type_lower for keyword in retryable_keywords):
        return ErrorCategory.RETRYABLE
    
    # 参数错误
    param_keywords = [
        "parameter", "argument", "invalid argument", "missing required",
        "type error", "value error", "keyerror", "attributeerror"
    ]
    if any(keyword in error_lower or keyword in error_type_lower for keyword in param_keywords):
        return ErrorCategory.PARAMETER_ERROR
    
    # 代码错误
    code_keywords = [
        "syntax", "indentation", "nameerror", "not defined",
        "logic error", "indexerror", "zerodivisionerror"
    ]
    if any(keyword in error_lower or keyword in error_type_lower for keyword in code_keywords):
        return ErrorCategory.CODE_ERROR
    
    # 默认：系统错误
    return ErrorCategory.SYSTEM_ERROR


# ===================== Executor 节点 =====================

def initialize_tasks_node(state: ExecutorState) -> ExecutorState:
    """
    初始化任务节点
    
    1. 展开并行任务组，将所有任务合并到subtasks中
    2. 初始化任务状态映射
    3. 标记无依赖任务为就绪态
    4. 标记有依赖任务为等待依赖态
    """
    # 展开并行任务组中的任务，合并到subtasks中
    all_tasks = list(state.subtasks)  # 复制列表，避免修改原列表
    
    # 从并行任务组中提取所有任务
    # 注意：parallel_task_groups 中的值可能是 ParallelTaskGroup 对象或字典
    parallel_tasks_count = 0
    for group_id, group in state.parallel_task_groups.items():
        group_subtasks = None
        
        # 处理 group 可能是对象或字典的情况
        if isinstance(group, dict):
            # 如果是字典，尝试获取 subtasks
            group_subtasks = group.get('subtasks', [])
            # 如果 subtasks 是字典列表，需要转换为 SubTask 对象
            if group_subtasks and isinstance(group_subtasks[0], dict):
                try:
                    from agent.state import SubTask
                    group_subtasks = [SubTask.model_validate(task_dict) for task_dict in group_subtasks]
                except Exception as e:
                    print(f"  ⚠ 无法将并行任务组 {group_id} 的任务转换为 SubTask 对象: {e}")
                    continue
        elif hasattr(group, 'subtasks'):
            # 如果是对象，直接获取 subtasks
            group_subtasks = group.subtasks
        
        if group_subtasks:
            for task in group_subtasks:
                # 确保 task 是 SubTask 对象
                if isinstance(task, dict):
                    try:
                        from agent.state import SubTask
                        task = SubTask.model_validate(task)
                    except Exception as e:
                        print(f"  ⚠ 无法将任务 {task.get('task_id', 'unknown')} 转换为 SubTask 对象: {e}")
                        continue
                
                # 检查任务是否已经在subtasks中（避免重复）
                if not any(t.task_id == task.task_id for t in all_tasks):
                    all_tasks.append(task)
                    parallel_tasks_count += 1
                    print(f"  [DEBUG] 从并行任务组 {group_id} 添加任务 {task.task_id}")
    
    # 更新subtasks为包含所有任务的完整列表
    state.subtasks = all_tasks
    state.total_tasks = len(state.subtasks)
    
    serial_tasks_count = len(state.subtasks) - parallel_tasks_count
    print(f"✓ 初始化任务：共 {state.total_tasks} 个任务（串行: {serial_tasks_count} 个，并行: {parallel_tasks_count} 个，并行组数: {len(state.parallel_task_groups)}）")
    
    # 初始化所有任务的状态
    for task in state.subtasks:
        if not task.dependencies:
            # 无依赖任务，直接标记为就绪
            state.task_status_map[task.task_id] = ExecutorTaskStatus.READY
        else:
            # 有依赖任务，标记为等待依赖
            state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_DEPENDENCY
    
    ready_count = sum(1 for s in state.task_status_map.values() if s == ExecutorTaskStatus.READY)
    print(f"  就绪任务: {ready_count} 个，等待依赖: {state.total_tasks - ready_count} 个")
    
    return state


def infer_parameters_node(state: ExecutorState) -> ExecutorState:
    """
    参数推断节点
    
    优先使用 task_decomposition 中已推断的参数结果。
    对于已推断的参数：
    - source_type 为 DETERMINED：直接使用推断值
    - source_type 为 FROM_TASK：等待依赖任务完成后获取
    - source_type 为 USER_REQUIRED：标记为缺失参数，触发 HITL
    对于没有推断结果的参数，使用 LLM 重新推断。
    """
    ready_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.READY
        and task.task_id not in state.running_tasks
    ]
    
    if not ready_tasks:
        return state
    
    # 从 global_state 中获取参数推断结果
    parameter_inference_results = {}
    if state.parent_state and state.parent_state.merged_result:
        parameter_inference_results = state.parent_state.merged_result.get("parameter_inference_results", {})
    
    llm = create_reasoning_llm()
    tools_params_map = _load_tools_params_table()
    
    for task in ready_tasks:
        # 初始化任务结果
        if task.task_id not in state.task_results:
            state.task_results[task.task_id] = TaskExecutionResult(
                task_id=task.task_id,
                status=ExecutorTaskStatus.READY,
                execution_mode=""
            )
        
        result = state.task_results[task.task_id]
        task_result = task.result if isinstance(task.result, dict) else {}
        tools = task_result.get("tools", [])
        inputs = task_result.get("inputs", [])
        
        if not tools:
            # 无工具任务，参数为空
            result.parameters = {}
            result.missing_parameters = []
            continue
        
        # 收集所有工具的参数需求
        all_params = {}
        missing_params = []
        
        # 首先使用 task_decomposition 的参数推断结果
        task_inference = parameter_inference_results.get(task.task_id)
        if task_inference:
            inference_params = task_inference.get("parameters", {})
            for param_name, param_info in inference_params.items():
                source_type = param_info.get("source_type", "")
                
                if source_type == "determined":
                    # 直接使用推断值
                    param_value = param_info.get("value")
                    if param_value is not None:
                        all_params[param_name] = param_value
                        print(f"  ✓ 任务 {task.task_id} 参数 {param_name} 使用推断值: {param_value}")
                    else:
                        # 推断值为空，标记为缺失
                        tool_name = task_inference.get("tool_name", "").split(",")[0].strip() if task_inference.get("tool_name") else ""
                        missing_params.append(f"{tool_name}.{param_name}" if tool_name else param_name)
                
                elif source_type == "from_task":
                    # 参数值来自依赖任务，需要等待依赖任务完成
                    source_task_id = param_info.get("source_task_id")
                    source_output_key = param_info.get("source_output_key", param_name)
                    
                    # 检查依赖任务是否已完成
                    if source_task_id in state.task_results:
                        dep_result = state.task_results[source_task_id]
                        if dep_result.status == ExecutorTaskStatus.COMPLETED:
                            # 从依赖任务的结果中获取参数值
                            if isinstance(dep_result.output, dict):
                                param_value = dep_result.output.get(source_output_key)
                            elif isinstance(dep_result.output, str):
                                # 尝试解析字符串输出
                                try:
                                    import json
                                    output_dict = json.loads(dep_result.output)
                                    param_value = output_dict.get(source_output_key)
                                except:
                                    param_value = dep_result.output
                            else:
                                param_value = dep_result.output
                            
                            if param_value is not None:
                                all_params[param_name] = param_value
                                print(f"  ✓ 任务 {task.task_id} 参数 {param_name} 从任务 {source_task_id} 获取: {param_value}")
                            else:
                                tool_name = task_inference.get("tool_name", "").split(",")[0].strip() if task_inference.get("tool_name") else ""
                                missing_params.append(f"{tool_name}.{param_name}" if tool_name else param_name)
                        else:
                            # 依赖任务未完成，标记为缺失（但这是正常的，等待依赖）
                            print(f"  ⏳ 任务 {task.task_id} 参数 {param_name} 等待依赖任务 {source_task_id} 完成")
                            # 不添加到 missing_params，因为这是依赖关系，不是真正的缺失
                    else:
                        # 依赖任务不存在或未执行，标记为缺失
                        tool_name = task_inference.get("tool_name", "").split(",")[0].strip() if task_inference.get("tool_name") else ""
                        missing_params.append(f"{tool_name}.{param_name}" if tool_name else param_name)
                
                elif source_type == "user_required":
                    # 需要用户提供，标记为缺失参数
                    tool_name = task_inference.get("tool_name", "").split(",")[0].strip() if task_inference.get("tool_name") else ""
                    missing_params.append(f"{tool_name}.{param_name}" if tool_name else param_name)
        
        for tool_item in tools:
            tool_name = None
            if isinstance(tool_item, str):
                tool_name = tool_item
            elif isinstance(tool_item, dict):
                tool_name = tool_item.get("tool_name") or tool_item.get("name", "")
            
            if not tool_name:
                continue
            
            # 查找工具参数定义
            tool_params = tools_params_map.get(tool_name)
            
            # 如果工具不在参数表中，使用 inputs 字段来推断参数
            if not tool_params:
                # 从 inputs 字段推断参数
                for input_param in inputs:
                    if input_param not in all_params:
                        # 尝试从任务描述中推断
                        if llm:
                            try:
                                from langchain_core.messages import SystemMessage, HumanMessage
                                inference_prompt = f"""
请根据以下信息推断参数值：

任务描述: {task.content}
工具名称: {tool_name}
参数名称: {input_param}
任务输入列表: {inputs}

请推断该参数的值。如果无法从任务描述中推断，或者需要用户提供（如文件路径、用户选择等），请返回 null。

返回JSON格式：
{{
    "value": <参数值或null>,
    "can_infer": <true/false>,
    "reason": "<推断理由或为什么需要用户提供>"
}}
"""
                                messages = [
                                    SystemMessage(content="你是一个专业的参数推断专家，能够从任务描述中提取参数值。"),
                                    HumanMessage(content=inference_prompt)
                                ]
                                response = llm.invoke(messages)
                                response_text = response.content.strip()
                                
                                import re
                                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                                if json_match:
                                    inference_data = json.loads(json_match.group())
                                    param_value = inference_data.get("value")
                                    can_infer = inference_data.get("can_infer", False)
                                    
                                    if param_value is not None and can_infer:
                                        all_params[input_param] = param_value
                                    else:
                                        missing_params.append(f"{tool_name}.{input_param}")
                                else:
                                    missing_params.append(f"{tool_name}.{input_param}")
                            except Exception as e:
                                print(f"  ⚠ 推断参数 {input_param} 失败: {e}")
                                missing_params.append(f"{tool_name}.{input_param}")
                        else:
                            # 没有 LLM，直接标记为缺失
                            missing_params.append(f"{tool_name}.{input_param}")
                continue
            
            input_params = tool_params.get("input_params", [])
            for param in input_params:
                param_name = param.get("name", "")
                param_type = param.get("type", "")
                param_desc = param.get("description", "")
                is_optional = "optional" in param_type.lower() or param_type.startswith("Optional")
                
                if not param_name:
                    continue
                
                # 如果参数已经在参数推断结果中，跳过（已经处理过了）
                if param_name in all_params or any(p.endswith(f".{param_name}") or p == param_name for p in missing_params):
                    continue
                
                # 使用 LLM 推断参数值（只对没有推断结果的参数）
                if llm and param_name not in all_params:
                    try:
                        from langchain_core.messages import SystemMessage, HumanMessage
                        
                        inference_prompt = f"""
请根据以下信息推断参数值：

任务描述: {task.content}
工具名称: {tool_name}
参数名称: {param_name}
参数类型: {param_type}
参数描述: {param_desc}
任务输入列表: {inputs}

请推断该参数的值。如果无法从任务描述中推断，或者需要用户提供（如文件路径、用户选择等），请返回 null。

返回JSON格式：
{{
    "value": <参数值或null>,
    "can_infer": <true/false>,
    "reason": "<推断理由或为什么需要用户提供>"
}}
"""
                        messages = [
                            SystemMessage(content="你是一个专业的参数推断专家，能够从任务描述中提取参数值。"),
                            HumanMessage(content=inference_prompt)
                        ]
                        
                        response = llm.invoke(messages)
                        response_text = response.content.strip()
                        
                        # 解析响应
                        import re
                        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                        if json_match:
                            inference_data = json.loads(json_match.group())
                            param_value = inference_data.get("value")
                            can_infer = inference_data.get("can_infer", False)
                            
                            if param_value is not None and can_infer:
                                all_params[param_name] = param_value
                            elif not is_optional:
                                missing_params.append(f"{tool_name}.{param_name}")
                        else:
                            if not is_optional:
                                missing_params.append(f"{tool_name}.{param_name}")
                    except Exception as e:
                        print(f"  ⚠ 推断参数 {param_name} 失败: {e}")
                        if not is_optional:
                            missing_params.append(f"{tool_name}.{param_name}")
        
        # 更新任务结果
        result.parameters = all_params
        result.missing_parameters = missing_params
        
        # 如果有缺失参数，触发 HITL
        if missing_params:
            state.hitl_requests[task.task_id] = {
                "type": "missing_parameters",
                "task_id": task.task_id,
                "task_description": task.content,
                "missing_parameters": missing_params,
                "message": f"任务 {task.task_id} 需要以下参数，请提供：{', '.join(missing_params)}"
            }
            state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_HITL_PARAMS
            print(f"  ⚠ 任务 {task.task_id} 需要用户提供参数: {', '.join(missing_params)}")
    
    return state


def check_hitl_params_node(state: ExecutorState) -> Literal["hitl_params", "execute"]:
    """检查是否需要 HITL 参数"""
    pending_hitl = [
        task_id for task_id, request in state.hitl_requests.items()
        if request.get("type") == "missing_parameters" and task_id not in state.hitl_responses
    ]
    
    if pending_hitl:
        return "hitl_params"
    else:
        return "execute"


def hitl_params_node(state: ExecutorState) -> ExecutorState:
    """
    HITL 参数请求节点
    
    向用户请求参数，并等待响应（使用 interrupt）
    支持从 interrupt 恢复时接收 resume 值
    
    工作流程：
    1. 首次执行：检查是否有未响应的 HITL 请求，如果有则触发 interrupt
    2. 恢复执行：从 interrupt 的 resume 值中获取用户响应，更新参数，继续执行
    """
    # 尝试获取 resume 值（如果是恢复执行）
    # 注意：在恢复执行时，interrupt() 会返回 Command(resume=...) 中的值
    # 首次调用时，interrupt() 会抛出异常（这是正常的）
    resume_value = None
    if INTERRUPT_AVAILABLE:
        try:
            # 先尝试获取 resume 值（不传参数）
            # 如果是恢复执行，这会返回 resume 值
            # 如果是首次调用，这会抛出异常，我们会在后面处理
            resume_value = interrupt()
        except Exception:
            # 首次调用时，interrupt() 会抛出异常（正常行为）
            # 我们会在后面需要中断时再次调用 interrupt(value)
            resume_value = None
    
    # 如果有 resume 值，说明这是从中断恢复，处理用户响应
    if resume_value is not None:
        # resume_value 可能是 Command 对象或字典
        # 如果是 Command 对象，需要提取 resume 字段
        if hasattr(resume_value, 'resume'):
            resume_data = resume_value.resume
        elif isinstance(resume_value, dict) and 'resume' in resume_value:
            resume_data = resume_value['resume']
        else:
            resume_data = resume_value
        
        if isinstance(resume_data, dict) and resume_data.get("type") == "response_parameters":
            responses = resume_data.get("responses", {})
            for task_id, response_data in responses.items():
                if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                    state.hitl_responses[task_id] = response_data
                    # 更新参数
                    if task_id in state.task_results:
                        result = state.task_results[task_id]
                        if "parameters" in response_data:
                            # 更新参数
                            result.parameters.update(response_data["parameters"])
                            # 从 missing_parameters 中移除已提供的参数
                            # missing_parameters 格式可能是 "tool_name.param_name" 或 "param_name"
                            provided_params = set(response_data["parameters"].keys())
                            result.missing_parameters = [
                                p for p in result.missing_parameters
                                if not any(
                                    p == param_name or p.endswith(f".{param_name}")
                                    for param_name in provided_params
                                )
                            ]
                        # 如果所有必需参数都已提供，标记为就绪
                        if not result.missing_parameters:
                            state.task_status_map[task_id] = ExecutorTaskStatus.READY
                            print(f"  ✓ 任务 {task_id} 已获得所有必需参数，标记为就绪")
    
    # 先检查是否有 HITL 响应（从 parent_state 中获取，作为降级方案）
    if state.parent_state and state.parent_state.hitl_status:
        try:
            hitl_data = json.loads(state.parent_state.hitl_status)
            if hitl_data.get("type") == "response_parameters":
                # 处理用户响应（降级方案：通过 parent_state 传递）
                responses = hitl_data.get("responses", {})
                for task_id, response_data in responses.items():
                    if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                        state.hitl_responses[task_id] = response_data
                        # 更新参数
                        if task_id in state.task_results:
                            result = state.task_results[task_id]
                            if "parameters" in response_data:
                                result.parameters.update(response_data["parameters"])
                                provided_params = set(response_data["parameters"].keys())
                                result.missing_parameters = [
                                    p for p in result.missing_parameters
                                    if not any(
                                        p == param_name or p.endswith(f".{param_name}")
                                        for param_name in provided_params
                                    )
                                ]
                            if not result.missing_parameters:
                                state.task_status_map[task_id] = ExecutorTaskStatus.READY
                                print(f"  ✓ 任务 {task_id} 已获得所有必需参数，标记为就绪")
        except Exception as e:
            print(f"  ⚠ 解析HITL响应失败: {e}")
    
    # 如果还有未响应的请求，设置 HITL 请求信息并触发 interrupt
    remaining_requests = [
        task_id for task_id in state.hitl_requests.keys()
        if task_id not in state.hitl_responses
    ]
    if remaining_requests and state.parent_state:
        hitl_messages = []
        for task_id in remaining_requests:
            request = state.hitl_requests[task_id]
            if request.get("type") == "missing_parameters":
                hitl_messages.append({
                    "task_id": task_id,
                    "message": request["message"],
                    "type": request["type"],
                    "missing_parameters": request.get("missing_parameters", [])
                })
        
        if hitl_messages:
            state.parent_state.hitl_status = json.dumps({
                "type": "missing_parameters",
                "requests": hitl_messages
            }, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"HITL请求：需要用户提供参数")
            print(f"{'='*60}")
            for msg in hitl_messages:
                print(f"任务 {msg['task_id']}: {msg['message']}")
            print(f"{'='*60}\n")
            
            # 触发 interrupt，暂停执行等待用户响应
            # 首次调用时，interrupt() 会抛出 GraphInterrupt 异常
            # LangGraph 会捕获这个异常，保存状态，并返回带有 __interrupt__ 字段的结果
            # 恢复时，调用者需要使用 Command(resume=...) 传递用户响应
            if INTERRUPT_AVAILABLE:
                try:
                    interrupt({
                        "type": "missing_parameters",
                        "requests": hitl_messages,
                        "message": "等待用户提供参数"
                    })
                except Exception as e:
                    # interrupt 会抛出异常（这是正常行为）
                    # LangGraph 会捕获并保存状态
                    # 异常会向上传播，让 LangGraph 处理
                    raise
            else:
                # 降级方案：使用状态标记
                print("  ⚠ 注意：interrupt 功能不可用，将使用状态标记方式")
    
    return state


def execute_tasks_node(state: ExecutorState) -> ExecutorState:
    """
    执行任务节点
    
    1. 获取所有就绪任务（排除正在运行的）
    2. 限制并行数量
    3. 并行执行任务
    4. 更新任务状态
    """
    ready_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.READY
        and task.task_id not in state.running_tasks
    ]
    
    if not ready_tasks:
        return state
    
    # 计算可执行的任务数量（考虑并行上限）
    available_slots = state.max_parallel_tasks - len(state.running_tasks)
    if available_slots <= 0:
        return state
    
    tasks_to_execute = ready_tasks[:available_slots]
    print(f"🔄 开始执行 {len(tasks_to_execute)} 个任务（当前运行: {len(state.running_tasks)}, 最大并行: {state.max_parallel_tasks}）")
    
    # 执行任务
    for task in tasks_to_execute:
        # 标记为运行中
        state.task_status_map[task.task_id] = ExecutorTaskStatus.RUNNING
        state.running_tasks.append(task.task_id)
        
        # 执行任务
        result = _execute_single_task(task, state)
        
        # 更新任务结果和状态
        state.task_results[result.task_id] = result
        state.task_status_map[result.task_id] = result.status
        state.running_tasks.remove(task.task_id)
        
        if result.status == ExecutorTaskStatus.COMPLETED:
            state.completed_count += 1
            print(f"  ✓ 任务 {result.task_id} 执行成功（耗时 {result.execution_time:.2f}秒）")
        else:
            state.failed_count += 1
            print(f"  ✗ 任务 {result.task_id} 执行失败: {result.error}")
    
    return state


def _execute_single_task(task: SubTask, state: ExecutorState) -> TaskExecutionResult:
    """执行单个任务"""
    start_time = time.time()
    result = state.task_results.get(task.task_id, TaskExecutionResult(
        task_id=task.task_id,
        status=ExecutorTaskStatus.RUNNING,
        execution_mode="",
        retry_count=0
    ))
    result.status = ExecutorTaskStatus.RUNNING
    
    try:
        # 确定执行模式
        task_result = task.result if isinstance(task.result, dict) else {}
        tools = task_result.get("tools", [])
        
        if tools and len(tools) > 0:
            execution_mode = CodeActExecutionMode.MCP_TOOL
        else:
            execution_mode = CodeActExecutionMode.CODEACT
        
        result.execution_mode = execution_mode.value
        
        # 获取已解析的参数
        parameters = result.parameters
        
        # 构建 CodeAct 子图输入
        codeact_input = codeact_input_mapper(
            executor_state=state,
            task=task,
            execution_mode=execution_mode,
            parameters=parameters
        )
        
        # 调用 CodeAct 子图
        codeact_graph = build_codeact_subgraph()
        codeact_output = codeact_graph.invoke(codeact_input)
        
        # 将字典输出转换为 CodeActState 对象（LangGraph 返回字典）
        if isinstance(codeact_output, dict):
            codeact_state = CodeActState.model_validate(codeact_output)
        else:
            codeact_state = codeact_output
        
        # 处理执行结果
        exec_result = codeact_output_mapper(codeact_state)
        if exec_result.get("status") == "success":
            result.status = ExecutorTaskStatus.COMPLETED
            result.code = exec_result.get("code")
            result.output = exec_result.get("output")
        else:
            result.status = ExecutorTaskStatus.FAILED
            result.error = exec_result.get("error", "执行失败")
            result.error_category = classify_error(
                result.error,
                exec_result.get("error_type", "UnknownError")
            )
    
    except Exception as e:
        result.status = ExecutorTaskStatus.FAILED
        error_msg = str(e)
        result.error_category = classify_error(error_msg, type(e).__name__)
        # 记录详细的错误信息以便调试
        import traceback
        error_traceback = traceback.format_exc()
        if len(error_traceback) > 1000:
            error_traceback = error_traceback[:1000] + "..."
        result.error = f"{error_msg}\n{error_traceback}"
        print(f"  ✗ 任务 {task.task_id} 执行异常: {error_msg}")
    
    result.execution_time = time.time() - start_time
    return result


def analyze_results_node(state: ExecutorState) -> ExecutorState:
    """
    结果推理节点
    
    使用 LLM 对任务执行结果进行推理，确定是否满足要求。
    如果不满足，触发 HITL 询问用户是否继续。
    """
    llm = create_reasoning_advanced_llm()
    
    # 分析已完成的任务
    for task in state.subtasks:
        result = state.task_results.get(task.task_id)
        if not result or result.status != ExecutorTaskStatus.COMPLETED:
            continue
        
        # 如果已经分析过，跳过
        if result.result_satisfied is not None:
            continue
        
        if llm and result.output:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                
                analysis_prompt = f"""
请评估以下任务执行结果是否满足要求：

任务ID: {result.task_id}
任务描述: {task.content}
执行模式: {result.execution_mode}
执行结果: {str(result.output)[:1000]}

请返回JSON格式：
{{
    "satisfied": <true/false>,
    "confidence": <0-1之间的浮点数>,
    "reason": "<评估理由>",
    "needs_user_confirmation": <true/false>
}}
"""
                messages = [
                    SystemMessage(content="你是一个专业的任务执行结果评估专家。"),
                    HumanMessage(content=analysis_prompt)
                ]
                
                response = llm.invoke(messages)
                response_text = response.content.strip()
                
                # 解析响应
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                    result.result_satisfied = analysis_data.get("satisfied", True)
                    result.confidence_score = analysis_data.get("confidence", 0.5)
                    needs_confirmation = analysis_data.get("needs_user_confirmation", False)
                    
                    # 如果不满足要求或需要用户确认，触发 HITL
                    if not result.result_satisfied or needs_confirmation:
                        state.hitl_requests[task.task_id] = {
                            "type": "result_confirmation",
                            "task_id": task.task_id,
                            "task_description": task.content,
                            "result": str(result.output)[:500],
                            "reason": analysis_data.get("reason", ""),
                            "message": f"任务 {task.task_id} 的执行结果可能不满足要求。是否继续执行后续任务？"
                        }
                        state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_HITL_CONFIRM
                        print(f"  ⚠ 任务 {task.task_id} 结果需要用户确认")
            except Exception as e:
                print(f"  ⚠ 分析结果失败: {e}")
                result.result_satisfied = True  # 默认认为满足要求
    
    return state


def check_hitl_confirm_node(state: ExecutorState) -> Literal["hitl_confirm", "activate"]:
    """检查是否需要 HITL 确认"""
    pending_hitl = [
        task_id for task_id, request in state.hitl_requests.items()
        if request.get("type") == "result_confirmation" and task_id not in state.hitl_responses
    ]
    
    if pending_hitl:
        return "hitl_confirm"
    else:
        return "activate"


def hitl_confirm_node(state: ExecutorState) -> ExecutorState:
    """
    HITL 确认节点
    
    向用户请求确认是否继续执行，并等待响应
    支持从 interrupt 恢复时接收 resume 值
    
    工作流程：
    1. 首次执行：检查是否有未响应的 HITL 确认请求，如果有则触发 interrupt
    2. 恢复执行：从 interrupt 的 resume 值中获取用户确认，更新状态，继续执行
    """
    # 尝试获取 resume 值（如果是恢复执行）
    resume_value = None
    if INTERRUPT_AVAILABLE:
        try:
            resume_value = interrupt()
        except Exception:
            # 首次调用时，interrupt() 会抛出异常（正常行为）
            resume_value = None
    
    # 如果有 resume 值，处理用户确认响应
    if resume_value is not None:
        # resume_value 可能是 Command 对象或字典
        if hasattr(resume_value, 'resume'):
            resume_data = resume_value.resume
        elif isinstance(resume_value, dict) and 'resume' in resume_value:
            resume_data = resume_value['resume']
        else:
            resume_data = resume_value
        
        if isinstance(resume_data, dict) and resume_data.get("type") == "response_confirmation":
            responses = resume_data.get("responses", {})
            for task_id, response_data in responses.items():
                if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                    state.hitl_responses[task_id] = response_data
                    # 更新用户选择
                    if task_id in state.task_results:
                        result = state.task_results[task_id]
                        result.user_continue = response_data.get("continue", True)
                        # 标记为已完成（无论用户选择继续与否）
                        state.task_status_map[task_id] = ExecutorTaskStatus.COMPLETED
                        print(f"  ✓ 任务 {task_id} 用户确认: {'继续' if result.user_continue else '停止'}")
    
    pending_requests = [
        (task_id, request) for task_id, request in state.hitl_requests.items()
        if request.get("type") == "result_confirmation" and task_id not in state.hitl_responses
    ]
    
    if not pending_requests:
        return state
    
    # 先检查是否有 HITL 响应（从 parent_state 中获取，作为降级方案）
    if state.parent_state and state.parent_state.hitl_status:
        try:
            hitl_data = json.loads(state.parent_state.hitl_status)
            if hitl_data.get("type") == "response_confirmation":
                responses = hitl_data.get("responses", {})
                for task_id, response_data in responses.items():
                    if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                        state.hitl_responses[task_id] = response_data
                        # 更新用户选择
                        if task_id in state.task_results:
                            result = state.task_results[task_id]
                            result.user_continue = response_data.get("continue", True)
                            state.task_status_map[task_id] = ExecutorTaskStatus.COMPLETED
                            print(f"  ✓ 任务 {task_id} 用户确认: {'继续' if result.user_continue else '停止'}")
        except Exception as e:
            print(f"  ⚠ 解析HITL响应失败: {e}")
    
    # 如果还有未响应的请求，设置 HITL 请求信息并触发 interrupt
    remaining_requests = [
        task_id for task_id in state.hitl_requests.keys()
        if task_id not in state.hitl_responses
    ]
    if remaining_requests and state.parent_state:
        hitl_messages = []
        for task_id in remaining_requests:
            request = state.hitl_requests[task_id]
            if request.get("type") == "result_confirmation":
                hitl_messages.append({
                    "task_id": task_id,
                    "message": request["message"],
                    "type": request["type"],
                    "result": request.get("result", ""),
                    "reason": request.get("reason", "")
                })
        
        if hitl_messages:
            state.parent_state.hitl_status = json.dumps({
                "type": "result_confirmation",
                "requests": hitl_messages
            }, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"HITL请求：需要用户确认是否继续")
            print(f"{'='*60}")
            for msg in hitl_messages:
                print(f"任务 {msg['task_id']}: {msg['message']}")
            print(f"{'='*60}\n")
            
            # 触发 interrupt，暂停执行等待用户确认
            if INTERRUPT_AVAILABLE:
                try:
                    interrupt({
                        "type": "result_confirmation",
                        "requests": hitl_messages,
                        "message": "等待用户确认是否继续"
                    })
                except Exception as e:
                    # interrupt 会抛出异常（这是正常行为）
                    # LangGraph 会捕获并保存状态
                    raise
            else:
                print("  ⚠ 注意：interrupt 功能不可用，将使用状态标记方式")
    
    return state


def activate_dependent_tasks_node(state: ExecutorState) -> ExecutorState:
    """
    激活依赖任务节点
    
    检查所有等待依赖的任务，如果其依赖都已完成，则标记为就绪
    如果依赖任务失败，也标记为就绪（允许继续执行，但可能失败）
    """
    waiting_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.WAITING_DEPENDENCY
    ]
    
    activated_count = 0
    for task in waiting_tasks:
        # 检查所有依赖是否都已完成或失败
        # 注意：即使依赖失败，也允许后续任务继续执行（可能用于错误处理或清理）
        all_deps_finished = all(
            dep_id in state.task_results and
            state.task_results[dep_id].status in [
                ExecutorTaskStatus.COMPLETED,
                ExecutorTaskStatus.FAILED
            ]
            for dep_id in task.dependencies
        )
        
        if all_deps_finished:
            # 检查是否有依赖失败
            has_failed_deps = any(
                dep_id in state.task_results and
                state.task_results[dep_id].status == ExecutorTaskStatus.FAILED
                for dep_id in task.dependencies
            )
            
            if has_failed_deps:
                # 如果依赖失败，仍然激活任务，但记录警告
                print(f"  ⚠ 任务 {task.task_id} 的依赖中有失败的任务，但仍将激活（允许错误处理）")
            
            state.task_status_map[task.task_id] = ExecutorTaskStatus.READY
            activated_count += 1
            print(f"  ✓ 任务 {task.task_id} 依赖已完成，已激活")
    
    # 更新循环计数器
    if activated_count > 0:
        # 如果激活了任务，重置计数器
        state.activate_iteration_count = 0
        print(f"✓ 激活了 {activated_count} 个新任务")
    else:
        # 如果没有激活任何任务，增加计数器
        state.activate_iteration_count += 1
        if waiting_tasks:
            print(f"  ⚠ 有 {len(waiting_tasks)} 个任务仍在等待依赖（连续激活迭代: {state.activate_iteration_count}/{state.max_activate_iterations}）")
            # 打印每个等待任务的依赖状态，便于调试
            for task in waiting_tasks[:3]:  # 只打印前3个，避免日志过长
                dep_statuses = []
                for dep_id in task.dependencies:
                    if dep_id in state.task_results:
                        dep_statuses.append(f"{dep_id}:{state.task_results[dep_id].status.value}")
                    else:
                        dep_statuses.append(f"{dep_id}:未执行")
                print(f"    任务 {task.task_id} 依赖状态: {', '.join(dep_statuses)}")
    
    return state


def check_completion_node(state: ExecutorState) -> Literal["infer_params", "activate", "summary"]:
    """
    检查完成状态节点
    
    判断是否所有任务都已完成，决定下一步操作
    
    返回逻辑：
    - "summary": 所有任务都已完成或失败，或者没有就绪任务且没有运行中任务（无法继续执行）
    - "infer_params": 有就绪任务需要执行
    - "activate": 没有就绪任务，但可能有等待依赖的任务需要激活，且还有运行中的任务（可能后续会有新的依赖任务被激活）
    """
    # 检查所有任务是否都已完成或失败
    all_completed = all(
        state.task_status_map.get(task.task_id) in [
            ExecutorTaskStatus.COMPLETED,
            ExecutorTaskStatus.FAILED
        ]
        for task in state.subtasks
    )
    
    if all_completed:
        print(f"  ✓ 所有任务已完成，准备汇总结果")
        return "summary"
    
    # 检查是否还有就绪任务（未运行）
    ready_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.READY
        and task.task_id not in state.running_tasks
    ]
    
    if ready_tasks:
        # 有新的就绪任务，重置循环计数器
        state.activate_iteration_count = 0
        print(f"  ✓ 有 {len(ready_tasks)} 个就绪任务，开始参数推断")
        return "infer_params"
    
    # 检查是否有运行中的任务
    running_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.RUNNING
        or task.task_id in state.running_tasks
    ]
    
    # 检查是否有等待依赖的任务
    waiting_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.WAITING_DEPENDENCY
    ]
    
    # 关键逻辑：如果没有就绪任务，也没有运行中的任务，说明不会有新的任务完成，也就不会有新的依赖任务被激活
    # 此时应该结束执行
    if not running_tasks:
        # 没有运行中的任务，说明不会有新的任务完成
        # 如果还有等待依赖的任务，说明它们的依赖永远不会完成（可能是死锁或依赖失败）
        if waiting_tasks:
            print(f"  ⚠ 没有就绪任务和运行中的任务，但有 {len(waiting_tasks)} 个等待依赖的任务，这些任务的依赖可能永远不会完成，标记为失败并结束")
            # 将所有等待依赖的任务标记为失败
            for task in waiting_tasks:
                state.task_status_map[task.task_id] = ExecutorTaskStatus.FAILED
                if task.task_id not in state.task_results:
                    state.task_results[task.task_id] = TaskExecutionResult(
                        task_id=task.task_id,
                        status=ExecutorTaskStatus.FAILED,
                        execution_mode="",
                        error="依赖任务无法完成，导致无法继续执行"
                    )
            return "summary"
        else:
            # 没有就绪任务，没有运行中的任务，也没有等待依赖的任务，但任务未全部完成
            # 检查是否有未初始化的任务或其他状态的任务
            uninitialized_tasks = [
                task for task in state.subtasks
                if task.task_id not in state.task_status_map
            ]
            
            other_state_tasks = [
                task for task in state.subtasks
                if state.task_status_map.get(task.task_id) in [
                    ExecutorTaskStatus.WAITING_HITL_PARAMS,
                    ExecutorTaskStatus.WAITING_HITL_CONFIRM
                ]
            ]
            
            if uninitialized_tasks:
                print(f"  ⚠ 检测到 {len(uninitialized_tasks)} 个未初始化的任务，标记为失败并结束")
                for task in uninitialized_tasks:
                    state.task_status_map[task.task_id] = ExecutorTaskStatus.FAILED
                    if task.task_id not in state.task_results:
                        state.task_results[task.task_id] = TaskExecutionResult(
                            task_id=task.task_id,
                            status=ExecutorTaskStatus.FAILED,
                            execution_mode="",
                            error="任务未正确初始化"
                        )
                return "summary"
            elif other_state_tasks:
                # 有 HITL 等待的任务，但没有运行中的任务，说明这些任务可能永远不会被响应
                # 为了安全，标记为失败并结束
                print(f"  ⚠ 检测到 {len(other_state_tasks)} 个 HITL 等待的任务，但没有运行中的任务，标记为失败并结束")
                for task in other_state_tasks:
                    state.task_status_map[task.task_id] = ExecutorTaskStatus.FAILED
                    if task.task_id not in state.task_results:
                        state.task_results[task.task_id] = TaskExecutionResult(
                            task_id=task.task_id,
                            status=ExecutorTaskStatus.FAILED,
                            execution_mode="",
                            error="HITL 等待超时，无法继续执行"
                        )
                return "summary"
            else:
                # 可能是状态不一致，强制结束
                print(f"  ⚠ 没有就绪任务、运行中的任务和等待依赖的任务，但任务未全部完成，可能是状态不一致，强制结束")
                return "summary"
    
    # 有运行中的任务，检查是否有等待依赖的任务可以激活
    if waiting_tasks:
        # 检查这些等待任务是否真的可以激活
        # 注意：即使依赖失败，也允许激活（activate_dependent_tasks_node 会处理）
        can_activate = False
        for task in waiting_tasks:
            all_deps_finished = all(
                dep_id in state.task_results and
                state.task_results[dep_id].status in [
                    ExecutorTaskStatus.COMPLETED,
                    ExecutorTaskStatus.FAILED
                ]
                for dep_id in task.dependencies
            )
            if all_deps_finished:
                can_activate = True
                break
        
        if can_activate:
            print(f"  ✓ 有等待依赖的任务可以激活（当前有 {len(running_tasks)} 个运行中的任务）")
            return "activate"
        else:
            # 所有等待任务都无法激活，但有运行中的任务，继续等待
            print(f"  ⚠ 有 {len(waiting_tasks)} 个等待依赖的任务，但依赖尚未完成，继续等待（当前有 {len(running_tasks)} 个运行中的任务）")
            return "activate"
    else:
        # 没有等待依赖的任务，但有运行中的任务，继续等待
        print(f"  ⚠ 没有等待依赖的任务，但有 {len(running_tasks)} 个运行中的任务，继续等待")
        return "activate"


def summary_results_node(state: ExecutorState) -> ExecutorState:
    """
    汇总结果节点
    
    汇总所有任务的执行结果
    """
    print(f"\n{'='*60}")
    print(f"执行完成汇总")
    print(f"{'='*60}")
    print(f"总任务数: {state.total_tasks}")
    print(f"已完成: {state.completed_count}")
    print(f"失败: {state.failed_count}")
    print(f"{'='*60}\n")
    
    return state


# ===================== 构建 Executor 子图 =====================

def build_executor_subgraph():
    """构建Executor子图（使用 LangGraph 1.0+ 特性）"""
    graph = StateGraph(ExecutorState)
    
    # 添加节点
    graph.add_node("initialize", initialize_tasks_node)
    graph.add_node("infer_params", infer_parameters_node)
    graph.add_node("check_hitl_params", check_hitl_params_node)  # 条件节点
    graph.add_node("hitl_params", hitl_params_node)
    graph.add_node("execute", execute_tasks_node)
    graph.add_node("analyze_results", analyze_results_node)
    graph.add_node("check_hitl_confirm", check_hitl_confirm_node)  # 条件节点
    graph.add_node("hitl_confirm", hitl_confirm_node)
    graph.add_node("activate", activate_dependent_tasks_node)
    graph.add_node("summary", summary_results_node)
    
    # 定义流程
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "infer_params")
    
    # 参数推断后，检查是否需要 HITL
    graph.add_conditional_edges(
        "infer_params",
        check_hitl_params_node,
        {
            "hitl_params": "hitl_params",
            "execute": "execute"
        }
    )
    
    # HITL 参数请求后，继续执行
    graph.add_edge("hitl_params", "execute")
    
    # 执行后，分析结果
    graph.add_edge("execute", "analyze_results")
    
    # 结果分析后，检查是否需要 HITL 确认
    graph.add_conditional_edges(
        "analyze_results",
        check_hitl_confirm_node,
        {
            "hitl_confirm": "hitl_confirm",
            "activate": "activate"
        }
    )
    
    # HITL 确认后，激活依赖任务
    graph.add_edge("hitl_confirm", "activate")
    
    # 激活依赖后，检查完成状态
    # 注意：这里直接路由，不需要额外的 check_completion 节点
    graph.add_conditional_edges(
        "activate",
        check_completion_node,
        {
            "infer_params": "infer_params",
            "activate": "activate",  # 如果还有依赖可以激活，继续激活
            "summary": "summary"  # 所有任务完成或无法继续，结束
        }
    )
    
    graph.add_edge("summary", END)
    
    # 使用 MemorySaver 作为 checkpoint（可以替换为持久化存储）
    if MemorySaver:
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    else:
        # 如果 MemorySaver 不可用，使用默认编译
        return graph.compile()


# ===================== 状态映射函数 =====================

def executor_input_mapper(global_state: GlobalState) -> ExecutorState:
    """
    将主图状态映射到Executor子图状态
    
    Args:
        global_state: 主图全局状态
    
    Returns:
        Executor子图状态
    """
    # 使用 model_construct 来绕过严格的验证，直接使用 SubTask 对象
    # 这样可以避免 Pydantic v2 对嵌套模型的严格验证问题
    executor_state = ExecutorState.model_construct(
        subtasks=global_state.subtasks,
        parallel_task_groups=global_state.parallel_task_groups,
        sandbox_dir=global_state.sandbox_dir,
        parent_state=global_state
    )
    
    return executor_state


def executor_output_mapper(executor_state: ExecutorState, global_state: GlobalState) -> GlobalState:
    """
    将Executor子图状态映射回主图状态
    
    Args:
        executor_state: Executor子图状态
        global_state: 主图全局状态
    
    Returns:
        更新后的主图状态
    """
    # 更新任务结果（包括subtasks和并行任务组中的任务）
    all_tasks_to_update = list(global_state.subtasks)
    
    # 从并行任务组中提取所有任务
    for group_id, group in global_state.parallel_task_groups.items():
        if hasattr(group, 'subtasks') and group.subtasks:
            for task in group.subtasks:
                if not any(t.task_id == task.task_id for t in all_tasks_to_update):
                    all_tasks_to_update.append(task)
    
    # 更新所有任务的结果
    for task in all_tasks_to_update:
        task_result = executor_state.task_results.get(task.task_id)
        if task_result and task_result.status == ExecutorTaskStatus.COMPLETED:
            # 更新任务结果
            if not task.result:
                task.result = {}
            if isinstance(task.result, dict):
                task.result["execution_result"] = task_result.output
                task.result["execution_mode"] = task_result.execution_mode
                task.result["code"] = task_result.code
                task.result["confidence_score"] = task_result.confidence_score
            
            # 标记任务为已完成
            global_state.completed_tasks[task.task_id] = task
    
    # 更新汇总结果
    global_state.merged_result["executor_results"] = {
        "total_tasks": executor_state.total_tasks,
        "completed": executor_state.completed_count,
        "failed": executor_state.failed_count,
        "task_results": {
            task_id: {
                "status": result.status.value,
                "execution_mode": result.execution_mode,
                "error": result.error,
                "error_category": result.error_category.value if result.error_category else None,
                "confidence_score": result.confidence_score,
                "failure_analysis": result.failure_analysis,
                "suggestions": result.suggestions
            }
            for task_id, result in executor_state.task_results.items()
        }
    }
    
    # 更新HITL状态（如果有）
    if executor_state.hitl_requests:
        pending_hitl = [
            task_id for task_id in executor_state.hitl_requests.keys()
            if task_id not in executor_state.hitl_responses
        ]
        if pending_hitl:
            global_state.hitl_status = json.dumps({
                "type": "request",
                "requests": [
                    executor_state.hitl_requests[task_id]
                    for task_id in pending_hitl
                ]
            }, ensure_ascii=False)
    
    return global_state


# ===================== Executor 子图执行包装函数（支持 Interrupt） =====================

def execute_executor_with_interrupt_support(
    executor_graph,
    initial_state: ExecutorState,
    thread_id: str = "default",
    resume_value: Optional[Any] = None
) -> Dict[str, Any]:
    """
    执行 Executor 子图，支持 interrupt 检测和恢复
    
    Args:
        executor_graph: 编译后的 Executor 子图
        initial_state: 初始状态
        thread_id: 线程ID（用于 checkpoint）
        resume_value: 恢复值（如果是恢复执行）
    
    Returns:
        包含执行结果和中断信息的字典：
        {
            "result": ExecutorState,  # 执行结果
            "interrupted": bool,  # 是否中断
            "interrupt_data": Any,  # 中断数据（如果有）
            "needs_resume": bool  # 是否需要恢复
        }
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # 如果是恢复执行，使用 Command(resume=...)
    if resume_value is not None and INTERRUPT_AVAILABLE and Command is not None:
        input_data = Command(resume=resume_value)
    else:
        # 首次执行，使用普通状态
        if isinstance(initial_state, dict):
            input_data = initial_state
        else:
            input_data = initial_state.model_dump(exclude={'parent_state'}, mode='json')
    
    # 使用 stream() 来检测中断
    interrupted = False
    interrupt_data = None
    final_result = None
    
    try:
        # 使用 stream 来逐步执行，可以检测中断
        for chunk in executor_graph.stream(input_data, config=config):
            # 检查是否有中断
            if "__interrupt__" in chunk:
                interrupted = True
                interrupt_data = chunk["__interrupt__"]
                # 获取当前状态（中断前的状态）
                # 注意：chunk 中可能包含状态信息
                for key, value in chunk.items():
                    if key != "__interrupt__":
                        # 这可能是状态更新
                        if isinstance(value, dict):
                            try:
                                final_result = ExecutorState.model_validate(value)
                            except:
                                pass
                break
            else:
                # 正常的状态更新
                for key, value in chunk.items():
                    if isinstance(value, dict):
                        try:
                            final_result = ExecutorState.model_validate(value)
                        except:
                            pass
        
        # 如果没有中断，获取最终结果
        if not interrupted and final_result is None:
            # 使用 invoke 获取最终结果
            output = executor_graph.invoke(input_data, config=config)
            if isinstance(output, dict):
                final_result = ExecutorState.model_validate(output)
            else:
                final_result = output
        
    except Exception as e:
        # 如果 interrupt 抛出异常，这是正常行为
        # LangGraph 会在 stream 中处理这个异常
        if "interrupt" in str(e).lower() or "GraphInterrupt" in str(type(e).__name__):
            interrupted = True
            # 尝试从异常中提取中断信息
            interrupt_data = getattr(e, 'value', None) or str(e)
        else:
            # 其他异常，重新抛出
            raise
    
    return {
        "result": final_result,
        "interrupted": interrupted,
        "interrupt_data": interrupt_data,
        "needs_resume": interrupted
    }


def resume_executor_after_interrupt(
    executor_graph,
    thread_id: str,
    resume_value: Any
) -> Dict[str, Any]:
    """
    恢复 Executor 子图的执行（在中断后）
    
    Args:
        executor_graph: 编译后的 Executor 子图
        thread_id: 线程ID（必须与中断时相同）
        resume_value: 恢复值（用户响应）
    
    Returns:
        执行结果字典（格式同 execute_executor_with_interrupt_support）
    """
    if not INTERRUPT_AVAILABLE or Command is None:
        raise ValueError("interrupt 功能不可用，无法恢复执行")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # 使用 Command(resume=...) 来恢复执行
    input_data = Command(resume=resume_value)
    
    # 继续执行
    interrupted = False
    interrupt_data = None
    final_result = None
    
    try:
        for chunk in executor_graph.stream(input_data, config=config):
            if "__interrupt__" in chunk:
                interrupted = True
                interrupt_data = chunk["__interrupt__"]
                break
            else:
                for key, value in chunk.items():
                    if isinstance(value, dict):
                        try:
                            final_result = ExecutorState.model_validate(value)
                        except:
                            pass
        
        if not interrupted and final_result is None:
            output = executor_graph.invoke(input_data, config=config)
            if isinstance(output, dict):
                final_result = ExecutorState.model_validate(output)
            else:
                final_result = output
                
    except Exception as e:
        if "interrupt" in str(e).lower() or "GraphInterrupt" in str(type(e).__name__):
            interrupted = True
            interrupt_data = getattr(e, 'value', None) or str(e)
        else:
            raise
    
    return {
        "result": final_result,
        "interrupted": interrupted,
        "interrupt_data": interrupt_data,
        "needs_resume": interrupted
    }
